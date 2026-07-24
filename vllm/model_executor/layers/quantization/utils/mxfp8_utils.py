# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from contextlib import nullcontext
from functools import cache
from typing import Any, NamedTuple

import torch

from vllm.compilation.passes.inductor_pass import InductorPass, get_pass_context
from vllm.utils.torch_utils import direct_register_custom_op

# MXFP8 constants
MXFP8_VALUE_DTYPE = torch.float8_e4m3fn
MXFP8_SCALE_DTYPE = torch.uint8
MXFP8_BLOCK_SIZE = 32
MXFP8_TRTLLM_8X4_MAX_M = 256
MXFP8_TRTLLM_LAYOUT_ENV = "VLLM_MXFP8_DENSE_TRTLLM_LAYOUT"
MXFP8_TRTLLM_SWITCH_M_ENV = "VLLM_MXFP8_DENSE_TRTLLM_SWITCH_M"
MXFP8_TRTLLM_HIGH_M_TACTIC_ENV = "VLLM_MXFP8_DENSE_TRTLLM_TACTIC"
MXFP8_TRTLLM_HIGH_M_TACTIC_HINTS_ENV = "VLLM_MXFP8_DENSE_TRTLLM_TACTIC_HINTS_128X4"


class _Mxfp8TrtllmLayoutConfig(NamedTuple):
    policy: str
    switch_m: int | None


@cache
def _mxfp8_trtllm_layout_config() -> _Mxfp8TrtllmLayoutConfig:
    policy = os.environ.get(MXFP8_TRTLLM_LAYOUT_ENV, "adaptive").strip().lower()
    normalized = policy.replace("_", "").replace("-", "")
    aliases = {
        "adaptive": "adaptive",
        "8x4": "8x4",
        "8by4": "8x4",
        "128x4": "128x4",
        "128by4": "128x4",
    }
    if normalized not in aliases:
        raise ValueError(
            f"{MXFP8_TRTLLM_LAYOUT_ENV} must be one of adaptive, 8x4, or "
            f"128x4; got {policy!r}."
        )
    resolved_policy = aliases[normalized]
    if resolved_policy != "adaptive":
        return _Mxfp8TrtllmLayoutConfig(resolved_policy, None)

    raw_switch_m = os.environ.get(
        MXFP8_TRTLLM_SWITCH_M_ENV,
        str(MXFP8_TRTLLM_8X4_MAX_M),
    )
    try:
        switch_m = int(raw_switch_m)
    except ValueError as exc:
        raise ValueError(
            f"{MXFP8_TRTLLM_SWITCH_M_ENV} must be a positive integer; "
            f"got {raw_switch_m!r}."
        ) from exc
    if switch_m <= 0:
        raise ValueError(
            f"{MXFP8_TRTLLM_SWITCH_M_ENV} must be positive; got {switch_m}."
        )
    return _Mxfp8TrtllmLayoutConfig(resolved_policy, switch_m)


def mxfp8_trtllm_layout_policy() -> str:
    return _mxfp8_trtllm_layout_config().policy


def mxfp8_trtllm_switch_m() -> int:
    switch_m = _mxfp8_trtllm_layout_config().switch_m
    if switch_m is None:
        return MXFP8_TRTLLM_8X4_MAX_M
    return switch_m


def mxfp8_trtllm_use_8x4_sf_layout(m: int) -> bool:
    config = _mxfp8_trtllm_layout_config()
    if config.policy == "8x4":
        return True
    if config.policy == "128x4":
        return False
    assert config.switch_m is not None
    return m <= config.switch_m


def mxfp8_trtllm_scale_numel(m: int, k: int, use_8x4: bool) -> int:
    if k % MXFP8_BLOCK_SIZE != 0:
        raise ValueError(f"MXFP8 K must be divisible by 32, got K={k}.")
    m_tile = 8 if use_8x4 else 128
    m_padded = (m + m_tile - 1) // m_tile * m_tile
    scale_k = k // MXFP8_BLOCK_SIZE
    scale_k_padded = (scale_k + 3) // 4 * 4
    return m_padded * scale_k_padded


def _parse_mxfp8_tactic_hints(raw: str) -> dict[tuple[int, int, int], int]:
    hints: dict[tuple[int, int, int], int] = {}
    for item in raw.split(";"):
        item = item.strip()
        if not item:
            continue
        shape_raw, separator, tactic_raw = item.partition(":")
        if not separator:
            raise ValueError(
                f"MXFP8 tactic hints must use M,N,K:tactic entries; got {item!r}."
            )
        shape = tuple(int(value.strip()) for value in shape_raw.split(","))
        if len(shape) != 3 or any(value <= 0 for value in shape):
            raise ValueError(f"Invalid MXFP8 tactic shape {shape_raw!r}.")
        tactic = int(tactic_raw.strip())
        if tactic < -1:
            raise ValueError(f"Invalid MXFP8 tactic {tactic}; expected -1 or greater.")
        hints[(shape[0], shape[1], shape[2])] = tactic
    return hints


def _resolve_mxfp8_high_m_tactic(
    m: int,
    n: int,
    k: int,
    hints: dict[tuple[int, int, int], int],
    fallback: int,
    *,
    use_global_fallback: bool,
) -> int | None:
    shape = (m, n, k)
    if shape in hints:
        return hints[shape]
    return fallback if use_global_fallback else None


def mxfp8_trtllm_high_m_static_tactics_enabled() -> bool:
    return MXFP8_TRTLLM_HIGH_M_TACTIC_ENV in os.environ or bool(
        os.environ.get(MXFP8_TRTLLM_HIGH_M_TACTIC_HINTS_ENV, "").strip()
    )


class _Mxfp8HighMTrtllmState(NamedTuple):
    workspace: torch.Tensor
    runner: Any
    fallback_tactic: int
    tactic_hints: dict[tuple[int, int, int], int]
    use_global_fallback: bool


_MXFP8_HIGH_M_TRTLLM_STATES: dict[tuple[str, int], _Mxfp8HighMTrtllmState] = {}


def _mxfp8_cuda_device_key(device: torch.device) -> tuple[str, int]:
    canonical = torch.device(device)
    if canonical.type != "cuda":
        raise RuntimeError(f"MXFP8 TRTLLM tactics require CUDA, got {canonical}.")
    index = canonical.index
    if index is None:
        index = torch.cuda.current_device()
    return canonical.type, index


def prepare_mxfp8_trtllm_high_m_tactic_state(
    device: torch.device,
) -> _Mxfp8HighMTrtllmState | None:
    if not mxfp8_trtllm_high_m_static_tactics_enabled():
        return None

    device_key = _mxfp8_cuda_device_key(device)
    existing = _MXFP8_HIGH_M_TRTLLM_STATES.get(device_key)
    if existing is not None:
        return existing
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "MXFP8 high-M TRTLLM tactic state must be prepared before "
            "CUDA Graph capture."
        )

    fallback_tactic = int(os.environ.get(MXFP8_TRTLLM_HIGH_M_TACTIC_ENV, "-1"))
    if fallback_tactic < -1:
        raise ValueError(f"{MXFP8_TRTLLM_HIGH_M_TACTIC_ENV} must be -1 or greater.")
    tactic_hints = _parse_mxfp8_tactic_hints(
        os.environ.get(MXFP8_TRTLLM_HIGH_M_TACTIC_HINTS_ENV, "")
    )

    from flashinfer.gemm.gemm_base import (
        DEFAULT_WORKSPACE_SIZE,
        _get_cache_buf,
        get_trtllm_gemm_module,
    )

    canonical = torch.device(device_key[0], device_key[1])
    with torch.cuda.device(canonical):
        workspace = _get_cache_buf(
            "vllm_mxfp8_trtllm_high_m_static_tactic_workspace",
            DEFAULT_WORKSPACE_SIZE,
            canonical,
        )
        runner = get_trtllm_gemm_module().trtllm_mxfp8_gemm_runner(
            use_8x4_sf_layout=False
        )

    state = _Mxfp8HighMTrtllmState(
        workspace=workspace,
        runner=runner,
        fallback_tactic=fallback_tactic,
        tactic_hints=tactic_hints,
        use_global_fallback=MXFP8_TRTLLM_HIGH_M_TACTIC_ENV in os.environ,
    )
    _MXFP8_HIGH_M_TRTLLM_STATES[device_key] = state
    return state


def _require_mxfp8_trtllm_high_m_tactic_state(
    device: torch.device,
) -> _Mxfp8HighMTrtllmState:
    device_key = _mxfp8_cuda_device_key(device)
    state = _MXFP8_HIGH_M_TRTLLM_STATES.get(device_key)
    if state is not None:
        return state
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "MXFP8 high-M TRTLLM tactic state was not prepared before "
            "CUDA Graph capture."
        )
    prepared = prepare_mxfp8_trtllm_high_m_tactic_state(device)
    if prepared is None:
        raise RuntimeError("MXFP8 high-M static tactic path is not enabled.")
    return prepared


def swizzle_mxfp8_scale(sf: torch.Tensor, M: int, K: int) -> torch.Tensor:
    """Swizzle MXFP8 scales from row-major 2D to F8_128x4 layout."""
    scaling_vector_size = MXFP8_BLOCK_SIZE  # 32 for MXFP8
    factor = scaling_vector_size * 4  # 128

    num_m_tiles = (M + 127) // 128
    num_k_tiles = (K + factor - 1) // factor

    m_padded = num_m_tiles * 128
    k_scale_padded = num_k_tiles * 4

    scale_cols = K // scaling_vector_size
    sf_padded = torch.zeros(
        (m_padded, k_scale_padded), dtype=sf.dtype, device=sf.device
    )
    sf_padded[:M, :scale_cols] = sf

    sf_reshaped = sf_padded.view(num_m_tiles, 4, 32, num_k_tiles, 4)

    sf_swizzled = sf_reshaped.transpose(1, 3)

    return sf_swizzled.contiguous().view(-1)


def _mxfp8_e4m3_quantize_torch(
    x: torch.Tensor,
    is_sf_swizzled_layout: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Naive MXFP8 quantization.
    For each block of 32 elements along the last dimension, compute a
    shared e8m0 scale (the biased exponent of the block-wise amax)
    and quantize each element to float8_e4m3fn.

    Returns (quantized_values [same shape, fp8], scales uint8).
    Scale shape depends on is_sf_swizzled_layout:
      False -> [..., K//32]  (row-major 2D)
      True  -> [flat swizzled 1D]
    """
    assert x.shape[-1] % MXFP8_BLOCK_SIZE == 0
    orig_shape = x.shape
    num_blocks = x.shape[-1] // MXFP8_BLOCK_SIZE

    x_fp32 = x.to(torch.float32)
    x_blocked = x_fp32.view(*orig_shape[:-1], num_blocks, MXFP8_BLOCK_SIZE)

    amax = x_blocked.abs().amax(dim=-1)
    amax = amax.clamp(min=torch.finfo(torch.float32).tiny)
    scale_biased = torch.floor(torch.log2(amax)) + 127.0
    scale_biased = scale_biased.clamp(0, 254)
    scales_uint8 = scale_biased.to(torch.uint8)

    descale = torch.exp2(scale_biased - 127.0)
    x_scaled = x_blocked / descale.unsqueeze(-1)

    x_fp8 = x_scaled.view(orig_shape).to(MXFP8_VALUE_DTYPE)

    if x.ndim == 2:
        M, K = x.shape
        scales_uint8 = scales_uint8.view(M, -1)
        if is_sf_swizzled_layout:
            scales_uint8 = swizzle_mxfp8_scale(scales_uint8, M=M, K=K)
    elif x.ndim == 3:
        B, M, K = x.shape
        scales_uint8 = scales_uint8.view(B, M, -1)
        if is_sf_swizzled_layout:
            swizzled = []
            for i in range(B):
                swizzled.append(swizzle_mxfp8_scale(scales_uint8[i], M=M, K=K))
            scales_uint8 = torch.cat(swizzled)

    return x_fp8, scales_uint8


def _mxfp8_quant_triton_kernel():
    """Lazily-built Triton kernel: per-32-block E8M0 scale + FP8-E4M3 quant.

    Fuses what ``_mxfp8_e4m3_quantize_torch`` does in several elementwise passes
    into one launch. Each program handles ``[BLOCK_M, 32]`` (one MX block).
    """
    from vllm.triton_utils import tl, triton

    @triton.jit
    def _kernel(
        x_ptr,
        xq_ptr,
        s_ptr,
        M,
        K,
        sxm,
        sxk,
        sqm,
        sqk,
        ssm,
        ssk,
        BLOCK_M: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_b = tl.program_id(1)  # which 32-element block along K
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_k = pid_b * 32 + tl.arange(0, 32)
        m_mask = offs_m < M
        x = tl.load(
            x_ptr + offs_m[:, None] * sxm + offs_k[None, :] * sxk,
            mask=m_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        amax = tl.maximum(tl.max(tl.abs(x), axis=1), 1e-30)  # [BLOCK_M]
        sb = tl.floor(tl.log2(amax)) + 127.0
        sb = tl.minimum(tl.maximum(sb, 0.0), 254.0)
        descale = tl.exp2(sb - 127.0)
        xq = (x / descale[:, None]).to(xq_ptr.dtype.element_ty)
        tl.store(
            xq_ptr + offs_m[:, None] * sqm + offs_k[None, :] * sqk,
            xq,
            mask=m_mask[:, None],
        )
        tl.store(s_ptr + offs_m * ssm + pid_b * ssk, sb.to(tl.uint8), mask=m_mask)

    return _kernel


_MXFP8_QUANT_KERNEL = None


def _mxfp8_e4m3_quantize_triton(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused 2D MXFP8 quant (non-swizzled, row-major [M, K//32] scales)."""
    from vllm.triton_utils import triton

    global _MXFP8_QUANT_KERNEL
    if _MXFP8_QUANT_KERNEL is None:
        _MXFP8_QUANT_KERNEL = _mxfp8_quant_triton_kernel()

    M, K = x.shape
    x = x.contiguous()
    xq = torch.empty((M, K), dtype=MXFP8_VALUE_DTYPE, device=x.device)
    scales = torch.empty(
        (M, K // MXFP8_BLOCK_SIZE), dtype=MXFP8_SCALE_DTYPE, device=x.device
    )
    BLOCK_M = 64
    grid = (triton.cdiv(M, BLOCK_M), K // MXFP8_BLOCK_SIZE)
    _MXFP8_QUANT_KERNEL[grid](
        x,
        xq,
        scales,
        M,
        K,
        x.stride(0),
        x.stride(1),
        xq.stride(0),
        xq.stride(1),
        scales.stride(0),
        scales.stride(1),
        BLOCK_M=BLOCK_M,
    )
    return xq, scales


def _mxfp8_e4m3_quantize_impl(
    x: torch.Tensor,
    is_sf_swizzled_layout: bool = False,
    alignment: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    from vllm.platforms import current_platform

    if current_platform.has_device_capability(100):
        from flashinfer import mxfp8_quantize as flashinfer_mxfp8_quantize

        x_q, x_scales = flashinfer_mxfp8_quantize(
            x,
            is_sf_swizzled_layout=is_sf_swizzled_layout,
            alignment=alignment if alignment > 0 else 32,
            backend="cute-dsl",
        )
        if x_scales.ndim == 1 and x.ndim == 2 and not is_sf_swizzled_layout:
            x_scales = x_scales.view(x.size(0), -1)
        return x_q, x_scales

    # ROCm: a single fused Triton kernel beats the multi-pass torch path for the
    # common 2D, non-swizzled activation-quant case (used by the native MX
    # linear/MoE). Falls back to torch otherwise (3D weights, swizzled layout).
    if (
        current_platform.is_rocm()
        and not is_sf_swizzled_layout
        and x.ndim == 2
        and x.shape[-1] % MXFP8_BLOCK_SIZE == 0
    ):
        return _mxfp8_e4m3_quantize_triton(x)

    return _mxfp8_e4m3_quantize_torch(x, is_sf_swizzled_layout)


def mxfp8_e4m3_quantize(
    x: torch.Tensor,
    is_sf_swizzled_layout: bool = False,
    alignment: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.vllm.mxfp8_quantize(x, is_sf_swizzled_layout, alignment)


def dequant_mxfp8_to_bf16(x: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Dequantize MXFP8 tensor to BF16."""
    x_float = x.to(torch.float32)

    num_blocks = x.shape[-1] // MXFP8_BLOCK_SIZE
    x_blocked = x_float.view(*x.shape[:-1], num_blocks, MXFP8_BLOCK_SIZE)

    descale = torch.exp2(scales.to(torch.float32) - 127.0)

    dequantized = x_blocked * descale.unsqueeze(-1)

    dequantized = dequantized.view(*x.shape)

    return dequantized.to(torch.bfloat16)


def mxfp8_e4m3_quantize_fake(
    x: torch.Tensor,
    is_sf_swizzled_layout: bool = False,
    alignment: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fake implementation for torch.compile tracing."""
    fp_data = torch.empty_like(x, dtype=MXFP8_VALUE_DTYPE)

    block_size = MXFP8_BLOCK_SIZE

    if x.ndim == 2:
        M, N = x.shape
        K = (N + block_size - 1) // block_size
        if is_sf_swizzled_layout:
            M_padded = ((M + 127) // 128) * 128
            K_padded = ((K + 3) // 4) * 4
            scales = torch.empty(
                M_padded * K_padded, dtype=MXFP8_SCALE_DTYPE, device=x.device
            )
        else:
            scales = torch.empty((M, K), dtype=MXFP8_SCALE_DTYPE, device=x.device)
    elif x.ndim == 3:
        B, M, N = x.shape
        K = (N + block_size - 1) // block_size
        if is_sf_swizzled_layout:
            M_padded = ((M + 127) // 128) * 128
            K_padded = ((K + 3) // 4) * 4
            scales = torch.empty(
                B * M_padded * K_padded, dtype=MXFP8_SCALE_DTYPE, device=x.device
            )
        else:
            scales = torch.empty((B, M, K), dtype=MXFP8_SCALE_DTYPE, device=x.device)
    else:
        scale_shape = list(x.shape)
        scale_shape[-1] = (x.shape[-1] + block_size - 1) // block_size
        scales = torch.empty(scale_shape, dtype=MXFP8_SCALE_DTYPE, device=x.device)

    return fp_data, scales


direct_register_custom_op(
    op_name="mxfp8_quantize",
    op_func=_mxfp8_e4m3_quantize_impl,
    fake_impl=mxfp8_e4m3_quantize_fake,
)


def _mxfp8_trtllm_linear_fixed_impl(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    output_features: int,
    *,
    use_8x4_sf_layout: bool,
) -> torch.Tensor:
    from flashinfer import SfLayout
    from flashinfer import autotune as flashinfer_autotune
    from flashinfer import mm_mxfp8 as flashinfer_mm_mxfp8
    from flashinfer import mxfp8_quantize as flashinfer_mxfp8_quantize

    sf_layout = SfLayout.layout_8x4 if use_8x4_sf_layout else SfLayout.layout_128x4
    input_mxfp8, input_scale = flashinfer_mxfp8_quantize(
        x,
        alignment=MXFP8_BLOCK_SIZE,
        backend="cuda",
        sf_swizzle_layout=sf_layout,
    )
    if not use_8x4_sf_layout and mxfp8_trtllm_high_m_static_tactics_enabled():
        state = _require_mxfp8_trtllm_high_m_tactic_state(x.device)
        logical_shape = (int(x.shape[0]), int(output_features), int(x.shape[1]))
        tactic = _resolve_mxfp8_high_m_tactic(
            *logical_shape,
            state.tactic_hints,
            state.fallback_tactic,
            use_global_fallback=state.use_global_fallback,
        )
        if tactic is not None:
            physical_n = int(weight.shape[0])
            output = torch.empty(
                (x.shape[0], physical_n), dtype=torch.bfloat16, device=x.device
            )
            output = state.runner.forward(
                [
                    input_mxfp8,
                    weight.t(),
                    input_scale,
                    weight_scale,
                    torch.bfloat16,
                    output,
                    state.workspace,
                ],
                tactic=tactic,
            )
            return output[:, :output_features].contiguous()

    # Exact-hint misses keep the serving-stable FlashInfer wrapper path.
    autotune_context = (
        nullcontext()
        if use_8x4_sf_layout
        else flashinfer_autotune(skip_ops={"mxfp8_gemm"})
    )
    with autotune_context:
        output = flashinfer_mm_mxfp8(
            input_mxfp8,
            weight.t(),
            input_scale,
            weight_scale,
            out_dtype=torch.bfloat16,
            use_8x4_sf_layout=use_8x4_sf_layout,
            backend="trtllm",
        )
    return output[:, :output_features].contiguous()


def _mxfp8_trtllm_adaptive_linear_impl(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    output_features: int,
) -> torch.Tensor:
    return _mxfp8_trtllm_linear_fixed_impl(
        x,
        weight,
        weight_scale,
        output_features,
        use_8x4_sf_layout=mxfp8_trtllm_use_8x4_sf_layout(int(x.shape[0])),
    )


def _mxfp8_trtllm_linear_8x4_impl(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    output_features: int,
) -> torch.Tensor:
    return _mxfp8_trtllm_linear_fixed_impl(
        x, weight, weight_scale, output_features, use_8x4_sf_layout=True
    )


def _mxfp8_trtllm_linear_128x4_impl(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    output_features: int,
) -> torch.Tensor:
    return _mxfp8_trtllm_linear_fixed_impl(
        x, weight, weight_scale, output_features, use_8x4_sf_layout=False
    )


def mxfp8_trtllm_adaptive_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    output_features: int,
) -> torch.Tensor:
    return torch.ops.vllm.mxfp8_trtllm_adaptive_linear(
        x, weight, weight_scale, output_features
    )


def mxfp8_trtllm_adaptive_linear_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    output_features: int,
) -> torch.Tensor:
    if x.ndim != 2:
        raise ValueError(f"TRTLLM MXFP8 linear requires 2D input, got {x.ndim}D.")
    return torch.empty(
        (x.shape[0], output_features), dtype=torch.bfloat16, device=x.device
    )


direct_register_custom_op(
    op_name="mxfp8_trtllm_adaptive_linear",
    op_func=_mxfp8_trtllm_adaptive_linear_impl,
    fake_impl=mxfp8_trtllm_adaptive_linear_fake,
)

direct_register_custom_op(
    op_name="mxfp8_trtllm_linear_8x4",
    op_func=_mxfp8_trtllm_linear_8x4_impl,
    fake_impl=mxfp8_trtllm_adaptive_linear_fake,
)

direct_register_custom_op(
    op_name="mxfp8_trtllm_linear_128x4",
    op_func=_mxfp8_trtllm_linear_128x4_impl,
    fake_impl=mxfp8_trtllm_adaptive_linear_fake,
)


def _mxfp8_layout_for_compile_range(
    range_start: int, range_end: int, switch_m: int
) -> bool:
    if range_end <= switch_m:
        return True
    if range_start > switch_m:
        return False
    raise RuntimeError(
        f"MXFP8 compile range [{range_start}, {range_end}] straddles "
        f"adaptive layout switch M={switch_m}."
    )


def _specialize_mxfp8_adaptive_layout_graph(
    graph: Any, *, marker_op: Any, fixed_op: Any
) -> int:
    replaced = 0
    for node in graph.nodes:
        if node.op == "call_function" and node.target == marker_op:
            node.target = fixed_op
            replaced += 1
    return replaced


class _Mxfp8AdaptiveLayoutSpecializationPass(InductorPass):
    def __init__(self, layout_policy: str, switch_m: int | None) -> None:
        self.layout_policy = layout_policy
        self.switch_m = switch_m

    def __call__(self, graph: torch.fx.Graph) -> None:
        compile_range = get_pass_context().compile_range
        if self.layout_policy == "8x4":
            use_8x4_sf_layout = True
        elif self.layout_policy == "128x4":
            use_8x4_sf_layout = False
        else:
            assert self.switch_m is not None
            use_8x4_sf_layout = _mxfp8_layout_for_compile_range(
                compile_range.start, compile_range.end, self.switch_m
            )
        replaced = _specialize_mxfp8_adaptive_layout_graph(
            graph,
            marker_op=torch.ops.vllm.mxfp8_trtllm_adaptive_linear.default,
            fixed_op=(
                torch.ops.vllm.mxfp8_trtllm_linear_8x4.default
                if use_8x4_sf_layout
                else torch.ops.vllm.mxfp8_trtllm_linear_128x4.default
            ),
        )
        if replaced == 0:
            return

    def uuid(self) -> str:
        return self.hash_dict(
            {
                "source": self.hash_source(self),
                "layout_policy": self.layout_policy,
                "switch_m": self.switch_m,
                "phase": "joint_custom_pre_pass",
                "schema_version": 2,
            }
        )


def configure_mxfp8_trtllm_adaptive_compilation() -> None:
    from vllm.config import get_current_vllm_config

    vllm_config = get_current_vllm_config()
    compilation_config = vllm_config.compilation_config
    max_num_batched_tokens = vllm_config.scheduler_config.max_num_batched_tokens
    layout_config = _mxfp8_trtllm_layout_config()
    layout_policy = layout_config.policy
    switch_m = layout_config.switch_m
    if layout_policy == "adaptive" and max_num_batched_tokens is None:
        raise RuntimeError(
            "TRTLLM MXFP8 adaptive layout requires finite max_num_batched_tokens."
        )

    endpoints = list(compilation_config.compile_ranges_endpoints or [])
    if (
        layout_policy == "adaptive"
        and max_num_batched_tokens is not None
        and switch_m is not None
        and max_num_batched_tokens > switch_m
    ):
        endpoints.append(switch_m)
    compilation_config.compile_ranges_endpoints = sorted(set(endpoints))

    pass_key = "joint_custom_pre_pass"
    existing_pass = compilation_config.inductor_compile_config.get(pass_key)
    if existing_pass is None:
        compilation_config.inductor_compile_config[pass_key] = (
            _Mxfp8AdaptiveLayoutSpecializationPass(layout_policy, switch_m)
        )
    elif not isinstance(existing_pass, _Mxfp8AdaptiveLayoutSpecializationPass):
        raise RuntimeError(
            "TRTLLM MXFP8 adaptive layout cannot replace an existing "
            "Inductor joint custom pre-pass."
        )
    elif (
        existing_pass.layout_policy != layout_policy
        or existing_pass.switch_m != switch_m
    ):
        raise RuntimeError("TRTLLM MXFP8 layout policy changed after setup.")


def xpu_mxfp8_quantize(
    x: torch.Tensor, dtype: torch.dtype | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.vllm.xpu_mxfp8_quantize(x, dtype)
