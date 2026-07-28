# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from typing import Protocol

import torch

from vllm.utils.torch_utils import direct_register_custom_op

# MXFP8 constants
MXFP8_VALUE_DTYPE = torch.float8_e4m3fn
MXFP8_SCALE_DTYPE = torch.uint8
MXFP8_BLOCK_SIZE = 32

_MXFP8_DENSE_QUANT_BACKEND: str | None = None


class _Mxfp8DenseRuntimeConfiguration(Protocol):
    layout_mode: str
    switch_m: int
    quant_backend: str
    require_8x4_quant: bool


def _mxfp8_dense_runtime_configuration(
) -> _Mxfp8DenseRuntimeConfiguration | None:
    if not os.environ.get("VLLM_MXFP8_DENSE_CONFIG_FILE", "").strip():
        return None
    from vllm.utils import flashinfer as vllm_flashinfer

    return vllm_flashinfer.get_mxfp8_trtllm_configuration()


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off", "")


def prepare_mxfp8_dense_quant_backend(backend: str | None = None) -> str:
    global _MXFP8_DENSE_QUANT_BACKEND
    runtime_configuration = _mxfp8_dense_runtime_configuration()
    active_backend = (
        (
            backend
            if backend is not None
            else (
                runtime_configuration.quant_backend
                if runtime_configuration is not None
                else os.environ.get("VLLM_MXFP8_DENSE_QUANT_BACKEND", "cuda")
            )
        )
        .strip()
        .lower()
    )
    if _MXFP8_DENSE_QUANT_BACKEND is None:
        _MXFP8_DENSE_QUANT_BACKEND = active_backend
    elif active_backend != _MXFP8_DENSE_QUANT_BACKEND:
        raise RuntimeError(
            "MXFP8 dense quantization backend changed after preparation; "
            "restart the worker before changing it"
        )
    return _MXFP8_DENSE_QUANT_BACKEND


def _mxfp8_dense_quant_backend() -> str:
    return prepare_mxfp8_dense_quant_backend()


def _mxfp8_dense_a_sf_layout() -> str:
    runtime_configuration = _mxfp8_dense_runtime_configuration()
    if runtime_configuration is not None:
        return runtime_configuration.layout_mode
    return os.environ.get("VLLM_MXFP8_DENSE_A_SF_LAYOUT", "128x4").strip().lower()


def mxfp8_dense_use_8x4_sf_layout(m: int) -> bool:
    layout = _mxfp8_dense_a_sf_layout()
    if layout in ("8x4", "layout_8x4", "true", "1"):
        return True
    if layout in ("128x4", "layout_128x4", "false", "0"):
        return False
    if layout in ("adaptive", "shape-aware", "shape_aware"):
        runtime_configuration = _mxfp8_dense_runtime_configuration()
        threshold = (
            runtime_configuration.switch_m
            if runtime_configuration is not None
            else int(
                os.environ.get(
                    "VLLM_MXFP8_DENSE_A_SF_LAYOUT_SWITCH_M", "256"
                )
            )
        )
        if threshold <= 0:
            raise ValueError("VLLM_MXFP8_DENSE_A_SF_LAYOUT_SWITCH_M must be positive")
        return int(m) <= threshold
    raise ValueError(
        f"VLLM_MXFP8_DENSE_A_SF_LAYOUT must be 8x4, 128x4, or adaptive; got {layout!r}"
    )


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


def _flashinfer_mxfp8_quantize_impl(
    x: torch.Tensor,
    is_sf_swizzled_layout: bool,
    alignment: int,
    use_8x4_sf_layout: bool,
    *,
    require_exact_8x4_layout: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    from flashinfer import mxfp8_quantize as flashinfer_mxfp8_quantize

    quant_kwargs = {
        "input": x,
        "is_sf_swizzled_layout": is_sf_swizzled_layout,
        "alignment": alignment if alignment > 0 else 32,
    }
    use_8x4 = is_sf_swizzled_layout and use_8x4_sf_layout
    if use_8x4:
        from flashinfer import SfLayout

        quant_kwargs["sf_swizzle_layout"] = SfLayout.layout_8x4
        quant_kwargs["backend"] = _mxfp8_dense_quant_backend()

    try:
        x_q, x_scales = flashinfer_mxfp8_quantize(**quant_kwargs)
    except TypeError:
        runtime_configuration = _mxfp8_dense_runtime_configuration()
        require_8x4_quant = (
            runtime_configuration.require_8x4_quant
            if runtime_configuration is not None
            else _env_flag("VLLM_MXFP8_DENSE_REQUIRE_8X4_QUANT")
        )
        if use_8x4 and (
            require_exact_8x4_layout or require_8x4_quant
        ):
            raise
        x_q, x_scales = flashinfer_mxfp8_quantize(
            x,
            is_sf_swizzled_layout=is_sf_swizzled_layout,
            alignment=alignment if alignment > 0 else 32,
        )
    if x_scales.ndim == 1 and x.ndim == 2 and not is_sf_swizzled_layout:
        x_scales = x_scales.view(x.size(0), -1)
    return x_q, x_scales


def _mxfp8_e4m3_quantize_impl(
    x: torch.Tensor,
    is_sf_swizzled_layout: bool = False,
    alignment: int = 0,
    use_8x4_sf_layout: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    from vllm.platforms import current_platform

    if current_platform.has_device_capability(100):
        return _flashinfer_mxfp8_quantize_impl(
            x,
            is_sf_swizzled_layout,
            alignment,
            use_8x4_sf_layout,
            require_exact_8x4_layout=False,
        )

    return _mxfp8_e4m3_quantize_torch(x, is_sf_swizzled_layout)


def mxfp8_e4m3_quantize(
    x: torch.Tensor,
    is_sf_swizzled_layout: bool = False,
    alignment: int = 0,
    use_8x4_sf_layout: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if use_8x4_sf_layout is None:
        use_8x4_sf_layout = mxfp8_dense_use_8x4_sf_layout(int(x.shape[-2]))
    return torch.ops.vllm.mxfp8_quantize(
        x, is_sf_swizzled_layout, alignment, use_8x4_sf_layout
    )


def _mxfp8_e4m3_quantize_fixed_layout_impl(
    x: torch.Tensor,
    *,
    use_8x4_sf_layout: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    from vllm.platforms import current_platform

    if not current_platform.has_device_capability(100):
        raise RuntimeError("Fixed-layout MXFP8 quantization requires >=sm_100")
    return _flashinfer_mxfp8_quantize_impl(
        x,
        is_sf_swizzled_layout=True,
        alignment=32,
        use_8x4_sf_layout=use_8x4_sf_layout,
        require_exact_8x4_layout=use_8x4_sf_layout,
    )


def mxfp8_e4m3_quantize_8x4_impl(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _mxfp8_e4m3_quantize_fixed_layout_impl(x, use_8x4_sf_layout=True)


def mxfp8_e4m3_quantize_128x4_impl(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _mxfp8_e4m3_quantize_fixed_layout_impl(x, use_8x4_sf_layout=False)


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
    use_8x4_sf_layout: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fake implementation for torch.compile tracing."""
    fp_data = torch.empty_like(x, dtype=MXFP8_VALUE_DTYPE)

    block_size = MXFP8_BLOCK_SIZE

    if x.ndim == 2:
        M, N = x.shape
        K = (N + block_size - 1) // block_size
        if is_sf_swizzled_layout:
            m_tile = 8 if use_8x4_sf_layout else 128
            M_padded = ((M + m_tile - 1) // m_tile) * m_tile
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
            m_tile = 8 if use_8x4_sf_layout else 128
            M_padded = ((M + m_tile - 1) // m_tile) * m_tile
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


def mxfp8_e4m3_quantize_8x4_fake(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return mxfp8_e4m3_quantize_fake(
        x,
        is_sf_swizzled_layout=True,
        alignment=32,
        use_8x4_sf_layout=True,
    )


def mxfp8_e4m3_quantize_128x4_fake(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return mxfp8_e4m3_quantize_fake(
        x,
        is_sf_swizzled_layout=True,
        alignment=32,
        use_8x4_sf_layout=False,
    )


direct_register_custom_op(
    op_name="mxfp8_quantize",
    op_func=_mxfp8_e4m3_quantize_impl,
    fake_impl=mxfp8_e4m3_quantize_fake,
)

direct_register_custom_op(
    op_name="mxfp8_quantize_8x4",
    op_func=mxfp8_e4m3_quantize_8x4_impl,
    fake_impl=mxfp8_e4m3_quantize_8x4_fake,
)

direct_register_custom_op(
    op_name="mxfp8_quantize_128x4",
    op_func=mxfp8_e4m3_quantize_128x4_impl,
    fake_impl=mxfp8_e4m3_quantize_128x4_fake,
)

def xpu_mxfp8_quantize(
    x: torch.Tensor, dtype: torch.dtype | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.vllm.xpu_mxfp8_quantize(x, dtype)
