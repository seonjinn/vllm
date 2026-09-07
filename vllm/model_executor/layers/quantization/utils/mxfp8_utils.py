# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from enum import Enum
import os

import torch

from vllm.logger import init_logger
from vllm.utils import flashinfer as vllm_flashinfer
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)


class Mxfp8LinearBackend(Enum):
    EMULATION = "emulation"
    FLASHINFER_CUTLASS = "flashinfer-cutlass"


# MXFP8 constants
MXFP8_VALUE_DTYPE = torch.float8_e4m3fn
MXFP8_SCALE_DTYPE = torch.uint8
MXFP8_BLOCK_SIZE = 32
MXFP8_INPUT_QUANT_KEY = "mxfp8_e4m3_block_scale"


_SUPPORTED_MXFP8_DENSE_BACKENDS = ("cutlass", "trtllm", "auto")


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off", "")


def _mxfp8_dense_backend() -> str:
    backend = os.environ.get("VLLM_MXFP8_DENSE_GEMM_BACKEND", "cutlass")
    backend = backend.strip().lower()
    if backend not in _SUPPORTED_MXFP8_DENSE_BACKENDS:
        raise ValueError(
            "VLLM_MXFP8_DENSE_GEMM_BACKEND must be one of "
            f"{_SUPPORTED_MXFP8_DENSE_BACKENDS}, got {backend!r}"
        )
    return backend


def _mxfp8_use_8x4_sf_layout() -> bool:
    raw = os.environ.get("VLLM_MXFP8_DENSE_A_SF_LAYOUT", "128x4")
    return raw.strip().lower() in ("8x4", "layout_8x4", "true", "1")


@dataclass(frozen=True)
class Mxfp8QuantizedActivation:
    """Pre-quantized MXFP8 activation handoff for ModelOpt MXFP8 linears.

    This is a narrow scaffold for the #42469/#42597 style QuantizedActivation
    contract. The producer owns activation quantization and scale layout; the
    linear consumes the result without re-running mxfp8_e4m3_quantize.
    """

    value: torch.Tensor
    scale: torch.Tensor
    orig_shape: tuple[int, ...]
    sf_layout: str = "128x4"
    input_quant_key: str = MXFP8_INPUT_QUANT_KEY


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


def _mxfp8_e4m3_quantize_impl(
    x: torch.Tensor, is_sf_swizzled_layout: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    from flashinfer import mxfp8_quantize as flashinfer_mxfp8_quantize

    kwargs = {
        "input": x,
        "is_sf_swizzled_layout": is_sf_swizzled_layout,
    }
    use_8x4 = is_sf_swizzled_layout and _mxfp8_use_8x4_sf_layout()
    if use_8x4:
        from flashinfer import SfLayout

        kwargs["sf_swizzle_layout"] = SfLayout.layout_8x4
        kwargs["backend"] = os.environ.get("VLLM_MXFP8_DENSE_QUANT_BACKEND", "cuda")

    try:
        x_q, x_scales = flashinfer_mxfp8_quantize(**kwargs)
    except TypeError:
        if use_8x4 and _env_flag("VLLM_MXFP8_DENSE_REQUIRE_8X4_QUANT"):
            raise
        x_q, x_scales = flashinfer_mxfp8_quantize(
            x, is_sf_swizzled_layout=is_sf_swizzled_layout
        )
    if x_scales.ndim == 1 and x.ndim == 2 and not is_sf_swizzled_layout:
        x_scales = x_scales.view(x.size(0), -1)
    return x_q, x_scales


def mxfp8_e4m3_quantize(
    x: torch.Tensor, is_sf_swizzled_layout: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.vllm.mxfp8_quantize(x, is_sf_swizzled_layout)


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
    x: torch.Tensor, is_sf_swizzled_layout: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fake implementation for torch.compile tracing."""
    fp_data = torch.empty_like(x, dtype=MXFP8_VALUE_DTYPE)

    block_size = MXFP8_BLOCK_SIZE

    if x.ndim == 2:
        M, N = x.shape
        K = (N + block_size - 1) // block_size
        if is_sf_swizzled_layout:
            m_tile = 8 if _mxfp8_use_8x4_sf_layout() else 128
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
            m_tile = 8 if _mxfp8_use_8x4_sf_layout() else 128
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


direct_register_custom_op(
    op_name="mxfp8_quantize",
    op_func=_mxfp8_e4m3_quantize_impl,
    fake_impl=mxfp8_e4m3_quantize_fake,
)


class Mxfp8LinearOp:
    def __init__(self, backend: Mxfp8LinearBackend):
        if backend not in Mxfp8LinearBackend:
            raise ValueError(f"Unsupported backend: {backend}")

        self.backend = backend

    def _apply_emulation(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        out_dtype: torch.dtype,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Validate weight_scale dtype and shape (must be 2D for TORCH backend)
        if weight_scale.dtype != MXFP8_SCALE_DTYPE:
            raise ValueError(
                f"TORCH backend requires {MXFP8_SCALE_DTYPE} weight_scale dtype, "
                f"got {weight_scale.dtype}."
            )
        if weight_scale.ndim != 2:
            raise ValueError(
                f"TORCH backend requires 2D weight_scale, got {weight_scale.ndim}D. "
                f"Ensure process_weights_after_loading was called."
            )

        weight_bf16 = dequant_mxfp8_to_bf16(weight, weight_scale)

        output = torch.nn.functional.linear(input, weight_bf16, bias)
        return output.to(out_dtype)

    def _apply_flashinfer_cutlass(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        out_dtype: torch.dtype,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        N, K = weight.shape

        input_shape = input.shape
        input_2d = input.view(-1, K)
        M_orig = input_2d.shape[0]

        # Minimum dimension size for the original F8_128x4 CUTLASS path.
        min_dim = 128
        backend = _mxfp8_dense_backend()

        assert min_dim <= K, (
            f"mm_mxfp8 requires K >= {min_dim}, got K={K}. "
            f"in_features is too small for mm_mxfp8."
        )
        assert K % MXFP8_BLOCK_SIZE == 0, (
            f"mm_mxfp8 requires K to be divisible by {MXFP8_BLOCK_SIZE}, got K={K}."
        )
        assert min_dim <= N, (
            f"mm_mxfp8 requires N >= {min_dim}, got N={N}. "
            f"out_features is too small for mm_mxfp8."
        )

        pad_to_128 = _env_flag("VLLM_MXFP8_DENSE_PAD_TO_128", backend != "trtllm")
        if backend in ("cutlass", "auto"):
            pad_to_128 = True

        if pad_to_128:
            M_padded = ((M_orig + min_dim - 1) // min_dim) * min_dim
        else:
            M_padded = M_orig
        pad_rows = M_padded - M_orig
        if pad_rows > 0:
            input_2d = torch.nn.functional.pad(input_2d, (0, 0, 0, pad_rows))

        input_mxfp8, input_scale = mxfp8_e4m3_quantize(
            input_2d,
            is_sf_swizzled_layout=True,  # Swizzled for best accuracy
        )

        if not weight.is_contiguous():
            weight = weight.contiguous()

        output = vllm_flashinfer.mm_mxfp8(
            input_mxfp8,
            weight.t(),
            input_scale,
            weight_scale,
            out_dtype=out_dtype,
            backend=backend,
        )

        if pad_rows > 0:
            output = output[:M_orig, :]

        if bias is not None:
            output = output + bias

        output_shape = (*input_shape[:-1], N)
        return output.view(output_shape)

    def apply_quantized(
        self,
        input: Mxfp8QuantizedActivation,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        out_dtype: torch.dtype,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if input.input_quant_key != MXFP8_INPUT_QUANT_KEY:
            raise ValueError(
                "Unsupported MXFP8 activation quant key: "
                f"{input.input_quant_key!r}"
            )
        if input.value.dtype != MXFP8_VALUE_DTYPE:
            raise ValueError(
                f"MXFP8 activation must be {MXFP8_VALUE_DTYPE}, "
                f"got {input.value.dtype}."
            )
        if input.scale.dtype != MXFP8_SCALE_DTYPE:
            raise ValueError(
                f"MXFP8 activation scale must be {MXFP8_SCALE_DTYPE}, "
                f"got {input.scale.dtype}."
            )

        N, K = weight.shape
        input_2d = input.value.view(-1, K)
        backend = _mxfp8_dense_backend()
        output = vllm_flashinfer.mm_mxfp8(
            input_2d,
            weight.t(),
            input.scale,
            weight_scale,
            out_dtype=out_dtype,
            backend=backend,
        )

        m_orig = 1
        for dim in input.orig_shape[:-1]:
            m_orig *= dim
        if output.shape[0] != m_orig:
            output = output[:m_orig, :]

        if bias is not None:
            output = output + bias

        return output.view(*input.orig_shape[:-1], N)

    def apply(
        self,
        input: torch.Tensor | Mxfp8QuantizedActivation,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        out_dtype: torch.dtype,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if isinstance(input, Mxfp8QuantizedActivation):
            return self.apply_quantized(input, weight, weight_scale, out_dtype, bias)

        if self.backend == Mxfp8LinearBackend.EMULATION:
            return self._apply_emulation(input, weight, weight_scale, out_dtype, bias)

        assert self.backend == Mxfp8LinearBackend.FLASHINFER_CUTLASS
        return self._apply_flashinfer_cutlass(
            input, weight, weight_scale, out_dtype, bias
        )
