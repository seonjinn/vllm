# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
import socket
from functools import cache
from pathlib import Path
from typing import Any, NamedTuple

import torch
from torch.nn.parameter import Parameter

from vllm import envs
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    MXFP8_BLOCK_SIZE,
    mxfp8_e4m3_quantize,
    swizzle_mxfp8_scale,
)
from vllm.platforms import current_platform
from vllm.utils import flashinfer as vllm_flashinfer
from vllm.utils.flashinfer import has_flashinfer, has_flashinfer_cutedsl
from vllm.utils.torch_utils import direct_register_custom_op

from .Mxfp8LinearKernel import Mxfp8LinearKernel, Mxfp8LinearLayerConfig

MXFP8_TRTLLM_LAYOUT_ENV = "VLLM_MXFP8_TRTLLM_LAYOUT"
MXFP8_TRTLLM_SWITCH_M_ENV = "VLLM_MXFP8_TRTLLM_SWITCH_M"
MXFP8_TRTLLM_TACTICS_ENV = "VLLM_MXFP8_TRTLLM_TACTICS"
_MXFP8_DENSE_TRACE_SEEN: set[tuple[str, int, int, int, int]] = set()


def _mxfp8_dense_family(layer: torch.nn.Module) -> str:
    prefix = str(getattr(layer, "prefix", "")).lower()
    if any(name in prefix for name in ("qkv_proj", "q_proj", "k_proj", "v_proj")):
        return "QKV"
    if any(name in prefix for name in ("o_proj", "out_proj", "attention.dense")):
        return "O"
    if any(name in prefix for name in ("gate_up_proj", "gate_proj", "up_proj")):
        return "FC1"
    if any(name in prefix for name in ("down_proj", "fc2", ".w2")):
        return "FC2"
    if "mamba" in prefix:
        return "MambaProjection"
    if any(name in prefix for name in ("mlp", "ffn", "expert")):
        return "MLPOrExpertDense"
    return "OtherDense"


def _trace_mxfp8_dense_shape(
    *,
    layer: torch.nn.Module,
    m: int,
    n_logical: int,
    n_physical: int,
    k: int,
) -> None:
    enabled = os.environ.get("VLLM_MXFP8_DENSE_SHAPE_TRACE", "").strip().lower()
    if enabled in ("", "0", "false", "no", "off"):
        return
    if torch.compiler.is_compiling() or torch.cuda.is_current_stream_capturing():
        return
    trace_dir = os.environ.get("VLLM_MXFP8_DENSE_SHAPE_TRACE_DIR", "").strip()
    if not trace_dir:
        return

    family = _mxfp8_dense_family(layer)
    prefix = str(getattr(layer, "prefix", "unknown"))
    shape = (family, int(m), int(n_logical), int(n_physical), int(k))
    max_records = int(os.environ.get("VLLM_MXFP8_DENSE_SHAPE_TRACE_MAX", "4096"))
    if shape in _MXFP8_DENSE_TRACE_SEEN or len(_MXFP8_DENSE_TRACE_SEEN) >= max_records:
        return
    _MXFP8_DENSE_TRACE_SEEN.add(shape)

    output_dir = Path(trace_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"dense_shapes_{socket.gethostname()}_{os.getpid()}.jsonl"
    record = {
        "event": "mxfp8_dense_shape",
        "family": family,
        "hostname": socket.gethostname(),
        "k": int(k),
        "m": int(m),
        "n_logical": int(n_logical),
        "n_physical": int(n_physical),
        "pid": os.getpid(),
        "prefix": prefix,
    }
    with output.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")


class _Mxfp8TrtllmLayoutConfig(NamedTuple):
    policy: str
    switch_m: int | None


@cache
def _mxfp8_trtllm_layout_config() -> _Mxfp8TrtllmLayoutConfig:
    policy = envs.VLLM_MXFP8_TRTLLM_LAYOUT.strip().lower()
    if policy != "adaptive":
        return _Mxfp8TrtllmLayoutConfig(policy, None)

    switch_m = envs.VLLM_MXFP8_TRTLLM_SWITCH_M
    if switch_m <= 0:
        raise ValueError(
            f"{MXFP8_TRTLLM_SWITCH_M_ENV} must be positive; got {switch_m}."
        )
    return _Mxfp8TrtllmLayoutConfig(policy, switch_m)


def mxfp8_trtllm_use_8x4_sf_layout(m: int) -> bool:
    config = _mxfp8_trtllm_layout_config()
    if config.policy == "8x4":
        return True
    if config.policy == "128x4":
        return False
    assert config.switch_m is not None
    return m <= config.switch_m


@cache
def _mxfp8_trtllm_tactics() -> dict[tuple[int, int, int], int]:
    tactics: dict[tuple[int, int, int], int] = {}
    for raw_entry in envs.VLLM_MXFP8_TRTLLM_TACTICS.split(";"):
        entry = raw_entry.strip()
        if not entry:
            continue
        shape_text, separator, tactic_text = entry.partition(":")
        try:
            shape = tuple(
                int(value) for value in shape_text.replace("x", ",").split(",")
            )
            tactic = int(tactic_text)
        except ValueError as exc:
            raise ValueError(
                f"Invalid {MXFP8_TRTLLM_TACTICS_ENV} entry: {entry!r}."
            ) from exc
        if separator != ":" or len(shape) != 3 or tactic < -1:
            raise ValueError(f"Invalid {MXFP8_TRTLLM_TACTICS_ENV} entry: {entry!r}.")
        key = (shape[0], shape[1], shape[2])
        if key in tactics:
            raise ValueError(
                f"Duplicate {MXFP8_TRTLLM_TACTICS_ENV} shape: {shape_text!r}."
            )
        tactics[key] = tactic
    return tactics


def mxfp8_trtllm_tactic(m: int, n: int, k: int) -> int | None:
    return _mxfp8_trtllm_tactics().get((m, n, k))


@cache
def _mxfp8_trtllm_runtime(
    device_type: str,
    device_index: int,
    use_8x4_sf_layout: bool,
) -> tuple[Any, torch.Tensor]:
    from flashinfer.gemm.gemm_base import (
        DEFAULT_WORKSPACE_SIZE,
        _get_cache_buf,
        get_trtllm_gemm_module,
    )

    device = torch.device(device_type, device_index)
    suffix = "8x4" if use_8x4_sf_layout else "128x4"
    workspace = _get_cache_buf(
        f"vllm_mxfp8_trtllm_tactic_workspace_{suffix}",
        DEFAULT_WORKSPACE_SIZE,
        device,
    )
    runner = get_trtllm_gemm_module().trtllm_mxfp8_gemm_runner(
        use_8x4_sf_layout=use_8x4_sf_layout
    )
    return runner, workspace


def _mxfp8_trtllm_tactic_linear_impl(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    output_features: int,
    use_8x4_sf_layout: bool,
    tactic: int,
) -> torch.Tensor:
    quantize = (
        vllm_flashinfer.flashinfer_mxfp8_quantize_8x4
        if use_8x4_sf_layout
        else vllm_flashinfer.flashinfer_mxfp8_quantize_128x4
    )
    input_mxfp8, input_scale = quantize(x)
    physical_output_features = int(weight.shape[0])
    output = torch.empty(
        (x.shape[0], physical_output_features),
        dtype=x.dtype,
        device=x.device,
    )
    runner, workspace = _mxfp8_trtllm_runtime(
        x.device.type,
        x.device.index
        if x.device.index is not None
        else torch.accelerator.current_device_index(),
        use_8x4_sf_layout,
    )
    result = runner.forward(
        [
            input_mxfp8,
            weight.t(),
            input_scale,
            weight_scale,
            x.dtype,
            output,
            workspace,
        ],
        tactic=tactic,
    )
    return result[:, :output_features].contiguous()


def _mxfp8_trtllm_linear_fixed_impl(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    output_features: int,
    *,
    use_8x4_sf_layout: bool,
) -> torch.Tensor:
    quantize = (
        vllm_flashinfer.flashinfer_mxfp8_quantize_8x4
        if use_8x4_sf_layout
        else vllm_flashinfer.flashinfer_mxfp8_quantize_128x4
    )
    input_mxfp8, input_scale = quantize(x)
    output = vllm_flashinfer.mm_mxfp8(
        input_mxfp8,
        weight.t(),
        input_scale,
        weight_scale,
        out_dtype=x.dtype,
        backend="trtllm",
        use_8x4_sf_layout=use_8x4_sf_layout,
    )
    return output[:, :output_features].contiguous()


def _mxfp8_trtllm_dispatch_linear_impl(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    output_features: int,
) -> torch.Tensor:
    config = _mxfp8_trtllm_layout_config()
    use_8x4_sf_layout = mxfp8_trtllm_use_8x4_sf_layout(int(x.shape[0]))
    physical_output_features = int(weight.shape[0])
    tactic = mxfp8_trtllm_tactic(
        int(x.shape[0]), physical_output_features, int(x.shape[1])
    )
    if tactic is not None:
        return _mxfp8_trtllm_tactic_linear_impl(
            x,
            weight,
            weight_scale,
            output_features,
            use_8x4_sf_layout,
            tactic,
        )
    return _mxfp8_trtllm_linear_fixed_impl(
        x,
        weight,
        weight_scale,
        output_features,
        use_8x4_sf_layout=(
            use_8x4_sf_layout if config.policy == "adaptive" else config.policy == "8x4"
        ),
    )


def mxfp8_trtllm_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    output_features: int,
) -> torch.Tensor:
    return torch.ops.vllm.mxfp8_trtllm_dispatch_linear(
        x, weight, weight_scale, output_features
    )


def _mxfp8_trtllm_linear_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    output_features: int,
) -> torch.Tensor:
    return torch.empty((x.shape[0], output_features), dtype=x.dtype, device=x.device)


def _mxfp8_trtllm_tactic_linear_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    output_features: int,
    use_8x4_sf_layout: bool,
    tactic: int,
) -> torch.Tensor:
    return torch.empty((x.shape[0], output_features), dtype=x.dtype, device=x.device)


direct_register_custom_op(
    op_name="mxfp8_trtllm_dispatch_linear",
    op_func=_mxfp8_trtllm_dispatch_linear_impl,
    fake_impl=_mxfp8_trtllm_linear_fake,
)
direct_register_custom_op(
    op_name="mxfp8_trtllm_tactic_linear",
    op_func=_mxfp8_trtllm_tactic_linear_impl,
    fake_impl=_mxfp8_trtllm_tactic_linear_fake,
)


class FlashInferCutlassMxfp8LinearKernel(Mxfp8LinearKernel):
    """MXFP8 W8A8 GEMM via FlashInfer CUTLASS (SM100+)."""

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if current_platform.has_device_capability(100):
            return True, None
        return False, "requires >=sm_100 (Blackwell)"

    @classmethod
    def can_implement(cls, c: Mxfp8LinearLayerConfig) -> tuple[bool, str | None]:
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight = layer.weight.data  # [N, K]
        N, K = weight.shape

        scale_k = K // MXFP8_BLOCK_SIZE
        weight_scale_2d = layer.weight_scale.data[:N, :scale_k].contiguous()
        weight_scale_swizzled = swizzle_mxfp8_scale(weight_scale_2d, M=N, K=K)

        layer.weight = Parameter(weight.contiguous(), requires_grad=False)
        layer.weight_scale = Parameter(
            weight_scale_swizzled.contiguous(), requires_grad=False
        )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        weight = layer.weight
        weight_scale = layer.weight_scale
        out_dtype = x.dtype
        N, K = weight.shape

        input_shape = x.shape
        input_2d = x.view(-1, K)
        min_dim = 128

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

        input_mxfp8, input_scale = mxfp8_e4m3_quantize(
            input_2d, is_sf_swizzled_layout=True
        )

        if not weight.is_contiguous():
            weight = weight.contiguous()

        output = vllm_flashinfer.mm_mxfp8(
            input_mxfp8,
            weight.t(),
            input_scale,
            weight_scale,
            out_dtype=out_dtype,
            backend="cutlass",
        )

        if bias is not None:
            output = output + bias

        output_shape = (*input_shape[:-1], N)
        return output.view(output_shape)


class FlashInferCutedslMxfp8LinearKernel(Mxfp8LinearKernel):
    """MXFP8 W8A8 GEMM via FlashInfer CuTe-DSL (SM100/SM103)."""

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not (
            current_platform.is_cuda()
            and current_platform.is_device_capability_family(100)
        ):
            return False, "requires sm_100/sm_103 (Blackwell)"
        if not has_flashinfer_cutedsl():
            return False, "requires FlashInfer CuTe-DSL module"
        return True, None

    @classmethod
    def can_implement(cls, c: Mxfp8LinearLayerConfig) -> tuple[bool, str | None]:
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight = layer.weight.data  # [N, K]
        N, K = weight.shape

        scale_k = K // MXFP8_BLOCK_SIZE
        weight_scale_2d = layer.weight_scale.data[:N, :scale_k].contiguous()
        weight_scale_swizzled = swizzle_mxfp8_scale(weight_scale_2d, M=N, K=K)

        # Store weight column-major [K, N] as mm_mxfp8 expects for operand B.
        layer.weight = Parameter(weight.contiguous().t(), requires_grad=False)
        layer.weight_scale = Parameter(
            weight_scale_swizzled.contiguous(), requires_grad=False
        )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        weight = layer.weight  # [K, N], column-major
        weight_scale = layer.weight_scale
        out_dtype = x.dtype
        K, N = weight.shape

        input_shape = x.shape
        input_2d = x.view(-1, K)
        min_dim = 128

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

        input_mxfp8, input_scale = mxfp8_e4m3_quantize(
            input_2d, is_sf_swizzled_layout=True
        )

        output = vllm_flashinfer.mm_mxfp8(
            input_mxfp8,
            weight,
            input_scale,
            weight_scale,
            out_dtype=out_dtype,
            backend="cute-dsl",
        )

        if bias is not None:
            output = output + bias

        output_shape = (*input_shape[:-1], N)
        return output.view(output_shape)


class FlashInferTrtllmMxfp8LinearKernel(Mxfp8LinearKernel):
    """MXFP8 W8A8 GEMM via FlashInfer's TensorRT-LLM wrapper."""

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not (
            current_platform.is_cuda()
            and current_platform.is_device_capability_family(100)
        ):
            return False, "requires SM100-family GPU"
        if not has_flashinfer():
            return False, "requires FlashInfer"
        return True, None

    @classmethod
    def can_implement(cls, c: Mxfp8LinearLayerConfig) -> tuple[bool, str | None]:
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        from flashinfer import shuffle_matrix_a, shuffle_matrix_sf_a

        if hasattr(layer, "_mxfp8_trtllm_output_size") and layer.weight_scale.ndim == 1:
            return

        weight = layer.weight.data  # [N, K]
        N, K = weight.shape
        if K % 256 != 0:
            raise ValueError(
                f"FlashInfer TRTLLM MXFP8 requires K to be divisible by 256, got K={K}."
            )

        scale_k = K // MXFP8_BLOCK_SIZE
        weight_scale = layer.weight_scale.data[:N, :scale_k].contiguous()
        padded_n = ((N + 127) // 128) * 128
        if padded_n != N:
            padded_weight = weight.new_zeros((padded_n, K))
            padded_weight[:N] = weight
            weight = padded_weight

            padded_scale = weight_scale.new_zeros((padded_n, scale_k))
            padded_scale[:N] = weight_scale
            weight_scale = padded_scale
        else:
            weight = weight.contiguous()

        layer.weight = Parameter(
            shuffle_matrix_a(weight, 128).reshape(padded_n, K),
            requires_grad=False,
        )
        layer.weight_scale = Parameter(
            shuffle_matrix_sf_a(
                weight_scale,
                128,
                num_elts_per_sf=MXFP8_BLOCK_SIZE,
            ).reshape(-1),
            requires_grad=False,
        )
        layer._mxfp8_trtllm_output_size = N

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert x.dtype == torch.bfloat16, (
            f"FlashInfer TRTLLM MXFP8 requires bfloat16 activations, got {x.dtype}."
        )

        weight = layer.weight  # shuffled [padded N, K]
        weight_scale = layer.weight_scale
        _, K = weight.shape
        output_size = layer._mxfp8_trtllm_output_size
        input_shape = x.shape
        input_2d = x.view(-1, K)

        _trace_mxfp8_dense_shape(
            layer=layer,
            m=input_2d.shape[0],
            n_logical=output_size,
            n_physical=weight.shape[0],
            k=K,
        )

        output = mxfp8_trtllm_linear(
            input_2d,
            weight,
            weight_scale,
            output_size,
        )

        if bias is not None:
            output = output + bias

        output_shape = (*input_shape[:-1], output_size)
        return output.view(output_shape)
