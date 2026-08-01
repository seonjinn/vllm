# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import base64
import json
import os
import socket
from pathlib import Path

import torch
from torch.nn.parameter import Parameter

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    MXFP8_BLOCK_SIZE,
    configure_mxfp8_trtllm_adaptive_compilation,
    mxfp8_e4m3_quantize,
    mxfp8_trtllm_adaptive_linear,
    mxfp8_trtllm_use_8x4_sf_layout,
    prepare_mxfp8_trtllm_exact_tactic_state,
    prepare_mxfp8_trtllm_high_m_tactic_state,
    swizzle_mxfp8_scale,
)
from vllm.platforms import current_platform
from vllm.utils import flashinfer as vllm_flashinfer
from vllm.utils.flashinfer import has_flashinfer, has_flashinfer_cutedsl

from .Mxfp8LinearKernel import Mxfp8LinearKernel, Mxfp8LinearLayerConfig

_MXFP8_DENSE_TRACE_SEEN: set[tuple[int, int, int, int, str]] = set()
_MXFP8_DENSE_TRACE_WRITTEN = 0
logger = init_logger(__name__)

_MXFP8_TRTLLM_LAYER_ALLOWLIST_ENV = "VLLM_MXFP8_DENSE_TRTLLM_LAYER_ALLOWLIST"
_MXFP8_TRTLLM_LAYER_ALLOWLIST_B64_ENV = "VLLM_MXFP8_DENSE_TRTLLM_LAYER_ALLOWLIST_B64"


def _parse_mxfp8_trtllm_layer_allowlist(value: str) -> set[tuple[int, int]]:
    entries: set[tuple[int, int]] = set()
    for raw_entry in value.split(";"):
        entry = raw_entry.strip()
        if not entry:
            continue
        fields = [field.strip() for field in entry.split(",")]
        if len(fields) != 2:
            raise ValueError(
                f"{_MXFP8_TRTLLM_LAYER_ALLOWLIST_ENV} entries must use N,K"
            )
        try:
            n, k = (int(field) for field in fields)
        except ValueError as error:
            raise ValueError(
                f"{_MXFP8_TRTLLM_LAYER_ALLOWLIST_ENV} entries must use N,K"
            ) from error
        if n <= 0 or k <= 0:
            raise ValueError(
                f"{_MXFP8_TRTLLM_LAYER_ALLOWLIST_ENV} entries must be positive N,K"
            )
        entries.add((n, k))
    return entries


def _mxfp8_trtllm_layer_is_qualified(n: int, k: int) -> bool:
    value = os.environ.get(_MXFP8_TRTLLM_LAYER_ALLOWLIST_ENV)
    encoded_value = os.environ.get(_MXFP8_TRTLLM_LAYER_ALLOWLIST_B64_ENV)
    if value is not None and encoded_value is not None:
        raise ValueError(
            f"Set only one of {_MXFP8_TRTLLM_LAYER_ALLOWLIST_ENV} and "
            f"{_MXFP8_TRTLLM_LAYER_ALLOWLIST_B64_ENV}"
        )
    if encoded_value is not None:
        try:
            value = base64.b64decode(encoded_value, validate=True).decode("ascii")
        except (ValueError, UnicodeDecodeError) as error:
            raise ValueError(
                f"{_MXFP8_TRTLLM_LAYER_ALLOWLIST_B64_ENV} must contain "
                "base64-encoded ASCII N,K pairs"
            ) from error
    if value is None:
        return True
    return (n, k) in _parse_mxfp8_trtllm_layer_allowlist(value)


def _mxfp8_dense_family(layer: torch.nn.Module) -> str:
    prefix = str(getattr(layer, "prefix", "")).lower()
    if any(token in prefix for token in ("qkv_proj", "query_key_value")):
        return "QKV"
    if any(token in prefix for token in ("q_proj", "k_proj", "v_proj")):
        return "QKV"
    if any(token in prefix for token in ("o_proj", "out_proj", "attention.dense")):
        return "O"
    if any(token in prefix for token in ("gate_up_proj", "gate_proj", "up_proj")):
        return "FC1"
    if any(token in prefix for token in ("fc1", ".w1", ".w3")):
        return "FC1"
    if any(token in prefix for token in ("down_proj", "fc2", ".w2")):
        return "FC2"
    if "mamba" in prefix and any(
        token in prefix
        for token in ("proj", "linear", "in_proj", "out_proj", "x_proj", "dt_proj")
    ):
        return "MambaProjection"
    if any(token in prefix for token in ("mlp", "ffn", "expert")):
        return "MLPOrExpertDense"
    return "OtherDense"


def _trace_mxfp8_dense_shape(
    *,
    prefix: str,
    family: str,
    m: int,
    n_logical: int,
    n_physical: int,
    k: int,
    layout: str,
) -> None:
    enabled = os.environ.get("VLLM_MXFP8_DENSE_SHAPE_TRACE", "")
    if enabled.strip().lower() in ("", "0", "false", "no", "off"):
        return
    if torch.compiler.is_compiling() or torch.cuda.is_current_stream_capturing():
        return
    trace_dir = os.environ.get("VLLM_MXFP8_DENSE_SHAPE_TRACE_DIR", "").strip()
    if not trace_dir:
        return

    key = (m, n_logical, n_physical, k, layout)
    max_records = int(os.environ.get("VLLM_MXFP8_DENSE_SHAPE_TRACE_MAX", "4096"))
    global _MXFP8_DENSE_TRACE_WRITTEN
    if key in _MXFP8_DENSE_TRACE_SEEN or max_records <= _MXFP8_DENSE_TRACE_WRITTEN:
        return
    _MXFP8_DENSE_TRACE_SEEN.add(key)
    _MXFP8_DENSE_TRACE_WRITTEN += 1

    output_dir = Path(trace_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "event": "mxfp8_dense_shape",
        "family": family,
        "hostname": socket.gethostname(),
        "k": int(k),
        "layout": layout,
        "m": int(m),
        "n_logical": int(n_logical),
        "n_physical": int(n_physical),
        "pid": os.getpid(),
        "prefix": prefix,
    }
    output = output_dir / f"dense_shapes_{record['hostname']}_{record['pid']}.jsonl"
    with output.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")


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
    """MXFP8 W8A8 GEMM via FlashInfer's TensorRT-LLM runner."""

    def __init__(self, c: Mxfp8LinearLayerConfig) -> None:
        super().__init__(c)
        configure_mxfp8_trtllm_adaptive_compilation()
        self._cutedsl_fallback: FlashInferCutedslMxfp8LinearKernel | None = None

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not has_flashinfer():
            return False, "requires FlashInfer"
        if current_platform.is_device_capability(
            100
        ) or current_platform.is_device_capability(103):
            return True, None
        return False, "requires sm_100 or sm_103 (Blackwell)"

    @classmethod
    def can_implement(cls, c: Mxfp8LinearLayerConfig) -> tuple[bool, str | None]:
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight = layer.weight.data  # [N, K]
        n, k = weight.shape
        if not _mxfp8_trtllm_layer_is_qualified(n, k):
            self._prepare_cutedsl_fallback(layer, n, k, "not allowlisted")
            return
        if k % 256 != 0:
            allow_fallback = os.environ.get(
                "VLLM_MXFP8_DENSE_TRTLLM_ALLOW_CUTEDSL_FALLBACK", ""
            ).strip().lower() in {"1", "true", "yes", "on"}
            if allow_fallback:
                self._prepare_cutedsl_fallback(
                    layer,
                    n,
                    k,
                    "TRTLLM K is not divisible by 256",
                )
                return
            raise ValueError(
                "TRTLLM MXFP8 dense weights require K to be divisible by 256, "
                f"got N={n}, K={k}."
            )

        from flashinfer import shuffle_matrix_a, shuffle_matrix_sf_a

        scale_k = k // MXFP8_BLOCK_SIZE
        weight_scale = layer.weight_scale.data[:n, :scale_k].contiguous()
        n_padded = (n + 127) // 128 * 128

        if n_padded != n:
            padded_weight = torch.zeros(
                (n_padded, k), dtype=weight.dtype, device=weight.device
            )
            padded_weight[:n].copy_(weight)
            padded_scale = torch.zeros(
                (n_padded, scale_k),
                dtype=weight_scale.dtype,
                device=weight_scale.device,
            )
            padded_scale[:n].copy_(weight_scale)
        else:
            padded_weight = weight.contiguous()
            padded_scale = weight_scale

        shuffled_weight = shuffle_matrix_a(padded_weight, 128).reshape(n_padded, k)
        shuffled_scale = shuffle_matrix_sf_a(
            padded_scale,
            128,
            num_elts_per_sf=MXFP8_BLOCK_SIZE,
        ).reshape(-1)

        layer.weight = Parameter(shuffled_weight.contiguous(), requires_grad=False)
        layer.weight_scale = Parameter(shuffled_scale.contiguous(), requires_grad=False)
        layer._mxfp8_trtllm_output_features = n
        layer._mxfp8_dense_backend = "trtllm"
        prepare_mxfp8_trtllm_exact_tactic_state(layer.weight.device)
        prepare_mxfp8_trtllm_high_m_tactic_state(layer.weight.device)

    def _prepare_cutedsl_fallback(
        self,
        layer: torch.nn.Module,
        n: int,
        k: int,
        reason: str,
    ) -> None:
        supported, failure_reason = FlashInferCutedslMxfp8LinearKernel.is_supported()
        if not supported:
            raise ValueError(
                f"{reason} and CuTeDSL fallback is unavailable: {failure_reason}"
            )
        if n < 128 or k < 128 or k % MXFP8_BLOCK_SIZE:
            raise ValueError(
                f"{reason} and the layer does not satisfy CuTeDSL constraints: "
                f"N={n}, K={k}."
            )
        if self._cutedsl_fallback is None:
            self._cutedsl_fallback = FlashInferCutedslMxfp8LinearKernel(self.config)
        self._cutedsl_fallback.process_weights_after_loading(layer)
        layer._mxfp8_dense_backend = "cute-dsl"
        logger.info(
            "Using CuTeDSL MXFP8 dense fallback for %s (N=%d, K=%d): %s",
            getattr(layer, "prefix", "unknown"),
            n,
            k,
            reason,
        )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if getattr(layer, "_mxfp8_dense_backend", "trtllm") == "cute-dsl":
            if self._cutedsl_fallback is None:
                raise RuntimeError("CuTeDSL fallback was not initialized")
            return self._cutedsl_fallback.apply_weights(layer, x, bias)
        if x.dtype != torch.bfloat16:
            raise ValueError(
                "FlashInfer TRTLLM MXFP8 dense GEMM requires BF16 activations, "
                f"got {x.dtype}."
            )
        weight = layer.weight  # shuffled [N_padded, K]
        weight_scale = layer.weight_scale
        n_padded, k = weight.shape
        output_features = layer._mxfp8_trtllm_output_features

        input_shape = x.shape
        input_2d = x.view(-1, k)
        if (
            not torch.compiler.is_compiling()
            and not torch.cuda.is_current_stream_capturing()
        ):
            use_8x4 = mxfp8_trtllm_use_8x4_sf_layout(int(input_2d.shape[0]))
            _trace_mxfp8_dense_shape(
                prefix=str(getattr(layer, "prefix", "unknown")),
                family=_mxfp8_dense_family(layer),
                m=int(input_2d.shape[0]),
                n_logical=int(output_features),
                n_physical=int(n_padded),
                k=int(k),
                layout="8x4" if use_8x4 else "128x4",
            )
        output = mxfp8_trtllm_adaptive_linear(
            input_2d,
            weight,
            weight_scale,
            output_features,
        )
        if bias is not None:
            output = output + bias

        output_shape = (*input_shape[:-1], output_features)
        return output.view(output_shape)
