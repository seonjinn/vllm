# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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
    mxfp8_trtllm_resolved_binding,
    mxfp8_trtllm_specialization_fingerprint,
    mxfp8_trtllm_use_8x4_sf_layout,
    prepare_mxfp8_trtllm_tactic_state,
    register_mxfp8_trtllm_trace_callback,
    swizzle_mxfp8_scale,
)
from vllm.platforms import current_platform
from vllm.utils import flashinfer as vllm_flashinfer
from vllm.utils.flashinfer import has_flashinfer, has_flashinfer_cutedsl

from .Mxfp8LinearKernel import Mxfp8LinearKernel, Mxfp8LinearLayerConfig

logger = init_logger(__name__)

_MXFP8_DENSE_TRACE_SEEN: set[tuple[object, ...]] = set()
_MXFP8_DENSE_TRACE_WRITTEN = 0
_MXFP8_DENSE_TRACE_WARNED = False
_MXFP8_CAPTURE_DEFAULT_BINDING = (-1, "capture_exact_miss")


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
    m_logical: int,
    m_physical: int,
    n_logical: int,
    n_physical: int,
    k_logical: int,
    k_physical: int,
    layout: str,
    output_dtype: str,
    tactic_source: str,
    selected_tactic: int,
    compilation_state: str,
    cuda_graph_state: str,
    runtime_provenance: dict[str, object],
) -> None:
    enabled = os.environ.get("VLLM_MXFP8_DENSE_SHAPE_TRACE", "")
    if enabled.strip().lower() in ("", "0", "false", "no", "off"):
        return
    if torch.compiler.is_compiling() or torch.cuda.is_current_stream_capturing():
        return
    trace_dir = os.environ.get("VLLM_MXFP8_DENSE_SHAPE_TRACE_DIR", "").strip()
    if not trace_dir:
        return

    workload = os.environ.get("VLLM_MXFP8_DENSE_TRACE_WORKLOAD", "").strip()
    raw_batch_size = os.environ.get("VLLM_MXFP8_DENSE_TRACE_BATCH_SIZE", "")
    serving_phase = os.environ.get("VLLM_MXFP8_DENSE_TRACE_SERVING_PHASE", "").strip()
    topology = runtime_provenance.get("topology")
    global _MXFP8_DENSE_TRACE_WARNED
    if (
        not workload
        or not raw_batch_size
        or not serving_phase
        or not isinstance(topology, str)
        or not topology
        or not runtime_provenance
    ):
        if not _MXFP8_DENSE_TRACE_WARNED:
            _MXFP8_DENSE_TRACE_WARNED = True
            logger.warning(
                "MXFP8 dense shape tracing requires workload, batch size, "
                "serving phase, and runtime provenance; skipping incomplete rows."
            )
        return
    try:
        batch_size = int(raw_batch_size)
    except ValueError:
        batch_size = 0
    if batch_size <= 0:
        if not _MXFP8_DENSE_TRACE_WARNED:
            _MXFP8_DENSE_TRACE_WARNED = True
            logger.warning(
                "MXFP8 dense shape trace batch size must be a positive integer."
            )
        return

    key = (
        prefix,
        family,
        m_logical,
        m_physical,
        n_logical,
        n_physical,
        k_logical,
        k_physical,
        layout,
        output_dtype,
        tactic_source,
        selected_tactic,
        compilation_state,
        cuda_graph_state,
        workload,
        batch_size,
        serving_phase,
    )
    max_records = int(os.environ.get("VLLM_MXFP8_DENSE_SHAPE_TRACE_MAX", "4096"))
    global _MXFP8_DENSE_TRACE_WRITTEN
    if key in _MXFP8_DENSE_TRACE_SEEN or max_records <= _MXFP8_DENSE_TRACE_WRITTEN:
        return
    _MXFP8_DENSE_TRACE_SEEN.add(key)
    _MXFP8_DENSE_TRACE_WRITTEN += 1

    output_dir = Path(trace_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    host = socket.gethostname()
    record = {
        "activation_scale_layout": layout,
        "batch_size": batch_size,
        "compilation_state": compilation_state,
        "cuda_graph_state": cuda_graph_state,
        "event": "mxfp8_dense_shape",
        "family": family,
        "host": host,
        "hostname": host,
        "k": int(k_logical),
        "k_logical": int(k_logical),
        "k_physical": int(k_physical),
        "layer_prefix": prefix,
        "layout": layout,
        "m": int(m_logical),
        "m_logical": int(m_logical),
        "m_physical": int(m_physical),
        "n_logical": int(n_logical),
        "n_physical": int(n_physical),
        "normalized_family": family,
        "output_dtype": output_dtype,
        "pid": os.getpid(),
        "prefix": prefix,
        "rank": int(os.environ.get("RANK", "0")),
        "runtime_provenance": runtime_provenance,
        "selected_tactic": int(selected_tactic),
        "serving_phase": serving_phase,
        "tactic_source": tactic_source,
        "topology": topology,
        "workload": workload,
    }
    output = output_dir / f"dense_shapes_{record['host']}_{record['pid']}.jsonl"
    with output.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")


register_mxfp8_trtllm_trace_callback(_trace_mxfp8_dense_shape)


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
        from flashinfer import shuffle_matrix_a, shuffle_matrix_sf_a

        weight = layer.weight.data  # [N, K]
        n, k = weight.shape
        if k % 256 != 0:
            raise ValueError(
                "TRTLLM MXFP8 dense weights require K to be divisible by 256, "
                f"got N={n}, K={k}."
            )

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
        prepare_mxfp8_trtllm_tactic_state(layer.weight.device)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.dtype != torch.bfloat16:
            raise ValueError(
                "FlashInfer TRTLLM MXFP8 dense GEMM requires BF16 activations, "
                f"got {x.dtype}."
            )
        weight = layer.weight  # shuffled [N_padded, K]
        weight_scale = layer.weight_scale
        _, k = weight.shape
        output_features = layer._mxfp8_trtllm_output_features

        input_shape = x.shape
        input_2d = x.view(-1, k)
        m = int(input_2d.shape[0])
        use_8x4_sf_layout = mxfp8_trtllm_use_8x4_sf_layout(m)
        is_capturing = torch.cuda.is_current_stream_capturing()
        if is_capturing:
            bindings = getattr(layer, "_mxfp8_trtllm_capture_bindings", None)
            tactic, tactic_source = (
                bindings.get(m, _MXFP8_CAPTURE_DEFAULT_BINDING)
                if bindings is not None
                else _MXFP8_CAPTURE_DEFAULT_BINDING
            )
            tactic_specialization_fingerprint = (
                mxfp8_trtllm_specialization_fingerprint(input_2d.device)
            )
        else:
            tactic = -2
            tactic_source = "unresolved_eager"
            tactic_specialization_fingerprint = ""
        output = mxfp8_trtllm_adaptive_linear(
            input_2d,
            weight,
            weight_scale,
            output_features,
            str(getattr(layer, "prefix", "unknown")),
            _mxfp8_dense_family(layer),
            tactic,
            tactic_source,
            tactic_specialization_fingerprint,
        )
        if not is_capturing and not torch.compiler.is_compiling():
            binding = mxfp8_trtllm_resolved_binding(
                input_2d,
                weight,
                output_features,
                use_8x4_sf_layout=use_8x4_sf_layout,
            )
            if binding is not None:
                bindings = getattr(layer, "_mxfp8_trtllm_capture_bindings", None)
                if bindings is None:
                    bindings = {}
                    layer._mxfp8_trtllm_capture_bindings = bindings
                bindings[m] = binding
        if bias is not None:
            output = output + bias

        output_shape = (*input_shape[:-1], output_features)
        return output.view(output_shape)
