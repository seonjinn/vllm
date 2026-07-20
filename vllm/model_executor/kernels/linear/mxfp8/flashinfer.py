# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import functools
import importlib
import importlib.metadata
import json
import os
import socket
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import torch
from torch.nn.parameter import Parameter

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    MXFP8_BLOCK_SIZE,
    configure_mxfp8_trtllm_adaptive_compilation,
    mxfp8_e4m3_quantize,
    mxfp8_trtllm_adaptive_linear,
    prepare_mxfp8_trtllm_high_m_tactic_state,
    swizzle_mxfp8_scale,
)
from vllm.platforms import current_platform
from vllm.utils import flashinfer as vllm_flashinfer
from vllm.utils.flashinfer import has_flashinfer, has_flashinfer_cutedsl

if TYPE_CHECKING:
    from vllm.v1.worker.workspace import WorkspaceManager

from .Mxfp8LinearKernel import Mxfp8LinearKernel, Mxfp8LinearLayerConfig

_MXFP8_DENSE_TRACE_SEEN: set[tuple[str, str, int, int, int, int, str]] = set()
_MXFP8_DENSE_TRACE_WRITTEN = 0
logger = init_logger(__name__)
_DYNAMIC_A_ARTIFACT_ENV = "VLLM_MXFP8_CUTEDSL_DYNAMIC_A_ARTIFACT"
_DYNAMIC_A_TELEMETRY_ENV = "VLLM_MXFP8_CUTEDSL_DYNAMIC_A_TELEMETRY"
_DYNAMIC_A_CUTLASS_VERSION_ENV = "VLLM_MXFP8_CUTEDSL_DYNAMIC_A_CUTLASS_VERSION"
_DYNAMIC_A_RUNTIME_FIELDS = frozenset(
    {
        "gpu_sm",
        "flashinfer_revision",
        "cuda_version",
        "cutlass_version",
        "activation_dtype",
        "output_dtype",
        "activation_scale_layout",
        "weight_scale_layout",
    }
)


def _env_flag_enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() not in (
        "",
        "0",
        "false",
        "no",
        "off",
    )


@dataclass(frozen=True)
class _DynamicAShapeConfig:
    tactic: int
    workspace_bytes: int
    status: str


class Mxfp8CutedslDynamicAPolicy:
    def __init__(
        self,
        shapes: Mapping[tuple[int, int, int], _DynamicAShapeConfig],
        *,
        runtime_compatible: bool,
    ) -> None:
        self._shapes = MappingProxyType(dict(shapes))
        self._runtime_compatible = runtime_compatible
        self._counters = {
            "dynamic_hits": 0,
            "stock_misses": 0,
            "incompatibility_fallbacks": 0,
        }

    @property
    def shapes(self) -> Mapping[tuple[int, int, int], _DynamicAShapeConfig]:
        return self._shapes

    @property
    def runtime_compatible(self) -> bool:
        return self._runtime_compatible

    def select(self, shape: tuple[int, int, int]) -> str:
        config = self._shapes.get(shape)
        if not self._runtime_compatible or (
            config is not None and config.status != "qualified"
        ):
            self._counters["incompatibility_fallbacks"] += 1
            return "stock_cutedsl"
        if config is None:
            self._counters["stock_misses"] += 1
            return "stock_cutedsl"
        self._counters["dynamic_hits"] += 1
        return "dynamic_cutlass_8x4"

    def dispatch_counters(self) -> dict[str, int]:
        return dict(self._counters)


def _parse_physical_shape(raw_shape: str) -> tuple[int, int, int]:
    try:
        values = tuple(int(value) for value in raw_shape.split(","))
    except ValueError as exc:
        raise ValueError(
            f"Invalid physical MXFP8 shape {raw_shape!r}; expected M,N,K"
        ) from exc
    if (
        len(values) != 3
        or any(value <= 0 for value in values)
        or raw_shape != ",".join(str(value) for value in values)
    ):
        raise ValueError(f"Invalid physical MXFP8 shape {raw_shape!r}; expected M,N,K")
    return values


def _require_json_object(value: object, context: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{context} must be a JSON object with string keys")
    return value


def load_mxfp8_cutedsl_dynamic_a_policy(
    path: str,
    *,
    runtime_metadata: Mapping[str, str],
) -> Mxfp8CutedslDynamicAPolicy:
    artifact_path = str(Path(path).expanduser().resolve(strict=True))
    runtime_items = tuple(sorted(runtime_metadata.items()))
    return _load_mxfp8_cutedsl_dynamic_a_policy(artifact_path, runtime_items)


@functools.cache
def _load_mxfp8_cutedsl_dynamic_a_policy(
    artifact_path: str,
    runtime_items: tuple[tuple[str, str], ...],
) -> Mxfp8CutedslDynamicAPolicy:
    runtime_metadata = dict(runtime_items)
    with Path(artifact_path).open(encoding="utf-8") as stream:
        artifact = _require_json_object(json.load(stream), "MXFP8 artifact")

    if set(artifact) != {"schema_version", "runtime", "shapes"}:
        raise ValueError(
            "MXFP8 artifact must contain exactly schema_version, runtime, and shapes"
        )
    if artifact["schema_version"] != 1:
        raise ValueError(
            f"Unsupported MXFP8 artifact schema {artifact['schema_version']!r}"
        )

    artifact_runtime = _require_json_object(
        artifact["runtime"], "MXFP8 artifact runtime"
    )
    if set(artifact_runtime) != _DYNAMIC_A_RUNTIME_FIELDS or not all(
        isinstance(value, str) and value for value in artifact_runtime.values()
    ):
        raise ValueError(
            "MXFP8 artifact runtime metadata is incomplete or contains "
            "unsupported fields"
        )
    if set(runtime_metadata) != _DYNAMIC_A_RUNTIME_FIELDS:
        raise ValueError("MXFP8 runtime metadata fields do not match schema version 1")

    raw_shapes = _require_json_object(artifact["shapes"], "MXFP8 artifact shapes")
    shapes: dict[tuple[int, int, int], _DynamicAShapeConfig] = {}
    for raw_shape, raw_config in raw_shapes.items():
        shape = _parse_physical_shape(raw_shape)
        config = _require_json_object(raw_config, f"MXFP8 artifact shape {raw_shape}")
        status = config.get("status")
        if status == "qualified":
            if set(config) != {"status", "tactic", "workspace_bytes"}:
                raise ValueError(
                    f"Qualified MXFP8 shape {raw_shape} has invalid fields"
                )
            tactic = config["tactic"]
            workspace_bytes = config["workspace_bytes"]
            if (
                isinstance(tactic, bool)
                or not isinstance(tactic, int)
                or tactic < -1
                or isinstance(workspace_bytes, bool)
                or not isinstance(workspace_bytes, int)
                or workspace_bytes < 0
            ):
                raise ValueError(
                    f"Qualified MXFP8 shape {raw_shape} has invalid resources"
                )
        elif status == "rejected":
            if set(config) != {"status"}:
                raise ValueError(f"Rejected MXFP8 shape {raw_shape} has invalid fields")
            tactic = -1
            workspace_bytes = 0
        else:
            raise ValueError(
                f"MXFP8 shape {raw_shape} has unsupported status {status!r}"
            )
        shapes[shape] = _DynamicAShapeConfig(
            tactic=tactic,
            workspace_bytes=workspace_bytes,
            status=status,
        )

    runtime_compatible = dict(artifact_runtime) == dict(runtime_metadata)
    return Mxfp8CutedslDynamicAPolicy(
        shapes,
        runtime_compatible=runtime_compatible,
    )


def _mxfp8_dynamic_a_runtime_metadata() -> dict[str, str]:
    flashinfer = importlib.import_module("flashinfer")
    flashinfer_revision = getattr(flashinfer, "__version__", None)
    if flashinfer_revision is None:
        try:
            flashinfer_revision = importlib.metadata.version("flashinfer-python")
        except importlib.metadata.PackageNotFoundError:
            flashinfer_revision = "unknown"
    major, minor = torch.cuda.get_device_capability()
    cutlass_version = os.environ.get(
        _DYNAMIC_A_CUTLASS_VERSION_ENV,
        str(getattr(flashinfer, "__cutlass_version__", "unknown")),
    )
    return {
        "gpu_sm": f"sm_{major}{minor}",
        "flashinfer_revision": str(flashinfer_revision),
        "cuda_version": str(torch.version.cuda or "unknown"),
        "cutlass_version": cutlass_version,
        "activation_dtype": "bfloat16",
        "output_dtype": "bfloat16",
        "activation_scale_layout": "8x4",
        "weight_scale_layout": "128x4",
    }


def _dynamic_a_buffer_specs(
    shape: tuple[int, int, int], workspace_bytes: int
) -> tuple[tuple[tuple[int, ...], torch.dtype], ...]:
    m, n, k = shape
    scale_rows = (m + 7) // 8 * 8
    scale_cols = ((k // MXFP8_BLOCK_SIZE + 3) // 4) * 4
    return (
        ((m, k), torch.float8_e4m3fn),
        ((scale_rows, scale_cols), torch.uint8),
        ((workspace_bytes,), torch.uint8),
        ((m, n), torch.bfloat16),
    )


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

    key = (prefix, family, m, n_logical, n_physical, k, layout)
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

    def __init__(self, c: Mxfp8LinearLayerConfig) -> None:
        super().__init__(c)
        artifact_path = os.environ.get(_DYNAMIC_A_ARTIFACT_ENV, "").strip()
        self._dynamic_a_policy = (
            load_mxfp8_cutedsl_dynamic_a_policy(
                artifact_path,
                runtime_metadata=_mxfp8_dynamic_a_runtime_metadata(),
            )
            if artifact_path
            else None
        )
        self._dynamic_a_buffers: dict[
            tuple[int, int, int],
            dict[str, torch.Tensor] | tuple[dict[str, torch.Tensor], ...],
        ] = {}

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

    def reserve_dynamic_a_workspaces(
        self, layer: torch.nn.Module, manager: "WorkspaceManager"
    ) -> None:
        policy = self._dynamic_a_policy
        if policy is None or not policy.runtime_compatible:
            return

        qualified = [
            (shape, config)
            for shape, config in policy.shapes.items()
            if config.status == "qualified"
        ]
        if not qualified:
            return

        specs = tuple(
            spec
            for shape, config in qualified
            for spec in _dynamic_a_buffer_specs(shape, config.workspace_bytes)
        )
        reserved_by_slot = manager.reserve_simultaneous_for_all_ubatches(*specs)
        buffers: dict[tuple[int, int, int], tuple[dict[str, torch.Tensor], ...]] = {}
        names = ("quant_out_value", "quant_out_scale", "workspace", "out")
        for shape_index, (shape, _) in enumerate(qualified):
            offset = shape_index * len(names)
            buffers[shape] = tuple(
                dict(zip(names, slot[offset : offset + len(names)]))
                for slot in reserved_by_slot
            )
        self._dynamic_a_buffers.update(buffers)

        if _env_flag_enabled(_DYNAMIC_A_TELEMETRY_ENV):
            logger.info_once(
                "MXFP8 dynamic-A reserved physical shapes: %s",
                ";".join(",".join(map(str, shape)) for shape, _ in qualified),
            )

    def _dynamic_a_buffers_for_shape(
        self, shape: tuple[int, int, int]
    ) -> dict[str, torch.Tensor] | None:
        buffers = self._dynamic_a_buffers.get(shape)
        if buffers is None or isinstance(buffers, dict):
            return buffers
        from vllm.v1.worker import workspace as workspace_module

        return buffers[workspace_module.dbo_current_ubatch_id()]

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
        problem = (input_2d.shape[0], N, K)
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

        dynamic_buffers = self._dynamic_a_buffers_for_shape(problem)
        use_dynamic_a = (
            self._dynamic_a_policy is not None
            and dynamic_buffers is not None
            and x.dtype == torch.bfloat16
            and current_platform.is_cuda()
            and current_platform.is_device_capability_family(100)
            and vllm_flashinfer.has_flashinfer_mxfp8_dynamic_a_cutlass()
            and self._dynamic_a_policy.select(problem) == "dynamic_cutlass_8x4"
        )
        if use_dynamic_a:
            assert dynamic_buffers is not None
            config = self._dynamic_a_policy.shapes[problem]
            output = vllm_flashinfer.mm_mxfp8_dynamic_a_cutlass(
                input_2d,
                weight,
                weight_scale,
                out=dynamic_buffers["out"],
                workspace=dynamic_buffers["workspace"],
                quant_out_value=dynamic_buffers["quant_out_value"],
                quant_out_scale=dynamic_buffers["quant_out_scale"],
                out_dtype=out_dtype,
                tactic=config.tactic,
            )
        else:
            if (
                self._dynamic_a_policy is not None
                and dynamic_buffers is not None
                and (
                    not current_platform.is_cuda()
                    or not current_platform.is_device_capability_family(100)
                    or x.dtype != torch.bfloat16
                    or not vllm_flashinfer.has_flashinfer_mxfp8_dynamic_a_cutlass()
                )
            ):
                self._dynamic_a_policy.select(problem)
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
        prepare_mxfp8_trtllm_high_m_tactic_state(layer.weight.device)

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
        n_padded, k = weight.shape
        output_features = layer._mxfp8_trtllm_output_features

        input_shape = x.shape
        input_2d = x.view(-1, k)
        use_8x4 = int(input_2d.shape[0]) <= 256
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
