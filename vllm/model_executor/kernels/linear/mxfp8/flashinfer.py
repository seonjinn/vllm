# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import functools
import importlib
import importlib.metadata
import json
import os
import socket
import threading
import weakref
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import count
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
from vllm.utils.torch_utils import direct_register_custom_op

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
_COUNTER_NAMES = (
    "dynamic_hits",
    "stock_misses",
    "incompatibility_fallbacks",
)
_TELEMETRY_LOCK = threading.Lock()
_POLICY_REQUEST_COUNTERS: dict[str, dict[str, int]] = defaultdict(
    lambda: dict.fromkeys(_COUNTER_NAMES, 0)
)
_EXECUTION_COUNTERS: dict[str, dict[str, int]] = defaultdict(
    lambda: dict.fromkeys(_COUNTER_NAMES, 0)
)
_EXECUTION_TELEMETRY_TENSORS: dict[
    str, list[weakref.ReferenceType[torch.Tensor]]
] = defaultdict(list)
_DYNAMIC_A_OWNER_IDS = count()


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
        runtime_metadata: Mapping[str, str],
        telemetry_key: str,
    ) -> None:
        self._shapes = MappingProxyType(dict(shapes))
        self._runtime_compatible = runtime_compatible
        self._runtime_metadata = MappingProxyType(dict(runtime_metadata))
        self._telemetry_key = telemetry_key

    @property
    def shapes(self) -> Mapping[tuple[int, int, int], _DynamicAShapeConfig]:
        return self._shapes

    @property
    def runtime_compatible(self) -> bool:
        return self._runtime_compatible

    @property
    def runtime_metadata(self) -> Mapping[str, str]:
        return self._runtime_metadata

    @property
    def telemetry_key(self) -> str:
        return self._telemetry_key

    def select(self, shape: tuple[int, int, int]) -> str:
        config = self._shapes.get(shape)
        if not self._runtime_compatible or (
            config is not None and config.status != "qualified"
        ):
            _record_host_counter(
                _POLICY_REQUEST_COUNTERS,
                self._telemetry_key,
                "incompatibility_fallbacks",
            )
            return "stock_cutedsl"
        if config is None:
            _record_host_counter(
                _POLICY_REQUEST_COUNTERS, self._telemetry_key, "stock_misses"
            )
            return "stock_cutedsl"
        _record_host_counter(
            _POLICY_REQUEST_COUNTERS, self._telemetry_key, "dynamic_hits"
        )
        return "dynamic_cutlass_8x4"

    def selection_counters(self) -> dict[str, int]:
        with _TELEMETRY_LOCK:
            return dict(_POLICY_REQUEST_COUNTERS[self._telemetry_key])

    def dispatch_counters(self) -> dict[str, int]:
        with _TELEMETRY_LOCK:
            counters = dict(_EXECUTION_COUNTERS[self._telemetry_key])
            refs = list(_EXECUTION_TELEMETRY_TENSORS[self._telemetry_key])
        for tensor_ref in refs:
            tensor = tensor_ref()
            if tensor is not None:
                values = tensor.detach().cpu().tolist()
                for name, value in zip(_COUNTER_NAMES, values):
                    counters[name] += int(value)
        return counters


def _record_host_counter(
    registry: dict[str, dict[str, int]], telemetry_key: str, counter: str
) -> None:
    with _TELEMETRY_LOCK:
        registry[telemetry_key][counter] += 1


def _register_execution_telemetry_tensor(
    telemetry_key: str, tensor: torch.Tensor
) -> None:
    with _TELEMETRY_LOCK:
        refs = _EXECUTION_TELEMETRY_TENSORS[telemetry_key]
        if not any(existing() is tensor for existing in refs):
            refs.append(weakref.ref(tensor))


def _record_execution(
    telemetry_key: str,
    counter: str,
    telemetry: torch.Tensor | None,
) -> None:
    if telemetry is None:
        _record_host_counter(_EXECUTION_COUNTERS, telemetry_key, counter)
        return
    telemetry[_COUNTER_NAMES.index(counter)].add_(1)


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


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key {key!r}")
        result[key] = value
    return result


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
        artifact = _require_json_object(
            json.load(stream, object_pairs_hook=_reject_duplicate_json_keys),
            "MXFP8 artifact",
        )

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
        runtime_metadata=runtime_metadata,
        telemetry_key=f"{artifact_path}|{runtime_items!r}",
    )


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _mxfp8_dynamic_a_runtime_metadata(
    activation_dtype: torch.dtype,
    output_dtype: torch.dtype,
) -> dict[str, str]:
    try:
        flashinfer = importlib.import_module("flashinfer")
    except ModuleNotFoundError:
        flashinfer = None
    flashinfer_revision = getattr(flashinfer, "__version__", None)
    if flashinfer_revision is None:
        try:
            flashinfer_revision = importlib.metadata.version("flashinfer-python")
        except importlib.metadata.PackageNotFoundError:
            flashinfer_revision = "unknown"
    major, minor = (
        torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
    )
    cutlass_version = os.environ.get(
        _DYNAMIC_A_CUTLASS_VERSION_ENV,
        str(getattr(flashinfer, "__cutlass_version__", "unknown")),
    )
    return {
        "gpu_sm": f"sm_{major}{minor}",
        "flashinfer_revision": str(flashinfer_revision),
        "cuda_version": str(torch.version.cuda or "unknown"),
        "cutlass_version": cutlass_version,
        "activation_dtype": _dtype_name(activation_dtype),
        "output_dtype": _dtype_name(output_dtype),
        "activation_scale_layout": "8x4",
        "weight_scale_layout": "128x4",
    }


def _dynamic_a_buffer_specs(
    shape: tuple[int, int, int], workspace_bytes: int, output_dtype: torch.dtype
) -> tuple[tuple[tuple[int, ...], torch.dtype], ...]:
    m, n, k = shape
    scale_rows = (m + 7) // 8 * 8
    scale_cols = ((k // MXFP8_BLOCK_SIZE + 3) // 4) * 4
    return (
        ((m, k), torch.float8_e4m3fn),
        ((scale_rows, scale_cols), torch.uint8),
        ((workspace_bytes,), torch.uint8),
        ((m, n), output_dtype),
    )


def _stock_cutedsl_mxfp8(
    input_2d: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    input_mxfp8, input_scale = mxfp8_e4m3_quantize(
        input_2d, is_sf_swizzled_layout=True
    )
    return vllm_flashinfer.mm_mxfp8(
        input_mxfp8,
        weight,
        input_scale,
        weight_scale,
        out_dtype=out_dtype,
        backend="cute-dsl",
    )


def _mxfp8_cutedsl_dynamic_a_hybrid_impl(
    input_2d: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    dynamic_buffers: list[torch.Tensor],
    qualified_shapes: list[int],
    rejected_shapes: list[int],
    buffer_offsets: list[int],
    tactics: list[int],
    dynamic_enabled: bool,
    telemetry_key: str,
    telemetry: torch.Tensor | None,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    problem = (input_2d.shape[0], weight.shape[1], weight.shape[0])

    if not dynamic_enabled:
        output = _stock_cutedsl_mxfp8(
            input_2d, weight, weight_scale, out_dtype
        )
        _record_execution(telemetry_key, "incompatibility_fallbacks", telemetry)
        return output

    for index in range(0, len(qualified_shapes), 3):
        if problem != tuple(qualified_shapes[index : index + 3]):
            continue
        shape_index = index // 3
        buffer_offset = buffer_offsets[shape_index]
        if buffer_offset < 0:
            output = _stock_cutedsl_mxfp8(
                input_2d, weight, weight_scale, out_dtype
            )
            _record_execution(
                telemetry_key, "incompatibility_fallbacks", telemetry
            )
            return output
        quant_out_value, quant_out_scale, workspace, out = dynamic_buffers[
            buffer_offset : buffer_offset + 4
        ]
        output = vllm_flashinfer.mm_mxfp8_dynamic_a_cutlass(
            input_2d,
            weight,
            weight_scale,
            out=out,
            workspace=workspace,
            quant_out_value=quant_out_value,
            quant_out_scale=quant_out_scale,
            out_dtype=out_dtype,
            tactic=tactics[shape_index],
        )
        _record_execution(telemetry_key, "dynamic_hits", telemetry)
        return output

    for index in range(0, len(rejected_shapes), 3):
        if problem == tuple(rejected_shapes[index : index + 3]):
            output = _stock_cutedsl_mxfp8(
                input_2d, weight, weight_scale, out_dtype
            )
            _record_execution(
                telemetry_key, "incompatibility_fallbacks", telemetry
            )
            return output

    output = _stock_cutedsl_mxfp8(input_2d, weight, weight_scale, out_dtype)
    _record_execution(telemetry_key, "stock_misses", telemetry)
    return output


def _mxfp8_cutedsl_dynamic_a_hybrid_fake(
    input_2d: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    dynamic_buffers: list[torch.Tensor],
    qualified_shapes: list[int],
    rejected_shapes: list[int],
    buffer_offsets: list[int],
    tactics: list[int],
    dynamic_enabled: bool,
    telemetry_key: str,
    telemetry: torch.Tensor | None,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    return torch.empty(
        input_2d.shape[0],
        weight.shape[1],
        dtype=out_dtype,
        device=input_2d.device,
    )


direct_register_custom_op(
    op_name="mxfp8_cutedsl_dynamic_a_hybrid",
    op_func=_mxfp8_cutedsl_dynamic_a_hybrid_impl,
    mutates_args=["dynamic_buffers", "telemetry"],
    fake_impl=_mxfp8_cutedsl_dynamic_a_hybrid_fake,
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
        self._dynamic_a_artifact_path = os.environ.get(
            _DYNAMIC_A_ARTIFACT_ENV, ""
        ).strip()
        self._dynamic_a_policy: Mxfp8CutedslDynamicAPolicy | None = None
        self._dynamic_a_buffers: dict[
            tuple[int, int, int],
            dict[str, torch.Tensor] | tuple[dict[str, torch.Tensor], ...],
        ] = {}
        self._dynamic_a_telemetry: tuple[torch.Tensor, ...] = ()
        self._dynamic_a_owner_key = f"mxfp8-dynamic-a:{next(_DYNAMIC_A_OWNER_IDS)}"

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
        self,
        layer: torch.nn.Module,
        manager: "WorkspaceManager",
        *,
        activation_dtype: torch.dtype,
        output_dtype: torch.dtype,
    ) -> None:
        artifact_path = self._dynamic_a_artifact_path
        if not artifact_path:
            return

        policy = load_mxfp8_cutedsl_dynamic_a_policy(
            artifact_path,
            runtime_metadata=_mxfp8_dynamic_a_runtime_metadata(
                activation_dtype, output_dtype
            ),
        )
        self._dynamic_a_policy = policy
        self._dynamic_a_buffers = {}
        if (
            not policy.runtime_compatible
            or not current_platform.is_cuda()
            or not current_platform.is_device_capability_family(100)
            or not vllm_flashinfer.has_flashinfer_mxfp8_dynamic_a_cutlass()
        ):
            return

        qualified = sorted(
            (shape, config)
            for shape, config in policy.shapes.items()
            if config.status == "qualified"
        )
        if not qualified:
            return

        queried_resources: dict[tuple[int, int, int], dict[str, int]] = {}
        for shape, config in qualified:
            m, n, k = shape
            resources = vllm_flashinfer.get_flashinfer_mxfp8_dynamic_a_resources(
                m,
                n,
                k,
                tactic=config.tactic,
                activation_dtype=activation_dtype,
                output_dtype=output_dtype,
            )
            buffer_specs = _dynamic_a_buffer_specs(
                shape, config.workspace_bytes, output_dtype
            )
            expected_resources = {
                "workspace_bytes": config.workspace_bytes,
                "activation_value_elements": m * k,
                "activation_scale_elements": int(
                    torch.Size(buffer_specs[1][0]).numel()
                ),
                "output_elements": m * n,
            }
            if resources is None or resources != expected_resources:
                logger.warning_once(
                    "MXFP8 dynamic-A disabled because runtime resources do not "
                    "match the artifact for physical shape %s: expected=%s, got=%s",
                    shape,
                    tuple(sorted(expected_resources.items())),
                    None if resources is None else tuple(sorted(resources.items())),
                )
                return
            queried_resources[shape] = resources

        specs = tuple(
            spec
            for shape, _ in qualified
            for spec in _dynamic_a_buffer_specs(
                shape, queried_resources[shape]["workspace_bytes"], output_dtype
            )
        ) + (((len(_COUNTER_NAMES),), torch.int64),)
        owner_key = self._dynamic_a_owner_key
        reserved_by_slot = manager.reserve_owner_for_all_ubatches(owner_key, *specs)
        buffers: dict[
            tuple[int, int, int],
            dict[str, torch.Tensor] | tuple[dict[str, torch.Tensor], ...],
        ] = {}
        names = ("quant_out_value", "quant_out_scale", "workspace", "out")
        for shape_index, (shape, _) in enumerate(qualified):
            offset = shape_index * len(names)
            shape_buffers: list[dict[str, torch.Tensor]] = []
            for slot in reserved_by_slot:
                slot_buffers: dict[str, torch.Tensor] = {}
                for name, tensor in zip(
                    names, slot[offset : offset + len(names)]
                ):
                    slot_buffers[name] = tensor
                shape_buffers.append(slot_buffers)
            buffers[shape] = tuple(shape_buffers)
        self._dynamic_a_buffers = buffers

        if not self._dynamic_a_telemetry:
            self._dynamic_a_telemetry = tuple(slot[-1] for slot in reserved_by_slot)
            for telemetry in self._dynamic_a_telemetry:
                telemetry.zero_()
                _register_execution_telemetry_tensor(
                    policy.telemetry_key, telemetry
                )

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

    def _dynamic_a_telemetry_for_current_slot(self) -> torch.Tensor | None:
        telemetry = getattr(self, "_dynamic_a_telemetry", ())
        if not telemetry:
            return None
        from vllm.v1.worker import workspace as workspace_module

        return telemetry[workspace_module.dbo_current_ubatch_id()]

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

        policy = getattr(self, "_dynamic_a_policy", None)
        if policy is None:
            output = _stock_cutedsl_mxfp8(
                input_2d, weight, weight_scale, out_dtype
            )
        else:
            qualified = sorted(
                (shape, config)
                for shape, config in policy.shapes.items()
                if config.status == "qualified"
            )
            rejected = sorted(
                shape
                for shape, config in policy.shapes.items()
                if config.status == "rejected"
            )
            dynamic_buffers: list[torch.Tensor] = []
            buffer_offsets: list[int] = []
            for shape, _ in qualified:
                buffers = self._dynamic_a_buffers_for_shape(shape)
                if buffers is None:
                    buffer_offsets.append(-1)
                    continue
                buffer_offsets.append(len(dynamic_buffers))
                dynamic_buffers.extend(
                    buffers[name]
                    for name in (
                        "quant_out_value",
                        "quant_out_scale",
                        "workspace",
                        "out",
                    )
                )

            runtime_metadata = policy.runtime_metadata
            dynamic_enabled = (
                policy.runtime_compatible
                and runtime_metadata["activation_dtype"] == _dtype_name(x.dtype)
                and runtime_metadata["output_dtype"] == _dtype_name(out_dtype)
                and current_platform.is_cuda()
                and current_platform.is_device_capability_family(100)
                and vllm_flashinfer.has_flashinfer_mxfp8_dynamic_a_cutlass()
            )
            output = torch.ops.vllm.mxfp8_cutedsl_dynamic_a_hybrid(
                input_2d,
                weight,
                weight_scale,
                dynamic_buffers,
                [dimension for shape, _ in qualified for dimension in shape],
                [dimension for shape in rejected for dimension in shape],
                buffer_offsets,
                [config.tactic for _, config in qualified],
                dynamic_enabled,
                policy.telemetry_key,
                self._dynamic_a_telemetry_for_current_slot(),
                out_dtype,
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
