# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compatibility wrapper for FlashInfer API changes.

Users of vLLM should always import **only** these wrappers.
"""

import contextlib
import functools
import importlib
import importlib.metadata
import importlib.util
import json
import os
import shutil
import socket
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple, NoReturn

import requests
import torch
from torch._higher_order_ops.auto_functionalize import auto_functionalized

import vllm
import vllm.envs as envs
from vllm.compilation.passes.inductor_pass import InductorPass, get_pass_context
from vllm.logger import init_logger
from vllm.platforms import current_platform

if TYPE_CHECKING:
    from vllm.model_executor.kernels.linear.mxfp8.tactic_config import (
        Mxfp8DenseRuntimeConfig,
    )

logger = init_logger(__name__)

_MXFP8_ADAPTIVE_DISPATCH_TRACE_SEEN: set[
    tuple[tuple[int, int, int], bool, int, bool, str | None]
] = set()


def _trace_mxfp8_adaptive_dispatch(
    *,
    shape_key: tuple[int, int, int],
    use_8x4_sf_layout: bool,
    tactic: int,
    tactic_hit: bool,
    config_sha256: str | None,
) -> None:
    raw_enabled = os.environ.get("VLLM_MXFP8_DENSE_SHAPE_TRACE", "")
    if raw_enabled.strip().lower() in ("", "0", "false", "no", "off"):
        return
    if torch.cuda.is_current_stream_capturing():
        return
    trace_dir = os.environ.get("VLLM_MXFP8_DENSE_SHAPE_TRACE_DIR", "").strip()
    if not trace_dir:
        return

    key = (shape_key, use_8x4_sf_layout, tactic, tactic_hit, config_sha256)
    if key in _MXFP8_ADAPTIVE_DISPATCH_TRACE_SEEN:
        return
    _MXFP8_ADAPTIVE_DISPATCH_TRACE_SEEN.add(key)

    path = Path(trace_dir)
    path.mkdir(parents=True, exist_ok=True)
    output = path / f"adaptive_dispatch_{socket.gethostname()}_{os.getpid()}.jsonl"
    record = {
        "event": "mxfp8_adaptive_dispatch",
        "config_sha256": config_sha256,
        "time": time.time(),
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "layout": "8x4" if use_8x4_sf_layout else "128x4",
        "m": shape_key[0],
        "n": shape_key[1],
        "k": shape_key[2],
        "tactic": tactic,
        "tactic_source": "static_hint" if tactic_hit else "runner_default",
    }
    with output.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")


def _mxfp8_env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off", "")


@functools.cache
def _parse_mxfp8_tactic_hints(
    raw_hints: str,
) -> dict[tuple[int, int, int], int]:
    tactic_table: dict[tuple[int, int, int], int] = {}
    for item in raw_hints.split(";"):
        item = item.strip()
        if not item or ":" not in item:
            continue
        shape_raw, tactic_raw = item.split(":", 1)
        try:
            shape = tuple(int(part.strip()) for part in shape_raw.split(","))
            if len(shape) != 3:
                continue
            tactic_table[(shape[0], shape[1], shape[2])] = int(tactic_raw.strip())
        except ValueError:
            continue
    return tactic_table


class _Mxfp8TrtllmConfigurationFingerprint(NamedTuple):
    config_path: str
    config_sha256: str
    qualification_scope: str
    model: str
    tensor_parallel_size: int
    layout_mode: str
    switch_m: int
    gemm_backend: str
    direct_trtllm: bool
    require_direct_trtllm: bool
    default_tactic: int
    low_tactic_hints_raw: str
    low_tactic_map: tuple[tuple[tuple[int, int, int], int], ...]
    high_tactic_hints_raw: str
    high_tactic_map: tuple[tuple[tuple[int, int, int], int], ...]
    quant_backend: str
    require_8x4_quant: bool
    pad_to_128: bool


def _active_mxfp8_model_and_tp() -> tuple[str, int]:
    from vllm.config import get_current_vllm_config

    vllm_config = get_current_vllm_config()
    return (
        str(vllm_config.model_config.model),
        int(vllm_config.parallel_config.tensor_parallel_size),
    )


def _load_mxfp8_dense_runtime_config(
    config_reference: str,
    *,
    actual_model: str,
    actual_tensor_parallel_size: int,
) -> "Mxfp8DenseRuntimeConfig":
    from vllm.model_executor.kernels.linear.mxfp8.tactic_config import (
        load_mxfp8_dense_runtime_config,
    )

    return load_mxfp8_dense_runtime_config(
        config_reference,
        actual_vllm_version=vllm.__version__,
        actual_flashinfer_version=importlib.metadata.version(
            "flashinfer-python"
        ),
        actual_compute_capability=torch.cuda.get_device_capability(),
        actual_model=actual_model,
        actual_tensor_parallel_size=actual_tensor_parallel_size,
    )


def _mxfp8_trtllm_configuration_fingerprint(
    prepared: _Mxfp8TrtllmConfigurationFingerprint | None = None,
) -> _Mxfp8TrtllmConfigurationFingerprint:
    def env_flag(name: str, default: bool = False) -> bool:
        raw = os.environ.get(name)
        if raw is None:
            return default
        return raw.strip().lower() not in ("0", "false", "no", "off", "")

    config_reference = os.environ.get("VLLM_MXFP8_DENSE_CONFIG_FILE", "").strip()
    low_tactic_hints_raw = os.environ.get(
        "VLLM_MXFP8_DENSE_TRTLLM_TACTIC_HINTS", ""
    )
    high_tactic_hints_raw = os.environ.get(
        "VLLM_MXFP8_DENSE_TRTLLM_TACTIC_HINTS_128X4", ""
    )
    if config_reference:
        for variable, value in (
            ("VLLM_MXFP8_DENSE_TRTLLM_TACTIC_HINTS", low_tactic_hints_raw),
            (
                "VLLM_MXFP8_DENSE_TRTLLM_TACTIC_HINTS_128X4",
                high_tactic_hints_raw,
            ),
        ):
            if value.strip():
                raise ValueError(
                    f"{variable} cannot be set with VLLM_MXFP8_DENSE_CONFIG_FILE"
                )
        if prepared is None:
            actual_model, actual_tensor_parallel_size = (
                _active_mxfp8_model_and_tp()
            )
        else:
            actual_model = prepared.model
            actual_tensor_parallel_size = prepared.tensor_parallel_size
        runtime_config = _load_mxfp8_dense_runtime_config(
            config_reference,
            actual_model=actual_model,
            actual_tensor_parallel_size=actual_tensor_parallel_size,
        )
        return _Mxfp8TrtllmConfigurationFingerprint(
            config_path=str(runtime_config.source_path),
            config_sha256=runtime_config.source_sha256,
            qualification_scope=str(
                runtime_config.provenance["qualification_scope"]
            ),
            model=actual_model,
            tensor_parallel_size=actual_tensor_parallel_size,
            layout_mode=runtime_config.layout,
            switch_m=runtime_config.switch_m,
            gemm_backend=runtime_config.gemm_backend,
            direct_trtllm=runtime_config.direct_trtllm,
            require_direct_trtllm=runtime_config.require_direct_trtllm,
            default_tactic=runtime_config.default_tactic,
            low_tactic_hints_raw="",
            low_tactic_map=runtime_config.tactics_8x4,
            high_tactic_hints_raw="",
            high_tactic_map=runtime_config.tactics_128x4,
            quant_backend=runtime_config.quant_backend,
            require_8x4_quant=runtime_config.require_8x4_quant,
            pad_to_128=runtime_config.pad_to_128,
        )

    layout_mode = (
        os.environ.get("VLLM_MXFP8_DENSE_A_SF_LAYOUT", "128x4").strip().lower()
    )
    if layout_mode not in ("adaptive", "shape-aware", "shape_aware"):
        raise RuntimeError(
            "MXFP8 TRTLLM adaptive execution state requires adaptive layout mode; "
            f"got {layout_mode!r}"
        )

    switch_m = int(os.environ.get("VLLM_MXFP8_DENSE_A_SF_LAYOUT_SWITCH_M", "256"))
    if switch_m <= 0:
        raise ValueError("VLLM_MXFP8_DENSE_A_SF_LAYOUT_SWITCH_M must be positive")
    if switch_m % 128:
        raise ValueError(
            "VLLM_MXFP8_DENSE_A_SF_LAYOUT_SWITCH_M must be a multiple of 128 "
            "because adaptive serving pads the physical M dimension to 128 rows"
        )

    gemm_backend = (
        os.environ.get("VLLM_MXFP8_DENSE_GEMM_BACKEND", "cutlass").strip().lower()
    )
    if gemm_backend != "trtllm":
        raise RuntimeError(
            f"MXFP8 adaptive layout requires backend='trtllm'; got {gemm_backend!r}"
        )

    return _Mxfp8TrtllmConfigurationFingerprint(
        config_path="",
        config_sha256="",
        qualification_scope="",
        model="",
        tensor_parallel_size=0,
        layout_mode="adaptive",
        switch_m=switch_m,
        gemm_backend=gemm_backend,
        direct_trtllm=env_flag("VLLM_MXFP8_DENSE_DIRECT_TRTLLM"),
        require_direct_trtllm=env_flag("VLLM_MXFP8_DENSE_REQUIRE_DIRECT_TRTLLM"),
        default_tactic=int(os.environ.get("VLLM_MXFP8_DENSE_TRTLLM_TACTIC", "-1")),
        low_tactic_hints_raw=low_tactic_hints_raw,
        low_tactic_map=tuple(
            sorted(_parse_mxfp8_tactic_hints(low_tactic_hints_raw).items())
        ),
        high_tactic_hints_raw=high_tactic_hints_raw,
        high_tactic_map=tuple(
            sorted(_parse_mxfp8_tactic_hints(high_tactic_hints_raw).items())
        ),
        quant_backend=os.environ.get("VLLM_MXFP8_DENSE_QUANT_BACKEND", "cuda"),
        require_8x4_quant=env_flag("VLLM_MXFP8_DENSE_REQUIRE_8X4_QUANT"),
        pad_to_128=env_flag("VLLM_MXFP8_DENSE_PAD_TO_128", True),
    )


def _validate_mxfp8_trtllm_configuration(
    prepared: _Mxfp8TrtllmConfigurationFingerprint,
    active: _Mxfp8TrtllmConfigurationFingerprint,
) -> None:
    if prepared != active:
        raise RuntimeError(
            "MXFP8 TRTLLM configuration changed after preparation; "
            "restart the worker before using the new layout or tactic configuration"
        )


_MXFP8_TRTLLM_CONFIGURATION: _Mxfp8TrtllmConfigurationFingerprint | None = None
_MXFP8_DENSE_CONFIG_LOGGED = False
_MXFP8_TRTLLM_SOURCE_VALIDATED = False


def _log_mxfp8_dense_config_once(
    configuration: _Mxfp8TrtllmConfigurationFingerprint,
) -> None:
    global _MXFP8_DENSE_CONFIG_LOGGED
    if not configuration.config_path or _MXFP8_DENSE_CONFIG_LOGGED:
        return
    _MXFP8_DENSE_CONFIG_LOGGED = True
    logger.info(
        "MXFP8 dense config path=%s sha256=%s mode=adaptive "
        "switch_m=%d tactics_8x4=%d tactics_128x4=%d "
        "qualification_scope=%s",
        configuration.config_path,
        configuration.config_sha256,
        configuration.switch_m,
        len(configuration.low_tactic_map),
        len(configuration.high_tactic_map),
        configuration.qualification_scope,
    )


def _freeze_mxfp8_trtllm_configuration(
    active: _Mxfp8TrtllmConfigurationFingerprint,
) -> _Mxfp8TrtllmConfigurationFingerprint:
    global _MXFP8_TRTLLM_CONFIGURATION
    if _MXFP8_TRTLLM_CONFIGURATION is None:
        _MXFP8_TRTLLM_CONFIGURATION = active
    else:
        _validate_mxfp8_trtllm_configuration(_MXFP8_TRTLLM_CONFIGURATION, active)
    _log_mxfp8_dense_config_once(_MXFP8_TRTLLM_CONFIGURATION)
    return _MXFP8_TRTLLM_CONFIGURATION


def get_mxfp8_trtllm_configuration() -> _Mxfp8TrtllmConfigurationFingerprint:
    if _MXFP8_TRTLLM_CONFIGURATION is not None:
        return _MXFP8_TRTLLM_CONFIGURATION
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "MXFP8 TRTLLM configuration must be loaded before CUDA Graph capture"
        )
    return _freeze_mxfp8_trtllm_configuration(
        _mxfp8_trtllm_configuration_fingerprint()
    )


def get_mxfp8_trtllm_file_configuration(
) -> _Mxfp8TrtllmConfigurationFingerprint | None:
    if _MXFP8_TRTLLM_CONFIGURATION is not None:
        if _MXFP8_TRTLLM_CONFIGURATION.config_path:
            return _MXFP8_TRTLLM_CONFIGURATION
        return None
    if not os.environ.get("VLLM_MXFP8_DENSE_CONFIG_FILE", "").strip():
        return None
    return get_mxfp8_trtllm_configuration()


def validate_mxfp8_trtllm_configuration(
    prepared: _Mxfp8TrtllmConfigurationFingerprint,
) -> None:
    if _MXFP8_TRTLLM_CONFIGURATION is None:
        raise RuntimeError(
            "MXFP8 TRTLLM configuration was not prepared before execution"
        )
    _validate_mxfp8_trtllm_configuration(
        prepared, _MXFP8_TRTLLM_CONFIGURATION
    )


def validate_mxfp8_trtllm_configuration_source(
    prepared: _Mxfp8TrtllmConfigurationFingerprint | None = None,
) -> None:
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "MXFP8 TRTLLM configuration source cannot be validated during "
            "CUDA Graph capture"
        )
    frozen = get_mxfp8_trtllm_configuration()
    if prepared is not None:
        _validate_mxfp8_trtllm_configuration(prepared, frozen)
    active = _mxfp8_trtllm_configuration_fingerprint(frozen)
    _validate_mxfp8_trtllm_configuration(frozen, active)


class _Mxfp8TrtllmDirectState(NamedTuple):
    workspace_8x4: torch.Tensor
    workspace_128x4: torch.Tensor
    runner_8x4: Any
    runner_128x4: Any
    configuration: _Mxfp8TrtllmConfigurationFingerprint | None
    tactic_map_8x4: dict[tuple[int, int, int], int]
    tactic_map_128x4: dict[tuple[int, int, int], int]


_MXFP8_TRTLLM_DIRECT_STATES: dict[tuple[str, int], _Mxfp8TrtllmDirectState] = {}


def _mxfp8_trtllm_device_key(device: torch.device) -> tuple[str, int]:
    canonical_device = torch.device(device)
    if canonical_device.type != "cuda":
        raise RuntimeError(
            f"MXFP8 TRTLLM direct state requires a CUDA device; got {canonical_device}"
        )
    device_index = canonical_device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    return canonical_device.type, device_index


def prepare_mxfp8_trtllm_direct_state(
    device: torch.device,
) -> _Mxfp8TrtllmDirectState:
    device_key = _mxfp8_trtllm_device_key(device)
    canonical_device = torch.device(device_key[0], device_key[1])
    with torch.cuda.device(canonical_device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "MXFP8 TRTLLM direct state must be prepared before CUDA Graph capture"
            )

        runtime_configuration = get_mxfp8_trtllm_file_configuration()
        layout_mode = (
            runtime_configuration.layout_mode
            if runtime_configuration is not None
            else os.environ.get(
                "VLLM_MXFP8_DENSE_A_SF_LAYOUT", "128x4"
            )
            .strip()
            .lower()
        )
        is_adaptive_layout = layout_mode in (
            "adaptive",
            "shape-aware",
            "shape_aware",
        )
        configuration = None
        tactic_map_8x4: dict[tuple[int, int, int], int] = {}
        tactic_map_128x4: dict[tuple[int, int, int], int] = {}
        if is_adaptive_layout:
            configuration = get_mxfp8_trtllm_configuration()

        prepared_state = _MXFP8_TRTLLM_DIRECT_STATES.get(device_key)
        if prepared_state is not None:
            if configuration is not None and prepared_state.configuration is None:
                raise RuntimeError(
                    "MXFP8 TRTLLM state was prepared without adaptive configuration; "
                    "restart the worker before enabling adaptive layout"
                )
            if prepared_state.configuration is not None:
                validate_mxfp8_trtllm_configuration(prepared_state.configuration)
            return prepared_state

        if configuration is not None:
            from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
                prepare_mxfp8_dense_quant_backend,
            )

            prepare_mxfp8_dense_quant_backend(configuration.quant_backend)
            tactic_map_8x4 = dict(configuration.low_tactic_map)
            tactic_map_128x4 = dict(configuration.high_tactic_map)

        from flashinfer.gemm.gemm_base import (  # type: ignore
            DEFAULT_WORKSPACE_SIZE,
            _get_cache_buf,
            get_trtllm_gemm_module,
        )

        workspace_8x4 = _get_cache_buf(
            "vllm_direct_trtllm_mxfp8_workspace_8x4",
            DEFAULT_WORKSPACE_SIZE,
            canonical_device,
        )
        workspace_128x4 = _get_cache_buf(
            "vllm_direct_trtllm_mxfp8_workspace_128x4",
            DEFAULT_WORKSPACE_SIZE,
            canonical_device,
        )
        gemm_module = get_trtllm_gemm_module()
        prepared_state = _Mxfp8TrtllmDirectState(
            workspace_8x4=workspace_8x4,
            workspace_128x4=workspace_128x4,
            runner_8x4=gemm_module.trtllm_mxfp8_gemm_runner(use_8x4_sf_layout=True),
            runner_128x4=gemm_module.trtllm_mxfp8_gemm_runner(use_8x4_sf_layout=False),
            configuration=configuration,
            tactic_map_8x4=tactic_map_8x4,
            tactic_map_128x4=tactic_map_128x4,
        )
        _MXFP8_TRTLLM_DIRECT_STATES[device_key] = prepared_state
        return prepared_state


def _require_mxfp8_trtllm_direct_state(
    device: torch.device,
) -> _Mxfp8TrtllmDirectState:
    device_key = _mxfp8_trtllm_device_key(device)
    prepared_state = _MXFP8_TRTLLM_DIRECT_STATES.get(device_key)
    if prepared_state is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "MXFP8 TRTLLM direct state was not prepared before CUDA Graph capture; "
                f"device={device_key}"
            )
        return prepare_mxfp8_trtllm_direct_state(device)
    runtime_configuration = get_mxfp8_trtllm_file_configuration()
    layout_mode = (
        runtime_configuration.layout_mode
        if runtime_configuration is not None
        else os.environ.get("VLLM_MXFP8_DENSE_A_SF_LAYOUT", "128x4")
        .strip()
        .lower()
    )
    is_adaptive_layout = layout_mode in (
        "adaptive",
        "shape-aware",
        "shape_aware",
    )
    if is_adaptive_layout and prepared_state.configuration is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "MXFP8 TRTLLM adaptive state was not prepared before "
                "CUDA Graph capture; "
                f"device={device_key}"
            )
        return prepare_mxfp8_trtllm_direct_state(device)
    if prepared_state.configuration is not None:
        validate_mxfp8_trtllm_configuration(prepared_state.configuration)
    return prepared_state


def _mxfp8_trtllm_run_prepared(
    A: torch.Tensor,
    B: torch.Tensor,
    A_scale: torch.Tensor,
    B_scale: torch.Tensor,
    out_dtype: torch.dtype,
    workspace: torch.Tensor,
    use_8x4_sf_layout: bool,
    tactic: int,
) -> torch.Tensor:
    state = _require_mxfp8_trtllm_direct_state(A.device)
    m = int(A.shape[0])
    n = int(B.shape[1])
    out = torch.empty((m, n), dtype=out_dtype, device=A.device)
    runner = state.runner_8x4 if use_8x4_sf_layout else state.runner_128x4
    return runner.forward(
        [A, B, A_scale, B_scale, out_dtype, out, workspace],
        tactic=tactic,
    )


def get_mxfp8_trtllm_prepared_workspaces(
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    state = _require_mxfp8_trtllm_direct_state(device)
    return state.workspace_8x4, state.workspace_128x4


def _mxfp8_quantize_mm_fixed_layout_impl(
    A_bf16: torch.Tensor,
    B: torch.Tensor,
    B_scale: torch.Tensor,
    out_dtype: torch.dtype,
    backend: str,
    workspace_8x4: torch.Tensor,
    workspace_128x4: torch.Tensor,
    *,
    use_8x4_sf_layout: bool,
) -> torch.Tensor:
    if backend != "trtllm":
        raise RuntimeError(
            f"MXFP8 adaptive layout requires backend='trtllm'; got {backend!r}"
        )
    state = _require_mxfp8_trtllm_direct_state(A_bf16.device)
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        mxfp8_e4m3_quantize_8x4_impl,
        mxfp8_e4m3_quantize_128x4_impl,
    )

    quantize = (
        mxfp8_e4m3_quantize_8x4_impl
        if use_8x4_sf_layout
        else mxfp8_e4m3_quantize_128x4_impl
    )
    A, A_scale = quantize(A_bf16)
    configuration = state.configuration
    if configuration is None:
        raise RuntimeError("MXFP8 adaptive layout requires a pre-capture configuration")

    tactic_table = state.tactic_map_8x4 if use_8x4_sf_layout else state.tactic_map_128x4
    shape_key = (int(A.shape[0]), int(B.shape[1]), int(A.shape[1]))
    tactic_hit = shape_key in tactic_table
    tactic = tactic_table.get(shape_key, -1)
    _trace_mxfp8_adaptive_dispatch(
        shape_key=shape_key,
        use_8x4_sf_layout=use_8x4_sf_layout,
        tactic=tactic,
        tactic_hit=tactic_hit,
        config_sha256=configuration.config_sha256 or None,
    )
    workspace = workspace_8x4 if use_8x4_sf_layout else workspace_128x4
    return _mxfp8_trtllm_run_prepared(
        A,
        B,
        A_scale,
        B_scale,
        out_dtype,
        workspace,
        use_8x4_sf_layout,
        tactic,
    )


def _mxfp8_layout_for_compile_range(
    range_start: int,
    range_end: int,
    switch_m: int,
) -> bool:
    if range_end <= switch_m:
        return True
    if range_start > switch_m:
        return False
    raise RuntimeError(
        "MXFP8 compile range "
        f"[{range_start}, {range_end}] straddles adaptive layout switch M={switch_m}"
    )


def _specialize_mxfp8_adaptive_layout_graph(
    graph: Any,
    *,
    marker_op: Any,
    fixed_op: Any,
    auto_functionalized_op: Any = None,
) -> int:
    replaced = 0
    for node in list(graph.nodes):
        if node.op != "call_function":
            continue
        if node.target == marker_op:
            node.target = fixed_op
            replaced += 1
        elif (
            node.target == auto_functionalized_op
            and node.args
            and node.args[0] == marker_op
        ):
            node.args = (fixed_op, *node.args[1:])
            replaced += 1
    return replaced


def _validate_mxfp8_adaptive_op_schemas() -> None:
    schema_pairs = (
        (
            torch.ops.vllm.mxfp8_adaptive_quantize_mm_marker.default._schema,
            torch.ops.vllm.mxfp8_quantize_mm_8x4.default._schema,
        ),
        (
            torch.ops.vllm.mxfp8_adaptive_quantize_mm_marker.default._schema,
            torch.ops.vllm.mxfp8_quantize_mm_128x4.default._schema,
        ),
    )
    for marker_schema, fixed_schema in schema_pairs:
        marker_signature = str(marker_schema)[str(marker_schema).index("(") :]
        fixed_signature = str(fixed_schema)[str(fixed_schema).index("(") :]
        if fixed_signature != marker_signature:
            raise RuntimeError(
                "MXFP8 adaptive fixed-op schema must match its marker schema: "
                f"marker={marker_schema}, fixed={fixed_schema}"
            )


class _Mxfp8AdaptiveLayoutSpecializationPass(InductorPass):
    def __init__(
        self,
        configuration: _Mxfp8TrtllmConfigurationFingerprint,
    ) -> None:
        self.configuration = configuration
        self.switch_m = configuration.switch_m

    def __call__(self, graph: torch.fx.Graph) -> None:
        compile_range = get_pass_context().compile_range
        use_8x4_sf_layout = _mxfp8_layout_for_compile_range(
            compile_range.start,
            compile_range.end,
            self.switch_m,
        )
        # Functionalization may wrap the workspace-mutating fused marker.
        _specialize_mxfp8_adaptive_layout_graph(
            graph,
            marker_op=torch.ops.vllm.mxfp8_adaptive_quantize_mm_marker.default,
            fixed_op=(
                torch.ops.vllm.mxfp8_quantize_mm_8x4.default
                if use_8x4_sf_layout
                else torch.ops.vllm.mxfp8_quantize_mm_128x4.default
            ),
            auto_functionalized_op=auto_functionalized,
        )

    def uuid(self) -> str:
        return self.hash_dict(
            {
                "source": self.hash_source(self),
                "configuration": self.configuration._asdict(),
                "phase": "joint_custom_pre_pass",
                "schema_version": 7,
            }
        )


def configure_mxfp8_adaptive_layout_compilation() -> None:
    global _MXFP8_TRTLLM_SOURCE_VALIDATED
    from vllm.config import get_current_vllm_config

    active_configuration = get_mxfp8_trtllm_configuration()
    if not _MXFP8_TRTLLM_SOURCE_VALIDATED:
        validate_mxfp8_trtllm_configuration_source(active_configuration)
        _MXFP8_TRTLLM_SOURCE_VALIDATED = True
    else:
        validate_mxfp8_trtllm_configuration(active_configuration)
    _validate_mxfp8_adaptive_op_schemas()
    vllm_config = get_current_vllm_config()
    compilation_config = vllm_config.compilation_config

    max_num_batched_tokens = vllm_config.scheduler_config.max_num_batched_tokens
    if max_num_batched_tokens is None:
        raise RuntimeError(
            "MXFP8 adaptive layout requires a finite max_num_batched_tokens"
        )
    endpoints = list(compilation_config.compile_ranges_endpoints or [])
    if active_configuration.switch_m < max_num_batched_tokens:
        endpoints.append(active_configuration.switch_m)
    compilation_config.compile_ranges_endpoints = sorted(set(endpoints))

    # Retarget existing traced markers before AOT partitioning and lifetime
    # analysis. No tuple-return node is injected after fake propagation.
    pass_key = "joint_custom_pre_pass"
    existing_pass = compilation_config.inductor_compile_config.get(pass_key)
    if existing_pass is None:
        compilation_config.inductor_compile_config[pass_key] = (
            _Mxfp8AdaptiveLayoutSpecializationPass(active_configuration)
        )
    elif not isinstance(existing_pass, _Mxfp8AdaptiveLayoutSpecializationPass):
        raise RuntimeError(
            "MXFP8 adaptive layout cannot replace an existing Inductor joint pre-pass"
        )
    else:
        _validate_mxfp8_trtllm_configuration(
            existing_pass.configuration,
            active_configuration,
        )


# This is the storage path for the cubins, it can be replaced
# with a local path for testing.
# Referenced from https://github.com/flashinfer-ai/flashinfer/blob/0c9a92c3d9a7e043ab6f3f7b2273269caf6ab044/flashinfer/jit/cubin_loader.py#L35  # noqa: E501
FLASHINFER_CUBINS_REPOSITORY = os.environ.get(
    "FLASHINFER_CUBINS_REPOSITORY",
    "https://edge.urm.nvidia.com/artifactory/sw-kernelinferencelibrary-public-generic-local/",  # noqa: E501
)


@functools.cache
def has_flashinfer_cubin() -> bool:
    """Return `True` if flashinfer-cubin package is available."""
    if envs.VLLM_HAS_FLASHINFER_CUBIN:
        return True
    if importlib.util.find_spec("flashinfer_cubin") is not None:
        return True
    logger.debug_once("flashinfer-cubin package was not found")
    return False


@functools.cache
def has_flashinfer() -> bool:
    """Return `True` if flashinfer-python package is available."""
    # Use find_spec to check if the module exists without importing it
    # This avoids potential CUDA initialization side effects
    if importlib.util.find_spec("flashinfer") is None:
        logger.debug_once("FlashInfer unavailable since package was not found")
        return False
    # When not using flashinfer cubin,
    # Also check if nvcc is available since it's required to JIT compile flashinfer
    if not has_flashinfer_cubin() and shutil.which("nvcc") is None:
        logger.debug_once(
            "FlashInfer unavailable since nvcc was not found "
            "and not using pre-downloaded cubins"
        )
        return False
    return True


def _missing(*_: Any, **__: Any) -> NoReturn:
    """Placeholder for unavailable FlashInfer backend."""
    raise RuntimeError(
        "FlashInfer backend is not available. Please install the package "
        "to enable FlashInfer kernels: "
        "https://github.com/flashinfer-ai/flashinfer"
    )


def _get_submodule(module_name: str) -> Any | None:
    """Safely import a submodule and return it, or None if not available."""
    try:
        return importlib.import_module(module_name)
    except (ImportError, ModuleNotFoundError):
        return None


# General lazy import wrapper
def _lazy_import_wrapper(
    module_name: str, attr_name: str, fallback_fn: Callable[..., Any] = _missing
):
    """Create a lazy import wrapper for a specific function."""

    @functools.cache
    def _get_impl():
        if not has_flashinfer():
            return None
        mod = _get_submodule(module_name)
        return getattr(mod, attr_name, None) if mod else None

    def wrapper(*args, **kwargs):
        impl = _get_impl()
        if impl is None:
            return fallback_fn(*args, **kwargs)
        return impl(*args, **kwargs)

    return wrapper


# Create lazy wrappers for each function
flashinfer_trtllm_bf16_moe = _lazy_import_wrapper(
    "flashinfer.fused_moe", "trtllm_bf16_moe"
)
flashinfer_trtllm_bf16_routed_moe = _lazy_import_wrapper(
    "flashinfer.fused_moe", "trtllm_bf16_routed_moe"
)
flashinfer_trtllm_fp8_block_scale_moe = _lazy_import_wrapper(
    "flashinfer.fused_moe", "trtllm_fp8_block_scale_moe"
)
flashinfer_trtllm_fp8_per_tensor_scale_moe = _lazy_import_wrapper(
    "flashinfer.fused_moe", "trtllm_fp8_per_tensor_scale_moe"
)
flashinfer_cutlass_fused_moe = _lazy_import_wrapper(
    "flashinfer.fused_moe", "cutlass_fused_moe"
)
flashinfer_cutedsl_grouped_gemm_nt_masked = _lazy_import_wrapper(
    "flashinfer.cute_dsl.blockscaled_gemm", "grouped_gemm_nt_masked"
)
flashinfer_fp4_quantize = _lazy_import_wrapper("flashinfer", "fp4_quantize")
nvfp4_batched_quantize = _lazy_import_wrapper("flashinfer", "nvfp4_batched_quantize")
silu_and_mul_scaled_nvfp4_experts_quantize = _lazy_import_wrapper(
    "flashinfer", "silu_and_mul_scaled_nvfp4_experts_quantize"
)
scaled_fp4_grouped_quantize = _lazy_import_wrapper(
    "flashinfer", "scaled_fp4_grouped_quantize"
)
nvfp4_block_scale_interleave = _lazy_import_wrapper(
    "flashinfer.fp4_quantization", "block_scale_interleave"
)
flashinfer_cute_dsl_fused_moe_nvfp4 = _lazy_import_wrapper(
    "flashinfer", "cute_dsl_fused_moe_nvfp4"
)
flashinfer_convert_sf_to_mma_layout = _lazy_import_wrapper(
    "flashinfer.cute_dsl.utils", "convert_sf_to_mma_layout"
)
trtllm_fp4_block_scale_moe = _lazy_import_wrapper(
    "flashinfer", "trtllm_fp4_block_scale_moe"
)
# Special case for autotune since it returns a context manager
autotune = _lazy_import_wrapper(
    "flashinfer.autotuner",
    "autotune",
    fallback_fn=lambda *args, **kwargs: contextlib.nullcontext(),
)
_is_fi_autotuning: bool = False


@functools.cache
def has_flashinfer_comm() -> bool:
    """Return `True` if FlashInfer comm module is available."""
    return has_flashinfer() and importlib.util.find_spec("flashinfer.comm") is not None


@functools.cache
def has_flashinfer_nvlink_two_sided() -> bool:
    """Return `True` if FlashInfer mnnvl all2all is available."""
    if not has_flashinfer_comm():
        return False

    # Check if all required functions are available
    required_functions = [
        ("flashinfer.comm", "Mapping"),
        ("flashinfer.comm.mnnvl", "MnnvlMemory"),
        ("flashinfer.comm.trtllm_alltoall", "MnnvlMoe"),
        ("flashinfer.comm.trtllm_alltoall", "MoEAlltoallInfo"),
    ]

    for module_name, attr_name in required_functions:
        mod = _get_submodule(module_name)
        if not mod or not hasattr(mod, attr_name):
            return False
    return True


@functools.cache
def has_flashinfer_nvlink_one_sided() -> bool:
    """Return `True` if FlashInfer trtllm_moe_alltoall module is available."""
    if not has_flashinfer_comm():
        return False
    return importlib.util.find_spec("flashinfer.comm.trtllm_moe_alltoall") is not None


@functools.cache
def has_flashinfer_moe() -> bool:
    """Return `True` if FlashInfer MoE module is available."""
    return (
        has_flashinfer()
        and importlib.util.find_spec("flashinfer.fused_moe") is not None
    )


@functools.cache
def has_flashinfer_cutedsl() -> bool:
    """Return ``True`` if FlashInfer cutedsl module is available."""
    return (
        has_flashinfer() and importlib.util.find_spec("flashinfer.cute_dsl") is not None
    )


@functools.cache
def has_flashinfer_trtllm_fused_moe() -> bool:
    """Return `True` if FlashInfer TRTLLM fused MoE is available."""
    if not has_flashinfer_moe():
        return False
    required_functions = [
        ("flashinfer.fused_moe", "trtllm_fp8_block_scale_moe"),
        ("flashinfer.fused_moe", "trtllm_fp8_per_tensor_scale_moe"),
        ("flashinfer.fused_moe", "trtllm_fp4_block_scale_moe"),
        ("flashinfer.fused_moe", "trtllm_mxint4_block_scale_moe"),
        ("flashinfer.fused_moe", "trtllm_bf16_moe"),
    ]
    for module_name, attr_name in required_functions:
        mod = _get_submodule(module_name)
        if not mod or not hasattr(mod, attr_name):
            return False
    return True


@functools.cache
def has_flashinfer_trtllm_bf16_routed_moe() -> bool:
    """Return `True` if FlashInfer exposes the routed bf16 MoE entrypoint.

    The routed variant takes pre-dispatched topk_ids and is required by the
    modular bf16 trtllm-gen MoE path (which supports EPLB / all2all).
    """
    if not has_flashinfer_trtllm_fused_moe():
        return False
    mod = _get_submodule("flashinfer.fused_moe")
    return mod is not None and hasattr(mod, "trtllm_bf16_routed_moe")


@functools.cache
def has_flashinfer_cutlass_fused_moe() -> bool:
    """Return `True` if FlashInfer CUTLASS fused MoE is available."""
    if not has_flashinfer_moe():
        return False

    # Check if all required functions are available
    required_functions = [
        ("flashinfer.fused_moe", "cutlass_fused_moe"),
        ("flashinfer", "fp4_quantize"),
        ("flashinfer", "nvfp4_block_scale_interleave"),
        ("flashinfer.fused_moe", "trtllm_fp4_block_scale_moe"),
    ]

    for module_name, attr_name in required_functions:
        mod = _get_submodule(module_name)
        if not mod or not hasattr(mod, attr_name):
            return False
    return True


@functools.cache
def has_flashinfer_cutedsl_grouped_gemm_nt_masked() -> bool:
    """Return ``True`` if FlashInfer CUTLASS fused MoE is available."""
    if not has_flashinfer_cutedsl():
        return False

    # Check if all required functions are available
    required_functions = [
        ("flashinfer.cute_dsl.blockscaled_gemm", "grouped_gemm_nt_masked"),
        ("flashinfer", "scaled_fp4_grouped_quantize"),
        ("flashinfer", "silu_and_mul_scaled_nvfp4_experts_quantize"),
    ]

    for module_name, attr_name in required_functions:
        mod = _get_submodule(module_name)
        if not mod or not hasattr(mod, attr_name):
            return False
    return True


@functools.cache
def has_flashinfer_cutedsl_moe_nvfp4() -> bool:
    """Return ``True`` if FlashInfer cute_dsl_fused_moe_nvfp4 is available."""
    if not has_flashinfer_cutedsl():
        return False
    mod = _get_submodule("flashinfer")
    return mod is not None and hasattr(mod, "cute_dsl_fused_moe_nvfp4")


@functools.cache
def has_nvidia_artifactory() -> bool:
    """Return `True` if NVIDIA's artifactory is accessible.

    This checks connectivity to the kernel inference library artifactory
    which is required for downloading certain cubin kernels like TRTLLM FHMA.
    """
    # If we have pre-downloaded cubins, we can assume the cubins are available.
    if has_flashinfer_cubin():
        return True

    try:
        # Use a short timeout to avoid blocking for too long
        response = requests.get(FLASHINFER_CUBINS_REPOSITORY, timeout=5)
        accessible = response.status_code == 200
        if accessible:
            logger.debug_once("NVIDIA artifactory is accessible")
        else:
            logger.warning_once(
                "NVIDIA artifactory returned failed status code: %d",
                response.status_code,
            )
        return accessible
    except Exception as e:
        logger.warning_once("Failed to connect to NVIDIA artifactory: %s", e)
        return False


@functools.cache
def supports_trtllm_attention() -> bool:
    """
    TRTLLM attention is supported if the platform is SM100,
    NVIDIA artifactory is accessible, and batch-invariant mode is not enabled.
    """
    # Batch-invariant mode disables TRTLLM attention
    if envs.VLLM_BATCH_INVARIANT:
        return False

    # Requires SM100 and NVIDIA artifactory to be accessible to download cubins
    return (
        current_platform.is_device_capability_family(100) and has_nvidia_artifactory()
    )


def force_use_trtllm_attention() -> bool | None:
    """
    This function should only be called during initialization stage when vllm config
    is set.
    Return `None` if --attention-config.use_trtllm_attention is not set,
    return `True` if TRTLLM attention is forced to be used,
    return `False` if TRTLLM attention is forced to be not used.
    """
    from vllm.config import get_current_vllm_config

    vllm_config = get_current_vllm_config()
    return vllm_config.attention_config.use_trtllm_attention


def can_use_trtllm_attention(num_qo_heads: int, num_kv_heads: int) -> bool:
    """Check if the current configuration supports TRTLLM attention."""
    if force_use_trtllm_attention() is False:
        return False
    has_trtllm = supports_trtllm_attention()
    return has_trtllm and (num_qo_heads % num_kv_heads == 0)


def use_trtllm_attention(
    num_qo_heads: int,
    num_kv_heads: int,
    num_tokens: int,
    max_seq_len: int,
    dcp_world_size: int,
    kv_cache_dtype: str,
    q_dtype: torch.dtype,
    is_prefill: bool,
    # None means auto-detection, True means force on, False means force off
    force_use_trtllm: bool | None = None,
    has_sinks: bool = False,
    has_spec: bool = False,
) -> bool:
    """Return `True` if TRTLLM attention is used."""

    # CLI argument is set to 0 - respect it
    if force_use_trtllm is not None and not force_use_trtllm:
        return False

    # Decode context parallel is not supported
    if dcp_world_size > 1:
        logger.warning_once(
            "Trtllm does not support returning LSE and as a result "
            "does not support DCP, reverting to FlashInfer"
        )
        return False

    # The platform is not supported
    if not supports_trtllm_attention():
        if force_use_trtllm:
            logger.warning_once(
                "TRTLLM attention is not supported on this platform, "
                "but --attention-config.use_trtllm_attention is set to 1"
            )
        return False

    # The combination of query and key heads is not supported
    if num_qo_heads % num_kv_heads != 0:
        if force_use_trtllm:
            logger.warning_once(
                "TRTLLM attention is not supported for this combination of "
                "query and key heads, but --attention-config.use_trtllm_attention is "
                "set to 1"
            )
        return False

    if has_spec and not is_prefill:
        # Speculative decoding requires TRTLLM attention for decodes
        logger.info_once("Using TRTLLM attention (enabled for speculative decoding).")
        return True

    # Must use TRTLLM attention if query is FP8 quantized
    if q_dtype == current_platform.fp8_dtype():
        logger.info_once("Using TRTLLM attention (query is quantized).")
        return True

    # If sinks are being used, we must use TRTLLM attention as it's
    # the only backend that supports them
    if has_sinks:
        logger.info_once("Using TRTLLM attention (required for attention sinks).")
        return True

    if force_use_trtllm is None:
        # CLI argument not set - use auto-detection
        if is_prefill:
            # Prefill auto-detection
            use_trtllm = kv_cache_dtype == "auto"
            if use_trtllm:
                logger.warning_once("Using TRTLLM prefill attention (auto-detected).")
        else:
            # Decode auto-detection
            use_trtllm = num_tokens <= 256 and kv_cache_dtype == "auto"
            if use_trtllm:
                logger.warning_once("Using TRTLLM decode attention (auto-detected).")
        return use_trtllm

    # CLI argument is set to 1 - respect it
    logger.info_once(
        "Using TRTLLM attention (--attention-config.use_trtllm_attention is set to 1)"
    )
    return True


if has_flashinfer():
    from vllm.utils.torch_utils import direct_register_custom_op

    def _flashinfer_concat_mla_k(
        k: torch.Tensor,
        k_nope: torch.Tensor,
        k_pe: torch.Tensor,
    ) -> None:
        """Custom op wrapper for flashinfer's concat_mla_k.

        This is an in-place operation that concatenates k_nope and k_pe into k.

        The kernel is optimized for DeepSeek V3 dimensions:
        - num_heads=128
        - nope_dim=128
        - rope_dim=64

        Key optimizations:
        - Warp-based processing with software pipelining
        - Vectorized memory access (int2 for nope, int for rope)
        - L2 prefetching for next row while processing current
        - Register reuse for rope values across all heads

        Args:
            k: Output tensor, shape [num_tokens, num_heads, nope_dim + rope_dim].
                Modified in-place.
            k_nope: The nope part of k, shape [num_tokens, num_heads, nope_dim].
            k_pe: The rope part of k (shared), shape [num_tokens, 1, rope_dim].
                  This is broadcast to all heads.
        """
        from flashinfer.concat_ops import concat_mla_k

        concat_mla_k(k, k_nope, k_pe)

    def _flashinfer_concat_mla_k_fake(
        k: torch.Tensor,
        k_nope: torch.Tensor,
        k_pe: torch.Tensor,
    ) -> None:
        return

    # Register flashinfer concat_mla_k custom op
    direct_register_custom_op(
        op_name="flashinfer_concat_mla_k",
        op_func=_flashinfer_concat_mla_k,
        mutates_args=["k"],  # k tensor is modified in-place
        fake_impl=_flashinfer_concat_mla_k_fake,
    )

    @torch.library.custom_op(
        "vllm::flashinfer_mm_fp4",
        mutates_args=[],
        device_types="cuda",
    )
    def flashinfer_mm_fp4(
        A: torch.Tensor,
        B: torch.Tensor,
        A_scale: torch.Tensor,
        B_scale: torch.Tensor,
        g_scale: torch.Tensor,
        dtype: torch.dtype,
        use_8x4_sf_layout: bool,
        backend: str,
    ) -> torch.Tensor:
        from flashinfer import mm_fp4 as flashinfer_mm_fp4_

        return flashinfer_mm_fp4_(
            A,
            B,
            A_scale,
            B_scale,
            g_scale,
            dtype,
            block_size=16,
            use_8x4_sf_layout=use_8x4_sf_layout,
            backend=backend,
        )

    @torch.library.register_fake(
        "vllm::flashinfer_mm_fp4",
    )
    def flashinfer_mm_fp4_fake(
        A: torch.Tensor,
        B: torch.Tensor,
        A_scale: torch.Tensor,
        B_scale: torch.Tensor,
        g_scale: torch.Tensor,
        dtype: torch.dtype,
        use_8x4_sf_layout: bool,
        backend: str,
    ) -> torch.Tensor:
        return torch.empty(A.shape[0], B.shape[1], dtype=dtype, device=A.device)

    @torch.library.custom_op(
        "vllm::bmm_fp8",
        mutates_args=[],
        device_types="cuda",
    )
    def bmm_fp8(
        A: torch.Tensor,
        B: torch.Tensor,
        A_scale: torch.Tensor,
        B_scale: torch.Tensor,
        dtype: torch.dtype,
        backend: str,
    ) -> torch.Tensor:
        from flashinfer import bmm_fp8 as bmm_fp8_

        return bmm_fp8_(A, B, A_scale, B_scale, dtype, None, backend)

    @torch.library.register_fake(
        "vllm::bmm_fp8",
    )
    def bmm_fp8_fake(
        A: torch.Tensor,
        B: torch.Tensor,
        A_scale: torch.Tensor,
        B_scale: torch.Tensor,
        dtype: torch.dtype,
        backend: str,
    ) -> torch.Tensor:
        return torch.empty(
            A.shape[0], A.shape[1], B.shape[2], dtype=dtype, device=A.device
        )

    @torch.library.custom_op(
        "vllm::flashinfer_nvfp4_quantize",
        mutates_args=[],
        device_types="cuda",
    )
    def flashinfer_nvfp4_quantize(
        a: torch.Tensor, a_global_sf: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from flashinfer import SfLayout
        from flashinfer import nvfp4_quantize as nvfp4_quantize_

        return nvfp4_quantize_(
            a, a_global_sf, sfLayout=SfLayout.layout_8x4, do_shuffle=False
        )

    @torch.library.register_fake(
        "vllm::flashinfer_nvfp4_quantize",
    )
    def flashinfer_nvfp4_quantize_fake(
        a: torch.Tensor, a_global_sf: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        m, n = a.shape

        round_up = lambda x, y: (x + y - 1) // y * y

        rounded_m = round_up(m, 8)
        scale_n = n // 16
        rounded_n = round_up(scale_n, 4)

        return torch.empty(m, n // 2, dtype=torch.uint8, device=a.device), torch.empty(
            rounded_m, rounded_n, dtype=torch.uint8, device=a.device
        )

    @torch.library.custom_op(
        "vllm::mm_mxfp8",
        mutates_args=[],
        device_types="cuda",
    )
    def mm_mxfp8(
        A: torch.Tensor,
        B: torch.Tensor,
        A_scale: torch.Tensor,
        B_scale: torch.Tensor,
        out_dtype: torch.dtype,
        backend: str = "cutlass",
        use_8x4_sf_layout: bool = False,
    ) -> torch.Tensor:
        from flashinfer import mm_mxfp8 as mm_mxfp8_

        runtime_configuration = get_mxfp8_trtllm_file_configuration()
        layout_mode = (
            runtime_configuration.layout_mode
            if runtime_configuration is not None
            else os.environ.get(
                "VLLM_MXFP8_DENSE_A_SF_LAYOUT", "128x4"
            )
            .strip()
            .lower()
        )
        is_adaptive_layout = layout_mode in (
            "adaptive",
            "shape-aware",
            "shape_aware",
        )
        if is_adaptive_layout and backend != "trtllm":
            raise RuntimeError(
                f"MXFP8 adaptive layout requires backend='trtllm'; got {backend!r}"
            )
        use_direct_trtllm = is_adaptive_layout or (
            backend == "trtllm" and _mxfp8_env_flag("VLLM_MXFP8_DENSE_DIRECT_TRTLLM")
        )
        if use_direct_trtllm:
            require_direct = is_adaptive_layout or _mxfp8_env_flag(
                "VLLM_MXFP8_DENSE_REQUIRE_DIRECT_TRTLLM"
            )
            try:
                state = _require_mxfp8_trtllm_direct_state(A.device)
                configuration = state.configuration
                tactic = (
                    configuration.default_tactic
                    if configuration is not None
                    else int(os.environ.get("VLLM_MXFP8_DENSE_TRTLLM_TACTIC", "-1"))
                )
                tactic_hints_name = (
                    "VLLM_MXFP8_DENSE_TRTLLM_TACTIC_HINTS"
                    if use_8x4_sf_layout
                    else "VLLM_MXFP8_DENSE_TRTLLM_TACTIC_HINTS_128X4"
                )
                if configuration is None:
                    tactic_table = _parse_mxfp8_tactic_hints(
                        os.environ.get(tactic_hints_name, "")
                    )
                else:
                    tactic_table = dict(
                        configuration.low_tactic_map
                        if use_8x4_sf_layout
                        else configuration.high_tactic_map
                    )
                m = int(A.shape[0])
                k = int(A.shape[1])
                n = int(B.shape[1])
                tactic = tactic_table.get((m, n, k), tactic)
                workspace = (
                    state.workspace_8x4 if use_8x4_sf_layout else state.workspace_128x4
                )
                return _mxfp8_trtllm_run_prepared(
                    A,
                    B,
                    A_scale,
                    B_scale,
                    out_dtype,
                    workspace,
                    use_8x4_sf_layout,
                    tactic,
                )
            except Exception:
                if require_direct:
                    raise

        if is_adaptive_layout:
            raise RuntimeError(
                "MXFP8 adaptive layout cannot fall back from direct TRTLLM"
            )
        return mm_mxfp8_(
            A,
            B,
            A_scale,
            B_scale,
            out=None,
            out_dtype=out_dtype,
            backend=backend,
        )

    @torch.library.register_fake(
        "vllm::mm_mxfp8",
    )
    def mm_mxfp8_fake(
        A: torch.Tensor,
        B: torch.Tensor,
        A_scale: torch.Tensor,
        B_scale: torch.Tensor,
        out_dtype: torch.dtype,
        backend: str = "cutlass",
        use_8x4_sf_layout: bool = False,
    ) -> torch.Tensor:
        # A is [m, k], B is [k, n] -> output [m, n]
        return torch.empty(A.shape[0], B.shape[1], dtype=out_dtype, device=A.device)

    @torch.library.custom_op(
        "vllm::mxfp8_adaptive_quantize_mm_marker",
        mutates_args=["workspace_8x4", "workspace_128x4"],
        device_types="cuda",
    )
    def mxfp8_adaptive_quantize_mm_marker(
        A_bf16: torch.Tensor,
        B: torch.Tensor,
        B_scale: torch.Tensor,
        out_dtype: torch.dtype,
        backend: str,
        workspace_8x4: torch.Tensor,
        workspace_128x4: torch.Tensor,
    ) -> torch.Tensor:
        raise RuntimeError(
            "MXFP8 adaptive quantize-MM marker must be specialized before execution"
        )

    @torch.library.register_fake(
        "vllm::mxfp8_adaptive_quantize_mm_marker",
    )
    def mxfp8_adaptive_quantize_mm_marker_fake(
        A_bf16: torch.Tensor,
        B: torch.Tensor,
        B_scale: torch.Tensor,
        out_dtype: torch.dtype,
        backend: str,
        workspace_8x4: torch.Tensor,
        workspace_128x4: torch.Tensor,
    ) -> torch.Tensor:
        return torch.empty(
            A_bf16.shape[0], B.shape[1], dtype=out_dtype, device=A_bf16.device
        )

    @torch.library.custom_op(
        "vllm::mxfp8_quantize_mm_8x4",
        mutates_args=["workspace_8x4", "workspace_128x4"],
        device_types="cuda",
    )
    def mxfp8_quantize_mm_8x4(
        A_bf16: torch.Tensor,
        B: torch.Tensor,
        B_scale: torch.Tensor,
        out_dtype: torch.dtype,
        backend: str,
        workspace_8x4: torch.Tensor,
        workspace_128x4: torch.Tensor,
    ) -> torch.Tensor:
        return _mxfp8_quantize_mm_fixed_layout_impl(
            A_bf16,
            B,
            B_scale,
            out_dtype,
            backend,
            workspace_8x4,
            workspace_128x4,
            use_8x4_sf_layout=True,
        )

    @torch.library.register_fake("vllm::mxfp8_quantize_mm_8x4")
    def mxfp8_quantize_mm_8x4_fake(
        A_bf16: torch.Tensor,
        B: torch.Tensor,
        B_scale: torch.Tensor,
        out_dtype: torch.dtype,
        backend: str,
        workspace_8x4: torch.Tensor,
        workspace_128x4: torch.Tensor,
    ) -> torch.Tensor:
        return torch.empty(
            A_bf16.shape[0], B.shape[1], dtype=out_dtype, device=A_bf16.device
        )

    @torch.library.custom_op(
        "vllm::mxfp8_quantize_mm_128x4",
        mutates_args=["workspace_8x4", "workspace_128x4"],
        device_types="cuda",
    )
    def mxfp8_quantize_mm_128x4(
        A_bf16: torch.Tensor,
        B: torch.Tensor,
        B_scale: torch.Tensor,
        out_dtype: torch.dtype,
        backend: str,
        workspace_8x4: torch.Tensor,
        workspace_128x4: torch.Tensor,
    ) -> torch.Tensor:
        return _mxfp8_quantize_mm_fixed_layout_impl(
            A_bf16,
            B,
            B_scale,
            out_dtype,
            backend,
            workspace_8x4,
            workspace_128x4,
            use_8x4_sf_layout=False,
        )

    @torch.library.register_fake("vllm::mxfp8_quantize_mm_128x4")
    def mxfp8_quantize_mm_128x4_fake(
        A_bf16: torch.Tensor,
        B: torch.Tensor,
        B_scale: torch.Tensor,
        out_dtype: torch.dtype,
        backend: str,
        workspace_8x4: torch.Tensor,
        workspace_128x4: torch.Tensor,
    ) -> torch.Tensor:
        return torch.empty(
            A_bf16.shape[0], B.shape[1], dtype=out_dtype, device=A_bf16.device
        )


def flashinfer_mm_mxfp8(
    a: torch.Tensor,
    b: torch.Tensor,
    block_scale_a: torch.Tensor,
    block_scale_b: torch.Tensor,
    out_dtype: torch.dtype,
    backend: str = "cutlass",
) -> torch.Tensor:
    """MXFP8 MM helper - mirrors flashinfer_scaled_fp4_mm API.

    Takes non-transposed weights and handles transpose internally.

    CRITICAL: mm_mxfp8 CUTLASS kernel requires SWIZZLED 1D scales for optimal
    performance and accuracy. Both input and weight scales should be in
    swizzled format from FlashInfer's mxfp8_quantize(is_sf_swizzled_layout=True).
    """
    # a shape [M, K]
    # b shape [K, N]
    assert a.ndim == 2 and b.ndim == 2
    assert a.shape[1] == b.shape[1]  # K dimension must match

    if block_scale_b.ndim != 1:
        raise ValueError(
            "mm_mxfp8 expects 1D swizzled weight scales for CUTLASS; "
            f"got shape={tuple(block_scale_b.shape)}"
        )

    # Output tensor [M, N]
    return mm_mxfp8(
        a,
        b.t(),  # Transpose weight: [N, K] -> [K, N]
        block_scale_a,
        block_scale_b,
        out_dtype,
        backend=backend,
    )


def flashinfer_scaled_fp4_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    block_scale_a: torch.Tensor,
    block_scale_b: torch.Tensor,
    alpha: torch.Tensor,
    out_dtype: torch.dtype,
    backend: str,
) -> torch.Tensor:
    assert a.ndim == 2 and b.ndim == 2
    assert block_scale_a.ndim == 2 and block_scale_b.ndim == 2
    assert a.stride(-1) == 1 and b.stride(-1) == 1
    assert a.shape[1] == b.shape[1]

    if backend in ("cutlass", "cudnn"):
        block_scale_a = block_scale_a.view(torch.uint8)
        block_scale_b = block_scale_b.view(torch.uint8)

    use_8x4_sf_layout = True if backend == "trtllm" and a.shape[0] <= 32 else False  # noqa: SIM210

    return flashinfer_mm_fp4(
        a,
        b.t(),
        block_scale_a,
        block_scale_b.t(),
        alpha,
        out_dtype,
        use_8x4_sf_layout=use_8x4_sf_layout,
        backend=backend,
    )


def flashinfer_scaled_fp8_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    assert a.ndim == 2 and b.ndim == 2
    assert a.shape[1] == b.shape[0]
    assert scale_a.numel() == 1 and scale_b.numel() == 1
    assert a.dtype == torch.float8_e4m3fn and b.dtype == torch.float8_e4m3fn
    assert a.device.type == "cuda" and b.device.type == "cuda"
    assert scale_a.dtype == torch.float32 and scale_b.dtype == torch.float32
    assert scale_a.device.type == "cuda" and scale_b.device.type == "cuda"

    output = bmm_fp8(
        a.unsqueeze(0),
        b.unsqueeze(0),
        scale_a,
        scale_b,
        out_dtype,
        "auto",
    ).view(a.shape[0], b.shape[1])

    if bias is not None:
        output = output + bias
    return output


def flashinfer_quant_nvfp4_8x4_sf_layout(
    a: torch.Tensor, a_global_sf: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    return flashinfer_nvfp4_quantize(a, a_global_sf)


flashinfer_fp8_blockscale_gemm = _lazy_import_wrapper(
    "flashinfer.gemm", "fp8_blockscale_gemm_sm90"
)


@functools.cache
def has_flashinfer_fp8_blockscale_gemm() -> bool:
    """Return `True` if FlashInfer block-scale FP8 GEMM is available."""
    return (
        has_flashinfer()
        and current_platform.is_device_capability(90)
        and hasattr(_get_submodule("flashinfer.gemm"), "fp8_blockscale_gemm_sm90")
    )


@functools.cache
def is_flashinfer_fp8_blockscale_gemm_supported() -> bool:
    """Return `True` if FlashInfer block-scale FP8 GEMM is supported."""
    return (
        envs.VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER
        and has_flashinfer_fp8_blockscale_gemm()
    )


def should_use_flashinfer_for_blockscale_fp8_gemm(
    is_flashinfer_supported: bool,
    output_dtype: torch.dtype,
    input_dtype: torch.dtype,
    weight_dtype: torch.dtype,
    weight_shape: tuple[int, int],
):
    if not is_flashinfer_supported:
        return False

    # Verify DeepGEMM N/K dims requirements
    # NOTE: Also synchronized with test_w8a8_block_fp8_deep_gemm_matmul
    # test inside kernels/quantization/test_block_fp8.py
    N_MULTIPLE = 64
    K_MULTIPLE = 128

    should_use_flashinfer = (
        output_dtype == torch.bfloat16
        and input_dtype == torch.bfloat16
        and weight_dtype == torch.float8_e4m3fn
        and weight_shape[0] % N_MULTIPLE == 0
        and weight_shape[1] % K_MULTIPLE == 0
    )

    return should_use_flashinfer


__all__ = [
    "has_flashinfer",
    "flashinfer_trtllm_fp8_block_scale_moe",
    "flashinfer_cutlass_fused_moe",
    "flashinfer_cutedsl_grouped_gemm_nt_masked",
    "flashinfer_fp4_quantize",
    "silu_and_mul_scaled_nvfp4_experts_quantize",
    "scaled_fp4_grouped_quantize",
    "nvfp4_block_scale_interleave",
    "flashinfer_cute_dsl_fused_moe_nvfp4",
    "flashinfer_convert_sf_to_mma_layout",
    "trtllm_fp4_block_scale_moe",
    "autotune",
    "has_flashinfer_moe",
    "has_flashinfer_comm",
    "has_flashinfer_nvlink_two_sided",
    "has_flashinfer_nvlink_one_sided",
    "has_flashinfer_cutlass_fused_moe",
    "has_flashinfer_cutedsl_grouped_gemm_nt_masked",
    "has_flashinfer_cutedsl_moe_nvfp4",
    "has_flashinfer_fp8_blockscale_gemm",
    "has_nvidia_artifactory",
    "supports_trtllm_attention",
    "can_use_trtllm_attention",
    "use_trtllm_attention",
    "flashinfer_scaled_fp4_mm",
    "flashinfer_scaled_fp8_mm",
    "flashinfer_quant_nvfp4_8x4_sf_layout",
    "flashinfer_fp8_blockscale_gemm",
    "should_use_flashinfer_for_blockscale_fp8_gemm",
    "is_flashinfer_fp8_blockscale_gemm_supported",
]
