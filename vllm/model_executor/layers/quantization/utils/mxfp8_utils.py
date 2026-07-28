# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import atexit
import hashlib
import json
import os
import socket
import threading
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, fields
from dataclasses import field as dataclass_field
from functools import cache
from pathlib import Path
from types import MappingProxyType
from typing import Any, NamedTuple

import torch

from vllm import envs
from vllm.compilation.passes.inductor_pass import InductorPass, get_pass_context
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.mxfp8_tactic_table import (
    Mxfp8TacticArtifact,
    Mxfp8TacticKey,
    RuntimeProvenance,
    load_mxfp8_tactic_artifact,
)
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)

# MXFP8 constants
MXFP8_VALUE_DTYPE = torch.float8_e4m3fn
MXFP8_SCALE_DTYPE = torch.uint8
MXFP8_BLOCK_SIZE = 32
MXFP8_TRTLLM_8X4_MAX_M = 256
MXFP8_TRTLLM_LAYOUT_ENV = "VLLM_MXFP8_DENSE_TRTLLM_LAYOUT"
MXFP8_TRTLLM_SWITCH_M_ENV = "VLLM_MXFP8_DENSE_TRTLLM_SWITCH_M"
MXFP8_TRTLLM_HIGH_M_TACTIC_ENV = "VLLM_MXFP8_DENSE_TRTLLM_TACTIC"
MXFP8_TRTLLM_HIGH_M_TACTIC_HINTS_ENV = "VLLM_MXFP8_DENSE_TRTLLM_TACTIC_HINTS_128X4"
_MXFP8_TRTLLM_UNRESOLVED_TACTIC = -2


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


class _Mxfp8TrtllmTacticState(NamedTuple):
    artifact: Mxfp8TacticArtifact | None
    runner_8x4: Any
    runner_128x4: Any
    workspace_8x4: torch.Tensor
    workspace_128x4: torch.Tensor
    resolved_tactics: dict[Mxfp8TacticKey, int]
    resolution_lock: threading.RLock


@dataclass(frozen=True)
class _Mxfp8TrtllmTacticSpecialization:
    fingerprint: str
    tactics: Mapping[Mxfp8TacticKey, tuple[int, str]]


class _Mxfp8RunnerProfile:
    def __init__(self, key: Mxfp8TacticKey) -> None:
        self._shapes = (
            (key.m_logical, key.k_physical),
            (key.k_physical, key.n_physical),
        )

    def get_opt_shapes(self) -> tuple[tuple[int, int], tuple[int, int]]:
        return self._shapes


@dataclass
class _Mxfp8TacticAudit:
    output_dir: Path | None
    expected_rank_count: int
    rank: int
    host: str
    pid: int
    registered_keys: dict[Mxfp8TacticKey, dict[str, Any]]
    rejected_artifact_reasons: list[str]
    hits: int = 0
    misses: int = 0
    defaults: int = 0
    complete: bool = False
    _lock: threading.RLock = dataclass_field(
        default_factory=threading.RLock,
        init=False,
        repr=False,
        compare=False,
    )

    def reject_artifact(self, reason: str) -> None:
        with self._lock:
            if reason not in self.rejected_artifact_reasons:
                self.rejected_artifact_reasons.append(reason)
                self.write()

    def register(
        self,
        key: Mxfp8TacticKey,
        *,
        selected_tactic: int,
        tactic_source: str,
        requested_tactic: int | None,
        artifact_lookup_hit: bool | None,
    ) -> None:
        with self._lock:
            if key in self.registered_keys:
                return
            if artifact_lookup_hit is True:
                self.hits += 1
            elif artifact_lookup_hit is False:
                self.misses += 1
            if selected_tactic == -1:
                self.defaults += 1
            self.registered_keys[key] = {
                **asdict(key),
                "artifact_lookup": (
                    "hit"
                    if artifact_lookup_hit is True
                    else "miss"
                    if artifact_lookup_hit is False
                    else None
                ),
                "requested_tactic": requested_tactic,
                "selected_tactic": selected_tactic,
                "tactic_source": tactic_source,
            }
            self.write()

    def finalize(self) -> None:
        with self._lock:
            self.complete = True
            self.write()

    def write(self) -> None:
        with self._lock:
            if self.output_dir is None:
                return
            self.output_dir.mkdir(parents=True, exist_ok=True)
            worker_name = f"rank-{self.rank}-pid-{self.pid}"
            temporary = self.output_dir / f"{worker_name}.json.tmp"
            destination = self.output_dir / f"{worker_name}.json"
            payload = {
                "complete": self.complete,
                "defaults": self.defaults,
                "expected_rank_count": self.expected_rank_count,
                "hits": self.hits,
                "host": self.host,
                "misses": self.misses,
                "pid": self.pid,
                "rank": self.rank,
                "registered_keys": sorted(
                    self.registered_keys.values(),
                    key=lambda row: (
                        row["m_logical"],
                        row["n_logical"],
                        row["k_logical"],
                        row["n_physical"],
                        row["k_physical"],
                        row["activation_scale_layout"],
                        row["output_dtype"],
                    ),
                ),
                "rejected_artifact_reasons": self.rejected_artifact_reasons,
            }
            with temporary.open("w", encoding="utf-8") as stream:
                json.dump(payload, stream, indent=2, sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)


_MXFP8_TRTLLM_STATE_LOCK = threading.RLock()
_MXFP8_TRTLLM_STATES: dict[tuple[str, int], _Mxfp8TrtllmTacticState] = {}
_MXFP8_TRTLLM_STATES_BY_DEVICE_INDEX: list[_Mxfp8TrtllmTacticState | None] = []
_MXFP8_TRTLLM_SPECIALIZATIONS_BY_DEVICE_INDEX: list[
    _Mxfp8TrtllmTacticSpecialization | None
] = []
_MXFP8_TACTIC_SOURCES: dict[
    tuple[int, Mxfp8TacticKey],
    tuple[str, int | None, bool | None],
] = {}
_MXFP8_RUNTIME_PROVENANCE: RuntimeProvenance | None = None
_MXFP8_ARTIFACT_CONFIGURATION_PRESENT = False
_MXFP8_ARTIFACT_REJECTION_WARNED = False
_MXFP8_LEGACY_TACTIC_HINTS: dict[tuple[int, int, int], int] = {}
_MXFP8_LEGACY_FALLBACK_TACTIC = -1
_MXFP8_LEGACY_GLOBAL_FALLBACK = False
_MXFP8_TACTIC_AUDIT: _Mxfp8TacticAudit | None = None
_MXFP8_TRTLLM_TRACE_CALLBACK: Callable[..., None] | None = None


def _mxfp8_cuda_device_key(device: torch.device) -> tuple[str, int]:
    canonical = torch.device(device)
    if canonical.type != "cuda":
        raise RuntimeError(f"MXFP8 TRTLLM tactics require CUDA, got {canonical}.")
    index = canonical.index
    if index is None:
        index = torch.cuda.current_device()
    return canonical.type, index


def _runtime_provenance_from_file(
    path: Path,
    expected_sha256: str,
) -> RuntimeProvenance:
    try:
        contents = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"Unable to read MXFP8 runtime provenance {path}.") from exc
    if hashlib.sha256(contents).hexdigest() != expected_sha256:
        raise ValueError(
            "MXFP8 runtime provenance SHA256 does not match configuration."
        )
    try:
        raw = json.loads(contents)
    except json.JSONDecodeError as exc:
        raise ValueError("MXFP8 runtime provenance is not valid JSON.") from exc
    if not isinstance(raw, dict):
        raise ValueError("MXFP8 runtime provenance root must be an object.")

    values: dict[str, Any] = {}
    for field in fields(RuntimeProvenance):
        value = raw.get(field.name)
        if field.name == "adaptive_switch_m":
            if type(value) is not int or value <= 0:
                raise ValueError(
                    "MXFP8 runtime provenance adaptive_switch_m must be positive."
                )
        elif not isinstance(value, str) or not value:
            raise ValueError(
                f"MXFP8 runtime provenance {field.name} must be non-empty."
            )
        values[field.name] = value
    return RuntimeProvenance(**values)


def _load_configured_mxfp8_tactic_artifact() -> tuple[
    Mxfp8TacticArtifact | None, RuntimeProvenance | None, str | None
]:
    table_path = envs.VLLM_MXFP8_DENSE_TRTLLM_TACTIC_TABLE_PATH
    table_sha256 = envs.VLLM_MXFP8_DENSE_TRTLLM_TACTIC_TABLE_SHA256
    provenance_path = envs.VLLM_MXFP8_DENSE_TRTLLM_RUNTIME_PROVENANCE_PATH
    provenance_sha256 = envs.VLLM_MXFP8_DENSE_TRTLLM_RUNTIME_PROVENANCE_SHA256
    table_configured = bool(table_path or table_sha256)
    provenance_configured = bool(provenance_path or provenance_sha256)
    global _MXFP8_ARTIFACT_CONFIGURATION_PRESENT
    _MXFP8_ARTIFACT_CONFIGURATION_PRESENT = table_configured
    if not table_configured and not provenance_configured:
        return None, None, None

    expected: RuntimeProvenance | None = None
    if provenance_path and provenance_sha256:
        try:
            expected = _runtime_provenance_from_file(
                Path(provenance_path),
                provenance_sha256,
            )
        except ValueError as exc:
            return None, None, str(exc)
    elif provenance_path or provenance_sha256:
        return (
            None,
            None,
            "incomplete runtime provenance configuration",
        )

    if not table_configured:
        return None, expected, None
    if not table_path or not table_sha256 or expected is None:
        return (
            None,
            expected,
            "incomplete tactic table or runtime provenance configuration",
        )

    try:
        artifact = load_mxfp8_tactic_artifact(
            Path(table_path),
            table_sha256,
            expected,
        )
    except ValueError as exc:
        safe_disable_errors = (
            "Unable to read MXFP8 tactic artifact ",
            "MXFP8 tactic artifact SHA256 does not match configuration.",
            "MXFP8 tactic artifact provenance does not match runtime for ",
            "MXFP8 tactic artifact provenance does not match worker-local runtime for ",
            "Unable to query MXFP8 worker-local runtime provenance.",
        )
        if not str(exc).startswith(safe_disable_errors):
            raise
        return None, expected, str(exc)
    return artifact, expected, None


def _new_mxfp8_tactic_audit() -> _Mxfp8TacticAudit:
    configured_path = envs.VLLM_MXFP8_DENSE_TACTIC_AUDIT_PATH
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        expected_rank_count = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
    else:
        expected_rank_count = int(os.environ.get("WORLD_SIZE", "1"))
        rank = int(os.environ.get("RANK", "0"))
    return _Mxfp8TacticAudit(
        output_dir=Path(configured_path) if configured_path else None,
        expected_rank_count=expected_rank_count,
        rank=rank,
        host=socket.gethostname(),
        pid=os.getpid(),
        registered_keys={},
        rejected_artifact_reasons=[],
    )


def finalize_mxfp8_trtllm_tactic_audit() -> None:
    with _MXFP8_TRTLLM_STATE_LOCK:
        audit = _MXFP8_TACTIC_AUDIT
    if audit is None:
        return
    audit.finalize()


def _mxfp8_tactic_rows(
    tactics: dict[Mxfp8TacticKey, int],
    state: _Mxfp8TrtllmTacticState,
) -> list[dict[str, Any]]:
    rows = []
    for key, selected_tactic in tactics.items():
        tactic_source, requested_tactic, artifact_lookup_hit = (
            _MXFP8_TACTIC_SOURCES.get(
                (id(state), key),
                ("pre_resolved", selected_tactic, None),
            )
        )
        rows.append(
            {
                "key": asdict(key),
                "selected_tactic": selected_tactic,
                "tactic_source": tactic_source,
                "requested_tactic": requested_tactic,
                "artifact_lookup_hit": artifact_lookup_hit,
            }
        )
    rows.sort(key=lambda row: json.dumps(row["key"], sort_keys=True))
    return rows


def _mxfp8_artifact_payload(
    artifact: Mxfp8TacticArtifact | None,
) -> dict[str, Any] | None:
    if artifact is None:
        return None
    entries = [
        {
            "key": asdict(key),
            "selected_tactic": selected_tactic,
        }
        for key, selected_tactic in artifact.tactics.items()
    ]
    entries.sort(key=lambda row: json.dumps(row["key"], sort_keys=True))
    return {
        "provenance": asdict(artifact.provenance),
        "entries": entries,
    }


def _mxfp8_tactic_specialization_fingerprint(
    state: _Mxfp8TrtllmTacticState,
    resolved_tactics: dict[Mxfp8TacticKey, int],
) -> str:
    payload = {
        "schema_version": 1,
        "artifact_configuration_present": _MXFP8_ARTIFACT_CONFIGURATION_PRESENT,
        "artifact_sha256": (envs.VLLM_MXFP8_DENSE_TRTLLM_TACTIC_TABLE_SHA256),
        "runtime_provenance_sha256": (
            envs.VLLM_MXFP8_DENSE_TRTLLM_RUNTIME_PROVENANCE_SHA256
        ),
        "runtime_provenance": (
            asdict(_MXFP8_RUNTIME_PROVENANCE)
            if _MXFP8_RUNTIME_PROVENANCE is not None
            else None
        ),
        "accepted_artifact": _mxfp8_artifact_payload(state.artifact),
        "resolved_legality": _mxfp8_tactic_rows(resolved_tactics, state),
        "legacy_fallback_tactic": _MXFP8_LEGACY_FALLBACK_TACTIC,
        "legacy_global_fallback": _MXFP8_LEGACY_GLOBAL_FALLBACK,
        "legacy_tactic_hints": sorted(
            (*shape, tactic) for shape, tactic in _MXFP8_LEGACY_TACTIC_HINTS.items()
        ),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def finalize_mxfp8_trtllm_tactic_specialization() -> None:
    """Freeze worker-validated static tactics for compilation and capture."""
    global _MXFP8_TRTLLM_SPECIALIZATIONS_BY_DEVICE_INDEX
    with _MXFP8_TRTLLM_STATE_LOCK:
        specializations: list[_Mxfp8TrtllmTacticSpecialization | None] = []
        for state in _MXFP8_TRTLLM_STATES_BY_DEVICE_INDEX:
            if state is None:
                specializations.append(None)
                continue
            with state.resolution_lock:
                resolved_tactics = dict(state.resolved_tactics)
                tactics = {
                    key: (
                        selected_tactic,
                        _MXFP8_TACTIC_SOURCES.get(
                            (id(state), key),
                            ("pre_resolved", selected_tactic, None),
                        )[0],
                    )
                    for key, selected_tactic in resolved_tactics.items()
                }
                fingerprint = _mxfp8_tactic_specialization_fingerprint(
                    state,
                    resolved_tactics,
                )
            specializations.append(
                _Mxfp8TrtllmTacticSpecialization(
                    fingerprint=fingerprint,
                    tactics=MappingProxyType(tactics),
                )
            )
        _MXFP8_TRTLLM_SPECIALIZATIONS_BY_DEVICE_INDEX = specializations


def mxfp8_trtllm_specialization_fingerprint(device: torch.device) -> str:
    device_index = _mxfp8_cuda_device_key(device)[1]
    specialization = (
        _MXFP8_TRTLLM_SPECIALIZATIONS_BY_DEVICE_INDEX[device_index]
        if 0 <= device_index < len(_MXFP8_TRTLLM_SPECIALIZATIONS_BY_DEVICE_INDEX)
        else None
    )
    if specialization is None:
        raise RuntimeError(
            "MXFP8 tactic specialization must be finalized before CUDA Graph capture."
        )
    return specialization.fingerprint


def mxfp8_trtllm_resolved_binding(
    x: torch.Tensor,
    weight: torch.Tensor,
    output_features: int,
    *,
    use_8x4_sf_layout: bool,
) -> tuple[int, str] | None:
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("MXFP8 tactic binding lookup is not allowed during capture.")
    state = _require_mxfp8_trtllm_tactic_state(x.device)
    key = Mxfp8TacticKey(
        m_logical=int(x.shape[0]),
        n_logical=int(output_features),
        k_logical=int(x.shape[1]),
        n_physical=int(weight.shape[0]),
        k_physical=int(weight.shape[1]),
        activation_scale_layout=("8x4" if use_8x4_sf_layout else "128x4"),
        output_dtype="bfloat16",
    )
    with state.resolution_lock:
        selected_tactic = state.resolved_tactics.get(key)
        if selected_tactic is None:
            return None
        tactic_source = _MXFP8_TACTIC_SOURCES.get(
            (id(state), key),
            ("pre_resolved", selected_tactic, None),
        )[0]
    return selected_tactic, tactic_source


def prewarm_mxfp8_trtllm_tactic_specializations(
    model_runner: Any,
    **dummy_run_kwargs: Any,
) -> None:
    """Resolve exact static contracts eagerly before the first compilation."""
    with _MXFP8_TRTLLM_STATE_LOCK:
        has_prepared_state = any(
            state is not None for state in _MXFP8_TRTLLM_STATES_BY_DEVICE_INDEX
        )
    if not has_prepared_state:
        return

    compile_sizes = model_runner.compilation_config.compile_sizes or []
    for size in sorted(set(compile_sizes), reverse=True):
        model_runner._dummy_run(
            size,
            **dummy_run_kwargs,
            skip_eplb=True,
            is_profile=True,
            skip_compiled_for_mxfp8_prewarm=True,
        )
    finalize_mxfp8_trtllm_tactic_specialization()


atexit.register(finalize_mxfp8_trtllm_tactic_audit)


def register_mxfp8_trtllm_trace_callback(
    callback: Callable[..., None],
) -> None:
    global _MXFP8_TRTLLM_TRACE_CALLBACK
    _MXFP8_TRTLLM_TRACE_CALLBACK = callback


def prepare_mxfp8_trtllm_tactic_state(
    device: torch.device,
) -> _Mxfp8TrtllmTacticState:
    device_key = _mxfp8_cuda_device_key(device)
    with _MXFP8_TRTLLM_STATE_LOCK:
        return _prepare_mxfp8_trtllm_tactic_state_locked(device, device_key)


def _prepare_mxfp8_trtllm_tactic_state_locked(
    device: torch.device,
    device_key: tuple[str, int],
) -> _Mxfp8TrtllmTacticState:
    global _MXFP8_ARTIFACT_REJECTION_WARNED
    global _MXFP8_LEGACY_FALLBACK_TACTIC
    global _MXFP8_LEGACY_GLOBAL_FALLBACK
    global _MXFP8_LEGACY_TACTIC_HINTS
    global _MXFP8_RUNTIME_PROVENANCE
    global _MXFP8_TACTIC_AUDIT
    global _MXFP8_TRTLLM_SPECIALIZATIONS_BY_DEVICE_INDEX
    existing = _MXFP8_TRTLLM_STATES.get(device_key)
    if existing is not None:
        return existing
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "MXFP8 TRTLLM tactic state must be prepared before CUDA Graph capture."
        )

    artifact, runtime_provenance, rejection = _load_configured_mxfp8_tactic_artifact()
    _MXFP8_RUNTIME_PROVENANCE = runtime_provenance
    if _MXFP8_TACTIC_AUDIT is None:
        _MXFP8_TACTIC_AUDIT = _new_mxfp8_tactic_audit()
    if rejection is not None:
        _MXFP8_TACTIC_AUDIT.reject_artifact(rejection)
        if not _MXFP8_ARTIFACT_REJECTION_WARNED:
            _MXFP8_ARTIFACT_REJECTION_WARNED = True
            logger.warning(
                "Disabling explicit MXFP8 tactics and using direct TRTLLM "
                "tactic=-1: %s",
                rejection,
            )

    fallback_tactic = int(os.environ.get(MXFP8_TRTLLM_HIGH_M_TACTIC_ENV, "-1"))
    if fallback_tactic < -1:
        raise ValueError(f"{MXFP8_TRTLLM_HIGH_M_TACTIC_ENV} must be -1 or greater.")
    _MXFP8_LEGACY_FALLBACK_TACTIC = fallback_tactic
    _MXFP8_LEGACY_TACTIC_HINTS = _parse_mxfp8_tactic_hints(
        os.environ.get(MXFP8_TRTLLM_HIGH_M_TACTIC_HINTS_ENV, "")
    )
    _MXFP8_LEGACY_GLOBAL_FALLBACK = MXFP8_TRTLLM_HIGH_M_TACTIC_ENV in os.environ

    from flashinfer.gemm.gemm_base import (
        DEFAULT_WORKSPACE_SIZE,
        _get_cache_buf,
        get_trtllm_gemm_module,
    )

    canonical = torch.device(device_key[0], device_key[1])
    with torch.cuda.device(canonical):
        workspace_8x4 = _get_cache_buf(
            "vllm_mxfp8_trtllm_tactic_workspace_8x4",
            DEFAULT_WORKSPACE_SIZE,
            canonical,
        )
        workspace_128x4 = _get_cache_buf(
            "vllm_mxfp8_trtllm_tactic_workspace_128x4",
            DEFAULT_WORKSPACE_SIZE,
            canonical,
        )
        module = get_trtllm_gemm_module()
        runner_8x4 = module.trtllm_mxfp8_gemm_runner(use_8x4_sf_layout=True)
        runner_128x4 = module.trtllm_mxfp8_gemm_runner(use_8x4_sf_layout=False)

    state = _Mxfp8TrtllmTacticState(
        artifact=artifact,
        runner_8x4=runner_8x4,
        runner_128x4=runner_128x4,
        workspace_8x4=workspace_8x4,
        workspace_128x4=workspace_128x4,
        resolved_tactics={},
        resolution_lock=threading.RLock(),
    )
    _MXFP8_TRTLLM_STATES[device_key] = state
    device_index = device_key[1]
    if device_index >= len(_MXFP8_TRTLLM_STATES_BY_DEVICE_INDEX):
        _MXFP8_TRTLLM_STATES_BY_DEVICE_INDEX.extend(
            [None] * (device_index + 1 - len(_MXFP8_TRTLLM_STATES_BY_DEVICE_INDEX))
        )
    _MXFP8_TRTLLM_STATES_BY_DEVICE_INDEX[device_index] = state
    _MXFP8_TRTLLM_SPECIALIZATIONS_BY_DEVICE_INDEX = [
        None for _ in _MXFP8_TRTLLM_STATES_BY_DEVICE_INDEX
    ]
    return state


def prepare_mxfp8_trtllm_high_m_tactic_state(
    device: torch.device,
) -> _Mxfp8TrtllmTacticState | None:
    if not mxfp8_trtllm_high_m_static_tactics_enabled():
        return None
    return prepare_mxfp8_trtllm_tactic_state(device)


def _require_mxfp8_trtllm_tactic_state(
    device: torch.device,
) -> _Mxfp8TrtllmTacticState:
    device_key = _mxfp8_cuda_device_key(device)
    state = _MXFP8_TRTLLM_STATES.get(device_key)
    if state is not None:
        return state
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("unresolved MXFP8 tactic before CUDA Graph capture")
    return prepare_mxfp8_trtllm_tactic_state(device)


def _runner_and_workspace(
    state: _Mxfp8TrtllmTacticState,
    use_8x4_sf_layout: bool,
) -> tuple[Any, torch.Tensor]:
    if use_8x4_sf_layout:
        return state.runner_8x4, state.workspace_8x4
    return state.runner_128x4, state.workspace_128x4


def _record_mxfp8_tactic_resolution(
    key: Mxfp8TacticKey,
    *,
    selected_tactic: int,
    tactic_source: str,
    requested_tactic: int | None,
    artifact_lookup_hit: bool | None,
) -> None:
    if torch.compiler.is_compiling() or torch.cuda.is_current_stream_capturing():
        return
    if _MXFP8_TACTIC_AUDIT is not None:
        _MXFP8_TACTIC_AUDIT.register(
            key,
            selected_tactic=selected_tactic,
            tactic_source=tactic_source,
            requested_tactic=requested_tactic,
            artifact_lookup_hit=artifact_lookup_hit,
        )


def _require_exact_mxfp8_tactic() -> bool:
    value = os.environ.get("VLLM_MXFP8_DENSE_REQUIRE_EXACT_TACTIC", "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _validate_required_exact_mxfp8_tactic(
    key: Mxfp8TacticKey,
    *,
    selected_tactic: int,
    tactic_source: str,
    artifact_lookup_hit: bool | None,
) -> None:
    if not _require_exact_mxfp8_tactic():
        return
    if artifact_lookup_hit is not True or selected_tactic < 0:
        raise RuntimeError(
            "MXFP8 exact tactic is required but unavailable for "
            f"{key}: source={tactic_source}, tactic={selected_tactic}"
        )


def _resolve_mxfp8_trtllm_tactic(
    state: _Mxfp8TrtllmTacticState,
    key: Mxfp8TacticKey,
    runner_inputs: list[Any],
) -> tuple[int, str]:
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("unresolved MXFP8 tactic before CUDA Graph capture")

    with state.resolution_lock:
        return _resolve_mxfp8_trtllm_tactic_locked(state, key, runner_inputs)


def _resolve_mxfp8_trtllm_tactic_locked(
    state: _Mxfp8TrtllmTacticState,
    key: Mxfp8TacticKey,
    runner_inputs: list[Any],
) -> tuple[int, str]:
    existing = state.resolved_tactics.get(key)
    if existing is not None:
        tactic_source, requested_tactic, artifact_lookup_hit = (
            _MXFP8_TACTIC_SOURCES.get(
                (id(state), key),
                ("pre_resolved", existing, None),
            )
        )
        _record_mxfp8_tactic_resolution(
            key,
            selected_tactic=existing,
            tactic_source=tactic_source,
            requested_tactic=requested_tactic,
            artifact_lookup_hit=artifact_lookup_hit,
        )
        _validate_required_exact_mxfp8_tactic(
            key,
            selected_tactic=existing,
            tactic_source=tactic_source,
            artifact_lookup_hit=artifact_lookup_hit,
        )
        return existing, tactic_source

    selected_tactic, tactic_source, requested_tactic, artifact_lookup_hit = (
        _select_mxfp8_trtllm_tactic_candidate(state, key)
    )

    if selected_tactic >= 0:
        runner, _ = _runner_and_workspace(
            state,
            key.activation_scale_layout == "8x4",
        )
        try:
            valid_tactics = runner.get_valid_tactics(
                runner_inputs,
                _Mxfp8RunnerProfile(key),
            )
        except Exception:
            logger.warning(
                "MXFP8 tactic %d could not be validated at runtime; using "
                "direct TRTLLM tactic=-1.",
                selected_tactic,
                exc_info=True,
            )
            valid_tactics = []
        if selected_tactic not in valid_tactics:
            selected_tactic = -1
            tactic_source = "runtime_illegal_tactic"

    state.resolved_tactics[key] = selected_tactic
    _MXFP8_TACTIC_SOURCES[(id(state), key)] = (
        tactic_source,
        requested_tactic,
        artifact_lookup_hit,
    )
    _record_mxfp8_tactic_resolution(
        key,
        selected_tactic=selected_tactic,
        tactic_source=tactic_source,
        requested_tactic=requested_tactic,
        artifact_lookup_hit=artifact_lookup_hit,
    )
    _validate_required_exact_mxfp8_tactic(
        key,
        selected_tactic=selected_tactic,
        tactic_source=tactic_source,
        artifact_lookup_hit=artifact_lookup_hit,
    )
    return selected_tactic, tactic_source


def _select_mxfp8_trtllm_tactic_candidate(
    state: _Mxfp8TrtllmTacticState,
    key: Mxfp8TacticKey,
) -> tuple[int, str, int | None, bool | None]:
    requested_tactic: int | None = None
    artifact_lookup_hit: bool | None = None
    if state.artifact is not None:
        requested_tactic = state.artifact.lookup(key)
        if requested_tactic is None:
            selected_tactic = -1
            tactic_source = "exact_miss"
            artifact_lookup_hit = False
        elif requested_tactic == -1:
            selected_tactic = -1
            tactic_source = "exact_table_default"
            artifact_lookup_hit = True
        else:
            selected_tactic = requested_tactic
            tactic_source = "exact_table"
            artifact_lookup_hit = True
    elif (
        not _MXFP8_ARTIFACT_CONFIGURATION_PRESENT
        and key.activation_scale_layout == "128x4"
    ):
        legacy_tactic = _resolve_mxfp8_high_m_tactic(
            key.m_logical,
            key.n_logical,
            key.k_logical,
            _MXFP8_LEGACY_TACTIC_HINTS,
            _MXFP8_LEGACY_FALLBACK_TACTIC,
            use_global_fallback=_MXFP8_LEGACY_GLOBAL_FALLBACK,
        )
        if legacy_tactic is None:
            selected_tactic = -1
            tactic_source = "artifact_disabled"
        else:
            requested_tactic = legacy_tactic
            selected_tactic = legacy_tactic
            tactic_source = "legacy_high_m"
    else:
        selected_tactic = -1
        tactic_source = "artifact_disabled"

    return (
        selected_tactic,
        tactic_source,
        requested_tactic,
        artifact_lookup_hit,
    )


def _run_mxfp8_trtllm_pre_resolved(
    state: _Mxfp8TrtllmTacticState,
    *,
    use_8x4_sf_layout: bool,
    runner_inputs: list[Any],
    tactic: int,
) -> torch.Tensor:
    runner, _ = _runner_and_workspace(state, use_8x4_sf_layout)
    return runner.forward(runner_inputs, tactic=tactic)


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
    tactic: int,
    tactic_source: str,
    tactic_specialization_fingerprint: str,
    layer_prefix: str,
    normalized_family: str,
    compilation_state: str,
    cuda_graph_state: str,
    *,
    use_8x4_sf_layout: bool,
) -> torch.Tensor:
    is_capturing = torch.cuda.is_current_stream_capturing()
    if is_capturing and tactic == _MXFP8_TRTLLM_UNRESOLVED_TACTIC:
        raise RuntimeError("unresolved MXFP8 tactic before CUDA Graph capture")

    device_index = x.get_device()
    state = (
        _MXFP8_TRTLLM_STATES_BY_DEVICE_INDEX[device_index]
        if 0 <= device_index < len(_MXFP8_TRTLLM_STATES_BY_DEVICE_INDEX)
        else None
    )
    if state is None:
        if is_capturing:
            raise RuntimeError("unresolved MXFP8 tactic before CUDA Graph capture")
        state = prepare_mxfp8_trtllm_tactic_state(x.device)

    if tactic != _MXFP8_TRTLLM_UNRESOLVED_TACTIC:
        specialization = (
            _MXFP8_TRTLLM_SPECIALIZATIONS_BY_DEVICE_INDEX[device_index]
            if 0 <= device_index < len(_MXFP8_TRTLLM_SPECIALIZATIONS_BY_DEVICE_INDEX)
            else None
        )
        if (
            specialization is None
            or not tactic_specialization_fingerprint
            or specialization.fingerprint != tactic_specialization_fingerprint
        ):
            raise RuntimeError(
                "stale MXFP8 tactic specialization does not match this worker"
            )
    elif tactic_specialization_fingerprint:
        raise RuntimeError("unresolved MXFP8 tactic has a specialization fingerprint")

    physical_n = int(weight.shape[0])
    key = (
        None
        if is_capturing
        else Mxfp8TacticKey(
            m_logical=int(x.shape[0]),
            n_logical=int(output_features),
            k_logical=int(x.shape[1]),
            n_physical=physical_n,
            k_physical=int(weight.shape[1]),
            activation_scale_layout=("8x4" if use_8x4_sf_layout else "128x4"),
            output_dtype="bfloat16",
        )
    )
    selected_tactic = tactic
    resolved_source = tactic_source

    from flashinfer import SfLayout
    from flashinfer import mxfp8_quantize as flashinfer_mxfp8_quantize

    sf_layout = SfLayout.layout_8x4 if use_8x4_sf_layout else SfLayout.layout_128x4
    input_mxfp8, input_scale = flashinfer_mxfp8_quantize(
        x,
        alignment=MXFP8_BLOCK_SIZE,
        backend="cuda",
        sf_swizzle_layout=sf_layout,
    )
    output = torch.empty(
        (x.shape[0], physical_n), dtype=torch.bfloat16, device=x.device
    )
    _, workspace = _runner_and_workspace(state, use_8x4_sf_layout)
    runner_inputs = [
        input_mxfp8,
        weight.t(),
        input_scale,
        weight_scale,
        torch.bfloat16,
        output,
        workspace,
    ]
    if not is_capturing:
        assert key is not None
        if tactic == _MXFP8_TRTLLM_UNRESOLVED_TACTIC:
            selected_tactic, resolved_source = _resolve_mxfp8_trtllm_tactic(
                state,
                key,
                runner_inputs,
            )
    if (
        not torch.compiler.is_compiling()
        and not is_capturing
        and _MXFP8_TRTLLM_TRACE_CALLBACK is not None
    ):
        assert key is not None
        _MXFP8_TRTLLM_TRACE_CALLBACK(
            prefix=layer_prefix,
            family=normalized_family,
            m_logical=key.m_logical,
            m_physical=key.m_logical,
            n_logical=key.n_logical,
            n_physical=key.n_physical,
            k_logical=key.k_logical,
            k_physical=key.k_physical,
            layout=key.activation_scale_layout,
            output_dtype=key.output_dtype,
            tactic_source=resolved_source,
            selected_tactic=selected_tactic,
            compilation_state=compilation_state,
            cuda_graph_state=cuda_graph_state,
            runtime_provenance=(
                asdict(_MXFP8_RUNTIME_PROVENANCE)
                if _MXFP8_RUNTIME_PROVENANCE is not None
                else {}
            ),
        )
    output = _run_mxfp8_trtllm_pre_resolved(
        state,
        use_8x4_sf_layout=use_8x4_sf_layout,
        runner_inputs=runner_inputs,
        tactic=selected_tactic,
    )
    return output[:, :output_features].contiguous()


def _mxfp8_trtllm_adaptive_linear_impl(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    output_features: int,
    tactic: int,
    tactic_source: str,
    tactic_specialization_fingerprint: str,
    layer_prefix: str,
    normalized_family: str,
    compilation_state: str,
    cuda_graph_state: str,
) -> torch.Tensor:
    return _mxfp8_trtllm_linear_fixed_impl(
        x,
        weight,
        weight_scale,
        output_features,
        tactic,
        tactic_source,
        tactic_specialization_fingerprint,
        layer_prefix,
        normalized_family,
        compilation_state,
        cuda_graph_state,
        use_8x4_sf_layout=mxfp8_trtllm_use_8x4_sf_layout(int(x.shape[0])),
    )


def _mxfp8_trtllm_linear_8x4_impl(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    output_features: int,
    tactic: int,
    tactic_source: str,
    tactic_specialization_fingerprint: str,
    layer_prefix: str,
    normalized_family: str,
    compilation_state: str,
    cuda_graph_state: str,
) -> torch.Tensor:
    return _mxfp8_trtllm_linear_fixed_impl(
        x,
        weight,
        weight_scale,
        output_features,
        tactic,
        tactic_source,
        tactic_specialization_fingerprint,
        layer_prefix,
        normalized_family,
        compilation_state,
        cuda_graph_state,
        use_8x4_sf_layout=True,
    )


def _mxfp8_trtllm_linear_128x4_impl(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    output_features: int,
    tactic: int,
    tactic_source: str,
    tactic_specialization_fingerprint: str,
    layer_prefix: str,
    normalized_family: str,
    compilation_state: str,
    cuda_graph_state: str,
) -> torch.Tensor:
    return _mxfp8_trtllm_linear_fixed_impl(
        x,
        weight,
        weight_scale,
        output_features,
        tactic,
        tactic_source,
        tactic_specialization_fingerprint,
        layer_prefix,
        normalized_family,
        compilation_state,
        cuda_graph_state,
        use_8x4_sf_layout=False,
    )


def mxfp8_trtllm_adaptive_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    output_features: int,
    layer_prefix: str = "unknown",
    normalized_family: str = "OtherDense",
    tactic: int = _MXFP8_TRTLLM_UNRESOLVED_TACTIC,
    tactic_source: str = "unresolved_eager",
    tactic_specialization_fingerprint: str = "",
) -> torch.Tensor:
    return torch.ops.vllm.mxfp8_trtllm_adaptive_linear(
        x,
        weight,
        weight_scale,
        output_features,
        tactic,
        tactic_source,
        tactic_specialization_fingerprint,
        layer_prefix,
        normalized_family,
        "eager",
        "eager",
    )


def mxfp8_trtllm_adaptive_linear_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    output_features: int,
    tactic: int,
    tactic_source: str,
    tactic_specialization_fingerprint: str,
    layer_prefix: str,
    normalized_family: str,
    compilation_state: str,
    cuda_graph_state: str,
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
    graph: Any,
    *,
    marker_op: Any,
    fixed_op: Any,
    tactic_override: tuple[int, str, str] | None = None,
    tactic_resolver: Callable[[Any], tuple[int, str, str]] | None = None,
    execution_context: tuple[str, str] | None = None,
) -> int:
    if tactic_override is not None and tactic_resolver is not None:
        raise ValueError("MXFP8 tactic override and resolver are mutually exclusive.")
    replaced = 0
    for node in graph.nodes:
        if node.op == "call_function" and node.target == marker_op:
            node.target = fixed_op
            node_tactic = (
                tactic_resolver(node)
                if tactic_resolver is not None
                else tactic_override
            )
            if node_tactic is not None:
                if len(node.args) < 7:
                    raise RuntimeError(
                        "MXFP8 adaptive marker is missing bound tactic arguments."
                    )
                node.args = (
                    *node.args[:4],
                    *node_tactic,
                    *node.args[7:],
                )
            if execution_context is not None:
                if len(node.args) < 11:
                    raise RuntimeError(
                        "MXFP8 adaptive marker is missing layer context arguments."
                    )
                node.args = (
                    *node.args[:9],
                    *execution_context,
                    *node.args[11:],
                )
            replaced += 1
    return replaced


def _mxfp8_node_tensor_value(node: Any, argument_name: str) -> Any:
    value = node.meta.get("val")
    if value is None:
        value = node.meta.get("tensor_meta")
    if value is None:
        raise RuntimeError(
            f"MXFP8 static specialization is missing {argument_name} metadata"
        )
    return value


def _mxfp8_static_tactic_binding(
    node: Any,
    *,
    m_logical: int,
    use_8x4_sf_layout: bool,
) -> tuple[int, str, str]:
    x = _mxfp8_node_tensor_value(node.args[0], "activation")
    weight = _mxfp8_node_tensor_value(node.args[1], "weight")
    device_index = _mxfp8_cuda_device_key(weight.device)[1]
    specialization = (
        _MXFP8_TRTLLM_SPECIALIZATIONS_BY_DEVICE_INDEX[device_index]
        if 0 <= device_index < len(_MXFP8_TRTLLM_SPECIALIZATIONS_BY_DEVICE_INDEX)
        else None
    )
    if specialization is None:
        raise RuntimeError(
            "MXFP8 static tactics must be legality-validated during eager "
            "pre-capture warmup before compilation"
        )
    key = Mxfp8TacticKey(
        m_logical=m_logical,
        n_logical=int(node.args[3]),
        k_logical=int(x.shape[-1]),
        n_physical=int(weight.shape[0]),
        k_physical=int(weight.shape[1]),
        activation_scale_layout=("8x4" if use_8x4_sf_layout else "128x4"),
        output_dtype="bfloat16",
    )
    binding = specialization.tactics.get(key)
    if binding is None:
        raise RuntimeError(
            f"MXFP8 static tactic has no exact worker-validated prewarm contract: {key}"
        )
    tactic, tactic_source = binding
    return tactic, tactic_source, specialization.fingerprint


class _Mxfp8AdaptiveLayoutSpecializationPass(InductorPass):
    def __init__(
        self,
        layout_policy: str,
        switch_m: int | None,
        capture_sizes: frozenset[int] = frozenset(),
    ) -> None:
        self.layout_policy = layout_policy
        self.switch_m = switch_m
        self.capture_sizes = capture_sizes

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
        is_single_size = compile_range.is_single_size()
        execution_context = (
            "compiled",
            (
                "pre_capture"
                if is_single_size and compile_range.start in self.capture_sizes
                else "not_captured"
            ),
        )
        replaced = _specialize_mxfp8_adaptive_layout_graph(
            graph,
            marker_op=torch.ops.vllm.mxfp8_trtllm_adaptive_linear.default,
            fixed_op=(
                torch.ops.vllm.mxfp8_trtllm_linear_8x4.default
                if use_8x4_sf_layout
                else torch.ops.vllm.mxfp8_trtllm_linear_128x4.default
            ),
            tactic_resolver=(
                (
                    lambda node: _mxfp8_static_tactic_binding(
                        node,
                        m_logical=compile_range.start,
                        use_8x4_sf_layout=use_8x4_sf_layout,
                    )
                )
                if is_single_size
                else None
            ),
            tactic_override=(
                None
                if is_single_size
                else (
                    _MXFP8_TRTLLM_UNRESOLVED_TACTIC,
                    "unresolved_dynamic_compile",
                    "",
                )
            ),
            execution_context=execution_context,
        )
        if replaced == 0:
            return

    def uuid(self) -> str:
        with _MXFP8_TRTLLM_STATE_LOCK:
            specialization_fingerprints = [
                specialization.fingerprint if specialization is not None else None
                for specialization in (_MXFP8_TRTLLM_SPECIALIZATIONS_BY_DEVICE_INDEX)
            ]
        return self.hash_dict(
            {
                "source": self.hash_source(self),
                "layout_policy": self.layout_policy,
                "switch_m": self.switch_m,
                "capture_sizes": sorted(self.capture_sizes),
                "tactic_binding": "worker_validated_static_snapshot",
                "specialization_fingerprints": specialization_fingerprints,
                "phase": "joint_custom_pre_pass",
                "schema_version": 5,
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
    capture_sizes = frozenset(compilation_config.cudagraph_capture_sizes or [])
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
    compilation_config.compile_sizes = sorted(
        set(compilation_config.compile_sizes or []).union(capture_sizes)
    )

    pass_key = "joint_custom_pre_pass"
    existing_pass = compilation_config.inductor_compile_config.get(pass_key)
    if existing_pass is None:
        compilation_config.inductor_compile_config[pass_key] = (
            _Mxfp8AdaptiveLayoutSpecializationPass(
                layout_policy,
                switch_m,
                capture_sizes,
            )
        )
    elif not isinstance(existing_pass, _Mxfp8AdaptiveLayoutSpecializationPass):
        raise RuntimeError(
            "TRTLLM MXFP8 adaptive layout cannot replace an existing "
            "Inductor joint custom pre-pass."
        )
    elif (
        existing_pass.layout_policy != layout_policy
        or existing_pass.switch_m != switch_m
        or existing_pass.capture_sizes != capture_sizes
    ):
        raise RuntimeError("TRTLLM MXFP8 layout policy changed after setup.")


def xpu_mxfp8_quantize(
    x: torch.Tensor, dtype: torch.dtype | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.vllm.xpu_mxfp8_quantize(x, dtype)
