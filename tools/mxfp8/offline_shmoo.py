# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Qualify exact-shape MXFP8 TRTLLM tactics from offline rollout traces."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import math
import re
import statistics
import sys
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, cast

from packaging.version import InvalidVersion, Version

Layout = Literal["8x4", "128x4"]
ShapeIdentity = tuple[Layout, int, int, int]
ObservationIdentity = tuple[Layout, int, int, int, str, int, int]

_LAYOUT_ORDER: dict[Layout, int] = {"8x4": 0, "128x4": 1}
_ELIGIBLE_TRACE_EVENTS = frozenset(
    {"mxfp8_adaptive_dispatch", "mxfp8_dense_shape"}
)
_OBSERVATION_FIELDS = frozenset(
    {
        "layout",
        "m",
        "n",
        "k",
        "config_sha256",
        "tactic",
        "repeat",
        "median_ms",
        "all_finite",
        "cosine_similarity",
        "status",
        "seed",
        "warmup",
        "iterations",
        "device_name",
        "compute_capability",
        "vllm_version",
        "flashinfer_version",
        "container_sha256",
    }
)
_SUCCESS_STATUS = "success"
_COMPATIBILITY_FIELDS = frozenset(
    {
        "vllm_version",
        "vllm_base_commit",
        "flashinfer_version",
        "compute_capability",
        "gpu_family",
        "model",
        "tensor_parallel_size",
    }
)
_GPU_FAMILY_ALIASES: dict[str, frozenset[str]] = {
    "B200": frozenset({"B200"}),
    "GB200": frozenset({"GB200"}),
    "GB300": frozenset({"GB300"}),
    "H100": frozenset({"H100"}),
    "H200": frozenset({"H200"}),
}


@dataclass(frozen=True)
class ShapeRecord:
    """One exact physical dense MXFP8 shape observed during rollout."""

    layout: Layout
    m: int
    n: int
    k: int
    config_sha256: str
    frequency: int

    @property
    def identity(self) -> ShapeIdentity:
        return (self.layout, self.m, self.n, self.k)


@dataclass(frozen=True)
class BenchmarkObservation:
    """One independent timing and correctness observation for one tactic."""

    layout: Layout
    m: int
    n: int
    k: int
    config_sha256: str
    tactic: int
    repeat: int
    median_ms: float | None
    all_finite: bool
    cosine_similarity: float | None
    status: str
    seed: int
    warmup: int
    iterations: int
    device_name: str
    compute_capability: str
    vllm_version: str
    flashinfer_version: str
    container_sha256: str
    num_valid_tactics: int | None = None
    error: str | None = None

    @property
    def shape_identity(self) -> ShapeIdentity:
        return (self.layout, self.m, self.n, self.k)

    @property
    def identity(self) -> ObservationIdentity:
        return (
            self.layout,
            self.m,
            self.n,
            self.k,
            self.config_sha256,
            self.tactic,
            self.repeat,
        )

    @property
    def runtime_identity(self) -> tuple[object, ...]:
        return (
            self.config_sha256,
            self.device_name,
            self.compute_capability,
            self.vllm_version,
            self.flashinfer_version,
            self.container_sha256,
            self.warmup,
            self.iterations,
        )


@dataclass(frozen=True)
class QualifiedShape:
    """Qualification decision for one inventory shape."""

    layout: Layout
    m: int
    n: int
    k: int
    config_sha256: str
    frequency: int
    tactic: int
    baseline_median_ms: float
    candidate_median_ms: float | None
    speedup_vs_default: float | None

    @property
    def identity(self) -> ShapeIdentity:
        return (self.layout, self.m, self.n, self.k)


@dataclass(frozen=True)
class BenchmarkPlan:
    """A deterministic independent repeat for one exact physical shape."""

    layout: Layout
    m: int
    n: int
    k: int
    config_sha256: str
    repeat: int
    seed: int

    @property
    def shape_identity(self) -> ShapeIdentity:
        return (self.layout, self.m, self.n, self.k)


@dataclass(frozen=True)
class InventoryArtifact:
    """Inventory plus the bootstrap runtime identity that produced its trace."""

    shapes: tuple[ShapeRecord, ...]
    source_manifest_sha256: str
    bootstrap_manifest_sha256: str
    compatibility: Mapping[str, object]


@dataclass(frozen=True)
class _GpuRuntimeIdentity:
    device_name: str
    compute_capability: str
    vllm_version: str
    flashinfer_version: str
    container_sha256: str


class _ShapeProfile:
    def __init__(self, a_shape: tuple[int, ...], b_shape: tuple[int, ...]) -> None:
        self._shapes = [a_shape, b_shape]

    def get_opt_shapes(self) -> list[tuple[int, ...]]:
        return self._shapes


def load_shape_inventory(
    paths: Sequence[Path],
    *,
    expected_config_sha256: str | None = None,
) -> tuple[ShapeRecord, ...]:
    """Load, validate, and deterministically aggregate trace JSONL files."""
    if not paths:
        raise ValueError("at least one trace input is required")
    frequencies: dict[tuple[ShapeIdentity, str], int] = defaultdict(int)
    config_hashes: set[str] = set()
    for path in paths:
        for line_number, document in _read_jsonl(path):
            event = document.get("event")
            if event not in _ELIGIBLE_TRACE_EVENTS:
                continue
            field = f"{path}:{line_number}"
            layout = _require_layout(document.get("layout"), f"{field}.layout")
            config_sha256 = _require_sha256(
                document.get("config_sha256"), f"{field}.config_sha256"
            )
            if event == "mxfp8_dense_shape":
                m = _require_positive_int(
                    document.get("m_physical"), f"{field}.m_physical"
                )
                n = _require_positive_int(
                    document.get("n_physical"), f"{field}.n_physical"
                )
            else:
                m = _require_positive_int(document.get("m"), f"{field}.m")
                n = _require_positive_int(document.get("n"), f"{field}.n")
            k = _require_positive_int(document.get("k"), f"{field}.k")
            frequency = _require_positive_int(
                document.get("frequency", 1), f"{field}.frequency"
            )
            identity = (layout, m, n, k)
            frequencies[(identity, config_sha256)] += frequency
            config_hashes.add(config_sha256)
    if not frequencies:
        raise ValueError("zero eligible dense MXFP8 trace records")
    if len(config_hashes) != 1:
        raise ValueError("all eligible trace records must share one config_sha256")
    if (
        expected_config_sha256 is not None
        and config_hashes != {
            _require_sha256(
                expected_config_sha256, "expected bootstrap manifest SHA256"
            )
        }
    ):
        raise ValueError(
            "trace config_sha256 does not match bootstrap manifest SHA256"
        )
    records = [
        ShapeRecord(
            layout=identity[0],
            m=identity[1],
            n=identity[2],
            k=identity[3],
            config_sha256=config_sha256,
            frequency=frequency,
        )
        for (identity, config_sha256), frequency in frequencies.items()
    ]
    return tuple(sorted(records, key=_shape_sort_key))


def load_benchmark_observations(
    paths: Sequence[Path],
) -> tuple[BenchmarkObservation, ...]:
    """Load append-only observation JSONL with fail-closed identities."""
    if not paths:
        raise ValueError("at least one benchmark observation input is required")
    observations: list[BenchmarkObservation] = []
    identities: set[ObservationIdentity] = set()
    runtime_identities: set[tuple[object, ...]] = set()
    for path in paths:
        for line_number, document in _read_jsonl(path):
            observation = _parse_observation(document, path, line_number)
            if observation.identity in identities:
                raise ValueError(
                    f"duplicate benchmark observation identity: "
                    f"{observation.identity!r}"
                )
            identities.add(observation.identity)
            runtime_identities.add(observation.runtime_identity)
            observations.append(observation)
    if not observations:
        raise ValueError("zero benchmark observations")
    if len(runtime_identities) > 1:
        raise ValueError("benchmark observations have mixed runtime identity")
    return tuple(sorted(observations, key=_observation_sort_key))


def qualify_observations(
    inventory: Sequence[ShapeRecord],
    observations: Sequence[BenchmarkObservation],
    *,
    minimum_repeat_count: int,
    minimum_cosine_similarity: float,
    minimum_speedup_vs_default: float,
) -> tuple[QualifiedShape, ...]:
    """Promote only complete, correct tactics faster than runner default."""
    if not inventory:
        raise ValueError("zero eligible dense MXFP8 inventory shapes")
    if (
        isinstance(minimum_repeat_count, bool)
        or not isinstance(minimum_repeat_count, int)
        or minimum_repeat_count < 3
    ):
        raise ValueError("minimum_repeat_count must be an integer of at least 3")
    _require_finite_range(
        minimum_cosine_similarity,
        "minimum_cosine_similarity",
        minimum=0.0,
        maximum=1.0,
        minimum_inclusive=False,
    )
    _require_finite_range(
        minimum_speedup_vs_default,
        "minimum_speedup_vs_default",
        minimum=0.0,
        minimum_inclusive=False,
    )

    inventory_by_identity: dict[ShapeIdentity, ShapeRecord] = {}
    config_hashes: set[str] = set()
    for record in inventory:
        _validate_shape_record(record)
        if record.identity in inventory_by_identity:
            raise ValueError(f"duplicate inventory shape: {record.identity!r}")
        inventory_by_identity[record.identity] = record
        config_hashes.add(record.config_sha256)
    if len(config_hashes) != 1:
        raise ValueError("inventory shapes must share one config_sha256")

    observation_ids: set[ObservationIdentity] = set()
    runtime_ids: set[tuple[object, ...]] = set()
    grouped: dict[
        tuple[ShapeIdentity, int], dict[int, BenchmarkObservation]
    ] = defaultdict(dict)
    for observation in observations:
        _validate_observation(observation, "observation")
        if observation.identity in observation_ids:
            raise ValueError(
                f"duplicate benchmark observation identity: "
                f"{observation.identity!r}"
            )
        observation_ids.add(observation.identity)
        runtime_ids.add(observation.runtime_identity)
        if observation.shape_identity not in inventory_by_identity:
            raise ValueError(
                "benchmark observation shape absent from inventory: "
                f"{observation.shape_identity!r}"
            )
        inventory_record = inventory_by_identity[observation.shape_identity]
        if observation.config_sha256 != inventory_record.config_sha256:
            raise ValueError("observation config_sha256 does not match inventory")
        grouped[(observation.shape_identity, observation.tactic)][
            observation.repeat
        ] = observation
    if len(runtime_ids) > 1:
        raise ValueError("benchmark observations have mixed runtime identity")

    required_repeats = range(minimum_repeat_count)
    results: list[QualifiedShape] = []
    for identity, record in sorted(
        inventory_by_identity.items(), key=lambda item: _identity_sort_key(item[0])
    ):
        defaults = grouped.get((identity, -1), {})
        if not all(
            repeat in defaults
            and _observation_passes(
                defaults[repeat], minimum_cosine_similarity
            )
            for repeat in required_repeats
        ):
            raise ValueError(
                f"default tactic -1 lacks passing repeats for shape {identity!r}"
            )
        repeat_seeds: list[int] = []
        for repeat in required_repeats:
            seeds = {
                observation.seed
                for (shape_identity, _tactic), repeats in grouped.items()
                if shape_identity == identity and repeat in repeats
                for observation in (repeats[repeat],)
            }
            if len(seeds) != 1:
                raise ValueError(
                    f"shape {identity!r} repeat {repeat} must use one seed "
                    "across all tactics"
                )
            repeat_seeds.append(next(iter(seeds)))
        if len(set(repeat_seeds)) != minimum_repeat_count:
            raise ValueError(
                f"shape {identity!r} required repeats must use distinct seeds"
            )
        default_median = statistics.median(
            cast(float, defaults[repeat].median_ms) for repeat in required_repeats
        )

        candidates: list[tuple[float, int]] = []
        tactics = sorted(
            tactic
            for shape_identity, tactic in grouped
            if shape_identity == identity and tactic != -1
        )
        for tactic in tactics:
            repeats = grouped[(identity, tactic)]
            if not all(
                repeat in repeats
                and _observation_passes(
                    repeats[repeat], minimum_cosine_similarity
                )
                for repeat in required_repeats
            ):
                continue
            candidate_median = statistics.median(
                cast(float, repeats[repeat].median_ms)
                for repeat in required_repeats
            )
            candidates.append((candidate_median, tactic))

        candidate_median_ms: float | None = None
        speedup: float | None = None
        tactic = -1
        if candidates:
            candidate_median_ms, candidate_tactic = min(candidates)
            candidate_speedup = default_median / candidate_median_ms
            if candidate_speedup >= minimum_speedup_vs_default:
                tactic = candidate_tactic
                speedup = candidate_speedup
            else:
                candidate_median_ms = None
        results.append(
            QualifiedShape(
                layout=record.layout,
                m=record.m,
                n=record.n,
                k=record.k,
                config_sha256=record.config_sha256,
                frequency=record.frequency,
                tactic=tactic,
                baseline_median_ms=default_median,
                candidate_median_ms=candidate_median_ms,
                speedup_vs_default=speedup,
            )
        )
    return tuple(results)


def build_qualified_manifest(
    qualified_shapes: Sequence[QualifiedShape],
    *,
    compatibility: Mapping[str, object],
    provenance: Mapping[str, object],
) -> dict[str, object]:
    """Build the immutable schema-1 runtime manifest from qualification decisions."""
    entries: dict[Layout, list[dict[str, int]]] = {"8x4": [], "128x4": []}
    seen: set[ShapeIdentity] = set()
    promoted_shapes: dict[tuple[int, int, int], Layout] = {}
    for qualified in sorted(qualified_shapes, key=_qualified_sort_key):
        if qualified.identity in seen:
            raise ValueError(f"duplicate qualified shape: {qualified.identity!r}")
        seen.add(qualified.identity)
        _validate_qualified_shape(qualified)
        if qualified.tactic == -1:
            continue
        dimensions = (qualified.m, qualified.n, qualified.k)
        previous_layout = promoted_shapes.get(dimensions)
        if previous_layout is not None and previous_layout != qualified.layout:
            raise ValueError(
                f"promoted shape appears in both layouts: {dimensions!r}"
            )
        promoted_shapes[dimensions] = qualified.layout
        entries[qualified.layout].append(
            {
                "m": qualified.m,
                "n": qualified.n,
                "k": qualified.k,
                "tactic": qualified.tactic,
            }
        )
    return {
        "schema_version": 1,
        "mode": "adaptive",
        "compatibility": dict(compatibility),
        "policy": {
            "gemm_backend": "trtllm",
            "layout": "adaptive",
            "switch_m": 256,
            "direct_trtllm": True,
            "require_direct_trtllm": True,
            "quant_backend": "cuda",
            "require_8x4_quant": True,
            "pad_to_128": False,
            "default_tactic": -1,
        },
        "tactics": entries,
        "provenance": dict(provenance),
    }


def build_benchmark_plan(
    inventory: Iterable[ShapeRecord],
    *,
    repeat_count: int,
    base_seed: int,
) -> tuple[BenchmarkPlan, ...]:
    """Create deterministic exact-layout GPU repeats without padding shapes."""
    if (
        isinstance(repeat_count, bool)
        or not isinstance(repeat_count, int)
        or repeat_count < 3
    ):
        raise ValueError("repeat_count must be an integer of at least 3")
    if (
        isinstance(base_seed, bool)
        or not isinstance(base_seed, int)
        or base_seed < 0
    ):
        raise ValueError("base_seed must be a non-negative integer")
    plans: list[BenchmarkPlan] = []
    for record in sorted(tuple(inventory), key=_shape_sort_key):
        _validate_shape_record(record)
        shape_seed = base_seed + record.m * 17 + record.n * 31 + record.k * 43
        for repeat in range(repeat_count):
            plans.append(
                BenchmarkPlan(
                    layout=record.layout,
                    m=record.m,
                    n=record.n,
                    k=record.k,
                    config_sha256=record.config_sha256,
                    repeat=repeat,
                    seed=shape_seed + repeat,
                )
            )
    return tuple(plans)


def build_tactic_plan(
    plan: BenchmarkPlan,
    *,
    valid_tactics: Iterable[int],
    completed: set[ObservationIdentity],
) -> tuple[int, ...]:
    """Select runner default and every valid tactic not already observed."""
    tactics: set[int] = {-1}
    for tactic in valid_tactics:
        tactics.add(_require_tactic(tactic, "valid_tactic"))
    ordered = (-1, *sorted(tactic for tactic in tactics if tactic != -1))
    return tuple(
        tactic
        for tactic in ordered
        if (
            plan.layout,
            plan.m,
            plan.n,
            plan.k,
            plan.config_sha256,
            tactic,
            plan.repeat,
        )
        not in completed
    )


def validate_resume_observations(
    plans: Sequence[BenchmarkPlan],
    observations: Sequence[BenchmarkObservation],
) -> None:
    """Reject append/resume data generated from another deterministic plan."""
    expected_seeds = {
        (plan.shape_identity, plan.repeat): plan.seed for plan in plans
    }
    if len(expected_seeds) != len(plans):
        raise ValueError("benchmark plan contains duplicate shape/repeat identities")
    seen_seeds: dict[tuple[ShapeIdentity, int], int] = {}
    for observation in observations:
        key = (observation.shape_identity, observation.repeat)
        expected_seed = expected_seeds.get(key)
        if expected_seed is None or observation.seed != expected_seed:
            raise ValueError(
                "existing observation does not match deterministic seed plan: "
                f"{observation.identity!r}"
            )
        previous_seed = seen_seeds.setdefault(key, observation.seed)
        if previous_seed != observation.seed:
            raise ValueError(
                "existing tactics for one shape/repeat do not share a seed"
            )
    repeats_by_shape: dict[ShapeIdentity, set[int]] = defaultdict(set)
    for (shape_identity, _repeat), seed in expected_seeds.items():
        repeats_by_shape[shape_identity].add(seed)
    if any(
        len(seeds) != len(
            {
                plan.repeat
                for plan in plans
                if plan.shape_identity == shape_identity
            }
        )
        for shape_identity, seeds in repeats_by_shape.items()
    ):
        raise ValueError("benchmark plan repeats must use distinct seeds")


def enumerate_valid_tactics(
    runner: object, inputs: list[object], shape_profile: object
) -> tuple[int, ...]:
    """Enumerate all private-runner tactics and preserve ABI diagnostics."""
    get_valid_tactics = getattr(runner, "get_valid_tactics", None)
    if not callable(get_valid_tactics):
        raise RuntimeError("TRTLLM runner has no callable get_valid_tactics")
    try:
        raw_tactics = cast(
            Iterable[object],
            get_valid_tactics(inputs, shape_profile),
        )
        tactics = {
            _require_tactic(tactic, "valid_tactic")
            for tactic in raw_tactics
        }
    except Exception as error:
        raise RuntimeError(
            f"TRTLLM get_valid_tactics failed: {error}"
        ) from error
    return tuple(sorted(tactic for tactic in tactics if tactic != -1))


def validate_active_runtime_identity(
    *,
    active_vllm_version: str,
    active_flashinfer_version: str,
    active_compute_capability: str,
    active_device_name: str,
    declared_vllm_version: str,
    declared_flashinfer_version: str,
    inventory_compatibility: Mapping[str, object],
) -> None:
    """Bind installed GPU runtime identity to the traced bootstrap manifest."""
    active_vllm_public = _public_version(
        active_vllm_version, "active vLLM version"
    )
    declared_vllm_public = _public_version(
        declared_vllm_version, "declared vLLM version"
    )
    active_flashinfer_public = _public_version(
        active_flashinfer_version, "active FlashInfer version"
    )
    declared_flashinfer_public = _public_version(
        declared_flashinfer_version, "declared FlashInfer version"
    )
    if active_vllm_public != declared_vllm_public:
        raise RuntimeError(
            "active vLLM version does not match --vllm-version"
        )
    if active_flashinfer_public != declared_flashinfer_public:
        raise RuntimeError(
            "active FlashInfer version does not match --flashinfer-version"
        )
    expected_vllm = inventory_compatibility.get("vllm_version")
    expected_flashinfer = inventory_compatibility.get("flashinfer_version")
    expected_capability = inventory_compatibility.get("compute_capability")
    expected_gpu_family = inventory_compatibility.get("gpu_family")
    if not isinstance(expected_vllm, str) or (
        _public_version(expected_vllm, "inventory vLLM version")
        != active_vllm_public
    ):
        raise RuntimeError(
            "active vLLM version does not match inventory compatibility"
        )
    if not isinstance(expected_flashinfer, str) or (
        _public_version(expected_flashinfer, "inventory FlashInfer version")
        != active_flashinfer_public
    ):
        raise RuntimeError(
            "active FlashInfer version does not match inventory compatibility"
        )
    if expected_capability != active_compute_capability:
        raise RuntimeError(
            "active compute capability does not match inventory compatibility"
        )
    if not isinstance(expected_gpu_family, str) or not expected_gpu_family:
        raise ValueError("inventory GPU family must be a non-empty string")
    normalized_family = "".join(
        character for character in expected_gpu_family.upper() if character.isalnum()
    )
    accepted_aliases = _GPU_FAMILY_ALIASES.get(
        normalized_family, frozenset({normalized_family})
    )
    device_tokens = frozenset(
        re.findall(r"[A-Z]+[0-9]+[A-Z0-9]*", active_device_name.upper())
    )
    if accepted_aliases.isdisjoint(device_tokens):
        raise RuntimeError(
            "active device does not match inventory GPU family"
        )


def _public_version(raw: str, field: str) -> str:
    try:
        return Version(raw).public
    except InvalidVersion as error:
        raise RuntimeError(f"{field} is not a valid version") from error


def _compatibility_values_match(
    field: str, expected: object, actual: object
) -> bool:
    if field in {"vllm_version", "flashinfer_version"}:
        return (
            isinstance(expected, str)
            and isinstance(actual, str)
            and _public_version(expected, f"compatibility.{field}")
            == _public_version(actual, f"observed {field}")
        )
    return expected == actual


def digest_input_paths(paths: Sequence[Path]) -> str:
    """Aggregate raw inputs by content digest, independent of paths and order."""
    if not paths:
        raise ValueError("at least one input path is required for provenance")
    member_digests = sorted(
        hashlib.sha256(path.read_bytes()).hexdigest() for path in paths
    )
    encoded = json.dumps(member_digests, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def canonical_json_bytes(document: object) -> bytes:
    """Serialize canonical two-space, key-sorted JSON with one final newline."""
    return (
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()


def regenerate_qualified_manifest(
    *,
    inventory_path: Path,
    observation_paths: Sequence[Path],
    minimum_repeat_count: int,
    minimum_cosine_similarity: float,
    minimum_speedup_vs_default: float,
    qualification_scope: str,
) -> tuple[dict[str, object], tuple[QualifiedShape, ...]]:
    """Regenerate the manifest through the same pure promotion data flow."""
    artifact = _load_inventory_document(inventory_path)
    observations = load_benchmark_observations(observation_paths)
    observed_runtime = observations[0]
    for field, observed in (
        ("vllm_version", observed_runtime.vllm_version),
        ("flashinfer_version", observed_runtime.flashinfer_version),
        ("compute_capability", observed_runtime.compute_capability),
    ):
        if not _compatibility_values_match(
            field, artifact.compatibility[field], observed
        ):
            raise ValueError(
                f"observed {field} does not match inventory compatibility"
            )
    qualified = qualify_observations(
        artifact.shapes,
        observations,
        minimum_repeat_count=minimum_repeat_count,
        minimum_cosine_similarity=minimum_cosine_similarity,
        minimum_speedup_vs_default=minimum_speedup_vs_default,
    )
    provenance: dict[str, object] = {
        "source_manifest_sha256": artifact.source_manifest_sha256,
        "source_hint_sha256": digest_input_paths(observation_paths),
        "container_sha256": observations[0].container_sha256,
        "qualification_scope": _require_nonempty_string(
            qualification_scope, "qualification_scope"
        ),
        "qualification_repeat_count": minimum_repeat_count,
        "minimum_cosine_similarity": minimum_cosine_similarity,
        "minimum_speedup_vs_default": minimum_speedup_vs_default,
    }
    return (
        build_qualified_manifest(
            qualified,
            compatibility=artifact.compatibility,
            provenance=provenance,
        ),
        qualified,
    )


def validate_manifest(
    path: Path,
    *,
    inventory_path: Path,
    observation_paths: Sequence[Path],
    minimum_repeat_count: int,
    minimum_cosine_similarity: float,
    minimum_speedup_vs_default: float,
    qualification_scope: str,
    actual_vllm_version: str,
    actual_flashinfer_version: str,
    actual_compute_capability: tuple[int, int],
    actual_model: str,
    actual_tensor_parallel_size: int,
    check: bool,
) -> object:
    """Validate compatibility and optionally byte-check raw regeneration."""
    regenerated, _qualified = regenerate_qualified_manifest(
        inventory_path=inventory_path,
        observation_paths=observation_paths,
        minimum_repeat_count=minimum_repeat_count,
        minimum_cosine_similarity=minimum_cosine_similarity,
        minimum_speedup_vs_default=minimum_speedup_vs_default,
        qualification_scope=qualification_scope,
    )
    loader = _load_production_loader()
    runtime_config = loader(
        str(path),
        actual_vllm_version=actual_vllm_version,
        actual_flashinfer_version=actual_flashinfer_version,
        actual_compute_capability=actual_compute_capability,
        actual_model=actual_model,
        actual_tensor_parallel_size=actual_tensor_parallel_size,
    )
    if check and path.read_bytes() != canonical_json_bytes(regenerated):
        raise ValueError(
            "manifest bytes differ from regenerated raw qualification inputs"
        )
    return runtime_config


def _read_jsonl(path: Path) -> Iterable[tuple[int, Mapping[str, object]]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise RuntimeError(f"unable to read JSONL input: {path}") from error
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            raise ValueError(f"{path}:{line_number} must not be blank")
        yield line_number, _decode_json_object(
            line, f"{path}:{line_number}"
        )


def _read_json_document(path: Path) -> dict[str, object]:
    try:
        source = path.read_text(encoding="utf-8")
    except OSError as error:
        raise RuntimeError(f"unable to read JSON input: {path}") from error
    return dict(_decode_json_object(source, str(path)))


def _decode_json_object(source: str, field: str) -> Mapping[str, object]:
    try:
        parsed = json.loads(
            source,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        raise ValueError(f"{field} must contain valid finite JSON") from error
    if not isinstance(parsed, dict) or not all(
        isinstance(key, str) for key in parsed
    ):
        raise ValueError(f"{field} must contain a JSON object")
    return cast(Mapping[str, object], parsed)


def _reject_duplicate_json_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"JSON must not contain non-finite constant: {value}")


def _parse_observation(
    document: Mapping[str, object], path: Path, line_number: int
) -> BenchmarkObservation:
    field = f"{path}:{line_number}"
    missing = _OBSERVATION_FIELDS - document.keys()
    if missing:
        raise ValueError(f"{field} missing observation fields: {sorted(missing)}")
    layout = _require_layout(document["layout"], f"{field}.layout")
    status = _require_nonempty_string(document["status"], f"{field}.status")
    if status not in {_SUCCESS_STATUS, "incorrect", "error"}:
        raise ValueError(f"{field}.status has unsupported value {status!r}")
    median_ms = _require_optional_float(
        document["median_ms"], f"{field}.median_ms"
    )
    cosine_similarity = _require_optional_float(
        document["cosine_similarity"], f"{field}.cosine_similarity"
    )
    observation = BenchmarkObservation(
        layout=layout,
        m=_require_positive_int(document["m"], f"{field}.m"),
        n=_require_positive_int(document["n"], f"{field}.n"),
        k=_require_positive_int(document["k"], f"{field}.k"),
        config_sha256=_require_sha256(
            document["config_sha256"], f"{field}.config_sha256"
        ),
        tactic=_require_tactic(document["tactic"], f"{field}.tactic"),
        repeat=_require_nonnegative_int(
            document["repeat"], f"{field}.repeat"
        ),
        median_ms=median_ms,
        all_finite=_require_bool(
            document["all_finite"], f"{field}.all_finite"
        ),
        cosine_similarity=cosine_similarity,
        status=status,
        seed=_require_nonnegative_int(document["seed"], f"{field}.seed"),
        warmup=_require_nonnegative_int(
            document["warmup"], f"{field}.warmup"
        ),
        iterations=_require_positive_int(
            document["iterations"], f"{field}.iterations"
        ),
        device_name=_require_nonempty_string(
            document["device_name"], f"{field}.device_name"
        ),
        compute_capability=_require_nonempty_string(
            document["compute_capability"], f"{field}.compute_capability"
        ),
        vllm_version=_require_nonempty_string(
            document["vllm_version"], f"{field}.vllm_version"
        ),
        flashinfer_version=_require_nonempty_string(
            document["flashinfer_version"], f"{field}.flashinfer_version"
        ),
        container_sha256=_require_sha256(
            document["container_sha256"], f"{field}.container_sha256"
        ),
        num_valid_tactics=(
            _require_nonnegative_int(
                document["num_valid_tactics"],
                f"{field}.num_valid_tactics",
            )
            if document.get("num_valid_tactics") is not None
            else None
        ),
        error=(
            _require_nonempty_string(document["error"], f"{field}.error")
            if document.get("error") is not None
            else None
        ),
    )
    _validate_observation(observation, field)
    return observation


def _validate_observation(
    observation: BenchmarkObservation, field: str
) -> None:
    _require_layout(observation.layout, f"{field}.layout")
    for name in ("m", "n", "k"):
        _require_positive_int(getattr(observation, name), f"{field}.{name}")
    _require_sha256(observation.config_sha256, f"{field}.config_sha256")
    _require_tactic(observation.tactic, f"{field}.tactic")
    _require_nonnegative_int(observation.repeat, f"{field}.repeat")
    _require_nonnegative_int(observation.seed, f"{field}.seed")
    _require_nonnegative_int(observation.warmup, f"{field}.warmup")
    _require_positive_int(observation.iterations, f"{field}.iterations")
    _require_bool(observation.all_finite, f"{field}.all_finite")
    for name in (
        "status",
        "device_name",
        "compute_capability",
        "vllm_version",
        "flashinfer_version",
    ):
        _require_nonempty_string(getattr(observation, name), f"{field}.{name}")
    _require_sha256(observation.container_sha256, f"{field}.container_sha256")
    if observation.num_valid_tactics is not None:
        _require_nonnegative_int(
            observation.num_valid_tactics, f"{field}.num_valid_tactics"
        )
    if observation.error is not None:
        _require_nonempty_string(observation.error, f"{field}.error")
    if observation.median_ms is not None:
        _require_positive_float(observation.median_ms, f"{field}.median_ms")
    if observation.cosine_similarity is not None:
        _require_finite_range(
            observation.cosine_similarity,
            f"{field}.cosine_similarity",
            minimum=-1.0,
            maximum=1.0,
        )
    if observation.status == _SUCCESS_STATUS and (
        observation.median_ms is None
        or observation.cosine_similarity is None
    ):
        raise ValueError(
            f"{field} successful observation requires timing and cosine"
        )


def _observation_passes(
    observation: BenchmarkObservation, minimum_cosine_similarity: float
) -> bool:
    return (
        observation.status == _SUCCESS_STATUS
        and observation.all_finite
        and observation.median_ms is not None
        and observation.median_ms > 0
        and observation.cosine_similarity is not None
        and observation.cosine_similarity >= minimum_cosine_similarity
    )


def _validate_shape_record(record: ShapeRecord) -> None:
    _require_layout(record.layout, "shape.layout")
    for name in ("m", "n", "k", "frequency"):
        _require_positive_int(getattr(record, name), f"shape.{name}")
    _require_sha256(record.config_sha256, "shape.config_sha256")


def _validate_qualified_shape(qualified: QualifiedShape) -> None:
    _require_layout(qualified.layout, "qualified.layout")
    for name in ("m", "n", "k", "frequency"):
        _require_positive_int(getattr(qualified, name), f"qualified.{name}")
    _require_sha256(qualified.config_sha256, "qualified.config_sha256")
    _require_tactic(qualified.tactic, "qualified.tactic")
    _require_positive_float(
        qualified.baseline_median_ms, "qualified.baseline_median_ms"
    )
    if qualified.tactic == -1:
        if (
            qualified.candidate_median_ms is not None
            or qualified.speedup_vs_default is not None
        ):
            raise ValueError("default qualified shape must not retain candidate timing")
    else:
        if (
            qualified.candidate_median_ms is None
            or qualified.speedup_vs_default is None
        ):
            raise ValueError("promoted qualified shape requires candidate metrics")
        _require_positive_float(
            qualified.candidate_median_ms, "qualified.candidate_median_ms"
        )
        _require_positive_float(
            qualified.speedup_vs_default, "qualified.speedup_vs_default"
        )


def _require_layout(value: object, field: str) -> Layout:
    if not isinstance(value, str) or value not in _LAYOUT_ORDER:
        raise ValueError(f"{field} must be exactly '8x4' or '128x4'")
    return cast(Layout, value)


def _require_positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive non-boolean integer")
    return value


def _require_nonnegative_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative non-boolean integer")
    return value


def _require_tactic(value: object, field: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < -1
    ):
        raise ValueError(f"{field} must be integer -1 or greater")
    return value


def _require_bool(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be a boolean")
    return value


def _require_nonempty_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _require_sha256(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be one lowercase SHA-256 digest")
    return value


def _require_optional_float(value: object, field: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a number or null")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{field} must be finite")
    return converted


def _require_positive_float(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a number")
    converted = float(value)
    if not math.isfinite(converted) or converted <= 0:
        raise ValueError(f"{field} must be finite and positive")
    return converted


def _require_finite_range(
    value: object,
    field: str,
    *,
    minimum: float,
    maximum: float | None = None,
    minimum_inclusive: bool = True,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a number")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{field} must be finite")
    below_minimum = (
        converted < minimum if minimum_inclusive else converted <= minimum
    )
    if below_minimum or (maximum is not None and converted > maximum):
        raise ValueError(f"{field} is outside the supported range")
    return converted


def _identity_sort_key(identity: ShapeIdentity) -> tuple[int, int, int, int]:
    return (_LAYOUT_ORDER[identity[0]], identity[1], identity[2], identity[3])


def _shape_sort_key(record: ShapeRecord) -> tuple[int, int, int, int]:
    return _identity_sort_key(record.identity)


def _qualified_sort_key(
    record: QualifiedShape,
) -> tuple[int, int, int, int]:
    return _identity_sort_key(record.identity)


def _observation_sort_key(
    observation: BenchmarkObservation,
) -> tuple[int, int, int, int, int, int]:
    return (
        *_identity_sort_key(observation.shape_identity),
        observation.tactic,
        observation.repeat,
    )


def _inventory_document(
    inventory: Sequence[ShapeRecord],
    source_manifest_sha256: str,
    bootstrap_manifest_sha256: str,
    compatibility: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "source_manifest_sha256": _require_sha256(
            source_manifest_sha256, "source_manifest_sha256"
        ),
        "bootstrap_manifest_sha256": _require_sha256(
            bootstrap_manifest_sha256, "bootstrap_manifest_sha256"
        ),
        "compatibility": _validate_compatibility_mapping(compatibility),
        "shapes": [asdict(record) for record in sorted(inventory, key=_shape_sort_key)],
    }


def _load_inventory_document(path: Path) -> InventoryArtifact:
    document = _read_json_document(path)
    if set(document) != {
        "schema_version",
        "source_manifest_sha256",
        "bootstrap_manifest_sha256",
        "compatibility",
        "shapes",
    }:
        raise ValueError("inventory document has unsupported fields")
    if document["schema_version"] != 1 or isinstance(
        document["schema_version"], bool
    ):
        raise ValueError("inventory schema_version must be 1")
    source_sha256 = _require_sha256(
        document["source_manifest_sha256"], "inventory.source_manifest_sha256"
    )
    bootstrap_sha256 = _require_sha256(
        document["bootstrap_manifest_sha256"],
        "inventory.bootstrap_manifest_sha256",
    )
    raw_compatibility = document["compatibility"]
    if not isinstance(raw_compatibility, dict):
        raise ValueError("inventory.compatibility must be an object")
    compatibility = _validate_compatibility_mapping(raw_compatibility)
    raw_shapes = document["shapes"]
    if not isinstance(raw_shapes, list):
        raise ValueError("inventory.shapes must be an array")
    records: list[ShapeRecord] = []
    for index, raw_shape in enumerate(raw_shapes):
        if not isinstance(raw_shape, dict):
            raise ValueError(f"inventory.shapes[{index}] must be an object")
        expected = {"layout", "m", "n", "k", "config_sha256", "frequency"}
        if set(raw_shape) != expected:
            raise ValueError(
                f"inventory.shapes[{index}] has unsupported fields"
            )
        record = ShapeRecord(
            layout=_require_layout(
                raw_shape["layout"], f"inventory.shapes[{index}].layout"
            ),
            m=_require_positive_int(
                raw_shape["m"], f"inventory.shapes[{index}].m"
            ),
            n=_require_positive_int(
                raw_shape["n"], f"inventory.shapes[{index}].n"
            ),
            k=_require_positive_int(
                raw_shape["k"], f"inventory.shapes[{index}].k"
            ),
            config_sha256=_require_sha256(
                raw_shape["config_sha256"],
                f"inventory.shapes[{index}].config_sha256",
            ),
            frequency=_require_positive_int(
                raw_shape["frequency"],
                f"inventory.shapes[{index}].frequency",
            ),
        )
        records.append(record)
    if not records:
        raise ValueError("zero eligible dense MXFP8 inventory shapes")
    sorted_records = tuple(sorted(records, key=_shape_sort_key))
    if len({record.identity for record in sorted_records}) != len(
        sorted_records
    ):
        raise ValueError("inventory contains duplicate shape identities")
    if len({record.config_sha256 for record in sorted_records}) != 1:
        raise ValueError("inventory shapes have mixed config_sha256")
    if {record.config_sha256 for record in sorted_records} != {
        bootstrap_sha256
    }:
        raise ValueError(
            "inventory shapes do not match bootstrap manifest SHA256"
        )
    return InventoryArtifact(
        shapes=sorted_records,
        source_manifest_sha256=source_sha256,
        bootstrap_manifest_sha256=bootstrap_sha256,
        compatibility=MappingProxyType(compatibility),
    )


def _validate_compatibility_mapping(
    compatibility: Mapping[str, object],
) -> dict[str, object]:
    if set(compatibility) != _COMPATIBILITY_FIELDS:
        raise ValueError("compatibility has unsupported fields")
    result = dict(compatibility)
    for key in (
        "vllm_version",
        "vllm_base_commit",
        "flashinfer_version",
        "compute_capability",
        "gpu_family",
        "model",
    ):
        _require_nonempty_string(result[key], f"compatibility.{key}")
    _require_positive_int(
        result["tensor_parallel_size"],
        "compatibility.tensor_parallel_size",
    )
    _parse_compute_capability(
        cast(str, result["compute_capability"])
    )
    return result


def _load_bootstrap_runtime_manifest(
    path: Path,
) -> tuple[str, dict[str, object]]:
    document = _read_json_document(path)
    raw_compatibility = document.get("compatibility")
    if not isinstance(raw_compatibility, dict):
        raise ValueError("bootstrap manifest compatibility must be an object")
    compatibility = _validate_compatibility_mapping(raw_compatibility)
    compute_capability = _parse_compute_capability(
        cast(str, compatibility["compute_capability"])
    )
    loader = _load_production_loader()
    runtime_config = loader(
        str(path),
        actual_vllm_version=cast(str, compatibility["vllm_version"]),
        actual_flashinfer_version=cast(
            str, compatibility["flashinfer_version"]
        ),
        actual_compute_capability=compute_capability,
        actual_model=cast(str, compatibility["model"]),
        actual_tensor_parallel_size=cast(
            int, compatibility["tensor_parallel_size"]
        ),
    )
    source_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    if runtime_config.source_sha256 != source_sha256:
        raise RuntimeError("bootstrap manifest SHA256 changed while loading")
    return source_sha256, compatibility


def _load_production_loader() -> Any:
    module_path = (
        Path(__file__).parents[2]
        / "vllm/model_executor/kernels/linear/mxfp8/tactic_config.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_mxfp8_offline_production_tactic_config", module_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load production MXFP8 tactic config loader")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.load_mxfp8_dense_runtime_config


def _import_gpu_dependencies() -> tuple[Any, ...]:
    try:
        torch = importlib.import_module("torch")
        flashinfer = importlib.import_module("flashinfer")
        active_vllm = importlib.import_module("vllm")
    except ImportError as error:
        raise RuntimeError(
            "GPU shmoo requires PyTorch, FlashInfer, and vLLM"
        ) from error
    if not torch.cuda.is_available():
        raise RuntimeError("GPU shmoo requires CUDA")

    mm_mxfp8 = getattr(flashinfer, "mm_mxfp8", None)
    mxfp8_quantize = getattr(flashinfer, "mxfp8_quantize", None)
    shuffle_matrix_a = getattr(flashinfer, "shuffle_matrix_a", None)
    shuffle_matrix_sf_a = getattr(flashinfer, "shuffle_matrix_sf_a", None)
    if mxfp8_quantize is None:
        mxfp8_quantize = importlib.import_module(
            "flashinfer.fp4_quantization"
        ).mxfp8_quantize
    if shuffle_matrix_a is None or shuffle_matrix_sf_a is None:
        fp4_quantization = importlib.import_module(
            "flashinfer.fp4_quantization"
        )
        shuffle_matrix_a = fp4_quantization.shuffle_matrix_a
        shuffle_matrix_sf_a = fp4_quantization.shuffle_matrix_sf_a
    if mm_mxfp8 is None:
        mm_mxfp8 = importlib.import_module("flashinfer.gemm").mm_mxfp8
    try:
        get_trtllm_gemm_module = importlib.import_module(
            "flashinfer.gemm.gemm_base"
        ).get_trtllm_gemm_module
    except (ImportError, AttributeError) as error:
        raise RuntimeError(
            "FlashInfer internal get_trtllm_gemm_module is unavailable"
        ) from error
    return (
        torch,
        flashinfer,
        active_vllm,
        mm_mxfp8,
        mxfp8_quantize,
        shuffle_matrix_a,
        shuffle_matrix_sf_a,
        get_trtllm_gemm_module,
    )


def _resolve_sf_layout(name: Layout | Literal["linear"]) -> object:
    attribute = {
        "linear": "layout_linear",
        "128x4": "layout_128x4",
        "8x4": "layout_8x4",
    }[name]
    for module_name in ("flashinfer.fp4_quantization", "flashinfer"):
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        enum = getattr(module, "SfLayout", None)
        if enum is not None and hasattr(enum, attribute):
            return getattr(enum, attribute)
    raise RuntimeError(f"FlashInfer SfLayout.{attribute} is unavailable")


def _quantize(
    mxfp8_quantize: Any,
    tensor: Any,
    *,
    layout: Layout | Literal["linear"],
) -> tuple[Any, Any]:
    try:
        return mxfp8_quantize(
            input=tensor,
            backend="cuda",
            sf_swizzle_layout=_resolve_sf_layout(layout),
        )
    except TypeError as error:
        if layout != "128x4":
            raise RuntimeError(
                f"mxfp8_quantize lacks exact {layout} CUDA layout support"
            ) from error
        try:
            return mxfp8_quantize(
                input=tensor, is_sf_swizzled_layout=True
            )
        except TypeError:
            return mxfp8_quantize(tensor, is_sf_swizzled_layout=True)


def _prepare_trtllm_weight(
    mxfp8_quantize: Any,
    shuffle_matrix_a: Any,
    shuffle_matrix_sf_a: Any,
    weight: Any,
) -> tuple[Any, Any]:
    """Quantize then shuffle B and B scales in validated TRTLLM order."""
    n, k = weight.shape
    weight_mxfp8, weight_scale = _quantize(
        mxfp8_quantize, weight, layout="linear"
    )
    weight_mxfp8 = shuffle_matrix_a(weight_mxfp8, 128).reshape(n, k)
    weight_scale = shuffle_matrix_sf_a(
        weight_scale.reshape(n, k // 32),
        128,
        num_elts_per_sf=32,
    ).reshape(-1)
    return weight_mxfp8, weight_scale


def _time_cuda(torch: Any, function: Any, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        if function() is None:
            raise RuntimeError("benchmark function returned None")
    torch.cuda.synchronize()
    times: list[float] = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = function()
        end.record()
        torch.cuda.synchronize()
        if result is None:
            raise RuntimeError("benchmark function returned None")
        times.append(float(start.elapsed_time(end)))
    return statistics.median(times)


def _compare_outputs(torch: Any, reference: Any, candidate: Any) -> tuple[bool, float]:
    if reference.shape != candidate.shape:
        raise ValueError(
            f"output shape mismatch: reference={reference.shape}, "
            f"candidate={candidate.shape}"
        )
    reference_float = reference.float()
    candidate_float = candidate.float()
    all_finite = bool(
        torch.isfinite(reference_float).all().item()
        and torch.isfinite(candidate_float).all().item()
    )
    if not all_finite:
        return False, 0.0
    cosine_similarity = float(
        torch.nn.functional.cosine_similarity(
            reference_float.reshape(-1),
            candidate_float.reshape(-1),
            dim=0,
        ).item()
    )
    return True, cosine_similarity


def _error_observation(
    plan: BenchmarkPlan,
    tactic: int,
    runtime: _GpuRuntimeIdentity,
    *,
    warmup: int,
    iterations: int,
    num_valid_tactics: int,
    error: Exception,
) -> BenchmarkObservation:
    return BenchmarkObservation(
        layout=plan.layout,
        m=plan.m,
        n=plan.n,
        k=plan.k,
        config_sha256=plan.config_sha256,
        tactic=tactic,
        repeat=plan.repeat,
        median_ms=None,
        all_finite=False,
        cosine_similarity=None,
        status="error",
        seed=plan.seed,
        warmup=warmup,
        iterations=iterations,
        device_name=runtime.device_name,
        compute_capability=runtime.compute_capability,
        vllm_version=runtime.vllm_version,
        flashinfer_version=runtime.flashinfer_version,
        container_sha256=runtime.container_sha256,
        num_valid_tactics=num_valid_tactics,
        error=str(error)[:1000] or error.__class__.__name__,
    )


def _benchmark_shape_repeat(
    plan: BenchmarkPlan,
    *,
    runtime: _GpuRuntimeIdentity,
    warmup: int,
    iterations: int,
    workspace_mb: int,
    minimum_cosine_similarity: float,
    completed: set[ObservationIdentity],
) -> tuple[BenchmarkObservation, ...]:
    (
        torch,
        _flashinfer,
        _active_vllm,
        _mm_mxfp8,
        mxfp8_quantize,
        shuffle_matrix_a,
        shuffle_matrix_sf_a,
        get_trtllm_gemm_module,
    ) = _import_gpu_dependencies()
    torch.manual_seed(plan.seed)
    a = torch.randn((plan.m, plan.k), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(
        (plan.n, plan.k), device="cuda", dtype=torch.bfloat16
    )
    reference = torch.mm(a, weight.t()).detach()
    weight_mxfp8, weight_scale = _prepare_trtllm_weight(
        mxfp8_quantize,
        shuffle_matrix_a,
        shuffle_matrix_sf_a,
        weight,
    )
    weight_mxfp8_t = weight_mxfp8.t()
    a_mxfp8, a_scale = _quantize(
        mxfp8_quantize, a, layout=plan.layout
    )
    output = torch.empty(
        (plan.m, plan.n), device="cuda", dtype=torch.bfloat16
    )
    workspace = torch.empty(
        (workspace_mb * 1024 * 1024,), device="cuda", dtype=torch.int8
    )
    runner = get_trtllm_gemm_module().trtllm_mxfp8_gemm_runner(
        use_8x4_sf_layout=plan.layout == "8x4"
    )
    inputs = [
        a_mxfp8,
        weight_mxfp8_t,
        a_scale,
        weight_scale,
        torch.bfloat16,
        output,
        workspace,
    ]
    valid_tactics = enumerate_valid_tactics(
        runner,
        inputs,
        _ShapeProfile(
            tuple(a_mxfp8.shape), tuple(weight_mxfp8_t.shape)
        ),
    )
    num_valid_tactics = len(valid_tactics)

    observations: list[BenchmarkObservation] = []
    for tactic in build_tactic_plan(
        plan, valid_tactics=valid_tactics, completed=completed
    ):
        try:
            candidate = runner.forward(inputs, tactic=tactic)
            torch.cuda.synchronize()
            all_finite, cosine_similarity = _compare_outputs(
                torch, reference, candidate
            )
            if (
                not all_finite
                or cosine_similarity < minimum_cosine_similarity
            ):
                observation = BenchmarkObservation(
                    layout=plan.layout,
                    m=plan.m,
                    n=plan.n,
                    k=plan.k,
                    config_sha256=plan.config_sha256,
                    tactic=tactic,
                    repeat=plan.repeat,
                    median_ms=None,
                    all_finite=all_finite,
                    cosine_similarity=cosine_similarity,
                    status="incorrect",
                    seed=plan.seed,
                    warmup=warmup,
                    iterations=iterations,
                    device_name=runtime.device_name,
                    compute_capability=runtime.compute_capability,
                    vllm_version=runtime.vllm_version,
                    flashinfer_version=runtime.flashinfer_version,
                    container_sha256=runtime.container_sha256,
                    num_valid_tactics=num_valid_tactics,
                )
            else:
                median_ms = _time_cuda(
                    torch,
                    lambda tactic=tactic: runner.forward(
                        inputs, tactic=tactic
                    ),
                    warmup,
                    iterations,
                )
                observation = BenchmarkObservation(
                    layout=plan.layout,
                    m=plan.m,
                    n=plan.n,
                    k=plan.k,
                    config_sha256=plan.config_sha256,
                    tactic=tactic,
                    repeat=plan.repeat,
                    median_ms=median_ms,
                    all_finite=True,
                    cosine_similarity=cosine_similarity,
                    status=_SUCCESS_STATUS,
                    seed=plan.seed,
                    warmup=warmup,
                    iterations=iterations,
                    device_name=runtime.device_name,
                    compute_capability=runtime.compute_capability,
                    vllm_version=runtime.vllm_version,
                    flashinfer_version=runtime.flashinfer_version,
                    container_sha256=runtime.container_sha256,
                    num_valid_tactics=num_valid_tactics,
                )
        except Exception as error:
            observation = _error_observation(
                plan,
                tactic,
                runtime,
                warmup=warmup,
                iterations=iterations,
                num_valid_tactics=num_valid_tactics,
                error=error,
            )
        observations.append(observation)
    return tuple(observations)


def _write_observation(path: Path, observation: BenchmarkObservation) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(asdict(observation), sort_keys=True, allow_nan=False)
            + "\n"
        )
        stream.flush()


def _reject_path_collisions(
    named_paths: Sequence[tuple[str, Path]],
) -> None:
    seen: dict[Path, str] = {}
    for name, path in named_paths:
        resolved = path.resolve()
        previous = seen.get(resolved)
        if previous is not None:
            raise ValueError(
                f"path collision between {previous} and {name}: {resolved}"
            )
        seen[resolved] = name


def _require_new_output(path: Path, field: str) -> None:
    if path.exists() or path.is_symlink():
        raise ValueError(f"{field} already exists: {path.resolve()}")


def _run_inventory(args: argparse.Namespace) -> int:
    trace_paths = tuple(args.trace)
    _reject_path_collisions(
        [
            ("bootstrap manifest", args.bootstrap_manifest),
            *((f"trace[{index}]", path) for index, path in enumerate(trace_paths)),
            ("inventory output", args.output),
        ]
    )
    _require_new_output(args.output, "inventory output")
    bootstrap_sha256, compatibility = _load_bootstrap_runtime_manifest(
        args.bootstrap_manifest
    )
    inventory = load_shape_inventory(
        trace_paths, expected_config_sha256=bootstrap_sha256
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(
        canonical_json_bytes(
            _inventory_document(
                inventory,
                digest_input_paths(trace_paths),
                bootstrap_sha256,
                compatibility,
            )
        )
    )
    return 0


def _run_shmoo(args: argparse.Namespace) -> int:
    _reject_path_collisions(
        [
            ("inventory input", args.inventory),
            ("shmoo output", args.output),
        ]
    )
    artifact = _load_inventory_document(args.inventory)
    plans = build_benchmark_plan(
        artifact.shapes,
        repeat_count=args.repeat_count,
        base_seed=args.base_seed,
    )
    _require_nonnegative_int(args.warmup, "--warmup")
    _require_positive_int(args.iterations, "--iterations")
    _require_positive_int(args.workspace_mb, "--workspace-mb")
    _require_finite_range(
        args.minimum_cosine_similarity,
        "--minimum-cosine-similarity",
        minimum=0.0,
        maximum=1.0,
        minimum_inclusive=False,
    )
    _require_sha256(args.container_sha256, "--container-sha256")
    _require_nonempty_string(args.vllm_version, "--vllm-version")
    _require_nonempty_string(
        args.flashinfer_version, "--flashinfer-version"
    )
    if args.output.exists() and args.output.is_dir():
        raise ValueError("--output must be a JSONL file, not a directory")

    existing: tuple[BenchmarkObservation, ...] = ()
    if args.output.exists() and args.output.stat().st_size:
        existing = load_benchmark_observations([args.output])
        validate_resume_observations(plans, existing)

    (
        torch,
        flashinfer,
        active_vllm,
        _mm_mxfp8,
        _mxfp8_quantize,
        _shuffle_matrix_a,
        _shuffle_matrix_sf_a,
        _get_trtllm_gemm_module,
    ) = _import_gpu_dependencies()
    major, minor = torch.cuda.get_device_capability()
    active_vllm_version = str(
        getattr(active_vllm, "__version__", "")
    )
    active_flashinfer_version = str(
        getattr(flashinfer, "__version__", "")
    )
    active_device_name = str(torch.cuda.get_device_name())
    active_compute_capability = f"{major}.{minor}"
    validate_active_runtime_identity(
        active_vllm_version=active_vllm_version,
        active_flashinfer_version=active_flashinfer_version,
        active_compute_capability=active_compute_capability,
        active_device_name=active_device_name,
        declared_vllm_version=args.vllm_version,
        declared_flashinfer_version=args.flashinfer_version,
        inventory_compatibility=artifact.compatibility,
    )
    runtime = _GpuRuntimeIdentity(
        device_name=active_device_name,
        compute_capability=active_compute_capability,
        vllm_version=active_vllm_version,
        flashinfer_version=active_flashinfer_version,
        container_sha256=args.container_sha256,
    )

    if existing and existing[0].runtime_identity != (
            artifact.shapes[0].config_sha256,
            runtime.device_name,
            runtime.compute_capability,
            runtime.vllm_version,
            runtime.flashinfer_version,
            runtime.container_sha256,
            args.warmup,
            args.iterations,
    ):
        raise RuntimeError(
            "existing shmoo output has incompatible runtime identity"
        )
    completed = {observation.identity for observation in existing}
    for plan in plans:
        for observation in _benchmark_shape_repeat(
            plan,
            runtime=runtime,
            warmup=args.warmup,
            iterations=args.iterations,
            workspace_mb=args.workspace_mb,
            minimum_cosine_similarity=args.minimum_cosine_similarity,
            completed=completed,
        ):
            _write_observation(args.output, observation)
            completed.add(observation.identity)
    return 0


def _compatibility_from_args(args: argparse.Namespace) -> dict[str, object]:
    return {
        "vllm_version": args.vllm_version,
        "vllm_base_commit": args.vllm_base_commit,
        "flashinfer_version": args.flashinfer_version,
        "compute_capability": args.compute_capability,
        "gpu_family": args.gpu_family,
        "model": args.model,
        "tensor_parallel_size": args.tensor_parallel_size,
    }


def _run_promote(args: argparse.Namespace) -> int:
    observation_paths = tuple(args.observations)
    named_paths: list[tuple[str, Path]] = [
        ("inventory input", args.inventory),
        *(
            (f"observation[{index}]", path)
            for index, path in enumerate(observation_paths)
        ),
        ("manifest output", args.output),
    ]
    if args.qualification_output is not None:
        named_paths.append(
            ("qualification output", args.qualification_output)
        )
    _reject_path_collisions(named_paths)
    _require_new_output(args.output, "manifest output")
    if args.qualification_output is not None:
        _require_new_output(
            args.qualification_output, "qualification output"
        )

    artifact = _load_inventory_document(args.inventory)
    declared_compatibility = _compatibility_from_args(args)
    for field, declared in declared_compatibility.items():
        if not _compatibility_values_match(
            field, artifact.compatibility[field], declared
        ):
            display = {
                "vllm_version": "observed vLLM version",
                "flashinfer_version": "observed FlashInfer version",
            }.get(field, f"declared {field}")
            raise ValueError(
                f"{display} does not match traced inventory compatibility"
            )
    manifest, qualified = regenerate_qualified_manifest(
        inventory_path=args.inventory,
        observation_paths=observation_paths,
        minimum_repeat_count=args.repeat_count,
        minimum_cosine_similarity=args.minimum_cosine_similarity,
        minimum_speedup_vs_default=args.minimum_speedup_vs_default,
        qualification_scope=args.qualification_scope,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json_bytes(manifest))
    if args.qualification_output is not None:
        args.qualification_output.parent.mkdir(parents=True, exist_ok=True)
        args.qualification_output.write_bytes(
            canonical_json_bytes(
                {
                    "schema_version": 1,
                    "qualified_shapes": [
                        asdict(item) for item in qualified
                    ],
                }
            )
        )
    return 0


def _run_validate(args: argparse.Namespace) -> int:
    observation_paths = tuple(args.observations)
    _reject_path_collisions(
        [
            ("manifest input", args.manifest),
            ("inventory input", args.inventory),
            *(
                (f"observation[{index}]", path)
                for index, path in enumerate(observation_paths)
            ),
        ]
    )
    validate_manifest(
        args.manifest,
        inventory_path=args.inventory,
        observation_paths=observation_paths,
        minimum_repeat_count=args.repeat_count,
        minimum_cosine_similarity=args.minimum_cosine_similarity,
        minimum_speedup_vs_default=args.minimum_speedup_vs_default,
        qualification_scope=args.qualification_scope,
        actual_vllm_version=args.vllm_version,
        actual_flashinfer_version=args.flashinfer_version,
        actual_compute_capability=_parse_compute_capability(
            args.compute_capability
        ),
        actual_model=args.model,
        actual_tensor_parallel_size=args.tensor_parallel_size,
        check=args.check,
    )
    return 0


def _parse_compute_capability(raw: str) -> tuple[int, int]:
    parts = raw.split(".")
    if (
        len(parts) != 2
        or not all(part.isdigit() for part in parts)
        or any(not part for part in parts)
    ):
        raise ValueError("compute capability must be MAJOR.MINOR")
    return int(parts[0]), int(parts[1])


def _add_compatibility_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--vllm-version", required=True)
    parser.add_argument("--vllm-base-commit", required=True)
    parser.add_argument("--flashinfer-version", required=True)
    parser.add_argument("--compute-capability", required=True)
    parser.add_argument("--gpu-family", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tensor-parallel-size", type=int, required=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    inventory = subparsers.add_parser(
        "inventory", help="aggregate exact physical shapes from trace JSONL"
    )
    inventory.add_argument("--trace", type=Path, action="append", required=True)
    inventory.add_argument("--bootstrap-manifest", type=Path, required=True)
    inventory.add_argument("--output", type=Path, required=True)
    inventory.set_defaults(handler=_run_inventory)

    shmoo = subparsers.add_parser(
        "shmoo", help="append direct TRTLLM tactic observations"
    )
    shmoo.add_argument("--inventory", type=Path, required=True)
    shmoo.add_argument("--output", type=Path, required=True)
    shmoo.add_argument("--repeat-count", type=int, default=3)
    shmoo.add_argument("--base-seed", type=int, default=1234)
    shmoo.add_argument("--warmup", type=int, default=10)
    shmoo.add_argument("--iterations", type=int, default=80)
    shmoo.add_argument("--workspace-mb", type=int, default=256)
    shmoo.add_argument("--minimum-cosine-similarity", type=float, default=0.999)
    shmoo.add_argument("--vllm-version", required=True)
    shmoo.add_argument("--flashinfer-version", required=True)
    shmoo.add_argument("--container-sha256", required=True)
    shmoo.set_defaults(handler=_run_shmoo)

    promote = subparsers.add_parser(
        "promote", help="qualify observations and write a runtime manifest"
    )
    promote.add_argument("--inventory", type=Path, required=True)
    promote.add_argument(
        "--observations", type=Path, action="append", required=True
    )
    promote.add_argument("--output", type=Path, required=True)
    promote.add_argument("--qualification-output", type=Path)
    promote.add_argument("--repeat-count", type=int, default=3)
    promote.add_argument(
        "--minimum-cosine-similarity", type=float, default=0.999
    )
    promote.add_argument(
        "--minimum-speedup-vs-default", type=float, default=1.02
    )
    promote.add_argument("--qualification-scope", required=True)
    _add_compatibility_arguments(promote)
    promote.set_defaults(handler=_run_promote)

    validate = subparsers.add_parser(
        "validate", help="load a manifest and check deterministic bytes"
    )
    validate.add_argument("--manifest", type=Path, required=True)
    validate.add_argument("--inventory", type=Path, required=True)
    validate.add_argument(
        "--observations", type=Path, action="append", required=True
    )
    validate.add_argument("--repeat-count", type=int, default=3)
    validate.add_argument(
        "--minimum-cosine-similarity", type=float, default=0.999
    )
    validate.add_argument(
        "--minimum-speedup-vs-default", type=float, default=1.02
    )
    validate.add_argument("--qualification-scope", required=True)
    validate.add_argument("--vllm-version", required=True)
    validate.add_argument("--flashinfer-version", required=True)
    validate.add_argument("--compute-capability", required=True)
    validate.add_argument("--model", required=True)
    validate.add_argument("--tensor-parallel-size", type=int, required=True)
    validate.add_argument("--check", action="store_true")
    validate.set_defaults(handler=_run_validate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one explicit offline qualification stage."""
    args = _parser().parse_args(argv)
    try:
        return cast(int, args.handler(args))
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
