# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure Dynamic SD profiling and schedule selection helpers."""

from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

MeasurementStatus = Literal["complete", "infeasible", "failed"]


@dataclass(frozen=True, init=False)
class ProfileIdentity:
    """Stable, JSON-compatible identity for a Dynamic SD profile."""

    _serialized_payload: str = field(repr=False)

    def __init__(self, payload: Mapping[str, object]) -> None:
        if not all(isinstance(key, str) for key in payload):
            raise ValueError("Profile identity keys must be strings.")
        try:
            serialized = json.dumps(
                dict(payload),
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Profile identity payload must be JSON-compatible."
            ) from exc
        object.__setattr__(self, "_serialized_payload", serialized)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> ProfileIdentity:
        """Create an identity detached from the caller's mutable mapping."""
        return cls(payload)

    @property
    def payload(self) -> dict[str, object]:
        """Return a detached JSON-compatible representation."""
        decoded = json.loads(self._serialized_payload)
        assert isinstance(decoded, dict)
        return decoded


@dataclass(frozen=True)
class Measurement:
    """One Dynamic SD throughput observation."""

    scheduler_key: int
    k: int
    repeat: int
    output_tokens_per_second: float | None
    status: MeasurementStatus
    workload_hash: str

    def __post_init__(self) -> None:
        if self.scheduler_key <= 0:
            raise ValueError("scheduler_key must be positive.")
        if self.k < 0:
            raise ValueError("k must be non-negative.")
        if self.repeat < 0:
            raise ValueError("repeat must be non-negative.")
        if not self.workload_hash:
            raise ValueError("workload_hash must not be empty.")
        if self.status not in {"complete", "infeasible", "failed"}:
            raise ValueError(f"Unknown measurement status: {self.status!r}.")
        if self.status == "complete":
            if self.output_tokens_per_second is None or not math.isfinite(
                self.output_tokens_per_second
            ):
                raise ValueError("Completed measurements require finite throughput.")
            if self.output_tokens_per_second <= 0:
                raise ValueError("Completed measurements require positive throughput.")
        elif self.output_tokens_per_second is not None:
            raise ValueError("Incomplete measurements must not include throughput.")


@dataclass(frozen=True)
class SelectionPolicy:
    """Thresholds and profiling grid used for Dynamic SD selection."""

    configured_ks: tuple[int, ...] = (0, 1, 2, 3, 4, 5)
    within_best_fraction: float = 0.02
    min_enable_gain: float = 0.05
    max_cv: float = 0.05
    confidence_level: float = 0.95

    def __post_init__(self) -> None:
        configured_ks = self.configured_ks
        if not configured_ks:
            raise ValueError("configured_ks must not be empty.")
        if len(set(configured_ks)) != len(configured_ks):
            raise ValueError("configured_ks must not contain duplicates.")
        if any(k < 0 for k in configured_ks):
            raise ValueError("configured_ks must be non-negative.")
        fractional_thresholds = {
            "within_best_fraction": self.within_best_fraction,
            "min_enable_gain": self.min_enable_gain,
            "max_cv": self.max_cv,
        }
        for name, value in fractional_thresholds.items():
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be a finite fraction in [0, 1].")
        if not math.isfinite(self.confidence_level) or not (
            0 < self.confidence_level < 1
        ):
            raise ValueError("confidence_level must be between zero and one.")


@dataclass(frozen=True)
class SelectionResult:
    """Selected Dynamic SD schedule and the statistics supporting it."""

    selected_k: dict[int, int]
    requires_k_extension: bool
    infeasible_ks: dict[int, tuple[int, ...]]
    median_throughputs: dict[int, dict[int, float]]
    gain_intervals: dict[int, dict[int, tuple[float, float]]]


def profile_id(identity: ProfileIdentity) -> str:
    """Return the stable short SHA-256 identifier for a profile identity."""
    payload = json.dumps(identity.payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def select_schedule(
    measurements: Sequence[Measurement], policy: SelectionPolicy
) -> SelectionResult:
    """Select a K for each scheduler key from complete profiling measurements.

    K candidates are compared by median throughput. When K=0 is feasible, a
    non-zero choice must also clear the lower confidence bound for its paired
    gain over K=0.
    """
    if not measurements:
        raise ValueError("At least one measurement is required.")
    configured_ks = policy.configured_ks

    rows_by_scheduler_key: dict[int, list[Measurement]] = defaultdict(list)
    for measurement in measurements:
        if measurement.k not in configured_ks:
            raise ValueError(
                f"scheduler key {measurement.scheduler_key} has unconfigured "
                f"K={measurement.k}."
            )
        rows_by_scheduler_key[measurement.scheduler_key].append(measurement)

    selected_k: dict[int, int] = {}
    infeasible_ks: dict[int, tuple[int, ...]] = {}
    median_throughputs: dict[int, dict[int, float]] = {}
    gain_intervals: dict[int, dict[int, tuple[float, float]]] = {}
    requires_k_extension = False

    for scheduler_key in sorted(rows_by_scheduler_key):
        key_result = _select_scheduler_key(rows_by_scheduler_key[scheduler_key], policy)
        infeasible_ks[scheduler_key] = key_result.infeasible_ks
        median_throughputs[scheduler_key] = key_result.median_throughputs
        gain_intervals[scheduler_key] = key_result.gain_intervals
        requires_k_extension |= key_result.requires_k_extension
        if key_result.selected_k is not None:
            selected_k[scheduler_key] = key_result.selected_k

    return SelectionResult(
        selected_k=selected_k,
        requires_k_extension=requires_k_extension,
        infeasible_ks=infeasible_ks,
        median_throughputs=median_throughputs,
        gain_intervals=gain_intervals,
    )


@dataclass(frozen=True)
class _SchedulerKeySelection:
    selected_k: int | None
    requires_k_extension: bool
    infeasible_ks: tuple[int, ...]
    median_throughputs: dict[int, float]
    gain_intervals: dict[int, tuple[float, float]]


def _select_scheduler_key(
    measurements: Sequence[Measurement], policy: SelectionPolicy
) -> _SchedulerKeySelection:
    scheduler_key = measurements[0].scheduler_key
    if any(measurement.scheduler_key != scheduler_key for measurement in measurements):
        raise ValueError("Measurements must share a scheduler key.")
    if len({measurement.workload_hash for measurement in measurements}) != 1:
        raise ValueError(
            f"scheduler key {scheduler_key} measurements must share a workload hash."
        )

    rows_by_k: dict[int, list[Measurement]] = defaultdict(list)
    for measurement in measurements:
        rows_by_k[measurement.k].append(measurement)

    complete_by_k: dict[int, dict[int, float]] = {}
    infeasible: list[int] = []
    configured_ks = policy.configured_ks
    for k in configured_ks:
        rows = rows_by_k.get(k)
        if not rows:
            raise ValueError(f"scheduler key {scheduler_key} missing K={k}.")

        statuses = {row.status for row in rows}
        if len(statuses) != 1:
            raise ValueError(
                f"scheduler key {scheduler_key} K={k} mixes measurement statuses."
            )
        status = rows[0].status
        if status == "infeasible":
            infeasible.append(k)
            continue
        if status == "failed":
            raise ValueError(f"scheduler key {scheduler_key} K={k} failed profiling.")

        complete_by_k[k] = _complete_repeats(rows, scheduler_key, k, policy)

    if not complete_by_k:
        return _SchedulerKeySelection(None, False, tuple(infeasible), {}, {})

    repeat_sets = {frozenset(repeats) for repeats in complete_by_k.values()}
    if len(repeat_sets) != 1:
        raise ValueError(f"scheduler key {scheduler_key} requires paired repeats.")
    repeat_ids = next(iter(repeat_sets))
    if len(repeat_ids) < 3:
        raise ValueError(
            f"scheduler key {scheduler_key} requires at least three paired "
            "complete repeats."
        )

    medians = {
        k: statistics.median(repeats.values()) for k, repeats in complete_by_k.items()
    }
    best_throughput = max(medians.values())
    maximum_configured_k = max(configured_ks)
    requires_k_extension = (
        maximum_configured_k > 0
        and medians.get(maximum_configured_k) == best_throughput
    )
    candidates = [
        k
        for k, throughput in medians.items()
        if throughput >= best_throughput * (1 - policy.within_best_fraction)
    ]
    selected = min(candidates)

    intervals: dict[int, tuple[float, float]] = {}
    baseline = complete_by_k.get(0)
    if baseline is not None:
        intervals[0] = (0.0, 0.0)
        for k, repeats in complete_by_k.items():
            if k == 0:
                continue
            intervals[k] = _bootstrap_paired_median_gain_interval(
                baseline,
                repeats,
                policy.confidence_level,
            )
        if selected != 0 and intervals[selected][0] < policy.min_enable_gain:
            selected = 0

    return _SchedulerKeySelection(
        selected,
        requires_k_extension,
        tuple(infeasible),
        medians,
        intervals,
    )


def _complete_repeats(
    rows: Sequence[Measurement], scheduler_key: int, k: int, policy: SelectionPolicy
) -> dict[int, float]:
    repeats: dict[int, float] = {}
    for row in rows:
        if row.repeat in repeats:
            raise ValueError(
                f"scheduler key {scheduler_key} K={k} has duplicate repeat "
                f"{row.repeat}."
            )
        assert row.output_tokens_per_second is not None
        repeats[row.repeat] = row.output_tokens_per_second

    mean = statistics.fmean(repeats.values())
    cv = statistics.stdev(repeats.values()) / mean if len(repeats) > 1 else 0.0
    if cv > policy.max_cv:
        raise ValueError(
            f"scheduler key {scheduler_key} K={k} exceeds the coefficient of "
            f"variation limit ({cv:.6f} > {policy.max_cv:.6f})."
        )
    return repeats


def _bootstrap_paired_median_gain_interval(
    baseline: Mapping[int, float],
    candidate: Mapping[int, float],
    confidence_level: float,
    *,
    samples: int = 2_000,
) -> tuple[float, float]:
    random_generator = random.Random(0)
    repeat_ids = sorted(baseline)
    count = len(repeat_ids)
    gains: list[float] = []
    for _ in range(samples):
        sampled_repeat_ids = random_generator.choices(repeat_ids, k=count)
        baseline_median = statistics.median(
            baseline[repeat] for repeat in sampled_repeat_ids
        )
        candidate_median = statistics.median(
            candidate[repeat] for repeat in sampled_repeat_ids
        )
        gains.append(candidate_median / baseline_median - 1)
    gains.sort()
    tail_probability = (1 - confidence_level) / 2
    return (
        _quantile(gains, tail_probability),
        _quantile(gains, 1 - tail_probability),
    )


def _quantile(values: Sequence[float], probability: float) -> float:
    position = probability * (len(values) - 1)
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return values[lower_index]
    fraction = position - lower_index
    return values[lower_index] + fraction * (values[upper_index] - values[lower_index])


def compress_schedule(
    selected_k: Mapping[int, int], max_num_seqs: int
) -> list[tuple[int, int, int]]:
    """Compress selected keys into sorted, inclusive scheduler ranges."""
    if max_num_seqs <= 0:
        raise ValueError("max_num_seqs must be positive.")
    if not selected_k:
        raise ValueError("selected_k must not be empty.")

    entries = sorted(selected_k.items())
    if entries[0][0] != 1:
        raise ValueError("selected_k must start at scheduler key 1.")
    if any(key <= 0 or key > max_num_seqs for key, _ in entries):
        raise ValueError("selected_k keys must be within [1, max_num_seqs].")
    if any(k < 0 for _, k in entries):
        raise ValueError("selected_k values must be non-negative.")

    schedule: list[tuple[int, int, int]] = []
    for index, (range_start, k) in enumerate(entries):
        range_end = (
            entries[index + 1][0] - 1 if index + 1 < len(entries) else max_num_seqs
        )
        if schedule and schedule[-1][2] == k:
            previous_start, _, _ = schedule[-1]
            schedule[-1] = (previous_start, range_end, k)
        else:
            schedule.append((range_start, range_end, k))
    return schedule
