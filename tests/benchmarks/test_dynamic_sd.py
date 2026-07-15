# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the pure Dynamic SD profile selection core."""

import math

import pytest

from vllm.benchmarks.dynamic_sd_core import (
    Measurement,
    ProfileIdentity,
    SelectionPolicy,
    compress_schedule,
    profile_id,
    select_schedule,
)


def measurements_for_key(
    scheduler_key: int,
    values_by_k: dict[int, list[float]],
    *,
    workload_hash: str = "workload-a",
) -> list[Measurement]:
    return [
        Measurement(
            scheduler_key=scheduler_key,
            k=k,
            repeat=repeat,
            output_tokens_per_second=value,
            status="complete",
            workload_hash=workload_hash,
        )
        for k, values in values_by_k.items()
        for repeat, value in enumerate(values)
    ]


def infeasible_measurement(scheduler_key: int, k: int) -> Measurement:
    return Measurement(
        scheduler_key=scheduler_key,
        k=k,
        repeat=0,
        output_tokens_per_second=None,
        status="infeasible",
        workload_hash="workload-a",
    )


def test_profile_id_is_stable_for_equivalent_identity():
    left = ProfileIdentity.from_mapping({"model": "Qwen/Qwen3-235B-A22B", "tp": 8})
    right = ProfileIdentity.from_mapping({"tp": 8, "model": "Qwen/Qwen3-235B-A22B"})

    assert profile_id(left) == profile_id(right)


def test_profile_identity_does_not_retain_mutable_input_mapping():
    payload = {"model": "Qwen/Qwen3-235B-A22B", "tp": 8}
    identity = ProfileIdentity.from_mapping(payload)
    payload["tp"] = 16

    assert identity.payload == {"model": "Qwen/Qwen3-235B-A22B", "tp": 8}


def test_selector_uses_runtime_k0_when_gain_is_below_margin():
    rows = measurements_for_key(1, {0: [100.0, 101.0, 99.0], 1: [104.0] * 3})

    result = select_schedule(rows, SelectionPolicy(min_enable_gain=0.05))

    assert result.selected_k == {1: 0}


def test_selector_requires_every_configured_k_or_infeasible_marker():
    rows = measurements_for_key(1, {0: [100.0, 100.0, 100.0]})

    with pytest.raises(ValueError, match="missing K=1"):
        select_schedule(rows, SelectionPolicy(configured_ks=(0, 1)))


def test_selector_records_infeasible_k_without_rejecting_the_grid():
    rows = measurements_for_key(1, {0: [100.0] * 3})
    rows.append(infeasible_measurement(1, 1))

    result = select_schedule(rows, SelectionPolicy(configured_ks=(0, 1)))

    assert result.selected_k == {1: 0}
    assert result.infeasible_ks == {1: (1,)}


def test_selector_rejects_measurements_from_different_workloads():
    rows = measurements_for_key(1, {0: [100.0], 1: [110.0]})
    rows[-1] = Measurement(
        scheduler_key=1,
        k=1,
        repeat=0,
        output_tokens_per_second=110.0,
        status="complete",
        workload_hash="workload-b",
    )

    with pytest.raises(ValueError, match="workload hash"):
        select_schedule(rows, SelectionPolicy(configured_ks=(0, 1)))


def test_selector_rejects_unpaired_complete_measurements():
    rows = measurements_for_key(1, {0: [100.0, 100.0], 1: [110.0]})

    with pytest.raises(ValueError, match="paired repeats"):
        select_schedule(rows, SelectionPolicy(configured_ks=(0, 1)))


def test_selector_rejects_unstable_measurements_by_coefficient_of_variation():
    rows = measurements_for_key(1, {0: [100.0] * 3, 1: [90.0, 110.0, 90.0]})

    with pytest.raises(ValueError, match="coefficient of variation"):
        select_schedule(rows, SelectionPolicy(configured_ks=(0, 1)))


def test_selector_uses_bootstrap_lower_bound_for_runtime_k0_gate():
    rows = measurements_for_key(1, {0: [100.0] * 3, 1: [110.0, 90.0, 110.0]})
    policy = SelectionPolicy(configured_ks=(0, 1), max_cv=0.2)

    first = select_schedule(rows, policy)
    second = select_schedule(rows, policy)

    assert first.selected_k == {1: 0}
    assert first.gain_intervals == second.gain_intervals
    assert first.gain_intervals[1][1][0] < policy.min_enable_gain


def test_selector_uses_smallest_k_within_best_fraction():
    rows = measurements_for_key(1, {0: [100.0] * 3, 1: [110.0] * 3, 2: [111.0] * 3})

    result = select_schedule(rows, SelectionPolicy(configured_ks=(0, 1, 2)))

    assert result.selected_k == {1: 1}


def test_selector_requires_grid_extension_when_kmax_wins():
    rows = measurements_for_key(1, {0: [100.0] * 3, 5: [120.0] * 3})

    result = select_schedule(rows, SelectionPolicy(configured_ks=(0, 5)))

    assert result.requires_k_extension


def test_selector_keeps_all_infeasible_keys_structured_in_the_result():
    rows = [infeasible_measurement(1, 0), infeasible_measurement(1, 1)]

    result = select_schedule(rows, SelectionPolicy(configured_ks=(0, 1)))

    assert result.selected_k == {}
    assert result.infeasible_ks == {1: (0, 1)}
    assert not result.requires_k_extension


def test_compress_schedule_emits_sorted_inclusive_ranges_and_fills_gaps():
    schedule = compress_schedule({4: 1, 1: 2, 3: 3}, max_num_seqs=5)

    assert schedule == [(1, 2, 2), (3, 3, 3), (4, 5, 1)]


def test_compress_schedule_requires_a_schedule_starting_at_one():
    with pytest.raises(ValueError, match="start at scheduler key 1"):
        compress_schedule({2: 1}, max_num_seqs=2)


def test_measurement_rejects_non_finite_completed_throughput():
    with pytest.raises(ValueError, match="finite"):
        Measurement(
            scheduler_key=1,
            k=0,
            repeat=0,
            output_tokens_per_second=math.nan,
            status="complete",
            workload_hash="workload-a",
        )
