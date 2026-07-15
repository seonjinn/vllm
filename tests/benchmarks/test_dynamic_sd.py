# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the pure Dynamic SD profile selection core."""

import hashlib
import json
import math
import os
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import vllm.benchmarks.dynamic_sd_worker as dynamic_sd_worker
from vllm.benchmarks.dynamic_sd_core import (
    Measurement,
    ProfileIdentity,
    SelectionPolicy,
    compress_schedule,
    profile_id,
    select_schedule,
)
from vllm.benchmarks.dynamic_sd_worker import (
    CUDAGraphStep,
    WorkerConfig,
    WorkerMetricSnapshot,
    derive_cudagraph_capture_sizes,
    run_worker,
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


def test_profile_identity_remains_stable_when_nested_payload_copy_is_mutated():
    identity = ProfileIdentity.from_mapping(
        {
            "model": "Qwen/Qwen3-235B-A22B",
            "parallelism": {"tp": 8},
            "tags": ["production"],
        }
    )
    original_profile_id = profile_id(identity)

    payload = identity.payload
    parallelism = payload["parallelism"]
    tags = payload["tags"]
    assert isinstance(parallelism, dict)
    assert isinstance(tags, list)
    parallelism["tp"] = 16
    tags.append("modified")

    assert identity.payload == {
        "model": "Qwen/Qwen3-235B-A22B",
        "parallelism": {"tp": 8},
        "tags": ["production"],
    }
    assert profile_id(identity) == original_profile_id


def test_selector_uses_runtime_k0_when_gain_is_below_margin():
    rows = measurements_for_key(1, {0: [100.0, 101.0, 99.0], 1: [104.0] * 3})

    result = select_schedule(
        rows,
        SelectionPolicy(configured_ks=(0, 1), min_enable_gain=0.05),
    )

    assert result.selected_k == {1: 0}


def test_selector_requires_every_configured_k_or_infeasible_marker():
    rows = measurements_for_key(1, {0: [100.0, 100.0, 100.0]})

    with pytest.raises(ValueError, match="missing K=1"):
        select_schedule(rows, SelectionPolicy(configured_ks=(0, 1)))


def test_selector_treats_default_configured_ks_as_a_strict_complete_grid():
    rows = measurements_for_key(1, {0: [100.0, 100.0, 100.0]})

    with pytest.raises(ValueError, match="missing K=1"):
        select_schedule(rows, SelectionPolicy())


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


def test_selector_requires_at_least_three_paired_complete_repeats():
    rows = measurements_for_key(1, {0: [100.0, 100.0], 1: [110.0, 110.0]})

    with pytest.raises(ValueError, match="at least three paired complete repeats"):
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


def test_selector_bootstraps_ratio_of_paired_sample_medians():
    rows = measurements_for_key(1, {0: [80.0, 80.0, 90.0], 1: [90.0, 80.0, 90.0]})
    policy = SelectionPolicy(
        configured_ks=(0, 1),
        min_enable_gain=0.0,
        max_cv=0.1,
        confidence_level=0.5,
    )

    result = select_schedule(rows, policy)

    assert result.gain_intervals[1][1][1] == pytest.approx(0.125)


def test_selector_uses_smallest_k_within_best_fraction():
    rows = measurements_for_key(1, {0: [100.0] * 3, 1: [110.0] * 3, 2: [111.0] * 3})

    result = select_schedule(rows, SelectionPolicy(configured_ks=(0, 1, 2)))

    assert result.selected_k == {1: 1}


def test_selector_requires_grid_extension_when_kmax_wins():
    rows = measurements_for_key(1, {0: [100.0] * 3, 5: [120.0] * 3})

    result = select_schedule(rows, SelectionPolicy(configured_ks=(0, 5)))

    assert result.requires_k_extension


def test_selector_requires_grid_extension_when_kmax_is_best_before_tolerance():
    rows = measurements_for_key(
        1,
        {0: [100.0] * 3, 1: [119.0] * 3, 2: [120.0] * 3},
    )

    result = select_schedule(rows, SelectionPolicy(configured_ks=(0, 1, 2)))

    assert result.selected_k == {1: 1}
    assert result.requires_k_extension


@pytest.mark.parametrize(
    "policy_factory",
    [
        lambda: SelectionPolicy(within_best_fraction=math.nan),
        lambda: SelectionPolicy(within_best_fraction=math.inf),
        lambda: SelectionPolicy(within_best_fraction=1.01),
        lambda: SelectionPolicy(min_enable_gain=math.nan),
        lambda: SelectionPolicy(min_enable_gain=math.inf),
        lambda: SelectionPolicy(min_enable_gain=1.01),
        lambda: SelectionPolicy(max_cv=math.nan),
        lambda: SelectionPolicy(max_cv=math.inf),
        lambda: SelectionPolicy(max_cv=1.01),
        lambda: SelectionPolicy(confidence_level=math.nan),
        lambda: SelectionPolicy(confidence_level=math.inf),
    ],
)
def test_selection_policy_rejects_invalid_fractional_thresholds(
    policy_factory: Callable[[], SelectionPolicy],
):
    with pytest.raises(ValueError):
        policy_factory()


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


class FakeMetricsReader:
    def __init__(self, llm: "FakeLLM") -> None:
        self.llm = llm

    def snapshot(self) -> WorkerMetricSnapshot:
        return WorkerMetricSnapshot(
            scheduler_steps=dict(self.llm.scheduler_steps),
            cudagraph_steps=dict(self.llm.cudagraph_steps),
        )

    def close(self) -> None:
        self.llm.metrics_reader_closed = True


DRAFT_COMMIT = "d" * 40


class FakeLLM:
    def __init__(
        self,
        engine_kwargs: dict[str, Any],
        *,
        output_token_delta: int = 0,
        scheduler_noise: bool = False,
        fallback_steps: int = 0,
        emit_cudagraph_metrics: bool = True,
        resolved_cudagraph_mode: str | None = None,
        resolved_capture_sizes: tuple[int, ...] | None = None,
        resolved_use_v2_model_runner: bool = True,
        resolved_speculative_overrides: dict[str, Any] | None = None,
        resolved_draft_model: str | None = None,
        resolved_draft_revision: str | None = None,
        resolved_draft_commit: str | None = DRAFT_COMMIT,
    ) -> None:
        self.engine_kwargs = engine_kwargs
        self.output_token_delta = output_token_delta
        self.scheduler_noise = scheduler_noise
        self.fallback_steps = fallback_steps
        self.emit_cudagraph_metrics = emit_cudagraph_metrics
        self.scheduler_steps: dict[tuple[int, int], int] = {}
        self.cudagraph_steps: dict[CUDAGraphStep, int] = {}
        self.generate_calls: list[tuple[list[dict[str, list[int]]], list[Any]]] = []
        self.metrics_reader_closed = False
        self.shutdown_calls = 0

        compilation = engine_kwargs["compilation_config"]
        assert isinstance(compilation, dict)
        capture_sizes = resolved_capture_sizes or tuple(
            compilation["cudagraph_capture_sizes"]
        )
        cudagraph_mode = resolved_cudagraph_mode or str(compilation["cudagraph_mode"])
        speculative_config = engine_kwargs.get("speculative_config")
        resolved_speculative_config = None
        if isinstance(speculative_config, dict):
            resolved_speculative_values = dict(speculative_config)
            resolved_speculative_values.update(resolved_speculative_overrides or {})
            resolved_speculative_values["draft_model_config"] = SimpleNamespace(
                model=resolved_draft_model or speculative_config["model"],
                revision=resolved_draft_revision or speculative_config["revision"],
                hf_config=SimpleNamespace(_commit_hash=resolved_draft_commit),
            )
            resolved_speculative_config = SimpleNamespace(**resolved_speculative_values)
        self.llm_engine = SimpleNamespace(
            engine_core=SimpleNamespace(shutdown=self._shutdown),
            vllm_config=SimpleNamespace(
                use_v2_model_runner=resolved_use_v2_model_runner,
                model_config=SimpleNamespace(
                    model=engine_kwargs["model"],
                    revision=engine_kwargs["revision"],
                    enforce_eager=engine_kwargs["enforce_eager"],
                ),
                parallel_config=SimpleNamespace(
                    data_parallel_size=engine_kwargs["data_parallel_size"],
                    tensor_parallel_size=engine_kwargs.get("tensor_parallel_size", 1),
                ),
                scheduler_config=SimpleNamespace(
                    max_num_seqs=engine_kwargs["max_num_seqs"],
                ),
                cache_config=SimpleNamespace(
                    cache_dtype="auto",
                    enable_prefix_caching=True,
                ),
                compilation_config=SimpleNamespace(
                    cudagraph_mode=cudagraph_mode,
                    cudagraph_capture_sizes=list(capture_sizes),
                    max_cudagraph_capture_size=max(capture_sizes),
                ),
                speculative_config=resolved_speculative_config,
            ),
        )

    def _shutdown(self) -> None:
        self.shutdown_calls += 1

    def generate(
        self,
        prompts: list[dict[str, list[int]]],
        sampling_params: list[Any],
        *,
        use_tqdm: bool,
    ) -> list[SimpleNamespace]:
        assert not use_tqdm
        self.generate_calls.append((prompts, sampling_params))
        scheduler_key = len(prompts)
        speculative_config = self.engine_kwargs.get("speculative_config")
        k = 0
        if isinstance(speculative_config, dict):
            schedule = speculative_config["num_speculative_tokens_per_batch_size"]
            k = schedule[0][2]

        if self.scheduler_noise:
            self.scheduler_steps[(scheduler_key, k)] = (
                self.scheduler_steps.get((scheduler_key, k), 0) + 94
            )
            self.scheduler_steps[(scheduler_key - 1, k)] = (
                self.scheduler_steps.get((scheduler_key - 1, k), 0) + 6
            )
        else:
            self.scheduler_steps[(scheduler_key, k)] = (
                self.scheduler_steps.get((scheduler_key, k), 0) + 1
            )

        num_tokens = scheduler_key * (k + 1)
        if self.emit_cudagraph_metrics:
            full_step = CUDAGraphStep(num_tokens, num_tokens, "FULL")
            self.cudagraph_steps[full_step] = self.cudagraph_steps.get(full_step, 0) + 1
            if self.fallback_steps:
                fallback = CUDAGraphStep(num_tokens, num_tokens, "PIECEWISE")
                self.cudagraph_steps[fallback] = (
                    self.cudagraph_steps.get(fallback, 0) + self.fallback_steps
                )

        return [
            SimpleNamespace(
                prompt_token_ids=prompt["prompt_token_ids"],
                outputs=[
                    SimpleNamespace(
                        token_ids=list(
                            range(
                                sampling_params[index].max_tokens
                                + self.output_token_delta
                            )
                        )
                    )
                ],
            )
            for index, prompt in enumerate(prompts)
        ]


class FakeLLMFactory:
    def __init__(self, **llm_options: Any) -> None:
        self.llm_options = llm_options
        self.calls: list[dict[str, Any]] = []
        self.instances: list[FakeLLM] = []

    def __call__(self, **engine_kwargs: Any) -> FakeLLM:
        self.calls.append(engine_kwargs)
        llm = FakeLLM(engine_kwargs, **self.llm_options)
        self.instances.append(llm)
        return llm


def worker_config(
    tmp_path: Path,
    *,
    k: int | None = 3,
    scheduler_keys: tuple[int, ...] = (2, 4),
    kmax: int = 5,
    data_parallel_size: int = 1,
) -> WorkerConfig:
    capture_schedule = {key: k or 0 for key in scheduler_keys}
    return WorkerConfig(
        output_path=tmp_path / "worker-result.json",
        profile_identity=ProfileIdentity.from_mapping(
            {
                "model": "target-model",
                "model_revision": "target-revision",
                "speculative_model": "draft-model",
                "speculative_revision": DRAFT_COMMIT,
            }
        ),
        model="target-model",
        model_revision="target-revision",
        speculative_model="draft-model",
        speculative_revision=DRAFT_COMMIT,
        speculative_method="eagle3",
        k=k,
        kmax=kmax,
        scheduler_keys=scheduler_keys,
        prompt_token_ids=tuple((10 + index, 20 + index) for index in range(4)),
        max_tokens=4,
        warmups=1,
        repeats=2,
        seed=17,
        data_parallel_size=data_parallel_size,
        cudagraph_capture_sizes=derive_cudagraph_capture_sizes(capture_schedule),
        engine_kwargs={
            "tensor_parallel_size": 2,
            "enable_prefix_caching": True,
        },
    )


@pytest.fixture
def v2_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")


@pytest.fixture(autouse=True)
def reset_worker_process_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dynamic_sd_worker, "_worker_invoked", False, raising=False)


def run_fake_worker(
    config: WorkerConfig,
    factory: FakeLLMFactory,
):
    return run_worker(
        config,
        factory,
        metrics_reader_factory=FakeMetricsReader,
    )


def test_capture_sizes_are_total_input_rows_for_each_dynamic_key():
    assert derive_cudagraph_capture_sizes({1: 0, 2: 3, 4: 1}) == (1, 8)


def test_vanilla_has_no_speculative_config_and_captures_request_counts(
    tmp_path: Path,
    v2_runner: None,
):
    config = worker_config(tmp_path, k=None)
    factory = FakeLLMFactory()

    result = run_fake_worker(config, factory)

    assert "speculative_config" not in result.engine_kwargs
    assert result.engine_kwargs["compilation_config"]["cudagraph_capture_sizes"] == [
        2,
        4,
    ]


def test_runtime_k0_keeps_common_kmax_and_drafter_loaded(
    tmp_path: Path,
    v2_runner: None,
):
    config = worker_config(tmp_path, k=0)
    factory = FakeLLMFactory()

    result = run_fake_worker(config, factory)

    speculative = result.engine_kwargs["speculative_config"]
    assert speculative["num_speculative_tokens"] == 5
    assert speculative["num_speculative_tokens_per_batch_size"] == [[1, 4, 0]]
    assert speculative["draft_sample_method"] == "probabilistic"
    assert speculative["rejection_sample_method"] == "standard"
    assert result.engine_kwargs["compilation_config"]["cudagraph_capture_sizes"] == [
        2,
        4,
    ]


def test_fixed_k_uses_one_range_schedule_and_fixed_output_sampling(
    tmp_path: Path,
    v2_runner: None,
):
    config = worker_config(tmp_path, k=3)
    factory = FakeLLMFactory()

    result = run_fake_worker(config, factory)

    speculative = result.engine_kwargs["speculative_config"]
    assert speculative["num_speculative_tokens"] == 5
    assert speculative["num_speculative_tokens_per_batch_size"] == [[1, 4, 3]]
    assert result.engine_kwargs["compilation_config"]["cudagraph_capture_sizes"] == [
        8,
        16,
    ]
    first_params = factory.instances[0].generate_calls[0][1]
    assert [params.seed for params in first_params] == [17, 18]
    assert all(params.temperature == 1.0 for params in first_params)
    assert all(params.top_p == 1.0 for params in first_params)
    assert all(params.min_tokens == config.max_tokens for params in first_params)
    assert all(params.max_tokens == config.max_tokens for params in first_params)
    assert all(params.ignore_eos for params in first_params)
    assert len(factory.instances[0].generate_calls) == len(config.scheduler_keys) * (
        config.warmups + config.repeats
    )


def test_worker_rejects_capture_sizes_not_matched_to_forced_k(
    tmp_path: Path,
    v2_runner: None,
):
    config = replace(
        worker_config(tmp_path, k=3),
        cudagraph_capture_sizes=(2, 4),
    )
    factory = FakeLLMFactory()

    with pytest.raises(ValueError, match="capture sizes"):
        run_fake_worker(config, factory)

    assert factory.calls == []


def test_worker_rejects_unstable_scheduler_key_coverage_and_records_failure(
    tmp_path: Path,
    v2_runner: None,
):
    config = worker_config(tmp_path, k=3, scheduler_keys=(4,))
    factory = FakeLLMFactory(scheduler_noise=True)

    with pytest.raises(ValueError, match="below 95%"):
        run_fake_worker(config, factory)

    payload = json.loads(config.output_path.read_text())
    assert payload["status"] == "failed"
    assert payload["output_tokens_per_second"] is None


def test_worker_rejects_vllm_data_parallelism(
    tmp_path: Path,
    v2_runner: None,
):
    config = worker_config(tmp_path, data_parallel_size=2)
    config.output_path.write_text('{"status": "stale"}\n')
    factory = FakeLLMFactory()

    with pytest.raises(ValueError, match="data_parallel_size must be 1"):
        run_fake_worker(config, factory)

    payload = json.loads(config.output_path.read_text())
    assert payload["status"] == "failed"
    assert payload["output_tokens_per_second"] is None
    assert "data_parallel_size must be 1" in payload["error"]
    assert factory.calls == []


def test_worker_engine_build_failure_replaces_stale_result_with_failure(
    tmp_path: Path,
    v2_runner: None,
):
    config = replace(
        worker_config(tmp_path),
        engine_kwargs={"observability_config": "invalid"},
    )
    config.output_path.write_text('{"status": "stale"}\n')
    factory = FakeLLMFactory()

    with pytest.raises(ValueError, match="observability_config must be a mapping"):
        run_fake_worker(config, factory)

    payload = json.loads(config.output_path.read_text())
    assert payload["status"] == "failed"
    assert payload["output_tokens_per_second"] is None
    assert "observability_config must be a mapping" in payload["error"]
    assert factory.calls == []


def test_worker_rejects_v1_model_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")

    with pytest.raises(ValueError, match="VLLM_USE_V2_MODEL_RUNNER=1"):
        run_fake_worker(worker_config(tmp_path), FakeLLMFactory())


def test_worker_rejects_resolved_v1_model_runner(
    tmp_path: Path,
    v2_runner: None,
):
    config = worker_config(tmp_path)

    with pytest.raises(ValueError, match="Resolved use_v2_model_runner must be true"):
        run_fake_worker(
            config,
            FakeLLMFactory(resolved_use_v2_model_runner=False),
        )

    payload = json.loads(config.output_path.read_text())
    assert payload["status"] == "failed"


def test_worker_is_one_shot_and_second_invocation_fails_closed(
    tmp_path: Path,
    v2_runner: None,
):
    config = worker_config(tmp_path, scheduler_keys=(2,))
    first_factory = FakeLLMFactory()
    second_factory = FakeLLMFactory()
    run_fake_worker(config, first_factory)

    with pytest.raises(RuntimeError, match="fresh process"):
        run_fake_worker(config, second_factory)

    payload = json.loads(config.output_path.read_text())
    assert payload["status"] == "failed"
    assert "fresh process" in payload["error"]
    assert len(first_factory.calls) == 1
    assert second_factory.calls == []


def test_worker_restores_environment_and_cleans_reader_and_engine(
    tmp_path: Path,
    v2_runner: None,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("VLLM_DYNAMIC_SD_PROFILE_METRICS", "original")
    factory = FakeLLMFactory()

    run_fake_worker(worker_config(tmp_path, scheduler_keys=(2,)), factory)

    llm = factory.instances[0]
    assert os.environ["VLLM_DYNAMIC_SD_PROFILE_METRICS"] == "original"
    assert llm.metrics_reader_closed
    assert llm.shutdown_calls == 1


def test_worker_rejects_resolved_piecewise_only_mode(
    tmp_path: Path,
    v2_runner: None,
):
    factory = FakeLLMFactory(resolved_cudagraph_mode="PIECEWISE")

    with pytest.raises(ValueError, match="FULL_AND_PIECEWISE"):
        run_fake_worker(worker_config(tmp_path), factory)


def test_worker_rejects_resolved_capture_set_without_full_shape_coverage(
    tmp_path: Path,
    v2_runner: None,
):
    factory = FakeLLMFactory(resolved_capture_sizes=(8,))

    with pytest.raises(ValueError, match="Resolved capture sizes"):
        run_fake_worker(worker_config(tmp_path), factory)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("model", "other-draft", "model"),
        ("revision", "e" * 40, "revision"),
        ("method", "draft_model", "method"),
        ("draft_sample_method", "greedy", "draft_sample_method"),
        ("rejection_sample_method", "block", "rejection_sample_method"),
        ("num_speculative_tokens", 4, "Kmax"),
        ("num_speculative_tokens_per_batch_size", [[1, 4, 2]], "schedule"),
    ],
)
def test_worker_rejects_changed_resolved_speculative_controls(
    tmp_path: Path,
    v2_runner: None,
    field: str,
    value: object,
    error: str,
):
    factory = FakeLLMFactory(resolved_speculative_overrides={field: value})

    with pytest.raises(ValueError, match=error):
        run_fake_worker(worker_config(tmp_path), factory)


def test_worker_requires_immutable_requested_drafter_revision(
    tmp_path: Path,
    v2_runner: None,
):
    config = replace(worker_config(tmp_path), speculative_revision="main")

    with pytest.raises(ValueError, match="immutable.*commit"):
        run_fake_worker(config, FakeLLMFactory())

    assert json.loads(config.output_path.read_text())["status"] == "failed"


def test_worker_rejects_mismatched_resolved_drafter_commit(
    tmp_path: Path,
    v2_runner: None,
):
    factory = FakeLLMFactory(resolved_draft_commit="e" * 40)

    with pytest.raises(ValueError, match="drafter commit"):
        run_fake_worker(worker_config(tmp_path), factory)


def test_worker_fails_closed_when_measured_cudagraph_metrics_are_empty(
    tmp_path: Path,
    v2_runner: None,
):
    config = worker_config(tmp_path, scheduler_keys=(2,))

    with pytest.raises(ValueError, match="No CUDA Graph metrics"):
        run_fake_worker(
            config,
            FakeLLMFactory(emit_cudagraph_metrics=False),
        )

    payload = json.loads(config.output_path.read_text())
    assert payload["status"] == "failed"
    assert payload["output_tokens_per_second"] is None


def test_worker_binds_canonical_workload_provenance_to_profile_identity(
    tmp_path: Path,
    v2_runner: None,
):
    config = worker_config(tmp_path, scheduler_keys=(2,))

    result = run_fake_worker(config, FakeLLMFactory())

    expected_workload = {
        "schema_version": 1,
        "scheduler_keys": [2],
        "prompt_token_ids": [[10, 20], [11, 21]],
        "max_output_tokens": 4,
        "warmups": 1,
        "repeats": 2,
        "sampling_parameters": {
            "temperature": 1.0,
            "top_p": 1.0,
            "min_tokens": 4,
            "max_tokens": 4,
            "ignore_eos": True,
        },
        "seed_provenance": {
            "base_seed": 17,
            "seed_stride": 4,
            "formula": "base_seed + repeat * seed_stride + request_index",
            "warmup_seeds": {"2": [[17, 18]]},
            "repeat_seeds": {"2": [[17, 18], [21, 22]]},
        },
    }
    expected_hash = hashlib.sha256(
        json.dumps(
            expected_workload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    assert result.workload_identity == expected_workload
    assert result.workload_hash == expected_hash
    assert result.profile_identity["workload_hash"] == expected_hash
    assert result.profile_id == profile_id(
        ProfileIdentity.from_mapping(result.profile_identity)
    )


def test_worker_rejects_declared_workload_hash_not_bound_to_exact_workload(
    tmp_path: Path,
    v2_runner: None,
):
    config = worker_config(tmp_path, scheduler_keys=(2,))
    mismatched_identity = config.profile_identity.payload
    mismatched_identity["workload_hash"] = "not-the-canonical-hash"
    config = replace(
        config,
        profile_identity=ProfileIdentity.from_mapping(mismatched_identity),
    )

    with pytest.raises(ValueError, match="workload_hash"):
        run_fake_worker(config, FakeLLMFactory())

    payload = json.loads(config.output_path.read_text())
    assert payload["status"] == "failed"
    assert payload["workload_hash"]


def test_worker_exact_output_work_failure_has_no_zero_throughput(
    tmp_path: Path,
    v2_runner: None,
):
    config = worker_config(tmp_path)
    factory = FakeLLMFactory(output_token_delta=-1)

    with pytest.raises(ValueError, match="exactly 4 output tokens"):
        run_fake_worker(config, factory)

    payload = json.loads(config.output_path.read_text())
    assert payload["status"] == "failed"
    assert payload["output_tokens_per_second"] is None


def test_worker_records_exact_work_identity_configs_and_fallback_histogram(
    tmp_path: Path,
    v2_runner: None,
):
    config = worker_config(tmp_path, k=3, scheduler_keys=(2,))
    factory = FakeLLMFactory(fallback_steps=1)

    result = run_fake_worker(config, factory)

    assert result.status == "complete"
    assert result.profile_id == profile_id(
        ProfileIdentity.from_mapping(result.profile_identity)
    )
    assert result.profile_identity["workload_hash"] == result.workload_hash
    assert result.target_revision == "target-revision"
    assert result.speculative_revision == DRAFT_COMMIT
    assert result.total_prompt_tokens == 8
    assert result.total_output_tokens == 16
    assert result.output_tokens_per_second is not None
    assert len(result.measurements) == 2
    assert result.measurements[0].scheduler_key_histogram == [
        {"scheduler_key": 2, "k": 3, "steps": 1}
    ]
    assert result.measurements[0].cudagraph_fallback_histogram == [
        {
            "num_unpadded_tokens": 8,
            "num_padded_tokens": 8,
            "runtime_mode": "PIECEWISE",
            "steps": 1,
        }
    ]
    assert (
        result.resolved_engine_config["compilation_config"]["cudagraph_mode"]
        == "FULL_AND_PIECEWISE"
    )
    assert result.resolved_engine_config["use_v2_model_runner"] is True
    assert result.resolved_engine_config["drafter_identity"] == {
        "model": "draft-model",
        "revision": DRAFT_COMMIT,
        "commit": DRAFT_COMMIT,
    }
    assert result.runtime_identity["python"]
    assert result.runtime_identity["vllm_package_commit"]
    assert result.runtime_identity["vllm_source_commit"]
    assert json.loads(config.output_path.read_text())["status"] == "complete"


def test_atomic_write_failure_leaves_no_result_or_temporary_file(
    tmp_path: Path,
    v2_runner: None,
    monkeypatch: pytest.MonkeyPatch,
):
    config = worker_config(tmp_path)
    config.output_path.write_text('{"status": "stale"}\n')

    def fail_replace(source: Path, target: Path) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(dynamic_sd_worker, "_replace_file", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        run_fake_worker(config, FakeLLMFactory())

    assert not config.output_path.exists()
    assert not config.output_path.with_suffix(".json.tmp").exists()
