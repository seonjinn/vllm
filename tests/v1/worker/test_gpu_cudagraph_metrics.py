# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any

import pytest
import torch

import vllm.v1.worker.gpu.model_runner as model_runner_module
import vllm.v1.worker.gpu.spec_decode.dflash.speculator as dflash_module
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.cudagraph_metrics import CUDAGraphDispatchMetrics
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator
from vllm.v1.worker.gpu_worker import Worker


def _empty_counts() -> dict[str, dict[str, int]]:
    return {
        "NONE": {"dispatches": 0, "requests": 0, "tokens": 0},
        "PIECEWISE": {"dispatches": 0, "requests": 0, "tokens": 0},
        "FULL": {"dispatches": 0, "requests": 0, "tokens": 0},
    }


def test_collector_snapshot_is_zero_populated_json_and_detached() -> None:
    metrics = CUDAGraphDispatchMetrics()

    first = metrics.snapshot()
    assert first == _empty_counts()
    json.dumps(first)

    metrics.observe(CUDAGraphMode.FULL, num_requests=2, num_tokens=7)

    assert first == _empty_counts()
    recorded = metrics.snapshot()
    assert recorded == {
        **_empty_counts(),
        "FULL": {"dispatches": 1, "requests": 2, "tokens": 7},
    }

    metrics.reset()
    assert recorded["FULL"] == {"dispatches": 1, "requests": 2, "tokens": 7}
    recorded["FULL"]["tokens"] = 99  # type: ignore[index]
    assert metrics.snapshot() == _empty_counts()


def test_collector_reset_clears_every_runtime_mode() -> None:
    metrics = CUDAGraphDispatchMetrics()
    for mode in CUDAGraphMode.valid_runtime_modes():
        metrics.observe(mode, num_requests=1, num_tokens=2)

    metrics.reset()

    assert metrics.snapshot() == _empty_counts()


@pytest.mark.parametrize(
    ("mode", "num_requests", "num_tokens"),
    [
        (CUDAGraphMode.FULL_DECODE_ONLY, 1, 1),
        (CUDAGraphMode.FULL_AND_PIECEWISE, 1, 1),
        (CUDAGraphMode.FULL, -1, 1),
        (CUDAGraphMode.FULL, 1, -1),
        (CUDAGraphMode.FULL, True, 1),
        (CUDAGraphMode.FULL, 1, False),
    ],
)
def test_collector_rejects_non_runtime_modes_and_invalid_counts(
    mode: CUDAGraphMode,
    num_requests: int,
    num_tokens: int,
) -> None:
    metrics = CUDAGraphDispatchMetrics()

    with pytest.raises(ValueError):
        metrics.observe(
            mode,
            num_requests=num_requests,
            num_tokens=num_tokens,
        )

    assert metrics.snapshot() == _empty_counts()


def test_collector_observe_is_thread_safe() -> None:
    metrics = CUDAGraphDispatchMetrics()

    def observe_many() -> None:
        for _ in range(500):
            metrics.observe(CUDAGraphMode.PIECEWISE, num_requests=2, num_tokens=3)

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(lambda _: observe_many(), range(8)))

    assert metrics.snapshot()["PIECEWISE"] == {
        "dispatches": 4000,
        "requests": 8000,
        "tokens": 12000,
    }


def _runner_for_execute_model() -> GPUModelRunner:
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.cudagraph_dispatch_metrics = CUDAGraphDispatchMetrics()
    runner.update_pp_decode_requests = lambda: None
    runner.finish_requests = lambda _scheduler_output: None
    runner.free_states = lambda _scheduler_output: None
    runner.add_requests = lambda _scheduler_output: None
    runner.update_requests = lambda _scheduler_output: None
    runner.block_tables = SimpleNamespace(apply_staged_writes=lambda: None)
    runner.kv_connector = SimpleNamespace(no_forward=lambda _scheduler_output: None)
    runner.lora_config = None
    runner.is_encoder_decoder = False
    runner.cudagraph_manager = SimpleNamespace(
        cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY
    )
    runner.dp_size = 1
    runner.dp_rank = 0
    return runner


def _scheduler_output(num_scheduled_tokens: dict[str, int]) -> SimpleNamespace:
    return SimpleNamespace(
        num_scheduled_tokens=num_scheduled_tokens,
        total_num_scheduled_tokens=sum(num_scheduled_tokens.values()),
        scheduled_encoder_inputs={},
    )


def test_target_records_final_dispatch_mode_with_unpadded_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner_for_execute_model()

    def final_dispatch(*args: Any, **kwargs: Any):
        assert runner.cudagraph_dispatch_metrics.snapshot() == _empty_counts()
        return (
            BatchExecutionDescriptor(
                cg_mode=CUDAGraphMode.FULL,
                num_tokens=0,
                num_reqs=8,
            ),
            None,
        )

    monkeypatch.setattr(
        model_runner_module,
        "dispatch_cg_and_sync_dp",
        final_dispatch,
    )

    runner.execute_model(_scheduler_output({"req-0": 2, "req-1": 3}))

    assert runner.cudagraph_dispatch_metrics.snapshot()["FULL"] == {
        "dispatches": 1,
        "requests": 2,
        "tokens": 5,
    }
    assert runner.cudagraph_dispatch_metrics.snapshot()["NONE"] == {
        "dispatches": 0,
        "requests": 0,
        "tokens": 0,
    }


@pytest.mark.parametrize(
    ("dummy_run", "is_profile"),
    [(True, False), (False, True), (True, True)],
)
def test_target_excludes_dummy_and_profile_dispatches(
    monkeypatch: pytest.MonkeyPatch,
    dummy_run: bool,
    is_profile: bool,
) -> None:
    runner = _runner_for_execute_model()
    monkeypatch.setattr(
        model_runner_module,
        "dispatch_cg_and_sync_dp",
        lambda *args, **kwargs: (
            BatchExecutionDescriptor(CUDAGraphMode.NONE, 0, 1),
            None,
        ),
    )

    runner.execute_model(
        _scheduler_output({"req-0": 1}),
        dummy_run=dummy_run,
        is_profile=is_profile,
    )

    assert runner.cudagraph_dispatch_metrics.snapshot() == _empty_counts()


def test_target_excludes_zero_real_token_calls() -> None:
    runner = _runner_for_execute_model()

    runner.execute_model(_scheduler_output({}))

    assert runner.cudagraph_dispatch_metrics.snapshot() == _empty_counts()


def _draft_speculator(speculator_type: type[DFlashSpeculator]) -> DFlashSpeculator:
    speculator: DFlashSpeculator = object.__new__(speculator_type)
    if speculator_type is DSparkSpeculator:
        speculator.cudagraph_dispatch_metrics = CUDAGraphDispatchMetrics()
    speculator.num_query_per_req = 3
    speculator.num_speculative_steps = 3
    speculator.max_model_len = 32
    speculator.hidden_states = torch.zeros((8, 4))
    speculator.context_positions = torch.zeros(8, dtype=torch.int64)
    speculator._copy_request_inputs = lambda *args, **kwargs: None
    speculator.draft_kv_cache_group_id = 0
    speculator.draft_kv_cache_group_ids = []
    speculator._layer_group_idx = []
    speculator._group_causal = False
    speculator.model = SimpleNamespace(
        precompute_and_store_context_kv=lambda *args, **kwargs: None
    )
    speculator.block_tables = SimpleNamespace(slot_mappings=torch.zeros((1, 16)))
    speculator.kv_cache_config = object()
    speculator._build_draft_attn_metadata = lambda **kwargs: None
    speculator._prepare_eplb_forward = lambda _num_tokens: None
    speculator._generate_draft = lambda *args, **kwargs: None
    speculator.query_cudagraph_manager = SimpleNamespace(
        cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
        run_fullgraph=lambda _batch_desc: None,
    )
    speculator.dp_size = 1
    speculator.dp_rank = 0
    speculator.draft_tokens = torch.zeros((8, 3), dtype=torch.int64)
    return speculator


def _draft_input_batch() -> SimpleNamespace:
    return SimpleNamespace(
        num_reqs=2,
        num_tokens=2,
        seq_lens_cpu_upper_bound=torch.tensor([1, 1]),
        idx_mapping=torch.tensor([0, 1]),
    )


def _propose(
    speculator: DFlashSpeculator,
    *,
    dummy_run: bool = False,
    is_profile: bool = False,
) -> torch.Tensor:
    return speculator.propose(
        _draft_input_batch(),
        attn_metadata={},
        slot_mappings={},
        last_hidden_states=torch.ones((2, 4)),
        aux_hidden_states=None,
        num_sampled=torch.zeros(2, dtype=torch.int32),
        num_rejected=torch.zeros(2, dtype=torch.int32),
        last_sampled=torch.zeros(8, dtype=torch.int64),
        next_prefill_tokens=torch.zeros(8, dtype=torch.int64),
        temperature=torch.ones(8),
        seeds=torch.zeros(8, dtype=torch.int64),
        dummy_run=dummy_run,
        is_profile=is_profile,
    )


def test_dspark_inherited_propose_records_final_draft_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    speculator = _draft_speculator(DSparkSpeculator)

    def final_dispatch(*args: Any, **kwargs: Any):
        assert speculator.cudagraph_dispatch_metrics.snapshot() == _empty_counts()
        return (
            BatchExecutionDescriptor(CUDAGraphMode.FULL, 12, 4),
            None,
        )

    monkeypatch.setattr(dflash_module, "dispatch_cg_and_sync_dp", final_dispatch)
    monkeypatch.setattr(
        dflash_module,
        "build_slot_mappings_by_layer",
        lambda *args, **kwargs: {},
    )

    _propose(speculator)

    assert speculator.cudagraph_dispatch_metrics.snapshot()["FULL"] == {
        "dispatches": 1,
        "requests": 2,
        "tokens": 6,
    }


def test_ordinary_dflash_inherited_propose_does_not_report_dspark_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    speculator = _draft_speculator(DFlashSpeculator)
    monkeypatch.setattr(
        dflash_module,
        "dispatch_cg_and_sync_dp",
        lambda *args, **kwargs: (
            BatchExecutionDescriptor(CUDAGraphMode.FULL, 12, 4),
            None,
        ),
    )
    monkeypatch.setattr(
        dflash_module,
        "build_slot_mappings_by_layer",
        lambda *args, **kwargs: {},
    )

    _propose(speculator)

    assert not hasattr(speculator, "cudagraph_dispatch_metrics")


@pytest.mark.parametrize(
    ("dummy_run", "is_profile"),
    [(True, False), (False, True), (True, True)],
)
def test_dspark_excludes_dummy_and_profile_draft_dispatches(
    monkeypatch: pytest.MonkeyPatch,
    dummy_run: bool,
    is_profile: bool,
) -> None:
    speculator = _draft_speculator(DSparkSpeculator)
    monkeypatch.setattr(
        dflash_module,
        "dispatch_cg_and_sync_dp",
        lambda *args, **kwargs: (
            BatchExecutionDescriptor(CUDAGraphMode.NONE, 6, 2),
            None,
        ),
    )
    monkeypatch.setattr(
        dflash_module,
        "build_slot_mappings_by_layer",
        lambda *args, **kwargs: {},
    )

    _propose(speculator, dummy_run=dummy_run, is_profile=is_profile)

    assert speculator.cudagraph_dispatch_metrics.snapshot() == _empty_counts()


def _runner_with_dspark_metrics(
    speculator_type: type[DFlashSpeculator],
) -> GPUModelRunner:
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.cudagraph_dispatch_metrics = CUDAGraphDispatchMetrics()
    runner.cudagraph_manager = SimpleNamespace(
        cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE
    )
    runner.speculator = _draft_speculator(speculator_type)
    return runner


def test_worker_snapshot_and_reset_cover_target_and_dspark_collectors() -> None:
    runner = _runner_with_dspark_metrics(DSparkSpeculator)
    runner.cudagraph_dispatch_metrics.observe(
        CUDAGraphMode.PIECEWISE,
        num_requests=2,
        num_tokens=5,
    )
    runner.speculator.cudagraph_dispatch_metrics.observe(
        CUDAGraphMode.FULL,
        num_requests=2,
        num_tokens=6,
    )
    worker = Worker.__new__(Worker)
    worker.rank = 3
    worker.model_runner = runner

    snapshot = worker.snapshot_cudagraph_dispatch_metrics()

    assert snapshot == {
        "rank": 3,
        "configured_modes": {
            "target": "FULL_AND_PIECEWISE",
            "dspark_draft": "FULL_DECODE_ONLY",
        },
        "target": {
            **_empty_counts(),
            "PIECEWISE": {"dispatches": 1, "requests": 2, "tokens": 5},
        },
        "dspark_draft": {
            **_empty_counts(),
            "FULL": {"dispatches": 1, "requests": 2, "tokens": 6},
        },
    }
    json.dumps(snapshot)

    worker.reset_cudagraph_dispatch_metrics()

    reset_snapshot = worker.snapshot_cudagraph_dispatch_metrics()
    assert reset_snapshot["target"] == _empty_counts()
    assert reset_snapshot["dspark_draft"] == _empty_counts()


def test_worker_reports_zero_dspark_bucket_for_ordinary_dflash() -> None:
    runner = _runner_with_dspark_metrics(DFlashSpeculator)
    worker = Worker.__new__(Worker)
    worker.rank = 0
    worker.model_runner = runner

    snapshot = worker.snapshot_cudagraph_dispatch_metrics()

    assert snapshot["configured_modes"]["dspark_draft"] is None
    assert snapshot["dspark_draft"] == _empty_counts()
