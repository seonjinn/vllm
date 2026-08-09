# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import sys
import types
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import pytest
import torch

from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.experts import trtllm_fp8_moe
from vllm.model_executor.layers.fused_moe.experts.trtllm_moe_trace import (
    MoeTraceMetadata,
    _reset_trace_sampling_for_testing,
    allocate_routing_replay,
    record_routing_signature,
    should_sample_routing_signature,
    trace_enabled,
)

BASE = MoeTraceMetadata(
    schema_version=1,
    model_revision="qwen3-30ba3b-test",
    layer_family="routed_experts",
    global_num_experts=128,
    local_num_experts=128,
    top_k=8,
    hidden_size=2048,
    intermediate_size=768,
    tp_size=1,
    ep_size=1,
    dp_size=16,
    cuda_graph_state="trace-eager",
    weight_layout="MajorK",
    quantization="MXFP8",
    runtime_fingerprint="runtime-sha256",
)


@pytest.fixture(autouse=True)
def reset_trace_sampling() -> None:
    _reset_trace_sampling_for_testing()


@dataclass(frozen=True)
class CallObservation:
    flashinfer_kwargs: dict[str, object]
    input_topk: torch.Tensor | None
    recorded_topk: torch.Tensor | None
    num_tokens: int
    top_k: int
    created_cuda_events: int


def _moe_config() -> FusedMoEConfig:
    parallel_config = FusedMoEParallelConfig(
        tp_size=1,
        pcp_size=1,
        dp_size=1,
        ep_size=1,
        tp_rank=0,
        pcp_rank=0,
        dp_rank=0,
        ep_rank=0,
        sp_size=1,
        use_ep=False,
        all2all_backend="naive",
        enable_eplb=False,
    )
    return FusedMoEConfig(
        num_experts=4,
        experts_per_token=2,
        hidden_dim=32,
        intermediate_size=32,
        num_local_experts=4,
        num_logical_experts=4,
        activation=MoEActivation.SWIGLUOAI_UNINTERLEAVE,
        device=torch.device("cpu"),
        routing_method=RoutingMethodType.Renormalize,
        moe_parallel_config=parallel_config,
        in_dtype=torch.bfloat16,
    )


def _quant_config() -> FusedMoEQuantConfig:
    return FusedMoEQuantConfig.make(
        quant_dtype="mxfp8",
        block_shape=[1, 32],
    )


def _install_flashinfer_fake(
    monkeypatch: pytest.MonkeyPatch,
    captured_kwargs: dict[str, object],
) -> None:
    def fake_moe(**kwargs: object) -> torch.Tensor:
        captured_kwargs.update(kwargs)
        routing_replay_out = kwargs.get("routing_replay_out")
        if isinstance(routing_replay_out, torch.Tensor):
            routing_replay_out.copy_(torch.tensor([[0, 1], [1, 2]], dtype=torch.int16))
        return torch.zeros((2, 32), dtype=torch.bfloat16)

    fused_moe = types.ModuleType("flashinfer.fused_moe")
    fused_moe.Fp8QuantizationType = types.SimpleNamespace(MxFp8=0, DeepSeekFp8=1)
    fused_moe.WeightLayout = types.SimpleNamespace(MajorK=0, BlockMajorK=1)
    fused_moe.trtllm_fp8_block_scale_moe = fake_moe
    fused_moe.trtllm_fp8_block_scale_routed_moe = fake_moe

    core = types.ModuleType("flashinfer.fused_moe.core")
    core.ActivationType = types.SimpleNamespace(
        Silu=types.SimpleNamespace(value=0),
        Gelu=types.SimpleNamespace(value=1),
        Swiglu=types.SimpleNamespace(value=2),
        Geglu=types.SimpleNamespace(value=3),
        Relu2=types.SimpleNamespace(value=4),
    )

    flashinfer = types.ModuleType("flashinfer")
    flashinfer.fused_moe = fused_moe
    monkeypatch.setitem(sys.modules, "flashinfer", flashinfer)
    monkeypatch.setitem(sys.modules, "flashinfer.fused_moe", fused_moe)
    monkeypatch.setitem(sys.modules, "flashinfer.fused_moe.core", core)


def _set_trace_directory(
    monkeypatch: pytest.MonkeyPatch,
    trace_dir: Path | None,
) -> None:
    if trace_dir is None:
        monkeypatch.delenv("VLLM_MXFP8_MOE_TRACE_DIR", raising=False)
    else:
        monkeypatch.setenv("VLLM_MXFP8_MOE_TRACE_DIR", str(trace_dir))


@pytest.fixture
def monolithic_call(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[[Path | None], CallObservation]:
    def call(trace_dir: Path | None) -> CallObservation:
        _set_trace_directory(monkeypatch, trace_dir)
        captured_kwargs: dict[str, object] = {}
        recorded_topk: torch.Tensor | None = None
        created_cuda_events = 0

        class CountingEvent:
            def __init__(self, **kwargs: object) -> None:
                nonlocal created_cuda_events
                created_cuda_events += 1

            def record(self) -> None:
                pass

            def synchronize(self) -> None:
                pass

            def elapsed_time(self, end_event: object) -> float:
                return 1.0

        def record_signature(
            topk_ids: torch.Tensor,
            metadata: MoeTraceMetadata,
            sampled_gpu_time_us: float,
        ) -> None:
            nonlocal recorded_topk
            recorded_topk = topk_ids

        _install_flashinfer_fake(monkeypatch, captured_kwargs)
        monkeypatch.setattr(torch.cuda, "Event", CountingEvent)
        monkeypatch.setattr(
            trtllm_fp8_moe,
            "record_routing_signature",
            record_signature,
            raising=False,
        )
        experts = trtllm_fp8_moe.TrtLlmFp8ExpertsMonolithic(
            _moe_config(), _quant_config()
        )
        hidden_states = torch.zeros((2, 32), dtype=torch.float8_e4m3fn)
        experts.apply(
            hidden_states,
            torch.zeros((4, 2, 32, 32), dtype=torch.float8_e4m3fn),
            torch.zeros((4, 32, 1, 32), dtype=torch.float8_e4m3fn),
            torch.zeros((2, 4), dtype=torch.bfloat16),
            MoEActivation.SWIGLUOAI_UNINTERLEAVE,
            global_num_experts=4,
            expert_map=None,
            a1q_scale=torch.ones((2, 1), dtype=torch.float8_e8m0fnu),
            apply_router_weight_on_input=False,
        )
        return CallObservation(
            flashinfer_kwargs=captured_kwargs,
            input_topk=None,
            recorded_topk=recorded_topk,
            num_tokens=2,
            top_k=2,
            created_cuda_events=created_cuda_events,
        )

    return call


@pytest.fixture
def modular_call(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[[Path | None], CallObservation]:
    def call(trace_dir: Path | None) -> CallObservation:
        _set_trace_directory(monkeypatch, trace_dir)
        captured_kwargs: dict[str, object] = {}
        recorded_topk: torch.Tensor | None = None
        created_cuda_events = 0
        input_topk = torch.tensor([[0, 1], [1, 2]], dtype=torch.int32)

        class CountingEvent:
            def __init__(self, **kwargs: object) -> None:
                nonlocal created_cuda_events
                created_cuda_events += 1

            def record(self) -> None:
                pass

            def synchronize(self) -> None:
                pass

            def elapsed_time(self, end_event: object) -> float:
                return 1.0

        def record_signature(
            topk_ids: torch.Tensor,
            metadata: MoeTraceMetadata,
            sampled_gpu_time_us: float,
        ) -> None:
            nonlocal recorded_topk
            recorded_topk = topk_ids

        _install_flashinfer_fake(monkeypatch, captured_kwargs)
        monkeypatch.setattr(torch.cuda, "Event", CountingEvent)
        monkeypatch.setattr(
            trtllm_fp8_moe,
            "record_routing_signature",
            record_signature,
            raising=False,
        )
        monkeypatch.setattr(
            trtllm_fp8_moe,
            "trtllm_moe_pack_topk_ids_weights",
            lambda topk_ids, topk_weights: topk_ids,
        )
        experts = trtllm_fp8_moe.TrtLlmFp8ExpertsModular(
            _moe_config(), _quant_config()
        )
        experts.apply(
            torch.zeros((2, 32), dtype=torch.bfloat16),
            torch.zeros((2, 32), dtype=torch.float8_e4m3fn),
            torch.zeros((4, 2, 32, 32), dtype=torch.float8_e4m3fn),
            torch.zeros((4, 32, 1, 32), dtype=torch.float8_e4m3fn),
            torch.ones((2, 2), dtype=torch.float32),
            input_topk,
            MoEActivation.SWIGLUOAI_UNINTERLEAVE,
            global_num_experts=4,
            expert_map=None,
            a1q_scale=torch.ones((2, 1), dtype=torch.float8_e8m0fnu),
            a2_scale=None,
            workspace13=torch.empty(0),
            workspace2=torch.empty(0),
            expert_tokens_meta=None,
            apply_router_weight_on_input=False,
        )
        return CallObservation(
            flashinfer_kwargs=captured_kwargs,
            input_topk=input_topk,
            recorded_topk=recorded_topk,
            num_tokens=2,
            top_k=2,
            created_cuda_events=created_cuda_events,
        )

    return call


def test_monolithic_trace_disabled_does_not_pass_replay_buffer(
    monolithic_call: Callable[[Path | None], CallObservation],
) -> None:
    result = monolithic_call(trace_dir=None)

    assert result.flashinfer_kwargs.get("routing_replay_out") is None
    assert result.created_cuda_events == 0


def test_monolithic_trace_enabled_passes_int16_replay_buffer(
    monolithic_call: Callable[[Path | None], CallObservation], tmp_path: Path
) -> None:
    result = monolithic_call(trace_dir=tmp_path)

    replay = result.flashinfer_kwargs["routing_replay_out"]
    assert isinstance(replay, torch.Tensor)
    assert replay.dtype == torch.int16
    assert replay.shape == (result.num_tokens, result.top_k)
    assert result.created_cuda_events == 2


def test_monolithic_unsampled_call_avoids_replay_and_events(
    monolithic_call: Callable[[Path | None], CallObservation],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_MXFP8_MOE_TRACE_INTERVAL", "2")

    assert monolithic_call(trace_dir=tmp_path).created_cuda_events == 2
    result = monolithic_call(trace_dir=tmp_path)

    assert result.flashinfer_kwargs.get("routing_replay_out") is None
    assert result.created_cuda_events == 0


def test_modular_trace_uses_existing_topk_without_replay_allocation(
    modular_call: Callable[[Path | None], CallObservation], tmp_path: Path
) -> None:
    result = modular_call(trace_dir=tmp_path)

    assert result.recorded_topk is not None
    assert result.input_topk is not None
    assert result.recorded_topk.data_ptr() == result.input_topk.data_ptr()
    assert "routing_replay_out" not in result.flashinfer_kwargs


def test_modular_unsampled_call_avoids_events(
    modular_call: Callable[[Path | None], CallObservation],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_MXFP8_MOE_TRACE_INTERVAL", "2")

    assert modular_call(trace_dir=tmp_path).created_cuda_events == 2
    result = modular_call(trace_dir=tmp_path)

    assert result.recorded_topk is None
    assert result.created_cuda_events == 0


def test_trace_is_disabled_without_directory(monkeypatch) -> None:
    monkeypatch.delenv("VLLM_MXFP8_MOE_TRACE_DIR", raising=False)

    assert not trace_enabled()
    assert allocate_routing_replay(4, 2, torch.device("cpu")) is None


def test_trace_sampling_honors_interval_and_limit(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("VLLM_MXFP8_MOE_TRACE_DIR", str(tmp_path))
    monkeypatch.setenv("VLLM_MXFP8_MOE_TRACE_INTERVAL", "3")
    monkeypatch.setenv("VLLM_MXFP8_MOE_TRACE_MAX_SAMPLES", "2")

    sampled = [should_sample_routing_signature() for _ in range(10)]

    assert sampled == [
        True,
        False,
        False,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
    ]


def test_trace_sampling_rejects_nonpositive_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VLLM_MXFP8_MOE_TRACE_DIR", str(tmp_path))
    monkeypatch.setenv("VLLM_MXFP8_MOE_TRACE_INTERVAL", "0")

    with pytest.raises(ValueError, match="VLLM_MXFP8_MOE_TRACE_INTERVAL"):
        should_sample_routing_signature()


def test_allocate_routing_replay_uses_int16_sentinel(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("VLLM_MXFP8_MOE_TRACE_DIR", str(tmp_path))

    replay = allocate_routing_replay(4, 2, torch.device("cpu"))

    assert replay is not None
    assert replay.dtype is torch.int16
    assert replay.shape == (4, 2)
    assert torch.equal(replay, torch.full((4, 2), -1, dtype=torch.int16))


def test_record_writes_histogram_without_payload(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("VLLM_MXFP8_MOE_TRACE_DIR", str(tmp_path))
    topk_ids = torch.tensor([[0, 1], [1, 2]], dtype=torch.int16)

    record_routing_signature(
        topk_ids,
        replace(BASE, global_num_experts=4, top_k=2),
        sampled_gpu_time_us=17.5,
    )

    row = json.loads(next(tmp_path.glob("*.jsonl")).read_text().strip())
    assert row["expert_counts"] == [1, 2, 1, 0]
    assert row["num_tokens"] == 2
    assert row["sampled_gpu_time_us"] == 17.5
    assert "topk_ids" not in row
    assert "hidden_states" not in row


def test_record_uses_replica_rank_from_environment(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("VLLM_MXFP8_MOE_TRACE_DIR", str(tmp_path))
    monkeypatch.setenv("RANK", "14")

    record_routing_signature(
        torch.tensor([[0, 1]], dtype=torch.int16),
        BASE,
        sampled_gpu_time_us=1.0,
    )

    assert next(tmp_path.glob("*.jsonl")).name.startswith("moe-routing-rank14-")


def test_record_accepts_unsigned_integer_tensor(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("VLLM_MXFP8_MOE_TRACE_DIR", str(tmp_path))
    topk_ids = torch.tensor([[0, 1]], dtype=torch.uint16)

    record_routing_signature(topk_ids, BASE, sampled_gpu_time_us=1.0)

    row = json.loads(next(tmp_path.glob("*.jsonl")).read_text().strip())
    assert row["expert_counts"][:2] == [1, 1]


def test_record_does_not_call_item_for_bounds_check(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("VLLM_MXFP8_MOE_TRACE_DIR", str(tmp_path))

    def fail_item(*args, **kwargs):
        raise AssertionError("record_routing_signature must not call Tensor.item()")

    monkeypatch.setattr(torch.Tensor, "item", fail_item)

    record_routing_signature(
        torch.tensor([[0, 1]], dtype=torch.int16),
        BASE,
        sampled_gpu_time_us=1.0,
    )


@pytest.mark.parametrize(
    "topk_ids",
    [
        torch.tensor([0, 1], dtype=torch.int16),
        torch.tensor([[0.0, 1.0]], dtype=torch.float32),
        torch.tensor([[0, 1], [2, 3]], dtype=torch.int16).t(),
    ],
)
def test_record_rejects_invalid_routing_tensor(topk_ids, tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("VLLM_MXFP8_MOE_TRACE_DIR", str(tmp_path))

    with pytest.raises(ValueError):
        record_routing_signature(topk_ids, BASE, sampled_gpu_time_us=1.0)


@pytest.mark.parametrize("sampled_gpu_time_us", [0.0, -1.0, float("inf"), float("nan")])
def test_record_rejects_invalid_gpu_time(
    sampled_gpu_time_us, tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("VLLM_MXFP8_MOE_TRACE_DIR", str(tmp_path))
    topk_ids = torch.tensor([[0]], dtype=torch.int16)

    with pytest.raises(ValueError):
        record_routing_signature(topk_ids, BASE, sampled_gpu_time_us)


@pytest.mark.parametrize("expert_id", [-2, -1, 128, 129])
def test_record_rejects_out_of_range_expert_id(
    expert_id, tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("VLLM_MXFP8_MOE_TRACE_DIR", str(tmp_path))
    topk_ids = torch.tensor([[expert_id]], dtype=torch.int16)

    with pytest.raises(ValueError):
        record_routing_signature(topk_ids, BASE, sampled_gpu_time_us=1.0)
