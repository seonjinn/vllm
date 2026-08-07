# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from dataclasses import replace

import pytest
import torch

from vllm.model_executor.layers.fused_moe.experts.trtllm_moe_trace import (
    MoeTraceMetadata,
    allocate_routing_replay,
    record_routing_signature,
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


def test_trace_is_disabled_without_directory(monkeypatch) -> None:
    monkeypatch.delenv("VLLM_MXFP8_MOE_TRACE_DIR", raising=False)

    assert not trace_enabled()
    assert allocate_routing_replay(4, 2, torch.device("cpu")) is None


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
