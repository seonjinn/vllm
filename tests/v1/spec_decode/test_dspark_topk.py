# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.model_executor.models import qwen3_dflash
from vllm.model_executor.models.qwen3_dspark import (
    DSparkFirstDraftAdapter,
    DSparkMarkovHead,
    Qwen3DSparkForCausalLM,
    Qwen3DSparkModel,
)
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import (
    DSparkSpeculator,
    _prepare_dspark_draft_hidden,
)


class _WeightOnlyNorm(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(width))


class _AdaptiveAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.target_k_proj = nn.Linear(3, 2, bias=True)
        self.target_v_proj = nn.Linear(3, 2, bias=True)


class _AdaptiveLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _AdaptiveAttention()


def _adaptive_dspark_causal_lm() -> Qwen3DSparkForCausalLM:
    model = Qwen3DSparkModel.__new__(Qwen3DSparkModel)
    nn.Module.__init__(model)
    model.layers = nn.ModuleList([_AdaptiveLayer()])
    model.target_fusion_type = "adaptive_layerwise"
    model.separate_target_kv = True
    model.config = type("Config", (), {"attention_bias": True})()
    model.target_fusion_logits = nn.Parameter(torch.empty(1, 2))
    model.target_hidden_norms = nn.ModuleList([_WeightOnlyNorm(3)])
    model.first_draft_adapter = DSparkFirstDraftAdapter.__new__(DSparkFirstDraftAdapter)
    nn.Module.__init__(model.first_draft_adapter)
    model.first_draft_adapter.down_proj = nn.Linear(3, 1, bias=False)
    model.first_draft_adapter.up_proj = nn.Linear(1, 3, bias=False)
    model.confidence_head = None
    model._build_fused_kv_buffers = lambda: None

    causal_lm = Qwen3DSparkForCausalLM.__new__(Qwen3DSparkForCausalLM)
    nn.Module.__init__(causal_lm)
    causal_lm.model = model
    return causal_lm


def _adaptive_weights() -> dict[str, torch.Tensor]:
    return {
        "target_fusion_logits": torch.tensor([[0.25, 0.75]]),
        "target_hidden_norms.0.weight": torch.tensor([1.0, 2.0, 3.0]),
        "layers.0.self_attn.target_k_proj.weight": torch.arange(6.0).view(2, 3),
        "layers.0.self_attn.target_k_proj.bias": torch.tensor([6.0, 7.0]),
        "layers.0.self_attn.target_v_proj.weight": torch.arange(8.0, 14.0).view(2, 3),
        "layers.0.self_attn.target_v_proj.bias": torch.tensor([14.0, 15.0]),
        "first_draft_adapter.down_proj.weight": torch.tensor([[16.0, 17.0, 18.0]]),
        "first_draft_adapter.up_proj.weight": torch.tensor([[19.0], [20.0], [21.0]]),
    }


def _markov_head(weight: torch.Tensor) -> DSparkMarkovHead:
    head = DSparkMarkovHead.__new__(DSparkMarkovHead)
    nn.Module.__init__(head)
    head.markov_w2 = nn.Linear(
        weight.shape[1], weight.shape[0], bias=False, dtype=weight.dtype
    )
    head.markov_w2.weight.data.copy_(weight)
    return head


def test_gathered_markov_bias_overwrites_dense_logits():
    weight = torch.arange(21, dtype=torch.float32).view(7, 3) / 10
    markov_embed = torch.tensor([[0.5, -1.0, 0.25], [1.0, 0.5, -0.5]])
    logits = torch.tensor(
        [
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
            [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
        ]
    )
    values, index = logits.topk(3, dim=-1)
    values = torch.stack((values, torch.zeros_like(values)), dim=1)[:, 0]
    expected = values + torch.bmm(weight[index], markov_embed.unsqueeze(-1)).squeeze(-1)
    logits.fill_(float("-inf"))

    result = _markov_head(weight).apply_bias_gathered(
        markov_embed, logits, values, index
    )

    assert result is logits
    torch.testing.assert_close(result.gather(1, index), expected)
    selected = torch.zeros_like(result, dtype=torch.bool).scatter_(1, index, True)
    assert torch.isneginf(result.masked_select(~selected)).all()


def test_gathered_markov_bias_matches_dense_at_full_vocab():
    weight = torch.arange(15, dtype=torch.float32).view(5, 3) / 10
    markov_embed = torch.tensor([[0.5, -1.0, 0.25]])
    logits = torch.tensor([[0.1, 0.4, -0.2, 0.3, 0.0]])
    original = logits.clone()
    values, index = logits.topk(logits.shape[-1], dim=-1)
    scale = 0.5
    logits.fill_(float("-inf"))

    result = _markov_head(weight).apply_bias_gathered(
        markov_embed, logits, values, index, scale
    )

    expected = original + markov_embed @ weight.T * scale
    torch.testing.assert_close(result, expected)


def test_first_draft_adapter_uses_explicit_semantic_step_ids():
    adapter = DSparkFirstDraftAdapter.__new__(DSparkFirstDraftAdapter)
    nn.Module.__init__(adapter)
    adapter.down_proj = nn.Linear(2, 1, bias=False)
    adapter.up_proj = nn.Linear(1, 2, bias=False)
    adapter.down_proj.weight.data.copy_(torch.tensor([[1.0, 0.0]]))
    adapter.up_proj.weight.data.copy_(torch.tensor([[2.0], [3.0]]))

    sampled = torch.arange(12, dtype=torch.float32).view(6, 2)
    # Deliberately not request-major. The serving path passes DFlash's
    # sample_col, whose values are the semantic DSpark draft steps.
    draft_steps = torch.tensor([2, 0, 1, 2, 1, 0])

    adapted = adapter.apply_to_steps(sampled, draft_steps)

    delta = adapted - sampled
    assert torch.count_nonzero(delta[draft_steps != 0]) == 0
    activated = torch.nn.functional.silu(sampled[draft_steps == 0, 0])
    torch.testing.assert_close(delta[draft_steps == 0, 0], activated * 2)
    torch.testing.assert_close(delta[draft_steps == 0, 1], activated * 3)


def test_first_draft_adapter_checkpoint_names_are_stable():
    adapter = DSparkFirstDraftAdapter.__new__(DSparkFirstDraftAdapter)
    nn.Module.__init__(adapter)
    adapter.down_proj = nn.Linear(4, 2, bias=False)
    adapter.up_proj = nn.Linear(2, 4, bias=False)

    assert set(adapter.state_dict()) == {
        "down_proj.weight",
        "up_proj.weight",
    }


def test_prepare_draft_hidden_preserves_one_argument_model_contract():
    class LegacyModel:
        def compute_draft_logits(self, hidden_states):
            return hidden_states + 1

    model = LegacyModel()
    hidden = torch.arange(6.0).view(3, 2)

    prepared = _prepare_dspark_draft_hidden(model, hidden, torch.tensor([0, 1, 2]))

    assert prepared is hidden
    torch.testing.assert_close(model.compute_draft_logits(prepared), hidden + 1)


def test_qwen_prepare_draft_hidden_is_capability_gated_for_gemma():
    causal_lm = Qwen3DSparkForCausalLM.__new__(Qwen3DSparkForCausalLM)
    nn.Module.__init__(causal_lm)
    causal_lm.model = nn.Module()
    hidden = torch.arange(6.0).view(3, 2)

    prepared = causal_lm.prepare_draft_hidden(hidden, torch.tensor([0, 1, 2]))

    assert prepared is hidden


@pytest.mark.parametrize("draft_topk", [None, 1])
def test_adapter_prepares_logits_and_confidence_hidden_once(draft_topk):
    class Model:
        def prepare_draft_hidden(self, hidden_states, draft_steps):
            self.prepared = hidden_states + 10
            return self.prepared

        def compute_draft_logits(self, hidden_states):
            self.logit_hidden = hidden_states
            return torch.zeros(hidden_states.shape[0], 3)

        def markov_embed(self, token_ids):
            return torch.zeros(token_ids.shape[0], 1)

        def markov_bias(self, markov_embed):
            return torch.zeros(markov_embed.shape[0], 3)

        def apply_markov_bias_gathered(self, markov_embed, logits, values, index):
            return logits.scatter(1, index, values)

        def compute_confidence(self, head_hidden, markov_embed):
            self.confidence_hidden = head_hidden
            return torch.ones(head_hidden.shape[0])

    speculator = DSparkSpeculator.__new__(DSparkSpeculator)
    speculator.model = Model()
    speculator._draft_topk = draft_topk
    speculator.num_speculative_steps = 2
    speculator.sample_indices = torch.tensor([0, 1])
    speculator.sample_col = torch.tensor([0, 1])
    speculator.sample_idx_mapping = torch.tensor([0, 0])
    speculator.sample_pos = torch.tensor([1, 2])
    speculator.input_buffers = SimpleNamespace(input_ids=torch.tensor([5]))
    speculator._anchor_idx = torch.tensor([0])
    speculator.enable_adaptive_verification = True
    speculator.draft_token_confidence_probs = torch.zeros(1, 2)
    speculator.draft_tokens = torch.zeros(1, 2, dtype=torch.long)
    speculator._sample_logits = lambda logits, idx_map, sample_pos, step: torch.zeros(
        logits.shape[0], dtype=torch.long
    )
    hidden = torch.arange(4.0).view(2, 2)

    speculator._sample_sequential(1, hidden)

    torch.testing.assert_close(speculator.model.logit_hidden, hidden + 10)
    torch.testing.assert_close(speculator.model.confidence_hidden, hidden + 10)


def test_first_draft_adapter_weights_load_without_mlp_name_mapping(monkeypatch):
    model = Qwen3DSparkModel.__new__(Qwen3DSparkModel)
    nn.Module.__init__(model)
    model.first_draft_adapter = DSparkFirstDraftAdapter.__new__(DSparkFirstDraftAdapter)
    nn.Module.__init__(model.first_draft_adapter)
    model.first_draft_adapter.down_proj = nn.Linear(4, 2, bias=False)
    model.first_draft_adapter.up_proj = nn.Linear(2, 4, bias=False)
    monkeypatch.setattr(qwen3_dflash, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(qwen3_dflash, "get_tensor_model_parallel_rank", lambda: 0)
    expected = {
        "first_draft_adapter.down_proj.weight": torch.arange(8.0).view(2, 4),
        "first_draft_adapter.up_proj.weight": torch.arange(8.0, 16.0).view(4, 2),
    }

    loaded = model.load_weights(expected.items())

    assert loaded == set(expected)
    for name, value in expected.items():
        torch.testing.assert_close(model.get_parameter(name), value)


def test_dspark_causal_lm_loader_consumes_adaptive_checkpoint_keys(monkeypatch):
    causal_lm = _adaptive_dspark_causal_lm()
    monkeypatch.setattr(qwen3_dflash, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(qwen3_dflash, "get_tensor_model_parallel_rank", lambda: 0)
    expected = _adaptive_weights()

    loaded = causal_lm.load_weights(expected.items())

    assert loaded is None
    for name, value in expected.items():
        torch.testing.assert_close(causal_lm.model.get_parameter(name), value)


@pytest.mark.parametrize(
    "missing_group",
    [
        "target_fusion_logits",
        "target_hidden_norms.0",
        "target_k_proj.weight",
        "target_k_proj.bias",
        "target_v_proj.weight",
        "target_v_proj.bias",
        "first_draft_adapter.down_proj",
        "first_draft_adapter.up_proj",
    ],
)
def test_dspark_adaptive_loader_fails_when_required_group_is_missing(
    monkeypatch, missing_group
):
    causal_lm = _adaptive_dspark_causal_lm()
    monkeypatch.setattr(qwen3_dflash, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(qwen3_dflash, "get_tensor_model_parallel_rank", lambda: 0)
    weights = {
        name: value
        for name, value in _adaptive_weights().items()
        if missing_group not in name
    }

    with pytest.raises(ValueError, match="missing required adaptive weights"):
        causal_lm.load_weights(weights.items())
