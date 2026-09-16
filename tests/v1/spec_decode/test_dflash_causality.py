# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Config-only DFlash behavior.

``dflash_has_any_non_causal`` decides pre-build whether the draft needs a
non-causal-capable backend, so its branch table (explicit override, SWA-derived
per-layer causality, and the no-``layer_types`` fallback) is worth pinning.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.models import qwen3_dflash
from vllm.model_executor.models.qwen3_dflash import (
    DFlashQwen3ForCausalLM,
    DFlashQwen3Model,
    _dflash_layer_causal,
    _fuse_adaptive_target_hidden_states,
    _get_dflash_context_input_size,
    _get_dflash_fc_input_size,
    _map_dflash_weights,
    _required_adaptive_checkpoint_weights,
    _validate_num_target_taps,
    dflash_has_any_non_causal,
)
from vllm.v1.worker.gpu.spec_decode.eagle.eagle3_utils import (
    get_eagle3_aux_layers_from_config,
)


def _config(num_hidden_layers, layer_types=None, causal_override=None, is_causal=None):
    dflash_config = None if causal_override is None else {"causal": causal_override}
    return SimpleNamespace(
        num_hidden_layers=num_hidden_layers,
        layer_types=layer_types,
        dflash_config=dflash_config,
        is_causal=is_causal,
    )


@pytest.mark.parametrize(
    "config,expected",
    [
        # Override forces causality on every layer, ignoring layer_types.
        (_config(2, layer_types=["full_attention"] * 2, causal_override=True), False),
        # Override forces non-causal on every layer.
        (
            _config(2, layer_types=["sliding_attention"] * 2, causal_override=False),
            True,
        ),
        # DFlash2 stores the explicit attention semantics at the top level.
        (
            _config(
                2,
                layer_types=["sliding_attention"] * 2,
                is_causal=False,
            ),
            True,
        ),
        (
            _config(2, layer_types=["full_attention"] * 2, is_causal=True),
            False,
        ),
        # SWA-derived: full-attention layers are non-causal.
        (_config(2, layer_types=["sliding_attention", "full_attention"]), True),
        # SWA-derived: all-sliding is fully causal.
        (_config(2, layer_types=["sliding_attention", "sliding_attention"]), False),
        # No layer_types -> non-causal fallback.
        (_config(2, layer_types=None), True),
        (_config(2, layer_types=[]), True),
    ],
)
def test_dflash_has_any_non_causal(config, expected):
    assert dflash_has_any_non_causal(config) is expected


def test_dflash_layer_causal_is_per_layer():
    config = _config(2, layer_types=["sliding_attention", "full_attention"])
    assert _dflash_layer_causal(config, 0) is True
    assert _dflash_layer_causal(config, 1) is False


def test_dflash_layer_causal_honors_top_level_override():
    config = _config(
        2,
        layer_types=["sliding_attention", "full_attention"],
        is_causal=False,
    )
    assert _dflash_layer_causal(config, 0) is False
    assert _dflash_layer_causal(config, 1) is False


def _vllm_config(**draft_config):
    config = SimpleNamespace(**draft_config)
    return SimpleNamespace(
        speculative_config=SimpleNamespace(
            draft_model_config=SimpleNamespace(hf_config=config)
        )
    )


def test_dflash_fc_uses_aux_layer_count():
    vllm_config = _vllm_config(
        num_hidden_layers=5,
        hidden_size=4096,
        target_hidden_size=None,
        target_layer_ids=[1, 17, 32],
    )

    assert _get_dflash_fc_input_size(vllm_config) == 3 * 4096


def test_adaptive_fusion_keeps_all_target_taps_as_context_input():
    vllm_config = _vllm_config(
        num_hidden_layers=5,
        hidden_size=4096,
        target_hidden_size=None,
        target_layer_ids=[1, 14, 27, 40, 52, 65, 78, 91],
        target_fusion_type="adaptive_layerwise",
        num_target_taps=8,
    )

    assert _get_dflash_context_input_size(vllm_config) == 8 * 4096


@pytest.mark.parametrize("value", [True, False, 0, -1, 1.5, "8"])
def test_num_target_taps_rejects_non_positive_and_non_integer_values(value):
    with pytest.raises(ValueError, match="positive integer"):
        _validate_num_target_taps(value, fallback=8)


def test_num_target_taps_uses_fallback_only_for_none():
    assert _validate_num_target_taps(None, fallback=8) == 8
    assert _validate_num_target_taps(3, fallback=8) == 3


def test_adaptive_fusion_is_independent_for_each_draft_layer():
    taps = torch.tensor(
        [
            [[1.0, 2.0], [10.0, 20.0], [100.0, 200.0]],
            [[3.0, 4.0], [30.0, 40.0], [300.0, 400.0]],
        ]
    )
    logits = torch.log(torch.tensor([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]]))

    fused = _fuse_adaptive_target_hidden_states(
        taps.flatten(1), logits, [torch.nn.Identity(), torch.nn.Identity()]
    )

    expected = torch.einsum("lt,nth->lnh", logits.softmax(-1), taps)
    torch.testing.assert_close(fused, expected)


def test_target_projection_weights_and_biases_bypass_legacy_qkv_stacking():
    weights = [
        ("layers.0.self_attn.q_proj.weight", torch.empty(1)),
        ("layers.0.self_attn.target_k_proj.weight", torch.empty(1)),
        ("layers.0.self_attn.target_k_proj.bias", torch.empty(1)),
        ("layers.0.self_attn.target_v_proj.weight", torch.empty(1)),
        ("layers.0.self_attn.target_v_proj.bias", torch.empty(1)),
        ("first_draft_adapter.up_proj.weight", torch.empty(1)),
    ]

    mapped = list(_map_dflash_weights(weights, DFlashQwen3Model.hf_to_vllm_mapper))

    assert [name for name, _ in mapped] == [
        "layers.0.self_attn.qkv_proj.weight",
        "layers.0.self_attn.target_k_proj.weight",
        "layers.0.self_attn.target_k_proj.bias",
        "layers.0.self_attn.target_v_proj.weight",
        "layers.0.self_attn.target_v_proj.bias",
        "first_draft_adapter.up_proj.weight",
    ]
    assert mapped[0][1].shard_id == "q"


def test_target_projection_weights_and_biases_load_by_exact_name(monkeypatch):
    class Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self.target_k_proj = nn.Linear(3, 2, bias=True)
            self.target_v_proj = nn.Linear(3, 2, bias=True)

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = Attention()

    model = DFlashQwen3Model.__new__(DFlashQwen3Model)
    nn.Module.__init__(model)
    model.layers = nn.ModuleList([Layer()])
    monkeypatch.setattr(qwen3_dflash, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(qwen3_dflash, "get_tensor_model_parallel_rank", lambda: 0)

    expected = {
        "layers.0.self_attn.target_k_proj.weight": torch.arange(6.0).view(2, 3),
        "layers.0.self_attn.target_k_proj.bias": torch.tensor([6.0, 7.0]),
        "layers.0.self_attn.target_v_proj.weight": torch.arange(8.0, 14.0).view(2, 3),
        "layers.0.self_attn.target_v_proj.bias": torch.tensor([14.0, 15.0]),
    }

    loaded = model.load_weights(expected.items())

    assert loaded == set(expected)
    for name, value in expected.items():
        torch.testing.assert_close(model.get_parameter(name), value)


def _adaptive_dflash_causal_lm() -> DFlashQwen3ForCausalLM:
    class WeightOnlyNorm(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.empty(3))

    class Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self.target_k_proj = nn.Linear(3, 2, bias=True)
            self.target_v_proj = nn.Linear(3, 2, bias=True)

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = Attention()

    model = DFlashQwen3Model.__new__(DFlashQwen3Model)
    nn.Module.__init__(model)
    model.layers = nn.ModuleList([Layer()])
    model.target_fusion_type = "adaptive_layerwise"
    model.separate_target_kv = True
    model.config = SimpleNamespace(attention_bias=True)
    model.target_fusion_logits = nn.Parameter(torch.empty(1, 2))
    model.target_hidden_norms = nn.ModuleList([WeightOnlyNorm()])
    model.use_aux_hidden_state = True
    model.has_separate_mask_embedding = False
    model.mask_token_id = None
    model._build_fused_kv_buffers = lambda: None

    causal_lm = DFlashQwen3ForCausalLM.__new__(DFlashQwen3ForCausalLM)
    nn.Module.__init__(causal_lm)
    causal_lm.model = model
    return causal_lm


def _adaptive_dflash_weights() -> dict[str, torch.Tensor]:
    return {
        "target_fusion_logits": torch.tensor([[0.25, 0.75]]),
        "target_hidden_norms.0.weight": torch.tensor([1.0, 2.0, 3.0]),
        "layers.0.self_attn.target_k_proj.weight": torch.arange(6.0).view(2, 3),
        "layers.0.self_attn.target_k_proj.bias": torch.tensor([6.0, 7.0]),
        "layers.0.self_attn.target_v_proj.weight": torch.arange(8.0, 14.0).view(2, 3),
        "layers.0.self_attn.target_v_proj.bias": torch.tensor([14.0, 15.0]),
    }


def test_dflash_adaptive_loader_preserves_none_return_for_weight_sharing(monkeypatch):
    causal_lm = _adaptive_dflash_causal_lm()
    monkeypatch.setattr(qwen3_dflash, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(qwen3_dflash, "get_tensor_model_parallel_rank", lambda: 0)

    loaded = causal_lm.load_weights(_adaptive_dflash_weights().items())

    assert loaded is None


def test_default_loader_does_not_enable_strict_tracking_for_adaptive_drafter(
    monkeypatch,
):
    causal_lm = _adaptive_dflash_causal_lm()
    monkeypatch.setattr(qwen3_dflash, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(qwen3_dflash, "get_tensor_model_parallel_rank", lambda: 0)
    loader = DefaultModelLoader.__new__(DefaultModelLoader)
    loader.load_config = SimpleNamespace()
    loader.enable_weights_track = None
    loader.counter_before_loading_weights = 0.0
    loader._init_ep_weight_filter = lambda model_config: None
    loader.get_all_weights = lambda model_config, model: iter(
        _adaptive_dflash_weights().items()
    )
    tracked = []
    loader.track_weights_loading = lambda model, loaded: tracked.append(loaded)

    loader.load_weights(causal_lm, SimpleNamespace(quantization=None))

    assert tracked == []


def test_adaptive_required_weights_follow_serialized_quantized_parameters():
    class QuantizedProjection(nn.Module):
        def __init__(self):
            super().__init__()
            self.qweight = nn.Parameter(torch.empty(2, 3), requires_grad=False)
            self.scales = nn.Parameter(torch.empty(2), requires_grad=False)
            self.quant_method = SimpleNamespace(quant_config=SimpleNamespace())

    causal_lm = _adaptive_dflash_causal_lm()
    for layer in causal_lm.model.layers:
        layer.self_attn.target_k_proj = QuantizedProjection()
        layer.self_attn.target_v_proj = QuantizedProjection()

    required = _required_adaptive_checkpoint_weights(causal_lm.model)

    assert "layers.0.self_attn.target_k_proj.qweight" in required
    assert "layers.0.self_attn.target_k_proj.scales" in required
    assert "layers.0.self_attn.target_k_proj.weight" not in required
    assert "layers.0.self_attn.target_k_proj.bias" not in required


def test_adaptive_loader_accepts_quantized_projection_keys(monkeypatch):
    class QuantizedProjection(nn.Module):
        def __init__(self):
            super().__init__()
            self.qweight = nn.Parameter(torch.empty(2, 3), requires_grad=False)
            self.scales = nn.Parameter(torch.empty(2), requires_grad=False)
            self.quant_method = SimpleNamespace(quant_config=SimpleNamespace())

    causal_lm = _adaptive_dflash_causal_lm()
    for layer in causal_lm.model.layers:
        layer.self_attn.target_k_proj = QuantizedProjection()
        layer.self_attn.target_v_proj = QuantizedProjection()
    monkeypatch.setattr(qwen3_dflash, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(qwen3_dflash, "get_tensor_model_parallel_rank", lambda: 0)
    weights = {
        "target_fusion_logits": torch.tensor([[0.25, 0.75]]),
        "target_hidden_norms.0.weight": torch.tensor([1.0, 2.0, 3.0]),
        "layers.0.self_attn.target_k_proj.qweight": torch.arange(6.0).view(2, 3),
        "layers.0.self_attn.target_k_proj.scales": torch.tensor([1.0, 2.0]),
        "layers.0.self_attn.target_v_proj.qweight": torch.arange(6.0, 12.0).view(2, 3),
        "layers.0.self_attn.target_v_proj.scales": torch.tensor([3.0, 4.0]),
    }

    loaded = causal_lm.load_weights(weights.items())

    assert loaded is None
    for name, value in weights.items():
        torch.testing.assert_close(causal_lm.model.get_parameter(name), value)


@pytest.mark.parametrize("missing_suffix", ["qweight", "scales"])
def test_adaptive_loader_requires_each_quantized_projection_parameter(
    monkeypatch, missing_suffix
):
    class QuantizedProjection(nn.Module):
        def __init__(self):
            super().__init__()
            self.qweight = nn.Parameter(torch.empty(2, 3), requires_grad=False)
            self.scales = nn.Parameter(torch.empty(2), requires_grad=False)
            self.quant_method = SimpleNamespace(quant_config=SimpleNamespace())

    causal_lm = _adaptive_dflash_causal_lm()
    for layer in causal_lm.model.layers:
        layer.self_attn.target_k_proj = QuantizedProjection()
        layer.self_attn.target_v_proj = QuantizedProjection()
    monkeypatch.setattr(qwen3_dflash, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(qwen3_dflash, "get_tensor_model_parallel_rank", lambda: 0)
    weights = {
        "target_fusion_logits": torch.tensor([[0.25, 0.75]]),
        "target_hidden_norms.0.weight": torch.tensor([1.0, 2.0, 3.0]),
        "layers.0.self_attn.target_k_proj.qweight": torch.empty(2, 3),
        "layers.0.self_attn.target_k_proj.scales": torch.empty(2),
        "layers.0.self_attn.target_v_proj.qweight": torch.empty(2, 3),
        "layers.0.self_attn.target_v_proj.scales": torch.empty(2),
    }
    weights.pop(f"layers.0.self_attn.target_k_proj.{missing_suffix}")

    with pytest.raises(ValueError, match="missing required adaptive weights"):
        causal_lm.load_weights(weights.items())


def test_adaptive_required_weights_exclude_online_generated_quant_scales():
    class OnlineQuantizedProjection(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.empty(2, 3), requires_grad=False)
            self.bias = nn.Parameter(torch.empty(2), requires_grad=False)
            self.weight_scale = nn.Parameter(torch.empty(1), requires_grad=False)
            quant_config = SimpleNamespace(is_checkpoint_fp8_serialized=False)
            self.quant_method = SimpleNamespace(quant_config=quant_config)

    causal_lm = _adaptive_dflash_causal_lm()
    for layer in causal_lm.model.layers:
        layer.self_attn.target_k_proj = OnlineQuantizedProjection()
        layer.self_attn.target_v_proj = OnlineQuantizedProjection()

    required = _required_adaptive_checkpoint_weights(causal_lm.model)

    assert "layers.0.self_attn.target_k_proj.weight" in required
    assert "layers.0.self_attn.target_k_proj.bias" in required
    assert "layers.0.self_attn.target_k_proj.weight_scale" not in required


@pytest.mark.parametrize(
    "missing_group",
    [
        "target_fusion_logits",
        "target_hidden_norms.0",
        "target_k_proj.weight",
        "target_k_proj.bias",
        "target_v_proj.weight",
        "target_v_proj.bias",
    ],
)
def test_dflash_adaptive_loader_fails_when_required_group_is_missing(
    monkeypatch, missing_group
):
    causal_lm = _adaptive_dflash_causal_lm()
    monkeypatch.setattr(qwen3_dflash, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(qwen3_dflash, "get_tensor_model_parallel_rank", lambda: 0)
    weights = {
        name: value
        for name, value in _adaptive_dflash_weights().items()
        if missing_group not in name
    }

    with pytest.raises(ValueError, match="missing required adaptive weights"):
        causal_lm.load_weights(weights.items())


@pytest.mark.parametrize("config_name", ["dflash_config", "eagle_config"])
def test_eagle_aux_layers_preserves_legacy_layer_ids(config_name):
    layer_ids = [1, 17, 32]
    vllm_config = _vllm_config(
        **{config_name: {"layer_ids": layer_ids}},
    )

    assert get_eagle3_aux_layers_from_config(vllm_config.speculative_config) == tuple(
        layer_ids
    )
