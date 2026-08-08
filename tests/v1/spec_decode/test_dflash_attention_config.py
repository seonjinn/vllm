# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import torch

from vllm.v1.worker.gpu.spec_decode import speculator as speculator_base
from vllm.v1.worker.gpu.spec_decode.dflash import speculator as dflash_speculator


def test_attn_vllm_config_only_replaces_attention_config(monkeypatch):
    attention_config = SimpleNamespace(use_non_causal=False)
    vllm_config = SimpleNamespace(
        attention_config=attention_config,
        untouched=object(),
    )
    speculator = object.__new__(dflash_speculator.DFlashSpeculator)
    speculator.vllm_config = vllm_config
    speculator.requires_non_causal = True

    replaced_objects = []

    def replace_attention_only(obj, **changes):
        replaced_objects.append(obj)
        assert obj is attention_config, "outer VllmConfig must be shallow-copied"
        values = vars(obj).copy()
        values.update(changes)
        return SimpleNamespace(**values)

    monkeypatch.setattr(dflash_speculator, "replace", replace_attention_only)

    attn_vllm_config = speculator.attn_vllm_config

    assert replaced_objects == [attention_config]
    assert attn_vllm_config is not vllm_config
    assert attn_vllm_config.untouched is vllm_config.untouched
    assert attn_vllm_config.attention_config.use_non_causal is True
    assert vllm_config.attention_config.use_non_causal is False


def test_set_attn_updates_all_fa3_builders_with_draft_geometry(monkeypatch):
    target_geometry = {"num_heads_q": 32, "num_heads_kv": 8, "headdim": 128}
    draft_geometry = {"num_heads_q": 16, "num_heads_kv": 4, "headdim": 64}
    builders = [SimpleNamespace(**target_geometry), SimpleNamespace(**target_geometry)]

    class AttentionGroup:
        def __init__(self, builder):
            self.builder = builder

        def get_metadata_builder(self, index):
            assert index == 0
            return self.builder

    speculator = object.__new__(dflash_speculator.DFlashSpeculator)
    speculator.attn_groups = [[AttentionGroup(builder)] for builder in builders]
    speculator.vllm_config = SimpleNamespace(parallel_config=object())
    speculator.draft_model_config = SimpleNamespace(
        get_num_attention_heads=lambda parallel_config: draft_geometry["num_heads_q"],
        get_num_kv_heads=lambda parallel_config: draft_geometry["num_heads_kv"],
        get_head_size=lambda: draft_geometry["headdim"],
    )
    speculator.block_tables = SimpleNamespace(block_sizes=[16, 16])
    speculator.max_num_tokens = 8
    speculator.device = torch.device("cpu")
    speculator.requires_non_causal = False
    speculator.model = object()

    monkeypatch.setattr(
        speculator_base.DraftModelSpeculator, "set_attn", lambda *args: None
    )

    speculator.set_attn(
        model_state=SimpleNamespace(),
        kv_cache_config=SimpleNamespace(kv_cache_groups=[]),
        block_tables=speculator.block_tables,
    )

    for builder in builders:
        assert builder.num_heads_q == draft_geometry["num_heads_q"]
        assert builder.num_heads_kv == draft_geometry["num_heads_kv"]
        assert builder.headdim == draft_geometry["headdim"]
