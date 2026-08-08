# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.worker.gpu import cudagraph_utils
from vllm.v1.worker.gpu.attn_utils import build_attn_metadata
from vllm.v1.worker.gpu.spec_decode import speculator as speculator_base
from vllm.v1.worker.gpu.spec_decode.dflash import speculator as dflash_speculator
from vllm.v1.worker.gpu.spec_decode.dflash.cudagraph import DFlashCudaGraphManager
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator


def test_build_attn_metadata_resolves_causality_per_kv_group():
    observed_causality = []

    class MetadataBuilder:
        def build(self, *, common_prefix_len, common_attn_metadata):
            assert common_prefix_len == 0
            observed_causality.append(common_attn_metadata.causal)
            return object()

    class AttentionGroup:
        def __init__(self, layer_name):
            self.layer_names = [layer_name]
            self.builder = MetadataBuilder()

        def get_metadata_builder(self, index):
            assert index == 0
            return self.builder

    build_attn_metadata(
        attn_groups=[
            [AttentionGroup("causal_layer")],
            [AttentionGroup("non_causal_layer")],
        ],
        num_reqs=1,
        num_tokens=1,
        query_start_loc_gpu=torch.tensor([0, 1]),
        query_start_loc_cpu=torch.tensor([0, 1]),
        max_query_len=1,
        seq_lens=torch.tensor([1]),
        max_seq_len=1,
        block_tables=[torch.tensor([[0]]), torch.tensor([[0]])],
        slot_mappings=torch.tensor([[0], [0]]),
        kv_cache_config=SimpleNamespace(kv_cache_groups=[object(), object()]),
        causal={0: True, 1: False},
    )

    assert observed_causality == [True, False]


@pytest.mark.parametrize(
    ("speculator_cls", "group_causal"),
    [
        (dflash_speculator.DFlashSpeculator, False),
        (DSparkSpeculator, {0: True, 1: False}),
    ],
)
def test_init_cudagraph_manager_does_not_store_group_causal(
    monkeypatch, speculator_cls, group_causal
):
    speculator = object.__new__(speculator_cls)
    speculator.attn_cg_support = SimpleNamespace(
        min_cg_support=AttentionCGSupport.UNIFORM_BATCH,
        min_cg_attn_backend="test",
    )
    speculator.vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_seqs=4),
        compilation_config=SimpleNamespace(cudagraph_capture_sizes=[]),
        parallel_config=SimpleNamespace(
            data_parallel_size=1,
            tensor_parallel_size=1,
        ),
        speculative_config=None,
    )
    speculator.device = torch.device("cpu")
    speculator.num_query_per_req = 3
    speculator._group_causal = group_causal

    monkeypatch.setattr(
        cudagraph_utils,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    monkeypatch.setattr(
        cudagraph_utils,
        "is_breakable_cudagraph_enabled",
        lambda: False,
    )

    speculator.init_cudagraph_manager(CUDAGraphMode.NONE)

    manager = speculator.query_cudagraph_manager
    assert isinstance(manager, DFlashCudaGraphManager)
    assert not hasattr(manager, "causal")


@pytest.mark.parametrize(
    ("speculator_cls", "group_causal"),
    [
        (dflash_speculator.DFlashSpeculator, False),
        (DSparkSpeculator, {0: True, 1: False}),
    ],
)
def test_capture_uses_group_causal(speculator_cls, group_causal):
    capture_kwargs = {}

    def capture(*args, **kwargs):
        capture_kwargs.update(kwargs)

    speculator = object.__new__(speculator_cls)
    speculator.sample_indices = torch.zeros(1)
    speculator.sample_pos = torch.zeros(1)
    speculator.sample_idx_mapping = torch.zeros(1)
    speculator.query_cudagraph_manager = SimpleNamespace(capture=capture)
    speculator.input_buffers = object()
    speculator.block_tables = object()
    speculator.attn_groups = []
    speculator.kv_cache_config = object()
    speculator.max_model_len = 1
    speculator._group_causal = group_causal

    speculator.capture()

    assert capture_kwargs["causal"] == group_causal


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
