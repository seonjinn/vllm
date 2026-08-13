# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.spec_decode.autoregressive import speculator as spec_module
from vllm.v1.worker.gpu.spec_decode.autoregressive.speculator import (
    AutoRegressiveSpeculator,
)


class _TestSpeculator(AutoRegressiveSpeculator):
    def load_draft_model(self, target_model, target_attn_layer_names):
        raise NotImplementedError


class _DraftModel(torch.nn.Module):
    def __init__(self, output: torch.Tensor | tuple[torch.Tensor, torch.Tensor]):
        super().__init__()
        self.output = output

    def forward(self, **kwargs):
        return self.output


def _make_speculator(
    monkeypatch,
    output: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
) -> _TestSpeculator:
    monkeypatch.setattr(
        spec_module,
        "set_forward_context",
        lambda *args, **kwargs: nullcontext(),
    )

    speculator = object.__new__(_TestSpeculator)
    speculator.supports_mm_inputs = False
    speculator.vllm_config = None
    speculator.input_buffers = SimpleNamespace(
        input_ids=torch.arange(4),
        positions=torch.arange(4),
    )
    speculator.hidden_states = torch.zeros(4, 3)
    speculator.model = _DraftModel(output)
    return speculator


def test_run_model_unpacks_tuple_return_for_mtp(monkeypatch):
    logits_hidden = torch.full((4, 3), 1.0)
    feedback_hidden = torch.full((4, 3), 2.0)
    speculator = _make_speculator(monkeypatch, (logits_hidden, feedback_hidden))

    actual_logits_hidden, actual_feedback_hidden = speculator._run_model(
        4,
        attn_metadata=None,
        slot_mappings=None,
        num_tokens_across_dp=None,
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
    )

    assert actual_logits_hidden is logits_hidden
    assert actual_feedback_hidden is feedback_hidden


def test_run_model_reuses_tensor_return_for_mtp(monkeypatch):
    hidden = torch.full((4, 3), 1.0)
    speculator = _make_speculator(monkeypatch, hidden)

    actual_logits_hidden, actual_feedback_hidden = speculator._run_model(
        4,
        attn_metadata=None,
        slot_mappings=None,
        num_tokens_across_dp=None,
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
    )

    assert actual_logits_hidden is hidden
    assert actual_feedback_hidden is hidden


def test_propose_k0_runs_prefill_without_draft_decode(monkeypatch):
    speculator = object.__new__(_TestSpeculator)
    speculator.num_speculative_steps = 2
    speculator.max_model_len = 32
    speculator.max_num_reqs = 2
    speculator.dp_size = 1
    speculator.dp_rank = 0
    speculator.supports_mm_inputs = False
    speculator.hidden_states = torch.zeros(2, 3)
    speculator.draft_tokens = torch.tensor([[11, 12], [21, 22]])
    speculator.last_token_indices = torch.zeros(2, dtype=torch.int64)
    speculator.current_draft_step = torch.tensor(0)
    speculator.input_buffers = SimpleNamespace()
    speculator.prefill_cudagraph_manager = None
    speculator.decode_cudagraph_manager = None
    speculator._copy_request_inputs = Mock()
    speculator._prepare_eplb_forward = Mock()
    speculator._prefill = Mock()
    speculator._multi_step_decode = Mock()

    prepare_prefill = Mock()
    prepare_decode = Mock()
    monkeypatch.setattr(spec_module, "prepare_prefill_inputs", prepare_prefill)
    monkeypatch.setattr(spec_module, "prepare_decode_inputs", prepare_decode)
    monkeypatch.setattr(spec_module, "get_uniform_token_count", Mock())
    monkeypatch.setattr(
        spec_module,
        "dispatch_cg_and_sync_dp",
        Mock(
            return_value=(
                BatchExecutionDescriptor(
                    cg_mode=CUDAGraphMode.NONE,
                    num_tokens=2,
                    num_reqs=2,
                ),
                None,
            )
        ),
    )

    input_batch = SimpleNamespace(
        num_tokens=2,
        num_tokens_after_padding=2,
        num_reqs=2,
        num_scheduled_tokens=torch.ones(2, dtype=torch.int32),
        seq_lens=torch.ones(2, dtype=torch.int32),
        seq_lens_cpu_upper_bound=torch.ones(2, dtype=torch.int32),
        idx_mapping=torch.arange(2),
        has_prefill=False,
    )
    output = speculator.propose(
        input_batch,
        attn_metadata={},
        slot_mappings={},
        last_hidden_states=torch.ones(2, 3),
        aux_hidden_states=None,
        num_sampled=torch.ones(2, dtype=torch.int32),
        num_rejected=torch.zeros(2, dtype=torch.int32),
        last_sampled=torch.zeros(2, dtype=torch.int64),
        next_prefill_tokens=torch.zeros(2, dtype=torch.int64),
        temperature=torch.zeros(2),
        seeds=torch.zeros(2, dtype=torch.int64),
        num_speculative_tokens=0,
    )

    assert output.shape == (2, 0)
    speculator._prefill.assert_called_once()
    prepare_decode.assert_not_called()
    speculator._multi_step_decode.assert_not_called()
