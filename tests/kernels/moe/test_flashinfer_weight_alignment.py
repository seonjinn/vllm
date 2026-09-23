# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from torch import nn

from vllm.model_executor.layers.fused_moe.config import FusedMoEParallelConfig
from vllm.model_executor.layers.fused_moe.oracle.unquantized import (
    UnquantizedMoeBackend,
)
from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
    UnquantizedFusedMoEMethod,
)
from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
    align_moe_weights_for_fi,
)


@pytest.mark.parametrize("intermediate", [192, 256, 928])
@pytest.mark.parametrize("gated", [False, True])
def test_align_moe_preserves_each_projection(intermediate: int, gated: bool) -> None:
    experts, hidden = 2, 16
    multiplier = 2 if gated else 1
    w13 = torch.empty(experts, multiplier * intermediate, hidden, dtype=torch.bfloat16)
    w2 = torch.empty(experts, hidden, intermediate, dtype=torch.bfloat16)
    for expert in range(experts):
        for projection in range(multiplier):
            w13[expert, projection * intermediate : (projection + 1) * intermediate] = (
                1 + expert * multiplier + projection
            )
        w2[expert].fill_(7 + expert)

    original_w13, original_w2 = w13.clone(), w2.clone()
    aligned_w13, aligned_w2, padded = align_moe_weights_for_fi(
        w13, w2, gated, min_alignment=128
    )

    assert padded == ((intermediate + 127) // 128) * 128
    assert aligned_w13.shape == (experts, multiplier * padded, hidden)
    assert aligned_w2.shape == (experts, hidden, padded)
    for projection in range(multiplier):
        actual = aligned_w13[:, projection * padded : (projection + 1) * padded]
        expected = original_w13[
            :, projection * intermediate : (projection + 1) * intermediate
        ]
        torch.testing.assert_close(actual[:, :intermediate], expected, rtol=0, atol=0)
        assert torch.count_nonzero(actual[:, intermediate:]).item() == 0
    torch.testing.assert_close(
        aligned_w2[:, :, :intermediate], original_w2, rtol=0, atol=0
    )
    assert torch.count_nonzero(aligned_w2[:, :, intermediate:]).item() == 0

    twice_w13, twice_w2, twice_padded = align_moe_weights_for_fi(
        aligned_w13, aligned_w2, gated, min_alignment=128
    )
    assert twice_padded == padded
    assert twice_w13.data_ptr() == aligned_w13.data_ptr()
    assert twice_w2.data_ptr() == aligned_w2.data_ptr()


def test_unquantized_trtllm_rounds_intermediate_before_kernel_setup() -> None:
    method = object.__new__(UnquantizedFusedMoEMethod)
    method.unquantized_backend = UnquantizedMoeBackend.FLASHINFER_TRTLLM

    hidden, intermediate = method.maybe_roundup_sizes(
        hidden_size=1024,
        intermediate_size_per_partition=64,
        act_dtype=torch.bfloat16,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
    )

    assert hidden == 1024
    assert intermediate == 128


def test_unquantized_trtllm_zeroes_gated_intermediate_padding() -> None:
    method = object.__new__(UnquantizedFusedMoEMethod)
    method.unquantized_backend = UnquantizedMoeBackend.FLASHINFER_TRTLLM
    method.moe = type(
        "MoeConfig",
        (),
        {
            "intermediate_size_per_partition_unpadded": 64,
            "is_act_and_mul": True,
        },
    )()
    layer = nn.Module()
    layer.moe_config = method.moe
    layer.register_parameter(
        "w13_weight",
        nn.Parameter(torch.ones(2, 256, 32), requires_grad=False),
    )
    layer.register_parameter(
        "w2_weight",
        nn.Parameter(torch.ones(2, 32, 128), requires_grad=False),
    )

    method._zero_trtllm_padding(layer)

    assert torch.all(layer.w13_weight[:, :64] == 1)
    assert torch.count_nonzero(layer.w13_weight[:, 64:128]).item() == 0
    assert torch.all(layer.w13_weight[:, 128:192] == 1)
    assert torch.count_nonzero(layer.w13_weight[:, 192:]).item() == 0
    assert torch.all(layer.w2_weight[:, :, :64] == 1)
    assert torch.count_nonzero(layer.w2_weight[:, :, 64:]).item() == 0
