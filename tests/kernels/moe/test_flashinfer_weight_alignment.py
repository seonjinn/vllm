# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

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
