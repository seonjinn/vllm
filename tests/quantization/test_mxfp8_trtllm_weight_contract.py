# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
import types

import pytest

from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    mxfp8_trtllm_padded_n,
    prepare_mxfp8_trtllm_weight,
)


@pytest.mark.parametrize(
    ("logical_n", "padded_n"),
    [(128, 128), (4384, 4480), (5120, 5120), (8768, 8832)],
)
def test_mxfp8_trtllm_padded_n(logical_n: int, padded_n: int) -> None:
    assert mxfp8_trtllm_padded_n(logical_n) == padded_n


def test_prepare_mxfp8_trtllm_weight_pads_and_shuffles(monkeypatch) -> None:
    calls: list[tuple[object, ...]] = []

    class FakeSlice:
        def copy_(self, source):
            calls.append(("copy", source.shape))

    class FakeTensor:
        def __init__(self, shape, dtype):
            self.shape = tuple(shape)
            self.dtype = dtype

        def new_zeros(self, shape):
            calls.append(("zeros", tuple(shape), self.dtype))
            return FakeTensor(shape, self.dtype)

        def __getitem__(self, _key):
            return FakeSlice()

        def view(self, dtype):
            calls.append(("view", self.shape, dtype))
            return FakeTensor(self.shape, dtype)

        def contiguous(self):
            calls.append(("contiguous", self.shape, self.dtype))
            return self

        def reshape(self, *shape):
            calls.append(("reshape", self.shape, tuple(shape)))
            return FakeTensor(shape, self.dtype)

    def shuffle_matrix_a(tensor, tile_m):
        calls.append(("shuffle_weight", tensor.shape, tile_m))
        return tensor

    def shuffle_matrix_sf_a(tensor, tile_m, *, num_elts_per_sf):
        calls.append(
            ("shuffle_scale", tensor.shape, tile_m, num_elts_per_sf)
        )
        return tensor

    flashinfer = types.ModuleType("flashinfer")
    flashinfer.shuffle_matrix_a = shuffle_matrix_a
    flashinfer.shuffle_matrix_sf_a = shuffle_matrix_sf_a
    monkeypatch.setitem(sys.modules, "flashinfer", flashinfer)

    weight = FakeTensor((4384, 8192), "fp8")
    scale = FakeTensor((4384, 256), "e8m0")
    shuffled_weight, shuffled_scale, logical_n = prepare_mxfp8_trtllm_weight(
        weight, scale
    )

    assert logical_n == 4384
    assert shuffled_weight.shape == (4480, 8192)
    assert shuffled_scale.shape == (4480 * 256,)
    assert ("shuffle_weight", (4480, 8192), 128) in calls
    assert ("shuffle_scale", (4480, 256), 128, 32) in calls
