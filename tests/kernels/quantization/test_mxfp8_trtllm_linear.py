# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from vllm.model_executor.kernels.linear.mxfp8 import Mxfp8LinearLayerConfig
from vllm.model_executor.kernels.linear.mxfp8.flashinfer import (
    FlashInferTrtllmMxfp8LinearKernel,
)
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    MXFP8_TRTLLM_HIGH_M_TACTIC_ENV,
    MXFP8_TRTLLM_HIGH_M_TACTIC_HINTS_ENV,
    _mxfp8_layout_for_compile_range,
    _parse_mxfp8_tactic_hints,
    _resolve_mxfp8_high_m_tactic,
    _specialize_mxfp8_adaptive_layout_graph,
    mxfp8_trtllm_high_m_static_tactics_enabled,
    mxfp8_trtllm_scale_numel,
    mxfp8_trtllm_use_8x4_sf_layout,
)


@pytest.mark.parametrize("m", [1, 2, 4, 8, 16, 32, 64, 128, 256])
def test_mxfp8_trtllm_uses_8x4_for_low_m(m: int) -> None:
    assert mxfp8_trtllm_use_8x4_sf_layout(m)


@pytest.mark.parametrize("m", [257, 512, 1024])
def test_mxfp8_trtllm_uses_128x4_above_threshold(m: int) -> None:
    assert not mxfp8_trtllm_use_8x4_sf_layout(m)


@pytest.mark.parametrize(
    ("m", "k", "use_8x4", "expected"),
    [
        (1, 5120, True, 8 * 160),
        (32, 5120, True, 32 * 160),
        (33, 5120, True, 40 * 160),
        (128, 5120, True, 128 * 160),
        (129, 5120, True, 136 * 160),
        (257, 5120, False, 384 * 160),
        (8, 5184, True, 8 * 164),
    ],
)
def test_mxfp8_trtllm_scale_numel(
    m: int,
    k: int,
    use_8x4: bool,
    expected: int,
) -> None:
    assert mxfp8_trtllm_scale_numel(m, k, use_8x4) == expected


def test_mxfp8_trtllm_scale_numel_rejects_invalid_k() -> None:
    with pytest.raises(ValueError, match="divisible by 32"):
        mxfp8_trtllm_scale_numel(8, 5130, True)


def test_mxfp8_layout_compile_ranges_do_not_straddle_switch() -> None:
    assert _mxfp8_layout_for_compile_range(1, 256, 256)
    assert not _mxfp8_layout_for_compile_range(257, 8480, 256)
    with pytest.raises(RuntimeError, match="straddles"):
        _mxfp8_layout_for_compile_range(1, 2048, 256)


def test_mxfp8_high_m_tactic_hints_use_logical_shape() -> None:
    hints = _parse_mxfp8_tactic_hints(
        "1000,8768,8192:92;4004,8192,4096:91"
    )
    assert hints == {
        (1000, 8768, 8192): 92,
        (4004, 8192, 4096): 91,
    }


def test_mxfp8_high_m_tactic_exact_hit_and_global_fallback() -> None:
    hints = {(1000, 8768, 8192): 92}
    assert _resolve_mxfp8_high_m_tactic(1000, 8768, 8192, hints, -1) == 92
    assert _resolve_mxfp8_high_m_tactic(2002, 8768, 8192, hints, 91) == 91


def test_mxfp8_high_m_static_tactics_are_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(MXFP8_TRTLLM_HIGH_M_TACTIC_ENV, raising=False)
    monkeypatch.delenv(MXFP8_TRTLLM_HIGH_M_TACTIC_HINTS_ENV, raising=False)
    assert not mxfp8_trtllm_high_m_static_tactics_enabled()

    monkeypatch.setenv(MXFP8_TRTLLM_HIGH_M_TACTIC_ENV, "92")
    assert mxfp8_trtllm_high_m_static_tactics_enabled()


@pytest.mark.parametrize(
    ("fixed_op", "expected_op"),
    [
        (
            torch.ops.vllm.mxfp8_trtllm_linear_8x4.default,
            torch.ops.vllm.mxfp8_trtllm_linear_8x4.default,
        ),
        (
            torch.ops.vllm.mxfp8_trtllm_linear_128x4.default,
            torch.ops.vllm.mxfp8_trtllm_linear_128x4.default,
        ),
    ],
)
def test_mxfp8_adaptive_marker_is_specialized(
    fixed_op: object, expected_op: object
) -> None:
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    weight = graph.placeholder("weight")
    scale = graph.placeholder("scale")
    node = graph.call_function(
        torch.ops.vllm.mxfp8_trtllm_adaptive_linear.default,
        (x, weight, scale, 512),
    )
    graph.output(node)

    replaced = _specialize_mxfp8_adaptive_layout_graph(
        graph,
        marker_op=torch.ops.vllm.mxfp8_trtllm_adaptive_linear.default,
        fixed_op=fixed_op,
    )

    assert replaced == 1
    assert node.target == expected_op


def test_mxfp8_trtllm_linear_rejects_fp16_activations() -> None:
    kernel = object.__new__(FlashInferTrtllmMxfp8LinearKernel)
    with pytest.raises(ValueError, match="requires BF16 activations"):
        kernel.apply_weights(torch.nn.Module(), torch.empty(1, 32, dtype=torch.float16))


@pytest.mark.parametrize(("m", "n"), [(4, 512), (64, 520)])
def test_mxfp8_trtllm_linear_matches_bf16(
    m: int, n: int, default_vllm_config: object
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("requires SM100/SM103")

    flashinfer = pytest.importorskip("flashinfer")
    torch.manual_seed(7)
    k = 512
    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16) * 0.1
    weight_bf16 = torch.randn((n, k), device="cuda", dtype=torch.bfloat16) * 0.02
    weight, weight_scale = flashinfer.mxfp8_quantize(
        weight_bf16,
        backend="cuda",
        sf_swizzle_layout=flashinfer.SfLayout.layout_linear,
    )

    layer = torch.nn.Module()
    layer.weight = Parameter(weight, requires_grad=False)
    layer.weight_scale = Parameter(weight_scale.view(n, k // 32), requires_grad=False)
    kernel = FlashInferTrtllmMxfp8LinearKernel(Mxfp8LinearLayerConfig())
    kernel.process_weights_after_loading(layer)

    compiled_apply = torch.compile(
        lambda input_: kernel.apply_weights(layer, input_), fullgraph=True, dynamic=True
    )
    with flashinfer.autotune(False):
        output = compiled_apply(x)
    reference = x @ weight_bf16.t()

    assert output.shape == (m, n)
    assert torch.isfinite(output).all()
    similarity = F.cosine_similarity(
        output.float().flatten(), reference.float().flatten(), dim=0
    )
    assert similarity > 0.95
