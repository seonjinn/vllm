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
    MXFP8_TRTLLM_LAYOUT_ENV,
    MXFP8_TRTLLM_SWITCH_M_ENV,
    _mxfp8_layout_for_compile_range,
    _mxfp8_trtllm_layout_config,
    _parse_mxfp8_tactic_hints,
    _resolve_mxfp8_high_m_tactic,
    _specialize_mxfp8_adaptive_layout_graph,
    mxfp8_trtllm_high_m_static_tactics_enabled,
    mxfp8_trtllm_scale_numel,
    mxfp8_trtllm_use_8x4_sf_layout,
)


@pytest.fixture(autouse=True)
def reset_mxfp8_layout_config() -> None:
    _mxfp8_trtllm_layout_config.cache_clear()
    yield
    _mxfp8_trtllm_layout_config.cache_clear()


@pytest.mark.parametrize(
    ("policy", "m", "expected"),
    [
        ("8x4", 1, True),
        ("8x4", 8480, True),
        ("128x4", 1, False),
        ("128x4", 8480, False),
        ("adaptive", 256, True),
        ("adaptive", 257, False),
    ],
)
def test_mxfp8_trtllm_layout_policy(
    monkeypatch: pytest.MonkeyPatch,
    policy: str,
    m: int,
    expected: bool,
) -> None:
    monkeypatch.setenv(MXFP8_TRTLLM_LAYOUT_ENV, policy)
    monkeypatch.delenv(MXFP8_TRTLLM_SWITCH_M_ENV, raising=False)
    assert mxfp8_trtllm_use_8x4_sf_layout(m) is expected


def test_mxfp8_trtllm_adaptive_switch_is_configurable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(MXFP8_TRTLLM_LAYOUT_ENV, "adaptive")
    monkeypatch.setenv(MXFP8_TRTLLM_SWITCH_M_ENV, "32")
    assert mxfp8_trtllm_use_8x4_sf_layout(32)
    assert not mxfp8_trtllm_use_8x4_sf_layout(33)


def test_mxfp8_trtllm_layout_policy_rejects_invalid_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(MXFP8_TRTLLM_LAYOUT_ENV, "invalid")
    with pytest.raises(ValueError, match=MXFP8_TRTLLM_LAYOUT_ENV):
        mxfp8_trtllm_use_8x4_sf_layout(1)


def test_mxfp8_trtllm_adaptive_switch_rejects_non_integer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(MXFP8_TRTLLM_LAYOUT_ENV, "adaptive")
    monkeypatch.setenv(MXFP8_TRTLLM_SWITCH_M_ENV, "not-an-integer")
    with pytest.raises(ValueError, match=MXFP8_TRTLLM_SWITCH_M_ENV):
        mxfp8_trtllm_use_8x4_sf_layout(1)


def test_mxfp8_trtllm_fixed_layout_ignores_adaptive_switch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(MXFP8_TRTLLM_LAYOUT_ENV, "8x4")
    monkeypatch.setenv(MXFP8_TRTLLM_SWITCH_M_ENV, "not-an-integer")
    assert mxfp8_trtllm_use_8x4_sf_layout(8480)


def test_mxfp8_trtllm_layout_config_is_process_stable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(MXFP8_TRTLLM_LAYOUT_ENV, "adaptive")
    monkeypatch.setenv(MXFP8_TRTLLM_SWITCH_M_ENV, "32")
    assert not mxfp8_trtllm_use_8x4_sf_layout(64)

    monkeypatch.setenv(MXFP8_TRTLLM_SWITCH_M_ENV, "128")
    assert not mxfp8_trtllm_use_8x4_sf_layout(64)


@pytest.mark.parametrize("m", [1, 2, 4, 8, 16, 32, 64, 128, 256])
def test_mxfp8_trtllm_uses_8x4_for_low_m(
    m: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(MXFP8_TRTLLM_LAYOUT_ENV, raising=False)
    monkeypatch.delenv(MXFP8_TRTLLM_SWITCH_M_ENV, raising=False)
    assert mxfp8_trtllm_use_8x4_sf_layout(m)


@pytest.mark.parametrize("m", [257, 512, 1024])
def test_mxfp8_trtllm_uses_128x4_above_threshold(
    m: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(MXFP8_TRTLLM_LAYOUT_ENV, raising=False)
    monkeypatch.delenv(MXFP8_TRTLLM_SWITCH_M_ENV, raising=False)
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
    hints = _parse_mxfp8_tactic_hints("1000,8768,8192:92;4004,8192,4096:91")
    assert hints == {
        (1000, 8768, 8192): 92,
        (4004, 8192, 4096): 91,
    }


def test_mxfp8_high_m_tactic_exact_hit_and_global_fallback() -> None:
    hints = {(1000, 8768, 8192): 92}
    assert (
        _resolve_mxfp8_high_m_tactic(
            1000,
            8768,
            8192,
            hints,
            -1,
            use_global_fallback=False,
        )
        == 92
    )
    assert (
        _resolve_mxfp8_high_m_tactic(
            2002,
            8768,
            8192,
            hints,
            91,
            use_global_fallback=False,
        )
        is None
    )
    assert (
        _resolve_mxfp8_high_m_tactic(
            2002,
            8768,
            8192,
            hints,
            91,
            use_global_fallback=True,
        )
        == 91
    )


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
