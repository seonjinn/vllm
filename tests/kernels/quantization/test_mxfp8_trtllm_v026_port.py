# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

ROOT = Path(__file__).parents[3]


def test_v026_registers_trtllm_as_an_mxfp8_linear_backend() -> None:
    registry_source = (
        ROOT / "vllm/model_executor/kernels/linear/__init__.py"
    ).read_text()
    kernel_source = (
        ROOT / "vllm/model_executor/kernels/linear/mxfp8/flashinfer.py"
    ).read_text()

    assert "class FlashInferTrtllmMxfp8LinearKernel" in kernel_source
    assert '"flashinfer_trtllm": {' in registry_source
    assert "FlashInferTrtllmMxfp8LinearKernel," in registry_source


def test_v026_adaptive_policy_has_exact_layout_and_safe_fallback_controls() -> None:
    source = (
        ROOT / "vllm/model_executor/layers/quantization/utils/mxfp8_utils.py"
    ).read_text()

    assert "VLLM_MXFP8_DENSE_TRTLLM_LAYOUT" in source
    assert "VLLM_MXFP8_DENSE_TRTLLM_SWITCH_M" in source
    assert "mxfp8_trtllm_use_8x4_sf_layout" in source
    assert "fallback_tactic = int(os.environ.get(" in source
    assert '"-1"' in source


def test_v026_exact_tactics_keep_both_layouts_on_direct_trtllm_runners() -> None:
    source = (
        ROOT / "vllm/model_executor/layers/quantization/utils/mxfp8_utils.py"
    ).read_text()

    assert "class _Mxfp8TrtllmTacticState" in source
    assert "runner_8x4" in source
    assert "runner_128x4" in source
    assert "runtime_illegal_tactic" in source
    assert "unresolved MXFP8 tactic before CUDA Graph capture" in source
    assert "flashinfer_mm_mxfp8" not in source
