from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
V020_ADAPTIVE_OVERRIDE = (
    ROOT / "vllm/model_executor/kernels/linear/mxfp8/flashinfer.py"
)


def test_v020_adaptive_override_inherits_repaired_weight_and_tail_contract() -> None:
    source = V020_ADAPTIVE_OVERRIDE.read_text(encoding="utf-8")

    assert "N_padded = ((N + 127) // 128) * 128" in source
    assert "weight_trtllm = shuffle_matrix_a(weight_padded, 128)" in source
    assert "scale_padded[:N, :].copy_(weight_scale_2d)" in source
    assert "shuffle_matrix_sf_a(scale_padded, 128, num_elts_per_sf=32)" in source
    assert "layer._mxfp8_dense_output_features = N" in source
    assert "output = output[:, :output_features]" in source
    assert "weight_trtllm_for_apply" not in source


def test_v020_adaptive_dispatch_keeps_only_shuffled_trtllm_weight() -> None:
    source = V020_ADAPTIVE_OVERRIDE.read_text(encoding="utf-8")

    assert "configure_mxfp8_adaptive_layout_compilation()" in source
    assert "prepare_mxfp8_trtllm_direct_state(" in source
    assert "weight.device" in source
    assert "torch.ops.vllm.mxfp8_adaptive_quantize_mm_marker" in source
    assert "torch.ops.vllm.mxfp8_adaptive_quantize_marker" not in source
    assert "torch.ops.vllm.mxfp8_adaptive_mm_marker" not in source
    assert "layer._mxfp8_trtllm_workspace_8x4" in source
    assert "layer._mxfp8_trtllm_workspace_128x4" in source
    assert "vllm_flashinfer.mm_mxfp8(" in source
    assert "mxfp8_e4m3_quantize(" in source
    assert "torch.ops.vllm.mxfp8_shape_specialized_quantize_mm" not in source
    assert "torch.ops.vllm.mxfp8_adaptive_quantize_mm(" not in source
    assert 'backend = "cutlass"' not in source
    assert "weight_cutlass" not in source


def test_v020_adaptive_apply_validates_prepared_fingerprint() -> None:
    source = V020_ADAPTIVE_OVERRIDE.read_text(encoding="utf-8")

    assert "layer._mxfp8_trtllm_configuration" in source
    assert "validate_mxfp8_trtllm_configuration(" in source


def test_v020_adaptive_layout_disables_bf16_small_m_fallback() -> None:
    source = V020_ADAPTIVE_OVERRIDE.read_text(encoding="utf-8")

    assert 'if os.environ.get("MXFP8_BF16_FALLBACK_SMALL_M") == "1":' not in source
    assert 'and os.environ.get("MXFP8_BF16_FALLBACK_SMALL_M") == "1"' in source
    assert 'if M_orig < 128 and hasattr(layer, "weight_bf16"):' not in source
    assert "not is_adaptive_layout and M_orig < 128" in source
