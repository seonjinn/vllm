# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
import json
import sys
import types
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import vllm.model_executor.kernels.linear.mxfp8.flashinfer as flashinfer_module
from vllm.model_executor.kernels.linear.mxfp8 import Mxfp8LinearLayerConfig
from vllm.model_executor.kernels.linear.mxfp8.flashinfer import (
    FlashInferCutedslMxfp8LinearKernel,
    FlashInferCutlassMxfp8LinearKernel,
    FlashInferTrtllmMxfp8LinearKernel,
)
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    MXFP8_TRTLLM_EXACT_TACTIC_FILE_ENV,
    MXFP8_TRTLLM_EXACT_TACTIC_SHA256_ENV,
    MXFP8_TRTLLM_HIGH_M_TACTIC_ENV,
    MXFP8_TRTLLM_HIGH_M_TACTIC_HINTS_ENV,
    MXFP8_TRTLLM_LAYOUT_ENV,
    MXFP8_TRTLLM_SWITCH_M_ENV,
    _load_mxfp8_exact_tactic_table,
    _mxfp8_layout_for_compile_range,
    _mxfp8_trtllm_layout_config,
    _parse_mxfp8_tactic_hints,
    _resolve_mxfp8_exact_tactic,
    _resolve_mxfp8_high_m_tactic,
    _specialize_mxfp8_adaptive_layout_graph,
    _validate_mxfp8_runtime_fingerprint,
    mxfp8_trtllm_exact_tactics_enabled,
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


def test_mxfp8_exact_tactic_table_uses_full_execution_signature(
    tmp_path: Path,
) -> None:
    table_path = tmp_path / "exact-tactics.json"
    payload = {
        "schema_version": 1,
        "metadata": {
            "runtime_fingerprint": {
                "vllm_version": "0.25.1",
                "flashinfer_version": "0.6.13",
                "cuda_version": "12.9",
                "device_name": "NVIDIA GB200",
                "compute_capability": [10, 0],
            }
        },
        "entries": [
            {
                "m": 8,
                "n_logical": 8768,
                "n_physical": 8832,
                "k": 8192,
                "layout": "8x4",
                "tactic": 65,
                "valid_tactics": [61, 65, 66],
            },
            {
                "m": 1000,
                "n_logical": 8768,
                "n_physical": 8832,
                "k": 8192,
                "layout": "128x4",
                "tactic": 92,
                "valid_tactics": [91, 92, 93],
            },
        ],
    }
    table_path.write_text(json.dumps(payload), encoding="utf-8")
    digest = hashlib.sha256(table_path.read_bytes()).hexdigest()

    table = _load_mxfp8_exact_tactic_table(str(table_path), digest)

    assert table.tactics[(8, 8768, 8832, 8192, "8x4")] == 65
    assert table.tactics[(1000, 8768, 8832, 8192, "128x4")] == 92
    assert (
        _resolve_mxfp8_exact_tactic(
            8,
            8768,
            8832,
            8192,
            "8x4",
            table.tactics,
        )
        == 65
    )
    assert (
        _resolve_mxfp8_exact_tactic(
            16,
            8768,
            8832,
            8192,
            "8x4",
            table.tactics,
        )
        == -1
    )


def test_mxfp8_exact_tactic_table_rejects_sha_mismatch(tmp_path: Path) -> None:
    table_path = tmp_path / "exact-tactics.json"
    table_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "metadata": {
                    "runtime_fingerprint": {
                        "vllm_version": "0.25.1",
                        "flashinfer_version": "0.6.13",
                        "cuda_version": "12.9",
                        "device_name": "NVIDIA GB200",
                        "compute_capability": [10, 0],
                    }
                },
                "entries": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=MXFP8_TRTLLM_EXACT_TACTIC_SHA256_ENV):
        _load_mxfp8_exact_tactic_table(str(table_path), "0" * 64)


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("m", True),
        ("n_logical", 8768.5),
        ("n_physical", "8832"),
        ("k", None),
        ("tactic", 65.0),
    ],
)
def test_mxfp8_exact_tactic_table_rejects_non_integer_fields(
    tmp_path: Path,
    field: str,
    invalid_value: object,
) -> None:
    table_path = tmp_path / f"invalid-{field}.json"
    entry = {
        "m": 8,
        "n_logical": 8768,
        "n_physical": 8832,
        "k": 8192,
        "layout": "8x4",
        "tactic": 65,
        "valid_tactics": [61, 65, 66],
    }
    entry[field] = invalid_value
    payload = {
        "schema_version": 1,
        "metadata": {
            "runtime_fingerprint": {
                "vllm_version": "0.25.1",
                "flashinfer_version": "0.6.13",
                "cuda_version": "12.9",
                "device_name": "NVIDIA GB200",
                "compute_capability": [10, 0],
            }
        },
        "entries": [entry],
    }
    table_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="must be a JSON integer"):
        _load_mxfp8_exact_tactic_table(
            str(table_path),
            hashlib.sha256(table_path.read_bytes()).hexdigest(),
        )


def test_mxfp8_exact_tactic_table_rejects_runtime_fingerprint_mismatch() -> None:
    expected = {
        "vllm_version": "0.25.1",
        "flashinfer_version": "0.6.13",
        "cuda_version": "12.9",
        "device_name": "NVIDIA GB200",
        "compute_capability": [10, 0],
    }
    current = dict(expected)
    current["flashinfer_version"] = "0.6.14"

    with pytest.raises(ValueError, match="flashinfer_version"):
        _validate_mxfp8_runtime_fingerprint(expected, current)


def test_mxfp8_exact_tactic_table_rejects_tactic_outside_valid_set(
    tmp_path: Path,
) -> None:
    table_path = tmp_path / "invalid-tactic.json"
    payload = {
        "schema_version": 1,
        "metadata": {
            "runtime_fingerprint": {
                "vllm_version": "0.25.1",
                "flashinfer_version": "0.6.13",
                "cuda_version": "12.9",
                "device_name": "NVIDIA GB200",
                "compute_capability": [10, 0],
            }
        },
        "entries": [
            {
                "m": 8,
                "n_logical": 8768,
                "n_physical": 8832,
                "k": 8192,
                "layout": "8x4",
                "tactic": 999,
                "valid_tactics": [61, 65, 66],
            }
        ],
    }
    table_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="not in valid_tactics"):
        _load_mxfp8_exact_tactic_table(
            str(table_path),
            hashlib.sha256(table_path.read_bytes()).hexdigest(),
        )


def test_mxfp8_exact_tactic_table_is_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(MXFP8_TRTLLM_EXACT_TACTIC_FILE_ENV, raising=False)
    monkeypatch.delenv(MXFP8_TRTLLM_EXACT_TACTIC_SHA256_ENV, raising=False)
    assert not mxfp8_trtllm_exact_tactics_enabled()

    monkeypatch.setenv(MXFP8_TRTLLM_EXACT_TACTIC_FILE_ENV, "/tmp/tactics.json")
    assert mxfp8_trtllm_exact_tactics_enabled()


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


@pytest.mark.parametrize(
    "kernel_type",
    [
        FlashInferCutlassMxfp8LinearKernel,
        FlashInferCutedslMxfp8LinearKernel,
        FlashInferTrtllmMxfp8LinearKernel,
    ],
)
def test_mxfp8_linear_kernels_declare_refit_safe_capability(kernel_type: type) -> None:
    assert kernel_type.preserves_checkpoint_weight_scale_for_refit is True


def test_mxfp8_cutlass_refit_preserves_checkpoint_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        flashinfer_module,
        "swizzle_mxfp8_scale",
        lambda scale, *, M, K: scale + M + K,
    )
    layer = torch.nn.Module()
    layer.weight = Parameter(torch.arange(256).reshape(4, 64), requires_grad=False)
    layer.weight_scale = Parameter(torch.ones((4, 2)), requires_grad=False)
    weight = layer.weight
    weight_scale = layer.weight_scale
    kernel = object.__new__(FlashInferCutlassMxfp8LinearKernel)

    kernel.process_weights_after_loading(layer)
    prepared_scale = layer.weight_scale_for_apply
    layer.weight.data.add_(1)
    layer.weight_scale.data.mul_(2)
    kernel.process_weights_after_loading(layer)

    assert layer.weight is weight
    assert layer.weight_scale is weight_scale
    assert layer.weight_scale_for_apply is prepared_scale
    assert torch.equal(layer.weight_scale_for_apply, torch.full((4, 2), 70.0))


def test_mxfp8_cutedsl_refit_refreshes_stable_prepared_buffers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        flashinfer_module,
        "swizzle_mxfp8_scale",
        lambda scale, *, M, K: scale + M + K,
    )
    layer = torch.nn.Module()
    layer.weight = Parameter(torch.arange(256).reshape(4, 64), requires_grad=False)
    layer.weight_scale = Parameter(torch.ones((4, 2)), requires_grad=False)
    weight = layer.weight
    weight_scale = layer.weight_scale
    kernel = object.__new__(FlashInferCutedslMxfp8LinearKernel)

    kernel.process_weights_after_loading(layer)
    prepared_weight = layer.weight_for_apply
    prepared_scale = layer.weight_scale_for_apply
    layer.weight.data.add_(1)
    layer.weight_scale.data.mul_(2)
    kernel.process_weights_after_loading(layer)

    assert layer.weight is weight
    assert layer.weight_scale is weight_scale
    assert layer.weight_for_apply is prepared_weight
    assert layer.weight_scale_for_apply is prepared_scale
    assert torch.equal(layer.weight_for_apply, layer.weight.t())
    assert torch.equal(layer.weight_scale_for_apply, torch.full((4, 2), 70.0))


def test_mxfp8_trtllm_refit_refreshes_stable_prepared_buffers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        sys.modules,
        "flashinfer",
        types.SimpleNamespace(
            shuffle_matrix_a=lambda weight, _: weight + 3,
            shuffle_matrix_sf_a=lambda scale, *_args, **_kwargs: scale + 5,
        ),
    )
    monkeypatch.setattr(
        flashinfer_module, "prepare_mxfp8_trtllm_exact_tactic_state", lambda _: None
    )
    monkeypatch.setattr(
        flashinfer_module, "prepare_mxfp8_trtllm_high_m_tactic_state", lambda _: None
    )
    layer = torch.nn.Module()
    layer.weight = Parameter(torch.arange(1024).reshape(4, 256), requires_grad=False)
    layer.weight_scale = Parameter(torch.ones((4, 8)), requires_grad=False)
    weight = layer.weight
    weight_scale = layer.weight_scale
    kernel = object.__new__(FlashInferTrtllmMxfp8LinearKernel)
    kernel._cutedsl_fallback = None

    kernel.process_weights_after_loading(layer)
    prepared_weight = layer.weight_for_apply
    prepared_scale = layer.weight_scale_for_apply
    layer.weight.data.add_(1)
    layer.weight_scale.data.mul_(2)
    kernel.process_weights_after_loading(layer)

    assert layer.weight is weight
    assert layer.weight_scale is weight_scale
    assert layer.weight_for_apply is prepared_weight
    assert layer.weight_scale_for_apply is prepared_scale
    assert torch.equal(layer.weight_for_apply[:4], layer.weight + 3)
    assert torch.equal(layer.weight_scale_for_apply[:32], torch.full((32,), 7.0))


def test_mxfp8_trtllm_unsupported_k_requires_explicit_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("VLLM_MXFP8_DENSE_TRTLLM_ALLOW_CUTEDSL_FALLBACK", raising=False)
    kernel = object.__new__(FlashInferTrtllmMxfp8LinearKernel)
    layer = torch.nn.Module()
    layer.weight = Parameter(
        torch.empty((128, 1344), dtype=torch.float8_e4m3fn), requires_grad=False
    )

    with pytest.raises(ValueError, match="K to be divisible by 256"):
        kernel.process_weights_after_loading(layer)


def test_mxfp8_trtllm_unsupported_k_uses_cutedsl_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_MXFP8_DENSE_TRTLLM_ALLOW_CUTEDSL_FALLBACK", "1")
    processed: list[torch.nn.Module] = []

    class Fallback:
        def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
            processed.append(layer)

        def apply_weights(
            self,
            layer: torch.nn.Module,
            x: torch.Tensor,
            bias: torch.Tensor | None = None,
        ) -> torch.Tensor:
            return x + 1

    kernel = object.__new__(FlashInferTrtllmMxfp8LinearKernel)
    kernel._cutedsl_fallback = Fallback()
    monkeypatch.setattr(
        flashinfer_module.FlashInferCutedslMxfp8LinearKernel,
        "is_supported",
        lambda: (True, None),
    )
    layer = torch.nn.Module()
    layer.weight = Parameter(
        torch.empty((128, 1344), dtype=torch.float8_e4m3fn), requires_grad=False
    )

    kernel.process_weights_after_loading(layer)
    output = kernel.apply_weights(
        layer,
        torch.zeros((2, 1344), dtype=torch.bfloat16),
    )

    assert processed == [layer]
    assert layer._mxfp8_dense_backend == "cute-dsl"
    assert torch.equal(output, torch.ones_like(output))


def test_mxfp8_trtllm_unsupported_k_rejects_unavailable_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_MXFP8_DENSE_TRTLLM_ALLOW_CUTEDSL_FALLBACK", "1")
    monkeypatch.setattr(
        flashinfer_module.FlashInferCutedslMxfp8LinearKernel,
        "is_supported",
        lambda: (False, "missing test backend"),
    )
    kernel = object.__new__(FlashInferTrtllmMxfp8LinearKernel)
    kernel._cutedsl_fallback = None
    layer = torch.nn.Module()
    layer.weight = Parameter(
        torch.empty((128, 1344), dtype=torch.float8_e4m3fn), requires_grad=False
    )

    with pytest.raises(ValueError, match="CuTeDSL fallback is unavailable"):
        kernel.process_weights_after_loading(layer)


def test_mxfp8_trtllm_layer_allowlist_parses_exact_nk_pairs() -> None:
    assert flashinfer_module._parse_mxfp8_trtllm_layer_allowlist(
        "5120,5120;8192,4096"
    ) == {(5120, 5120), (8192, 4096)}


def test_mxfp8_trtllm_layer_allowlist_accepts_line_delimited_artifact() -> None:
    assert flashinfer_module._parse_mxfp8_trtllm_layer_allowlist(
        "1280,8192\n2048,8192\n8192,1024\n"
    ) == {(1280, 8192), (2048, 8192), (8192, 1024)}


def test_mxfp8_trtllm_layer_allowlist_rejects_malformed_entry() -> None:
    with pytest.raises(ValueError, match="N,K"):
        flashinfer_module._parse_mxfp8_trtllm_layer_allowlist("5120")


def test_mxfp8_trtllm_layer_allowlist_decodes_base64_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(
        "VLLM_MXFP8_DENSE_TRTLLM_LAYER_ALLOWLIST",
        raising=False,
    )
    monkeypatch.setenv(
        "VLLM_MXFP8_DENSE_TRTLLM_LAYER_ALLOWLIST_B64",
        "NTEyMCw1MTIwOzgyMTIsNDA5Ng==",
    )

    assert flashinfer_module._mxfp8_trtllm_layer_is_qualified(5120, 5120)
    assert not flashinfer_module._mxfp8_trtllm_layer_is_qualified(4096, 4096)


def test_mxfp8_trtllm_layer_allowlist_rejects_two_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "VLLM_MXFP8_DENSE_TRTLLM_LAYER_ALLOWLIST",
        "5120,5120",
    )
    monkeypatch.setenv(
        "VLLM_MXFP8_DENSE_TRTLLM_LAYER_ALLOWLIST_B64",
        "NTEyMCw1MTIw",
    )

    with pytest.raises(ValueError, match="only one"):
        flashinfer_module._mxfp8_trtllm_layer_is_qualified(5120, 5120)


def test_mxfp8_trtllm_unqualified_layer_uses_cutedsl_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "VLLM_MXFP8_DENSE_TRTLLM_LAYER_ALLOWLIST",
        "256,1280",
    )
    processed: list[torch.nn.Module] = []

    class Fallback:
        def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
            processed.append(layer)

    kernel = object.__new__(FlashInferTrtllmMxfp8LinearKernel)
    kernel._cutedsl_fallback = Fallback()
    monkeypatch.setattr(
        flashinfer_module.FlashInferCutedslMxfp8LinearKernel,
        "is_supported",
        lambda: (True, None),
    )
    layer = torch.nn.Module()
    layer.weight = Parameter(
        torch.empty((128, 1280), dtype=torch.float8_e4m3fn), requires_grad=False
    )

    kernel.process_weights_after_loading(layer)

    assert processed == [layer]
    assert layer._mxfp8_dense_backend == "cute-dsl"


def test_mxfp8_shape_trace_policy_is_not_evaluated_during_compile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kernel = object.__new__(FlashInferTrtllmMxfp8LinearKernel)
    layer = torch.nn.Module()
    layer.weight = Parameter(
        torch.empty((128, 256), dtype=torch.float8_e4m3fn), requires_grad=False
    )
    layer.weight_scale = Parameter(
        torch.empty(1024, dtype=torch.uint8), requires_grad=False
    )
    layer._mxfp8_trtllm_output_features = 120

    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    monkeypatch.setattr(
        flashinfer_module,
        "mxfp8_trtllm_use_8x4_sf_layout",
        lambda _: pytest.fail("compile must not evaluate the shape-trace layout"),
    )
    monkeypatch.setattr(
        flashinfer_module,
        "mxfp8_trtllm_adaptive_linear",
        lambda x, *_: torch.empty(
            (x.shape[0], 120), dtype=torch.bfloat16, device=x.device
        ),
    )

    output = kernel.apply_weights(
        layer,
        torch.empty((4, 256), dtype=torch.bfloat16),
    )
    assert output.shape == (4, 120)


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
