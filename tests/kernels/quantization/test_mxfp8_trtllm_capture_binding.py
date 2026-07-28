# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.kernels.linear.mxfp8 import flashinfer


def _layer(m: int, binding: tuple[int, str] | None) -> SimpleNamespace:
    layer = SimpleNamespace(
        weight=torch.empty((8, 4), dtype=torch.bfloat16),
        weight_scale=torch.empty(1, dtype=torch.uint8),
        _mxfp8_trtllm_output_features=7,
        prefix="model.layers.0.mlp.fc1",
    )
    if binding is not None:
        layer._mxfp8_trtllm_capture_bindings = {m: binding}
    return layer


def test_capture_uses_layer_prewarmed_tactic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    monkeypatch.setattr(
        flashinfer,
        "mxfp8_trtllm_specialization_fingerprint",
        lambda _device: "worker-fingerprint",
    )
    monkeypatch.setattr(
        flashinfer,
        "mxfp8_trtllm_adaptive_linear",
        lambda *args: (
            calls.append(args)
            or torch.empty((args[0].shape[0], args[3]), dtype=torch.bfloat16)
        ),
    )
    kernel = object.__new__(flashinfer.FlashInferTrtllmMxfp8LinearKernel)

    kernel.apply_weights(
        _layer(2, (65, "exact_table")),
        torch.empty((2, 4), dtype=torch.bfloat16),
    )

    assert calls[0][6:9] == (65, "exact_table", "worker-fingerprint")


def test_capture_unseen_m_uses_default_tactic_without_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[object, ...]] = []
    layer = _layer(2, None)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    monkeypatch.setattr(
        flashinfer,
        "mxfp8_trtllm_specialization_fingerprint",
        lambda _device: "worker-fingerprint",
    )
    monkeypatch.setattr(
        flashinfer,
        "mxfp8_trtllm_adaptive_linear",
        lambda *args: (
            calls.append(args)
            or torch.empty((args[0].shape[0], args[3]), dtype=torch.bfloat16)
        ),
    )
    kernel = object.__new__(flashinfer.FlashInferTrtllmMxfp8LinearKernel)

    kernel.apply_weights(
        layer,
        torch.empty((2, 4), dtype=torch.bfloat16),
    )

    assert calls[0][6:9] == (-1, "capture_exact_miss", "worker-fingerprint")
    assert not hasattr(layer, "_mxfp8_trtllm_capture_bindings")


def test_capture_unseen_m_fails_when_exact_tactic_is_required(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layer = _layer(2, None)
    monkeypatch.setenv("VLLM_MXFP8_DENSE_REQUIRE_EXACT_TACTIC", "1")
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    kernel = object.__new__(flashinfer.FlashInferTrtllmMxfp8LinearKernel)

    with pytest.raises(RuntimeError, match="exact tactic is required"):
        kernel.apply_weights(
            layer,
            torch.empty((2, 4), dtype=torch.bfloat16),
        )


def test_eager_prewarm_caches_resolved_tactic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layer = _layer(2, None)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: False)
    monkeypatch.setattr(
        flashinfer,
        "mxfp8_trtllm_adaptive_linear",
        lambda *args: torch.empty(
            (args[0].shape[0], args[3]),
            dtype=torch.bfloat16,
        ),
    )
    monkeypatch.setattr(
        flashinfer,
        "mxfp8_trtllm_resolved_binding",
        lambda *_args, **_kwargs: (65, "exact_table"),
    )
    kernel = object.__new__(flashinfer.FlashInferTrtllmMxfp8LinearKernel)

    kernel.apply_weights(
        layer,
        torch.empty((2, 4), dtype=torch.bfloat16),
    )

    assert layer._mxfp8_trtllm_capture_bindings == {2: (65, "exact_table")}
