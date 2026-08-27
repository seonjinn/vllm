# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
import sys
import types
from contextlib import AbstractContextManager
from types import SimpleNamespace

import pytest
import torch

import vllm.envs as envs
from vllm.model_executor.warmup import kernel_warmup


@pytest.mark.parametrize(
    ("top_k", "rounds"),
    [(1, 1), (3, 3)],
)
def test_flashinfer_v2_policy_matches_serving_objective(
    monkeypatch: pytest.MonkeyPatch,
    top_k: int,
    rounds: int,
) -> None:
    captured: dict[str, object] = {}

    class FakeMeasurementPolicy:
        def __init__(self, **kwargs: object) -> None:
            captured["policy"] = kwargs

    def fake_autotune_v2(**kwargs: object) -> AbstractContextManager[None]:
        captured["autotune"] = kwargs
        return contextlib.nullcontext()

    fake_module = types.ModuleType("flashinfer.autotune_cache")
    fake_module.MeasurementPolicy = FakeMeasurementPolicy  # type: ignore[attr-defined]
    fake_module.autotune_v2 = fake_autotune_v2  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "flashinfer.autotune_cache", fake_module)
    monkeypatch.setattr(envs, "VLLM_FLASHINFER_AUTOTUNE_USE_V2", True)
    monkeypatch.setattr(envs, "VLLM_FLASHINFER_AUTOTUNE_REFINEMENT_TOP_K", top_k)
    monkeypatch.setattr(envs, "VLLM_FLASHINFER_AUTOTUNE_REFINEMENT_ROUNDS", rounds)

    with kernel_warmup._flashinfer_autotune_context({"skip_ops": {"fp4_gemm"}}):
        pass

    assert captured["policy"] == {
        "execution_mode": "cuda_graph",
        "cold_l2": True,
        "refinement_top_k": top_k,
        "refinement_rounds": rounds,
    }
    autotune_call = captured["autotune"]
    assert isinstance(autotune_call, dict)
    assert autotune_call == {
        "mode": "tune",
        "persistent_cache": True,
        "measurement_policy": autotune_call["measurement_policy"],
        "skip_ops": {"fp4_gemm"},
    }


def test_flashinfer_v2_policy_is_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[bool, dict[str, object]]] = []

    def fake_autotune(
        *, tune_mode: bool, **kwargs: object
    ) -> AbstractContextManager[None]:
        calls.append((tune_mode, kwargs))
        return contextlib.nullcontext()

    monkeypatch.setattr(envs, "VLLM_FLASHINFER_AUTOTUNE_USE_V2", False)
    monkeypatch.setattr(kernel_warmup.fi_utils, "autotune", fake_autotune)

    with kernel_warmup._flashinfer_autotune_context({"skip_ops": {"fp4_gemm"}}):
        pass

    assert calls == [(True, {"skip_ops": {"fp4_gemm"}})]


@pytest.mark.parametrize(
    ("layout", "switch_m", "max_tokens", "expected"),
    [
        ("8x4", 256, 16384, None),
        ("128x4", 256, 16384, None),
        ("adaptive", 256, 16384, 256),
        ("adaptive", 256, 128, None),
    ],
)
def test_flashinfer_adaptive_mxfp8_warmup_m_covers_small_layout(
    monkeypatch: pytest.MonkeyPatch,
    layout: str,
    switch_m: int,
    max_tokens: int,
    expected: int | None,
) -> None:
    monkeypatch.setattr(envs, "VLLM_MXFP8_TRTLLM_LAYOUT", layout)
    monkeypatch.setattr(envs, "VLLM_MXFP8_TRTLLM_SWITCH_M", switch_m)

    assert kernel_warmup._flashinfer_adaptive_mxfp8_warmup_m(max_tokens) == expected


def test_flashinfer_adaptive_mxfp8_layers_support_kernel_holder_names() -> None:
    class FakeKernel:
        pass

    kernel = FakeKernel()
    modules = [
        SimpleNamespace(
            quant_method=SimpleNamespace(kernel=kernel),
            weight=torch.empty((128, 256)),
        ),
        SimpleNamespace(
            scheme=SimpleNamespace(fp8_linear=kernel),
            weight=torch.empty((512, 256)),
        ),
        SimpleNamespace(
            quant_method=SimpleNamespace(kernel=kernel),
            weight=torch.empty((128, 256)),
        ),
    ]
    model = SimpleNamespace(modules=lambda: modules)

    layers = kernel_warmup._linear_kernel_layers(model, FakeKernel)

    assert tuple(layers) == ((128, 256), (512, 256))
    assert all(found_kernel is kernel for _, found_kernel in layers.values())
