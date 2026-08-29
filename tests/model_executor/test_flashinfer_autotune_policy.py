# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
import sys
import types
from contextlib import AbstractContextManager
from types import SimpleNamespace
from typing import Any

import pytest

import vllm.envs as envs
from vllm.model_executor.kernels.linear.mxfp8.flashinfer import (
    FlashInferTrtllmMxfp8LinearKernel,
)
from vllm.model_executor.warmup import kernel_warmup


class _FakeModel:
    def __init__(self, kernel: object) -> None:
        self._kernel = kernel

    def modules(self) -> list[SimpleNamespace]:
        return [
            SimpleNamespace(
                quant_method=SimpleNamespace(kernel=self._kernel),
            )
        ]


def _fake_runner(kernel: object, max_num_batched_tokens: int = 16_384) -> Any:
    return SimpleNamespace(
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=max_num_batched_tokens,
        ),
        get_model=lambda: _FakeModel(kernel),
    )


def test_adaptive_mxfp8_trtllm_autotune_warms_both_layouts_at_max_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(envs, "VLLM_MXFP8_TRTLLM_LAYOUT", "adaptive")
    monkeypatch.setattr(envs, "VLLM_MXFP8_TRTLLM_SWITCH_M", 256)
    kernel = object.__new__(FlashInferTrtllmMxfp8LinearKernel)

    runs = kernel_warmup._flashinfer_autotune_runs(_fake_runner(kernel))

    assert runs == ((16_384, None), (16_384, True))


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
