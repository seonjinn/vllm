# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import contextmanager
from types import ModuleType, SimpleNamespace
from typing import Any

import vllm.envs as envs
from vllm.model_executor.warmup import kernel_warmup


def test_flashinfer_autotune_context_uses_v1_by_default(monkeypatch, tmp_path):
    calls = []

    @contextmanager
    def autotune(**kwargs):
        calls.append(kwargs)
        yield

    monkeypatch.setattr(envs, "VLLM_FLASHINFER_AUTOTUNE_USE_V2", False)
    fi_utils = SimpleNamespace(autotune=autotune)

    with kernel_warmup._flashinfer_autotune_context(
        fi_utils,
        tmp_path / "cache.json",
        {"skip_ops": {"fp4_gemm"}},
    ):
        pass

    assert calls == [{"tune_mode": True, "skip_ops": {"fp4_gemm"}}]


def test_flashinfer_autotune_context_uses_production_v2_policy(monkeypatch, tmp_path):
    calls = []

    class MeasurementPolicy:
        def __init__(self, **kwargs):
            calls.append(("policy", kwargs))

    @contextmanager
    def autotune_v2(**kwargs):
        calls.append(("autotune_v2", kwargs))
        yield

    module: Any = ModuleType("flashinfer.autotune_cache")
    module.MeasurementPolicy = MeasurementPolicy
    module.autotune_v2 = autotune_v2
    monkeypatch.setitem(__import__("sys").modules, module.__name__, module)
    monkeypatch.setattr(envs, "VLLM_FLASHINFER_AUTOTUNE_USE_V2", True)

    cache_path = tmp_path / "model" / "cache.json"
    with kernel_warmup._flashinfer_autotune_context(
        SimpleNamespace(),
        cache_path,
        {"skip_ops": {"fp4_gemm"}},
    ):
        pass

    assert calls[0] == (
        "policy",
        {"execution_mode": "cuda_graph", "cold_l2": True},
    )
    assert calls[1][0] == "autotune_v2"
    assert calls[1][1]["mode"] == "tune"
    assert calls[1][1]["persistent_cache"] is True
    assert calls[1][1]["cache_root"] == cache_path.parent
    assert calls[1][1]["skip_ops"] == {"fp4_gemm"}


def test_flashinfer_autotune_v2_reload_is_synchronized(monkeypatch):
    events = []

    def autotune_v2_reload():
        events.append("reload")

    module: Any = ModuleType("flashinfer.autotune_cache")
    module.autotune_v2_reload = autotune_v2_reload
    monkeypatch.setitem(__import__("sys").modules, module.__name__, module)

    class World:
        world_size = 4

        def barrier(self):
            events.append("barrier")

    kernel_warmup._reload_flashinfer_autotune_v2(World())

    assert events == ["barrier", "reload", "barrier"]
