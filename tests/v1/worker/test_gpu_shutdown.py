# SPDX-License-Identifier: Apache-2.0

import sys
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import pytest

import vllm.v1.worker.gpu.shutdown as shutdown_module


def test_normal_gpu_shutdown_finalizes_mxfp8_audit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    finalized = False

    def finalize() -> None:
        nonlocal finalized
        finalized = True

    rotary_embedding = ModuleType("vllm.model_executor.layers.rotary_embedding")
    cast(Any, rotary_embedding)._ROPE_DICT = {}
    workspace = ModuleType("vllm.v1.worker.workspace")
    cast(Any, workspace).reset_workspace_manager = lambda: None
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.layers.rotary_embedding",
        rotary_embedding,
    )
    monkeypatch.setitem(sys.modules, "vllm.v1.worker.workspace", workspace)
    monkeypatch.setattr(
        shutdown_module,
        "finalize_mxfp8_trtllm_tactic_audit",
        finalize,
    )
    config = cast(
        Any,
        SimpleNamespace(
            cache_config=SimpleNamespace(num_gpu_blocks=1),
            compilation_config=SimpleNamespace(static_forward_context={}),
        ),
    )

    shutdown_module.free_before_shutdown(config)

    assert finalized
