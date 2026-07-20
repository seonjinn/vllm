# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest
import torch

import vllm.v1.worker.gpu_model_runner as gpu_model_runner_module
import vllm.v1.worker.workspace as workspace
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.v1.worker.workspace import WorkspaceManager


def _allowlisted_dynamic_a_buffers() -> tuple[tuple[tuple[int, ...], torch.dtype], ...]:
    return (
        ((8, 512), torch.float8_e4m3fn),
        ((8, 16), torch.uint8),
        ((4096,), torch.uint8),
        ((8, 130), torch.bfloat16),
        ((32, 512), torch.float8_e4m3fn),
        ((32, 16), torch.uint8),
        ((8192,), torch.uint8),
        ((32, 130), torch.bfloat16),
    )


def test_pre_capture_reservation_allocates_buffers_per_dbo_slot() -> None:
    manager = WorkspaceManager(torch.device("cpu"), num_ubatches=2)

    reserved = manager.reserve_simultaneous_for_all_ubatches(
        *_allowlisted_dynamic_a_buffers()
    )

    assert [
        [buffer.untyped_storage().data_ptr() for buffer in slot] for slot in reserved
    ] != [
        [buffer.untyped_storage().data_ptr() for buffer in reserved[1]],
        [buffer.untyped_storage().data_ptr() for buffer in reserved[0]],
    ]


def test_locked_workspace_rejects_growth_after_pre_capture_reservation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = WorkspaceManager(torch.device("cpu"), num_ubatches=2)
    manager.reserve_simultaneous_for_all_ubatches(*_allowlisted_dynamic_a_buffers())
    manager.lock()
    monkeypatch.setattr("vllm.v1.worker.workspace.dbo_current_ubatch_id", lambda: 1)

    with pytest.raises(AssertionError, match="Workspace growth is not allowed"):
        manager.get_simultaneous(((64, 512), torch.float8_e4m3fn))


class _DynamicAReservationProbe:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    def reserve_dynamic_a_workspaces(
        self, layer: torch.nn.Module, manager: WorkspaceManager
    ) -> None:
        self.calls.append((tuple(layer.weight.shape), tuple(layer.weight_scale.shape)))
        manager.reserve_simultaneous_for_all_ubatches(*_allowlisted_dynamic_a_buffers())


def test_capture_model_reserves_dynamic_a_before_workspace_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace.reset_workspace_manager()
    workspace.init_workspace_manager(torch.device("cpu"), num_ubatches=2)
    manager = workspace.current_workspace_manager()
    probe = _DynamicAReservationProbe()
    layer = torch.nn.Module()
    layer.weight = torch.empty((512, 130), dtype=torch.float8_e4m3fn)
    layer.weight_scale = torch.empty((4096,), dtype=torch.float8_e8m0fnu)
    layer.quant_method = probe
    model = torch.nn.Module()
    model.add_module("physical_n_tail", layer)
    runner = object.__new__(GPUModelRunner)
    runner.model = model
    runner.compilation_config = SimpleNamespace(cudagraph_mode=object())
    runner.device = torch.device("cpu")
    runner.encoder_cudagraph_manager = None
    runner.cudagraph_dispatcher = SimpleNamespace(get_capture_descs=lambda: [])
    runner._maybe_init_encoder_cudagraph_manager = lambda: None
    runner._freeze_gc = nullcontext
    slot = 0
    lifecycle: list[str] = []
    monkeypatch.setattr(workspace, "dbo_current_ubatch_id", lambda: slot)

    original_reserve = runner._reserve_mxfp8_dynamic_a_workspaces

    def reserve_spy() -> None:
        lifecycle.append("reserve")
        original_reserve()

    def lock_spy() -> None:
        lifecycle.append("lock")
        manager.lock()

    monkeypatch.setattr(runner, "_reserve_mxfp8_dynamic_a_workspaces", reserve_spy)
    monkeypatch.setattr(gpu_model_runner_module, "lock_workspace", lock_spy)

    @contextmanager
    def graph_capture_spy(**_: object):
        lifecycle.append("capture_enter")
        try:
            yield
        finally:
            lifecycle.append("capture_exit")

    monkeypatch.setattr(gpu_model_runner_module, "graph_capture", graph_capture_spy)
    monkeypatch.setattr(
        gpu_model_runner_module.torch,
        "accelerator",
        SimpleNamespace(
            synchronize=lambda: None,
            empty_cache=lambda: None,
            get_memory_info=lambda: (1024, 1024),
        ),
    )

    try:
        runner.capture_model()
        before = [
            current.untyped_storage().data_ptr()
            for current in manager._current_workspaces
            if current is not None
        ]
        for slot in range(2):
            manager.get_simultaneous(*_allowlisted_dynamic_a_buffers())
        after = [
            current.untyped_storage().data_ptr()
            for current in manager._current_workspaces
            if current is not None
        ]
    finally:
        workspace.reset_workspace_manager()

    assert (
        probe.calls,
        lifecycle,
        manager.is_locked(),
        len(before),
        before == after,
    ) == (
        [((512, 130), (4096,))],
        ["reserve", "capture_enter", "capture_exit", "lock"],
        True,
        2,
        True,
    )
