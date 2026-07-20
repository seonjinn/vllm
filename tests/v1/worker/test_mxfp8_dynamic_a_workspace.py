# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest
import torch

import vllm.v1.worker.gpu.model_runner as gpu_model_runner_v2_module
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


def test_owner_reservations_are_disjoint_and_stable() -> None:
    manager = WorkspaceManager(torch.device("cpu"), num_ubatches=2)

    first = manager.reserve_owner_for_all_ubatches(
        "layer.0", *_allowlisted_dynamic_a_buffers()
    )
    second = manager.reserve_owner_for_all_ubatches(
        "layer.1", *_allowlisted_dynamic_a_buffers()
    )
    first_again = manager.reserve_owner_for_all_ubatches(
        "layer.0", *_allowlisted_dynamic_a_buffers()
    )

    first_storages = {
        buffer.untyped_storage().data_ptr() for slot in first for buffer in slot
    }
    second_storages = {
        buffer.untyped_storage().data_ptr() for slot in second for buffer in slot
    }
    assert first_storages.isdisjoint(second_storages)
    assert [buffer.data_ptr() for slot in first for buffer in slot] == [
        buffer.data_ptr() for slot in first_again for buffer in slot
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


def test_v1_reservation_traverses_separately_held_drafter_model() -> None:
    workspace.reset_workspace_manager()
    workspace.init_workspace_manager(torch.device("cpu"), num_ubatches=1)
    probe = _DynamicAReservationProbe()

    def make_model(n: int) -> torch.nn.Module:
        model = torch.nn.Module()
        layer = torch.nn.Module()
        layer.weight = torch.empty((512, n), dtype=torch.float8_e4m3fn)
        layer.weight_scale = torch.empty((4096,), dtype=torch.uint8)
        layer.quant_method = probe
        model.add_module("linear", layer)
        return model

    runner = object.__new__(GPUModelRunner)
    runner.model = make_model(130)
    runner.drafter = SimpleNamespace(model=make_model(256))
    try:
        runner._reserve_mxfp8_dynamic_a_workspaces()
    finally:
        workspace.reset_workspace_manager()

    assert probe.calls == [((512, 130), (4096,)), ((512, 256), (4096,))]


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


def test_v2_capture_reserves_model_and_speculator_before_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace.reset_workspace_manager()
    workspace.init_workspace_manager(torch.device("cpu"), num_ubatches=1)
    manager = workspace.current_workspace_manager()
    calls: list[str] = []

    def make_model(name: str) -> torch.nn.Module:
        model = torch.nn.Module()
        layer = torch.nn.Module()
        layer.weight = torch.empty((512, 130), dtype=torch.float8_e4m3fn)
        layer.quant_method = SimpleNamespace(
            reserve_dynamic_a_workspaces=lambda *_: calls.append(name)
        )
        model.add_module("linear", layer)
        return model

    runner = object.__new__(gpu_model_runner_v2_module.GPUModelRunner)
    runner.model = make_model("model")
    runner.speculator = SimpleNamespace(
        model=make_model("speculator"), capture=lambda _: None
    )
    runner.cudagraph_manager = SimpleNamespace(
        needs_capture=lambda: True,
        capture=lambda *_args, **_kwargs: None,
    )
    runner.model_state = object()
    runner.input_buffers = object()
    runner.intermediate_tensors = object()
    runner.block_tables = object()
    runner.attn_groups = object()
    runner.kv_cache_config = object()
    runner.lora_config = None
    runner.use_aux_hidden_state_outputs = False
    runner.device = torch.device("cpu")
    runner.maybe_setup_dummy_loras = lambda _: nullcontext()

    monkeypatch.setattr(gpu_model_runner_v2_module.gc, "collect", lambda: None)
    monkeypatch.setattr(
        gpu_model_runner_v2_module.torch,
        "accelerator",
        SimpleNamespace(
            empty_cache=lambda: None,
            get_memory_info=lambda: (1024, 1024),
        ),
    )
    monkeypatch.setattr(
        gpu_model_runner_v2_module,
        "create_lora_capture_hook",
        lambda *_: None,
    )

    try:
        runner.capture_model()
    finally:
        workspace.reset_workspace_manager()

    assert calls == ["model", "speculator"]
    assert manager.is_locked()
