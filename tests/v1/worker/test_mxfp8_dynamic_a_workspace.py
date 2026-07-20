# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

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
