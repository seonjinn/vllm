# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import logging
from types import SimpleNamespace

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.example_hidden_states_connector import (  # noqa: E501
    ExampleHiddenStatesConnector,
)


def _connector() -> ExampleHiddenStatesConnector:
    connector = ExampleHiddenStatesConnector.__new__(ExampleHiddenStatesConnector)
    connector._request_filenames = {}
    connector._pending_saves = {}
    return connector


def _request(req_id: str):
    return SimpleNamespace(
        request_id=req_id,
        kv_transfer_params={},
        prompt_token_ids=[11, 12],
        all_token_ids=[11, 12, 13],
    )


def test_registered_request_finish_creates_pending_save(tmp_path):
    connector = _connector()
    request = _request("registered")
    filename = str(tmp_path / "registered.safetensors")
    connector._request_filenames[request.request_id] = filename

    result = connector.request_finished(request, block_ids=[2, 4])

    assert result == (True, {"hidden_states_path": filename})
    pending = connector._pending_saves[request.request_id]
    assert pending.filename == filename
    assert pending.block_ids == [2, 4]
    assert torch.equal(pending.token_ids, torch.tensor([11, 12]))
    assert request.request_id not in connector._request_filenames


def test_unregistered_request_abort_is_ignored(caplog):
    connector = _connector()
    request = _request("unregistered-abort")

    with caplog.at_level(logging.WARNING):
        result = connector.request_finished(request, block_ids=[])

    assert result == (False, None)
    assert request.request_id in caplog.text
    assert connector._pending_saves == {}


def test_duplicate_finish_returns_existing_pending_path(tmp_path):
    connector = _connector()
    request = _request("duplicate")
    filename = str(tmp_path / "duplicate.safetensors")
    connector._request_filenames[request.request_id] = filename

    first = connector.request_finished(request, block_ids=[3])
    second = connector.request_finished(request, block_ids=[99])

    assert first == second == (True, {"hidden_states_path": filename})
    pending = connector._pending_saves[request.request_id]
    assert pending.block_ids == [3]


def test_finish_after_pending_save_is_drained_is_ignored(tmp_path, caplog):
    connector = _connector()
    request = _request("drained")
    filename = str(tmp_path / "drained.safetensors")
    connector._request_filenames[request.request_id] = filename

    connector.request_finished(request, block_ids=[5])
    metadata = connector.build_connector_meta(SimpleNamespace(scheduled_new_reqs=[]))
    assert metadata.pending_saves[0].filename == filename
    assert connector._pending_saves == {}

    with caplog.at_level(logging.WARNING):
        result = connector.request_finished(request, block_ids=[])

    assert result == (False, None)
    assert request.request_id in caplog.text
