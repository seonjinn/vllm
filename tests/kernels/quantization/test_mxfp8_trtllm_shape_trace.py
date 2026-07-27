# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from pathlib import Path

import pytest
import torch

from vllm.model_executor.kernels.linear.mxfp8 import flashinfer as flashinfer_kernel


@pytest.fixture(autouse=True)
def reset_shape_trace_state(monkeypatch: pytest.MonkeyPatch) -> None:
    flashinfer_kernel._MXFP8_DENSE_TRACE_SEEN.clear()
    flashinfer_kernel._MXFP8_DENSE_TRACE_WRITTEN = 0
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)


def trace_once(tmp_path: Path) -> None:
    flashinfer_kernel._trace_mxfp8_dense_shape(
        prefix="model.layers.0.mlp.fc1",
        family="FC1",
        m=1000,
        n_logical=8768,
        n_physical=8832,
        k=8192,
        layout="128x4",
    )


def test_shape_trace_is_disabled_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("VLLM_MXFP8_DENSE_SHAPE_TRACE", raising=False)
    monkeypatch.setenv("VLLM_MXFP8_DENSE_SHAPE_TRACE_DIR", str(tmp_path))

    trace_once(tmp_path)

    assert list(tmp_path.iterdir()) == []


def test_shape_trace_writes_exact_high_m_record_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VLLM_MXFP8_DENSE_SHAPE_TRACE", "1")
    monkeypatch.setenv("VLLM_MXFP8_DENSE_SHAPE_TRACE_DIR", str(tmp_path))
    monkeypatch.setenv("VLLM_MXFP8_DENSE_SHAPE_TRACE_MAX", "8")

    trace_once(tmp_path)
    trace_once(tmp_path)

    paths = list(tmp_path.glob("dense_shapes_*.jsonl"))
    assert len(paths) == 1
    records = [json.loads(line) for line in paths[0].read_text().splitlines()]
    assert records == [
        {
            "event": "mxfp8_dense_shape",
            "family": "FC1",
            "hostname": records[0]["hostname"],
            "k": 8192,
            "layout": "128x4",
            "m": 1000,
            "n_logical": 8768,
            "n_physical": 8832,
            "pid": records[0]["pid"],
            "prefix": "model.layers.0.mlp.fc1",
        }
    ]


def test_shape_trace_honors_record_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VLLM_MXFP8_DENSE_SHAPE_TRACE", "1")
    monkeypatch.setenv("VLLM_MXFP8_DENSE_SHAPE_TRACE_DIR", str(tmp_path))
    monkeypatch.setenv("VLLM_MXFP8_DENSE_SHAPE_TRACE_MAX", "1")

    trace_once(tmp_path)
    flashinfer_kernel._trace_mxfp8_dense_shape(
        prefix="model.layers.0.mlp.fc2",
        family="FC2",
        m=2000,
        n_logical=8192,
        n_physical=8192,
        k=4480,
        layout="128x4",
    )

    path = next(tmp_path.glob("dense_shapes_*.jsonl"))
    assert len(path.read_text().splitlines()) == 1


def test_shape_trace_skips_compile_and_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VLLM_MXFP8_DENSE_SHAPE_TRACE", "1")
    monkeypatch.setenv("VLLM_MXFP8_DENSE_SHAPE_TRACE_DIR", str(tmp_path))
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    trace_once(tmp_path)
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    trace_once(tmp_path)

    assert list(tmp_path.iterdir()) == []
