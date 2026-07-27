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
    flashinfer_kernel._MXFP8_DENSE_TRACE_WARNED = False
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)


def trace_once() -> None:
    flashinfer_kernel._trace_mxfp8_dense_shape(
        prefix="model.layers.0.mlp.fc1",
        family="FC1",
        m_logical=1000,
        m_physical=1000,
        n_logical=8768,
        n_physical=8832,
        k_logical=8192,
        k_physical=8192,
        layout="128x4",
        output_dtype="bfloat16",
        tactic_source="exact_table",
        selected_tactic=17,
        runtime_provenance={
            "vllm_version": "0.26.0",
            "flashinfer_version": "0.6.14",
            "torch_version": "2.11.0+cu130",
            "cuda_version": "13.0",
            "driver_version": "580.65.06",
            "gpu": "NVIDIA GB200",
            "topology": "tp4",
        },
    )


def test_shape_trace_is_disabled_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("VLLM_MXFP8_DENSE_SHAPE_TRACE", raising=False)
    monkeypatch.setenv("VLLM_MXFP8_DENSE_SHAPE_TRACE_DIR", str(tmp_path))

    trace_once()

    assert list(tmp_path.iterdir()) == []


def test_shape_trace_writes_exact_high_m_record_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VLLM_MXFP8_DENSE_SHAPE_TRACE", "1")
    monkeypatch.setenv("VLLM_MXFP8_DENSE_SHAPE_TRACE_DIR", str(tmp_path))
    monkeypatch.setenv("VLLM_MXFP8_DENSE_SHAPE_TRACE_MAX", "8")
    monkeypatch.setenv("VLLM_MXFP8_DENSE_TRACE_WORKLOAD", "1024/10240")
    monkeypatch.setenv("VLLM_MXFP8_DENSE_TRACE_BATCH_SIZE", "32")
    monkeypatch.setenv("VLLM_MXFP8_DENSE_TRACE_SERVING_PHASE", "decode")
    monkeypatch.setenv("RANK", "2")

    trace_once()
    trace_once()

    paths = list(tmp_path.glob("dense_shapes_*.jsonl"))
    assert len(paths) == 1
    records = [json.loads(line) for line in paths[0].read_text().splitlines()]
    assert records == [
        {
            "activation_scale_layout": "128x4",
            "batch_size": 32,
            "compilation_state": "eager",
            "cuda_graph_state": "eager",
            "event": "mxfp8_dense_shape",
            "family": "FC1",
            "host": records[0]["host"],
            "hostname": records[0]["host"],
            "k": 8192,
            "k_logical": 8192,
            "k_physical": 8192,
            "layer_prefix": "model.layers.0.mlp.fc1",
            "layout": "128x4",
            "m": 1000,
            "m_logical": 1000,
            "m_physical": 1000,
            "n_logical": 8768,
            "n_physical": 8832,
            "normalized_family": "FC1",
            "output_dtype": "bfloat16",
            "pid": records[0]["pid"],
            "prefix": "model.layers.0.mlp.fc1",
            "rank": 2,
            "runtime_provenance": {
                "cuda_version": "13.0",
                "driver_version": "580.65.06",
                "flashinfer_version": "0.6.14",
                "gpu": "NVIDIA GB200",
                "topology": "tp4",
                "torch_version": "2.11.0+cu130",
                "vllm_version": "0.26.0",
            },
            "selected_tactic": 17,
            "serving_phase": "decode",
            "tactic_source": "exact_table",
            "topology": "tp4",
            "workload": "1024/10240",
        }
    ]


def test_shape_trace_honors_record_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VLLM_MXFP8_DENSE_SHAPE_TRACE", "1")
    monkeypatch.setenv("VLLM_MXFP8_DENSE_SHAPE_TRACE_DIR", str(tmp_path))
    monkeypatch.setenv("VLLM_MXFP8_DENSE_SHAPE_TRACE_MAX", "1")
    monkeypatch.setenv("VLLM_MXFP8_DENSE_TRACE_WORKLOAD", "1024/10240")
    monkeypatch.setenv("VLLM_MXFP8_DENSE_TRACE_BATCH_SIZE", "32")
    monkeypatch.setenv("VLLM_MXFP8_DENSE_TRACE_SERVING_PHASE", "decode")

    trace_once()
    flashinfer_kernel._trace_mxfp8_dense_shape(
        prefix="model.layers.0.mlp.fc2",
        family="FC2",
        m_logical=2000,
        m_physical=2000,
        n_logical=8192,
        n_physical=8192,
        k_logical=4096,
        k_physical=4096,
        layout="128x4",
        output_dtype="bfloat16",
        tactic_source="exact_miss",
        selected_tactic=-1,
        runtime_provenance={"topology": "tp4"},
    )

    path = next(tmp_path.glob("dense_shapes_*.jsonl"))
    assert len(path.read_text().splitlines()) == 1


def test_shape_trace_skips_compile_and_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VLLM_MXFP8_DENSE_SHAPE_TRACE", "1")
    monkeypatch.setenv("VLLM_MXFP8_DENSE_SHAPE_TRACE_DIR", str(tmp_path))
    monkeypatch.setenv("VLLM_MXFP8_DENSE_TRACE_WORKLOAD", "1024/10240")
    monkeypatch.setenv("VLLM_MXFP8_DENSE_TRACE_BATCH_SIZE", "32")
    monkeypatch.setenv("VLLM_MXFP8_DENSE_TRACE_SERVING_PHASE", "decode")
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    trace_once()
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    trace_once()

    assert list(tmp_path.iterdir()) == []
