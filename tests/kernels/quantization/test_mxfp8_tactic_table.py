# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from vllm import envs
from vllm.model_executor.layers.quantization.utils import mxfp8_tactic_table
from vllm.model_executor.layers.quantization.utils.mxfp8_tactic_table import (
    Mxfp8TacticKey,
    RuntimeProvenance,
    load_mxfp8_tactic_artifact,
)


def provenance() -> RuntimeProvenance:
    return RuntimeProvenance(
        vllm_version="0.26.0",
        flashinfer_version="0.6.14",
        torch_version="2.11.0+cu130",
        cuda_version="13.0",
        driver_version="580.65.06",
        gpu="NVIDIA GB200",
        topology="tp4",
        checkpoint_id="nvidia/Nemotron3-Ultra-30B-A3B-MXFP8",
        source_commit="0123456789abcdef0123456789abcdef01234567",
        container_digest=(
            "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
        ),
        adaptive_switch_m=256,
        weight_contract="trtllm_shuffled_b_sf_a_n128_v1",
    )


@pytest.fixture(autouse=True)
def matching_local_runtime(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> None:
    if request.node.name.startswith("test_wraps_worker_local_evidence_failures"):
        return
    monkeypatch.setattr(
        mxfp8_tactic_table,
        "_local_runtime_values",
        lambda: {
            "vllm_version": provenance().vllm_version,
            "flashinfer_version": provenance().flashinfer_version,
            "torch_version": provenance().torch_version,
            "cuda_version": provenance().cuda_version,
            "driver_version": provenance().driver_version,
            "gpu": provenance().gpu,
        },
    )


def key(
    *,
    m: int = 32,
    n_logical: int = 4384,
    n_physical: int = 4480,
    k: int = 8192,
    layout: str = "8x4",
) -> Mxfp8TacticKey:
    return Mxfp8TacticKey(
        m_logical=m,
        n_logical=n_logical,
        k_logical=k,
        n_physical=n_physical,
        k_physical=k,
        activation_scale_layout=layout,
        output_dtype="bfloat16",
    )


def entry(
    *,
    m: int = 32,
    n_logical: int = 4384,
    n_physical: int = 4480,
    k: int = 8192,
    layout: str = "8x4",
    tactic: int = 65,
    legal_tactics: list[int] | None = None,
) -> dict[str, Any]:
    return {
        "key": asdict(
            key(
                m=m,
                n_logical=n_logical,
                n_physical=n_physical,
                k=k,
                layout=layout,
            )
        ),
        "selected_tactic": tactic,
        "selection": "qualified",
        "legal_tactics": legal_tactics if legal_tactics is not None else [61, 65, 66],
        "default_runner_median_ms": 1.0,
        "selected_runner_median_ms": 0.9,
        "runner_speedup_vs_default": 1.111111,
        "quant_plus_gemm_speedup_vs_default": 1.03,
        "coefficient_of_variation": 0.01,
        "repeat_count": 3,
        "numerical": {
            "finite": True,
            "default_allclose": True,
            "bf16_cosine_min": 0.99,
            "bf16_max_abs_error": 0.1,
            "bf16_mean_abs_error": 0.01,
        },
        "cuda_graph": {
            "capture_ok": True,
            "replay_ok": True,
            "changed_input_ok": True,
            "output_pointer_stable": True,
        },
    }


def load_fixture(
    tmp_path: Path,
    *,
    entries: list[dict[str, Any]] | None = None,
    metadata: dict[str, Any] | None = None,
    schema_version: object = 1,
) -> Any:
    artifact_path = tmp_path / "tactic-table.json"
    artifact_metadata = {
        **asdict(provenance()),
        "manifest_sha256": "b" * 64,
        "trace_sha256": {"/trace/rank0.jsonl": "a" * 64},
        "repeat_sources": [],
        "job_ids": [],
        "selection_policy": {},
        "generated_at": "2026-07-26T00:00:00Z",
    }
    if metadata is not None:
        artifact_metadata.update(metadata)
    artifact_path.write_text(
        json.dumps(
            {
                "schema_version": schema_version,
                "metadata": artifact_metadata,
                "entries": entries if entries is not None else [entry()],
            },
            indent=2,
        )
    )
    return load_mxfp8_tactic_artifact(
        artifact_path,
        hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
        provenance(),
    )


def test_exact_lookup_distinguishes_layout_and_physical_n(tmp_path: Path) -> None:
    artifact = load_fixture(
        tmp_path,
        entries=[
            entry(
                m=32,
                n_logical=4384,
                n_physical=4480,
                k=8192,
                layout="8x4",
                tactic=65,
            ),
            entry(
                m=32,
                n_logical=4384,
                n_physical=4480,
                k=8192,
                layout="128x4",
                tactic=17,
                legal_tactics=[17],
            ),
        ],
    )

    assert artifact.lookup(key(layout="8x4")) == 65
    assert artifact.lookup(key(layout="128x4")) == 17
    assert artifact.lookup(key(n_physical=4384, layout="8x4")) is None


@pytest.mark.parametrize(
    ("entries", "schema_version", "metadata"),
    [
        ([entry(), entry()], 1, None),
        ([entry(tactic=-2)], 1, None),
        ([entry(layout="invalid")], 1, None),
        ([entry(n_physical=4416)], 1, None),
        ([entry(k=8064)], 1, None),
        ([entry(tactic=64)], 1, None),
        ([entry()], 2, None),
    ],
    ids=[
        "duplicate-key",
        "invalid-tactic",
        "unknown-layout",
        "unaligned-physical-n",
        "unaligned-k",
        "selected-tactic-not-legal",
        "unsupported-schema",
    ],
)
def test_rejects_malformed_artifacts(
    tmp_path: Path,
    entries: list[dict[str, Any]],
    schema_version: int,
    metadata: dict[str, Any] | None,
) -> None:
    with pytest.raises(ValueError):
        load_fixture(
            tmp_path,
            entries=entries,
            schema_version=schema_version,
            metadata=metadata,
        )


@pytest.mark.parametrize("schema_version", [True, 1.0], ids=["bool", "float"])
def test_rejects_non_integer_schema_version(
    tmp_path: Path, schema_version: object
) -> None:
    with pytest.raises(ValueError):
        load_fixture(tmp_path, schema_version=schema_version)


def test_rejects_unhashable_activation_scale_layout(tmp_path: Path) -> None:
    malformed_entry = entry()
    malformed_entry["key"]["activation_scale_layout"] = []

    with pytest.raises(ValueError):
        load_fixture(tmp_path, entries=[malformed_entry])


@pytest.mark.parametrize(
    "updated_expected",
    [
        {"vllm_version": "0.26.1"},
        {"flashinfer_version": "0.6.15"},
        {"torch_version": "2.11.1+cu130"},
        {"cuda_version": "13.1"},
        {"driver_version": "580.66.06"},
        {"gpu": "NVIDIA B200"},
        {"topology": "tp8"},
        {"checkpoint_id": "other/checkpoint"},
        {"source_commit": "f" * 40},
        {"container_digest": "sha256:" + "e" * 64},
        {"adaptive_switch_m": 128},
        {"weight_contract": "other-layout"},
    ],
)
def test_rejects_provenance_mismatch(
    tmp_path: Path, updated_expected: dict[str, Any]
) -> None:
    artifact_path = tmp_path / "tactic-table.json"
    artifact_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "metadata": {
                    **asdict(provenance()),
                    "manifest_sha256": "b" * 64,
                    "trace_sha256": {"/trace/rank0.jsonl": "a" * 64},
                    "repeat_sources": [],
                    "job_ids": [],
                    "selection_policy": {},
                    "generated_at": "2026-07-26T00:00:00Z",
                },
                "entries": [entry()],
            },
            indent=2,
        )
    )

    with pytest.raises(ValueError):
        load_mxfp8_tactic_artifact(
            artifact_path,
            hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
            replace(provenance(), **updated_expected),
        )


def test_rejects_worker_local_runtime_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        mxfp8_tactic_table,
        "_local_runtime_values",
        lambda: {
            "vllm_version": provenance().vllm_version,
            "flashinfer_version": provenance().flashinfer_version,
            "torch_version": "2.11.1+cu130",
            "cuda_version": provenance().cuda_version,
            "driver_version": provenance().driver_version,
            "gpu": provenance().gpu,
        },
    )

    with pytest.raises(ValueError, match="worker-local runtime"):
        load_fixture(tmp_path)


@pytest.mark.parametrize(
    "flashinfer,torch",
    [
        (
            SimpleNamespace(),
            SimpleNamespace(
                __version__="2.11.0+cu130",
                version=SimpleNamespace(cuda="13.0"),
                cuda=SimpleNamespace(
                    is_available=lambda: True,
                    get_device_name=lambda: "NVIDIA GB200",
                ),
            ),
        ),
        (
            SimpleNamespace(__version__="0.6.14"),
            SimpleNamespace(
                version=SimpleNamespace(cuda="13.0"),
                cuda=SimpleNamespace(
                    is_available=lambda: True,
                    get_device_name=lambda: "NVIDIA GB200",
                ),
            ),
        ),
        (
            SimpleNamespace(__version__="0.6.14"),
            SimpleNamespace(
                __version__="2.11.0+cu130",
                version=SimpleNamespace(cuda="13.0"),
                cuda=SimpleNamespace(
                    is_available=lambda: True,
                    get_device_name=lambda: (_ for _ in ()).throw(
                        RuntimeError("device query failed")
                    ),
                ),
            ),
        ),
    ],
    ids=["flashinfer-version", "torch-version", "gpu-name"],
)
def test_wraps_worker_local_evidence_failures(
    monkeypatch: pytest.MonkeyPatch,
    flashinfer: object,
    torch: object,
) -> None:
    monkeypatch.setitem(sys.modules, "flashinfer", flashinfer)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(
        sys.modules,
        "vllm",
        SimpleNamespace(__version__="0.26.0"),
    )
    monkeypatch.setattr(
        mxfp8_tactic_table.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout="580.65.06\\n"),
    )

    with pytest.raises(ValueError, match="Unable to query"):
        mxfp8_tactic_table._local_runtime_values()


def test_rejects_mismatched_artifact_digest(tmp_path: Path) -> None:
    artifact_path = tmp_path / "tactic-table.json"
    artifact_path.write_text("{}")

    with pytest.raises(ValueError):
        load_mxfp8_tactic_artifact(artifact_path, "0" * 64, provenance())


def test_environment_registry_validates_paths_and_digests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    names = {
        "VLLM_MXFP8_DENSE_TRTLLM_TACTIC_TABLE_PATH": "/tmp/tactics.json",
        "VLLM_MXFP8_DENSE_TRTLLM_TACTIC_TABLE_SHA256": "a" * 64,
        "VLLM_MXFP8_DENSE_TRTLLM_RUNTIME_PROVENANCE_PATH": "/tmp/provenance.json",
        "VLLM_MXFP8_DENSE_TRTLLM_RUNTIME_PROVENANCE_SHA256": "b" * 64,
        "VLLM_MXFP8_DENSE_TACTIC_AUDIT_PATH": "/tmp/audit.jsonl",
    }
    for name, value in names.items():
        monkeypatch.setenv(name, value)
        assert getattr(envs, name) == value

    monkeypatch.setenv("VLLM_MXFP8_DENSE_TRTLLM_TACTIC_TABLE_SHA256", "A" * 64)
    with pytest.raises(ValueError):
        _ = envs.VLLM_MXFP8_DENSE_TRTLLM_TACTIC_TABLE_SHA256
