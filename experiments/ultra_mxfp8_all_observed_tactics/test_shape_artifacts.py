# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import csv
import json
from pathlib import Path

import pytest

from experiments.ultra_mxfp8_all_observed_tactics.backend_config import (
    resolve_backend,
)
from experiments.ultra_mxfp8_all_observed_tactics.build_lookup import build_lookup
from experiments.ultra_mxfp8_all_observed_tactics.merge_shape_traces import (
    merge_traces,
)
from experiments.ultra_mxfp8_all_observed_tactics.shard_shapes import shard_shapes


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _write_complete_snapshot(directory: Path, pid: int, row: dict) -> None:
    payload = json.dumps(row) + "\n"
    (directory / f"trace.{pid}.jsonl").write_text(payload)
    (directory / f"counts.{pid}.jsonl").write_text(payload)
    (directory / f"counts.{pid}.complete").touch()


@pytest.mark.parametrize(
    ("name", "linear_backend", "oracle_backend", "scale_layout"),
    [
        ("cute-dsl", "flashinfer_cutedsl", "cute-dsl", "128x4"),
        ("cutlass", "flashinfer_cutlass", "cutlass", "128x4"),
        ("trtllm-128x4", "flashinfer_trtllm", "trtllm", "128x4"),
        ("trtllm-8x4", "flashinfer_trtllm", "trtllm", "8x4"),
    ],
)
def test_resolve_backend_returns_matching_serving_and_oracle_config(
    name: str,
    linear_backend: str,
    oracle_backend: str,
    scale_layout: str,
) -> None:
    config = resolve_backend(name)

    assert config.linear_backend == linear_backend
    assert config.oracle_backend == oracle_backend
    assert config.scale_layout == scale_layout


def test_resolve_backend_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError, match="unsupported backend"):
        resolve_backend("unknown")


def test_merge_traces_deduplicates_workers_and_preserves_phases(
    tmp_path: Path,
) -> None:
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    rows = [
        {
            "m": 1001,
            "n": 2304,
            "k": 8192,
            "runner": "CuteRunner",
            "phase": "eager",
            "tactic": 7,
            "selection_source": "default_autotuner",
            "invocation_count": 3,
            "pid": 10,
            "rank": "0",
        },
        {
            "m": 1001,
            "n": 2304,
            "k": 8192,
            "runner": "CuteRunner",
            "phase": "graph",
            "tactic": 7,
            "selection_source": "default_autotuner",
            "invocation_count": 5,
            "pid": 20,
            "rank": "1",
        },
    ]
    _write_complete_snapshot(trace_dir, 10, rows[0])
    _write_complete_snapshot(trace_dir, 20, rows[1])
    output_csv = tmp_path / "observed.csv"
    summary_path = tmp_path / "summary.json"

    summary = merge_traces(trace_dir, output_csv, summary_path)

    assert summary["shape_count"] == 1
    assert summary["irregular_m_count"] == 1
    assert _read_csv(output_csv) == [
        {
            "m": "1001",
            "n": "2304",
            "k": "8192",
            "runner": "CuteRunner",
            "phases": "eager,graph",
            "selected_phase": "graph",
            "selected_tactic": "7",
            "selection_call_count": "5",
            "process_count": "2",
            "rank_count": "2",
        }
    ]


def test_merge_traces_rejects_conflicting_tactics_in_selected_phase(
    tmp_path: Path,
) -> None:
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    for pid, tactic in ((10, 7), (20, 9)):
        row = {
            "m": 1001,
            "n": 2304,
            "k": 8192,
            "runner": "CuteRunner",
            "phase": "baseline",
            "tactic": tactic,
            "selection_source": "default_autotuner",
            "invocation_count": 1,
            "pid": pid,
            "rank": str(pid),
        }
        _write_complete_snapshot(trace_dir, pid, row)

    with pytest.raises(ValueError, match="conflicting serving tactics"):
        merge_traces(trace_dir, tmp_path / "observed.csv", tmp_path / "summary.json")


def test_merge_traces_uses_final_tactic_after_startup_fallback(
    tmp_path: Path,
) -> None:
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    for rank in range(2):
        pid = rank + 10
        rows = [
            {
                "m": 2,
                "n": 2048,
                "k": 8192,
                "runner": "CutlassRunner",
                "phase": "baseline",
                "tactic": -1,
                "selection_source": "default_autotuner",
                "invocation_count": 3,
                "first_invocation_index": 1,
                "last_invocation_index": 3,
                "pid": pid,
                "rank": str(rank),
            },
            {
                "m": 2,
                "n": 2048,
                "k": 8192,
                "runner": "CutlassRunner",
                "phase": "baseline",
                "tactic": 5,
                "selection_source": "default_autotuner",
                "invocation_count": 7,
                "first_invocation_index": 4,
                "last_invocation_index": 10,
                "pid": pid,
                "rank": str(rank),
            },
        ]
        payload = "".join(json.dumps(row) + "\n" for row in rows)
        (trace_dir / f"trace.{pid}.jsonl").write_text(payload)
        (trace_dir / f"counts.{pid}.jsonl").write_text(payload)
        (trace_dir / f"counts.{pid}.complete").touch()

    output = tmp_path / "observed.csv"
    merge_traces(
        trace_dir,
        output,
        tmp_path / "summary.json",
        expected_rank_count=2,
    )

    assert _read_csv(output)[0]["selected_tactic"] == "5"
    assert _read_csv(output)[0]["selection_call_count"] == "14"


def test_merge_traces_rejects_incomplete_expected_rank_coverage(
    tmp_path: Path,
) -> None:
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    for rank in range(3):
        row = {
            "m": 1001,
            "n": 2304,
            "k": 8192,
            "runner": "CuteRunner",
            "phase": "baseline",
            "tactic": 7,
            "selection_source": "default_autotuner",
            "invocation_count": 1,
            "pid": rank + 10,
            "rank": str(rank),
        }
        _write_complete_snapshot(trace_dir, rank + 10, row)

    with pytest.raises(ValueError, match="incomplete rank coverage.*baseline"):
        merge_traces(
            trace_dir,
            tmp_path / "observed.csv",
            tmp_path / "summary.json",
            expected_rank_count=4,
        )


def test_merge_traces_prefers_compact_count_snapshot_for_same_process(
    tmp_path: Path,
) -> None:
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    row = {
        "m": 1001,
        "n": 2304,
        "k": 8192,
        "runner": "CuteRunner",
        "phase": "baseline",
        "tactic": 7,
        "selection_source": "default_autotuner",
        "pid": 10,
        "rank": "0",
        "invocation_count": 1,
    }
    (trace_dir / "trace.10.jsonl").write_text(json.dumps(row) + "\n")
    (trace_dir / "counts.10.jsonl").write_text(
        json.dumps({**row, "invocation_count": 99}) + "\n"
    )
    (trace_dir / "counts.10.complete").touch()

    output_csv = tmp_path / "observed.csv"
    summary = merge_traces(trace_dir, output_csv, tmp_path / "summary.json")

    assert summary["trace_file_count"] == 1
    assert _read_csv(output_csv)[0]["selection_call_count"] == "99"


def test_merge_traces_keeps_same_pid_from_different_capture_runs(
    tmp_path: Path,
) -> None:
    trace_dir = tmp_path / "serving"
    for run_name, phase, count in (
        ("capture-graph", "graph", 5),
        ("capture-eager", "eager", 3),
    ):
        run_traces = trace_dir / run_name / "1" / "traces"
        run_traces.mkdir(parents=True)
        row = {
            "m": 1001,
            "n": 2304,
            "k": 8192,
            "runner": "CuteRunner",
            "phase": phase,
            "tactic": 7,
            "selection_source": "default_autotuner",
            "pid": 10,
            "rank": "0",
            "invocation_count": count,
        }
        (run_traces / "counts.10.jsonl").write_text(json.dumps(row) + "\n")
        (run_traces / "counts.10.complete").touch()

    output_csv = tmp_path / "observed.csv"
    summary = merge_traces(trace_dir, output_csv, tmp_path / "summary.json")

    assert summary["trace_file_count"] == 2
    assert _read_csv(output_csv)[0]["phases"] == "eager,graph"
    assert _read_csv(output_csv)[0]["selection_call_count"] == "5"


def test_merge_traces_rejects_unfinished_count_snapshot(tmp_path: Path) -> None:
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    row = {
        "m": 1001,
        "n": 2304,
        "k": 8192,
        "runner": "CuteRunner",
        "phase": "baseline",
        "tactic": 7,
        "selection_source": "default_autotuner",
        "pid": 10,
        "rank": "0",
        "invocation_count": 99,
    }
    (trace_dir / "trace.10.jsonl").write_text(json.dumps(row) + "\n")
    (trace_dir / "counts.10.jsonl").write_text(json.dumps(row) + "\n")

    with pytest.raises(ValueError, match="unfinished count snapshot"):
        merge_traces(trace_dir, tmp_path / "observed.csv", tmp_path / "summary.json")


def test_build_lookup_requires_complete_observed_shape_coverage(
    tmp_path: Path,
) -> None:
    observed = tmp_path / "observed.csv"
    observed.write_text(
        "m,n,k,runner,selected_tactic\n"
        "1001,2304,8192,CuteRunner,7\n"
        "4004,8192,2560,CuteRunner,7\n"
    )
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "backend": "cute-dsl",
                "scale_layout": "128x4",
                "flashinfer_commit": "def456",
                "flashinfer_version": "0.6.18",
                "flashinfer_file": "/installed/flashinfer/__init__.py",
                "container_sha256": "container-sha",
                "gpu": "NVIDIA GB200",
                "shapes": [
                    {
                        "m": 1001,
                        "n": 2304,
                        "k": 8192,
                        "runner": "CuteRunner",
                        "selected_tactic": 7,
                        "oracle_tactic": [[128, 32], [1, 1], True, False, 1],
                        "oracle_cosine_similarity": 0.999,
                        "selected_ms": 1.2,
                        "oracle_ms": 1.0,
                        "speedup": 1.2,
                    }
                ],
            }
        )
    )

    with pytest.raises(ValueError, match="missing oracle rows"):
        build_lookup(observed, [report], tmp_path / "lookup.json")


def test_build_lookup_uses_observed_runner_and_oracle_tactic(tmp_path: Path) -> None:
    observed = tmp_path / "observed.csv"
    observed.write_text("m,n,k,runner,selected_tactic\n1001,2304,8192,CuteRunner,7\n")
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "backend": "cute-dsl",
                "scale_layout": "128x4",
                "flashinfer_commit": "def456",
                "flashinfer_version": "0.6.18",
                "flashinfer_file": "/installed/flashinfer/__init__.py",
                "container_sha256": "container-sha",
                "gpu": "NVIDIA GB200",
                "shapes": [
                    {
                        "m": 1001,
                        "n": 2304,
                        "k": 8192,
                        "runner": "CuteRunner",
                        "selected_tactic": 7,
                        "oracle_tactic": [[128, 32], [1, 1], True, False, 1],
                        "oracle_cosine_similarity": 0.999,
                        "oracle_finite": True,
                        "oracle_matches_selected": True,
                        "selected_ms": 1.2,
                        "oracle_ms": 1.0,
                        "speedup": 1.2,
                    }
                ],
            }
        )
    )
    output = tmp_path / "lookup.json"

    lookup = build_lookup(observed, [report], output)

    assert lookup["entry_count"] == 1
    assert lookup["backend"] == "cute-dsl"
    assert lookup["flashinfer_commit"] == "def456"
    assert lookup["flashinfer_version"] == "0.6.18"
    assert lookup["container_sha256"] == "container-sha"
    assert lookup["entries"][0]["runner"] == "CuteRunner"
    assert lookup["entries"][0]["tactic"] == [
        [128, 32],
        [1, 1],
        True,
        False,
        1,
    ]


def test_build_lookup_requires_explicit_correctness_fields(tmp_path: Path) -> None:
    observed = tmp_path / "observed.csv"
    observed.write_text("m,n,k,runner,selected_tactic\n1001,2304,8192,CuteRunner,7\n")
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "backend": "cute-dsl",
                "scale_layout": "128x4",
                "flashinfer_commit": "def456",
                "flashinfer_version": "0.6.18",
                "flashinfer_file": "/installed/flashinfer/__init__.py",
                "container_sha256": "container-sha",
                "gpu": "NVIDIA GB200",
                "shapes": [
                    {
                        "m": 1001,
                        "n": 2304,
                        "k": 8192,
                        "runner": "CuteRunner",
                        "selected_tactic": 7,
                        "oracle_tactic": 7,
                        "oracle_cosine_similarity": 0.999,
                        "selected_ms": 1.0,
                        "oracle_ms": 1.0,
                        "speedup": 1.0,
                    }
                ],
            }
        )
    )

    with pytest.raises(ValueError, match="elementwise parity"):
        build_lookup(observed, [report], tmp_path / "lookup.json")


def test_build_lookup_rejects_oracle_for_different_serving_tactic(
    tmp_path: Path,
) -> None:
    observed = tmp_path / "observed.csv"
    observed.write_text("m,n,k,runner,selected_tactic\n1001,2304,8192,CuteRunner,7\n")
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "backend": "cute-dsl",
                "scale_layout": "128x4",
                "flashinfer_commit": "def456",
                "flashinfer_version": "0.6.18",
                "flashinfer_file": "/installed/flashinfer/__init__.py",
                "container_sha256": "container-sha",
                "gpu": "NVIDIA GB200",
                "shapes": [
                    {
                        "m": 1001,
                        "n": 2304,
                        "k": 8192,
                        "runner": "CuteRunner",
                        "selected_tactic": 9,
                        "oracle_tactic": 7,
                        "oracle_cosine_similarity": 0.999,
                        "oracle_finite": True,
                        "oracle_matches_selected": True,
                        "selected_ms": 1.2,
                        "oracle_ms": 1.0,
                        "speedup": 1.2,
                    }
                ],
            }
        )
    )

    with pytest.raises(ValueError, match="observed serving tactic"):
        build_lookup(observed, [report], tmp_path / "lookup.json")


def test_shard_shapes_keeps_nk_families_together(tmp_path: Path) -> None:
    observed = tmp_path / "observed.csv"
    observed.write_text(
        "m,n,k,runner,phases,selected_phase,selected_tactic,selection_call_count,process_count,rank_count\n"
        "1,2304,8192,CuteRunner,eager,eager,7,10,1,1\n"
        "1001,2304,8192,CuteRunner,eager,eager,7,10,1,1\n"
        "2,8192,2560,CuteRunner,eager,eager,7,10,1,1\n"
        "4004,8192,2560,CuteRunner,eager,eager,7,10,1,1\n"
        "8,1024,8192,CuteRunner,eager,eager,7,10,1,1\n"
    )

    outputs = shard_shapes(observed, tmp_path / "shards", shard_count=2)

    assert len(outputs) == 2
    locations: dict[tuple[int, int], set[int]] = {}
    for shard_index, path in enumerate(outputs):
        for row in _read_csv(path):
            locations.setdefault((int(row["n"]), int(row["k"])), set()).add(shard_index)
    assert all(len(shards) == 1 for shards in locations.values())
    assert sum(len(_read_csv(path)) for path in outputs) == 5
    assert all("selected_tactic" in row for path in outputs for row in _read_csv(path))
