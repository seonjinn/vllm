# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import csv
import json
from pathlib import Path

import pytest

from experiments.ultra_mxfp8_all_observed_tactics.build_lookup import build_lookup
from experiments.ultra_mxfp8_all_observed_tactics.merge_shape_traces import (
    merge_traces,
)
from experiments.ultra_mxfp8_all_observed_tactics.shard_shapes import shard_shapes


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


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
            "pid": 10,
            "rank": "0",
        },
        {
            "m": 1001,
            "n": 2304,
            "k": 8192,
            "runner": "CuteRunner",
            "phase": "graph",
            "pid": 20,
            "rank": "1",
        },
    ]
    (trace_dir / "trace.10.jsonl").write_text(json.dumps(rows[0]) + "\n")
    (trace_dir / "trace.20.jsonl").write_text(json.dumps(rows[1]) + "\n")
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
            "process_count": "2",
            "rank_count": "2",
        }
    ]


def test_build_lookup_requires_complete_observed_shape_coverage(
    tmp_path: Path,
) -> None:
    observed = tmp_path / "observed.csv"
    observed.write_text(
        "m,n,k,runner,phases,process_count,rank_count\n"
        "1001,2304,8192,CuteRunner,eager,1,1\n"
        "4004,8192,2560,CuteRunner,eager,1,1\n"
    )
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "backend": "cute-dsl",
                "scale_layout": "128x4",
                "shapes": [
                    {
                        "m": 1001,
                        "n": 2304,
                        "k": 8192,
                        "oracle_tactic": [[128, 32], [1, 1], True, False, 1],
                        "oracle_cosine_similarity": 0.999,
                    }
                ],
            }
        )
    )

    with pytest.raises(ValueError, match="missing oracle rows"):
        build_lookup(observed, [report], tmp_path / "lookup.json")


def test_build_lookup_uses_observed_runner_and_oracle_tactic(tmp_path: Path) -> None:
    observed = tmp_path / "observed.csv"
    observed.write_text(
        "m,n,k,runner,phases,process_count,rank_count\n"
        "1001,2304,8192,CuteRunner,eager,1,1\n"
    )
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "backend": "cute-dsl",
                "scale_layout": "128x4",
                "gpu": "NVIDIA GB200",
                "shapes": [
                    {
                        "m": 1001,
                        "n": 2304,
                        "k": 8192,
                        "oracle_tactic": [[128, 32], [1, 1], True, False, 1],
                        "oracle_cosine_similarity": 0.999,
                    }
                ],
            }
        )
    )
    output = tmp_path / "lookup.json"

    lookup = build_lookup(observed, [report], output)

    assert lookup["entry_count"] == 1
    assert lookup["backend"] == "cute-dsl"
    assert lookup["entries"][0]["runner"] == "CuteRunner"
    assert lookup["entries"][0]["tactic"] == [
        [128, 32],
        [1, 1],
        True,
        False,
        1,
    ]


def test_shard_shapes_keeps_nk_families_together(tmp_path: Path) -> None:
    observed = tmp_path / "observed.csv"
    observed.write_text(
        "m,n,k,runner,phases,process_count,rank_count\n"
        "1,2304,8192,CuteRunner,eager,1,1\n"
        "1001,2304,8192,CuteRunner,eager,1,1\n"
        "2,8192,2560,CuteRunner,eager,1,1\n"
        "4004,8192,2560,CuteRunner,eager,1,1\n"
        "8,1024,8192,CuteRunner,eager,1,1\n"
    )

    outputs = shard_shapes(observed, tmp_path / "shards", shard_count=2)

    assert len(outputs) == 2
    locations: dict[tuple[int, int], set[int]] = {}
    for shard_index, path in enumerate(outputs):
        for row in _read_csv(path):
            locations.setdefault((int(row["n"]), int(row["k"])), set()).add(shard_index)
    assert all(len(shards) == 1 for shards in locations.values())
    assert sum(len(_read_csv(path)) for path in outputs) == 5
