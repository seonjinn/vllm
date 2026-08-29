# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.ultra_mxfp8_tp8_serving_crossover.summarize_e2e import (
    collect_rows,
    compare_to_baseline,
)


def write_result(
    root: Path,
    arm: str,
    repetition: int,
    concurrency: int,
    throughput: float,
    *,
    completed: int | None = None,
    failed: int = 0,
) -> None:
    expected_requests = concurrency * 10
    run_dir = root / "runs" / arm / f"rep{repetition}" / "stamp" / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / f"raw_bench_serve_bs{concurrency}.json").write_text(
        json.dumps(
            {
                "completed": expected_requests if completed is None else completed,
                "failed": failed,
                "duration": 10.0,
                "total_input_tokens": expected_requests * 1000,
                "total_output_tokens": expected_requests * 10000,
                "output_throughput": throughput,
                "total_token_throughput": throughput * 1.1,
                "request_throughput": throughput / 10000,
                "mean_ttft_ms": 12.0,
                "median_ttft_ms": 11.0,
                "p99_ttft_ms": 15.0,
                "mean_tpot_ms": 2.0,
                "median_tpot_ms": 1.9,
                "p99_tpot_ms": 2.4,
            }
        )
    )


def test_collects_only_token_valid_rows_and_normalizes_per_gpu(
    tmp_path: Path,
) -> None:
    write_result(tmp_path, "cutedsl", 0, 2, 800.0)
    write_result(tmp_path, "adaptive", 0, 2, 880.0)
    write_result(tmp_path, "broken", 0, 2, 1.0, completed=0, failed=20)

    rows = collect_rows(tmp_path, waves=10, isl=1000, osl=10000, gpus=8)

    assert [row["arm"] for row in rows] == ["adaptive", "cutedsl"]
    assert all(row["tokens_ok"] for row in rows)
    assert rows[0]["output_tok_s_per_gpu"] == 110.0


def test_compares_matched_repetition_and_concurrency_to_cutedsl(
    tmp_path: Path,
) -> None:
    write_result(tmp_path, "cutedsl", 0, 8, 1000.0)
    write_result(tmp_path, "adaptive", 0, 8, 1120.0)

    rows = collect_rows(tmp_path, waves=10, isl=1000, osl=10000, gpus=8)
    compared = compare_to_baseline(rows, "cutedsl")
    adaptive = next(row for row in compared if row["arm"] == "adaptive")

    assert adaptive["throughput_vs_cutedsl"] == pytest.approx(1.12)
    assert adaptive["duration_vs_cutedsl"] == pytest.approx(1.0)
