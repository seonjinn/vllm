#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).with_name("summarize_paired_e2e.py")
FORWARD = ("cutedsl", "adaptive-lookup")
REVERSE = tuple(reversed(FORWARD))


def _raw_result(*, concurrency: int, waves: int, osl: int, throughput: float) -> dict:
    requests = concurrency * waves
    output_tokens = requests * osl
    return {
        "completed": requests,
        "failed": 0,
        "total_output_tokens": output_tokens,
        "duration": output_tokens / throughput,
        "output_throughput": throughput,
    }


def _outer_row(*, concurrency: int, waves: int, osl: int, throughput: float) -> dict:
    raw = _raw_result(
        concurrency=concurrency,
        waves=waves,
        osl=osl,
        throughput=throughput,
    )
    return {
        "bs": concurrency,
        "osl": osl,
        "actual_output_tokens": raw["total_output_tokens"],
        "expected_output_tokens": raw["total_output_tokens"],
        "tokens_ok": True,
        "output_tok_s": throughput,
    }


def _write_allocation(
    root: Path,
    *,
    order: tuple[str, str],
    throughputs: dict[str, dict[int, float]],
    waves: int = 10,
    osl: int = 10_000,
    raw_updates: dict[tuple[str, int], dict] | None = None,
    duplicate_raw: tuple[str, int] | None = None,
) -> Path:
    slug = "-then-".join(order)
    allocation = root / "runs" / slug / "flashinfer_cutedsl" / "rep0" / "shortin"
    allocation.mkdir(parents=True)
    runs = []
    for arm in order:
        arm_dir = allocation / "paired" / arm
        arm_dir.mkdir(parents=True)
        result_rows = []
        for concurrency, throughput in sorted(throughputs[arm].items()):
            raw = _raw_result(
                concurrency=concurrency,
                waves=waves,
                osl=osl,
                throughput=throughput,
            )
            if raw_updates and (arm, concurrency) in raw_updates:
                raw.update(raw_updates[(arm, concurrency)])
            raw_path = arm_dir / f"raw_bench_serve_bs{concurrency}.json"
            raw_path.write_text(json.dumps(raw, sort_keys=True) + "\n")
            if duplicate_raw == (arm, concurrency):
                duplicate_path = (
                    arm_dir / f"raw_bench_serve_bs{concurrency}_retry1.json"
                )
                duplicate_path.write_text(json.dumps(raw, sort_keys=True) + "\n")
            result_rows.append(
                _outer_row(
                    concurrency=concurrency,
                    waves=waves,
                    osl=osl,
                    throughput=throughput,
                )
            )
        result_path = arm_dir / "result.json"
        embedded = {"config": {"batch_sizes": [8, 32]}, "results": result_rows}
        result_path.write_text(json.dumps(embedded, sort_keys=True) + "\n")
        runs.append(
            {
                "arm": arm,
                "result_path": str(result_path),
                "result": embedded,
            }
        )
    outer_path = allocation / "result_shortin.json"
    outer_path.write_text(json.dumps({"order": list(order), "runs": runs}) + "\n")
    return outer_path


def _valid_tree(root: Path) -> None:
    _write_allocation(
        root,
        order=FORWARD,
        throughputs={
            "cutedsl": {8: 100.0, 32: 200.0},
            "adaptive-lookup": {8: 110.0, 32: 220.0},
        },
    )
    _write_allocation(
        root,
        order=REVERSE,
        throughputs={
            "cutedsl": {8: 100.0, 32: 200.0},
            "adaptive-lookup": {8: 120.0, 32: 240.0},
        },
    )


def _run(
    root: Path,
    output_dir: Path,
    *,
    concurrencies: tuple[int, ...] | None = None,
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        str(SCRIPT),
        "--root",
        str(root),
        "--output-dir",
        str(output_dir),
    ]
    if concurrencies is not None:
        command.extend(["--concurrencies", *(str(value) for value in concurrencies)])
    return subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )


def test_cli_summarizes_two_reverse_order_allocations_deterministically(
    tmp_path: Path,
) -> None:
    root = tmp_path / "results"
    output_dir = tmp_path / "summary"
    _valid_tree(root)

    first = _run(root, output_dir)

    assert first.returncode == 0, first.stderr
    json_bytes = (output_dir / "summary.json").read_bytes()
    csv_bytes = (output_dir / "summary.csv").read_bytes()
    summary = json.loads(json_bytes)
    assert summary["orders"] == [list(FORWARD), list(REVERSE)]
    assert [row["concurrency"] for row in summary["comparisons"]] == [8, 32]
    for row in summary["comparisons"]:
        assert row["cutedsl_then_adaptive_lookup"]["throughput_ratio"] == 1.1
        assert row["adaptive_lookup_then_cutedsl"]["throughput_ratio"] == 1.2
        assert row["geometric_mean_throughput_ratio"] == pytest.approx(math.sqrt(1.32))

    csv_rows = list(csv.DictReader((output_dir / "summary.csv").open()))
    assert [int(row["concurrency"]) for row in csv_rows] == [8, 32]
    assert float(csv_rows[0]["geometric_mean_throughput_ratio"]) == pytest.approx(
        math.sqrt(1.32)
    )

    second = _run(root, output_dir)
    assert second.returncode == 0, second.stderr
    assert (output_dir / "summary.json").read_bytes() == json_bytes
    assert (output_dir / "summary.csv").read_bytes() == csv_bytes


def test_cli_rejects_missing_reverse_order_allocation(tmp_path: Path) -> None:
    root = tmp_path / "results"
    _write_allocation(
        root,
        order=FORWARD,
        throughputs={
            "cutedsl": {8: 100.0, 32: 200.0},
            "adaptive-lookup": {8: 110.0, 32: 220.0},
        },
    )

    result = _run(root, tmp_path / "summary")

    assert result.returncode != 0
    assert "exactly two reverse-order allocations" in result.stderr


def test_cli_rejects_duplicate_order_allocations(tmp_path: Path) -> None:
    root = tmp_path / "results"
    first = _write_allocation(
        root,
        order=FORWARD,
        throughputs={
            "cutedsl": {8: 100.0, 32: 200.0},
            "adaptive-lookup": {8: 110.0, 32: 220.0},
        },
    )
    duplicate = root / "duplicate" / "result_shortin.json"
    duplicate.parent.mkdir(parents=True)
    duplicate.write_text(first.read_text())

    result = _run(root, tmp_path / "summary")

    assert result.returncode != 0
    assert "exactly two reverse-order allocations" in result.stderr


def test_cli_rejects_missing_requested_concurrency(tmp_path: Path) -> None:
    root = tmp_path / "results"
    _valid_tree(root)
    missing = next(root.rglob("paired/cutedsl/raw_bench_serve_bs32.json"))
    missing.unlink()

    result = _run(root, tmp_path / "summary")

    assert result.returncode != 0
    assert "missing concurrency 32" in result.stderr


def test_cli_validates_only_explicitly_requested_concurrency(tmp_path: Path) -> None:
    root = tmp_path / "results"
    _write_allocation(
        root,
        order=FORWARD,
        throughputs={
            "cutedsl": {8: 100.0, 32: 200.0},
            "adaptive-lookup": {8: 110.0, 32: 220.0},
        },
        raw_updates={("adaptive-lookup", 32): {"failed": 1}},
    )
    _write_allocation(
        root,
        order=REVERSE,
        throughputs={
            "cutedsl": {8: 100.0},
            "adaptive-lookup": {8: 120.0},
        },
    )

    result = _run(root, tmp_path / "summary", concurrencies=(8,))

    assert result.returncode == 0, result.stderr
    summary = json.loads((tmp_path / "summary" / "summary.json").read_text())
    assert summary["concurrencies"] == [8]
    assert summary["comparisons"][0][
        "geometric_mean_throughput_ratio"
    ] == pytest.approx(math.sqrt(1.32))


def test_cli_rejects_duplicate_concurrency_row(tmp_path: Path) -> None:
    root = tmp_path / "results"
    _write_allocation(
        root,
        order=FORWARD,
        throughputs={
            "cutedsl": {8: 100.0, 32: 200.0},
            "adaptive-lookup": {8: 110.0, 32: 220.0},
        },
        duplicate_raw=("cutedsl", 8),
    )
    _write_allocation(
        root,
        order=REVERSE,
        throughputs={
            "cutedsl": {8: 100.0, 32: 200.0},
            "adaptive-lookup": {8: 120.0, 32: 240.0},
        },
    )

    result = _run(root, tmp_path / "summary")

    assert result.returncode != 0
    assert "duplicate concurrency 8" in result.stderr


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ({"completed": 79}, "completed=79, expected 80"),
        ({"failed": 1}, "failed=1, expected 0"),
        ({"total_output_tokens": 799_999}, "output tokens=799999, expected 800000"),
    ],
)
def test_cli_rejects_invalid_raw_row(
    tmp_path: Path, update: dict, message: str
) -> None:
    root = tmp_path / "results"
    _write_allocation(
        root,
        order=FORWARD,
        throughputs={
            "cutedsl": {8: 100.0, 32: 200.0},
            "adaptive-lookup": {8: 110.0, 32: 220.0},
        },
        raw_updates={("adaptive-lookup", 8): update},
    )
    _write_allocation(
        root,
        order=REVERSE,
        throughputs={
            "cutedsl": {8: 100.0, 32: 200.0},
            "adaptive-lookup": {8: 120.0, 32: 240.0},
        },
    )

    result = _run(root, tmp_path / "summary")

    assert result.returncode != 0
    assert message in result.stderr
