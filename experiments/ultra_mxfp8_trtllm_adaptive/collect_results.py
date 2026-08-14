#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import regex as re

RESULT_PATTERN = re.compile(r"result_isl(?P<isl>\d+)_osl(?P<osl>\d+)_c(?P<c>\d+)\.json")


@dataclass(frozen=True)
class Result:
    layout: str
    isl: int
    osl: int
    concurrency: int
    completed: int
    failed: int
    total_input_tokens: int
    total_output_tokens: int
    request_throughput: float
    output_throughput: float
    total_token_throughput: float
    mean_ttft_ms: float
    p99_ttft_ms: float
    mean_tpot_ms: float
    p99_tpot_ms: float


def parse_result(path: Path, layout: str) -> Result:
    match = RESULT_PATTERN.fullmatch(path.name)
    if match is None:
        raise ValueError(f"Unexpected result filename: {path}")

    data: dict[str, Any] = json.loads(path.read_text())
    isl = int(match.group("isl"))
    osl = int(match.group("osl"))
    concurrency = int(match.group("c"))
    completed = int(data["completed"])
    failed = int(data["failed"])
    expected_input_tokens = completed * isl
    expected_output_tokens = completed * osl
    if failed != 0:
        raise ValueError(f"{path}: {failed} requests failed")
    if int(data["total_input_tokens"]) != expected_input_tokens:
        raise ValueError(f"{path}: input token count does not match fixed ISL")
    if int(data["total_output_tokens"]) != expected_output_tokens:
        raise ValueError(f"{path}: output token count does not match fixed OSL")

    return Result(
        layout=layout,
        isl=isl,
        osl=osl,
        concurrency=concurrency,
        completed=completed,
        failed=failed,
        total_input_tokens=expected_input_tokens,
        total_output_tokens=expected_output_tokens,
        request_throughput=float(data["request_throughput"]),
        output_throughput=float(data["output_throughput"]),
        total_token_throughput=float(data["total_token_throughput"]),
        mean_ttft_ms=float(data["mean_ttft_ms"]),
        p99_ttft_ms=float(data["p99_ttft_ms"]),
        mean_tpot_ms=float(data["mean_tpot_ms"]),
        p99_tpot_ms=float(data["p99_tpot_ms"]),
    )


def collect(root: Path) -> list[Result]:
    results: list[Result] = []
    for layout in ("8x4", "adaptive"):
        paths = sorted((root / layout).glob("*/result_*.json"))
        results.extend(parse_result(path, layout) for path in paths)
    return sorted(
        results, key=lambda row: (row.isl, row.osl, row.concurrency, row.layout)
    )


def write_csv(results: list[Result], path: Path) -> None:
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(asdict(results[0])))
        writer.writeheader()
        writer.writerows(asdict(result) for result in results)


def percent_change(value: float, baseline: float) -> float:
    return (value / baseline - 1.0) * 100.0


def print_comparison(results: list[Result]) -> None:
    by_key = {
        (result.layout, result.isl, result.osl, result.concurrency): result
        for result in results
    }
    print(
        "| ISL / OSL | C | Fixed output tok/s | Adaptive output tok/s | "
        "Change | TTFT change | TPOT change |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|")
    for isl, osl, concurrency in sorted(
        {(result.isl, result.osl, result.concurrency) for result in results}
    ):
        fixed = by_key[("8x4", isl, osl, concurrency)]
        adaptive = by_key[("adaptive", isl, osl, concurrency)]
        throughput_change = percent_change(
            adaptive.output_throughput, fixed.output_throughput
        )
        print(
            f"| {isl} / {osl} | {concurrency} | {fixed.output_throughput:.2f} | "
            f"{adaptive.output_throughput:.2f} | "
            f"{throughput_change:+.2f}% | "
            f"{percent_change(adaptive.mean_ttft_ms, fixed.mean_ttft_ms):+.2f}% | "
            f"{percent_change(adaptive.mean_tpot_ms, fixed.mean_tpot_ms):+.2f}% |"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_root", type=Path)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()

    results = collect(args.result_root)
    if len(results) != 18:
        raise ValueError(f"Expected 18 result files, found {len(results)}")
    if args.csv is not None:
        write_csv(results, args.csv)
    print_comparison(results)


if __name__ == "__main__":
    main()
