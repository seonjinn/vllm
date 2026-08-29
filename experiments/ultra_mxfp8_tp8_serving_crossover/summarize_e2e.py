#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

RESULT_PREFIX = "raw_bench_serve_bs"
RESULT_SUFFIX = ".json"


def collect_rows(
    root: Path,
    *,
    waves: int,
    isl: int,
    osl: int,
    gpus: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result_path in root.glob("runs/*/rep*/*/*/raw_bench_serve_bs*.json"):
        relative = result_path.relative_to(root)
        arm = relative.parts[1]
        repetition_text = relative.parts[2]
        has_prefix = result_path.name.startswith(RESULT_PREFIX)
        has_suffix = result_path.name.endswith(RESULT_SUFFIX)
        if not has_prefix or not has_suffix:
            continue
        if not repetition_text.startswith("rep"):
            continue
        repetition = int(repetition_text.removeprefix("rep"))
        concurrency = int(
            result_path.name.removeprefix(RESULT_PREFIX).removesuffix(RESULT_SUFFIX)
        )
        expected_requests = concurrency * waves
        expected_output_tokens = expected_requests * osl
        result = json.loads(result_path.read_text())
        tokens_ok = (
            result.get("completed") == expected_requests
            and result.get("failed") == 0
            and result.get("total_output_tokens") == expected_output_tokens
            and result.get("total_input_tokens")
            in (expected_requests * isl, expected_requests * (isl + 1))
        )
        if not tokens_ok:
            continue
        output_throughput = float(result["output_throughput"])
        rows.append(
            {
                "arm": arm,
                "repetition": repetition,
                "concurrency": concurrency,
                "completed": result["completed"],
                "failed": result["failed"],
                "tokens_ok": True,
                "duration_s": float(result["duration"]),
                "output_tok_s": output_throughput,
                "output_tok_s_per_gpu": output_throughput / gpus,
                "total_tok_s": float(result["total_token_throughput"]),
                "request_throughput": float(result["request_throughput"]),
                "mean_ttft_ms": float(result["mean_ttft_ms"]),
                "median_ttft_ms": float(result["median_ttft_ms"]),
                "p99_ttft_ms": float(result["p99_ttft_ms"]),
                "mean_tpot_ms": float(result["mean_tpot_ms"]),
                "median_tpot_ms": float(result["median_tpot_ms"]),
                "p99_tpot_ms": float(result["p99_tpot_ms"]),
                "result_path": str(result_path),
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            str(row["arm"]),
            int(row["repetition"]),
            int(row["concurrency"]),
        ),
    )


def compare_to_baseline(
    rows: list[dict[str, Any]], baseline_arm: str
) -> list[dict[str, Any]]:
    baseline = {
        (int(row["repetition"]), int(row["concurrency"])): row
        for row in rows
        if row["arm"] == baseline_arm
    }
    compared: list[dict[str, Any]] = []
    for row in rows:
        output = dict(row)
        key = (int(row["repetition"]), int(row["concurrency"]))
        reference = baseline.get(key)
        if reference is not None:
            output[f"throughput_vs_{baseline_arm}"] = float(
                row["output_tok_s"]
            ) / float(reference["output_tok_s"])
            output[f"duration_vs_{baseline_arm}"] = float(row["duration_s"]) / float(
                reference["duration_s"]
            )
        else:
            output[f"throughput_vs_{baseline_arm}"] = None
            output[f"duration_vs_{baseline_arm}"] = None
        compared.append(output)
    return compared


def write_summary(rows: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    if not rows:
        (output_dir / "summary.csv").write_text("")
        return
    with (output_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline", default="cutedsl")
    parser.add_argument("--waves", type=int, default=10)
    parser.add_argument("--isl", type=int, default=1000)
    parser.add_argument("--osl", type=int, default=10000)
    parser.add_argument("--gpus", type=int, default=8)
    args = parser.parse_args()

    rows = collect_rows(
        args.root,
        waves=args.waves,
        isl=args.isl,
        osl=args.osl,
        gpus=args.gpus,
    )
    write_summary(compare_to_baseline(rows, args.baseline), args.output_dir)
    print(json.dumps({"valid_rows": len(rows), "output_dir": str(args.output_dir)}))


if __name__ == "__main__":
    main()
