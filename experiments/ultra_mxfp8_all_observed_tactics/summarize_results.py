#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Summarize exact-shape MXFP8 oracle and serving A/B artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _geomean(values: list[float]) -> float:
    if not values:
        raise ValueError("geomean requires at least one value")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _complete_run(root: Path, kind: str) -> Path:
    markers = sorted((root / "serving" / kind).glob("*/COMPLETE"))
    if len(markers) != 1:
        raise ValueError(
            f"expected one complete {kind} run below {root}, found {len(markers)}"
        )
    return markers[0].parent


def _result_files(run_dir: Path) -> dict[tuple[int, int, int], Path]:
    results: dict[tuple[int, int, int], Path] = {}
    for path in sorted(run_dir.glob("result_*.json")):
        parts = path.stem.split("_")
        if len(parts) != 4 or parts[0] != "result":
            continue
        prefixes = ("isl", "osl", "c")
        if any(
            not part.startswith(prefix) for part, prefix in zip(parts[1:], prefixes)
        ):
            continue
        key = tuple(
            int(part[len(prefix) :]) for part, prefix in zip(parts[1:], prefixes)
        )
        results[key] = path
    return results


def _summarize_e2e(root: Path) -> dict[str, Any]:
    baseline_dir = _complete_run(root, "baseline")
    lookup_dir = _complete_run(root, "lookup")
    baseline_files = _result_files(baseline_dir)
    lookup_files = _result_files(lookup_dir)
    if baseline_files.keys() != lookup_files.keys():
        raise ValueError(
            "baseline and lookup result files do not match: "
            f"baseline={sorted(baseline_files)}, lookup={sorted(lookup_files)}"
        )

    workloads = []
    for (isl, osl, concurrency), baseline_path in baseline_files.items():
        baseline = _load(baseline_path)
        lookup = _load(lookup_files[(isl, osl, concurrency)])
        for label, result in (("baseline", baseline), ("lookup", lookup)):
            if result["failed"] != 0:
                raise ValueError(f"{label} has failed requests: {result}")
        workloads.append(
            {
                "isl": isl,
                "osl": osl,
                "concurrency": concurrency,
                "completed": lookup["completed"],
                "total_input_tokens": lookup["total_input_tokens"],
                "total_output_tokens": lookup["total_output_tokens"],
                "baseline_output_throughput": baseline["output_throughput"],
                "lookup_output_throughput": lookup["output_throughput"],
                "output_throughput_speedup": (
                    lookup["output_throughput"] / baseline["output_throughput"]
                ),
                "baseline_total_token_throughput": baseline["total_token_throughput"],
                "lookup_total_token_throughput": lookup["total_token_throughput"],
                "total_token_throughput_speedup": (
                    lookup["total_token_throughput"]
                    / baseline["total_token_throughput"]
                ),
                "baseline_mean_ttft_ms": baseline["mean_ttft_ms"],
                "lookup_mean_ttft_ms": lookup["mean_ttft_ms"],
                "baseline_mean_tpot_ms": baseline["mean_tpot_ms"],
                "lookup_mean_tpot_ms": lookup["mean_tpot_ms"],
                "baseline_duration_s": baseline["duration"],
                "lookup_duration_s": lookup["duration"],
                "duration_reduction_pct": 100.0
                * (1.0 - lookup["duration"] / baseline["duration"]),
            }
        )

    return {
        "baseline_run": str(baseline_dir),
        "lookup_run": str(lookup_dir),
        "workload_count": len(workloads),
        "geomean_output_throughput_speedup": _geomean(
            [row["output_throughput_speedup"] for row in workloads]
        ),
        "geomean_total_token_throughput_speedup": _geomean(
            [row["total_token_throughput_speedup"] for row in workloads]
        ),
        "workloads": workloads,
    }


def _summarize_oracle(root: Path) -> dict[str, Any]:
    report_paths = sorted(root.glob("oracle/shards/*/oracle/report.json"))
    if not report_paths:
        raise ValueError(f"no oracle reports below {root}")
    rows = [row for path in report_paths for row in _load(path)["shapes"]]
    cache_metadata = [
        _load(path)
        for path in sorted(root.glob("oracle/shards/*/cache/exact_cache_metadata.json"))
    ]
    speedups = [float(row["speedup"]) for row in rows]
    regrets = [100.0 * (speedup - 1.0) for speedup in speedups]
    candidate_counts = [int(row["candidate_count"]) for row in rows]
    top_regrets = []
    for row in sorted(rows, key=lambda item: float(item["speedup"]), reverse=True)[:20]:
        top_regrets.append(
            {
                "m": int(row["m"]),
                "n": int(row["n"]),
                "k": int(row["k"]),
                "candidate_count": int(row["candidate_count"]),
                "selected_ms": float(row["selected_ms"]),
                "oracle_ms": float(row["oracle_ms"]),
                "regret_pct": 100.0 * (float(row["speedup"]) - 1.0),
                "selected_tactic": row["selected_tactic"],
                "oracle_tactic": row["oracle_tactic"],
            }
        )
    return {
        "shape_count": len(rows),
        "geomean_speedup": _geomean(speedups),
        "max_speedup": max(speedups),
        "max_regret_pct": 100.0 * (max(speedups) - 1.0),
        "different_tactic_count": sum(not row["same_tactic"] for row in rows),
        "candidate_count_min": min(candidate_counts),
        "candidate_count_max": max(candidate_counts),
        "regret_pct_p50": _percentile(regrets, 50.0),
        "regret_pct_p90": _percentile(regrets, 90.0),
        "regret_pct_p99": _percentile(regrets, 99.0),
        "minimum_oracle_cosine_similarity": min(
            float(row["oracle_cosine_similarity"]) for row in rows
        ),
        "tuning_time_s": sum(
            float(metadata.get("tuning_time_s", 0.0)) for metadata in cache_metadata
        ),
        "top_regrets": top_regrets,
        "reports": [str(path) for path in report_paths],
    }


def _summarize_lookup(root: Path) -> dict[str, Any]:
    lookup_dir = _complete_run(root, "lookup")
    sources: dict[tuple[int, int, int, str], set[str]] = {}
    trace_paths = sorted((lookup_dir / "traces").glob("trace.*.jsonl"))
    if not trace_paths:
        raise ValueError(f"no lookup traces below {lookup_dir}")
    for path in trace_paths:
        for line in path.read_text().splitlines():
            row = json.loads(line)
            key = (
                int(row["m"]),
                int(row["n"]),
                int(row["k"]),
                str(row["runner"]),
            )
            sources.setdefault(key, set()).add(str(row["selection_source"]))
    conflicts = {key: value for key, value in sources.items() if len(value) != 1}
    if conflicts:
        raise ValueError(
            f"lookup source changed for the same dispatch key: {conflicts}"
        )
    hit_count = sum(next(iter(value)) == "offline_lookup" for value in sources.values())
    miss_count = len(sources) - hit_count
    return {
        "trace_process_count": len(trace_paths),
        "unique_dispatch_count": len(sources),
        "unique_hit_count": hit_count,
        "unique_miss_count": miss_count,
        "unique_hit_rate": hit_count / len(sources),
    }


def summarize_backend(root: Path, backend: str) -> dict[str, Any]:
    return {
        "backend": backend,
        "result_root": str(root),
        "e2e": _summarize_e2e(root),
        "oracle": _summarize_oracle(root),
        "lookup": _summarize_lookup(root),
    }


def _write_csv(path: Path, summaries: list[dict[str, Any]]) -> None:
    rows = []
    for summary in summaries:
        for workload in summary["e2e"]["workloads"]:
            rows.append({"backend": summary["backend"], **workload})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result",
        action="append",
        required=True,
        metavar="BACKEND=PATH",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()
    summaries = []
    for value in args.result:
        backend, separator, root = value.partition("=")
        if not separator:
            parser.error(f"invalid --result {value!r}; expected BACKEND=PATH")
        summaries.append(summarize_backend(Path(root), backend))
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps({"backends": summaries}, indent=2) + "\n")
    _write_csv(args.output_csv, summaries)


if __name__ == "__main__":
    main()
