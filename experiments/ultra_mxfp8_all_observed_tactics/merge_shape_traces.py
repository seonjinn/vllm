#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Merge per-process MXFP8 runtime traces into an oracle input CSV."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

ShapeKey = tuple[int, int, int, str]


def _is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


def merge_traces(
    trace_dir: Path, output_csv: Path, summary_path: Path
) -> dict[str, Any]:
    records: defaultdict[ShapeKey, dict[str, set[str]]] = defaultdict(
        lambda: {"phases": set(), "processes": set(), "ranks": set()}
    )
    files = sorted(trace_dir.rglob("trace.*.jsonl"))
    if not files:
        raise ValueError(f"no trace files found under {trace_dir}")

    for path in files:
        for line_number, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                key = (
                    int(row["m"]),
                    int(row["n"]),
                    int(row["k"]),
                    str(row["runner"]),
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(f"invalid trace row {path}:{line_number}") from error
            records[key]["phases"].add(str(row.get("phase", "unknown")))
            records[key]["processes"].add(str(row.get("pid", "unknown")))
            records[key]["ranks"].add(str(row.get("rank", "unknown")))

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "m",
                "n",
                "k",
                "runner",
                "phases",
                "process_count",
                "rank_count",
            ),
        )
        writer.writeheader()
        for (m, n, k, runner), provenance in sorted(records.items()):
            writer.writerow(
                {
                    "m": m,
                    "n": n,
                    "k": k,
                    "runner": runner,
                    "phases": ",".join(sorted(provenance["phases"])),
                    "process_count": len(provenance["processes"]),
                    "rank_count": len(provenance["ranks"]),
                }
            )

    m_values = {key[0] for key in records}
    summary = {
        "trace_file_count": len(files),
        "shape_count": len(records),
        "unique_m_count": len(m_values),
        "irregular_m_count": sum(not _is_power_of_two(m) for m in m_values),
        "min_m": min(m_values),
        "max_m": max(m_values),
        "output_csv": str(output_csv),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(merge_traces(args.trace_dir, args.output_csv, args.summary)))


if __name__ == "__main__":
    main()
