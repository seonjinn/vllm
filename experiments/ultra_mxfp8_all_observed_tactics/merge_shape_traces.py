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
_PHASE_PRECEDENCE = ("baseline", "graph", "eager")


def _is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


def _positive_count(row: dict[str, Any], path: Path, line_number: int) -> int:
    value = row.get("invocation_count")
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"invalid invocation_count {value!r} at {path}:{line_number}")
    return value


def merge_traces(
    trace_dir: Path, output_csv: Path, summary_path: Path
) -> dict[str, Any]:
    records: defaultdict[ShapeKey, dict[str, Any]] = defaultdict(
        lambda: {
            "phases": set(),
            "processes": set(),
            "ranks": set(),
            "tactics_by_phase": defaultdict(lambda: defaultdict(int)),
        }
    )
    count_files = {
        (
            path.parent.relative_to(trace_dir),
            path.name.removeprefix("counts.").removesuffix(".jsonl"),
        ): path
        for path in trace_dir.rglob("counts.*.jsonl")
    }
    unfinished = [
        path
        for (_, pid), path in count_files.items()
        if not path.with_name(f"counts.{pid}.complete").is_file()
    ]
    if unfinished:
        raise ValueError(f"unfinished count snapshot: {unfinished}")
    trace_files = {
        (
            path.parent.relative_to(trace_dir),
            path.name.removeprefix("trace.").removesuffix(".jsonl"),
        ): path
        for path in trace_dir.rglob("trace.*.jsonl")
    }
    missing_counts = sorted(
        path for process, path in trace_files.items() if process not in count_files
    )
    if missing_counts:
        raise ValueError(f"trace files are missing count snapshots: {missing_counts}")
    files = sorted(count_files.values())
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
            relative_parent = path.parent.relative_to(trace_dir)
            records[key]["processes"].add(
                f"{relative_parent}:{row.get('pid', 'unknown')}"
            )
            records[key]["ranks"].add(str(row.get("rank", "unknown")))
            if row.get("selection_source", "default_autotuner") == "default_autotuner":
                phase = str(row.get("phase", "unknown"))
                tactic = json.dumps(row["tactic"], separators=(",", ":"))
                count = _positive_count(row, path, line_number)
                records[key]["tactics_by_phase"][phase][tactic] += count

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
                "selected_phase",
                "selected_tactic",
                "selection_call_count",
                "process_count",
                "rank_count",
            ),
        )
        writer.writeheader()
        for (m, n, k, runner), provenance in sorted(records.items()):
            available_phases = provenance["tactics_by_phase"]
            if not available_phases:
                raise ValueError(
                    f"no default serving tactic recorded for {(m, n, k, runner)}"
                )
            selected_phase = next(
                (phase for phase in _PHASE_PRECEDENCE if phase in available_phases),
                None,
            )
            if selected_phase is None:
                selected_phase = min(available_phases)
            tactics = available_phases[selected_phase]
            if len(tactics) != 1:
                raise ValueError(
                    "conflicting serving tactics for "
                    f"{(m, n, k, runner)} in phase {selected_phase}: {dict(tactics)}"
                )
            selected_tactic = next(iter(tactics))
            selection_call_count = sum(tactics.values())
            writer.writerow(
                {
                    "m": m,
                    "n": n,
                    "k": k,
                    "runner": runner,
                    "phases": ",".join(sorted(provenance["phases"])),
                    "selected_phase": selected_phase,
                    "selected_tactic": selected_tactic,
                    "selection_call_count": selection_call_count,
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
