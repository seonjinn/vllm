#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Build a strict exact-shape runtime lookup from oracle reports."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

Shape = tuple[int, int, int]


def _read_observed(path: Path) -> list[tuple[Shape, str]]:
    rows: list[tuple[Shape, str]] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                (
                    (int(row["m"]), int(row["n"]), int(row["k"])),
                    row["runner"],
                )
            )
    if not rows:
        raise ValueError(f"observed shape CSV is empty: {path}")
    return rows


def build_lookup(
    observed_path: Path, report_paths: list[Path], output_path: Path
) -> dict[str, Any]:
    observed = _read_observed(observed_path)
    oracle: dict[Shape, dict[str, Any]] = {}
    metadata: list[dict[str, Any]] = []
    for report_path in report_paths:
        report = json.loads(report_path.read_text())
        metadata.append(
            {
                "path": str(report_path),
                "backend": report.get("backend"),
                "scale_layout": report.get("scale_layout"),
                "gpu": report.get("gpu"),
            }
        )
        for row in report.get("shapes", []):
            shape = (int(row["m"]), int(row["n"]), int(row["k"]))
            candidate = {
                "tactic": row["oracle_tactic"],
                "cosine_similarity": float(row["oracle_cosine_similarity"]),
            }
            previous = oracle.get(shape)
            if previous is not None and previous != candidate:
                raise ValueError(f"conflicting oracle rows for {shape}")
            oracle[shape] = candidate

    missing = sorted({shape for shape, _ in observed} - oracle.keys())
    if missing:
        raise ValueError(
            f"missing oracle rows for {len(missing)} observed shapes: {missing[:8]}"
        )

    entries = []
    for shape, runner in sorted(observed):
        row = oracle[shape]
        if row["cosine_similarity"] < 0.98:
            raise ValueError(
                f"oracle tactic failed correctness for {shape}: "
                f"{row['cosine_similarity']}"
            )
        entries.append(
            {
                "m": shape[0],
                "n": shape[1],
                "k": shape[2],
                "runner": runner,
                "tactic": row["tactic"],
            }
        )

    backends = {item["backend"] for item in metadata}
    layouts = {item["scale_layout"] for item in metadata}
    payload = {
        "format_version": 1,
        "created_utc": datetime.now(UTC).isoformat(),
        "backend": next(iter(backends)) if len(backends) == 1 else "mixed",
        "scale_layout": next(iter(layouts)) if len(layouts) == 1 else "mixed",
        "entry_count": len(entries),
        "observed_shapes": str(observed_path),
        "oracle_reports": metadata,
        "entries": entries,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--observed", type=Path, required=True)
    parser.add_argument("--report", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_lookup(args.observed, args.report, args.output)
    print(json.dumps({"entry_count": result["entry_count"]}))


if __name__ == "__main__":
    main()
