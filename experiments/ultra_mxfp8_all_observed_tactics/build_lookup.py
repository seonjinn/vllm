#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Build a strict exact-shape runtime lookup from oracle reports."""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

Shape = tuple[int, int, int]


def _tactic_key(tactic: Any) -> str:
    return json.dumps(tactic, separators=(",", ":"), sort_keys=True)


def _read_observed(path: Path) -> dict[tuple[Shape, str], str]:
    rows: dict[tuple[Shape, str], str] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            key = (
                (int(row["m"]), int(row["n"]), int(row["k"])),
                row["runner"],
            )
            tactic = _tactic_key(json.loads(row["selected_tactic"]))
            previous = rows.get(key)
            if previous is not None and previous != tactic:
                raise ValueError(f"conflicting observed tactics for {key}")
            rows[key] = tactic
    if not rows:
        raise ValueError(f"observed shape CSV is empty: {path}")
    return rows


def build_lookup(
    observed_path: Path, report_paths: list[Path], output_path: Path
) -> dict[str, Any]:
    observed = _read_observed(observed_path)
    oracle: dict[tuple[Shape, str], dict[str, Any]] = {}
    metadata: list[dict[str, Any]] = []
    for report_path in report_paths:
        report = json.loads(report_path.read_text())
        metadata.append(
            {
                "path": str(report_path),
                "backend": report.get("backend"),
                "scale_layout": report.get("scale_layout"),
                "flashinfer_commit": report.get("flashinfer_commit"),
                "flashinfer_version": report.get("flashinfer_version"),
                "flashinfer_file": report.get("flashinfer_file"),
                "container_sha256": report.get("container_sha256"),
                "gpu": report.get("gpu"),
            }
        )
        for row in report.get("shapes", []):
            shape = (int(row["m"]), int(row["n"]), int(row["k"]))
            runner = str(row["runner"])
            metrics = {
                "cosine_similarity": float(row["oracle_cosine_similarity"]),
                "selected_ms": float(row["selected_ms"]),
                "oracle_ms": float(row["oracle_ms"]),
                "speedup": float(row["speedup"]),
            }
            if not all(math.isfinite(value) for value in metrics.values()) or any(
                metrics[name] <= 0 for name in ("selected_ms", "oracle_ms", "speedup")
            ):
                raise ValueError(
                    f"non-finite or non-positive oracle metrics for {shape}"
                )
            candidate = {
                "tactic": row["oracle_tactic"],
                "selected_tactic": _tactic_key(row["selected_tactic"]),
                **metrics,
                "finite": row.get("oracle_finite") is True,
                "matches_selected": row.get("oracle_matches_selected") is True,
            }
            previous = oracle.get((shape, runner))
            if previous is not None and previous != candidate:
                raise ValueError(f"conflicting oracle rows for {(shape, runner)}")
            oracle[(shape, runner)] = candidate

    missing = sorted(observed.keys() - oracle.keys())
    if missing:
        raise ValueError(
            f"missing oracle rows for {len(missing)} observed shapes: {missing[:8]}"
        )

    entries = []
    for shape, runner in sorted(observed):
        row = oracle[(shape, runner)]
        if row["selected_tactic"] != observed[(shape, runner)]:
            raise ValueError(
                f"oracle did not profile the observed serving tactic for {shape}"
            )
        if row["cosine_similarity"] < 0.98:
            raise ValueError(
                f"oracle tactic failed correctness for {shape}: "
                f"{row['cosine_similarity']}"
            )
        if not row["finite"] or not row["matches_selected"]:
            raise ValueError(f"oracle tactic failed elementwise parity for {shape}")
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
    flashinfer_commits = {item["flashinfer_commit"] for item in metadata}
    flashinfer_versions = {item["flashinfer_version"] for item in metadata}
    flashinfer_files = {item["flashinfer_file"] for item in metadata}
    container_hashes = {item["container_sha256"] for item in metadata}
    gpus = {item["gpu"] for item in metadata}
    if (
        len(backends) != 1
        or len(layouts) != 1
        or len(flashinfer_commits) != 1
        or len(flashinfer_versions) != 1
        or len(flashinfer_files) != 1
        or len(container_hashes) != 1
        or len(gpus) != 1
        or any(
            value in (None, "")
            for values in (
                backends,
                layouts,
                flashinfer_commits,
                flashinfer_versions,
                flashinfer_files,
                container_hashes,
                gpus,
            )
            for value in values
        )
    ):
        raise ValueError(
            "oracle reports must have one backend/layout/FlashInfer runtime/"
            "container/GPU: "
            f"{backends}, {layouts}, {flashinfer_commits}, "
            f"{flashinfer_versions}, {flashinfer_files}, {container_hashes}, {gpus}"
        )
    payload = {
        "format_version": 1,
        "created_utc": datetime.now(UTC).isoformat(),
        "backend": next(iter(backends)),
        "scale_layout": next(iter(layouts)),
        "flashinfer_commit": next(iter(flashinfer_commits)),
        "flashinfer_version": next(iter(flashinfer_versions)),
        "flashinfer_file": next(iter(flashinfer_files)),
        "container_sha256": next(iter(container_hashes)),
        "gpu": next(iter(gpus)),
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
