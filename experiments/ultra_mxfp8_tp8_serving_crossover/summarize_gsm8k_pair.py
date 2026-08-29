#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read JSON {path}: {error}") from error


def _load_arm(
    path: Path, name: str
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    result = _load_json(path / "results.json")
    if not isinstance(result, dict):
        raise ValueError(f"{name}: results.json must contain an object")
    rows: dict[str, dict[str, Any]] = {}
    per_example = path / "per_example.jsonl"
    try:
        lines = per_example.read_text().splitlines()
    except OSError as error:
        raise ValueError(f"could not read {per_example}: {error}") from error
    for line_number, line in enumerate(lines, start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"{name}: invalid JSON on line {line_number} of {per_example}"
            ) from error
        example_id = row.get("id") if isinstance(row, dict) else None
        if not isinstance(example_id, str):
            raise ValueError(f"{name}: invalid example ID on line {line_number}")
        if example_id in rows:
            raise ValueError(f"{name}: duplicate example ID {example_id}")
        if type(row.get("correct")) is not bool:
            raise ValueError(f"{name}: invalid correctness for {example_id}")
        rows[example_id] = row

    total = result.get("total")
    correct = result.get("correct")
    if type(total) is not int or total != len(rows):
        raise ValueError(f"{name}: result total does not match per-example rows")
    measured_correct = sum(row["correct"] for row in rows.values())
    if type(correct) is not int or correct != measured_correct:
        raise ValueError(f"{name}: result correct count does not match rows")
    exact_match = result.get("exact_match")
    if not isinstance(exact_match, (int, float)) or not math.isclose(
        float(exact_match), correct / total, rel_tol=0, abs_tol=1e-12
    ):
        raise ValueError(f"{name}: exact_match does not match correct / total")
    return result, rows


def _mcnemar_exact_two_sided(adaptive_only: int, cutedsl_only: int) -> float:
    discordant = adaptive_only + cutedsl_only
    if discordant == 0:
        return 1.0
    tail = min(adaptive_only, cutedsl_only)
    probability = sum(math.comb(discordant, k) for k in range(tail + 1)) / (
        2**discordant
    )
    return min(1.0, 2 * probability)


def _arm_summary(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "correct": result["correct"],
        "empty_predictions": result.get("empty_predictions"),
        "exact_match": result["exact_match"],
        "invalid_predictions": result.get("invalid_predictions"),
    }


def summarize_pair(cutedsl_dir: Path, adaptive_dir: Path) -> dict[str, Any]:
    cutedsl_result, cutedsl_rows = _load_arm(cutedsl_dir, "cutedsl")
    adaptive_result, adaptive_rows = _load_arm(adaptive_dir, "adaptive-lookup")

    if set(cutedsl_rows) != set(adaptive_rows):
        raise ValueError("GSM8K example ID sets do not match")
    for key in (
        "checkpoint",
        "container_sha256",
        "evaluation_contract",
        "source_commit",
    ):
        cutedsl_value = cutedsl_result.get("provenance", {}).get(key)
        adaptive_value = adaptive_result.get("provenance", {}).get(key)
        if cutedsl_value != adaptive_value:
            raise ValueError(f"provenance mismatch for {key}")

    both_correct = 0
    both_wrong = 0
    adaptive_only = 0
    cutedsl_only = 0
    prediction_matches = 0
    for example_id in sorted(cutedsl_rows):
        cutedsl = cutedsl_rows[example_id]
        adaptive = adaptive_rows[example_id]
        if cutedsl.get("question_sha256") != adaptive.get("question_sha256"):
            raise ValueError(f"question hash mismatch for {example_id}")
        if cutedsl.get("target") != adaptive.get("target"):
            raise ValueError(f"target mismatch for {example_id}")
        cutedsl_correct = cutedsl["correct"]
        adaptive_correct = adaptive["correct"]
        if cutedsl_correct and adaptive_correct:
            both_correct += 1
        elif cutedsl_correct:
            cutedsl_only += 1
        elif adaptive_correct:
            adaptive_only += 1
        else:
            both_wrong += 1
        prediction_matches += cutedsl.get("prediction") == adaptive.get("prediction")

    total = len(cutedsl_rows)
    return {
        "accuracy_difference_percentage_points": 100
        * (adaptive_result["exact_match"] - cutedsl_result["exact_match"]),
        "adaptive_lookup": _arm_summary(adaptive_result),
        "cutedsl": _arm_summary(cutedsl_result),
        "paired": {
            "adaptive_only_correct": adaptive_only,
            "both_correct": both_correct,
            "both_wrong": both_wrong,
            "correctness_agreement": (both_correct + both_wrong) / total,
            "cutedsl_only_correct": cutedsl_only,
            "mcnemar_exact_two_sided_p": _mcnemar_exact_two_sided(
                adaptive_only, cutedsl_only
            ),
            "prediction_agreement": prediction_matches / total,
        },
        "schema_version": 1,
        "total": total,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cutedsl-dir", type=Path, required=True)
    parser.add_argument("--adaptive-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        summary = summarize_pair(args.cutedsl_dir, args.adaptive_dir)
    except ValueError as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
