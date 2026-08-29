# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.ultra_mxfp8_tp8_serving_crossover.summarize_gsm8k_pair import (
    summarize_pair,
)


def _write_arm(
    root: Path,
    name: str,
    correctness: list[bool],
    predictions: list[str | None],
) -> Path:
    arm = root / name
    arm.mkdir()
    rows = [
        {
            "correct": correct,
            "empty": prediction is None,
            "id": f"gsm8k-{index:04d}",
            "invalid": prediction is None,
            "prediction": prediction,
            "question_sha256": f"question-{index}",
            "target": str(index),
        }
        for index, (correct, prediction) in enumerate(
            zip(correctness, predictions, strict=True)
        )
    ]
    (arm / "per_example.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    (arm / "results.json").write_text(
        json.dumps(
            {
                "correct": sum(correctness),
                "empty_predictions": sum(
                    prediction is None for prediction in predictions
                ),
                "exact_match": sum(correctness) / len(correctness),
                "invalid_predictions": sum(
                    prediction is None for prediction in predictions
                ),
                "provenance": {
                    "checkpoint": "model",
                    "container_sha256": "container",
                    "evaluation_contract": {
                        "dataset": "gsm8k.jsonl",
                        "limit": len(correctness),
                        "topology": "TP8/DP1/EP8",
                    },
                    "source_commit": "source",
                },
                "total": len(correctness),
            }
        )
        + "\n"
    )
    return arm


def test_summarize_pair_reports_paired_correctness_and_prediction_agreement(
    tmp_path: Path,
) -> None:
    cutedsl = _write_arm(
        tmp_path,
        "cutedsl",
        [True, True, False, False],
        ["0", "1", "wrong", None],
    )
    adaptive = _write_arm(
        tmp_path,
        "adaptive",
        [True, False, True, False],
        ["0", "other", "2", None],
    )

    summary = summarize_pair(cutedsl, adaptive)

    assert summary["total"] == 4
    assert summary["cutedsl"]["correct"] == 2
    assert summary["adaptive_lookup"]["correct"] == 2
    assert summary["paired"] == {
        "adaptive_only_correct": 1,
        "both_correct": 1,
        "both_wrong": 1,
        "correctness_agreement": 0.5,
        "cutedsl_only_correct": 1,
        "mcnemar_exact_two_sided_p": 1.0,
        "prediction_agreement": 0.5,
    }


def test_summarize_pair_rejects_mismatched_example_identity(tmp_path: Path) -> None:
    cutedsl = _write_arm(tmp_path, "cutedsl", [True], ["0"])
    adaptive = _write_arm(tmp_path, "adaptive", [True], ["0"])
    path = adaptive / "per_example.jsonl"
    row = json.loads(path.read_text())
    row["question_sha256"] = "different"
    path.write_text(json.dumps(row) + "\n")

    with pytest.raises(ValueError, match="question hash mismatch"):
        summarize_pair(cutedsl, adaptive)
