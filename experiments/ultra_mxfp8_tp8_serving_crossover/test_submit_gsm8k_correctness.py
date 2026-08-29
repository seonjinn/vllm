# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import os
import subprocess
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parent
SUBMIT_SCRIPT = EXPERIMENT_DIR / "submit_gsm8k_correctness.sh"


def test_smoke_plan_keeps_matched_tp8_correctness_contract() -> None:
    result = subprocess.run(
        ["bash", str(SUBMIT_SCRIPT)],
        env={**os.environ, "PRINT_PLAN": "1", "PHASE": "smoke"},
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.splitlines() == [
        "phase=smoke",
        "examples=32",
        "parallelism=TP8,DP1,EP8",
        "decoding=temperature0,seed0,max_tokens256",
        "cuda_graph=true",
        "arm=cutedsl linear_backend=flashinfer_cutedsl layout=8x4 lookup=none",
        "arm=adaptive-lookup linear_backend=flashinfer_trtllm "
        "layout=adaptive lookup=exact",
    ]


def test_full_plan_uses_all_1319_examples() -> None:
    result = subprocess.run(
        ["bash", str(SUBMIT_SCRIPT)],
        env={
            **os.environ,
            "PRINT_PLAN": "1",
            "PHASE": "full",
            "ARMS": "adaptive",
        },
        check=True,
        capture_output=True,
        text=True,
    )

    assert "examples=1319" in result.stdout.splitlines()
    assert (
        "arm=adaptive linear_backend=flashinfer_trtllm layout=adaptive lookup=none"
    ) in result.stdout.splitlines()


def test_plan_rejects_unknown_correctness_arm() -> None:
    result = subprocess.run(
        ["bash", str(SUBMIT_SCRIPT)],
        env={**os.environ, "PRINT_PLAN": "1", "ARMS": "unknown"},
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "Unsupported correctness arm: unknown" in result.stderr


def test_server_uses_installed_vllm_extensions_and_tactic_hook() -> None:
    source = SUBMIT_SCRIPT.read_text()

    assert 'PYTHONPATH="${TACTIC_RUNTIME_ROOT}:${BENCH_ROOT}"' in source
    assert "VLLM_SUBPROCESS_PYTHONPATH=" in source
    assert "PYTHONDONTWRITEBYTECODE=1" in source
    assert 'PYTHONPATH="${BENCH_ROOT}:${SOURCE_ROOT}"' not in source
    assert 'PYTHONPATH="${SOURCE_ROOT}' not in source


def test_correctness_run_disables_unrelated_flashinfer_autotuning() -> None:
    source = SUBMIT_SCRIPT.read_text()

    assert "ENABLE_FLASHINFER_AUTOTUNE=0" in source
