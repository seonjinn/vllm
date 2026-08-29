# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).with_name("submit_shape_census.sh")


def _print_plan(phases: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(SCRIPT)],
        env={**os.environ, "PRINT_PLAN": "1", "PHASES": phases},
        check=False,
        capture_output=True,
        text=True,
    )


def test_plan_uses_tp8_and_ten_low_concurrency_waves() -> None:
    result = _print_plan("low")

    assert result.returncode == 0, result.stderr
    assert "parallelism=TP8,DP1,EP8" in result.stdout
    assert "workload=ISL1000,OSL10000" in result.stdout
    assert "concurrencies=1 2 4 8 16 32 waves=10" in result.stdout


def test_plan_gates_high_concurrency_with_one_wave_smoke() -> None:
    result = _print_plan("high-smoke")

    assert result.returncode == 0, result.stderr
    assert "concurrencies=128 512 waves=1" in result.stdout


def test_plan_rejects_unknown_phase() -> None:
    result = _print_plan("unknown")

    assert result.returncode == 2
    assert "Unsupported phase" in result.stderr
