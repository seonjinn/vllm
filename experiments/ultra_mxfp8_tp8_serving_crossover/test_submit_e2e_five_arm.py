# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import os
import subprocess
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parent


def test_five_arm_plan_keeps_the_serving_setup_matched() -> None:
    result = subprocess.run(
        ["bash", str(EXPERIMENT_DIR / "submit_e2e_five_arm.sh")],
        env={**os.environ, "PRINT_PLAN": "1"},
        check=True,
        capture_output=True,
        text=True,
    )

    lines = result.stdout.splitlines()
    assert "parallelism=TP8,DP1,EP8" in lines
    assert "workload=ISL1000,OSL10000" in lines
    assert "concurrencies=1 2 4 8 16 32" in lines
    assert "waves=10" in lines
    assert "cuda_graph=true" in lines

    arms = [line for line in lines if line.startswith("arm=")]
    assert arms == [
        "arm=cutedsl linear_backend=flashinfer_cutedsl layout=8x4 lookup=none",
        "arm=trtllm-8x4 linear_backend=flashinfer_trtllm layout=8x4 lookup=none",
        "arm=trtllm-128x4 linear_backend=flashinfer_trtllm layout=128x4 lookup=none",
        "arm=adaptive linear_backend=flashinfer_trtllm layout=adaptive lookup=none",
        "arm=adaptive-lookup linear_backend=flashinfer_trtllm "
        "layout=adaptive lookup=exact",
    ]


def test_plan_rejects_unknown_arm() -> None:
    result = subprocess.run(
        ["bash", str(EXPERIMENT_DIR / "submit_e2e_five_arm.sh")],
        env={**os.environ, "PRINT_PLAN": "1", "ARMS": "adaptive unknown"},
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "Unsupported arm: unknown" in result.stderr
