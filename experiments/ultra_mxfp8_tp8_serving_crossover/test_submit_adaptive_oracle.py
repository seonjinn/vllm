# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).with_name("submit_adaptive_oracle.sh")


def test_adaptive_oracle_submitter_is_executable() -> None:
    assert SCRIPT.stat().st_mode & stat.S_IXUSR


def test_adaptive_oracle_plan_is_layout_partitioned() -> None:
    result = subprocess.run(
        ["bash", str(SCRIPT)],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "PRINT_PLAN": "1", "SWITCH_M": "256"},
    )

    assert result.returncode == 0, result.stderr
    assert "layouts=8x4 128x4" in result.stdout
    assert "switch_m=256" in result.stdout
    assert "oracle=cuda_graph,cold_l2,rounds=2,repeat_iters=10" in result.stdout
    assert "hardware=1xGB200-node-per-layout,4-GPUs" in result.stdout


def test_oracle_job_uses_node_local_scratch_worker() -> None:
    text = SCRIPT.with_name("run_adaptive_oracle.sbatch").read_text()

    assert "oracle_worker.sh" in text
    assert '--ntasks-per-node="${SHARD_COUNT}"' in text
    assert "SCALE_LAYOUT" in text
