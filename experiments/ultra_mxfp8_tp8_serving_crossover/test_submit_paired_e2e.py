# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parent
SUBMIT_SCRIPT = EXPERIMENT_DIR / "submit_paired_e2e.sh"


def test_submit_script_is_executable() -> None:
    assert SUBMIT_SCRIPT.stat().st_mode & stat.S_IXUSR


def test_plan_uses_reverse_order_same_allocation_contract() -> None:
    result = subprocess.run(
        ["bash", str(SUBMIT_SCRIPT)],
        env={**os.environ, "PRINT_PLAN": "1"},
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.splitlines() == [
        "parallelism=TP8,DP1,EP8",
        "workload=ISL1000,OSL10000",
        "concurrencies=8 32",
        "waves=10",
        "cuda_graph=true",
        "allocation=2nodes,4gpus_per_node",
        "order=cutedsl adaptive-lookup",
        "order=adaptive-lookup cutedsl",
    ]


def test_script_loads_tactic_hook_without_shadowing_installed_extensions() -> None:
    source = SUBMIT_SCRIPT.read_text()

    assert 'PYTHONPATH="${TACTIC_RUNTIME_ROOT}:${BENCH_ROOT}"' in source
    assert "VLLM_SUBPROCESS_PYTHONPATH=" in source
    assert 'test -s "${TACTIC_RUNTIME_ROOT}/sitecustomize.py"' in source
    assert 'PYTHONPATH="${SOURCE_ROOT}' not in source
    assert "STRICT_RESULT_TOKENS=1" in source
    assert "PYTHONDONTWRITEBYTECODE=1" in source
    assert 'BSIZES="${CONCURRENCIES}"' in source
    assert 'MULT="${WAVES}"' in source
    assert "sbatch --test-only" not in source
    assert 'SBATCH_TEST_ONLY="${test_only}"' in source
