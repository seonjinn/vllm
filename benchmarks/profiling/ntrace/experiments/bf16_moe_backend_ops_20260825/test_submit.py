# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import os
import subprocess
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).parent
REPO_ROOT = EXPERIMENT_DIR.parents[4]


def _touch(path: Path, content: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def test_dry_run_emits_matched_triton_and_trtllm_arms(tmp_path: Path) -> None:
    bench_root = tmp_path / "bench"
    runtime = tmp_path / "runtime"
    result_root = tmp_path / "results"
    container = tmp_path / "vllm.sqsh"
    fake_submit = bench_root / "submit_bench_lyris_nemotron3_ultra_w4a16.sh"

    _touch(
        fake_submit,
        "#!/usr/bin/env bash\n"
        "printf 'backend=%s\\n' \"$BF16_MOE_BACKEND\"\n"
        "printf 'account=%s partition=%s qos=%s\\n' "
        '"$ACCOUNT" "$PARTITION" "$QOS"\n'
        "printf 'shape=%s/%s c=%s mult=%s\\n' "
        '"$ISL_SHORT_VALUE" "$OSL_LONG_VALUE" "$BSIZES" "$MULT"\n'
        "printf 'profiler=%s ranks=%s\\n' "
        '"$SERVER_PROFILER" "$NTRACE_ROLLOUT_RANKS"\n',
    )
    fake_submit.chmod(0o755)
    _touch(bench_root / "vllm-ultra-ray-bench-serve-static.sh")
    _touch(bench_root / "benchmark_vllm_bench_serve_static.py")
    _touch(runtime / "ntrace" / "_cupti_cpp.fake.so")
    _touch(container, "pinned-image")

    env = {
        **os.environ,
        "BENCH_ROOT": str(bench_root),
        "VLLM_SOURCE_ROOT": str(REPO_ROOT),
        "CONTAINER_IMAGE": str(container),
        "NTRACE_RUNTIME": str(runtime),
        "RUN_ROOT_BASE": str(result_root),
        "STAMP": "test",
        "DRY_RUN": "1",
        "SBATCH_TEST_ONLY": "1",
    }
    completed = subprocess.run(
        ["bash", str(EXPERIMENT_DIR / "submit.sh")],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.count("shape=1000/256 c=8 mult=1") == 2
    scheduler_contract = "account=coreai_dlalgo_llm partition=batch qos=normal"
    assert completed.stdout.count(scheduler_contract) == 2
    assert completed.stdout.count("profiler=ntrace ranks=0") == 2
    assert "backend=triton" in completed.stdout
    assert "backend=flashinfer_trtllm" in completed.stdout

    for backend in ("triton", "flashinfer_trtllm"):
        metadata = result_root / f"test_decode_{backend}_c8" / "metadata.env"
        assert metadata.is_file()
        text = metadata.read_text()
        assert f"moe_backend={backend}" in text
        assert "precision=bf16" in text
        assert "tp=8" in text
        assert "cuda_graph=FULL_AND_PIECEWISE" in text


def test_runtime_build_uses_oci_srun_container(tmp_path: Path) -> None:
    source = tmp_path / "ntrace"
    runtime = tmp_path / "runtime"
    container = tmp_path / "vllm.sqsh"
    patch = tmp_path / "ntrace.patch"
    inner_script = tmp_path / "build_inner.sh"
    source.mkdir()
    _touch(container)
    _touch(patch)
    _touch(inner_script, "#!/usr/bin/env bash\n")

    env = {
        **os.environ,
        "NTRACE_SOURCE": str(source),
        "NTRACE_RUNTIME": str(runtime),
        "NTRACE_PATCH": str(patch),
        "NTRACE_REVISION": "165ae08",
        "CONTAINER_IMAGE": str(container),
        "BUILD_INNER_SCRIPT": str(inner_script),
        "LOG_DIR": str(tmp_path / "logs"),
        "DRY_RUN": "1",
        "SBATCH_TEST_ONLY": "1",
    }
    completed = subprocess.run(
        ["bash", str(EXPERIMENT_DIR / "build_runtime_oci_hsg.sh")],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert completed.returncode == 0, completed.stderr
    assert "sbatch" in completed.stdout
    assert "--test-only" in completed.stdout
    assert "srun" in completed.stdout
    rendered = completed.stdout.replace("\\", "")
    assert f"--container-image={container}" in rendered
    assert "--container-mounts=/home:/home,/lustre:/lustre" in rendered
    assert "--gpus-per-node=1" in rendered
