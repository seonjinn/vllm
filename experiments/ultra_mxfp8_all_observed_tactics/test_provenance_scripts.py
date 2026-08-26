# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

EXPERIMENT_DIR = Path(__file__).parent
SERVER_ENTRYPOINT = EXPERIMENT_DIR / "server_entrypoint.sh"
PREPARE_ORACLE_INPUTS = EXPERIMENT_DIR / "prepare_oracle_inputs.sh"
RUN_PAIR = EXPERIMENT_DIR / "run_pair.sbatch"
RUN_SERVER = EXPERIMENT_DIR / "run_server.sbatch"
SUBMIT_PIPELINE = EXPERIMENT_DIR / "submit_pipeline.sh"


def test_request_trace_flush_finalizes_live_worker_without_stopping_it(
    tmp_path: Path,
) -> None:
    worker = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import os, signal, sys\n"
                "from pathlib import Path\n"
                "from experiments.ultra_mxfp8_all_observed_tactics."
                "shape_tactic_runtime import ShapeTrace\n"
                "trace = ShapeTrace(Path(sys.argv[1]), 'baseline')\n"
                "signal.signal(signal.SIGUSR1, signal.SIG_IGN)\n"
                "trace.record((2, 2048, 8192), object(), 5, "
                "'default_autotuner')\n"
                "print(os.getpid(), flush=True)\n"
                "while True:\n"
                "    signal.pause()\n"
            ),
            str(tmp_path),
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert worker.stdout is not None
        pid = int(worker.stdout.readline())
        deadline = time.monotonic() + 2
        trace_path = tmp_path / f"trace.{pid}.jsonl"
        while not trace_path.is_file() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert trace_path.is_file()

        result = subprocess.run(
            [
                "bash",
                str(SERVER_ENTRYPOINT),
                "--request-trace-flush",
                str(tmp_path),
                "2",
            ],
            check=False,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, result.stderr
        assert (tmp_path / f"counts.{pid}.complete").is_file()
        count = json.loads((tmp_path / f"counts.{pid}.jsonl").read_text())
        assert count["invocation_count"] == 1
        assert worker.poll() is None
    finally:
        worker.terminate()
        worker.wait(timeout=5)


def test_request_trace_flush_rejects_stale_completion_marker(
    tmp_path: Path,
) -> None:
    worker = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import os, signal, sys\n"
                "from pathlib import Path\n"
                "root = Path(sys.argv[1])\n"
                "pid = os.getpid()\n"
                "(root / f'trace.{pid}.jsonl').write_text('{}\\n')\n"
                "(root / f'counts.{pid}.jsonl').write_text('{}\\n')\n"
                "(root / f'counts.{pid}.complete').touch()\n"
                "signal.signal(signal.SIGUSR1, signal.SIG_IGN)\n"
                "print(pid, flush=True)\n"
                "while True:\n"
                "    signal.pause()\n"
            ),
            str(tmp_path),
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert worker.stdout is not None
        pid = int(worker.stdout.readline())
        result = subprocess.run(
            [
                "bash",
                str(SERVER_ENTRYPOINT),
                "--request-trace-flush",
                str(tmp_path),
                "1",
            ],
            check=False,
            capture_output=True,
            text=True,
        )

        assert result.returncode != 0
        assert f"{pid}" in result.stderr
        assert "Timed out" in result.stderr
    finally:
        worker.terminate()
        worker.wait(timeout=5)


def _record_cudagraph_evidence(
    tmp_path: Path,
    *,
    run_kind: str,
    enforce_eager: bool,
    server_log: str,
) -> tuple[subprocess.CompletedProcess[str], str]:
    log_path = tmp_path / "server.log"
    metadata_path = tmp_path / "metadata.txt"
    log_path.write_text(server_log)
    metadata_path.write_text(f"run_kind={run_kind}\n")

    result = subprocess.run(
        [
            "bash",
            str(SERVER_ENTRYPOINT),
            "--record-cudagraph-evidence",
            run_kind,
            "1" if enforce_eager else "0",
            str(log_path),
            str(metadata_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    return result, metadata_path.read_text()


@pytest.mark.parametrize("run_kind", ["baseline", "lookup"])
def test_graph_serving_runs_require_capture_completion_marker(
    tmp_path: Path, run_kind: str
) -> None:
    result, metadata = _record_cudagraph_evidence(
        tmp_path,
        run_kind=run_kind,
        enforce_eager=False,
        server_log="server became healthy\n",
    )

    assert result.returncode != 0
    assert "Graph capturing finished" in result.stderr
    assert "cudagraph_configured=true" in metadata
    assert "cudagraph_capture_status=configured_not_observed" in metadata
    assert "cudagraph_capture_evidence=none" in metadata
    assert "executed" not in metadata


@pytest.mark.parametrize("run_kind", ["baseline", "lookup"])
def test_graph_serving_runs_record_capture_completion_marker(
    tmp_path: Path, run_kind: str
) -> None:
    result, metadata = _record_cudagraph_evidence(
        tmp_path,
        run_kind=run_kind,
        enforce_eager=False,
        server_log=(
            "INFO worker: Graph capturing finished in 12 secs, took 3.50 GiB\n"
        ),
    )

    assert result.returncode == 0, result.stderr
    assert "cudagraph_configured=true" in metadata
    assert "cudagraph_capture_status=capture_completed" in metadata
    assert "cudagraph_capture_evidence=server_log_completion_marker" in metadata
    assert "cudagraph_capture_marker=Graph capturing finished" in metadata


def test_eager_run_records_graph_capture_as_disabled(tmp_path: Path) -> None:
    result, metadata = _record_cudagraph_evidence(
        tmp_path,
        run_kind="capture-eager",
        enforce_eager=True,
        server_log="server became healthy\n",
    )

    assert result.returncode == 0, result.stderr
    assert "cudagraph_configured=false" in metadata
    assert "cudagraph_capture_status=disabled_eager" in metadata
    assert "cudagraph_capture_evidence=not_applicable" in metadata


def test_configured_graph_capture_without_marker_fails_closed(
    tmp_path: Path,
) -> None:
    result, metadata = _record_cudagraph_evidence(
        tmp_path,
        run_kind="capture-graph",
        enforce_eager=False,
        server_log="server became healthy\n",
    )

    assert result.returncode != 0
    assert "Graph capturing finished" in result.stderr
    assert "cudagraph_configured=true" in metadata
    assert "cudagraph_capture_status=configured_not_observed" in metadata
    assert "cudagraph_capture_evidence=none" in metadata
    assert "executed" not in metadata


def _write_completed_run(
    result_root: Path,
    phase: str,
    job_id: str,
    capture_status: str,
    *,
    server_log: str | None = None,
) -> None:
    run_dir = result_root / "serving" / phase / job_id
    run_dir.mkdir(parents=True)
    configured = "false" if capture_status == "disabled_eager" else "true"
    evidence = {
        "capture_completed": "server_log_completion_marker",
        "configured_not_observed": "none",
        "disabled_eager": "not_applicable",
    }[capture_status]
    (run_dir / "metadata.txt").write_text(
        f"run_kind={phase}\n"
        f"cudagraph_configured={configured}\n"
        f"cudagraph_capture_status={capture_status}\n"
        f"cudagraph_capture_evidence={evidence}\n"
        "cudagraph_capture_marker=Graph capturing finished\n"
    )
    if server_log is None:
        server_log = (
            "Graph capturing finished in 12 secs\n"
            if capture_status == "capture_completed"
            else "server became healthy\n"
        )
    (run_dir / "server.log").write_text(server_log)
    (run_dir / "COMPLETE").touch()


def _validate_oracle_metadata(
    result_root: Path,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "bash",
            str(PREPARE_ORACLE_INPUTS),
            "--validate-cudagraph-metadata",
            str(result_root),
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def test_prepare_oracle_accepts_completed_graph_capture_and_disabled_eager(
    tmp_path: Path,
) -> None:
    _write_completed_run(tmp_path, "capture-graph", "11", "capture_completed")
    _write_completed_run(tmp_path, "capture-eager", "12", "disabled_eager")

    result = _validate_oracle_metadata(tmp_path)

    assert result.returncode == 0, result.stderr


def test_prepare_oracle_rejects_graph_capture_without_runtime_evidence(
    tmp_path: Path,
) -> None:
    _write_completed_run(tmp_path, "capture-graph", "11", "configured_not_observed")
    _write_completed_run(tmp_path, "capture-eager", "12", "disabled_eager")

    result = _validate_oracle_metadata(tmp_path)

    assert result.returncode != 0
    assert "capture-graph" in result.stderr
    assert "capture_completed" in result.stderr


def test_prepare_oracle_rechecks_graph_capture_server_log_marker(
    tmp_path: Path,
) -> None:
    _write_completed_run(
        tmp_path,
        "capture-graph",
        "11",
        "capture_completed",
        server_log="server became healthy\n",
    )
    _write_completed_run(tmp_path, "capture-eager", "12", "disabled_eager")

    result = _validate_oracle_metadata(tmp_path)

    assert result.returncode != 0
    assert "Graph capturing finished" in result.stderr


@pytest.mark.parametrize(
    ("pair_order", "expected_kinds"),
    [
        ("baseline-lookup", ["baseline", "lookup"]),
        ("lookup-baseline", ["lookup", "baseline"]),
    ],
)
def test_pair_runner_keeps_both_orders_in_one_allocation(
    tmp_path: Path, pair_order: str, expected_kinds: list[str]
) -> None:
    invocation_log = tmp_path / "invocations.jsonl"
    stub = tmp_path / "run_server.sh"
    stub.write_text(
        "#!/usr/bin/env bash\n"
        "python3 - <<'PY'\n"
        "import json, os\n"
        "keys = ('RUN_KIND', 'RUN_INSTANCE', 'RESULT_ROOT', 'PAIR_ID', "
        "'PAIR_ORDER', 'PAIR_POSITION', 'LOOKUP_PATH')\n"
        "with open(os.environ['INVOCATION_LOG'], 'a') as handle:\n"
        "    handle.write(json.dumps({key: os.environ.get(key) for key in keys}) "
        "+ '\\n')\n"
        "PY\n"
    )
    stub.chmod(0o755)
    result_root = tmp_path / "results"
    lookup_path = result_root / "oracle" / "lookup.json"
    lookup_path.parent.mkdir(parents=True)
    lookup_path.write_text("{}\n")
    env = {
        **os.environ,
        "SLURM_JOB_ID": "42",
        "PAIR_ORDER": pair_order,
        "RESULT_ROOT": str(result_root),
        "LOOKUP_PATH": str(lookup_path),
        "RUN_SERVER_SCRIPT": str(stub),
        "ALLOW_RUN_SERVER_SCRIPT_TEST_HOOK": "1",
        "INVOCATION_LOG": str(invocation_log),
    }

    result = subprocess.run(
        ["bash", str(RUN_PAIR)],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    rows = [json.loads(line) for line in invocation_log.read_text().splitlines()]
    assert [row["RUN_KIND"] for row in rows] == expected_kinds
    assert [row["PAIR_POSITION"] for row in rows] == ["1", "2"]
    assert {row["PAIR_ORDER"] for row in rows} == {pair_order}
    assert {row["PAIR_ID"] for row in rows} == {f"42-{pair_order}"}
    assert {row["RESULT_ROOT"] for row in rows} == {
        str(result_root / "pairs" / pair_order)
    }
    assert [row["RUN_INSTANCE"] for row in rows] == [
        f"42-{pair_order}-1-{expected_kinds[0]}",
        f"42-{pair_order}-2-{expected_kinds[1]}",
    ]
    assert rows[expected_kinds.index("baseline")]["LOOKUP_PATH"] is None
    assert rows[expected_kinds.index("lookup")]["LOOKUP_PATH"] == str(lookup_path)


def test_pair_runner_rejects_unknown_order(tmp_path: Path) -> None:
    result = subprocess.run(
        ["bash", str(RUN_PAIR)],
        env={
            **os.environ,
            "SLURM_JOB_ID": "42",
            "PAIR_ORDER": "baseline-baseline",
            "RESULT_ROOT": str(tmp_path),
            "LOOKUP_PATH": str(tmp_path / "lookup.json"),
        },
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "PAIR_ORDER" in result.stderr


def test_pair_runner_rejects_unapproved_server_script_hook(tmp_path: Path) -> None:
    lookup_path = tmp_path / "lookup.json"
    lookup_path.write_text("{}\n")

    result = subprocess.run(
        ["bash", str(RUN_PAIR)],
        env={
            **os.environ,
            "SLURM_JOB_ID": "42",
            "PAIR_ORDER": "baseline-lookup",
            "RESULT_ROOT": str(tmp_path / "result"),
            "LOOKUP_PATH": str(lookup_path),
            "RUN_SERVER_SCRIPT": str(tmp_path / "untrusted.sh"),
        },
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "test hook" in result.stderr


def test_submit_pipeline_uses_order_balanced_pair_jobs() -> None:
    script = SUBMIT_PIPELINE.read_text()

    assert "PAIR_ORDER=baseline-lookup" in script
    assert "PAIR_ORDER=lookup-baseline" in script
    assert script.count("run_pair.sbatch") >= 3
    assert "pair_baseline_lookup_job=" in script
    assert "pair_lookup_baseline_job=" in script


def test_submit_pipeline_uses_full_graph_capture_phase() -> None:
    script = SUBMIT_PIPELINE.read_text()

    assert "RUN_KIND=capture-graph" in script
    assert "RUN_KIND=baseline" not in script


def test_submit_pipeline_binds_container_and_model_bytes() -> None:
    script = SUBMIT_PIPELINE.read_text()

    assert 'sha256sum "${CONTAINER_IMAGE}"' in script
    assert 'sha256sum "${MODEL_PATH}/config.json"' in script
    assert 'sha256sum "${MODEL_PATH}/model.safetensors.index.json"' in script
    assert "EXPECTED_CONTAINER_SIZE=" in script
    assert "EXPECTED_CONTAINER_MTIME=" in script
    assert "EXPECTED_MODEL_CONFIG_SHA256=" in script
    assert "EXPECTED_MODEL_INDEX_SHA256=" in script
    assert "EXPECTED_MODEL_WEIGHTS_MANIFEST_SHA256=" in script
    assert "RUN_SERVER_SCRIPT=" in script
    assert "ALLOW_RUN_SERVER_SCRIPT_TEST_HOOK=0" in script


def test_run_server_refuses_existing_run_directory() -> None:
    script = RUN_SERVER.read_text()

    assert 'if [[ -e "${RUN_DIR}" ]]' in script
    assert "Refusing to reuse run directory" in script


def test_lookup_manifest_is_validated_before_server_launch() -> None:
    script = SERVER_ENTRYPOINT.read_text()

    preflight = "validate_lookup_from_environment"
    server_launch = 'setsid "${server_env[@]}" "${server_cmd[@]}"'
    assert preflight in script
    assert script.index(preflight) < script.index(server_launch)
