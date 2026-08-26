# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import subprocess
import sys
import time
from pathlib import Path

import pytest

EXPERIMENT_DIR = Path(__file__).parent
SERVER_ENTRYPOINT = EXPERIMENT_DIR / "server_entrypoint.sh"
PREPARE_ORACLE_INPUTS = EXPERIMENT_DIR / "prepare_oracle_inputs.sh"


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
        assert worker.poll() is None
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


def test_configured_graph_capture_without_marker_is_not_called_executed(
    tmp_path: Path,
) -> None:
    result, metadata = _record_cudagraph_evidence(
        tmp_path,
        run_kind="capture-graph",
        enforce_eager=False,
        server_log="server became healthy\n",
    )

    assert result.returncode == 0, result.stderr
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


def test_prepare_oracle_accepts_completed_baseline_and_disabled_eager(
    tmp_path: Path,
) -> None:
    _write_completed_run(tmp_path, "baseline", "11", "capture_completed")
    _write_completed_run(tmp_path, "capture-eager", "12", "disabled_eager")

    result = _validate_oracle_metadata(tmp_path)

    assert result.returncode == 0, result.stderr


def test_prepare_oracle_rejects_baseline_without_runtime_evidence(
    tmp_path: Path,
) -> None:
    _write_completed_run(tmp_path, "baseline", "11", "configured_not_observed")
    _write_completed_run(tmp_path, "capture-eager", "12", "disabled_eager")

    result = _validate_oracle_metadata(tmp_path)

    assert result.returncode != 0
    assert "baseline" in result.stderr
    assert "capture_completed" in result.stderr


def test_prepare_oracle_rechecks_baseline_server_log_marker(tmp_path: Path) -> None:
    _write_completed_run(
        tmp_path,
        "baseline",
        "11",
        "capture_completed",
        server_log="server became healthy\n",
    )
    _write_completed_run(tmp_path, "capture-eager", "12", "disabled_eager")

    result = _validate_oracle_metadata(tmp_path)

    assert result.returncode != 0
    assert "Graph capturing finished" in result.stderr
