# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.ultra_mxfp8_all_observed_tactics.shape_tactic_runtime import (
    ShapeTrace,
    TacticLookup,
    extract_mnk,
    make_dispatcher,
    restore_tactic,
)


class FakeTensor:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape


class CuteRunner:
    pass


class TrtRunner:
    pass


class FakeTuner:
    def __init__(self, *, is_tuning_mode: bool) -> None:
        self.is_tuning_mode = is_tuning_mode


def _wait_for_path(path: Path, timeout_s: float = 2.0) -> None:
    deadline = time.monotonic() + timeout_s
    while not path.is_file() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert path.is_file()


def _wait_for_content(path: Path, expected: str, timeout_s: float = 2.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if path.is_file() and path.read_text().strip() == expected:
            return
        time.sleep(0.01)
    assert path.is_file()
    assert path.read_text().strip() == expected


def _artifact_path(root: Path, kind: str, trace: ShapeTrace, suffix: str) -> Path:
    return root / f"{kind}.{trace.process_id}.{suffix}"


def test_resolve_rank_prefers_initialized_distributed_rank(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    distributed = SimpleNamespace(
        is_available=lambda: True,
        is_initialized=lambda: True,
        get_rank=lambda: 3,
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(distributed=distributed))
    monkeypatch.setenv("RANK", "0")
    trace = ShapeTrace(tmp_path, "baseline")

    trace.record((1, 2304, 8192), CuteRunner(), 7, "default_autotuner")

    row = json.loads(_artifact_path(tmp_path, "trace", trace, "jsonl").read_text())
    assert row["rank"] == "3"


def test_shape_trace_counts_repeated_dispatches_without_repeating_trace_rows(
    tmp_path: Path,
) -> None:
    trace = ShapeTrace(tmp_path, "baseline")
    runner = CuteRunner()

    trace.record((1001, 2304, 8192), runner, 7, "default_autotuner")
    trace.record((1001, 2304, 8192), runner, 7, "default_autotuner")
    trace.record((1001, 2304, 8192), runner, 7, "default_autotuner")
    trace.finalize("test")

    trace_rows = (
        _artifact_path(tmp_path, "trace", trace, "jsonl").read_text().splitlines()
    )
    count_rows = (
        _artifact_path(tmp_path, "counts", trace, "jsonl").read_text().splitlines()
    )
    assert len(trace_rows) == 1
    assert len(count_rows) == 1
    count = json.loads(count_rows[0])
    assert count["invocation_count"] == 3
    assert count["first_invocation_index"] == 1
    assert count["last_invocation_index"] == 3
    assert _artifact_path(tmp_path, "counts", trace, "complete").is_file()


def test_shape_trace_orders_fallback_before_tuned_tactic(tmp_path: Path) -> None:
    trace = ShapeTrace(tmp_path, "baseline")
    runner = CuteRunner()

    trace.record((2, 2048, 8192), runner, -1, "default_autotuner")
    trace.record((2, 2048, 8192), runner, 5, "default_autotuner")
    trace.record((2, 2048, 8192), runner, 5, "default_autotuner")
    trace.finalize("test")

    rows = [
        json.loads(line)
        for line in _artifact_path(tmp_path, "counts", trace, "jsonl")
        .read_text()
        .splitlines()
    ]
    by_tactic = {row["tactic"]: row for row in rows}
    assert by_tactic[-1]["first_invocation_index"] == 1
    assert by_tactic[-1]["last_invocation_index"] == 1
    assert by_tactic[5]["first_invocation_index"] == 2
    assert by_tactic[5]["last_invocation_index"] == 3


def test_shape_trace_acknowledges_each_snapshot_generation(
    tmp_path: Path,
) -> None:
    trace = ShapeTrace(tmp_path, "baseline")
    runner = CuteRunner()
    complete_path = _artifact_path(tmp_path, "counts", trace, "complete")
    request_path = _artifact_path(tmp_path, "flush", trace, "request")

    trace.record((2, 2048, 8192), runner, 5, "default_autotuner")
    request_path.write_text("first\n")
    _wait_for_content(complete_path, "first")

    trace.record((2, 2048, 8192), runner, 5, "default_autotuner")
    request_path.write_text("second\n")
    _wait_for_content(complete_path, "second")
    count = json.loads(_artifact_path(tmp_path, "counts", trace, "jsonl").read_text())
    assert count["invocation_count"] == 2


def test_shape_trace_does_not_drop_request_replaced_during_snapshot_read(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    trace = ShapeTrace(tmp_path, "baseline")
    trace.record((2, 2048, 8192), CuteRunner(), 5, "default_autotuner")
    request_path = _artifact_path(tmp_path, "flush", trace, "request")
    complete_path = _artifact_path(tmp_path, "counts", trace, "complete")
    original_read_text = Path.read_text
    replacement_sent = False

    def read_and_replace(path: Path, *args, **kwargs) -> str:
        nonlocal replacement_sent
        content = original_read_text(path, *args, **kwargs)
        if path.name.endswith(".processing") and not replacement_sent:
            replacement_sent = True
            request_path.write_text("second\n")
        return content

    monkeypatch.setattr(Path, "read_text", read_and_replace)
    request_path.write_text("first\n")

    _wait_for_content(complete_path, "second")
    assert replacement_sent


def test_exit_finalize_preserves_request_acknowledgement(tmp_path: Path) -> None:
    trace = ShapeTrace(tmp_path, "baseline")
    trace.record((2, 2048, 8192), CuteRunner(), 5, "default_autotuner")

    assert trace.finalize("request-token")
    assert trace.finalize()

    complete_path = _artifact_path(tmp_path, "counts", trace, "complete")
    assert complete_path.read_text().strip() == "request-token"


def test_shape_trace_flush_request_does_not_reenter_snapshot_write(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    trace = ShapeTrace(tmp_path, "baseline")
    trace.record((2, 2048, 8192), CuteRunner(), 5, "default_autotuner")
    original_replace = Path.replace
    request_sent = False

    def replace_and_request(path: Path, target: Path) -> Path:
        nonlocal request_sent
        if not request_sent:
            request_sent = True
            _artifact_path(tmp_path, "flush", trace, "request").write_text(
                "during-flush\n"
            )
            time.sleep(0.05)
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", replace_and_request)

    trace.flush()

    _wait_for_content(
        _artifact_path(tmp_path, "counts", trace, "complete"), "during-flush"
    )


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires fork")
def test_shape_trace_resets_process_local_state_after_fork(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import os, sys\n"
                "from pathlib import Path\n"
                "from experiments.ultra_mxfp8_all_observed_tactics."
                "shape_tactic_runtime import ShapeTrace\n"
                "trace = ShapeTrace(Path(sys.argv[1]), 'baseline')\n"
                "parent_pid = trace.pid\n"
                "child_pid = os.fork()\n"
                "if child_pid == 0:\n"
                "    trace.record((2, 2048, 8192), object(), 5, "
                "'default_autotuner')\n"
                "    trace.finalize('test')\n"
                "    os._exit(0)\n"
                "_, status = os.waitpid(child_pid, 0)\n"
                "print(parent_pid, child_pid, os.waitstatus_to_exitcode(status))\n"
            ),
            str(tmp_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    parent_pid, child_pid, exit_code = map(int, result.stdout.split())
    assert exit_code == 0
    assert child_pid != parent_pid
    traces = list(tmp_path.glob(f"trace.*.{child_pid}.jsonl"))
    completions = list(tmp_path.glob(f"counts.*.{child_pid}.complete"))
    assert len(traces) == 1
    assert len(completions) == 1
    assert not list(tmp_path.glob(f"trace.*.{parent_pid}.jsonl"))


def test_shape_trace_uses_host_and_pid_in_artifact_names(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("socket.gethostname", lambda: "gb200/node:7")

    trace = ShapeTrace(tmp_path, "baseline")
    trace.record((2, 2048, 8192), CuteRunner(), 5, "default_autotuner")
    trace.finalize()

    assert trace.process_id == f"gb200_node_7.{trace.pid}"
    assert _artifact_path(tmp_path, "trace", trace, "jsonl").is_file()
    complete = _artifact_path(tmp_path, "counts", trace, "complete")
    assert complete.read_text().strip() == "atexit"


def test_extract_mnk_flattens_all_activation_batch_dimensions() -> None:
    inputs = [FakeTensor((7, 11, 8192)), FakeTensor((8192, 2304))]

    assert extract_mnk(inputs) == (77, 2304, 8192)


def test_extract_mnk_rejects_weight_with_incompatible_k() -> None:
    inputs = [FakeTensor((1001, 8192)), FakeTensor((4096, 2304))]

    with pytest.raises(ValueError, match="incompatible K"):
        extract_mnk(inputs)


def test_restore_tactic_recovers_nested_tuple_structure() -> None:
    serialized = [[128, 32], [1, 1], True, False, 1]

    assert restore_tactic(serialized) == (
        (128, 32),
        (1, 1),
        True,
        False,
        1,
    )


def test_lookup_returns_only_exact_shape_and_runner_match(tmp_path: Path) -> None:
    lookup_path = tmp_path / "lookup.json"
    lookup_path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "backend": "cute-dsl",
                "scale_layout": "128x4",
                "entries": [
                    {
                        "m": 1001,
                        "n": 2304,
                        "k": 8192,
                        "runner": "CuteRunner",
                        "tactic": [[128, 32], [1, 1], True, False, 1],
                    }
                ],
            }
        )
    )
    lookup = TacticLookup.load(lookup_path)

    hit = lookup.choose((1001, 2304, 8192), [TrtRunner(), CuteRunner()])
    assert hit is not None
    assert isinstance(hit[0], CuteRunner)
    assert hit[1] == ((128, 32), (1, 1), True, False, 1)
    assert lookup.choose((1002, 2304, 8192), [CuteRunner()]) is None
    assert lookup.choose((1001, 2304, 8192), [TrtRunner()]) is None


def test_lookup_rejects_duplicate_shape_entries(tmp_path: Path) -> None:
    lookup_path = tmp_path / "lookup.json"
    entry = {
        "m": 1001,
        "n": 2304,
        "k": 8192,
        "runner": "CuteRunner",
        "tactic": 7,
    }
    lookup_path.write_text(json.dumps({"format_version": 1, "entries": [entry, entry]}))

    with pytest.raises(ValueError, match="duplicate lookup entry"):
        TacticLookup.load(lookup_path)


def test_lookup_rejects_backend_or_layout_mismatch(tmp_path: Path) -> None:
    lookup_path = tmp_path / "lookup.json"
    lookup_path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "backend": "trtllm",
                "scale_layout": "8x4",
                "entries": [
                    {"m": 1, "n": 2304, "k": 8192, "runner": "TrtRunner", "tactic": 7}
                ],
            }
        )
    )

    with pytest.raises(ValueError, match="lookup backend mismatch"):
        TacticLookup.load(
            lookup_path, expected_backend="cute-dsl", expected_scale_layout="128x4"
        )


def test_lookup_rejects_flashinfer_commit_mismatch(tmp_path: Path) -> None:
    lookup_path = tmp_path / "lookup.json"
    lookup_path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "backend": "trtllm",
                "scale_layout": "8x4",
                "flashinfer_commit": "def456",
                "entries": [
                    {"m": 1, "n": 2304, "k": 8192, "runner": "TrtRunner", "tactic": 7}
                ],
            }
        )
    )

    with pytest.raises(ValueError, match="FlashInfer commit mismatch"):
        TacticLookup.load(
            lookup_path,
            expected_backend="trtllm",
            expected_scale_layout="8x4",
            expected_flashinfer_commit="different",
        )


def test_lookup_rejects_flashinfer_runtime_mismatch(tmp_path: Path) -> None:
    lookup_path = tmp_path / "lookup.json"
    lookup_path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "flashinfer_version": "0.6.18",
                "flashinfer_file": "/installed/flashinfer/__init__.py",
                "container_sha256": "container-sha",
                "entries": [
                    {"m": 1, "n": 2304, "k": 8192, "runner": "TrtRunner", "tactic": 7}
                ],
            }
        )
    )

    with pytest.raises(ValueError, match="FlashInfer version mismatch"):
        TacticLookup.load(lookup_path, expected_flashinfer_version="0.6.17")
    with pytest.raises(ValueError, match="FlashInfer file mismatch"):
        TacticLookup.load(
            lookup_path,
            expected_flashinfer_file="/other/flashinfer/__init__.py",
        )
    with pytest.raises(ValueError, match="container SHA256 mismatch"):
        TacticLookup.load(lookup_path, expected_container_sha256="different")


def test_lookup_rejects_gpu_mismatch(tmp_path: Path) -> None:
    lookup_path = tmp_path / "lookup.json"
    lookup_path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "gpu": "NVIDIA GB200",
                "entries": [],
            }
        )
    )

    with pytest.raises(ValueError, match="lookup GPU mismatch"):
        TacticLookup.load(lookup_path, expected_gpu="NVIDIA H100")


def test_dispatcher_uses_lookup_hit_without_calling_default(tmp_path: Path) -> None:
    lookup_path = tmp_path / "lookup.json"
    lookup_path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "backend": "cute-dsl",
                "scale_layout": "128x4",
                "entries": [
                    {
                        "m": 1001,
                        "n": 2304,
                        "k": 8192,
                        "runner": "CuteRunner",
                        "tactic": 17,
                    }
                ],
            }
        )
    )
    default_calls = 0

    def default(*args, **kwargs):
        nonlocal default_calls
        default_calls += 1
        return TrtRunner(), -1

    dispatch = make_dispatcher(default, TacticLookup.load(lookup_path), None)
    runner, tactic = dispatch(
        object(),
        "mxfp8_gemm",
        [CuteRunner(), TrtRunner()],
        object(),
        [FakeTensor((1001, 8192)), FakeTensor((8192, 2304))],
    )

    assert isinstance(runner, CuteRunner)
    assert tactic == 17
    assert default_calls == 0


def test_dispatcher_delegates_lookup_miss_to_default(tmp_path: Path) -> None:
    lookup_path = tmp_path / "lookup.json"
    lookup_path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "backend": "cute-dsl",
                "scale_layout": "128x4",
                "entries": [
                    {
                        "m": 1001,
                        "n": 2304,
                        "k": 8192,
                        "runner": "CuteRunner",
                        "tactic": 17,
                    }
                ],
            }
        )
    )
    default_runner = TrtRunner()

    def default(*args, **kwargs):
        return default_runner, -1

    dispatch = make_dispatcher(default, TacticLookup.load(lookup_path), None)

    assert dispatch(
        object(),
        "mxfp8_gemm",
        [CuteRunner(), TrtRunner()],
        object(),
        [FakeTensor((1002, 8192)), FakeTensor((8192, 2304))],
    ) == (default_runner, -1)


def test_dispatcher_does_not_trace_autotuner_profiling_calls(tmp_path: Path) -> None:
    trace = ShapeTrace(tmp_path, "baseline")
    runner = CuteRunner()

    def default(*args, **kwargs):
        return runner, 7

    dispatch = make_dispatcher(default, None, trace)
    selected = dispatch(
        FakeTuner(is_tuning_mode=True),
        "mxfp8_gemm",
        [runner],
        object(),
        [FakeTensor((1024, 8192)), FakeTensor((8192, 2304))],
    )
    trace.finalize()

    assert selected == (runner, 7)
    assert not _artifact_path(tmp_path, "trace", trace, "jsonl").exists()
    assert not _artifact_path(tmp_path, "counts", trace, "jsonl").exists()


def test_dispatcher_delegates_lookup_hit_during_autotuner_profiling(
    tmp_path: Path,
) -> None:
    lookup_path = tmp_path / "lookup.json"
    lookup_path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "entries": [
                    {
                        "m": 1024,
                        "n": 2304,
                        "k": 8192,
                        "runner": "CuteRunner",
                        "tactic": 17,
                    }
                ],
            }
        )
    )
    default_runner = TrtRunner()
    default_calls = 0

    def default(*args, **kwargs):
        nonlocal default_calls
        default_calls += 1
        return default_runner, 23

    dispatch = make_dispatcher(default, TacticLookup.load(lookup_path), None)
    selected = dispatch(
        FakeTuner(is_tuning_mode=True),
        "mxfp8_gemm",
        [CuteRunner(), default_runner],
        object(),
        [FakeTensor((1024, 8192)), FakeTensor((8192, 2304))],
    )

    assert selected == (default_runner, 23)
    assert default_calls == 1
