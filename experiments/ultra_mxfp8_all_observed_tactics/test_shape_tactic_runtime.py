# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import sys
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

    row = json.loads((tmp_path / f"trace.{trace.pid}.jsonl").read_text())
    assert row["rank"] == "3"


def test_shape_trace_counts_repeated_dispatches_without_repeating_trace_rows(
    tmp_path: Path,
) -> None:
    trace = ShapeTrace(tmp_path, "baseline")
    runner = CuteRunner()

    trace.record((1001, 2304, 8192), runner, 7, "default_autotuner")
    trace.record((1001, 2304, 8192), runner, 7, "default_autotuner")
    trace.record((1001, 2304, 8192), runner, 7, "default_autotuner")
    trace.finalize()

    trace_rows = (tmp_path / f"trace.{trace.pid}.jsonl").read_text().splitlines()
    count_rows = (tmp_path / f"counts.{trace.pid}.jsonl").read_text().splitlines()
    assert len(trace_rows) == 1
    assert len(count_rows) == 1
    count = json.loads(count_rows[0])
    assert count["invocation_count"] == 3
    assert count["first_invocation_index"] == 1
    assert count["last_invocation_index"] == 3
    assert (tmp_path / f"counts.{trace.pid}.complete").is_file()


def test_shape_trace_orders_fallback_before_tuned_tactic(tmp_path: Path) -> None:
    trace = ShapeTrace(tmp_path, "baseline")
    runner = CuteRunner()

    trace.record((2, 2048, 8192), runner, -1, "default_autotuner")
    trace.record((2, 2048, 8192), runner, 5, "default_autotuner")
    trace.record((2, 2048, 8192), runner, 5, "default_autotuner")
    trace.finalize()

    rows = [
        json.loads(line)
        for line in (tmp_path / f"counts.{trace.pid}.jsonl").read_text().splitlines()
    ]
    by_tactic = {row["tactic"]: row for row in rows}
    assert by_tactic[-1]["first_invocation_index"] == 1
    assert by_tactic[-1]["last_invocation_index"] == 1
    assert by_tactic[5]["first_invocation_index"] == 2
    assert by_tactic[5]["last_invocation_index"] == 3


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
    assert not (tmp_path / f"trace.{trace.pid}.jsonl").exists()
    assert not (tmp_path / f"counts.{trace.pid}.jsonl").exists()


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
