# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Trace MXFP8 GEMM shapes and apply an exact-shape tactic lookup."""

from __future__ import annotations

import atexit
import json
import math
import os
import socket
import sys
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

Shape = tuple[int, int, int]


def _shape(value: Any) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is None:
        raise ValueError("MXFP8 input does not expose a shape")
    return tuple(int(dimension) for dimension in shape)


def extract_mnk(inputs: Sequence[Any]) -> Shape:
    """Return flattened activation M and GEMM N/K from FlashInfer inputs."""
    if len(inputs) < 2:
        raise ValueError("MXFP8 GEMM requires activation and weight inputs")
    activation = _shape(inputs[0])
    weight = _shape(inputs[1])
    if len(activation) < 2 or len(weight) != 2:
        raise ValueError(
            f"unsupported activation/weight shapes: {activation}, {weight}"
        )

    m = math.prod(activation[:-1])
    k = activation[-1]
    if weight[0] != k:
        raise ValueError(
            f"weight has incompatible K: activation={activation}, weight={weight}"
        )
    return m, weight[1], k


def restore_tactic(value: Any) -> Any:
    """Restore tuple-based FlashInfer tactics after JSON serialization."""
    if isinstance(value, list):
        return tuple(restore_tactic(item) for item in value)
    return value


def _jsonable(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def resolve_rank() -> str:
    """Return the initialized distributed rank, falling back to process env."""
    try:
        import torch

        distributed = torch.distributed
        if distributed.is_available() and distributed.is_initialized():
            return str(distributed.get_rank())
    except (AttributeError, ImportError):
        pass
    return os.getenv("RANK", os.getenv("SLURM_PROCID", "unknown"))


@dataclass(frozen=True)
class LookupEntry:
    runner: str
    tactic: Any


class TacticLookup:
    """Exact `(M, N, K, runner)` mapping produced by offline profiling."""

    def __init__(self, entries: dict[tuple[int, int, int, str], LookupEntry]):
        self._entries = entries

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        expected_backend: str | None = None,
        expected_scale_layout: str | None = None,
        expected_flashinfer_commit: str | None = None,
        expected_flashinfer_version: str | None = None,
        expected_flashinfer_file: str | None = None,
        expected_container_sha256: str | None = None,
        expected_gpu: str | None = None,
    ) -> TacticLookup:
        payload = json.loads(path.read_text())
        if payload.get("format_version") != 1:
            raise ValueError(f"unsupported lookup format: {path}")
        if expected_backend is not None and payload.get("backend") != expected_backend:
            raise ValueError(
                "lookup backend mismatch: "
                f"expected={expected_backend}, actual={payload.get('backend')}"
            )
        if (
            expected_scale_layout is not None
            and payload.get("scale_layout") != expected_scale_layout
        ):
            raise ValueError(
                "lookup scale-layout mismatch: "
                f"expected={expected_scale_layout}, "
                f"actual={payload.get('scale_layout')}"
            )
        if (
            expected_flashinfer_commit is not None
            and payload.get("flashinfer_commit") != expected_flashinfer_commit
        ):
            raise ValueError(
                "lookup FlashInfer commit mismatch: "
                f"expected={expected_flashinfer_commit}, "
                f"actual={payload.get('flashinfer_commit')}"
            )
        if (
            expected_flashinfer_version is not None
            and payload.get("flashinfer_version") != expected_flashinfer_version
        ):
            raise ValueError(
                "lookup FlashInfer version mismatch: "
                f"expected={expected_flashinfer_version}, "
                f"actual={payload.get('flashinfer_version')}"
            )
        if expected_flashinfer_file is not None:
            actual_flashinfer_file = payload.get("flashinfer_file")
            if (
                actual_flashinfer_file is None
                or Path(actual_flashinfer_file).resolve()
                != Path(expected_flashinfer_file).resolve()
            ):
                raise ValueError(
                    "lookup FlashInfer file mismatch: "
                    f"expected={expected_flashinfer_file}, "
                    f"actual={actual_flashinfer_file}"
                )
        if (
            expected_container_sha256 is not None
            and payload.get("container_sha256") != expected_container_sha256
        ):
            raise ValueError(
                "lookup container SHA256 mismatch: "
                f"expected={expected_container_sha256}, "
                f"actual={payload.get('container_sha256')}"
            )
        if expected_gpu is not None and payload.get("gpu") != expected_gpu:
            raise ValueError(
                "lookup GPU mismatch: "
                f"expected={expected_gpu}, actual={payload.get('gpu')}"
            )

        entries: dict[tuple[int, int, int, str], LookupEntry] = {}
        for row in payload.get("entries", []):
            key = (
                int(row["m"]),
                int(row["n"]),
                int(row["k"]),
                str(row["runner"]),
            )
            if key in entries:
                raise ValueError(f"duplicate lookup entry: {key}")
            entries[key] = LookupEntry(
                runner=key[-1], tactic=restore_tactic(row["tactic"])
            )
        if not entries:
            raise ValueError(f"lookup contains no entries: {path}")
        return cls(entries)

    def choose(self, shape: Shape, runners: Sequence[Any]) -> tuple[Any, Any] | None:
        m, n, k = shape
        for runner in runners:
            runner_name = runner.__class__.__name__
            entry = self._entries.get((m, n, k, runner_name))
            if entry is not None:
                return runner, entry.tactic
        return None


class ShapeTrace:
    """Append each unique per-process MXFP8 dispatch shape once."""

    def __init__(self, directory: Path, phase: str) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self._directory = directory
        self._phase = phase
        self._state_lock = threading.Lock()
        self._initialize_process_state(os.getpid())
        if hasattr(os, "register_at_fork"):
            os.register_at_fork(after_in_child=self._mark_fork_child)

    def _initialize_process_state(self, pid: int) -> None:
        self._path = self._directory / f"trace.{pid}.jsonl"
        self._count_path = self._directory / f"counts.{pid}.jsonl"
        self._complete_path = self._directory / f"counts.{pid}.complete"
        self._flush_request_path = self._directory / f"flush.{pid}.request"
        self._flush_processing_path = self._directory / f"flush.{pid}.processing"
        self._seen: set[tuple[Shape, str, str, str]] = set()
        self._counts: dict[tuple[Shape, str, str, str], dict[str, Any]] = {}
        self._invocation_index = 0
        self._calls_since_flush = 0
        self._rank: str | None = None
        self._lock = threading.Lock()
        self._snapshot_thread: threading.Thread | None = None
        self.pid = pid
        atexit.register(self.finalize)

    def _mark_fork_child(self) -> None:
        self._state_lock = threading.Lock()
        self._snapshot_thread = None
        self.pid = -1

    def _ensure_process_state(self) -> None:
        current_pid = os.getpid()
        if self.pid == current_pid:
            return
        with self._state_lock:
            if self.pid != current_pid:
                self._initialize_process_state(current_pid)

    def _ensure_snapshot_thread(self) -> None:
        thread = self._snapshot_thread
        if thread is not None and thread.is_alive():
            return
        with self._state_lock:
            if self._snapshot_thread is None or not self._snapshot_thread.is_alive():
                self._snapshot_thread = threading.Thread(
                    target=self._snapshot_loop,
                    args=(self.pid,),
                    name=f"mxfp8-trace-{self.pid}",
                    daemon=True,
                )
                self._snapshot_thread.start()

    def _snapshot_loop(self, owner_pid: int) -> None:
        last_token: str | None = None
        while self.pid == owner_pid and os.getpid() == owner_pid:
            try:
                self._flush_request_path.replace(self._flush_processing_path)
            except FileNotFoundError:
                time.sleep(0.05)
                continue
            token = self._flush_processing_path.read_text().strip()
            self._flush_processing_path.unlink(missing_ok=True)
            if token and token != last_token and self.finalize(token):
                last_token = token

    def _write_counts_locked(self) -> bool:
        rows = [
            {**record, "invocation_count": record["invocation_count"]}
            for _, record in sorted(self._counts.items(), key=lambda item: str(item[0]))
        ]
        if not rows:
            return False
        temporary = self._count_path.with_suffix(".tmp")
        temporary.write_text(
            "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows)
        )
        temporary.replace(self._count_path)
        self._calls_since_flush = 0
        return True

    def flush(self) -> None:
        self._ensure_process_state()
        with self._lock:
            self._write_counts_locked()

    def finalize(self, token: str | None = None) -> bool:
        self._ensure_process_state()
        with self._lock:
            if not self._write_counts_locked():
                return False
            if token is None:
                return True
            temporary = Path(f"{self._complete_path}.tmp")
            temporary.write_text(f"{token}\n")
            temporary.replace(self._complete_path)
            return True

    def record(
        self,
        shape: Shape,
        runner: Any,
        tactic: Any,
        selection_source: str,
    ) -> None:
        self._ensure_process_state()
        self._ensure_snapshot_thread()
        runner_name = runner.__class__.__name__
        tactic_json = json.dumps(_jsonable(tactic), separators=(",", ":"))
        key = (shape, runner_name, selection_source, tactic_json)
        with self._lock:
            self._invocation_index += 1
            record = self._counts.get(key)
            if record is None:
                if self._rank is None:
                    self._rank = resolve_rank()
                m, n, k = shape
                record = {
                    "m": m,
                    "n": n,
                    "k": k,
                    "runner": runner_name,
                    "tactic": _jsonable(tactic),
                    "selection_source": selection_source,
                    "phase": self._phase,
                    "pid": self.pid,
                    "rank": self._rank,
                    "host": socket.gethostname(),
                    "invocation_count": 0,
                    "first_invocation_index": self._invocation_index,
                    "last_invocation_index": self._invocation_index,
                }
                self._counts[key] = record
            record["invocation_count"] += 1
            record["last_invocation_index"] = self._invocation_index
            self._calls_since_flush += 1
            if key in self._seen:
                should_flush = self._calls_since_flush >= 4096
            else:
                self._seen.add(key)
                with self._path.open("a") as handle:
                    handle.write(json.dumps(record, separators=(",", ":")) + "\n")
                should_flush = self._calls_since_flush >= 4096
        if should_flush:
            self.flush()


_PATCHED = False


def make_dispatcher(
    original: Callable[..., tuple[Any, Any]],
    lookup: TacticLookup | None,
    trace: ShapeTrace | None,
) -> Callable[..., tuple[Any, Any]]:
    """Wrap `AutoTuner.choose_one` with exact lookup and tracing behavior."""

    def choose_one(
        self: Any,
        custom_op: str,
        runners: Sequence[Any],
        tuning_config: Any,
        inputs: Sequence[Any],
        **kwargs: Any,
    ) -> tuple[Any, Any]:
        if getattr(self, "is_tuning_mode", False):
            return original(self, custom_op, runners, tuning_config, inputs, **kwargs)

        shape = None
        if custom_op == "mxfp8_gemm":
            try:
                shape = extract_mnk(inputs)
            except ValueError:
                shape = None

        selected = lookup.choose(shape, runners) if lookup and shape else None
        selection_source = "offline_lookup" if selected else "default_autotuner"
        if selected is None:
            selected = original(
                self, custom_op, runners, tuning_config, inputs, **kwargs
            )
        if trace is not None and shape is not None:
            trace.record(shape, selected[0], selected[1], selection_source)
        return selected

    return choose_one


def validate_lookup_from_environment() -> TacticLookup | None:
    """Load and validate the opt-in lookup against the active runtime."""
    lookup_path_raw = os.getenv("MXFP8_TACTIC_LOOKUP")
    if not lookup_path_raw:
        return None

    import flashinfer

    required = {
        "MXFP8_TACTIC_BACKEND": os.getenv("MXFP8_TACTIC_BACKEND"),
        "MXFP8_TACTIC_SCALE_LAYOUT": os.getenv("MXFP8_TACTIC_SCALE_LAYOUT"),
        "FLASHINFER_COMMIT": os.getenv("FLASHINFER_COMMIT"),
        "EXPECTED_CONTAINER_SHA256": os.getenv("EXPECTED_CONTAINER_SHA256"),
        "MXFP8_TACTIC_GPU": os.getenv("MXFP8_TACTIC_GPU"),
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise ValueError(f"lookup provenance environment is missing: {missing}")
    return TacticLookup.load(
        Path(lookup_path_raw),
        expected_backend=required["MXFP8_TACTIC_BACKEND"],
        expected_scale_layout=required["MXFP8_TACTIC_SCALE_LAYOUT"],
        expected_flashinfer_commit=required["FLASHINFER_COMMIT"],
        expected_flashinfer_version=flashinfer.__version__,
        expected_flashinfer_file=str(Path(flashinfer.__file__).resolve()),
        expected_container_sha256=required["EXPECTED_CONTAINER_SHA256"],
        expected_gpu=required["MXFP8_TACTIC_GPU"],
    )


def install_from_environment() -> None:
    """Install the opt-in tracing and exact lookup dispatch wrapper."""
    global _PATCHED
    if _PATCHED:
        return

    trace_dir_raw = os.getenv("MXFP8_TACTIC_TRACE_DIR")
    lookup_path_raw = os.getenv("MXFP8_TACTIC_LOOKUP")
    if not trace_dir_raw and not lookup_path_raw:
        return

    from flashinfer.autotuner import AutoTuner

    trace = None
    if trace_dir_raw:
        trace = ShapeTrace(
            Path(trace_dir_raw), os.getenv("MXFP8_TACTIC_TRACE_PHASE", "unknown")
        )
    lookup = validate_lookup_from_environment()
    AutoTuner.choose_one = make_dispatcher(AutoTuner.choose_one, lookup, trace)
    _PATCHED = True
    print(
        "MXFP8 exact-shape runtime hook enabled "
        f"(trace={bool(trace)}, lookup={bool(lookup)})",
        file=sys.stderr,
    )
