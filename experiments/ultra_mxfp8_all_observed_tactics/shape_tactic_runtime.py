# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Trace MXFP8 GEMM shapes and apply an exact-shape tactic lookup."""

from __future__ import annotations

import json
import math
import os
import socket
import sys
import threading
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


@dataclass(frozen=True)
class LookupEntry:
    runner: str
    tactic: Any


class TacticLookup:
    """Exact `(M, N, K, runner)` mapping produced by offline profiling."""

    def __init__(self, entries: dict[tuple[int, int, int, str], LookupEntry]):
        self._entries = entries

    @classmethod
    def load(cls, path: Path) -> TacticLookup:
        payload = json.loads(path.read_text())
        if payload.get("format_version") != 1:
            raise ValueError(f"unsupported lookup format: {path}")

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
        self._path = directory / f"trace.{os.getpid()}.jsonl"
        self._phase = phase
        self._seen: set[tuple[Shape, str, str]] = set()
        self._lock = threading.Lock()

    def record(
        self,
        shape: Shape,
        runner: Any,
        tactic: Any,
        selection_source: str,
    ) -> None:
        runner_name = runner.__class__.__name__
        key = (shape, runner_name, selection_source)
        with self._lock:
            if key in self._seen:
                return
            self._seen.add(key)
            m, n, k = shape
            row = {
                "m": m,
                "n": n,
                "k": k,
                "runner": runner_name,
                "tactic": _jsonable(tactic),
                "selection_source": selection_source,
                "phase": self._phase,
                "pid": os.getpid(),
                "rank": os.getenv("RANK", os.getenv("SLURM_PROCID", "unknown")),
                "host": socket.gethostname(),
            }
            with self._path.open("a") as handle:
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")


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
    lookup = TacticLookup.load(Path(lookup_path_raw)) if lookup_path_raw else None
    AutoTuner.choose_one = make_dispatcher(AutoTuner.choose_one, lookup, trace)
    _PATCHED = True
    print(
        "MXFP8 exact-shape runtime hook enabled "
        f"(trace={bool(trace)}, lookup={bool(lookup)})",
        file=sys.stderr,
    )
