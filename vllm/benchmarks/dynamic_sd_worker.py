# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline worker for one vanilla or fixed-K Dynamic SD profile."""

from __future__ import annotations

import dataclasses
import importlib
import importlib.metadata
import json
import os
import platform
import socket
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Protocol

from vllm.benchmarks.dynamic_sd_core import ProfileIdentity, profile_id

WorkerStatus = Literal["complete", "infeasible", "failed"]


@dataclass(frozen=True, order=True)
class CUDAGraphStep:
    """One cumulative CUDA Graph runtime histogram key."""

    num_unpadded_tokens: int
    num_padded_tokens: int
    runtime_mode: str


@dataclass(frozen=True)
class WorkerMetricSnapshot:
    """Cumulative scheduler and CUDA Graph counters."""

    scheduler_steps: Mapping[tuple[int, int], int] = field(default_factory=dict)
    cudagraph_steps: Mapping[CUDAGraphStep, int] = field(default_factory=dict)


class WorkerMetricsReader(Protocol):
    """Reads cumulative metrics from one offline engine."""

    def snapshot(self) -> WorkerMetricSnapshot: ...


@dataclass(frozen=True)
class WorkerConfig:
    """Configuration for one vanilla or fixed-K worker process."""

    output_path: Path
    profile_identity: ProfileIdentity
    model: str
    model_revision: str
    speculative_model: str | None
    speculative_revision: str | None
    speculative_method: str
    k: int | None
    kmax: int
    scheduler_keys: tuple[int, ...]
    prompt_token_ids: tuple[tuple[int, ...], ...]
    max_tokens: int
    warmups: int
    repeats: int
    seed: int
    data_parallel_size: int
    cudagraph_capture_sizes: tuple[int, ...]
    engine_kwargs: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.model or not self.model_revision:
            raise ValueError("model and model_revision must not be empty.")
        if self.k is not None:
            if self.k < 0:
                raise ValueError("k must be non-negative or None for vanilla.")
            if not self.speculative_model or not self.speculative_revision:
                raise ValueError(
                    "Speculative workers require model and revision identities."
                )
        if self.kmax <= 0:
            raise ValueError("kmax must be positive.")
        if self.k is not None and self.k > self.kmax:
            raise ValueError("k must not exceed kmax.")
        if not self.scheduler_keys:
            raise ValueError("scheduler_keys must not be empty.")
        if len(set(self.scheduler_keys)) != len(self.scheduler_keys):
            raise ValueError("scheduler_keys must not contain duplicates.")
        if any(key <= 0 for key in self.scheduler_keys):
            raise ValueError("scheduler_keys must be positive.")
        if len(self.prompt_token_ids) < max(self.scheduler_keys):
            raise ValueError("prompt_token_ids must cover the largest scheduler key.")
        if any(not prompt for prompt in self.prompt_token_ids):
            raise ValueError("Every tokenized prompt must be non-empty.")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive.")
        if self.warmups < 0 or self.repeats <= 0:
            raise ValueError("warmups must be non-negative and repeats positive.")
        if not self.cudagraph_capture_sizes or any(
            size <= 0 for size in self.cudagraph_capture_sizes
        ):
            raise ValueError("cudagraph_capture_sizes must be positive.")
        if len(set(self.cudagraph_capture_sizes)) != len(self.cudagraph_capture_sizes):
            raise ValueError("cudagraph_capture_sizes must not contain duplicates.")
        reserved_engine_keys = {
            "compilation_config",
            "data_parallel_size",
            "disable_log_stats",
            "enforce_eager",
            "max_num_seqs",
            "model",
            "revision",
            "speculative_config",
        }
        conflicts = reserved_engine_keys.intersection(self.engine_kwargs)
        if conflicts:
            raise ValueError(
                "engine_kwargs contains worker-owned keys: "
                + ", ".join(sorted(conflicts))
            )


@dataclass(frozen=True)
class WorkerMeasurement:
    """One measured repeat for one requested scheduler key."""

    scheduler_key: int
    repeat: int
    elapsed_seconds: float
    prompt_tokens: int
    output_tokens: int
    output_tokens_per_second: float
    scheduler_key_coverage: float
    scheduler_key_histogram: list[dict[str, object]]
    cudagraph_histogram: list[dict[str, object]]
    cudagraph_fallback_histogram: list[dict[str, object]]


@dataclass(frozen=True)
class WorkerResult:
    """Atomic raw result emitted by one worker process."""

    status: WorkerStatus
    profile_id: str
    profile_identity: dict[str, object]
    variant: str
    forced_k: int | None
    common_kmax: int | None
    target_revision: str
    speculative_revision: str | None
    engine_kwargs: dict[str, object]
    runtime_identity: dict[str, object]
    resolved_engine_config: dict[str, object]
    measurements: tuple[WorkerMeasurement, ...]
    total_prompt_tokens: int
    total_output_tokens: int
    elapsed_seconds: float
    output_tokens_per_second: float | None
    error: str | None = None

    def to_payload(self) -> dict[str, object]:
        """Return a JSON-compatible raw result."""
        return _json_compatible(dataclasses.asdict(self))


class InfeasibleWorkerError(RuntimeError):
    """Marks a worker point as explicitly infeasible."""


def derive_cudagraph_capture_sizes(
    k_by_scheduler_key: Mapping[int, int],
) -> tuple[int, ...]:
    """Derive exact FULL capture rows for a fixed or dynamic K schedule.

    Dynamic SD's scheduler key counts requests, while a uniform decode CUDA
    Graph shape counts all input-token rows. A scheduler key ``B`` using ``K``
    speculative tokens therefore needs ``B * (K + 1)`` rows. Vanilla and
    runtime K=0 both reduce to ``B``.
    """
    if not k_by_scheduler_key:
        raise ValueError("A capture schedule must not be empty.")
    if any(key <= 0 for key in k_by_scheduler_key):
        raise ValueError("Capture schedule keys must be positive.")
    if any(k < 0 for k in k_by_scheduler_key.values()):
        raise ValueError("Capture schedule K values must be non-negative.")
    return tuple(sorted({key * (k + 1) for key, k in k_by_scheduler_key.items()}))


def run_worker(
    config: WorkerConfig,
    llm_factory: Callable[..., Any],
    *,
    metrics_reader_factory: Callable[[Any], WorkerMetricsReader] | None = None,
) -> WorkerResult:
    """Run one vanilla or forced-K profile and atomically write its result."""
    _clear_previous_result(config.output_path)
    _validate_preflight(config)
    engine_kwargs = _build_engine_kwargs(config)
    os.environ["VLLM_DYNAMIC_SD_PROFILE_METRICS"] = "1"

    try:
        result = _run_measurements(
            config,
            engine_kwargs,
            llm_factory,
            metrics_reader_factory or _DefaultMetricsReader,
        )
    except Exception as exc:
        status: WorkerStatus = (
            "infeasible" if isinstance(exc, InfeasibleWorkerError) else "failed"
        )
        write_json_atomic(
            config.output_path,
            _failure_payload(config, engine_kwargs, status, exc),
        )
        raise

    write_json_atomic(config.output_path, result.to_payload())
    return result


def _clear_previous_result(path: Path) -> None:
    path.unlink(missing_ok=True)
    path.with_suffix(path.suffix + ".tmp").unlink(missing_ok=True)


def write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    """Write one JSON result by replacing it from the same directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"
        )
        _replace_file(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _replace_file(source: Path, target: Path) -> None:
    source.replace(target)


def _validate_preflight(config: WorkerConfig) -> None:
    if config.data_parallel_size != 1:
        raise ValueError("data_parallel_size must be 1 for Dynamic SD profiling.")
    if os.environ.get("VLLM_USE_V2_MODEL_RUNNER") != "1":
        raise ValueError("Dynamic SD profiling requires VLLM_USE_V2_MODEL_RUNNER=1.")

    forced_k = config.k or 0
    expected_capture_sizes = derive_cudagraph_capture_sizes(
        {key: forced_k for key in config.scheduler_keys}
    )
    if tuple(sorted(config.cudagraph_capture_sizes)) != expected_capture_sizes:
        raise ValueError(
            "Explicit capture sizes do not match the forced-K token-row shapes: "
            f"expected {expected_capture_sizes}, got "
            f"{tuple(sorted(config.cudagraph_capture_sizes))}."
        )


def _build_engine_kwargs(config: WorkerConfig) -> dict[str, object]:
    engine_kwargs = dict(config.engine_kwargs)
    observability = engine_kwargs.get("observability_config", {})
    if not isinstance(observability, Mapping):
        raise ValueError("observability_config must be a mapping.")
    engine_kwargs.update(
        {
            "model": config.model,
            "revision": config.model_revision,
            "data_parallel_size": config.data_parallel_size,
            "max_num_seqs": max(config.scheduler_keys),
            "enforce_eager": False,
            "disable_log_stats": False,
            "observability_config": {
                **observability,
                "cudagraph_metrics": True,
            },
            "compilation_config": {
                "cudagraph_mode": "FULL_AND_PIECEWISE",
                "cudagraph_capture_sizes": list(sorted(config.cudagraph_capture_sizes)),
            },
        }
    )
    if config.k is not None:
        engine_kwargs["speculative_config"] = {
            "method": config.speculative_method,
            "model": config.speculative_model,
            "revision": config.speculative_revision,
            "num_speculative_tokens": config.kmax,
            "num_speculative_tokens_per_batch_size": [
                [1, max(config.scheduler_keys), config.k]
            ],
            "draft_sample_method": "probabilistic",
            "rejection_sample_method": "standard",
        }
    return engine_kwargs


def _run_measurements(
    config: WorkerConfig,
    engine_kwargs: dict[str, object],
    llm_factory: Callable[..., Any],
    metrics_reader_factory: Callable[[Any], WorkerMetricsReader],
) -> WorkerResult:
    llm = llm_factory(**engine_kwargs)
    resolved_config = _resolved_engine_config(llm)
    _validate_resolved_config(config, resolved_config)
    metrics_reader = metrics_reader_factory(llm)

    measurements: list[WorkerMeasurement] = []
    total_prompt_tokens = 0
    total_output_tokens = 0
    total_elapsed = 0.0

    for scheduler_key in config.scheduler_keys:
        prompts = [
            {"prompt_token_ids": list(token_ids)}
            for token_ids in config.prompt_token_ids[:scheduler_key]
        ]
        for warmup in range(config.warmups):
            llm.generate(
                prompts,
                _sampling_params(config, scheduler_key, warmup),
                use_tqdm=False,
            )

        for repeat in range(config.repeats):
            before = metrics_reader.snapshot()
            started_at = time.perf_counter()
            outputs = llm.generate(
                prompts,
                _sampling_params(config, scheduler_key, repeat),
                use_tqdm=False,
            )
            elapsed = time.perf_counter() - started_at
            after = metrics_reader.snapshot()
            if elapsed <= 0:
                raise ValueError("Measured elapsed time must be positive.")

            prompt_tokens, output_tokens = _validate_exact_work(
                prompts, outputs, config.max_tokens
            )
            metric_delta = _metric_delta(before, after)
            measurement = _build_measurement(
                config,
                scheduler_key,
                repeat,
                elapsed,
                prompt_tokens,
                output_tokens,
                metric_delta,
            )
            measurements.append(measurement)
            total_prompt_tokens += prompt_tokens
            total_output_tokens += output_tokens
            total_elapsed += elapsed

    return WorkerResult(
        status="complete",
        profile_id=profile_id(config.profile_identity),
        profile_identity=config.profile_identity.payload,
        variant="vanilla" if config.k is None else f"k{config.k}",
        forced_k=config.k,
        common_kmax=None if config.k is None else config.kmax,
        target_revision=config.model_revision,
        speculative_revision=(
            None if config.k is None else config.speculative_revision
        ),
        engine_kwargs=_json_compatible(engine_kwargs),
        runtime_identity=_runtime_identity(),
        resolved_engine_config=resolved_config,
        measurements=tuple(measurements),
        total_prompt_tokens=total_prompt_tokens,
        total_output_tokens=total_output_tokens,
        elapsed_seconds=total_elapsed,
        output_tokens_per_second=total_output_tokens / total_elapsed,
    )


def _sampling_params(
    config: WorkerConfig, scheduler_key: int, repeat: int
) -> list[Any]:
    from vllm.sampling_params import SamplingParams

    return [
        SamplingParams(
            temperature=1.0,
            top_p=1.0,
            seed=config.seed + repeat * len(config.prompt_token_ids) + index,
            min_tokens=config.max_tokens,
            max_tokens=config.max_tokens,
            ignore_eos=True,
        )
        for index in range(scheduler_key)
    ]


def _validate_exact_work(
    prompts: Sequence[Mapping[str, Sequence[int]]],
    outputs: Sequence[Any],
    max_tokens: int,
) -> tuple[int, int]:
    if len(outputs) != len(prompts):
        raise ValueError(
            f"Expected {len(prompts)} request outputs, got {len(outputs)}."
        )

    prompt_tokens = 0
    output_tokens = 0
    for index, (prompt, request_output) in enumerate(zip(prompts, outputs)):
        expected_prompt = list(prompt["prompt_token_ids"])
        actual_prompt = list(getattr(request_output, "prompt_token_ids", []))
        if actual_prompt != expected_prompt:
            raise ValueError(f"Request {index} returned mismatched prompt token IDs.")
        candidates = getattr(request_output, "outputs", None)
        if not isinstance(candidates, Sequence) or len(candidates) != 1:
            raise ValueError(f"Request {index} must return exactly one completion.")
        token_ids = getattr(candidates[0], "token_ids", None)
        if not isinstance(token_ids, Sequence) or len(token_ids) != max_tokens:
            actual_count = len(token_ids) if isinstance(token_ids, Sequence) else 0
            raise ValueError(
                f"Request {index} must return exactly {max_tokens} output tokens; "
                f"got {actual_count}."
            )
        prompt_tokens += len(expected_prompt)
        output_tokens += len(token_ids)
    return prompt_tokens, output_tokens


def _metric_delta(
    before: WorkerMetricSnapshot, after: WorkerMetricSnapshot
) -> WorkerMetricSnapshot:
    return WorkerMetricSnapshot(
        scheduler_steps=_counter_delta(
            before.scheduler_steps, after.scheduler_steps, "scheduler"
        ),
        cudagraph_steps=_counter_delta(
            before.cudagraph_steps, after.cudagraph_steps, "CUDA Graph"
        ),
    )


def _counter_delta(
    before: Mapping[Any, int], after: Mapping[Any, int], name: str
) -> dict[Any, int]:
    delta: dict[Any, int] = {}
    for key in set(before) | set(after):
        value = after.get(key, 0) - before.get(key, 0)
        if value < 0:
            raise ValueError(f"Cumulative {name} counter decreased for {key!r}.")
        if value:
            delta[key] = value
    return delta


def _build_measurement(
    config: WorkerConfig,
    scheduler_key: int,
    repeat: int,
    elapsed: float,
    prompt_tokens: int,
    output_tokens: int,
    metrics: WorkerMetricSnapshot,
) -> WorkerMeasurement:
    requested_k = config.k or 0
    measured_steps = sum(metrics.scheduler_steps.values())
    requested_steps = metrics.scheduler_steps.get((scheduler_key, requested_k), 0)
    if measured_steps <= 0:
        raise ValueError("No scheduler-key metrics were recorded for the point.")
    scheduler_key_coverage = requested_steps / measured_steps
    if scheduler_key_coverage < 0.95:
        raise ValueError(
            f"Scheduler-key coverage is below 95% for B={scheduler_key}, "
            f"K={requested_k}: {scheduler_key_coverage:.2%}."
        )

    scheduler_histogram: list[dict[str, object]] = [
        {"scheduler_key": key, "k": k, "steps": steps}
        for (key, k), steps in sorted(metrics.scheduler_steps.items())
    ]
    cudagraph_histogram = [
        {
            "num_unpadded_tokens": step.num_unpadded_tokens,
            "num_padded_tokens": step.num_padded_tokens,
            "runtime_mode": step.runtime_mode,
            "steps": count,
        }
        for step, count in sorted(metrics.cudagraph_steps.items())
    ]
    fallback_histogram = [
        row for row in cudagraph_histogram if row["runtime_mode"] != "FULL"
    ]
    return WorkerMeasurement(
        scheduler_key=scheduler_key,
        repeat=repeat,
        elapsed_seconds=elapsed,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        output_tokens_per_second=output_tokens / elapsed,
        scheduler_key_coverage=scheduler_key_coverage,
        scheduler_key_histogram=scheduler_histogram,
        cudagraph_histogram=cudagraph_histogram,
        cudagraph_fallback_histogram=fallback_histogram,
    )


def _resolved_engine_config(llm: Any) -> dict[str, object]:
    llm_engine = getattr(llm, "llm_engine", None)
    vllm_config = getattr(llm_engine, "vllm_config", None)
    if vllm_config is None:
        raise ValueError("LLM does not expose its resolved vLLM configuration.")
    sections = (
        "model_config",
        "parallel_config",
        "scheduler_config",
        "cache_config",
        "compilation_config",
        "speculative_config",
    )
    return {
        section: _json_compatible(getattr(vllm_config, section, None))
        for section in sections
    }


def _validate_resolved_config(
    config: WorkerConfig, resolved: Mapping[str, object]
) -> None:
    parallel = _require_mapping(resolved, "parallel_config")
    if parallel.get("data_parallel_size") != 1:
        raise ValueError("Resolved data_parallel_size must be 1.")

    model = _require_mapping(resolved, "model_config")
    if model.get("enforce_eager") is not False:
        raise ValueError("Resolved engine must have enforce_eager=false.")

    compilation = _require_mapping(resolved, "compilation_config")
    mode = _enum_name(compilation.get("cudagraph_mode"))
    if mode != "FULL_AND_PIECEWISE":
        raise ValueError(
            f"Resolved cudagraph mode must be FULL_AND_PIECEWISE; got {mode or None}."
        )
    expected_capture_sizes = tuple(sorted(config.cudagraph_capture_sizes))
    resolved_sizes_value = compilation.get("cudagraph_capture_sizes")
    if not isinstance(resolved_sizes_value, Sequence) or isinstance(
        resolved_sizes_value, (str, bytes)
    ):
        raise ValueError("Resolved capture sizes are not an integer sequence.")
    resolved_capture_sizes = tuple(sorted(int(size) for size in resolved_sizes_value))
    if resolved_capture_sizes != expected_capture_sizes:
        raise ValueError(
            "Resolved capture sizes do not provide exact FULL shape coverage: "
            f"expected {expected_capture_sizes}, got {resolved_capture_sizes}."
        )

    speculative = resolved.get("speculative_config")
    if config.k is None:
        if speculative is not None:
            raise ValueError("Vanilla worker resolved a speculative configuration.")
        return
    if not isinstance(speculative, Mapping):
        raise ValueError("Speculative worker did not resolve speculative_config.")
    if speculative.get("num_speculative_tokens") != config.kmax:
        raise ValueError("Resolved speculative config changed common Kmax.")
    expected_schedule = [[1, max(config.scheduler_keys), config.k]]
    resolved_schedule = speculative.get("num_speculative_tokens_per_batch_size")
    if _json_compatible(resolved_schedule) != expected_schedule:
        raise ValueError("Resolved speculative config changed the forced-K schedule.")


def _require_mapping(mapping: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = mapping.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"Resolved engine config is missing {key}.")
    return value


def _failure_payload(
    config: WorkerConfig,
    engine_kwargs: Mapping[str, object],
    status: WorkerStatus,
    error: Exception,
) -> dict[str, object]:
    return {
        "status": status,
        "profile_id": profile_id(config.profile_identity),
        "profile_identity": config.profile_identity.payload,
        "variant": "vanilla" if config.k is None else f"k{config.k}",
        "forced_k": config.k,
        "common_kmax": None if config.k is None else config.kmax,
        "target_revision": config.model_revision,
        "speculative_revision": (
            None if config.k is None else config.speculative_revision
        ),
        "engine_kwargs": _json_compatible(engine_kwargs),
        "runtime_identity": _runtime_identity(),
        "resolved_engine_config": {},
        "measurements": [],
        "total_prompt_tokens": 0,
        "total_output_tokens": 0,
        "elapsed_seconds": 0.0,
        "output_tokens_per_second": None,
        "error": f"{type(error).__name__}: {error}",
    }


def _runtime_identity() -> dict[str, object]:
    try:
        vllm_version = importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError:
        vllm_version = "unknown"
    try:
        version_module = importlib.import_module("vllm._version")
        package_commit = getattr(version_module, "commit_id", None) or "unknown"
    except ImportError:
        package_commit = "unknown"
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "vllm_package_commit": package_commit,
        "vllm_source_commit": _source_commit(),
        "vllm_version": vllm_version,
        "vllm_use_v2_model_runner": os.environ.get("VLLM_USE_V2_MODEL_RUNNER"),
    }


def _source_commit() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    commit = completed.stdout.strip()
    return commit if completed.returncode == 0 and commit else "unknown"


def _json_compatible(value: Any, _seen: set[int] | None = None) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Enum):
        return value.name
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_compatible(item, _seen) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_compatible(item, _seen) for item in value]

    seen = _seen if _seen is not None else set()
    value_id = id(value)
    if value_id in seen:
        return f"<{type(value).__name__}:recursive>"
    seen.add(value_id)
    try:
        if dataclasses.is_dataclass(value):
            return {
                field.name: _json_compatible(getattr(value, field.name), seen)
                for field in dataclasses.fields(value)
                if not field.name.startswith("_")
            }
        attributes = getattr(value, "__dict__", None)
        if isinstance(attributes, Mapping):
            return {
                str(key): _json_compatible(item, seen)
                for key, item in attributes.items()
                if not str(key).startswith("_") and not callable(item)
            }
        return str(value)
    finally:
        seen.remove(value_id)


def _enum_name(value: object) -> str:
    if isinstance(value, Enum):
        return value.name
    return str(value).rsplit(".", 1)[-1]


class _CUDAGraphCollector:
    def __init__(self) -> None:
        self.cudagraph_steps: dict[CUDAGraphStep, int] = {}

    def record(
        self,
        scheduler_stats: Any,
        iteration_stats: Any,
        mm_cache_stats: Any = None,
        engine_idx: int = 0,
    ) -> None:
        del iteration_stats, mm_cache_stats, engine_idx
        if scheduler_stats is None:
            return
        stats = getattr(scheduler_stats, "cudagraph_stats", None)
        if stats is None:
            return
        step = CUDAGraphStep(
            num_unpadded_tokens=int(stats.num_unpadded_tokens),
            num_padded_tokens=int(stats.num_padded_tokens),
            runtime_mode=_enum_name(stats.runtime_mode),
        )
        self.cudagraph_steps[step] = self.cudagraph_steps.get(step, 0) + 1

    def record_sleep_state(self, sleep: int = 0, level: int = 0) -> None:
        del sleep, level

    def log(self) -> None:
        pass

    def log_engine_initialized(self) -> None:
        pass


class _DefaultMetricsReader:
    def __init__(self, llm: Any) -> None:
        self.llm = llm
        self.cudagraph_collector = _CUDAGraphCollector()
        logger_manager = getattr(
            getattr(llm, "llm_engine", None), "logger_manager", None
        )
        stat_loggers = getattr(logger_manager, "stat_loggers", None)
        if not isinstance(stat_loggers, list):
            raise ValueError("LLM does not expose an enabled statistics logger.")
        stat_loggers.append(self.cudagraph_collector)

    def snapshot(self) -> WorkerMetricSnapshot:
        scheduler_steps: dict[tuple[int, int], int] = {}
        for metric in self.llm.get_metrics():
            if getattr(metric, "name", None) != "vllm:dynamic_sd_scheduler_steps":
                continue
            labels = getattr(metric, "labels", {})
            scheduler_key = int(labels["scheduler_batch_size"])
            k = int(labels["k"])
            scheduler_steps[(scheduler_key, k)] = scheduler_steps.get(
                (scheduler_key, k), 0
            ) + int(metric.value)
        return WorkerMetricSnapshot(
            scheduler_steps=scheduler_steps,
            cudagraph_steps=dict(self.cudagraph_collector.cudagraph_steps),
        )
