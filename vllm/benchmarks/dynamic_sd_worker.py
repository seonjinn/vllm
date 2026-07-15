# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline worker for one vanilla or fixed-K Dynamic SD profile."""

from __future__ import annotations

import dataclasses
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Protocol

from vllm.benchmarks.dynamic_sd_core import ProfileIdentity, profile_id

WorkerStatus = Literal["complete", "infeasible", "failed"]
_PROFILE_METRICS_ENV = "VLLM_DYNAMIC_SD_PROFILE_METRICS"
_worker_invocation_lock = threading.Lock()
_worker_invoked = False


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
    workload_hash: str
    workload_identity: dict[str, object]
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


def workload_identity(config: WorkerConfig) -> dict[str, object]:
    """Return the exact canonical workload and deterministic sampling inputs."""
    max_scheduler_key = max(config.scheduler_keys)
    seed_stride = len(config.prompt_token_ids)
    return {
        "schema_version": 1,
        "scheduler_keys": list(config.scheduler_keys),
        "prompt_token_ids": [
            list(token_ids) for token_ids in config.prompt_token_ids[:max_scheduler_key]
        ],
        "max_output_tokens": config.max_tokens,
        "warmups": config.warmups,
        "repeats": config.repeats,
        "sampling_parameters": {
            "temperature": 1.0,
            "top_p": 1.0,
            "min_tokens": config.max_tokens,
            "max_tokens": config.max_tokens,
            "ignore_eos": True,
        },
        "seed_provenance": {
            "base_seed": config.seed,
            "seed_stride": seed_stride,
            "formula": "base_seed + repeat * seed_stride + request_index",
            "warmup_seeds": {
                str(key): [
                    [_sampling_seed(config, warmup, index) for index in range(key)]
                    for warmup in range(config.warmups)
                ]
                for key in config.scheduler_keys
            },
            "repeat_seeds": {
                str(key): [
                    [_sampling_seed(config, repeat, index) for index in range(key)]
                    for repeat in range(config.repeats)
                ]
                for key in config.scheduler_keys
            },
        },
    }


def workload_hash(config: WorkerConfig) -> str:
    """Hash the canonical workload used by this worker."""
    canonical = json.dumps(
        workload_identity(config), sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def _bound_profile_identity(config: WorkerConfig) -> ProfileIdentity:
    payload = config.profile_identity.payload
    payload["workload_hash"] = workload_hash(config)
    return ProfileIdentity.from_mapping(payload)


def run_worker(
    config: WorkerConfig,
    llm_factory: Callable[..., Any],
    *,
    metrics_reader_factory: Callable[[Any], WorkerMetricsReader] | None = None,
) -> WorkerResult:
    """Run one profile in a fresh process and atomically write its result.

    A process may call this function exactly once, even when that invocation
    fails. The caller must launch every additional profile in a fresh process.
    """
    _clear_previous_result(config.output_path)
    engine_kwargs: dict[str, object] = {}
    previous_metrics_env = os.environ.get(_PROFILE_METRICS_ENV)

    try:
        _claim_worker_process()
        _validate_preflight(config)
        engine_kwargs = _build_engine_kwargs(config)
        os.environ[_PROFILE_METRICS_ENV] = "1"
        result = _run_measurements(
            config,
            engine_kwargs,
            llm_factory,
            metrics_reader_factory or _DefaultMetricsReader,
        )
        write_json_atomic(config.output_path, result.to_payload())
        return result
    except Exception as exc:
        status: WorkerStatus = (
            "infeasible" if isinstance(exc, InfeasibleWorkerError) else "failed"
        )
        write_json_atomic(
            config.output_path,
            _failure_payload(config, engine_kwargs, status, exc),
        )
        raise
    finally:
        if previous_metrics_env is None:
            os.environ.pop(_PROFILE_METRICS_ENV, None)
        else:
            os.environ[_PROFILE_METRICS_ENV] = previous_metrics_env


def _claim_worker_process() -> None:
    global _worker_invoked
    with _worker_invocation_lock:
        if _worker_invoked:
            raise RuntimeError(
                "Dynamic SD workers are one-shot; launch each profile in a fresh "
                "process."
            )
        _worker_invoked = True


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
    if config.k is not None and not _is_full_commit_hash(config.speculative_revision):
        raise ValueError(
            "speculative_revision must be an immutable 40-character commit hash."
        )

    declared_workload_hash = config.profile_identity.payload.get("workload_hash")
    canonical_workload_hash = workload_hash(config)
    if (
        declared_workload_hash is not None
        and declared_workload_hash != canonical_workload_hash
    ):
        raise ValueError(
            "profile_identity workload_hash is not bound to the exact canonical "
            "worker workload."
        )

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
    llm: Any = None
    metrics_reader: WorkerMetricsReader | None = None
    try:
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

        bound_profile_identity = _bound_profile_identity(config)
        return WorkerResult(
            status="complete",
            profile_id=profile_id(bound_profile_identity),
            profile_identity=bound_profile_identity.payload,
            workload_hash=workload_hash(config),
            workload_identity=workload_identity(config),
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
    finally:
        _cleanup_runtime_state(metrics_reader, llm)


def _cleanup_runtime_state(
    metrics_reader: WorkerMetricsReader | None, llm: Any
) -> None:
    cleanup_error: Exception | None = None
    close = getattr(metrics_reader, "close", None)
    if callable(close):
        try:
            close()
        except Exception as exc:
            cleanup_error = exc

    if llm is not None:
        engine = getattr(llm, "llm_engine", None)
        engine_core = getattr(engine, "engine_core", None)
        for owner in (llm, engine, engine_core):
            shutdown = getattr(owner, "shutdown", None)
            if not callable(shutdown):
                continue
            try:
                shutdown()
            except Exception as exc:
                if cleanup_error is None:
                    cleanup_error = exc
            break

    if cleanup_error is not None:
        raise cleanup_error


def _sampling_params(
    config: WorkerConfig, scheduler_key: int, repeat: int
) -> list[Any]:
    from vllm.sampling_params import SamplingParams

    return [
        SamplingParams(
            temperature=1.0,
            top_p=1.0,
            seed=_sampling_seed(config, repeat, index),
            min_tokens=config.max_tokens,
            max_tokens=config.max_tokens,
            ignore_eos=True,
        )
        for index in range(scheduler_key)
    ]


def _sampling_seed(config: WorkerConfig, repeat: int, request_index: int) -> int:
    return config.seed + repeat * len(config.prompt_token_ids) + request_index


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
    if not metrics.cudagraph_steps:
        raise ValueError("No CUDA Graph metrics were recorded for the point.")
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
    resolved = {
        section: _json_compatible(getattr(vllm_config, section, None))
        for section in sections
    }
    speculative_config = getattr(vllm_config, "speculative_config", None)
    resolved["use_v2_model_runner"] = _json_compatible(
        getattr(vllm_config, "use_v2_model_runner", None)
    )
    resolved["drafter_identity"] = _resolved_drafter_identity(speculative_config)
    return resolved


def _resolved_drafter_identity(speculative_config: Any) -> dict[str, object] | None:
    if speculative_config is None:
        return None
    draft_model_config = _config_field(speculative_config, "draft_model_config")
    if draft_model_config is None:
        return None
    hf_config = _config_field(draft_model_config, "hf_config")
    return {
        "model": _json_compatible(_config_field(draft_model_config, "model")),
        "revision": _json_compatible(_config_field(draft_model_config, "revision")),
        "commit": _json_compatible(_config_field(hf_config, "_commit_hash")),
    }


def _config_field(config: Any, field_name: str) -> Any:
    if isinstance(config, Mapping):
        return config.get(field_name)
    return getattr(config, field_name, None)


def _validate_resolved_config(
    config: WorkerConfig, resolved: Mapping[str, object]
) -> None:
    if resolved.get("use_v2_model_runner") is not True:
        raise ValueError("Resolved use_v2_model_runner must be true.")

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
        if resolved.get("drafter_identity") is not None:
            raise ValueError("Vanilla worker resolved a drafter identity.")
        return
    if not isinstance(speculative, Mapping):
        raise ValueError("Speculative worker did not resolve speculative_config.")

    expected_controls = {
        "model": config.speculative_model,
        "revision": config.speculative_revision,
        "method": config.speculative_method,
        "draft_sample_method": "probabilistic",
        "rejection_sample_method": "standard",
    }
    for field_name, expected in expected_controls.items():
        actual = speculative.get(field_name)
        if field_name in {
            "method",
            "draft_sample_method",
            "rejection_sample_method",
        }:
            actual = _enum_name(actual)
        if actual != expected:
            raise ValueError(
                f"Resolved speculative {field_name} changed: expected "
                f"{expected!r}, got {actual!r}."
            )
    if speculative.get("num_speculative_tokens") != config.kmax:
        raise ValueError("Resolved speculative config changed common Kmax.")
    expected_schedule = [[1, max(config.scheduler_keys), config.k]]
    resolved_schedule = speculative.get("num_speculative_tokens_per_batch_size")
    if _json_compatible(resolved_schedule) != expected_schedule:
        raise ValueError("Resolved speculative config changed the forced-K schedule.")

    drafter = _require_mapping(resolved, "drafter_identity")
    if drafter.get("model") != config.speculative_model:
        raise ValueError("Resolved drafter model does not match the requested model.")
    if drafter.get("revision") != config.speculative_revision:
        raise ValueError(
            "Resolved drafter revision does not match the requested immutable revision."
        )
    if drafter.get("commit") != config.speculative_revision:
        raise ValueError(
            "Resolved drafter commit does not match the requested immutable commit."
        )


def _is_full_commit_hash(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


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
    bound_profile_identity = _bound_profile_identity(config)
    return {
        "status": status,
        "profile_id": profile_id(bound_profile_identity),
        "profile_identity": bound_profile_identity.payload,
        "workload_hash": workload_hash(config),
        "workload_identity": workload_identity(config),
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
        self.stat_loggers = stat_loggers

    def close(self) -> None:
        if self.cudagraph_collector in self.stat_loggers:
            self.stat_loggers.remove(self.cudagraph_collector)

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
