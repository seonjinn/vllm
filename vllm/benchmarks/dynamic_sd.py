# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Profile fixed-K candidates and emit a Dynamic SD runtime schedule."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from vllm.benchmarks.dynamic_sd_core import (
    Measurement,
    ProfileIdentity,
    SelectionPolicy,
    compress_schedule,
    select_schedule,
)
from vllm.benchmarks.dynamic_sd_worker import (
    WorkerConfig,
    derive_cudagraph_capture_sizes,
    worker_config_to_payload,
    write_json_atomic,
)
from vllm.v1.spec_decode.dynamic.utils import (
    validate_and_normalize_dynamic_sd_schedule,
)

DYNAMIC_SD_PROFILE_CONTRACT_VERSION = 1
DYNAMIC_SD_PROFILE_CAPABILITIES = frozenset(
    {
        "deployment_engine_identity",
        "external_ray_cluster",
        "fixed_output_calibration",
        "full_and_piecewise_capture",
        "process_isolated_candidates",
    }
)

ProcessRunner = Callable[..., subprocess.CompletedProcess[str]]


@dataclass(frozen=True)
class ProfileArgs:
    model: str
    model_revision: str
    speculative_model: str
    speculative_model_revision: str
    speculative_method: str
    tensor_parallel_size: int
    draft_tensor_parallel_size: int
    data_parallel_size: int
    engine_kwargs: Mapping[str, object]
    scheduler_keys: tuple[int, ...]
    k_values: tuple[int, ...]
    common_kmax: int
    profile_backend: str
    temperature: float
    top_p: float
    draft_sample_method: str
    rejection_sample_method: str
    cuda_graph_mode: str
    enforce_eager: bool
    warmups: int
    repeats: int
    seed: int
    min_tokens: int
    max_tokens: int
    ignore_eos: bool
    prompt_workload: Path
    natural_stopping_validation: bool
    natural_stopping_seed: int
    natural_stopping_repeats: int
    output_dir: Path
    worker_timeout_seconds: float | None
    selection_policy: SelectionPolicy

    @classmethod
    def from_namespace(cls, namespace: argparse.Namespace) -> ProfileArgs:
        scheduler_keys = _parse_positive_ints(namespace.scheduler_batch_sizes)
        k_values = _parse_nonnegative_ints(namespace.k_values)
        profile = cls(
            model=namespace.model,
            model_revision=namespace.model_revision,
            speculative_model=namespace.speculative_model,
            speculative_model_revision=namespace.speculative_model_revision,
            speculative_method=namespace.speculative_method,
            tensor_parallel_size=namespace.tensor_parallel_size,
            draft_tensor_parallel_size=namespace.draft_tensor_parallel_size,
            data_parallel_size=namespace.data_parallel_size,
            engine_kwargs=_parse_engine_kwargs_json(namespace.engine_kwargs_json),
            scheduler_keys=scheduler_keys,
            k_values=k_values,
            common_kmax=namespace.common_kmax,
            profile_backend=namespace.profile_backend,
            temperature=namespace.temperature,
            top_p=namespace.top_p,
            draft_sample_method=namespace.draft_sample_method,
            rejection_sample_method=namespace.rejection_sample_method,
            cuda_graph_mode=namespace.cuda_graph_mode,
            enforce_eager=namespace.enforce_eager,
            warmups=namespace.warmups,
            repeats=namespace.repeats,
            seed=namespace.seed,
            min_tokens=namespace.min_tokens,
            max_tokens=namespace.max_tokens,
            ignore_eos=namespace.ignore_eos,
            prompt_workload=namespace.prompt_workload,
            natural_stopping_validation=namespace.natural_stopping_validation,
            natural_stopping_seed=namespace.natural_stopping_seed,
            natural_stopping_repeats=namespace.natural_stopping_repeats,
            output_dir=namespace.output_dir,
            worker_timeout_seconds=namespace.worker_timeout_seconds,
            selection_policy=SelectionPolicy(
                configured_ks=k_values,
                within_best_fraction=namespace.within_best_fraction,
                min_enable_gain=namespace.min_enable_gain,
                max_cv=namespace.max_cv,
                confidence_level=namespace.confidence_level,
            ),
        )
        profile.validate()
        return profile

    def validate(self) -> None:
        if self.data_parallel_size != 1:
            raise ValueError("Dynamic SD profiling currently requires DP=1.")
        if self.profile_backend != "offline-sync":
            raise ValueError("Only profile-backend=offline-sync is supported.")
        if self.cuda_graph_mode != "FULL_AND_PIECEWISE" or self.enforce_eager:
            raise ValueError(
                "Profiling requires CUDA Graph mode FULL_AND_PIECEWISE and "
                "enforce_eager=false."
            )
        if self.k_values[0] != 0 or self.common_kmax != max(self.k_values):
            raise ValueError("k-values must start at 0 and end at common-kmax.")
        if self.min_tokens != self.max_tokens or not self.ignore_eos:
            raise ValueError(
                "Calibration requires fixed output: min_tokens=max_tokens and "
                "ignore_eos=true."
            )
        if self.natural_stopping_validation:
            raise ValueError(
                "Natural-stopping validation is not implemented by this profiler; "
                "validate the emitted schedule in the downstream workload."
            )
        if self.repeats < 3:
            raise ValueError("Schedule selection requires at least three repeats.")
        if self.max_tokens <= 0 or self.warmups < 0:
            raise ValueError("Invalid token or warmup count.")
        if self.tensor_parallel_size <= 0 or self.draft_tensor_parallel_size <= 0:
            raise ValueError("Tensor parallel sizes must be positive.")
        if "tensor_parallel_size" in self.engine_kwargs:
            raise ValueError(
                "engine-kwargs-json must not override tensor_parallel_size."
            )


def add_cli_args(parser: argparse.ArgumentParser) -> None:
    actions = parser.add_subparsers(required=True, dest="dynamic_sd_action")
    profile = actions.add_parser("profile", help="Profile and select a schedule.")
    _add_profile_args(profile)
    profile.set_defaults(dynamic_sd_dispatch=_profile_command)

    select = actions.add_parser("select", help="Re-select from existing raw results.")
    select.add_argument("--output-dir", type=Path, required=True)
    select.add_argument("--k-values", default="0,1,2,3")
    select.add_argument("--within-best-fraction", type=float, default=0.02)
    select.add_argument("--min-enable-gain", type=float, default=0.05)
    select.add_argument("--max-cv", type=float, default=0.05)
    select.add_argument("--confidence-level", type=float, default=0.95)
    select.set_defaults(dynamic_sd_dispatch=_select_command)


def _add_profile_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--speculative-model", required=True)
    parser.add_argument("--speculative-model-revision", required=True)
    parser.add_argument("--speculative-method", required=True)
    parser.add_argument("--tensor-parallel-size", type=int, required=True)
    parser.add_argument("--draft-tensor-parallel-size", type=int, required=True)
    parser.add_argument("--data-parallel-size", type=int, default=1)
    parser.add_argument("--engine-kwargs-json", default="{}")
    parser.add_argument("--scheduler-batch-sizes", required=True)
    parser.add_argument("--k-values", required=True)
    parser.add_argument("--common-kmax", type=int, required=True)
    parser.add_argument("--profile-backend", default="offline-sync")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--draft-sample-method", default="probabilistic")
    parser.add_argument("--rejection-sample-method", default="standard")
    parser.add_argument("--cuda-graph-mode", default="FULL_AND_PIECEWISE")
    parser.add_argument("--enforce-eager", type=_parse_bool, default=False)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-tokens", type=int, required=True)
    parser.add_argument("--max-tokens", type=int, required=True)
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--prompt-workload", type=Path, required=True)
    parser.add_argument("--natural-stopping-validation", action="store_true")
    parser.add_argument("--natural-stopping-seed", type=int, default=0)
    parser.add_argument("--natural-stopping-repeats", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--worker-timeout-seconds", type=float)
    parser.add_argument("--within-best-fraction", type=float, default=0.02)
    parser.add_argument("--min-enable-gain", type=float, default=0.05)
    parser.add_argument("--max-cv", type=float, default=0.05)
    parser.add_argument("--confidence-level", type=float, default=0.95)


def main(namespace: argparse.Namespace) -> None:
    namespace.dynamic_sd_dispatch(namespace)


def build_worker_configs(profile: ProfileArgs) -> list[WorkerConfig]:
    prompts = _load_prompt_workload(profile.prompt_workload)
    if len(prompts) < max(profile.scheduler_keys):
        raise ValueError("Prompt workload does not cover the largest scheduler key.")
    identity = ProfileIdentity.from_mapping(_profile_identity(profile))
    common = {
        "profile_identity": identity,
        "model": profile.model,
        "model_revision": profile.model_revision,
        "speculative_model": profile.speculative_model,
        "speculative_revision": profile.speculative_model_revision,
        "speculative_method": profile.speculative_method,
        "kmax": profile.common_kmax,
        "scheduler_keys": profile.scheduler_keys,
        "prompt_token_ids": prompts,
        "max_tokens": profile.max_tokens,
        "warmups": profile.warmups,
        "repeats": profile.repeats,
        "seed": profile.seed,
        "data_parallel_size": profile.data_parallel_size,
        "engine_kwargs": {
            **profile.engine_kwargs,
            "tensor_parallel_size": profile.tensor_parallel_size,
        },
        "draft_tensor_parallel_size": profile.draft_tensor_parallel_size,
        "temperature": profile.temperature,
        "top_p": profile.top_p,
        "draft_sample_method": profile.draft_sample_method,
        "rejection_sample_method": profile.rejection_sample_method,
    }
    configs: list[WorkerConfig] = []
    for k in profile.k_values:
        label = f"k{k}"
        configs.append(
            WorkerConfig(
                output_path=profile.output_dir / "raw" / f"{label}.json",
                k=k,
                cudagraph_capture_sizes=derive_cudagraph_capture_sizes(
                    {key: k for key in profile.scheduler_keys}
                ),
                **common,
            )
        )
    return configs


def run_profile_driver(
    profile: ProfileArgs,
    *,
    runner: ProcessRunner = subprocess.run,
) -> dict[str, object]:
    profile.output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = profile.output_dir / "raw"
    config_dir = profile.output_dir / "worker-configs"
    if (
        raw_dir.exists()
        or config_dir.exists()
        or (profile.output_dir / "manifest.json").exists()
    ):
        raise ValueError("Profile output directory contains an existing run.")
    raw_dir.mkdir()
    config_dir.mkdir()
    configs = build_worker_configs(profile)
    for config in configs:
        assert config.k is not None
        label = f"k{config.k}"
        config_path = config_dir / f"{label}.json"
        write_json_atomic(config_path, worker_config_to_payload(config))
        worker_environment = dict(os.environ)
        worker_environment["VLLM_USE_V2_MODEL_RUNNER"] = "1"
        completed = runner(
            [
                sys.executable,
                "-m",
                "vllm.benchmarks.dynamic_sd_worker",
                "--config",
                str(config_path),
            ],
            check=False,
            text=True,
            capture_output=True,
            timeout=profile.worker_timeout_seconds,
            env=worker_environment,
        )
        payload = _read_result(config.output_path)
        if completed.returncode != 0 and payload.get("status") != "infeasible":
            raise RuntimeError(
                f"Dynamic SD worker {label} failed: "
                f"{payload.get('error') or completed.stderr.strip()}"
            )
    payloads = [_read_result(config.output_path) for config in configs]
    manifest = _profile_manifest(profile.output_dir, payloads, profile)
    write_json_atomic(profile.output_dir / "manifest.json", manifest)
    return select_profile(profile.output_dir, profile.selection_policy)


def select_profile(
    output_dir: Path,
    policy: SelectionPolicy,
) -> dict[str, object]:
    manifest, payloads = _load_manifest_results(output_dir)
    profile_ids = {str(payload["profile_id"]) for payload in payloads}
    workload_hashes = {str(payload["workload_hash"]) for payload in payloads}
    manifest_ks = tuple(int(k) for k in manifest["k_values"])
    if policy.configured_ks != manifest_ks:
        raise ValueError("Selection K values do not match the measured manifest grid.")
    measurements = _selection_measurements(payloads, policy)
    selection = select_schedule(measurements, policy)
    expected_keys = {int(key) for key in manifest["scheduler_keys"]}
    if set(selection.selected_k) != expected_keys:
        missing = sorted(expected_keys - set(selection.selected_k))
        raise ValueError(f"No feasible K was selected for scheduler keys: {missing}")
    max_num_seqs = max(expected_keys)
    schedule = validate_and_normalize_dynamic_sd_schedule(
        [list(entry) for entry in compress_schedule(selection.selected_k, max_num_seqs)]
    )
    summary = {
        "schema_version": 1,
        "profile_id": next(iter(profile_ids)),
        "workload_hash": next(iter(workload_hashes)),
        "selected_k": {str(key): value for key, value in selection.selected_k.items()},
        "schedule": [list(entry) for entry in schedule],
        "requires_k_extension": selection.requires_k_extension,
        "median_throughputs": selection.median_throughputs,
        "gain_intervals": selection.gain_intervals,
        "infeasible_ks": selection.infeasible_ks,
        "selection_policy": asdict(policy),
    }
    _write_grid(output_dir / "grid.csv", payloads)
    write_json_atomic(output_dir / "summary.json", summary)
    runtime_path = output_dir / "dynamic_speculative_config.json"
    runtime_path.unlink(missing_ok=True)
    if selection.requires_k_extension:
        raise ValueError(
            "The maximum profiled K won; extend the K grid before deployment."
        )
    dynamic_config = _dynamic_config(payloads, policy, schedule)
    write_json_atomic(runtime_path, dynamic_config)
    return summary


def _selection_measurements(
    payloads: Sequence[Mapping[str, object]], policy: SelectionPolicy
) -> list[Measurement]:
    rows: list[Measurement] = []
    by_k: dict[object, Mapping[str, object]] = {}
    for payload in payloads:
        k = payload.get("forced_k")
        if k in by_k:
            raise ValueError(f"Duplicate raw result for K={k}.")
        by_k[k] = payload
    for k in policy.configured_ks:
        payload = by_k.get(k)
        if payload is None:
            raise ValueError(f"Missing raw result for K={k}.")
        status = str(payload["status"])
        if status == "complete":
            for measurement in payload["measurements"]:
                rows.append(
                    Measurement(
                        scheduler_key=int(measurement["scheduler_key"]),
                        k=k,
                        repeat=int(measurement["repeat"]),
                        output_tokens_per_second=float(
                            measurement["output_tokens_per_second"]
                        ),
                        status="complete",
                        workload_hash=str(payload["workload_hash"]),
                    )
                )
        elif status == "infeasible":
            identity = payload["profile_identity"]
            for key in identity["scheduler_keys"]:
                for repeat in range(int(identity["repeats"])):
                    rows.append(
                        Measurement(
                            scheduler_key=int(key),
                            k=k,
                            repeat=repeat,
                            output_tokens_per_second=None,
                            status="infeasible",
                            workload_hash=str(payload["workload_hash"]),
                        )
                    )
        else:
            raise ValueError(f"K={k} has failed profiling.")
    return rows


def _dynamic_config(
    payloads: Sequence[Mapping[str, object]],
    policy: SelectionPolicy,
    schedule: Sequence[tuple[int, int, int]],
) -> dict[str, object]:
    speculative = next(payload for payload in payloads if payload.get("forced_k") == 0)
    engine = speculative["engine_kwargs"]
    spec = engine["speculative_config"]
    max_k = max(policy.configured_ks)
    capture_sizes = derive_cudagraph_capture_sizes(
        {
            batch_size: k
            for start, end, k in schedule
            for batch_size in range(start, end + 1)
        }
    )
    return {
        "schema_version": 1,
        "speculative_config": {
            "method": spec["method"],
            "model": spec["model"],
            "revision": spec["revision"],
            "num_speculative_tokens": max_k,
            "num_speculative_tokens_per_batch_size": [
                list(entry) for entry in schedule
            ],
            "draft_tensor_parallel_size": spec["draft_tensor_parallel_size"],
            "draft_sample_method": spec["draft_sample_method"],
            "rejection_sample_method": spec["rejection_sample_method"],
        },
        "compilation_config": {
            "cudagraph_mode": "FULL_AND_PIECEWISE",
            "cudagraph_capture_sizes": list(capture_sizes),
        },
        "metadata": {
            "downstream_natural_stopping_validation_required": True,
            "max_profiled_batch_size": max(end for _, end, _ in schedule),
        },
    }


def _write_grid(path: Path, payloads: Sequence[Mapping[str, object]]) -> None:
    rows: list[dict[str, object]] = []
    for payload in payloads:
        for measurement in payload.get("measurements", []):
            rows.append(
                {
                    "variant": payload["variant"],
                    "k": payload["forced_k"],
                    "scheduler_key": measurement["scheduler_key"],
                    "repeat": measurement["repeat"],
                    "output_tokens_per_second": measurement["output_tokens_per_second"],
                    "scheduler_key_coverage": measurement["scheduler_key_coverage"],
                    "status": payload["status"],
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _profile_manifest(
    output_dir: Path,
    payloads: Sequence[Mapping[str, object]],
    profile: ProfileArgs,
) -> dict[str, object]:
    expected_ks = list(profile.k_values)
    if sorted(payload.get("forced_k") for payload in payloads) != expected_ks:
        raise ValueError("Raw results do not match the exact requested K grid.")
    profile_ids = {str(payload.get("profile_id")) for payload in payloads}
    workload_hashes = {str(payload.get("workload_hash")) for payload in payloads}
    if len(profile_ids) != 1 or len(workload_hashes) != 1:
        raise ValueError("Raw results mix profile IDs or workload hashes.")
    raw_results = []
    for k in expected_ks:
        path = output_dir / "raw" / f"k{k}.json"
        if not path.is_file():
            raise ValueError(f"Missing raw result for K={k}.")
        raw_results.append(
            {
                "k": k,
                "path": str(path.relative_to(output_dir)),
                "sha256": _sha256(path),
            }
        )
    return {
        "schema_version": 1,
        "contract_version": DYNAMIC_SD_PROFILE_CONTRACT_VERSION,
        "capabilities": sorted(DYNAMIC_SD_PROFILE_CAPABILITIES),
        "profile_id": next(iter(profile_ids)),
        "workload_hash": next(iter(workload_hashes)),
        "k_values": expected_ks,
        "common_kmax": profile.common_kmax,
        "scheduler_keys": list(profile.scheduler_keys),
        "raw_results": raw_results,
        "variants": [payload["variant"] for payload in payloads],
    }


def _load_manifest_results(
    output_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = _read_result(output_dir / "manifest.json")
    entries = manifest.get("raw_results")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Manifest raw_results must be a non-empty list.")
    expected_paths: set[Path] = set()
    payloads: list[dict[str, Any]] = []
    seen_ks: set[int] = set()
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("Manifest raw result entry must be an object.")
        k = int(entry["k"])
        if k in seen_ks:
            raise ValueError(f"Manifest contains duplicate K={k}.")
        seen_ks.add(k)
        relative_path = Path(str(entry["path"]))
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError("Manifest raw result path escapes the output directory.")
        path = output_dir / relative_path
        expected_paths.add(path.resolve())
        if _sha256(path) != entry["sha256"]:
            raise ValueError(f"Raw result hash mismatch for K={k}.")
        payload = _read_result(path)
        if payload.get("forced_k") != k:
            raise ValueError(f"Raw result identity mismatch for K={k}.")
        if payload.get("profile_id") != manifest.get("profile_id"):
            raise ValueError(f"Raw result profile ID mismatch for K={k}.")
        if payload.get("workload_hash") != manifest.get("workload_hash"):
            raise ValueError(f"Raw result workload hash mismatch for K={k}.")
        payloads.append(payload)
    actual_paths = {path.resolve() for path in (output_dir / "raw").glob("*.json")}
    if actual_paths != expected_paths:
        raise ValueError("Raw result directory contains missing or extra JSON files.")
    if seen_ks != {int(k) for k in manifest.get("k_values", [])}:
        raise ValueError("Manifest K grid does not match its raw results.")
    return manifest, payloads


def _profile_identity(profile: ProfileArgs) -> dict[str, object]:
    return {
        "schema_version": 1,
        "model": profile.model,
        "model_revision": profile.model_revision,
        "speculative_model": profile.speculative_model,
        "speculative_revision": profile.speculative_model_revision,
        "speculative_method": profile.speculative_method,
        "tensor_parallel_size": profile.tensor_parallel_size,
        "draft_tensor_parallel_size": profile.draft_tensor_parallel_size,
        "data_parallel_size": profile.data_parallel_size,
        "engine_kwargs": dict(profile.engine_kwargs),
        "scheduler_keys": list(profile.scheduler_keys),
        "configured_ks": list(profile.k_values),
        "common_kmax": profile.common_kmax,
        "temperature": profile.temperature,
        "top_p": profile.top_p,
        "draft_sample_method": profile.draft_sample_method,
        "rejection_sample_method": profile.rejection_sample_method,
        "max_tokens": profile.max_tokens,
        "warmups": profile.warmups,
        "repeats": profile.repeats,
        "seed": profile.seed,
    }


def _load_prompt_workload(path: Path) -> tuple[tuple[int, ...], ...]:
    prompts: list[tuple[int, ...]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        payload = json.loads(line)
        tokens = (
            payload.get("prompt_token_ids") if isinstance(payload, dict) else payload
        )
        if not isinstance(tokens, list) or not tokens:
            raise ValueError(f"Invalid prompt_token_ids on line {line_number}.")
        prompts.append(tuple(int(token) for token in tokens))
    if not prompts:
        raise ValueError("Prompt workload must not be empty.")
    return tuple(prompts)


def _read_result(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Result must be a JSON object: {path}")
    return payload


def _parse_ints(value: str, *, allow_zero: bool) -> tuple[int, ...]:
    values: list[int] = []
    for raw_part in value.split(","):
        part = raw_part.strip()
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start, end = int(start_text), int(end_text)
            if start > end:
                raise ValueError(f"Invalid integer range: {part}")
            values.extend(range(start, end + 1))
        else:
            values.append(int(part))
    if not values or len(set(values)) != len(values):
        raise ValueError("Integer list must be non-empty and unique.")
    minimum = 0 if allow_zero else 1
    if any(value < minimum for value in values):
        raise ValueError(f"Integer values must be >= {minimum}.")
    return tuple(sorted(values))


def _parse_positive_ints(value: str) -> tuple[int, ...]:
    return _parse_ints(value, allow_zero=False)


def _parse_nonnegative_ints(value: str) -> tuple[int, ...]:
    return _parse_ints(value, allow_zero=True)


def _parse_bool(value: str) -> bool:
    normalized = value.lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def _parse_engine_kwargs_json(value: str) -> dict[str, object]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError(f"engine-kwargs-json is invalid JSON: {error}") from None
    if not isinstance(parsed, dict) or any(not isinstance(key, str) for key in parsed):
        raise ValueError("engine-kwargs-json must be a JSON object with string keys.")
    return parsed


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _profile_command(namespace: argparse.Namespace) -> None:
    run_profile_driver(ProfileArgs.from_namespace(namespace))


def _select_command(namespace: argparse.Namespace) -> None:
    policy = SelectionPolicy(
        configured_ks=_parse_nonnegative_ints(namespace.k_values),
        within_best_fraction=namespace.within_best_fraction,
        min_enable_gain=namespace.min_enable_gain,
        max_cv=namespace.max_cv,
        confidence_level=namespace.confidence_level,
    )
    select_profile(
        namespace.output_dir,
        policy,
    )
