#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Summarize exact-shape MXFP8 oracle and serving A/B artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

_CROSS_BACKEND_METADATA = (
    "source_commit",
    "flashinfer_commit",
    "expected_vllm_version",
    "flashinfer",
    "flashinfer_file",
    "vllm_version",
    "vllm_file",
    "vllm_compiled_file",
    "gpu_name",
    "driver_version",
    "container",
    "container_sha256",
    "container_size",
    "container_mtime",
    "model",
    "model_config_sha256",
    "model_index_sha256",
    "model_weights_manifest_sha256",
    "tp",
    "moe_backend",
    "attention_backend",
    "kv_cache_dtype",
    "max_model_len",
    "max_num_batched_tokens",
    "max_num_seqs",
    "gpu_memory_utilization",
    "enable_chunked_prefill",
    "enable_prefix_caching",
    "mamba_cache_mode",
    "mamba_ssm_cache_dtype",
    "enforce_eager",
    "cudagraph_capture_sizes",
    "cudagraph_configured",
    "cudagraph_capture_status",
    "cudagraph_capture_evidence",
    "cudagraph_capture_marker",
    "workloads",
    "concurrencies",
    "prompt_multiplier",
)
_ORACLE_TIMING_FIELDS = (
    "cuda_graph",
    "cold_l2_cache",
    "rounds",
    "dry_run_iters",
    "repeat_iters",
    "calls_per_graph",
)
_EXPECTED_RECIPES = {
    "cute-dsl": {
        "backend_name": "cute-dsl",
        "linear_backend": "flashinfer_cutedsl",
        "oracle_backend": "cute-dsl",
        "scale_layout": "128x4",
        "trtllm_layout": "8x4",
    },
    "cutlass": {
        "backend_name": "cutlass",
        "linear_backend": "flashinfer_cutlass",
        "oracle_backend": "cutlass",
        "scale_layout": "128x4",
        "trtllm_layout": "8x4",
    },
    "trtllm-128x4": {
        "backend_name": "trtllm-128x4",
        "linear_backend": "flashinfer_trtllm",
        "oracle_backend": "trtllm",
        "scale_layout": "128x4",
        "trtllm_layout": "128x4",
    },
    "trtllm-8x4": {
        "backend_name": "trtllm-8x4",
        "linear_backend": "flashinfer_trtllm",
        "oracle_backend": "trtllm",
        "scale_layout": "8x4",
        "trtllm_layout": "8x4",
    },
}
_EXPECTED_BACKENDS = tuple(_EXPECTED_RECIPES)
_UNSUPPORTED_PROVENANCE_METADATA = (
    "source_commit",
    "flashinfer_commit",
    "expected_vllm_version",
    "flashinfer",
    "flashinfer_file",
    "vllm_version",
    "vllm_file",
    "vllm_compiled_file",
    "gpu_name",
    "driver_version",
    "container",
    "container_sha256",
    "container_size",
    "container_mtime",
    "model",
    "model_config_sha256",
    "model_index_sha256",
    "model_weights_manifest_sha256",
    "tp",
    "moe_backend",
    "attention_backend",
    "kv_cache_dtype",
    "max_model_len",
    "max_num_batched_tokens",
    "max_num_seqs",
    "gpu_memory_utilization",
    "enable_chunked_prefill",
    "enable_prefix_caching",
    "mamba_cache_mode",
    "mamba_ssm_cache_dtype",
    "cudagraph_capture_sizes",
    "prompt_multiplier",
)
_UNSUPPORTED_COMPLETED_RUN_METADATA = frozenset(
    {
        "cudagraph_configured",
        "cudagraph_capture_status",
        "cudagraph_capture_evidence",
        "cudagraph_capture_marker",
    }
)
_ALLOWED_UNSUPPORTED_PROVENANCE_METADATA = frozenset(
    (*_UNSUPPORTED_PROVENANCE_METADATA, "enforce_eager", "workloads", "concurrencies")
)
_ALLOWED_UNSUPPORTED_STAGES = frozenset({"server_startup", "serving_capture"})
_ALLOWED_UNSUPPORTED_ATTEMPT_MODES = frozenset({"eager", "cuda_graph"})
_ALLOWED_UNSUPPORTED_ATTEMPT_OUTCOMES = frozenset(
    {
        "failed",
        "timed_out",
        "cancelled_after_stall",
        "engine_dead",
        "out_of_memory",
        "initialization_error",
    }
)
_REQUIRED_UNSUPPORTED_ATTEMPT_FIELDS = frozenset({"job_id", "mode", "outcome"})
_ALLOWED_UNSUPPORTED_ATTEMPT_FIELDS = frozenset(
    {*_REQUIRED_UNSUPPORTED_ATTEMPT_FIELDS, "node", "elapsed"}
)
_METRIC_SECTIONS = frozenset({"e2e", "oracle", "lookup"})


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _nonempty_string(mapping: dict[str, Any], key: str, label: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} {key} must be a non-empty string")
    return value


def _validate_unsupported_record(
    payload: dict[str, Any],
    backend: str,
    evidence_root: Path | None,
) -> None:
    if backend not in _EXPECTED_RECIPES:
        raise ValueError(f"unexpected unsupported backend: {backend}")
    metric_sections = sorted(_METRIC_SECTIONS & payload.keys())
    if metric_sections:
        raise ValueError(
            f"unsupported status must not contain metric sections: {metric_sections}"
        )
    expected_fields = {"backend", "status", "recipe", "provenance", "failure"}
    if payload.keys() != expected_fields:
        raise ValueError(
            "unsupported status fields do not match schema: "
            f"missing={sorted(expected_fields - payload.keys())}, "
            f"unexpected={sorted(payload.keys() - expected_fields)}"
        )
    if payload.get("backend") != backend:
        raise ValueError(
            "unsupported backend label mismatch: "
            f"expected={backend}, actual={payload.get('backend')}"
        )
    if payload.get("status") != "empirically_unsupported":
        raise ValueError(
            "unsupported status must identify status=empirically_unsupported"
        )

    recipe = payload.get("recipe")
    if not isinstance(recipe, dict):
        raise ValueError("unsupported recipe must be a JSON object")
    expected_recipe = _EXPECTED_RECIPES[backend]
    recipe_mismatches = {
        key: (expected, recipe.get(key))
        for key, expected in expected_recipe.items()
        if recipe.get(key) != expected
    }
    if recipe.keys() != expected_recipe.keys() or recipe_mismatches:
        raise ValueError(f"unsupported recipe mismatch: {recipe_mismatches}")

    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("unsupported provenance must be a JSON object")
    completed_run_metadata = sorted(
        _UNSUPPORTED_COMPLETED_RUN_METADATA & provenance.keys()
    )
    if completed_run_metadata:
        raise ValueError(
            "unsupported status contains completed-run provenance: "
            f"{completed_run_metadata}"
        )
    unexpected_provenance = sorted(
        provenance.keys() - _ALLOWED_UNSUPPORTED_PROVENANCE_METADATA
    )
    if unexpected_provenance:
        raise ValueError(
            f"unexpected unsupported provenance fields: {unexpected_provenance}"
        )
    for key in _UNSUPPORTED_PROVENANCE_METADATA:
        _nonempty_string(provenance, key, "unsupported provenance")
    for key in ("enforce_eager", "workloads", "concurrencies"):
        if key in provenance:
            _nonempty_string(provenance, key, "unsupported provenance")

    failure = payload.get("failure")
    if not isinstance(failure, dict):
        raise ValueError("unsupported failure must be a JSON object")
    expected_failure_fields = {
        "stage",
        "reason_code",
        "message",
        "attempts",
        "evidence",
    }
    if failure.keys() != expected_failure_fields:
        raise ValueError(
            "unsupported failure fields do not match schema: "
            f"missing={sorted(expected_failure_fields - failure.keys())}, "
            f"unexpected={sorted(failure.keys() - expected_failure_fields)}"
        )
    for key in ("reason_code", "message"):
        _nonempty_string(failure, key, "unsupported failure")
    stage = _nonempty_string(failure, "stage", "unsupported failure")
    if stage not in _ALLOWED_UNSUPPORTED_STAGES:
        raise ValueError(f"unsupported failure stage is not empirical: {stage!r}")
    attempts = failure.get("attempts")
    if not isinstance(attempts, list) or not attempts:
        raise ValueError("unsupported failure attempts must be a non-empty list")
    if not all(isinstance(attempt, dict) and attempt for attempt in attempts):
        raise ValueError("unsupported failure attempts must contain non-empty objects")
    observed_modes = set()
    attempt_links = set()
    job_ids = set()
    for index, attempt in enumerate(attempts):
        missing_fields = _REQUIRED_UNSUPPORTED_ATTEMPT_FIELDS - attempt.keys()
        unexpected_fields = attempt.keys() - _ALLOWED_UNSUPPORTED_ATTEMPT_FIELDS
        if missing_fields or unexpected_fields:
            raise ValueError(
                f"unsupported attempt schema mismatch at {index}: "
                f"missing={sorted(missing_fields)}, "
                f"unexpected={sorted(unexpected_fields)}"
            )
        for key in attempt:
            _nonempty_string(attempt, key, f"unsupported attempt {index}")
        mode = attempt["mode"]
        outcome = attempt["outcome"]
        if mode not in _ALLOWED_UNSUPPORTED_ATTEMPT_MODES:
            raise ValueError(f"unsupported attempt mode at {index}: {mode!r}")
        if outcome not in _ALLOWED_UNSUPPORTED_ATTEMPT_OUTCOMES:
            raise ValueError(f"unsupported attempt outcome at {index}: {outcome!r}")
        job_id = attempt["job_id"]
        if job_id in job_ids:
            raise ValueError("unsupported attempt job_id values must be unique")
        job_ids.add(job_id)
        observed_modes.add(mode)
        attempt_links.add((job_id, mode))
    if observed_modes != _ALLOWED_UNSUPPORTED_ATTEMPT_MODES:
        raise ValueError(
            "unsupported failure must include eager and cuda_graph attempts: "
            f"{sorted(observed_modes)}"
        )

    evidence = failure.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        raise ValueError("unsupported evidence must be a non-empty list")
    evidence_links = set()
    evidence_paths = set()
    for index, entry in enumerate(evidence):
        if not isinstance(entry, dict):
            raise ValueError(f"unsupported evidence entry {index} must be an object")
        expected_evidence_fields = {"job_id", "mode", "path", "sha256"}
        if entry.keys() != expected_evidence_fields:
            raise ValueError(
                f"unsupported evidence entry fields do not match schema at {index}: "
                f"missing={sorted(expected_evidence_fields - entry.keys())}, "
                f"unexpected={sorted(entry.keys() - expected_evidence_fields)}"
            )
        evidence_job_id = _nonempty_string(
            entry, "job_id", f"unsupported evidence entry {index}"
        )
        evidence_mode = _nonempty_string(
            entry, "mode", f"unsupported evidence entry {index}"
        )
        evidence_link = (evidence_job_id, evidence_mode)
        if evidence_link not in attempt_links:
            raise ValueError(
                f"unsupported evidence entry {index} does not match an attempt: "
                f"{evidence_link}"
            )
        evidence_value = _nonempty_string(
            entry, "path", f"unsupported evidence entry {index}"
        )
        expected_sha256 = _nonempty_string(
            entry, "sha256", f"unsupported evidence entry {index}"
        )
        if len(expected_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in expected_sha256
        ):
            raise ValueError(f"unsupported evidence entry {index} has invalid sha256")
        evidence_path = Path(evidence_value)
        if not evidence_path.is_absolute():
            evidence_path = (evidence_root or Path.cwd()) / evidence_path
        if not evidence_path.is_file():
            raise ValueError(f"evidence file does not exist: {evidence_path}")
        evidence_path = evidence_path.resolve(strict=True)
        if evidence_path in evidence_paths:
            raise ValueError(
                f"unsupported evidence paths must be unique: {evidence_path}"
            )
        evidence_paths.add(evidence_path)
        actual_sha256 = _sha256(evidence_path)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                "evidence checksum mismatch: "
                f"path={evidence_path}, expected={expected_sha256}, "
                f"actual={actual_sha256}"
            )
        entry["path"] = str(evidence_path)
        evidence_links.add(evidence_link)
    if evidence_links != attempt_links:
        missing_links = sorted(attempt_links - evidence_links)
        raise ValueError(
            f"unsupported evidence must cover every attempt: missing={missing_links}"
        )


def summarize_unsupported_backend(status_path: Path, backend: str) -> dict[str, Any]:
    payload = _load(status_path)
    if not isinstance(payload, dict):
        raise ValueError(f"unsupported status must be a JSON object: {status_path}")
    _validate_unsupported_record(payload, backend, status_path.parent)
    return dict(payload)


def _load_metadata(path: Path) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for line in path.read_text().splitlines():
        key, separator, value = line.partition("=")
        if separator:
            metadata[key] = value
    return metadata


def _positive_metric(result: dict[str, Any], field: str, label: str) -> float:
    value = float(result[field])
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label} {field} must be finite and positive: {value}")
    return value


def _positive_count(row: dict[str, Any], path: Path) -> int:
    value = row.get("invocation_count")
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"invalid invocation_count {value!r} in {path}")
    return value


def _geomean(values: list[float]) -> float:
    if not values:
        raise ValueError("geomean requires at least one value")
    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError(f"geomean requires finite positive values: {values}")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _regret_by_m(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(int(row["m"]), []).append(row)

    summary = []
    for m, group in sorted(grouped.items()):
        speedups = [float(row["speedup"]) for row in group]
        regrets = [100.0 * (speedup - 1.0) for speedup in speedups]
        summary.append(
            {
                "m": m,
                "shape_count": len(group),
                "geomean_speedup": _geomean(speedups),
                "regret_pct_p50": _percentile(regrets, 50.0),
                "regret_pct_p90": _percentile(regrets, 90.0),
                "max_regret_pct": max(regrets),
                "different_tactic_rate": sum(not row["same_tactic"] for row in group)
                / len(group),
            }
        )
    return summary


def _complete_run(root: Path, kind: str) -> Path:
    markers = sorted((root / "serving" / kind).glob("*/COMPLETE"))
    if len(markers) != 1:
        raise ValueError(
            f"expected one complete {kind} run below {root}, found {len(markers)}"
        )
    return markers[0].parent


def _result_files(run_dir: Path) -> dict[tuple[int, int, int], Path]:
    results: dict[tuple[int, int, int], Path] = {}
    for path in sorted(run_dir.glob("result_*.json")):
        parts = path.stem.split("_")
        if len(parts) != 4 or parts[0] != "result":
            continue
        prefixes = ("isl", "osl", "c")
        if any(
            not part.startswith(prefix) for part, prefix in zip(parts[1:], prefixes)
        ):
            continue
        key = (
            int(parts[1][len(prefixes[0]) :]),
            int(parts[2][len(prefixes[1]) :]),
            int(parts[3][len(prefixes[2]) :]),
        )
        results[key] = path
    return results


def _artifact_signature(result: dict[str, Any]) -> str:
    payload = {
        key: result[key]
        for key in (
            "completed",
            "total_input_tokens",
            "total_output_tokens",
            "input_lens",
            "output_lens",
            "generated_texts",
        )
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _summarize_e2e_single(root: Path) -> dict[str, Any]:
    baseline_dir = _complete_run(root, "baseline")
    lookup_dir = _complete_run(root, "lookup")
    baseline_metadata = _load_metadata(baseline_dir / "metadata.txt")
    lookup_metadata = _load_metadata(lookup_dir / "metadata.txt")
    comparable_metadata = (
        "backend_name",
        "oracle_backend",
        "scale_layout",
        "source_commit",
        "flashinfer_commit",
        "expected_vllm_version",
        "flashinfer",
        "flashinfer_file",
        "vllm_version",
        "vllm_file",
        "vllm_compiled_file",
        "gpu_name",
        "driver_version",
        "container",
        "container_sha256",
        "container_size",
        "container_mtime",
        "model",
        "model_config_sha256",
        "model_index_sha256",
        "model_weights_manifest_sha256",
        "tp",
        "linear_backend",
        "trtllm_layout",
        "moe_backend",
        "attention_backend",
        "kv_cache_dtype",
        "max_model_len",
        "max_num_batched_tokens",
        "max_num_seqs",
        "gpu_memory_utilization",
        "enable_chunked_prefill",
        "enable_prefix_caching",
        "mamba_cache_mode",
        "mamba_ssm_cache_dtype",
        "enforce_eager",
        "cudagraph_capture_sizes",
        "cudagraph_configured",
        "cudagraph_capture_status",
        "cudagraph_capture_evidence",
        "cudagraph_capture_marker",
        "workloads",
        "concurrencies",
        "prompt_multiplier",
    )
    missing_metadata = {
        label: [key for key in comparable_metadata if not metadata.get(key)]
        for label, metadata in (
            ("baseline", baseline_metadata),
            ("lookup", lookup_metadata),
        )
    }
    missing_metadata = {label: keys for label, keys in missing_metadata.items() if keys}
    if missing_metadata:
        raise ValueError(f"required execution metadata is missing: {missing_metadata}")
    metadata_mismatches = {
        key: (baseline_metadata.get(key), lookup_metadata.get(key))
        for key in comparable_metadata
        if baseline_metadata.get(key) != lookup_metadata.get(key)
    }
    if metadata_mismatches:
        raise ValueError(
            f"baseline and lookup metadata do not match: {metadata_mismatches}"
        )
    if baseline_metadata["expected_vllm_version"] != baseline_metadata["vllm_version"]:
        raise ValueError(
            "vLLM version mismatch: "
            f"expected={baseline_metadata['expected_vllm_version']}, "
            f"actual={baseline_metadata['vllm_version']}"
        )

    baseline_files = _result_files(baseline_dir)
    lookup_files = _result_files(lookup_dir)
    if baseline_files.keys() != lookup_files.keys():
        raise ValueError(
            "baseline and lookup result files do not match: "
            f"baseline={sorted(baseline_files)}, lookup={sorted(lookup_files)}"
        )

    workloads = []
    prompt_multiplier = int(baseline_metadata["prompt_multiplier"])
    for (isl, osl, concurrency), baseline_path in baseline_files.items():
        baseline = _load(baseline_path)
        lookup = _load(lookup_files[(isl, osl, concurrency)])
        for label, result in (("baseline", baseline), ("lookup", lookup)):
            if result["failed"] != 0:
                raise ValueError(f"{label} has failed requests: {result}")
            expected_requests = concurrency * prompt_multiplier
            if result["completed"] != expected_requests:
                raise ValueError(
                    f"{label} completed {result['completed']} requests, "
                    f"expected {expected_requests}"
                )
            generated_texts = result.get("generated_texts")
            if (
                not isinstance(generated_texts, list)
                or len(generated_texts) != result["completed"]
            ):
                raise ValueError(
                    f"{label} is missing generated texts for "
                    f"ISL={isl}, OSL={osl}, C={concurrency}"
                )
            input_lens = result.get("input_lens")
            output_lens = result.get("output_lens")
            if (
                not isinstance(input_lens, list)
                or len(input_lens) != expected_requests
                or not all(isl <= length <= isl + 1 for length in input_lens)
            ):
                raise ValueError(
                    f"{label} input lengths do not match ISL={isl}, C={concurrency}"
                )
            if (
                not isinstance(output_lens, list)
                or len(output_lens) != expected_requests
                or not all(length == osl for length in output_lens)
            ):
                raise ValueError(
                    f"{label} output lengths do not match OSL={osl}, C={concurrency}"
                )
            if result.get("total_input_tokens") != sum(input_lens):
                raise ValueError(f"{label} total input tokens are inconsistent")
            if result.get("total_output_tokens") != expected_requests * osl:
                raise ValueError(f"{label} total output tokens are inconsistent")
        parity_fields = (
            "completed",
            "total_input_tokens",
            "total_output_tokens",
            "input_lens",
            "output_lens",
        )
        mismatches = {
            field: (baseline.get(field), lookup.get(field))
            for field in parity_fields
            if baseline.get(field) != lookup.get(field)
        }
        if mismatches:
            raise ValueError(
                "baseline and lookup token results do not match for "
                f"ISL={isl}, OSL={osl}, C={concurrency}: {mismatches}"
            )
        if baseline.get("generated_texts") != lookup.get("generated_texts"):
            raise ValueError(
                "baseline and lookup generated text mismatch for "
                f"ISL={isl}, OSL={osl}, C={concurrency}"
            )
        baseline_metrics = {
            field: _positive_metric(baseline, field, "baseline")
            for field in (
                "output_throughput",
                "total_token_throughput",
                "mean_ttft_ms",
                "mean_tpot_ms",
                "duration",
            )
        }
        lookup_metrics = {
            field: _positive_metric(lookup, field, "lookup")
            for field in baseline_metrics
        }
        workloads.append(
            {
                "isl": isl,
                "osl": osl,
                "concurrency": concurrency,
                "completed": lookup["completed"],
                "total_input_tokens": lookup["total_input_tokens"],
                "total_output_tokens": lookup["total_output_tokens"],
                "artifact_signature": _artifact_signature(lookup),
                "baseline_output_throughput": baseline_metrics["output_throughput"],
                "lookup_output_throughput": lookup_metrics["output_throughput"],
                "output_throughput_speedup": (
                    lookup_metrics["output_throughput"]
                    / baseline_metrics["output_throughput"]
                ),
                "baseline_total_token_throughput": baseline_metrics[
                    "total_token_throughput"
                ],
                "lookup_total_token_throughput": lookup_metrics[
                    "total_token_throughput"
                ],
                "total_token_throughput_speedup": (
                    lookup_metrics["total_token_throughput"]
                    / baseline_metrics["total_token_throughput"]
                ),
                "baseline_mean_ttft_ms": baseline_metrics["mean_ttft_ms"],
                "lookup_mean_ttft_ms": lookup_metrics["mean_ttft_ms"],
                "baseline_mean_tpot_ms": baseline_metrics["mean_tpot_ms"],
                "lookup_mean_tpot_ms": lookup_metrics["mean_tpot_ms"],
                "baseline_duration_s": baseline_metrics["duration"],
                "lookup_duration_s": lookup_metrics["duration"],
                "duration_reduction_pct": 100.0
                * (1.0 - lookup_metrics["duration"] / baseline_metrics["duration"]),
            }
        )

    return {
        "baseline_run": str(baseline_dir),
        "lookup_run": str(lookup_dir),
        "run_metadata": {
            "baseline": baseline_metadata,
            "lookup": lookup_metadata,
        },
        "metadata": {key: baseline_metadata.get(key) for key in comparable_metadata},
        "workload_count": len(workloads),
        "geomean_output_throughput_speedup": _geomean(
            [row["output_throughput_speedup"] for row in workloads]
        ),
        "geomean_total_token_throughput_speedup": _geomean(
            [row["total_token_throughput_speedup"] for row in workloads]
        ),
        "workloads": workloads,
    }


def _summarize_e2e(root: Path) -> dict[str, Any]:
    pair_orders = ("baseline-lookup", "lookup-baseline")
    pair_roots = {order: root / "pairs" / order for order in pair_orders}
    present_orders = [order for order, path in pair_roots.items() if path.is_dir()]
    if not present_orders:
        return _summarize_e2e_single(root)
    if present_orders != list(pair_orders):
        raise ValueError(
            "order-balanced comparison requires both pair orders: "
            f"found={present_orders}"
        )

    pairs = {order: _summarize_e2e_single(pair_roots[order]) for order in pair_orders}
    canonical_lookup_path = (root / "oracle" / "lookup.json").resolve(strict=True)
    canonical_lookup_sha256 = hashlib.sha256(
        canonical_lookup_path.read_bytes()
    ).hexdigest()
    reference = pairs[pair_orders[0]]
    reference_workloads = {
        (row["isl"], row["osl"], row["concurrency"]): row
        for row in reference["workloads"]
    }
    for order, pair in pairs.items():
        baseline_metadata = pair["run_metadata"]["baseline"]
        lookup_metadata = pair["run_metadata"]["lookup"]
        hosts = {
            baseline_metadata.get("compute_host"),
            lookup_metadata.get("compute_host"),
        }
        if None in hosts or "" in hosts or len(hosts) != 1:
            raise ValueError(
                f"paired baseline and lookup must use the same compute host: {hosts}"
            )
        job_ids = {baseline_metadata.get("job_id"), lookup_metadata.get("job_id")}
        if None in job_ids or "" in job_ids or len(job_ids) != 1:
            raise ValueError(
                f"paired runs must share the same SLURM allocation: {job_ids}"
            )
        job_id = next(iter(job_ids))
        pair_ids = {baseline_metadata.get("pair_id"), lookup_metadata.get("pair_id")}
        if None in pair_ids or "" in pair_ids or len(pair_ids) != 1:
            raise ValueError(f"paired runs must share one pair_id: {pair_ids}")
        expected_pair_id = f"{job_id}-{order}"
        if pair_ids != {expected_pair_id}:
            raise ValueError(
                f"paired run pair_id must be {expected_pair_id}: {pair_ids}"
            )
        if {
            baseline_metadata.get("pair_order"),
            lookup_metadata.get("pair_order"),
        } != {order}:
            raise ValueError(f"paired run order metadata does not match {order}")
        expected_positions = (
            {"baseline": "1", "lookup": "2"}
            if order == "baseline-lookup"
            else {"baseline": "2", "lookup": "1"}
        )
        actual_positions = {
            "baseline": baseline_metadata.get("pair_position"),
            "lookup": lookup_metadata.get("pair_position"),
        }
        if actual_positions != expected_positions:
            raise ValueError(
                f"paired run positions do not match {order}: {actual_positions}"
            )
        baseline_has_lookup = (
            baseline_metadata.get("lookup_path") != "not_applicable"
            or baseline_metadata.get("lookup_sha256") != "not_applicable"
        )
        if baseline_has_lookup:
            raise ValueError("baseline pair run must not bind a lookup manifest")
        lookup_path_value = lookup_metadata.get("lookup_path")
        lookup_sha256 = lookup_metadata.get("lookup_sha256")
        if not lookup_path_value or not lookup_sha256:
            raise ValueError("lookup pair run is missing lookup manifest provenance")
        lookup_path = Path(lookup_path_value)
        if not lookup_path.is_file():
            raise ValueError(f"lookup manifest is missing: {lookup_path}")
        if lookup_path.resolve() != canonical_lookup_path:
            raise ValueError(
                "measured lookup path is not the canonical lookup manifest: "
                f"measured={lookup_path}, canonical={canonical_lookup_path}"
            )
        actual_lookup_sha256 = hashlib.sha256(lookup_path.read_bytes()).hexdigest()
        if (
            actual_lookup_sha256 != lookup_sha256
            or lookup_sha256 != canonical_lookup_sha256
        ):
            raise ValueError(
                "lookup manifest checksum does not match measured run: "
                f"expected={lookup_sha256}, actual={actual_lookup_sha256}"
            )
        if pair["metadata"] != reference["metadata"]:
            raise ValueError(f"paired execution metadata does not match for {order}")
        workloads = {
            (row["isl"], row["osl"], row["concurrency"]): row
            for row in pair["workloads"]
        }
        if workloads.keys() != reference_workloads.keys():
            raise ValueError(f"paired workloads do not match for {order}")
        mismatched_artifacts = {
            key
            for key in workloads
            if workloads[key]["artifact_signature"]
            != reference_workloads[key]["artifact_signature"]
        }
        if mismatched_artifacts:
            mismatches = sorted(mismatched_artifacts)
            raise ValueError(f"paired workload artifacts do not match: {mismatches}")

    workloads = []
    for key in sorted(reference_workloads):
        pair_rows = [
            (
                order,
                next(
                    row
                    for row in pairs[order]["workloads"]
                    if (row["isl"], row["osl"], row["concurrency"]) == key
                ),
            )
            for order in pair_orders
        ]
        speedups = [row["output_throughput_speedup"] for _, row in pair_rows]
        total_speedups = [row["total_token_throughput_speedup"] for _, row in pair_rows]
        duration_ratios = [
            row["lookup_duration_s"] / row["baseline_duration_s"]
            for _, row in pair_rows
        ]
        first = pair_rows[0][1]
        workloads.append(
            {
                **{
                    field: first[field]
                    for field in (
                        "isl",
                        "osl",
                        "concurrency",
                        "completed",
                        "total_input_tokens",
                        "total_output_tokens",
                        "artifact_signature",
                    )
                },
                "baseline_output_throughput": _geomean(
                    [row["baseline_output_throughput"] for _, row in pair_rows]
                ),
                "lookup_output_throughput": _geomean(
                    [row["lookup_output_throughput"] for _, row in pair_rows]
                ),
                "output_throughput_speedup": _geomean(speedups),
                "output_throughput_speedup_min": min(speedups),
                "output_throughput_speedup_max": max(speedups),
                "baseline_total_token_throughput": _geomean(
                    [row["baseline_total_token_throughput"] for _, row in pair_rows]
                ),
                "lookup_total_token_throughput": _geomean(
                    [row["lookup_total_token_throughput"] for _, row in pair_rows]
                ),
                "total_token_throughput_speedup": _geomean(total_speedups),
                "baseline_mean_ttft_ms": _geomean(
                    [row["baseline_mean_ttft_ms"] for _, row in pair_rows]
                ),
                "lookup_mean_ttft_ms": _geomean(
                    [row["lookup_mean_ttft_ms"] for _, row in pair_rows]
                ),
                "baseline_mean_tpot_ms": _geomean(
                    [row["baseline_mean_tpot_ms"] for _, row in pair_rows]
                ),
                "lookup_mean_tpot_ms": _geomean(
                    [row["lookup_mean_tpot_ms"] for _, row in pair_rows]
                ),
                "baseline_duration_s": _geomean(
                    [row["baseline_duration_s"] for _, row in pair_rows]
                ),
                "lookup_duration_s": _geomean(
                    [row["lookup_duration_s"] for _, row in pair_rows]
                ),
                "duration_reduction_pct": 100.0 * (1.0 - _geomean(duration_ratios)),
                "paired_measurements": [
                    {
                        "pair_order": order,
                        "compute_host": pairs[order]["run_metadata"]["baseline"][
                            "compute_host"
                        ],
                        **row,
                    }
                    for order, row in pair_rows
                ],
            }
        )

    return {
        "baseline_run": [pairs[order]["baseline_run"] for order in pair_orders],
        "lookup_run": [pairs[order]["lookup_run"] for order in pair_orders],
        "pair_count": len(pair_orders),
        "pair_orders": list(pair_orders),
        "metadata": reference["metadata"],
        "workload_count": len(workloads),
        "geomean_output_throughput_speedup": _geomean(
            [row["output_throughput_speedup"] for row in workloads]
        ),
        "geomean_total_token_throughput_speedup": _geomean(
            [row["total_token_throughput_speedup"] for row in workloads]
        ),
        "workloads": workloads,
    }


def validate_comparison_summaries(
    summaries: list[dict[str, Any]],
    unsupported_summaries: list[dict[str, Any]] | None = None,
) -> None:
    if not summaries:
        raise ValueError("comparison requires at least one measured backend summary")

    unsupported_summaries = unsupported_summaries or []
    if len(unsupported_summaries) > 1:
        raise ValueError("comparison supports at most one unsupported backend arm")
    for summary in unsupported_summaries:
        backend = summary.get("backend")
        if not isinstance(backend, str):
            raise ValueError(f"unsupported backend must be a string: {backend!r}")
        _validate_unsupported_record(summary, backend, None)
    measured_backends = [str(summary["backend"]) for summary in summaries]
    unsupported_backends = [
        str(summary["backend"]) for summary in unsupported_summaries
    ]
    all_backends = measured_backends + unsupported_backends
    if len(set(measured_backends)) != len(measured_backends):
        raise ValueError(f"duplicate measured backend summaries: {measured_backends}")
    if len(set(unsupported_backends)) != len(unsupported_backends):
        raise ValueError(
            f"duplicate unsupported backend summaries: {unsupported_backends}"
        )
    if len(set(all_backends)) != len(all_backends):
        raise ValueError(f"backend appears as measured and unsupported: {all_backends}")
    if set(all_backends) != set(_EXPECTED_BACKENDS):
        raise ValueError(
            "comparison requires exactly these backend arms: "
            f"expected={sorted(_EXPECTED_BACKENDS)}, actual={sorted(all_backends)}"
        )

    invalid_measured_statuses = {
        summary["backend"]: summary.get("status")
        for summary in summaries
        if summary.get("status", "measured") != "measured"
    }
    if invalid_measured_statuses:
        raise ValueError(f"invalid measured statuses: {invalid_measured_statuses}")

    reference = summaries[0]
    reference_metadata = reference["e2e"]["metadata"]
    reference_workloads = {
        (row["isl"], row["osl"], row["concurrency"]): row["artifact_signature"]
        for row in reference["e2e"]["workloads"]
    }
    for summary in summaries[1:]:
        metadata = summary["e2e"]["metadata"]
        mismatches = {
            key: (reference_metadata.get(key), metadata.get(key))
            for key in _CROSS_BACKEND_METADATA
            if reference_metadata.get(key) != metadata.get(key)
        }
        if mismatches:
            raise ValueError(
                "cross-backend provenance mismatch: "
                f"{reference['backend']} vs {summary['backend']}: {mismatches}"
            )

        workloads = {
            (row["isl"], row["osl"], row["concurrency"]): row["artifact_signature"]
            for row in summary["e2e"]["workloads"]
        }
        if workloads.keys() != reference_workloads.keys():
            raise ValueError(
                "cross-backend workload mismatch: "
                f"{reference['backend']}={sorted(reference_workloads)}, "
                f"{summary['backend']}={sorted(workloads)}"
            )
        artifact_mismatches = {
            key: (reference_workloads[key], workloads[key])
            for key in reference_workloads
            if reference_workloads[key] != workloads[key]
        }
        if artifact_mismatches:
            raise ValueError(
                "cross-backend artifact mismatch: "
                f"{reference['backend']} vs {summary['backend']}: "
                f"{artifact_mismatches}"
            )

    for summary in unsupported_summaries:
        provenance = summary["provenance"]
        mismatches = {
            key: (reference_metadata.get(key), provenance.get(key))
            for key in _UNSUPPORTED_PROVENANCE_METADATA
            if reference_metadata.get(key) != provenance.get(key)
        }
        if mismatches:
            raise ValueError(
                "unsupported provenance mismatch: "
                f"{reference['backend']} vs {summary['backend']}: {mismatches}"
            )


def build_study_summary(
    summaries: list[dict[str, Any]],
    unsupported_summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    validate_comparison_summaries(summaries, unsupported_summaries)
    measured_by_backend = {summary["backend"]: summary for summary in summaries}
    unsupported_by_backend = {
        summary["backend"]: summary for summary in unsupported_summaries
    }
    measured_backends = [
        backend for backend in _EXPECTED_BACKENDS if backend in measured_by_backend
    ]
    unsupported_backends = [
        backend for backend in _EXPECTED_BACKENDS if backend in unsupported_by_backend
    ]
    backends = []
    for backend in _EXPECTED_BACKENDS:
        if backend in measured_by_backend:
            backends.append({**measured_by_backend[backend], "status": "measured"})
        else:
            backends.append(dict(unsupported_by_backend[backend]))
    has_unsupported = bool(unsupported_backends)
    return {
        "study_status": (
            "complete_with_unsupported_arm" if has_unsupported else "complete"
        ),
        "measured_backends": measured_backends,
        "unsupported_backends": unsupported_backends,
        "metric_comparison_status": "partial" if has_unsupported else "complete",
        "backends": backends,
    }


def _summarize_oracle(root: Path) -> dict[str, Any]:
    report_paths = sorted(root.glob("oracle/shards/*/oracle/report.json"))
    if not report_paths:
        raise ValueError(f"no oracle reports below {root}")
    reports = [_load(path) for path in report_paths]
    rows = [row for report in reports for row in report["shapes"]]
    if not rows:
        raise ValueError("oracle reports contain no shape results")
    backends = {report.get("backend") for report in reports}
    layouts = {report.get("scale_layout") for report in reports}
    flashinfer_commits = {report.get("flashinfer_commit") for report in reports}
    flashinfer_versions = {report.get("flashinfer_version") for report in reports}
    flashinfer_files = {report.get("flashinfer_file") for report in reports}
    container_hashes = {report.get("container_sha256") for report in reports}
    gpus = {report.get("gpu") for report in reports}
    correctness_configs = {
        (
            report.get("correctness", {}).get("minimum_cosine_similarity"),
            report.get("correctness", {}).get("rtol"),
            report.get("correctness", {}).get("atol"),
        )
        for report in reports
    }
    timing_blocks = []
    for path, report in zip(report_paths, reports):
        timing = report.get("timing")
        if not isinstance(timing, dict):
            raise ValueError(f"oracle timing block is incomplete in {path}: {timing}")
        missing_timing = [
            field for field in _ORACLE_TIMING_FIELDS if field not in timing
        ]
        if missing_timing:
            raise ValueError(
                f"oracle timing block is incomplete in {path}: {missing_timing}"
            )
        timing_blocks.append(timing)
    timing = timing_blocks[0]
    mismatched_timing = [
        str(path)
        for path, block in zip(report_paths[1:], timing_blocks[1:])
        if block != timing
    ]
    if mismatched_timing:
        raise ValueError(
            "oracle timing blocks do not match: "
            f"reference={timing}, mismatched={mismatched_timing}"
        )
    if timing["cuda_graph"] is not True or timing["cold_l2_cache"] is not True:
        raise ValueError(
            f"oracle timing requires cuda_graph=true and cold_l2_cache=true: {timing}"
        )
    invalid_timing_counts = {
        field: timing[field]
        for field in _ORACLE_TIMING_FIELDS[2:]
        if isinstance(timing[field], bool)
        or not isinstance(timing[field], int)
        or timing[field] <= 0
    }
    if invalid_timing_counts:
        raise ValueError(
            f"oracle timing counts must be positive integers: {invalid_timing_counts}"
        )
    if (
        len(backends) != 1
        or len(layouts) != 1
        or len(flashinfer_commits) != 1
        or len(flashinfer_versions) != 1
        or len(flashinfer_files) != 1
        or len(container_hashes) != 1
        or len(gpus) != 1
        or len(correctness_configs) != 1
        or any(
            value in (None, "")
            for values in (
                backends,
                layouts,
                flashinfer_commits,
                flashinfer_versions,
                flashinfer_files,
                container_hashes,
                gpus,
            )
            for value in values
        )
        or any(
            value in (None, "") for config in correctness_configs for value in config
        )
    ):
        raise ValueError(
            "oracle report provenance mismatch: "
            f"{backends}, {layouts}, {flashinfer_commits}, {flashinfer_versions}, "
            f"{flashinfer_files}, {container_hashes}, {gpus}, {correctness_configs}"
        )
    min_cosine, rtol, atol = next(iter(correctness_configs))
    correctness_values = [float(min_cosine), float(rtol), float(atol)]
    if any(not math.isfinite(value) or value < 0 for value in correctness_values):
        raise ValueError(f"invalid oracle correctness thresholds: {correctness_values}")
    speedups = [float(row["speedup"]) for row in rows]
    selected_times = [float(row["selected_ms"]) for row in rows]
    oracle_times = [float(row["oracle_ms"]) for row in rows]
    cosine_values = [float(row["oracle_cosine_similarity"]) for row in rows]
    metrics = speedups + selected_times + oracle_times + cosine_values
    if any(not math.isfinite(value) for value in metrics) or any(
        value <= 0 for value in speedups + selected_times + oracle_times
    ):
        raise ValueError("oracle reports contain non-finite or non-positive metrics")
    profiling_wall_times = [float(report["profiling_wall_s"]) for report in reports]
    measured_candidate_gpu_times = [
        float(report["measured_candidate_gpu_s"]) for report in reports
    ]
    if any(
        not math.isfinite(value) or value < 0
        for value in profiling_wall_times + measured_candidate_gpu_times
    ):
        raise ValueError("oracle reports contain invalid profiling times")
    regrets = [100.0 * (speedup - 1.0) for speedup in speedups]
    candidate_counts = [int(row["candidate_count"]) for row in rows]
    finite_pass_count = sum(row.get("oracle_finite") is True for row in rows)
    selected_allclose_pass_count = sum(
        row.get("oracle_matches_selected") is True for row in rows
    )
    bf16_cosine_pass_count = sum(value >= float(min_cosine) for value in cosine_values)
    if (
        finite_pass_count != len(rows)
        or selected_allclose_pass_count != len(rows)
        or bf16_cosine_pass_count != len(rows)
    ):
        raise ValueError(
            "oracle correctness incomplete: "
            f"shapes={len(rows)}, finite={finite_pass_count}, "
            f"selected_allclose={selected_allclose_pass_count}, "
            f"bf16_cosine={bf16_cosine_pass_count}"
        )
    top_regrets = []
    for row in sorted(rows, key=lambda item: float(item["speedup"]), reverse=True)[:20]:
        top_regrets.append(
            {
                "m": int(row["m"]),
                "n": int(row["n"]),
                "k": int(row["k"]),
                "candidate_count": int(row["candidate_count"]),
                "selected_ms": float(row["selected_ms"]),
                "oracle_ms": float(row["oracle_ms"]),
                "regret_pct": 100.0 * (float(row["speedup"]) - 1.0),
                "selected_tactic": row["selected_tactic"],
                "oracle_tactic": row["oracle_tactic"],
            }
        )
    return {
        "backend": next(iter(backends)),
        "scale_layout": next(iter(layouts)),
        "flashinfer_commit": next(iter(flashinfer_commits)),
        "flashinfer_version": next(iter(flashinfer_versions)),
        "flashinfer_file": next(iter(flashinfer_files)),
        "container_sha256": next(iter(container_hashes)),
        "gpu": next(iter(gpus)),
        "timing": timing,
        "shape_count": len(rows),
        "geomean_speedup": _geomean(speedups),
        "max_speedup": max(speedups),
        "max_regret_pct": 100.0 * (max(speedups) - 1.0),
        "different_tactic_count": sum(not row["same_tactic"] for row in rows),
        "candidate_count_min": min(candidate_counts),
        "candidate_count_max": max(candidate_counts),
        "regret_pct_p50": _percentile(regrets, 50.0),
        "regret_pct_p90": _percentile(regrets, 90.0),
        "regret_pct_p99": _percentile(regrets, 99.0),
        "regret_by_m": _regret_by_m(rows),
        "minimum_oracle_cosine_similarity": min(
            float(row["oracle_cosine_similarity"]) for row in rows
        ),
        "correctness": {
            "minimum_cosine_similarity_required": float(min_cosine),
            "rtol": float(rtol),
            "atol": float(atol),
            "finite_pass_count": finite_pass_count,
            "selected_allclose_pass_count": selected_allclose_pass_count,
            "bf16_cosine_pass_count": bf16_cosine_pass_count,
        },
        "measured_candidate_gpu_s": sum(measured_candidate_gpu_times),
        "profiling_wall_s_estimate": max(profiling_wall_times),
        "top_regrets": top_regrets,
        "shapes": rows,
        "reports": [str(path) for path in report_paths],
    }


def _summarize_lookup_single(
    root: Path, *, expected_rank_count: int, manifest_root: Path
) -> dict[str, Any]:
    lookup_dir = _complete_run(root, "lookup")
    sources: dict[tuple[int, int, int, str], set[str]] = {}
    selection_call_counts: dict[tuple[int, int, int, str, str], int] = {}
    ranks: set[str] = set()
    trace_dir = lookup_dir / "traces"
    count_paths = sorted(trace_dir.glob("counts.*.jsonl"))
    unfinished = [
        path for path in count_paths if not path.with_suffix(".complete").is_file()
    ]
    if unfinished:
        raise ValueError(f"unfinished lookup count snapshot: {unfinished}")
    count_pids = {path.name.split(".")[1] for path in count_paths}
    missing_counts = [
        path
        for path in sorted(trace_dir.glob("trace.*.jsonl"))
        if path.name.split(".")[1] not in count_pids
    ]
    if missing_counts:
        raise ValueError(f"lookup traces are missing count snapshots: {missing_counts}")
    trace_paths = count_paths
    if not trace_paths:
        raise ValueError(f"no lookup traces below {lookup_dir}")
    for path in trace_paths:
        for line in path.read_text().splitlines():
            row = json.loads(line)
            key = (
                int(row["m"]),
                int(row["n"]),
                int(row["k"]),
                str(row["runner"]),
            )
            source = str(row["selection_source"])
            ranks.add(str(row.get("rank", "unknown")))
            sources.setdefault(key, set()).add(source)
            count_key = (*key, source)
            selection_call_counts[count_key] = selection_call_counts.get(
                count_key, 0
            ) + _positive_count(row, path)
    expected_ranks = {str(rank) for rank in range(expected_rank_count)}
    if ranks != expected_ranks:
        raise ValueError(
            "incomplete lookup rank coverage: "
            f"expected={sorted(expected_ranks)}, actual={sorted(ranks)}"
        )
    allowed_sources = {"offline_lookup", "default_autotuner"}
    unexpected_sources = {
        source for values in sources.values() for source in values
    } - allowed_sources
    if unexpected_sources:
        raise ValueError(
            f"unexpected lookup selection sources: {sorted(unexpected_sources)}"
        )
    conflicts = {key: value for key, value in sources.items() if len(value) != 1}
    if conflicts:
        raise ValueError(
            f"lookup source changed for the same dispatch key: {conflicts}"
        )
    hit_count = sum(next(iter(value)) == "offline_lookup" for value in sources.values())
    if hit_count == 0:
        raise ValueError(
            "lookup run has zero offline_lookup hits; no lookup performance "
            "comparison is valid"
        )
    miss_count = len(sources) - hit_count
    selection_call_hit_count = sum(
        count
        for (*_, source), count in selection_call_counts.items()
        if source == "offline_lookup"
    )
    selection_call_count = sum(selection_call_counts.values())
    selection_call_miss_count = selection_call_count - selection_call_hit_count
    selection_source_counts = {
        source: sum(
            count
            for (*_, observed_source), count in selection_call_counts.items()
            if observed_source == source
        )
        for source in sorted(allowed_sources)
        if any(key[-1] == source for key in selection_call_counts)
    }
    dispatches = []
    for (m, n, k, runner), source_values in sorted(sources.items()):
        source = next(iter(source_values))
        dispatches.append(
            {
                "m": m,
                "n": n,
                "k": k,
                "runner": runner,
                "selection_source": source,
                "invocation_count": selection_call_counts[(m, n, k, runner, source)],
            }
        )
    lookup_path = manifest_root / "oracle" / "lookup.json"
    if not lookup_path.is_file():
        raise ValueError(f"lookup manifest is missing: {lookup_path}")
    lookup_bytes = lookup_path.read_bytes()
    lookup_payload = json.loads(lookup_bytes)
    manifest_fields = (
        "backend",
        "scale_layout",
        "flashinfer_commit",
        "flashinfer_version",
        "flashinfer_file",
        "container_sha256",
        "gpu",
        "entry_count",
    )
    missing_manifest = [key for key in manifest_fields if not lookup_payload.get(key)]
    if missing_manifest:
        raise ValueError(f"lookup manifest is incomplete: {missing_manifest}")
    coverage_by_m = []
    for m in sorted({key[0] for key in sources}):
        group = [value for key, value in sources.items() if key[0] == m]
        group_hits = sum(next(iter(value)) == "offline_lookup" for value in group)
        coverage_by_m.append(
            {
                "m": m,
                "unique_dispatch_count": len(group),
                "hit_rate": group_hits / len(group),
            }
        )
    return {
        "trace_process_count": len(trace_paths),
        "ranks": sorted(ranks),
        "unique_dispatch_count": len(sources),
        "unique_hit_count": hit_count,
        "unique_miss_count": miss_count,
        "unique_hit_rate": hit_count / len(sources),
        "selection_call_count": selection_call_count,
        "selection_call_hit_count": selection_call_hit_count,
        "selection_call_miss_count": selection_call_miss_count,
        "selection_call_weighted_hit_rate": selection_call_hit_count
        / selection_call_count,
        "selection_source_counts": selection_source_counts,
        "dispatches": dispatches,
        "manifest": {
            "path": str(lookup_path),
            "sha256": hashlib.sha256(lookup_bytes).hexdigest(),
            **{key: lookup_payload[key] for key in manifest_fields},
        },
        "coverage_by_m": coverage_by_m,
    }


def _summarize_lookup(root: Path, *, expected_rank_count: int) -> dict[str, Any]:
    pair_orders = ("baseline-lookup", "lookup-baseline")
    pair_roots = {order: root / "pairs" / order for order in pair_orders}
    present_orders = [order for order, path in pair_roots.items() if path.is_dir()]
    if not present_orders:
        return _summarize_lookup_single(
            root, expected_rank_count=expected_rank_count, manifest_root=root
        )
    if present_orders != list(pair_orders):
        raise ValueError(
            f"lookup summary requires both pair orders: found={present_orders}"
        )

    pairs = {
        order: _summarize_lookup_single(
            pair_roots[order],
            expected_rank_count=expected_rank_count,
            manifest_root=root,
        )
        for order in pair_orders
    }
    reference = pairs[pair_orders[0]]
    dispatch_sources = {
        (row["m"], row["n"], row["k"], row["runner"]): row["selection_source"]
        for row in reference["dispatches"]
    }
    for order, pair in pairs.items():
        observed_sources = {
            (row["m"], row["n"], row["k"], row["runner"]): row["selection_source"]
            for row in pair["dispatches"]
        }
        if observed_sources != dispatch_sources:
            raise ValueError(f"lookup dispatch coverage differs for {order}")
        if pair["manifest"] != reference["manifest"]:
            raise ValueError(f"lookup manifest differs for {order}")

    combined_dispatches = []
    for key, source in sorted(dispatch_sources.items()):
        invocation_count = 0
        for pair in pairs.values():
            row = next(
                row
                for row in pair["dispatches"]
                if (row["m"], row["n"], row["k"], row["runner"]) == key
            )
            invocation_count += int(row["invocation_count"])
        m, n, k, runner = key
        combined_dispatches.append(
            {
                "m": m,
                "n": n,
                "k": k,
                "runner": runner,
                "selection_source": source,
                "invocation_count": invocation_count,
            }
        )

    selection_call_count = sum(pair["selection_call_count"] for pair in pairs.values())
    selection_call_hit_count = sum(
        pair["selection_call_hit_count"] for pair in pairs.values()
    )
    return {
        **reference,
        "pair_count": len(pair_orders),
        "trace_process_count": sum(
            pair["trace_process_count"] for pair in pairs.values()
        ),
        "selection_call_count": selection_call_count,
        "selection_call_hit_count": selection_call_hit_count,
        "selection_call_miss_count": selection_call_count - selection_call_hit_count,
        "selection_call_weighted_hit_rate": selection_call_hit_count
        / selection_call_count,
        "selection_source_counts": {
            source: sum(
                pair["selection_source_counts"].get(source, 0)
                for pair in pairs.values()
            )
            for source in {
                source
                for pair in pairs.values()
                for source in pair["selection_source_counts"]
            }
        },
        "dispatches": combined_dispatches,
    }


def summarize_backend(root: Path, backend: str) -> dict[str, Any]:
    e2e = _summarize_e2e(root)
    oracle = _summarize_oracle(root)
    expected = e2e["metadata"]
    if backend != expected["backend_name"]:
        expected_backend = expected["backend_name"]
        raise ValueError(
            f"backend label mismatch: expected={expected_backend}, got={backend}"
        )
    oracle_metadata = {
        "oracle_backend": oracle["backend"],
        "scale_layout": oracle["scale_layout"],
        "flashinfer_commit": oracle["flashinfer_commit"],
        "flashinfer_version": oracle["flashinfer_version"],
        "flashinfer_file": oracle["flashinfer_file"],
        "container_sha256": oracle["container_sha256"],
        "gpu": oracle["gpu"],
    }
    expected_oracle_metadata = {
        "oracle_backend": expected["oracle_backend"],
        "scale_layout": expected["scale_layout"],
        "flashinfer_commit": expected["flashinfer_commit"],
        "flashinfer_version": expected["flashinfer"],
        "flashinfer_file": expected["flashinfer_file"],
        "container_sha256": expected["container_sha256"],
        "gpu": expected["gpu_name"],
    }
    if oracle_metadata != expected_oracle_metadata:
        raise ValueError(
            "serving and oracle provenance do not match: "
            f"serving={expected_oracle_metadata}, oracle={oracle_metadata}"
        )
    lookup = _summarize_lookup(root, expected_rank_count=int(expected["tp"]))
    expected_lookup_manifest = {
        "backend": expected["oracle_backend"],
        "scale_layout": expected["scale_layout"],
        "flashinfer_commit": expected["flashinfer_commit"],
        "flashinfer_version": expected["flashinfer"],
        "flashinfer_file": expected["flashinfer_file"],
        "container_sha256": expected["container_sha256"],
        "gpu": expected["gpu_name"],
        "entry_count": oracle["shape_count"],
    }
    manifest_mismatches = {
        key: (value, lookup["manifest"].get(key))
        for key, value in expected_lookup_manifest.items()
        if lookup["manifest"].get(key) != value
    }
    if manifest_mismatches:
        raise ValueError(
            f"serving and lookup provenance do not match: {manifest_mismatches}"
        )
    return {
        "backend": backend,
        "result_root": str(root),
        "e2e": e2e,
        "oracle": oracle,
        "lookup": lookup,
    }


def _write_csv(path: Path, summaries: list[dict[str, Any]]) -> None:
    rows = []
    for summary in summaries:
        if summary.get("status", "measured") != "measured":
            continue
        for workload in summary["e2e"]["workloads"]:
            rows.append({"backend": summary["backend"], **workload})
    if not rows:
        raise ValueError("E2E CSV requires at least one measured workload")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result",
        action="append",
        default=[],
        metavar="BACKEND=PATH",
    )
    parser.add_argument(
        "--unsupported",
        action="append",
        default=[],
        metavar="BACKEND=STATUS_JSON",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()
    summaries = []
    for value in args.result:
        backend, separator, root = value.partition("=")
        if not separator:
            parser.error(f"invalid --result {value!r}; expected BACKEND=PATH")
        summaries.append(summarize_backend(Path(root), backend))
    unsupported_summaries = []
    for value in args.unsupported:
        backend, separator, status_path = value.partition("=")
        if not separator:
            parser.error(
                f"invalid --unsupported {value!r}; expected BACKEND=STATUS_JSON"
            )
        unsupported_summaries.append(
            summarize_unsupported_backend(Path(status_path), backend)
        )
    if unsupported_summaries:
        output = build_study_summary(summaries, unsupported_summaries)
        csv_records = output["backends"]
    else:
        validate_comparison_summaries(summaries)
        output = {"backends": summaries}
        csv_records = summaries
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2) + "\n")
    _write_csv(args.output_csv, csv_records)


if __name__ == "__main__":
    main()
