#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Summarize exact-shape MXFP8 oracle and serving A/B artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


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


def _summarize_e2e(root: Path) -> dict[str, Any]:
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
        "gpu_name",
        "driver_version",
        "container",
        "container_sha256",
        "model",
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
    gpus = {report.get("gpu") for report in reports}
    if (
        len(backends) != 1
        or len(layouts) != 1
        or len(flashinfer_commits) != 1
        or len(gpus) != 1
        or any(
            value in (None, "")
            for values in (backends, layouts, flashinfer_commits, gpus)
            for value in values
        )
    ):
        raise ValueError(
            "oracle report provenance mismatch: "
            f"{backends}, {layouts}, {flashinfer_commits}, {gpus}"
        )
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
        "gpu": next(iter(gpus)),
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
        "measured_candidate_gpu_s": sum(measured_candidate_gpu_times),
        "profiling_wall_s_estimate": max(profiling_wall_times),
        "top_regrets": top_regrets,
        "reports": [str(path) for path in report_paths],
    }


def _summarize_lookup(root: Path) -> dict[str, Any]:
    lookup_dir = _complete_run(root, "lookup")
    sources: dict[tuple[int, int, int, str], set[str]] = {}
    selection_call_counts: dict[tuple[int, int, int, str, str], int] = {}
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
            sources.setdefault(key, set()).add(source)
            count_key = (*key, source)
            selection_call_counts[count_key] = selection_call_counts.get(
                count_key, 0
            ) + _positive_count(row, path)
    conflicts = {key: value for key, value in sources.items() if len(value) != 1}
    if conflicts:
        raise ValueError(
            f"lookup source changed for the same dispatch key: {conflicts}"
        )
    hit_count = sum(next(iter(value)) == "offline_lookup" for value in sources.values())
    miss_count = len(sources) - hit_count
    selection_call_hit_count = sum(
        count
        for (*_, source), count in selection_call_counts.items()
        if source == "offline_lookup"
    )
    selection_call_count = sum(selection_call_counts.values())
    selection_call_miss_count = selection_call_count - selection_call_hit_count
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
        "unique_dispatch_count": len(sources),
        "unique_hit_count": hit_count,
        "unique_miss_count": miss_count,
        "unique_hit_rate": hit_count / len(sources),
        "selection_call_count": selection_call_count,
        "selection_call_hit_count": selection_call_hit_count,
        "selection_call_miss_count": selection_call_miss_count,
        "selection_call_weighted_hit_rate": selection_call_hit_count
        / selection_call_count,
        "coverage_by_m": coverage_by_m,
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
        "gpu": oracle["gpu"],
    }
    expected_oracle_metadata = {
        "oracle_backend": expected["oracle_backend"],
        "scale_layout": expected["scale_layout"],
        "flashinfer_commit": expected["flashinfer_commit"],
        "gpu": expected["gpu_name"],
    }
    if oracle_metadata != expected_oracle_metadata:
        raise ValueError(
            "serving and oracle provenance do not match: "
            f"serving={expected_oracle_metadata}, oracle={oracle_metadata}"
        )
    return {
        "backend": backend,
        "result_root": str(root),
        "e2e": e2e,
        "oracle": oracle,
        "lookup": _summarize_lookup(root),
    }


def _write_csv(path: Path, summaries: list[dict[str, Any]]) -> None:
    rows = []
    for summary in summaries:
        for workload in summary["e2e"]["workloads"]:
            rows.append({"backend": summary["backend"], **workload})
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
        required=True,
        metavar="BACKEND=PATH",
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
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps({"backends": summaries}, indent=2) + "\n")
    _write_csv(args.output_csv, summaries)


if __name__ == "__main__":
    main()
