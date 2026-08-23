#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# pyright: reportMissingImports=false

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

PHASES = ("prefill", "decode")
SCOPES = ("bf16", "mxfp8")
ARM_ORDER = tuple((phase, scope) for phase in PHASES for scope in SCOPES)

REQUIRED_METADATA_KEYS = (
    "phase",
    "scope",
    "precision",
    "vllm_head",
    "container",
    "container_sha256",
    "model",
    "ntrace_runtime",
    "ntrace_native",
    "tp",
    "dp",
    "pp",
    "expert_parallel",
    "batch_size",
    "concurrency",
    "num_requests",
    "isl",
    "osl",
    "cuda_graph",
    "linear_backend",
    "moe_backend",
    "kv_cache_dtype",
    "bench_seed",
    "warmup_seed",
)
INTEGER_METADATA_KEYS = {
    "tp",
    "dp",
    "pp",
    "expert_parallel",
    "batch_size",
    "concurrency",
    "num_requests",
    "isl",
    "osl",
    "bench_seed",
    "warmup_seed",
}
EXPECTED_PHASE_SHAPES = {
    "prefill": {"isl": 10_000, "osl": 1},
    "decode": {"isl": 1_000, "osl": 256},
}
EXPECTED_FIXED_METADATA: dict[str, object] = {
    "tp": 8,
    "dp": 1,
    "pp": 1,
    "expert_parallel": 1,
    "batch_size": 8,
    "concurrency": 8,
    "num_requests": 8,
    "cuda_graph": "FULL_AND_PIECEWISE",
    "moe_backend": "flashinfer_trtllm",
    "kv_cache_dtype": "auto",
    "bench_seed": 17,
    "warmup_seed": 117,
}
SHARED_METADATA_KEYS = (
    "vllm_head",
    "container",
    "container_sha256",
    "ntrace_runtime",
    "ntrace_native",
    *EXPECTED_FIXED_METADATA,
)
SCOPE_METADATA_KEYS = ("precision", "model", "linear_backend")

BENCHMARK_CONFIG_FIELDS = (
    "isl",
    "osl",
    "batch_sizes",
)
BENCHMARK_RESULT_FIELDS = (
    "bs",
    "isl",
    "osl",
    "actual_output_tokens",
    "expected_output_tokens",
    "tokens_ok",
    "latency_med_s",
    "output_tok_s",
)
BENCHMARK_DELTA_FIELDS = (
    "latency_med_s",
    "output_tok_s",
    "total_tok_s",
    "request_throughput",
    "mean_ttft_ms",
    "mean_tpot_ms",
)
TIMING_DELTA_FIELDS = (
    "window_sum_ns",
    "window_union_ns",
    "gpu_sum_ns",
    "gpu_union_ns",
    "memop_union_ns",
    "activity_union_ns",
    "no_recorded_activity_ns",
    "overlap_factor",
)


class ValidationError(ValueError):
    """Raised when an arm does not satisfy this experiment's contract."""


def classify_kernel(name: str) -> str:
    lower = name.lower()
    if "moe::dev::routing" in lower:
        return "moe_routing"
    if "moe::dev::finalize" in lower:
        return "moe_finalize"
    if "bmm_" in lower or "trtllm_fp8_block_scale_moe" in lower:
        return "moe_gemm"
    if (
        "sm100blockscaled" in lower
        or "dense_blockscaled_gemm" in lower
        or ("gemm_" in lower and "mxe4m3" in lower)
    ):
        return "dense_mxfp8_gemm"
    if "nvjet_sm100_" in lower or "cublaslt::splitkreduce_kernel" in lower:
        return "dense_bf16_gemm"
    if "quantize" in lower and ("mxfp8" in lower or "block_size" in lower):
        return "mxfp8_quantize"
    if "nccl" in lower or "allreduce" in lower or "all_reduce" in lower:
        return "communication"
    if any(token in lower for token in ("fmha", "attention", "softmax")):
        return "attention"
    if any(
        token in lower
        for token in (
            "causal_conv",
            "selective_scan",
            "selective_state",
            "state_passing",
            "chunk_scan",
            "chunk_state",
        )
    ):
        return "mamba"
    if any(
        token in lower for token in ("rms_norm", "rmsnorm", "layer_norm", "layernorm")
    ):
        return "normalization"
    if "direct_copy" in lower or "copy_kernel" in lower:
        return "copy"
    return "other"


def union_duration_ns(intervals: Iterable[tuple[int, int]]) -> int:
    total = 0
    merged_end: int | None = None
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if merged_end is None or start >= merged_end:
            total += end - start
            merged_end = end
        elif end > merged_end:
            total += end - merged_end
            merged_end = end
    return total


def _find_one(root: Path, name: str, *, required: bool) -> Path | None:
    candidates = sorted(root.rglob(name))
    if not candidates and required:
        raise ValidationError(f"missing required artifact {name} under {root}")
    if len(candidates) > 1:
        joined = ", ".join(str(path) for path in candidates)
        raise ValidationError(f"ambiguous {name} under {root}: {joined}")
    return candidates[0] if candidates else None


def _parse_metadata(
    run_dir: Path, expected_phase: str, expected_scope: str
) -> dict[str, Any]:
    path = run_dir / "metadata.env"
    if not path.is_file():
        raise ValidationError(f"missing required metadata: {path}")

    raw: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            raise ValidationError(f"invalid metadata line {path}:{line_number}")
        key, value = stripped.split("=", 1)
        if key in raw:
            raise ValidationError(f"duplicate metadata key {key!r} in {path}")
        raw[key] = value

    missing = [key for key in REQUIRED_METADATA_KEYS if not raw.get(key)]
    if missing:
        raise ValidationError(
            f"missing required metadata in {path}: {', '.join(missing)}"
        )

    metadata: dict[str, Any] = dict(raw)
    for key in INTEGER_METADATA_KEYS:
        try:
            metadata[key] = int(raw[key])
        except ValueError as error:
            raise ValidationError(
                f"metadata {key} must be an integer in {path}: {raw[key]!r}"
            ) from error

    expected_values: dict[str, object] = {
        "phase": expected_phase,
        "scope": expected_scope,
        "precision": expected_scope,
        **EXPECTED_PHASE_SHAPES[expected_phase],
        **EXPECTED_FIXED_METADATA,
        "linear_backend": "vllm_default" if expected_scope == "bf16" else "auto",
    }
    errors = [
        f"expected {key}={expected!r}, found {metadata[key]!r}"
        for key, expected in expected_values.items()
        if metadata[key] != expected
    ]
    if errors:
        raise ValidationError(f"metadata mismatch in {path}: {'; '.join(errors)}")
    return metadata


def _optional_fields(
    source: Mapping[str, Any], fields: Sequence[str], prefix: str
) -> tuple[dict[str, Any], list[str]]:
    selected = {field: source[field] for field in fields if field in source}
    unavailable = [f"{prefix}.{field}" for field in fields if field not in source]
    return selected, unavailable


def _validate_benchmark(
    path: Path,
    metadata: Mapping[str, Any],
    config: Mapping[str, Any],
    result: Mapping[str, Any],
) -> None:
    comparisons = (
        ("config.isl", config.get("isl"), metadata["isl"]),
        ("config.osl", config.get("osl"), metadata["osl"]),
        ("result.bs", result.get("bs"), metadata["batch_size"]),
        ("result.isl", result.get("isl"), metadata["isl"]),
        ("result.osl", result.get("osl"), metadata["osl"]),
    )
    errors = [
        f"{field}={actual!r} contradicts metadata "
        f"{field.rsplit('.', 1)[-1]}={expected!r}"
        for field, actual, expected in comparisons
        if actual is not None and actual != expected
    ]

    batch_sizes = config.get("batch_sizes")
    if batch_sizes is not None and metadata["batch_size"] not in batch_sizes:
        errors.append(
            f"config.batch_sizes={batch_sizes!r} does not contain metadata "
            f"batch_size={metadata['batch_size']!r}"
        )

    expected_tokens = metadata["num_requests"] * metadata["osl"]
    for field in ("actual_output_tokens", "expected_output_tokens"):
        value = result.get(field)
        if value is not None and value != expected_tokens:
            errors.append(
                f"result.{field}={value!r} contradicts metadata-derived "
                f"output_tokens={expected_tokens!r}"
            )
    if result.get("tokens_ok") is False:
        errors.append("result.tokens_ok is false")

    if errors:
        raise ValidationError(f"benchmark mismatch in {path}: {'; '.join(errors)}")


def _read_benchmark(run_dir: Path, metadata: Mapping[str, Any]) -> dict[str, Any]:
    path = _find_one(run_dir, "result_bench_*.json", required=False)
    all_optional = [
        *(f"config.{field}" for field in BENCHMARK_CONFIG_FIELDS),
        *(f"result.{field}" for field in BENCHMARK_RESULT_FIELDS),
    ]
    if path is None:
        return {
            "available": False,
            "path": None,
            "config": {},
            "result": {},
            "unavailable_fields": all_optional,
        }

    try:
        document = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValidationError(f"cannot parse benchmark JSON {path}: {error}") from error
    if not isinstance(document, dict):
        raise ValidationError(f"benchmark JSON must contain an object: {path}")

    config_raw = document.get("config", {})
    results_raw = document.get("results", [])
    if not isinstance(config_raw, dict) or not isinstance(results_raw, list):
        raise ValidationError(f"invalid config/results structure in {path}")
    if any(not isinstance(item, dict) for item in results_raw):
        raise ValidationError(f"benchmark results must contain objects in {path}")

    matching = [
        item for item in results_raw if item.get("bs") == metadata["batch_size"]
    ]
    if len(matching) > 1:
        raise ValidationError(
            "multiple benchmark results for batch size "
            f"{metadata['batch_size']} in {path}"
        )
    if matching:
        result_raw = matching[0]
    elif len(results_raw) == 1:
        result_raw = results_raw[0]
    elif results_raw:
        raise ValidationError(
            f"no benchmark result for batch size {metadata['batch_size']} in {path}"
        )
    else:
        result_raw = {}

    _validate_benchmark(path, metadata, config_raw, result_raw)
    config, unavailable_config = _optional_fields(
        config_raw, BENCHMARK_CONFIG_FIELDS, "config"
    )
    result_fields = tuple(
        dict.fromkeys((*BENCHMARK_RESULT_FIELDS, *BENCHMARK_DELTA_FIELDS))
    )
    result = {
        field: result_raw[field] for field in result_fields if field in result_raw
    }
    unavailable_result = [
        f"result.{field}"
        for field in BENCHMARK_RESULT_FIELDS
        if field not in result_raw
    ]
    return {
        "available": True,
        "path": str(path),
        "config": config,
        "result": result,
        "unavailable_fields": unavailable_config + unavailable_result,
    }


def _decode_windows(metadata: Mapping[bytes, bytes] | None) -> list[tuple[int, int]]:
    metadata = metadata or {}
    try:
        starts = json.loads(metadata[b"ntrace.iter_start_ns"])
        ends = json.loads(metadata[b"ntrace.iter_end_ns"])
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValidationError(
            "trace is missing valid ntrace iteration windows"
        ) from error
    if (
        not isinstance(starts, list)
        or not isinstance(ends, list)
        or not starts
        or len(starts) != len(ends)
    ):
        raise ValidationError(
            "trace iteration windows must be non-empty equal-length lists"
        )

    windows: list[tuple[int, int]] = []
    for start, end in zip(starts, ends, strict=True):
        if not isinstance(start, int) or not isinstance(end, int) or end <= start:
            raise ValidationError(f"invalid trace iteration window: {(start, end)!r}")
        windows.append((start, end))
    return sorted(windows)


def _intersections(
    start: int, end: int, windows: Sequence[tuple[int, int]]
) -> list[tuple[int, int]]:
    return [
        (max(start, window_start), min(end, window_end))
        for window_start, window_end in windows
        if start < window_end and end > window_start
    ]


def _nearest_rank(values: Sequence[int], percentile: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def _quantiles_ns(values: Sequence[int]) -> dict[str, int | None]:
    return {
        "count": len(values),
        "p50": _nearest_rank(values, 0.50),
        "p95": _nearest_rank(values, 0.95),
        "p99": _nearest_rank(values, 0.99),
    }


def _compress_sequence(
    records: Sequence[dict[str, Any]],
    origin_ns: int,
    limit: int,
    *,
    order_scope: str,
    causal: bool,
    caveat: str,
) -> dict[str, Any]:
    segments: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    total_segments = 0
    for record in records:
        if current is not None and current["name"] == record["name"]:
            current["consecutive_count"] += 1
            current["total_ns"] += record["duration_ns"]
            current["end_offset_ns"] = max(
                current["end_offset_ns"], record["end_ns"] - origin_ns
            )
            continue
        if current is not None:
            total_segments += 1
            if len(segments) < limit:
                segments.append(current)
        current = {
            "name": record["name"],
            "category": record["category"],
            "consecutive_count": 1,
            "total_ns": record["duration_ns"],
            "start_offset_ns": record["start_ns"] - origin_ns,
            "end_offset_ns": record["end_ns"] - origin_ns,
        }
    if current is not None:
        total_segments += 1
        if len(segments) < limit:
            segments.append(current)
    return {
        "order_scope": order_scope,
        "ordering": "chronological by clipped kernel start; row order breaks ties",
        "causal": causal,
        "caveat": caveat,
        "compression": "adjacent identical profiler-visible kernel names",
        "total_segments": total_segments,
        "emitted_segments": len(segments),
        "truncated": len(segments) < total_segments,
        "segments": segments,
    }


def _summarize_memops(
    records_path: Path, windows: Sequence[tuple[int, int]]
) -> tuple[dict[str, Any], list[tuple[int, int]]]:
    path = records_path.with_name("ntrace_memops_rank0.parquet")
    if not path.is_file():
        return {
            "available": False,
            "path": None,
            "record_counts": None,
        }, []

    required_columns = ("start_ns", "end_ns")
    try:
        schema = pq.read_schema(path)
    except Exception as error:
        raise ValidationError(
            f"cannot read rank-0 memops schema {path}: {error}"
        ) from error
    missing = [name for name in required_columns if name not in schema.names]
    if missing:
        raise ValidationError(
            f"memops trace {path} is missing columns: {', '.join(missing)}"
        )
    try:
        table = pq.read_table(path, columns=list(required_columns))
    except Exception as error:
        raise ValidationError(f"cannot read rank-0 memops {path}: {error}") from error

    data = table.to_pydict()
    intervals: list[tuple[int, int]] = []
    included = 0
    excluded_nonpositive = 0
    excluded_outside = 0
    for start, end in zip(data["start_ns"], data["end_ns"], strict=True):
        if start is None or end is None or end <= start:
            excluded_nonpositive += 1
            continue
        clipped = _intersections(start, end, windows)
        if not clipped:
            excluded_outside += 1
            continue
        included += 1
        intervals.extend(clipped)
    return {
        "available": True,
        "path": str(path),
        "record_counts": {
            "parquet_rows": table.num_rows,
            "included_memops": included,
            "excluded_nonpositive_duration": excluded_nonpositive,
            "excluded_outside_windows": excluded_outside,
        },
    }, intervals


def _summarize_graph_replays(
    records: Sequence[dict[str, Any]],
    available_columns: set[str],
    expected_replays: int | None,
) -> dict[str, Any]:
    if expected_replays is None:
        return {
            "status": "not_applicable",
            "expected_replays": None,
            "reason": "decode replay validation is only required for decode arms",
        }
    if expected_replays <= 0:
        raise ValidationError(
            "expected decode graph replay count must be positive, "
            f"got {expected_replays}"
        )

    provenance_columns = {"source", "graph_id", "graph_node_id"}
    missing = sorted(provenance_columns - available_columns)
    if missing:
        raise ValidationError(
            "decode graph replay validation requires trace columns: "
            + ", ".join(missing)
        )

    replay_records = [
        record
        for record in records
        if str(record.get("source") or "").lower() in {"replay", "graph_replay"}
    ]
    incomplete = [
        record
        for record in replay_records
        if record.get("graph_id") is None or record.get("graph_node_id") is None
    ]
    if incomplete:
        raise ValidationError(
            "decode graph replay validation found "
            f"{len(incomplete)} replay records without graph provenance"
        )
    if not replay_records:
        raise ValidationError(
            "decode graph replay validation found no timed replay records"
        )

    by_graph: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for record in replay_records:
        by_graph[record["graph_id"]].append(record)
    dominant_graph_id = max(
        by_graph,
        key=lambda graph_id: (len(by_graph[graph_id]), str(graph_id)),
    )
    dominant_graph_records = by_graph[dominant_graph_id]

    by_node: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for record in dominant_graph_records:
        by_node[record["graph_node_id"]].append(record)
    for occurrences in by_node.values():
        occurrences.sort(key=lambda item: (item["start_ns"], item["row_index"]))

    occurrence_histogram = Counter(len(occurrences) for occurrences in by_node.values())
    observed_replays = max(
        occurrence_histogram,
        key=lambda occurrence_count: (
            occurrence_histogram[occurrence_count],
            occurrence_count,
        ),
    )
    dominant_node_ids = [
        node_id
        for node_id, occurrences in by_node.items()
        if len(occurrences) == observed_replays
    ]
    if observed_replays != expected_replays:
        raise ValidationError(
            "decode graph replay count mismatch: "
            f"expected {expected_replays} from metadata OSL-1, "
            f"observed {observed_replays} on dominant graph {dominant_graph_id}"
        )

    anchor_node_id = min(
        dominant_node_ids,
        key=lambda node_id: (by_node[node_id][0]["start_ns"], str(node_id)),
    )
    replay_spans = [
        max(by_node[node_id][index]["end_ns"] for node_id in dominant_node_ids)
        - min(by_node[node_id][index]["start_ns"] for node_id in dominant_node_ids)
        for index in range(observed_replays)
    ]
    anchor_starts = [record["start_ns"] for record in by_node[anchor_node_id]]
    anchor_periods = [
        current - previous
        for previous, current in zip(anchor_starts, anchor_starts[1:])
    ]
    category_totals: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    kernel_totals: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    graph_kernel_sum = 0
    for record in dominant_graph_records:
        duration = record["duration_ns"]
        category_totals[record["category"]][0] += 1
        category_totals[record["category"]][1] += duration
        kernel_totals[record["name"]][0] += 1
        kernel_totals[record["name"]][1] += duration
        graph_kernel_sum += duration

    categories = {
        category: {
            "count": count,
            "total_ns": total_ns,
            "mean_per_replay_ns": total_ns / observed_replays,
            "share_of_graph_kernel_sum": (
                total_ns / graph_kernel_sum if graph_kernel_sum else None
            ),
        }
        for category, (count, total_ns) in sorted(category_totals.items())
    }
    top_kernels = [
        {
            "name": name,
            "count": count,
            "total_ns": total_ns,
            "mean_per_replay_ns": total_ns / observed_replays,
        }
        for name, (count, total_ns) in sorted(
            kernel_totals.items(), key=lambda item: (-item[1][1], item[0])
        )[:20]
    ]
    return {
        "status": "validated",
        "expected_replays": expected_replays,
        "expected_definition": "metadata.osl - 1",
        "observed_replays": observed_replays,
        "replay_source_values": sorted(
            {str(record["source"]) for record in replay_records}
        ),
        "graph_count": len(by_graph),
        "dominant_graph_id": dominant_graph_id,
        "dominant_graph_replay_records": len(dominant_graph_records),
        "dominant_graph_nodes_per_replay": len(dominant_node_ids),
        "non_dominant_graph_nodes": len(by_node) - len(dominant_node_ids),
        "graph_node_occurrence_histogram": {
            str(count): node_count
            for count, node_count in sorted(occurrence_histogram.items())
        },
        "anchor_graph_node_id": anchor_node_id,
        "replay_span_ns": _quantiles_ns(replay_spans),
        "anchor_period_ns": _quantiles_ns(anchor_periods),
        "graph_kernel_sum_ns": graph_kernel_sum,
        "graph_kernel_sum_per_replay_ns": graph_kernel_sum / observed_replays,
        "categories": categories,
        "top_kernels": top_kernels,
        "alignment": (
            "nth chronological occurrence of each dominant graph node; replay span "
            "is max(end)-min(start), anchor is the earliest dominant node"
        ),
    }


def summarize_trace(
    path: Path,
    *,
    sequence_limit: int = 200,
    expected_decode_replays: int | None = None,
) -> dict[str, Any]:
    if sequence_limit < 0:
        raise ValidationError("sequence_limit must be non-negative")
    required_columns = (
        "kernel_name_demangled",
        "symbol_name",
        "start_ns",
        "duration_ns",
    )
    optional_columns = (
        "source",
        "stream_id",
        "graph_id",
        "graph_node_id",
    )
    try:
        schema = pq.read_schema(path)
    except Exception as error:
        raise ValidationError(
            f"cannot read rank-0 trace schema {path}: {error}"
        ) from error
    missing_columns = [name for name in required_columns if name not in schema.names]
    if missing_columns:
        raise ValidationError(
            f"trace {path} is missing columns: {', '.join(missing_columns)}"
        )
    selected_columns = [
        *required_columns,
        *(name for name in optional_columns if name in schema.names),
    ]
    try:
        table = pq.read_table(path, columns=selected_columns)
    except Exception as error:
        raise ValidationError(f"cannot read rank-0 trace {path}: {error}") from error

    windows = _decode_windows(table.schema.metadata)
    available_columns = set(table.column_names)
    data = table.to_pydict()
    categories: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    kernels: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    included: list[dict[str, Any]] = []
    gpu_intervals: list[tuple[int, int]] = []
    excluded_nonpositive = 0
    excluded_outside = 0

    for row_index in range(table.num_rows):
        demangled = data["kernel_name_demangled"][row_index]
        symbol = data["symbol_name"][row_index]
        start = data["start_ns"][row_index]
        duration = data["duration_ns"][row_index]
        if start is None or duration is None or duration <= 0:
            excluded_nonpositive += 1
            continue
        clipped = _intersections(start, start + duration, windows)
        if not clipped:
            excluded_outside += 1
            continue

        clipped_duration = sum(end - clipped_start for clipped_start, end in clipped)
        name = demangled or symbol or "<unknown>"
        category = classify_kernel(name)
        categories[category][0] += 1
        categories[category][1] += clipped_duration
        kernels[name][0] += 1
        kernels[name][1] += clipped_duration
        gpu_intervals.extend(clipped)
        included.append(
            {
                "name": name,
                "category": category,
                "start_ns": clipped[0][0],
                "end_ns": clipped[-1][1],
                "duration_ns": clipped_duration,
                "row_index": row_index,
                "source": data["source"][row_index]
                if "source" in available_columns
                else None,
                "stream_id": data["stream_id"][row_index]
                if "stream_id" in available_columns
                else None,
                "graph_id": data["graph_id"][row_index]
                if "graph_id" in available_columns
                else None,
                "graph_node_id": data["graph_node_id"][row_index]
                if "graph_node_id" in available_columns
                else None,
            }
        )

    included.sort(key=lambda item: (item["start_ns"], item["row_index"]))
    window_sum = sum(end - start for start, end in windows)
    window_union = union_duration_ns(windows)
    gpu_sum = sum(item["duration_ns"] for item in included)
    gpu_union = union_duration_ns(gpu_intervals)
    memops, memop_intervals = _summarize_memops(path, windows)
    memop_union = union_duration_ns(memop_intervals) if memops["available"] else None
    activity_union = union_duration_ns([*gpu_intervals, *memop_intervals])
    no_recorded_activity = max(0, window_union - activity_union)
    overlap_factor = gpu_sum / gpu_union if gpu_union else None
    category_report = {
        category: {
            "count": count,
            "total_ns": total_ns,
            "share_of_gpu_sum": total_ns / gpu_sum if gpu_sum else None,
        }
        for category, (count, total_ns) in sorted(categories.items())
    }
    top_kernels = [
        {"name": name, "count": count, "total_ns": total_ns}
        for name, (count, total_ns) in sorted(
            kernels.items(), key=lambda item: (-item[1][1], item[0])
        )[:40]
    ]
    global_sequence = _compress_sequence(
        included,
        windows[0][0],
        limit=sequence_limit,
        order_scope="global_observed_timestamp",
        causal=False,
        caveat=(
            "global timestamp order is non-causal across CUDA streams; equal starts "
            "use Parquet row order"
        ),
    )
    per_stream: dict[str, Any] = {}
    if "stream_id" in available_columns:
        by_stream: dict[Any, list[dict[str, Any]]] = defaultdict(list)
        for record in included:
            by_stream[record["stream_id"]].append(record)
        for stream_id in sorted(
            by_stream, key=lambda value: (value is None, str(value))
        ):
            key = "unknown" if stream_id is None else str(stream_id)
            per_stream[key] = _compress_sequence(
                by_stream[stream_id],
                windows[0][0],
                limit=sequence_limit,
                order_scope="per_stream_observed_timestamp",
                causal=True,
                caveat=(
                    "chronological order is causal within this CUDA stream; it does "
                    "not establish dependencies with other streams"
                ),
            )
    replay_report = _summarize_graph_replays(
        included, available_columns, expected_decode_replays
    )
    return {
        "path": str(path),
        "iteration_count": len(windows),
        "iteration_windows_ns": [[start, end] for start, end in windows],
        "record_counts": {
            "parquet_rows": table.num_rows,
            "included_kernels": len(included),
            "excluded_nonpositive_duration": excluded_nonpositive,
            "excluded_outside_windows": excluded_outside,
        },
        "memops": memops,
        "timing": {
            "window_sum_ns": window_sum,
            "window_union_ns": window_union,
            "gpu_sum_ns": gpu_sum,
            "gpu_union_ns": gpu_union,
            "memop_union_ns": memop_union,
            "activity_union_ns": activity_union,
            "no_recorded_activity_ns": no_recorded_activity,
            "overlap_factor": overlap_factor,
            "gpu_sum_share_of_window": gpu_sum / window_union if window_union else None,
            "gpu_union_share_of_window": gpu_union / window_union
            if window_union
            else None,
            "activity_union_share_of_window": activity_union / window_union
            if window_union
            else None,
            "no_recorded_activity_share_of_window": no_recorded_activity / window_union
            if window_union
            else None,
        },
        "timing_semantics": {
            "window_sum_ns": "sum of profiler iteration-window durations",
            "window_union_ns": (
                "wall time covered by iteration windows with overlaps merged"
            ),
            "gpu_sum_ns": (
                "sum of clipped kernel durations; concurrent streams can "
                "double-count wall time"
            ),
            "gpu_union_ns": (
                "clipped kernel interval union across streams; defensible "
                "rank-0 GPU busy wall time"
            ),
            "memop_union_ns": (
                "clipped rank-0 memop interval union; null when the optional memops "
                "Parquet is absent"
            ),
            "activity_union_ns": (
                "union of clipped rank-0 kernel and available memop intervals"
            ),
            "no_recorded_activity_ns": (
                "iteration-window union minus recorded activity union; this is not "
                "proof that the device was idle"
            ),
            "overlap_factor": (
                "gpu_sum_ns / gpu_union_ns; values above one reflect concurrent "
                "kernel intervals"
            ),
        },
        "categories": category_report,
        "top_kernels": top_kernels,
        "kernel_sequence": global_sequence,
        "per_stream_kernel_sequences": {
            "available": "stream_id" in available_columns,
            "order_scope": "per_stream_observed_timestamp",
            "causal_within_stream": True,
            "caveat": (
                "CUDA stream order is causal only within each stream; these summaries "
                "do not infer cross-stream dependencies"
            ),
            "streams": per_stream,
        },
        "decode_graph_replay": replay_report,
        "limitations": [
            (
                "Stack-qualified routed/shared MoE ownership is not emitted: "
                "graph-node and stack provenance are retained by ntrace, but "
                "model-specific clone-aware ownership rules are outside this "
                "concise analyzer."
            )
        ],
    }


def analyze_run(
    run_dir: Path,
    expected_phase: str,
    expected_scope: str,
    *,
    sequence_limit: int = 200,
) -> dict[str, Any]:
    if not run_dir.is_dir():
        raise ValidationError(f"run directory does not exist: {run_dir}")
    metadata = _parse_metadata(run_dir, expected_phase, expected_scope)
    benchmark = _read_benchmark(run_dir, metadata)
    trace_path = _find_one(run_dir, "ntrace_records_rank0.parquet", required=True)
    assert trace_path is not None
    expected_decode_replays = (
        metadata["osl"] - 1 if expected_phase == "decode" else None
    )
    return {
        "label": f"{expected_phase}_{expected_scope}",
        "run_dir": str(run_dir),
        "metadata": metadata,
        "benchmark": benchmark,
        "trace": summarize_trace(
            trace_path,
            sequence_limit=sequence_limit,
            expected_decode_replays=expected_decode_replays,
        ),
    }


def _require_equal(
    runs: Mapping[tuple[str, str], Mapping[str, Any]],
    arms: Sequence[tuple[str, str]],
    keys: Sequence[str],
    context: str,
) -> None:
    reference_arm = arms[0]
    reference = runs[reference_arm]["metadata"]
    errors: list[str] = []
    for key in keys:
        values = {
            f"{phase}_{scope}": runs[(phase, scope)]["metadata"][key]
            for phase, scope in arms
        }
        if any(value != reference[key] for value in values.values()):
            errors.append(f"{key}={values!r}")
    if errors:
        raise ValidationError(f"{context} metadata mismatch: {'; '.join(errors)}")


def _delta(bf16: int | float, mxfp8: int | float) -> dict[str, int | float | None]:
    delta = mxfp8 - bf16
    return {
        "bf16": bf16,
        "mxfp8": mxfp8,
        "delta": delta,
        "delta_pct": delta / bf16 * 100 if bf16 else None,
    }


def compute_deltas(bf16: Mapping[str, Any], mxfp8: Mapping[str, Any]) -> dict[str, Any]:
    bf16_trace = bf16["trace"]
    mxfp8_trace = mxfp8["trace"]
    timing = {
        field: _delta(bf16_trace["timing"][field], mxfp8_trace["timing"][field])
        for field in TIMING_DELTA_FIELDS
        if isinstance(bf16_trace["timing"].get(field), (int, float))
        and isinstance(mxfp8_trace["timing"].get(field), (int, float))
    }

    category_names = sorted(
        set(bf16_trace["categories"]) | set(mxfp8_trace["categories"])
    )
    categories: dict[str, Any] = {}
    for category in category_names:
        baseline = bf16_trace["categories"].get(category, {})
        variant = mxfp8_trace["categories"].get(category, {})
        categories[category] = {
            field: _delta(baseline.get(field, 0), variant.get(field, 0))
            for field in ("count", "total_ns", "share_of_gpu_sum")
        }

    bf16_result = bf16.get("benchmark", {}).get("result", {})
    mxfp8_result = mxfp8.get("benchmark", {}).get("result", {})
    benchmark = {
        field: _delta(bf16_result[field], mxfp8_result[field])
        for field in BENCHMARK_DELTA_FIELDS
        if isinstance(bf16_result.get(field), (int, float))
        and isinstance(mxfp8_result.get(field), (int, float))
    }
    return {
        "direction": "mxfp8_minus_bf16",
        "timing": timing,
        "categories": categories,
        "benchmark": benchmark,
    }


def analyze_experiment(
    paths: Mapping[tuple[str, str], Path], *, sequence_limit: int = 200
) -> dict[str, Any]:
    missing_arms = [arm for arm in ARM_ORDER if arm not in paths]
    if missing_arms:
        raise ValidationError(f"missing run arms: {missing_arms!r}")
    runs = {
        arm: analyze_run(
            paths[arm],
            expected_phase=arm[0],
            expected_scope=arm[1],
            sequence_limit=sequence_limit,
        )
        for arm in ARM_ORDER
    }

    _require_equal(runs, ARM_ORDER, SHARED_METADATA_KEYS, "cross-arm")
    for scope in SCOPES:
        scope_arms = [(phase, scope) for phase in PHASES]
        _require_equal(runs, scope_arms, SCOPE_METADATA_KEYS, f"{scope} cross-phase")
    for phase in PHASES:
        phase_arms = [(phase, scope) for scope in SCOPES]
        _require_equal(runs, phase_arms, ("isl", "osl"), f"{phase} cross-scope")

    labeled_runs = {
        f"{phase}_{scope}": runs[(phase, scope)] for phase, scope in ARM_ORDER
    }
    return {
        "validation": {
            "status": "ok",
            "metadata_contract": (
                "all required arm, fixed, phase, scope, and cross-arm checks passed"
            ),
        },
        "runs": labeled_runs,
        "bf16_to_mxfp8_deltas": {
            phase: compute_deltas(runs[(phase, "bf16")], runs[(phase, "mxfp8")])
            for phase in PHASES
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze matched BF16/MXFP8 rank-0 ntrace runs."
    )
    for phase, scope in ARM_ORDER:
        parser.add_argument(f"--{phase}-{scope}", type=Path, required=True)
    parser.add_argument("--sequence-limit", type=int, default=200)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    paths = {
        (phase, scope): getattr(args, f"{phase}_{scope}") for phase, scope in ARM_ORDER
    }
    try:
        report = analyze_experiment(paths, sequence_limit=args.sequence_limit)
    except ValidationError as error:
        print(f"validation error: {error}", file=sys.stderr)
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
