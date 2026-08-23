# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# pyright: reportMissingImports=false

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import analyze_runs
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

EXPERIMENT_DIR = Path(__file__).parent


def _metadata(phase: str, scope: str) -> dict[str, str]:
    isl, osl = {
        "prefill": ("10000", "1"),
        "decode": ("1000", "256"),
    }[phase]
    return {
        "phase": phase,
        "scope": scope,
        "precision": scope,
        "vllm_head": "abc123",
        "container": "/images/vllm.sqsh",
        "container_sha256": "f" * 64,
        "model": f"/models/{scope}",
        "ntrace_runtime": "/opt/ntrace",
        "ntrace_native": "/opt/ntrace/_cupti_cpp.so",
        "tp": "8",
        "dp": "1",
        "pp": "1",
        "expert_parallel": "1",
        "batch_size": "8",
        "concurrency": "8",
        "num_requests": "8",
        "isl": isl,
        "osl": osl,
        "cuda_graph": "FULL_AND_PIECEWISE",
        "linear_backend": "vllm_default" if scope == "bf16" else "auto",
        "moe_backend": "flashinfer_trtllm",
        "kv_cache_dtype": "auto",
        "bench_seed": "17",
        "warmup_seed": "117",
    }


def _write_metadata(run_dir: Path, values: dict[str, str]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metadata.env").write_text(
        "".join(f"{key}={value}\n" for key, value in values.items())
    )


def _write_trace(
    run_dir: Path,
    records: Sequence[Mapping[str, object]],
    windows: list[tuple[int, int]],
) -> None:
    trace_dir = run_dir / "ntrace"
    trace_dir.mkdir(parents=True, exist_ok=True)
    schema = pa.schema(
        [
            ("kernel_name_demangled", pa.string()),
            ("symbol_name", pa.string()),
            ("start_ns", pa.int64()),
            ("duration_ns", pa.int64()),
            ("source", pa.string()),
            ("stream_id", pa.int64()),
            ("graph_id", pa.int64()),
            ("graph_node_id", pa.int64()),
        ],
        metadata={
            b"ntrace.iter_start_ns": json.dumps([start for start, _ in windows]),
            b"ntrace.iter_end_ns": json.dumps([end for _, end in windows]),
        },
    )
    table = pa.Table.from_pylist(list(records), schema=schema)
    pq.write_table(table, trace_dir / "ntrace_records_rank0.parquet")


def _write_memops(
    run_dir: Path,
    records: Sequence[Mapping[str, object]],
) -> None:
    trace_dir = run_dir / "ntrace"
    trace_dir.mkdir(parents=True, exist_ok=True)
    schema = pa.schema(
        [
            ("op_type", pa.string()),
            ("start_ns", pa.int64()),
            ("end_ns", pa.int64()),
            ("duration_ns", pa.int64()),
            ("stream_id", pa.int64()),
        ]
    )
    table = pa.Table.from_pylist(list(records), schema=schema)
    pq.write_table(table, trace_dir / "ntrace_memops_rank0.parquet")


def _default_records(phase: str = "prefill") -> list[dict[str, object]]:
    if phase == "decode":
        return [
            {
                "kernel_name_demangled": "decode_graph_kernel",
                "symbol_name": None,
                "start_ns": 100 + replay * 10,
                "duration_ns": 5,
                "source": "replay",
                "stream_id": 7,
                "graph_id": 42,
                "graph_node_id": 3,
            }
            for replay in range(255)
        ]
    return [
        {
            "kernel_name_demangled": "bmm_mxfp8_expert",
            "symbol_name": None,
            "start_ns": 100,
            "duration_ns": 10,
            "source": "non_graph",
            "stream_id": 7,
            "graph_id": None,
            "graph_node_id": None,
        }
    ]


def _write_benchmark(
    run_dir: Path,
    metadata: dict[str, str],
    *,
    include_optional_fields: bool = True,
) -> Path:
    config: dict[str, object] = {"driver": "vllm_bench_serve_static"}
    result: dict[str, object] = {}
    if include_optional_fields:
        config.update(
            {
                "isl": int(metadata["isl"]),
                "osl": int(metadata["osl"]),
                "batch_sizes": [int(metadata["batch_size"])],
            }
        )
        expected = int(metadata["num_requests"]) * int(metadata["osl"])
        result.update(
            {
                "bs": int(metadata["batch_size"]),
                "isl": int(metadata["isl"]),
                "osl": int(metadata["osl"]),
                "actual_output_tokens": expected,
                "expected_output_tokens": expected,
                "tokens_ok": True,
                "latency_med_s": 1.25,
                "output_tok_s": 100.0,
            }
        )
    path = run_dir / "results" / "result_bench_short.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"config": config, "results": [result]}))
    return path


def _write_run(
    root: Path,
    phase: str,
    scope: str,
    *,
    records: list[dict[str, object]] | None = None,
    benchmark: bool = True,
) -> Path:
    run_dir = root / f"{phase}_{scope}"
    metadata = _metadata(phase, scope)
    _write_metadata(run_dir, metadata)
    _write_trace(
        run_dir,
        records if records is not None else _default_records(phase),
        [(100, 3_000)] if phase == "decode" else [(100, 200)],
    )
    if benchmark:
        _write_benchmark(run_dir, metadata)
    return run_dir


def _hierarchy_node(
    node_id: int,
    *,
    name: str,
    start_ns: int,
    direct_module: str | None = None,
    direct_operation: str | None = None,
    is_norm: bool = False,
    is_dense: bool = False,
    is_communication: bool = False,
) -> dict[str, object]:
    return {
        "node_id": node_id,
        "name": name,
        "start_ns": start_ns,
        "stream_id": 7,
        "mean_duration_ns": 10,
        "direct_module": direct_module,
        "direct_operation": direct_operation,
        "is_norm": is_norm,
        "is_dense": is_dense,
        "is_communication": is_communication,
        "mixer_id": node_id if direct_module in {"mamba", "attention"} else None,
    }


def test_hierarchy_classifier_splits_routed_w13_and_w2() -> None:
    frames = [
        {
            "filename": "/opt/vllm/fused_moe/routed_experts.py",
            "funcname": "forward_monolithic",
        }
    ]

    assert analyze_runs.classify_hierarchy_direct(
        "bmm_Bfloat16_Bfloat16Bfloat16_Fp32_relu2_sm100f", frames
    ) == ("moe", "routed W13 + activation")
    assert analyze_runs.classify_hierarchy_direct(
        "bmm_Bfloat16_Bfloat16Bfloat16_Fp32_sm100f", frames
    ) == ("moe", "routed W2")
    assert analyze_runs.classify_hierarchy_direct(
        "bmm_MxE4m3_MxE4m3MxE4m3_Fp32_relu2_sm100f", frames
    ) == ("moe", "routed W13 + activation")
    assert analyze_runs.classify_hierarchy_direct(
        "bmm_Bfloat16_MxE4m3MxE4m3_Fp32_sm100f", frames
    ) == ("moe", "routed W2")


def test_hierarchy_segments_modules_and_leaves_final_norm_outside_moe() -> None:
    nodes = [
        _hierarchy_node(0, name="embedding", start_ns=0),
        _hierarchy_node(1, name="rms_norm", start_ns=10, is_norm=True),
        _hierarchy_node(2, name="mamba_in", start_ns=20, is_dense=True),
        _hierarchy_node(
            3,
            name="causal_conv",
            start_ns=30,
            direct_module="mamba",
            direct_operation="causal conv",
        ),
        _hierarchy_node(4, name="mamba_out", start_ns=40, is_dense=True),
        _hierarchy_node(5, name="all_reduce", start_ns=50, is_communication=True),
        _hierarchy_node(6, name="rms_norm", start_ns=60, is_norm=True),
        _hierarchy_node(7, name="qkv", start_ns=70, is_dense=True),
        _hierarchy_node(
            8,
            name="fmha",
            start_ns=80,
            direct_module="attention",
            direct_operation="attention core",
        ),
        _hierarchy_node(9, name="o_proj", start_ns=90, is_dense=True),
        _hierarchy_node(10, name="all_reduce", start_ns=100, is_communication=True),
        _hierarchy_node(11, name="rms_norm", start_ns=110, is_norm=True),
        _hierarchy_node(12, name="router", start_ns=120, is_dense=True),
        _hierarchy_node(
            13,
            name="routing",
            start_ns=130,
            direct_module="moe",
            direct_operation="routing/top-k",
        ),
        _hierarchy_node(
            19,
            name="routing_cooperative",
            start_ns=135,
            direct_module="moe",
            direct_operation="routing/top-k",
        ),
        _hierarchy_node(
            14,
            name="w13",
            start_ns=140,
            direct_module="moe",
            direct_operation="routed W13 + activation",
        ),
        _hierarchy_node(
            15,
            name="w2",
            start_ns=150,
            direct_module="moe",
            direct_operation="routed W2",
        ),
        _hierarchy_node(
            16,
            name="finalize",
            start_ns=160,
            direct_module="moe",
            direct_operation="finalize/scatter",
        ),
        _hierarchy_node(17, name="all_reduce", start_ns=170, is_communication=True),
        _hierarchy_node(18, name="rms_norm", start_ns=180, is_norm=True),
    ]

    assignments, component_counts = analyze_runs.attribute_main_stream_hierarchy(nodes)

    assert component_counts == {"mamba": 1, "attention": 1, "moe": 1}
    assert assignments[2] == ("mamba", "input projection")
    assert assignments[4] == ("mamba", "output projection")
    assert assignments[5] == ("mamba", "TP all-reduce")
    assert assignments[7] == ("attention", "QKV projection")
    assert assignments[9] == ("attention", "O projection")
    assert assignments[12] == ("moe", "router projection")
    assert assignments[17] == ("moe", "TP all-reduce")
    assert 0 not in assignments
    assert 18 not in assignments


def test_summarize_trace_clips_filters_orders_and_aggregates(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    records = [
        {
            "kernel_name_demangled": "bmm_mxfp8_expert",
            "symbol_name": None,
            "start_ns": 90,
            "duration_ns": 20,
        },
        {
            "kernel_name_demangled": "quantize_with_block_size_mxfp8",
            "symbol_name": None,
            "start_ns": 120,
            "duration_ns": 20,
        },
        {
            "kernel_name_demangled": "fmhaSm100fKernel",
            "symbol_name": None,
            "start_ns": 130,
            "duration_ns": 30,
        },
        {
            "kernel_name_demangled": "outside",
            "symbol_name": None,
            "start_ns": 250,
            "duration_ns": 20,
        },
        {
            "kernel_name_demangled": None,
            "symbol_name": "direct_copy_kernel",
            "start_ns": 390,
            "duration_ns": 20,
        },
        {
            "kernel_name_demangled": "zero_duration",
            "symbol_name": None,
            "start_ns": 150,
            "duration_ns": 0,
        },
    ]
    _write_trace(run_dir, records, [(100, 200), (300, 400)])

    summary = analyze_runs.summarize_trace(
        run_dir / "ntrace" / "ntrace_records_rank0.parquet"
    )

    assert summary["iteration_count"] == 2
    assert summary["timing"] == {
        "window_sum_ns": 200,
        "window_union_ns": 200,
        "gpu_sum_ns": 70,
        "gpu_union_ns": 60,
        "memop_union_ns": None,
        "activity_union_ns": 60,
        "no_recorded_activity_ns": 140,
        "overlap_factor": pytest.approx(7 / 6),
        "gpu_sum_share_of_window": 0.35,
        "gpu_union_share_of_window": 0.3,
        "activity_union_share_of_window": 0.3,
        "no_recorded_activity_share_of_window": 0.7,
    }
    assert summary["record_counts"] == {
        "parquet_rows": 6,
        "included_kernels": 4,
        "excluded_nonpositive_duration": 1,
        "excluded_outside_windows": 1,
    }
    sequence = summary["kernel_sequence"]
    assert sequence["total_segments"] == 4
    assert sequence["truncated"] is False
    assert [entry["category"] for entry in sequence["segments"]] == [
        "moe_gemm",
        "mxfp8_quantize",
        "attention",
        "copy",
    ]
    assert [entry["start_offset_ns"] for entry in sequence["segments"]] == [
        0,
        20,
        30,
        290,
    ]
    assert summary["categories"]["attention"] == {
        "count": 1,
        "total_ns": 30,
        "share_of_gpu_sum": pytest.approx(3 / 7),
    }
    assert summary["top_kernels"][0]["name"] == "fmhaSm100fKernel"
    assert summary["memops"]["available"] is False
    assert summary["kernel_sequence"]["causal"] is False
    assert "non-causal" in summary["kernel_sequence"]["caveat"]


def test_summarize_trace_clips_memops_and_computes_activity_union(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    _write_trace(
        run_dir,
        [
            {
                "kernel_name_demangled": "kernel_a",
                "symbol_name": None,
                "start_ns": 100,
                "duration_ns": 30,
                "stream_id": 1,
            },
            {
                "kernel_name_demangled": "kernel_b",
                "symbol_name": None,
                "start_ns": 120,
                "duration_ns": 30,
                "stream_id": 2,
            },
        ],
        [(100, 200), (300, 400)],
    )
    _write_memops(
        run_dir,
        [
            {
                "op_type": "HtoD",
                "start_ns": 125,
                "end_ns": 160,
                "duration_ns": 35,
                "stream_id": 3,
            },
            {
                "op_type": "DtoH",
                "start_ns": 180,
                "end_ns": 220,
                "duration_ns": 40,
                "stream_id": 3,
            },
            {
                "op_type": "DtoD",
                "start_ns": 250,
                "end_ns": 270,
                "duration_ns": 20,
                "stream_id": 3,
            },
            {
                "op_type": "DtoD",
                "start_ns": 350,
                "end_ns": 350,
                "duration_ns": 0,
                "stream_id": 3,
            },
        ],
    )

    summary = analyze_runs.summarize_trace(
        run_dir / "ntrace" / "ntrace_records_rank0.parquet"
    )

    assert summary["memops"] == {
        "available": True,
        "path": str(run_dir / "ntrace" / "ntrace_memops_rank0.parquet"),
        "record_counts": {
            "parquet_rows": 4,
            "included_memops": 2,
            "excluded_nonpositive_duration": 1,
            "excluded_outside_windows": 1,
        },
    }
    assert summary["timing"]["gpu_sum_ns"] == 60
    assert summary["timing"]["gpu_union_ns"] == 50
    assert summary["timing"]["memop_union_ns"] == 55
    assert summary["timing"]["activity_union_ns"] == 80
    assert summary["timing"]["no_recorded_activity_ns"] == 120
    assert summary["timing"]["overlap_factor"] == pytest.approx(1.2)


def test_kernel_sequence_compresses_adjacent_identical_kernels(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    records = [
        {
            "kernel_name_demangled": "quantize_with_block_size_mxfp8",
            "symbol_name": None,
            "start_ns": 110,
            "duration_ns": 5,
        },
        {
            "kernel_name_demangled": "quantize_with_block_size_mxfp8",
            "symbol_name": None,
            "start_ns": 120,
            "duration_ns": 7,
        },
        {
            "kernel_name_demangled": "fmhaSm100fKernel",
            "symbol_name": None,
            "start_ns": 130,
            "duration_ns": 11,
        },
    ]
    _write_trace(run_dir, records, [(100, 200)])

    sequence_report = analyze_runs.summarize_trace(
        run_dir / "ntrace" / "ntrace_records_rank0.parquet"
    )["kernel_sequence"]

    assert sequence_report["total_segments"] == 2
    assert sequence_report["truncated"] is False
    assert sequence_report["segments"] == [
        {
            "name": "quantize_with_block_size_mxfp8",
            "category": "mxfp8_quantize",
            "consecutive_count": 2,
            "total_ns": 12,
            "start_offset_ns": 10,
            "end_offset_ns": 27,
        },
        {
            "name": "fmhaSm100fKernel",
            "category": "attention",
            "consecutive_count": 1,
            "total_ns": 11,
            "start_offset_ns": 30,
            "end_offset_ns": 41,
        },
    ]


def test_kernel_sequence_reports_total_while_bounding_emitted_prefix(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    records = [
        {
            "kernel_name_demangled": name,
            "symbol_name": None,
            "start_ns": start,
            "duration_ns": 5,
        }
        for name, start in (("kernel_a", 110), ("kernel_b", 120), ("kernel_c", 130))
    ]
    _write_trace(run_dir, records, [(100, 200)])

    sequence = analyze_runs.summarize_trace(
        run_dir / "ntrace" / "ntrace_records_rank0.parquet", sequence_limit=2
    )["kernel_sequence"]

    assert sequence["total_segments"] == 3
    assert sequence["emitted_segments"] == 2
    assert sequence["truncated"] is True
    assert [segment["name"] for segment in sequence["segments"]] == [
        "kernel_a",
        "kernel_b",
    ]


def test_summarize_trace_preserves_per_stream_chronology(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    records = [
        {
            "kernel_name_demangled": "stream_1_first",
            "symbol_name": None,
            "start_ns": 100,
            "duration_ns": 5,
            "stream_id": 1,
        },
        {
            "kernel_name_demangled": "stream_2_first",
            "symbol_name": None,
            "start_ns": 110,
            "duration_ns": 5,
            "stream_id": 2,
        },
        {
            "kernel_name_demangled": "stream_2_second",
            "symbol_name": None,
            "start_ns": 120,
            "duration_ns": 5,
            "stream_id": 2,
        },
        {
            "kernel_name_demangled": "stream_1_second",
            "symbol_name": None,
            "start_ns": 130,
            "duration_ns": 5,
            "stream_id": 1,
        },
    ]
    _write_trace(run_dir, records, [(100, 200)])

    summary = analyze_runs.summarize_trace(
        run_dir / "ntrace" / "ntrace_records_rank0.parquet"
    )

    observed = summary["kernel_sequence"]
    assert observed["order_scope"] == "global_observed_timestamp"
    assert observed["causal"] is False
    assert [segment["name"] for segment in observed["segments"]] == [
        "stream_1_first",
        "stream_2_first",
        "stream_2_second",
        "stream_1_second",
    ]
    per_stream = summary["per_stream_kernel_sequences"]
    assert per_stream["order_scope"] == "per_stream_observed_timestamp"
    assert per_stream["causal_within_stream"] is True
    assert [segment["name"] for segment in per_stream["streams"]["1"]["segments"]] == [
        "stream_1_first",
        "stream_1_second",
    ]
    assert [segment["name"] for segment in per_stream["streams"]["2"]["segments"]] == [
        "stream_2_first",
        "stream_2_second",
    ]


def test_summarize_trace_validates_decode_graph_replays(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    records = []
    for replay, (anchor_start, second_start) in enumerate(
        ((100, 110), (200, 215), (330, 350))
    ):
        records.extend(
            [
                {
                    "kernel_name_demangled": "anchor_kernel",
                    "symbol_name": None,
                    "start_ns": anchor_start,
                    "duration_ns": 5,
                    "source": "replay",
                    "stream_id": 1,
                    "graph_id": 7,
                    "graph_node_id": 1,
                },
                {
                    "kernel_name_demangled": "second_kernel",
                    "symbol_name": None,
                    "start_ns": second_start,
                    "duration_ns": 5,
                    "source": "replay",
                    "stream_id": 2,
                    "graph_id": 7,
                    "graph_node_id": 2,
                },
            ]
        )
    _write_trace(run_dir, records, [(100, 400)])

    replay = analyze_runs.summarize_trace(
        run_dir / "ntrace" / "ntrace_records_rank0.parquet",
        expected_decode_replays=3,
    )["decode_graph_replay"]

    assert replay["status"] == "validated"
    assert replay["expected_replays"] == 3
    assert replay["observed_replays"] == 3
    assert replay["dominant_graph_id"] == 7
    assert replay["dominant_graph_nodes_per_replay"] == 2
    assert replay["non_dominant_graph_nodes"] == 0
    assert replay["anchor_graph_node_id"] == 1
    assert replay["replay_span_ns"] == {
        "count": 3,
        "p50": 20,
        "p95": 25,
        "p99": 25,
    }
    assert replay["anchor_period_ns"] == {
        "count": 2,
        "p50": 100,
        "p95": 130,
        "p99": 130,
    }
    assert replay["graph_kernel_sum_ns"] == 30
    assert replay["graph_kernel_sum_per_replay_ns"] == 10
    assert replay["categories"] == {
        "other": {
            "count": 6,
            "total_ns": 30,
            "mean_per_replay_ns": 10,
            "share_of_graph_kernel_sum": 1.0,
        }
    }
    assert replay["top_kernels"] == [
        {
            "name": "anchor_kernel",
            "count": 3,
            "total_ns": 15,
            "mean_per_replay_ns": 5,
        },
        {
            "name": "second_kernel",
            "count": 3,
            "total_ns": 15,
            "mean_per_replay_ns": 5,
        },
    ]


def test_summarize_trace_rejects_decode_graph_replay_count_mismatch(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    records = [
        {
            "kernel_name_demangled": "decode_kernel",
            "symbol_name": None,
            "start_ns": 100 + replay * 10,
            "duration_ns": 5,
            "source": "replay",
            "stream_id": 1,
            "graph_id": 7,
            "graph_node_id": 1,
        }
        for replay in range(2)
    ]
    _write_trace(run_dir, records, [(100, 200)])

    with pytest.raises(analyze_runs.ValidationError, match="expected 3.*observed 2"):
        analyze_runs.summarize_trace(
            run_dir / "ntrace" / "ntrace_records_rank0.parquet",
            expected_decode_replays=3,
        )


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("bmm_foo", "moe_gemm"),
        ("bmm_Bfloat16_Bfloat16Bfloat16_Fp32", "moe_gemm"),
        ("trtllm_fp8_block_scale_moe", "moe_gemm"),
        ("nvjet_sm100_tst_64x8_64x16", "dense_bf16_gemm"),
        ("void cublasLt::splitKreduce_kernel<32, 16>", "dense_bf16_gemm"),
        ("sm100BlockScaledGemm", "dense_mxfp8_gemm"),
        ("gemm_kernel_mxe4m3", "dense_mxfp8_gemm"),
        ("quantize_with_block_size", "mxfp8_quantize"),
        ("ncclDevKernel_AllReduce", "communication"),
        ("multimem_all_reduce_kernel<c10::BFloat16>", "communication"),
        (
            "moe::dev::routing::routingCustom::KernelParams<ScaledSumNormalizePostprocess>",
            "moe_routing",
        ),
        ("moe::dev::finalize::finalizeKernel", "moe_finalize"),
        ("fmhaSm100fKernel", "attention"),
        ("selective_state_update", "mamba"),
        ("_selective_scan_update_kernel", "mamba"),
        ("_causal_conv1d_update_kernel", "mamba"),
        ("triton_red_fused_fused_add_rms_norm_1", "normalization"),
        ("layer_norm_kernel", "normalization"),
        ("direct_copy_kernel", "copy"),
        ("elementwise_kernel", "other"),
    ],
)
def test_classify_kernel(name: str, expected: str) -> None:
    assert analyze_runs.classify_kernel(name) == expected


def test_summarize_trace_emits_refined_profiler_categories(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    names = (
        "bmm_Bfloat16_Bfloat16Bfloat16_Fp32",
        "nvjet_sm100_tst_64x8_64x16",
        "void cublasLt::splitKreduce_kernel<32, 16>",
        "multimem_all_reduce_kernel<c10::BFloat16>",
        "moe::dev::routing::routingCustom::KernelParams<ScaledSumNormalizePostprocess>",
        "moe::dev::finalize::finalizeKernel",
        "_selective_scan_update_kernel",
        "triton_red_fused_fused_add_rms_norm_1",
        "unclassified_elementwise_kernel",
    )
    records = [
        {
            "kernel_name_demangled": name,
            "symbol_name": None,
            "start_ns": 100 + index * 10,
            "duration_ns": 5,
        }
        for index, name in enumerate(names)
    ]
    _write_trace(run_dir, records, [(100, 200)])

    categories = analyze_runs.summarize_trace(
        run_dir / "ntrace" / "ntrace_records_rank0.parquet"
    )["categories"]

    assert {category: values["count"] for category, values in categories.items()} == {
        "communication": 1,
        "dense_bf16_gemm": 2,
        "mamba": 1,
        "moe_finalize": 1,
        "moe_gemm": 1,
        "moe_routing": 1,
        "normalization": 1,
        "other": 1,
    }


def test_analyze_run_reads_metadata_benchmark_and_trace(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path, "prefill", "bf16")

    report = analyze_runs.analyze_run(run_dir, "prefill", "bf16")

    assert report["label"] == "prefill_bf16"
    assert report["metadata"]["isl"] == 10000
    assert report["benchmark"]["available"] is True
    assert report["benchmark"]["result"]["actual_output_tokens"] == 8
    assert report["benchmark"]["unavailable_fields"] == []
    assert report["trace"]["categories"]["moe_gemm"]["count"] == 1


def test_analyze_run_allows_unavailable_optional_benchmark_fields(
    tmp_path: Path,
) -> None:
    run_dir = _write_run(tmp_path, "decode", "bf16", benchmark=False)

    report = analyze_runs.analyze_run(run_dir, "decode", "bf16")

    assert report["benchmark"]["available"] is False
    assert "config.isl" in report["benchmark"]["unavailable_fields"]
    assert report["trace"]["record_counts"]["included_kernels"] == 255
    assert report["trace"]["decode_graph_replay"]["status"] == "validated"
    assert report["trace"]["decode_graph_replay"]["expected_replays"] == 255


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda values: values.pop("vllm_head"), "missing required metadata"),
        (lambda values: values.__setitem__("phase", "decode"), "expected phase"),
    ],
)
def test_analyze_run_rejects_missing_or_mislabeled_metadata(
    tmp_path: Path,
    mutation: Callable[[dict[str, str]], object],
    message: str,
) -> None:
    run_dir = tmp_path / "prefill_bf16"
    metadata = _metadata("prefill", "bf16")
    mutation(metadata)
    _write_metadata(run_dir, metadata)
    _write_trace(run_dir, _default_records(), [(100, 200)])

    with pytest.raises(analyze_runs.ValidationError, match=message):
        analyze_runs.analyze_run(run_dir, "prefill", "bf16")


def test_analyze_run_rejects_benchmark_values_that_contradict_metadata(
    tmp_path: Path,
) -> None:
    run_dir = _write_run(tmp_path, "decode", "bf16", benchmark=False)
    metadata = _metadata("decode", "bf16")
    benchmark_path = _write_benchmark(run_dir, metadata)
    benchmark = json.loads(benchmark_path.read_text())
    benchmark["config"]["osl"] = 1000
    benchmark_path.write_text(json.dumps(benchmark))

    with pytest.raises(analyze_runs.ValidationError, match="config.osl=1000.*osl=256"):
        analyze_runs.analyze_run(run_dir, "decode", "bf16")


def test_compute_deltas_uses_bf16_as_baseline() -> None:
    bf16 = {
        "trace": {
            "timing": {"window_union_ns": 100, "gpu_sum_ns": 80, "gpu_union_ns": 70},
            "categories": {
                "moe_gemm": {"count": 2, "total_ns": 60},
                "other": {"count": 1, "total_ns": 20},
            },
        }
    }
    mxfp8 = {
        "trace": {
            "timing": {"window_union_ns": 90, "gpu_sum_ns": 60, "gpu_union_ns": 55},
            "categories": {
                "moe_gemm": {"count": 3, "total_ns": 30},
                "mxfp8_quantize": {"count": 1, "total_ns": 10},
                "other": {"count": 1, "total_ns": 20},
            },
        }
    }

    deltas = analyze_runs.compute_deltas(bf16, mxfp8)

    assert deltas["timing"]["gpu_sum_ns"] == {
        "bf16": 80,
        "mxfp8": 60,
        "delta": -20,
        "delta_pct": -25.0,
    }
    assert deltas["categories"]["moe_gemm"]["total_ns"]["delta_pct"] == -50.0
    assert deltas["categories"]["mxfp8_quantize"]["total_ns"] == {
        "bf16": 0,
        "mxfp8": 10,
        "delta": 10,
        "delta_pct": None,
    }


def test_analyze_experiment_rejects_cross_arm_contract_mismatch(tmp_path: Path) -> None:
    paths = {
        (phase, scope): _write_run(tmp_path, phase, scope)
        for phase in ("prefill", "decode")
        for scope in ("bf16", "mxfp8")
    }
    mismatched = _metadata("decode", "mxfp8")
    mismatched["container_sha256"] = "0" * 64
    _write_metadata(paths[("decode", "mxfp8")], mismatched)

    with pytest.raises(analyze_runs.ValidationError, match="container_sha256"):
        analyze_runs.analyze_experiment(paths)


def test_cli_writes_four_arm_report_and_phase_deltas(tmp_path: Path) -> None:
    paths = {
        (phase, scope): _write_run(tmp_path, phase, scope)
        for phase in ("prefill", "decode")
        for scope in ("bf16", "mxfp8")
    }
    output = tmp_path / "analysis.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(EXPERIMENT_DIR / "analyze_runs.py"),
            "--prefill-bf16",
            str(paths[("prefill", "bf16")]),
            "--prefill-mxfp8",
            str(paths[("prefill", "mxfp8")]),
            "--decode-bf16",
            str(paths[("decode", "bf16")]),
            "--decode-mxfp8",
            str(paths[("decode", "mxfp8")]),
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads(output.read_text())
    assert list(report["runs"]) == [
        "prefill_bf16",
        "prefill_mxfp8",
        "decode_bf16",
        "decode_mxfp8",
    ]
    assert set(report["bf16_to_mxfp8_deltas"]) == {"prefill", "decode"}
