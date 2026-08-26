# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import csv
import hashlib
import json
import sys
from pathlib import Path

import pytest

from experiments.ultra_mxfp8_all_observed_tactics import summarize_results
from experiments.ultra_mxfp8_all_observed_tactics.summarize_results import (
    _summarize_e2e,
    _summarize_lookup,
    _summarize_oracle,
    summarize_backend,
)

ORACLE_TIMING = {
    "cuda_graph": True,
    "cold_l2_cache": True,
    "rounds": 3,
    "dry_run_iters": 5,
    "repeat_iters": 20,
    "calls_per_graph": 4,
}

EXPECTED_BACKENDS = (
    "cute-dsl",
    "cutlass",
    "trtllm-128x4",
    "trtllm-8x4",
)
UNSUPPORTED_PROVENANCE_FIELDS = (
    "source_commit",
    "flashinfer_commit",
    "expected_vllm_version",
    "flashinfer",
    "flashinfer_file",
    "vllm_version",
    "vllm_file",
    "vllm_compiled_file",
    "container",
    "container_sha256",
    "container_size",
    "container_mtime",
    "model",
    "model_config_sha256",
    "model_index_sha256",
    "model_weights_manifest_sha256",
    "gpu_name",
    "driver_version",
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


def _write_result(
    path: Path,
    *,
    output_tps: float,
    duration: float,
    generated_texts: list[str] | None = None,
) -> None:
    path.write_text(
        json.dumps(
            {
                "completed": 10,
                "failed": 0,
                "total_input_tokens": 10_000,
                "total_output_tokens": 10_000,
                "input_lens": [1_000] * 10,
                "output_lens": [1_000] * 10,
                "generated_texts": generated_texts or ["same"] * 10,
                "output_throughput": output_tps,
                "total_token_throughput": output_tps * 2,
                "mean_ttft_ms": 100.0,
                "mean_tpot_ms": 10.0,
                "duration": duration,
            }
        )
    )


def _write_metadata(path: Path, run_kind: str) -> None:
    path.write_text(
        "\n".join(
            [
                f"run_kind={run_kind}",
                "backend_name=cute-dsl",
                "oracle_backend=cute-dsl",
                "scale_layout=128x4",
                "source_commit=abc123",
                "flashinfer_commit=def456",
                "expected_vllm_version=0.27.1",
                "flashinfer=0.6.18",
                "flashinfer_file=/flashinfer/__init__.py",
                "vllm_version=0.27.1",
                "vllm_file=/vllm/__init__.py",
                "vllm_compiled_file=/site-packages/vllm/_C.abi3.so",
                "gpu_name=NVIDIA GB200",
                "driver_version=590.00",
                "container=/container.sqsh",
                "container_sha256=container-sha",
                "container_size=1234",
                "container_mtime=5678",
                "model=/model",
                "model_config_sha256=model-config-sha",
                "model_index_sha256=model-index-sha",
                "model_weights_manifest_sha256=model-weights-manifest-sha",
                "tp=1",
                "linear_backend=flashinfer_cutedsl",
                "trtllm_layout=8x4",
                "moe_backend=flashinfer_trtllm",
                "attention_backend=FLASHINFER",
                "kv_cache_dtype=auto",
                "max_model_len=12024",
                "max_num_batched_tokens=16384",
                "max_num_seqs=32",
                "gpu_memory_utilization=0.95",
                "enable_chunked_prefill=true",
                "enable_prefix_caching=true",
                "mamba_cache_mode=all",
                "mamba_ssm_cache_dtype=float32",
                "enforce_eager=0",
                "cudagraph_capture_sizes=1,2,4,8,16,32",
                "cudagraph_configured=true",
                "cudagraph_capture_status=capture_completed",
                "cudagraph_capture_evidence=server_log_completion_marker",
                "cudagraph_capture_marker=Graph capturing finished",
                "workloads=1000:1000",
                "concurrencies=1",
                "prompt_multiplier=10",
            ]
        )
        + "\n"
    )


def _comparison_metadata(tmp_path: Path) -> dict[str, str]:
    path = tmp_path / "comparison_metadata.txt"
    _write_metadata(path, "baseline")
    return dict(line.split("=", 1) for line in path.read_text().splitlines())


def _comparison_summary(
    backend: str,
    metadata: dict[str, str],
) -> dict:
    return {
        "backend": backend,
        "e2e": {
            "metadata": {**metadata, "backend_name": backend},
            "workloads": [
                {
                    "isl": 1000,
                    "osl": 1000,
                    "concurrency": 1,
                    "artifact_signature": "same",
                    "output_throughput_speedup": 1.1,
                }
            ],
        },
        "oracle": {"shape_count": 1},
        "lookup": {"entry_count": 1},
    }


def _write_unsupported_status(
    tmp_path: Path,
    metadata: dict[str, str],
    *,
    backend: str = "cutlass",
    extra: dict | None = None,
) -> Path:
    evidence_path = tmp_path / f"{backend}.stderr"
    evidence_path.write_text("backend compilation failed\n")
    provenance = {key: metadata[key] for key in UNSUPPORTED_PROVENANCE_FIELDS}
    provenance.update(
        {
            "enforce_eager": "1",
            "workloads": "1000:64,10000:64",
            "concurrencies": "1,2,4,8,16,32",
        }
    )
    payload = {
        "backend": backend,
        "status": "empirically_unsupported",
        "recipe": {
            "backend_name": backend,
            "linear_backend": "flashinfer_cutlass",
            "oracle_backend": "cutlass",
            "scale_layout": "128x4",
            "trtllm_layout": "8x4",
        },
        "provenance": provenance,
        "failure": {
            "stage": "server_startup",
            "reason_code": "backend_initialization_failed",
            "message": "CUTLASS is unavailable for this model recipe",
            "attempts": [
                {"job_id": "100", "mode": "eager", "outcome": "failed"},
                {"job_id": "101", "mode": "cuda_graph", "outcome": "failed"},
            ],
            "evidence": [
                {
                    "path": evidence_path.name,
                    "sha256": hashlib.sha256(evidence_path.read_bytes()).hexdigest(),
                }
            ],
        },
        **(extra or {}),
    }
    status_path = tmp_path / f"{backend}.status.json"
    status_path.write_text(json.dumps(payload))
    return status_path


def _write_paired_run(
    pair_root: Path,
    *,
    pair_order: str,
    run_kind: str,
    position: int,
    output_tps: float,
    host: str,
    job_id: str = "42",
) -> None:
    lookup_path = pair_root.parents[1] / "oracle" / "lookup.json"
    lookup_path.parent.mkdir(exist_ok=True)
    if not lookup_path.exists():
        lookup_path.write_text("{}\n")
    lookup_sha256 = hashlib.sha256(lookup_path.read_bytes()).hexdigest()
    run_dir = pair_root / "serving" / run_kind / f"{pair_order}-{run_kind}"
    run_dir.mkdir(parents=True)
    (run_dir / "COMPLETE").touch()
    _write_metadata(run_dir / "metadata.txt", run_kind)
    with (run_dir / "metadata.txt").open("a") as handle:
        handle.write(
            f"compute_host={host}\n"
            f"job_id={job_id}\n"
            f"pair_id={job_id}-{pair_order}\n"
            f"pair_order={pair_order}\n"
            f"pair_position={position}\n"
            f"lookup_path={lookup_path if run_kind == 'lookup' else 'not_applicable'}\n"
            "lookup_sha256="
            f"{lookup_sha256 if run_kind == 'lookup' else 'not_applicable'}\n"
        )
    _write_result(
        run_dir / "result_isl1000_osl1000_c1.json",
        output_tps=output_tps,
        duration=20_000.0 / output_tps,
    )


def test_summarize_e2e_aggregates_order_balanced_node_matched_pairs(
    tmp_path: Path,
) -> None:
    baseline_lookup = tmp_path / "pairs" / "baseline-lookup"
    _write_paired_run(
        baseline_lookup,
        pair_order="baseline-lookup",
        run_kind="baseline",
        position=1,
        output_tps=100.0,
        host="node-a",
    )
    _write_paired_run(
        baseline_lookup,
        pair_order="baseline-lookup",
        run_kind="lookup",
        position=2,
        output_tps=110.0,
        host="node-a",
    )
    lookup_baseline = tmp_path / "pairs" / "lookup-baseline"
    _write_paired_run(
        lookup_baseline,
        pair_order="lookup-baseline",
        run_kind="lookup",
        position=1,
        output_tps=120.0,
        host="node-b",
    )
    _write_paired_run(
        lookup_baseline,
        pair_order="lookup-baseline",
        run_kind="baseline",
        position=2,
        output_tps=100.0,
        host="node-b",
    )

    summary = _summarize_e2e(tmp_path)

    workload = summary["workloads"][0]
    assert summary["pair_count"] == 2
    assert summary["pair_orders"] == ["baseline-lookup", "lookup-baseline"]
    assert workload["output_throughput_speedup"] == pytest.approx((1.1 * 1.2) ** 0.5)
    assert workload["output_throughput_speedup_min"] == pytest.approx(1.1)
    assert workload["output_throughput_speedup_max"] == pytest.approx(1.2)
    assert [row["pair_order"] for row in workload["paired_measurements"]] == [
        "baseline-lookup",
        "lookup-baseline",
    ]

    (tmp_path / "oracle" / "lookup.json").write_text('{"changed": true}\n')
    with pytest.raises(ValueError, match="checksum"):
        _summarize_e2e(tmp_path)


def test_summarize_e2e_rejects_cross_node_pair(tmp_path: Path) -> None:
    pair_root = tmp_path / "pairs" / "baseline-lookup"
    _write_paired_run(
        pair_root,
        pair_order="baseline-lookup",
        run_kind="baseline",
        position=1,
        output_tps=100.0,
        host="node-a",
    )
    _write_paired_run(
        pair_root,
        pair_order="baseline-lookup",
        run_kind="lookup",
        position=2,
        output_tps=110.0,
        host="node-b",
    )
    reverse_root = tmp_path / "pairs" / "lookup-baseline"
    _write_paired_run(
        reverse_root,
        pair_order="lookup-baseline",
        run_kind="lookup",
        position=1,
        output_tps=110.0,
        host="node-c",
    )
    _write_paired_run(
        reverse_root,
        pair_order="lookup-baseline",
        run_kind="baseline",
        position=2,
        output_tps=100.0,
        host="node-c",
    )

    with pytest.raises(ValueError, match="same compute host"):
        _summarize_e2e(tmp_path)


def test_summarize_e2e_rejects_cross_allocation_pair(tmp_path: Path) -> None:
    pair_root = tmp_path / "pairs" / "baseline-lookup"
    _write_paired_run(
        pair_root,
        pair_order="baseline-lookup",
        run_kind="baseline",
        position=1,
        output_tps=100.0,
        host="node-a",
        job_id="41",
    )
    _write_paired_run(
        pair_root,
        pair_order="baseline-lookup",
        run_kind="lookup",
        position=2,
        output_tps=110.0,
        host="node-a",
        job_id="42",
    )
    reverse_root = tmp_path / "pairs" / "lookup-baseline"
    for kind, position in (("lookup", 1), ("baseline", 2)):
        _write_paired_run(
            reverse_root,
            pair_order="lookup-baseline",
            run_kind=kind,
            position=position,
            output_tps=110.0 if kind == "lookup" else 100.0,
            host="node-b",
            job_id="43",
        )

    with pytest.raises(ValueError, match="same SLURM allocation"):
        _summarize_e2e(tmp_path)


def test_summarize_e2e_rejects_noncanonical_lookup_path(tmp_path: Path) -> None:
    for order, job_id in (
        ("baseline-lookup", "42"),
        ("lookup-baseline", "43"),
    ):
        pair_root = tmp_path / "pairs" / order
        positions = (
            {"baseline": 1, "lookup": 2}
            if order == "baseline-lookup"
            else {"baseline": 2, "lookup": 1}
        )
        for kind in ("baseline", "lookup"):
            _write_paired_run(
                pair_root,
                pair_order=order,
                run_kind=kind,
                position=positions[kind],
                output_tps=100.0 if kind == "baseline" else 110.0,
                host=f"node-{job_id}",
                job_id=job_id,
            )
    canonical = tmp_path / "oracle" / "lookup.json"
    alternate = tmp_path / "alternate-lookup.json"
    alternate.write_text(canonical.read_text())
    lookup_metadata = next(
        (tmp_path / "pairs" / "baseline-lookup" / "serving" / "lookup").glob(
            "*/metadata.txt"
        )
    )
    lookup_metadata.write_text(
        lookup_metadata.read_text().replace(str(canonical), str(alternate))
    )

    with pytest.raises(ValueError, match="canonical lookup manifest"):
        _summarize_e2e(tmp_path)


def test_summarize_lookup_combines_both_pair_orders(tmp_path: Path) -> None:
    for order in ("baseline-lookup", "lookup-baseline"):
        lookup = tmp_path / "pairs" / order / "serving" / "lookup" / order
        traces = lookup / "traces"
        traces.mkdir(parents=True)
        (lookup / "COMPLETE").touch()
        (traces / "counts.1.jsonl").write_text(
            json.dumps(
                {
                    "m": 1,
                    "n": 2304,
                    "k": 8192,
                    "runner": "CuteRunner",
                    "selection_source": "offline_lookup",
                    "invocation_count": 3,
                    "rank": "0",
                }
            )
            + "\n"
        )
        (traces / "counts.1.complete").write_text("token\n")
    oracle = tmp_path / "oracle"
    oracle.mkdir()
    (oracle / "lookup.json").write_text(
        json.dumps(
            {
                "backend": "cute-dsl",
                "scale_layout": "128x4",
                "flashinfer_commit": "def456",
                "flashinfer_version": "0.6.18",
                "flashinfer_file": "/flashinfer/__init__.py",
                "container_sha256": "container-sha",
                "gpu": "NVIDIA GB200",
                "entry_count": 1,
            }
        )
    )

    summary = _summarize_lookup(tmp_path, expected_rank_count=1)

    assert summary["pair_count"] == 2
    assert summary["unique_hit_count"] == 1
    assert summary["selection_call_count"] == 6
    assert summary["selection_call_hit_count"] == 6
    assert summary["selection_call_weighted_hit_rate"] == 1.0


def test_summarize_backend_combines_e2e_oracle_and_lookup_coverage(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "serving" / "baseline" / "11"
    lookup = tmp_path / "serving" / "lookup" / "22"
    baseline.mkdir(parents=True)
    lookup.mkdir(parents=True)
    (baseline / "COMPLETE").touch()
    (lookup / "COMPLETE").touch()
    _write_metadata(baseline / "metadata.txt", "baseline")
    _write_metadata(lookup / "metadata.txt", "lookup")
    _write_result(
        baseline / "result_isl1000_osl1000_c1.json",
        output_tps=100.0,
        duration=200.0,
    )
    _write_result(
        lookup / "result_isl1000_osl1000_c1.json",
        output_tps=110.0,
        duration=180.0,
    )
    traces = lookup / "traces"
    traces.mkdir()
    (traces / "counts.1.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "m": 1,
                        "n": 2304,
                        "k": 8192,
                        "runner": "CuteRunner",
                        "selection_source": "offline_lookup",
                        "invocation_count": 3,
                        "rank": "0",
                    }
                ),
                json.dumps(
                    {
                        "m": 2,
                        "n": 2304,
                        "k": 8192,
                        "runner": "CuteRunner",
                        "selection_source": "default_autotuner",
                        "invocation_count": 1,
                        "rank": "0",
                    }
                ),
            ]
        )
        + "\n"
    )
    (traces / "counts.1.complete").touch()
    oracle_root = tmp_path / "oracle"
    oracle_root.mkdir()
    (oracle_root / "lookup.json").write_text(
        json.dumps(
            {
                "backend": "cute-dsl",
                "scale_layout": "128x4",
                "flashinfer_commit": "def456",
                "flashinfer_version": "0.6.18",
                "flashinfer_file": "/flashinfer/__init__.py",
                "container_sha256": "container-sha",
                "gpu": "NVIDIA GB200",
                "entry_count": 2,
            }
        )
    )
    report_dir = tmp_path / "oracle" / "shards" / "0" / "oracle"
    report_dir.mkdir(parents=True)
    (report_dir / "report.json").write_text(
        json.dumps(
            {
                "backend": "cute-dsl",
                "scale_layout": "128x4",
                "flashinfer_commit": "def456",
                "flashinfer_version": "0.6.18",
                "flashinfer_file": "/flashinfer/__init__.py",
                "container_sha256": "container-sha",
                "gpu": "NVIDIA GB200",
                "correctness": {
                    "minimum_cosine_similarity": 0.98,
                    "rtol": 0.02,
                    "atol": 0.1,
                },
                "timing": ORACLE_TIMING,
                "profiling_wall_s": 12.5,
                "measured_candidate_gpu_s": 15.0,
                "shapes": [
                    {
                        "m": 1,
                        "n": 2304,
                        "k": 8192,
                        "speedup": 1.2,
                        "selected_ms": 1.2,
                        "oracle_ms": 1.0,
                        "candidate_count": 4,
                        "same_tactic": False,
                        "selected_tactic": 3,
                        "oracle_tactic": 7,
                        "oracle_cosine_similarity": 0.999,
                        "oracle_finite": True,
                        "oracle_matches_selected": True,
                    },
                    {
                        "m": 2,
                        "n": 2304,
                        "k": 8192,
                        "speedup": 1.0,
                        "selected_ms": 1.0,
                        "oracle_ms": 1.0,
                        "candidate_count": 2,
                        "same_tactic": True,
                        "selected_tactic": 3,
                        "oracle_tactic": 3,
                        "oracle_cosine_similarity": 0.998,
                        "oracle_finite": True,
                        "oracle_matches_selected": True,
                    },
                ],
            }
        )
    )
    summary = summarize_backend(tmp_path, "cute-dsl")

    assert summary["e2e"]["workloads"][0]["output_throughput_speedup"] == pytest.approx(
        1.1
    )
    assert summary["e2e"]["workloads"][0]["duration_reduction_pct"] == pytest.approx(
        10.0
    )
    assert len(summary["e2e"]["workloads"][0]["artifact_signature"]) == 64
    assert summary["oracle"]["shape_count"] == 2
    assert summary["oracle"]["geomean_speedup"] == pytest.approx(1.2**0.5)
    assert summary["oracle"]["different_tactic_count"] == 1
    assert summary["oracle"]["flashinfer_version"] == "0.6.18"
    assert summary["oracle"]["flashinfer_file"] == "/flashinfer/__init__.py"
    assert summary["oracle"]["container_sha256"] == "container-sha"
    assert summary["oracle"]["timing"] == ORACLE_TIMING
    assert summary["oracle"]["correctness"] == {
        "minimum_cosine_similarity_required": pytest.approx(0.98),
        "rtol": pytest.approx(0.02),
        "atol": pytest.approx(0.1),
        "finite_pass_count": 2,
        "selected_allclose_pass_count": 2,
        "bf16_cosine_pass_count": 2,
    }
    assert summary["oracle"]["measured_candidate_gpu_s"] == pytest.approx(15.0)
    assert summary["oracle"]["profiling_wall_s_estimate"] == pytest.approx(12.5)
    assert summary["oracle"]["candidate_count_min"] == 2
    assert summary["oracle"]["candidate_count_max"] == 4
    assert summary["oracle"]["regret_pct_p50"] == pytest.approx(10.0)
    assert summary["oracle"]["regret_pct_p90"] == pytest.approx(18.0)
    assert summary["oracle"]["regret_by_m"] == [
        {
            "m": 1,
            "shape_count": 1,
            "geomean_speedup": pytest.approx(1.2),
            "regret_pct_p50": pytest.approx(20.0),
            "regret_pct_p90": pytest.approx(20.0),
            "max_regret_pct": pytest.approx(20.0),
            "different_tactic_rate": pytest.approx(1.0),
        },
        {
            "m": 2,
            "shape_count": 1,
            "geomean_speedup": pytest.approx(1.0),
            "regret_pct_p50": pytest.approx(0.0),
            "regret_pct_p90": pytest.approx(0.0),
            "max_regret_pct": pytest.approx(0.0),
            "different_tactic_rate": pytest.approx(0.0),
        },
    ]
    assert summary["oracle"]["top_regrets"][0] == {
        "m": 1,
        "n": 2304,
        "k": 8192,
        "candidate_count": 4,
        "selected_ms": 1.2,
        "oracle_ms": 1.0,
        "regret_pct": pytest.approx(20.0),
        "selected_tactic": 3,
        "oracle_tactic": 7,
    }
    assert summary["lookup"]["unique_hit_count"] == 1
    assert summary["lookup"]["unique_miss_count"] == 1
    assert summary["lookup"]["unique_hit_rate"] == pytest.approx(0.5)
    assert summary["lookup"]["selection_call_weighted_hit_rate"] == pytest.approx(0.75)
    assert summary["lookup"]["selection_source_counts"] == {
        "default_autotuner": 1,
        "offline_lookup": 3,
    }
    assert summary["lookup"]["dispatches"] == [
        {
            "m": 1,
            "n": 2304,
            "k": 8192,
            "runner": "CuteRunner",
            "selection_source": "offline_lookup",
            "invocation_count": 3,
        },
        {
            "m": 2,
            "n": 2304,
            "k": 8192,
            "runner": "CuteRunner",
            "selection_source": "default_autotuner",
            "invocation_count": 1,
        },
    ]
    assert summary["lookup"]["manifest"]["backend"] == "cute-dsl"
    assert len(summary["lookup"]["manifest"]["sha256"]) == 64
    assert summary["lookup"]["coverage_by_m"] == [
        {"m": 1, "unique_dispatch_count": 1, "hit_rate": pytest.approx(1.0)},
        {"m": 2, "unique_dispatch_count": 1, "hit_rate": pytest.approx(0.0)},
    ]


def test_summarize_backend_requires_matched_result_files(tmp_path: Path) -> None:
    baseline = tmp_path / "serving" / "baseline" / "11"
    lookup = tmp_path / "serving" / "lookup" / "22"
    baseline.mkdir(parents=True)
    lookup.mkdir(parents=True)
    (baseline / "COMPLETE").touch()
    (lookup / "COMPLETE").touch()
    _write_metadata(baseline / "metadata.txt", "baseline")
    _write_metadata(lookup / "metadata.txt", "lookup")
    _write_result(
        baseline / "result_isl1000_osl1000_c1.json",
        output_tps=100.0,
        duration=200.0,
    )
    _write_result(
        lookup / "result_isl1000_osl1000_c8.json",
        output_tps=110.0,
        duration=180.0,
    )

    with pytest.raises(ValueError, match="result files do not match"):
        summarize_backend(tmp_path, "cute-dsl")


def test_summarize_e2e_rejects_expected_vllm_version_mismatch(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "serving" / "baseline" / "11"
    lookup = tmp_path / "serving" / "lookup" / "22"
    baseline.mkdir(parents=True)
    lookup.mkdir(parents=True)
    (baseline / "COMPLETE").touch()
    (lookup / "COMPLETE").touch()
    _write_metadata(baseline / "metadata.txt", "baseline")
    _write_metadata(lookup / "metadata.txt", "lookup")
    for metadata_path in (baseline / "metadata.txt", lookup / "metadata.txt"):
        metadata_path.write_text(
            metadata_path.read_text().replace(
                "\nvllm_version=0.27.1", "\nvllm_version=0.27.0"
            )
        )
    _write_result(
        baseline / "result_isl1000_osl1000_c1.json",
        output_tps=100.0,
        duration=200.0,
    )
    _write_result(
        lookup / "result_isl1000_osl1000_c1.json",
        output_tps=110.0,
        duration=180.0,
    )

    with pytest.raises(ValueError, match="vLLM version mismatch"):
        _summarize_e2e(tmp_path)


def test_summarize_oracle_rejects_incomplete_correctness(tmp_path: Path) -> None:
    report_dir = tmp_path / "oracle" / "shards" / "0" / "oracle"
    report_dir.mkdir(parents=True)
    (report_dir / "report.json").write_text(
        json.dumps(
            {
                "backend": "cute-dsl",
                "scale_layout": "128x4",
                "flashinfer_commit": "def456",
                "flashinfer_version": "0.6.18",
                "flashinfer_file": "/flashinfer/__init__.py",
                "container_sha256": "container-sha",
                "gpu": "NVIDIA GB200",
                "correctness": {
                    "minimum_cosine_similarity": 0.98,
                    "rtol": 0.02,
                    "atol": 0.1,
                },
                "timing": ORACLE_TIMING,
                "profiling_wall_s": 1.0,
                "measured_candidate_gpu_s": 1.0,
                "shapes": [
                    {
                        "m": 1,
                        "n": 2304,
                        "k": 8192,
                        "speedup": 1.0,
                        "selected_ms": 1.0,
                        "oracle_ms": 1.0,
                        "candidate_count": 1,
                        "same_tactic": True,
                        "selected_tactic": 3,
                        "oracle_tactic": 3,
                        "oracle_cosine_similarity": 0.99,
                        "oracle_finite": True,
                        "oracle_matches_selected": False,
                    }
                ],
            }
        )
    )

    with pytest.raises(ValueError, match="oracle correctness incomplete"):
        _summarize_oracle(tmp_path)


def test_summarize_backend_requires_matching_cudagraph_evidence(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "serving" / "baseline" / "11"
    lookup = tmp_path / "serving" / "lookup" / "22"
    baseline.mkdir(parents=True)
    lookup.mkdir(parents=True)
    (baseline / "COMPLETE").touch()
    (lookup / "COMPLETE").touch()
    _write_metadata(baseline / "metadata.txt", "baseline")
    _write_metadata(lookup / "metadata.txt", "lookup")
    lookup_metadata = lookup / "metadata.txt"
    lookup_metadata.write_text(
        lookup_metadata.read_text().replace(
            "cudagraph_capture_status=capture_completed",
            "cudagraph_capture_status=configured_not_observed",
        )
    )
    _write_result(
        baseline / "result_isl1000_osl1000_c1.json",
        output_tps=100.0,
        duration=200.0,
    )
    _write_result(
        lookup / "result_isl1000_osl1000_c1.json",
        output_tps=110.0,
        duration=180.0,
    )

    with pytest.raises(ValueError, match="metadata do not match"):
        summarize_backend(tmp_path, "cute-dsl")


def test_summarize_lookup_rejects_incomplete_expected_rank_coverage(
    tmp_path: Path,
) -> None:
    lookup = tmp_path / "serving" / "lookup" / "22"
    traces = lookup / "traces"
    traces.mkdir(parents=True)
    (lookup / "COMPLETE").touch()
    (traces / "counts.1.jsonl").write_text(
        json.dumps(
            {
                "m": 1,
                "n": 2304,
                "k": 8192,
                "runner": "CuteRunner",
                "selection_source": "offline_lookup",
                "invocation_count": 1,
                "rank": "0",
            }
        )
        + "\n"
    )
    (traces / "counts.1.complete").touch()

    with pytest.raises(ValueError, match="incomplete lookup rank coverage"):
        _summarize_lookup(tmp_path, expected_rank_count=2)


def test_summarize_lookup_rejects_unexpected_selection_source(tmp_path: Path) -> None:
    lookup = tmp_path / "serving" / "lookup" / "22"
    traces = lookup / "traces"
    traces.mkdir(parents=True)
    (lookup / "COMPLETE").touch()
    (traces / "counts.1.jsonl").write_text(
        json.dumps(
            {
                "m": 1,
                "n": 2304,
                "k": 8192,
                "runner": "CuteRunner",
                "selection_source": "forced_tactic",
                "invocation_count": 1,
                "rank": "0",
            }
        )
        + "\n"
    )
    (traces / "counts.1.complete").touch()

    with pytest.raises(ValueError, match="unexpected lookup selection sources"):
        _summarize_lookup(tmp_path, expected_rank_count=1)


def test_summarize_lookup_rejects_zero_offline_lookup_hits(tmp_path: Path) -> None:
    lookup = tmp_path / "serving" / "lookup" / "22"
    traces = lookup / "traces"
    traces.mkdir(parents=True)
    (lookup / "COMPLETE").touch()
    (traces / "counts.1.jsonl").write_text(
        json.dumps(
            {
                "m": 1,
                "n": 2304,
                "k": 8192,
                "runner": "CuteRunner",
                "selection_source": "default_autotuner",
                "invocation_count": 4,
                "rank": "0",
            }
        )
        + "\n"
    )
    (traces / "counts.1.complete").touch()

    with pytest.raises(ValueError, match="zero offline_lookup hits"):
        _summarize_lookup(tmp_path, expected_rank_count=1)


def _write_oracle_report(path: Path, timing: dict) -> None:
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "backend": "cute-dsl",
                "scale_layout": "128x4",
                "flashinfer_commit": "def456",
                "flashinfer_version": "0.6.18",
                "flashinfer_file": "/flashinfer/__init__.py",
                "container_sha256": "container-sha",
                "gpu": "NVIDIA GB200",
                "correctness": {
                    "minimum_cosine_similarity": 0.98,
                    "rtol": 0.02,
                    "atol": 0.1,
                },
                "timing": timing,
                "profiling_wall_s": 1.0,
                "measured_candidate_gpu_s": 1.0,
                "shapes": [
                    {
                        "m": 1,
                        "n": 2304,
                        "k": 8192,
                        "speedup": 1.0,
                        "selected_ms": 1.0,
                        "oracle_ms": 1.0,
                        "candidate_count": 1,
                        "same_tactic": True,
                        "selected_tactic": 3,
                        "oracle_tactic": 3,
                        "oracle_cosine_similarity": 0.99,
                        "oracle_finite": True,
                        "oracle_matches_selected": True,
                    }
                ],
            }
        )
    )


def test_summarize_oracle_rejects_incomplete_timing_block(tmp_path: Path) -> None:
    timing = dict(ORACLE_TIMING)
    del timing["calls_per_graph"]
    _write_oracle_report(
        tmp_path / "oracle" / "shards" / "0" / "oracle" / "report.json",
        timing,
    )

    with pytest.raises(ValueError, match="oracle timing block is incomplete"):
        _summarize_oracle(tmp_path)


def test_summarize_oracle_rejects_timing_mismatch_across_shards(
    tmp_path: Path,
) -> None:
    _write_oracle_report(
        tmp_path / "oracle" / "shards" / "0" / "oracle" / "report.json",
        dict(ORACLE_TIMING),
    )
    different_timing = {**ORACLE_TIMING, "repeat_iters": 21}
    _write_oracle_report(
        tmp_path / "oracle" / "shards" / "1" / "oracle" / "report.json",
        different_timing,
    )

    with pytest.raises(ValueError, match="oracle timing blocks do not match"):
        _summarize_oracle(tmp_path)


@pytest.mark.parametrize("field", ["cuda_graph", "cold_l2_cache"])
def test_summarize_oracle_requires_cuda_graph_and_cold_l2(
    tmp_path: Path,
    field: str,
) -> None:
    timing = {**ORACLE_TIMING, field: False}
    _write_oracle_report(
        tmp_path / "oracle" / "shards" / "0" / "oracle" / "report.json",
        timing,
    )

    with pytest.raises(ValueError, match="oracle timing requires"):
        _summarize_oracle(tmp_path)


def test_comparison_validation_rejects_cross_backend_provenance_mismatch(
    tmp_path: Path,
) -> None:
    metadata_path = tmp_path / "metadata.txt"
    _write_metadata(metadata_path, "baseline")
    metadata = dict(
        line.split("=", 1) for line in metadata_path.read_text().splitlines()
    )

    def summary(backend: str, model: str) -> dict:
        backend_metadata = {**metadata, "backend_name": backend, "model": model}
        return {
            "backend": backend,
            "e2e": {
                "metadata": backend_metadata,
                "workloads": [
                    {
                        "isl": 1000,
                        "osl": 1000,
                        "concurrency": 1,
                        "artifact_signature": "same",
                    }
                ],
            },
        }

    with pytest.raises(ValueError, match="cross-backend provenance mismatch"):
        summarize_results.validate_comparison_summaries(
            [
                summary("cute-dsl", "/model-a"),
                summary("cutlass", "/model-b"),
                summary("trtllm-128x4", "/model-a"),
                summary("trtllm-8x4", "/model-a"),
            ]
        )


def test_comparison_validation_requires_all_four_backend_arms() -> None:
    with pytest.raises(ValueError, match="requires exactly these backend arms"):
        summarize_results.validate_comparison_summaries(
            [
                {"backend": "cute-dsl", "e2e": {}},
                {"backend": "cutlass", "e2e": {}},
            ]
        )


def test_comparison_validation_rejects_cross_backend_artifact_mismatch(
    tmp_path: Path,
) -> None:
    metadata_path = tmp_path / "metadata.txt"
    _write_metadata(metadata_path, "baseline")
    metadata = dict(
        line.split("=", 1) for line in metadata_path.read_text().splitlines()
    )

    def summary(backend: str, signature: str) -> dict:
        return {
            "backend": backend,
            "e2e": {
                "metadata": {**metadata, "backend_name": backend},
                "workloads": [
                    {
                        "isl": 1000,
                        "osl": 1000,
                        "concurrency": 1,
                        "artifact_signature": signature,
                    }
                ],
            },
        }

    with pytest.raises(ValueError, match="cross-backend artifact mismatch"):
        summarize_results.validate_comparison_summaries(
            [
                summary("cute-dsl", "one"),
                summary("cutlass", "two"),
                summary("trtllm-128x4", "one"),
                summary("trtllm-8x4", "one"),
            ]
        )


def test_summarize_backend_rejects_generated_text_mismatch(tmp_path: Path) -> None:
    baseline = tmp_path / "serving" / "baseline" / "11"
    lookup = tmp_path / "serving" / "lookup" / "22"
    baseline.mkdir(parents=True)
    lookup.mkdir(parents=True)
    (baseline / "COMPLETE").touch()
    (lookup / "COMPLETE").touch()
    _write_metadata(baseline / "metadata.txt", "baseline")
    _write_metadata(lookup / "metadata.txt", "lookup")
    _write_result(
        baseline / "result_isl1000_osl1000_c1.json",
        output_tps=100.0,
        duration=200.0,
        generated_texts=["baseline"] * 10,
    )
    _write_result(
        lookup / "result_isl1000_osl1000_c1.json",
        output_tps=110.0,
        duration=180.0,
        generated_texts=["lookup"] * 10,
    )

    with pytest.raises(ValueError, match="generated text mismatch"):
        summarize_backend(tmp_path, "cute-dsl")


def test_summarize_backend_rejects_execution_metadata_mismatch(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "serving" / "baseline" / "11"
    lookup = tmp_path / "serving" / "lookup" / "22"
    baseline.mkdir(parents=True)
    lookup.mkdir(parents=True)
    (baseline / "COMPLETE").touch()
    (lookup / "COMPLETE").touch()
    _write_metadata(baseline / "metadata.txt", "baseline")
    _write_metadata(lookup / "metadata.txt", "lookup")
    lookup_metadata = lookup / "metadata.txt"
    lookup_metadata.write_text(
        lookup_metadata.read_text().replace(
            "flashinfer_commit=def456", "flashinfer_commit=different"
        )
    )

    with pytest.raises(ValueError, match="metadata do not match"):
        summarize_backend(tmp_path, "cute-dsl")


def test_summarize_backend_requires_generated_texts(tmp_path: Path) -> None:
    baseline = tmp_path / "serving" / "baseline" / "11"
    lookup = tmp_path / "serving" / "lookup" / "22"
    baseline.mkdir(parents=True)
    lookup.mkdir(parents=True)
    (baseline / "COMPLETE").touch()
    (lookup / "COMPLETE").touch()
    _write_metadata(baseline / "metadata.txt", "baseline")
    _write_metadata(lookup / "metadata.txt", "lookup")
    baseline_result = baseline / "result_isl1000_osl1000_c1.json"
    lookup_result = lookup / "result_isl1000_osl1000_c1.json"
    _write_result(baseline_result, output_tps=100.0, duration=200.0)
    _write_result(lookup_result, output_tps=110.0, duration=180.0)
    for path in (baseline_result, lookup_result):
        result = json.loads(path.read_text())
        del result["generated_texts"]
        path.write_text(json.dumps(result))

    with pytest.raises(ValueError, match="missing generated texts"):
        summarize_backend(tmp_path, "cute-dsl")


def test_build_study_summary_preserves_four_measured_path(tmp_path: Path) -> None:
    metadata = _comparison_metadata(tmp_path)
    summaries = [
        _comparison_summary(backend, metadata)
        for backend in reversed(EXPECTED_BACKENDS)
    ]

    study = summarize_results.build_study_summary(summaries, [])

    assert study["study_status"] == "complete"
    assert study["measured_backends"] == list(EXPECTED_BACKENDS)
    assert study["unsupported_backends"] == []
    assert study["metric_comparison_status"] == "complete"
    assert [record["backend"] for record in study["backends"]] == list(
        EXPECTED_BACKENDS
    )
    assert {record["status"] for record in study["backends"]} == {"measured"}


def test_main_emits_evidence_backed_partial_comparison(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = _comparison_metadata(tmp_path)
    measured_backends = ("cute-dsl", "trtllm-128x4", "trtllm-8x4")
    summaries = {
        backend: _comparison_summary(backend, metadata) for backend in measured_backends
    }
    status_path = _write_unsupported_status(tmp_path, metadata)
    output_json = tmp_path / "summary.json"
    output_csv = tmp_path / "e2e.csv"
    monkeypatch.setattr(
        summarize_results,
        "summarize_backend",
        lambda _root, backend: summaries[backend],
    )
    argv = ["summarize_results.py"]
    for backend in reversed(measured_backends):
        argv.extend(("--result", f"{backend}={tmp_path / backend}"))
    argv.extend(
        (
            "--unsupported",
            f"cutlass={status_path}",
            "--output-json",
            str(output_json),
            "--output-csv",
            str(output_csv),
        )
    )
    monkeypatch.setattr(sys, "argv", argv)

    summarize_results.main()

    study = json.loads(output_json.read_text())
    assert study["study_status"] == "complete_with_unsupported_arm"
    assert study["measured_backends"] == [
        "cute-dsl",
        "trtllm-128x4",
        "trtllm-8x4",
    ]
    assert study["unsupported_backends"] == ["cutlass"]
    assert study["metric_comparison_status"] == "partial"
    assert [record["backend"] for record in study["backends"]] == list(
        EXPECTED_BACKENDS
    )
    unsupported = study["backends"][1]
    assert unsupported["status"] == "empirically_unsupported"
    assert set(unsupported) >= {
        "backend",
        "status",
        "recipe",
        "provenance",
        "failure",
    }
    assert "evidence" not in unsupported
    assert unsupported["failure"]["evidence"]
    assert Path(unsupported["failure"]["evidence"][0]["path"]).is_absolute()
    assert not {"e2e", "oracle", "lookup"} & unsupported.keys()
    with output_csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["backend"] for row in rows] == list(study["measured_backends"])


def test_main_preserves_legacy_four_measured_json_shape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = _comparison_metadata(tmp_path)
    summaries = {
        backend: _comparison_summary(backend, metadata) for backend in EXPECTED_BACKENDS
    }
    output_json = tmp_path / "summary.json"
    output_csv = tmp_path / "e2e.csv"
    monkeypatch.setattr(
        summarize_results,
        "summarize_backend",
        lambda _root, backend: summaries[backend],
    )
    argv = ["summarize_results.py"]
    for backend in reversed(EXPECTED_BACKENDS):
        argv.extend(("--result", f"{backend}={tmp_path / backend}"))
    argv.extend(
        (
            "--output-json",
            str(output_json),
            "--output-csv",
            str(output_csv),
        )
    )
    monkeypatch.setattr(sys, "argv", argv)

    summarize_results.main()

    payload = json.loads(output_json.read_text())
    assert set(payload) == {"backends"}
    assert [record["backend"] for record in payload["backends"]] == list(
        reversed(EXPECTED_BACKENDS)
    )
    assert all("status" not in record for record in payload["backends"])


def test_unsupported_status_rejects_metric_sections(tmp_path: Path) -> None:
    metadata = _comparison_metadata(tmp_path)
    status_path = _write_unsupported_status(
        tmp_path,
        metadata,
        extra={"oracle": {"shape_count": 0}},
    )

    with pytest.raises(ValueError, match="metric sections"):
        summarize_results.summarize_unsupported_backend(status_path, "cutlass")


def test_unsupported_status_rejects_missing_evidence_file(tmp_path: Path) -> None:
    metadata = _comparison_metadata(tmp_path)
    status_path = _write_unsupported_status(tmp_path, metadata)
    payload = json.loads(status_path.read_text())
    payload["failure"]["evidence"][0]["path"] = "missing.stderr"
    status_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="evidence file does not exist"):
        summarize_results.summarize_unsupported_backend(status_path, "cutlass")


def test_unsupported_status_rejects_evidence_checksum_mismatch(
    tmp_path: Path,
) -> None:
    metadata = _comparison_metadata(tmp_path)
    status_path = _write_unsupported_status(tmp_path, metadata)
    payload = json.loads(status_path.read_text())
    payload["failure"]["evidence"][0]["sha256"] = "0" * 64
    status_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="evidence checksum mismatch"):
        summarize_results.summarize_unsupported_backend(status_path, "cutlass")


def test_unsupported_status_requires_exact_evidence_entry_schema(
    tmp_path: Path,
) -> None:
    metadata = _comparison_metadata(tmp_path)
    status_path = _write_unsupported_status(tmp_path, metadata)
    payload = json.loads(status_path.read_text())
    payload["failure"]["evidence"][0]["kind"] = "stderr"
    status_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="evidence entry fields"):
        summarize_results.summarize_unsupported_backend(status_path, "cutlass")


def test_unsupported_status_rejects_recipe_identity_mismatch(tmp_path: Path) -> None:
    metadata = _comparison_metadata(tmp_path)
    status_path = _write_unsupported_status(tmp_path, metadata)
    payload = json.loads(status_path.read_text())
    payload["recipe"]["backend_name"] = "cute-dsl"
    status_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="recipe mismatch"):
        summarize_results.summarize_unsupported_backend(status_path, "cutlass")


def test_unsupported_status_requires_matching_measured_provenance(
    tmp_path: Path,
) -> None:
    metadata = _comparison_metadata(tmp_path)
    status_path = _write_unsupported_status(tmp_path, metadata)
    unsupported = summarize_results.summarize_unsupported_backend(
        status_path, "cutlass"
    )
    unsupported["provenance"]["gpu_name"] = "NVIDIA H100"
    measured = [
        _comparison_summary(backend, metadata)
        for backend in ("cute-dsl", "trtllm-128x4", "trtllm-8x4")
    ]

    with pytest.raises(ValueError, match="unsupported provenance mismatch"):
        summarize_results.build_study_summary(measured, [unsupported])


def test_partial_comparison_still_requires_four_backend_arms(
    tmp_path: Path,
) -> None:
    metadata = _comparison_metadata(tmp_path)
    status_path = _write_unsupported_status(tmp_path, metadata)
    unsupported = summarize_results.summarize_unsupported_backend(
        status_path, "cutlass"
    )
    measured = [
        _comparison_summary(backend, metadata)
        for backend in ("cute-dsl", "trtllm-128x4")
    ]

    with pytest.raises(ValueError, match="requires exactly these backend arms"):
        summarize_results.build_study_summary(measured, [unsupported])


def test_partial_comparison_supports_at_most_one_unsupported_arm(
    tmp_path: Path,
) -> None:
    metadata = _comparison_metadata(tmp_path)
    cutlass = summarize_results.summarize_unsupported_backend(
        _write_unsupported_status(tmp_path, metadata, backend="cutlass"),
        "cutlass",
    )
    trtllm_128x4_path = _write_unsupported_status(
        tmp_path,
        metadata,
        backend="trtllm-128x4",
    )
    trtllm_128x4_payload = json.loads(trtllm_128x4_path.read_text())
    trtllm_128x4_payload["recipe"] = {
        "backend_name": "trtllm-128x4",
        "linear_backend": "flashinfer_trtllm",
        "oracle_backend": "trtllm",
        "scale_layout": "128x4",
        "trtllm_layout": "128x4",
    }
    trtllm_128x4_path.write_text(json.dumps(trtllm_128x4_payload))
    trtllm_128x4 = summarize_results.summarize_unsupported_backend(
        trtllm_128x4_path,
        "trtllm-128x4",
    )
    measured = [
        _comparison_summary(backend, metadata) for backend in ("cute-dsl", "trtllm-8x4")
    ]

    with pytest.raises(ValueError, match="at most one unsupported backend arm"):
        summarize_results.build_study_summary(
            measured,
            [cutlass, trtllm_128x4],
        )


def test_unsupported_provenance_constant_is_stable_cross_backend_subset() -> None:
    excluded = {
        "enforce_eager",
        "cudagraph_configured",
        "cudagraph_capture_status",
        "cudagraph_capture_evidence",
        "cudagraph_capture_marker",
        "workloads",
        "concurrencies",
    }

    assert (
        tuple(
            key
            for key in summarize_results._CROSS_BACKEND_METADATA
            if key not in excluded
        )
        == summarize_results._UNSUPPORTED_PROVENANCE_METADATA
    )


def test_unsupported_status_rejects_completed_run_provenance(tmp_path: Path) -> None:
    metadata = _comparison_metadata(tmp_path)
    status_path = _write_unsupported_status(tmp_path, metadata)
    payload = json.loads(status_path.read_text())
    payload["provenance"]["cudagraph_capture_status"] = "capture_completed"
    status_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="completed-run provenance"):
        summarize_results.summarize_unsupported_backend(status_path, "cutlass")


def test_unsupported_status_requires_nonempty_attempt_list(tmp_path: Path) -> None:
    metadata = _comparison_metadata(tmp_path)
    status_path = _write_unsupported_status(tmp_path, metadata)
    payload = json.loads(status_path.read_text())
    payload["failure"]["attempts"] = []
    status_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="attempts must be a non-empty list"):
        summarize_results.summarize_unsupported_backend(status_path, "cutlass")


def test_unsupported_status_requires_nonempty_attempt_objects(tmp_path: Path) -> None:
    metadata = _comparison_metadata(tmp_path)
    status_path = _write_unsupported_status(tmp_path, metadata)
    payload = json.loads(status_path.read_text())
    payload["failure"]["attempts"] = [None, {}]
    status_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="attempts must contain non-empty objects"):
        summarize_results.summarize_unsupported_backend(status_path, "cutlass")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (("stage", "success"), "unsupported failure stage"),
        (("outcome", "passed"), "unsupported attempt outcome"),
    ],
)
def test_unsupported_status_requires_empirical_failure_attempts(
    tmp_path: Path,
    mutation: tuple[str, str],
    message: str,
) -> None:
    metadata = _comparison_metadata(tmp_path)
    status_path = _write_unsupported_status(tmp_path, metadata)
    payload = json.loads(status_path.read_text())
    field, value = mutation
    if field == "stage":
        payload["failure"][field] = value
    else:
        payload["failure"]["attempts"][0][field] = value
    status_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match=message):
        summarize_results.summarize_unsupported_backend(status_path, "cutlass")


def test_unsupported_status_rejects_unexpected_provenance_fields(
    tmp_path: Path,
) -> None:
    metadata = _comparison_metadata(tmp_path)
    status_path = _write_unsupported_status(tmp_path, metadata)
    payload = json.loads(status_path.read_text())
    payload["provenance"]["backend_name"] = "cute-dsl"
    status_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="unexpected unsupported provenance fields"):
        summarize_results.summarize_unsupported_backend(status_path, "cutlass")


def test_build_study_summary_revalidates_unsupported_schema(tmp_path: Path) -> None:
    metadata = _comparison_metadata(tmp_path)
    measured = [
        _comparison_summary(backend, metadata)
        for backend in ("cute-dsl", "trtllm-128x4", "trtllm-8x4")
    ]
    malformed = {
        "backend": "cutlass",
        "status": "empirically_unsupported",
        "provenance": {key: metadata[key] for key in UNSUPPORTED_PROVENANCE_FIELDS},
    }

    with pytest.raises(ValueError, match="unsupported status fields"):
        summarize_results.build_study_summary(measured, [malformed])
