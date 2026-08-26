# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from pathlib import Path

import pytest

from experiments.ultra_mxfp8_all_observed_tactics import summarize_results
from experiments.ultra_mxfp8_all_observed_tactics.summarize_results import (
    _summarize_lookup,
    summarize_backend,
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
                "gpu_name=NVIDIA GB200",
                "driver_version=590.00",
                "container=/container.sqsh",
                "container_sha256=container-sha",
                "model=/model",
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
    report_dir = tmp_path / "oracle" / "shards" / "0" / "oracle"
    report_dir.mkdir(parents=True)
    (report_dir / "report.json").write_text(
        json.dumps(
            {
                "backend": "cute-dsl",
                "scale_layout": "128x4",
                "flashinfer_commit": "def456",
                "gpu": "NVIDIA GB200",
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
            [summary("cute-dsl", "/model-a"), summary("cutlass", "/model-b")]
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
            [summary("cute-dsl", "one"), summary("cutlass", "two")]
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
