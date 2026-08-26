# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from pathlib import Path

import pytest

from experiments.ultra_mxfp8_all_observed_tactics.summarize_results import (
    summarize_backend,
)


def _write_result(path: Path, *, output_tps: float, duration: float) -> None:
    path.write_text(
        json.dumps(
            {
                "completed": 10,
                "failed": 0,
                "total_input_tokens": 10_000,
                "total_output_tokens": 10_000,
                "output_throughput": output_tps,
                "total_token_throughput": output_tps * 2,
                "mean_ttft_ms": 100.0,
                "mean_tpot_ms": 10.0,
                "duration": duration,
            }
        )
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
    (traces / "trace.1.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "m": 1,
                        "n": 2304,
                        "k": 8192,
                        "runner": "CuteRunner",
                        "selection_source": "offline_lookup",
                    }
                ),
                json.dumps(
                    {
                        "m": 2,
                        "n": 2304,
                        "k": 8192,
                        "runner": "CuteRunner",
                        "selection_source": "default_autotuner",
                    }
                ),
            ]
        )
        + "\n"
    )
    report_dir = tmp_path / "oracle" / "shards" / "0" / "oracle"
    report_dir.mkdir(parents=True)
    (report_dir / "report.json").write_text(
        json.dumps(
            {
                "backend": "cute-dsl",
                "scale_layout": "128x4",
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
    cache_dir = tmp_path / "oracle" / "shards" / "0" / "cache"
    cache_dir.mkdir(parents=True)
    (cache_dir / "exact_cache_metadata.json").write_text(
        json.dumps({"tuning_time_s": 12.5})
    )
    second_cache_dir = tmp_path / "oracle" / "shards" / "1" / "cache"
    second_cache_dir.mkdir(parents=True)
    (second_cache_dir / "exact_cache_metadata.json").write_text(
        json.dumps({"tuning_time_s": 2.5})
    )

    summary = summarize_backend(tmp_path, "cute-dsl")

    assert summary["e2e"]["workloads"][0]["output_throughput_speedup"] == pytest.approx(
        1.1
    )
    assert summary["e2e"]["workloads"][0]["duration_reduction_pct"] == pytest.approx(
        10.0
    )
    assert summary["oracle"]["shape_count"] == 2
    assert summary["oracle"]["geomean_speedup"] == pytest.approx(1.2**0.5)
    assert summary["oracle"]["different_tactic_count"] == 1
    assert summary["oracle"]["cache_tuning_gpu_s"] == pytest.approx(15.0)
    assert summary["oracle"]["cache_tuning_wall_s_estimate"] == pytest.approx(12.5)
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
