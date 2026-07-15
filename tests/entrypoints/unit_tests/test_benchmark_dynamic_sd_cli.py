# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import hashlib
import json
from pathlib import Path

import pytest

from vllm.benchmarks.dynamic_sd import (
    DYNAMIC_SD_PROFILE_CAPABILITIES,
    DYNAMIC_SD_PROFILE_CONTRACT_VERSION,
    ProfileArgs,
    build_worker_configs,
    select_profile,
)
from vllm.benchmarks.dynamic_sd_core import SelectionPolicy
from vllm.entrypoints.cli.benchmark.dynamic_sd import (
    BenchmarkDynamicSDSubcommand,
)


def _profile_argv(tmp_path: Path) -> list[str]:
    workload = tmp_path / "prompts.jsonl"
    workload.write_text('{"prompt_token_ids":[1,2,3]}\n' * 4)
    return [
        "profile",
        "--model",
        "Qwen/Qwen3-235B-A22B",
        "--model-revision",
        "a" * 40,
        "--speculative-model",
        "nvidia/Qwen3-235B-A22B-Eagle3",
        "--speculative-model-revision",
        "b" * 40,
        "--speculative-method",
        "eagle3",
        "--tensor-parallel-size",
        "8",
        "--draft-tensor-parallel-size",
        "1",
        "--data-parallel-size",
        "1",
        "--engine-kwargs-json",
        (
            '{"dtype":"bfloat16","enable_chunked_prefill":true,'
            '"enable_prefix_caching":true,"gpu_memory_utilization":0.4,'
            '"kv_cache_dtype":"auto","max_model_len":8192,'
            '"max_num_batched_tokens":2048,"moe_backend":"triton"}'
        ),
        "--scheduler-batch-sizes",
        "1,2,4",
        "--k-values",
        "0,1,2,3",
        "--common-kmax",
        "3",
        "--profile-backend",
        "offline-sync",
        "--temperature",
        "1.0",
        "--top-p",
        "1.0",
        "--draft-sample-method",
        "probabilistic",
        "--rejection-sample-method",
        "standard",
        "--cuda-graph-mode",
        "FULL_AND_PIECEWISE",
        "--enforce-eager",
        "false",
        "--warmups",
        "1",
        "--repeats",
        "3",
        "--seed",
        "17",
        "--min-tokens",
        "32",
        "--max-tokens",
        "32",
        "--ignore-eos",
        "--prompt-workload",
        str(workload),
        "--output-dir",
        str(tmp_path / "profile"),
    ]


def test_dynamic_sd_profile_cli_contract_and_normalization(tmp_path: Path):
    parser = argparse.ArgumentParser()
    BenchmarkDynamicSDSubcommand.add_cli_args(parser)

    namespace = parser.parse_args(_profile_argv(tmp_path))
    profile = ProfileArgs.from_namespace(namespace)

    assert namespace.dynamic_sd_action == "profile"
    assert callable(namespace.dynamic_sd_dispatch)
    assert profile.scheduler_keys == (1, 2, 4)
    assert profile.k_values == (0, 1, 2, 3)
    assert profile.common_kmax == 3
    assert profile.enforce_eager is False
    assert profile.engine_kwargs == {
        "dtype": "bfloat16",
        "enable_chunked_prefill": True,
        "enable_prefix_caching": True,
        "gpu_memory_utilization": 0.4,
        "kv_cache_dtype": "auto",
        "max_model_len": 8192,
        "max_num_batched_tokens": 2048,
        "moe_backend": "triton",
    }
    assert DYNAMIC_SD_PROFILE_CONTRACT_VERSION == 1
    assert (
        frozenset(
            {
                "deployment_engine_identity",
                "external_ray_cluster",
                "fixed_output_calibration",
                "full_and_piecewise_capture",
                "process_isolated_candidates",
            }
        )
        == DYNAMIC_SD_PROFILE_CAPABILITIES
    )


def test_profile_builds_runtime_k_grid_with_common_kmax(tmp_path: Path):
    parser = argparse.ArgumentParser()
    BenchmarkDynamicSDSubcommand.add_cli_args(parser)
    profile = ProfileArgs.from_namespace(parser.parse_args(_profile_argv(tmp_path)))

    configs = build_worker_configs(profile)

    assert [config.k for config in configs] == [0, 1, 2, 3]
    assert all(config.kmax == 3 for config in configs)
    assert all(config.scheduler_keys == (1, 2, 4) for config in configs)
    assert configs[0].cudagraph_capture_sizes == (1, 2, 4)
    assert configs[-1].cudagraph_capture_sizes == (4, 8, 16)
    assert configs[-1].engine_kwargs["tensor_parallel_size"] == 8
    assert configs[-1].engine_kwargs["max_model_len"] == 8192
    assert configs[-1].engine_kwargs["gpu_memory_utilization"] == 0.4
    assert configs[-1].profile_identity.payload["engine_kwargs"] == {
        "dtype": "bfloat16",
        "enable_chunked_prefill": True,
        "enable_prefix_caching": True,
        "gpu_memory_utilization": 0.4,
        "kv_cache_dtype": "auto",
        "max_model_len": 8192,
        "max_num_batched_tokens": 2048,
        "moe_backend": "triton",
    }
    assert configs[-1].draft_tensor_parallel_size == 1


def test_profile_rejects_non_object_engine_kwargs_json(tmp_path: Path):
    parser = argparse.ArgumentParser()
    BenchmarkDynamicSDSubcommand.add_cli_args(parser)
    argv = _profile_argv(tmp_path)
    value_index = argv.index("--engine-kwargs-json") + 1
    argv[value_index] = "[]"

    with pytest.raises(ValueError, match="JSON object"):
        ProfileArgs.from_namespace(parser.parse_args(argv))


def _write_raw_grid(tmp_path: Path, throughputs: dict[int, float]) -> Path:
    output_dir = tmp_path / "profile"
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True)
    raw_results = []
    for k, throughput in sorted(throughputs.items()):
        path = raw_dir / f"k{k}.json"
        speculative_config = {
            "method": "eagle3",
            "model": "draft-model",
            "revision": "b" * 40,
            "num_speculative_tokens": 3,
            "draft_tensor_parallel_size": 1,
            "draft_sample_method": "probabilistic",
            "rejection_sample_method": "standard",
        }
        payload = {
            "status": "complete",
            "profile_id": "profile-a",
            "profile_identity": {"scheduler_keys": [1, 2, 4], "repeats": 3},
            "workload_hash": "workload-a",
            "variant": f"k{k}",
            "forced_k": k,
            "engine_kwargs": {"speculative_config": speculative_config},
            "measurements": [
                {
                    "scheduler_key": key,
                    "repeat": repeat,
                    "output_tokens_per_second": throughput,
                    "scheduler_key_coverage": 1.0,
                }
                for key in (1, 2, 4)
                for repeat in range(3)
            ],
        }
        path.write_text(json.dumps(payload))
        raw_results.append(
            {
                "k": k,
                "path": f"raw/k{k}.json",
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    manifest = {
        "schema_version": 1,
        "profile_id": "profile-a",
        "workload_hash": "workload-a",
        "k_values": [0, 1, 2, 3],
        "common_kmax": 3,
        "scheduler_keys": [1, 2, 4],
        "raw_results": raw_results,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest))
    return output_dir


def test_select_profile_emits_separate_runtime_config_sections(tmp_path: Path):
    output_dir = _write_raw_grid(tmp_path, {0: 100, 1: 120, 2: 110, 3: 105})

    summary = select_profile(output_dir, SelectionPolicy(configured_ks=(0, 1, 2, 3)))

    runtime = json.loads((output_dir / "dynamic_speculative_config.json").read_text())
    assert summary["schedule"] == [[1, 4, 1]]
    assert runtime["speculative_config"]["num_speculative_tokens_per_batch_size"] == [
        [1, 4, 1]
    ]
    assert runtime["compilation_config"]["cudagraph_capture_sizes"] == [2, 4, 6, 8]
    assert "cudagraph_capture_sizes" not in runtime["speculative_config"]


def test_select_profile_rejects_extra_raw_result(tmp_path: Path):
    output_dir = _write_raw_grid(tmp_path, {0: 100, 1: 120, 2: 110, 3: 105})
    (output_dir / "raw" / "stale.json").write_text("{}")

    with pytest.raises(ValueError, match="missing or extra"):
        select_profile(output_dir, SelectionPolicy(configured_ks=(0, 1, 2, 3)))


def test_boundary_k_winner_writes_diagnostic_but_no_runtime_config(
    tmp_path: Path,
):
    output_dir = _write_raw_grid(tmp_path, {0: 100, 1: 110, 2: 120, 3: 140})

    with pytest.raises(ValueError, match="extend the K grid"):
        select_profile(output_dir, SelectionPolicy(configured_ks=(0, 1, 2, 3)))

    assert (output_dir / "summary.json").is_file()
    assert not (output_dir / "dynamic_speculative_config.json").exists()
