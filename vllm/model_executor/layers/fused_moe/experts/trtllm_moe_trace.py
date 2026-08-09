# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import math
import os
import threading
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

_TRACE_DIR_ENV_VAR = "VLLM_MXFP8_MOE_TRACE_DIR"
_TRACE_INTERVAL_ENV_VAR = "VLLM_MXFP8_MOE_TRACE_INTERVAL"
_TRACE_MAX_SAMPLES_ENV_VAR = "VLLM_MXFP8_MOE_TRACE_MAX_SAMPLES"
_TRACE_SAMPLING_LOCK = threading.Lock()
_TRACE_CALL_COUNT = 0
_TRACE_SAMPLE_COUNT = 0


@dataclass(frozen=True)
class MoeTraceMetadata:
    schema_version: int
    model_revision: str
    layer_family: str
    global_num_experts: int
    local_num_experts: int
    top_k: int
    hidden_size: int
    intermediate_size: int
    tp_size: int
    ep_size: int
    dp_size: int
    cuda_graph_state: str
    weight_layout: str
    quantization: str
    runtime_fingerprint: str


def trace_enabled() -> bool:
    return bool(os.getenv(_TRACE_DIR_ENV_VAR))


def _positive_env_int(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def should_sample_routing_signature() -> bool:
    """Return whether this process should trace the current MoE call."""
    if not trace_enabled():
        return False

    interval = _positive_env_int(_TRACE_INTERVAL_ENV_VAR, 1)
    max_samples = _positive_env_int(_TRACE_MAX_SAMPLES_ENV_VAR, 2048)
    global _TRACE_CALL_COUNT, _TRACE_SAMPLE_COUNT
    with _TRACE_SAMPLING_LOCK:
        call_index = _TRACE_CALL_COUNT
        _TRACE_CALL_COUNT += 1
        if _TRACE_SAMPLE_COUNT >= max_samples or call_index % interval != 0:
            return False
        _TRACE_SAMPLE_COUNT += 1
        return True


def _reset_trace_sampling_for_testing() -> None:
    global _TRACE_CALL_COUNT, _TRACE_SAMPLE_COUNT
    with _TRACE_SAMPLING_LOCK:
        _TRACE_CALL_COUNT = 0
        _TRACE_SAMPLE_COUNT = 0


def _trace_process_rank() -> int:
    replica_rank = os.getenv("RANK")
    if replica_rank is not None:
        return int(replica_rank)
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def allocate_routing_replay(
    num_tokens: int, top_k: int, device: torch.device
) -> torch.Tensor | None:
    if not trace_enabled():
        return None
    return torch.full(
        (num_tokens, top_k), -1, dtype=torch.int16, device=device
    )


def record_routing_signature(
    topk_ids: torch.Tensor,
    metadata: MoeTraceMetadata,
    sampled_gpu_time_us: float,
) -> None:
    if not trace_enabled():
        return
    if topk_ids.ndim != 2 or not topk_ids.is_contiguous():
        raise ValueError("topk_ids must be a contiguous rank-2 tensor")
    if (
        topk_ids.dtype is torch.bool
        or topk_ids.is_floating_point()
        or topk_ids.is_complex()
    ):
        raise ValueError("topk_ids must have an integer dtype")
    if not math.isfinite(sampled_gpu_time_us) or sampled_gpu_time_us <= 0:
        raise ValueError("sampled_gpu_time_us must be finite and positive")
    if metadata.global_num_experts <= 0:
        raise ValueError("global_num_experts must be positive")
    flattened_ids = topk_ids.flatten().to(torch.int64)
    histogram = torch.bincount(
        flattened_ids.clamp(-1, metadata.global_num_experts) + 1,
        minlength=metadata.global_num_experts + 2,
    )
    histogram_on_cpu = histogram.cpu().tolist()
    if histogram_on_cpu[0] or histogram_on_cpu[-1]:
        raise ValueError("topk_ids contain an expert ID outside the valid range")
    row = {
        **asdict(metadata),
        "num_tokens": topk_ids.shape[0],
        "sampled_gpu_time_us": sampled_gpu_time_us,
        "expert_counts": histogram_on_cpu[1:-1],
    }
    trace_dir = Path(os.environ[_TRACE_DIR_ENV_VAR])
    trace_dir.mkdir(parents=True, exist_ok=True)
    rank = _trace_process_rank()
    trace_path = trace_dir / f"moe-routing-rank{rank}-pid{os.getpid()}.jsonl"
    with trace_path.open("a", encoding="ascii") as trace_file:
        trace_file.write(json.dumps(row, ensure_ascii=True, allow_nan=False) + "\n")
