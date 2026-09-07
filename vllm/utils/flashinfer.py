# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compatibility wrapper for FlashInfer API changes.

Users of vLLM should always import **only** these wrappers.
"""

import contextlib
import functools
import importlib
import importlib.util
import os
import shutil
from collections.abc import Callable
from typing import Any, NoReturn

import requests
import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

# This is the storage path for the cubins, it can be replaced
# with a local path for testing.
# Referenced from https://github.com/flashinfer-ai/flashinfer/blob/0c9a92c3d9a7e043ab6f3f7b2273269caf6ab044/flashinfer/jit/cubin_loader.py#L35  # noqa: E501
FLASHINFER_CUBINS_REPOSITORY = os.environ.get(
    "FLASHINFER_CUBINS_REPOSITORY",
    "https://edge.urm.nvidia.com/artifactory/sw-kernelinferencelibrary-public-generic-local/",  # noqa: E501
)


@functools.cache
def has_flashinfer_cubin() -> bool:
    """Return `True` if flashinfer-cubin package is available."""
    if envs.VLLM_HAS_FLASHINFER_CUBIN:
        return True
    if importlib.util.find_spec("flashinfer_cubin") is not None:
        return True
    logger.debug_once("flashinfer-cubin package was not found")
    return False


@functools.cache
def has_flashinfer() -> bool:
    """Return `True` if flashinfer-python package is available."""
    # Use find_spec to check if the module exists without importing it
    # This avoids potential CUDA initialization side effects
    if importlib.util.find_spec("flashinfer") is None:
        logger.debug_once("FlashInfer unavailable since package was not found")
        return False
    # When not using flashinfer cubin,
    # Also check if nvcc is available since it's required to JIT compile flashinfer
    if not has_flashinfer_cubin() and shutil.which("nvcc") is None:
        logger.debug_once(
            "FlashInfer unavailable since nvcc was not found "
            "and not using pre-downloaded cubins"
        )
        return False
    return True


def _missing(*_: Any, **__: Any) -> NoReturn:
    """Placeholder for unavailable FlashInfer backend."""
    raise RuntimeError(
        "FlashInfer backend is not available. Please install the package "
        "to enable FlashInfer kernels: "
        "https://github.com/flashinfer-ai/flashinfer"
    )


def _get_submodule(module_name: str) -> Any | None:
    """Safely import a submodule and return it, or None if not available."""
    try:
        return importlib.import_module(module_name)
    except (ImportError, ModuleNotFoundError):
        return None


# General lazy import wrapper
def _lazy_import_wrapper(
    module_name: str, attr_name: str, fallback_fn: Callable[..., Any] = _missing
):
    """Create a lazy import wrapper for a specific function."""

    @functools.cache
    def _get_impl():
        if not has_flashinfer():
            return None
        mod = _get_submodule(module_name)
        return getattr(mod, attr_name, None) if mod else None

    def wrapper(*args, **kwargs):
        impl = _get_impl()
        if impl is None:
            return fallback_fn(*args, **kwargs)
        return impl(*args, **kwargs)

    return wrapper


# Create lazy wrappers for each function
flashinfer_trtllm_bf16_moe = _lazy_import_wrapper(
    "flashinfer.fused_moe", "trtllm_bf16_moe"
)
flashinfer_trtllm_fp8_block_scale_moe = _lazy_import_wrapper(
    "flashinfer.fused_moe", "trtllm_fp8_block_scale_moe"
)
flashinfer_trtllm_fp8_per_tensor_scale_moe = _lazy_import_wrapper(
    "flashinfer.fused_moe", "trtllm_fp8_per_tensor_scale_moe"
)
flashinfer_cutlass_fused_moe = _lazy_import_wrapper(
    "flashinfer.fused_moe", "cutlass_fused_moe"
)
flashinfer_cutedsl_grouped_gemm_nt_masked = _lazy_import_wrapper(
    "flashinfer.cute_dsl.blockscaled_gemm", "grouped_gemm_nt_masked"
)
flashinfer_fp4_quantize = _lazy_import_wrapper("flashinfer", "fp4_quantize")
nvfp4_batched_quantize = _lazy_import_wrapper("flashinfer", "nvfp4_batched_quantize")
silu_and_mul_scaled_nvfp4_experts_quantize = _lazy_import_wrapper(
    "flashinfer", "silu_and_mul_scaled_nvfp4_experts_quantize"
)
scaled_fp4_grouped_quantize = _lazy_import_wrapper(
    "flashinfer", "scaled_fp4_grouped_quantize"
)
nvfp4_block_scale_interleave = _lazy_import_wrapper(
    "flashinfer.fp4_quantization", "block_scale_interleave"
)
trtllm_fp4_block_scale_moe = _lazy_import_wrapper(
    "flashinfer", "trtllm_fp4_block_scale_moe"
)
# Special case for autotune since it returns a context manager
autotune = _lazy_import_wrapper(
    "flashinfer.autotuner",
    "autotune",
    fallback_fn=lambda *args, **kwargs: contextlib.nullcontext(),
)
_is_fi_autotuning: bool = False


@functools.cache
def has_flashinfer_comm() -> bool:
    """Return `True` if FlashInfer comm module is available."""
    return has_flashinfer() and importlib.util.find_spec("flashinfer.comm") is not None


@functools.cache
def has_flashinfer_nvlink_two_sided() -> bool:
    """Return `True` if FlashInfer mnnvl all2all is available."""
    if not has_flashinfer_comm():
        return False

    # Check if all required functions are available
    required_functions = [
        ("flashinfer.comm", "Mapping"),
        ("flashinfer.comm.mnnvl", "MnnvlMemory"),
        ("flashinfer.comm.trtllm_alltoall", "MnnvlMoe"),
        ("flashinfer.comm.trtllm_alltoall", "MoEAlltoallInfo"),
    ]

    for module_name, attr_name in required_functions:
        mod = _get_submodule(module_name)
        if not mod or not hasattr(mod, attr_name):
            return False
    return True


@functools.cache
def has_flashinfer_nvlink_one_sided() -> bool:
    """Return `True` if FlashInfer trtllm_moe_alltoall module is available."""
    if not has_flashinfer_comm():
        return False
    return importlib.util.find_spec("flashinfer.comm.trtllm_moe_alltoall") is not None


@functools.cache
def has_flashinfer_moe() -> bool:
    """Return `True` if FlashInfer MoE module is available."""
    return (
        has_flashinfer()
        and importlib.util.find_spec("flashinfer.fused_moe") is not None
    )


@functools.cache
def has_flashinfer_cutedsl() -> bool:
    """Return ``True`` if FlashInfer cutedsl module is available."""
    return (
        has_flashinfer() and importlib.util.find_spec("flashinfer.cute_dsl") is not None
    )


@functools.cache
def has_flashinfer_trtllm_fused_moe() -> bool:
    """Return `True` if FlashInfer TRTLLM fused MoE is available."""
    if not has_flashinfer_moe():
        return False
    required_functions = [
        ("flashinfer.fused_moe", "trtllm_fp8_block_scale_moe"),
        ("flashinfer.fused_moe", "trtllm_fp8_per_tensor_scale_moe"),
        ("flashinfer.fused_moe", "trtllm_fp4_block_scale_moe"),
        ("flashinfer.fused_moe", "trtllm_mxint4_block_scale_moe"),
    ]
    for module_name, attr_name in required_functions:
        mod = _get_submodule(module_name)
        if not mod or not hasattr(mod, attr_name):
            return False
    return True


@functools.cache
def has_flashinfer_cutlass_fused_moe() -> bool:
    """Return `True` if FlashInfer CUTLASS fused MoE is available."""
    if not has_flashinfer_moe():
        return False

    # Check if all required functions are available
    required_functions = [
        ("flashinfer.fused_moe", "cutlass_fused_moe"),
        ("flashinfer", "fp4_quantize"),
        ("flashinfer", "nvfp4_block_scale_interleave"),
        ("flashinfer.fused_moe", "trtllm_fp4_block_scale_moe"),
    ]

    for module_name, attr_name in required_functions:
        mod = _get_submodule(module_name)
        if not mod or not hasattr(mod, attr_name):
            return False
    return True


@functools.cache
def has_flashinfer_cutedsl_grouped_gemm_nt_masked() -> bool:
    """Return ``True`` if FlashInfer CUTLASS fused MoE is available."""
    if not has_flashinfer_cutedsl():
        return False

    # Check if all required functions are available
    required_functions = [
        ("flashinfer.cute_dsl.blockscaled_gemm", "grouped_gemm_nt_masked"),
        ("flashinfer", "scaled_fp4_grouped_quantize"),
        ("flashinfer", "silu_and_mul_scaled_nvfp4_experts_quantize"),
    ]

    for module_name, attr_name in required_functions:
        mod = _get_submodule(module_name)
        if not mod or not hasattr(mod, attr_name):
            return False
    return True


@functools.cache
def has_nvidia_artifactory() -> bool:
    """Return `True` if NVIDIA's artifactory is accessible.

    This checks connectivity to the kernel inference library artifactory
    which is required for downloading certain cubin kernels like TRTLLM FHMA.
    """
    # If we have pre-downloaded cubins, we can assume the cubins are available.
    if has_flashinfer_cubin():
        return True

    try:
        # Use a short timeout to avoid blocking for too long
        response = requests.get(FLASHINFER_CUBINS_REPOSITORY, timeout=5)
        accessible = response.status_code == 200
        if accessible:
            logger.debug_once("NVIDIA artifactory is accessible")
        else:
            logger.warning_once(
                "NVIDIA artifactory returned failed status code: %d",
                response.status_code,
            )
        return accessible
    except Exception as e:
        logger.warning_once("Failed to connect to NVIDIA artifactory: %s", e)
        return False


@functools.cache
def supports_trtllm_attention() -> bool:
    """
    TRTLLM attention is supported if the platform is SM100,
    NVIDIA artifactory is accessible, and batch-invariant mode is not enabled.
    """
    # Batch-invariant mode disables TRTLLM attention
    if envs.VLLM_BATCH_INVARIANT:
        return False

    # Requires SM100 and NVIDIA artifactory to be accessible to download cubins
    return (
        current_platform.is_device_capability_family(100) and has_nvidia_artifactory()
    )


def force_use_trtllm_attention() -> bool | None:
    """
    This function should only be called during initialization stage when vllm config
    is set.
    Return `None` if --attention-config.use_trtllm_attention is not set,
    return `True` if TRTLLM attention is forced to be used,
    return `False` if TRTLLM attention is forced to be not used.
    """
    from vllm.config import get_current_vllm_config

    vllm_config = get_current_vllm_config()
    return vllm_config.attention_config.use_trtllm_attention


def can_use_trtllm_attention(num_qo_heads: int, num_kv_heads: int) -> bool:
    """Check if the current configuration supports TRTLLM attention."""
    if force_use_trtllm_attention() is False:
        return False
    has_trtllm = supports_trtllm_attention()
    return has_trtllm and (num_qo_heads % num_kv_heads == 0)


def use_trtllm_attention(
    num_qo_heads: int,
    num_kv_heads: int,
    num_tokens: int,
    max_seq_len: int,
    dcp_world_size: int,
    kv_cache_dtype: str,
    q_dtype: torch.dtype,
    is_prefill: bool,
    # None means auto-detection, True means force on, False means force off
    force_use_trtllm: bool | None = None,
    has_sinks: bool = False,
    has_spec: bool = False,
) -> bool:
    """Return `True` if TRTLLM attention is used."""

    # CLI argument is set to 0 - respect it
    if force_use_trtllm is not None and not force_use_trtllm:
        return False

    # Decode context parallel is not supported
    if dcp_world_size > 1:
        logger.warning_once(
            "Trtllm does not support returning LSE and as a result "
            "does not support DCP, reverting to FlashInfer"
        )
        return False

    # The platform is not supported
    if not supports_trtllm_attention():
        if force_use_trtllm:
            logger.warning_once(
                "TRTLLM attention is not supported on this platform, "
                "but --attention-config.use_trtllm_attention is set to 1"
            )
        return False

    # The combination of query and key heads is not supported
    if num_qo_heads % num_kv_heads != 0:
        if force_use_trtllm:
            logger.warning_once(
                "TRTLLM attention is not supported for this combination of "
                "query and key heads, but --attention-config.use_trtllm_attention is "
                "set to 1"
            )
        return False

    if has_spec and not is_prefill:
        # Speculative decoding requires TRTLLM attention for decodes
        logger.info_once("Using TRTLLM attention (enabled for speculative decoding).")
        return True

    # Must use TRTLLM attention if query is FP8 quantized
    if q_dtype == current_platform.fp8_dtype():
        logger.info_once("Using TRTLLM attention (query is quantized).")
        return True

    # If sinks are being used, we must use TRTLLM attention as it's
    # the only backend that supports them
    if has_sinks:
        logger.info_once("Using TRTLLM attention (required for attention sinks).")
        return True

    if force_use_trtllm is None:
        # CLI argument not set - use auto-detection
        if is_prefill:
            # Prefill auto-detection
            use_trtllm = kv_cache_dtype == "auto"
            if use_trtllm:
                logger.warning_once("Using TRTLLM prefill attention (auto-detected).")
        else:
            # Decode auto-detection
            use_trtllm = num_tokens <= 256 and kv_cache_dtype == "auto"
            if use_trtllm:
                logger.warning_once("Using TRTLLM decode attention (auto-detected).")
        return use_trtllm

    # CLI argument is set to 1 - respect it
    logger.info_once(
        "Using TRTLLM attention (--attention-config.use_trtllm_attention is set to 1)"
    )
    return True


if has_flashinfer():
    from vllm.utils.torch_utils import direct_register_custom_op

    def _flashinfer_concat_mla_k(
        k: torch.Tensor,
        k_nope: torch.Tensor,
        k_pe: torch.Tensor,
    ) -> None:
        """Custom op wrapper for flashinfer's concat_mla_k.

        This is an in-place operation that concatenates k_nope and k_pe into k.

        The kernel is optimized for DeepSeek V3 dimensions:
        - num_heads=128
        - nope_dim=128
        - rope_dim=64

        Key optimizations:
        - Warp-based processing with software pipelining
        - Vectorized memory access (int2 for nope, int for rope)
        - L2 prefetching for next row while processing current
        - Register reuse for rope values across all heads

        Args:
            k: Output tensor, shape [num_tokens, num_heads, nope_dim + rope_dim].
                Modified in-place.
            k_nope: The nope part of k, shape [num_tokens, num_heads, nope_dim].
            k_pe: The rope part of k (shared), shape [num_tokens, 1, rope_dim].
                  This is broadcast to all heads.
        """
        from flashinfer.concat_ops import concat_mla_k

        concat_mla_k(k, k_nope, k_pe)

    def _flashinfer_concat_mla_k_fake(
        k: torch.Tensor,
        k_nope: torch.Tensor,
        k_pe: torch.Tensor,
    ) -> None:
        return

    # Register flashinfer concat_mla_k custom op
    direct_register_custom_op(
        op_name="flashinfer_concat_mla_k",
        op_func=_flashinfer_concat_mla_k,
        mutates_args=["k"],  # k tensor is modified in-place
        fake_impl=_flashinfer_concat_mla_k_fake,
    )

    @torch.library.custom_op(
        "vllm::flashinfer_mm_fp4",
        mutates_args=[],
        device_types="cuda",
    )
    def flashinfer_mm_fp4(
        A: torch.Tensor,
        B: torch.Tensor,
        A_scale: torch.Tensor,
        B_scale: torch.Tensor,
        g_scale: torch.Tensor,
        dtype: torch.dtype,
        use_8x4_sf_layout: bool,
        backend: str,
    ) -> torch.Tensor:
        from flashinfer import mm_fp4 as flashinfer_mm_fp4_

        return flashinfer_mm_fp4_(
            A,
            B,
            A_scale,
            B_scale,
            g_scale,
            dtype,
            block_size=16,
            use_8x4_sf_layout=use_8x4_sf_layout,
            backend=backend,
        )

    @torch.library.register_fake(
        "vllm::flashinfer_mm_fp4",
    )
    def flashinfer_mm_fp4_fake(
        A: torch.Tensor,
        B: torch.Tensor,
        A_scale: torch.Tensor,
        B_scale: torch.Tensor,
        g_scale: torch.Tensor,
        dtype: torch.dtype,
        use_8x4_sf_layout: bool,
        backend: str,
    ) -> torch.Tensor:
        return torch.empty(A.shape[0], B.shape[1], dtype=dtype, device=A.device)

    @torch.library.custom_op(
        "vllm::bmm_fp8",
        mutates_args=[],
        device_types="cuda",
    )
    def bmm_fp8(
        A: torch.Tensor,
        B: torch.Tensor,
        A_scale: torch.Tensor,
        B_scale: torch.Tensor,
        dtype: torch.dtype,
        backend: str,
    ) -> torch.Tensor:
        from flashinfer import bmm_fp8 as bmm_fp8_

        return bmm_fp8_(A, B, A_scale, B_scale, dtype, None, backend)

    @torch.library.register_fake(
        "vllm::bmm_fp8",
    )
    def bmm_fp8_fake(
        A: torch.Tensor,
        B: torch.Tensor,
        A_scale: torch.Tensor,
        B_scale: torch.Tensor,
        dtype: torch.dtype,
        backend: str,
    ) -> torch.Tensor:
        return torch.empty(
            A.shape[0], A.shape[1], B.shape[2], dtype=dtype, device=A.device
        )

    @torch.library.custom_op(
        "vllm::flashinfer_nvfp4_quantize",
        mutates_args=[],
        device_types="cuda",
    )
    def flashinfer_nvfp4_quantize(
        a: torch.Tensor, a_global_sf: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from flashinfer import SfLayout
        from flashinfer import nvfp4_quantize as nvfp4_quantize_

        return nvfp4_quantize_(
            a, a_global_sf, sfLayout=SfLayout.layout_8x4, do_shuffle=False
        )

    @torch.library.register_fake(
        "vllm::flashinfer_nvfp4_quantize",
    )
    def flashinfer_nvfp4_quantize_fake(
        a: torch.Tensor, a_global_sf: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        m, n = a.shape

        round_up = lambda x, y: (x + y - 1) // y * y

        rounded_m = round_up(m, 8)
        scale_n = n // 16
        rounded_n = round_up(scale_n, 4)

        return torch.empty(m, n // 2, dtype=torch.uint8, device=a.device), torch.empty(
            rounded_m, rounded_n, dtype=torch.uint8, device=a.device
        )

    @torch.library.custom_op(
        "vllm::mm_mxfp8",
        mutates_args=[],
        device_types="cuda",
    )
    def mm_mxfp8(
        A: torch.Tensor,
        B: torch.Tensor,
        A_scale: torch.Tensor,
        B_scale: torch.Tensor,
        out_dtype: torch.dtype,
        backend: str = "cutlass",
    ) -> torch.Tensor:
        from flashinfer import mm_mxfp8 as mm_mxfp8_

        use_direct_trtllm = (
            backend == "trtllm"
            and os.environ.get("VLLM_MXFP8_DENSE_DIRECT_TRTLLM", "0")
            .strip()
            .lower()
            not in ("0", "false", "no", "off", "")
        )
        if use_direct_trtllm:
            require_direct = (
                os.environ.get("VLLM_MXFP8_DENSE_REQUIRE_DIRECT_TRTLLM", "0")
                .strip()
                .lower()
                not in ("0", "false", "no", "off", "")
            )
            try:
                from flashinfer.gemm.gemm_base import (  # type: ignore
                    DEFAULT_WORKSPACE_SIZE,
                    _get_cache_buf,
                    get_trtllm_gemm_module,
                )

                use_8x4 = (
                    os.environ.get("VLLM_MXFP8_DENSE_A_SF_LAYOUT", "128x4")
                    .strip()
                    .lower()
                    in ("8x4", "layout_8x4", "true", "1")
                )
                out = torch.empty(
                    (A.shape[0], B.shape[1]), dtype=out_dtype, device=A.device
                )
                workspace = _get_cache_buf(
                    "vllm_direct_trtllm_mxfp8_workspace",
                    DEFAULT_WORKSPACE_SIZE,
                    A.device,
                )
                runner = get_trtllm_gemm_module().trtllm_mxfp8_gemm_runner(
                    use_8x4_sf_layout=use_8x4
                )
                tactic_raw = os.environ.get(
                    "VLLM_MXFP8_DENSE_TRTLLM_TACTIC", "-1"
                )
                tactic_hints_raw = os.environ.get(
                    "VLLM_MXFP8_DENSE_TRTLLM_TACTIC_HINTS", ""
                )
                tactic_policy = os.environ.get(
                    "VLLM_MXFP8_DENSE_TRTLLM_TACTIC_POLICY", ""
                )
                tactic_raw_norm = tactic_raw.strip().lower()
                tactic_policy = tactic_policy.strip().lower()
                use_shape_cache = tactic_raw_norm in (
                    "auto",
                    "auto-best",
                    "shape-cache",
                    "shape_cache",
                ) or tactic_policy in (
                    "auto",
                    "auto-best",
                    "shape-cache",
                    "shape_cache",
                )

                def _fallback_tactic() -> int:
                    try:
                        return int(tactic_raw)
                    except ValueError:
                        return -1

                tactic = _fallback_tactic()

                def _shape_hint_tactic() -> int | None:
                    raw = tactic_hints_raw.strip()
                    if not raw:
                        return None
                    shape_key = (
                        int(A.shape[0]),
                        int(B.shape[1]),
                        int(A.shape[1]),
                    )
                    cached_raw = getattr(
                        mm_mxfp8, "_trtllm_tactic_hints_raw", None
                    )
                    cached_map = getattr(
                        mm_mxfp8, "_trtllm_tactic_hints_map", None
                    )
                    if cached_raw == raw and isinstance(cached_map, dict):
                        return cached_map.get(shape_key)

                    hint_map: dict[tuple[int, int, int], int] = {}
                    for item in raw.split(";"):
                        item = item.strip()
                        if not item or ":" not in item:
                            continue
                        shape, hint = item.rsplit(":", 1)
                        try:
                            m, n, k = (
                                int(part.strip())
                                for part in shape.split(",", 2)
                            )
                            hint_tactic = int(hint.strip())
                        except ValueError:
                            continue
                        hint_map[(m, n, k)] = hint_tactic
                    mm_mxfp8._trtllm_tactic_hints_raw = raw
                    mm_mxfp8._trtllm_tactic_hints_map = hint_map
                    return hint_map.get(shape_key)

                def _trace_shape(tactic_value: int, tactic_source: str) -> None:
                    trace_flag = (
                        os.environ.get("VLLM_MXFP8_DENSE_SHAPE_TRACE", "0")
                        .strip()
                        .lower()
                    )
                    if trace_flag in ("0", "false", "no", "off", ""):
                        return
                    trace_dir = os.environ.get(
                        "VLLM_MXFP8_DENSE_SHAPE_TRACE_DIR", ""
                    ).strip()
                    if not trace_dir:
                        return
                    shape_key = (
                        int(A.shape[0]),
                        int(B.shape[1]),
                        int(A.shape[1]),
                        str(out_dtype),
                        bool(use_8x4),
                    )
                    seen = getattr(mm_mxfp8, "_trtllm_shape_trace_seen", None)
                    if seen is None:
                        seen = set()
                        mm_mxfp8._trtllm_shape_trace_seen = seen
                    if shape_key in seen:
                        return
                    max_shapes = int(
                        os.environ.get(
                            "VLLM_MXFP8_DENSE_SHAPE_TRACE_MAX", "4096"
                        )
                    )
                    if len(seen) >= max_shapes:
                        return
                    seen.add(shape_key)

                    try:
                        import json
                        import socket
                        import time

                        os.makedirs(trace_dir, exist_ok=True)
                        rank = (
                            os.environ.get("RANK")
                            or os.environ.get("LOCAL_RANK")
                            or os.environ.get("SLURM_PROCID")
                            or "na"
                        )
                        path = os.path.join(
                            trace_dir,
                            f"mxfp8_dense_shapes_rank{rank}_pid{os.getpid()}.jsonl",
                        )
                        record = {
                            "event": "mxfp8_dense_shape",
                            "time": time.time(),
                            "pid": os.getpid(),
                            "hostname": socket.gethostname(),
                            "rank": rank,
                            "m": int(A.shape[0]),
                            "n": int(B.shape[1]),
                            "k": int(A.shape[1]),
                            "a_shape": [int(x) for x in A.shape],
                            "b_shape": [int(x) for x in B.shape],
                            "a_dtype": str(A.dtype),
                            "b_dtype": str(B.dtype),
                            "out_dtype": str(out_dtype),
                            "use_8x4_sf_layout": bool(use_8x4),
                            "tactic": int(tactic_value),
                            "tactic_source": tactic_source,
                        }
                        with open(path, "a", encoding="utf-8") as f:
                            f.write(json.dumps(record, sort_keys=True) + "\n")
                    except Exception:
                        return

                hint_tactic = _shape_hint_tactic()
                tactic_source = "env"
                if hint_tactic is not None:
                    tactic = hint_tactic
                    tactic_source = "hint"
                    use_shape_cache = False

                if use_shape_cache:
                    class _ShapeProfile:
                        def __init__(
                            self,
                            a_shape: tuple[int, ...],
                            b_shape: tuple[int, ...],
                        ) -> None:
                            self._shapes = [a_shape, b_shape]

                        def get_opt_shapes(self) -> list[tuple[int, ...]]:
                            return self._shapes

                    cache = getattr(mm_mxfp8, "_trtllm_tactic_cache", None)
                    if cache is None:
                        cache = {}
                        mm_mxfp8._trtllm_tactic_cache = cache

                    cache_key = (
                        int(A.shape[0]),
                        int(A.shape[1]),
                        int(B.shape[1]),
                        str(A.dtype),
                        str(B.dtype),
                        str(out_dtype),
                        bool(use_8x4),
                    )
                    tactic_is_cached = cache_key in cache
                    tactic = cache.get(cache_key, -1)
                    tactic_source = (
                        "auto_cache" if tactic_is_cached else "auto_default"
                    )
                    is_capturing = getattr(
                        torch.cuda, "is_current_stream_capturing", lambda: False
                    )
                    if not tactic_is_cached and not is_capturing():
                        inputs = [A, B, A_scale, B_scale, out_dtype, out, workspace]
                        try:
                            valid = list(
                                runner.get_valid_tactics(
                                    inputs,
                                    _ShapeProfile(tuple(A.shape), tuple(B.shape)),
                                )
                            )
                        except Exception:
                            valid = []
                        candidates = [-1] + [x for x in valid if x != -1]
                        warmup = int(os.environ.get(
                            "VLLM_MXFP8_DENSE_TRTLLM_TACTIC_WARMUP", "2"
                        ))
                        iters = int(os.environ.get(
                            "VLLM_MXFP8_DENSE_TRTLLM_TACTIC_ITERS", "8"
                        ))
                        trials: list[tuple[int, float]] = []
                        for candidate in candidates:
                            try:
                                for _ in range(warmup):
                                    runner.forward(inputs, tactic=candidate)
                                torch.cuda.synchronize()
                                start = torch.cuda.Event(enable_timing=True)
                                end = torch.cuda.Event(enable_timing=True)
                                start.record()
                                for _ in range(iters):
                                    runner.forward(inputs, tactic=candidate)
                                end.record()
                                torch.cuda.synchronize()
                                trials.append((
                                    int(candidate),
                                    float(start.elapsed_time(end)) / max(1, iters),
                                ))
                            except Exception:
                                continue
                        if trials:
                            tactic = min(trials, key=lambda item: item[1])[0]
                            cache[cache_key] = tactic
                            tactic_source = "auto_tuned"
                    elif not tactic_is_cached:
                        tactic_source = "auto_capture_default"

                _trace_shape(tactic, tactic_source)
                return runner.forward(
                    [A, B, A_scale, B_scale, out_dtype, out, workspace],
                    tactic=tactic,
                )
            except Exception:
                if require_direct:
                    raise

        return mm_mxfp8_(
            A,
            B,
            A_scale,
            B_scale,
            out=None,
            out_dtype=out_dtype,
            backend=backend,
        )

    @torch.library.register_fake(
        "vllm::mm_mxfp8",
    )
    def mm_mxfp8_fake(
        A: torch.Tensor,
        B: torch.Tensor,
        A_scale: torch.Tensor,
        B_scale: torch.Tensor,
        out_dtype: torch.dtype,
        backend: str = "cutlass",
    ) -> torch.Tensor:
        # A is [m, k], B is [k, n] -> output [m, n]
        return torch.empty(A.shape[0], B.shape[1], dtype=out_dtype, device=A.device)


def flashinfer_mm_mxfp8(
    a: torch.Tensor,
    b: torch.Tensor,
    block_scale_a: torch.Tensor,
    block_scale_b: torch.Tensor,
    out_dtype: torch.dtype,
    backend: str = "cutlass",
) -> torch.Tensor:
    """MXFP8 MM helper - mirrors flashinfer_scaled_fp4_mm API.

    Takes non-transposed weights and handles transpose internally.

    CRITICAL: mm_mxfp8 CUTLASS kernel requires SWIZZLED 1D scales for optimal
    performance and accuracy. Both input and weight scales should be in
    swizzled format from FlashInfer's mxfp8_quantize(is_sf_swizzled_layout=True).
    """
    # a shape [M, K]
    # b shape [K, N]
    assert a.ndim == 2 and b.ndim == 2
    assert a.shape[1] == b.shape[1]  # K dimension must match

    if block_scale_b.ndim != 1:
        raise ValueError(
            "mm_mxfp8 expects 1D swizzled weight scales for CUTLASS; "
            f"got shape={tuple(block_scale_b.shape)}"
        )

    # Output tensor [M, N]
    return mm_mxfp8(
        a,
        b.t(),  # Transpose weight: [N, K] -> [K, N]
        block_scale_a,
        block_scale_b,
        out_dtype,
        backend=backend,
    )


def flashinfer_scaled_fp4_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    block_scale_a: torch.Tensor,
    block_scale_b: torch.Tensor,
    alpha: torch.Tensor,
    out_dtype: torch.dtype,
    backend: str,
) -> torch.Tensor:
    assert a.ndim == 2 and b.ndim == 2
    assert block_scale_a.ndim == 2 and block_scale_b.ndim == 2
    assert a.stride(-1) == 1 and b.stride(-1) == 1
    assert a.shape[1] == b.shape[1]

    if backend in ("cutlass", "cudnn"):
        block_scale_a = block_scale_a.view(torch.uint8)
        block_scale_b = block_scale_b.view(torch.uint8)

    use_8x4_sf_layout = True if backend == "trtllm" and a.shape[0] <= 32 else False  # noqa: SIM210

    return flashinfer_mm_fp4(
        a,
        b.t(),
        block_scale_a,
        block_scale_b.t(),
        alpha,
        out_dtype,
        use_8x4_sf_layout=use_8x4_sf_layout,
        backend=backend,
    )


def flashinfer_scaled_fp8_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    assert a.ndim == 2 and b.ndim == 2
    assert a.shape[1] == b.shape[0]
    assert scale_a.numel() == 1 and scale_b.numel() == 1
    assert a.dtype == torch.float8_e4m3fn and b.dtype == torch.float8_e4m3fn
    assert a.device.type == "cuda" and b.device.type == "cuda"
    assert scale_a.dtype == torch.float32 and scale_b.dtype == torch.float32
    assert scale_a.device.type == "cuda" and scale_b.device.type == "cuda"

    output = bmm_fp8(
        a.unsqueeze(0),
        b.unsqueeze(0),
        scale_a,
        scale_b,
        out_dtype,
        "auto",
    ).view(a.shape[0], b.shape[1])

    if bias is not None:
        output = output + bias
    return output


def flashinfer_quant_nvfp4_8x4_sf_layout(
    a: torch.Tensor, a_global_sf: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    return flashinfer_nvfp4_quantize(a, a_global_sf)


flashinfer_fp8_blockscale_gemm = _lazy_import_wrapper(
    "flashinfer.gemm", "fp8_blockscale_gemm_sm90"
)


@functools.cache
def has_flashinfer_fp8_blockscale_gemm() -> bool:
    """Return `True` if FlashInfer block-scale FP8 GEMM is available."""
    return (
        has_flashinfer()
        and current_platform.is_device_capability(90)
        and hasattr(_get_submodule("flashinfer.gemm"), "fp8_blockscale_gemm_sm90")
    )


@functools.cache
def is_flashinfer_fp8_blockscale_gemm_supported() -> bool:
    """Return `True` if FlashInfer block-scale FP8 GEMM is supported."""
    return (
        envs.VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER
        and has_flashinfer_fp8_blockscale_gemm()
    )


def should_use_flashinfer_for_blockscale_fp8_gemm(
    is_flashinfer_supported: bool,
    output_dtype: torch.dtype,
    input: torch.Tensor,
    weight: torch.Tensor,
):
    if not is_flashinfer_supported:
        return False

    # Verify DeepGEMM N/K dims requirements
    # NOTE: Also synchronized with test_w8a8_block_fp8_deep_gemm_matmul
    # test inside kernels/quantization/test_block_fp8.py
    N_MULTIPLE = 64
    K_MULTIPLE = 128

    weight_dtype = weight.dtype
    input_dtype = input.dtype

    should_use_flashinfer = (
        output_dtype == torch.bfloat16
        and input_dtype == torch.bfloat16
        and weight_dtype == torch.float8_e4m3fn
        and weight.shape[0] % N_MULTIPLE == 0
        and weight.shape[1] % K_MULTIPLE == 0
    )

    return should_use_flashinfer


__all__ = [
    "has_flashinfer",
    "flashinfer_trtllm_fp8_block_scale_moe",
    "flashinfer_cutlass_fused_moe",
    "flashinfer_cutedsl_grouped_gemm_nt_masked",
    "flashinfer_fp4_quantize",
    "silu_and_mul_scaled_nvfp4_experts_quantize",
    "scaled_fp4_grouped_quantize",
    "nvfp4_block_scale_interleave",
    "trtllm_fp4_block_scale_moe",
    "autotune",
    "has_flashinfer_moe",
    "has_flashinfer_comm",
    "has_flashinfer_nvlink_two_sided",
    "has_flashinfer_nvlink_one_sided",
    "has_flashinfer_cutlass_fused_moe",
    "has_flashinfer_cutedsl_grouped_gemm_nt_masked",
    "has_flashinfer_fp8_blockscale_gemm",
    "has_nvidia_artifactory",
    "supports_trtllm_attention",
    "can_use_trtllm_attention",
    "use_trtllm_attention",
    "flashinfer_scaled_fp4_mm",
    "flashinfer_scaled_fp8_mm",
    "flashinfer_quant_nvfp4_8x4_sf_layout",
    "flashinfer_fp8_blockscale_gemm",
    "should_use_flashinfer_for_blockscale_fp8_gemm",
    "is_flashinfer_fp8_blockscale_gemm_supported",
]
