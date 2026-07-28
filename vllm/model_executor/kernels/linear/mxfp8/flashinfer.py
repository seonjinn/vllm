# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
import socket
import time
from pathlib import Path

import torch
from flashinfer import shuffle_matrix_a, shuffle_matrix_sf_a
from torch.nn.parameter import Parameter

from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    MXFP8_BLOCK_SIZE,
    _mxfp8_dense_a_sf_layout,
    mxfp8_dense_use_8x4_sf_layout,
    mxfp8_e4m3_quantize,
    swizzle_mxfp8_scale,
)
from vllm.platforms import current_platform
from vllm.utils import flashinfer as vllm_flashinfer

from .Mxfp8LinearKernel import Mxfp8LinearKernel, Mxfp8LinearLayerConfig


_SUPPORTED_MXFP8_DENSE_BACKENDS = ("cutlass", "trtllm", "cute-dsl", "auto")
_MXFP8_DENSE_TRACE_SEEN: set[tuple[object, ...]] = set()
_MXFP8_DENSE_TRACE_WRITTEN = 0


def _mxfp8_dense_is_compiling() -> bool:
    for namespace_name in ("compiler", "_dynamo"):
        namespace = getattr(torch, namespace_name, None)
        is_compiling = getattr(namespace, "is_compiling", None)
        try:
            if is_compiling is not None and is_compiling():
                return True
        except Exception:
            pass
    return False


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off", "")


def _mxfp8_dense_family(layer: torch.nn.Module) -> str:
    prefix = str(getattr(layer, "prefix", "")).lower()
    if any(tok in prefix for tok in ("qkv_proj", "query_key_value")):
        return "QKV"
    if any(tok in prefix for tok in ("q_proj", "k_proj", "v_proj")):
        return "QKV"
    if any(tok in prefix for tok in ("o_proj", "out_proj", "attention.dense")):
        return "O"
    if any(tok in prefix for tok in ("gate_up_proj", "gate_proj", "up_proj")):
        return "FC1"
    if any(tok in prefix for tok in ("fc1", ".w1", ".w3")):
        return "FC1"
    if any(tok in prefix for tok in ("down_proj", "fc2", ".w2")):
        return "FC2"
    if "mamba" in prefix and any(
        tok in prefix
        for tok in ("proj", "linear", "in_proj", "out_proj", "x_proj", "dt_proj")
    ):
        return "MambaProjection"
    if any(tok in prefix for tok in ("mlp", "ffn", "expert")):
        return "MLPOrExpertDense"
    return "OtherDense"


def _mxfp8_dense_shape_trace(
    *,
    layer: torch.nn.Module,
    family: str,
    m_logical: int,
    m_physical: int,
    n_logical: int,
    n_physical: int,
    k: int,
    backend: str,
    input_shape: torch.Size,
    weight_shape: torch.Size,
) -> None:
    if not _env_flag("VLLM_MXFP8_DENSE_SHAPE_TRACE", False):
        return
    if _mxfp8_dense_is_compiling():
        return
    if torch.cuda.is_current_stream_capturing():
        return
    trace_dir = os.environ.get("VLLM_MXFP8_DENSE_SHAPE_TRACE_DIR", "").strip()
    if not trace_dir:
        return

    prefix = str(getattr(layer, "prefix", "unknown"))
    key = (
        family,
        prefix,
        m_logical,
        m_physical,
        n_logical,
        n_physical,
        k,
        backend,
    )
    max_records = int(os.environ.get("VLLM_MXFP8_DENSE_SHAPE_TRACE_MAX", "4096"))

    global _MXFP8_DENSE_TRACE_WRITTEN
    if key in _MXFP8_DENSE_TRACE_SEEN:
        return
    if _MXFP8_DENSE_TRACE_WRITTEN >= max_records:
        return
    _MXFP8_DENSE_TRACE_SEEN.add(key)
    _MXFP8_DENSE_TRACE_WRITTEN += 1

    path = Path(trace_dir)
    path.mkdir(parents=True, exist_ok=True)
    output = path / f"dense_shapes_{socket.gethostname()}_{os.getpid()}.jsonl"
    record = {
        "event": "mxfp8_dense_shape",
        "time": time.time(),
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "family": family,
        "prefix": prefix,
        "layer_class": layer.__class__.__name__,
        "m_logical": int(m_logical),
        "m_physical": int(m_physical),
        "n_logical": int(n_logical),
        "n_physical": int(n_physical),
        "k": int(k),
        "backend": backend,
        "input_shape": list(input_shape),
        "weight_shape": list(weight_shape),
    }
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _mxfp8_dense_nvtx_push(message: str) -> bool:
    if not _env_flag("VLLM_MXFP8_DENSE_NVTX", False):
        return False
    try:
        torch.cuda.nvtx.range_push(message)
        return True
    except Exception:
        return False


def _mxfp8_dense_nvtx_pop(enabled: bool) -> None:
    if not enabled:
        return
    try:
        torch.cuda.nvtx.range_pop()
    except Exception:
        pass


def _mxfp8_dense_backend() -> str:
    backend = os.environ.get("VLLM_MXFP8_DENSE_GEMM_BACKEND", "cutlass")
    backend = backend.strip().lower()
    if backend not in _SUPPORTED_MXFP8_DENSE_BACKENDS:
        raise ValueError(
            "VLLM_MXFP8_DENSE_GEMM_BACKEND must be one of "
            f"{_SUPPORTED_MXFP8_DENSE_BACKENDS}, got {backend!r}"
        )
    return backend


class FlashInferCutlassMxfp8LinearKernel(Mxfp8LinearKernel):
    """MXFP8 W8A8 GEMM via FlashInfer CUTLASS (SM100+)."""

    def __init__(self, c: Mxfp8LinearLayerConfig) -> None:
        super().__init__(c)
        layout_mode = _mxfp8_dense_a_sf_layout()
        if layout_mode in ("adaptive", "shape-aware", "shape_aware"):
            if _mxfp8_dense_backend() != "trtllm":
                raise RuntimeError("MXFP8 adaptive layout requires the TRTLLM backend")
            vllm_flashinfer.configure_mxfp8_adaptive_layout_compilation()

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if current_platform.has_device_capability(100):
            return True, None
        return False, "requires >=sm_100 (Blackwell)"

    @classmethod
    def can_implement(cls, c: Mxfp8LinearLayerConfig) -> tuple[bool, str | None]:
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight = layer.weight.data  # [N, K]
        N, K = weight.shape

        scale_k = K // MXFP8_BLOCK_SIZE
        weight_scale_2d = layer.weight_scale.data[:N, :scale_k].contiguous()
        backend = _mxfp8_dense_backend()
        layout_mode = _mxfp8_dense_a_sf_layout()
        is_adaptive_layout = layout_mode in (
            "adaptive",
            "shape-aware",
            "shape_aware",
        )
        if is_adaptive_layout and backend != "trtllm":
            raise RuntimeError("MXFP8 adaptive layout requires the TRTLLM backend")
        layer._mxfp8_dense_backend_for_apply = backend

        if backend == "trtllm" and (K % 256 != 0 or (K // MXFP8_BLOCK_SIZE) % 4 != 0):
            raise ValueError(
                "TRTLLM MXFP8 dense weight requires K % 256 == 0 and "
                "(K / 32) % 4 == 0; "
                f"got N={N}, K={K}"
            )

        # iter8 nemo-speed (opt-in via MXFP8_BF16_FALLBACK_SMALL_M=1):
        # cache a BF16-dequantized copy of the weight for use at small M
        # where mm_mxfp8 has to pad the input up to 128 rows and waste 75%
        # of GEMM compute. Doubles linear-weight memory but unlocks the
        # mid-concurrency regime where harness configs ab_mid/ab_decode_heavy/
        # swe_192k_512 spend most of their time.
        if (
            not is_adaptive_layout
            and os.environ.get("MXFP8_BF16_FALLBACK_SMALL_M") == "1"
        ):
            # Dequantize:  bf16 = fp8.to(bf16) * 2^(scale_biased - 127)
            # weight_scale_2d is e8m0 biased exponent stored in uint8.
            descale = torch.exp2(weight_scale_2d.to(torch.float32) - 127.0).to(
                torch.bfloat16
            )  # [N, K/32]
            w_bf16 = weight.to(torch.bfloat16).view(N, scale_k, MXFP8_BLOCK_SIZE)
            w_bf16 = w_bf16 * descale.unsqueeze(-1)
            w_bf16 = w_bf16.view(N, K).contiguous()
            if hasattr(layer, "weight_bf16"):
                layer.weight_bf16.data.copy_(w_bf16)
            else:
                layer.weight_bf16 = Parameter(w_bf16, requires_grad=False)

        if backend == "trtllm":
            N_padded = ((N + 127) // 128) * 128
            layer._mxfp8_dense_output_features = N
            if N_padded == N:
                weight_trtllm = shuffle_matrix_a(weight, 128).reshape(N, K)
                weight.copy_(weight_trtllm)
                del weight_trtllm
            else:
                weight_padded = torch.zeros(
                    (N_padded, K), dtype=weight.dtype, device=weight.device
                )
                weight_padded[:N, :].copy_(weight)
                weight_trtllm = shuffle_matrix_a(weight_padded, 128).reshape(
                    N_padded, K
                )
                layer.weight = Parameter(weight_trtllm, requires_grad=False)
                del weight_padded
            if N_padded != N:
                scale_padded = torch.zeros(
                    (N_padded, scale_k),
                    dtype=weight_scale_2d.dtype,
                    device=weight_scale_2d.device,
                )
                scale_padded[:N, :].copy_(weight_scale_2d)
            else:
                scale_padded = weight_scale_2d
            weight_scale_for_apply = (
                shuffle_matrix_sf_a(scale_padded, 128, num_elts_per_sf=32)
                .reshape(-1)
                .contiguous()
            )
            if N_padded != N:
                del scale_padded
        else:
            weight_scale_for_apply = swizzle_mxfp8_scale(
                weight_scale_2d, M=N, K=K
            ).contiguous()

        if (
            hasattr(layer, "weight_scale_for_apply")
            and layer.weight_scale_for_apply.shape == weight_scale_for_apply.shape
        ):
            layer.weight_scale_for_apply.data.copy_(weight_scale_for_apply)
        else:
            layer.weight_scale_for_apply = Parameter(
                weight_scale_for_apply, requires_grad=False
            )

        if is_adaptive_layout:
            direct_state = vllm_flashinfer.prepare_mxfp8_trtllm_direct_state(
                weight.device
            )
            layer._mxfp8_trtllm_configuration = direct_state.configuration
            (
                layer._mxfp8_trtllm_workspace_8x4,
                layer._mxfp8_trtllm_workspace_128x4,
            ) = vllm_flashinfer.get_mxfp8_trtllm_prepared_workspaces(weight.device)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        weight = layer.weight
        weight_scale = layer.weight_scale_for_apply
        prepared_configuration = getattr(layer, "_mxfp8_trtllm_configuration", None)
        if prepared_configuration is not None:
            vllm_flashinfer.validate_mxfp8_trtllm_configuration(prepared_configuration)
        backend = _mxfp8_dense_backend()
        loaded_backend = layer._mxfp8_dense_backend_for_apply
        if backend != loaded_backend:
            raise RuntimeError(
                "MXFP8 dense backend changed after weight preparation: "
                f"loaded with {loaded_backend!r}, applying with {backend!r}"
            )
        out_dtype = x.dtype
        N, K = weight.shape
        output_features = getattr(layer, "_mxfp8_dense_output_features", N)

        input_shape = x.shape
        input_2d = x.view(-1, K)
        M_orig = input_2d.shape[0]
        layout_mode = _mxfp8_dense_a_sf_layout()
        is_adaptive_layout = layout_mode in (
            "adaptive",
            "shape-aware",
            "shape_aware",
        )
        if is_adaptive_layout and backend != "trtllm":
            raise RuntimeError("MXFP8 adaptive layout requires the TRTLLM backend")

        # iter8 nemo-speed: BF16 fallback for small M. mm_mxfp8 pads M up to
        # 128 and processes a full 128-row tile regardless, wasting compute
        # at M < 128 (smoke = 32, ab_mid/ab_decode/swe = 64). The cached
        # weight_bf16 lets us do a plain bf16 matmul instead. Only enabled
        # if MXFP8_BF16_FALLBACK_SMALL_M=1 was set at startup so
        # process_weights_after_loading allocated weight_bf16.
        if not is_adaptive_layout and M_orig < 128 and hasattr(layer, "weight_bf16"):
            input_bf16 = input_2d.to(torch.bfloat16)
            output = torch.matmul(input_bf16, layer.weight_bf16.t())
            if bias is not None:
                output = output + bias
            return output.view(*input_shape[:-1], output_features).to(out_dtype)

        min_dim = 128

        # The original CUTLASS path pads low-M inputs to 128 rows before
        # GEMM. Keep that default for correctness and only let opt-in smoke
        # runs test TRTLLM/CuTe-DSL on the real low-M shape.
        pad_to_128 = _env_flag("VLLM_MXFP8_DENSE_PAD_TO_128", True)
        if backend in ("cutlass", "auto"):
            pad_to_128 = True

        assert min_dim <= K, (
            f"mm_mxfp8 requires K >= {min_dim}, got K={K}. "
            f"in_features is too small for mm_mxfp8."
        )
        assert K % MXFP8_BLOCK_SIZE == 0, (
            f"mm_mxfp8 requires K to be divisible by {MXFP8_BLOCK_SIZE}, got K={K}."
        )
        assert min_dim <= N, (
            f"mm_mxfp8 requires N >= {min_dim}, got N={N}. "
            f"out_features is too small for mm_mxfp8."
        )

        if pad_to_128:
            M_padded = ((M_orig + min_dim - 1) // min_dim) * min_dim
        else:
            M_padded = M_orig
        pad_rows = M_padded - M_orig
        if pad_rows > 0:
            input_2d = torch.nn.functional.pad(input_2d, (0, 0, 0, pad_rows))

        _mxfp8_dense_shape_trace(
            layer=layer,
            family=_mxfp8_dense_family(layer),
            m_logical=M_orig,
            m_physical=M_padded,
            n_logical=output_features,
            n_physical=N,
            k=K,
            backend=backend,
            input_shape=input_shape,
            weight_shape=weight.shape,
        )

        if not weight.is_contiguous():
            weight = weight.contiguous()

        is_compiling = _mxfp8_dense_is_compiling()
        use_8x4_sf_layout = None

        sync_nvtx = False
        nvtx_enabled = False
        if (
            _env_flag("VLLM_MXFP8_DENSE_NVTX", False)
            and not torch.compiler.is_compiling()
        ):
            marker = (
                f"mxfp8_dense::{_mxfp8_dense_family(layer)}::"
                f"{str(getattr(layer, 'prefix', 'unknown')).replace('::', '/')}"
                f"::M={M_orig},N={N},K={K}"
                f"::M_padded={M_padded}"
                f"::backend={backend}"
                "::tactic=delegated_to_mm_mxfp8"
                f"::pad_to_128={int(pad_to_128)}"
                f"::layer={layer.__class__.__name__}"
            )
            sync_nvtx = _env_flag("VLLM_MXFP8_DENSE_NVTX_SYNC", False)
            nvtx_enabled = _mxfp8_dense_nvtx_push(marker)
        if sync_nvtx:
            torch.cuda.synchronize(input_2d.device)
        try:
            if is_adaptive_layout and is_compiling:
                output = torch.ops.vllm.mxfp8_adaptive_quantize_mm_marker(
                    input_2d,
                    weight.t(),
                    weight_scale,
                    out_dtype,
                    backend,
                    layer._mxfp8_trtllm_workspace_8x4,
                    layer._mxfp8_trtllm_workspace_128x4,
                )
            else:
                use_8x4_sf_layout = mxfp8_dense_use_8x4_sf_layout(M_padded)
                input_mxfp8, input_scale = mxfp8_e4m3_quantize(
                    input_2d,
                    is_sf_swizzled_layout=True,
                    use_8x4_sf_layout=use_8x4_sf_layout,
                )
                output = vllm_flashinfer.mm_mxfp8(
                    input_mxfp8,
                    weight.t(),
                    input_scale,
                    weight_scale,
                    out_dtype=out_dtype,
                    backend=backend,
                    use_8x4_sf_layout=use_8x4_sf_layout,
                )
            if sync_nvtx:
                torch.cuda.synchronize(input_2d.device)
        finally:
            _mxfp8_dense_nvtx_pop(nvtx_enabled)

        if pad_rows > 0:
            output = output[:M_orig, :]
        if output_features != N:
            output = output[:, :output_features]

        if bias is not None:
            output = output + bias

        output_shape = (*input_shape[:-1], output_features)
        return output.view(output_shape)
