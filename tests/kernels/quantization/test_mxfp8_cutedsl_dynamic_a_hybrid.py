# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.nn.parameter import Parameter

from vllm.model_executor.kernels.linear.mxfp8 import flashinfer as hybrid
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    MXFP8_BLOCK_SIZE,
)
from vllm.utils import flashinfer as vllm_flashinfer

_RUNTIME = {
    "gpu_sm": "sm_100",
    "flashinfer_revision": "0.6.13",
    "cuda_version": "12.8",
    "cutlass_version": "4.0",
    "activation_dtype": "bfloat16",
    "output_dtype": "bfloat16",
    "activation_scale_layout": "8x4",
    "weight_scale_layout": "128x4",
}


def _write_artifact(tmp_path: Path) -> str:
    artifact = {
        "schema_version": 1,
        "runtime": _RUNTIME,
        "shapes": {
            "8,130,512": {
                "status": "qualified",
                "tactic": 7,
                "workspace_bytes": 4096,
            },
            "16,130,512": {"status": "rejected"},
        },
    }
    path = tmp_path / "dynamic_a_allowlist.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    return str(path)


@pytest.mark.parametrize(
    ("shape", "expected"),
    [
        ((8, 130, 512), "dynamic_cutlass_8x4"),
        ((8, 256, 512), "stock_cutedsl"),
        ((16, 130, 512), "stock_cutedsl"),
    ],
)
def test_exact_physical_shape_selects_only_qualified_backend(
    tmp_path: Path,
    shape: tuple[int, int, int],
    expected: str,
) -> None:
    policy = hybrid.load_mxfp8_cutedsl_dynamic_a_policy(
        _write_artifact(tmp_path), runtime_metadata=_RUNTIME
    )

    assert policy.select(shape) == expected


def test_incompatible_runtime_falls_back_to_stock_cutedsl(
    tmp_path: Path,
) -> None:
    incompatible_runtime = _RUNTIME | {"flashinfer_revision": "0.6.14"}
    policy = hybrid.load_mxfp8_cutedsl_dynamic_a_policy(
        _write_artifact(tmp_path), runtime_metadata=incompatible_runtime
    )

    assert policy.select((8, 130, 512)) == "stock_cutedsl"


def test_malformed_physical_shape_key_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "malformed_dynamic_a_allowlist.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "runtime": _RUNTIME,
                "shapes": {"8,130": {"status": "qualified"}},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="physical.*M,N,K"):
        hybrid.load_mxfp8_cutedsl_dynamic_a_policy(str(path), runtime_metadata=_RUNTIME)


def test_policy_records_dynamic_hits_stock_misses_and_incompatibility_fallbacks(
    tmp_path: Path,
) -> None:
    policy = hybrid.load_mxfp8_cutedsl_dynamic_a_policy(
        _write_artifact(tmp_path), runtime_metadata=_RUNTIME
    )
    policy.select((8, 130, 512))
    policy.select((8, 256, 512))
    policy.select((16, 130, 512))

    assert policy.dispatch_counters() == {
        "dynamic_hits": 1,
        "stock_misses": 1,
        "incompatibility_fallbacks": 1,
    }


def test_post_load_keeps_one_unpadded_stock_b_and_b_scale_storage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    n, k = 130, 512
    layer = torch.nn.Module()
    layer.weight = Parameter(
        torch.ones((n, k), dtype=torch.float8_e4m3fn), requires_grad=False
    )
    layer.weight_scale = Parameter(
        torch.ones((n, k // MXFP8_BLOCK_SIZE), dtype=torch.float8_e8m0fnu),
        requires_grad=False,
    )
    monkeypatch.setitem(
        sys.modules,
        "flashinfer",
        SimpleNamespace(
            shuffle_matrix_a=lambda *_: pytest.fail("TRTLLM shuffle called"),
            shuffle_matrix_sf_a=lambda *_: pytest.fail("TRTLLM scale shuffle called"),
        ),
    )

    kernel = object.__new__(hybrid.FlashInferCutedslMxfp8LinearKernel)
    kernel.process_weights_after_loading(layer)

    assert {
        name: (
            tuple(parameter.shape),
            parameter.ndim,
            parameter.untyped_storage().nbytes(),
        )
        for name, parameter in layer.named_parameters()
    } == {
        "weight": ((k, n), 2, n * k),
        "weight_scale": ((4096,), 1, 4096),
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_dynamic_custom_op_forwards_all_caller_owned_buffers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = torch.device("cuda")
    a = torch.empty((8, 512), dtype=torch.bfloat16, device=device)
    weight = torch.empty((512, 130), dtype=torch.float8_e4m3fn, device=device)
    weight_scale = torch.empty((4096,), dtype=torch.float8_e8m0fnu, device=device)
    out = torch.empty((8, 130), dtype=torch.bfloat16, device=device)
    workspace = torch.empty((4096,), dtype=torch.uint8, device=device)
    quantized_a = torch.empty((8, 512), dtype=torch.float8_e4m3fn, device=device)
    quantized_a_scale = torch.empty((8, 16), dtype=torch.uint8, device=device)
    forwarded: dict[str, int | torch.dtype] = {}

    def dynamic_a_cutlass_mxfp8(
        a_bf16: torch.Tensor,
        weight_col_major: torch.Tensor,
        weight_scale_128x4: torch.Tensor,
        *,
        out: torch.Tensor,
        workspace: torch.Tensor,
        quant_out_value: torch.Tensor,
        quant_out_scale: torch.Tensor,
        out_dtype: torch.dtype,
    ) -> torch.Tensor:
        forwarded.update(
            {
                "a": a_bf16.data_ptr(),
                "weight": weight_col_major.data_ptr(),
                "weight_scale": weight_scale_128x4.data_ptr(),
                "out": out.data_ptr(),
                "workspace": workspace.data_ptr(),
                "quantized_a": quant_out_value.data_ptr(),
                "quantized_a_scale": quant_out_scale.data_ptr(),
                "out_dtype": out_dtype,
            }
        )
        return out

    monkeypatch.setitem(
        sys.modules,
        "flashinfer",
        SimpleNamespace(dynamic_a_cutlass_mxfp8=dynamic_a_cutlass_mxfp8),
    )

    result = vllm_flashinfer.mm_mxfp8_dynamic_a_cutlass(
        a,
        weight,
        weight_scale,
        out=out,
        workspace=workspace,
        quant_out_value=quantized_a,
        quant_out_scale=quantized_a_scale,
        out_dtype=torch.bfloat16,
    )

    assert (result.data_ptr(), forwarded) == (
        out.data_ptr(),
        {
            "a": a.data_ptr(),
            "weight": weight.data_ptr(),
            "weight_scale": weight_scale.data_ptr(),
            "out": out.data_ptr(),
            "workspace": workspace.data_ptr(),
            "quantized_a": quantized_a.data_ptr(),
            "quantized_a_scale": quantized_a_scale.data_ptr(),
            "out_dtype": torch.bfloat16,
        },
    )
