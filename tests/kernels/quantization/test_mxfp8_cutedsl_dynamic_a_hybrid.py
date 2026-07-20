# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import sys
from collections.abc import Iterator
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


def _make_physical_layer() -> torch.nn.Module:
    layer = torch.nn.Module()
    layer.weight = Parameter(
        torch.ones((512, 130), dtype=torch.float8_e4m3fn), requires_grad=False
    )
    layer.weight_scale = Parameter(
        torch.ones((4096,), dtype=torch.float8_e8m0fnu), requires_grad=False
    )
    # This is intentionally not the physical output width. Dispatch must use
    # layer.weight.shape, not a logical/checkpoint-derived output width.
    layer.output_size_per_partition = 256
    return layer


def _guard_trtllm_helpers() -> SimpleNamespace:
    def fail_helper_use(*_: object, **__: object) -> None:
        pytest.fail("TRTLLM weight helper was invoked")

    return SimpleNamespace(
        shuffle_matrix_a=fail_helper_use,
        shuffle_matrix_sf_a=fail_helper_use,
    )


def _guard_imported_trtllm_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_helper_use(*_: object, **__: object) -> None:
        pytest.fail("imported TRTLLM helper was invoked")

    for helper_name in (
        "configure_mxfp8_trtllm_adaptive_compilation",
        "mxfp8_trtllm_adaptive_linear",
        "prepare_mxfp8_trtllm_high_m_tactic_state",
    ):
        monkeypatch.setattr(hybrid, helper_name, fail_helper_use)


def _run_apply_weights(
    monkeypatch: pytest.MonkeyPatch,
    policy: object | None,
    *,
    api_available: bool = True,
    is_sm100: bool = True,
    dtype: torch.dtype = torch.bfloat16,
    m: int = 8,
) -> tuple[torch.Tensor, list[dict[str, object]], list[tuple[int, int, int]]]:
    stock_calls: list[dict[str, object]] = []
    dynamic_shapes: list[tuple[int, int, int]] = []
    layer = _make_physical_layer()
    kernel = object.__new__(hybrid.FlashInferCutedslMxfp8LinearKernel)
    kernel._dynamic_a_policy = policy
    x = torch.ones((m, 512), dtype=dtype)
    pre_reserved = {
        "out": torch.empty((m, 130), dtype=dtype),
        "workspace": torch.empty((8192,), dtype=torch.uint8),
        "quant_out_value": torch.empty((m, 512), dtype=torch.float8_e4m3fn),
        "quant_out_scale": torch.empty((m, 16), dtype=torch.uint8),
    }
    kernel._dynamic_a_buffers = {(m, 130, 512): pre_reserved}

    monkeypatch.setitem(sys.modules, "flashinfer", _guard_trtllm_helpers())
    _guard_imported_trtllm_helpers(monkeypatch)
    monkeypatch.setattr(
        hybrid,
        "current_platform",
        SimpleNamespace(
            is_cuda=lambda: is_sm100,
            is_device_capability_family=lambda capability: (
                is_sm100 and capability == 100
            ),
        ),
    )
    monkeypatch.setattr(
        hybrid,
        "mxfp8_e4m3_quantize",
        lambda value, **_: (
            value.to(torch.float8_e4m3fn),
            torch.ones((4096,), dtype=torch.float8_e8m0fnu),
        ),
    )
    monkeypatch.setattr(
        vllm_flashinfer,
        "has_flashinfer_mxfp8_dynamic_a_cutlass",
        lambda: api_available,
    )

    def stock_mm(
        a: torch.Tensor,
        b: torch.Tensor,
        a_scale: torch.Tensor,
        b_scale: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:
        stock_calls.append({"backend": kwargs["backend"], "b_shape": tuple(b.shape)})
        return torch.zeros((a.shape[0], b.shape[1]), dtype=kwargs["out_dtype"])

    def dynamic_mm(
        a: torch.Tensor,
        b: torch.Tensor,
        b_scale: torch.Tensor,
        *,
        out: torch.Tensor,
        workspace: torch.Tensor,
        quant_out_value: torch.Tensor,
        quant_out_scale: torch.Tensor,
        **_: object,
    ) -> torch.Tensor:
        assert {
            "out": out.data_ptr(),
            "workspace": workspace.data_ptr(),
            "quant_out_value": quant_out_value.data_ptr(),
            "quant_out_scale": quant_out_scale.data_ptr(),
        } == {name: buffer.data_ptr() for name, buffer in pre_reserved.items()}
        dynamic_shapes.append((a.shape[0], b.shape[1], b.shape[0]))
        out.zero_()
        return out

    monkeypatch.setattr(vllm_flashinfer, "mm_mxfp8", stock_mm)
    monkeypatch.setattr(vllm_flashinfer, "mm_mxfp8_dynamic_a_cutlass", dynamic_mm)

    return kernel.apply_weights(layer, x), stock_calls, dynamic_shapes


def _persistent_tensor_storages(module: torch.nn.Module) -> set[tuple[int, int]]:
    seen_objects: set[int] = set()
    storages: set[tuple[int, int]] = set()

    def visit(value: object) -> Iterator[torch.Tensor]:
        if id(value) in seen_objects:
            return
        seen_objects.add(id(value))
        if isinstance(value, torch.Tensor):
            yield value
        elif isinstance(value, dict):
            for child in value.values():
                yield from visit(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                yield from visit(child)

    values: list[object] = [
        dict(module.named_parameters(remove_duplicate=False)),
        dict(module.named_buffers(remove_duplicate=False)),
        {
            name: value
            for name, value in vars(module).items()
            if name not in {"_parameters", "_buffers", "_modules"}
        },
    ]
    for value in values:
        for tensor in visit(value):
            storage = tensor.untyped_storage()
            storages.add((storage.data_ptr(), storage.nbytes()))
    return storages


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


def test_apply_weights_uses_physical_n_for_dynamic_a_dispatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    policy = hybrid.load_mxfp8_cutedsl_dynamic_a_policy(
        _write_artifact(tmp_path), runtime_metadata=_RUNTIME
    )

    output, stock_calls, dynamic_shapes = _run_apply_weights(monkeypatch, policy)

    assert (tuple(output.shape), stock_calls, dynamic_shapes) == (
        (8, 130),
        [],
        [(8, 130, 512)],
    )


@pytest.mark.parametrize(
    "runtime_override",
    [{field: "mismatch"} for field in _RUNTIME],
)
def test_apply_weights_uses_stock_for_each_runtime_metadata_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    runtime_override: dict[str, str],
) -> None:
    policy = hybrid.load_mxfp8_cutedsl_dynamic_a_policy(
        _write_artifact(tmp_path), runtime_metadata=_RUNTIME | runtime_override
    )

    _, stock_calls, dynamic_shapes = _run_apply_weights(monkeypatch, policy)

    assert (stock_calls, dynamic_shapes) == (
        [{"backend": "cute-dsl", "b_shape": (512, 130)}],
        [],
    )


@pytest.mark.parametrize(
    ("policy", "api_available", "is_sm100", "dtype"),
    [
        (None, True, True, torch.bfloat16),
        ("matching", True, False, torch.bfloat16),
        ("matching", True, True, torch.float16),
        ("matching", False, True, torch.bfloat16),
    ],
    ids=["absent-artifact", "unsupported-sm", "unsupported-dtype", "missing-api"],
)
def test_apply_weights_falls_back_to_stock_when_dynamic_a_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    policy: object | str | None,
    api_available: bool,
    is_sm100: bool,
    dtype: torch.dtype,
) -> None:
    if policy is None:
        monkeypatch.delenv("VLLM_MXFP8_CUTEDSL_DYNAMIC_A_ARTIFACT", raising=False)
    resolved_policy = (
        hybrid.load_mxfp8_cutedsl_dynamic_a_policy(
            _write_artifact(tmp_path), runtime_metadata=_RUNTIME
        )
        if policy == "matching"
        else policy
    )

    _, stock_calls, dynamic_shapes = _run_apply_weights(
        monkeypatch,
        resolved_policy,
        api_available=api_available,
        is_sm100=is_sm100,
        dtype=dtype,
    )

    assert (stock_calls, dynamic_shapes) == (
        [{"backend": "cute-dsl", "b_shape": (512, 130)}],
        [],
    )


def test_apply_weights_updates_telemetry_for_dynamic_miss_and_incompatible_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    qualified = hybrid.load_mxfp8_cutedsl_dynamic_a_policy(
        _write_artifact(tmp_path), runtime_metadata=_RUNTIME
    )
    incompatible = hybrid.load_mxfp8_cutedsl_dynamic_a_policy(
        _write_artifact(tmp_path),
        runtime_metadata=_RUNTIME | {"cuda_version": "mismatch"},
    )
    _run_apply_weights(monkeypatch, qualified)
    _run_apply_weights(monkeypatch, qualified, m=32)
    _run_apply_weights(monkeypatch, qualified)
    _run_apply_weights(monkeypatch, incompatible)

    assert (qualified.dispatch_counters(), incompatible.dispatch_counters()) == (
        {
            "dynamic_hits": 2,
            "stock_misses": 1,
            "incompatibility_fallbacks": 0,
        },
        {
            "dynamic_hits": 0,
            "stock_misses": 0,
            "incompatibility_fallbacks": 1,
        },
    )


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
    monkeypatch.setitem(sys.modules, "flashinfer", _guard_trtllm_helpers())
    _guard_imported_trtllm_helpers(monkeypatch)

    kernel = object.__new__(hybrid.FlashInferCutedslMxfp8LinearKernel)
    kernel.process_weights_after_loading(layer)

    assert _persistent_tensor_storages(layer) == {
        (layer.weight.untyped_storage().data_ptr(), n * k),
        (layer.weight_scale.untyped_storage().data_ptr(), 4096),
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_dynamic_custom_op_is_compile_and_cuda_graph_safe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = torch.device("cuda")
    a = torch.ones((8, 512), dtype=torch.bfloat16, device=device)
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
        out.copy_(a_bf16[:, :1].expand_as(out))
        return out

    monkeypatch.setitem(
        sys.modules,
        "flashinfer",
        SimpleNamespace(dynamic_a_cutlass_mxfp8=dynamic_a_cutlass_mxfp8),
    )

    def run(a_bf16: torch.Tensor) -> torch.Tensor:
        return vllm_flashinfer.mm_mxfp8_dynamic_a_cutlass(
            a_bf16,
            weight,
            weight_scale,
            out=out,
            workspace=workspace,
            quant_out_value=quantized_a,
            quant_out_scale=quantized_a_scale,
            out_dtype=torch.bfloat16,
        )

    compiled = torch.compile(run, fullgraph=True, dynamic=True)
    compiled(a)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = compiled(a)
    first_output = out.clone()
    expected_pointers = {
        "a": a.data_ptr(),
        "weight": weight.data_ptr(),
        "weight_scale": weight_scale.data_ptr(),
        "out": out.data_ptr(),
        "workspace": workspace.data_ptr(),
        "quantized_a": quantized_a.data_ptr(),
        "quantized_a_scale": quantized_a_scale.data_ptr(),
        "out_dtype": torch.bfloat16,
    }
    a.fill_(2)
    graph.replay()

    assert result.data_ptr() == out.data_ptr()
    assert result.shape == (8, 130)
    assert forwarded == expected_pointers
    torch.testing.assert_close(first_output, torch.ones_like(first_output))
    torch.testing.assert_close(out, torch.full_like(out, 2))
