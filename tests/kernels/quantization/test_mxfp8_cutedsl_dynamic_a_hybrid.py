# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import inspect
import json
import logging
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
from vllm.v1.worker.workspace import WorkspaceManager

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


def _write_artifact_shapes(
    tmp_path: Path, shapes: dict[str, dict[str, object]]
) -> str:
    path = tmp_path / "dynamic_a_shapes.json"
    path.write_text(
        json.dumps({"schema_version": 1, "runtime": _RUNTIME, "shapes": shapes}),
        encoding="utf-8",
    )
    return str(path)


def _make_reservation_kernel(
    artifact_path: str,
) -> hybrid.FlashInferCutedslMxfp8LinearKernel:
    kernel = object.__new__(hybrid.FlashInferCutedslMxfp8LinearKernel)
    kernel._dynamic_a_artifact_path = artifact_path
    kernel._dynamic_a_policy = None
    kernel._dynamic_a_buffers = {}
    kernel._dynamic_a_telemetry = ()
    kernel._dynamic_a_owner_key = "layer.0"
    return kernel


def _runtime_resources(
    m: int, n: int, k: int, workspace_bytes: int
) -> dict[str, int]:
    scale_rows = (m + 7) // 8 * 8
    scale_cols = ((k // MXFP8_BLOCK_SIZE + 3) // 4) * 4
    return {
        "workspace_bytes": workspace_bytes,
        "activation_value_elements": m * k,
        "activation_scale_elements": scale_rows * scale_cols,
        "output_elements": m * n,
    }


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


def test_duplicate_physical_shape_key_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "duplicate_dynamic_a_allowlist.json"
    path.write_text(
        '{"schema_version":1,"runtime":'
        + json.dumps(_RUNTIME)
        + ',"shapes":{"8,130,512":{"status":"rejected"},'
        '"8,130,512":{"status":"rejected"}}}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate JSON key"):
        hybrid.load_mxfp8_cutedsl_dynamic_a_policy(
            str(path), runtime_metadata=_RUNTIME
        )


def test_runtime_metadata_uses_configured_dtypes() -> None:
    metadata = hybrid._mxfp8_dynamic_a_runtime_metadata(
        torch.float16, torch.float32
    )

    assert (metadata["activation_dtype"], metadata["output_dtype"]) == (
        "float16",
        "float32",
    )


def test_dynamic_resources_require_same_module_runtime_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resources = {
        "workspace_bytes": 4096,
        "activation_value_elements": 4096,
        "activation_scale_elements": 128,
        "output_elements": 1040,
    }
    module = SimpleNamespace(
        mm_mxfp8_dynamic_activation=lambda *_args, **_kwargs: None,
        get_mm_mxfp8_dynamic_activation_resources=lambda *_args, **_kwargs: resources,
    )
    monkeypatch.setitem(sys.modules, "flashinfer", module)
    vllm_flashinfer.has_flashinfer_mxfp8_dynamic_a_cutlass.cache_clear()

    assert vllm_flashinfer.get_flashinfer_mxfp8_dynamic_a_resources(
        8,
        130,
        512,
        tactic=7,
        activation_dtype=torch.bfloat16,
        output_dtype=torch.bfloat16,
    ) == resources

    delattr(module, "get_mm_mxfp8_dynamic_activation_resources")
    assert (
        vllm_flashinfer.get_flashinfer_mxfp8_dynamic_a_resources(
            8,
            130,
            512,
            tactic=7,
            activation_dtype=torch.bfloat16,
            output_dtype=torch.bfloat16,
        )
        is None
    )


def test_dynamic_api_prefers_complete_gemm_pair_over_partial_top_level(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resources = {
        "workspace_bytes": 4096,
        "activation_value_elements": 4096,
        "activation_scale_elements": 128,
        "output_elements": 1040,
    }
    monkeypatch.setitem(
        sys.modules,
        "flashinfer",
        SimpleNamespace(mm_mxfp8_dynamic_activation=lambda *_args, **_kwargs: None),
    )
    monkeypatch.setitem(
        sys.modules,
        "flashinfer.gemm",
        SimpleNamespace(
            mm_mxfp8_dynamic_activation=lambda *_args, **_kwargs: None,
            get_mm_mxfp8_dynamic_activation_resources=(
                lambda *_args, **_kwargs: resources
            ),
        ),
    )

    assert vllm_flashinfer.get_flashinfer_mxfp8_dynamic_a_resources(
        8,
        130,
        512,
        tactic=7,
        activation_dtype=torch.bfloat16,
        output_dtype=torch.bfloat16,
    ) == resources


def test_compatibility_dynamic_api_forwards_qualified_tactic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    forwarded_tactics: list[int] = []

    def compatibility_api(
        *_: object,
        out: torch.Tensor,
        tactic: int,
        **__: object,
    ) -> torch.Tensor:
        forwarded_tactics.append(tactic)
        return out

    monkeypatch.setitem(
        sys.modules,
        "flashinfer",
        SimpleNamespace(dynamic_a_cutlass_mxfp8=compatibility_api),
    )
    out = torch.empty((8, 130), dtype=torch.bfloat16)
    vllm_flashinfer.mm_mxfp8_dynamic_a_cutlass(
        torch.ones((8, 512), dtype=torch.bfloat16),
        torch.empty((512, 130), dtype=torch.float8_e4m3fn),
        torch.empty((4096,), dtype=torch.uint8),
        out=out,
        workspace=torch.empty((4096,), dtype=torch.uint8),
        quant_out_value=torch.empty((8, 512), dtype=torch.float8_e4m3fn),
        quant_out_scale=torch.empty((8, 16), dtype=torch.uint8),
        out_dtype=torch.bfloat16,
        tactic=7,
    )

    assert forwarded_tactics == [7]


def test_compatibility_dynamic_api_without_tactic_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def compatibility_api(
        *_: object,
        out: torch.Tensor,
        workspace: torch.Tensor,
        quant_out_value: torch.Tensor,
        quant_out_scale: torch.Tensor,
        out_dtype: torch.dtype,
    ) -> torch.Tensor:
        return out

    monkeypatch.setitem(
        sys.modules,
        "flashinfer",
        SimpleNamespace(
            dynamic_a_cutlass_mxfp8=compatibility_api,
            get_mm_mxfp8_dynamic_activation_resources=lambda *_args, **_kwargs: (
                _runtime_resources(8, 130, 512, 4096)
            ),
        ),
    )

    assert (
        vllm_flashinfer.get_flashinfer_mxfp8_dynamic_a_resources(
            8,
            130,
            512,
            tactic=7,
            activation_dtype=torch.bfloat16,
            output_dtype=torch.bfloat16,
        )
        is None
    )


def test_reservation_filters_artifact_shapes_by_layer_physical_nk(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    artifact_path = _write_artifact_shapes(
        tmp_path,
        {
            "8,130,512": {
                "status": "qualified",
                "tactic": 7,
                "workspace_bytes": 4096,
            },
            "32,130,512": {
                "status": "qualified",
                "tactic": 8,
                "workspace_bytes": 8192,
            },
            "8,256,512": {
                "status": "qualified",
                "tactic": 9,
                "workspace_bytes": 4096,
            },
            "8,130,1024": {
                "status": "qualified",
                "tactic": 10,
                "workspace_bytes": 4096,
            },
        },
    )
    kernel = _make_reservation_kernel(artifact_path)
    manager = WorkspaceManager(torch.device("cpu"), num_ubatches=2)
    queried: list[tuple[int, int, int]] = []
    monkeypatch.setattr(
        hybrid, "_mxfp8_dynamic_a_runtime_metadata", lambda *_: _RUNTIME
    )
    monkeypatch.setattr(
        hybrid,
        "current_platform",
        SimpleNamespace(
            is_cuda=lambda: True,
            is_device_capability_family=lambda capability: capability == 100,
        ),
    )
    monkeypatch.setattr(
        vllm_flashinfer, "has_flashinfer_mxfp8_dynamic_a_cutlass", lambda: True
    )
    monkeypatch.setattr(
        vllm_flashinfer,
        "get_flashinfer_mxfp8_dynamic_a_resources",
        lambda m, n, k, **_: (
            queried.append((m, n, k))
            or _runtime_resources(m, n, k, 4096 if m == 8 else 8192)
        ),
    )

    kernel.reserve_dynamic_a_workspaces(
        _make_physical_layer(),
        manager,
        activation_dtype=torch.bfloat16,
        output_dtype=torch.bfloat16,
    )

    assert queried == [(8, 130, 512), (32, 130, 512)]
    assert set(kernel._dynamic_a_buffers) == {(8, 130, 512), (32, 130, 512)}
    assert len(manager._owner_specs["layer.0"]) == 9


def test_incompatible_reservation_keeps_per_slot_device_telemetry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    kernel = _make_reservation_kernel(_write_artifact(tmp_path))
    manager = WorkspaceManager(torch.device("cpu"), num_ubatches=2)
    monkeypatch.setattr(
        hybrid,
        "_mxfp8_dynamic_a_runtime_metadata",
        lambda *_: _RUNTIME | {"cuda_version": "mismatch"},
    )
    monkeypatch.setattr(
        hybrid,
        "_record_host_counter",
        lambda *_args, **_kwargs: pytest.fail(
            "capturable fallback must use device telemetry"
        ),
    )
    monkeypatch.setattr(
        hybrid,
        "_stock_cutedsl_mxfp8",
        lambda input_2d, weight, *_: torch.zeros(
            (input_2d.shape[0], weight.shape[1]), dtype=input_2d.dtype
        ),
    )

    kernel.reserve_dynamic_a_workspaces(
        _make_physical_layer(),
        manager,
        activation_dtype=torch.bfloat16,
        output_dtype=torch.bfloat16,
    )

    assert kernel._dynamic_a_buffers == {}
    assert len(kernel._dynamic_a_telemetry) == 2
    assert manager._owner_specs["layer.0"] == (((3,), torch.int64),)

    telemetry = kernel._dynamic_a_telemetry[0]
    torch.ops.vllm.mxfp8_cutedsl_dynamic_a_hybrid(
        torch.ones((8, 512), dtype=torch.bfloat16),
        _make_physical_layer().weight,
        torch.empty((4096,), dtype=torch.uint8),
        [],
        [8, 130, 512],
        [],
        [-1],
        [7],
        False,
        kernel._dynamic_a_policy.telemetry_key,
        telemetry,
        torch.bfloat16,
    )
    assert telemetry.tolist() == [0, 0, 1]


def test_runtime_resource_query_exception_fails_closed_with_telemetry(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    kernel = _make_reservation_kernel(_write_artifact(tmp_path))
    manager = WorkspaceManager(torch.device("cpu"), num_ubatches=1)
    monkeypatch.setattr(
        hybrid, "_mxfp8_dynamic_a_runtime_metadata", lambda *_: _RUNTIME
    )
    monkeypatch.setattr(
        hybrid,
        "current_platform",
        SimpleNamespace(
            is_cuda=lambda: True,
            is_device_capability_family=lambda capability: capability == 100,
        ),
    )
    monkeypatch.setattr(
        vllm_flashinfer, "has_flashinfer_mxfp8_dynamic_a_cutlass", lambda: True
    )

    def raise_query(*_: object, **__: object) -> None:
        raise RuntimeError("query failed")

    monkeypatch.setattr(
        vllm_flashinfer,
        "get_flashinfer_mxfp8_dynamic_a_resources",
        raise_query,
    )

    with caplog.at_level(logging.WARNING):
        kernel.reserve_dynamic_a_workspaces(
            _make_physical_layer(),
            manager,
            activation_dtype=torch.bfloat16,
            output_dtype=torch.bfloat16,
        )

    assert kernel._dynamic_a_buffers == {}
    assert len(kernel._dynamic_a_telemetry) == 1
    assert manager._owner_specs["layer.0"] == (((3,), torch.int64),)
    assert "runtime resource query failed" in caplog.text


def test_runtime_resource_mismatch_fails_before_dynamic_allocation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    kernel = _make_reservation_kernel(_write_artifact(tmp_path))
    manager = WorkspaceManager(torch.device("cpu"), num_ubatches=1)
    monkeypatch.setattr(
        hybrid, "_mxfp8_dynamic_a_runtime_metadata", lambda *_: _RUNTIME
    )
    monkeypatch.setattr(
        hybrid,
        "current_platform",
        SimpleNamespace(
            is_cuda=lambda: True,
            is_device_capability_family=lambda capability: capability == 100,
        ),
    )
    monkeypatch.setattr(
        vllm_flashinfer, "has_flashinfer_mxfp8_dynamic_a_cutlass", lambda: True
    )
    monkeypatch.setattr(
        vllm_flashinfer,
        "get_flashinfer_mxfp8_dynamic_a_resources",
        lambda *_args, **_kwargs: _runtime_resources(8, 130, 512, 8192),
    )

    kernel.reserve_dynamic_a_workspaces(
        _make_physical_layer(),
        manager,
        activation_dtype=torch.bfloat16,
        output_dtype=torch.bfloat16,
    )

    assert kernel._dynamic_a_buffers == {}
    assert manager._owner_specs["layer.0"] == (((3,), torch.int64),)


def test_policy_records_dynamic_hits_stock_misses_and_incompatibility_fallbacks(
    tmp_path: Path,
) -> None:
    policy = hybrid.load_mxfp8_cutedsl_dynamic_a_policy(
        _write_artifact(tmp_path), runtime_metadata=_RUNTIME
    )
    policy.select((8, 130, 512))
    policy.select((8, 256, 512))
    policy.select((16, 130, 512))

    assert policy.selection_counters() == {
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


def test_apply_weights_delegates_runtime_shape_choice_to_opaque_op() -> None:
    source = inspect.getsource(hybrid.FlashInferCutedslMxfp8LinearKernel.apply_weights)

    assert ".select(" not in source
    assert "mxfp8_cutedsl_dynamic_a_hybrid" in source


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


def test_opaque_dynamic_dispatch_propagates_execution_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_execution(*_: object, **__: object) -> torch.Tensor:
        raise RuntimeError("dynamic execution failed")

    monkeypatch.setattr(
        vllm_flashinfer, "mm_mxfp8_dynamic_a_cutlass", raise_execution
    )
    dynamic_buffers = [
        torch.empty((8, 512), dtype=torch.float8_e4m3fn),
        torch.empty((8, 16), dtype=torch.uint8),
        torch.empty((4096,), dtype=torch.uint8),
        torch.empty((8, 130), dtype=torch.bfloat16),
    ]

    with pytest.raises(RuntimeError, match="dynamic execution failed"):
        torch.ops.vllm.mxfp8_cutedsl_dynamic_a_hybrid(
            torch.ones((8, 512), dtype=torch.bfloat16),
            torch.empty((512, 130), dtype=torch.float8_e4m3fn),
            torch.empty((4096,), dtype=torch.uint8),
            dynamic_buffers,
            [8, 130, 512],
            [],
            [0],
            [7],
            True,
            "execution-error",
            torch.zeros((3,), dtype=torch.int64),
            torch.bfloat16,
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


def test_dynamic_custom_op_returns_fresh_output_and_validates_upstream_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    a = torch.ones((8, 512), dtype=torch.bfloat16)
    weight = torch.empty((512, 130), dtype=torch.float8_e4m3fn)
    weight_scale = torch.empty((4096,), dtype=torch.uint8)
    out = torch.empty((8, 130), dtype=torch.bfloat16)
    workspace = torch.empty((4096,), dtype=torch.uint8)
    quantized_a = torch.empty((8, 512), dtype=torch.float8_e4m3fn)
    quantized_a_scale = torch.empty((8, 16), dtype=torch.uint8)

    def dynamic_a_cutlass_mxfp8(*_: object, out: torch.Tensor, **__: object):
        out.fill_(3)
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

    assert result.data_ptr() != out.data_ptr()
    torch.testing.assert_close(result, out)

    def nonalias_dynamic_a(*_: object, out: torch.Tensor, **__: object):
        return torch.empty_like(out)

    monkeypatch.setitem(
        sys.modules,
        "flashinfer",
        SimpleNamespace(dynamic_a_cutlass_mxfp8=nonalias_dynamic_a),
    )
    with pytest.raises(RuntimeError, match="must alias caller-owned out"):
        vllm_flashinfer.mm_mxfp8_dynamic_a_cutlass(
            a,
            weight,
            weight_scale,
            out=out,
            workspace=workspace,
            quant_out_value=quantized_a,
            quant_out_scale=quantized_a_scale,
            out_dtype=torch.bfloat16,
        )


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
    bias = torch.ones((130,), dtype=torch.bfloat16, device=device)
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
        tactic: int,
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
                "tactic": tactic,
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
        result = vllm_flashinfer.mm_mxfp8_dynamic_a_cutlass(
            a_bf16,
            weight,
            weight_scale,
            out=out,
            workspace=workspace,
            quant_out_value=quantized_a,
            quant_out_scale=quantized_a_scale,
            out_dtype=torch.bfloat16,
        )
        torch._assert(result.ndim == 2, "dynamic-A custom op must return [M, N]")
        return (result + bias).sum(dim=1)

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
        "tactic": -1,
    }
    a.fill_(2)
    graph.replay()

    assert result.shape == (8,)
    assert forwarded == expected_pointers
    torch.testing.assert_close(first_output, torch.ones_like(first_output))
    torch.testing.assert_close(out, torch.full_like(out, 2))
    torch.testing.assert_close(result, torch.full_like(result, 390))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_opaque_hybrid_op_fullgraph_dispatch_and_fallback_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = torch.device("cuda")
    activation = torch.ones((8, 512), dtype=torch.bfloat16, device=device)
    weight = torch.empty((512, 130), dtype=torch.float8_e4m3fn, device=device)
    weight_scale = torch.empty((4096,), dtype=torch.uint8, device=device)
    owner_out = torch.empty((8, 130), dtype=torch.bfloat16, device=device)
    dynamic_buffers = [
        torch.empty((8, 512), dtype=torch.float8_e4m3fn, device=device),
        torch.empty((8, 16), dtype=torch.uint8, device=device),
        torch.empty((4096,), dtype=torch.uint8, device=device),
        owner_out,
    ]
    dynamic_telemetry = torch.zeros((3,), dtype=torch.int64, device=device)
    fallback_telemetry = torch.zeros((3,), dtype=torch.int64, device=device)

    def dynamic_mm(
        input_2d: torch.Tensor,
        *_: object,
        out: torch.Tensor,
        tactic: int,
        **__: object,
    ) -> torch.Tensor:
        assert tactic == 7
        out.copy_(input_2d[:, :1].expand_as(out))
        return out.clone()

    def stock_mm(
        input_2d: torch.Tensor,
        weight_tensor: torch.Tensor,
        *_: object,
    ) -> torch.Tensor:
        return input_2d[:, :1].expand(
            input_2d.shape[0], weight_tensor.shape[1]
        ) + 4

    monkeypatch.setattr(
        vllm_flashinfer, "mm_mxfp8_dynamic_a_cutlass", dynamic_mm
    )
    monkeypatch.setattr(hybrid, "_stock_cutedsl_mxfp8", stock_mm)

    schema = torch.ops.vllm.mxfp8_cutedsl_dynamic_a_hybrid.default._schema
    arguments = {argument.name: argument for argument in schema.arguments}
    assert arguments["dynamic_buffers"].alias_info.is_write
    assert arguments["telemetry"].alias_info.is_write
    assert schema.returns[0].alias_info is None

    def run_dynamic(input_2d: torch.Tensor) -> torch.Tensor:
        return torch.ops.vllm.mxfp8_cutedsl_dynamic_a_hybrid(
            input_2d,
            weight,
            weight_scale,
            dynamic_buffers,
            [8, 130, 512],
            [],
            [0],
            [7],
            True,
            "cuda-dynamic",
            dynamic_telemetry,
            torch.bfloat16,
        )

    compiled_dynamic = torch.compile(run_dynamic, fullgraph=True, dynamic=True)
    dynamic_result = compiled_dynamic(activation)
    assert dynamic_result.data_ptr() != owner_out.data_ptr()
    torch.testing.assert_close(dynamic_result, torch.ones_like(dynamic_result))
    assert dynamic_telemetry.tolist() == [1, 0, 0]

    def run_fallback(input_2d: torch.Tensor) -> torch.Tensor:
        return torch.ops.vllm.mxfp8_cutedsl_dynamic_a_hybrid(
            input_2d,
            weight,
            weight_scale,
            [],
            [8, 130, 512],
            [],
            [-1],
            [7],
            False,
            "cuda-fallback",
            fallback_telemetry,
            torch.bfloat16,
        )

    compiled_fallback = torch.compile(run_fallback, fullgraph=True, dynamic=True)
    compiled_fallback(activation)
    fallback_telemetry.zero_()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fallback_result = compiled_fallback(activation)
    activation.fill_(2)
    graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(fallback_result, torch.full_like(fallback_result, 6))
    assert fallback_telemetry.tolist() == [0, 0, 2]
