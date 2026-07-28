import ast
import builtins
import functools
import hashlib
import importlib.util
import json
import os
import socket
import sys
import time
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, NamedTuple

import pytest

ROOT = Path(__file__).resolve().parents[3]
OVERRIDE = ROOT / "vllm"
UTILS = OVERRIDE / "model_executor/layers/quantization/utils/mxfp8_utils.py"
FLASHINFER_UTILS = OVERRIDE / "utils/flashinfer.py"
LINEAR = OVERRIDE / "model_executor/kernels/linear/mxfp8/flashinfer.py"
TACTIC_CONFIG = (
    OVERRIDE / "model_executor/kernels/linear/mxfp8/tactic_config.py"
)

_TACTIC_CONFIG_SPEC = importlib.util.spec_from_file_location(
    "task4_mxfp8_tactic_config", TACTIC_CONFIG
)
if _TACTIC_CONFIG_SPEC is None or _TACTIC_CONFIG_SPEC.loader is None:
    raise RuntimeError("cannot load MXFP8 tactic config module")
_TACTIC_CONFIG_MODULE = importlib.util.module_from_spec(_TACTIC_CONFIG_SPEC)
sys.modules[_TACTIC_CONFIG_SPEC.name] = _TACTIC_CONFIG_MODULE
_TACTIC_CONFIG_SPEC.loader.exec_module(_TACTIC_CONFIG_MODULE)


def _isolated_import(
    name: str,
    globals_: dict[str, object] | None = None,
    locals_: dict[str, object] | None = None,
    fromlist: tuple[str, ...] = (),
    level: int = 0,
) -> object:
    if name == "vllm.utils":
        runtime_module = sys.modules.get("vllm.utils")
        if runtime_module is not None:
            return runtime_module
        return SimpleNamespace(
            flashinfer=SimpleNamespace(
                get_mxfp8_trtllm_file_configuration=lambda: None
            )
        )
    return builtins.__import__(name, globals_, locals_, fromlist, level)


def _load_layout_policy() -> Callable[[int], bool]:
    tree = ast.parse(UTILS.read_text(encoding="utf-8"))
    names = {
        "_mxfp8_dense_runtime_configuration",
        "_mxfp8_dense_a_sf_layout",
        "mxfp8_dense_use_8x4_sf_layout",
    }
    selected = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in names
    ]
    assert {
        "_mxfp8_dense_a_sf_layout",
        "mxfp8_dense_use_8x4_sf_layout",
    } <= {node.name for node in selected}
    namespace: dict[str, object] = {
        "__builtins__": {**vars(builtins), "__import__": _isolated_import},
        "os": os,
        "_Mxfp8DenseRuntimeConfiguration": object,
    }
    exec(
        compile(ast.Module(body=selected, type_ignores=[]), str(UTILS), "exec"),
        namespace,
    )
    return namespace["mxfp8_dense_use_8x4_sf_layout"]  # type: ignore[return-value]


def _load_trtllm_configuration_contract() -> tuple[type, Callable, Callable]:
    tree = ast.parse(FLASHINFER_UTILS.read_text(encoding="utf-8"))
    names = {
        "_Mxfp8TrtllmConfigurationFingerprint",
        "_parse_mxfp8_tactic_hints",
        "_mxfp8_trtllm_configuration_fingerprint",
        "_validate_mxfp8_trtllm_configuration",
    }
    selected = [
        node
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names
    ]
    assert {node.name for node in selected} == names
    namespace: dict[str, object] = {
        "functools": functools,
        "importlib": SimpleNamespace(
            metadata=SimpleNamespace(
                version=lambda _package: "0.6.8.post1+cu129"
            )
        ),
        "_load_mxfp8_dense_runtime_config": (
            lambda reference, *, actual_model, actual_tensor_parallel_size: (
                _TACTIC_CONFIG_MODULE.load_mxfp8_dense_runtime_config(
                    reference,
                    actual_vllm_version="0.20.2+local",
                    actual_flashinfer_version="0.6.8.post1+cu129",
                    actual_compute_capability=(10, 0),
                    actual_model=actual_model,
                    actual_tensor_parallel_size=actual_tensor_parallel_size,
                )
            )
        ),
        "_active_mxfp8_model_and_tp": lambda: (
            "Nemotron 3 Ultra MXFP8",
            4,
        ),
        "NamedTuple": NamedTuple,
        "os": os,
        "torch": SimpleNamespace(
            cuda=SimpleNamespace(
                get_device_capability=lambda: (10, 0),
                is_current_stream_capturing=lambda: False,
            )
        ),
        "vllm": SimpleNamespace(__version__="0.20.2+local"),
    }
    exec(
        compile(
            ast.Module(body=selected, type_ignores=[]), str(FLASHINFER_UTILS), "exec"
        ),
        namespace,
    )
    return (
        namespace["_Mxfp8TrtllmConfigurationFingerprint"],  # type: ignore[return-value]
        namespace["_mxfp8_trtllm_configuration_fingerprint"],  # type: ignore[return-value]
        namespace["_validate_mxfp8_trtllm_configuration"],  # type: ignore[return-value]
    )


def _load_trtllm_runtime_freeze_contract(
    messages: list[str],
) -> Callable[[], object]:
    tree = ast.parse(FLASHINFER_UTILS.read_text(encoding="utf-8"))
    names = {
        "_Mxfp8TrtllmConfigurationFingerprint",
        "_parse_mxfp8_tactic_hints",
        "_mxfp8_trtllm_configuration_fingerprint",
        "_validate_mxfp8_trtllm_configuration",
        "_log_mxfp8_dense_config_once",
        "_freeze_mxfp8_trtllm_configuration",
        "get_mxfp8_trtllm_configuration",
    }
    selected = [
        node
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names
    ]
    assert {node.name for node in selected} == names

    class CapturingLogger:
        def info(self, message: str, *args: object) -> None:
            messages.append(message % args)

    namespace: dict[str, object] = {
        "functools": functools,
        "importlib": SimpleNamespace(
            metadata=SimpleNamespace(
                version=lambda _package: "0.6.8.post1+cu129"
            )
        ),
        "_load_mxfp8_dense_runtime_config": (
            lambda reference, *, actual_model, actual_tensor_parallel_size: (
                _TACTIC_CONFIG_MODULE.load_mxfp8_dense_runtime_config(
                    reference,
                    actual_vllm_version="0.20.2+local",
                    actual_flashinfer_version="0.6.8.post1+cu129",
                    actual_compute_capability=(10, 0),
                    actual_model=actual_model,
                    actual_tensor_parallel_size=actual_tensor_parallel_size,
                )
            )
        ),
        "_active_mxfp8_model_and_tp": lambda: (
            "Nemotron 3 Ultra MXFP8",
            4,
        ),
        "logger": CapturingLogger(),
        "NamedTuple": NamedTuple,
        "os": os,
        "torch": SimpleNamespace(
            cuda=SimpleNamespace(
                get_device_capability=lambda: (10, 0),
                is_current_stream_capturing=lambda: False,
            )
        ),
        "vllm": SimpleNamespace(__version__="0.20.2+local"),
        "_MXFP8_DENSE_CONFIG_LOGGED": False,
        "_MXFP8_TRTLLM_CONFIGURATION": None,
    }
    exec(
        compile(
            ast.Module(body=selected, type_ignores=[]), str(FLASHINFER_UTILS), "exec"
        ),
        namespace,
    )
    return namespace["get_mxfp8_trtllm_configuration"]  # type: ignore[return-value]


def _load_post_freeze_validation_contract(
    loader_calls: list[str],
    context_calls: list[None],
) -> dict[str, object]:
    tree = ast.parse(FLASHINFER_UTILS.read_text(encoding="utf-8"))
    names = {
        "_Mxfp8TrtllmConfigurationFingerprint",
        "_parse_mxfp8_tactic_hints",
        "_mxfp8_trtllm_configuration_fingerprint",
        "_validate_mxfp8_trtllm_configuration",
        "_log_mxfp8_dense_config_once",
        "_freeze_mxfp8_trtllm_configuration",
        "get_mxfp8_trtllm_configuration",
        "get_mxfp8_trtllm_file_configuration",
        "validate_mxfp8_trtllm_configuration",
        "validate_mxfp8_trtllm_configuration_source",
    }
    selected = [
        node
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names
    ]
    assert {node.name for node in selected} == names
    namespace: dict[str, object] = {
        "functools": functools,
        "_load_mxfp8_dense_runtime_config": lambda reference,
        *,
        actual_model,
        actual_tensor_parallel_size: (
            loader_calls.append(reference)
            or _TACTIC_CONFIG_MODULE.load_mxfp8_dense_runtime_config(
                reference,
                actual_vllm_version="0.20.2+local",
                actual_flashinfer_version="0.6.8.post1+cu129",
                actual_compute_capability=(10, 0),
                actual_model=actual_model,
                actual_tensor_parallel_size=actual_tensor_parallel_size,
            )
        ),
        "_active_mxfp8_model_and_tp": lambda: (
            context_calls.append(None) or ("Nemotron 3 Ultra MXFP8", 4)
        ),
        "logger": SimpleNamespace(info=lambda *_args: None),
        "NamedTuple": NamedTuple,
        "os": os,
        "torch": SimpleNamespace(
            cuda=SimpleNamespace(is_current_stream_capturing=lambda: False)
        ),
        "_MXFP8_DENSE_CONFIG_LOGGED": False,
        "_MXFP8_TRTLLM_CONFIGURATION": None,
    }
    exec(
        compile(
            ast.Module(body=selected, type_ignores=[]), str(FLASHINFER_UTILS), "exec"
        ),
        namespace,
    )
    return namespace


def _function_source(path: Path, name: str) -> str:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
    )
    segment = ast.get_source_segment(source, function)
    assert segment is not None
    return segment


class _FakeCudaDevice:
    def __init__(self, device: object, index: int | None = None) -> None:
        if isinstance(device, _FakeCudaDevice):
            self.type = device.type
            self.index = device.index
        elif isinstance(device, str):
            device_type, separator, device_index = device.partition(":")
            self.type = device_type
            self.index = int(device_index) if separator else index
        else:
            raise TypeError(f"unsupported fake device {device!r}")


class _FakeCuda:
    def __init__(self, *, current_device: int, capturing_devices: set[int]) -> None:
        self._current_device = current_device
        self._capturing_devices = capturing_devices

    def current_device(self) -> int:
        return self._current_device

    def is_current_stream_capturing(self) -> bool:
        return self._current_device in self._capturing_devices

    @contextmanager
    def device(self, device: _FakeCudaDevice):
        previous_device = self._current_device
        self._current_device = device.index
        try:
            yield
        finally:
            self._current_device = previous_device


def _load_direct_state_contract(cuda: _FakeCuda) -> dict[str, object]:
    tree = ast.parse(FLASHINFER_UTILS.read_text(encoding="utf-8"))
    names = {
        "_Mxfp8TrtllmConfigurationFingerprint",
        "_Mxfp8TrtllmDirectState",
        "_parse_mxfp8_tactic_hints",
        "_mxfp8_trtllm_configuration_fingerprint",
        "_validate_mxfp8_trtllm_configuration",
        "_freeze_mxfp8_trtllm_configuration",
        "get_mxfp8_trtllm_configuration",
        "get_mxfp8_trtllm_file_configuration",
        "validate_mxfp8_trtllm_configuration",
        "_mxfp8_trtllm_device_key",
        "prepare_mxfp8_trtllm_direct_state",
    }
    selected = [
        node
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names
    ]
    assert {node.name for node in selected} == names
    fake_torch = SimpleNamespace(Tensor=object, device=_FakeCudaDevice, cuda=cuda)
    namespace: dict[str, object] = {
        "functools": functools,
        "importlib": SimpleNamespace(
            metadata=SimpleNamespace(
                version=lambda _package: "0.6.8.post1+cu129"
            )
        ),
        "_load_mxfp8_dense_runtime_config": (
            lambda reference, *, actual_model, actual_tensor_parallel_size: (
                _TACTIC_CONFIG_MODULE.load_mxfp8_dense_runtime_config(
                    reference,
                    actual_vllm_version="0.20.2+local",
                    actual_flashinfer_version="0.6.8.post1+cu129",
                    actual_compute_capability=(10, 0),
                    actual_model=actual_model,
                    actual_tensor_parallel_size=actual_tensor_parallel_size,
                )
            )
        ),
        "_active_mxfp8_model_and_tp": lambda: (
            "Nemotron 3 Ultra MXFP8",
            4,
        ),
        "NamedTuple": NamedTuple,
        "os": os,
        "torch": fake_torch,
        "vllm": SimpleNamespace(__version__="0.20.2+local"),
        "Any": Any,
        "_log_mxfp8_dense_config_once": lambda _configuration: None,
        "_MXFP8_TRTLLM_CONFIGURATION": None,
        "_MXFP8_TRTLLM_DIRECT_STATES": {},
    }
    exec(
        compile(
            ast.Module(body=selected, type_ignores=[]), str(FLASHINFER_UTILS), "exec"
        ),
        namespace,
    )
    return namespace


def _install_fake_flashinfer_gemm(monkeypatch: pytest.MonkeyPatch) -> None:
    flashinfer = SimpleNamespace(__path__=[])
    gemm = SimpleNamespace(__path__=[])
    gemm_base = SimpleNamespace(
        DEFAULT_WORKSPACE_SIZE=1,
        _get_cache_buf=lambda *_args: object(),
        get_trtllm_gemm_module=lambda: SimpleNamespace(
            trtllm_mxfp8_gemm_runner=lambda **_kwargs: object()
        ),
    )
    monkeypatch.setitem(sys.modules, "flashinfer", flashinfer)
    monkeypatch.setitem(sys.modules, "flashinfer.gemm", gemm)
    monkeypatch.setitem(sys.modules, "flashinfer.gemm.gemm_base", gemm_base)


def _load_fixed_quantizer(
    name: str,
    torch_module: object,
) -> Callable[[object], tuple[object, object]]:
    tree = ast.parse(UTILS.read_text(encoding="utf-8"))
    names = {
        "_mxfp8_dense_runtime_configuration",
        "_env_flag",
        "prepare_mxfp8_dense_quant_backend",
        "_mxfp8_dense_quant_backend",
        "_flashinfer_mxfp8_quantize_impl",
        "_mxfp8_e4m3_quantize_fixed_layout_impl",
        "mxfp8_e4m3_quantize_8x4_impl",
        "mxfp8_e4m3_quantize_128x4_impl",
    }
    selected = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert {node.name for node in selected} == names
    namespace: dict[str, object] = {
        "__builtins__": {**vars(builtins), "__import__": _isolated_import},
        "os": os,
        "torch": torch_module,
        "_Mxfp8DenseRuntimeConfiguration": object,
        "_MXFP8_DENSE_QUANT_BACKEND": None,
    }
    exec(
        compile(ast.Module(body=selected, type_ignores=[]), str(UTILS), "exec"),
        namespace,
    )
    return namespace[name]  # type: ignore[return-value]


class _FakeScale:
    def __init__(self, numel: int) -> None:
        self._numel = numel
        self.ndim = 1
        self.dtype = object()
        self.device = object()

    def numel(self) -> int:
        return self._numel

    def reshape(self, *_shape: int) -> "_FakeScale":
        return self

    def __getitem__(self, _index: object) -> "_FakeScale":
        return self

    def copy_(self, _source: object) -> "_FakeScale":
        return self


class _FakeTorch:
    Tensor = object


def _runtime_manifest() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "mode": "adaptive",
        "compatibility": {
            "vllm_version": "0.20.2",
            "vllm_base_commit": "5246e3c5df5fb8266b50ceaa6eca2836fb2d13b1",
            "flashinfer_version": "0.6.8.post1",
            "compute_capability": "10.0",
            "gpu_family": "GB200",
            "model": "Nemotron 3 Ultra MXFP8",
            "tensor_parallel_size": 4,
        },
        "policy": {
            "gemm_backend": "trtllm",
            "layout": "adaptive",
            "switch_m": 256,
            "direct_trtllm": True,
            "require_direct_trtllm": True,
            "quant_backend": "cuda",
            "require_8x4_quant": True,
            "pad_to_128": False,
            "default_tactic": -1,
        },
        "tactics": {
            "8x4": [
                {"m": 1, "n": 2048, "k": 8192, "tactic": 66},
                {"m": 32, "n": 8192, "k": 2048, "tactic": 71},
            ],
            "128x4": [
                {"m": 1000, "n": 2048, "k": 8192, "tactic": 70},
            ],
        },
        "provenance": {
            "source_manifest_sha256": "a" * 64,
            "source_hint_sha256": "b" * 64,
            "container_sha256": "c" * 64,
            "qualification_scope": "standalone_serving_seed",
            "qualification_repeat_count": 3,
            "minimum_cosine_similarity": 0.999,
            "minimum_speedup_vs_default": 1.02,
        },
    }


@pytest.mark.parametrize("m", [1, 32, 128, 16960])
def test_fixed_8x4_layout_ignores_m(monkeypatch: pytest.MonkeyPatch, m: int) -> None:
    monkeypatch.setenv("VLLM_MXFP8_DENSE_A_SF_LAYOUT", "8x4")

    assert _load_layout_policy()(m) is True


@pytest.mark.parametrize("m", [1, 32, 128, 16960])
def test_fixed_128x4_layout_ignores_m(monkeypatch: pytest.MonkeyPatch, m: int) -> None:
    monkeypatch.setenv("VLLM_MXFP8_DENSE_A_SF_LAYOUT", "128x4")

    assert _load_layout_policy()(m) is False


def test_adaptive_layout_uses_configured_m_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_MXFP8_DENSE_A_SF_LAYOUT", "adaptive")
    monkeypatch.setenv("VLLM_MXFP8_DENSE_A_SF_LAYOUT_SWITCH_M", "128")
    policy = _load_layout_policy()

    assert policy(128) is True
    assert policy(129) is False


def test_adaptive_layout_defaults_to_256_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_MXFP8_DENSE_A_SF_LAYOUT", "adaptive")
    monkeypatch.delenv("VLLM_MXFP8_DENSE_A_SF_LAYOUT_SWITCH_M", raising=False)
    policy = _load_layout_policy()

    assert policy(256) is True
    assert policy(257) is False


def test_adaptive_layout_accepts_submitter_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_MXFP8_DENSE_A_SF_LAYOUT", "shape_aware")
    policy = _load_layout_policy()

    assert policy(256) is True
    assert policy(257) is False


def test_adaptive_layout_rejects_nonpositive_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_MXFP8_DENSE_A_SF_LAYOUT", "adaptive")
    monkeypatch.setenv("VLLM_MXFP8_DENSE_A_SF_LAYOUT_SWITCH_M", "0")

    with pytest.raises(ValueError, match="must be positive"):
        _load_layout_policy()(32)


def test_config_file_alone_selects_adaptive_layout_policy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path = tmp_path / "adaptive.json"
    config_path.write_text(json.dumps(_runtime_manifest()), encoding="utf-8")
    monkeypatch.setenv("VLLM_MXFP8_DENSE_CONFIG_FILE", str(config_path))
    for legacy_name in (
        "VLLM_MXFP8_DENSE_A_SF_LAYOUT",
        "VLLM_MXFP8_DENSE_A_SF_LAYOUT_SWITCH_M",
        "VLLM_MXFP8_DENSE_GEMM_BACKEND",
    ):
        monkeypatch.delenv(legacy_name, raising=False)
    _, current_fingerprint, _ = _load_trtllm_configuration_contract()
    prepared = current_fingerprint()
    runtime_module = SimpleNamespace(
        get_mxfp8_trtllm_file_configuration=lambda: prepared
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.utils.flashinfer",
        runtime_module,
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.utils",
        SimpleNamespace(flashinfer=runtime_module),
    )

    policy = _load_layout_policy()

    assert policy(1) is True
    assert policy(256) is True
    assert policy(257) is False


def test_frozen_config_keeps_layout_policy_after_environment_change(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path = tmp_path / "adaptive.json"
    config_path.write_text(json.dumps(_runtime_manifest()), encoding="utf-8")
    monkeypatch.setenv("VLLM_MXFP8_DENSE_CONFIG_FILE", str(config_path))
    _, current_fingerprint, _ = _load_trtllm_configuration_contract()
    prepared = current_fingerprint()
    runtime_module = SimpleNamespace(
        get_mxfp8_trtllm_file_configuration=lambda: prepared
    )
    monkeypatch.setitem(sys.modules, "vllm.utils.flashinfer", runtime_module)
    monkeypatch.setitem(
        sys.modules,
        "vllm.utils",
        SimpleNamespace(flashinfer=runtime_module),
    )
    monkeypatch.delenv("VLLM_MXFP8_DENSE_CONFIG_FILE")
    monkeypatch.setenv("VLLM_MXFP8_DENSE_A_SF_LAYOUT", "128x4")

    policy = _load_layout_policy()

    assert policy(1) is True
    assert policy(256) is True
    assert policy(257) is False


def test_config_file_alone_selects_linear_backend_and_unpadded_shape(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path = tmp_path / "adaptive.json"
    config_path.write_text(json.dumps(_runtime_manifest()), encoding="utf-8")
    monkeypatch.setenv("VLLM_MXFP8_DENSE_CONFIG_FILE", str(config_path))
    monkeypatch.delenv("VLLM_MXFP8_DENSE_GEMM_BACKEND", raising=False)
    monkeypatch.delenv("VLLM_MXFP8_DENSE_PAD_TO_128", raising=False)
    _, current_fingerprint, _ = _load_trtllm_configuration_contract()
    prepared = current_fingerprint()
    fake_runtime = SimpleNamespace(
        get_mxfp8_trtllm_file_configuration=lambda: prepared
    )
    tree = ast.parse(LINEAR.read_text(encoding="utf-8"))
    names = {"_mxfp8_dense_backend", "_mxfp8_dense_pad_to_128"}
    selected = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert {node.name for node in selected} == names
    namespace: dict[str, object] = {
        "os": os,
        "vllm_flashinfer": fake_runtime,
        "_SUPPORTED_MXFP8_DENSE_BACKENDS": (
            "cutlass",
            "trtllm",
            "cute-dsl",
            "auto",
        ),
    }
    exec(
        compile(ast.Module(body=selected, type_ignores=[]), str(LINEAR), "exec"),
        namespace,
    )

    assert namespace["_mxfp8_dense_backend"]() == "trtllm"  # type: ignore[operator]
    assert namespace["_mxfp8_dense_pad_to_128"]() is False  # type: ignore[operator]


def test_frozen_config_keeps_linear_policy_after_environment_change(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path = tmp_path / "adaptive.json"
    config_path.write_text(json.dumps(_runtime_manifest()), encoding="utf-8")
    monkeypatch.setenv("VLLM_MXFP8_DENSE_CONFIG_FILE", str(config_path))
    _, current_fingerprint, _ = _load_trtllm_configuration_contract()
    prepared = current_fingerprint()
    fake_runtime = SimpleNamespace(
        get_mxfp8_trtllm_file_configuration=lambda: prepared
    )
    monkeypatch.delenv("VLLM_MXFP8_DENSE_CONFIG_FILE")
    monkeypatch.setenv("VLLM_MXFP8_DENSE_GEMM_BACKEND", "cutlass")
    monkeypatch.setenv("VLLM_MXFP8_DENSE_PAD_TO_128", "1")
    tree = ast.parse(LINEAR.read_text(encoding="utf-8"))
    names = {"_mxfp8_dense_backend", "_mxfp8_dense_pad_to_128"}
    selected = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    namespace: dict[str, object] = {
        "os": os,
        "vllm_flashinfer": fake_runtime,
        "_SUPPORTED_MXFP8_DENSE_BACKENDS": (
            "cutlass",
            "trtllm",
            "cute-dsl",
            "auto",
        ),
    }
    exec(
        compile(ast.Module(body=selected, type_ignores=[]), str(LINEAR), "exec"),
        namespace,
    )

    assert namespace["_mxfp8_dense_backend"]() == "trtllm"  # type: ignore[operator]
    assert namespace["_mxfp8_dense_pad_to_128"]() is False  # type: ignore[operator]


def test_config_file_alone_selects_quant_backend_and_strict_8x4(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest = _runtime_manifest()
    manifest["policy"]["quant_backend"] = "flashinfer"
    config_path = tmp_path / "adaptive.json"
    config_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setenv("VLLM_MXFP8_DENSE_CONFIG_FILE", str(config_path))
    monkeypatch.delenv("VLLM_MXFP8_DENSE_QUANT_BACKEND", raising=False)
    monkeypatch.delenv("VLLM_MXFP8_DENSE_REQUIRE_8X4_QUANT", raising=False)
    _, current_fingerprint, _ = _load_trtllm_configuration_contract()
    prepared = current_fingerprint()
    runtime_module = SimpleNamespace(
        get_mxfp8_trtllm_file_configuration=lambda: prepared
    )
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(__path__=[]))
    monkeypatch.setitem(
        sys.modules,
        "vllm.utils",
        SimpleNamespace(flashinfer=runtime_module),
    )
    native_scales = _FakeScale(32)

    def mxfp8_quantize(*args: object, **kwargs: object) -> tuple[object, object]:
        if kwargs.get("sf_swizzle_layout") is not None:
            raise TypeError("native 8x4 selection unavailable")
        return object(), native_scales

    monkeypatch.setitem(
        sys.modules,
        "flashinfer",
        SimpleNamespace(
            SfLayout=SimpleNamespace(layout_8x4=object()),
            mxfp8_quantize=mxfp8_quantize,
        ),
    )
    quantize = _load_fixed_quantizer(
        "_flashinfer_mxfp8_quantize_impl", _FakeTorch()
    )
    fake_input = SimpleNamespace(ndim=2, shape=(1, 128))

    with pytest.raises(TypeError, match="native 8x4"):
        quantize(  # type: ignore[call-arg]
            fake_input,
            True,
            32,
            True,
            require_exact_8x4_layout=False,
        )


def test_quantizer_accepts_explicit_layout_choice() -> None:
    source = UTILS.read_text(encoding="utf-8")

    assert "use_8x4_sf_layout: bool" in source
    assert "sf_swizzle_layout" in source
    assert "m_tile = 8 if use_8x4_sf_layout else 128" in source


def test_runner_and_linear_path_share_the_same_layout_choice() -> None:
    runner_source = FLASHINFER_UTILS.read_text(encoding="utf-8")
    linear_source = LINEAR.read_text(encoding="utf-8")

    assert "use_8x4_sf_layout: bool" in runner_source
    assert "VLLM_MXFP8_DENSE_TRTLLM_TACTIC_HINTS_128X4" in runner_source
    assert "mxfp8_dense_use_8x4_sf_layout(M_padded)" in linear_source
    assert "use_8x4_sf_layout=use_8x4_sf_layout" in linear_source


def test_tactic_hint_table_is_parsed_once_and_looked_up_by_shape() -> None:
    source = FLASHINFER_UTILS.read_text(encoding="utf-8")

    assert "@functools.cache\ndef _parse_mxfp8_tactic_hints" in source
    assert "tactic_table.get((m, n, k), tactic)" in source
    direct_path = source.split("def mm_mxfp8(", 1)[1].split(
        "@torch.library.register_fake", 1
    )[0]
    assert 'for item in tactic_hints_raw.split(";")' not in direct_path


def test_adaptive_layout_has_separate_fixed_direct_quantizers() -> None:
    quantizer_source = UTILS.read_text(encoding="utf-8")

    assert "def mxfp8_e4m3_quantize_8x4_impl(" in quantizer_source
    assert "def mxfp8_e4m3_quantize_128x4_impl(" in quantizer_source
    assert 'op_name="mxfp8_quantize_8x4"' in quantizer_source
    assert 'op_name="mxfp8_quantize_128x4"' in quantizer_source
    assert "use_8x4_sf_layout=True" in quantizer_source
    assert "use_8x4_sf_layout=False" in quantizer_source

    fixed_quantizers = quantizer_source.split("def mxfp8_e4m3_quantize_8x4_impl(", 1)[1]
    fixed_quantizers = fixed_quantizers.split("def dequant_mxfp8_to_bf16(", 1)[0]
    assert "torch.ops" not in fixed_quantizers


def _fixed_quantizer_result(
    monkeypatch: pytest.MonkeyPatch,
    *,
    quantizer_name: str,
    m: int,
    native_scale_numel: int,
) -> tuple[object, _FakeScale]:
    native_scales = _FakeScale(native_scale_numel)
    flashinfer = SimpleNamespace(
        SfLayout=SimpleNamespace(layout_8x4=object()),
        mxfp8_quantize=lambda **_kwargs: (object(), native_scales),
    )
    monkeypatch.setitem(sys.modules, "flashinfer", flashinfer)
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(__path__=[]))
    monkeypatch.setitem(
        sys.modules,
        "vllm.platforms",
        SimpleNamespace(
            current_platform=SimpleNamespace(
                has_device_capability=lambda _capability: True
            )
        ),
    )
    quantize = _load_fixed_quantizer(quantizer_name, _FakeTorch())
    scales = quantize(SimpleNamespace(ndim=2, shape=(m, 128)))[1]

    return scales, native_scales


def test_low_m_fixed_8x4_quantizer_returns_compact_native_scales(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scales, native_scales = _fixed_quantizer_result(
        monkeypatch,
        quantizer_name="mxfp8_e4m3_quantize_8x4_impl",
        m=1,
        native_scale_numel=32,
    )

    assert scales is native_scales


def test_high_m_fixed_128x4_quantizer_returns_native_scales(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scales, native_scales = _fixed_quantizer_result(
        monkeypatch,
        quantizer_name="mxfp8_e4m3_quantize_128x4_impl",
        m=128,
        native_scale_numel=512,
    )

    assert scales is native_scales


@pytest.mark.parametrize("forbidden_normalization", ["torch.full(", ".copy_("])
def test_fixed_layout_quantization_source_has_no_scale_normalization(
    forbidden_normalization: str,
) -> None:
    source = UTILS.read_text(encoding="utf-8")

    assert forbidden_normalization not in source


def test_adaptive_layout_removes_composite_runner_state() -> None:
    runner_source = FLASHINFER_UTILS.read_text(encoding="utf-8")

    assert "_mxfp8_shape_specialized_quantize_mm_impl" not in runner_source
    assert '"vllm::mxfp8_shape_specialized_quantize_mm"' not in runner_source


def test_adaptive_layout_prepares_both_direct_runners_before_capture() -> None:
    runner_source = FLASHINFER_UTILS.read_text(encoding="utf-8")
    linear_source = LINEAR.read_text(encoding="utf-8")
    prepare = _function_source(FLASHINFER_UTILS, "prepare_mxfp8_trtllm_direct_state")
    process_weights = _function_source(LINEAR, "process_weights_after_loading")

    assert "_MXFP8_TRTLLM_DIRECT_STATES" in runner_source
    assert "_get_cache_buf(" in prepare
    assert "use_8x4_sf_layout=True" in prepare
    assert "use_8x4_sf_layout=False" in prepare
    assert "torch.cuda.is_current_stream_capturing()" in prepare
    assert "prepare_mxfp8_trtllm_direct_state(" in process_weights
    assert "weight.device" in process_weights
    assert "prepare_mxfp8_trtllm_direct_state(" in linear_source


def test_direct_trtllm_capture_requires_prepared_device_state() -> None:
    require_state = _function_source(
        FLASHINFER_UTILS, "_require_mxfp8_trtllm_direct_state"
    )
    direct_run = _function_source(FLASHINFER_UTILS, "_mxfp8_trtllm_run_prepared")

    assert "_MXFP8_TRTLLM_DIRECT_STATES.get(device_key)" in require_state
    assert "was not prepared before CUDA Graph capture" in require_state
    assert "_require_mxfp8_trtllm_direct_state(A.device)" in direct_run


def test_prepare_direct_state_checks_capture_on_target_cuda_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cuda = _FakeCuda(current_device=0, capturing_devices={1})
    contract = _load_direct_state_contract(cuda)
    _install_fake_flashinfer_gemm(monkeypatch)
    prepare = contract["prepare_mxfp8_trtllm_direct_state"]

    with pytest.raises(RuntimeError, match="before CUDA Graph capture"):
        prepare(_FakeCudaDevice("cuda:1"))  # type: ignore[operator]


def test_prepare_direct_state_rejects_promoting_nonadaptive_state_to_adaptive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cuda = _FakeCuda(current_device=1, capturing_devices=set())
    contract = _load_direct_state_contract(cuda)
    prepared_state_type = contract["_Mxfp8TrtllmDirectState"]
    prepared_states = contract["_MXFP8_TRTLLM_DIRECT_STATES"]
    prepared_states[("cuda", 1)] = prepared_state_type(  # type: ignore[index,operator]
        object(), object(), object(), object(), None, {}, {}
    )
    monkeypatch.setenv("VLLM_MXFP8_DENSE_A_SF_LAYOUT", "adaptive")
    monkeypatch.setenv("VLLM_MXFP8_DENSE_GEMM_BACKEND", "trtllm")
    prepare = contract["prepare_mxfp8_trtllm_direct_state"]

    with pytest.raises(RuntimeError):
        prepare(_FakeCudaDevice("cuda:1"))  # type: ignore[operator]


def test_fixed_8x4_quantization_rejects_backend_mutation_after_preparation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backends: list[str] = []
    fake_scales = _FakeScale(32)
    flashinfer = SimpleNamespace(
        SfLayout=SimpleNamespace(layout_8x4=object()),
        mxfp8_quantize=lambda **kwargs: (
            backends.append(kwargs["backend"]),
            fake_scales,
        ),
    )
    monkeypatch.setitem(sys.modules, "flashinfer", flashinfer)
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(__path__=[]))
    monkeypatch.setitem(
        sys.modules,
        "vllm.platforms",
        SimpleNamespace(
            current_platform=SimpleNamespace(
                has_device_capability=lambda _capability: True
            )
        ),
    )
    quantize_8x4 = _load_fixed_quantizer(
        "mxfp8_e4m3_quantize_8x4_impl", _FakeTorch()
    )
    fake_input = SimpleNamespace(ndim=2, shape=(1, 128))
    monkeypatch.setenv("VLLM_MXFP8_DENSE_QUANT_BACKEND", "cuda")
    quantize_8x4(fake_input)
    monkeypatch.setenv("VLLM_MXFP8_DENSE_QUANT_BACKEND", "triton")

    with pytest.raises(RuntimeError):
        quantize_8x4(fake_input)


def test_adaptive_layout_forces_direct_trtllm_without_generic_fallback() -> None:
    direct_path = (
        FLASHINFER_UTILS.read_text(encoding="utf-8")
        .split("def mm_mxfp8(", 1)[1]
        .split("@torch.library.register_fake", 1)[0]
    )

    assert "use_direct_trtllm = is_adaptive_layout or (" in direct_path
    assert "require_direct = is_adaptive_layout or" in direct_path
    assert "MXFP8 adaptive layout requires backend='trtllm'" in direct_path
    fallback_index = direct_path.index("return mm_mxfp8_(")
    assert direct_path.index("if is_adaptive_layout:", fallback_index - 250) < (
        fallback_index
    )


def test_adaptive_fingerprint_covers_every_direct_execution_input() -> None:
    fingerprint_source = _function_source(
        FLASHINFER_UTILS, "_mxfp8_trtllm_configuration_fingerprint"
    )
    fingerprint_type = next(
        node
        for node in ast.parse(FLASHINFER_UTILS.read_text(encoding="utf-8")).body
        if isinstance(node, ast.ClassDef)
        and node.name == "_Mxfp8TrtllmConfigurationFingerprint"
    )
    field_names = {
        node.target.id
        for node in fingerprint_type.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    }

    assert {
        "model",
        "tensor_parallel_size",
        "layout_mode",
        "switch_m",
        "gemm_backend",
        "direct_trtllm",
        "require_direct_trtllm",
        "default_tactic",
        "low_tactic_hints_raw",
        "low_tactic_map",
        "high_tactic_hints_raw",
        "high_tactic_map",
        "quant_backend",
        "require_8x4_quant",
        "pad_to_128",
    } <= field_names
    assert "MXFP8 adaptive layout requires backend='trtllm'" in fingerprint_source


def test_adaptive_layout_configures_pre_capture_range_specialization() -> None:
    runner_source = FLASHINFER_UTILS.read_text(encoding="utf-8")
    linear_source = LINEAR.read_text(encoding="utf-8")

    assert "def configure_mxfp8_adaptive_layout_compilation(" in runner_source
    assert "compile_ranges_endpoints" in runner_source
    assert "active_configuration.switch_m" in runner_source
    configure = _function_source(
        FLASHINFER_UTILS, "configure_mxfp8_adaptive_layout_compilation"
    )
    assert 'pass_key = "joint_custom_pre_pass"' in configure
    assert "current_platform.pass_key" not in configure
    assert "_Mxfp8AdaptiveLayoutSpecializationPass" in runner_source
    assert '"phase": "joint_custom_pre_pass"' in runner_source
    assert '"schema_version": 7' in runner_source
    assert "configure_mxfp8_adaptive_layout_compilation()" in linear_source


def test_prepared_state_fingerprint_fails_closed_on_configuration_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_MXFP8_DENSE_A_SF_LAYOUT", "adaptive")
    monkeypatch.setenv("VLLM_MXFP8_DENSE_GEMM_BACKEND", "trtllm")
    monkeypatch.setenv("VLLM_MXFP8_DENSE_DIRECT_TRTLLM", "0")
    monkeypatch.setenv("VLLM_MXFP8_DENSE_REQUIRE_DIRECT_TRTLLM", "0")
    monkeypatch.setenv("VLLM_MXFP8_DENSE_A_SF_LAYOUT_SWITCH_M", "128")
    monkeypatch.setenv("VLLM_MXFP8_DENSE_TRTLLM_TACTIC", "-1")
    monkeypatch.setenv(
        "VLLM_MXFP8_DENSE_TRTLLM_TACTIC_HINTS",
        " 32,64,256:7 ; ignored ",
    )
    monkeypatch.setenv(
        "VLLM_MXFP8_DENSE_TRTLLM_TACTIC_HINTS_128X4",
        "1000,64,256:11",
    )
    monkeypatch.setenv("VLLM_MXFP8_DENSE_QUANT_BACKEND", "cuda")
    fingerprint_type, current_fingerprint, validate = (
        _load_trtllm_configuration_contract()
    )

    prepared = current_fingerprint()
    assert isinstance(prepared, fingerprint_type)
    assert prepared.switch_m == 128
    assert prepared.gemm_backend == "trtllm"
    assert prepared.direct_trtllm is False
    assert prepared.require_direct_trtllm is False
    assert prepared.default_tactic == -1
    assert prepared.low_tactic_hints_raw == " 32,64,256:7 ; ignored "
    assert prepared.low_tactic_map == (((32, 64, 256), 7),)
    assert prepared.high_tactic_map == (((1000, 64, 256), 11),)
    assert prepared.quant_backend == "cuda"
    with pytest.raises(AttributeError):
        prepared.switch_m = 65

    monkeypatch.setenv("VLLM_MXFP8_DENSE_A_SF_LAYOUT_SWITCH_M", "256")
    with pytest.raises(RuntimeError, match="configuration changed after preparation"):
        validate(prepared, current_fingerprint())

    monkeypatch.setenv("VLLM_MXFP8_DENSE_A_SF_LAYOUT_SWITCH_M", "128")
    monkeypatch.setenv(
        "VLLM_MXFP8_DENSE_TRTLLM_TACTIC_HINTS",
        "32,64,256:7",
    )
    raw_changed = current_fingerprint()
    assert raw_changed.low_tactic_map == prepared.low_tactic_map
    with pytest.raises(RuntimeError, match="configuration changed after preparation"):
        validate(prepared, raw_changed)

    monkeypatch.setenv("VLLM_MXFP8_DENSE_TRTLLM_TACTIC", "7")
    with pytest.raises(RuntimeError, match="configuration changed after preparation"):
        validate(prepared, current_fingerprint())


def test_config_file_fingerprint_freezes_resolved_path_sha_and_complete_policy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path = tmp_path / "qualified.json"
    config_path.write_text(json.dumps(_runtime_manifest()), encoding="utf-8")
    monkeypatch.setenv("VLLM_MXFP8_DENSE_CONFIG_FILE", str(config_path))
    for legacy_name in (
        "VLLM_MXFP8_DENSE_A_SF_LAYOUT",
        "VLLM_MXFP8_DENSE_A_SF_LAYOUT_SWITCH_M",
        "VLLM_MXFP8_DENSE_GEMM_BACKEND",
        "VLLM_MXFP8_DENSE_DIRECT_TRTLLM",
        "VLLM_MXFP8_DENSE_REQUIRE_DIRECT_TRTLLM",
        "VLLM_MXFP8_DENSE_TRTLLM_TACTIC",
        "VLLM_MXFP8_DENSE_QUANT_BACKEND",
        "VLLM_MXFP8_DENSE_REQUIRE_8X4_QUANT",
        "VLLM_MXFP8_DENSE_PAD_TO_128",
    ):
        monkeypatch.delenv(legacy_name, raising=False)
    _, current_fingerprint, _ = _load_trtllm_configuration_contract()

    fingerprint = current_fingerprint()

    assert fingerprint.config_path == str(config_path.resolve())
    assert fingerprint.config_sha256 == hashlib.sha256(
        config_path.read_bytes()
    ).hexdigest()
    assert fingerprint.model == "Nemotron 3 Ultra MXFP8"
    assert fingerprint.tensor_parallel_size == 4
    assert fingerprint.layout_mode == "adaptive"
    assert fingerprint.switch_m == 256
    assert fingerprint.gemm_backend == "trtllm"
    assert fingerprint.direct_trtllm is True
    assert fingerprint.require_direct_trtllm is True
    assert fingerprint.quant_backend == "cuda"
    assert fingerprint.require_8x4_quant is True
    assert fingerprint.pad_to_128 is False
    assert fingerprint.low_tactic_hints_raw == ""
    assert fingerprint.low_tactic_map == (
        ((1, 2048, 8192), 66),
        ((32, 8192, 2048), 71),
    )
    assert fingerprint.high_tactic_hints_raw == ""
    assert fingerprint.high_tactic_map == (((1000, 2048, 8192), 70),)
    assert fingerprint.qualification_scope == "standalone_serving_seed"


@pytest.mark.parametrize(
    "legacy_tactics",
    (
        "VLLM_MXFP8_DENSE_TRTLLM_TACTIC_HINTS",
        "VLLM_MXFP8_DENSE_TRTLLM_TACTIC_HINTS_128X4",
    ),
)
def test_config_file_rejects_simultaneous_inline_tactics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    legacy_tactics: str,
) -> None:
    config_path = tmp_path / "qualified.json"
    config_path.write_text(json.dumps(_runtime_manifest()), encoding="utf-8")
    monkeypatch.setenv("VLLM_MXFP8_DENSE_CONFIG_FILE", str(config_path))
    monkeypatch.setenv(legacy_tactics, "1,2048,8192:66")
    _, current_fingerprint, _ = _load_trtllm_configuration_contract()

    with pytest.raises(ValueError, match=legacy_tactics):
        current_fingerprint()


def test_config_file_runtime_rejects_non_minus_one_default_tactic(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest = _runtime_manifest()
    manifest["policy"]["default_tactic"] = 7
    config_path = tmp_path / "invalid-default.json"
    config_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setenv("VLLM_MXFP8_DENSE_CONFIG_FILE", str(config_path))
    _, current_fingerprint, _ = _load_trtllm_configuration_contract()

    with pytest.raises(ValueError, match="policy.default_tactic"):
        current_fingerprint()


def test_config_file_byte_mutation_fails_frozen_fingerprint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path = tmp_path / "qualified.json"
    manifest = _runtime_manifest()
    config_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setenv("VLLM_MXFP8_DENSE_CONFIG_FILE", str(config_path))
    _, current_fingerprint, validate = _load_trtllm_configuration_contract()
    prepared = current_fingerprint()

    config_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    active = current_fingerprint()

    assert active.config_sha256 != prepared.config_sha256
    with pytest.raises(RuntimeError, match="configuration changed after preparation"):
        validate(prepared, active)


def test_hot_path_uses_frozen_configuration_without_source_io(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path = tmp_path / "qualified.json"
    manifest = _runtime_manifest()
    config_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setenv("VLLM_MXFP8_DENSE_CONFIG_FILE", str(config_path))
    loader_calls: list[str] = []
    context_calls: list[None] = []
    namespace = _load_post_freeze_validation_contract(
        loader_calls, context_calls
    )
    get_configuration = namespace["get_mxfp8_trtllm_configuration"]
    get_file_configuration = namespace[
        "get_mxfp8_trtllm_file_configuration"
    ]
    validate = namespace["validate_mxfp8_trtllm_configuration"]
    validate_source = namespace[
        "validate_mxfp8_trtllm_configuration_source"
    ]

    prepared = get_configuration()  # type: ignore[operator]
    validate_source(prepared)  # type: ignore[operator]
    assert len(loader_calls) == 2
    assert len(context_calls) == 1

    for _ in range(5):
        validate(prepared)  # type: ignore[operator]
        assert get_configuration() is prepared  # type: ignore[operator]
        assert get_file_configuration() is prepared  # type: ignore[operator]
    assert len(loader_calls) == 2
    assert len(context_calls) == 1

    config_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    monkeypatch.delenv("VLLM_MXFP8_DENSE_CONFIG_FILE")
    validate(prepared)  # type: ignore[operator]
    assert get_configuration() is prepared  # type: ignore[operator]
    assert get_file_configuration() is prepared  # type: ignore[operator]
    assert len(loader_calls) == 2
    assert len(context_calls) == 1

    monkeypatch.setenv("VLLM_MXFP8_DENSE_CONFIG_FILE", str(config_path))
    with pytest.raises(RuntimeError, match="configuration changed after preparation"):
        validate_source(prepared)  # type: ignore[operator]
    assert len(loader_calls) == 3
    assert len(context_calls) == 1

    other_path = tmp_path / "other-qualified.json"
    other_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setenv("VLLM_MXFP8_DENSE_CONFIG_FILE", str(other_path))
    validate(prepared)  # type: ignore[operator]
    assert get_file_configuration() is prepared  # type: ignore[operator]
    assert len(loader_calls) == 3
    with pytest.raises(RuntimeError, match="configuration changed after preparation"):
        validate_source(prepared)  # type: ignore[operator]
    assert len(loader_calls) == 4
    assert len(context_calls) == 1


def test_eager_gemm_checks_only_frozen_configuration() -> None:
    apply_weights = _function_source(LINEAR, "apply_weights")
    require_state = _function_source(
        FLASHINFER_UTILS, "_require_mxfp8_trtllm_direct_state"
    )
    run_prepared = _function_source(
        FLASHINFER_UTILS, "_mxfp8_trtllm_run_prepared"
    )

    for source in (apply_weights, require_state, run_prepared):
        assert "validate_mxfp8_trtllm_configuration_source" not in source
        assert "_mxfp8_trtllm_configuration_fingerprint" not in source


def test_config_file_logs_startup_provenance_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path = tmp_path / "qualified.json"
    config_path.write_text(json.dumps(_runtime_manifest()), encoding="utf-8")
    monkeypatch.setenv("VLLM_MXFP8_DENSE_CONFIG_FILE", str(config_path))
    messages: list[str] = []
    get_configuration = _load_trtllm_runtime_freeze_contract(messages)

    get_configuration()
    get_configuration()

    assert len(messages) == 1
    assert f"path={config_path.resolve()}" in messages[0]
    assert (
        f"sha256={hashlib.sha256(config_path.read_bytes()).hexdigest()}"
        in messages[0]
    )
    assert "mode=adaptive" in messages[0]
    assert "switch_m=256" in messages[0]
    assert "tactics_8x4=2" in messages[0]
    assert "tactics_128x4=1" in messages[0]
    assert "qualification_scope=standalone_serving_seed" in messages[0]


def test_frozen_configuration_getter_does_not_reload_during_capture() -> None:
    getter_source = _function_source(
        FLASHINFER_UTILS, "get_mxfp8_trtllm_configuration"
    )
    frozen = object()
    namespace: dict[str, object] = {
        "_Mxfp8TrtllmConfigurationFingerprint": object,
        "_MXFP8_TRTLLM_CONFIGURATION": frozen,
        "_mxfp8_trtllm_configuration_fingerprint": (
            lambda: pytest.fail("capture path reloaded the config file")
        ),
        "_freeze_mxfp8_trtllm_configuration": (
            lambda _configuration: pytest.fail("capture path refroze configuration")
        ),
        "torch": SimpleNamespace(
            cuda=SimpleNamespace(is_current_stream_capturing=lambda: True)
        ),
    }
    exec(getter_source, namespace)

    assert namespace["get_mxfp8_trtllm_configuration"]() is frozen  # type: ignore[operator]


def test_adaptive_switch_m_must_match_physical_128_row_padding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_MXFP8_DENSE_A_SF_LAYOUT", "adaptive")
    monkeypatch.setenv("VLLM_MXFP8_DENSE_GEMM_BACKEND", "trtllm")
    monkeypatch.setenv("VLLM_MXFP8_DENSE_A_SF_LAYOUT_SWITCH_M", "129")
    _, current_fingerprint, _ = _load_trtllm_configuration_contract()

    with pytest.raises(ValueError, match="multiple of 128"):
        current_fingerprint()


def test_adaptive_fingerprint_rejects_non_trtllm_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_MXFP8_DENSE_A_SF_LAYOUT", "adaptive")
    monkeypatch.setenv("VLLM_MXFP8_DENSE_GEMM_BACKEND", "cutlass")
    _, current_fingerprint, _ = _load_trtllm_configuration_contract()

    with pytest.raises(RuntimeError, match="requires backend='trtllm'"):
        current_fingerprint()


def test_compilation_configuration_validates_active_environment() -> None:
    source = FLASHINFER_UTILS.read_text(encoding="utf-8")
    configure = _function_source(
        FLASHINFER_UTILS, "configure_mxfp8_adaptive_layout_compilation"
    )

    assert "_Mxfp8TrtllmConfigurationFingerprint" in source
    assert configure.index(
        "validate_mxfp8_trtllm_configuration_source("
    ) < configure.index("compilation_config.inductor_compile_config")
    assert "get_mxfp8_trtllm_configuration()" in configure


def test_adaptive_layout_uses_separate_tactic_tables_with_unknown_minus_one() -> None:
    source = FLASHINFER_UTILS.read_text(encoding="utf-8")

    direct_path = source.split("def mm_mxfp8(", 1)[1].split(
        "@torch.library.register_fake", 1
    )[0]
    assert '"VLLM_MXFP8_DENSE_TRTLLM_TACTIC_HINTS"' in direct_path
    assert '"VLLM_MXFP8_DENSE_TRTLLM_TACTIC_HINTS_128X4"' in direct_path
    assert "tactic_table.get((m, n, k), tactic)" in direct_path
    fixed_path = _function_source(
        FLASHINFER_UTILS, "_mxfp8_quantize_mm_fixed_layout_impl"
    )
    assert "tactic_table.get(shape_key, -1)" in fixed_path
    assert "configuration.default_tactic" not in fixed_path


def test_adaptive_dispatch_trace_records_layout_tactic_and_lookup_source(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    trace_source = _function_source(
        FLASHINFER_UTILS, "_trace_mxfp8_adaptive_dispatch"
    )
    namespace: dict[str, object] = {
        "json": json,
        "os": os,
        "Path": Path,
        "socket": socket,
        "time": time,
        "torch": SimpleNamespace(
            cuda=SimpleNamespace(is_current_stream_capturing=lambda: False)
        ),
        "_MXFP8_ADAPTIVE_DISPATCH_TRACE_SEEN": set(),
    }
    exec(trace_source, namespace)
    trace = namespace["_trace_mxfp8_adaptive_dispatch"]
    monkeypatch.setenv("VLLM_MXFP8_DENSE_SHAPE_TRACE", "1")
    monkeypatch.setenv("VLLM_MXFP8_DENSE_SHAPE_TRACE_DIR", str(tmp_path))

    trace(  # type: ignore[operator]
        shape_key=(8480, 8832, 8192),
        use_8x4_sf_layout=False,
        tactic=73,
        tactic_hit=True,
        config_sha256="d" * 64,
    )
    trace(  # type: ignore[operator]
        shape_key=(8480, 8832, 8192),
        use_8x4_sf_layout=False,
        tactic=73,
        tactic_hit=True,
        config_sha256="d" * 64,
    )

    paths = list(tmp_path.glob("adaptive_dispatch_*.jsonl"))
    assert len(paths) == 1
    rows = paths[0].read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1
    assert json.loads(rows[0]) | {"time": None} == {
        "event": "mxfp8_adaptive_dispatch",
        "config_sha256": "d" * 64,
        "hostname": socket.gethostname(),
        "layout": "128x4",
        "m": 8480,
        "n": 8832,
        "k": 8192,
        "pid": os.getpid(),
        "tactic": 73,
        "tactic_source": "static_hint",
        "time": None,
    }


def test_adaptive_dispatch_trace_performs_no_io_during_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace_source = _function_source(
        FLASHINFER_UTILS, "_trace_mxfp8_adaptive_dispatch"
    )
    namespace: dict[str, object] = {
        "json": json,
        "os": os,
        "Path": lambda _path: pytest.fail("capture trace attempted filesystem I/O"),
        "socket": socket,
        "time": time,
        "torch": SimpleNamespace(
            cuda=SimpleNamespace(is_current_stream_capturing=lambda: True)
        ),
        "_MXFP8_ADAPTIVE_DISPATCH_TRACE_SEEN": set(),
    }
    exec(trace_source, namespace)
    monkeypatch.setenv("VLLM_MXFP8_DENSE_SHAPE_TRACE", "1")
    monkeypatch.setenv("VLLM_MXFP8_DENSE_SHAPE_TRACE_DIR", "/capture-forbidden")

    namespace["_trace_mxfp8_adaptive_dispatch"](  # type: ignore[operator]
        shape_key=(32, 8192, 2048),
        use_8x4_sf_layout=True,
        tactic=71,
        tactic_hit=True,
        config_sha256="f" * 64,
    )


def test_dense_shape_trace_records_frozen_config_sha256(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    trace_source = _function_source(LINEAR, "_mxfp8_dense_shape_trace")
    fake_torch = SimpleNamespace(
        nn=SimpleNamespace(Module=object),
        Size=tuple,
        cuda=SimpleNamespace(is_current_stream_capturing=lambda: False),
    )
    namespace: dict[str, object] = {
        "json": json,
        "os": os,
        "Path": Path,
        "socket": socket,
        "time": time,
        "torch": fake_torch,
        "_env_flag": lambda _name, _default: True,
        "_mxfp8_dense_is_compiling": lambda: False,
        "_MXFP8_DENSE_TRACE_SEEN": set(),
        "_MXFP8_DENSE_TRACE_WRITTEN": 0,
    }
    exec(trace_source, namespace)
    monkeypatch.setenv("VLLM_MXFP8_DENSE_SHAPE_TRACE_DIR", str(tmp_path))
    trace = namespace["_mxfp8_dense_shape_trace"]

    trace(  # type: ignore[operator]
        layer=SimpleNamespace(prefix="model.layers.0.mlp.down_proj"),
        family="FC2",
        m_logical=32,
        m_physical=32,
        n_logical=8192,
        n_physical=8192,
        k=2048,
        backend="trtllm",
        input_shape=(32, 2048),
        weight_shape=(8192, 2048),
        config_sha256="e" * 64,
    )

    trace_path = next(tmp_path.glob("dense_shapes_*.jsonl"))
    record = json.loads(trace_path.read_text(encoding="utf-8"))
    assert record["config_sha256"] == "e" * 64


def test_adaptive_fixed_ops_validate_marker_compatible_schemas() -> None:
    source = FLASHINFER_UTILS.read_text(encoding="utf-8")
    configure = _function_source(
        FLASHINFER_UTILS, "configure_mxfp8_adaptive_layout_compilation"
    )

    assert "def _validate_mxfp8_adaptive_op_schemas(" in source
    assert "_validate_mxfp8_adaptive_op_schemas()" in configure
    assert "mxfp8_adaptive_quantize_mm_marker.default._schema" in source
    assert "mxfp8_quantize_mm_8x4.default._schema" in source
    assert "mxfp8_quantize_mm_128x4.default._schema" in source


def test_direct_trtllm_workspace_is_owned_by_fixed_layout() -> None:
    prepare = _function_source(FLASHINFER_UTILS, "prepare_mxfp8_trtllm_direct_state")
    direct_run = _function_source(FLASHINFER_UTILS, "_mxfp8_trtllm_run_prepared")
    workspaces = _function_source(
        FLASHINFER_UTILS, "get_mxfp8_trtllm_prepared_workspaces"
    )
    assert '"vllm_direct_trtllm_mxfp8_workspace_8x4"' in prepare
    assert '"vllm_direct_trtllm_mxfp8_workspace_128x4"' in prepare
    assert "state.workspace_8x4" in workspaces
    assert "state.workspace_128x4" in workspaces
    assert "workspace: torch.Tensor" in direct_run
    assert "_get_cache_buf" not in direct_run


def test_adaptive_fused_ops_surface_mutable_prepared_workspaces() -> None:
    runner_source = FLASHINFER_UTILS.read_text(encoding="utf-8")
    linear_source = LINEAR.read_text(encoding="utf-8")

    assert "def get_mxfp8_trtllm_prepared_workspaces(" in runner_source
    assert "return state.workspace_8x4, state.workspace_128x4" in runner_source
    assert "layer._mxfp8_trtllm_workspace_8x4" in linear_source
    assert "layer._mxfp8_trtllm_workspace_128x4" in linear_source
    assert "torch.ops.vllm.mxfp8_adaptive_quantize_mm_marker(" in linear_source
    assert "torch.ops.vllm.mxfp8_adaptive_quantize_marker(" not in linear_source
    assert "torch.ops.vllm.mxfp8_adaptive_mm_marker(" not in linear_source
    assert '"vllm::mxfp8_quantize_mm_8x4"' in runner_source
    assert '"vllm::mxfp8_quantize_mm_128x4"' in runner_source
    assert 'mutates_args=["workspace_8x4", "workspace_128x4"]' in runner_source
    assert "[A, B, A_scale, B_scale, out_dtype, out, workspace]" in runner_source


def test_adaptive_layout_serving_graph_has_one_output_only_fused_op() -> None:
    runner_source = FLASHINFER_UTILS.read_text(encoding="utf-8")
    linear_source = LINEAR.read_text(encoding="utf-8")

    specialization = _function_source(
        FLASHINFER_UTILS, "_specialize_mxfp8_adaptive_layout_graph"
    )
    assert "mxfp8_adaptive_quantize_mm_marker.default" in runner_source
    assert "mxfp8_quantize_mm_8x4.default" in runner_source
    assert "mxfp8_quantize_mm_128x4.default" in runner_source
    assert "node.target = fixed_op" in specialization
    assert "graph.call_function(" not in specialization
    assert "torch.cond(" not in runner_source
    assert "torch.ops.vllm.mxfp8_adaptive_quantize_mm_marker" in linear_source
    assert "torch.ops.vllm.mxfp8_shape_specialized_quantize_mm" not in linear_source
    assert "input_scale" not in linear_source.split(
        "if is_adaptive_layout and is_compiling:", 1
    )[1].split("else:", 1)[0]


def test_adaptive_layout_fixed_fused_ops_use_native_quantizers() -> None:
    runner_source = FLASHINFER_UTILS.read_text(encoding="utf-8")
    fused_impl = _function_source(
        FLASHINFER_UTILS, "_mxfp8_quantize_mm_fixed_layout_impl"
    )

    assert '"vllm::mxfp8_quantize_mm_8x4"' in runner_source
    assert '"vllm::mxfp8_quantize_mm_128x4"' in runner_source
    assert "mxfp8_e4m3_quantize_8x4_impl" in fused_impl
    assert "mxfp8_e4m3_quantize_128x4_impl" in fused_impl
    assert "_mxfp8_trtllm_run_prepared(" in fused_impl
    assert "_normalize_mxfp8_adaptive_scale" not in fused_impl
    assert fused_impl.index("_require_mxfp8_trtllm_direct_state(") < fused_impl.index(
        "from vllm.model_executor.layers.quantization.utils.mxfp8_utils import"
    )


def test_adaptive_layout_fails_closed_on_backend_or_unspecialized_marker() -> None:
    runner_source = FLASHINFER_UTILS.read_text(encoding="utf-8")
    linear_source = LINEAR.read_text(encoding="utf-8")

    assert "MXFP8 adaptive layout requires the TRTLLM backend" in linear_source
    assert "must be specialized before execution" in runner_source
    assert "straddles adaptive layout switch" in runner_source


def test_compile_range_layout_choice_is_fixed_and_fails_on_straddle() -> None:
    runner_source = FLASHINFER_UTILS.read_text(encoding="utf-8")
    selector_source = _function_source(
        FLASHINFER_UTILS, "_mxfp8_layout_for_compile_range"
    )
    namespace: dict[str, object] = {}
    exec(selector_source, namespace)
    selector = namespace["_mxfp8_layout_for_compile_range"]

    assert selector(1, 256, 256) is True  # type: ignore[operator]
    assert selector(257, 16384, 256) is False  # type: ignore[operator]
    with pytest.raises(RuntimeError, match="straddles adaptive layout switch"):
        selector(1, 16384, 256)  # type: ignore[operator]
    assert "get_pass_context().compile_range" in runner_source


def test_graph_specialization_retargets_existing_fused_marker() -> None:
    transform_source = _function_source(
        FLASHINFER_UTILS, "_specialize_mxfp8_adaptive_layout_graph"
    )
    namespace: dict[str, object] = {"Any": Any}
    exec(transform_source, namespace)
    transform = namespace["_specialize_mxfp8_adaptive_layout_graph"]

    marker_op = object()
    fixed_op = object()
    marker_args = tuple(object() for _ in range(7))

    class FakeNode:
        def __init__(self, target: object, args: tuple[object, ...]) -> None:
            self.op = "call_function"
            self.target = target
            self.args = args
            self.meta: dict[str, object] = {}

    class FakeGraph:
        def __init__(self) -> None:
            self.marker = FakeNode(marker_op, marker_args)
            self.nodes = [self.marker]

    graph = FakeGraph()
    replaced = transform(  # type: ignore[operator]
        graph,
        marker_op=marker_op,
        fixed_op=fixed_op,
    )

    assert replaced == 1
    assert graph.marker.target is fixed_op
    assert graph.marker.args == marker_args


def test_graph_specialization_retargets_auto_functionalized_fused_marker() -> None:
    transform_source = _function_source(
        FLASHINFER_UTILS, "_specialize_mxfp8_adaptive_layout_graph"
    )
    namespace: dict[str, object] = {"Any": Any}
    exec(transform_source, namespace)
    transform = namespace["_specialize_mxfp8_adaptive_layout_graph"]

    marker_op = object()
    fixed_op = object()
    auto_functionalized_op = object()
    mm_kwargs = {
        "A": object(),
        "B": object(),
        "A_scale": object(),
        "B_scale": object(),
        "out_dtype": object(),
        "backend": "trtllm",
        "workspace_8x4": object(),
        "workspace_128x4": object(),
    }

    class FakeNode:
        def __init__(self) -> None:
            self.op = "call_function"
            self.target = auto_functionalized_op
            self.args = (marker_op,)
            self.kwargs = mm_kwargs
            self.meta = {"stack_trace": "auto-functionalized-mm"}

    class FakeGraph:
        def __init__(self) -> None:
            self.mm_marker = FakeNode()
            self.nodes = [self.mm_marker]

    graph = FakeGraph()
    replaced = transform(  # type: ignore[operator]
        graph,
        marker_op=marker_op,
        fixed_op=fixed_op,
        auto_functionalized_op=auto_functionalized_op,
    )

    assert replaced == 1
    assert graph.mm_marker.target is auto_functionalized_op
    assert graph.mm_marker.args == (fixed_op,)
    assert graph.mm_marker.kwargs is mm_kwargs
    assert graph.mm_marker.meta == {"stack_trace": "auto-functionalized-mm"}


def test_specialization_allows_quantize_and_mm_in_separate_compiler_graphs() -> None:
    source = FLASHINFER_UTILS.read_text(encoding="utf-8")
    specialization_pass = source.split(
        "class _Mxfp8AdaptiveLayoutSpecializationPass", 1
    )[1].split("def configure_mxfp8_adaptive_layout_compilation", 1)[0]

    assert "quantize_count != mm_count" not in specialization_pass


def test_graph_specialization_preserves_marker_metadata() -> None:
    transform_source = _function_source(
        FLASHINFER_UTILS, "_specialize_mxfp8_adaptive_layout_graph"
    )
    namespace: dict[str, object] = {"Any": Any}
    exec(transform_source, namespace)
    transform = namespace["_specialize_mxfp8_adaptive_layout_graph"]

    marker_op = object()
    output_value = object()
    fixed_op = object()

    class FakeNode:
        def __init__(self, target: object, args: tuple[object, ...]) -> None:
            self.op = "call_function"
            self.target = target
            self.args = args
            self.meta: dict[str, object] = {}

    input_node = FakeNode(object(), ())
    marker_args = (input_node, *tuple(object() for _ in range(6)))

    class FakeGraph:
        def __init__(self) -> None:
            self.marker = FakeNode(marker_op, marker_args)
            self.marker.meta = {
                "val": output_value,
                "stack_trace": "quantize-mm-marker",
                "nn_module_stack": {"linear": object()},
            }
            self.nodes = [self.marker]

    graph = FakeGraph()
    transform(  # type: ignore[operator]
        graph,
        marker_op=marker_op,
        fixed_op=fixed_op,
    )

    assert graph.marker.target is fixed_op
    assert graph.marker.meta["val"] is output_value
    assert graph.marker.meta["stack_trace"] == "quantize-mm-marker"
    assert "nn_module_stack" in graph.marker.meta
