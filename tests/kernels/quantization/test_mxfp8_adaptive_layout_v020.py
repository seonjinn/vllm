import ast
from contextlib import contextmanager
import functools
import json
import os
from pathlib import Path
import socket
import sys
import time
from typing import Any, Callable, NamedTuple
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[3]
OVERRIDE = ROOT / "vllm"
UTILS = OVERRIDE / "model_executor/layers/quantization/utils/mxfp8_utils.py"
FLASHINFER_UTILS = OVERRIDE / "utils/flashinfer.py"
LINEAR = OVERRIDE / "model_executor/kernels/linear/mxfp8/flashinfer.py"
def _load_layout_policy() -> Callable[[int], bool]:
    tree = ast.parse(UTILS.read_text(encoding="utf-8"))
    names = {
        "_mxfp8_dense_a_sf_layout",
        "mxfp8_dense_use_8x4_sf_layout",
    }
    selected = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in names
    ]
    namespace: dict[str, object] = {"os": os}
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
        "NamedTuple": NamedTuple,
        "os": os,
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
        "NamedTuple": NamedTuple,
        "os": os,
        "torch": fake_torch,
        "Any": Any,
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
        "os": os,
        "torch": torch_module,
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
        "_mxfp8_trtllm_configuration_fingerprint()"
    ) < configure.index("compilation_config.inductor_compile_config")
    assert "_validate_mxfp8_trtllm_configuration(" in configure


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
    )
    trace(  # type: ignore[operator]
        shape_key=(8480, 8832, 8192),
        use_8x4_sf_layout=False,
        tactic=73,
        tactic_hit=True,
    )

    paths = list(tmp_path.glob("adaptive_dispatch_*.jsonl"))
    assert len(paths) == 1
    rows = paths[0].read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1
    assert json.loads(rows[0]) | {"time": None} == {
        "event": "mxfp8_adaptive_dispatch",
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
