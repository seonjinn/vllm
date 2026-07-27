# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from vllm.model_executor.layers.quantization.utils import mxfp8_utils
from vllm.model_executor.layers.quantization.utils.mxfp8_tactic_table import (
    Mxfp8TacticArtifact,
    Mxfp8TacticKey,
    RuntimeProvenance,
)
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    _load_configured_mxfp8_tactic_artifact,
    _mxfp8_trtllm_linear_fixed_impl,
    _Mxfp8TacticAudit,
    _Mxfp8TrtllmTacticState,
    _resolve_mxfp8_trtllm_tactic,
    _run_mxfp8_trtllm_pre_resolved,
)


class FakeRunner:
    def __init__(self, valid_tactics: list[int]) -> None:
        self.valid_tactics = valid_tactics
        self.valid_tactic_inputs: list[list[Any]] = []
        self.valid_tactic_shapes: list[object] = []
        self.forward_inputs: list[list[Any]] = []
        self.forward_tactics: list[int] = []

    def get_valid_tactics(self, inputs: list[Any], profile: Any) -> list[int]:
        self.valid_tactic_inputs.append(inputs)
        self.valid_tactic_shapes.append(profile.get_opt_shapes())
        return self.valid_tactics

    def forward(self, inputs: list[Any], tactic: int) -> torch.Tensor:
        self.forward_inputs.append(inputs)
        self.forward_tactics.append(tactic)
        return inputs[-2]


def provenance() -> RuntimeProvenance:
    return RuntimeProvenance(
        vllm_version="0.26.0",
        flashinfer_version="0.6.14",
        torch_version="2.11.0+cu130",
        cuda_version="13.0",
        driver_version="580.65.06",
        gpu="NVIDIA GB200",
        topology="tp4",
        checkpoint_id="nvidia/Nemotron3-Ultra-30B-A3B-MXFP8",
        source_commit="0123456789abcdef0123456789abcdef01234567",
        container_digest=(
            "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
        ),
        adaptive_switch_m=256,
        weight_contract="trtllm_shuffled_b_sf_a_n128_v1",
    )


def key(*, layout: str, m: int = 32) -> Mxfp8TacticKey:
    return Mxfp8TacticKey(
        m_logical=m,
        n_logical=4384,
        k_logical=8192,
        n_physical=4480,
        k_physical=8192,
        activation_scale_layout=layout,
        output_dtype="bfloat16",
    )


def state(
    *,
    tactics: dict[Mxfp8TacticKey, int],
    runner_8x4: FakeRunner,
    runner_128x4: FakeRunner,
    artifact_enabled: bool = True,
) -> _Mxfp8TrtllmTacticState:
    return _Mxfp8TrtllmTacticState(
        artifact=(
            Mxfp8TacticArtifact(
                provenance=provenance(),
                tactics=MappingProxyType(tactics),
            )
            if artifact_enabled
            else None
        ),
        runner_8x4=runner_8x4,
        runner_128x4=runner_128x4,
        workspace_8x4=torch.Tensor(),
        workspace_128x4=torch.Tensor(),
        resolved_tactics={},
    )


def runner_inputs() -> list[object]:
    return [
        object(),
        object(),
        object(),
        object(),
        torch.bfloat16,
        object(),
        object(),
    ]


def configure_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    table_path: Path,
    table_sha256: str,
) -> None:
    monkeypatch.setattr(
        mxfp8_utils,
        "_MXFP8_ARTIFACT_CONFIGURATION_PRESENT",
        False,
    )
    provenance_contents = (
        json.dumps(asdict(provenance()), sort_keys=True).encode("utf-8") + b"\n"
    )
    provenance_path = tmp_path / "runtime-provenance.json"
    provenance_path.write_bytes(provenance_contents)
    monkeypatch.setattr(
        mxfp8_utils.envs,
        "VLLM_MXFP8_DENSE_TRTLLM_TACTIC_TABLE_PATH",
        str(table_path),
    )
    monkeypatch.setattr(
        mxfp8_utils.envs,
        "VLLM_MXFP8_DENSE_TRTLLM_TACTIC_TABLE_SHA256",
        table_sha256,
    )
    monkeypatch.setattr(
        mxfp8_utils.envs,
        "VLLM_MXFP8_DENSE_TRTLLM_RUNTIME_PROVENANCE_PATH",
        str(provenance_path),
    )
    monkeypatch.setattr(
        mxfp8_utils.envs,
        "VLLM_MXFP8_DENSE_TRTLLM_RUNTIME_PROVENANCE_SHA256",
        hashlib.sha256(provenance_contents).hexdigest(),
    )


def test_missing_artifact_is_safely_disabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_artifact(
        monkeypatch,
        tmp_path,
        table_path=tmp_path / "missing.json",
        table_sha256="0" * 64,
    )

    artifact, runtime_provenance, rejection = _load_configured_mxfp8_tactic_artifact()

    assert artifact is None
    assert runtime_provenance == provenance()
    assert rejection is not None
    assert "Unable to read MXFP8 tactic artifact" in rejection


def test_hash_mismatched_artifact_is_safely_disabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table_path = tmp_path / "tactics.json"
    table_path.write_text("{}\n", encoding="utf-8")
    configure_artifact(
        monkeypatch,
        tmp_path,
        table_path=table_path,
        table_sha256="0" * 64,
    )

    artifact, runtime_provenance, rejection = _load_configured_mxfp8_tactic_artifact()

    assert artifact is None
    assert runtime_provenance == provenance()
    assert rejection == "MXFP8 tactic artifact SHA256 does not match configuration."


def test_provenance_rejected_artifact_is_safely_disabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table_contents = b"{}\n"
    table_path = tmp_path / "tactics.json"
    table_path.write_bytes(table_contents)
    configure_artifact(
        monkeypatch,
        tmp_path,
        table_path=table_path,
        table_sha256=hashlib.sha256(table_contents).hexdigest(),
    )

    def reject_provenance(*_: object) -> None:
        raise ValueError(
            "MXFP8 tactic artifact provenance does not match runtime for gpu."
        )

    monkeypatch.setattr(
        mxfp8_utils,
        "load_mxfp8_tactic_artifact",
        reject_provenance,
    )

    artifact, runtime_provenance, rejection = _load_configured_mxfp8_tactic_artifact()

    assert artifact is None
    assert runtime_provenance == provenance()
    assert rejection is not None
    assert "provenance does not match runtime" in rejection


def test_structurally_invalid_artifact_remains_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table_contents = b"{}\n"
    table_path = tmp_path / "tactics.json"
    table_path.write_bytes(table_contents)
    configure_artifact(
        monkeypatch,
        tmp_path,
        table_path=table_path,
        table_sha256=hashlib.sha256(table_contents).hexdigest(),
    )

    def reject_duplicate(*_: object) -> None:
        raise ValueError("Duplicate MXFP8 tactic artifact key at entry 1.")

    monkeypatch.setattr(
        mxfp8_utils,
        "load_mxfp8_tactic_artifact",
        reject_duplicate,
    )

    with pytest.raises(ValueError, match="Duplicate MXFP8 tactic artifact key"):
        _load_configured_mxfp8_tactic_artifact()


def test_capture_rejects_unprepared_state_before_startup_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        mxfp8_utils,
        "_mxfp8_cuda_device_key",
        lambda _: ("cuda", 0),
    )
    monkeypatch.setattr(mxfp8_utils, "_MXFP8_TRTLLM_STATES", {})
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    monkeypatch.setattr(
        mxfp8_utils,
        "_load_configured_mxfp8_tactic_artifact",
        lambda: pytest.fail("capture must not load the tactic artifact"),
    )

    with pytest.raises(
        RuntimeError,
        match="must be prepared before CUDA Graph capture",
    ):
        mxfp8_utils.prepare_mxfp8_trtllm_tactic_state(torch.device("cuda", 0))


@pytest.mark.parametrize(
    ("m", "layout", "expected_tactic"),
    [(32, "8x4", 65), (1024, "128x4", 17)],
)
def test_dispatch_uses_exact_table_for_both_layouts(
    monkeypatch: pytest.MonkeyPatch,
    m: int,
    layout: str,
    expected_tactic: int,
) -> None:
    runner_8x4 = FakeRunner([61, 65, 66])
    runner_128x4 = FakeRunner([17])
    dispatch_state = state(
        tactics={key(layout=layout, m=m): expected_tactic},
        runner_8x4=runner_8x4,
        runner_128x4=runner_128x4,
    )
    inputs = runner_inputs()
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)

    tactic, source = _resolve_mxfp8_trtllm_tactic(
        dispatch_state,
        key(layout=layout, m=m),
        inputs,
    )
    _run_mxfp8_trtllm_pre_resolved(
        dispatch_state,
        use_8x4_sf_layout=layout == "8x4",
        runner_inputs=inputs,
        tactic=tactic,
    )

    active_runner = runner_8x4 if layout == "8x4" else runner_128x4
    assert active_runner.forward_tactics == [expected_tactic]
    assert active_runner.valid_tactic_inputs == [inputs]
    assert active_runner.valid_tactic_shapes == [((m, 8192), (8192, 4480))]
    assert source == "exact_table"


def test_exact_miss_uses_matching_direct_runner_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner_8x4 = FakeRunner([61, 65, 66])
    runner_128x4 = FakeRunner([17])
    dispatch_state = state(
        tactics={key(layout="8x4"): 65},
        runner_8x4=runner_8x4,
        runner_128x4=runner_128x4,
    )
    inputs = runner_inputs()
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)

    tactic, source = _resolve_mxfp8_trtllm_tactic(
        dispatch_state,
        key(layout="8x4", m=33),
        inputs,
    )
    _run_mxfp8_trtllm_pre_resolved(
        dispatch_state,
        use_8x4_sf_layout=True,
        runner_inputs=inputs,
        tactic=tactic,
    )

    assert runner_8x4.forward_tactics == [-1]
    assert runner_128x4.forward_tactics == []
    assert source == "exact_miss"


def test_runtime_illegal_tactic_is_downgraded_to_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner_8x4 = FakeRunner([61, 66])
    runner_128x4 = FakeRunner([17])
    dispatch_state = state(
        tactics={key(layout="8x4"): 65},
        runner_8x4=runner_8x4,
        runner_128x4=runner_128x4,
    )
    inputs = runner_inputs()
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)

    tactic, source = _resolve_mxfp8_trtllm_tactic(
        dispatch_state,
        key(layout="8x4"),
        inputs,
    )

    assert tactic == -1
    assert source == "runtime_illegal_tactic"
    assert dispatch_state.resolved_tactics[key(layout="8x4")] == -1


@pytest.mark.parametrize("layout", ["8x4", "128x4"])
def test_safely_disabled_artifact_uses_matching_direct_runner_default(
    monkeypatch: pytest.MonkeyPatch,
    layout: str,
) -> None:
    runner_8x4 = FakeRunner([65])
    runner_128x4 = FakeRunner([17])
    dispatch_state = state(
        tactics={},
        runner_8x4=runner_8x4,
        runner_128x4=runner_128x4,
        artifact_enabled=False,
    )
    inputs = runner_inputs()
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)

    tactic, source = _resolve_mxfp8_trtllm_tactic(
        dispatch_state,
        key(layout=layout),
        inputs,
    )
    _run_mxfp8_trtllm_pre_resolved(
        dispatch_state,
        use_8x4_sf_layout=layout == "8x4",
        runner_inputs=inputs,
        tactic=tactic,
    )

    active_runner = runner_8x4 if layout == "8x4" else runner_128x4
    assert active_runner.valid_tactic_inputs == []
    assert active_runner.forward_tactics == [-1]
    assert source == "artifact_disabled"


def test_capture_executes_only_pre_resolved_scalar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner_8x4 = FakeRunner([65])
    runner_128x4 = FakeRunner([17])
    dispatch_state = state(
        tactics={key(layout="8x4"): 65},
        runner_8x4=runner_8x4,
        runner_128x4=runner_128x4,
    )
    inputs = runner_inputs()
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)

    _run_mxfp8_trtllm_pre_resolved(
        dispatch_state,
        use_8x4_sf_layout=True,
        runner_inputs=inputs,
        tactic=65,
    )

    assert runner_8x4.valid_tactic_inputs == []
    assert runner_8x4.forward_tactics == [65]


def test_capture_rejects_unresolved_key_before_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner_8x4 = FakeRunner([65])
    runner_128x4 = FakeRunner([17])
    dispatch_state = state(
        tactics={key(layout="8x4"): 65},
        runner_8x4=runner_8x4,
        runner_128x4=runner_128x4,
    )
    unresolved_key = key(layout="8x4")
    inputs = runner_inputs()

    class LookupBomb(dict[Mxfp8TacticKey, int]):
        def get(self, *_: object, **__: object) -> int:
            pytest.fail("capture must not look up resolved tactics")

    dispatch_state = dispatch_state._replace(resolved_tactics=LookupBomb())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)

    with pytest.raises(
        RuntimeError,
        match="unresolved MXFP8 tactic before CUDA Graph capture",
    ):
        _resolve_mxfp8_trtllm_tactic(
            dispatch_state,
            unresolved_key,
            inputs,
        )


def test_fixed_impl_validates_exact_forward_contract_before_registration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exact_key = Mxfp8TacticKey(
        m_logical=2,
        n_logical=7,
        k_logical=4,
        n_physical=8,
        k_physical=4,
        activation_scale_layout="8x4",
        output_dtype="bfloat16",
    )
    runner_8x4 = FakeRunner([65])
    dispatch_state = state(
        tactics={exact_key: 65},
        runner_8x4=runner_8x4,
        runner_128x4=FakeRunner([]),
    )
    monkeypatch.setattr(
        mxfp8_utils,
        "_MXFP8_TRTLLM_STATES_BY_DEVICE_INDEX",
        [dispatch_state],
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: False)
    monkeypatch.setattr(mxfp8_utils, "_MXFP8_TACTIC_AUDIT", None)
    monkeypatch.setattr(mxfp8_utils, "_MXFP8_TRTLLM_TRACE_CALLBACK", None)

    x = torch.empty((2, 4), dtype=torch.bfloat16)
    weight = torch.empty((8, 4), dtype=torch.float8_e4m3fn)
    weight_scale = torch.empty(1, dtype=torch.uint8)
    input_mxfp8 = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    input_scale = torch.empty(1, dtype=torch.uint8)
    monkeypatch.setattr(
        mxfp8_utils,
        "prepare_mxfp8_trtllm_tactic_state",
        lambda _device: dispatch_state,
    )
    monkeypatch.setitem(
        sys.modules,
        "flashinfer",
        SimpleNamespace(
            SfLayout=SimpleNamespace(layout_8x4="8x4", layout_128x4="128x4"),
            mxfp8_quantize=lambda *_args, **_kwargs: (
                input_mxfp8,
                input_scale,
            ),
        ),
    )

    _mxfp8_trtllm_linear_fixed_impl(
        x,
        weight,
        weight_scale,
        7,
        65,
        "exact_table",
        "model.layers.0.mlp.fc1",
        "FC1",
        "compiled",
        "pre_capture",
        use_8x4_sf_layout=True,
    )

    assert runner_8x4.valid_tactic_inputs == runner_8x4.forward_inputs
    active_inputs = runner_8x4.forward_inputs[0]
    assert active_inputs[0] is input_mxfp8
    assert active_inputs[1].data_ptr() == weight.data_ptr()
    assert active_inputs[2] is input_scale
    assert active_inputs[3] is weight_scale
    assert active_inputs[4] is torch.bfloat16
    assert active_inputs[5].shape == (2, 8)
    assert active_inputs[6] is dispatch_state.workspace_8x4


def test_fixed_impl_rejects_capture_without_concrete_key_registration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dispatch_state = state(
        tactics={},
        runner_8x4=FakeRunner([65]),
        runner_128x4=FakeRunner([]),
    )
    monkeypatch.setattr(
        mxfp8_utils,
        "_MXFP8_TRTLLM_STATES_BY_DEVICE_INDEX",
        [dispatch_state],
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)

    with FakeTensorMode():
        x = torch.empty((2, 4), device="cuda", dtype=torch.bfloat16)
        weight = torch.empty((8, 4), device="cuda", dtype=torch.float8_e4m3fn)
        weight_scale = torch.empty(1, device="cuda", dtype=torch.uint8)
        monkeypatch.setitem(
            sys.modules,
            "flashinfer",
            SimpleNamespace(
                SfLayout=SimpleNamespace(layout_8x4="8x4", layout_128x4="128x4"),
                mxfp8_quantize=lambda *_args, **_kwargs: pytest.fail(
                    "capture-only key must fail before activation quantization"
                ),
            ),
        )

        with pytest.raises(
            RuntimeError,
            match="unresolved MXFP8 tactic before CUDA Graph capture",
        ):
            _mxfp8_trtllm_linear_fixed_impl(
                x,
                weight,
                weight_scale,
                7,
                65,
                "exact_table",
                "model.layers.0.mlp.fc1",
                "FC1",
                "compiled",
                "pre_capture",
                use_8x4_sf_layout=True,
            )


def test_audit_atomically_records_runtime_illegal_default(
    tmp_path: Path,
) -> None:
    audit = _Mxfp8TacticAudit(
        output_dir=tmp_path,
        expected_rank_count=4,
        rank=2,
        host="gb200-0",
        pid=1234,
        registered_keys={},
        rejected_artifact_reasons=[],
    )

    audit.register(
        key(layout="8x4"),
        selected_tactic=-1,
        tactic_source="runtime_illegal_tactic",
        requested_tactic=65,
    )

    assert not list(tmp_path.glob("*.tmp"))
    payload = json.loads((tmp_path / "rank-2-pid-1234.json").read_text())
    assert payload["complete"] is False
    assert payload["defaults"] == 1
    assert payload["expected_rank_count"] == 4
    assert payload["registered_keys"][0]["requested_tactic"] == 65
    assert payload["registered_keys"][0]["tactic_source"] == "runtime_illegal_tactic"


def test_audit_rejection_and_normal_shutdown_are_durable(tmp_path: Path) -> None:
    audit = _Mxfp8TacticAudit(
        output_dir=tmp_path,
        expected_rank_count=1,
        rank=0,
        host="gb200-0",
        pid=4321,
        registered_keys={},
        rejected_artifact_reasons=[],
    )

    audit.reject_artifact("MXFP8 tactic artifact SHA256 does not match")
    audit.complete = True
    audit.write()

    payload = json.loads((tmp_path / "rank-0-pid-4321.json").read_text())
    assert payload["complete"] is True
    assert payload["rejected_artifact_reasons"] == [
        "MXFP8 tactic artifact SHA256 does not match"
    ]
