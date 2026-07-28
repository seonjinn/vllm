"""Tests for the offline MXFP8 trace-to-shmoo qualification pipeline."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, TypedDict, Unpack

import pytest

_MODULE_PATH = Path(__file__).parents[3] / "tools/mxfp8/offline_shmoo.py"
_SPEC = importlib.util.spec_from_file_location("mxfp8_offline_shmoo", _MODULE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("cannot load MXFP8 offline shmoo module")
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

BenchmarkObservation = _MODULE.BenchmarkObservation
ShapeRecord = _MODULE.ShapeRecord
build_benchmark_plan = _MODULE.build_benchmark_plan
build_tactic_plan = _MODULE.build_tactic_plan
build_qualified_manifest = _MODULE.build_qualified_manifest
canonical_json_bytes = _MODULE.canonical_json_bytes
digest_input_paths = _MODULE.digest_input_paths
load_benchmark_observations = _MODULE.load_benchmark_observations
load_shape_inventory = _MODULE.load_shape_inventory
main = _MODULE.main
qualify_observations = _MODULE.qualify_observations
validate_manifest = _MODULE.validate_manifest

_CONFIG_SHA256 = "d" * 64
_CONTAINER_SHA256 = "c" * 64
_COMPATIBILITY: dict[str, object] = {
    "vllm_version": "0.20.2",
    "vllm_base_commit": "5246e3c5df5fb8266b50ceaa6eca2836fb2d13b1",
    "flashinfer_version": "0.6.8.post1",
    "compute_capability": "10.0",
    "gpu_family": "GB200",
    "model": "Qwen/Qwen3-30B-A3B",
    "tensor_parallel_size": 1,
}
_PROVENANCE: dict[str, object] = {
    "source_manifest_sha256": "a" * 64,
    "source_hint_sha256": "b" * 64,
    "container_sha256": _CONTAINER_SHA256,
    "qualification_scope": "nemo_rl_rollout",
    "qualification_repeat_count": 3,
    "minimum_cosine_similarity": 0.999,
    "minimum_speedup_vs_default": 1.02,
}


class _ObservationOverrides(TypedDict, total=False):
    layout: str
    m: int
    n: int
    k: int
    config_sha256: str
    cosine_similarity: float | None
    all_finite: bool
    status: str
    device_name: str
    vllm_version: str
    flashinfer_version: str
    seed: int


def _bootstrap_manifest() -> dict[str, object]:
    return {
        "schema_version": 1,
        "mode": "adaptive",
        "compatibility": dict(_COMPATIBILITY),
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
        "tactics": {"8x4": [], "128x4": []},
        "provenance": dict(_PROVENANCE),
    }


def _write_bootstrap_manifest(path: Path) -> str:
    path.write_bytes(canonical_json_bytes(_bootstrap_manifest()))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_inventory_artifact(
    path: Path,
    *,
    bootstrap_sha256: str = _CONFIG_SHA256,
    source_manifest_sha256: str = "a" * 64,
) -> None:
    path.write_bytes(
        canonical_json_bytes(
            {
                "schema_version": 1,
                "source_manifest_sha256": source_manifest_sha256,
                "bootstrap_manifest_sha256": bootstrap_sha256,
                "compatibility": dict(_COMPATIBILITY),
                "shapes": [
                    {
                        "layout": "8x4",
                        "m": 8,
                        "n": 2048,
                        "k": 8192,
                        "config_sha256": bootstrap_sha256,
                        "frequency": 1,
                    }
                ],
            }
        )
    )


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def _shape(
    *,
    layout: str = "8x4",
    m: int = 8,
    n: int = 2048,
    k: int = 8192,
    frequency: int = 1,
) -> Any:
    return ShapeRecord(
        layout=layout,
        m=m,
        n=n,
        k=k,
        config_sha256=_CONFIG_SHA256,
        frequency=frequency,
    )


def _observation(
    *,
    layout: str = "8x4",
    m: int = 8,
    n: int = 2048,
    k: int = 8192,
    config_sha256: str = _CONFIG_SHA256,
    tactic: int = -1,
    repeat: int = 0,
    median_ms: float | None = 10.0,
    all_finite: bool = True,
    cosine_similarity: float | None = 0.9999,
    status: str = "success",
    device_name: str = "NVIDIA GB200",
    vllm_version: str = "0.20.2",
    flashinfer_version: str = "0.6.8.post1",
    seed: int | None = None,
) -> Any:
    return BenchmarkObservation(
        layout=layout,
        m=m,
        n=n,
        k=k,
        config_sha256=config_sha256,
        tactic=tactic,
        repeat=repeat,
        median_ms=median_ms,
        all_finite=all_finite,
        cosine_similarity=cosine_similarity,
        status=status,
        seed=1000 + repeat if seed is None else seed,
        warmup=10,
        iterations=80,
        device_name=device_name,
        compute_capability="10.0",
        vllm_version=vllm_version,
        flashinfer_version=flashinfer_version,
        container_sha256=_CONTAINER_SHA256,
    )


def _observations_for_tactic(
    tactic: int,
    medians: tuple[float, float, float],
    **overrides: Unpack[_ObservationOverrides],
) -> list[Any]:
    return [
        _observation(
            tactic=tactic,
            repeat=repeat,
            median_ms=median,
            **overrides,
        )
        for repeat, median in enumerate(medians)
    ]


def test_inventory_aggregates_dispatch_and_dense_physical_shapes(
    tmp_path: Path,
) -> None:
    """Dropping frequency or logical-to-physical normalization would skew shmoo."""
    dispatch = tmp_path / "dispatch.jsonl"
    dense = tmp_path / "dense.jsonl"
    _write_jsonl(
        dispatch,
        [
            {
                "event": "mxfp8_adaptive_dispatch",
                "layout": "8x4",
                "m": 8,
                "n": 2048,
                "k": 8192,
                "config_sha256": _CONFIG_SHA256,
            },
            {
                "event": "mxfp8_adaptive_dispatch",
                "layout": "8x4",
                "m": 8,
                "n": 2048,
                "k": 8192,
                "frequency": 4,
                "config_sha256": _CONFIG_SHA256,
            },
        ],
    )
    _write_jsonl(
        dense,
        [
            {
                "event": "mxfp8_dense_shape",
                "layout": "128x4",
                "m_logical": 257,
                "m_physical": 384,
                "n_logical": 2047,
                "n_physical": 2048,
                "k": 8192,
                "frequency": 2,
                "config_sha256": _CONFIG_SHA256,
            }
        ],
    )

    inventory = load_shape_inventory([dense, dispatch])

    assert inventory == (
        _shape(frequency=5),
        _shape(layout="128x4", m=384, frequency=2),
    )


@pytest.mark.parametrize(
    "record",
    [
        {
            "event": "mxfp8_adaptive_dispatch",
            "layout": "16x8",
            "m": 8,
            "n": 2048,
            "k": 8192,
            "config_sha256": _CONFIG_SHA256,
        },
        {
            "event": "mxfp8_adaptive_dispatch",
            "layout": "8x4",
            "m": True,
            "n": 2048,
            "k": 8192,
            "config_sha256": _CONFIG_SHA256,
        },
        {
            "event": "mxfp8_adaptive_dispatch",
            "layout": "8x4",
            "m": 0,
            "n": 2048,
            "k": 8192,
            "config_sha256": _CONFIG_SHA256,
        },
        {
            "event": "mxfp8_adaptive_dispatch",
            "layout": "8x4",
            "m": 8,
            "n": 2048,
            "k": 8192,
            "frequency": False,
            "config_sha256": _CONFIG_SHA256,
        },
        {
            "event": "mxfp8_adaptive_dispatch",
            "layout": "8x4",
            "m": 8,
            "n": 2048,
            "k": 8192,
            "config_sha256": _CONFIG_SHA256.upper(),
        },
    ],
)
def test_inventory_rejects_malformed_exact_shape_fields(
    tmp_path: Path, record: dict[str, object]
) -> None:
    """Invalid physical identities cannot be benchmarked under another key."""
    path = tmp_path / "invalid.jsonl"
    _write_jsonl(path, [record])

    with pytest.raises(ValueError):
        load_shape_inventory([path])


def test_inventory_rejects_mixed_config_hashes(tmp_path: Path) -> None:
    """Shapes traced under different runtime policies cannot share qualification."""
    path = tmp_path / "mixed.jsonl"
    _write_jsonl(
        path,
        [
            {
                "event": "mxfp8_adaptive_dispatch",
                "layout": "8x4",
                "m": 8,
                "n": 2048,
                "k": 8192,
                "config_sha256": _CONFIG_SHA256,
            },
            {
                "event": "mxfp8_adaptive_dispatch",
                "layout": "8x4",
                "m": 16,
                "n": 2048,
                "k": 8192,
                "config_sha256": "e" * 64,
            },
        ],
    )

    with pytest.raises(ValueError, match="config_sha256"):
        load_shape_inventory([path])


def test_inventory_rejects_zero_eligible_dense_shapes(tmp_path: Path) -> None:
    """A Qwen rollout that bypasses dense MXFP8 must fail rather than optimize zero."""
    path = tmp_path / "no-hits.jsonl"
    _write_jsonl(path, [{"event": "unrelated", "config_sha256": _CONFIG_SHA256}])

    with pytest.raises(ValueError, match="zero eligible"):
        load_shape_inventory([path])


def test_inventory_requires_explicit_layout_for_dense_trace(tmp_path: Path) -> None:
    """Dense layout must never be inferred only from M."""
    path = tmp_path / "missing-layout.jsonl"
    _write_jsonl(
        path,
        [
            {
                "event": "mxfp8_dense_shape",
                "m_physical": 8,
                "n_physical": 2048,
                "k": 8192,
                "config_sha256": _CONFIG_SHA256,
            }
        ],
    )

    with pytest.raises(ValueError, match="layout"):
        load_shape_inventory([path])


def test_inventory_cli_rejects_trace_config_not_bound_to_bootstrap_manifest(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Trace rows cannot be relabeled with an unrelated bootstrap manifest."""
    bootstrap = tmp_path / "bootstrap.json"
    _write_bootstrap_manifest(bootstrap)
    trace = tmp_path / "trace.jsonl"
    _write_jsonl(
        trace,
        [
            {
                "event": "mxfp8_adaptive_dispatch",
                "layout": "8x4",
                "m": 8,
                "n": 2048,
                "k": 8192,
                "config_sha256": _CONFIG_SHA256,
            }
        ],
    )

    assert (
        main(
            [
                "inventory",
                "--trace",
                str(trace),
                "--bootstrap-manifest",
                str(bootstrap),
                "--output",
                str(tmp_path / "inventory.json"),
            ]
        )
        == 2
    )
    assert "bootstrap manifest SHA256" in capsys.readouterr().err


def test_observation_loader_rejects_duplicate_exact_identity(tmp_path: Path) -> None:
    """Resume collisions cannot silently replace one repeat's measurement."""
    path = tmp_path / "duplicate.jsonl"
    record = _observation().__dict__
    _write_jsonl(path, [record, record])

    with pytest.raises(ValueError, match="duplicate"):
        load_benchmark_observations([path])


def test_observation_loader_rejects_empty_shmoo_output(tmp_path: Path) -> None:
    """Promotion must fail clearly when the GPU stage produced no observations."""
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="zero benchmark observations"):
        load_benchmark_observations([path])


def test_observation_loader_rejects_mixed_runtime_identity(tmp_path: Path) -> None:
    """Measurements from different devices or versions cannot share a manifest."""
    path = tmp_path / "mixed-runtime.jsonl"
    _write_jsonl(
        path,
        [
            _observation(tactic=-1, repeat=0).__dict__,
            _observation(
                tactic=-1,
                repeat=1,
                device_name="Different GB200",
            ).__dict__,
        ],
    )

    with pytest.raises(ValueError, match="runtime identity"):
        load_benchmark_observations([path])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("tactic", True),
        ("repeat", False),
        ("median_ms", float("inf")),
        ("cosine_similarity", float("nan")),
        ("warmup", True),
        ("iterations", 0),
    ],
)
def test_observation_loader_rejects_boolean_or_nonfinite_numeric_fields(
    tmp_path: Path, field: str, value: object
) -> None:
    """Malformed timing data cannot accidentally pass numeric qualification."""
    path = tmp_path / "invalid-observation.jsonl"
    record = dict(_observation().__dict__)
    record[field] = value
    path.write_text(json.dumps(record, allow_nan=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError):
        load_benchmark_observations([path])


def test_qualification_selects_fastest_correct_complete_candidate() -> None:
    """The winner must use medians across complete independent repeats."""
    observations = [
        *_observations_for_tactic(-1, (10.0, 12.0, 11.0)),
        *_observations_for_tactic(7, (9.0, 8.0, 8.5)),
        *_observations_for_tactic(9, (9.0, 9.1, 8.9)),
    ]

    qualified = qualify_observations(
        [_shape(frequency=17)],
        observations,
        minimum_repeat_count=3,
        minimum_cosine_similarity=0.999,
        minimum_speedup_vs_default=1.02,
    )

    assert len(qualified) == 1
    assert qualified[0].tactic == 7
    assert qualified[0].baseline_median_ms == 11.0
    assert qualified[0].candidate_median_ms == 8.5
    assert qualified[0].speedup_vs_default == pytest.approx(11.0 / 8.5)
    assert qualified[0].frequency == 17


def test_qualification_timing_tie_uses_lower_tactic_id() -> None:
    """Equal candidate medians must have a stable tactic-ID tie break."""
    observations = [
        *_observations_for_tactic(-1, (10.0, 10.0, 10.0)),
        *_observations_for_tactic(9, (8.0, 8.0, 8.0)),
        *_observations_for_tactic(7, (8.0, 8.0, 8.0)),
    ]

    qualified = qualify_observations(
        [_shape()],
        observations,
        minimum_repeat_count=3,
        minimum_cosine_similarity=0.999,
        minimum_speedup_vs_default=1.02,
    )

    assert qualified[0].tactic == 7


@pytest.mark.parametrize(
    "candidate",
    [
        [
            *_observations_for_tactic(7, (9.9, 9.9, 9.9)),
        ],
        [
            *_observations_for_tactic(7, (8.0, 8.0, 8.0))[:2],
        ],
        [
            *_observations_for_tactic(
                7,
                (8.0, 8.0, 8.0),
                cosine_similarity=0.9,
            ),
        ],
        [
            *_observations_for_tactic(
                7,
                (8.0, 8.0, 8.0),
                all_finite=False,
            ),
        ],
    ],
)
def test_qualification_leaves_slow_incomplete_or_incorrect_candidate_at_default(
    candidate: list[Any],
) -> None:
    """A candidate missing any promotion gate must stay on runner default -1."""
    qualified = qualify_observations(
        [_shape()],
        [*_observations_for_tactic(-1, (10.0, 10.0, 10.0)), *candidate],
        minimum_repeat_count=3,
        minimum_cosine_similarity=0.999,
        minimum_speedup_vs_default=1.02,
    )

    assert qualified[0].tactic == -1
    assert qualified[0].candidate_median_ms is None
    assert qualified[0].speedup_vs_default is None


def test_qualification_requires_passing_default_for_every_repeat() -> None:
    """No shape can be promoted without a complete correct runner-default baseline."""
    observations = _observations_for_tactic(-1, (10.0, 10.0, 10.0))[:2]

    with pytest.raises(ValueError, match="default tactic -1"):
        qualify_observations(
            [_shape()],
            observations,
            minimum_repeat_count=3,
            minimum_cosine_similarity=0.999,
            minimum_speedup_vs_default=1.02,
        )


def test_qualification_requires_shared_seed_per_repeat_and_distinct_repeats() -> None:
    """Externally supplied observations must preserve independent repeat identity."""
    observations = [
        *_observations_for_tactic(-1, (10.0, 10.0, 10.0)),
        *_observations_for_tactic(7, (8.0, 8.0, 8.0)),
    ]
    observations[3] = _observation(
        tactic=7,
        repeat=0,
        median_ms=8.0,
        seed=9999,
    )

    with pytest.raises(ValueError, match="one seed"):
        qualify_observations(
            [_shape()],
            observations,
            minimum_repeat_count=3,
            minimum_cosine_similarity=0.999,
            minimum_speedup_vs_default=1.02,
        )

    repeated_seed = [
        _observation(tactic=-1, repeat=repeat, seed=1000)
        for repeat in range(3)
    ]
    with pytest.raises(ValueError, match="distinct seeds"):
        qualify_observations(
            [_shape()],
            repeated_seed,
            minimum_repeat_count=3,
            minimum_cosine_similarity=0.999,
            minimum_speedup_vs_default=1.02,
        )


def test_qualification_rejects_observation_shape_absent_from_inventory() -> None:
    """Benchmark results cannot introduce an untraced physical shape."""
    observations = _observations_for_tactic(-1, (10.0, 10.0, 10.0), m=16)

    with pytest.raises(ValueError, match="absent from inventory"):
        qualify_observations(
            [_shape()],
            observations,
            minimum_repeat_count=3,
            minimum_cosine_similarity=0.999,
            minimum_speedup_vs_default=1.02,
        )


def test_qualified_manifest_preserves_layout_and_omits_default_shapes() -> None:
    """Layout-table crossing or serializing -1 would bypass runtime fallback."""
    inventory = [
        _shape(layout="8x4", m=8),
        _shape(layout="128x4", m=384),
        _shape(layout="8x4", m=16),
    ]
    observations = [
        *_observations_for_tactic(-1, (10.0, 10.0, 10.0), m=8),
        *_observations_for_tactic(7, (8.0, 8.0, 8.0), m=8),
        *_observations_for_tactic(
            -1,
            (20.0, 20.0, 20.0),
            layout="128x4",
            m=384,
        ),
        *_observations_for_tactic(
            11,
            (15.0, 15.0, 15.0),
            layout="128x4",
            m=384,
        ),
        *_observations_for_tactic(-1, (5.0, 5.0, 5.0), m=16),
        *_observations_for_tactic(13, (4.99, 4.99, 4.99), m=16),
    ]
    qualified = qualify_observations(
        inventory,
        observations,
        minimum_repeat_count=3,
        minimum_cosine_similarity=0.999,
        minimum_speedup_vs_default=1.02,
    )

    manifest = build_qualified_manifest(
        qualified,
        compatibility=_COMPATIBILITY,
        provenance=_PROVENANCE,
    )

    assert manifest["policy"] == {
        "gemm_backend": "trtllm",
        "layout": "adaptive",
        "switch_m": 256,
        "direct_trtllm": True,
        "require_direct_trtllm": True,
        "quant_backend": "cuda",
        "require_8x4_quant": True,
        "pad_to_128": False,
        "default_tactic": -1,
    }
    assert manifest["tactics"] == {
        "8x4": [{"m": 8, "n": 2048, "k": 8192, "tactic": 7}],
        "128x4": [{"m": 384, "n": 2048, "k": 8192, "tactic": 11}],
    }


def test_benchmark_plan_keeps_exact_unpadded_layout_and_deterministic_seeds() -> None:
    """GPU jobs must not turn a traced 8x4 M=8 into an M=128 benchmark."""
    inventory = [_shape(layout="128x4", m=384), _shape(layout="8x4", m=8)]

    first = build_benchmark_plan(inventory, repeat_count=3, base_seed=1234)
    second = build_benchmark_plan(reversed(inventory), repeat_count=3, base_seed=1234)

    assert first == second
    assert [(job.layout, job.m, job.repeat) for job in first] == [
        ("8x4", 8, 0),
        ("8x4", 8, 1),
        ("8x4", 8, 2),
        ("128x4", 384, 0),
        ("128x4", 384, 1),
        ("128x4", 384, 2),
    ]
    assert [job.seed for job in first] == [
        417114,
        417115,
        417116,
        423506,
        423507,
        423508,
    ]


def test_resume_rejects_changed_base_seed_and_mixed_repeat_seeds() -> None:
    """Resume identity must bind every existing tactic to the deterministic seed."""
    plans = build_benchmark_plan([_shape()], repeat_count=3, base_seed=1234)
    observations = [
        *[
            _observation(
                tactic=-1,
                repeat=repeat,
                median_ms=10.0,
                seed=plans[repeat].seed,
            )
            for repeat in range(3)
        ],
        *[
            _observation(
                tactic=7,
                repeat=repeat,
                median_ms=8.0,
                seed=plans[repeat].seed,
            )
            for repeat in range(3)
        ],
    ]

    _MODULE.validate_resume_observations(plans, observations)

    changed = list(observations)
    changed[3] = _observation(
        tactic=7,
        repeat=0,
        median_ms=8.0,
        seed=changed[3].seed + 100,
    )
    with pytest.raises(ValueError, match="deterministic seed"):
        _MODULE.validate_resume_observations(plans, changed)

    changed_plan = build_benchmark_plan(
        [_shape()], repeat_count=3, base_seed=9999
    )
    with pytest.raises(ValueError, match="deterministic seed"):
        _MODULE.validate_resume_observations(changed_plan, observations)


def test_tactic_plan_includes_default_all_valid_and_exact_identity_resume() -> None:
    """The GPU stage must benchmark -1 plus every valid tactic exactly once."""
    plan = build_benchmark_plan(
        [_shape()], repeat_count=3, base_seed=1234
    )[0]
    completed = {
        (
            plan.layout,
            plan.m,
            plan.n,
            plan.k,
            plan.config_sha256,
            7,
            plan.repeat,
        )
    }

    tactics = build_tactic_plan(
        plan,
        valid_tactics=[9, -1, 7, 9, 11],
        completed=completed,
    )

    assert tactics == (-1, 9, 11)


def test_valid_tactic_enumeration_error_aborts_with_underlying_diagnostic() -> None:
    """A private-runner ABI error cannot masquerade as zero valid tactics."""

    class BrokenRunner:
        def get_valid_tactics(
            self, _inputs: list[object], _profile: object
        ) -> list[int]:
            raise RuntimeError("enumeration ABI mismatch")

    with pytest.raises(RuntimeError, match="enumeration ABI mismatch"):
        _MODULE.enumerate_valid_tactics(BrokenRunner(), [], object())

    class EmptyRunner:
        def get_valid_tactics(
            self, _inputs: list[object], _profile: object
        ) -> list[int]:
            return []

    assert _MODULE.enumerate_valid_tactics(EmptyRunner(), [], object()) == ()


def test_gpu_error_observation_retains_actionable_diagnostic() -> None:
    """Append-only failures must explain the failing private runner operation."""
    plan = build_benchmark_plan(
        [_shape()], repeat_count=3, base_seed=1234
    )[0]
    runtime = _MODULE._GpuRuntimeIdentity(
        device_name="NVIDIA GB200",
        compute_capability="10.0",
        vllm_version="0.20.2",
        flashinfer_version="0.6.8.post1",
        container_sha256=_CONTAINER_SHA256,
    )

    observation = _MODULE._error_observation(
        plan,
        7,
        runtime,
        warmup=10,
        iterations=80,
        num_valid_tactics=12,
        error=RuntimeError("runner forward ABI mismatch"),
    )

    assert observation.status == "error"
    assert observation.num_valid_tactics == 12
    assert observation.error == "runner forward ABI mismatch"


def test_active_runtime_identity_rejects_declared_version_or_gpu_mismatch() -> None:
    """The GPU stage must bind CLI declarations to installed/runtime identity."""
    _MODULE.validate_active_runtime_identity(
        active_vllm_version="0.20.2",
        active_flashinfer_version="0.6.8.post1",
        active_compute_capability="10.0",
        active_device_name="NVIDIA GB200",
        declared_vllm_version="0.20.2",
        declared_flashinfer_version="0.6.8.post1",
        inventory_compatibility=_COMPATIBILITY,
    )

    with pytest.raises(RuntimeError, match="active vLLM"):
        _MODULE.validate_active_runtime_identity(
            active_vllm_version="0.20.1",
            active_flashinfer_version="0.6.8.post1",
            active_compute_capability="10.0",
            active_device_name="NVIDIA GB200",
            declared_vllm_version="0.20.2",
            declared_flashinfer_version="0.6.8.post1",
            inventory_compatibility=_COMPATIBILITY,
        )
    with pytest.raises(RuntimeError, match="GPU family"):
        _MODULE.validate_active_runtime_identity(
            active_vllm_version="0.20.2",
            active_flashinfer_version="0.6.8.post1",
            active_compute_capability="10.0",
            active_device_name="NVIDIA H100",
            declared_vllm_version="0.20.2",
            declared_flashinfer_version="0.6.8.post1",
            inventory_compatibility=_COMPATIBILITY,
        )


def test_active_runtime_identity_accepts_compatible_local_version_labels() -> None:
    """Container-local build labels must follow the production loader contract."""
    validate_active_runtime_identity = (
        _MODULE.validate_active_runtime_identity
    )

    validate_active_runtime_identity(
        active_vllm_version="0.20.2+local.cu130",
        active_flashinfer_version="0.6.8.post1+cu130",
        active_compute_capability="10.0",
        active_device_name="NVIDIA GB200",
        declared_vllm_version="0.20.2+custom",
        declared_flashinfer_version="0.6.8.post1",
        inventory_compatibility=_COMPATIBILITY,
    )

    with pytest.raises(RuntimeError, match="active vLLM"):
        validate_active_runtime_identity(
            active_vllm_version="0.20.2rc1",
            active_flashinfer_version="0.6.8.post1+cu130",
            active_compute_capability="10.0",
            active_device_name="NVIDIA GB200",
            declared_vllm_version="0.20.2",
            declared_flashinfer_version="0.6.8.post1",
            inventory_compatibility=_COMPATIBILITY,
        )


def test_regeneration_accepts_observation_local_version_labels(
    tmp_path: Path,
) -> None:
    """Shmoo observations with local tags must match base manifest versions."""
    inventory = tmp_path / "inventory.json"
    _write_inventory_artifact(inventory)
    observations = tmp_path / "observations.jsonl"
    records = [
        *_observations_for_tactic(
            -1,
            (10.0, 10.0, 10.0),
            vllm_version="0.20.2+local.cu130",
            flashinfer_version="0.6.8.post1+cu130",
        ),
        *_observations_for_tactic(
            7,
            (8.0, 8.0, 8.0),
            vllm_version="0.20.2+local.cu130",
            flashinfer_version="0.6.8.post1+cu130",
        ),
    ]
    _write_jsonl(
        observations, [observation.__dict__ for observation in records]
    )

    manifest, _qualified = _MODULE.regenerate_qualified_manifest(
        inventory_path=inventory,
        observation_paths=[observations],
        minimum_repeat_count=3,
        minimum_cosine_similarity=0.999,
        minimum_speedup_vs_default=1.02,
        qualification_scope="nemo_rl_rollout",
    )

    assert manifest["compatibility"] == _COMPATIBILITY


def test_gpu_family_matching_uses_exact_alias_tokens_not_substrings() -> None:
    """B200 must never satisfy a GB200 bootstrap identity, or vice versa."""
    b200_compatibility = dict(_COMPATIBILITY)
    b200_compatibility["gpu_family"] = "B200"

    with pytest.raises(RuntimeError, match="GPU family"):
        _MODULE.validate_active_runtime_identity(
            active_vllm_version="0.20.2",
            active_flashinfer_version="0.6.8.post1",
            active_compute_capability="10.0",
            active_device_name="NVIDIA GB200",
            declared_vllm_version="0.20.2",
            declared_flashinfer_version="0.6.8.post1",
            inventory_compatibility=b200_compatibility,
        )
    with pytest.raises(RuntimeError, match="GPU family"):
        _MODULE.validate_active_runtime_identity(
            active_vllm_version="0.20.2",
            active_flashinfer_version="0.6.8.post1",
            active_compute_capability="10.0",
            active_device_name="NVIDIA B200",
            declared_vllm_version="0.20.2",
            declared_flashinfer_version="0.6.8.post1",
            inventory_compatibility=_COMPATIBILITY,
        )


def test_qwen_trace_bootstrap_cli_writes_canonical_loader_validated_bytes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The fixed Qwen trace bootstrap must be reproducible and runtime-valid."""
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    common = [
        "trace-bootstrap-qwen3-30ba3b-tp1",
        "--source-manifest-sha256",
        "a" * 64,
        "--source-hint-sha256",
        "b" * 64,
        "--container-sha256",
        _CONTAINER_SHA256,
        "--output",
    ]
    expected = {
        "schema_version": 1,
        "mode": "adaptive",
        "compatibility": {
            "vllm_version": "0.20.2",
            "vllm_base_commit": (
                "5246e3c5df5fb8266b50ceaa6eca2836fb2d13b1"
            ),
            "flashinfer_version": "0.6.8.post1",
            "compute_capability": "10.0",
            "gpu_family": "GB200",
            "model": "Qwen/Qwen3-30B-A3B",
            "tensor_parallel_size": 1,
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
        "tactics": {"8x4": [], "128x4": []},
        "provenance": {
            "source_manifest_sha256": "a" * 64,
            "source_hint_sha256": "b" * 64,
            "container_sha256": _CONTAINER_SHA256,
            "qualification_scope": (
                "nemo_rl_qwen3_30ba3b_mxfp8_rollout_trace_bootstrap"
            ),
            "qualification_repeat_count": 3,
            "minimum_cosine_similarity": 0.999,
            "minimum_speedup_vs_default": 1.02,
        },
    }

    assert main([*common, str(first)]) == 0
    first_sha256 = capsys.readouterr().out.strip()
    assert main([*common, str(second)]) == 0
    second_sha256 = capsys.readouterr().out.strip()

    expected_bytes = (
        json.dumps(expected, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    assert first.read_bytes() == expected_bytes
    assert second.read_bytes() == expected_bytes
    assert first_sha256 == hashlib.sha256(expected_bytes).hexdigest()
    assert second_sha256 == first_sha256
    loaded_sha256, compatibility = _MODULE._load_bootstrap_runtime_manifest(
        first
    )
    assert loaded_sha256 == first_sha256
    assert compatibility == expected["compatibility"]


@pytest.mark.parametrize(
    ("invalid_flag", "invalid_value"),
    [
        ("--source-manifest-sha256", "not-a-digest"),
        ("--source-hint-sha256", "A" * 64),
        ("--container-sha256", "f" * 63),
    ],
)
def test_qwen_trace_bootstrap_cli_rejects_invalid_provenance_before_output(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    invalid_flag: str,
    invalid_value: str,
) -> None:
    """Malformed provenance must not leave a bootstrap artifact."""
    output = tmp_path / "bootstrap.json"
    arguments = {
        "--source-manifest-sha256": "a" * 64,
        "--source-hint-sha256": "b" * 64,
        "--container-sha256": _CONTAINER_SHA256,
    }
    arguments[invalid_flag] = invalid_value
    cli = ["trace-bootstrap-qwen3-30ba3b-tp1"]
    for flag, value in arguments.items():
        cli.extend((flag, value))
    cli.extend(("--output", str(output)))

    assert main(cli) == 2
    assert not output.exists()
    assert "lowercase SHA-256" in capsys.readouterr().err


def test_qwen_trace_bootstrap_cli_fails_closed_on_loader_or_output_conflict(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A loader rejection or existing target must never be overwritten."""
    output = tmp_path / "bootstrap.json"
    common = [
        "trace-bootstrap-qwen3-30ba3b-tp1",
        "--source-manifest-sha256",
        "a" * 64,
        "--source-hint-sha256",
        "b" * 64,
        "--container-sha256",
        _CONTAINER_SHA256,
        "--output",
        str(output),
    ]

    def reject_manifest(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("simulated production loader rejection")

    monkeypatch.setattr(
        _MODULE, "_load_production_loader", lambda: reject_manifest
    )
    assert main(common) == 2
    assert not output.exists()
    assert "loader rejection" in capsys.readouterr().err

    output.write_text("preserve", encoding="utf-8")
    assert main(common) == 2
    assert output.read_text(encoding="utf-8") == "preserve"
    assert "already exists" in capsys.readouterr().err


def test_aggregate_input_digest_is_path_and_argument_order_independent(
    tmp_path: Path,
) -> None:
    """Moving identical raw artifacts cannot change manifest provenance."""
    first = tmp_path / "z.jsonl"
    second = tmp_path / "a.jsonl"
    first.write_bytes(b"first\n")
    second.write_bytes(b"second\n")
    expected_members = sorted(
        hashlib.sha256(path.read_bytes()).hexdigest() for path in (first, second)
    )
    expected = hashlib.sha256(
        json.dumps(expected_members, separators=(",", ":")).encode()
    ).hexdigest()

    assert digest_input_paths([first, second]) == expected
    assert digest_input_paths([second, first]) == expected


def test_inventory_cli_writes_canonical_json_and_zero_hit_returns_nonzero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The explicit inventory stage must be reproducible and fail loudly on no hits."""
    bootstrap = tmp_path / "bootstrap.json"
    bootstrap_sha256 = _write_bootstrap_manifest(bootstrap)
    trace = tmp_path / "trace.jsonl"
    _write_jsonl(
        trace,
        [
            {
                "event": "mxfp8_adaptive_dispatch",
                "layout": "8x4",
                "m": 8,
                "n": 2048,
                "k": 8192,
                "config_sha256": bootstrap_sha256,
            }
        ],
    )
    output = tmp_path / "inventory.json"

    assert (
        main(
            [
                "inventory",
                "--trace",
                str(trace),
                "--bootstrap-manifest",
                str(bootstrap),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert output.read_bytes() == canonical_json_bytes(payload)
    assert payload["source_manifest_sha256"] == digest_input_paths([trace])
    assert payload["bootstrap_manifest_sha256"] == bootstrap_sha256
    assert payload["compatibility"] == _COMPATIBILITY
    assert payload["shapes"] == [
        {
            "config_sha256": bootstrap_sha256,
            "frequency": 1,
            "k": 8192,
            "layout": "8x4",
            "m": 8,
            "n": 2048,
        }
    ]

    empty = tmp_path / "empty.jsonl"
    _write_jsonl(empty, [{"event": "unrelated"}])
    assert (
        main(
            [
                "inventory",
                "--trace",
                str(empty),
                "--bootstrap-manifest",
                str(bootstrap),
                "--output",
                str(tmp_path / "must-not-exist.json"),
            ]
        )
        == 2
    )
    assert "zero eligible" in capsys.readouterr().err


def test_generated_outputs_refuse_aliases_and_existing_files(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Generated artifacts cannot overwrite trace evidence or prior artifacts."""
    bootstrap = tmp_path / "bootstrap.json"
    bootstrap_sha256 = _write_bootstrap_manifest(bootstrap)
    trace = tmp_path / "trace.jsonl"
    _write_jsonl(
        trace,
        [
            {
                "event": "mxfp8_adaptive_dispatch",
                "layout": "8x4",
                "m": 8,
                "n": 2048,
                "k": 8192,
                "config_sha256": bootstrap_sha256,
            }
        ],
    )
    common = [
        "inventory",
        "--trace",
        str(trace),
        "--bootstrap-manifest",
        str(bootstrap),
        "--output",
    ]

    assert main([*common, str(trace)]) == 2
    assert "path collision" in capsys.readouterr().err
    existing = tmp_path / "existing.json"
    existing.write_text("keep", encoding="utf-8")
    assert main([*common, str(existing)]) == 2
    assert existing.read_text(encoding="utf-8") == "keep"
    assert "already exists" in capsys.readouterr().err


@pytest.mark.parametrize(
    "invalid_args",
    [
        ["--repeat-count", "2"],
        ["--base-seed", "-1"],
        ["--warmup", "-1"],
        ["--iterations", "0"],
        ["--workspace-mb", "0"],
        ["--minimum-cosine-similarity", "0"],
        ["--container-sha256", "bad"],
    ],
)
def test_shmoo_rejects_invalid_contract_arguments_before_gpu_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    invalid_args: list[str],
) -> None:
    """Invalid settings must not touch CUDA or create unreadable append logs."""
    inventory = tmp_path / "inventory.json"
    _write_inventory_artifact(inventory)
    output = tmp_path / "observations.jsonl"
    imports = 0

    def forbidden_gpu_import() -> tuple[object, ...]:
        nonlocal imports
        imports += 1
        raise AssertionError("GPU import must occur after argument validation")

    monkeypatch.setattr(_MODULE, "_import_gpu_dependencies", forbidden_gpu_import)
    base_args = [
        "shmoo",
        "--inventory",
        str(inventory),
        "--output",
        str(output),
        "--vllm-version",
        "0.20.2",
        "--flashinfer-version",
        "0.6.8.post1",
        "--container-sha256",
        _CONTAINER_SHA256,
    ]

    assert main([*base_args, *invalid_args]) == 2
    assert imports == 0
    assert not output.exists()


def test_promote_cli_binds_manifest_compatibility_to_observed_runtime(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A caller cannot relabel shmoo results as another vLLM runtime."""
    bootstrap = tmp_path / "bootstrap.json"
    bootstrap_sha256 = _write_bootstrap_manifest(bootstrap)
    trace = tmp_path / "trace.jsonl"
    _write_jsonl(
        trace,
        [
            {
                "event": "mxfp8_adaptive_dispatch",
                "layout": "8x4",
                "m": 8,
                "n": 2048,
                "k": 8192,
                "config_sha256": bootstrap_sha256,
            }
        ],
    )
    inventory = tmp_path / "inventory.json"
    assert (
        main(
            [
                "inventory",
                "--trace",
                str(trace),
                "--bootstrap-manifest",
                str(bootstrap),
                "--output",
                str(inventory),
            ]
        )
        == 0
    )
    observations = tmp_path / "observations.jsonl"
    _write_jsonl(
        observations,
        [
            *(
                observation.__dict__
                for observation in _observations_for_tactic(
                    -1,
                    (10.0, 10.0, 10.0),
                    config_sha256=bootstrap_sha256,
                )
            ),
            *(
                observation.__dict__
                for observation in _observations_for_tactic(
                    7,
                    (8.0, 8.0, 8.0),
                    config_sha256=bootstrap_sha256,
                )
            ),
        ],
    )
    output = tmp_path / "qualified.json"
    common_args = [
        "promote",
        "--inventory",
        str(inventory),
        "--observations",
        str(observations),
        "--output",
        str(output),
        "--qualification-scope",
        "nemo_rl_rollout",
        "--vllm-base-commit",
        str(_COMPATIBILITY["vllm_base_commit"]),
        "--flashinfer-version",
        str(_COMPATIBILITY["flashinfer_version"]),
        "--compute-capability",
        str(_COMPATIBILITY["compute_capability"]),
        "--gpu-family",
        str(_COMPATIBILITY["gpu_family"]),
        "--model",
        str(_COMPATIBILITY["model"]),
        "--tensor-parallel-size",
        str(_COMPATIBILITY["tensor_parallel_size"]),
    ]

    assert main([*common_args, "--vllm-version", "0.20.2"]) == 0
    manifest = json.loads(output.read_text(encoding="utf-8"))
    assert output.read_bytes() == canonical_json_bytes(manifest)
    assert manifest["provenance"]["source_manifest_sha256"] == (
        digest_input_paths([trace])
    )
    assert manifest["provenance"]["source_hint_sha256"] == (
        digest_input_paths([observations])
    )

    collision_args = list(common_args)
    output_index = collision_args.index("--output") + 1
    collision_args[output_index] = str(observations)
    raw_observations = observations.read_bytes()
    assert main([*collision_args, "--vllm-version", "0.20.2"]) == 2
    assert observations.read_bytes() == raw_observations
    assert "path collision" in capsys.readouterr().err

    mismatch_args = list(common_args)
    mismatch_args[output_index] = str(tmp_path / "mismatch.json")
    assert main([*mismatch_args, "--vllm-version", "0.20.1"]) == 2
    assert "observed vLLM version" in capsys.readouterr().err


def test_validate_manifest_uses_production_loader_and_checks_canonical_bytes(
    tmp_path: Path,
) -> None:
    """Validation must regenerate from raw inputs and catch canonical tactic edits."""
    inventory_path = tmp_path / "inventory.json"
    _write_inventory_artifact(inventory_path)
    observations = [
        *_observations_for_tactic(-1, (10.0, 10.0, 10.0)),
        *_observations_for_tactic(7, (8.0, 8.0, 8.0)),
    ]
    observations_path = tmp_path / "observations.jsonl"
    _write_jsonl(
        observations_path,
        [observation.__dict__ for observation in observations],
    )
    qualified = qualify_observations(
        [_shape()],
        observations,
        minimum_repeat_count=3,
        minimum_cosine_similarity=0.999,
        minimum_speedup_vs_default=1.02,
    )
    provenance = dict(_PROVENANCE)
    provenance["source_hint_sha256"] = digest_input_paths(
        [observations_path]
    )
    manifest = build_qualified_manifest(
        qualified,
        compatibility=_COMPATIBILITY,
        provenance=provenance,
    )
    path = tmp_path / "qualified.json"
    path.write_bytes(canonical_json_bytes(manifest))

    validate_manifest(
        path,
        inventory_path=inventory_path,
        observation_paths=[observations_path],
        minimum_repeat_count=3,
        minimum_cosine_similarity=0.999,
        minimum_speedup_vs_default=1.02,
        qualification_scope="nemo_rl_rollout",
        actual_vllm_version="0.20.2+local",
        actual_flashinfer_version="0.6.8.post1+cu130",
        actual_compute_capability=(10, 0),
        actual_model="Qwen/Qwen3-30B-A3B",
        actual_tensor_parallel_size=1,
        check=True,
    )

    manifest["tactics"]["8x4"][0]["tactic"] = 10007
    path.write_bytes(canonical_json_bytes(manifest))
    with pytest.raises(ValueError, match="regenerated"):
        validate_manifest(
            path,
            inventory_path=inventory_path,
            observation_paths=[observations_path],
            minimum_repeat_count=3,
            minimum_cosine_similarity=0.999,
            minimum_speedup_vs_default=1.02,
            qualification_scope="nemo_rl_rollout",
            actual_vllm_version="0.20.2",
            actual_flashinfer_version="0.6.8.post1",
            actual_compute_capability=(10, 0),
            actual_model="Qwen/Qwen3-30B-A3B",
            actual_tensor_parallel_size=1,
            check=True,
        )
