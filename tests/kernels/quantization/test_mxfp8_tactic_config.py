"""Tests for loading fail-closed MXFP8 dense tactic manifests."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

_MODULE_PATH = (
    Path(__file__).parents[3]
    / "vllm/model_executor/kernels/linear/mxfp8/tactic_config.py"
)
_SPEC = importlib.util.spec_from_file_location("mxfp8_tactic_config", _MODULE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("cannot load MXFP8 tactic config module")
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
load_mxfp8_dense_runtime_config = _MODULE.load_mxfp8_dense_runtime_config


def _manifest() -> dict[str, Any]:
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
            "pad_to_128": True,
            "default_tactic": -1,
        },
        "tactics": {
            "8x4": [{"m": 1, "n": 2048, "k": 8192, "tactic": 66}],
            "128x4": [
                {"m": 1000, "n": 2048, "k": 8192, "tactic": 70}
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


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.write_text(json.dumps(manifest), encoding="utf-8")


def _load(
    reference: str,
    *,
    actual_vllm_version: str = "0.20.2+local",
    actual_flashinfer_version: str = "0.6.8.post1+cu129",
    config_dir: Path | None = None,
) -> Any:
    return load_mxfp8_dense_runtime_config(
        reference,
        actual_vllm_version=actual_vllm_version,
        actual_flashinfer_version=actual_flashinfer_version,
        actual_compute_capability=(10, 0),
        package_config_dir=config_dir,
    )


def test_loads_absolute_manifest_with_immutable_metadata(tmp_path: Path) -> None:
    """A valid manifest produces the exact frozen runtime configuration."""
    path = tmp_path / "qualified.json"
    _write_manifest(path, _manifest())

    config = _load(str(path))

    assert config.switch_m == 256
    assert config.tactics_8x4 == (((1, 2048, 8192), 66),)
    assert config.tactics_128x4 == (((1000, 2048, 8192), 70),)
    assert config.source_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert config.provenance["qualification_scope"] == "standalone_serving_seed"
    with pytest.raises(TypeError):
        config.compatibility["model"] = "other"
    with pytest.raises(TypeError):
        config.provenance["qualification_repeat_count"] = 1


def test_loads_relative_manifest_only_from_injected_config_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A relative name cannot accidentally resolve from a worker's CWD."""
    config_dir = tmp_path / "tactic_configs"
    config_dir.mkdir()
    _write_manifest(config_dir / "qualified.json", _manifest())
    cwd_file = tmp_path / "qualified.json"
    _write_manifest(cwd_file, {"schema_version": 999})
    monkeypatch.chdir(tmp_path)

    config = _load("qualified.json", config_dir=config_dir)

    assert config.source_path == (config_dir / "qualified.json").resolve()


@pytest.mark.parametrize(
    ("mutate", "error"),
    [
        (lambda data: data.__setitem__("schema_version", 2), "schema_version"),
        (lambda data: data.__setitem__("mode", "static"), "mode"),
        (
            lambda data: data["compatibility"].__setitem__(
                "vllm_version", "0.20.1"
            ),
            "compatibility.vllm_version",
        ),
        (
            lambda data: data["compatibility"].__setitem__(
                "flashinfer_version", "0.6.7"
            ),
            "compatibility.flashinfer_version",
        ),
        (
            lambda data: data["compatibility"].__setitem__(
                "compute_capability", "9.0"
            ),
            "compatibility.compute_capability",
        ),
        (lambda data: data["policy"].__setitem__("switch_m", 0), "policy.switch_m"),
        (lambda data: data["policy"].__setitem__("switch_m", 129), "policy.switch_m"),
        (
            lambda data: data["tactics"]["8x4"][0].__setitem__("m", 0),
            "tactics.8x4[0].m",
        ),
        (
            lambda data: data["tactics"]["8x4"][0].__setitem__("n", "2048"),
            "tactics.8x4[0].n",
        ),
        (
            lambda data: data["tactics"]["8x4"][0].__setitem__("tactic", 1.5),
            "tactics.8x4[0].tactic",
        ),
    ],
)
def test_rejects_invalid_manifest_fields(
    tmp_path: Path,
    mutate: Callable[[dict[str, Any]], None],
    error: str,
) -> None:
    """Each critical field rejects malformed or incompatible values."""
    path = tmp_path / "invalid.json"
    manifest = _manifest()
    mutate(manifest)
    _write_manifest(path, manifest)

    with pytest.raises((ValueError, RuntimeError), match=re.escape(error)):
        _load(str(path))


def test_rejects_duplicate_shape_within_layout(tmp_path: Path) -> None:
    """A duplicate layout shape cannot silently choose one tactic."""
    path = tmp_path / "duplicate.json"
    manifest = _manifest()
    manifest["tactics"]["8x4"].append(
        {"m": 1, "n": 2048, "k": 8192, "tactic": 71}
    )
    _write_manifest(path, manifest)

    with pytest.raises(ValueError, match="tactics.8x4"):
        _load(str(path))


def test_rejects_shape_shared_between_layouts(tmp_path: Path) -> None:
    """A shape must unambiguously select exactly one layout's tactic."""
    path = tmp_path / "overlap.json"
    manifest = _manifest()
    manifest["tactics"]["128x4"][0]["m"] = 1
    _write_manifest(path, manifest)

    with pytest.raises(ValueError, match="tactics"):
        _load(str(path))


@pytest.mark.parametrize("section", ["policy", "provenance"])
def test_rejects_missing_required_metadata(tmp_path: Path, section: str) -> None:
    """Required runtime policy and qualification evidence cannot be omitted."""
    path = tmp_path / f"missing-{section}.json"
    manifest = _manifest()
    del manifest[section]
    _write_manifest(path, manifest)

    with pytest.raises(ValueError, match=section):
        _load(str(path))


@pytest.mark.parametrize(
    ("section", "field"),
    [
        ("policy", "switch_m"),
        ("provenance", "source_manifest_sha256"),
        ("provenance", "qualification_scope"),
    ],
)
def test_rejects_missing_required_metadata_field(
    tmp_path: Path, section: str, field: str
) -> None:
    """A manifest cannot omit one required policy or provenance value."""
    path = tmp_path / f"missing-{section}-{field}.json"
    manifest = _manifest()
    del manifest[section][field]
    _write_manifest(path, manifest)

    with pytest.raises(ValueError, match=re.escape(section)):
        _load(str(path))


@pytest.mark.parametrize(
    ("actual_vllm_version", "actual_flashinfer_version", "error"),
    [
        ("0.20.2rc1", "0.6.8.post1+cu129", "compatibility.vllm_version"),
        ("0.20.2.dev1", "0.6.8.post1+cu129", "compatibility.vllm_version"),
        ("0.20.2+local", "0.6.8", "compatibility.flashinfer_version"),
        ("0.20.2+local", "0.6.8rc1", "compatibility.flashinfer_version"),
        ("0.20.2+local", "0.6.8.post2", "compatibility.flashinfer_version"),
    ],
)
def test_rejects_version_qualifier_mismatches(
    tmp_path: Path,
    actual_vllm_version: str,
    actual_flashinfer_version: str,
    error: str,
) -> None:
    """Only a local version label may differ from a qualified manifest version."""
    path = tmp_path / "version-mismatch.json"
    _write_manifest(path, _manifest())

    with pytest.raises(RuntimeError, match=re.escape(error)):
        _load(
            str(path),
            actual_vllm_version=actual_vllm_version,
            actual_flashinfer_version=actual_flashinfer_version,
        )


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_rejects_non_finite_json_constants(constant: str) -> None:
    """JSON parser constants cannot create non-finite qualification values."""
    raw_json = f'{{"minimum_speedup_vs_default": {constant}}}'.encode()

    with pytest.raises(ValueError, match="non-finite"):
        _MODULE._load_document(raw_json)


def test_rejects_duplicate_json_object_keys() -> None:
    """The parser cannot silently overwrite a duplicate manifest field."""
    raw_json = b'{"schema_version": 1, "schema_version": 2}'

    with pytest.raises(ValueError, match="duplicate JSON key: schema_version"):
        _MODULE._load_document(raw_json)


def test_rejects_relative_path_traversal(tmp_path: Path) -> None:
    """A relative config name is confined to the package tactic directory."""
    config_dir = tmp_path / "tactic_configs"
    config_dir.mkdir()
    _write_manifest(tmp_path / "outside.json", _manifest())

    with pytest.raises(ValueError, match="relative MXFP8 config"):
        _load("../outside.json", config_dir=config_dir)
