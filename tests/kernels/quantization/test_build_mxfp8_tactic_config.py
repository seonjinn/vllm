"""Tests for deterministic MXFP8 tactic manifest generation."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).parents[3] / "tools/mxfp8/build_tactic_config.py"
_SPEC = importlib.util.spec_from_file_location(
    "build_mxfp8_tactic_config", _MODULE_PATH
)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("cannot load MXFP8 tactic config builder")
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

build_manifest = _MODULE.build_manifest
main = _MODULE.main
parse_legacy_hints = _MODULE.parse_legacy_hints

_HINTS = "1,2048,8192:66;256,8192,2048:71;1000,2048,8192:70"
_COMPATIBILITY: dict[str, object] = {
    "vllm_version": "0.20.2",
    "vllm_base_commit": "5246e3c5df5fb8266b50ceaa6eca2836fb2d13b1",
    "flashinfer_version": "0.6.8.post1",
    "compute_capability": "10.0",
    "gpu_family": "GB200",
    "model": "Nemotron 3 Ultra MXFP8",
    "tensor_parallel_size": 4,
}
_PROVENANCE: dict[str, object] = {
    "source_manifest_sha256": "a" * 64,
    "source_hint_sha256": "b" * 64,
    "container_sha256": "c" * 64,
    "qualification_scope": "standalone_serving_seed",
    "qualification_repeat_count": 3,
    "minimum_cosine_similarity": 0.999,
    "minimum_speedup_vs_default": 1.02,
}


def test_parse_legacy_hints_returns_shapes_in_sorted_order() -> None:
    """Reordered legacy input cannot change generated tactic ordering."""
    entries = parse_legacy_hints(_HINTS)

    assert entries == (
        ((1, 2048, 8192), 66),
        ((256, 8192, 2048), 71),
        ((1000, 2048, 8192), 70),
    )


@pytest.mark.parametrize(
    "raw_hints",
    [
        "1,2048,8192:66;1,2048,8192:71",
        "1,2048:66",
        "0,2048,8192:66",
        "1,2048,8192",
    ],
)
def test_parse_legacy_hints_rejects_ambiguous_or_invalid_shapes(
    raw_hints: str,
) -> None:
    """Malformed, duplicate, and non-positive shapes cannot reach a manifest."""
    with pytest.raises(ValueError):
        parse_legacy_hints(raw_hints)


def test_build_manifest_omits_high_m_entries_for_empty_policy() -> None:
    """Unqualified high-M tactics are not silently retained in a seed manifest."""
    manifest = build_manifest(
        parse_legacy_hints(_HINTS),
        switch_m=256,
        high_m_policy="empty",
        compatibility=_COMPATIBILITY,
        provenance=_PROVENANCE,
    )

    assert manifest["tactics"] == {
        "8x4": [
            {"m": 1, "n": 2048, "k": 8192, "tactic": 66},
            {"m": 256, "n": 8192, "k": 2048, "tactic": 71},
        ],
        "128x4": [],
    }


def test_build_manifest_includes_high_m_entries_only_when_requested() -> None:
    """The explicit include policy is the sole route for high-M tactics."""
    manifest = build_manifest(
        parse_legacy_hints(_HINTS),
        switch_m=256,
        high_m_policy="include",
        compatibility=_COMPATIBILITY,
        provenance=_PROVENANCE,
    )

    assert manifest["tactics"]["128x4"] == [
        {"m": 1000, "n": 2048, "k": 8192, "tactic": 70}
    ]


def test_low_m_tactic_keys_remain_reachable_under_generated_policy() -> None:
    """Adaptive low-M lookup must use the same unpadded M keys as qualification."""
    manifest = build_manifest(
        parse_legacy_hints(_HINTS),
        switch_m=256,
        high_m_policy="empty",
        compatibility=_COMPATIBILITY,
        provenance=_PROVENANCE,
    )
    policy = manifest["policy"]
    tactics = manifest["tactics"]["8x4"]

    assert policy["pad_to_128"] is False
    tactic_shapes = {
        (entry["m"], entry["n"], entry["k"])
        for entry in tactics
    }
    for entry in tactics:
        logical_m = entry["m"]
        physical_m = (
            ((logical_m + 127) // 128) * 128
            if policy["pad_to_128"]
            else logical_m
        )
        assert (physical_m, entry["n"], entry["k"]) in tactic_shapes


def test_checked_in_low_m_seed_keys_remain_reachable() -> None:
    """The shipped 63-entry seed must preserve actual low-M benchmark shapes."""
    seed_path = (
        Path(__file__).parents[3]
        / "vllm/model_executor/kernels/linear/mxfp8/tactic_configs"
        / "nemotron3_ultra_tp4_v0202_standalone_seed.json"
    )
    seed = json.loads(seed_path.read_text(encoding="utf-8"))
    tactics = seed["tactics"]["8x4"]

    assert len(tactics) == 63
    assert seed["policy"]["pad_to_128"] is False
    assert {entry["m"] for entry in tactics} == {1, 2, 4, 6, 7, 8, 15, 23, 32}


def test_cli_writes_key_sorted_json_with_single_trailing_newline(
    tmp_path: Path,
) -> None:
    """CLI output remains reproducible for auditing and manifest hashing."""
    hints_file = tmp_path / "hints.txt"
    hints_file.write_text(_HINTS, encoding="utf-8")
    output_path = tmp_path / "seed.json"

    main(
        [
            "--hints-file",
            str(hints_file),
            "--output",
            str(output_path),
            "--switch-m",
            "256",
            "--high-m-policy",
            "empty",
            "--vllm-version",
            str(_COMPATIBILITY["vllm_version"]),
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
            "--source-manifest-sha256",
            str(_PROVENANCE["source_manifest_sha256"]),
            "--source-hint-sha256",
            str(_PROVENANCE["source_hint_sha256"]),
            "--container-sha256",
            str(_PROVENANCE["container_sha256"]),
            "--qualification-scope",
            str(_PROVENANCE["qualification_scope"]),
            "--qualification-repeat-count",
            str(_PROVENANCE["qualification_repeat_count"]),
            "--minimum-cosine-similarity",
            str(_PROVENANCE["minimum_cosine_similarity"]),
            "--minimum-speedup-vs-default",
            str(_PROVENANCE["minimum_speedup_vs_default"]),
        ]
    )

    serialized = output_path.read_text(encoding="utf-8")
    assert serialized.endswith("\n")
    assert not serialized.endswith("\n\n")
    assert serialized == json.dumps(
        json.loads(serialized), indent=2, sort_keys=True
    ) + "\n"
