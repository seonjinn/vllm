# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Fail-closed loading of qualified MXFP8 dense tactic manifests."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Literal, cast

from packaging.version import InvalidVersion, Version

Shape = tuple[int, int, int]

_TOP_LEVEL_FIELDS = frozenset(
    {"schema_version", "mode", "compatibility", "policy", "tactics", "provenance"}
)
_COMPATIBILITY_FIELDS = frozenset(
    {
        "vllm_version",
        "vllm_base_commit",
        "flashinfer_version",
        "compute_capability",
        "gpu_family",
        "model",
        "tensor_parallel_size",
    }
)
_POLICY_FIELDS = frozenset(
    {
        "gemm_backend",
        "layout",
        "switch_m",
        "direct_trtllm",
        "require_direct_trtllm",
        "quant_backend",
        "require_8x4_quant",
        "pad_to_128",
        "default_tactic",
    }
)
_TACTIC_FIELDS = frozenset({"m", "n", "k", "tactic"})
_PROVENANCE_FIELDS = frozenset(
    {
        "source_manifest_sha256",
        "source_hint_sha256",
        "container_sha256",
        "qualification_scope",
        "qualification_repeat_count",
        "minimum_cosine_similarity",
        "minimum_speedup_vs_default",
    }
)


@dataclass(frozen=True)
class Mxfp8DenseRuntimeConfig:
    source_path: Path
    source_sha256: str
    mode: Literal["adaptive"]
    switch_m: int
    gemm_backend: Literal["trtllm"]
    layout: Literal["adaptive"]
    direct_trtllm: bool
    require_direct_trtllm: bool
    quant_backend: Literal["cuda", "flashinfer"]
    require_8x4_quant: bool
    pad_to_128: bool
    default_tactic: int
    tactics_8x4: tuple[tuple[Shape, int], ...]
    tactics_128x4: tuple[tuple[Shape, int], ...]
    compatibility: Mapping[str, object]
    provenance: Mapping[str, object]


def load_mxfp8_dense_runtime_config(
    reference: str,
    *,
    actual_vllm_version: str,
    actual_flashinfer_version: str,
    actual_compute_capability: tuple[int, int],
    actual_model: str,
    actual_tensor_parallel_size: int,
    package_config_dir: Path | None = None,
) -> Mxfp8DenseRuntimeConfig:
    """Load a qualified tactic manifest and reject every incompatible input."""
    source_path, source_bytes = _read_config(reference, package_config_dir)
    document = _load_document(source_bytes)

    _require_exact_fields(document, _TOP_LEVEL_FIELDS, "config")
    _require_equal(document, "schema_version", 1, "schema_version")
    _require_equal(document, "mode", "adaptive", "mode")

    compatibility = _require_mapping(document, "compatibility", "compatibility")
    _validate_compatibility(
        compatibility,
        actual_vllm_version,
        actual_flashinfer_version,
        actual_compute_capability,
        actual_model,
        actual_tensor_parallel_size,
    )
    policy = _require_mapping(document, "policy", "policy")
    validated_policy = _validate_policy(policy)
    tactics = _require_mapping(document, "tactics", "tactics")
    tactics_8x4, tactics_128x4 = _validate_tactics(tactics)
    provenance = _require_mapping(document, "provenance", "provenance")
    _validate_provenance(provenance)

    return Mxfp8DenseRuntimeConfig(
        source_path=source_path,
        source_sha256=hashlib.sha256(source_bytes).hexdigest(),
        mode="adaptive",
        switch_m=validated_policy.switch_m,
        gemm_backend=validated_policy.gemm_backend,
        layout="adaptive",
        direct_trtllm=validated_policy.direct_trtllm,
        require_direct_trtllm=validated_policy.require_direct_trtllm,
        quant_backend=validated_policy.quant_backend,
        require_8x4_quant=validated_policy.require_8x4_quant,
        pad_to_128=validated_policy.pad_to_128,
        default_tactic=validated_policy.default_tactic,
        tactics_8x4=tactics_8x4,
        tactics_128x4=tactics_128x4,
        compatibility=MappingProxyType(dict(compatibility)),
        provenance=MappingProxyType(dict(provenance)),
    )


@dataclass(frozen=True)
class _ValidatedPolicy:
    switch_m: int
    gemm_backend: Literal["trtllm"]
    direct_trtllm: bool
    require_direct_trtllm: bool
    quant_backend: Literal["cuda", "flashinfer"]
    require_8x4_quant: bool
    pad_to_128: bool
    default_tactic: int


def _read_config(reference: str, package_config_dir: Path | None) -> tuple[Path, bytes]:
    if not reference:
        raise ValueError("reference must name an MXFP8 config file")
    config_dir = package_config_dir or Path(__file__).with_name("tactic_configs")
    config_dir = config_dir.resolve()
    reference_path = Path(reference)
    if reference_path.is_absolute():
        candidate = reference_path.resolve()
    else:
        candidate = (config_dir / reference).resolve()
        if not candidate.is_relative_to(config_dir):
            raise ValueError("relative MXFP8 config must stay inside tactic_configs")
    try:
        return candidate, candidate.read_bytes()
    except OSError as error:
        raise RuntimeError(f"unable to read MXFP8 config: {candidate}") from error


def _load_document(source_bytes: bytes) -> Mapping[str, object]:
    try:
        parsed = cast(
            object,
            json.loads(
                source_bytes,
                object_pairs_hook=_reject_duplicate_json_keys,
                parse_constant=_reject_non_finite_json_constant,
            ),
        )
    except (TypeError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("MXFP8 config must contain valid JSON") from error
    if not isinstance(parsed, dict) or not all(
        isinstance(key, str) for key in parsed
    ):
        raise ValueError("config must be a JSON object")
    return cast(Mapping[str, object], parsed)


def _reject_duplicate_json_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    parsed: dict[str, object] = {}
    for key, value in pairs:
        if key in parsed:
            raise ValueError(f"duplicate JSON key: {key}")
        parsed[key] = value
    return parsed


def _reject_non_finite_json_constant(value: str) -> None:
    raise ValueError(f"JSON must not contain non-finite constant: {value}")


def _require_exact_fields(
    values: Mapping[str, object], expected: frozenset[str], field: str
) -> None:
    missing = expected - values.keys()
    extra = values.keys() - expected
    if missing:
        raise ValueError(f"{field} missing required fields: {sorted(missing)}")
    if extra:
        raise ValueError(f"{field} has unsupported fields: {sorted(extra)}")


def _require_equal(
    values: Mapping[str, object], key: str, expected: object, field: str
) -> None:
    if values.get(key) != expected or (
        isinstance(expected, int) and isinstance(values.get(key), bool)
    ):
        raise ValueError(f"{field} must be {expected!r}")


def _require_mapping(
    values: Mapping[str, object], key: str, field: str
) -> Mapping[str, object]:
    value = values.get(key)
    if not isinstance(value, dict) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{field} must be an object")
    return cast(Mapping[str, object], value)


def _validate_compatibility(
    compatibility: Mapping[str, object],
    actual_vllm_version: str,
    actual_flashinfer_version: str,
    actual_compute_capability: tuple[int, int],
    actual_model: str,
    actual_tensor_parallel_size: int,
) -> None:
    _require_exact_fields(compatibility, _COMPATIBILITY_FIELDS, "compatibility")
    for key in (
        "vllm_version",
        "vllm_base_commit",
        "flashinfer_version",
        "compute_capability",
        "gpu_family",
        "model",
    ):
        _require_nonempty_string(compatibility, key, f"compatibility.{key}")
    _require_positive_integer(
        compatibility, "tensor_parallel_size", "compatibility.tensor_parallel_size"
    )
    _require_matching_base_version(
        compatibility["vllm_version"], actual_vllm_version, "compatibility.vllm_version"
    )
    _require_matching_base_version(
        compatibility["flashinfer_version"],
        actual_flashinfer_version,
        "compatibility.flashinfer_version",
    )
    major, minor = actual_compute_capability
    expected_capability = f"{major}.{minor}"
    if compatibility["compute_capability"] != expected_capability:
        raise RuntimeError(
            "compatibility.compute_capability does not match actual compute "
            "capability"
        )
    if compatibility["model"] != actual_model:
        raise RuntimeError("compatibility.model does not match the active model")
    if compatibility["tensor_parallel_size"] != actual_tensor_parallel_size:
        raise RuntimeError(
            "compatibility.tensor_parallel_size does not match the active "
            "tensor parallel size"
        )


def _require_matching_base_version(
    expected: object, actual: str, field: str
) -> None:
    if not isinstance(expected, str):
        raise ValueError(f"{field} must be a version string")
    try:
        expected_public = Version(expected).public
        actual_public = Version(actual).public
    except InvalidVersion as error:
        raise ValueError(f"{field} must be a valid version") from error
    if expected_public != actual_public:
        raise RuntimeError(f"{field} does not match the active public version")


def _validate_policy(policy: Mapping[str, object]) -> _ValidatedPolicy:
    _require_exact_fields(policy, _POLICY_FIELDS, "policy")
    _require_equal(policy, "gemm_backend", "trtllm", "policy.gemm_backend")
    _require_equal(policy, "layout", "adaptive", "policy.layout")
    _require_positive_integer(policy, "switch_m", "policy.switch_m")
    switch_m = cast(int, policy["switch_m"])
    if switch_m % 128:
        raise ValueError("policy.switch_m must be divisible by 128")
    for key in (
        "direct_trtllm",
        "require_direct_trtllm",
        "require_8x4_quant",
        "pad_to_128",
    ):
        _require_bool(policy, key, f"policy.{key}")
    for key in ("direct_trtllm", "require_direct_trtllm"):
        if policy[key] is not True:
            raise ValueError(f"policy.{key} must be true for adaptive execution")
    quant_backend = policy["quant_backend"]
    if not isinstance(quant_backend, str) or quant_backend not in {
        "cuda",
        "flashinfer",
    }:
        raise ValueError("policy.quant_backend must be 'cuda' or 'flashinfer'")
    _require_integer(policy, "default_tactic", "policy.default_tactic")
    if policy["default_tactic"] != -1:
        raise ValueError("policy.default_tactic must be -1")
    return _ValidatedPolicy(
        switch_m=switch_m,
        gemm_backend="trtllm",
        direct_trtllm=cast(bool, policy["direct_trtllm"]),
        require_direct_trtllm=cast(bool, policy["require_direct_trtllm"]),
        quant_backend=cast(Literal["cuda", "flashinfer"], quant_backend),
        require_8x4_quant=cast(bool, policy["require_8x4_quant"]),
        pad_to_128=cast(bool, policy["pad_to_128"]),
        default_tactic=cast(int, policy["default_tactic"]),
    )


def _validate_tactics(
    tactics: Mapping[str, object],
) -> tuple[tuple[tuple[Shape, int], ...], tuple[tuple[Shape, int], ...]]:
    _require_exact_fields(tactics, frozenset({"8x4", "128x4"}), "tactics")
    tactics_8x4 = _validate_tactic_layout(tactics, "8x4")
    tactics_128x4 = _validate_tactic_layout(tactics, "128x4")
    overlap = {shape for shape, _ in tactics_8x4} & {
        shape for shape, _ in tactics_128x4
    }
    if overlap:
        raise ValueError("tactics must not share a shape between layouts")
    return tactics_8x4, tactics_128x4


def _validate_tactic_layout(
    tactics: Mapping[str, object], layout: Literal["8x4", "128x4"]
) -> tuple[tuple[Shape, int], ...]:
    entries = tactics[layout]
    if not isinstance(entries, list):
        raise ValueError(f"tactics.{layout} must be an array")
    parsed: list[tuple[Shape, int]] = []
    shapes: set[Shape] = set()
    for index, entry in enumerate(entries):
        field = f"tactics.{layout}[{index}]"
        if not isinstance(entry, dict) or not all(
            isinstance(key, str) for key in entry
        ):
            raise ValueError(f"{field} must be an object")
        mapping = cast(Mapping[str, object], entry)
        _require_exact_fields(mapping, _TACTIC_FIELDS, field)
        dimensions: list[int] = []
        for dimension in ("m", "n", "k"):
            dimension_field = f"{field}.{dimension}"
            _require_positive_integer(mapping, dimension, dimension_field)
            dimensions.append(cast(int, mapping[dimension]))
        _require_integer(mapping, "tactic", f"{field}.tactic")
        shape = cast(Shape, tuple(dimensions))
        if shape in shapes:
            raise ValueError(f"tactics.{layout} has duplicate shape {shape}")
        shapes.add(shape)
        parsed.append((shape, cast(int, mapping["tactic"])))
    return tuple(sorted(parsed, key=lambda item: item[0]))


def _validate_provenance(provenance: Mapping[str, object]) -> None:
    _require_exact_fields(provenance, _PROVENANCE_FIELDS, "provenance")
    for key in (
        "source_manifest_sha256",
        "source_hint_sha256",
        "container_sha256",
    ):
        value = provenance[key]
        if not isinstance(value, str) or len(value) != 64 or any(
            character not in "0123456789abcdefABCDEF" for character in value
        ):
            raise ValueError(f"provenance.{key} must be a SHA-256 hexadecimal digest")
    _require_nonempty_string(
        provenance,
        "qualification_scope",
        "provenance.qualification_scope",
    )
    _require_positive_integer(
        provenance,
        "qualification_repeat_count",
        "provenance.qualification_repeat_count",
    )
    for key in ("minimum_cosine_similarity", "minimum_speedup_vs_default"):
        value = provenance[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"provenance.{key} must be a number")
        if not math.isfinite(value):
            raise ValueError(f"provenance.{key} must be finite")
    cosine = cast(float | int, provenance["minimum_cosine_similarity"])
    if not 0 <= cosine <= 1:
        raise ValueError("provenance.minimum_cosine_similarity must be between 0 and 1")
    speedup = cast(float | int, provenance["minimum_speedup_vs_default"])
    if speedup <= 0:
        raise ValueError("provenance.minimum_speedup_vs_default must be positive")


def _require_nonempty_string(
    values: Mapping[str, object], key: str, field: str
) -> None:
    value = values.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")


def _require_integer(values: Mapping[str, object], key: str, field: str) -> None:
    value = values.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")


def _require_positive_integer(
    values: Mapping[str, object], key: str, field: str
) -> None:
    _require_integer(values, key, field)
    if cast(int, values[key]) <= 0:
        raise ValueError(f"{field} must be positive")


def _require_bool(values: Mapping[str, object], key: str, field: str) -> None:
    if not isinstance(values.get(key), bool):
        raise ValueError(f"{field} must be a boolean")
