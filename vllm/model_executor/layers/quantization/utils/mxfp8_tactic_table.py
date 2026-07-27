# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Fail-closed loader for versioned MXFP8 TRTLLM tactic artifacts."""

import hashlib
import json
import re
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from types import MappingProxyType
from typing import Any

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_SCHEMA_VERSION = 1
_LAYOUTS = frozenset(("8x4", "128x4"))


@dataclass(frozen=True)
class Mxfp8TacticKey:
    """Execution signature used to select an MXFP8 TRTLLM tactic."""

    m_logical: int
    n_logical: int
    k_logical: int
    n_physical: int
    k_physical: int
    activation_scale_layout: str
    output_dtype: str


@dataclass(frozen=True)
class RuntimeProvenance:
    """Immutable runtime identity that must match a tactic artifact."""

    vllm_version: str
    flashinfer_version: str
    torch_version: str
    cuda_version: str
    driver_version: str
    gpu: str
    topology: str
    checkpoint_id: str
    source_commit: str
    container_digest: str
    adaptive_switch_m: int
    weight_contract: str


@dataclass(frozen=True)
class Mxfp8TacticArtifact:
    """Validated MXFP8 tactic table with immutable exact-key lookup."""

    provenance: RuntimeProvenance
    tactics: Mapping[Mxfp8TacticKey, int]

    def lookup(self, key: Mxfp8TacticKey) -> int | None:
        """Return the selected tactic for an exact execution signature."""
        return self.tactics.get(key)


def load_mxfp8_tactic_artifact(
    path: Path,
    expected_sha256: str,
    expected: RuntimeProvenance,
) -> Mxfp8TacticArtifact:
    """Load a schema-1 tactic artifact after integrity and provenance checks.

    Args:
        path: Artifact JSON file produced by offline tactic selection.
        expected_sha256: Lowercase SHA-256 digest configured by the launcher.
        expected: Immutable serving provenance configured by the launcher.

    Returns:
        An immutable exact-match tactic artifact.

    Raises:
        ValueError: If the artifact is malformed or does not match its runtime.
    """
    _require_sha256(expected_sha256, "expected SHA256")
    try:
        contents = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"Unable to read MXFP8 tactic artifact {path}.") from exc

    actual_sha256 = hashlib.sha256(contents).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError("MXFP8 tactic artifact SHA256 does not match configuration.")

    try:
        artifact = json.loads(contents)
    except json.JSONDecodeError as exc:
        raise ValueError("MXFP8 tactic artifact is not valid JSON.") from exc
    if not isinstance(artifact, dict):
        raise ValueError("MXFP8 tactic artifact root must be an object.")

    schema_version = artifact.get("schema_version")
    if type(schema_version) is not int or schema_version != _SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported MXFP8 tactic artifact schema {schema_version!r}."
        )

    provenance = _parse_provenance(artifact.get("metadata"))
    _require_matching_provenance(provenance, expected)
    _require_matching_local_runtime(expected)

    entries = artifact.get("entries")
    if not isinstance(entries, list):
        raise ValueError("MXFP8 tactic artifact entries must be a list.")

    tactics: dict[Mxfp8TacticKey, int] = {}
    for index, raw_entry in enumerate(entries):
        key, tactic = _parse_entry(raw_entry, index)
        if key in tactics:
            raise ValueError(f"Duplicate MXFP8 tactic artifact key at entry {index}.")
        tactics[key] = tactic

    return Mxfp8TacticArtifact(provenance, MappingProxyType(tactics))


def _parse_provenance(raw_metadata: Any) -> RuntimeProvenance:
    if not isinstance(raw_metadata, dict):
        raise ValueError("MXFP8 tactic artifact metadata must be an object.")

    values: dict[str, Any] = {}
    for field in fields(RuntimeProvenance):
        value = raw_metadata.get(field.name)
        if field.name == "adaptive_switch_m":
            _require_positive_int(value, f"metadata.{field.name}")
        elif not isinstance(value, str) or not value:
            raise ValueError(f"metadata.{field.name} must be a non-empty string.")
        values[field.name] = value
    return RuntimeProvenance(**values)


def _require_matching_provenance(
    artifact: RuntimeProvenance,
    expected: RuntimeProvenance,
) -> None:
    for field in fields(RuntimeProvenance):
        if getattr(artifact, field.name) != getattr(expected, field.name):
            raise ValueError(
                "MXFP8 tactic artifact provenance does not match runtime "
                f"for {field.name}."
            )


def _require_matching_local_runtime(expected: RuntimeProvenance) -> None:
    for field_name, actual in _local_runtime_values().items():
        if actual != getattr(expected, field_name):
            raise ValueError(
                "MXFP8 tactic artifact provenance does not match worker-local "
                f"runtime for {field_name}."
            )


def _local_runtime_values() -> dict[str, str]:
    try:
        import flashinfer
        import torch

        from vllm import __version__ as vllm_version

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        cuda_version = torch.version.cuda
        if cuda_version is None:
            raise RuntimeError("PyTorch does not report a CUDA version")
        driver_version = (
            subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                check=True,
                capture_output=True,
                text=True,
            )
            .stdout.splitlines()[0]
            .strip()
        )
        if not driver_version:
            raise RuntimeError("nvidia-smi did not report a driver version")
        return {
            "vllm_version": vllm_version,
            "flashinfer_version": flashinfer.__version__,
            "torch_version": torch.__version__,
            "cuda_version": cuda_version,
            "driver_version": driver_version,
            "gpu": torch.cuda.get_device_name(),
        }
    except Exception as exc:
        raise ValueError(
            "Unable to query MXFP8 worker-local runtime provenance."
        ) from exc


def _parse_entry(raw_entry: Any, index: int) -> tuple[Mxfp8TacticKey, int]:
    if not isinstance(raw_entry, dict):
        raise ValueError(f"MXFP8 tactic artifact entry {index} must be an object.")

    raw_key = raw_entry.get("key")
    if not isinstance(raw_key, dict):
        raise ValueError(f"MXFP8 tactic artifact entry {index} key must be an object.")
    key = _parse_key(raw_key, index)

    tactic = _require_tactic(raw_entry.get("selected_tactic"), f"entry {index}")

    legal_tactics = raw_entry.get("legal_tactics")
    if not isinstance(legal_tactics, list):
        raise ValueError(f"entry {index} legal_tactics must be a list.")
    validated_legal_tactics = [
        _require_tactic(legal_tactic, f"entry {index} legal_tactics")
        for legal_tactic in legal_tactics
    ]
    if tactic >= 0 and tactic not in validated_legal_tactics:
        raise ValueError(f"entry {index} selected_tactic must appear in legal_tactics.")

    return key, tactic


def _parse_key(raw_key: dict[str, Any], index: int) -> Mxfp8TacticKey:
    integer_fields = (
        "m_logical",
        "n_logical",
        "k_logical",
        "n_physical",
        "k_physical",
    )
    for field_name in integer_fields:
        _require_positive_int(raw_key.get(field_name), f"entry {index} {field_name}")

    n_logical = raw_key["n_logical"]
    n_physical = raw_key["n_physical"]
    k_logical = raw_key["k_logical"]
    k_physical = raw_key["k_physical"]
    if n_physical < n_logical:
        raise ValueError(f"entry {index} n_physical cannot be less than n_logical.")
    if k_physical < k_logical:
        raise ValueError(f"entry {index} k_physical cannot be less than k_logical.")
    if n_physical % 128:
        raise ValueError(f"entry {index} n_physical must be divisible by 128.")
    if k_physical % 256:
        raise ValueError(f"entry {index} k_physical must be divisible by 256.")

    layout = raw_key.get("activation_scale_layout")
    if not isinstance(layout, str) or layout not in _LAYOUTS:
        raise ValueError(
            f"entry {index} has unsupported MXFP8 scale layout {layout!r}."
        )
    output_dtype = raw_key.get("output_dtype")
    if output_dtype != "bfloat16":
        raise ValueError(
            f"entry {index} has unsupported output dtype {output_dtype!r}."
        )

    return Mxfp8TacticKey(
        m_logical=raw_key["m_logical"],
        n_logical=n_logical,
        k_logical=k_logical,
        n_physical=n_physical,
        k_physical=k_physical,
        activation_scale_layout=layout,
        output_dtype=output_dtype,
    )


def _require_sha256(value: str, name: str) -> None:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase 64-character hexadecimal string.")


def _require_positive_int(value: Any, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")


def _require_tactic(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < -1:
        raise ValueError(f"{name} must be -1 or a nonnegative integer.")
    return value
