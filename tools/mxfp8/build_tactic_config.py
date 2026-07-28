# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Build versioned MXFP8 dense tactic manifests from legacy hints."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal, cast

Shape = tuple[int, int, int]


def parse_legacy_hints(raw_hints: str) -> tuple[tuple[Shape, int], ...]:
    """Parse non-empty ``M,N,K:tactic`` records into a sorted unique tuple."""
    entries: list[tuple[Shape, int]] = []
    shapes: set[Shape] = set()
    for raw_entry in raw_hints.split(";"):
        entry = raw_entry.strip()
        if not entry:
            raise ValueError("legacy MXFP8 tactic hints must not contain empty entries")
        shape_text, separator, tactic_text = entry.partition(":")
        if not separator or ":" in tactic_text:
            raise ValueError(f"invalid legacy MXFP8 tactic hint: {entry!r}")
        dimensions = shape_text.split(",")
        if len(dimensions) != 3:
            raise ValueError(f"invalid legacy MXFP8 shape: {shape_text!r}")
        try:
            shape = cast(Shape, tuple(int(dimension) for dimension in dimensions))
            tactic = int(tactic_text)
        except ValueError as error:
            raise ValueError(f"invalid legacy MXFP8 tactic hint: {entry!r}") from error
        if any(dimension <= 0 for dimension in shape):
            raise ValueError(f"legacy MXFP8 shape must be positive: {shape!r}")
        if shape in shapes:
            raise ValueError(f"duplicate legacy MXFP8 shape: {shape!r}")
        shapes.add(shape)
        entries.append((shape, tactic))
    return tuple(sorted(entries, key=lambda item: item[0]))


def build_manifest(
    entries: tuple[tuple[Shape, int], ...],
    *,
    switch_m: int,
    high_m_policy: Literal["empty", "include"],
    compatibility: Mapping[str, object],
    provenance: Mapping[str, object],
) -> dict[str, object]:
    """Build a schema-1 adaptive manifest with an explicit high-M policy."""
    if isinstance(switch_m, bool) or not isinstance(switch_m, int) or switch_m <= 0:
        raise ValueError("switch_m must be a positive integer")
    if switch_m % 128:
        raise ValueError("switch_m must be divisible by 128")
    if high_m_policy not in {"empty", "include"}:
        raise ValueError("high_m_policy must be 'empty' or 'include'")

    low_entries: list[dict[str, int]] = []
    high_entries: list[dict[str, int]] = []
    shapes: set[Shape] = set()
    for shape, tactic in entries:
        _validate_entry(shape, tactic)
        if shape in shapes:
            raise ValueError(f"duplicate MXFP8 shape: {shape!r}")
        shapes.add(shape)
        serialized = {"m": shape[0], "n": shape[1], "k": shape[2], "tactic": tactic}
        if shape[0] <= switch_m:
            low_entries.append(serialized)
        elif high_m_policy == "include":
            high_entries.append(serialized)

    return {
        "schema_version": 1,
        "mode": "adaptive",
        "compatibility": dict(compatibility),
        "policy": {
            "gemm_backend": "trtllm",
            "layout": "adaptive",
            "switch_m": switch_m,
            "direct_trtllm": True,
            "require_direct_trtllm": True,
            "quant_backend": "cuda",
            "require_8x4_quant": True,
            "pad_to_128": True,
            "default_tactic": -1,
        },
        "tactics": {"8x4": low_entries, "128x4": high_entries},
        "provenance": dict(provenance),
    }


def _validate_entry(shape: Shape, tactic: int) -> None:
    if len(shape) != 3 or any(
        isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0
        for dimension in shape
    ):
        raise ValueError(f"MXFP8 shape must contain three positive integers: {shape!r}")
    if isinstance(tactic, bool) or not isinstance(tactic, int):
        raise ValueError(f"MXFP8 tactic must be an integer: {tactic!r}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hints-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--switch-m", type=int, required=True)
    parser.add_argument(
        "--high-m-policy", choices=("empty", "include"), required=True
    )
    parser.add_argument("--vllm-version", required=True)
    parser.add_argument("--vllm-base-commit", required=True)
    parser.add_argument("--flashinfer-version", required=True)
    parser.add_argument("--compute-capability", required=True)
    parser.add_argument("--gpu-family", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tensor-parallel-size", type=int, required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--source-hint-sha256", required=True)
    parser.add_argument("--container-sha256", required=True)
    parser.add_argument("--qualification-scope", required=True)
    parser.add_argument("--qualification-repeat-count", type=int, required=True)
    parser.add_argument("--minimum-cosine-similarity", type=float, required=True)
    parser.add_argument("--minimum-speedup-vs-default", type=float, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Build a manifest from command-line arguments and write canonical JSON."""
    args = _parser().parse_args(argv)
    compatibility: dict[str, object] = {
        "vllm_version": args.vllm_version,
        "vllm_base_commit": args.vllm_base_commit,
        "flashinfer_version": args.flashinfer_version,
        "compute_capability": args.compute_capability,
        "gpu_family": args.gpu_family,
        "model": args.model,
        "tensor_parallel_size": args.tensor_parallel_size,
    }
    provenance: dict[str, object] = {
        "source_manifest_sha256": args.source_manifest_sha256,
        "source_hint_sha256": args.source_hint_sha256,
        "container_sha256": args.container_sha256,
        "qualification_scope": args.qualification_scope,
        "qualification_repeat_count": args.qualification_repeat_count,
        "minimum_cosine_similarity": args.minimum_cosine_similarity,
        "minimum_speedup_vs_default": args.minimum_speedup_vs_default,
    }
    manifest = build_manifest(
        parse_legacy_hints(args.hints_file.read_text(encoding="utf-8").strip()),
        switch_m=args.switch_m,
        high_m_policy=cast(Literal["empty", "include"], args.high_m_policy),
        compatibility=compatibility,
        provenance=provenance,
    )
    args.output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
