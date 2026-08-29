#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Combine layout-specific MXFP8 tactic lookups under one adaptive policy."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_PROVENANCE_KEYS = (
    "backend",
    "flashinfer_commit",
    "flashinfer_version",
    "flashinfer_file",
    "container_sha256",
    "gpu",
)


def _load_component(path: Path, expected_layout: str) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("format_version") != 1:
        raise ValueError(f"unsupported lookup format: {path}")
    if payload.get("backend") != "trtllm":
        raise ValueError(f"adaptive lookup requires TRTLLM backend: {path}")
    if payload.get("scale_layout") != expected_layout:
        raise ValueError(
            f"expected {expected_layout} lookup, got {payload.get('scale_layout')}"
        )
    entries = payload.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"lookup contains no entries: {path}")
    if payload.get("entry_count") != len(entries):
        raise ValueError(f"lookup entry_count mismatch: {path}")
    return payload


def combine_adaptive_lookups(
    lookup_8x4_path: Path,
    lookup_128x4_path: Path,
    output_path: Path,
    *,
    switch_m: int,
) -> dict[str, Any]:
    if switch_m <= 0:
        raise ValueError("switch_m must be positive")
    components = {
        "8x4": _load_component(lookup_8x4_path, "8x4"),
        "128x4": _load_component(lookup_128x4_path, "128x4"),
    }
    for key in _PROVENANCE_KEYS:
        values = {payload.get(key) for payload in components.values()}
        if len(values) != 1 or None in values or "" in values:
            raise ValueError(
                f"component lookup provenance mismatch for {key}: {values}"
            )

    entries = []
    seen = set()
    for layout, payload in components.items():
        for entry in payload["entries"]:
            m = int(entry["m"])
            if layout == "8x4" and m > switch_m:
                raise ValueError(f"8x4 entry exceeds switch_m: M={m}")
            if layout == "128x4" and m <= switch_m:
                raise ValueError(f"128x4 entry does not exceed switch_m: M={m}")
            key = (m, int(entry["n"]), int(entry["k"]), str(entry["runner"]))
            if key in seen:
                raise ValueError(f"duplicate adaptive lookup entry: {key}")
            seen.add(key)
            entries.append(entry)
    entries.sort(
        key=lambda entry: (
            int(entry["m"]),
            int(entry["n"]),
            int(entry["k"]),
            str(entry["runner"]),
        )
    )

    payload = {
        "format_version": 1,
        "created_utc": datetime.now(UTC).isoformat(),
        "backend": components["8x4"]["backend"],
        "scale_layout": "adaptive",
        "scale_layout_components": ["8x4", "128x4"],
        "switch_m": switch_m,
        **{key: components["8x4"][key] for key in _PROVENANCE_KEYS if key != "backend"},
        "entry_count": len(entries),
        "component_lookups": {
            "8x4": str(lookup_8x4_path),
            "128x4": str(lookup_128x4_path),
        },
        "entries": entries,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lookup-8x4", type=Path, required=True)
    parser.add_argument("--lookup-128x4", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--switch-m", type=int, required=True)
    args = parser.parse_args()
    result = combine_adaptive_lookups(
        args.lookup_8x4,
        args.lookup_128x4,
        args.output,
        switch_m=args.switch_m,
    )
    print(json.dumps({"entry_count": result["entry_count"]}))


if __name__ == "__main__":
    main()
