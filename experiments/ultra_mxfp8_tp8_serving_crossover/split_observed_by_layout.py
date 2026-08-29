#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Partition observed MXFP8 shapes by the adaptive activation-scale layout."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def split_observed_by_layout(
    observed_path: Path, output_dir: Path, *, switch_m: int
) -> dict[str, Path]:
    if switch_m <= 0:
        raise ValueError("switch_m must be positive")
    with observed_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = list(reader)
    if not fieldnames or not rows:
        raise ValueError(f"observed shape CSV is empty: {observed_path}")

    partitions = {
        "8x4": [row for row in rows if int(row["m"]) <= switch_m],
        "128x4": [row for row in rows if int(row["m"]) > switch_m],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for layout, layout_rows in partitions.items():
        if not layout_rows:
            raise ValueError(f"adaptive {layout} partition is empty")
        path = output_dir / f"observed_{layout}.csv"
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(layout_rows)
        outputs[layout] = path
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--observed", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--switch-m", type=int, required=True)
    args = parser.parse_args()
    outputs = split_observed_by_layout(
        args.observed, args.output_dir, switch_m=args.switch_m
    )
    for layout, path in outputs.items():
        print(f"{layout}={path}")


if __name__ == "__main__":
    main()
