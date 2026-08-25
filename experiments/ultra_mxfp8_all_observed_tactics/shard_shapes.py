#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Split observed shapes across GPUs without splitting `(N, K)` families."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def shard_shapes(observed_path: Path, output_dir: Path, shard_count: int) -> list[Path]:
    if shard_count <= 0:
        raise ValueError("shard_count must be positive")

    with observed_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"observed shape CSV is empty: {observed_path}")

    families: defaultdict[tuple[int, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        families[(int(row["n"]), int(row["k"]))].append(row)

    shards: list[list[dict[str, str]]] = [[] for _ in range(shard_count)]
    loads = [0] * shard_count
    for _, family_rows in sorted(
        families.items(), key=lambda item: (-len(item[1]), item[0])
    ):
        target = min(range(shard_count), key=lambda index: (loads[index], index))
        shards[target].extend(family_rows)
        loads[target] += len(family_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    fieldnames = ("m", "n", "k")
    for index, shard in enumerate(shards):
        path = output_dir / f"shard_{index}.csv"
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in sorted(
                shard, key=lambda item: tuple(int(item[name]) for name in fieldnames)
            ):
                writer.writerow({name: row[name] for name in fieldnames})
        outputs.append(path)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--observed", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shards", type=int, required=True)
    args = parser.parse_args()
    for path in shard_shapes(args.observed, args.output_dir, args.shards):
        print(path)


if __name__ == "__main__":
    main()
