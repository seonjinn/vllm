#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tune an exact-M FlashInfer cache for one observed-shape shard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shapes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backend", default="cute-dsl")
    parser.add_argument("--scale-layout", default="128x4")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    import torch
    from benchmarks.bench_cutedsl_mxfp8_serving_shapes import (
        _configure_source_jit_paths,
        _tune_mode,
        group_shapes,
        load_shapes,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("exact cache generation requires a CUDA GPU")
    _configure_source_jit_paths()
    shapes = load_shapes(args.shapes)
    result = _tune_mode(
        mode="exact",
        groups=group_shapes(shapes),
        output_dir=args.output_dir,
        seed=args.seed,
        backend=args.backend,
        scale_layout=args.scale_layout,
    )
    (args.output_dir / "exact_cache_metadata.json").write_text(
        json.dumps(
            {
                "gpu": torch.cuda.get_device_name(0),
                "shape_count": len(shapes),
                **result,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
