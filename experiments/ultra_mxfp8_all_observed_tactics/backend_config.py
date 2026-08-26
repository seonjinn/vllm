#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Resolve matching vLLM and FlashInfer settings for one MXFP8 backend."""

from __future__ import annotations

import argparse
import shlex
from dataclasses import dataclass


@dataclass(frozen=True)
class BackendConfig:
    name: str
    linear_backend: str
    oracle_backend: str
    scale_layout: str
    trtllm_layout: str


_BACKENDS = {
    "cute-dsl": BackendConfig(
        name="cute-dsl",
        linear_backend="flashinfer_cutedsl",
        oracle_backend="cute-dsl",
        scale_layout="128x4",
        trtllm_layout="8x4",
    ),
    "cutlass": BackendConfig(
        name="cutlass",
        linear_backend="flashinfer_cutlass",
        oracle_backend="cutlass",
        scale_layout="128x4",
        trtllm_layout="8x4",
    ),
    "trtllm": BackendConfig(
        name="trtllm",
        linear_backend="flashinfer_trtllm",
        oracle_backend="trtllm",
        scale_layout="8x4",
        trtllm_layout="8x4",
    ),
}


def resolve_backend(name: str) -> BackendConfig:
    try:
        return _BACKENDS[name]
    except KeyError as error:
        choices = ", ".join(sorted(_BACKENDS))
        raise ValueError(
            f"unsupported backend {name!r}; choose one of: {choices}"
        ) from error


def _shell(config: BackendConfig) -> str:
    values = {
        "BACKEND_NAME": config.name,
        "LINEAR_BACKEND": config.linear_backend,
        "ORACLE_BACKEND": config.oracle_backend,
        "SCALE_LAYOUT": config.scale_layout,
        "TRTLLM_LAYOUT": config.trtllm_layout,
    }
    return "\n".join(f"{key}={shlex.quote(value)}" for key, value in values.items())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("backend")
    parser.add_argument("--format", choices=("shell",), default="shell")
    args = parser.parse_args()
    print(_shell(resolve_backend(args.backend)))


if __name__ == "__main__":
    main()
