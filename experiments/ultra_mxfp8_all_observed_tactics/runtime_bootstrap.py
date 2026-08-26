# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Verify the container vLLM and configure the pinned FlashInfer overlay."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType


def _verify_import(module: ModuleType, root: Path, name: str) -> None:
    module_path = Path(module.__file__ or "").resolve()
    if not module_path.is_relative_to(root.resolve()):
        raise RuntimeError(
            f"{name} imported outside pinned checkout: module={module_path}, "
            f"checkout={root}"
        )


def _configure_flashinfer_source_jit(root: Path) -> None:
    helper_path = root / "benchmarks" / "bench_cutedsl_mxfp8_serving_shapes.py"
    spec = importlib.util.spec_from_file_location(
        "_mxfp8_flashinfer_source_jit", helper_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load FlashInfer JIT helper: {helper_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module._configure_source_jit_paths()


def configure_runtime() -> None:
    flashinfer_root = Path(os.environ["FLASHINFER_ROOT"])

    import flashinfer

    _verify_import(flashinfer, flashinfer_root, "FlashInfer")
    _configure_flashinfer_source_jit(flashinfer_root)

    import vllm

    expected_vllm_version = os.environ["EXPECTED_VLLM_VERSION"]
    if vllm.__version__ != expected_vllm_version:
        raise RuntimeError(
            "vLLM version mismatch: "
            f"expected={expected_vllm_version}, actual={vllm.__version__}"
        )
