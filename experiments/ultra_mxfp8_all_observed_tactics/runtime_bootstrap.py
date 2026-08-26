# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Join the pinned vLLM Python tree to the container's compiled runtime."""

from __future__ import annotations

import os
import site
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


def _installed_package_path(name: str, excluded_roots: tuple[Path, ...]) -> Path:
    candidates = [Path(root) / name for root in site.getsitepackages()]
    candidates.extend(Path(root) / name for root in sys.path if root)
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_dir() and not any(
            resolved.is_relative_to(root.resolve()) for root in excluded_roots
        ):
            return resolved
    raise RuntimeError(f"cannot find installed {name} package outside {excluded_roots}")


def configure_runtime() -> None:
    source_root = Path(os.environ["SOURCE_ROOT"])
    flashinfer_root = Path(os.environ["FLASHINFER_ROOT"])

    import vllm

    _verify_import(vllm, source_root, "vLLM")
    installed_vllm = _installed_package_path("vllm", (source_root,))
    installed_vllm_text = str(installed_vllm)
    if installed_vllm_text not in vllm.__path__:
        vllm.__path__.append(installed_vllm_text)

    import vllm._C

    _verify_import(vllm._C, installed_vllm, "vLLM compiled extension")

    import flashinfer

    installed_flashinfer = _installed_package_path(
        "flashinfer", (source_root, flashinfer_root)
    )
    _verify_import(flashinfer, installed_flashinfer, "FlashInfer")

    expected_vllm_version = os.environ["EXPECTED_VLLM_VERSION"]
    if vllm.__version__ != expected_vllm_version:
        raise RuntimeError(
            "vLLM version mismatch: "
            f"expected={expected_vllm_version}, actual={vllm.__version__}"
        )
