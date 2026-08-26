# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Join the pinned vLLM Python tree to the container's compiled runtime."""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import os
import site
import sys
from pathlib import Path
from types import ModuleType


class _CombinedVllmPackageFinder(importlib.abc.MetaPathFinder):
    def __init__(self, source_package: Path, installed_package: Path) -> None:
        self.source_package = source_package
        self.installed_package = installed_package

    def find_spec(
        self,
        fullname: str,
        path: object = None,
        target: ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        del path, target
        prefix = "vllm."
        if not fullname.startswith(prefix):
            return None

        relative_parts = fullname[len(prefix) :].split(".")
        source_dir = self.source_package.joinpath(*relative_parts)
        installed_dir = self.installed_package.joinpath(*relative_parts)
        source_init = source_dir / "__init__.py"
        if not source_init.is_file() or not installed_dir.is_dir():
            return None

        return importlib.util.spec_from_file_location(
            fullname,
            source_init,
            submodule_search_locations=[str(source_dir), str(installed_dir)],
        )


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


def _load_vllm(source_root: Path, installed_root: Path) -> ModuleType:
    source_package = source_root / "vllm"
    finder = _CombinedVllmPackageFinder(source_package, installed_root)
    sys.meta_path.insert(0, finder)
    spec = importlib.util.spec_from_file_location(
        "vllm",
        source_package / "__init__.py",
        submodule_search_locations=[str(source_package), str(installed_root)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot build vLLM module spec from {source_package}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(spec.name, None)
        sys.meta_path.remove(finder)
        raise
    return module


def configure_runtime() -> None:
    source_root = Path(os.environ["SOURCE_ROOT"])
    flashinfer_root = Path(os.environ["FLASHINFER_ROOT"])

    installed_vllm = _installed_package_path("vllm", (source_root,))
    vllm = _load_vllm(source_root, installed_vllm)
    _verify_import(vllm, source_root, "vLLM")

    compiled_extension = importlib.import_module("vllm._C_stable_libtorch")
    _verify_import(compiled_extension, installed_vllm, "vLLM compiled extension")

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
