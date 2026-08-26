# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
from pathlib import Path

from runtime_bootstrap import _load_vllm


def test_load_vllm_combines_source_and_installed_package_paths(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    installed_root = tmp_path / "installed" / "vllm"
    source_package = source_root / "vllm"
    source_package.mkdir(parents=True)
    installed_root.mkdir(parents=True)
    (source_package / "__init__.py").write_text("from .generated import VALUE\n")
    (installed_root / "generated.py").write_text("VALUE = 17\n")

    previous = sys.modules.pop("vllm", None)
    previous_meta_path = list(sys.meta_path)
    try:
        module = _load_vllm(source_root, installed_root)
        assert module.VALUE == 17
        assert list(module.__path__) == [str(source_package), str(installed_root)]
    finally:
        sys.meta_path[:] = previous_meta_path
        sys.modules.pop("vllm", None)
        sys.modules.pop("vllm.generated", None)
        if previous is not None:
            sys.modules["vllm"] = previous


def test_load_vllm_combines_nested_package_paths(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    installed_root = tmp_path / "installed" / "vllm"
    source_package = source_root / "vllm"
    source_nested = source_package / "nested"
    installed_nested = installed_root / "nested"
    source_nested.mkdir(parents=True)
    installed_nested.mkdir(parents=True)
    (source_package / "__init__.py").write_text("")
    (source_nested / "__init__.py").write_text("from .generated import VALUE\n")
    (installed_nested / "generated.py").write_text("VALUE = 23\n")

    previous = sys.modules.pop("vllm", None)
    previous_meta_path = list(sys.meta_path)
    try:
        _load_vllm(source_root, installed_root)
        nested = __import__("vllm.nested", fromlist=["nested"])
        assert nested.VALUE == 23
        assert list(nested.__path__) == [str(source_nested), str(installed_nested)]
    finally:
        sys.meta_path[:] = previous_meta_path
        sys.modules.pop("vllm", None)
        sys.modules.pop("vllm.nested", None)
        sys.modules.pop("vllm.nested.generated", None)
        if previous is not None:
            sys.modules["vllm"] = previous
