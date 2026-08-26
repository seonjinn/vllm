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
    try:
        module = _load_vllm(source_root, installed_root)
        assert module.VALUE == 17
        assert list(module.__path__) == [str(source_package), str(installed_root)]
    finally:
        sys.modules.pop("vllm", None)
        sys.modules.pop("vllm.generated", None)
        if previous is not None:
            sys.modules["vllm"] = previous
