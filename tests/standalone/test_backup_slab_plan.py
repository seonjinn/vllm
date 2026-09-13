# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run on the cluster without importing the complete vLLM package."""

import importlib.util
import sys
import unittest
from collections.abc import Callable
from pathlib import Path
from typing import Any


def load_planner() -> Callable[..., Any]:
    path = Path(__file__).parents[2] / "vllm/device_allocator/backup_slabs.py"
    spec = importlib.util.spec_from_file_location("backup_slabs", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.plan_backup_slabs


class BackupSlabPlanTest(unittest.TestCase):
    def test_empty(self) -> None:
        self.assertEqual(load_planner()([], limit=1024), [])

    def test_coverage_determinism_and_isolation(self) -> None:
        plan = load_planner()
        handles = [
            (10, 600, 0, "weights"),
            (11, 300, 0, "weights"),
            (12, 100, 0, "weights"),
            (13, 300, 1, "weights"),
            (14, 300, 0, "draft"),
            (15, 2048, 0, "weights"),
        ]
        slabs = plan(handles, limit=1024)
        self.assertEqual(slabs, plan(list(reversed(handles)), limit=1024))
        observed: dict[int, tuple[int, int, str]] = {}
        for slab in slabs:
            end = 0
            self.assertEqual(slab.capacity & (slab.capacity - 1), 0)
            for ptr, offset, size in slab.views:
                self.assertGreaterEqual(offset, end)
                self.assertLessEqual(offset + size, slab.capacity)
                self.assertNotIn(ptr, observed)
                observed[ptr] = (size, slab.device, slab.tag)
                end = offset + size
            if slab.capacity > 1024:
                self.assertEqual(len(slab.views), 1)
                self.assertGreater(slab.views[0][2], 1024)
        self.assertEqual(observed, {p: (n, d, t) for p, n, d, t in handles})
        self.assertEqual(len(slabs), 4)

    def test_rounding_and_invalid_requests(self) -> None:
        plan = load_planner()
        self.assertEqual(plan([(1, 512, 0, "x")], limit=1024)[0].capacity, 512)
        self.assertEqual(plan([(1, 513, 0, "x")], limit=1024)[0].capacity, 1024)
        for handles, limit in (
            ([(1, 0, 0, "x")], 1024),
            ([(1, -1, 0, "x")], 1024),
            ([(1, 1, 0, "x"), (1, 2, 0, "x")], 1024),
            ([], 0),
            ([], 1000),
        ):
            with (
                self.subTest(handles=handles, limit=limit),
                self.assertRaises(ValueError),
            ):
                plan(handles, limit=limit)


if __name__ == "__main__":
    unittest.main()
