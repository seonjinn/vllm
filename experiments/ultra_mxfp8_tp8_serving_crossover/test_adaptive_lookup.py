# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from experiments.ultra_mxfp8_tp8_serving_crossover.combine_adaptive_lookup import (
    combine_adaptive_lookups,
)
from experiments.ultra_mxfp8_tp8_serving_crossover.split_observed_by_layout import (
    split_observed_by_layout,
)


def _lookup(layout: str, entries: list[dict]) -> dict:
    return {
        "format_version": 1,
        "backend": "trtllm",
        "scale_layout": layout,
        "flashinfer_commit": "flashinfer-sha",
        "flashinfer_version": "0.6.18",
        "flashinfer_file": "/runtime/flashinfer/__init__.py",
        "container_sha256": "container-sha",
        "gpu": "NVIDIA GB200",
        "entry_count": len(entries),
        "entries": entries,
    }


def test_split_observed_uses_adaptive_m_boundary(tmp_path: Path) -> None:
    observed = tmp_path / "observed.csv"
    observed.write_text(
        "m,n,k,runner,selected_tactic\n"
        "1,1280,8192,Runner,7\n"
        "256,1280,8192,Runner,8\n"
        "257,1280,8192,Runner,9\n"
        "1000,1280,8192,Runner,10\n"
    )

    outputs = split_observed_by_layout(observed, tmp_path / "split", switch_m=256)

    with outputs["8x4"].open(newline="") as handle:
        rows_8x4 = list(csv.DictReader(handle))
    with outputs["128x4"].open(newline="") as handle:
        rows_128x4 = list(csv.DictReader(handle))
    assert [int(row["m"]) for row in rows_8x4] == [1, 256]
    assert [int(row["m"]) for row in rows_128x4] == [257, 1000]


def test_combine_adaptive_lookup_preserves_provenance_and_partition(
    tmp_path: Path,
) -> None:
    lookup_8x4 = tmp_path / "8x4.json"
    lookup_128x4 = tmp_path / "128x4.json"
    output = tmp_path / "adaptive.json"
    lookup_8x4.write_text(
        json.dumps(
            _lookup(
                "8x4",
                [{"m": 32, "n": 1280, "k": 8192, "runner": "Runner", "tactic": 7}],
            )
        )
    )
    lookup_128x4.write_text(
        json.dumps(
            _lookup(
                "128x4",
                [
                    {
                        "m": 1000,
                        "n": 1280,
                        "k": 8192,
                        "runner": "Runner",
                        "tactic": 9,
                    }
                ],
            )
        )
    )

    combined = combine_adaptive_lookups(lookup_8x4, lookup_128x4, output, switch_m=256)

    assert combined["scale_layout"] == "adaptive"
    assert combined["scale_layout_components"] == ["8x4", "128x4"]
    assert combined["switch_m"] == 256
    assert combined["entry_count"] == 2
    assert [entry["m"] for entry in combined["entries"]] == [32, 1000]
    assert json.loads(output.read_text()) == combined


def test_combine_adaptive_lookup_rejects_entry_on_wrong_side(tmp_path: Path) -> None:
    lookup_8x4 = tmp_path / "8x4.json"
    lookup_128x4 = tmp_path / "128x4.json"
    lookup_8x4.write_text(
        json.dumps(
            _lookup(
                "8x4",
                [
                    {
                        "m": 512,
                        "n": 1280,
                        "k": 8192,
                        "runner": "Runner",
                        "tactic": 7,
                    }
                ],
            )
        )
    )
    lookup_128x4.write_text(
        json.dumps(
            _lookup(
                "128x4",
                [
                    {
                        "m": 1000,
                        "n": 1280,
                        "k": 8192,
                        "runner": "Runner",
                        "tactic": 9,
                    }
                ],
            )
        )
    )

    with pytest.raises(ValueError, match="8x4 entry exceeds switch_m"):
        combine_adaptive_lookups(
            lookup_8x4, lookup_128x4, tmp_path / "adaptive.json", switch_m=256
        )
