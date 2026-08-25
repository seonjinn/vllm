# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from pathlib import Path

import pytest

from experiments.ultra_mxfp8_all_observed_tactics.shape_tactic_runtime import (
    TacticLookup,
    extract_mnk,
    make_dispatcher,
    restore_tactic,
)


class FakeTensor:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape


class CuteRunner:
    pass


class TrtRunner:
    pass


def test_extract_mnk_flattens_all_activation_batch_dimensions() -> None:
    inputs = [FakeTensor((7, 11, 8192)), FakeTensor((8192, 2304))]

    assert extract_mnk(inputs) == (77, 2304, 8192)


def test_extract_mnk_rejects_weight_with_incompatible_k() -> None:
    inputs = [FakeTensor((1001, 8192)), FakeTensor((4096, 2304))]

    with pytest.raises(ValueError, match="incompatible K"):
        extract_mnk(inputs)


def test_restore_tactic_recovers_nested_tuple_structure() -> None:
    serialized = [[128, 32], [1, 1], True, False, 1]

    assert restore_tactic(serialized) == (
        (128, 32),
        (1, 1),
        True,
        False,
        1,
    )


def test_lookup_returns_only_exact_shape_and_runner_match(tmp_path: Path) -> None:
    lookup_path = tmp_path / "lookup.json"
    lookup_path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "entries": [
                    {
                        "m": 1001,
                        "n": 2304,
                        "k": 8192,
                        "runner": "CuteRunner",
                        "tactic": [[128, 32], [1, 1], True, False, 1],
                    }
                ],
            }
        )
    )
    lookup = TacticLookup.load(lookup_path)

    hit = lookup.choose((1001, 2304, 8192), [TrtRunner(), CuteRunner()])
    assert hit is not None
    assert isinstance(hit[0], CuteRunner)
    assert hit[1] == ((128, 32), (1, 1), True, False, 1)
    assert lookup.choose((1002, 2304, 8192), [CuteRunner()]) is None
    assert lookup.choose((1001, 2304, 8192), [TrtRunner()]) is None


def test_lookup_rejects_duplicate_shape_entries(tmp_path: Path) -> None:
    lookup_path = tmp_path / "lookup.json"
    entry = {
        "m": 1001,
        "n": 2304,
        "k": 8192,
        "runner": "CuteRunner",
        "tactic": 7,
    }
    lookup_path.write_text(json.dumps({"format_version": 1, "entries": [entry, entry]}))

    with pytest.raises(ValueError, match="duplicate lookup entry"):
        TacticLookup.load(lookup_path)


def test_dispatcher_uses_lookup_hit_without_calling_default(tmp_path: Path) -> None:
    lookup_path = tmp_path / "lookup.json"
    lookup_path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "entries": [
                    {
                        "m": 1001,
                        "n": 2304,
                        "k": 8192,
                        "runner": "CuteRunner",
                        "tactic": 17,
                    }
                ],
            }
        )
    )
    default_calls = 0

    def default(*args, **kwargs):
        nonlocal default_calls
        default_calls += 1
        return TrtRunner(), -1

    dispatch = make_dispatcher(default, TacticLookup.load(lookup_path), None)
    runner, tactic = dispatch(
        object(),
        "mxfp8_gemm",
        [CuteRunner(), TrtRunner()],
        object(),
        [FakeTensor((1001, 8192)), FakeTensor((8192, 2304))],
    )

    assert isinstance(runner, CuteRunner)
    assert tactic == 17
    assert default_calls == 0


def test_dispatcher_delegates_lookup_miss_to_default(tmp_path: Path) -> None:
    lookup_path = tmp_path / "lookup.json"
    lookup_path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "entries": [
                    {
                        "m": 1001,
                        "n": 2304,
                        "k": 8192,
                        "runner": "CuteRunner",
                        "tactic": 17,
                    }
                ],
            }
        )
    )
    default_runner = TrtRunner()

    def default(*args, **kwargs):
        return default_runner, -1

    dispatch = make_dispatcher(default, TacticLookup.load(lookup_path), None)

    assert dispatch(
        object(),
        "mxfp8_gemm",
        [CuteRunner(), TrtRunner()],
        object(),
        [FakeTensor((1002, 8192)), FakeTensor((8192, 2304))],
    ) == (default_runner, -1)
