#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

ARMS = ("cutedsl", "adaptive-lookup")
FORWARD = ARMS
REVERSE = tuple(reversed(ARMS))
RAW_RESULT_PREFIX = "raw_bench_serve_bs"
RAW_RESULT_SUFFIX = ".json"


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read JSON {path}: {error}") from error


def _find_outer_results(root: Path) -> list[tuple[Path, dict[str, Any]]]:
    results = []
    for path in sorted(root.rglob("*.json")):
        payload = _load_json(path)
        if isinstance(payload, dict) and "order" in payload and "runs" in payload:
            results.append((path, payload))
    return results


def _validate_orders(
    candidates: list[tuple[Path, dict[str, Any]]],
) -> dict[tuple[str, str], tuple[Path, dict[str, Any]]]:
    expected = {FORWARD, REVERSE}
    actual: dict[tuple[str, str], tuple[Path, dict[str, Any]]] = {}
    for path, payload in candidates:
        raw_order = payload.get("order")
        if not isinstance(raw_order, list) or not all(
            isinstance(arm, str) for arm in raw_order
        ):
            raise ValueError(f"invalid paired order in {path}: {raw_order!r}")
        order = tuple(raw_order)
        if order in actual:
            raise ValueError(
                "expected exactly two reverse-order allocations; "
                f"duplicate order {' -> '.join(order)}"
            )
        actual[order] = (path, payload)
    if len(candidates) != 2 or set(actual) != expected:
        rendered = sorted(" -> ".join(order) for order in actual)
        raise ValueError(
            "expected exactly two reverse-order allocations "
            f"({ARMS[0]} -> {ARMS[1]} and {ARMS[1]} -> {ARMS[0]}); found {rendered}"
        )
    return actual


def _index_embedded_rows(
    result: dict[str, Any],
    *,
    arm: str,
    requested: tuple[int, ...],
) -> dict[int, dict[str, Any]]:
    config = result.get("config")
    if not isinstance(config, dict):
        raise ValueError(f"{arm}: embedded result is missing config")
    batch_sizes = config.get("batch_sizes")
    if batch_sizes is not None and not set(requested).issubset(batch_sizes):
        raise ValueError(
            f"{arm}: embedded batch_sizes={batch_sizes} does not contain "
            f"requested {list(requested)}"
        )
    raw_rows = result.get("results")
    if not isinstance(raw_rows, list):
        raise ValueError(f"{arm}: embedded result is missing results rows")
    rows: dict[int, dict[str, Any]] = {}
    for row in raw_rows:
        if not isinstance(row, dict) or type(row.get("bs")) is not int:
            raise ValueError(f"{arm}: invalid embedded result row: {row!r}")
        concurrency = row["bs"]
        if concurrency in rows:
            raise ValueError(
                f"{arm}: duplicate concurrency {concurrency} in embedded rows"
            )
        rows[concurrency] = row
    _validate_concurrency_set(
        rows,
        arm=arm,
        requested=requested,
        source="embedded rows",
        allow_unexpected=True,
    )
    return rows


def _validate_concurrency_set(
    rows: dict[int, Any],
    *,
    arm: str,
    requested: tuple[int, ...],
    source: str,
    allow_unexpected: bool = False,
) -> None:
    expected = set(requested)
    actual = set(rows)
    missing = sorted(expected - actual)
    if missing:
        raise ValueError(f"{arm}: missing concurrency {missing[0]} in {source}")
    unexpected = sorted(actual - expected)
    if unexpected and not allow_unexpected:
        raise ValueError(f"{arm}: unexpected concurrency {unexpected[0]} in {source}")


def _resolve_arm_result(
    outer_path: Path,
    run: dict[str, Any],
    *,
    arm: str,
) -> tuple[Path, dict[str, Any]]:
    embedded = run.get("result")
    if not isinstance(embedded, dict):
        raise ValueError(f"{arm}: outer result is missing embedded result")
    candidates = []
    result_path = run.get("result_path")
    if isinstance(result_path, str):
        candidates.append(Path(result_path))
    candidates.append(outer_path.parent / "paired" / arm / "result.json")
    existing = next((path for path in candidates if path.is_file()), None)
    if existing is None:
        raise ValueError(f"{arm}: result.json does not exist in paired output tree")
    stored = _load_json(existing)
    if stored != embedded:
        raise ValueError(f"{arm}: embedded result does not match {existing}")
    return existing.parent, embedded


def _index_raw_results(arm_dir: Path, *, arm: str) -> dict[int, Path]:
    rows: dict[int, Path] = {}
    for path in sorted(arm_dir.glob("raw_bench_serve_bs*.json")):
        body = path.name.removeprefix(RAW_RESULT_PREFIX).removesuffix(RAW_RESULT_SUFFIX)
        concurrency_text, separator, retry_text = body.partition("_retry")
        if not concurrency_text.isdigit() or (separator and not retry_text.isdigit()):
            raise ValueError(f"{arm}: invalid raw result filename {path.name}")
        concurrency = int(concurrency_text)
        if concurrency in rows:
            raise ValueError(
                f"{arm}: duplicate concurrency {concurrency} in raw results"
            )
        rows[concurrency] = path
    return rows


def _require_int(raw: dict[str, Any], key: str, *, context: str) -> int:
    value = raw.get(key)
    if type(value) is not int:
        raise ValueError(f"{context}: {key} must be an integer, got {value!r}")
    return value


def _require_positive_float(raw: dict[str, Any], key: str, *, context: str) -> float:
    value = raw.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{context}: {key} must be numeric, got {value!r}")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0:
        raise ValueError(f"{context}: {key} must be finite and positive, got {value!r}")
    return numeric


def _validate_measurement(
    path: Path,
    embedded: dict[str, Any],
    *,
    arm: str,
    concurrency: int,
    waves: int,
    osl: int,
) -> dict[str, Any]:
    context = f"{arm} concurrency {concurrency} ({path})"
    raw = _load_json(path)
    if not isinstance(raw, dict):
        raise ValueError(f"{context}: raw result must be a JSON object")
    expected_requests = concurrency * waves
    completed = _require_int(raw, "completed", context=context)
    if completed != expected_requests:
        raise ValueError(
            f"{context}: completed={completed}, expected {expected_requests}"
        )
    failed = _require_int(raw, "failed", context=context)
    if failed != 0:
        raise ValueError(f"{context}: failed={failed}, expected 0")
    expected_output_tokens = expected_requests * osl
    output_tokens = _require_int(raw, "total_output_tokens", context=context)
    if output_tokens != expected_output_tokens:
        raise ValueError(
            f"{context}: output tokens={output_tokens}, "
            f"expected {expected_output_tokens}"
        )
    throughput = _require_positive_float(raw, "output_throughput", context=context)
    duration = _require_positive_float(raw, "duration", context=context)

    if embedded.get("tokens_ok") is not True:
        raise ValueError(f"{context}: embedded tokens_ok is not true")
    if embedded.get("actual_output_tokens") != expected_output_tokens:
        raise ValueError(f"{context}: embedded actual_output_tokens is invalid")
    if embedded.get("expected_output_tokens") != expected_output_tokens:
        raise ValueError(f"{context}: embedded expected_output_tokens is invalid")
    if embedded.get("osl") != osl:
        raise ValueError(
            f"{context}: embedded OSL={embedded.get('osl')}, expected {osl}"
        )
    embedded_throughput = embedded.get("output_tok_s")
    if not isinstance(embedded_throughput, (int, float)) or not math.isclose(
        float(embedded_throughput), throughput, rel_tol=1e-12, abs_tol=1e-12
    ):
        raise ValueError(f"{context}: embedded throughput does not match raw result")
    return {
        "completed": completed,
        "duration_s": duration,
        "failed": failed,
        "output_tokens": output_tokens,
        "output_tok_s": throughput,
    }


def _summarize_allocation(
    outer_path: Path,
    payload: dict[str, Any],
    *,
    order: tuple[str, str],
    requested: tuple[int, ...],
    waves: int,
    osl: int,
    root: Path,
) -> dict[str, Any]:
    raw_runs = payload.get("runs")
    if not isinstance(raw_runs, list) or len(raw_runs) != 2:
        raise ValueError(f"{' -> '.join(order)}: expected exactly two runs")
    runs: dict[str, dict[str, Any]] = {}
    for run in raw_runs:
        if not isinstance(run, dict) or run.get("arm") not in ARMS:
            raise ValueError(f"{' -> '.join(order)}: invalid run entry {run!r}")
        arm = run["arm"]
        if arm in runs:
            raise ValueError(f"{' -> '.join(order)}: duplicate arm {arm}")
        runs[arm] = run
    if tuple(run["arm"] for run in raw_runs) != order:
        raise ValueError(f"{' -> '.join(order)}: run order does not match outer order")

    by_arm: dict[str, dict[int, dict[str, Any]]] = {}
    for arm in ARMS:
        arm_dir, embedded_result = _resolve_arm_result(outer_path, runs[arm], arm=arm)
        embedded_rows = _index_embedded_rows(
            embedded_result, arm=arm, requested=requested
        )
        raw_paths = _index_raw_results(arm_dir, arm=arm)
        _validate_concurrency_set(
            raw_paths,
            arm=arm,
            requested=requested,
            source="raw results",
            allow_unexpected=True,
        )
        by_arm[arm] = {
            concurrency: _validate_measurement(
                raw_paths[concurrency],
                embedded_rows[concurrency],
                arm=arm,
                concurrency=concurrency,
                waves=waves,
                osl=osl,
            )
            for concurrency in requested
        }

    measurements = []
    for concurrency in requested:
        baseline = by_arm["cutedsl"][concurrency]
        candidate = by_arm["adaptive-lookup"][concurrency]
        measurements.append(
            {
                "adaptive_lookup_output_tok_s": candidate["output_tok_s"],
                "concurrency": concurrency,
                "cutedsl_output_tok_s": baseline["output_tok_s"],
                "throughput_ratio": candidate["output_tok_s"]
                / baseline["output_tok_s"],
            }
        )
    try:
        relative_path = str(outer_path.relative_to(root))
    except ValueError:
        relative_path = str(outer_path)
    return {
        "measurements": measurements,
        "order": list(order),
        "outer_result_path": relative_path,
    }


def summarize(
    root: Path,
    *,
    concurrencies: tuple[int, ...] = (8, 32),
    waves: int = 10,
    osl: int = 10_000,
) -> dict[str, Any]:
    if not root.is_dir():
        raise ValueError(f"result root does not exist: {root}")
    requested = tuple(sorted(concurrencies))
    if not requested or any(value <= 0 for value in requested):
        raise ValueError("concurrencies must contain positive integers")
    if len(set(requested)) != len(requested):
        raise ValueError("concurrencies must not contain duplicates")
    if waves <= 0 or osl <= 0:
        raise ValueError("waves and OSL must be positive")

    ordered = _validate_orders(_find_outer_results(root))
    allocations = [
        _summarize_allocation(
            *ordered[order],
            order=order,
            requested=requested,
            waves=waves,
            osl=osl,
            root=root,
        )
        for order in (FORWARD, REVERSE)
    ]
    by_order = {tuple(allocation["order"]): allocation for allocation in allocations}
    comparisons = []
    for index, concurrency in enumerate(requested):
        forward = by_order[FORWARD]["measurements"][index]
        reverse = by_order[REVERSE]["measurements"][index]
        comparisons.append(
            {
                "adaptive_lookup_then_cutedsl": {
                    "adaptive_lookup_output_tok_s": reverse[
                        "adaptive_lookup_output_tok_s"
                    ],
                    "cutedsl_output_tok_s": reverse["cutedsl_output_tok_s"],
                    "throughput_ratio": reverse["throughput_ratio"],
                },
                "concurrency": concurrency,
                "cutedsl_then_adaptive_lookup": {
                    "adaptive_lookup_output_tok_s": forward[
                        "adaptive_lookup_output_tok_s"
                    ],
                    "cutedsl_output_tok_s": forward["cutedsl_output_tok_s"],
                    "throughput_ratio": forward["throughput_ratio"],
                },
                "geometric_mean_throughput_ratio": math.sqrt(
                    forward["throughput_ratio"] * reverse["throughput_ratio"]
                ),
            }
        )
    return {
        "allocations": allocations,
        "comparisons": comparisons,
        "concurrencies": list(requested),
        "orders": [list(FORWARD), list(REVERSE)],
        "osl": osl,
        "schema_version": 1,
        "waves": waves,
    }


def _format_float(value: float) -> str:
    return format(value, ".12g")


def write_outputs(summary: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    (output_dir / "summary.json").write_text(json_text)
    forward_cutedsl = "cutedsl_then_adaptive_lookup_cutedsl_output_tok_s"
    forward_adaptive = "cutedsl_then_adaptive_lookup_adaptive_lookup_output_tok_s"
    forward_ratio = "cutedsl_then_adaptive_lookup_throughput_ratio"
    reverse_cutedsl = "adaptive_lookup_then_cutedsl_cutedsl_output_tok_s"
    reverse_adaptive = "adaptive_lookup_then_cutedsl_adaptive_lookup_output_tok_s"
    reverse_ratio = "adaptive_lookup_then_cutedsl_throughput_ratio"
    fieldnames = [
        "concurrency",
        forward_cutedsl,
        forward_adaptive,
        forward_ratio,
        reverse_cutedsl,
        reverse_adaptive,
        reverse_ratio,
        "geometric_mean_throughput_ratio",
    ]
    with (output_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for comparison in summary["comparisons"]:
            forward = comparison["cutedsl_then_adaptive_lookup"]
            reverse = comparison["adaptive_lookup_then_cutedsl"]
            writer.writerow(
                {
                    "concurrency": comparison["concurrency"],
                    forward_cutedsl: _format_float(forward["cutedsl_output_tok_s"]),
                    forward_adaptive: _format_float(
                        forward["adaptive_lookup_output_tok_s"]
                    ),
                    forward_ratio: _format_float(forward["throughput_ratio"]),
                    reverse_cutedsl: _format_float(reverse["cutedsl_output_tok_s"]),
                    reverse_adaptive: _format_float(
                        reverse["adaptive_lookup_output_tok_s"]
                    ),
                    reverse_ratio: _format_float(reverse["throughput_ratio"]),
                    "geometric_mean_throughput_ratio": _format_float(
                        comparison["geometric_mean_throughput_ratio"]
                    ),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--concurrencies", nargs="+", type=int, default=[8, 32])
    parser.add_argument("--waves", type=int, default=10)
    parser.add_argument("--osl", type=int, default=10_000)
    args = parser.parse_args()
    try:
        summary = summarize(
            args.root,
            concurrencies=tuple(args.concurrencies),
            waves=args.waves,
            osl=args.osl,
        )
    except ValueError as error:
        parser.error(str(error))
    write_outputs(summary, args.output_dir)
    print(
        json.dumps(
            {
                "allocations": len(summary["allocations"]),
                "concurrencies": summary["concurrencies"],
                "output_dir": str(args.output_dir),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
