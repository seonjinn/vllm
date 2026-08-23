#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import sys
from pathlib import Path


def _arg_value(args: list[str], name: str) -> str | None:
    try:
        return args[args.index(name) + 1]
    except (ValueError, IndexError):
        return None


def _validate_shape_contract(args: list[str]) -> None:
    for flag, env_name in (
        ("--isl", "NTRACE_EXPECTED_ISL"),
        ("--osl", "NTRACE_EXPECTED_OSL"),
    ):
        expected = os.environ.get(env_name)
        if expected is None:
            continue
        actual = _arg_value(args, flag)
        if actual != expected:
            raise ValueError(
                f"ntrace workload mismatch: expected {flag}={expected}, got {actual}"
            )


def main() -> None:
    target = Path(os.environ["NTRACE_BENCH_TARGET"])
    if not target.is_file():
        raise FileNotFoundError(f"benchmark harness not found: {target}")

    args = [sys.executable, str(target), *sys.argv[1:]]
    if "--server-profiler" not in args:
        args.extend(["--server-profiler", "ntrace"])
    if "--client-profile" not in args:
        args.append("--client-profile")
    if "--profile-first-attempt" not in args:
        args.append("--profile-first-attempt")
    _validate_shape_contract(args)
    os.execv(sys.executable, args)


if __name__ == "__main__":
    main()
