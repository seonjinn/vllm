#!/usr/bin/env python3

import os
import sys
from pathlib import Path


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
    os.execv(sys.executable, args)


if __name__ == "__main__":
    main()

