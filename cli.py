#!/usr/bin/env python3
"""
Wrapper script for backwards compatibility.
Import the actual CLI from the new package structure.
"""

import sys


def _runtime_guard() -> int:
    min_version = (3, 14)
    if sys.version_info < min_version:
        current = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        required = f"{min_version[0]}.{min_version[1]}"
        print(
            f"Unsupported Python runtime: {current}. mtdata CLI requires Python {required}+.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    code = _runtime_guard()
    if code != 0:
        raise SystemExit(code)
    from src.mtdata.core.cli import main
    main()
