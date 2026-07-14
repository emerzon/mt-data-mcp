"""Lightweight command-line entry point."""

import sys
from importlib import metadata as importlib_metadata
from typing import Optional, Sequence


def _installed_version() -> str:
    try:
        return importlib_metadata.version("mtdata")
    except importlib_metadata.PackageNotFoundError:
        return "unknown"


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Handle cheap entry-point modes before importing the full tool graph."""
    effective_argv = list(sys.argv[1:] if argv is None else argv)
    if effective_argv in (["--version"], ["-V"]):
        print(f"mtdata-cli {_installed_version()}")
        return 0

    from . import api

    if effective_argv == ["shell"]:
        return api.run_shell()
    if argv is None:
        return api.main()

    original_argv = list(sys.argv)
    try:
        sys.argv = [original_argv[0], *effective_argv]
        return api.main()
    finally:
        sys.argv = original_argv

__all__ = ["main"]
