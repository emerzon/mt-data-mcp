"""Lightweight command-line entry point."""

import sys
from difflib import get_close_matches
from importlib import metadata as importlib_metadata
from typing import Optional, Sequence

from .catalog import available_command_names, format_root_help


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

    program = str(sys.argv[0] or "mtdata-cli").rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    if effective_argv in (["--help"], ["-h"]):
        print(format_root_help(program))
        return 0
    if not effective_argv:
        print(format_root_help(program))
        return 1

    raw_command = effective_argv[0]
    normalized_command = raw_command.replace("-", "_")
    known_commands = {*available_command_names(), "shell"}
    if not raw_command.startswith("-") and normalized_command not in known_commands:
        message = f"Unknown command: {raw_command}"
        suggestions = get_close_matches(normalized_command, sorted(known_commands), n=3)
        if suggestions:
            message += f". Did you mean: {', '.join(suggestions)}?"
        if "--json" in effective_argv:
            import json

            print(json.dumps({"success": False, "error": message, "error_code": "cli_unknown_command"}))
        else:
            print(message, file=sys.stderr)
            print(f"Run '{program} --help' to list commands.", file=sys.stderr)
        return 2

    from . import api

    if effective_argv == ["shell"]:
        return api.run_shell(interactive=sys.stdin.isatty())
    if argv is None:
        return api.main()

    original_argv = list(sys.argv)
    try:
        sys.argv = [original_argv[0], *effective_argv]
        return api.main()
    finally:
        sys.argv = original_argv

__all__ = ["main"]
