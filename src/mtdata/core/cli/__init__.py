"""Lightweight command-line entry point."""

import json
import os
import sys
from difflib import get_close_matches
from typing import Optional, Sequence

from ..error_envelope import build_error_payload
from .catalog import format_root_help, known_command_names
from .version import cli_version

_GLOBAL_OPTIONS_WITH_VALUES = frozenset(
    {"--output-fields", "--precision", "--timeframe"}
)
_GLOBAL_FLAG_OPTIONS = frozenset({"--json"})


def _json_output_requested(argv: Sequence[str]) -> bool:
    """Resolve the lightweight entry point's output mode without loading tools."""
    if "--json" in argv:
        return True
    return str(os.getenv("MTDATA_OUTPUT_FORMAT") or "").strip().lower() == "json"


def _leading_command_token(argv: Sequence[str]) -> Optional[str]:
    """Return the command token after any supported leading global options."""
    index = 0
    while index < len(argv):
        token = str(argv[index])
        option = token.split("=", 1)[0]
        if option in _GLOBAL_FLAG_OPTIONS:
            index += 1
            continue
        if option in _GLOBAL_OPTIONS_WITH_VALUES:
            index += 1 if "=" in token else 2
            continue
        return token
    return None


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Handle cheap entry-point modes before importing the full tool graph."""
    effective_argv = list(sys.argv[1:] if argv is None else argv)
    if effective_argv in (["--version"], ["-V"]):
        print(f"mtdata-cli {cli_version()}")
        return 0

    program = str(sys.argv[0] or "mtdata-cli").rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    if effective_argv in (["--help"], ["-h"]):
        print(format_root_help(program))
        return 0
    if not effective_argv:
        print(format_root_help(program))
        return 1

    raw_command = _leading_command_token(effective_argv)
    if raw_command is None:
        from . import api

        if argv is None:
            return api.main()
        original_argv = list(sys.argv)
        try:
            sys.argv = [original_argv[0], *effective_argv]
            return api.main()
        finally:
            sys.argv = original_argv
    normalized_command = raw_command.replace("-", "_")
    known_commands = {*known_command_names(), "shell"}
    if not raw_command.startswith("-") and normalized_command not in known_commands:
        message = f"Unknown command: {raw_command}"
        suggestions = get_close_matches(normalized_command, sorted(known_commands), n=3)
        if suggestions:
            message += f". Did you mean: {', '.join(suggestions)}?"
        if _json_output_requested(effective_argv):
            print(
                json.dumps(
                    build_error_payload(
                        message,
                        code="cli_unknown_command",
                        operation="cli",
                        remediation=f"Run '{program} --help' to list commands.",
                        documentation="docs/CLI.md",
                    )
                )
            )
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
