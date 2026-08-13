"""Lightweight CLI output-format configuration validation."""

import os
from typing import Any

CLI_FORMAT_TOON = "toon"
CLI_FORMAT_JSON = "json"
CLI_OUTPUT_FORMAT_ENV = "MTDATA_OUTPUT_FORMAT"
CLI_OUTPUT_FORMAT_CHOICES = (CLI_FORMAT_JSON, CLI_FORMAT_TOON)


def normalize_cli_output_format(value: Any) -> str:
    """Normalize a configured format and reject unsupported nonblank values."""
    raw = str(value or "").strip().lower()
    if not raw:
        return CLI_FORMAT_TOON
    if raw in CLI_OUTPUT_FORMAT_CHOICES:
        return raw
    raise ValueError(
        f"Invalid {CLI_OUTPUT_FORMAT_ENV} value {value!r}. "
        f"Expected one of: {', '.join(CLI_OUTPUT_FORMAT_CHOICES)}."
    )


def resolve_cli_output_format_env() -> str:
    """Resolve and validate the environment-driven CLI format."""
    return normalize_cli_output_format(os.getenv(CLI_OUTPUT_FORMAT_ENV))


__all__ = [
    "CLI_FORMAT_JSON",
    "CLI_FORMAT_TOON",
    "CLI_OUTPUT_FORMAT_CHOICES",
    "CLI_OUTPUT_FORMAT_ENV",
    "normalize_cli_output_format",
    "resolve_cli_output_format_env",
]
