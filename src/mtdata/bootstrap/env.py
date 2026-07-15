"""Shared environment-variable parsing helpers for bootstrap modules."""

from __future__ import annotations

import os


def get_bool_env(name: str, default: bool = False) -> bool:
    """Read a boolean environment variable using the project's truthy values."""
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}
