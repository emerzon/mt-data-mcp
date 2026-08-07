"""Shared environment-variable parsing helpers for bootstrap modules."""

from __future__ import annotations

import logging
import os


_LOGGER = logging.getLogger(__name__)
_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}


def get_bool_env(name: str, default: bool = False) -> bool:
    """Read a boolean environment variable, warning before using a default."""
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    normalized = str(raw).strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    _LOGGER.warning(
        "Invalid boolean %s=%r; using default %s. Accepted values are: %s.",
        name,
        raw,
        bool(default),
        ", ".join(sorted(_TRUE_VALUES | _FALSE_VALUES)),
    )
    return bool(default)
