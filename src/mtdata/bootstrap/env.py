"""Shared environment-variable parsing helpers for bootstrap modules."""

from __future__ import annotations

import logging
import os

from ..utils.coercion import UNPARSED_BOOL, parse_bool_like

_LOGGER = logging.getLogger(__name__)
_BOOL_VALUES = "0, 1, false, n, no, off, on, true, y, yes"


def get_bool_env(name: str, default: bool = False) -> bool:
    """Read a boolean environment variable, warning before using a default."""
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    parsed = parse_bool_like(raw)
    if parsed is not UNPARSED_BOOL:
        return bool(parsed)
    _LOGGER.warning(
        "Invalid boolean %s=%r; using default %s. Accepted values are: %s.",
        name,
        raw,
        bool(default),
        _BOOL_VALUES,
    )
    return bool(default)
