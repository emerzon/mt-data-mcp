"""Shared scalar coercion helpers."""

from __future__ import annotations

import math
from typing import Any, Optional


def coerce_finite_float(value: Any) -> Optional[float]:
    """Best-effort float coercion that rejects missing and non-finite values."""
    if value is None:
        return None
    try:
        out = float(value)
    except Exception:
        try:
            out = float(str(value))
        except Exception:
            return None
    if not math.isfinite(out):
        return None
    return float(out)


def safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    """Best-effort finite float coercion with an optional fallback."""
    out = coerce_finite_float(value)
    return default if out is None else out


def is_explicit_false(value: Any) -> bool:
    """Return True only when a supplied value is explicitly falsy."""
    if value is None:
        return False
    try:
        return not bool(value)
    except Exception:
        return False
