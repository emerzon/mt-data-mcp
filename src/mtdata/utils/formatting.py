"""Shared formatting helpers for consistent text output."""
from typing import Any, Optional
import math

from .constants import (
    DISPLAY_MAX_DECIMALS,
    PRECISION_ABS_TOL,
    PRECISION_MAX_DECIMALS,
    PRECISION_MAX_LOSS_PCT,
    PRECISION_REL_TOL,
)


def _adaptive_decimals(num: float, max_decimals: int = PRECISION_MAX_DECIMALS) -> int:
    scale = max(1.0, abs(num))
    tol = max(PRECISION_ABS_TOL, PRECISION_REL_TOL * scale, abs(num) * PRECISION_MAX_LOSS_PCT)
    for d in range(0, max_decimals + 1):
        factor = 10.0 ** d
        rv = round(num * factor) / factor
        if abs(rv - num) <= tol:
            return d
    return max_decimals


def format_number(value: Any, decimals: Optional[int] = None) -> str:
    """Render scalars with consistent numeric/boolean/null representation."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    try:
        num = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(num):
        return str(value)
    if decimals is None:
        decimals = _adaptive_decimals(num)
    text = f"{num:.{int(decimals)}f}".rstrip('0').rstrip('.')
    if text in ("", "-0"):
        text = "0"
    return text
