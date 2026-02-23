"""Shared formatting helpers for consistent text output."""
from typing import Any, List, Optional
import math

from .constants import (
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


def format_float(value: float, decimals: int) -> str:
    """Format a float with fixed decimals and trimmed trailing zeros."""
    text = f"{float(value):.{int(decimals)}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    if text in ("", "-0"):
        return "0"
    return text


def optimal_decimals(
    values: List[float],
    rel_tol: float = PRECISION_REL_TOL,
    abs_tol: float = PRECISION_ABS_TOL,
    max_decimals: int = PRECISION_MAX_DECIMALS,
    max_loss_pct: float = PRECISION_MAX_LOSS_PCT,
) -> int:
    """Infer minimal decimals that preserve numeric variation under tolerance."""
    if not values:
        return 0
    nums: List[float] = []
    for v in values:
        try:
            fv = float(v)
        except Exception:
            continue
        if math.isfinite(fv):
            nums.append(fv)
    if not nums:
        return 0

    vmin = min(nums)
    vmax = max(nums)
    value_range = vmax - vmin
    if value_range <= 0.0:
        scale = max(1.0, max(abs(v) for v in nums))
        tol = max(abs_tol, rel_tol * scale)
    else:
        tol = max(abs_tol, value_range * max_loss_pct)

    for d in range(0, int(max_decimals) + 1):
        factor = 10.0 ** d
        max_diff = 0.0
        for v in nums:
            rv = round(v * factor) / factor
            diff = abs(rv - v)
            if diff > max_diff:
                max_diff = diff
            if max_diff > tol:
                break
        if max_diff <= tol:
            return d
    return int(max_decimals)


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
    return format_float(num, int(decimals))
