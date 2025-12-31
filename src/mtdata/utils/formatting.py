"""Shared formatting helpers for consistent text output."""
from typing import Any
import math

from .constants import DISPLAY_MAX_DECIMALS


def format_number(value: Any, decimals: int = DISPLAY_MAX_DECIMALS) -> str:
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
    text = f"{num:.{int(decimals)}f}".rstrip('0').rstrip('.')
    if text in ("", "-0"):
        text = "0"
    return text
