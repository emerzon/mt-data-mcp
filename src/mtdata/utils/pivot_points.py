"""Pure pivot point formulas shared by pivot-related tools."""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Optional

from ..shared.schema import _PIVOT_METHODS
from .coercion import round_finite


def _round_price(value: float, digits: int) -> float:
    if digits < 0:
        return float(value)
    rounded = round_finite(value, digits, on_invalid="passthrough")
    return float(rounded) if isinstance(rounded, (int, float)) and not isinstance(rounded, bool) else float(value)


def compute_pivot_method_levels(
    method_name: str,
    *,
    open_price: float,
    high_price: float,
    low_price: float,
    close_price: float,
    digits: int = 6,
) -> Optional[Dict[str, Any]]:
    """Compute one supported pivot method from a completed OHLC bar."""
    name = str(method_name).lower().strip()
    high = float(high_price)
    low = float(low_price)
    close = float(close_price)
    open_ = float(open_price)
    if any(not math.isfinite(v) for v in (high, low, close)):
        raise ValueError("Pivot calculation requires high, low, and close prices")
    if name == "demark" and not math.isfinite(open_):
        raise ValueError("DeMark pivot calculation requires a finite open price")

    range_value = high - low
    if name == "classic":
        pivot = (high + low + close) / 3.0
        levels_raw = {
            "PP": pivot,
            "R1": 2 * pivot - low,
            "S1": 2 * pivot - high,
            "R2": pivot + range_value,
            "S2": pivot - range_value,
            "R3": high + 2 * (pivot - low),
            "S3": low - 2 * (high - pivot),
        }
    elif name == "fibonacci":
        pivot = (high + low + close) / 3.0
        levels_raw = {
            "PP": pivot,
            "R1": pivot + 0.382 * range_value,
            "S1": pivot - 0.382 * range_value,
            "R2": pivot + 0.618 * range_value,
            "S2": pivot - 0.618 * range_value,
            "R3": pivot + range_value,
            "S3": pivot - range_value,
        }
    elif name == "camarilla":
        k = 1.1
        levels_raw = {
            "PP": (high + low + close) / 3.0,
            "R1": close + (k * range_value) / 12.0,
            "S1": close - (k * range_value) / 12.0,
            "R2": close + (k * range_value) / 6.0,
            "S2": close - (k * range_value) / 6.0,
            "R3": close + (k * range_value) / 4.0,
            "S3": close - (k * range_value) / 4.0,
            "R4": close + (k * range_value) / 2.0,
            "S4": close - (k * range_value) / 2.0,
        }
        pivot = levels_raw["PP"]
    elif name == "woodie":
        pivot = (high + low + 2 * close) / 4.0
        levels_raw = {
            "PP": pivot,
            "R1": 2 * pivot - low,
            "S1": 2 * pivot - high,
            "R2": pivot + range_value,
            "S2": pivot - range_value,
        }
    elif name == "demark":
        if close < open_:
            x_value = high + 2 * low + close
        elif close > open_:
            x_value = 2 * high + low + close
        else:
            x_value = high + low + 2 * close
        pivot = x_value / 4.0
        levels_raw = {
            "PP": pivot,
            "R1": x_value / 2.0 - low,
            "S1": x_value / 2.0 - high,
        }
    else:
        return None

    levels = {key: _round_price(value, int(digits)) for key, value in levels_raw.items()}
    return {
        "method": name,
        "pivot": _round_price(pivot, int(digits)),
        "levels": levels,
        "level_set": list(levels.keys()),
        **(
            {"pivot_convention": "retail_x_over_4_extension"}
            if name == "demark"
            else {}
        ),
    }


def compute_pivot_methods(
    *,
    open_price: float,
    high_price: float,
    low_price: float,
    close_price: float,
    digits: int = 6,
    methods: Optional[Iterable[str]] = None,
) -> list[Dict[str, Any]]:
    """Compute all requested supported pivot methods."""
    out: list[Dict[str, Any]] = []
    for method_name in methods or _PIVOT_METHODS:
        method_info = compute_pivot_method_levels(
            str(method_name),
            open_price=open_price,
            high_price=high_price,
            low_price=low_price,
            close_price=close_price,
            digits=int(digits),
        )
        if method_info:
            out.append(method_info)
    return out
