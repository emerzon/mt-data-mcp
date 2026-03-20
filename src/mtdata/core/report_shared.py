"""Shared report formatting and preview helpers."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from ..utils.formatting import format_float, format_number as _format_number
from ..utils.minimal_output import format_table_toon as _format_table_toon


def format_number(value: Any) -> str:
    return _format_number(value)


def _indicator_key_variants(key: str) -> List[str]:
    if not key:
        return []
    base = str(key)
    keys = [base, base.lower()]
    parts = base.split("_")
    if parts:
        changed = False
        alt_parts: List[str] = []
        for part in parts:
            if part.isdigit():
                alt_parts.append(f"{part}.0")
                changed = True
            else:
                alt_parts.append(part)
        if changed:
            alt = "_".join(alt_parts)
            keys.extend([alt, alt.lower()])
    return keys


def _get_indicator_value(row: Optional[Dict[str, Any]], base_key: str) -> Any:
    if not isinstance(row, dict):
        return None
    for key in _indicator_key_variants(base_key):
        if key in row:
            val = row.get(key)
            if val not in (None, ""):
                return val
    return None


def _format_series_preview(
    values: Any, decimals: int = 6, head: int = 3, tail: int = 3
) -> Optional[str]:
    if not isinstance(values, list):
        return None
    n = len(values)
    if n == 0:
        return "n=0 []"

    numeric: List[float] = []
    for val in values:
        try:
            num = float(val)
        except (TypeError, ValueError):
            numeric = []
            break
        if not math.isfinite(num):
            numeric = []
            break
        numeric.append(num)

    if numeric:
        start = _format_decimal(numeric[0], decimals) or format_number(numeric[0])
        end = _format_decimal(numeric[-1], decimals) or format_number(numeric[-1])
        min_val = min(numeric)
        max_val = max(numeric)
        min_txt = _format_decimal(min_val, decimals) or format_number(min_val)
        max_txt = _format_decimal(max_val, decimals) or format_number(max_val)
        return f"n={n} start={start}, end={end}, min={min_txt}, max={max_txt}"

    def _fmt(val: Any) -> str:
        if isinstance(val, (int, float)):
            fmt = _format_decimal(val, decimals)
            return fmt if fmt is not None else format_number(val)
        return str(val)

    if n <= head + tail:
        items = [_fmt(v) for v in values]
    else:
        items = (
            [_fmt(v) for v in values[:head]]
            + ["..."]
            + [_fmt(v) for v in values[-tail:]]
        )
    return f"n={n} [" + ", ".join(items) + "]"


def _format_state_shares(shares: Any) -> Optional[str]:
    if not isinstance(shares, dict) or not shares:
        return None
    parts: List[str] = []
    for key in sorted(
        shares.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x)
    ):
        try:
            pct = float(shares[key]) * 100.0
            parts.append(f"{key}:{pct:.1f}%")
        except Exception:
            parts.append(f"{key}:{shares[key]}")
    return ", ".join(parts) if parts else None


def _format_table(
    headers: List[str], rows: List[List[Optional[Any]]], name: str = "data"
) -> List[str]:
    return _format_table_toon(headers, rows, name=name)


def _format_signed(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{value:+.1f}"
    except Exception:
        return str(value)


def _as_float(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _format_decimal(value: Any, decimals: int = 4) -> Optional[str]:
    val = _as_float(value)
    if val is None:
        return None
    return format_float(val, decimals)


def _format_probability(value: Optional[Any]) -> str:
    prob = _as_float(value)
    if prob is None:
        return "n/a"
    return f"{prob * 100:.1f}%"
