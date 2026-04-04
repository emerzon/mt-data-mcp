"""Date normalization utilities for Finviz service."""
import datetime
from typing import Any, Dict, List, Optional


def normalize_finviz_date_string(value: Any) -> Any:
    """Normalize Finviz short dates like `Nov 07 '25` to ISO 8601."""
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return value
    text = text.replace("'", "'")
    for fmt in ("%b %d '%y", "%b %d %Y", "%Y-%m-%d"):
        try:
            return datetime.datetime.strptime(text, fmt).date().isoformat()
        except Exception:
            continue
    return value


def normalize_finviz_dates_in_rows(
    rows: List[Dict[str, Any]], *keys: str
) -> List[Dict[str, Any]]:
    """Normalize date strings in specified columns of row dictionaries."""
    out: List[Dict[str, Any]] = []
    wanted = set(keys)
    for row in rows:
        if not isinstance(row, dict):
            continue
        row_out = dict(row)
        for key in wanted:
            if key in row_out:
                row_out[key] = normalize_finviz_date_string(row_out.get(key))
        out.append(row_out)
    return out


def strip_string_fields_in_rows(
    rows: List[Dict[str, Any]], *keys: str
) -> List[Dict[str, Any]]:
    """Strip whitespace from specified string columns in row dictionaries."""
    out: List[Dict[str, Any]] = []
    wanted = set(keys)
    for row in rows:
        if not isinstance(row, dict):
            continue
        row_out = dict(row)
        for key in wanted:
            value = row_out.get(key)
            if isinstance(value, str):
                row_out[key] = value.strip()
        out.append(row_out)
    return out


def resolve_date_range(
    *, date_from: Optional[str], date_to: Optional[str], default_days: int = 7
) -> tuple[str, str]:
    """Resolve date range with defaults."""
    today = datetime.date.today()
    if date_from:
        from_str = str(date_from)
    else:
        from_date = today - datetime.timedelta(days=default_days)
        from_str = from_date.isoformat()
    if date_to:
        to_str = str(date_to)
    else:
        to_str = today.isoformat()
    return from_str, to_str


def align_to_next_monday_if_weekend(date_from: str) -> str:
    """Align date to next Monday if it falls on weekend."""
    try:
        d = datetime.date.fromisoformat(date_from)
        # If Saturday (5) or Sunday (6), move to next Monday
        if d.weekday() == 5:  # Saturday
            d = d + datetime.timedelta(days=2)
        elif d.weekday() == 6:  # Sunday
            d = d + datetime.timedelta(days=1)
        return d.isoformat()
    except Exception:
        return date_from


__all__ = [
    "normalize_finviz_date_string",
    "normalize_finviz_dates_in_rows",
    "strip_string_fields_in_rows",
    "resolve_date_range",
    "align_to_next_monday_if_weekend",
]
