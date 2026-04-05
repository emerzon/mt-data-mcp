"""Date normalization utilities for Finviz service."""
import datetime
from typing import Any, Dict, List, Optional


def parse_iso_date_input(value: Any, *, field_name: str) -> datetime.date:
    text = str(value).strip()
    if not text:
        raise ValueError(f"Invalid {field_name} '{value}'. Expected YYYY-MM-DD or ISO datetime")
    normalized = text[:-1] + "+00:00" if text.endswith(("Z", "z")) else text
    try:
        return datetime.date.fromisoformat(normalized)
    except ValueError:
        pass
    try:
        return datetime.datetime.fromisoformat(normalized).date()
    except ValueError as exc:
        raise ValueError(
            f"Invalid {field_name} '{value}'. Expected YYYY-MM-DD or ISO datetime"
        ) from exc


def normalize_finviz_date_string(value: Any) -> Any:
    """Normalize Finviz short dates like `Nov 07 '25` to ISO 8601."""
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return value
    text = text.replace("’", "'")
    for fmt in ("%b %d '%y", "%b %d %Y"):
        try:
            return datetime.datetime.strptime(text, fmt).date().isoformat()
        except Exception:
            continue
    try:
        return parse_iso_date_input(text, field_name="date").isoformat()
    except ValueError:
        pass
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
        from_str = parse_iso_date_input(date_from, field_name="date_from").isoformat()
    else:
        from_date = today - datetime.timedelta(days=default_days)
        from_str = from_date.isoformat()
    if date_to:
        to_str = parse_iso_date_input(date_to, field_name="date_to").isoformat()
    else:
        to_str = today.isoformat()
    return from_str, to_str


def align_to_next_monday_if_weekend(date_from: str) -> str:
    """Align date to next Monday if it falls on weekend."""
    d = parse_iso_date_input(date_from, field_name="date_from")
    # If Saturday (5) or Sunday (6), move to next Monday
    if d.weekday() == 5:  # Saturday
        d = d + datetime.timedelta(days=2)
    elif d.weekday() == 6:  # Sunday
        d = d + datetime.timedelta(days=1)
    return d.isoformat()


__all__ = [
    "parse_iso_date_input",
    "normalize_finviz_date_string",
    "normalize_finviz_dates_in_rows",
    "strip_string_fields_in_rows",
    "resolve_date_range",
    "align_to_next_monday_if_weekend",
]
