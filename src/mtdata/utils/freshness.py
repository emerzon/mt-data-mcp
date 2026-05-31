from __future__ import annotations

import math
from typing import Any, Optional


def format_age_seconds(seconds: Any) -> Optional[str]:
    try:
        total = max(0, int(round(float(seconds))))
    except Exception:
        return None
    days, remainder = divmod(total, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, secs = divmod(remainder, 60)
    if days:
        return f"{days}d {hours}h"
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _clean_status(value: Any) -> str:
    return str(value or "").strip().lower().replace("_", " ")


def _coerce_bool_flag(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (str, bytes, bytearray, list, tuple, dict, set)):
        return None
    if not hasattr(value, "__bool__"):
        return None
    try:
        return bool(value)
    except Exception:
        return None


def format_freshness_label(
    *,
    data_stale: Any = None,
    market_status: Any = None,
    market_status_reason: Any = None,
    age_seconds: Any = None,
    age_text: Any = None,
    item: str = "data",
    delayed: bool = False,
    delay_minutes: Any = None,
    timestamp_available: bool = True,
) -> Optional[str]:
    if delayed:
        delay = None
        try:
            numeric_delay = float(delay_minutes)
            if math.isfinite(numeric_delay) and numeric_delay > 0:
                delay = int(round(numeric_delay))
        except Exception:
            delay = None
        label = f"delayed {delay}m" if delay else "delayed"
        if not timestamp_available:
            return f"{label}, timestamp unavailable"
        return label

    status = _clean_status(market_status)
    reason = _clean_status(market_status_reason)
    if status == "closed":
        label = "closed"
        if reason:
            label = f"{label} {reason}"
    elif status and status not in {"open", "live"}:
        label = status
    else:
        stale_flag = _coerce_bool_flag(data_stale)
        if stale_flag is True:
            label = "stale"
        elif stale_flag is False:
            label = "fresh"
        else:
            return None

    age = str(age_text or "").strip() or format_age_seconds(age_seconds)
    if age:
        item_label = str(item or "data").strip() or "data"
        return f"{label}, {item_label} {age} ago"
    return label
