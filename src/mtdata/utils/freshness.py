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
    elif data_stale is True:
        label = "stale"
    elif data_stale is False:
        label = "fresh"
    else:
        return None

    age = str(age_text or "").strip() or format_age_seconds(age_seconds)
    if age:
        item_label = str(item or "data").strip() or "data"
        return f"{label}, {item_label} {age} ago"
    return label
