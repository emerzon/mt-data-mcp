from __future__ import annotations

import math
from datetime import datetime, time, timedelta, timezone
from typing import Any, Optional
from zoneinfo import ZoneInfo

from ..shared.symbols import is_probably_crypto_symbol

# Keep market-data readiness aligned with the pre-trade validator so the same
# quote age is not rejected by one public surface and accepted by another.
QUOTE_LIVE_SECONDS = 30
QUOTE_RECENT_SECONDS = 60
QUOTE_STALE_SECONDS = 300
_NEW_YORK = ZoneInfo("America/New_York")


def standard_weekend_window(now_utc: datetime) -> Optional[tuple[datetime, datetime]]:
    """Return the DST-aware FX weekend window containing ``now_utc``."""
    utc_value = now_utc if now_utc.tzinfo else now_utc.replace(tzinfo=timezone.utc)
    new_york = utc_value.astimezone(_NEW_YORK)
    days_since_friday = (new_york.weekday() - 4) % 7
    friday = new_york.date() - timedelta(days=days_since_friday)
    close_local = datetime.combine(friday, time(17), tzinfo=_NEW_YORK)
    open_local = datetime.combine(friday + timedelta(days=2), time(17), tzinfo=_NEW_YORK)
    close_utc = close_local.astimezone(timezone.utc)
    open_utc = open_local.astimezone(timezone.utc)
    if close_utc <= utc_value < open_utc:
        return close_utc, open_utc
    return None


def is_standard_weekend_closure(now_utc: datetime) -> bool:
    return standard_weekend_window(now_utc) is not None


def closed_session_context(
    symbol: Any,
    *,
    now_epoch: Any,
    item: str = "tick",
    data_age_seconds: Any = None,
) -> Optional[dict[str, Any]]:
    if (
        not str(symbol or "").strip()
        or is_probably_crypto_symbol(symbol)
    ):
        return None
    try:
        now_utc = datetime.fromtimestamp(float(now_epoch), tz=timezone.utc)
    except Exception:
        return None
    closure_window = standard_weekend_window(now_utc)
    if closure_window is None:
        return None
    item_label = str(item or "data").strip() or "data"
    out = {
        "market_status": "closed",
        "market_status_reason": "weekend",
        "market_status_source": "standard_weekend_hours",
        "note": f"Market is closed; showing the latest completed session {item_label}.",
    }
    if data_age_seconds is not None:
        try:
            age_seconds = max(0.0, float(data_age_seconds))
        except (TypeError, ValueError):
            age_seconds = float("inf")
        close_utc, open_utc = closure_window
        closure_seconds = (open_utc - close_utc).total_seconds()
        out.update(
            {
                "data_age_seconds": None if not math.isfinite(age_seconds) else age_seconds,
                "assumed_closure_start": close_utc.isoformat().replace("+00:00", "Z"),
                "assumed_closure_end": open_utc.isoformat().replace("+00:00", "Z"),
                "assumed_closure_seconds": closure_seconds,
                "freshness_policy_relaxed": age_seconds <= closure_seconds,
            }
        )
    return out


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
