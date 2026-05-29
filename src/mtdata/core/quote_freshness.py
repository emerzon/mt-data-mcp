from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from ..shared.symbols import is_probably_crypto_symbol

QUOTE_STALE_SECONDS = 300


def is_standard_weekend_closure(now_utc: datetime) -> bool:
    weekday = now_utc.weekday()
    if weekday == 5:
        return True
    if weekday == 6 and now_utc.hour < 22:
        return True
    if weekday == 4 and now_utc.hour >= 22:
        return True
    return False


def quote_closed_session_context(
    symbol: Any,
    *,
    now_epoch: Any,
) -> Optional[Dict[str, Any]]:
    if not str(symbol or "").strip():
        return None
    if is_probably_crypto_symbol(symbol):
        return None
    try:
        now_utc = datetime.fromtimestamp(float(now_epoch), tz=timezone.utc)
    except Exception:
        return None
    if not is_standard_weekend_closure(now_utc):
        return None
    return {
        "market_status": "closed",
        "market_status_reason": "weekend",
        "market_status_source": "standard_weekend_hours",
        "note": "Market is closed; showing the latest completed session tick.",
    }
