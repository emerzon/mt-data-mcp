from __future__ import annotations

from typing import Any

_DEFAULT_TRADE_FLAG_VALUES = {
    "TICK_FLAG_LAST": 8,
    "TICK_FLAG_VOLUME": 16,
    "TICK_FLAG_BUY": 32,
    "TICK_FLAG_SELL": 64,
    "TICK_FLAG_VOLUME_REAL": 1024,
}


def mt5_trade_event_mask(gateway: Any = None) -> int:
    """Return the MT5 flag mask that identifies last-trade state changes."""
    mask = 0
    for name, default in _DEFAULT_TRADE_FLAG_VALUES.items():
        try:
            candidate = getattr(gateway, name) if gateway is not None else default
            value = int(candidate) if isinstance(candidate, (int, float)) else default
        except (AttributeError, TypeError, ValueError):
            value = default
        mask |= value
    return mask


def is_mt5_trade_event(flags: Any, gateway: Any = None) -> bool:
    try:
        value = int(flags)
    except (TypeError, ValueError):
        return False
    return bool(value & mt5_trade_event_mask(gateway))
