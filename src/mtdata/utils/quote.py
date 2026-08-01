from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from .freshness import QUOTE_STALE_SECONDS, standard_weekend_window
from .market_metadata import build_tick_freshness_context


def tick_value(tick: Any, field: str) -> Any:
    if isinstance(tick, dict):
        return tick.get(field)
    try:
        return tick[field]
    except Exception:
        return getattr(tick, field, None)


def tick_epoch(tick: Any) -> Optional[float]:
    time_msc = tick_value(tick, "time_msc")
    try:
        epoch = float(time_msc) / 1000.0
        if math.isfinite(epoch) and epoch > 0.0:
            return epoch
    except (TypeError, ValueError):
        pass
    try:
        epoch = float(tick_value(tick, "time"))
    except (TypeError, ValueError):
        return None
    return epoch if math.isfinite(epoch) and epoch > 0.0 else None


def _latest_stream_tick(gateway: Any, symbol: str, *, now_epoch: float) -> Any:
    end = datetime.fromtimestamp(now_epoch, tz=timezone.utc) + timedelta(seconds=5)
    start = end - timedelta(minutes=15, seconds=5)
    closure = standard_weekend_window(datetime.fromtimestamp(now_epoch, tz=timezone.utc))
    if closure is not None:
        # Include only the final active portion of Friday, not an arbitrary
        # multi-day tick history, when reconciling a frozen weekend quote.
        start = closure[0] - timedelta(minutes=15)
    try:
        rows = gateway.copy_ticks_range(
            symbol,
            start,
            end,
            gateway.COPY_TICKS_ALL,
        )
    except Exception:
        return None
    if rows is None:
        return None
    try:
        candidates = [row for row in rows if tick_epoch(row) is not None]
    except (TypeError, ValueError):
        return None
    if not candidates:
        return None
    return max(candidates, key=lambda row: float(tick_epoch(row) or 0.0))


def _quote_pair(tick: Any) -> tuple[Optional[float], Optional[float]]:
    values = []
    for field in ("bid", "ask"):
        try:
            value = float(tick_value(tick, field))
        except (TypeError, ValueError):
            value = float("nan")
        values.append(value if math.isfinite(value) and value > 0.0 else None)
    return values[0], values[1]


def resolve_quote_tick(
    gateway: Any,
    symbol: str,
    tick: Any = None,
    *,
    now_epoch: float,
    stale_after_seconds: int = QUOTE_STALE_SECONDS,
) -> tuple[Any, Dict[str, Any]]:
    """Reconcile MT5's cached symbol tick with its authoritative tick stream."""
    raw_tick = tick if tick is not None else gateway.symbol_info_tick(symbol)
    raw_epoch = tick_epoch(raw_tick)
    stream_tick = _latest_stream_tick(gateway, symbol, now_epoch=now_epoch)
    stream_epoch = tick_epoch(stream_tick)
    metadata: Dict[str, Any] = {
        "quote_source": "mt5.symbol_info_tick",
        "quote_refresh_attempted": True,
    }

    raw_freshness = build_tick_freshness_context(
        symbol,
        tick_epoch=raw_epoch,
        now_epoch=now_epoch,
        item="tick",
        stale_after_seconds=stale_after_seconds,
    )
    raw_live_ready = raw_freshness.get("usable_for_live_trading") is True

    if stream_tick is None or stream_epoch is None:
        metadata["quote_source_state"] = (
            "current" if raw_live_ready else "unverified_stale"
        )
        return raw_tick, metadata

    stream_freshness = build_tick_freshness_context(
        symbol,
        tick_epoch=stream_epoch,
        now_epoch=now_epoch,
        item="tick",
        stale_after_seconds=stale_after_seconds,
    )
    stream_live_ready = stream_freshness.get("usable_for_live_trading") is True
    same_epoch = raw_epoch is not None and abs(stream_epoch - raw_epoch) <= 0.001
    quote_conflict = same_epoch and _quote_pair(raw_tick) != _quote_pair(stream_tick)
    use_stream = (
        raw_tick is None
        or quote_conflict
        or (raw_epoch is not None and stream_epoch > raw_epoch + 0.001)
        or (not raw_live_ready and stream_live_ready)
    )
    if not use_stream:
        metadata["quote_source_state"] = (
            "current" if raw_live_ready else "unverified_stale"
        )
        metadata["stream_tick_time_epoch"] = stream_epoch
        return raw_tick, metadata

    metadata.update(
        {
            "quote_source": "mt5.copy_ticks_range",
            "quote_source_state": (
                "reconciled_equal_timestamp_conflict"
                if quote_conflict
                else "refreshed_from_tick_stream"
            ),
            "symbol_info_tick_time_epoch": raw_epoch,
            "stream_tick_time_epoch": stream_epoch,
        }
    )
    if quote_conflict:
        metadata["quote_source_conflict"] = {
            "reason": "equal_timestamp_bid_ask_disagreement",
            "time_epoch": stream_epoch,
            "selected_source": "mt5.copy_ticks_range",
            "symbol_info_tick": dict(zip(("bid", "ask"), _quote_pair(raw_tick))),
            "stream_tick": dict(zip(("bid", "ask"), _quote_pair(stream_tick))),
        }
    return stream_tick, metadata
