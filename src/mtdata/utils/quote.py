from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from numbers import Real
from typing import Any, Dict, Optional

from .freshness import QUOTE_STALE_SECONDS, standard_weekend_window
from .market_metadata import build_tick_freshness_context


def tick_value(tick: Any, field: str) -> Any:
    if isinstance(tick, dict):
        return tick.get(field)
    try:
        value = tick[field]
        if type(value).__module__ != "unittest.mock":
            return value
    except Exception:
        pass
    return getattr(tick, field, None)


def tick_epoch(tick: Any) -> Optional[float]:
    def _number(value: Any) -> Optional[float]:
        # Mock placeholders and arbitrary objects may implement ``__float__``;
        # accepting them can turn a missing attribute into a plausible epoch.
        if isinstance(value, bool) or not isinstance(value, (Real, str)):
            return None
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        return number if math.isfinite(number) and number > 0.0 else None

    time_msc = _number(tick_value(tick, "time_msc"))
    if time_msc is not None:
        return time_msc / 1000.0
    return _number(tick_value(tick, "time"))


def canonical_quote_midpoint(bid: Any, ask: Any) -> Optional[float]:
    """Return the exact decimal midpoint represented by broker quote strings.

    Converting each quote through ``str`` first removes binary-float addition
    artifacts while retaining legitimate half-tick precision.
    """
    try:
        bid_decimal = Decimal(str(bid))
        ask_decimal = Decimal(str(ask))
        midpoint = (bid_decimal + ask_decimal) / Decimal(2)
    except (InvalidOperation, TypeError, ValueError):
        return None
    if not midpoint.is_finite() or midpoint <= 0:
        return None
    return float(midpoint)


def canonical_quote_spread(bid: Any, ask: Any) -> Optional[float]:
    """Return ask minus bid without leaking binary subtraction artifacts."""
    try:
        spread = Decimal(str(ask)) - Decimal(str(bid))
    except (InvalidOperation, TypeError, ValueError):
        return None
    return float(spread) if spread.is_finite() else None


def compute_spread_metrics(
    bid: Any,
    ask: Any,
    *,
    point: Any = None,
    points_per_pip: Any = None,
    tick_size: Any = None,
    tick_value_money: Any = None,
    account_currency: Optional[str] = None,
) -> Dict[str, Any]:
    """Return unrounded spread measurements and a canonical quote-quality label."""

    def _positive(value: Any) -> Optional[float]:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) and number > 0.0 else None

    bid_value = _positive(bid)
    ask_value = _positive(ask)
    if bid_value is None or ask_value is None:
        quality = "one_sided"
    elif ask_value < bid_value:
        quality = "inverted"
    elif ask_value == bid_value:
        quality = "locked"
    else:
        quality = "two_sided"

    out: Dict[str, Any] = {
        "mid": None,
        "spread": None,
        "spread_points": None,
        "spread_pips": None,
        "spread_pct": None,
        "spread_cost_per_lot": None,
        "spread_valid": quality == "two_sided",
        "spread_quality": quality,
        "pricing_basis": "quote_only",
    }
    if quality not in {"two_sided", "locked"}:
        return out

    spread = canonical_quote_spread(bid_value, ask_value)
    mid = canonical_quote_midpoint(bid_value, ask_value)
    if spread is None or mid is None:
        return out
    point_value = _positive(point)
    pip_points = _positive(points_per_pip)
    tick_size_value = _positive(tick_size)
    tick_money_value = _positive(tick_value_money)
    spread_points = spread / point_value if point_value is not None else None
    spread_pips = (
        spread_points / pip_points
        if spread_points is not None and pip_points is not None
        else None
    )
    spread_cost = (
        (spread / tick_size_value) * tick_money_value
        if tick_size_value is not None
        and tick_money_value is not None
        and account_currency
        else None
    )
    out.update(
        {
            "mid": mid,
            "spread": spread,
            "spread_points": spread_points,
            "spread_pips": spread_pips,
            "spread_pct": (spread / mid) * 100.0,
            "spread_cost_per_lot": spread_cost,
            "pricing_basis": (
                "per_1_lot_estimate" if spread_cost is not None else "quote_only"
            ),
        }
    )
    return out


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
    # CopyTicksRange returns ticks from oldest to newest.  Several updates can
    # share a millisecond, so preserve that ordering when timestamps tie: max()
    # alone would retain the first (stale) row with the newest timestamp.
    _, latest_tick = max(
        enumerate(candidates),
        key=lambda item: (float(tick_epoch(item[1]) or 0.0), item[0]),
    )
    return latest_tick


def _quote_pair(tick: Any) -> tuple[Optional[float], Optional[float]]:
    values = []
    for field in ("bid", "ask"):
        try:
            value = float(tick_value(tick, field))
        except (TypeError, ValueError):
            value = float("nan")
        values.append(value if math.isfinite(value) and value > 0.0 else None)
    return values[0], values[1]


def _quote_pair_quality(tick: Any) -> str:
    bid, ask = _quote_pair(tick)
    if bid is None or ask is None:
        return "one_sided"
    if ask < bid:
        return "inverted"
    if ask == bid:
        return "locked"
    return "two_sided"


def _quote_pair_quality_rank(tick: Any) -> int:
    return {"inverted": 0, "one_sided": 1, "locked": 2, "two_sided": 3}[
        _quote_pair_quality(tick)
    ]


def resolve_quote_tick(
    gateway: Any,
    symbol: str,
    tick: Any = None,
    *,
    now_epoch: float,
    stale_after_seconds: int = QUOTE_STALE_SECONDS,
) -> tuple[Any, Dict[str, Any]]:
    """Reconcile MT5's cached symbol tick with its authoritative tick stream."""
    if tick is not None:
        raw_tick = tick
    else:
        try:
            raw_tick = gateway.symbol_info_tick(symbol)
        except Exception:
            raw_tick = None
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
    use_stream_for_conflict = quote_conflict and (
        _quote_pair_quality_rank(stream_tick) >= _quote_pair_quality_rank(raw_tick)
    )
    raw_ahead_of_observation = (
        raw_epoch is not None and raw_epoch > float(now_epoch)
    )
    stream_not_ahead_of_observation = stream_epoch <= float(now_epoch)
    use_stream = (
        raw_tick is None
        or use_stream_for_conflict
        or (
            raw_ahead_of_observation
            and stream_not_ahead_of_observation
            and stream_live_ready
        )
        or (raw_epoch is not None and stream_epoch > raw_epoch + 0.001)
        or (not raw_live_ready and stream_live_ready)
    )
    if not use_stream:
        metadata["quote_source_state"] = (
            "reconciled_equal_timestamp_conflict"
            if quote_conflict
            else ("current" if raw_live_ready else "unverified_stale")
        )
        metadata["stream_tick_time_epoch"] = stream_epoch
        if quote_conflict:
            metadata["quote_source_conflict"] = {
                "reason": "equal_timestamp_bid_ask_disagreement",
                "time_epoch": stream_epoch,
                "selected_source": "mt5.symbol_info_tick",
                "symbol_info_tick": dict(zip(("bid", "ask"), _quote_pair(raw_tick))),
                "stream_tick": dict(zip(("bid", "ask"), _quote_pair(stream_tick))),
            }
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
