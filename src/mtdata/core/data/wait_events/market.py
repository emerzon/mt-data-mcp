"""Market spike, drought, level, near-fill, and stop-threat evaluators."""

from __future__ import annotations

import statistics
from datetime import datetime
from typing import (
    Any,
    Dict,
    List,
    Optional,
)

from mtdata.core.data.wait_events.account import (
    _matches_account_filters,
    _row_side,
    _row_time_iso,
)
from mtdata.core.data.wait_events.ticks import (
    _apply_window_metric_threshold,
    _current_price_change,
    _current_range_metric,
    _current_spread_metric,
    _current_volume_metric,
    _finite_number,
    _first_int,
    _market_price_points,
    _price_change_baseline_samples,
    _range_baseline_samples,
    _resolve_market_volume_source,
    _row_int,
    _row_value,
    _spread_baseline_samples,
    _tick_price,
    _volume_baseline_samples,
    _zscore,
)

_MARKET_EVENT_TYPES = {
    "price_change",
    "volume_spike",
    "tick_count_spike",
    "spread_spike",
    "tick_count_drought",
    "range_expansion",
    "price_touch_level",
    "price_break_level",
    "price_enter_zone",
    "pending_near_fill",
    "stop_threat",
}

_MARKET_METRIC_EVENT_TYPES = {
    "price_change",
    "volume_spike",
    "tick_count_spike",
    "spread_spike",
    "tick_count_drought",
    "range_expansion",
}

def _evaluate_market_event(
    spec: Dict[str, Any],
    market_data: Any,
    *,
    snapshot: Dict[str, Any],
    gateway: Any,
    live_state_cutoff_utc: Optional[datetime] = None,
    event_start_utc: Optional[datetime] = None,
) -> Optional[Dict[str, Any]]:
    event_type = str(spec.get("type") or "")
    match: Optional[Dict[str, Any]] = None
    if event_type == "price_change":
        match = _evaluate_price_change(spec, market_data)
    elif event_type in {"volume_spike", "tick_count_spike"}:
        match = _evaluate_volume_spike(spec, market_data)
    elif event_type == "spread_spike":
        match = _evaluate_spread_spike(spec, market_data)
    elif event_type == "tick_count_drought":
        match = _evaluate_tick_count_drought(spec, market_data)
    elif event_type == "range_expansion":
        match = _evaluate_range_expansion(spec, market_data)
    elif event_type == "price_touch_level":
        match = _evaluate_price_touch_level(
            spec,
            market_data,
            event_start_utc=event_start_utc,
        )
    elif event_type == "price_break_level":
        match = _evaluate_price_break_level(
            spec,
            market_data,
            event_start_utc=event_start_utc,
        )
    elif event_type == "price_enter_zone":
        match = _evaluate_price_enter_zone(
            spec,
            market_data,
            event_start_utc=event_start_utc,
        )
    elif event_type == "pending_near_fill":
        if live_state_cutoff_utc is not None:
            return None
        match = _evaluate_pending_near_fill(
            spec,
            snapshot.get("orders", []),
            market_data,
            gateway=gateway,
        )
    elif event_type == "stop_threat":
        if live_state_cutoff_utc is not None:
            return None
        match = _evaluate_stop_threat(
            spec,
            snapshot.get("positions", []),
            market_data,
            gateway=gateway,
        )
    if event_type in _MARKET_METRIC_EVENT_TYPES and event_start_utc is not None:
        if bool(spec.get("_preexisting_match_latched")):
            if match is None:
                spec["_preexisting_match_latched"] = False
            return None
    return match

def _prime_market_metric_latches(
    *,
    watch_for: List[Dict[str, Any]],
    market_state: Dict[str, Any],
    gateway: Any,
) -> None:
    """Suppress already-satisfied rolling metrics until they clear once."""
    for spec in watch_for:
        event_type = str(spec.get("type") or "")
        if event_type not in _MARKET_METRIC_EVENT_TYPES:
            continue
        match = _evaluate_market_event(
            spec,
            (market_state or {}).get(spec.get("symbol")),
            snapshot={"baseline": {}},
            gateway=gateway,
        )
        spec["_preexisting_match_latched"] = match is not None

def _evaluate_price_change(spec: Dict[str, Any], market_data: Any) -> Optional[Dict[str, Any]]:
    ticks = list((market_data or {}).get("ticks", []))
    if not ticks:
        return None

    prices = _market_price_points(ticks, source=str(spec.get("price_source") or "auto"))
    current_change = _current_price_change(spec, prices)
    if current_change is None:
        return None
    magnitude = abs(current_change)
    if not _price_direction_matches(spec["direction"], current_change):
        return None

    observed: Dict[str, Any] = {
        "symbol": spec["symbol"],
        "window": spec["window"],
        "baseline_window": spec["baseline_window"],
        "price_source": spec["price_source"],
        "current_change_pct": round(current_change, 6),
        "absolute_change_pct": round(magnitude, 6),
    }

    threshold_mode = spec["threshold_mode"]
    threshold_value = float(spec["threshold_value"])
    if threshold_mode == "fixed_pct":
        if magnitude < threshold_value:
            return None
        observed["threshold_value"] = threshold_value
    else:
        samples = _price_change_baseline_samples(spec, prices)
        if not samples:
            return None
        baseline_center = statistics.median(samples)
        observed["baseline_median_abs_change_pct"] = round(baseline_center, 6)
        if threshold_mode == "ratio_to_baseline":
            if baseline_center <= 0.0:
                return None
            ratio = magnitude / baseline_center
            observed["ratio"] = round(ratio, 6)
            if ratio < threshold_value:
                return None
        elif threshold_mode == "zscore":
            zscore = _zscore(magnitude, samples)
            if zscore is None or zscore < threshold_value:
                return None
            observed["zscore"] = round(zscore, 6)
        else:
            return None

    return {
        "type": spec["type"],
        "criteria": {
            "symbol": spec["symbol"],
            "price_source": spec["price_source"],
            "direction": spec["direction"],
            "threshold_mode": spec["threshold_mode"],
            "threshold_value": threshold_value,
            "window": spec["window"],
            "baseline_window": spec["baseline_window"],
        },
        "observed": observed,
    }

def _evaluate_volume_spike(spec: Dict[str, Any], market_data: Any) -> Optional[Dict[str, Any]]:
    ticks = list((market_data or {}).get("ticks", []))
    if not ticks:
        return None

    volume_source = _resolve_market_volume_source(ticks, preferred=spec["source"], window_kind=spec["window"]["kind"])
    current_volume = _current_volume_metric(spec, ticks, source=volume_source)
    if current_volume is None:
        return None

    observed: Dict[str, Any] = {
        "symbol": spec["symbol"],
        "window": spec["window"],
        "baseline_window": spec["baseline_window"],
        "volume_source": volume_source,
    }
    samples = _volume_baseline_samples(spec, ticks, source=volume_source)
    threshold_value = _apply_window_metric_threshold(
        spec,
        current_value=current_volume,
        samples=samples,
        observed=observed,
        current_label="current_window_volume",
        baseline_label="baseline_median_window_volume",
        mode="spike",
    )
    if threshold_value is None:
        return None

    return {
        "type": spec["type"],
        "criteria": {
            "symbol": spec["symbol"],
            "source": spec["source"],
            "threshold_mode": spec["threshold_mode"],
            "threshold_value": threshold_value,
            "window": spec["window"],
            "baseline_window": spec["baseline_window"],
        },
        "observed": observed,
    }

def _evaluate_spread_spike(spec: Dict[str, Any], market_data: Any) -> Optional[Dict[str, Any]]:
    ticks = list((market_data or {}).get("ticks", []))
    current_spread = _current_spread_metric(spec, ticks)
    if current_spread is None:
        return None
    observed: Dict[str, Any] = {
        "symbol": spec["symbol"],
        "window": spec["window"],
        "baseline_window": spec["baseline_window"],
    }
    threshold_value = _apply_window_metric_threshold(
        spec,
        current_value=current_spread,
        samples=_spread_baseline_samples(spec, ticks),
        observed=observed,
        current_label="current_window_max_spread",
        baseline_label="baseline_median_window_max_spread",
        mode="spike",
    )
    if threshold_value is None:
        return None
    return {
        "type": spec["type"],
        "criteria": {
            "symbol": spec["symbol"],
            "threshold_mode": spec["threshold_mode"],
            "threshold_value": threshold_value,
            "window": spec["window"],
            "baseline_window": spec["baseline_window"],
        },
        "observed": observed,
    }

def _evaluate_tick_count_drought(spec: Dict[str, Any], market_data: Any) -> Optional[Dict[str, Any]]:
    ticks = list((market_data or {}).get("ticks", []))
    current_volume = _current_volume_metric(spec, ticks, source="tick_count")
    if current_volume is None:
        return None
    observed: Dict[str, Any] = {
        "symbol": spec["symbol"],
        "window": spec["window"],
        "baseline_window": spec["baseline_window"],
        "volume_source": "tick_count",
    }
    threshold_value = _apply_window_metric_threshold(
        spec,
        current_value=current_volume,
        samples=_volume_baseline_samples(spec, ticks, source="tick_count"),
        observed=observed,
        current_label="current_window_volume",
        baseline_label="baseline_median_window_volume",
        mode="drought",
    )
    if threshold_value is None:
        return None
    return {
        "type": spec["type"],
        "criteria": {
            "symbol": spec["symbol"],
            "threshold_mode": spec["threshold_mode"],
            "threshold_value": threshold_value,
            "window": spec["window"],
            "baseline_window": spec["baseline_window"],
        },
        "observed": observed,
    }

def _evaluate_range_expansion(spec: Dict[str, Any], market_data: Any) -> Optional[Dict[str, Any]]:
    ticks = list((market_data or {}).get("ticks", []))
    prices = _market_price_points(ticks, source=str(spec.get("price_source") or "auto"))
    current_range_pct = _current_range_metric(spec, prices)
    if current_range_pct is None:
        return None
    observed: Dict[str, Any] = {
        "symbol": spec["symbol"],
        "window": spec["window"],
        "baseline_window": spec["baseline_window"],
        "price_source": spec["price_source"],
    }
    threshold_value = _apply_window_metric_threshold(
        spec,
        current_value=current_range_pct,
        samples=_range_baseline_samples(spec, prices),
        observed=observed,
        current_label="current_window_range_pct",
        baseline_label="baseline_median_window_range_pct",
        mode="spike",
    )
    if threshold_value is None:
        return None
    return {
        "type": spec["type"],
        "criteria": {
            "symbol": spec["symbol"],
            "price_source": spec["price_source"],
            "threshold_mode": spec["threshold_mode"],
            "threshold_value": threshold_value,
            "window": spec["window"],
            "baseline_window": spec["baseline_window"],
        },
        "observed": observed,
    }

def _event_price_points(spec: Dict[str, Any], market_data: Any) -> List[tuple[float, float]]:
    prices = _market_price_points(
        list((market_data or {}).get("ticks", [])),
        source=str(spec.get("price_source") or "auto"),
    )
    return prices

def _evaluate_price_touch_level(
    spec: Dict[str, Any],
    market_data: Any,
    *,
    event_start_utc: Optional[datetime] = None,
) -> Optional[Dict[str, Any]]:
    prices = _event_price_points(spec, market_data)
    if len(prices) < 2:
        return None
    level = float(spec["level"])
    tolerance = float(spec.get("tolerance") or 0.0)
    lower = level - tolerance
    upper = level + tolerance
    direction = str(spec.get("direction") or "either")
    matched_pair = None
    for previous, current in zip(prices, prices[1:]):
        if event_start_utc is not None and float(current[0]) <= event_start_utc.timestamp():
            continue
        previous_price = float(previous[1])
        current_price = float(current[1])
        upward_touch = previous_price < lower and current_price >= lower
        downward_touch = previous_price > upper and current_price <= upper
        if (
            (direction == "up" and upward_touch)
            or (direction == "down" and downward_touch)
            or (direction == "either" and (upward_touch or downward_touch))
        ):
            matched_pair = (previous_price, current_price)
            break
    if matched_pair is None:
        return None
    previous_price, current_price = matched_pair
    return {
        "type": spec["type"],
        "criteria": {
            "symbol": spec["symbol"],
            "level": level,
            "tolerance": tolerance,
            "direction": direction,
            "price_source": spec["price_source"],
        },
        "observed": {
            "symbol": spec["symbol"],
            "price_source": spec["price_source"],
            "previous_price": round(previous_price, 8),
            "current_price": round(current_price, 8),
            "level": round(level, 8),
            "tolerance": round(tolerance, 8),
            "distance": round(abs(current_price - level), 8),
        },
    }

def _evaluate_price_break_level(
    spec: Dict[str, Any],
    market_data: Any,
    *,
    event_start_utc: Optional[datetime] = None,
) -> Optional[Dict[str, Any]]:
    prices = _event_price_points(spec, market_data)
    confirm_ticks = max(1, int(spec.get("confirm_ticks") or 1))
    if len(prices) < confirm_ticks + 1:
        return None
    level = float(spec["level"])
    tolerance = float(spec.get("tolerance") or 0.0)
    upper = level + tolerance
    lower = level - tolerance
    direction = str(spec.get("direction") or "either")
    matched_window = None
    for end in range(confirm_ticks + 1, len(prices) + 1):
        previous_price = float(prices[end - confirm_ticks - 1][1])
        confirmed_points = prices[end - confirm_ticks : end]
        if event_start_utc is not None and any(
            float(epoch) <= event_start_utc.timestamp()
            for epoch, _ in confirmed_points
        ):
            continue
        confirmed_prices = [float(price) for _, price in confirmed_points]
        breakout_up = previous_price < lower and all(price >= upper for price in confirmed_prices)
        breakout_down = previous_price > upper and all(price <= lower for price in confirmed_prices)
        if (
            (direction == "up" and breakout_up)
            or (direction == "down" and breakout_down)
            or (direction == "either" and (breakout_up or breakout_down))
        ):
            matched_window = (previous_price, confirmed_prices)
            break
    if matched_window is None:
        return None
    previous_price, confirmed_prices = matched_window
    return {
        "type": spec["type"],
        "criteria": {
            "symbol": spec["symbol"],
            "level": level,
            "tolerance": tolerance,
            "direction": direction,
            "confirm_ticks": confirm_ticks,
            "price_source": spec["price_source"],
        },
        "observed": {
            "symbol": spec["symbol"],
            "price_source": spec["price_source"],
            "previous_price": round(previous_price, 8),
            "current_price": round(confirmed_prices[-1], 8),
            "level": round(level, 8),
            "tolerance": round(tolerance, 8),
            "confirm_ticks": confirm_ticks,
        },
    }

def _evaluate_price_enter_zone(
    spec: Dict[str, Any],
    market_data: Any,
    *,
    event_start_utc: Optional[datetime] = None,
) -> Optional[Dict[str, Any]]:
    prices = _event_price_points(spec, market_data)
    if len(prices) < 2:
        return None
    lower = float(spec["lower"])
    upper = float(spec["upper"])
    direction = str(spec.get("direction") or "either")
    matched_pair = None
    for previous, current in zip(prices, prices[1:]):
        if event_start_utc is not None and float(current[0]) <= event_start_utc.timestamp():
            continue
        previous_price = float(previous[1])
        current_price = float(current[1])
        crosses_zone = not (
            max(previous_price, current_price) < lower
            or min(previous_price, current_price) > upper
        )
        enter_up = previous_price < lower
        enter_down = previous_price > upper
        if (
            crosses_zone
            and not _price_within_band(previous_price, lower=lower, upper=upper)
            and (
                (direction == "up" and enter_up)
                or (direction == "down" and enter_down)
                or (direction == "either" and (enter_up or enter_down))
            )
        ):
            matched_pair = (previous_price, current_price)
            break
    if matched_pair is None:
        return None
    previous_price, current_price = matched_pair
    return {
        "type": spec["type"],
        "criteria": {
            "symbol": spec["symbol"],
            "lower": lower,
            "upper": upper,
            "direction": direction,
            "price_source": spec["price_source"],
        },
        "observed": {
            "symbol": spec["symbol"],
            "price_source": spec["price_source"],
            "previous_price": round(previous_price, 8),
            "current_price": round(current_price, 8),
            "lower": round(lower, 8),
            "upper": round(upper, 8),
        },
    }

def _evaluate_pending_near_fill(
    spec: Dict[str, Any],
    orders: List[Any],
    market_data: Any,
    *,
    gateway: Any,
) -> Optional[Dict[str, Any]]:
    for row in orders:
        if not _matches_account_filters(row, spec, gateway=gateway):
            continue
        order_price = _order_reference_price(row)
        if order_price is None:
            continue
        side = _row_side(row, gateway=gateway)
        current_price = _latest_market_price(
            market_data,
            price_source=str(spec.get("price_source") or "auto"),
            side=side,
            fallback_row=row,
        )
        if current_price is None:
            continue
        distance = abs(float(current_price) - float(order_price))
        max_distance = float(spec.get("distance") or 0.0)
        if distance > max_distance + 1e-12:
            continue
        return {
            "type": spec["type"],
            "criteria": {
                "symbol": spec["symbol"],
                "distance": max_distance,
                "price_source": spec["price_source"],
                "order_ticket": spec.get("order_ticket"),
                "magic": spec.get("magic"),
                "side": spec.get("side"),
            },
            "observed": {
                "ticket": _row_int(row, "ticket"),
                "order_ticket": _first_int(_row_int(row, "ticket"), _row_int(row, "order")),
                "symbol": _row_value(row, "symbol"),
                "side": side,
                "order_price": round(float(order_price), 8),
                "current_price": round(float(current_price), 8),
                "distance": round(distance, 8),
                "time_utc": _row_time_iso(row),
            },
        }
    return None

def _evaluate_stop_threat(
    spec: Dict[str, Any],
    positions: List[Any],
    market_data: Any,
    *,
    gateway: Any,
) -> Optional[Dict[str, Any]]:
    for row in positions:
        if not _matches_account_filters(row, spec, gateway=gateway):
            continue
        stop_price = _finite_number(_row_value(row, "sl"))
        if stop_price is None or float(stop_price) <= 0.0:
            continue
        side = _row_side(row, gateway=gateway)
        price_source = str(spec.get("price_source") or "auto")
        if price_source == "auto":
            if side == "buy":
                price_source = "bid"
            elif side == "sell":
                price_source = "ask"
        current_price = _latest_market_price(
            market_data,
            price_source=price_source,
            side=side,
            fallback_row=row,
        )
        if current_price is None:
            continue
        current_price = float(current_price)
        stop_price = float(stop_price)
        max_distance = float(spec.get("distance") or 0.0)
        distance = abs(current_price - stop_price)
        if side == "buy":
            threatened = current_price <= stop_price + max_distance
        elif side == "sell":
            threatened = current_price >= stop_price - max_distance
        else:
            threatened = distance <= max_distance + 1e-12
        if not threatened:
            continue
        return {
            "type": spec["type"],
            "criteria": {
                "symbol": spec["symbol"],
                "distance": max_distance,
                "price_source": spec["price_source"],
                "position_ticket": spec.get("position_ticket"),
                "magic": spec.get("magic"),
                "side": spec.get("side"),
            },
            "observed": {
                "ticket": _row_int(row, "ticket"),
                "position_ticket": _first_int(
                    _row_int(row, "ticket"),
                    _row_int(row, "position_id"),
                    _row_int(row, "position"),
                ),
                "symbol": _row_value(row, "symbol"),
                "side": side,
                "stop_price": round(stop_price, 8),
                "current_price": round(current_price, 8),
                "distance": round(distance, 8),
                "time_utc": _row_time_iso(row),
            },
        }
    return None

def _price_within_band(price: float, *, lower: float, upper: float) -> bool:
    return float(lower) - 1e-12 <= float(price) <= float(upper) + 1e-12

def _latest_market_price(
    market_data: Any,
    *,
    price_source: str,
    side: Optional[str],
    fallback_row: Any,
) -> Optional[float]:
    ticks = list((market_data or {}).get("ticks", []))
    effective_source = price_source
    if effective_source == "auto":
        if side == "buy":
            effective_source = "ask"
        elif side == "sell":
            effective_source = "bid"
    if ticks:
        price = _tick_price(ticks[-1], source=effective_source)
        if price is not None:
            return float(price)
    for key in ("price_current", "price_open", "price"):
        value = _finite_number(_row_value(fallback_row, key))
        if value is not None:
            return float(value)
    return None

def _order_reference_price(row: Any) -> Optional[float]:
    for key in ("price_open", "price_current", "price"):
        value = _finite_number(_row_value(row, key))
        if value is not None:
            return float(value)
    return None

def _price_direction_matches(direction: str, current_change: float) -> bool:
    if direction == "either":
        return True
    if direction == "up":
        return current_change > 0.0
    if direction == "down":
        return current_change < 0.0
    return False
