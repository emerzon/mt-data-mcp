from __future__ import annotations

from datetime import datetime, timedelta, timezone
import math
import re
import statistics
from typing import Any, Callable, Dict, List, Optional

from .data_requests import (
    CandleCloseEventSpec,
    OrderCancelledEventSpec,
    OrderCreatedEventSpec,
    OrderFilledEventSpec,
    PendingNearFillEventSpec,
    PositionClosedEventSpec,
    PositionOpenedEventSpec,
    PriceBreakLevelEventSpec,
    PriceChangeEventSpec,
    PriceEnterZoneEventSpec,
    PriceTouchLevelEventSpec,
    RangeExpansionEventSpec,
    SpreadSpikeEventSpec,
    SlHitEventSpec,
    StopThreatEventSpec,
    TickCountDroughtEventSpec,
    TickCountSpikeEventSpec,
    TpHitEventSpec,
    VolumeSpikeEventSpec,
    WaitEventRequest,
    WaitEventWindow,
)
from .trading_time import _next_candle_wait_payload, _sleep_until_next_candle
from ..utils.mt5 import _mt5_epoch_to_utc, _to_server_naive_dt

_MARKET_BOOTSTRAP_MIN_SECONDS = 60.0
_MARKET_BOOTSTRAP_MAX_SECONDS = 14400.0
_MARKET_ESTIMATED_SECONDS_PER_TICK = 2.0
_MARKET_BUFFER_EXTRA_TICKS = 32
_ACCOUNT_HISTORY_SEED_LOOKBACK_SECONDS = 5.0
_ORDER_STATE_EVENT_TYPES = {"order_created", "pending_near_fill"}
_POSITION_STATE_EVENT_TYPES = {"position_opened", "position_closed", "stop_threat"}
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


def _wait_event_connection_error(gateway: Any) -> Optional[Dict[str, Any]]:
    try:
        if hasattr(gateway, "ensure_connection"):
            gateway.ensure_connection()
    except Exception as exc:
        return {"error": f"MT5 connection lost while waiting for events: {exc}"}
    return None


def run_wait_event_loop(
    request: WaitEventRequest,
    *,
    gateway: Any,
    sleep_impl: Callable[[float], None],
    monotonic_impl: Callable[[], float],
    now_utc_impl: Callable[[], datetime],
) -> Dict[str, Any]:
    started_at_utc = _normalize_utc_datetime(now_utc_impl())
    compiled = _compile_request(request, started_at_utc=started_at_utc)
    if "error" in compiled:
        return compiled

    watch_for = compiled["watch_for"]
    boundaries = compiled["end_on"]
    watch_for_inferred = bool(compiled.get("watch_for_inferred"))
    end_on_inferred = bool(compiled.get("end_on_inferred"))
    watch_for_payload = list(compiled.get("watch_for_payload", []))
    end_on_payload = list(compiled.get("end_on_payload", []))
    max_wait_seconds = (
        None if request.max_wait_seconds is None else float(request.max_wait_seconds)
    )
    poll_interval_seconds = float(request.poll_interval_seconds)

    if not watch_for and len(boundaries) == 1 and boundaries[0]["type"] == "candle_close":
        return _run_candle_boundary_only(
            request=request,
            boundary=boundaries[0],
            sleep_impl=sleep_impl,
            now_utc=started_at_utc,
        )
    connection_error = _wait_event_connection_error(gateway)
    if connection_error is not None:
        return connection_error

    history_state = _build_account_history_state(
        gateway=gateway,
        watch_for=watch_for,
        started_at_utc=started_at_utc,
    )
    if isinstance(history_state, dict) and "error" in history_state:
        return history_state

    baseline = _build_baseline(gateway, watch_for) if _watchers_need_current_state(watch_for) else {}
    market_state = _build_market_state(
        gateway=gateway,
        watch_for=watch_for,
        observed_at_utc=started_at_utc,
        poll_interval_seconds=poll_interval_seconds,
    )
    if isinstance(market_state, dict) and "error" in market_state:
        return market_state
    if request.accept_preexisting:
        preexisting_match = _find_preexisting_match(
            watch_for=watch_for,
            baseline=baseline,
            market_state=market_state,
            gateway=gateway,
        )
        if preexisting_match is not None:
            observed_at = _normalize_utc_datetime(now_utc_impl())
            return _build_wait_result(
                request=request,
                status="already_satisfied",
                started_at_utc=started_at_utc,
                observed_at_utc=observed_at,
                polls=0,
                matched_event=preexisting_match,
                boundary_event=None,
                watch_for_payload=watch_for_payload,
                end_on_payload=end_on_payload,
                watch_for_inferred=watch_for_inferred,
                end_on_inferred=end_on_inferred,
            )

    started_at_monotonic = float(monotonic_impl())
    polls = 0
    while True:
        polls += 1
        observed_at_utc = _normalize_utc_datetime(now_utc_impl())
        connection_error = _wait_event_connection_error(gateway)
        if connection_error is not None:
            return connection_error
        snapshot = _collect_snapshot(
            gateway=gateway,
            watch_for=watch_for,
            baseline=baseline,
            history_state=history_state,
            market_state=market_state,
            started_at_utc=started_at_utc,
            observed_at_utc=observed_at_utc,
        )
        if "error" in snapshot:
            return snapshot

        matched_event = _evaluate_watch_events(
            watch_for=watch_for,
            snapshot=snapshot,
            gateway=gateway,
        )
        if matched_event is not None:
            return _build_wait_result(
                request=request,
                status="matched",
                started_at_utc=started_at_utc,
                observed_at_utc=observed_at_utc,
                polls=polls,
                matched_event=matched_event,
                boundary_event=None,
                watch_for_payload=watch_for_payload,
                end_on_payload=end_on_payload,
                watch_for_inferred=watch_for_inferred,
                end_on_inferred=end_on_inferred,
            )

        boundary_event = _evaluate_boundaries(boundaries, observed_at_utc=observed_at_utc)
        if boundary_event is not None:
            return _build_wait_result(
                request=request,
                status="boundary_reached",
                started_at_utc=started_at_utc,
                observed_at_utc=observed_at_utc,
                polls=polls,
                matched_event=None,
                boundary_event=boundary_event,
                watch_for_payload=watch_for_payload,
                end_on_payload=end_on_payload,
                watch_for_inferred=watch_for_inferred,
                end_on_inferred=end_on_inferred,
            )

        elapsed_seconds = max(0.0, float(monotonic_impl()) - started_at_monotonic)
        if max_wait_seconds is not None and elapsed_seconds >= max_wait_seconds:
            return _build_wait_result(
                request=request,
                status="timeout",
                started_at_utc=started_at_utc,
                observed_at_utc=observed_at_utc,
                polls=polls,
                matched_event=None,
                boundary_event=None,
                watch_for_payload=watch_for_payload,
                end_on_payload=end_on_payload,
                watch_for_inferred=watch_for_inferred,
                end_on_inferred=end_on_inferred,
            )

        sleep_seconds = _next_poll_sleep_seconds(
            poll_interval_seconds=poll_interval_seconds,
            max_wait_seconds=max_wait_seconds,
            elapsed_seconds=elapsed_seconds,
            boundaries=boundaries,
            observed_at_utc=observed_at_utc,
        )
        if sleep_seconds <= 0.0:
            continue
        sleep_impl(sleep_seconds)


def _compile_request(
    request: WaitEventRequest,
    *,
    started_at_utc: datetime,
) -> Dict[str, Any]:
    raw_watch_specs = request.watch_for
    watch_for_inferred = raw_watch_specs is None
    source_watch_specs = _default_watch_specs(request) if raw_watch_specs is None else list(raw_watch_specs)
    source_end_specs: List[Any]
    end_on_inferred = False
    if request.end_on:
        source_end_specs = list(request.end_on)
    elif request.timeframe is not None:
        source_end_specs = [CandleCloseEventSpec(timeframe=request.timeframe)]
        end_on_inferred = True
    else:
        source_end_specs = []
    watch_for: List[Dict[str, Any]] = []
    for spec in source_watch_specs:
        compiled = _compile_watch_event(spec, request=request)
        if "error" in compiled:
            return compiled
        watch_for.append(compiled)

    end_on: List[Dict[str, Any]] = []
    for spec in source_end_specs:
        compiled = _compile_boundary_event(
            spec,
            request=request,
            started_at_utc=started_at_utc,
        )
        if "error" in compiled:
            return compiled
        end_on.append(compiled)

    return {
        "watch_for": watch_for,
        "watch_for_inferred": watch_for_inferred,
        "watch_for_payload": [_public_watch_spec_payload(spec, request=request) for spec in source_watch_specs],
        "end_on_inferred": end_on_inferred,
        "end_on": sorted(
            end_on,
            key=lambda item: (
                float(item.get("boundary_at_epoch", math.inf)),
                str(item.get("timeframe") or ""),
            ),
        ),
        "end_on_payload": [_public_boundary_spec_payload(spec, request=request) for spec in source_end_specs],
    }


def _compile_watch_event(spec: Any, *, request: WaitEventRequest) -> Dict[str, Any]:
    if isinstance(spec, OrderCreatedEventSpec):
        return _compile_account_event(spec, request=request)
    if isinstance(spec, OrderFilledEventSpec):
        return _compile_account_event(spec, request=request)
    if isinstance(spec, OrderCancelledEventSpec):
        return _compile_account_event(spec, request=request)
    if isinstance(spec, PositionOpenedEventSpec):
        return _compile_account_event(spec, request=request)
    if isinstance(spec, PositionClosedEventSpec):
        return _compile_account_event(spec, request=request)
    if isinstance(spec, TpHitEventSpec):
        return _compile_account_event(spec, request=request)
    if isinstance(spec, SlHitEventSpec):
        return _compile_account_event(spec, request=request)
    if isinstance(spec, PendingNearFillEventSpec):
        compiled = _compile_account_market_event(spec, request=request)
        if "error" in compiled:
            return compiled
        compiled.update(
            {
                "distance": float(spec.distance),
                "price_source": str(spec.price_source),
                "required_tick_count": 1,
                "required_history_seconds": _MARKET_BOOTSTRAP_MIN_SECONDS,
            }
        )
        return compiled
    if isinstance(spec, StopThreatEventSpec):
        compiled = _compile_account_market_event(spec, request=request)
        if "error" in compiled:
            return compiled
        compiled.update(
            {
                "distance": float(spec.distance),
                "price_source": str(spec.price_source),
                "required_tick_count": 1,
                "required_history_seconds": _MARKET_BOOTSTRAP_MIN_SECONDS,
            }
        )
        return compiled
    if isinstance(spec, PriceChangeEventSpec):
        return _compile_window_metric_event(
            spec,
            request=request,
            required_tick_count=_required_tick_count_for_price_change(spec),
        )
    if isinstance(spec, (VolumeSpikeEventSpec, TickCountSpikeEventSpec)):
        extra = {
            "source": "tick_count" if isinstance(spec, TickCountSpikeEventSpec) else str(spec.source),
        }
        compiled = _compile_window_metric_event(
            spec,
            request=request,
            required_tick_count=_required_tick_count_for_volume_spike(spec),
            extra=extra,
        )
        if (
            "error" not in compiled
            and compiled.get("source") == "tick_count"
            and str(spec.window.kind) == "ticks"
        ):
            return {
                "error": (
                    f"{spec.type} with source='tick_count' requires a minutes window. "
                    "A tick-count metric over a fixed tick window is constant."
                )
            }
        return compiled
    if isinstance(spec, (SpreadSpikeEventSpec, TickCountDroughtEventSpec, RangeExpansionEventSpec)):
        extra: Dict[str, Any] = {}
        if hasattr(spec, "price_source"):
            extra["price_source"] = str(spec.price_source)
        return _compile_window_metric_event(
            spec,
            request=request,
            required_tick_count=_required_tick_count_for_volume_spike(spec),
            extra=extra,
        )
    if isinstance(spec, (PriceTouchLevelEventSpec, PriceBreakLevelEventSpec, PriceEnterZoneEventSpec)):
        return _compile_price_level_event(spec, request=request)
    return {"error": f"Unsupported wait event type: {getattr(spec, 'type', type(spec).__name__)}"}


def _compile_account_event(spec: Any, *, request: WaitEventRequest) -> Dict[str, Any]:
    symbol = _resolved_value(spec, request, "symbol")
    side = _normalize_side(_resolved_value(spec, request, "side"))
    return {
        "type": str(spec.type),
        "symbol": str(symbol).upper() if symbol else None,
        "order_ticket": _resolved_value(spec, request, "order_ticket"),
        "position_ticket": _resolved_value(spec, request, "position_ticket"),
        "magic": _resolved_value(spec, request, "magic"),
        "side": side,
    }


def _compile_account_market_event(spec: Any, *, request: WaitEventRequest) -> Dict[str, Any]:
    compiled = _compile_account_event(spec, request=request)
    if "error" in compiled:
        return compiled
    if not compiled.get("symbol"):
        return {"error": f"{spec.type} events require symbol at the event or request level."}
    return compiled


def _compile_window_metric_event(
    spec: Any,
    *,
    request: WaitEventRequest,
    required_tick_count: int,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    symbol = _resolved_value(spec, request, "symbol")
    event_type = str(spec.type)
    if not symbol:
        return {"error": f"{event_type} events require symbol at the event or request level."}
    if (
        spec.threshold_mode in {"ratio_to_baseline", "zscore"}
        and str(spec.baseline_window.kind) != str(spec.window.kind)
    ):
        return {
            "error": (
                f"{event_type} baseline_window.kind must match window.kind when "
                "threshold_mode is ratio_to_baseline or zscore."
            )
        }
    if spec.threshold_mode in {"ratio_to_baseline", "zscore"} and float(spec.baseline_window.value) <= float(spec.window.value):
        return {
            "error": (
                f"{event_type} baseline_window must be larger than window when "
                "threshold_mode is ratio_to_baseline or zscore."
            )
        }
    payload: Dict[str, Any] = {
        "type": event_type,
        "symbol": str(symbol).upper(),
        "threshold_mode": spec.threshold_mode,
        "threshold_value": float(spec.threshold_value),
        "window": _window_payload(spec.window),
        "baseline_window": _window_payload(spec.baseline_window),
        "required_tick_count": int(required_tick_count),
        "required_history_seconds": _required_history_seconds(
            window=spec.window,
            baseline_window=spec.baseline_window,
            poll_interval_seconds=float(request.poll_interval_seconds),
            adaptive=spec.threshold_mode in {"ratio_to_baseline", "zscore"},
        ),
    }
    if hasattr(spec, "direction"):
        payload["direction"] = str(spec.direction)
    if hasattr(spec, "price_source"):
        payload["price_source"] = str(spec.price_source)
    if extra:
        payload.update(extra)
    return payload


def _compile_price_level_event(spec: Any, *, request: WaitEventRequest) -> Dict[str, Any]:
    symbol = _resolved_value(spec, request, "symbol")
    event_type = str(spec.type)
    if not symbol:
        return {"error": f"{event_type} events require symbol at the event or request level."}
    payload: Dict[str, Any] = {
        "type": event_type,
        "symbol": str(symbol).upper(),
        "price_source": str(spec.price_source),
        "required_tick_count": 2,
        "required_history_seconds": _MARKET_BOOTSTRAP_MIN_SECONDS,
    }
    if hasattr(spec, "direction"):
        payload["direction"] = str(spec.direction)
    if hasattr(spec, "tolerance"):
        payload["tolerance"] = float(spec.tolerance)
    if hasattr(spec, "level"):
        payload["level"] = float(spec.level)
    if hasattr(spec, "lower"):
        payload["lower"] = float(spec.lower)
    if hasattr(spec, "upper"):
        payload["upper"] = float(spec.upper)
    if hasattr(spec, "confirm_ticks"):
        payload["confirm_ticks"] = int(spec.confirm_ticks)
        payload["required_tick_count"] = max(2, int(spec.confirm_ticks) + 1)
    return payload


def _compile_boundary_event(
    spec: CandleCloseEventSpec,
    *,
    request: WaitEventRequest,
    started_at_utc: datetime,
) -> Dict[str, Any]:
    timeframe = str(_resolved_value(spec, request, "timeframe", default="H1")).upper().strip()
    buffer_seconds = float(
        spec.buffer_seconds if spec.buffer_seconds is not None else request.buffer_seconds
    )
    preview = _next_candle_wait_payload(
        timeframe,
        buffer_seconds=buffer_seconds,
        now_utc=started_at_utc,
    )
    boundary_at_utc = _normalize_utc_datetime(preview["next_candle_close_utc"])
    return {
        "type": spec.type,
        "timeframe": timeframe,
        "buffer_seconds": buffer_seconds,
        "preview": preview,
        "boundary_at_utc": boundary_at_utc,
        "boundary_at_epoch": boundary_at_utc.timestamp() + float(buffer_seconds),
    }


def _default_watch_specs(request: WaitEventRequest) -> List[Any]:
    specs: List[Any] = [
        OrderCreatedEventSpec(),
        OrderFilledEventSpec(),
        OrderCancelledEventSpec(),
        PositionOpenedEventSpec(),
        PositionClosedEventSpec(),
        TpHitEventSpec(),
        SlHitEventSpec(),
    ]
    if request.symbol:
        specs.append(PriceChangeEventSpec(symbol=request.symbol))
        specs.append(VolumeSpikeEventSpec(symbol=request.symbol))
    return specs


def _public_watch_spec_payload(spec: Any, *, request: WaitEventRequest) -> Dict[str, Any]:
    if hasattr(spec, "model_dump"):
        payload = spec.model_dump(mode="json")
    else:
        payload = dict(spec)
    payload["type"] = str(payload.get("type") or getattr(spec, "type", ""))
    for field_name in ("symbol", "order_ticket", "position_ticket", "magic", "side"):
        if payload.get(field_name) is None:
            resolved = getattr(request, field_name, None)
            if resolved is not None:
                payload[field_name] = resolved
    return {key: value for key, value in payload.items() if value is not None}


def _public_boundary_spec_payload(spec: Any, *, request: WaitEventRequest) -> Dict[str, Any]:
    if hasattr(spec, "model_dump"):
        payload = spec.model_dump(mode="json")
    else:
        payload = dict(spec)
    payload["type"] = str(payload.get("type") or getattr(spec, "type", ""))
    if payload.get("timeframe") is None and request.timeframe is not None:
        payload["timeframe"] = request.timeframe
    if payload.get("buffer_seconds") is None:
        payload["buffer_seconds"] = request.buffer_seconds
    return {key: value for key, value in payload.items() if value is not None}


def _run_candle_boundary_only(
    *,
    request: WaitEventRequest,
    boundary: Dict[str, Any],
    sleep_impl: Callable[[float], None],
    now_utc: datetime,
) -> Dict[str, Any]:
    preview = dict(boundary["preview"])
    max_wait_seconds = request.max_wait_seconds
    if max_wait_seconds is not None and float(preview["sleep_seconds"]) > float(max_wait_seconds):
        preview["success"] = True
        preview["status"] = "deferred_timeout_risk"
        preview["slept"] = False
        preview["slept_seconds"] = 0.0
        preview["remaining_seconds"] = float(preview["sleep_seconds"])
        preview["max_wait_seconds"] = float(max_wait_seconds)
        preview["warning"] = (
            "Skipping blocking wait because the remaining candle wait exceeds max_wait_seconds. "
            "Increase max_wait_seconds in clients that allow longer MCP tool timeouts."
        )
        preview["event"] = "candle_close"
        preview["boundary_event"] = {
            "type": "candle_close",
            "timeframe": boundary["timeframe"],
            "buffer_seconds": boundary["buffer_seconds"],
        }
        return preview

    payload = _sleep_until_next_candle(
        boundary["timeframe"],
        buffer_seconds=boundary["buffer_seconds"],
        sleep_impl=sleep_impl,
        now_utc=now_utc,
    )
    payload["event"] = "candle_close"
    payload["boundary_event"] = {
        "type": "candle_close",
        "timeframe": boundary["timeframe"],
        "buffer_seconds": boundary["buffer_seconds"],
    }
    payload["max_wait_seconds"] = (
        None if request.max_wait_seconds is None else float(request.max_wait_seconds)
    )
    payload["success"] = True
    return payload


def _build_baseline(gateway: Any, watch_for: List[Dict[str, Any]]) -> Dict[str, Any]:
    baseline: Dict[str, Any] = {}
    if any(item["type"] in _ORDER_STATE_EVENT_TYPES for item in watch_for):
        baseline["orders"] = _coerce_rows(gateway.orders_get())
    if any(item["type"] in _POSITION_STATE_EVENT_TYPES for item in watch_for):
        baseline["positions"] = _coerce_rows(gateway.positions_get())
    return baseline


def _watchers_need_current_state(watch_for: List[Dict[str, Any]]) -> bool:
    return any(item["type"] in (_ORDER_STATE_EVENT_TYPES | _POSITION_STATE_EVENT_TYPES) for item in watch_for)


def _watchers_need_history_deals(watch_for: List[Dict[str, Any]]) -> bool:
    return any(
        item["type"] in {"order_filled", "position_opened", "position_closed", "tp_hit", "sl_hit"}
        for item in watch_for
    )


def _watchers_need_history_orders(watch_for: List[Dict[str, Any]]) -> bool:
    return any(item["type"] == "order_cancelled" for item in watch_for)


def _build_account_history_state(
    *,
    gateway: Any,
    watch_for: List[Dict[str, Any]],
    started_at_utc: datetime,
) -> Dict[str, Any]:
    state: Dict[str, Any] = {}
    if _watchers_need_history_deals(watch_for):
        seeded = _seed_account_history_keys(
            fetch_impl=gateway.history_deals_get,
            started_at_utc=started_at_utc,
            row_kind="deal",
            label="deal history",
        )
        if isinstance(seeded, dict) and "error" in seeded:
            return seeded
        state["history_deals"] = {"seen_keys": seeded}
    if _watchers_need_history_orders(watch_for):
        seeded = _seed_account_history_keys(
            fetch_impl=gateway.history_orders_get,
            started_at_utc=started_at_utc,
            row_kind="order",
            label="order history",
        )
        if isinstance(seeded, dict) and "error" in seeded:
            return seeded
        state["history_orders"] = {"seen_keys": seeded}
    return state


def _seed_account_history_keys(
    *,
    fetch_impl: Any,
    started_at_utc: datetime,
    row_kind: str,
    label: str,
) -> set[tuple[Any, ...]] | Dict[str, Any]:
    seed_from_utc = started_at_utc - timedelta(seconds=_ACCOUNT_HISTORY_SEED_LOOKBACK_SECONDS)
    try:
        rows = fetch_impl(
            _to_server_naive_dt(seed_from_utc),
            _to_server_naive_dt(started_at_utc),
        )
    except Exception as exc:
        return {"error": f"Failed to fetch {label}: {exc}"}
    seen_keys: set[tuple[Any, ...]] = set()
    for row in _coerce_rows(rows):
        row_key = _account_history_row_key(row, row_kind=row_kind)
        if row_key is not None:
            seen_keys.add(row_key)
    return seen_keys


def _find_preexisting_match(
    *,
    watch_for: List[Dict[str, Any]],
    baseline: Dict[str, Any],
    market_state: Dict[str, Any],
    gateway: Any,
) -> Optional[Dict[str, Any]]:
    for spec in watch_for:
        if spec["type"] == "order_created":
            rows = baseline.get("orders") or _coerce_rows(gateway.orders_get())
            for row in rows:
                if _matches_account_filters(row, spec, gateway=gateway):
                    return _format_account_match(spec["type"], row, gateway=gateway)
        elif spec["type"] == "pending_near_fill":
            match = _evaluate_pending_near_fill(
                spec,
                baseline.get("orders", []),
                (market_state or {}).get(spec["symbol"]),
                gateway=gateway,
            )
            if match is not None:
                return match
        elif spec["type"] in {"position_opened", "position_closed"}:
            rows = baseline.get("positions") or _coerce_rows(gateway.positions_get())
            for row in rows:
                if _matches_account_filters(row, spec, gateway=gateway):
                    return _format_account_match(spec["type"], row, gateway=gateway)
        elif spec["type"] == "stop_threat":
            match = _evaluate_stop_threat(
                spec,
                baseline.get("positions", []),
                (market_state or {}).get(spec["symbol"]),
                gateway=gateway,
            )
            if match is not None:
                return match
        elif spec["type"] in _MARKET_EVENT_TYPES:
            match = _evaluate_market_event(
                spec,
                (market_state or {}).get(spec["symbol"]),
                snapshot={"baseline": baseline},
                gateway=gateway,
            )
            if match is not None:
                return match
    return None


def _collect_snapshot(
    *,
    gateway: Any,
    watch_for: List[Dict[str, Any]],
    baseline: Dict[str, Any],
    history_state: Dict[str, Any],
    market_state: Dict[str, Any],
    started_at_utc: datetime,
    observed_at_utc: datetime,
) -> Dict[str, Any]:
    snapshot: Dict[str, Any] = {
        "observed_at_utc": observed_at_utc,
        "baseline": baseline,
    }

    if any(item["type"] in _ORDER_STATE_EVENT_TYPES for item in watch_for):
        try:
            snapshot["orders"] = _coerce_rows(gateway.orders_get())
        except Exception as exc:
            return {"error": f"Failed to fetch open orders: {exc}"}

    if any(item["type"] in _POSITION_STATE_EVENT_TYPES for item in watch_for):
        try:
            snapshot["positions"] = _coerce_rows(gateway.positions_get())
        except Exception as exc:
            return {"error": f"Failed to fetch open positions: {exc}"}

    if _watchers_need_history_deals(watch_for):
        rows = _collect_new_account_history_rows(
            fetch_impl=gateway.history_deals_get,
            started_at_utc=started_at_utc,
            observed_at_utc=observed_at_utc,
            state=history_state.setdefault("history_deals", {}),
            row_kind="deal",
            label="deal history",
        )
        if isinstance(rows, dict) and "error" in rows:
            return rows
        snapshot["history_deals"] = rows

    if _watchers_need_history_orders(watch_for):
        rows = _collect_new_account_history_rows(
            fetch_impl=gateway.history_orders_get,
            started_at_utc=started_at_utc,
            observed_at_utc=observed_at_utc,
            state=history_state.setdefault("history_orders", {}),
            row_kind="order",
            label="order history",
        )
        if isinstance(rows, dict) and "error" in rows:
            return rows
        snapshot["history_orders"] = rows

    market_specs = [item for item in watch_for if item["type"] in _MARKET_EVENT_TYPES]
    if market_specs:
        refreshed = _refresh_market_state(
            market_state=market_state,
            gateway=gateway,
            watch_for=market_specs,
            observed_at_utc=observed_at_utc,
        )
        if isinstance(refreshed, dict) and "error" in refreshed:
            return refreshed
        market_data: Dict[str, Any] = {}
        for symbol in _market_symbols(market_specs):
            state = refreshed.get(symbol) or {}
            market_data[symbol] = state
        snapshot["market_data"] = market_data

    return snapshot


def _collect_new_account_history_rows(
    *,
    fetch_impl: Any,
    started_at_utc: datetime,
    observed_at_utc: datetime,
    state: Dict[str, Any],
    row_kind: str,
    label: str,
) -> List[Any] | Dict[str, Any]:
    try:
        rows = _coerce_rows(
            fetch_impl(
                _to_server_naive_dt(started_at_utc),
                _to_server_naive_dt(observed_at_utc),
            )
        )
    except Exception as exc:
        return {"error": f"Failed to fetch {label}: {exc}"}

    seen_keys = state.setdefault("seen_keys", set())
    started_at_millis = _datetime_epoch_millis(started_at_utc)
    fresh_rows: List[Any] = []
    for row in rows:
        row_key = _account_history_row_key(row, row_kind=row_kind)
        if row_key is not None and row_key in seen_keys:
            continue
        row_time_millis = _row_event_time_millis(row)
        if row_time_millis is not None and row_time_millis < started_at_millis:
            if row_key is not None:
                seen_keys.add(row_key)
            continue
        if row_key is not None:
            seen_keys.add(row_key)
        fresh_rows.append(row)
    return fresh_rows


def _build_market_state(
    *,
    gateway: Any,
    watch_for: List[Dict[str, Any]],
    observed_at_utc: datetime,
    poll_interval_seconds: float,
) -> Dict[str, Any]:
    market_specs = [item for item in watch_for if item["type"] in _MARKET_EVENT_TYPES]
    if not market_specs:
        return {}

    state: Dict[str, Any] = {}
    for symbol in _market_symbols(market_specs):
        symbol_specs = [item for item in market_specs if item["symbol"] == symbol]
        bootstrap = _bootstrap_market_ticks(
            gateway=gateway,
            symbol=symbol,
            specs=symbol_specs,
            observed_at_utc=observed_at_utc,
            poll_interval_seconds=poll_interval_seconds,
        )
        if isinstance(bootstrap, dict) and "error" in bootstrap:
            return bootstrap
        state[symbol] = bootstrap
    return state


def _refresh_market_state(
    *,
    market_state: Dict[str, Any],
    gateway: Any,
    watch_for: List[Dict[str, Any]],
    observed_at_utc: datetime,
) -> Dict[str, Any]:
    for symbol in _market_symbols(watch_for):
        state = market_state.get(symbol)
        if state is None:
            continue
        last_epoch = float(state.get("last_epoch") or observed_at_utc.timestamp())
        from_dt = datetime.fromtimestamp(max(0.0, last_epoch - 1e-6), tz=timezone.utc)
        ticks_or_error = _fetch_market_ticks_range(
            gateway=gateway,
            symbol=symbol,
            from_dt_utc=from_dt,
            to_dt_utc=observed_at_utc,
        )
        if isinstance(ticks_or_error, dict) and "error" in ticks_or_error:
            return ticks_or_error
        merged = _merge_market_ticks(state.get("ticks", []), ticks_or_error)
        trimmed = _trim_market_ticks(
            ticks=merged,
            specs=[item for item in watch_for if item["symbol"] == symbol],
            observed_at_utc=observed_at_utc,
        )
        state["ticks"] = trimmed
        state["last_epoch"] = float(trimmed[-1]["epoch"]) if trimmed else last_epoch
    return market_state


def _evaluate_watch_events(
    *,
    watch_for: List[Dict[str, Any]],
    snapshot: Dict[str, Any],
    gateway: Any,
) -> Optional[Dict[str, Any]]:
    for spec in watch_for:
        event_type = spec["type"]
        if event_type == "order_created":
            current_orders = snapshot.get("orders", [])
            baseline_orders = snapshot.get("baseline", {}).get("orders", [])
            baseline_tickets = {
                _row_int(row, "ticket")
                for row in baseline_orders
                if _row_int(row, "ticket") is not None
            }
            for row in current_orders:
                ticket = _row_int(row, "ticket")
                if ticket in baseline_tickets:
                    continue
                if _matches_account_filters(row, spec, gateway=gateway):
                    return _format_account_match(event_type, row, gateway=gateway)
        elif event_type == "order_filled":
            for row in snapshot.get("history_deals", []):
                if not _is_deal_entry_in(row, gateway=gateway):
                    continue
                if _matches_account_filters(row, spec, gateway=gateway):
                    return _format_account_match(event_type, row, gateway=gateway)
        elif event_type == "order_cancelled":
            for row in snapshot.get("history_orders", []):
                if not _is_order_cancelled(row, gateway=gateway):
                    continue
                if _matches_account_filters(row, spec, gateway=gateway):
                    return _format_account_match(event_type, row, gateway=gateway)
        elif event_type == "position_opened":
            for row in snapshot.get("history_deals", []):
                if not _is_deal_entry_in(row, gateway=gateway):
                    continue
                if _matches_account_filters(row, spec, gateway=gateway):
                    return _format_account_match(event_type, row, gateway=gateway)
            current_positions = snapshot.get("positions", [])
            baseline_positions = snapshot.get("baseline", {}).get("positions", [])
            baseline_tickets = {
                _row_int(row, "ticket")
                for row in baseline_positions
                if _row_int(row, "ticket") is not None
            }
            for row in current_positions:
                ticket = _row_int(row, "ticket")
                if ticket in baseline_tickets:
                    continue
                if _matches_account_filters(row, spec, gateway=gateway):
                    return _format_account_match(event_type, row, gateway=gateway)
        elif event_type == "position_closed":
            for row in snapshot.get("history_deals", []):
                if not _is_deal_entry_out(row, gateway=gateway):
                    continue
                if _matches_account_filters(row, spec, gateway=gateway):
                    return _format_account_match(event_type, row, gateway=gateway)
            current_positions = snapshot.get("positions", [])
            baseline_positions = snapshot.get("baseline", {}).get("positions", [])
            current_tickets = {
                _row_int(row, "ticket")
                for row in current_positions
                if _row_int(row, "ticket") is not None
            }
            for row in baseline_positions:
                ticket = _row_int(row, "ticket")
                if ticket is not None and ticket in current_tickets:
                    continue
                if _matches_account_filters(row, spec, gateway=gateway):
                    return _format_inferred_position_closed(
                        row,
                        gateway=gateway,
                        observed_at_utc=snapshot.get("observed_at_utc", datetime.now(timezone.utc)),
                    )
        elif event_type == "tp_hit":
            for row in snapshot.get("history_deals", []):
                if not _is_deal_entry_out(row, gateway=gateway):
                    continue
                if not _is_exit_trigger(row, gateway=gateway, trigger="tp"):
                    continue
                if _matches_account_filters(row, spec, gateway=gateway):
                    return _format_account_match(event_type, row, gateway=gateway)
        elif event_type == "sl_hit":
            for row in snapshot.get("history_deals", []):
                if not _is_deal_entry_out(row, gateway=gateway):
                    continue
                if not _is_exit_trigger(row, gateway=gateway, trigger="sl"):
                    continue
                if _matches_account_filters(row, spec, gateway=gateway):
                    return _format_account_match(event_type, row, gateway=gateway)
        elif event_type in _MARKET_EVENT_TYPES:
            market_data = snapshot.get("market_data", {}).get(spec["symbol"])
            match = _evaluate_market_event(spec, market_data, snapshot=snapshot, gateway=gateway)
            if match is not None:
                return match
    return None


def _evaluate_boundaries(
    boundaries: List[Dict[str, Any]],
    *,
    observed_at_utc: datetime,
) -> Optional[Dict[str, Any]]:
    current_epoch = observed_at_utc.timestamp()
    for boundary in boundaries:
        if current_epoch + 1e-9 >= float(boundary["boundary_at_epoch"]):
            return {
                "type": "candle_close",
                "timeframe": boundary["timeframe"],
                "buffer_seconds": boundary["buffer_seconds"],
                "next_candle_close_utc": boundary["preview"]["next_candle_close_utc"],
                "next_candle_close_server": boundary["preview"]["next_candle_close_server"],
                "server_timezone": boundary["preview"]["server_timezone"],
            }
    return None


def _next_poll_sleep_seconds(
    *,
    poll_interval_seconds: float,
    max_wait_seconds: Optional[float],
    elapsed_seconds: float,
    boundaries: List[Dict[str, Any]],
    observed_at_utc: datetime,
) -> float:
    sleep_seconds = max(0.0, float(poll_interval_seconds))
    if max_wait_seconds is not None:
        sleep_seconds = min(
            sleep_seconds,
            max(0.0, float(max_wait_seconds) - float(elapsed_seconds)),
        )
    current_epoch = observed_at_utc.timestamp()
    for boundary in boundaries:
        boundary_remaining = float(boundary["boundary_at_epoch"]) - current_epoch
        if boundary_remaining > 0.0:
            sleep_seconds = min(sleep_seconds, boundary_remaining)
    return max(0.0, sleep_seconds)


def _evaluate_market_event(
    spec: Dict[str, Any],
    market_data: Any,
    *,
    snapshot: Dict[str, Any],
    gateway: Any,
) -> Optional[Dict[str, Any]]:
    event_type = str(spec.get("type") or "")
    if event_type == "price_change":
        return _evaluate_price_change(spec, market_data)
    if event_type in {"volume_spike", "tick_count_spike"}:
        return _evaluate_volume_spike(spec, market_data)
    if event_type == "spread_spike":
        return _evaluate_spread_spike(spec, market_data)
    if event_type == "tick_count_drought":
        return _evaluate_tick_count_drought(spec, market_data)
    if event_type == "range_expansion":
        return _evaluate_range_expansion(spec, market_data)
    if event_type == "price_touch_level":
        return _evaluate_price_touch_level(spec, market_data)
    if event_type == "price_break_level":
        return _evaluate_price_break_level(spec, market_data)
    if event_type == "price_enter_zone":
        return _evaluate_price_enter_zone(spec, market_data)
    if event_type == "pending_near_fill":
        return _evaluate_pending_near_fill(
            spec,
            snapshot.get("orders", []),
            market_data,
            gateway=gateway,
        )
    if event_type == "stop_threat":
        return _evaluate_stop_threat(
            spec,
            snapshot.get("positions", []),
            market_data,
            gateway=gateway,
        )
    return None


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


def _evaluate_price_touch_level(spec: Dict[str, Any], market_data: Any) -> Optional[Dict[str, Any]]:
    prices = _market_price_points(list((market_data or {}).get("ticks", [])), source=str(spec.get("price_source") or "auto"))
    if len(prices) < 2:
        return None
    previous_price = float(prices[-2][1])
    current_price = float(prices[-1][1])
    level = float(spec["level"])
    tolerance = float(spec.get("tolerance") or 0.0)
    lower = level - tolerance
    upper = level + tolerance
    upward_touch = previous_price < lower and current_price >= lower
    downward_touch = previous_price > upper and current_price <= upper
    direction = str(spec.get("direction") or "either")
    if direction == "up" and not upward_touch:
        return None
    if direction == "down" and not downward_touch:
        return None
    if direction == "either" and not (upward_touch or downward_touch):
        return None
    if not _price_within_band(current_price, lower=lower, upper=upper):
        return None
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


def _evaluate_price_break_level(spec: Dict[str, Any], market_data: Any) -> Optional[Dict[str, Any]]:
    prices = _market_price_points(list((market_data or {}).get("ticks", [])), source=str(spec.get("price_source") or "auto"))
    confirm_ticks = max(1, int(spec.get("confirm_ticks") or 1))
    if len(prices) < confirm_ticks + 1:
        return None
    level = float(spec["level"])
    tolerance = float(spec.get("tolerance") or 0.0)
    upper = level + tolerance
    lower = level - tolerance
    previous_price = float(prices[-(confirm_ticks + 1)][1])
    confirmed_prices = [float(price) for _, price in prices[-confirm_ticks:]]
    breakout_up = previous_price < lower and all(price >= upper for price in confirmed_prices)
    breakout_down = previous_price > upper and all(price <= lower for price in confirmed_prices)
    direction = str(spec.get("direction") or "either")
    if direction == "up" and not breakout_up:
        return None
    if direction == "down" and not breakout_down:
        return None
    if direction == "either" and not (breakout_up or breakout_down):
        return None
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


def _evaluate_price_enter_zone(spec: Dict[str, Any], market_data: Any) -> Optional[Dict[str, Any]]:
    prices = _market_price_points(list((market_data or {}).get("ticks", [])), source=str(spec.get("price_source") or "auto"))
    if len(prices) < 2:
        return None
    previous_price = float(prices[-2][1])
    current_price = float(prices[-1][1])
    lower = float(spec["lower"])
    upper = float(spec["upper"])
    if not _price_within_band(current_price, lower=lower, upper=upper):
        return None
    if _price_within_band(previous_price, lower=lower, upper=upper):
        return None
    direction = str(spec.get("direction") or "either")
    enter_up = previous_price < lower
    enter_down = previous_price > upper
    if direction == "up" and not enter_up:
        return None
    if direction == "down" and not enter_down:
        return None
    if direction == "either" and not (enter_up or enter_down):
        return None
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


def _market_symbols(watch_for: List[Dict[str, Any]]) -> List[str]:
    seen: set[str] = set()
    symbols: List[str] = []
    for spec in watch_for:
        symbol = str(spec.get("symbol") or "").upper().strip()
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        symbols.append(symbol)
    return symbols


def _bootstrap_market_ticks(
    *,
    gateway: Any,
    symbol: str,
    specs: List[Dict[str, Any]],
    observed_at_utc: datetime,
    poll_interval_seconds: float,
) -> Dict[str, Any] | Dict[str, str]:
    required_tick_count = max(int(spec.get("required_tick_count") or 0) for spec in specs)
    required_history_seconds = max(float(spec.get("required_history_seconds") or 0.0) for spec in specs)
    duration_seconds = _bootstrap_duration_seconds(
        required_tick_count=required_tick_count,
        required_history_seconds=required_history_seconds,
        poll_interval_seconds=poll_interval_seconds,
    )
    ticks: List[Dict[str, Any]] = []
    while True:
        from_dt = observed_at_utc - timedelta(seconds=duration_seconds)
        ticks_or_error = _fetch_market_ticks_range(
            gateway=gateway,
            symbol=symbol,
            from_dt_utc=from_dt,
            to_dt_utc=observed_at_utc,
        )
        if isinstance(ticks_or_error, dict) and "error" in ticks_or_error:
            return ticks_or_error
        ticks = ticks_or_error
        if required_tick_count <= 0 or len(ticks) >= required_tick_count or duration_seconds >= _MARKET_BOOTSTRAP_MAX_SECONDS:
            break
        duration_seconds = min(duration_seconds * 2.0, _MARKET_BOOTSTRAP_MAX_SECONDS)

    trimmed = _trim_market_ticks(
        ticks=ticks,
        specs=specs,
        observed_at_utc=observed_at_utc,
    )
    last_epoch = float(trimmed[-1]["epoch"]) if trimmed else observed_at_utc.timestamp()
    return {"ticks": trimmed, "last_epoch": last_epoch}


def _fetch_market_ticks_range(
    *,
    gateway: Any,
    symbol: str,
    from_dt_utc: datetime,
    to_dt_utc: datetime,
) -> List[Dict[str, Any]] | Dict[str, Any]:
    try:
        if hasattr(gateway, "symbol_select"):
            try:
                gateway.symbol_select(symbol, True)
            except Exception:
                pass
        flags = getattr(gateway, "COPY_TICKS_ALL", 0)
        rows = gateway.copy_ticks_range(
            symbol,
            _to_server_naive_dt(from_dt_utc),
            _to_server_naive_dt(to_dt_utc),
            flags,
        )
    except Exception as exc:
        return {"error": f"Failed to fetch tick data for {symbol}: {exc}"}
    return _normalize_tick_rows(rows)


def _build_wait_result(
    *,
    request: WaitEventRequest,
    status: str,
    started_at_utc: datetime,
    observed_at_utc: datetime,
    polls: int,
    matched_event: Optional[Dict[str, Any]],
    boundary_event: Optional[Dict[str, Any]],
    watch_for_payload: List[Dict[str, Any]],
    end_on_payload: List[Dict[str, Any]],
    watch_for_inferred: bool,
    end_on_inferred: bool,
) -> Dict[str, Any]:
    elapsed_seconds = max(0.0, (observed_at_utc - started_at_utc).total_seconds())
    return {
        "success": True,
        "status": status,
        "matched": status in {"matched", "already_satisfied"},
        "event": matched_event["type"] if matched_event is not None else None,
        "matched_event": matched_event,
        "boundary_event": boundary_event,
        "started_at_utc": started_at_utc.isoformat(),
        "observed_at_utc": observed_at_utc.isoformat(),
        "elapsed_seconds": round(elapsed_seconds, 6),
        "polls": int(polls),
        "poll_interval_seconds": float(request.poll_interval_seconds),
        "max_wait_seconds": None
        if request.max_wait_seconds is None
        else float(request.max_wait_seconds),
        "criteria": {
            "watch_for": list(watch_for_payload),
            "watch_for_inferred": bool(watch_for_inferred),
            "end_on": list(end_on_payload),
            "end_on_inferred": bool(end_on_inferred),
            "accept_preexisting": bool(request.accept_preexisting),
        },
    }

def _matches_account_filters(row: Any, spec: Dict[str, Any], *, gateway: Any) -> bool:
    symbol = spec.get("symbol")
    if symbol:
        row_symbol = str(_row_value(row, "symbol") or "").upper()
        if row_symbol != str(symbol).upper():
            return False

    magic = spec.get("magic")
    if magic is not None:
        row_magic = _row_int(row, "magic")
        if row_magic != int(magic):
            return False

    side = spec.get("side")
    if side:
        row_side = _row_side(row, gateway=gateway)
        if row_side != side:
            return False

    order_ticket = spec.get("order_ticket")
    if order_ticket is not None:
        row_order_ticket = _first_int(
            _row_int(row, "order"),
            _row_int(row, "ticket"),
            _row_int(row, "order_ticket"),
        )
        if row_order_ticket != int(order_ticket):
            return False

    position_ticket = spec.get("position_ticket")
    if position_ticket is not None:
        row_position_ticket = _first_int(
            _row_int(row, "position_id"),
            _row_int(row, "position"),
            _row_int(row, "position_by_id"),
            _row_int(row, "ticket"),
        )
        if row_position_ticket != int(position_ticket):
            return False

    return True


def _format_account_match(event_type: str, row: Any, *, gateway: Any) -> Dict[str, Any]:
    return {
        "type": event_type,
        "observed": {
            "ticket": _row_int(row, "ticket"),
            "order_ticket": _first_int(
                _row_int(row, "order"),
                _row_int(row, "order_ticket"),
                _row_int(row, "ticket"),
            ),
            "position_ticket": _first_int(
                _row_int(row, "position_id"),
                _row_int(row, "position"),
                _row_int(row, "position_by_id"),
                _row_int(row, "ticket"),
            ),
            "symbol": _row_value(row, "symbol"),
            "magic": _row_int(row, "magic"),
            "side": _row_side(row, gateway=gateway),
            "reason": _row_value(row, "reason"),
            "comment": _row_value(row, "comment"),
            "time_utc": _row_time_iso(row),
        },
    }


def _format_inferred_position_closed(row: Any, *, gateway: Any, observed_at_utc: datetime) -> Dict[str, Any]:
    return {
        "type": "position_closed",
        "observed": {
            "ticket": None,
            "order_ticket": _first_int(
                _row_int(row, "order"),
                _row_int(row, "order_ticket"),
            ),
            "position_ticket": _first_int(
                _row_int(row, "position_id"),
                _row_int(row, "position"),
                _row_int(row, "position_by_id"),
                _row_int(row, "ticket"),
            ),
            "symbol": _row_value(row, "symbol"),
            "magic": _row_int(row, "magic"),
            "side": _row_side(row, gateway=gateway),
            "reason": None,
            "comment": None,
            "time_utc": _normalize_utc_datetime(observed_at_utc).isoformat(),
            "inferred": True,
            "source": "position_disappeared",
        },
    }


def _matches_exit_trigger_text(text: str, *, trigger: str) -> bool:
    text_norm = str(text or "").strip().lower()
    if not text_norm:
        return False
    if trigger == "tp":
        phrases = ("take profit", "tp hit", "hit tp", "closed by tp", "tp")
    elif trigger == "sl":
        phrases = ("stop loss", "sl hit", "hit sl", "closed by sl", "sl")
    else:
        return False
    for phrase in phrases:
        if " " in phrase:
            if re.search(rf"\b{re.escape(phrase)}\b", text_norm):
                return True
            continue
        if text_norm == phrase:
            return True
        if re.search(rf"\b(?:hit|closed by)\s+{re.escape(phrase)}\b", text_norm):
            return True
    return False


def _is_deal_entry_in(row: Any, *, gateway: Any) -> bool:
    return _row_enum_matches(
        row,
        "entry",
        text_patterns=("deal_entry_in", "entry_in", " in"),
        numeric_constants=("DEAL_ENTRY_IN", "ENTRY_IN"),
        gateway=gateway,
    )


def _is_deal_entry_out(row: Any, *, gateway: Any) -> bool:
    return _row_enum_matches(
        row,
        "entry",
        text_patterns=("deal_entry_out", "deal_entry_out_by", "entry_out", "entry_out_by", " out"),
        numeric_constants=("DEAL_ENTRY_OUT", "DEAL_ENTRY_OUT_BY", "DEAL_ENTRY_INOUT", "ENTRY_OUT"),
        gateway=gateway,
    )


def _is_order_cancelled(row: Any, *, gateway: Any) -> bool:
    return _row_enum_matches(
        row,
        "state",
        text_patterns=("canceled", "cancelled"),
        numeric_constants=("ORDER_STATE_CANCELED", "ORDER_STATE_CANCELLED"),
        gateway=gateway,
    )


def _is_exit_trigger(row: Any, *, gateway: Any, trigger: str) -> bool:
    trigger_txt = str(trigger or "").strip().lower()
    comment = str(_row_value(row, "comment") or "").lower()
    reason = str(_row_value(row, "reason") or "").lower()
    if trigger_txt == "tp":
        if _matches_exit_trigger_text(comment, trigger="tp") or _matches_exit_trigger_text(reason, trigger="tp"):
            return True
        return _row_enum_matches(
            row,
            "reason",
            text_patterns=("deal_reason_tp", "take profit", "tp"),
            numeric_constants=("DEAL_REASON_TP",),
            gateway=gateway,
        )
    if trigger_txt == "sl":
        if _matches_exit_trigger_text(comment, trigger="sl") or _matches_exit_trigger_text(reason, trigger="sl"):
            return True
        return _row_enum_matches(
            row,
            "reason",
            text_patterns=("deal_reason_sl", "stop loss", "sl"),
            numeric_constants=("DEAL_REASON_SL",),
            gateway=gateway,
        )
    return False


def _row_enum_matches(
    row: Any,
    column: str,
    *,
    text_patterns: tuple[str, ...],
    numeric_constants: tuple[str, ...],
    gateway: Any,
) -> bool:
    value = _row_value(row, column)
    text = str(value or "").strip().lower()
    if text:
        for pattern in text_patterns:
            if pattern.strip() and pattern.strip() in text:
                return True
    try:
        numeric = int(value)
    except Exception:
        return False
    for constant_name in numeric_constants:
        constant_value = getattr(gateway, constant_name, None)
        if constant_value is None:
            continue
        try:
            if int(constant_value) == numeric:
                return True
        except Exception:
            continue
    return False


def _row_side(row: Any, *, gateway: Any) -> Optional[str]:
    candidates = (
        _row_value(row, "type"),
        _row_value(row, "order_type"),
        _row_value(row, "position_type"),
    )
    buy_values = {
        int(value)
        for value in (
            getattr(gateway, "POSITION_TYPE_BUY", None),
            getattr(gateway, "ORDER_TYPE_BUY", None),
            getattr(gateway, "ORDER_TYPE_BUY_LIMIT", None),
            getattr(gateway, "ORDER_TYPE_BUY_STOP", None),
            getattr(gateway, "ORDER_TYPE_BUY_STOP_LIMIT", None),
            getattr(gateway, "DEAL_TYPE_BUY", None),
        )
        if value is not None
    }
    sell_values = {
        int(value)
        for value in (
            getattr(gateway, "POSITION_TYPE_SELL", None),
            getattr(gateway, "ORDER_TYPE_SELL", None),
            getattr(gateway, "ORDER_TYPE_SELL_LIMIT", None),
            getattr(gateway, "ORDER_TYPE_SELL_STOP", None),
            getattr(gateway, "ORDER_TYPE_SELL_STOP_LIMIT", None),
            getattr(gateway, "DEAL_TYPE_SELL", None),
        )
        if value is not None
    }
    for value in candidates:
        text = str(value or "").strip().lower()
        if "buy" in text:
            return "buy"
        if "sell" in text:
            return "sell"
        try:
            numeric = int(value)
        except Exception:
            continue
        if numeric in buy_values or numeric in {0, 2, 4, 6}:
            return "buy"
        if numeric in sell_values or numeric in {1, 3, 5, 7}:
            return "sell"
    return None


def _required_tick_count_for_price_change(spec: PriceChangeEventSpec) -> int:
    if str(spec.window.kind) != "ticks":
        return 0
    current_points = max(2, int(math.ceil(float(spec.window.value))) + 1)
    if spec.threshold_mode not in {"ratio_to_baseline", "zscore"}:
        return current_points
    baseline_points = max(0, int(math.ceil(float(spec.baseline_window.value))))
    return current_points + baseline_points


def _required_tick_count_for_volume_spike(
    spec: VolumeSpikeEventSpec
    | TickCountSpikeEventSpec
    | SpreadSpikeEventSpec
    | TickCountDroughtEventSpec
    | RangeExpansionEventSpec
) -> int:
    if str(spec.window.kind) != "ticks":
        return 0
    current_points = max(1, int(math.ceil(float(spec.window.value))))
    if spec.threshold_mode not in {"ratio_to_baseline", "zscore"}:
        return current_points
    baseline_points = max(0, int(math.ceil(float(spec.baseline_window.value))))
    return current_points + baseline_points


def _required_history_seconds(
    *,
    window: WaitEventWindow,
    baseline_window: WaitEventWindow,
    poll_interval_seconds: float,
    adaptive: bool,
) -> float:
    total = 0.0
    if str(window.kind) == "minutes":
        total += float(window.value) * 60.0
    if adaptive and str(baseline_window.kind) == "minutes":
        total += float(baseline_window.value) * 60.0
    if total > 0.0:
        return total
    tick_count = float(window.value)
    if adaptive:
        tick_count += float(baseline_window.value)
    estimated = max(float(poll_interval_seconds), _MARKET_ESTIMATED_SECONDS_PER_TICK) * tick_count
    return max(_MARKET_BOOTSTRAP_MIN_SECONDS, estimated)


def _bootstrap_duration_seconds(
    *,
    required_tick_count: int,
    required_history_seconds: float,
    poll_interval_seconds: float,
) -> float:
    duration = max(_MARKET_BOOTSTRAP_MIN_SECONDS, float(required_history_seconds))
    if required_tick_count > 0:
        duration = max(
            duration,
            float(required_tick_count)
            * max(float(poll_interval_seconds), _MARKET_ESTIMATED_SECONDS_PER_TICK),
        )
    return min(duration, _MARKET_BOOTSTRAP_MAX_SECONDS)


def _normalize_tick_rows(rows: Any) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for row in _coerce_rows(rows):
        epoch = _tick_epoch(row)
        if epoch is None:
            continue
        tick = {
            "epoch": float(epoch),
            "time_msc": _tick_time_msc(row, fallback_epoch=float(epoch)),
            "bid": _tick_float(row, "bid"),
            "ask": _tick_float(row, "ask"),
            "last": _tick_float(row, "last"),
            "volume": _tick_float(row, "volume"),
            "volume_real": _tick_float(row, "volume_real"),
            "flags": _tick_int(row, "flags") or 0,
        }
        tick["key"] = (
            int(tick["time_msc"]),
            _tick_key_component(tick["bid"]),
            _tick_key_component(tick["ask"]),
            _tick_key_component(tick["last"]),
            _tick_key_component(tick["volume"]),
            _tick_key_component(tick["volume_real"]),
            int(tick["flags"]),
        )
        normalized.append(tick)
    normalized.sort(key=lambda item: (int(item["time_msc"]), float(item["epoch"])))
    return normalized


def _merge_market_ticks(existing: List[Dict[str, Any]], new_ticks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not existing:
        return list(new_ticks)
    out = list(existing)
    seen = {tuple(item["key"]) for item in existing}
    for tick in new_ticks:
        key = tuple(tick["key"])
        if key in seen:
            continue
        out.append(tick)
        seen.add(key)
    return out


def _trim_market_ticks(
    *,
    ticks: List[Dict[str, Any]],
    specs: List[Dict[str, Any]],
    observed_at_utc: datetime,
) -> List[Dict[str, Any]]:
    if not ticks:
        return []
    keep_seconds = max(float(spec.get("required_history_seconds") or 0.0) for spec in specs)
    keep_ticks = max(int(spec.get("required_tick_count") or 0) for spec in specs) + _MARKET_BUFFER_EXTRA_TICKS
    start_idx = 0
    if keep_seconds > 0.0:
        cutoff = observed_at_utc.timestamp() - keep_seconds - max(1.0, _MARKET_ESTIMATED_SECONDS_PER_TICK)
        start_idx = len(ticks)
        for idx, tick in enumerate(ticks):
            if float(tick["epoch"]) >= cutoff:
                start_idx = idx
                break
    if keep_ticks > 0:
        start_idx = min(start_idx, max(0, len(ticks) - keep_ticks))
    return ticks[start_idx:]


def _market_price_points(ticks: List[Dict[str, Any]], *, source: str) -> List[tuple[float, float]]:
    points: List[tuple[float, float]] = []
    for tick in ticks:
        price = _tick_price(tick, source=source)
        if price is None:
            continue
        points.append((float(tick["epoch"]), float(price)))
    return points


def _current_price_change(spec: Dict[str, Any], prices: List[tuple[float, float]]) -> Optional[float]:
    if not prices:
        return None
    if spec["window"]["kind"] == "ticks":
        window_ticks = max(1, int(math.ceil(float(spec["window"]["value"]))))
        if len(prices) <= window_ticks:
            return None
        return _pct_change(prices[-(window_ticks + 1)][1], prices[-1][1])
    window_seconds = float(spec["window"]["value"]) * 60.0
    end_epoch = prices[-1][0]
    start_epoch = end_epoch - window_seconds
    window_points = [(epoch, price) for epoch, price in prices if epoch >= start_epoch]
    if len(window_points) < 2:
        return None
    return _pct_change(window_points[0][1], window_points[-1][1])


def _price_change_baseline_samples(
    spec: Dict[str, Any],
    prices: List[tuple[float, float]],
) -> List[float]:
    if spec["window"]["kind"] == "ticks":
        return _tick_price_change_baseline_samples(spec, prices)
    return _duration_price_change_baseline_samples(spec, prices)


def _tick_price_change_baseline_samples(
    spec: Dict[str, Any],
    prices: List[tuple[float, float]],
) -> List[float]:
    window_ticks = max(1, int(math.ceil(float(spec["window"]["value"]))))
    baseline_ticks = max(1, int(math.ceil(float(spec["baseline_window"]["value"]))))
    end_idx = len(prices) - window_ticks - 1
    start_idx = max(window_ticks, end_idx - baseline_ticks + 1)
    samples: List[float] = []
    for idx in range(start_idx, end_idx + 1):
        change = _pct_change(prices[idx - window_ticks][1], prices[idx][1])
        if change is None:
            continue
        samples.append(abs(change))
    return samples


def _duration_price_change_baseline_samples(
    spec: Dict[str, Any],
    prices: List[tuple[float, float]],
) -> List[float]:
    window_seconds = float(spec["window"]["value"]) * 60.0
    baseline_seconds = float(spec["baseline_window"]["value"]) * 60.0
    latest_epoch = prices[-1][0]
    current_start = latest_epoch - window_seconds
    baseline_start = current_start - baseline_seconds
    sample_count = max(1, int(math.floor(baseline_seconds / max(window_seconds, 1.0))))
    samples: List[float] = []
    for sample_idx in range(sample_count):
        window_start = baseline_start + sample_idx * window_seconds
        window_end = min(window_start + window_seconds, current_start)
        if window_end <= window_start:
            continue
        window_points = [(epoch, price) for epoch, price in prices if window_start <= epoch <= window_end]
        if len(window_points) < 2:
            continue
        change = _pct_change(window_points[0][1], window_points[-1][1])
        if change is None:
            continue
        samples.append(abs(change))
    return samples


def _resolve_market_volume_source(
    ticks: List[Dict[str, Any]],
    *,
    preferred: str,
    window_kind: str,
) -> str:
    if preferred != "auto":
        return str(preferred)
    has_real = any(_finite_number(tick.get("volume_real")) not in (None, 0.0) for tick in ticks)
    if has_real:
        return "volume_real"
    has_volume = any(_finite_number(tick.get("volume")) not in (None, 0.0) for tick in ticks)
    if has_volume:
        return "volume"
    if window_kind == "minutes":
        return "tick_count"
    return "volume"


def _current_volume_metric(
    spec: Dict[str, Any],
    ticks: List[Dict[str, Any]],
    *,
    source: str,
) -> Optional[float]:
    if not ticks:
        return None
    if spec["window"]["kind"] == "ticks":
        window_ticks = max(1, int(math.ceil(float(spec["window"]["value"]))))
        if len(ticks) < window_ticks:
            return None
        return _volume_metric_for_ticks(ticks[-window_ticks:], source=source)
    window_seconds = float(spec["window"]["value"]) * 60.0
    end_epoch = ticks[-1]["epoch"]
    start_epoch = end_epoch - window_seconds
    window_ticks_rows = [tick for tick in ticks if float(tick["epoch"]) >= start_epoch]
    if not window_ticks_rows:
        return None
    return _volume_metric_for_ticks(window_ticks_rows, source=source)


def _volume_baseline_samples(
    spec: Dict[str, Any],
    ticks: List[Dict[str, Any]],
    *,
    source: str,
) -> List[float]:
    if not ticks:
        return []
    if spec["window"]["kind"] == "ticks":
        window_ticks = max(1, int(math.ceil(float(spec["window"]["value"]))))
        baseline_ticks = max(1, int(math.ceil(float(spec["baseline_window"]["value"]))))
        end_idx = len(ticks) - window_ticks
        start_idx = max(0, end_idx - baseline_ticks)
        samples: List[float] = []
        for idx in range(start_idx + window_ticks, end_idx + 1):
            metric = _volume_metric_for_ticks(ticks[idx - window_ticks : idx], source=source)
            if metric is None:
                continue
            samples.append(metric)
        return samples
    window_seconds = float(spec["window"]["value"]) * 60.0
    baseline_seconds = float(spec["baseline_window"]["value"]) * 60.0
    latest_epoch = float(ticks[-1]["epoch"])
    current_start = latest_epoch - window_seconds
    baseline_start = current_start - baseline_seconds
    sample_count = max(1, int(math.floor(baseline_seconds / max(window_seconds, 1.0))))
    samples: List[float] = []
    for sample_idx in range(sample_count):
        window_start = baseline_start + sample_idx * window_seconds
        window_end = min(window_start + window_seconds, current_start)
        if window_end <= window_start:
            continue
        window_ticks_rows = [
            tick for tick in ticks if window_start <= float(tick["epoch"]) <= window_end
        ]
        metric = _volume_metric_for_ticks(window_ticks_rows, source=source)
        if metric is None:
            continue
        samples.append(metric)
    return samples


def _volume_metric_for_ticks(ticks: List[Dict[str, Any]], *, source: str) -> Optional[float]:
    if not ticks:
        return None
    if source == "tick_count":
        return float(len(ticks))
    if source == "volume_real":
        values = [_finite_number(tick.get("volume_real")) for tick in ticks]
    else:
        values = [_finite_number(tick.get("volume")) for tick in ticks]
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None
    return float(sum(clean))


def _current_spread_metric(spec: Dict[str, Any], ticks: List[Dict[str, Any]]) -> Optional[float]:
    current_window = _window_ticks(ticks, spec["window"])
    if not current_window:
        return None
    spreads = _spread_values_for_ticks(current_window)
    if not spreads:
        return None
    return max(spreads)


def _spread_baseline_samples(spec: Dict[str, Any], ticks: List[Dict[str, Any]]) -> List[float]:
    return _window_metric_baseline_samples(spec, ticks, metric_fn=lambda window: _max_spread_for_ticks(window))


def _current_range_metric(spec: Dict[str, Any], prices: List[tuple[float, float]]) -> Optional[float]:
    current_window = _window_prices(prices, spec["window"])
    return _price_range_pct_for_points(current_window)


def _range_baseline_samples(spec: Dict[str, Any], prices: List[tuple[float, float]]) -> List[float]:
    return _window_metric_baseline_samples_for_prices(
        spec,
        prices,
        metric_fn=_price_range_pct_for_points,
    )


def _apply_window_metric_threshold(
    spec: Dict[str, Any],
    *,
    current_value: float,
    samples: List[float],
    observed: Dict[str, Any],
    current_label: str,
    baseline_label: str,
    mode: str,
) -> Optional[float]:
    observed[current_label] = round(float(current_value), 6)
    if not samples:
        return None
    threshold_mode = str(spec["threshold_mode"])
    threshold_value = float(spec["threshold_value"])
    baseline_center = statistics.median(samples)
    observed[baseline_label] = round(float(baseline_center), 6)
    if threshold_mode == "ratio_to_baseline":
        if baseline_center <= 0.0:
            return None
        ratio = float(current_value) / float(baseline_center)
        observed["ratio"] = round(ratio, 6)
        if mode == "spike" and ratio < threshold_value:
            return None
        if mode == "drought" and ratio > threshold_value:
            return None
        return threshold_value
    if threshold_mode == "zscore":
        zscore = _zscore(float(current_value), samples)
        if zscore is None:
            return None
        observed["zscore"] = round(zscore, 6)
        if mode == "spike" and zscore < threshold_value:
            return None
        if mode == "drought" and zscore > -threshold_value:
            return None
        return threshold_value
    return None


def _window_metric_baseline_samples(
    spec: Dict[str, Any],
    ticks: List[Dict[str, Any]],
    *,
    metric_fn: Callable[[List[Dict[str, Any]]], Optional[float]],
) -> List[float]:
    if spec["window"]["kind"] == "ticks":
        window_ticks = max(1, int(math.ceil(float(spec["window"]["value"]))))
        baseline_ticks = max(1, int(math.ceil(float(spec["baseline_window"]["value"]))))
        end_idx = len(ticks) - window_ticks
        start_idx = max(0, end_idx - baseline_ticks)
        samples: List[float] = []
        for idx in range(start_idx + window_ticks, end_idx + 1):
            metric = metric_fn(ticks[idx - window_ticks : idx])
            if metric is not None:
                samples.append(metric)
        return samples
    window_seconds = float(spec["window"]["value"]) * 60.0
    baseline_seconds = float(spec["baseline_window"]["value"]) * 60.0
    latest_epoch = float(ticks[-1]["epoch"])
    current_start = latest_epoch - window_seconds
    baseline_start = current_start - baseline_seconds
    sample_count = max(1, int(math.floor(baseline_seconds / max(window_seconds, 1.0))))
    samples: List[float] = []
    for sample_idx in range(sample_count):
        window_start = baseline_start + sample_idx * window_seconds
        window_end = min(window_start + window_seconds, current_start)
        if window_end <= window_start:
            continue
        metric = metric_fn(
            [
                tick
                for tick in ticks
                if window_start <= float(tick["epoch"]) <= window_end
            ]
        )
        if metric is not None:
            samples.append(metric)
    return samples


def _window_metric_baseline_samples_for_prices(
    spec: Dict[str, Any],
    prices: List[tuple[float, float]],
    *,
    metric_fn: Callable[[List[tuple[float, float]]], Optional[float]],
) -> List[float]:
    if spec["window"]["kind"] == "ticks":
        window_ticks = max(1, int(math.ceil(float(spec["window"]["value"]))))
        baseline_ticks = max(1, int(math.ceil(float(spec["baseline_window"]["value"]))))
        end_idx = len(prices) - window_ticks
        start_idx = max(0, end_idx - baseline_ticks)
        samples: List[float] = []
        for idx in range(start_idx + window_ticks, end_idx + 1):
            metric = metric_fn(prices[idx - window_ticks : idx])
            if metric is not None:
                samples.append(metric)
        return samples
    window_seconds = float(spec["window"]["value"]) * 60.0
    baseline_seconds = float(spec["baseline_window"]["value"]) * 60.0
    latest_epoch = float(prices[-1][0])
    current_start = latest_epoch - window_seconds
    baseline_start = current_start - baseline_seconds
    sample_count = max(1, int(math.floor(baseline_seconds / max(window_seconds, 1.0))))
    samples: List[float] = []
    for sample_idx in range(sample_count):
        window_start = baseline_start + sample_idx * window_seconds
        window_end = min(window_start + window_seconds, current_start)
        if window_end <= window_start:
            continue
        metric = metric_fn(
            [
                (epoch, price)
                for epoch, price in prices
                if window_start <= epoch <= window_end
            ]
        )
        if metric is not None:
            samples.append(metric)
    return samples


def _window_ticks(ticks: List[Dict[str, Any]], window: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not ticks:
        return []
    if window["kind"] == "ticks":
        window_ticks = max(1, int(math.ceil(float(window["value"]))))
        if len(ticks) < window_ticks:
            return []
        return ticks[-window_ticks:]
    window_seconds = float(window["value"]) * 60.0
    end_epoch = float(ticks[-1]["epoch"])
    start_epoch = end_epoch - window_seconds
    return [tick for tick in ticks if float(tick["epoch"]) >= start_epoch]


def _window_prices(prices: List[tuple[float, float]], window: Dict[str, Any]) -> List[tuple[float, float]]:
    if not prices:
        return []
    if window["kind"] == "ticks":
        window_ticks = max(1, int(math.ceil(float(window["value"]))))
        if len(prices) < window_ticks:
            return []
        return prices[-window_ticks:]
    window_seconds = float(window["value"]) * 60.0
    end_epoch = float(prices[-1][0])
    start_epoch = end_epoch - window_seconds
    return [(epoch, price) for epoch, price in prices if epoch >= start_epoch]


def _spread_values_for_ticks(ticks: List[Dict[str, Any]]) -> List[float]:
    values: List[float] = []
    for tick in ticks:
        bid = _finite_number(tick.get("bid"))
        ask = _finite_number(tick.get("ask"))
        if bid is None or ask is None:
            continue
        spread = float(ask) - float(bid)
        if math.isfinite(spread) and spread >= 0.0:
            values.append(spread)
    return values


def _max_spread_for_ticks(ticks: List[Dict[str, Any]]) -> Optional[float]:
    spreads = _spread_values_for_ticks(ticks)
    if not spreads:
        return None
    return max(spreads)


def _price_range_pct_for_points(points: List[tuple[float, float]]) -> Optional[float]:
    if len(points) < 2:
        return None
    values = [float(price) for _, price in points if math.isfinite(float(price))]
    if len(values) < 2:
        return None
    base = abs(values[0])
    if base <= 0.0:
        return None
    return ((max(values) - min(values)) / base) * 100.0


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


def _window_payload(window: WaitEventWindow) -> Dict[str, Any]:
    return {
        "kind": str(window.kind),
        "value": float(window.value),
    }


def _price_direction_matches(direction: str, current_change: float) -> bool:
    if direction == "either":
        return True
    if direction == "up":
        return current_change > 0.0
    if direction == "down":
        return current_change < 0.0
    return False


def _pct_change(base_value: float, current_value: float) -> Optional[float]:
    try:
        base = float(base_value)
        current = float(current_value)
    except Exception:
        return None
    if not math.isfinite(base) or not math.isfinite(current) or base == 0.0:
        return None
    return ((current / base) - 1.0) * 100.0


def _zscore(current_value: float, samples: List[float]) -> Optional[float]:
    finite_samples: List[float] = []
    for value in samples:
        try:
            numeric = float(value)
        except Exception:
            continue
        if math.isfinite(numeric):
            finite_samples.append(numeric)
    if len(finite_samples) < 2:
        return None
    try:
        mean_value = statistics.mean(finite_samples)
        stdev_value = statistics.pstdev(finite_samples)
    except statistics.StatisticsError:
        return None
    if not math.isfinite(stdev_value) or stdev_value <= 0.0:
        return None
    return (float(current_value) - mean_value) / stdev_value


def _tick_value(row: Any, key: str) -> Any:
    if isinstance(row, dict):
        return row.get(key)
    if hasattr(row, "_asdict"):
        try:
            return row._asdict().get(key)
        except Exception:
            pass
    dtype_names = getattr(getattr(row, "dtype", None), "names", None)
    if dtype_names and key in dtype_names:
        try:
            value = row[key]
            return value.item() if hasattr(value, "item") else value
        except Exception:
            return None
    if hasattr(row, key):
        return getattr(row, key)
    return None


def _tick_key_component(value: Any) -> Any:
    numeric = _finite_number(value)
    if numeric is None:
        return None
    return float(numeric)


def _finite_number(value: Any) -> Optional[float]:
    try:
        numeric = float(value)
    except Exception:
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _tick_float(row: Any, key: str) -> float:
    value = _finite_number(_tick_value(row, key))
    return float("nan") if value is None else float(value)


def _tick_int(row: Any, key: str) -> Optional[int]:
    value = _tick_value(row, key)
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _tick_epoch(row: Any) -> Optional[float]:
    value = _tick_value(row, "time")
    if value is None:
        return None
    try:
        return float(_mt5_epoch_to_utc(float(value)))
    except Exception:
        return None


def _mt5_millis_to_utc(value_millis: float) -> int:
    try:
        return int(round(float(_mt5_epoch_to_utc(float(value_millis) / 1000.0)) * 1000.0))
    except Exception:
        return int(round(float(value_millis)))


def _tick_time_msc(row: Any, *, fallback_epoch: float) -> int:
    value = _tick_int(row, "time_msc")
    if value is not None:
        return _mt5_millis_to_utc(value)
    return int(round(float(fallback_epoch) * 1000.0))


def _tick_price(tick: Dict[str, Any], *, source: str) -> Optional[float]:
    price_candidates: List[Optional[float]]
    if source == "bid":
        price_candidates = [_finite_number(tick.get("bid"))]
    elif source == "ask":
        price_candidates = [_finite_number(tick.get("ask"))]
    elif source == "last":
        price_candidates = [_finite_number(tick.get("last"))]
    elif source == "mid":
        bid = _finite_number(tick.get("bid"))
        ask = _finite_number(tick.get("ask"))
        price_candidates = [None if bid is None or ask is None else (bid + ask) / 2.0]
    else:
        bid = _finite_number(tick.get("bid"))
        ask = _finite_number(tick.get("ask"))
        mid = None if bid is None or ask is None else (bid + ask) / 2.0
        price_candidates = [
            mid,
            _finite_number(tick.get("last")),
            bid,
            ask,
        ]
    for candidate in price_candidates:
        if candidate is not None:
            return float(candidate)
    return None


def _coerce_rows(rows: Any) -> List[Any]:
    if rows is None:
        return []
    if isinstance(rows, list):
        return rows
    try:
        return list(rows)
    except Exception:
        return []


def _row_value(row: Any, key: str) -> Any:
    if isinstance(row, dict):
        return row.get(key)
    if hasattr(row, "_asdict"):
        try:
            return row._asdict().get(key)
        except Exception:
            pass
    dtype_names = getattr(getattr(row, "dtype", None), "names", None)
    if dtype_names and key in dtype_names:
        try:
            value = row[key]
            return value.item() if hasattr(value, "item") else value
        except Exception:
            return None
    if hasattr(row, key):
        return getattr(row, key)
    return None


def _row_int(row: Any, key: str) -> Optional[int]:
    value = _row_value(row, key)
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _first_int(*values: Optional[int]) -> Optional[int]:
    for value in values:
        if value is not None:
            return int(value)
    return None


def _datetime_epoch_millis(value: datetime) -> int:
    return int(round(_normalize_utc_datetime(value).timestamp() * 1000.0))


def _row_event_time_millis(row: Any) -> Optional[int]:
    for key in ("time_msc", "time_done_msc", "time_setup_msc", "time_update_msc"):
        value = _row_int(row, key)
        if value is not None:
            return _mt5_millis_to_utc(value)
    for key in ("time", "time_done", "time_setup", "time_update"):
        value = _row_value(row, key)
        if value is None:
            continue
        dt = _normalize_optional_utc_datetime(value)
        if dt is not None:
            return _datetime_epoch_millis(dt)
    return None


def _account_history_row_key(row: Any, *, row_kind: str) -> Optional[tuple[Any, ...]]:
    ticket = _row_int(row, "ticket")
    order_ticket = _first_int(
        _row_int(row, "order"),
        _row_int(row, "order_ticket"),
    )
    position_ticket = _first_int(
        _row_int(row, "position_id"),
        _row_int(row, "position"),
        _row_int(row, "position_by_id"),
    )
    time_millis = _row_event_time_millis(row)
    symbol = str(_row_value(row, "symbol") or "").upper().strip() or None
    entry = _row_value(row, "entry")
    state = _row_value(row, "state")
    side = _row_value(row, "type")
    key = (
        str(row_kind),
        ticket,
        order_ticket,
        position_ticket,
        time_millis,
        symbol,
        None if entry is None else str(entry),
        None if state is None else str(state),
        None if side is None else str(side),
    )
    if not any(value is not None for value in key[1:]):
        return None
    return key


def _row_time_iso(row: Any) -> Optional[str]:
    for key in ("time", "time_done", "time_setup", "time_update"):
        value = _row_value(row, key)
        if value is None:
            continue
        dt = _normalize_optional_utc_datetime(value)
        if dt is not None:
            return dt.isoformat()
    return None


def _normalize_optional_utc_datetime(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        return _normalize_utc_datetime(value)
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(_mt5_epoch_to_utc(float(value)), tz=timezone.utc)
        except Exception:
            return None
    if isinstance(value, str):
        try:
            return _normalize_utc_datetime(datetime.fromisoformat(value))
        except Exception:
            return None
    return None


def _normalize_utc_datetime(value: Any) -> datetime:
    if isinstance(value, str):
        value = datetime.fromisoformat(value)
    if not isinstance(value, datetime):
        raise TypeError(f"Expected datetime-compatible value, got {type(value).__name__}.")
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _resolved_value(spec: Any, request: WaitEventRequest, field_name: str, default: Any = None) -> Any:
    value = getattr(spec, field_name, None)
    if value is not None:
        return value
    request_value = getattr(request, field_name, None)
    if request_value is not None:
        return request_value
    return default


def _normalize_side(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"buy", "sell"}:
        return text
    return None
