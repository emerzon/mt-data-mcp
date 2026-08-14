from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np

from ...shared.result import Err, Ok, Result, to_dict
from ...utils.freshness import format_age_seconds as _format_age_seconds
from ...utils.freshness import format_freshness_label
from ...utils.market_metadata import (
    FRESHNESS_ANCHOR_QUERY_EXPECTED_END,
    FRESHNESS_ANCHOR_WALL_CLOCK,
    FRESHNESS_METRIC_LAST_COMPLETED_BAR_AGE,
    FRESHNESS_METRIC_LAST_TICK_AGE,
    FRESHNESS_METRIC_REQUESTED_RANGE_END_GAP,
    attach_candle_volume_semantics,
    build_tick_freshness_context,
    normalize_policy_relaxed,
)
from ...utils.quote import (
    canonical_quote_midpoint,
    enforce_quote_execution_readiness,
    resolve_quote_tick,
    tick_epoch,
    tick_value,
)
from ...utils.symbol import symbol_suggestions_from_gateway
from ..error_envelope import build_error_payload
from ..execution_logging import run_logged_operation
from ..mt5_gateway import mt5_connection_error
from ..output_contract import attach_collection_contract
from ..runtime_metadata import attach_mt5_source
from ..trading.time import _next_candle_wait_payload, _sleep_until_next_candle
from .requests import (
    DATA_FETCH_CANDLES_DEFAULT_LIMIT,
    DataFetchCandlesRequest,
    DataFetchTicksRequest,
    WaitCandleRequest,
    WaitEventRequest,
)
from .wait_events import run_wait_event_loop

logger = logging.getLogger(__name__)

_TICK_DETAIL_FORMATS = {
    "compact": "rows",
    "summary": "summary",
    "standard": "stats",
    "full": "full_rows",
}

_COMPACT_TICK_TOP_LEVEL_FIELDS = (
    "success",
    "symbol",
    "count",
    "tick_count",
    "trade_event_count",
    "quote_update_count",
    "feed_tier",
    "data",
    "empty",
    "empty_reason",
    "last_quote",
    "execution_quote",
    "timezone",
    "price_precision",
    "price_point",
    "price_currency",
    "units",
    "spread_statistics_basis",
    "freshness",
    "freshness_state",
    "freshness_reason",
    "data_age_seconds",
    "data_age_anchor",
    "data_age_metric",
    "data_stale",
    "timestamp_ahead_of_wall_clock",
    "timestamp_in_future",
    "timestamp_skew_seconds",
    "timestamp_skew_tolerance_seconds",
    "timestamp_warning",
    "history_policy_ok",
    "usable_for_live_trading",
    "usable_for_live_trading_basis",
    "live_max_age_seconds",
    "market_status",
    "market_status_reason",
    "market_status_source",
    "freshness_policy_relaxed",
    "note",
    "simplified",
    "simplify",
    "query_applied",
    "history_window_truncated",
    "history_window_limit_days",
    "history_window_floor",
    "effective_start",
    "data_quality",
    "last_unavailable",
    "warnings",
)

_ANALYSIS_CANDLE_DEFAULT_LIMIT = 100
_RANGE_CANDLE_DEFAULT_LIMIT = DATA_FETCH_CANDLES_DEFAULT_LIMIT


def _ensure_gateway_connection(gateway: Any) -> Dict[str, Any] | None:
    return mt5_connection_error(gateway)


def run_data_fetch_candles(
    request: DataFetchCandlesRequest,
    *,
    gateway: Any,
    fetch_candles_impl: Any,
) -> Dict[str, Any]:
    effective_limit = _effective_candle_limit(request)
    return run_logged_operation(
        logger,
        operation="data_fetch_candles",
        symbol=request.symbol,
        timeframe=request.timeframe,
        limit=effective_limit,
        func=lambda: _run_data_fetch_candles_impl(
            request=request,
            gateway=gateway,
            fetch_candles_impl=fetch_candles_impl,
            effective_limit=effective_limit,
        ),
    )


def run_data_fetch_ticks(
    request: DataFetchTicksRequest,
    *,
    gateway: Any,
    fetch_ticks_impl: Any,
) -> Dict[str, Any]:
    return run_logged_operation(
        logger,
        operation="data_fetch_ticks",
        symbol=request.symbol,
        limit=request.limit,
        detail=request.detail,
        func=lambda: _run_data_fetch_ticks_impl(
            request=request,
            gateway=gateway,
            fetch_ticks_impl=fetch_ticks_impl,
        ),
    )


def run_wait_candle(
    request: WaitCandleRequest,
    *,
    sleep_impl: Any = time.sleep,
) -> Dict[str, Any]:
    result = run_logged_operation(
        logger,
        operation="wait_candle",
        timeframe=request.timeframe,
        buffer_seconds=request.buffer_seconds,
        func=lambda: _run_wait_candle_impl(
            request=request,
            sleep_impl=sleep_impl,
        ),
    )
    return to_dict(result) if isinstance(result, (Ok, Err)) else result


def run_wait_event(
    request: WaitEventRequest,
    *,
    gateway: Any,
    sleep_impl: Any = time.sleep,
    monotonic_impl: Any = time.monotonic,
    now_utc_impl: Any = lambda: datetime.now(timezone.utc),
) -> Dict[str, Any]:
    result = run_logged_operation(
        logger,
        operation="wait_event",
        watch_for=len(request.watch_for or []),
        end_on=len(request.end_on),
        poll_interval_seconds=request.poll_interval_seconds,
        func=lambda: _run_wait_event_impl(
            request=request,
            gateway=gateway,
            sleep_impl=sleep_impl,
            monotonic_impl=monotonic_impl,
            now_utc_impl=now_utc_impl,
        ),
        success_eval=lambda r: (
            (
                isinstance(r, Ok)
                and (
                    not isinstance(r.value, dict)
                    or r.value.get("success") is not False
                )
            )
            or (isinstance(r, dict) and "error" not in r)
        ),
    )
    payload = to_dict(result) if isinstance(result, (Ok, Err)) else result
    return attach_mt5_source(payload, gateway=gateway)


def _run_data_fetch_candles_impl(
    *,
    request: DataFetchCandlesRequest,
    gateway: Any,
    fetch_candles_impl: Any,
    effective_limit: Optional[int] = None,
) -> Dict[str, Any]:
    connection_error = _ensure_gateway_connection(gateway)
    if connection_error is not None:
        return connection_error
    result = fetch_candles_impl(
        symbol=request.symbol,
        timeframe=request.timeframe,
        limit=effective_limit if effective_limit is not None else request.limit,
        start=request.start,
        end=request.end,
        ohlcv=request.ohlcv,
        indicators=request.indicators,
        denoise=request.denoise,
        simplify=request.simplify,
        time_as_epoch=str(request.timestamp_format).strip().lower() != "iso",
        include_spread=request.include_spread,
        include_incomplete=request.include_incomplete,
        allow_stale=request.allow_stale,
    )
    result = _normalize_candle_query_error(
        result,
        request=request,
        gateway=gateway,
    )
    detail_mode = str(request.detail or "compact").strip().lower()
    if isinstance(result, dict):
        _normalize_public_candle_timestamp_mode(
            result,
            include_raw=detail_mode == "full",
        )
        limit_explicit = "limit" in getattr(request, "model_fields_set", set())
        applied_limit = (
            effective_limit if effective_limit is not None else request.limit
        )
        if bool(getattr(request, "explain_indicators", False)):
            _attach_indicator_explanations(result)
        _apply_range_limit_cap(
            result,
            limit=applied_limit,
            limit_explicit=limit_explicit,
            start=request.start,
            end=request.end,
        )
        if request.start or request.end:
            _normalize_range_limit_contract(
                result,
                effective_limit=applied_limit,
                limit_explicit=limit_explicit,
            )
        _annotate_empty_candle_result(result)
        _normalize_candle_count_field(result)
        _prune_zero_candle_exclusions(result)
        if detail_mode == "compact":
            result = _compact_candles_payload(result)
            _slim_projected_candles_payload(result)
            _drop_redundant_session_gap_warnings(result)
        elif detail_mode == "summary":
            result = _summary_candles_payload(result)
        elif detail_mode == "standard":
            result = _standard_candles_payload(result)
        _attach_candle_machine_freshness(result)
        _attach_latest_candle_quote_freshness(
            result,
            request=request,
            gateway=gateway,
        )
        _attach_forming_candle_update_freshness(
            result,
            request=request,
            gateway=gateway,
        )
        result = attach_mt5_source(result, gateway=gateway)
    if isinstance(result, dict) and isinstance(result.get("data"), list):
        out = attach_collection_contract(
            result,
            collection_kind="time_series",
            series=result["data"],
            include_contract_meta=detail_mode == "full",
        )
        if detail_mode == "full" and isinstance(out, dict):
            out.pop("series", None)
            out.pop("canonical_source", None)
        return out
    return result


def _attach_forming_candle_update_freshness(
    payload: Dict[str, Any],
    *,
    request: DataFetchCandlesRequest,
    gateway: Any,
) -> None:
    if not request.include_incomplete or payload.get("error"):
        return
    data_window = payload.get("data_window")
    if not isinstance(data_window, dict) or data_window.get("latest_bar_complete") is not False:
        return
    try:
        tick = gateway.symbol_info_tick(request.symbol)
    except Exception:
        return
    tick_msc = getattr(tick, "time_msc", None) if tick is not None else None
    tick_seconds = getattr(tick, "time", None) if tick is not None else None
    try:
        tick_epoch = float(tick_msc) / 1000.0 if tick_msc else float(tick_seconds)
    except (TypeError, ValueError):
        return
    if not np.isfinite(tick_epoch) or tick_epoch <= 0:
        return
    update_age = max(0.0, float(time.time()) - tick_epoch)
    bar_open_age = payload.get("data_age_seconds")
    try:
        bar_open_age_value = max(0.0, float(bar_open_age))
    except (TypeError, ValueError):
        bar_open_age_value = None
    if bar_open_age_value is not None:
        payload["bar_open_age_seconds"] = round(bar_open_age_value, 3)
        data_window["latest_bar_open_age_seconds"] = round(bar_open_age_value, 3)
    payload["last_update_age_seconds"] = round(update_age, 3)
    payload["data_age_seconds"] = round(update_age, 3)
    payload["data_age_anchor"] = FRESHNESS_ANCHOR_WALL_CLOCK
    payload["data_age_metric"] = FRESHNESS_METRIC_LAST_TICK_AGE
    data_window["latest_bar_update_age_seconds"] = round(update_age, 3)
    update_text = _format_age_seconds(update_age)
    if bar_open_age_value is not None:
        payload["freshness"] = (
            f"forming bar open {_format_age_seconds(bar_open_age_value)} ago; "
            f"last update {update_text} ago"
        )
    else:
        payload["freshness"] = f"forming bar; last update {update_text} ago"


def _attach_latest_candle_quote_freshness(
    payload: Dict[str, Any],
    *,
    request: DataFetchCandlesRequest,
    gateway: Any,
) -> None:
    """Prevent a stale latest quote from being presented as a fresh candle mark."""
    if request.start or request.end or request.include_incomplete or payload.get("error"):
        return
    rows = payload.get("data")
    if not isinstance(rows, list) or not rows:
        return
    try:
        now_epoch = time.time()
        tick, _ = resolve_quote_tick(gateway, request.symbol, now_epoch=now_epoch)
        quote_context = build_tick_freshness_context(
            request.symbol,
            tick_epoch=tick_epoch(tick),
            now_epoch=now_epoch,
            item="tick",
        )
    except Exception:
        return
    if quote_context.get("data_stale") is not True:
        return
    quote_age = quote_context.get("data_age_seconds")
    payload["latest_quote_stale"] = True
    if quote_age is not None:
        payload["latest_quote_age_seconds"] = quote_age
    payload["data_stale"] = True
    payload["freshness_basis"] = "bar_policy_and_latest_quote"
    payload["freshness_reason"] = str(
        quote_context.get("freshness_reason") or "latest_quote_stale"
    )
    for key in ("market_status", "market_status_reason", "market_status_source"):
        if quote_context.get(key) is not None:
            payload[key] = quote_context[key]
    freshness_label = format_freshness_label(
        data_stale=True,
        market_status=payload.get("market_status"),
        market_status_reason=payload.get("market_status_reason"),
        age_seconds=payload.get("data_age_seconds"),
        item="bar",
    )
    if freshness_label:
        payload["freshness"] = freshness_label
    payload["stale_warning"] = (
        "The latest quote is stale, so the last candle must not be treated as a "
        "live mark even though completed-bar history is within policy."
    )


def _normalize_candle_query_error(
    result: Any,
    *,
    request: DataFetchCandlesRequest,
    gateway: Any = None,
) -> Any:
    if not isinstance(result, dict) or not result.get("error"):
        return result
    if result.get("error_code") == "data_fetch_candles_no_data":
        details = result.get("details")
        details = dict(details) if isinstance(details, dict) else {}
        empty_reason = str(details.get("no_data_reason") or "no_candles_in_range")
        payload: Dict[str, Any] = {
            "success": True,
            "symbol": request.symbol,
            "timeframe": request.timeframe,
            "count": 0,
            "data": [],
            "empty": True,
            "empty_reason": empty_reason,
        }
        if details.get("no_data_reason") is not None:
            payload["no_data_reason"] = details["no_data_reason"]
        for key in (
            "market_status_reason",
            "note",
            "requested_range",
            "available_range",
        ):
            if details.get(key) is not None:
                payload[key] = details[key]
        for key in ("query_applied", "warnings", "diagnostics"):
            if result.get(key) is not None:
                payload[key] = result[key]
        return payload
    if result.get("error_code"):
        return result

    message = str(result["error"])
    normalized = message.lower()
    error_code: Optional[str] = None
    remediation: Optional[str] = None

    if "not found" in normalized and "symbol" in normalized:
        error_code = "symbol_not_found"
        message = f"Symbol '{request.symbol}' was not found in MT5."
        remediation = (
            f"Use symbols_list(search_term='{request.symbol}') to find the broker's "
            "exact MT5 symbol name, including any suffixes or aliases."
        )
    elif "could not parse date" in normalized or "invalid date" in normalized:
        error_code = "data_fetch_candles_invalid_date"
        remediation = (
            "Use an ISO 8601 date or timestamp, for example 2026-08-03 or "
            "2026-08-03T14:30:00Z."
        )
    elif (
        "start_datetime must be before end_datetime" in normalized
        or "start must be before or equal to end" in normalized
    ):
        error_code = "data_fetch_candles_invalid_date_range"
        remediation = "Set start to a timestamp earlier than or equal to end."
    elif "in the future" in normalized and "start" in normalized:
        error_code = "data_fetch_candles_future_date_range"
        remediation = "Use a start timestamp at or before the current time."
    elif "data appears stale" in normalized:
        error_code = "data_fetch_candles_stale_data"
        remediation = (
            "Confirm the market session and broker feed, or set allow_stale=true "
            "when historical data is intentionally acceptable."
        )

    if error_code is None:
        return result

    details = {
        "symbol": request.symbol,
        "timeframe": request.timeframe,
    }
    if error_code == "symbol_not_found":
        details["did_you_mean"] = symbol_suggestions_from_gateway(
            gateway,
            request.symbol,
        )
    elif error_code == "data_fetch_candles_stale_data":
        related_live_symbols = _find_live_extended_session_symbols(
            gateway,
            request.symbol,
        )
        if related_live_symbols:
            details["related_live_symbols"] = related_live_symbols
            live_symbol = related_live_symbols[0]["symbol"]
            remediation = (
                f"A live extended-session contract is available: call market_ticker "
                f"for {live_symbol}, then use that exact symbol for current data. "
                "Set allow_stale=true only when the regular-session history is "
                "intentionally acceptable."
            )
    if request.start is not None:
        details["start"] = str(request.start)
    if request.end is not None:
        details["end"] = str(request.end)

    payload = build_error_payload(
        message,
        code=error_code,
        operation="data_fetch_candles",
        details=details,
        remediation=remediation,
    )
    for key in ("warnings", "diagnostics"):
        if key in result:
            payload[key] = result[key]
    return payload


def _find_live_extended_session_symbols(
    gateway: Any,
    requested_symbol: str,
    *,
    limit: int = 3,
) -> List[Dict[str, str]]:
    """Find visible, executable extended-session siblings for a stale symbol."""
    requested = str(requested_symbol or "").strip()
    if not requested or gateway is None:
        return []
    try:
        symbol_infos = list(gateway.symbols_get() or [])
    except Exception:
        return []

    requested_upper = requested.upper()
    now_epoch = time.time()
    matches: List[Dict[str, str]] = []
    for info in symbol_infos:
        name = str(getattr(info, "name", "") or "").strip()
        if not name or name.casefold() == requested.casefold():
            continue
        name_upper = name.upper()
        descriptor = " ".join(
            str(getattr(info, field, "") or "").upper()
            for field in ("name", "description", "path")
        )
        is_related = name_upper.startswith(requested_upper)
        is_extended = any(
            marker in descriptor for marker in ("-24", "24HR", "24/5", "24H")
        )
        if not is_related or not is_extended:
            continue
        if getattr(info, "visible", True) is False:
            continue
        try:
            resolved_tick, quote_meta = resolve_quote_tick(
                gateway,
                name,
                now_epoch=now_epoch,
            )
            freshness = build_tick_freshness_context(
                name,
                tick_epoch=tick_epoch(resolved_tick),
                now_epoch=now_epoch,
                item="tick",
            )
            enforce_quote_execution_readiness(
                freshness,
                bid=tick_value(resolved_tick, "bid"),
                ask=tick_value(resolved_tick, "ask"),
                quote_source_conflict=quote_meta.get("quote_source_conflict"),
            )
        except Exception:
            continue
        if freshness.get("usable_for_live_trading") is not True:
            continue
        matches.append(
            {
                "symbol": name,
                "session_type": "extended_24h",
                "quote_tool": "market_ticker",
            }
        )
        if len(matches) >= max(1, int(limit)):
            break
    return matches


def _effective_candle_limit(request: DataFetchCandlesRequest) -> int:
    try:
        limit = max(1, int(request.limit))
    except Exception:
        limit = DATA_FETCH_CANDLES_DEFAULT_LIMIT
    fields_set = getattr(request, "model_fields_set", set())
    limit_explicit = "limit" in fields_set
    if (request.start or request.end) and not limit_explicit:
        return _RANGE_CANDLE_DEFAULT_LIMIT
    has_indicators = request.indicators not in (None, "", [], {})
    if has_indicators and not limit_explicit:
        return max(limit, _ANALYSIS_CANDLE_DEFAULT_LIMIT)
    return limit


def _annotate_empty_candle_result(result: Dict[str, Any]) -> None:
    if (
        result.get("error")
        or not isinstance(result.get("data"), list)
        or result["data"]
    ):
        return
    reported_count = result.get("count", result.get("candles"))
    try:
        if reported_count is not None and int(reported_count) > 0:
            return
    except (TypeError, ValueError):
        pass
    result["empty"] = True
    result.setdefault(
        "empty_reason",
        result.get("range_incomplete_reason") or "no_candles_in_range",
    )


def _latest_numeric_row_value(rows: Any, column: str) -> Optional[float]:
    if not isinstance(rows, list):
        return None
    for row in reversed(rows):
        if not isinstance(row, dict) or column not in row:
            continue
        try:
            value = float(row.get(column))
        except Exception:
            continue
        if np.isfinite(value):
            return value
    return None


def _indicator_family(column: str) -> str:
    name = str(column or "").strip().upper()
    if name.startswith("MACD"):
        return "MACD"
    return name.split("_", 1)[0]


def _indicator_reading(column: str, value: float, *, latest_close: Optional[float]) -> str:
    family = _indicator_family(column)
    if family == "RSI":
        if value >= 70.0:
            state = "overbought"
        elif value <= 30.0:
            state = "oversold"
        else:
            state = "neutral"
        return f"RSI {value:.2f}: {state}; common bands are 30/70."
    if family in {"EMA", "SMA", "WMA", "HMA"}:
        if latest_close is None:
            return f"{family} {value:.5g}: moving-average trend reference."
        side = "above" if latest_close > value else "below" if latest_close < value else "at"
        return f"Close is {side} {family} ({value:.5g}); above often supports bullish trend context."
    if family == "MACD":
        if str(column).upper().startswith("MACDH"):
            side = "positive" if value > 0 else "negative" if value < 0 else "flat"
            return f"MACD histogram {value:.5g}: {side} momentum."
        side = "above zero" if value > 0 else "below zero" if value < 0 else "at zero"
        return f"MACD {value:.5g}: {side}; compare line/signal/histogram together."
    if family == "ATR":
        return f"ATR {value:.5g}: volatility/range estimate in price units."
    if family in {"BBL", "BBM", "BBU"}:
        return f"{family} {value:.5g}: Bollinger Band level; compare close to lower/mid/upper bands."
    return f"{column} {value:.5g}: see indicators_describe for detailed interpretation."


def _attach_indicator_explanations(result: Dict[str, Any]) -> None:
    meta = result.get("meta")
    diagnostics = meta.get("diagnostics") if isinstance(meta, dict) else None
    indicators = diagnostics.get("indicators") if isinstance(diagnostics, dict) else None
    added_columns = indicators.get("added_columns") if isinstance(indicators, dict) else None
    if not isinstance(added_columns, list) or not added_columns:
        return
    rows = result.get("data")
    latest_close = _latest_numeric_row_value(rows, "close")
    explanations: List[Dict[str, Any]] = []
    for column in added_columns:
        column_name = str(column or "").strip()
        if not column_name:
            continue
        value = _latest_numeric_row_value(rows, column_name)
        if value is None:
            continue
        explanations.append(
            {
                "column": column_name,
                "family": _indicator_family(column_name),
                "latest": round(float(value), 6),
                "reading": _indicator_reading(column_name, value, latest_close=latest_close),
            }
        )
    if explanations:
        result["indicator_explanations"] = explanations


def _apply_range_limit_cap(
    result: Dict[str, Any],
    *,
    limit: int,
    limit_explicit: bool,
    start: Optional[str],
    end: Optional[str],
) -> None:
    data = result.get("data")
    if not isinstance(data, list):
        return
    meta = result.get("meta")
    diagnostics = meta.get("diagnostics") if isinstance(meta, dict) else None
    query = diagnostics.get("query") if isinstance(diagnostics, dict) else None
    if not isinstance(query, dict) or query.get("mode") != "range":
        return
    query_applied = result.get("query_applied")
    if not isinstance(query_applied, dict):
        query_applied = {}
        result["query_applied"] = query_applied
    query_applied.setdefault("mode", "range")
    if start not in (None, ""):
        query_applied.setdefault("start", str(start))
    if end not in (None, ""):
        query_applied.setdefault("end", str(end))
    start_anchored = start not in (None, "")
    query_applied["limit_anchor"] = "start" if start_anchored else "end"
    query_applied["selection"] = "first_n" if start_anchored else "last_n"
    query_applied["order"] = "ascending"
    query_applied["limit_source"] = "user" if limit_explicit else "default"
    try:
        limit_value = max(1, int(limit))
    except Exception:
        return
    available = len(data)
    provider_bounded = bool(query.get("provider_bounded"))
    if available <= limit_value and not provider_bounded:
        spacing_mismatch = bool(result.get("timeframe_spacing_mismatch"))
        forming_bar_excluded = bool(
            result.get("forming_candle_status") == "skipped"
            and (
                result.get("has_forming_candle") is True
                or int(result.get("incomplete_candles_skipped") or 0) > 0
            )
        )
        result["range_complete"] = not spacing_mismatch and not forming_bar_excluded
        if forming_bar_excluded:
            result["range_incomplete_reason"] = "forming_bar_excluded"
            data_window = result.get("data_window")
            if isinstance(data_window, dict):
                data_window["latest_bar_complete"] = False
        elif spacing_mismatch:
            result["range_incomplete_reason"] = "timeframe_spacing_mismatch"
        return

    retained = (
        data[:limit_value]
        if start_anchored and available > limit_value
        else data[-limit_value:]
        if available > limit_value
        else data
    )
    result["data"] = retained
    result["count"] = len(retained)
    result["limit_applied"] = limit_value
    result["truncated"] = True
    result["truncation"] = {
        "reason": "limit",
        "retained": "first" if start_anchored else "last",
    }
    result["range_complete"] = False
    if available > limit_value:
        result["available_count"] = available
        result["truncation"]["excluded_count"] = available - len(retained)
        retained_label = "earliest" if start_anchored else "latest"
        warning = (
            f"Fetched range contained {available} bars; returned the {retained_label} "
            f"{len(retained)} because limit={limit_value}."
        )
        result["pagination"] = {
            "total": available,
            "returned": len(retained),
            "offset": 0,
            "limit": limit_value,
            "has_more": True,
            "more_available": available - len(retained),
        }
    else:
        result["truncation"]["excluded_count"] = None
        warning = (
            "The requested range began before the bounded provider window; "
            f"returned up to the latest {limit_value} bars. Increase limit or "
            "move the range start forward to retrieve an earlier page."
        )
        result["pagination"] = {
            "total": None,
            "total_lower_bound": len(retained) + 1,
            "returned": len(retained),
            "offset": 0,
            "limit": limit_value,
            "has_more": True,
            "more_available": None,
        }
    result.setdefault("warnings", []).append(warning)
    data_window = result.get("data_window")
    if isinstance(data_window, dict) and retained:
        first_row = retained[0]
        last_row = retained[-1]
        if isinstance(first_row, dict) and first_row.get("time") is not None:
            data_window["start"] = first_row["time"]
        if isinstance(last_row, dict) and last_row.get("time") is not None:
            data_window["end"] = last_row["time"]
    candle_counts = result.get("candle_counts")
    if isinstance(candle_counts, dict):
        candle_counts["returned"] = len(retained)
        excluded = candle_counts.get("excluded")
        if not isinstance(excluded, dict):
            excluded = {}
            candle_counts["excluded"] = excluded
        excluded_count = max(0, available - len(retained))
        excluded["limit_truncated"] = excluded_count
        excluded["total"] = int(excluded.get("total") or 0) + excluded_count
    query["limit_applied_to_range"] = True
    query["available_rows_before_limit"] = available
    query["returned_rows_after_limit"] = len(retained)


def _normalize_range_limit_contract(
    result: Dict[str, Any],
    *,
    effective_limit: int,
    limit_explicit: bool,
) -> None:
    query_applied = result.get("query_applied")
    if not isinstance(query_applied, dict):
        return
    result["limit_explicit"] = bool(limit_explicit)
    if limit_explicit:
        return
    result.pop("requested_limit", None)
    result.pop("candles_requested", None)
    result["default_limit"] = int(effective_limit)
    query_applied.pop("limit", None)
    query_applied["default_limit"] = int(effective_limit)
    candle_counts = result.get("candle_counts")
    if isinstance(candle_counts, dict):
        candle_counts.pop("requested", None)
        excluded = candle_counts.get("excluded")
        if isinstance(excluded, dict):
            excluded.pop("window_or_source_shortfall", None)
            excluded["total"] = sum(
                int(value)
                for key, value in excluded.items()
                if key != "total" and isinstance(value, int) and value > 0
            )


def _normalize_candle_count_field(result: Dict[str, Any]) -> None:
    candles_value = result.pop("candles", None)
    if "count" not in result and candles_value is not None:
        result["count"] = candles_value
    elif "count" not in result:
        data = result.get("data")
        if isinstance(data, list):
            result["count"] = len(data)
    result.pop("returned_count", None)
    data_window = result.get("data_window")
    if isinstance(data_window, dict):
        data_window.pop("requested_limit", None)
        data_window.pop("returned_count", None)


def _compact_candles_payload(
    result: Dict[str, Any],
    *,
    include_forming_booleans: bool = False,
) -> Dict[str, Any]:
    compact = dict(result)
    compact_time_normalization = result.get("time_normalization")
    public_diagnostics = _public_candle_diagnostics(result)
    try:
        requested_count = int(result["candles_requested"])
        returned_count = int(compact["count"])
    except (KeyError, TypeError, ValueError):
        pass
    else:
        query_applied = result.get("query_applied")
        is_range = (
            isinstance(query_applied, dict)
            and query_applied.get("mode") == "range"
        )
        if is_range and requested_count >= 0 and returned_count >= 0:
            compact["limit_reached"] = returned_count >= requested_count
            compact["range_complete"] = bool(result.get("range_complete", False))
        elif requested_count >= 0 and returned_count >= 0:
            # Compact responses omit the detailed exclusion breakdown, but a
            # caller must still be able to distinguish a complete response
            # from one shortened by the source, filters, or a forming bar.
            compact["limit_satisfied"] = returned_count >= requested_count
    for key in (
        "candles_requested",
        "candle_counts",
        "candles_excluded",
        "hint",
        "incomplete_candles_skipped",
        "spread_note",
        "volume_note",
        "bar_time_convention",
        "meta",
        "raw_time_basis",
        "raw_timestamp_mode",
        "time_normalization",
        "broker_server_tz",
        "broker_utc_offset_seconds",
        "timezone_note",
        "volume_semantics",
        "data_age_anchor",
        "data_age_metric",
        "query_end_gap_anchor",
        "query_end_gap_metric",
        "mt5_time_alignment",
    ):
        compact.pop(key, None)
    if not bool(compact.get("has_forming_candle")):
        compact.pop("has_forming_candle", None)
        compact["forming_candle_status"] = str(
            compact.get("forming_candle_status") or "none"
        )
        compact.pop("forming_candle_included", None)
        compact.pop("forming_candle_skipped", None)
    elif not include_forming_booleans:
        compact.pop("has_forming_candle", None)
        compact.pop("forming_candle_included", None)
        compact.pop("forming_candle_skipped", None)
    if result.get("forming_candle_status") == "skipped" and result.get("hint"):
        compact["hint"] = result["hint"]
    _attach_candle_timestamp_metadata(compact)
    if compact_time_normalization not in (None, ""):
        compact["time_normalization"] = compact_time_normalization
    for key in (
        "query_type",
        "freshness",
        "data_age_seconds",
        "data_stale",
        "history_policy_ok",
        "usable_for_live_trading",
        "usable_for_live_trading_basis",
        "freshness_policy_relaxed",
        "market_status",
        "market_status_reason",
        "market_status_source",
        "note",
        "query_end_gap_seconds",
        "query_end_gap",
        "indicator_warmup_bars",
        "history_bars_fetched",
    ):
        if key in public_diagnostics:
            compact[key] = public_diagnostics[key]
    if "spread_estimate" in public_diagnostics:
        compact["spread_estimate"] = public_diagnostics["spread_estimate"]
    _attach_denoise_disclosure(compact)
    attach_candle_volume_semantics(compact)
    return compact


def _attach_candle_timestamp_metadata(payload: Dict[str, Any]) -> None:
    rows = payload.get("data")
    if not isinstance(rows, list):
        latest = payload.get("latest_candle")
        rows = [latest] if isinstance(latest, dict) else []
    for row in rows:
        if not isinstance(row, dict) or "time" not in row:
            continue
        timestamp_value = row.get("time")
        if isinstance(timestamp_value, bool):
            continue
        if isinstance(timestamp_value, (int, float)) and np.isfinite(float(timestamp_value)):
            payload["timestamp_format"] = "epoch_seconds"
            payload.pop("timestamp_format_hint", None)
            return
        if isinstance(timestamp_value, str) and timestamp_value.strip():
            payload["timestamp_format"] = "iso_utc"
            payload.pop("timestamp_format_hint", None)
            return


def _normalize_public_candle_timestamp_mode(
    payload: Dict[str, Any],
    *,
    include_raw: bool,
) -> None:
    """Name the clock used by emitted timestamps, not the raw MT5 epoch axis."""
    raw_mode = str(payload.get("timestamp_mode") or "").strip()
    time_basis = str(payload.get("time_basis") or "").strip().lower()
    if not raw_mode or time_basis != "utc":
        return
    payload["timestamp_mode"] = "utc"
    payload["public_timestamp_mode"] = "utc"
    if include_raw:
        payload["raw_timestamp_mode"] = raw_mode
    else:
        payload.pop("raw_timestamp_mode", None)


def _attach_denoise_disclosure(payload: Dict[str, Any]) -> None:
    denoise_info = payload.get("denoise")
    applications = denoise_info.get("applications") if isinstance(denoise_info, dict) else None
    if not isinstance(applications, list) or not applications:
        return

    methods: List[str] = []
    overwritten: List[str] = []
    causalities: List[str] = []
    for app in applications:
        if not isinstance(app, dict):
            continue
        added_columns = app.get("added_columns")
        overwritten_columns = app.get("overwrote_columns")
        added = added_columns if isinstance(added_columns, list) else []
        overwritten_for_app = (
            overwritten_columns if isinstance(overwritten_columns, list) else []
        )
        if not added and not overwritten_for_app:
            continue
        method = str(app.get("method") or "").strip().lower()
        if method and method != "none" and method not in methods:
            methods.append(method)
        causality = str(app.get("causality") or "").strip().lower()
        if causality and causality not in causalities:
            causalities.append(causality)
        if bool(app.get("keep_original")):
            continue
        for column in overwritten_for_app:
            column = str(column).strip()
            if column and column not in overwritten:
                overwritten.append(column)

    if not methods and not overwritten:
        return
    payload["denoise_applied"] = True
    payload["denoise_status"] = "applied"
    if methods:
        payload["denoise_method"] = methods[0] if len(methods) == 1 else methods
    if overwritten:
        payload["denoise_overwrote_columns"] = overwritten
        if "close" in overwritten and methods:
            payload["price_column"] = f"close ({methods[0]}-smoothed)"
            payload["price_is_synthetic"] = True
    if "zero_phase" in causalities:
        payload["denoise_live_safe"] = False
        payload.setdefault("warnings", []).append(
            "Zero-phase denoise uses future observations and is not usable for live trading."
        )
    elif causalities:
        payload["denoise_live_safe"] = True
    payload.pop("denoise", None)


def _slim_projected_candles_payload(payload: Dict[str, Any]) -> None:
    if not bool(payload.get("ohlcv_filter_applied")):
        return
    rows = payload.get("data")
    projected_fields: set[str] = set()
    if isinstance(rows, list):
        for row in rows:
            if isinstance(row, dict):
                projected_fields.update(str(key) for key in row if str(key) != "time")
    payload.pop("ohlcv_filter_applied", None)
    if not projected_fields or projected_fields.isdisjoint({"tick_volume", "volume"}):
        for key in ("volume_type", "volume_unit", "volume_semantics"):
            payload.pop(key, None)
    if not projected_fields or "real_volume" not in projected_fields:
        for key in ("real_volume_type", "real_volume_unit"):
            payload.pop(key, None)
    if projected_fields.isdisjoint({"spread", "spread_points"}):
        payload.pop("spread_estimate", None)
        payload.pop("spread_unavailable", None)
    _filter_candle_units_to_projected_fields(payload, projected_fields)
    if not bool(payload.get("forming_candle_included")):
        payload.pop("has_forming_candle", None)
        payload.pop("forming_candle_included", None)
        payload.pop("forming_candle_skipped", None)


def _filter_candle_units_to_projected_fields(
    payload: Dict[str, Any],
    projected_fields: set[str],
) -> None:
    units = payload.get("units")
    if not isinstance(units, dict):
        return
    allowed_fields = set(projected_fields)
    if "volume" in allowed_fields:
        allowed_fields.update({"tick_volume", "real_volume"})
    filtered_units = {
        key: value
        for key, value in units.items()
        if key in allowed_fields
    }
    if filtered_units:
        payload["units"] = filtered_units
    else:
        payload.pop("units", None)


def _standard_candles_payload(result: Dict[str, Any]) -> Dict[str, Any]:
    standard = _compact_candles_payload(
        result,
        include_forming_booleans=True,
    )
    public_diagnostics = _public_candle_diagnostics(result)
    for key in (
        "query_type",
        "freshness",
        "data_stale",
        "history_policy_ok",
        "usable_for_live_trading",
        "usable_for_live_trading_basis",
        "data_age_seconds",
        "data_age_anchor",
        "data_age_metric",
        "freshness_policy_relaxed",
        "market_status",
        "market_status_reason",
        "market_status_source",
        "note",
        "query_end_gap_seconds",
        "query_end_gap",
        "query_end_gap_anchor",
        "query_end_gap_metric",
        "mt5_time_alignment",
        "stale_warning",
        "spread_estimate",
        "indicator_warmup_bars",
        "history_bars_fetched",
    ):
        if key in public_diagnostics:
            standard[key] = public_diagnostics[key]
    return standard


def _attach_candle_machine_freshness(payload: Dict[str, Any]) -> None:
    public_diagnostics = _public_candle_diagnostics(payload)
    for key in (
        "query_type",
        "data_age_seconds",
        "data_stale",
        "history_policy_ok",
        "usable_for_live_trading",
        "usable_for_live_trading_basis",
        "freshness_policy_relaxed",
        "query_end_gap_seconds",
        "query_end_gap",
    ):
        if key in public_diagnostics:
            payload.setdefault(key, public_diagnostics[key])


def _summary_candles_payload(result: Dict[str, Any]) -> Dict[str, Any]:
    summary = _compact_candles_payload(
        result,
        include_forming_booleans=True,
    )
    for key, value in _public_candle_diagnostics(result).items():
        summary[key] = value
    summary["output"] = "summary"
    rows = result.get("data")
    if isinstance(rows, list) and rows:
        latest = rows[-1]
        if isinstance(latest, dict):
            summary["latest_candle"] = {
                key: latest[key]
                for key in (
                    "time",
                    "open",
                    "high",
                    "low",
                    "close",
                    "tick_volume",
                    "real_volume",
                    "spread",
                    "spread_points",
                    "bar_state",
                    "broker_session_date",
                    "broker_trading_day",
                )
                if key in latest
            }
        statistics = _candle_summary_statistics(rows)
        if statistics:
            summary["summary_statistics"] = statistics
        _attach_candle_timestamp_metadata(summary)
    summary.pop("data", None)
    summary.pop("session_gaps", None)
    for key in (
        "candles_requested",
        "candles_excluded",
        "candle_counts",
        "incomplete_candles_skipped",
    ):
        value = result.get(key)
        if value not in (None, 0, [], {}):
            summary[key] = value
    return summary


def _finite_candle_values(rows: List[Any], key: str) -> List[float]:
    values: List[float] = []
    for row in rows:
        if not isinstance(row, dict) or key not in row:
            continue
        try:
            value = float(row.get(key))
        except Exception:
            continue
        if np.isfinite(value):
            values.append(value)
    return values


def _round_candle_stat(value: float) -> float:
    rounded = round(float(value), 6)
    return 0.0 if rounded == -0.0 else rounded


def _candle_summary_statistics(rows: List[Any]) -> Dict[str, Any]:
    stats: Dict[str, Any] = {}
    for field in ("open", "high", "low", "close"):
        values = _finite_candle_values(rows, field)
        if not values:
            continue
        stats[field] = {
            "min": _round_candle_stat(min(values)),
            "max": _round_candle_stat(max(values)),
            "mean": _round_candle_stat(float(np.mean(values))),
        }

    close_values = _finite_candle_values(rows, "close")
    if len(close_values) >= 2:
        first_close = close_values[0]
        last_close = close_values[-1]
        change = last_close - first_close
        close_stats = stats.setdefault("close", {})
        close_stats["change"] = _round_candle_stat(change)
        if first_close:
            close_stats["change_pct"] = _round_candle_stat((change / first_close) * 100.0)

    high_values = _finite_candle_values(rows, "high")
    low_values = _finite_candle_values(rows, "low")
    if high_values and low_values:
        paired_ranges: List[float] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            try:
                high = float(row.get("high"))
                low = float(row.get("low"))
            except Exception:
                continue
            if np.isfinite(high) and np.isfinite(low):
                paired_ranges.append(high - low)
        if paired_ranges:
            stats["range"] = {
                "min": _round_candle_stat(min(paired_ranges)),
                "max": _round_candle_stat(max(paired_ranges)),
                "mean": _round_candle_stat(float(np.mean(paired_ranges))),
            }

    for field in ("tick_volume", "real_volume", "volume"):
        values = _finite_candle_values(rows, field)
        if values:
            stats[field] = {
                "min": _round_candle_stat(min(values)),
                "max": _round_candle_stat(max(values)),
                "mean": _round_candle_stat(float(np.mean(values))),
                "sum": _round_candle_stat(float(np.sum(values))),
            }
    return stats


def _public_candle_diagnostics(result: Dict[str, Any]) -> Dict[str, Any]:  # noqa: C901
    meta = result.get("meta")
    diagnostics = meta.get("diagnostics") if isinstance(meta, dict) else None
    if not isinstance(diagnostics, dict):
        return {}

    public: Dict[str, Any] = {}
    query = diagnostics.get("query")
    query_mode = query.get("mode") if isinstance(query, dict) else None
    if query_mode == "range":
        public["query_type"] = "historical"
    elif query_mode == "latest":
        public["query_type"] = "latest"
    if isinstance(query, dict) and query.get("latency_ms") is not None:
        public["latency_ms"] = query["latency_ms"]
    indicators = diagnostics.get("indicators")
    if isinstance(indicators, dict) and indicators.get("requested") is True:
        if isinstance(query, dict) and query.get("warmup_bars") is not None:
            public["indicator_warmup_bars"] = int(query["warmup_bars"])
        if isinstance(query, dict) and query.get("raw_bars_fetched") is not None:
            public["history_bars_fetched"] = int(query["raw_bars_fetched"])

    spread_estimate = diagnostics.get("spread_estimate")
    if isinstance(spread_estimate, dict):
        value = spread_estimate.get("estimated_mean")
        source = spread_estimate.get("source")
        unit = spread_estimate.get("unit")
        if value is not None or source:
            public_estimate: Dict[str, Any] = {}
            if value is not None:
                public_estimate["value"] = value
            if source:
                public_estimate["source"] = source
            if unit:
                public_estimate["unit"] = unit
            public["spread_estimate"] = public_estimate

    freshness = diagnostics.get("freshness")
    if isinstance(freshness, dict):
        public["freshness_basis"] = "bar_policy"
        within_policy = freshness.get("last_bar_within_policy_window")
        if freshness.get("last_bar_within_policy_window") is not None:
            public["last_bar_within_policy_window"] = bool(
                freshness["last_bar_within_policy_window"]
            )
        if "freshness_policy_relaxed" in freshness:
            public["freshness_policy_relaxed"] = normalize_policy_relaxed(
                freshness.get("freshness_policy_relaxed")
            )
        if (
            query_mode == "range"
            and result.get("range_incomplete_reason") != "forming_bar_excluded"
            and freshness.get("data_freshness_seconds") is not None
        ):
            try:
                seconds = max(0.0, float(freshness["data_freshness_seconds"]))
            except Exception:
                seconds = freshness["data_freshness_seconds"]
            public["query_end_gap_seconds"] = seconds
            public["query_end_gap_anchor"] = (
                freshness.get("data_freshness_anchor")
                or FRESHNESS_ANCHOR_QUERY_EXPECTED_END
            )
            public["query_end_gap_metric"] = (
                freshness.get("data_freshness_metric")
                or FRESHNESS_METRIC_REQUESTED_RANGE_END_GAP
            )
            gap_text = _format_age_seconds(seconds)
            if gap_text is not None:
                public["query_end_gap"] = gap_text
        elif freshness.get("data_freshness_seconds") is not None:
            try:
                seconds = max(0.0, float(freshness["data_freshness_seconds"]))
            except Exception:
                seconds = freshness["data_freshness_seconds"]
            public.setdefault("data_age_seconds", seconds)
            public["data_age_anchor"] = (
                freshness.get("data_freshness_anchor")
                or FRESHNESS_ANCHOR_WALL_CLOCK
            )
            public["data_age_metric"] = (
                freshness.get("data_freshness_metric")
                or FRESHNESS_METRIC_LAST_COMPLETED_BAR_AGE
            )
            age_text = _format_age_seconds(seconds)
            if age_text is not None:
                public["data_age"] = age_text
            relaxed_policy = normalize_policy_relaxed(
                freshness.get("freshness_policy_relaxed")
            )
            if relaxed_policy:
                public["market_status"] = (
                    freshness.get("market_session_status") or "closed_or_idle"
                )
                if freshness.get("market_session_reason"):
                    public["market_status_reason"] = freshness[
                        "market_session_reason"
                    ]
                if freshness.get("market_session_source"):
                    public["market_status_source"] = freshness[
                        "market_session_source"
                    ]
                note = freshness.get("freshness_note")
                if note:
                    public["note"] = note
            stale = (
                within_policy is not None
                and not bool(within_policy)
            )
            history_policy_ok = not stale and not relaxed_policy
            public["history_policy_ok"] = history_policy_ok
            public["data_stale"] = stale
            freshness_label = format_freshness_label(
                data_stale=stale,
                market_status=public.get("market_status"),
                market_status_reason=public.get("market_status_reason"),
                age_seconds=seconds,
                item="bar",
            )
            if freshness_label:
                public["freshness"] = freshness_label
            if stale:
                public["stale_warning"] = (
                    "Latest completed candle is outside the freshness policy window; "
                    "market may be closed or broker data may be stale."
                )
    mt5_time_alignment = diagnostics.get("mt5_time_alignment")
    if isinstance(mt5_time_alignment, dict):
        status = str(mt5_time_alignment.get("status") or "").strip().lower()
        if status and status != "ok":
            public["mt5_time_alignment"] = {
                key: mt5_time_alignment.get(key)
                for key in (
                    "status",
                    "reason",
                    "warning",
                    "probe_timeframe",
                    "timestamp_contract",
                    "tick_age_seconds",
                    "current_bar_delta_seconds",
                )
                if mt5_time_alignment.get(key) is not None
            }
    return public


def _drop_redundant_session_gap_warnings(result: Dict[str, Any]) -> None:
    if not result.get("session_gaps"):
        return
    warnings = result.get("warnings")
    if not isinstance(warnings, list):
        return
    filtered = [
        warning
        for warning in warnings
        if not (
            isinstance(warning, str)
            and (
                warning.startswith("Detected session gaps larger than expected bar spacing")
                or warning.startswith("Example gap:")
            )
        )
    ]
    if filtered:
        result["warnings"] = filtered
    else:
        result.pop("warnings", None)


def _prune_zero_candle_exclusions(result: Dict[str, Any]) -> None:
    candle_counts = result.get("candle_counts")
    if not isinstance(candle_counts, dict):
        return
    excluded = candle_counts.get("excluded")
    if not isinstance(excluded, dict):
        return
    candle_counts["excluded"] = {
        key: value
        for key, value in excluded.items()
        if key == "total" or value not in (None, 0)
    }


def _run_data_fetch_ticks_impl(
    *,
    request: DataFetchTicksRequest,
    gateway: Any,
    fetch_ticks_impl: Any,
) -> Dict[str, Any]:
    connection_error = _ensure_gateway_connection(gateway)
    if connection_error is not None:
        return connection_error
    result = fetch_ticks_impl(
        symbol=request.symbol,
        limit=request.limit,
        start=request.start,
        end=request.end,
        simplify=request.simplify,
        time_as_epoch=str(request.timestamp_format).strip().lower() != "iso",
        format=_TICK_DETAIL_FORMATS.get(request.detail, "summary"),
    )
    result = _normalize_tick_query_error(
        result,
        request=request,
        gateway=gateway,
    )
    if isinstance(result, dict):
        warnings = result.get("warnings")
        if isinstance(warnings, list):
            result["warnings"] = list(dict.fromkeys(warnings))
    if str(request.detail or "compact").strip().lower() == "compact":
        result = _compact_tick_rows_payload(result)
    _attach_tick_freshness_contract(result)
    _attach_tick_pagination(result, requested_limit=request.limit)
    return attach_mt5_source(result, gateway=gateway)


def _normalize_tick_query_error(
    result: Any,
    *,
    request: DataFetchTicksRequest,
    gateway: Any = None,
) -> Any:
    if not isinstance(result, dict) or not result.get("error"):
        return result
    if result.get("error_code"):
        return result

    message = str(result["error"])
    normalized = message.lower()
    error_code = "data_fetch_ticks_provider_failure"
    remediation = "Check the MT5 connection and broker data feed, then retry."

    if (
        ("not found" in normalized and "symbol" in normalized)
        or "failed to select symbol" in normalized
        or "unknown symbol" in normalized
    ):
        error_code = "symbol_not_found"
        message = f"Symbol '{request.symbol}' was not found in MT5."
        remediation = (
            "Use symbols_list to find the broker's exact symbol name, including "
            "any suffix or alias."
        )
    elif "could not parse" in normalized and "date" in normalized:
        error_code = "data_fetch_ticks_invalid_date"
        remediation = "Use an ISO-8601 timestamp such as 2026-07-16T12:00:00Z."
    elif "start must be before or equal to end" in normalized:
        error_code = "data_fetch_ticks_invalid_date_range"
        remediation = "Set start to a timestamp earlier than or equal to end."
    elif "start datetime" in normalized and "in the future" in normalized:
        error_code = "data_fetch_ticks_future_date_range"
        remediation = "Use a start timestamp at or before the current time."
    elif "no tick data" in normalized:
        if _tick_request_is_future_only(request):
            error_code = "data_fetch_ticks_future_date_range"
            message = (
                f"start datetime {request.start or request.end} is in the future; "
                "no historical tick data is available for future dates."
            )
            remediation = "Use a start and end timestamp at or before the current time."
        else:
            empty: Dict[str, Any] = {
                "success": True,
                "symbol": request.symbol,
                "count": 0,
                "tick_count": 0,
                "data": [],
                "empty": True,
                "empty_reason": "no_ticks_in_range",
                "timezone": "UTC",
            }
            if request.start is not None:
                empty["start"] = str(request.start)
            if request.end is not None:
                empty["end"] = str(request.end)
            return empty

    details: Dict[str, Any] = {
        "symbol": request.symbol,
        "timezone": "UTC",
    }
    if error_code == "symbol_not_found":
        details["did_you_mean"] = symbol_suggestions_from_gateway(
            gateway,
            request.symbol,
        )
    if request.start is not None:
        details["start"] = str(request.start)
    if request.end is not None:
        details["end"] = str(request.end)
    return build_error_payload(
        message,
        code=error_code,
        operation="data_fetch_ticks",
        details=details,
        remediation=remediation,
        related_tools=["symbols_list"] if error_code == "symbol_not_found" else None,
    )


def _tick_request_is_future_only(request: DataFetchTicksRequest) -> bool:
    value = request.start or request.end
    if value in (None, ""):
        return False
    try:
        parsed = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc) > datetime.now(timezone.utc)
    except (TypeError, ValueError):
        return False


def _attach_tick_pagination(payload: Any, *, requested_limit: int) -> None:
    """Echo the requested limit and disclose whether the source cap was reached."""
    if not isinstance(payload, dict) or payload.get("error"):
        return
    count = payload.get("tick_count", payload.get("count"))
    if not isinstance(count, int):
        return
    try:
        limit_value = int(requested_limit)
    except (TypeError, ValueError):
        return
    payload["requested_limit"] = limit_value
    payload["limit_reached"] = bool(count >= limit_value)


def _attach_tick_freshness_contract(payload: Any) -> None:
    if not isinstance(payload, dict) or payload.get("error"):
        return
    if payload.get("data_age_seconds") is None:
        return
    payload.setdefault("data_age_anchor", FRESHNESS_ANCHOR_WALL_CLOCK)
    payload.setdefault("data_age_metric", FRESHNESS_METRIC_LAST_TICK_AGE)


def _compact_tick_rows_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict) or payload.get("error"):
        return payload
    compact = {
        key: payload[key]
        for key in _COMPACT_TICK_TOP_LEVEL_FIELDS
        if key in payload
        and (key == "data" or payload[key] not in (None, "", [], {}))
    }
    rows = compact.get("data")
    if isinstance(rows, list):
        compact_rows: List[Any] = []
        last_spread: Optional[float] = None
        for row in rows:
            compact_row, row_spread = _compact_tick_row(
                row,
                last_spread=last_spread,
            )
            if row_spread is not None:
                last_spread = row_spread
            compact_rows.append(compact_row)
        compact["data"] = compact_rows
        compact["count"] = len(compact["data"])
        if compact.get("tick_count") == compact["count"]:
            compact.pop("tick_count", None)
        units = compact.get("units")
        present_fields = {
            key
            for row in compact["data"]
            if isinstance(row, dict)
            for key in row.keys()
        }
        compact_units = (
            {
                key: value
                for key, value in units.items()
                if key in present_fields
            }
            if isinstance(units, dict)
            else {}
        )
        for field in ("bid", "ask", "mid", "spread"):
            if any(isinstance(row, dict) and field in row for row in compact["data"]):
                compact_units.setdefault(field, "absolute_price")
        if compact_units:
            compact["units"] = compact_units
        compact["volume_fields"] = [
            field
            for field in ("volume", "volume_real")
            if field in present_fields
        ]
    quote_completeness = _tick_quote_completeness_pct(payload)
    if quote_completeness is not None:
        compact["quote_completeness_pct"] = quote_completeness
    quality = _compact_tick_quality(payload)
    if quality:
        compact["quality"] = quality
    return compact


def _tick_quote_completeness_pct(payload: Dict[str, Any]) -> Optional[float]:
    data_quality = payload.get("data_quality")
    if not isinstance(data_quality, dict):
        return None
    complete = _as_nonnegative_int(data_quality.get("complete_ticks"))
    total = _as_nonnegative_int(data_quality.get("total_ticks"))
    if complete is None or not total:
        return None
    return round((float(complete) / float(total)) * 100.0, 2)


def _compact_tick_quality(payload: Dict[str, Any]) -> Optional[str]:
    notes: List[str] = []
    data_quality = payload.get("data_quality")
    if isinstance(data_quality, dict):
        incomplete = _as_nonnegative_int(data_quality.get("incomplete_ticks"))
        total = _as_nonnegative_int(data_quality.get("total_ticks"))
        if total is None:
            total = _as_nonnegative_int(payload.get("count"))
        if incomplete is not None and incomplete > 0 and total:
            notes.append(f"partial_quotes={incomplete}/{total}")
        else:
            status = str(data_quality.get("incomplete_quote_status") or "").strip().lower()
            if status and status not in {"ok", "info"}:
                notes.append(f"quote_quality={status}")
    quote_only = payload.get("feed_tier") == "quote_only"
    if payload.get("last_unavailable") is True and not quote_only:
        notes.append("last=unavailable")
    warnings = payload.get("warnings")
    if not notes and isinstance(warnings, list) and warnings:
        notes.append(f"warnings={len(warnings)}")
    if notes:
        return "; ".join(notes)
    return "ok" if quote_only else None


def _as_nonnegative_int(value: Any) -> Optional[int]:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number >= 0 else None


def _compact_tick_row(
    row: Any,
    *,
    last_spread: Optional[float] = None,
) -> tuple[Any, Optional[float]]:
    if not isinstance(row, dict):
        return row, None
    compact = {
        "time": row.get("time"),
        "bid": row.get("bid"),
        "ask": row.get("ask"),
    }
    if row.get("quote_type") not in (None, ""):
        compact["quote_type"] = row.get("quote_type")
    spread = row.get("spread")
    if spread in (None, ""):
        spread = _tick_row_spread(row.get("bid"), row.get("ask"))
    bid = _tick_row_price(row.get("bid"))
    ask = _tick_row_price(row.get("ask"))
    numeric_spread = _tick_row_price(spread)
    spread_valid = bool(
        bid is not None
        and ask is not None
        and numeric_spread is not None
        and ask > bid
        and numeric_spread > 0.0
    )
    if spread_valid:
        compact["spread"] = numeric_spread
    if spread_valid:
        midpoint = canonical_quote_midpoint(bid, ask)
        if midpoint is not None:
            compact["mid"] = midpoint
    elif last_spread is not None and bid is not None and ask is None:
        compact["mid"] = round(bid + (last_spread / 2.0), 10)
        compact["mid_inferred"] = True
    elif last_spread is not None and ask is not None and bid is None:
        compact["mid"] = round(ask - (last_spread / 2.0), 10)
        compact["mid_inferred"] = True
    last = _tick_row_price(row.get("last"))
    if last is not None and last > 0.0:
        compact["last"] = last
    for field in ("volume", "volume_real"):
        volume = _tick_row_price(row.get(field))
        if volume is not None and volume != 0.0:
            compact[field] = volume
    decoded = row.get("flags_decoded")
    if isinstance(decoded, list) and decoded:
        quote_flags = {str(value).strip().lower() for value in decoded}
        bid_updated = "bid" in quote_flags
        ask_updated = "ask" in quote_flags
        if bid_updated != ask_updated:
            compact["quote_update_type"] = (
                "bid_only_update" if bid_updated else "ask_only_update"
            )
    elif str(row.get("quote_update_type") or "") in {
        "bid_only_update",
        "ask_only_update",
    }:
        compact["quote_update_type"] = row["quote_update_type"]
    return compact, numeric_spread if spread_valid else None


def _tick_row_spread(bid: Any, ask: Any) -> Optional[float]:
    try:
        if bid in (None, "") or ask in (None, ""):
            return None
        return round(float(ask) - float(bid), 10)
    except (TypeError, ValueError):
        return None


def _tick_row_price(value: Any) -> Optional[float]:
    try:
        if value in (None, ""):
            return None
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric):
        return None
    return numeric


def _run_wait_candle_impl(
    *,
    request: WaitCandleRequest,
    sleep_impl: Any,
) -> Result[Dict[str, Any]]:
    try:
        preview = _next_candle_wait_payload(
            request.timeframe,
            buffer_seconds=request.buffer_seconds,
        )
        max_wait_seconds = request.max_wait_seconds
        if max_wait_seconds is not None and float(preview["sleep_seconds"]) > float(max_wait_seconds):
            preview["success"] = False
            preview["status"] = "wait_budget_exceeded"
            preview["error_code"] = "wait_budget_exceeded"
            preview["error"] = (
                "The next candle boundary is beyond max_wait_seconds; no wait was "
                "performed and no candle-close event was observed."
            )
            preview["not_waited"] = True
            preview["slept"] = False
            preview["slept_seconds"] = 0.0
            preview["remaining_seconds"] = float(preview["sleep_seconds"])
            preview["max_wait_seconds"] = float(max_wait_seconds)
            preview["remediation"] = (
                "Increase max_wait_seconds beyond remaining_seconds and retry."
            )
            return Ok(preview)

        payload = _sleep_until_next_candle(
            request.timeframe,
            buffer_seconds=request.buffer_seconds,
            sleep_impl=sleep_impl,
        )
    except ValueError as exc:
        return Err(str(exc))

    payload["max_wait_seconds"] = (
        None if request.max_wait_seconds is None else float(request.max_wait_seconds)
    )
    payload["success"] = True
    return Ok(payload)


def _run_wait_event_impl(
    *,
    request: WaitEventRequest,
    gateway: Any,
    sleep_impl: Any,
    monotonic_impl: Any,
    now_utc_impl: Any,
) -> Result[Dict[str, Any]]:
    try:
        return Ok(run_wait_event_loop(
            request,
            gateway=gateway,
            sleep_impl=sleep_impl,
            monotonic_impl=monotonic_impl,
            now_utc_impl=now_utc_impl,
        ))
    except ValueError as exc:
        return Err(str(exc))


def _wait_event_needs_gateway(request: WaitEventRequest) -> bool:
    if request.max_wait_seconds is not None and not request.watch_for:
        return False
    if request.watch_for is None:
        return request.symbol is not None or bool(request.symbols)
    if request.watch_for:
        return True
    if request.symbol is not None or request.symbols:
        return True
    return any(getattr(item, "type", None) != "candle_close" for item in (request.end_on or ()))
