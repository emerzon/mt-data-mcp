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
    normalize_policy_relaxed,
)
from ..error_envelope import build_error_payload
from ..execution_logging import run_logged_operation
from ..mt5_gateway import mt5_connection_error
from ..output_contract import attach_collection_contract
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
    "data",
    "timezone",
    "price_precision",
    "price_point",
    "price_currency",
    "units",
    "freshness",
    "freshness_state",
    "data_age_seconds",
    "data_age_anchor",
    "data_age_metric",
    "data_stale",
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
)

_ANALYSIS_CANDLE_DEFAULT_LIMIT = 100


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
            isinstance(r, Ok) or (isinstance(r, dict) and "error" not in r)
        ),
    )
    return to_dict(result) if isinstance(result, (Ok, Err)) else result


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
    result = _normalize_candle_query_error(result, request=request)
    # Detect missing or all-zero spread when include_spread requested.
    # MT5 often only reports spread at tick level; aggregated higher-timeframe bars may show spread==0.
    if isinstance(result, dict) and getattr(request, "include_spread", False):
        data = result.get("data")
        if isinstance(data, list) and len(data) > 0:
            has_spread_values = False
            spread_all_zero = True
            for bar in data:
                if isinstance(bar, dict):
                    if "spread" in bar and bar.get("spread") is not None:
                        has_spread_values = True
                        try:
                            if float(bar.get("spread", 0)) != 0.0:
                                spread_all_zero = False
                                break
                        except Exception:
                            # non-numeric spread; treat as available
                            has_spread_values = True
                            spread_all_zero = False
                            break
                # If bars are lists/sequences, skip detection here.
            if not has_spread_values:
                # No spread values present at all
                result.setdefault("warnings", []).append(
                    "include_spread requested but returned bars do not contain 'spread' values; spread unavailable at this timeframe or source."
                )
                result["spread_unavailable"] = True
            elif spread_all_zero:
                result.setdefault("warnings", []).append(
                    "include_spread requested but all returned spread values are zero; spread likely unavailable at this timeframe/source."
                )
                result["spread_unavailable"] = True
    detail_mode = str(request.detail or "compact").strip().lower()
    if isinstance(result, dict):
        if bool(getattr(request, "explain_indicators", False)):
            _attach_indicator_explanations(result)
        _apply_range_limit_cap(
            result,
            limit=effective_limit if effective_limit is not None else request.limit,
        )
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


def _normalize_candle_query_error(
    result: Any,
    *,
    request: DataFetchCandlesRequest,
) -> Any:
    if not isinstance(result, dict) or not result.get("error"):
        return result
    if result.get("error_code"):
        return result

    message = str(result["error"])
    normalized = message.lower()
    error_code: Optional[str] = None
    remediation: Optional[str] = None

    if "not found" in normalized and "symbol" in normalized:
        error_code = "symbol_not_found"
        remediation = (
            "Use the broker's exact MT5 symbol name; call market_ticker for symbol "
            "discovery when the broker uses suffixes or aliases."
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


def _effective_candle_limit(request: DataFetchCandlesRequest) -> int:
    try:
        limit = max(1, int(request.limit))
    except Exception:
        limit = DATA_FETCH_CANDLES_DEFAULT_LIMIT
    fields_set = getattr(request, "model_fields_set", set())
    limit_explicit = "limit" in fields_set
    has_indicators = request.indicators not in (None, "", [], {})
    if has_indicators and not limit_explicit:
        return max(limit, _ANALYSIS_CANDLE_DEFAULT_LIMIT)
    return limit


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


def _apply_range_limit_cap(result: Dict[str, Any], *, limit: int) -> None:
    data = result.get("data")
    if not isinstance(data, list):
        return
    meta = result.get("meta")
    diagnostics = meta.get("diagnostics") if isinstance(meta, dict) else None
    query = diagnostics.get("query") if isinstance(diagnostics, dict) else None
    if not isinstance(query, dict) or query.get("mode") != "range":
        return
    try:
        limit_value = max(1, int(limit))
    except Exception:
        return
    available = len(data)
    if available <= limit_value:
        return

    retained = data[-limit_value:]
    result["data"] = retained
    result["count"] = len(retained)
    result["available_count"] = available
    result["limit_applied"] = limit_value
    result["truncated"] = True
    result["truncation"] = {
        "reason": "limit",
        "retained": "last",
        "excluded_count": available - len(retained),
    }
    result.setdefault("warnings", []).append(
        f"Range contained {available} bars; returned the latest {len(retained)} "
        f"because limit={limit_value}. Set limit>={available} to return the full range."
    )
    candle_counts = result.get("candle_counts")
    if isinstance(candle_counts, dict):
        candle_counts["returned"] = len(retained)
        excluded = candle_counts.get("excluded")
        if not isinstance(excluded, dict):
            excluded = {}
            candle_counts["excluded"] = excluded
        excluded["limit_truncated"] = max(0, available - len(retained))
        excluded["total"] = int(excluded.get("total") or 0) + max(0, available - len(retained))
    query["limit_applied_to_range"] = True
    query["available_rows_before_limit"] = available
    query["returned_rows_after_limit"] = len(retained)


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
    public_diagnostics = _public_candle_diagnostics(result)
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
        "time_basis",
        "time_normalization",
        "timestamp_mode",
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
        compact.pop("forming_candle_status", None)
        compact.pop("forming_candle_included", None)
        compact.pop("forming_candle_skipped", None)
    elif not include_forming_booleans:
        compact.pop("has_forming_candle", None)
        compact.pop("forming_candle_included", None)
        compact.pop("forming_candle_skipped", None)
    if result.get("forming_candle_status") == "skipped" and result.get("hint"):
        compact["hint"] = result["hint"]
    _attach_candle_timestamp_metadata(compact)
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
    if "spread" not in projected_fields:
        payload.pop("spread_estimate", None)
        payload.pop("spread_unavailable", None)
    _filter_candle_units_to_projected_fields(payload, projected_fields)
    if not bool(payload.get("forming_candle_included")):
        payload.pop("forming_candle_status", None)
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
                for key in ("time", "open", "high", "low", "close", "tick_volume", "real_volume")
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


def _public_candle_diagnostics(result: Dict[str, Any]) -> Dict[str, Any]:
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
        if query_mode == "range" and freshness.get("data_freshness_seconds") is not None:
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
    if str(request.detail or "compact").strip().lower() == "compact":
        result = _compact_tick_rows_payload(result)
    _attach_tick_freshness_contract(result)
    _attach_tick_pagination(result, requested_limit=request.limit)
    return result


def _attach_tick_pagination(payload: Any, *, requested_limit: int) -> None:
    """Echo the requested limit and disclose whether the source cap was reached."""
    if not isinstance(payload, dict) or payload.get("error"):
        return
    count = payload.get("count")
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
        if key in payload and payload[key] not in (None, "", [], {})
    }
    rows = compact.get("data")
    price_point = _tick_price_point(payload)
    if isinstance(rows, list):
        compact_rows: List[Any] = []
        last_spread: Optional[float] = None
        for row in rows:
            compact_row, row_spread = _compact_tick_row(
                row,
                last_spread=last_spread,
                price_point=price_point,
            )
            if row_spread is not None:
                last_spread = row_spread
            compact_rows.append(compact_row)
        compact["data"] = compact_rows
        compact["count"] = len(compact["data"])
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
        for field, unit in (
            ("spread_points", "broker_points"),
            ("spread_pips", "pips"),
            ("spread_pct", "percentage_points (1.0 = 1%)"),
        ):
            if any(isinstance(row, dict) and field in row for row in compact["data"]):
                compact_units.setdefault(field, unit)
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
    if payload.get("last_unavailable") is True:
        notes.append("last=unavailable")
    warnings = payload.get("warnings")
    if not notes and isinstance(warnings, list) and warnings:
        notes.append(f"warnings={len(warnings)}")
    return "; ".join(notes) if notes else None


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
    price_point: Optional[float] = None,
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
    compact["spread"] = spread if spread not in ("",) else None
    bid = _tick_row_price(row.get("bid"))
    ask = _tick_row_price(row.get("ask"))
    numeric_spread = _tick_row_price(spread)
    if bid is not None and ask is not None:
        compact["mid"] = round((bid + ask) / 2.0, 10)
    elif last_spread is not None and bid is not None:
        compact["mid"] = round(bid + (last_spread / 2.0), 10)
        compact["mid_inferred"] = True
    elif last_spread is not None and ask is not None:
        compact["mid"] = round(ask - (last_spread / 2.0), 10)
        compact["mid_inferred"] = True
    row_spread_points = _tick_row_price(row.get("spread_points"))
    if row_spread_points is not None:
        compact["spread_points"] = row_spread_points
    elif numeric_spread is not None and price_point is not None and price_point > 0.0:
        compact["spread_points"] = round(numeric_spread / price_point, 4)
    row_spread_pct = _tick_row_price(row.get("spread_pct"))
    if row_spread_pct is not None:
        compact["spread_pct"] = row_spread_pct
    elif numeric_spread is not None:
        spread_mid = _tick_row_price(compact.get("mid"))
        if spread_mid is not None and spread_mid > 0.0:
            compact["spread_pct"] = round((numeric_spread / spread_mid) * 100.0, 6)
    last = _tick_row_price(row.get("last"))
    if last is not None and last > 0.0:
        compact["last"] = last
    for field in ("volume", "volume_real"):
        volume = _tick_row_price(row.get(field))
        if volume is not None and volume != 0.0:
            compact[field] = volume
    flags = _as_nonnegative_int(row.get("flags"))
    if flags is not None:
        compact["flags"] = flags
    decoded = row.get("flags_decoded")
    if isinstance(decoded, list) and decoded:
        compact["flags_decoded"] = list(decoded)
        quote_flags = {str(value).strip().lower() for value in decoded}
        bid_updated = "bid" in quote_flags
        ask_updated = "ask" in quote_flags
        if bid_updated != ask_updated:
            compact["quote_update_type"] = (
                "bid_only_update" if bid_updated else "ask_only_update"
            )
            compact["spread_valid"] = False
            compact["ask" if bid_updated else "bid"] = None
            compact["spread"] = None
            for field in ("mid", "mid_inferred", "spread_points", "spread_pct"):
                compact.pop(field, None)
            numeric_spread = None
        elif bid_updated and ask_updated:
            compact["quote_update_type"] = "bid_ask_update"
            compact["spread_valid"] = True
    return compact, numeric_spread


def _tick_price_point(payload: Dict[str, Any]) -> Optional[float]:
    point = _tick_row_price(payload.get("price_point"))
    if point is not None and point > 0.0:
        return point
    return None


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
        if _wait_event_needs_gateway(request):
            connection_error = _ensure_gateway_connection(gateway)
            if connection_error is not None:
                return Err(
                    str(connection_error.get("error", "MT5 connection failed")),
                    code="MT5_CONNECTION",
                )
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
    if request.watch_for is None:
        return True
    if request.watch_for:
        return True
    return any(getattr(item, "type", None) != "candle_close" for item in (request.end_on or ()))
