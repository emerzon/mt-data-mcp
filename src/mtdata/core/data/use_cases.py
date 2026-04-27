from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict

from ...shared.result import Err, Ok, Result, to_dict
from ..execution_logging import run_logged_operation
from ..mt5_gateway import mt5_connection_error
from ..output_contract import attach_collection_contract
from ..trading.time import _next_candle_wait_payload, _sleep_until_next_candle
from .requests import (
    DataFetchCandlesRequest,
    DataFetchTicksRequest,
    WaitCandleRequest,
    WaitEventRequest,
)
from .wait_events import run_wait_event_loop

logger = logging.getLogger(__name__)


def _ensure_gateway_connection(gateway: Any) -> Dict[str, Any] | None:
    return mt5_connection_error(gateway)


def run_data_fetch_candles(
    request: DataFetchCandlesRequest,
    *,
    gateway: Any,
    fetch_candles_impl: Any,
) -> Dict[str, Any]:
    return run_logged_operation(
        logger,
        operation="data_fetch_candles",
        symbol=request.symbol,
        timeframe=request.timeframe,
        limit=request.limit,
        func=lambda: _run_data_fetch_candles_impl(
            request=request,
            gateway=gateway,
            fetch_candles_impl=fetch_candles_impl,
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
        output_mode=request.output_mode,
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
) -> Dict[str, Any]:
    connection_error = _ensure_gateway_connection(gateway)
    if connection_error is not None:
        return connection_error
    result = fetch_candles_impl(
        symbol=request.symbol,
        timeframe=request.timeframe,
        limit=request.limit,
        start=request.start,
        end=request.end,
        ohlcv=request.ohlcv,
        indicators=request.indicators,
        denoise=request.denoise,
        simplify=request.simplify,
        include_spread=request.include_spread,
        include_incomplete=request.include_incomplete,
        allow_stale=request.allow_stale,
    )
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
    if isinstance(result, dict) and detail_mode == "compact":
        result = _compact_candles_payload(result)
    if isinstance(result, dict) and isinstance(result.get("data"), list):
        out = attach_collection_contract(
            result,
            collection_kind="time_series",
            series=result["data"],
            include_contract_meta=detail_mode == "full",
        )
        if detail_mode == "full" and isinstance(out, dict):
            out.pop("data", None)
        return out
    return result


def _compact_candles_payload(result: Dict[str, Any]) -> Dict[str, Any]:
    compact = dict(result)
    for key in ("symbol", "timeframe", "candles_requested"):
        compact.pop(key, None)
    candle_counts = compact.get("candle_counts")
    if isinstance(candle_counts, dict):
        excluded = candle_counts.get("excluded")
        total_excluded = excluded.get("total") if isinstance(excluded, dict) else None
        if total_excluded in (None, 0):
            compact.pop("candle_counts", None)
    for key in ("candles_excluded", "incomplete_candles_skipped"):
        if "candle_counts" in compact or compact.get(key) in (None, 0):
            compact.pop(key, None)
    for key in ("last_candle_open", "has_forming_candle"):
        if not bool(compact.get(key)):
            compact.pop(key, None)
    return compact


def _run_data_fetch_ticks_impl(
    *,
    request: DataFetchTicksRequest,
    gateway: Any,
    fetch_ticks_impl: Any,
) -> Dict[str, Any]:
    connection_error = _ensure_gateway_connection(gateway)
    if connection_error is not None:
        return connection_error
    return fetch_ticks_impl(
        symbol=request.symbol,
        limit=request.limit,
        start=request.start,
        end=request.end,
        simplify=request.simplify,
        format=request.output_mode,
    )


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
