from __future__ import annotations

import logging
import time
from typing import Any, Dict

from ..utils.mt5 import MT5ConnectionError
from .data_requests import DataFetchCandlesRequest, DataFetchTicksRequest, WaitCandleRequest
from .execution_logging import run_logged_operation
from .trading_time import _next_candle_wait_payload, _sleep_until_next_candle

logger = logging.getLogger(__name__)


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
        output=request.output,
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
    return run_logged_operation(
        logger,
        operation="wait_candle",
        timeframe=request.timeframe,
        buffer_seconds=request.buffer_seconds,
        func=lambda: _run_wait_candle_impl(
            request=request,
            sleep_impl=sleep_impl,
        ),
    )


def _run_data_fetch_candles_impl(
    *,
    request: DataFetchCandlesRequest,
    gateway: Any,
    fetch_candles_impl: Any,
) -> Dict[str, Any]:
    try:
        gateway.ensure_connection()
    except MT5ConnectionError as exc:
        return {"error": str(exc)}
    return fetch_candles_impl(
        symbol=request.symbol,
        timeframe=request.timeframe,
        limit=request.limit,
        start=request.start,
        end=request.end,
        ohlcv=request.ohlcv,
        indicators=request.indicators,
        denoise=request.denoise,
        simplify=request.simplify,
    )


def _run_data_fetch_ticks_impl(
    *,
    request: DataFetchTicksRequest,
    gateway: Any,
    fetch_ticks_impl: Any,
) -> Dict[str, Any]:
    try:
        gateway.ensure_connection()
    except MT5ConnectionError as exc:
        return {"error": str(exc)}
    return fetch_ticks_impl(
        symbol=request.symbol,
        limit=request.limit,
        start=request.start,
        end=request.end,
        simplify=request.simplify,
        output=request.output,
    )


def _run_wait_candle_impl(
    *,
    request: WaitCandleRequest,
    sleep_impl: Any,
) -> Dict[str, Any]:
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
            return preview

        payload = _sleep_until_next_candle(
            request.timeframe,
            buffer_seconds=request.buffer_seconds,
            sleep_impl=sleep_impl,
        )
    except ValueError as exc:
        return {"error": str(exc)}

    payload["max_wait_seconds"] = (
        None if request.max_wait_seconds is None else float(request.max_wait_seconds)
    )
    payload["success"] = True
    return payload
