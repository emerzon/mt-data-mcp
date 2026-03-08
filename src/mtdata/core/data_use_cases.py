from __future__ import annotations

import logging
from typing import Any, Dict

from ..utils.mt5 import MT5ConnectionError
from .data_requests import DataFetchCandlesRequest, DataFetchTicksRequest
from .execution_logging import run_logged_operation

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
