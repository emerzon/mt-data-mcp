from __future__ import annotations

from typing import Any, Dict

from ..utils.mt5 import MT5ConnectionError
from .data_requests import DataFetchCandlesRequest, DataFetchTicksRequest


def run_data_fetch_candles(
    request: DataFetchCandlesRequest,
    *,
    ensure_connection: Any,
    fetch_candles_impl: Any,
) -> Dict[str, Any]:
    try:
        ensure_connection()
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


def run_data_fetch_ticks(
    request: DataFetchTicksRequest,
    *,
    ensure_connection: Any,
    fetch_ticks_impl: Any,
) -> Dict[str, Any]:
    try:
        ensure_connection()
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
