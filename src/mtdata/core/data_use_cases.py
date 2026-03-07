from __future__ import annotations

import logging
import time
from typing import Any, Dict

from ..utils.mt5 import MT5ConnectionError
from .data_requests import DataFetchCandlesRequest, DataFetchTicksRequest
from .execution_logging import infer_result_success, log_operation_finish, log_operation_start

logger = logging.getLogger(__name__)


def run_data_fetch_candles(
    request: DataFetchCandlesRequest,
    *,
    gateway: Any,
    fetch_candles_impl: Any,
) -> Dict[str, Any]:
    started_at = time.perf_counter()
    log_operation_start(
        logger,
        operation="data_fetch_candles",
        symbol=request.symbol,
        timeframe=request.timeframe,
        limit=request.limit,
    )
    try:
        gateway.ensure_connection()
    except MT5ConnectionError as exc:
        result = {"error": str(exc)}
        log_operation_finish(
            logger,
            operation="data_fetch_candles",
            started_at=started_at,
            success=False,
            symbol=request.symbol,
            timeframe=request.timeframe,
            limit=request.limit,
        )
        return result
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
    )
    log_operation_finish(
        logger,
        operation="data_fetch_candles",
        started_at=started_at,
        success=infer_result_success(result),
        symbol=request.symbol,
        timeframe=request.timeframe,
        limit=request.limit,
    )
    return result


def run_data_fetch_ticks(
    request: DataFetchTicksRequest,
    *,
    gateway: Any,
    fetch_ticks_impl: Any,
) -> Dict[str, Any]:
    started_at = time.perf_counter()
    log_operation_start(
        logger,
        operation="data_fetch_ticks",
        symbol=request.symbol,
        limit=request.limit,
        output=request.output,
    )
    try:
        gateway.ensure_connection()
    except MT5ConnectionError as exc:
        result = {"error": str(exc)}
        log_operation_finish(
            logger,
            operation="data_fetch_ticks",
            started_at=started_at,
            success=False,
            symbol=request.symbol,
            limit=request.limit,
            output=request.output,
        )
        return result
    result = fetch_ticks_impl(
        symbol=request.symbol,
        limit=request.limit,
        start=request.start,
        end=request.end,
        simplify=request.simplify,
        output=request.output,
    )
    log_operation_finish(
        logger,
        operation="data_fetch_ticks",
        started_at=started_at,
        success=infer_result_success(result),
        symbol=request.symbol,
        limit=request.limit,
        output=request.output,
    )
    return result
