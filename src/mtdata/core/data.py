
from typing import Any, Dict
import logging

from ._mcp_instance import mcp
from .data_requests import (
    DataFetchCandlesRequest,
    DataFetchTicksRequest,
    WaitEventRequest,
)
from .schema import TimeframeLiteral
from .data_use_cases import (
    run_data_fetch_candles,
    run_data_fetch_ticks,
    run_wait_event,
)
from .execution_logging import run_logged_operation
from .mt5_gateway import get_mt5_gateway
from ..services.data_service import fetch_candles, fetch_ticks
from ..utils.mt5 import ensure_mt5_connection_or_raise

# Explicitly define what should be exported for '*' imports
__all__ = ['data_fetch_candles', 'data_fetch_ticks', 'wait_event']

logger = logging.getLogger(__name__)


@mcp.tool()
def data_fetch_candles(
    request: DataFetchCandlesRequest,
) -> Dict[str, Any]:
    """Fetch historical candle data with optional technical indicators and denoising.
    
    **REQUIRED**: symbol parameter must be provided (e.g., "EURUSD", "BTCUSD")
    
    Features:
    ---------
    - OHLCV data as tabular rows
    - Technical indicators (RSI, MACD, EMA, SMA, etc.)
    - Data denoising and smoothing
    - Data simplification for large datasets
    - Includes metadata: last_candle_open (true if last candle is still forming)
    
    Parameters:
    -----------
    symbol : str (REQUIRED)
        Trading symbol (e.g., "EURUSD", "GBPUSD", "BTCUSD")
    
    timeframe : str, optional (default="H1")
        Candle timeframe: "M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1", "MN1"
    
    limit : int, optional (default=200)
        Maximum number of candles to return
    
    start : str, optional
        Start time (dateparser)

    end : str, optional
        End time (dateparser)
    
    ohlcv : str, optional
        Fields to include: "close", "ohlc", "ohlcv", "all"
    
    indicators : list, optional
        Technical indicators list, e.g., [{"name": "rsi", "params": [14]}]
        Or compact string: "rsi(14),ema(20),macd(12,26,9)"
    
    denoise : dict, optional
        Denoising configuration to smooth price data
    
    simplify : dict, optional
        Data reduction options for large datasets
    
    Returns:
    --------
    dict
        - success: bool
        - symbol: str
        - timeframe: str
        - candles: int (number of candles returned)
        - last_candle_open: bool (true if last candle is still forming)
        - data: list[dict] (tabular candle rows)
    
    Examples:
    ---------
    # Get last 200 H1 candles
    data_fetch_candles(symbol="EURUSD")
    
    # Get 100 M15 candles with RSI indicator
    data_fetch_candles(
        symbol="EURUSD",
        timeframe="M15",
        limit=100,
        indicators="rsi(14)"
    )
    
    # Get date range with multiple indicators
    data_fetch_candles(
        symbol="GBPUSD",
        start="2025-11-01",
        end="2025-11-30",
        indicators="rsi(14),ema(20),macd(12,26,9)"
    )
    """
    return run_logged_operation(
        logger,
        operation="data_fetch_candles",
        symbol=request.symbol,
        timeframe=request.timeframe,
        limit=request.limit,
        func=lambda: run_data_fetch_candles(
            request,
            gateway=get_mt5_gateway(ensure_connection_impl=ensure_mt5_connection_or_raise),
            fetch_candles_impl=fetch_candles,
        ),
    )

@mcp.tool()
def data_fetch_ticks(
    request: DataFetchTicksRequest,
) -> Dict[str, Any]:
    """Fetch tick data for a symbol.

    By default (`output="summary"`), returns a compact set of descriptive stats
    over the fetched ticks (bid/ask/mid/spread, plus last and volume; volume uses real
    volume when available, otherwise tick_volume).

    Use `output="stats"` for a more detailed stats payload.
    Use `output="rows"` to return raw tick rows as structured data.
    `simplify` only applies to row output.
    """
    return run_logged_operation(
        logger,
        operation="data_fetch_ticks",
        symbol=request.symbol,
        limit=request.limit,
        output=request.output,
        func=lambda: run_data_fetch_ticks(
            request,
            gateway=get_mt5_gateway(ensure_connection_impl=ensure_mt5_connection_or_raise),
            fetch_ticks_impl=fetch_ticks,
        ),
    )


@mcp.tool()
def wait_event(
    instrument: str,
    timeframe: TimeframeLiteral = "M1",
    watch_tick_count_spike: bool = True,
) -> Dict[str, Any]:
    """Wait for position lifecycle events on an instrument until the next timeframe boundary.

    The public tool focuses on position state changes for the instrument:
    `position_opened`, `position_closed`, `tp_hit`, and `sl_hit`.
    By default it also wakes on a significant tick-count activity surge
    for the instrument; set `watch_tick_count_spike=false` to disable it.
    It stops at the next candle close for `timeframe` if no watched event
    happens first.
    """
    watch_for = [
        {"type": "position_opened", "symbol": instrument},
        {"type": "position_closed", "symbol": instrument},
        {"type": "tp_hit", "symbol": instrument},
        {"type": "sl_hit", "symbol": instrument},
    ]
    if watch_tick_count_spike:
        watch_for.append({"type": "tick_count_spike", "symbol": instrument})

    request = WaitEventRequest(
        symbol=instrument,
        timeframe=timeframe,
        watch_for=watch_for,
    )
    def _run() -> Dict[str, Any]:
        result = run_wait_event(
            request,
            gateway=get_mt5_gateway(ensure_connection_impl=ensure_mt5_connection_or_raise),
        )
        if isinstance(result, dict):
            result = dict(result)
            result.pop("max_wait_seconds", None)
        return result

    return run_logged_operation(
        logger,
        operation="wait_event",
        instrument=instrument,
        timeframe=timeframe,
        watch_tick_count_spike=watch_tick_count_spike,
        func=_run,
    )
