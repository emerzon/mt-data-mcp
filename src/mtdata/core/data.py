
from typing import Any, Dict
import logging

from ._mcp_instance import mcp
from .data_requests import DataFetchCandlesRequest, DataFetchTicksRequest
from .data_use_cases import run_data_fetch_candles, run_data_fetch_ticks
from .execution_logging import run_logged_operation
from .mt5_gateway import create_mt5_gateway
from ..services.data_service import fetch_candles, fetch_ticks
from ..utils.mt5 import ensure_mt5_connection_or_raise

# Explicitly define what should be exported for '*' imports
__all__ = ['data_fetch_candles', 'data_fetch_ticks']

logger = logging.getLogger(__name__)


def _get_mt5_gateway():
    return create_mt5_gateway(ensure_connection_impl=ensure_mt5_connection_or_raise)

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
    
    limit : int, optional (default=25)
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
    # Get last 25 H1 candles
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
            gateway=_get_mt5_gateway(),
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
            gateway=_get_mt5_gateway(),
            fetch_ticks_impl=fetch_ticks,
        ),
    )
