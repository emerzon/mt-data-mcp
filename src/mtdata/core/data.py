

from typing import Any, Dict, Optional, List
from .schema import TimeframeLiteral, IndicatorSpec, DenoiseSpec, SimplifySpec
from .server import mcp, _auto_connect_wrapper
from ..services.data_service import fetch_candles, fetch_ticks

# Explicitly define what should be exported for '*' imports
__all__ = ['data_fetch_candles', 'data_fetch_ticks']

@mcp.tool()
@_auto_connect_wrapper
def data_fetch_candles(
    symbol: str,
    timeframe: TimeframeLiteral = "H1",
    limit: int = 10,
    start: Optional[str] = None,
    end: Optional[str] = None,
    ohlcv: Optional[str] = None,
    indicators: Optional[List[IndicatorSpec]] = None,
    denoise: Optional[DenoiseSpec] = None,
    simplify: Optional[SimplifySpec] = None,
) -> Dict[str, Any]:
    """Fetch historical candle data with optional technical indicators and denoising.
    
    **REQUIRED**: symbol parameter must be provided (e.g., "EURUSD", "BTCUSD")
    
    Features:
    ---------
    - OHLCV data in CSV format
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
    
    limit : int, optional (default=10)
        Maximum number of candles to return
    
    start : str, optional
        Start date (e.g., "2025-08-29", "yesterday 14:00", "2 days ago")
    
    end : str, optional
        End date (same formats as start)
    
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
        - csv: str (CSV formatted candle data)
    
    Examples:
    ---------
    # Get last 10 H1 candles
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
    return fetch_candles(
        symbol=symbol,
        timeframe=timeframe,
        limit=limit,
        start=start,
        end=end,
        ohlcv=ohlcv,
        indicators=indicators,
        denoise=denoise,
        simplify=simplify
    )

@mcp.tool()
@_auto_connect_wrapper
def data_fetch_ticks(
    symbol: str,
    limit: int = 100,
    start: Optional[str] = None,
    end: Optional[str] = None,
    simplify: Optional[SimplifySpec] = None,
) -> Dict[str, Any]:
    """Return latest ticks as CSV with columns: time,bid,ask and optional last,volume,flags.
    Parameters: symbol, limit, start?, end?, simplify?
    - `limit` limits the number of rows.
    - `start` starts from a flexible date/time; optional `end` enables range.
    - `simplify`: Optional dict to reduce or aggregate rows (select/approximate/resample).
    """
    return fetch_ticks(
        symbol=symbol,
        limit=limit,
        start=start,
        end=end,
        simplify=simplify
    )
