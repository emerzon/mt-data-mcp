

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
    """Return historical candles as CSV.
       Parameters: symbol, timeframe, limit, start?, end?, ohlcv?, indicators?, denoise?, simplify?
       Can include OHLCV data, optionally along with technical indicators.
       Returns the last candles by default, unless a date range is specified.
         Parameters:
         - symbol: The symbol to retrieve data for (e.g., "EURUSD").
         - timeframe: The timeframe to use (e.g., "H1", "M30").
         - limit: The maximum number of bars to return when not using a date range (default 10).
         - start: Optional start date (e.g., "2025-08-29" or "yesterday 14:00").
         - end: Optional end date.
         - ohlcv: Optional fields to include.
             Accepts friendly forms like: 'close', 'price', 'ohlc', 'ohlcv', 'all',
             compact 'cl' or letters 'OHLCV', or names 'open,high,low,close'.
         - indicators: Optional technical indicators to include (e.g., "rsi(20),macd(12,26,9),ema(26)")
         - denoise: Optional denoising spec to smooth selected columns either pre‑ or post‑TI
         - simplify: Optional dict to reduce or transform rows.
             keys:
               - mode: 'select' (default, select points), 'approximate' (aggregate segments),
                       'encode' (transform data), 'segment' (detect turning points), 'symbolic' (SAX transform).
               - method: (for 'select'/'approximate' modes) 'lttb' (default), 'rdp', 'pla', 'apca'.
               - points: Target number of data points for LTTB, RDP, PLA, APCA, and encode modes.
               - ratio: Alternative to points (0.0 to 1.0).
               - For 'rdp': epsilon (tolerance in y-units).
               - For 'pla'/'apca': max_error (in y-units) or 'segments'.
               - For 'encode' mode:
                 - schema: 'envelope' (OHLC -> high/low/o_pos/c_pos) or 'delta' (OHLC -> d_open/d_high/d_low/d_close).
               - For 'segment' mode:
                 - algo: 'zigzag'.
                 - threshold_pct: Reversal threshold (e.g., 0.5 for 0.5%).
               - For 'symbolic' mode:
                 - schema: 'sax'.
                 - paa: Number of PAA segments (defaults from 'points').
       The full list of supported technical indicators can be retrieved from `get_indicators`.
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
