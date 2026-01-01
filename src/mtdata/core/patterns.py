from datetime import datetime
from typing import Any, Dict, Optional, List, Tuple, Literal
import pandas as pd
import warnings
import numpy as np

from .schema import TimeframeLiteral
from .constants import TIMEFRAME_MAP
from ..utils.mt5 import _mt5_copy_rates_from
from ..utils.utils import _format_time_minimal, to_float_np as __to_float_np
from ..patterns.candlestick import detect_candlestick_patterns as _detect_candlestick_patterns
from ..patterns.classic import detect_classic_patterns as _detect_classic_patterns, ClassicDetectorConfig as _ClassicCfg
from ..patterns.eliott import detect_elliott_waves as _detect_elliott_waves, ElliottWaveConfig as _ElliottCfg
from .server import mcp, _auto_connect_wrapper
from ..utils.denoise import _apply_denoise as _apply_denoise_util, normalize_denoise_spec as _normalize_denoise_spec
import MetaTrader5 as mt5


def _round_value(x: Any) -> Any:
    """Round numeric values to 8 decimal places."""
    try:
        return float(np.round(float(x), 8))
    except Exception:
        return x


def _fetch_pattern_data(
    symbol: str,
    timeframe: str,
    limit: int,
    denoise: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, Any]]]:
    """Fetch and prepare OHLCV data for pattern detection.
    
    Returns (df, error_dict) where error_dict is None on success.
    """
    if timeframe not in TIMEFRAME_MAP:
        return None, {"error": f"Invalid timeframe: {timeframe}. Valid options: {list(TIMEFRAME_MAP.keys())}"}
    
    mt5_tf = TIMEFRAME_MAP[timeframe]
    _info = mt5.symbol_info(symbol)
    _was_visible = bool(_info.visible) if _info is not None else None
    try:
        if _was_visible is False:
            mt5.symbol_select(symbol, True)
    except Exception:
        pass
    
    utc_now = datetime.utcnow()
    count = max(400, int(limit) + 2)
    rates = _mt5_copy_rates_from(symbol, mt5_tf, utc_now, count)
    
    if rates is None or len(rates) < 100:
        return None, {"error": f"Failed to fetch sufficient bars for {symbol}"}
    
    df = pd.DataFrame(rates)
    if 'volume' not in df.columns and 'tick_volume' in df.columns:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df['volume'] = df['tick_volume']
    
    # Drop the last (potentially incomplete) bar
    if len(df) >= 2:
        df = df.iloc[:-1]
    
    # Apply denoising if requested
    if denoise:
        try:
            dn = _normalize_denoise_spec(denoise, default_when='pre_ti')
            if dn:
                _apply_denoise_util(df, dn, default_when='pre_ti')
        except Exception:
            pass
    
    # Trim to requested limit
    if len(df) > int(limit):
        df = df.iloc[-int(limit):].copy()
    
    return df, None


def _build_pattern_response(
    symbol: str,
    timeframe: str,
    limit: int,
    mode: str,
    patterns: List[Dict[str, Any]],
    include_completed: bool,
    include_series: bool,
    series_time: str,
    df: pd.DataFrame,
) -> Dict[str, Any]:
    """Build the response dict for pattern detection results."""
    # Filter patterns based on include_completed
    filtered = patterns if include_completed else [
        d for d in patterns if str(d.get('status', '')).lower() == 'forming'
    ]
    
    resp: Dict[str, Any] = {
        "success": True,
        "symbol": symbol,
        "timeframe": timeframe,
        "lookback": int(limit),
        "mode": mode,
        "patterns": filtered,
        "n_patterns": int(len(filtered)),
    }
    
    # Include series data if requested
    if include_series:
        resp["series_close"] = [float(v) for v in __to_float_np(df.get('close')).tolist()]
        if 'time' in df.columns:
            if str(series_time).lower() == 'epoch':
                resp["series_epoch"] = [float(v) for v in __to_float_np(df.get('time')).tolist()]
            else:
                resp["series_time"] = [
                    _format_time_minimal(float(v)) for v in __to_float_np(df.get('time')).tolist()
                ]
    
    return resp


def _format_pattern_dates(start_time: Optional[float], end_time: Optional[float]) -> Tuple[Optional[str], Optional[str]]:
    """Format epoch times to date strings."""
    st_epoch = float(start_time) if start_time is not None else None
    et_epoch = float(end_time) if end_time is not None else None
    
    try:
        start_date = _format_time_minimal(st_epoch) if st_epoch is not None else None
    except Exception:
        start_date = None
    
    try:
        end_date = _format_time_minimal(et_epoch) if et_epoch is not None else None
    except Exception:
        end_date = None
    
    return start_date, end_date


def _apply_config_to_obj(cfg: Any, config: Optional[Dict[str, Any]]) -> None:
    """Apply config dict values to a config object's attributes."""
    if not isinstance(config, dict):
        return
    for k, v in config.items():
        if hasattr(cfg, k):
            try:
                setattr(cfg, k, type(getattr(cfg, k))(v))
            except Exception:
                try:
                    setattr(cfg, k, v)
                except Exception:
                    pass

@mcp.tool()
@_auto_connect_wrapper
def patterns_detect(
    symbol: str,
    timeframe: TimeframeLiteral = "H1",
    mode: Literal['candlestick', 'classic', 'elliott'] = 'candlestick',  # type: ignore
    limit: int = 1000,
    # Candlestick specific
    min_strength: float = 0.95,
    min_gap: int = 3,
    robust_only: bool = True,
    whitelist: Optional[str] = None,
    top_k: int = 1,
    # Classic/Elliott specific
    denoise: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
    include_series: bool = False,
    series_time: str = "string",
    include_completed: bool = False,
) -> Dict[str, Any]:
    """Detect chart patterns (candlestick, classic chart patterns, or Elliott Wave).
    
    **REQUIRED**: symbol parameter must be provided (e.g., "EURUSD", "BTCUSD")
    
    Parameters:
    -----------
    symbol : str (REQUIRED)
        Trading symbol to analyze (e.g., "EURUSD", "GBPUSD", "BTCUSD")
    
    timeframe : str, optional (default="H1")
        Chart timeframe: "M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1", "MN1"
    
    mode : str, optional (default="candlestick")
        Pattern detection method:
        - "candlestick": Japanese candlestick patterns (Doji, Hammer, Engulfing, etc.)
        - "classic": Chart patterns (Head & Shoulders, Triangles, Flags, etc.)
        - "elliott": Elliott Wave patterns
    
    limit : int, optional (default=1000)
        Number of historical bars to analyze
    
    Candlestick Mode Parameters:
    ----------------------------
    min_strength : float, optional (default=0.95)
        Minimum pattern strength threshold (0.0 to 1.0)
    
    min_gap : int, optional (default=3)
        Minimum gap between patterns (in bars)
    
    robust_only : bool, optional (default=True)
        Only return high-confidence patterns
    
    whitelist : str, optional
        Comma-separated list of specific patterns to detect (e.g., "doji,hammer,engulfing")
    
    top_k : int, optional (default=1)
        Return only the top K strongest patterns
    
    Classic/Elliott Mode Parameters:
    ---------------------------------
    denoise : dict, optional
        Denoising configuration to smooth price data
    
    config : dict, optional
        Pattern-specific configuration parameters
    
    include_series : bool, optional (default=False)
        Include the price series data in the response
    
    series_time : str, optional (default="string")
        Time format for series data
    
    include_completed : bool, optional (default=False)
        Include only completed patterns
    
    Returns:
    --------
    dict
        Pattern detection results including:
        - success: bool
        - symbol: str
        - timeframe: str
        - patterns: list of detected patterns with metadata
    
    Examples:
    ---------
    # Detect candlestick patterns
    patterns_detect(symbol="EURUSD")
    
    # Detect candlestick patterns on M15 with custom parameters
    patterns_detect(symbol="EURUSD", timeframe="M15", min_strength=0.90, top_k=3)
    
    # Detect classic chart patterns
    patterns_detect(symbol="GBPUSD", mode="classic", limit=500)
    
    # Detect Elliott Wave patterns
    patterns_detect(symbol="BTCUSD", mode="elliott", timeframe="H4")
    """
    try:
        if mode == 'candlestick':
            return _detect_candlestick_patterns(
                symbol=symbol,
                timeframe=timeframe,
                limit=limit,
                min_strength=min_strength,
                min_gap=min_gap,
                robust_only=robust_only,
                whitelist=whitelist,
                top_k=top_k,
            )

        elif mode == 'classic':
            # Fetch and prepare data using shared helper
            df, err = _fetch_pattern_data(symbol, timeframe, limit, denoise)
            if err:
                return err

            cfg = _ClassicCfg()
            _apply_config_to_obj(cfg, config)

            pats = _detect_classic_patterns(df, cfg)

            out_list = []
            n_bars = len(df)

            def _estimate_bars_to_completion(name: str, details: Dict[str, Any], start_idx: int, end_idx: int) -> Optional[int]:
                try:
                    length = max(1, int(end_idx) - int(start_idx) + 1)
                    nm = str(name).lower()
                    if all(k in details for k in ("top_slope", "top_intercept", "bottom_slope", "bottom_intercept")):
                        s_top = float(details.get("top_slope"))
                        b_top = float(details.get("top_intercept"))
                        s_bot = float(details.get("bottom_slope"))
                        b_bot = float(details.get("bottom_intercept"))
                        denom = (s_top - s_bot)
                        if abs(denom) <= 1e-12:
                            return None
                        t_star = (b_bot - b_top) / denom
                        bars = int(max(0, int(round(t_star - (n_bars - 1)))))
                        return int(min(max(0, bars), 3 * length))
                    if all(k in details for k in ("upper_slope", "upper_intercept", "lower_slope", "lower_intercept")):
                        s_top = float(details.get("upper_slope"))
                        b_top = float(details.get("upper_intercept"))
                        s_bot = float(details.get("lower_slope"))
                        b_bot = float(details.get("lower_intercept"))
                        denom = (s_top - s_bot)
                        if abs(denom) <= 1e-12:
                            return None
                        t_star = (b_bot - b_top) / denom
                        bars = int(max(0, int(round(t_star - (n_bars - 1)))))
                        return int(min(max(0, bars), 3 * length))
                    if nm in ("pennants", "flag", "bull pennants", "bear pennants", "bull flag", "bear flag") or ("pennant" in nm or "flag" in nm):
                        return int(max(1, min(2 * length, int(round(0.3 * length)))))
                except Exception:
                    return None
                return None
            
            for p in pats:
                try:
                    start_date, end_date = _format_pattern_dates(p.start_time, p.end_time)
                    d = {
                        "name": p.name,
                        "status": p.status,
                        "confidence": float(max(0.0, min(1.0, p.confidence))),
                        "start_index": int(p.start_index),
                        "end_index": int(p.end_index),
                        "start_date": start_date,
                        "end_date": end_date,
                        "details": {k: _round_value(v) for k, v in (p.details or {}).items()},
                    }
                    if p.status == 'forming':
                        est = _estimate_bars_to_completion(p.name, d["details"], d["start_index"], d["end_index"])
                        if est is not None:
                            d["bars_to_completion"] = int(est)
                    out_list.append(d)
                except Exception:
                    continue

            return _build_pattern_response(
                symbol, timeframe, limit, mode, out_list,
                include_completed, include_series, series_time, df
            )

        elif mode == 'elliott':
            # Fetch and prepare data using shared helper
            df, err = _fetch_pattern_data(symbol, timeframe, limit, denoise)
            if err:
                return err

            cfg = _ElliottCfg()
            _apply_config_to_obj(cfg, config)

            pats = _detect_elliott_waves(df, cfg)

            out_list = []
            n_bars = len(df)
            for p in pats:
                try:
                    start_date, end_date = _format_pattern_dates(p.start_time, p.end_time)
                    recent_bars = 3
                    status = 'forming' if int(p.end_index) >= int(n_bars - recent_bars) else 'completed'

                    d = {
                        "wave_type": p.wave_type,
                        "status": status,
                        "confidence": float(max(0.0, min(1.0, p.confidence))),
                        "start_index": int(p.start_index),
                        "end_index": int(p.end_index),
                        "start_date": start_date,
                        "end_date": end_date,
                        "details": {k: _round_value(v) for k, v in (p.details or {}).items()},
                    }
                    out_list.append(d)
                except Exception:
                    continue

            return _build_pattern_response(
                symbol, timeframe, limit, mode, out_list,
                include_completed, include_series, series_time, df
            )
        
        else:
            return {"error": f"Unknown mode: {mode}"}

    except Exception as e:
        return {"error": f"Error detecting patterns: {str(e)}"}
