from datetime import datetime
from typing import Any, Dict, Optional, List, Tuple, Literal
import pandas as pd
import warnings
import numpy as np

from .schema import TimeframeLiteral
from .constants import TIMEFRAME_MAP
from ..utils.mt5 import _mt5_copy_rates_from, _mt5_epoch_to_utc
from ..utils.utils import _table_from_rows, _format_time_minimal, _format_time_minimal_local, _use_client_tz, _time_format_from_epochs, _maybe_strip_year, _style_time_format, to_float_np as __to_float_np
from ..patterns.classic import detect_classic_patterns as _detect_classic_patterns, ClassicDetectorConfig as _ClassicCfg
from ..patterns.eliott import detect_elliott_waves as _detect_elliott_waves, ElliottWaveConfig as _ElliottCfg
from .server import mcp, _auto_connect_wrapper, _ensure_symbol_ready
from ..utils.denoise import _apply_denoise as _apply_denoise_util, normalize_denoise_spec as _normalize_denoise_spec
import MetaTrader5 as mt5

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
            # Reuse the logic from original patterns_detect_candlesticks
            # We need to handle the fact that 'limit' here might be large (1000) default, 
            # but candlesticks usually want fewer. If user didn't specify, maybe clamp it?
            # But the original default was 10. Let's respect the passed limit.
            
            # ... (Logic from patterns_detect_candlesticks) ...
            # To avoid code duplication, I will inline the logic or call a helper if I had one.
            # Since I am replacing the file content, I must implement the logic here.
            
            if timeframe not in TIMEFRAME_MAP:
                return {"error": f"Invalid timeframe: {timeframe}. Valid options: {list(TIMEFRAME_MAP.keys())}"}
            mt5_timeframe = TIMEFRAME_MAP[timeframe]

            _info_before = mt5.symbol_info(symbol)
            _was_visible = bool(_info_before.visible) if _info_before is not None else None
            err = _ensure_symbol_ready(symbol)
            if err:
                return {"error": err}

            try:
                utc_now = datetime.utcnow()
                # For candlesticks, if limit is huge (default 1000), maybe reduce it if not explicitly set?
                # But we can't know if it was explicitly set easily. 
                # Let's just use it. 1000 candles is fine for TA-Lib.
                rates = _mt5_copy_rates_from(symbol, mt5_timeframe, utc_now, limit)
            finally:
                if _was_visible is False:
                    try:
                        mt5.symbol_select(symbol, False)
                    except Exception:
                        pass

            if rates is None:
                return {"error": f"Failed to get rates for {symbol}: {mt5.last_error()}"}
            if len(rates) == 0:
                return {"error": "No candle data available"}

            df = pd.DataFrame(rates)
            try:
                if 'time' in df.columns:
                    df['time'] = df['time'].astype(float).apply(_mt5_epoch_to_utc)
            except Exception:
                pass
            epochs = [float(t) for t in df['time'].tolist()] if 'time' in df.columns else []
            _use_ctz = _use_client_tz()
            if _use_ctz:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    df['time'] = df['time'].apply(_format_time_minimal_local)
            else:
                time_fmt = _time_format_from_epochs(epochs) if epochs else "%Y-%m-%d %H:%M"
                time_fmt = _maybe_strip_year(time_fmt, epochs)
                time_fmt = _style_time_format(time_fmt)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    df['time'] = df['time'].apply(lambda t: datetime.utcfromtimestamp(float(t)).strftime(time_fmt))

            for col in ['open', 'high', 'low', 'close']:
                if col not in df.columns:
                    return {"error": f"Missing '{col}' data from rates"}

            try:
                temp = df.copy()
                temp['__epoch'] = [float(e) for e in epochs]
                temp.index = pd.to_datetime(temp['__epoch'], unit='s')
            except Exception:
                temp = df.copy()

            pattern_methods: List[str] = []
            try:
                for attr in dir(temp.ta):
                    if not attr.startswith('cdl_'):
                        continue
                    func = getattr(temp.ta, attr, None)
                    if callable(func):
                        pattern_methods.append(attr)
            except Exception:
                pass

            if not pattern_methods:
                return {"error": "No candlestick pattern detectors (cdl_*) found in pandas_ta."}

            before_cols = set(temp.columns)
            for name in sorted(pattern_methods):
                try:
                    method = getattr(temp.ta, name)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        method(append=True)
                except Exception:
                    continue

            pattern_cols = [c for c in temp.columns if c not in before_cols and c.lower().startswith('cdl_')]
            if not pattern_cols:
                return {"error": "No candle patterns produced any outputs."}

            rows: List[List[Any]] = []
            try:
                thr = float(min_strength)
            except Exception:
                thr = 0.95
            if thr > 1.0:
                thr = thr / 100.0
            thr = max(0.0, min(1.0, thr))
            
            _robust_whitelist = {
                'engulfing','harami','3inside','3outside','eveningstar','morningstar',
                'darkcloudcover','piercing','inside','outside','hikkake'
            }
            if whitelist and isinstance(whitelist, str):
                try:
                    parts = [p.strip() for p in whitelist.split(',') if p.strip()]
                    if parts:
                        _robust_whitelist = {p.replace('_','').replace(' ','').lower() for p in parts}
                except Exception:
                    pass
            def _norm_name(n: str) -> str:
                return str(n).replace('_','').replace(' ','').lower()
            
            try:
                gap = max(0, int(min_gap))
            except Exception:
                gap = 3
            last_pick_idx = -10**9
            _deprioritize = {
                'shortline', 'longline', 'spinningtop', 'highwave',
                'marubozu', 'closingmarubozu', 'doji', 'gravestonedoji', 'longleggeddoji', 'rickshawman'
            }
            
            # Use limit as the tail size, but since we fetched 'limit' bars, we process all of them?
            # Original logic: fetched 'limit' bars, then processed 'limit' bars.
            # So we just iterate over the whole df.
            
            df_tail = df
            temp_tail = temp
            
            for i in range(len(temp_tail)):
                hits: List[Tuple[str, float]] = []
                for col in pattern_cols:
                    try:
                        val = float(temp_tail.iloc[i][col])
                    except Exception:
                        continue
                    if abs(val) >= (thr * 100.0):
                        name = col
                        if name.lower().startswith('cdl_'):
                            name = name[len('cdl_'):]
                        if (not robust_only) or (_norm_name(name) in _robust_whitelist):
                            hits.append((name, val))
                if not hits:
                    continue
                if i - last_pick_idx < gap:
                    continue
                non_dep = [(n, v) for (n, v) in hits if n.split('_')[0].lower() not in _deprioritize]
                pool = non_dep if non_dep else hits
                try:
                    k = max(1, int(top_k))
                except Exception:
                    k = 1
                picks = sorted(pool, key=lambda x: abs(x[1]), reverse=True)[:k]
                t_val = str(df_tail.iloc[i].get('time')) if 'time' in df_tail.columns else ''
                for name, value in picks:
                    label_core = name.replace('_', ' ').strip().upper()
                    dir_title = 'Bullish' if value > 0 else 'Bearish'
                    rows.append([t_val, f"{dir_title} {label_core}" if label_core else dir_title])
                last_pick_idx = i

            headers = ["time", "pattern"]
            payload = _table_from_rows(headers, rows)
            payload.update({
                "success": True,
                "symbol": symbol,
                "timeframe": timeframe,
                "candles": int(limit),
                "mode": mode,
            })
            if not _use_ctz:
                payload["timezone"] = "UTC"
            return payload

        elif mode == 'classic':
            # Logic from patterns_detect_classic
            if timeframe not in TIMEFRAME_MAP:
                return {"error": f"Invalid timeframe: {timeframe}. Valid options: {list(TIMEFRAME_MAP.keys())}"}
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
                return {"error": f"Failed to fetch sufficient bars for {symbol}"}
            df = pd.DataFrame(rates)
            if 'volume' not in df.columns and 'tick_volume' in df.columns:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    df['volume'] = df['tick_volume']
            if len(df) >= 2:
                df = df.iloc[:-1]
            if denoise:
                try:
                    dn = _normalize_denoise_spec(denoise, default_when='pre_ti')
                    if dn:
                        _apply_denoise_util(df, dn, default_when='pre_ti')
                except Exception:
                    pass
            if len(df) > int(limit):
                df = df.iloc[-int(limit):].copy()

            cfg = _ClassicCfg()
            if isinstance(config, dict):
                for k, v in config.items():
                    if hasattr(cfg, k):
                        try:
                            setattr(cfg, k, type(getattr(cfg, k))(v))
                        except Exception:
                            try:
                                setattr(cfg, k, v)
                            except Exception:
                                pass

            pats = _detect_classic_patterns(df, cfg)

            def _round(x):
                try:
                    return float(np.round(float(x), 8))
                except Exception:
                    return x
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
                    st_epoch = float(p.start_time) if p.start_time is not None else None
                    et_epoch = float(p.end_time) if p.end_time is not None else None
                    try:
                        start_date = _format_time_minimal(st_epoch) if st_epoch is not None else None
                    except Exception:
                        start_date = None
                    try:
                        end_date = _format_time_minimal(et_epoch) if et_epoch is not None else None
                    except Exception:
                        end_date = None
                    d = {
                        "name": p.name,
                        "status": p.status,
                        "confidence": float(max(0.0, min(1.0, p.confidence))),
                        "start_index": int(p.start_index),
                        "end_index": int(p.end_index),
                        "start_date": start_date,
                        "end_date": end_date,
                        "details": {k: _round(v) for k, v in (p.details or {}).items()},
                    }
                    if p.status == 'forming':
                        est = _estimate_bars_to_completion(p.name, d["details"], d["start_index"], d["end_index"])
                        if est is not None:
                            d["bars_to_completion"] = int(est)
                    out_list.append(d)
                except Exception:
                    continue

            filtered = out_list if include_completed else [d for d in out_list if str(d.get('status','')).lower() == 'forming']

            resp: Dict[str, Any] = {
                "success": True,
                "symbol": symbol,
                "timeframe": timeframe,
                "lookback": int(limit),
                "mode": mode,
                "patterns": filtered,
                "n_patterns": int(len(filtered)),
            }
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

        elif mode == 'elliott':
            # Logic from patterns_detect_elliott_wave
            if timeframe not in TIMEFRAME_MAP:
                return {"error": f"Invalid timeframe: {timeframe}. Valid options: {list(TIMEFRAME_MAP.keys())}"}
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
                return {"error": f"Failed to fetch sufficient bars for {symbol}"}
            df = pd.DataFrame(rates)
            if 'volume' not in df.columns and 'tick_volume' in df.columns:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    df['volume'] = df['tick_volume']
            if len(df) >= 2:
                df = df.iloc[:-1]
            if denoise:
                try:
                    dn = _normalize_denoise_spec(denoise, default_when='pre_ti')
                    if dn:
                        _apply_denoise_util(df, dn, default_when='pre_ti')
                except Exception:
                    pass
            if len(df) > int(limit):
                df = df.iloc[-int(limit):].copy()

            cfg = _ElliottCfg()
            if isinstance(config, dict):
                for k, v in config.items():
                    if hasattr(cfg, k):
                        try:
                            setattr(cfg, k, type(getattr(cfg, k))(v))
                        except Exception:
                            try:
                                setattr(cfg, k, v)
                            except Exception:
                                pass

            pats = _detect_elliott_waves(df, cfg)

            def _round(x):
                try:
                    return float(np.round(float(x), 8))
                except Exception:
                    return x
            out_list = []
            n_bars = len(df)
            for p in pats:
                try:
                    st_epoch = float(p.start_time) if p.start_time is not None else None
                    et_epoch = float(p.end_time) if p.end_time is not None else None
                    try:
                        start_date = _format_time_minimal(st_epoch) if st_epoch is not None else None
                    except Exception:
                        start_date = None
                    try:
                        end_date = _format_time_minimal(et_epoch) if et_epoch is not None else None
                    except Exception:
                        end_date = None
                    try:
                        recent_bars = 3
                    except Exception:
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
                        "details": {k: _round(v) for k, v in (p.details or {}).items()},
                    }
                    out_list.append(d)
                except Exception:
                    continue
            filtered = out_list if include_completed else [d for d in out_list if str(d.get('status','')).lower() == 'forming']

            resp: Dict[str, Any] = {
                "success": True,
                "symbol": symbol,
                "timeframe": timeframe,
                "lookback": int(limit),
                "mode": mode,
                "patterns": filtered,
                "n_patterns": int(len(filtered)),
            }
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
        
        else:
            return {"error": f"Unknown mode: {mode}"}

    except Exception as e:
        return {"error": f"Error detecting patterns: {str(e)}"}
