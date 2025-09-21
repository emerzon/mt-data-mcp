
from datetime import datetime
from typing import Any, Dict, Optional, List, Tuple, Set
import pandas as pd
import warnings
import numpy as np

from .schema import TimeframeLiteral
from .constants import TIMEFRAME_MAP
from ..utils.mt5 import _mt5_copy_rates_from, _mt5_epoch_to_utc
from ..utils.utils import _csv_from_rows_util, _format_time_minimal_util, _format_time_minimal_local_util, _use_client_tz_util, _time_format_from_epochs_util, _maybe_strip_year_util, _style_time_format_util, to_float_np as __to_float_np
from ..patterns.classic import detect_classic_patterns as _detect_classic_patterns, ClassicDetectorConfig as _ClassicCfg
from ..patterns.eliott import detect_elliott_waves as _detect_elliott_waves, ElliottWaveConfig as _ElliottCfg
from .server import mcp, _auto_connect_wrapper, _ensure_symbol_ready
from ..utils.denoise import _apply_denoise as _apply_denoise_util, normalize_denoise_spec as _normalize_denoise_spec
import MetaTrader5 as mt5

@mcp.tool()
@_auto_connect_wrapper
def patterns_detect_candlesticks(
    symbol: str,
    timeframe: TimeframeLiteral = "H1",
    limit: int = 10,
    min_strength: float = 0.95,
    min_gap: int = 3,
    robust_only: bool = True,
    whitelist: Optional[str] = None,  # comma-separated names, e.g. "engulfing,harami"
    top_k: int = 1,
) -> Dict[str, Any]:
    """Detect candlestick patterns and return CSV rows of detections.
    Parameters: symbol, timeframe, limit

    Inputs:
    - `symbol`: Trading symbol (e.g., "EURUSD").
    - `timeframe`: One of the supported MT5 timeframes (e.g., "M15", "H1").
    - `limit`: Number of most recent candles to analyze.

    Output CSV columns:
    - `time`: UTC timestamp of the bar (compact format)
    - `pattern`: Human-friendly pattern label (includes direction, e.g., "Bearish ENGULFING BEAR")
    """
    try:
        # Validate timeframe
        if timeframe not in TIMEFRAME_MAP:
            return {"error": f"Invalid timeframe: {timeframe}. Valid options: {list(TIMEFRAME_MAP.keys())}"}
        mt5_timeframe = TIMEFRAME_MAP[timeframe]

        # Ensure symbol is ready; remember original visibility to restore later
        _info_before = mt5.symbol_info(symbol)
        _was_visible = bool(_info_before.visible) if _info_before is not None else None
        err = _ensure_symbol_ready(symbol)
        if err:
            return {"error": err}

        try:
            # Fetch last `limit` bars from now (UTC anchor)
            utc_now = datetime.utcnow()
            rates = _mt5_copy_rates_from(symbol, mt5_timeframe, utc_now, limit)
        finally:
            # Restore original visibility if we changed it
            if _was_visible is False:
                try:
                    mt5.symbol_select(symbol, False)
                except Exception:
                    pass

        if rates is None:
            return {"error": f"Failed to get rates for {symbol}: {mt5.last_error()}"}
        if len(rates) == 0:
            return {"error": "No candle data available"}

        # Build DataFrame and format time
        df = pd.DataFrame(rates)
        # Normalize epochs to UTC if server offset configured
        try:
            if 'time' in df.columns:
                df['time'] = df['time'].astype(float).apply(_mt5_epoch_to_utc)
        except Exception:
            pass
        epochs = [float(t) for t in df['time'].tolist()] if 'time' in df.columns else []
        _use_ctz = _use_client_tz_util()
        if _use_ctz:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                df['time'] = df['time'].apply(_format_time_minimal_local_util)
        else:
            time_fmt = _time_format_from_epochs_util(epochs) if epochs else "%Y-%m-%d %H:%M"
            time_fmt = _maybe_strip_year_util(time_fmt, epochs)
            time_fmt = _style_time_format_util(time_fmt)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                df['time'] = df['time'].apply(lambda t: datetime.utcfromtimestamp(float(t)).strftime(time_fmt))

        # Ensure required OHLC columns exist
        for col in ['open', 'high', 'low', 'close']:
            if col not in df.columns:
                return {"error": f"Missing '{col}' data from rates"}

        # Prepare temp DataFrame with DatetimeIndex for pandas_ta compatibility
        try:
            temp = df.copy()
            temp['__epoch'] = [float(e) for e in epochs]
            temp.index = pd.to_datetime(temp['__epoch'], unit='s')
        except Exception:
            temp = df.copy()

        # Discover callable cdl_* methods
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
            return {"error": "No candlestick pattern detectors (cdl_*) found in pandas_ta. Ensure pandas_ta (and TA-Lib if required) are installed."}

        before_cols = set(temp.columns)
        for name in sorted(pattern_methods):
            try:
                method = getattr(temp.ta, name)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    method(append=True)
            except Exception:
                continue

        # Identify newly added pattern columns
        pattern_cols = [c for c in temp.columns if c not in before_cols and c.lower().startswith('cdl_')]
        if not pattern_cols:
            return {"error": "No candle patterns produced any outputs."}

        # Compile detection rows per candle (choose at most one pattern per bar)
        rows: List[List[Any]] = []
        # Threshold (accept 0..1 or 0..100 scales)
        try:
            thr = float(min_strength)
        except Exception:
            thr = 0.95
        if thr > 1.0:
            thr = thr / 100.0
        thr = max(0.0, min(1.0, thr))
        # Robust multi-candle patterns (default whitelist)
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
        # Enforce minimal spacing between consecutive signal bars
        try:
            gap = max(0, int(min_gap))
        except Exception:
            gap = 3
        last_pick_idx = -10**9
        # Patterns to deprioritize when more specific ones co-occur
        _deprioritize = {
            'shortline', 'longline', 'spinningtop', 'highwave',
            'marubozu', 'closingmarubozu', 'doji', 'gravestonedoji', 'longleggeddoji', 'rickshawman'
        }
        # Work on the last `limit` bars
        try:
            tail_n = max(1, int(limit))
        except Exception:
            tail_n = int(limit) if isinstance(limit, int) else 10
        df_tail = df.iloc[-tail_n:]
        temp_tail = temp.iloc[-tail_n:]
        for i in range(len(temp_tail)):
            # Gather all hits at this bar
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
            # Respect minimal spacing across candles
            if i - last_pick_idx < gap:
                continue
            # Prefer non-deprioritized patterns by absolute strength
            non_dep = [(n, v) for (n, v) in hits if n.split('_')[0].lower() not in _deprioritize]
            pool = non_dep if non_dep else hits
            # Select up to top_k by absolute strength
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
        payload = _csv_from_rows_util(headers, rows)
        payload.update({
            "success": True,
            "symbol": symbol,
            "timeframe": timeframe,
            "candles": int(limit),
        })
        if not _use_ctz:
            payload["timezone"] = "UTC"
        return payload
    except Exception as e:
        return {"error": f"Error detecting candlestick patterns: {str(e)}"}

@mcp.tool()
@_auto_connect_wrapper
def patterns_detect_classic(
    symbol: str,
    timeframe: TimeframeLiteral = "H1",
    lookback: int = 1500,
    denoise: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
    include_series: bool = False,
    series_time: str = "string",  # 'string' or 'epoch'
    include_completed: bool = False,
) -> Dict[str, Any]:
    """Detect classic chart patterns (triangles, flags, wedges, H&S, channels, rectangles, etc.).

    - Pulls last `lookback` bars for `symbol`/`timeframe`.
    - Applies optional denoise.
    - Returns a list of patterns with status (completed|forming), confidence, time bounds, and pattern-specific levels.
    """
    try:
        if timeframe not in TIMEFRAME_MAP:
            return {"error": f"Invalid timeframe: {timeframe}. Valid options: {list(TIMEFRAME_MAP.keys())}"}
        mt5_tf = TIMEFRAME_MAP[timeframe]
        # Ensure symbol is visible
        _info = mt5.symbol_info(symbol)
        _was_visible = bool(_info.visible) if _info is not None else None
        try:
            if _was_visible is False:
                mt5.symbol_select(symbol, True)
        except Exception:
            pass
        # Fetch bars
        utc_now = datetime.utcnow()
        count = max(400, int(lookback) + 2)
        rates = _mt5_copy_rates_from(symbol, mt5_tf, utc_now, count)
        if rates is None or len(rates) < 100:
            return {"error": f"Failed to fetch sufficient bars for {symbol}"}
        df = pd.DataFrame(rates)
        if 'volume' not in df.columns and 'tick_volume' in df.columns:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                df['volume'] = df['tick_volume']
        # Drop forming last bar for stability
        if len(df) >= 2:
            df = df.iloc[:-1]
        if denoise:
            try:
                dn = _normalize_denoise_spec(denoise, default_when='pre_ti')
                if dn:
                    _apply_denoise_util(df, dn, default_when='pre_ti')
            except Exception:
                pass
        # Clip to lookback
        if len(df) > int(lookback):
            df = df.iloc[-int(lookback):].copy()

        # Build config
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

        # Detect
        pats = _detect_classic_patterns(df, cfg)

        # Serialize
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
                # Triangles/Wedges with explicit line params
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
                    # Clamp to a reasonable window
                    return int(min(max(0, bars), 3 * length))
                # Channels with explicit line params (rarely converging)
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
                # Flags/Pennants: heuristic fraction of current consolidation length
                if nm in ("pennants", "flag", "bull pennants", "bear pennants", "bull flag", "bear flag") or ("pennant" in nm or "flag" in nm):
                    return int(max(1, min(2 * length, int(round(0.3 * length)))))
            except Exception:
                return None
            return None
        for p in pats:
            try:
                # Format times per global time format
                st_epoch = float(p.start_time) if p.start_time is not None else None
                et_epoch = float(p.end_time) if p.end_time is not None else None
                try:
                    start_date = _format_time_minimal_util(st_epoch) if st_epoch is not None else None
                except Exception:
                    start_date = None
                try:
                    end_date = _format_time_minimal_util(et_epoch) if et_epoch is not None else None
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

        # Optionally filter out completed patterns by default
        filtered = out_list if include_completed else [d for d in out_list if str(d.get('status','')).lower() == 'forming']

        resp: Dict[str, Any] = {
            "success": True,
            "symbol": symbol,
            "timeframe": timeframe,
            "lookback": int(lookback),
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
                        _format_time_minimal_util(float(v)) for v in __to_float_np(df.get('time')).tolist()
                    ]
        return resp
    except Exception as e:
        return {"error": f"Error detecting classic patterns: {str(e)}"}


@mcp.tool()
@_auto_connect_wrapper
def patterns_detect_elliott_wave(
    symbol: str,
    timeframe: TimeframeLiteral = "H1",
    lookback: int = 1500,
    denoise: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
    include_series: bool = False,
    series_time: str = "string",  # 'string' or 'epoch'
    include_completed: bool = False,
) -> Dict[str, Any]:
    """Detect Elliott Wave patterns.

    - Pulls last `lookback` bars for `symbol`/`timeframe`.
    - Applies optional denoise.
    - Returns a list of detected Elliott Wave patterns with confidence scores.
    """
    try:
        if timeframe not in TIMEFRAME_MAP:
            return {"error": f"Invalid timeframe: {timeframe}. Valid options: {list(TIMEFRAME_MAP.keys())}"}
        mt5_tf = TIMEFRAME_MAP[timeframe]
        # Ensure symbol is visible
        _info = mt5.symbol_info(symbol)
        _was_visible = bool(_info.visible) if _info is not None else None
        try:
            if _was_visible is False:
                mt5.symbol_select(symbol, True)
        except Exception:
            pass
        # Fetch bars
        utc_now = datetime.utcnow()
        count = max(400, int(lookback) + 2)
        rates = _mt5_copy_rates_from(symbol, mt5_tf, utc_now, count)
        if rates is None or len(rates) < 100:
            return {"error": f"Failed to fetch sufficient bars for {symbol}"}
        df = pd.DataFrame(rates)
        if 'volume' not in df.columns and 'tick_volume' in df.columns:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                df['volume'] = df['tick_volume']
        # Drop forming last bar for stability
        if len(df) >= 2:
            df = df.iloc[:-1]
        if denoise:
            try:
                dn = _normalize_denoise_spec(denoise, default_when='pre_ti')
                if dn:
                    _apply_denoise_util(df, dn, default_when='pre_ti')
            except Exception:
                pass
        # Clip to lookback
        if len(df) > int(lookback):
            df = df.iloc[-int(lookback):].copy()

        # Build config
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

        # Detect
        pats = _detect_elliott_waves(df, cfg)

        # Serialize
        def _round(x):
            try:
                return float(np.round(float(x), 8))
            except Exception:
                return x
        out_list = []
        n_bars = len(df)
        for p in pats:
            try:
                # Format times per global time format
                st_epoch = float(p.start_time) if p.start_time is not None else None
                et_epoch = float(p.end_time) if p.end_time is not None else None
                try:
                    start_date = _format_time_minimal_util(st_epoch) if st_epoch is not None else None
                except Exception:
                    start_date = None
                try:
                    end_date = _format_time_minimal_util(et_epoch) if et_epoch is not None else None
                except Exception:
                    end_date = None
                # Status heuristic: treat recent structures as 'forming', older as 'completed'
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
        # Filter out completed patterns by default, similar to classic detector
        filtered = out_list if include_completed else [d for d in out_list if str(d.get('status','')).lower() == 'forming']

        resp: Dict[str, Any] = {
            "success": True,
            "symbol": symbol,
            "timeframe": timeframe,
            "lookback": int(lookback),
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
                        _format_time_minimal_util(float(v)) for v in __to_float_np(df.get('time')).tolist()
                    ]
        return resp
    except Exception as e:
        return {"error": f"Error detecting Elliott Wave patterns: {str(e)}"}
