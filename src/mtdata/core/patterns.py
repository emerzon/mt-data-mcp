from datetime import datetime, timezone
from typing import Any, Dict, Optional, List, Tuple, Literal
import importlib
import copy
import pandas as pd
import warnings
import numpy as np

from .schema import TimeframeLiteral
from .constants import TIMEFRAME_MAP
from ..utils.mt5 import _mt5_copy_rates_from
from ..utils.utils import _format_time_minimal, to_float_np as __to_float_np
from ..patterns.candlestick import detect_candlestick_patterns as _detect_candlestick_patterns
from ..patterns.classic import detect_classic_patterns as _detect_classic_patterns, ClassicDetectorConfig as _ClassicCfg
from ..patterns.elliott import detect_elliott_waves as _detect_elliott_waves, ElliottWaveConfig as _ElliottCfg
from .server import mcp, _auto_connect_wrapper
from ..utils.denoise import _apply_denoise as _apply_denoise_util, normalize_denoise_spec as _normalize_denoise_spec
import MetaTrader5 as mt5

_CLASSIC_ENGINE_ORDER = ("native", "stock_pattern", "precise_patterns")
_STOCK_PATTERN_CODE_TO_NAME = {
    "TRNG": "Triangle",
    "DTOP": "Double Top",
    "DBOT": "Double Bottom",
    "HNSD": "Head and Shoulders",
    "HNSU": "Inverse Head and Shoulders",
    "UPTL": "Ascending Trend Line",
    "DNTL": "Descending Trend Line",
    "FLAGU": "Bull Flag",
    "FLAGD": "Bear Flag",
    "VCPU": "Bull VCP",
    "VCPD": "Bear VCP",
    "ABCDU": "Bull AB=CD",
    "ABCDD": "Bear AB=CD",
    "BATU": "Bull Bat",
    "BATD": "Bear Bat",
    "GARTU": "Bull Gartley",
    "GARTD": "Bear Gartley",
    "CRABU": "Bull Crab",
    "CRABD": "Bear Crab",
    "BFLYU": "Bull Butterfly",
    "BFLYD": "Bear Butterfly",
}
_STOCK_PATTERN_UTILS_CACHE: Dict[str, Any] = {}


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
    
    utc_now = datetime.now(timezone.utc)
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


def _format_elliott_patterns(df: pd.DataFrame, cfg: _ElliottCfg) -> List[Dict[str, Any]]:
    """Run Elliott detection on prepared data and normalize result rows."""
    pats = _detect_elliott_waves(df, cfg)
    out_list: List[Dict[str, Any]] = []
    n_bars = len(df)

    for p in pats:
        try:
            start_date, end_date = _format_pattern_dates(p.start_time, p.end_time)
            recent_bars = 3
            status = 'forming' if int(p.end_index) >= int(n_bars - recent_bars) else 'completed'

            out_list.append(
                {
                    "wave_type": p.wave_type,
                    "status": status,
                    "confidence": float(max(0.0, min(1.0, p.confidence))),
                    "start_index": int(p.start_index),
                    "end_index": int(p.end_index),
                    "start_date": start_date,
                    "end_date": end_date,
                    "details": {k: _round_value(v) for k, v in (p.details or {}).items()},
                }
            )
        except Exception:
            continue
    return out_list


def _estimate_classic_bars_to_completion(
    name: str,
    details: Dict[str, Any],
    start_idx: int,
    end_idx: int,
    n_bars: int,
) -> Optional[int]:
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


def _format_classic_native_patterns(df: pd.DataFrame, cfg: _ClassicCfg) -> List[Dict[str, Any]]:
    pats = _detect_classic_patterns(df, cfg)
    out_list: List[Dict[str, Any]] = []
    n_bars = len(df)
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
                est = _estimate_classic_bars_to_completion(
                    p.name, d["details"], int(d["start_index"]), int(d["end_index"]), n_bars
                )
                if est is not None:
                    d["bars_to_completion"] = int(est)
            out_list.append(d)
        except Exception:
            continue
    return out_list


def _normalize_engine_name(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_")


def _parse_engine_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        parts = [p.strip() for p in value.replace(";", ",").split(",") if p.strip()]
        return [_normalize_engine_name(p) for p in parts]
    if isinstance(value, (list, tuple, set)):
        return [_normalize_engine_name(p) for p in value if str(p).strip()]
    return [_normalize_engine_name(value)]


def _select_classic_engines(engine: str, ensemble: bool) -> Tuple[List[str], List[str]]:
    requested = _parse_engine_list(engine)
    if not requested:
        requested = ["native"]
    if ensemble and requested == ["native"]:
        requested = list(_CLASSIC_ENGINE_ORDER)
    if ensemble and "native" not in requested:
        requested = ["native"] + requested
    unique: List[str] = []
    invalid: List[str] = []
    for e in requested:
        if e in unique:
            continue
        if e in _CLASSIC_ENGINE_ORDER:
            unique.append(e)
        else:
            invalid.append(e)
    if not unique:
        unique = ["native"]
    return unique, invalid


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d %H:%M")
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(v) for v in value]
    return value


def _timestamp_to_label(ts: Any) -> Optional[str]:
    try:
        if isinstance(ts, pd.Timestamp):
            return ts.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return None
    return None


def _load_stock_pattern_utils(config: Optional[Dict[str, Any]]) -> Tuple[Optional[Any], Optional[str]]:
    _ = config
    # Resolve from currently active Python environment only.
    candidate_modules = (
        "stock_pattern.utils",
        "stockpattern.utils",
    )
    required = ("get_max_min", "find_double_top", "find_double_bottom")
    last_err: Optional[str] = None
    for mod_name in candidate_modules:
        if mod_name in _STOCK_PATTERN_UTILS_CACHE:
            return _STOCK_PATTERN_UTILS_CACHE[mod_name], None
        try:
            mod = importlib.import_module(mod_name)
        except Exception as ex:
            last_err = str(ex)
            continue
        if not all(callable(getattr(mod, fn, None)) for fn in required):
            last_err = f"module '{mod_name}' missing required stock-pattern functions"
            continue
        _STOCK_PATTERN_UTILS_CACHE[mod_name] = mod
        return mod, None

    tail = f" Last import error: {last_err}" if last_err else ""
    return None, (
        "stock-pattern engine unavailable in current environment; "
        "install an importable stock-pattern module exposing stock_pattern.utils." + tail
    )


def _index_pos_for_timestamp(index: pd.Index, ts: Any) -> Optional[int]:
    try:
        loc = index.get_loc(pd.Timestamp(ts))
        if isinstance(loc, slice):
            return int(loc.start)
        if isinstance(loc, np.ndarray):
            return int(loc[0]) if loc.size else None
        if isinstance(loc, list):
            return int(loc[0]) if loc else None
        return int(loc)
    except Exception:
        return None


def _build_stock_pattern_frame(df: pd.DataFrame) -> pd.DataFrame:
    src = df.copy()
    close_col = "close" if "close" in src.columns else "Close"
    open_col = "open" if "open" in src.columns else "Open"
    high_col = "high" if "high" in src.columns else "High"
    low_col = "low" if "low" in src.columns else "Low"
    vol_col = "volume" if "volume" in src.columns else ("tick_volume" if "tick_volume" in src.columns else "Volume")

    out = pd.DataFrame(
        {
            "Open": pd.to_numeric(src.get(open_col), errors="coerce"),
            "High": pd.to_numeric(src.get(high_col), errors="coerce"),
            "Low": pd.to_numeric(src.get(low_col), errors="coerce"),
            "Close": pd.to_numeric(src.get(close_col), errors="coerce"),
            "Volume": pd.to_numeric(src.get(vol_col), errors="coerce"),
        }
    )

    if "time" in src.columns:
        try:
            idx = pd.to_datetime(pd.to_numeric(src["time"], errors="coerce"), unit="s", utc=True)
            out.index = pd.DatetimeIndex(idx).tz_localize(None)
        except Exception:
            out.index = pd.RangeIndex(start=0, stop=len(out), step=1)
    else:
        out.index = pd.RangeIndex(start=0, stop=len(out), step=1)
    return out.dropna(subset=["Open", "High", "Low", "Close"]).copy()


def _map_stock_pattern_name(row: Dict[str, Any]) -> str:
    code = str(row.get("pattern", "")).upper().strip()
    alt = str(row.get("alt_name", "")).strip()
    if code == "TRNG" and alt:
        return f"{alt} Triangle"
    if alt:
        return alt
    return _STOCK_PATTERN_CODE_TO_NAME.get(code, code or "Unknown")


def _to_float_safe(value: Any, default: float = 0.6) -> float:
    try:
        v = float(value)
        if np.isfinite(v):
            return v
    except Exception:
        pass
    return float(default)


def _infer_stock_pattern_confidence(row: Dict[str, Any]) -> float:
    if "confidence" in row:
        return float(max(0.0, min(1.0, _to_float_safe(row.get("confidence"), 0.6))))
    touches = row.get("touches")
    if touches is not None:
        t = _to_float_safe(touches, 0.0)
        return float(max(0.35, min(0.95, 0.5 + 0.05 * t)))
    return 0.6


def _parse_native_scale_factors(config: Optional[Dict[str, Any]]) -> List[float]:
    cfg_map = config if isinstance(config, dict) else {}
    raw = cfg_map.get("native_scale_factors", cfg_map.get("native_scales"))
    vals: List[float] = []
    if isinstance(raw, str):
        parts = [p.strip() for p in raw.replace(";", ",").split(",") if p.strip()]
        for p in parts:
            try:
                vals.append(float(p))
            except Exception:
                continue
    elif isinstance(raw, (list, tuple, set)):
        for item in raw:
            try:
                vals.append(float(item))
            except Exception:
                continue
    if not vals:
        vals = [0.8, 1.0, 1.25]
    out: List[float] = []
    seen = set()
    for v in vals:
        if not np.isfinite(v) or v <= 0:
            continue
        clamped = float(max(0.3, min(3.0, v)))
        key = round(clamped, 4)
        if key in seen:
            continue
        seen.add(key)
        out.append(clamped)
    if 1.0 not in [round(v, 4) for v in out]:
        out.insert(0, 1.0)
    return out


def _run_classic_engine_native(
    symbol: str,
    df: pd.DataFrame,
    cfg: _ClassicCfg,
    config: Optional[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    _ = symbol
    cfg_map = config if isinstance(config, dict) else {}
    if not bool(cfg_map.get("native_multiscale", False)):
        return _format_classic_native_patterns(df, cfg), None

    scales = _parse_native_scale_factors(config)
    if len(scales) <= 1:
        return _format_classic_native_patterns(df, cfg), None

    per_scale: Dict[str, List[Dict[str, Any]]] = {}
    scale_by_key: Dict[str, float] = {}
    base_min_dist = int(max(2, getattr(cfg, "min_distance", 5)))
    base_prom = float(max(1e-6, getattr(cfg, "min_prominence_pct", 0.5)))
    for scale in scales:
        key = f"native_scale_{scale:.2f}"
        cfg_i = copy.deepcopy(cfg)
        try:
            cfg_i.min_distance = max(2, int(round(base_min_dist * float(scale))))
        except Exception:
            cfg_i.min_distance = base_min_dist
        try:
            cfg_i.min_prominence_pct = max(0.05, float(base_prom * float(scale)))
        except Exception:
            cfg_i.min_prominence_pct = base_prom
        rows = _format_classic_native_patterns(df, cfg_i)
        for row in rows:
            d = row.get("details")
            if not isinstance(d, dict):
                d = {}
            d = dict(d)
            d["native_scale_factor"] = float(scale)
            row["details"] = d
        per_scale[key] = rows
        scale_by_key[key] = float(scale)

    non_empty = {k: v for k, v in per_scale.items() if v}
    if not non_empty:
        return [], None
    if len(non_empty) == 1:
        return list(next(iter(non_empty.values()))), None

    overlap = 0.45
    try:
        overlap = float(cfg_map.get("native_multiscale_overlap", overlap))
    except Exception:
        overlap = 0.45
    overlap = float(max(0.2, min(0.9, overlap)))

    merged = _merge_classic_ensemble(non_empty, {k: 1.0 for k in non_empty.keys()}, overlap_threshold=overlap)
    for row in merged:
        details = row.get("details")
        if not isinstance(details, dict):
            details = {}
        details = dict(details)
        src = [str(x) for x in row.get("source_engines", [])]
        details["native_multiscale"] = True
        details["native_multiscale_overlap"] = float(overlap)
        details["native_scale_support"] = int(len(src))
        details["native_scale_factors"] = [float(scale_by_key[s]) for s in src if s in scale_by_key]
        row["details"] = details
    return merged, None


def _run_classic_engine_stock_pattern(
    symbol: str,
    df: pd.DataFrame,
    cfg: _ClassicCfg,
    config: Optional[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    _ = cfg
    sp_utils, load_err = _load_stock_pattern_utils(config)
    if load_err:
        return [], load_err
    if sp_utils is None:
        return [], "stock-pattern module unavailable"

    try:
        sp_df = _build_stock_pattern_frame(df)
    except Exception as ex:
        return [], f"Failed preparing data for stock-pattern: {ex}"

    if sp_df.empty:
        return [], "No valid candles after stock-pattern dataframe normalization"

    bars_left = 6
    bars_right = 6
    cfg_map = config if isinstance(config, dict) else {}
    try:
        bars_left = int(cfg_map.get("stock_bars_left", bars_left))
        bars_right = int(cfg_map.get("stock_bars_right", bars_right))
    except Exception:
        pass

    pivots_cache: Dict[str, pd.DataFrame] = {}

    def _get_pivots(pivot_type: str) -> pd.DataFrame:
        if pivot_type not in pivots_cache:
            piv = sp_utils.get_max_min(sp_df, barsLeft=bars_left, barsRight=bars_right, pivot_type=pivot_type)
            pivots_cache[pivot_type] = piv if isinstance(piv, pd.DataFrame) else pd.DataFrame()
        return pivots_cache[pivot_type]

    fn_specs = [
        ("find_double_top", "both"),
        ("find_double_bottom", "both"),
        ("find_triangles", "both"),
        ("find_hns", "both"),
        ("find_reverse_hns", "both"),
        ("find_bullish_flag", "low"),
        ("find_bearish_flag", "high"),
        ("find_uptrend_line", "low"),
        ("find_downtrend_line", "high"),
    ]
    out_list: List[Dict[str, Any]] = []
    n_bars = len(df)
    for fn_name, pivot_type in fn_specs:
        fn = getattr(sp_utils, fn_name, None)
        if not callable(fn):
            continue
        pivots = _get_pivots(pivot_type)
        if pivots.empty:
            continue
        try:
            res = fn(symbol, sp_df, pivots, cfg_map)
        except Exception:
            continue
        if not isinstance(res, dict):
            continue

        start_ts = res.get("start")
        end_ts = res.get("end")
        s_idx = _index_pos_for_timestamp(sp_df.index, start_ts)
        e_idx = _index_pos_for_timestamp(sp_df.index, end_ts)
        if s_idx is None:
            s_idx = 0
        if e_idx is None:
            e_idx = len(sp_df) - 1
        if e_idx < s_idx:
            s_idx, e_idx = e_idx, s_idx

        name = _map_stock_pattern_name(res)
        details = _to_jsonable(
            {
                k: v
                for k, v in res.items()
                if k
                not in {
                    "sym",
                    "pattern",
                    "alt_name",
                    "start",
                    "end",
                    "df_start",
                    "df_end",
                }
            }
        )
        d: Dict[str, Any] = {
            "name": name,
            "status": "forming",
            "confidence": float(max(0.0, min(1.0, _infer_stock_pattern_confidence(res)))),
            "start_index": int(s_idx),
            "end_index": int(e_idx),
            "start_date": _timestamp_to_label(start_ts),
            "end_date": _timestamp_to_label(end_ts),
            "details": {k: _round_value(v) for k, v in dict(details).items()} if isinstance(details, dict) else {"raw": details},
        }
        est = _estimate_classic_bars_to_completion(name, d["details"], int(d["start_index"]), int(d["end_index"]), n_bars)
        if est is not None:
            d["bars_to_completion"] = int(est)
        out_list.append(d)
    return out_list, None


def _run_classic_engine_precise_patterns(
    symbol: str,
    df: pd.DataFrame,
    cfg: _ClassicCfg,
    config: Optional[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    _ = symbol
    _ = df
    _ = cfg
    _ = config
    try:
        patterns_mod = importlib.import_module("precise_patterns.patterns")
    except Exception:
        return [], "precise-patterns engine unavailable (install package to enable)"

    reg = getattr(patterns_mod, "Registry", None)
    if reg is None or not hasattr(reg, "all"):
        return [], "precise-patterns has no stable Registry API for pattern extraction"
    try:
        names = list((reg.all() or {}).keys())  # type: ignore[call-arg]
    except Exception:
        names = []
    if not names:
        return [], "precise-patterns has no registered pattern implementations"
    return [], "precise-patterns adapter is experimental and currently yields no classic detections"


def _run_classic_engine(
    engine: str,
    symbol: str,
    df: pd.DataFrame,
    cfg: _ClassicCfg,
    config: Optional[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    if engine == "native":
        return _run_classic_engine_native(symbol, df, cfg, config)
    if engine == "stock_pattern":
        return _run_classic_engine_stock_pattern(symbol, df, cfg, config)
    if engine == "precise_patterns":
        return _run_classic_engine_precise_patterns(symbol, df, cfg, config)
    return [], f"Unsupported classic engine: {engine}"


def _interval_overlap_ratio(a_start: int, a_end: int, b_start: int, b_end: int) -> float:
    lo = max(int(a_start), int(b_start))
    hi = min(int(a_end), int(b_end))
    inter = max(0, hi - lo + 1)
    union = max(int(a_end), int(b_end)) - min(int(a_start), int(b_start)) + 1
    if union <= 0:
        return 0.0
    return float(inter) / float(union)


def _resolve_engine_weights(
    engines: List[str],
    ensemble_weights: Optional[Dict[str, Any]],
) -> Dict[str, float]:
    out = {e: 1.0 for e in engines}
    if not isinstance(ensemble_weights, dict):
        return out
    for k, v in ensemble_weights.items():
        ek = _normalize_engine_name(k)
        if ek not in out:
            continue
        try:
            w = float(v)
            if np.isfinite(w) and w > 0:
                out[ek] = w
        except Exception:
            continue
    return out


def _merge_classic_ensemble(
    engine_patterns: Dict[str, List[Dict[str, Any]]],
    weights: Dict[str, float],
    overlap_threshold: float = 0.5,
) -> List[Dict[str, Any]]:
    groups: List[Dict[str, Any]] = []
    for engine, pats in engine_patterns.items():
        for p in pats:
            try:
                nm = str(p.get("name", "")).strip().lower()
                s_idx = int(p.get("start_index", 0))
                e_idx = int(p.get("end_index", s_idx))
            except Exception:
                continue
            target: Optional[Dict[str, Any]] = None
            for g in groups:
                if g["name_norm"] != nm:
                    continue
                if _interval_overlap_ratio(s_idx, e_idx, int(g["start_index"]), int(g["end_index"])) >= float(overlap_threshold):
                    target = g
                    break
            if target is None:
                target = {
                    "name_norm": nm,
                    "start_index": s_idx,
                    "end_index": e_idx,
                    "items": [],
                }
                groups.append(target)
            target["start_index"] = min(int(target["start_index"]), s_idx)
            target["end_index"] = max(int(target["end_index"]), e_idx)
            target["items"].append((engine, p))

    merged: List[Dict[str, Any]] = []
    for g in groups:
        items: List[Tuple[str, Dict[str, Any]]] = g.get("items", [])
        if not items:
            continue
        by_engine: Dict[str, Dict[str, Any]] = {}
        for eng, p in items:
            # Keep the strongest candidate per engine inside a merge group.
            prev = by_engine.get(eng)
            if prev is None or float(p.get("confidence", 0.0)) > float(prev.get("confidence", 0.0)):
                by_engine[eng] = p
        engines = list(by_engine.keys())
        total_w = float(sum(weights.get(e, 1.0) for e in engines)) or 1.0
        conf = float(
            sum(float(by_engine[e].get("confidence", 0.0)) * float(weights.get(e, 1.0)) for e in engines) / total_w
        )
        anchor_engine = max(engines, key=lambda e: float(by_engine[e].get("confidence", 0.0)))
        anchor = dict(by_engine[anchor_engine])
        statuses = [str(by_engine[e].get("status", "forming")).lower() for e in engines]
        anchor["status"] = "completed" if statuses and all(s == "completed" for s in statuses) else "forming"
        anchor["confidence"] = float(max(0.0, min(1.0, conf)))
        anchor["support_count"] = int(len(engines))
        anchor["source_engines"] = engines
        details = anchor.get("details")
        if not isinstance(details, dict):
            details = {}
        details = dict(details)
        details["engine_confidences"] = {
            e: float(max(0.0, min(1.0, float(by_engine[e].get("confidence", 0.0)))))
            for e in engines
        }
        details["consensus_support"] = int(len(engines))
        anchor["details"] = details
        merged.append(anchor)

    merged.sort(
        key=lambda p: (
            int(p.get("support_count", 1)),
            float(p.get("confidence", 0.0)),
            int(p.get("end_index", -1)),
        ),
        reverse=True,
    )
    return merged


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

    def _coerce_bool(value: Any) -> Any:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            s = value.strip().lower()
            if s in {"true", "1", "yes", "y", "on"}:
                return True
            if s in {"false", "0", "no", "n", "off"}:
                return False
        return value

    if not isinstance(config, dict):
        return
    for k, v in config.items():
        if hasattr(cfg, k):
            current = getattr(cfg, k)
            try:
                # Handle list-like attrs from common CLI forms, e.g. "impulse,correction".
                if isinstance(current, list):
                    if isinstance(v, str):
                        parsed = [p.strip() for p in v.replace(";", ",").split(",") if p.strip()]
                        setattr(cfg, k, parsed)
                    elif isinstance(v, (list, tuple, set)):
                        setattr(cfg, k, [x for x in v])
                    else:
                        setattr(cfg, k, [v])
                elif isinstance(current, bool):
                    coerced = _coerce_bool(v)
                    setattr(cfg, k, bool(coerced) if isinstance(coerced, bool) else current)
                else:
                    setattr(cfg, k, type(current)(v))
            except Exception:
                try:
                    setattr(cfg, k, v)
                except Exception:
                    pass

@mcp.tool()
@_auto_connect_wrapper
def patterns_detect(
    symbol: str,
    timeframe: Optional[TimeframeLiteral] = None,
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
    engine: str = "native",
    ensemble: bool = False,
    ensemble_weights: Optional[Dict[str, Any]] = None,
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
    
    timeframe : str, optional
        Chart timeframe: "M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1", "MN1"
        For `mode="elliott"`, when omitted, all available timeframes are scanned and
        returned in a single aggregated output.
    
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
        Pattern-specific configuration parameters.
        Useful classic options include:
        - native_multiscale: bool
        - native_scale_factors: list[float] (e.g. [0.8, 1.0, 1.25])
        - pivot_use_hl, pivot_use_atr_adaptive_prominence, pivot_use_atr_adaptive_distance
        - calibrate_confidence, confidence_calibration_map, confidence_calibration_blend

    engine : str, optional (default="native")
        Classic engine selection: "native", "stock_pattern", "precise_patterns",
        or comma-separated list when `ensemble=True`.

    ensemble : bool, optional (default=False)
        For classic mode, merge detections from multiple engines into consensus results.

    ensemble_weights : dict, optional
        Per-engine weights used for consensus confidence, e.g.
        {"native": 1.0, "stock_pattern": 0.8}
    
    include_series : bool, optional (default=False)
        Include the price series data in the response
    
    series_time : str, optional (default="string")
        Time format for series data
    
    include_completed : bool, optional (default=False)
        Include completed patterns alongside forming results (when False, only forming patterns are returned)
    
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
        tf_norm: Optional[str] = str(timeframe).strip().upper() if timeframe is not None else None
        if not tf_norm:
            tf_norm = None

        if mode == 'candlestick':
            tf_single = tf_norm or "H1"
            return _detect_candlestick_patterns(
                symbol=symbol,
                timeframe=tf_single,
                limit=limit,
                min_strength=min_strength,
                min_gap=min_gap,
                robust_only=robust_only,
                whitelist=whitelist,
                top_k=top_k,
            )

        elif mode == 'classic':
            tf_single = tf_norm or "H1"
            # Fetch and prepare data using shared helper
            df, err = _fetch_pattern_data(symbol, tf_single, limit, denoise)
            if err:
                return err

            cfg = _ClassicCfg()
            _apply_config_to_obj(cfg, config)
            engines, invalid_engines = _select_classic_engines(engine, ensemble)
            if invalid_engines:
                return {
                    "error": (
                        f"Invalid classic engine(s): {invalid_engines}. "
                        f"Valid options: {list(_CLASSIC_ENGINE_ORDER)}"
                    )
                }

            per_engine: Dict[str, List[Dict[str, Any]]] = {}
            engine_errors: Dict[str, str] = {}
            for eng in engines:
                patt_rows, eng_err = _run_classic_engine(eng, symbol, df, cfg, config)
                if eng_err:
                    engine_errors[eng] = eng_err
                per_engine[eng] = patt_rows

            non_empty = {e: p for e, p in per_engine.items() if p}
            if not non_empty:
                if engine_errors and len(engine_errors) == len(engines):
                    return {
                        "error": "No classic engines produced results",
                        "engines_run": engines,
                        "engine_errors": engine_errors,
                    }
                resp = _build_pattern_response(
                    symbol, tf_single, limit, mode, [],
                    include_completed, include_series, series_time, df
                )
                resp["engine"] = "ensemble" if (bool(ensemble) or len(engines) > 1) else engines[0]
                resp["engines_run"] = engines
                resp["engine_findings"] = [
                    {
                        "engine": e,
                        "n_patterns": int(len(per_engine.get(e, []))),
                        "n_forming": int(sum(1 for p in per_engine.get(e, []) if str(p.get("status", "")).lower() == "forming")),
                        "n_completed": int(sum(1 for p in per_engine.get(e, []) if str(p.get("status", "")).lower() == "completed")),
                    }
                    for e in engines
                ]
                if engine_errors:
                    resp["engine_errors"] = engine_errors
                return resp

            run_ensemble = bool(ensemble) or len(non_empty) > 1
            if run_ensemble:
                weight_map = _resolve_engine_weights(
                    engines,
                    ensemble_weights if isinstance(ensemble_weights, dict) else (
                        (config or {}).get("ensemble_weights") if isinstance(config, dict) else None
                    ),
                )
                out_list = _merge_classic_ensemble(non_empty, weight_map)
            else:
                # keep single-engine behavior when only one engine produces output
                only_engine = next(iter(non_empty.keys()))
                out_list = list(non_empty.get(only_engine, []))

            resp = _build_pattern_response(
                symbol, tf_single, limit, mode, out_list,
                include_completed, include_series, series_time, df
            )
            resp["engine"] = "ensemble" if run_ensemble else next(iter(non_empty.keys()))
            resp["engines_run"] = engines
            resp["engine_findings"] = [
                {
                    "engine": e,
                    "n_patterns": int(len(per_engine.get(e, []))),
                    "n_forming": int(sum(1 for p in per_engine.get(e, []) if str(p.get("status", "")).lower() == "forming")),
                    "n_completed": int(sum(1 for p in per_engine.get(e, []) if str(p.get("status", "")).lower() == "completed")),
                }
                for e in engines
            ]
            if engine_errors:
                resp["engine_errors"] = engine_errors
            return resp

        elif mode == 'elliott':
            cfg = _ElliottCfg()
            _apply_config_to_obj(cfg, config)

            if tf_norm:
                # Fetch and prepare data using shared helper
                df, err = _fetch_pattern_data(symbol, tf_norm, limit, denoise)
                if err:
                    return err

                out_list = _format_elliott_patterns(df, cfg)
                return _build_pattern_response(
                    symbol, tf_norm, limit, mode, out_list,
                    include_completed, include_series, series_time, df
                )

            # If timeframe is omitted for Elliott mode, scan all available timeframes.
            scanned_timeframes = list(TIMEFRAME_MAP.keys())
            findings: List[Dict[str, Any]] = []
            combined_patterns: List[Dict[str, Any]] = []
            failed_timeframes: Dict[str, str] = {}
            series_by_timeframe: Dict[str, Dict[str, Any]] = {}

            for tf in scanned_timeframes:
                df, err = _fetch_pattern_data(symbol, tf, limit, denoise)
                if err:
                    failed_timeframes[tf] = str(err.get("error", "Unknown error"))
                    continue

                tf_patterns = _format_elliott_patterns(df, cfg)
                filtered = tf_patterns if include_completed else [
                    d for d in tf_patterns if str(d.get('status', '')).lower() == 'forming'
                ]
                findings.append({
                    "timeframe": tf,
                    "n_patterns": int(len(filtered)),
                    "patterns": filtered,
                })

                for row in filtered:
                    merged = dict(row)
                    merged["timeframe"] = tf
                    combined_patterns.append(merged)

                if include_series:
                    series_payload: Dict[str, Any] = {
                        "series_close": [float(v) for v in __to_float_np(df.get('close')).tolist()]
                    }
                    if 'time' in df.columns:
                        if str(series_time).lower() == 'epoch':
                            series_payload["series_epoch"] = [float(v) for v in __to_float_np(df.get('time')).tolist()]
                        else:
                            series_payload["series_time"] = [
                                _format_time_minimal(float(v)) for v in __to_float_np(df.get('time')).tolist()
                            ]
                    series_by_timeframe[tf] = series_payload

            if not findings:
                return {
                    "error": f"Failed to fetch sufficient bars for {symbol} across all timeframes",
                    "failed_timeframes": failed_timeframes,
                }

            resp: Dict[str, Any] = {
                "success": True,
                "symbol": symbol,
                "timeframe": "ALL",
                "lookback": int(limit),
                "mode": mode,
                "scanned_timeframes": scanned_timeframes,
                "findings": findings,
                "patterns": combined_patterns,
                "n_patterns": int(len(combined_patterns)),
            }
            if failed_timeframes:
                resp["failed_timeframes"] = failed_timeframes
            if include_series:
                resp["series_by_timeframe"] = series_by_timeframe
            return resp
        
        else:
            return {"error": f"Unknown mode: {mode}"}

    except Exception as e:
        return {"error": f"Error detecting patterns: {str(e)}"}
