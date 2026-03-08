from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional, List, Tuple, Literal
import importlib
import copy
import logging
import pandas as pd
import warnings

from .constants import TIMEFRAME_MAP
from .execution_logging import run_logged_operation
from .mt5_gateway import create_mt5_gateway
from .patterns_support import (
    _STOCK_PATTERN_UTILS_CACHE,
    _build_stock_pattern_frame,
    _compact_patterns_payload,
    _enrich_classic_patterns,
    _estimate_classic_bars_to_completion,
    _format_pattern_dates,
    _index_pos_for_timestamp,
    _infer_stock_pattern_confidence,
    _interval_overlap_ratio,
    _load_stock_pattern_utils,
    _map_stock_pattern_name,
    _merge_classic_ensemble,
    _normalize_engine_name,
    _parse_engine_list,
    _parse_native_scale_factors,
    _resolve_engine_weights,
    _round_value,
    _summarize_engine_findings,
    _summarize_pattern_bias,
    _timestamp_to_label,
    _to_float_safe,
    _to_jsonable,
)
from ..utils.mt5 import _mt5_copy_rates_from
from ..utils.utils import _format_time_minimal, to_float_np as __to_float_np
from ..patterns.candlestick import detect_candlestick_patterns as _detect_candlestick_patterns
from ..patterns.classic import detect_classic_patterns as _detect_classic_patterns, ClassicDetectorConfig as _ClassicCfg
from ..patterns.elliott import detect_elliott_waves as _detect_elliott_waves, ElliottWaveConfig as _ElliottCfg
from ._mcp_instance import mcp
from .patterns_requests import PatternsDetectRequest
from .patterns_use_cases import run_patterns_detect
from ..utils.denoise import _apply_denoise as _apply_denoise_util, normalize_denoise_spec as _normalize_denoise_spec
from ..utils.mt5 import MT5ConnectionError, ensure_mt5_connection_or_raise, mt5

logger = logging.getLogger(__name__)

_CLASSIC_ENGINE_ORDER = ("native", "stock_pattern", "precise_patterns")
ClassicEngineRunner = Callable[
    [str, pd.DataFrame, _ClassicCfg, Optional[Dict[str, Any]]],
    Tuple[List[Dict[str, Any]], Optional[str]],
]
_CLASSIC_ENGINE_REGISTRY: Dict[str, ClassicEngineRunner] = {}


def _patterns_connection_error() -> Optional[Dict[str, Any]]:
    mt5_gateway = _get_mt5_gateway()
    try:
        mt5_gateway.ensure_connection()
    except MT5ConnectionError as exc:
        return {"error": str(exc)}
    return None


def _get_mt5_gateway():
    return create_mt5_gateway(adapter=mt5, ensure_connection_impl=ensure_mt5_connection_or_raise)


def _fetch_pattern_data(
    symbol: str,
    timeframe: str,
    limit: int,
    denoise: Optional[Dict[str, Any]] = None,
    gateway: Any = None,
) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, Any]]]:
    """Fetch and prepare OHLCV data for pattern detection.
    
    Returns (df, error_dict) where error_dict is None on success.
    """
    if timeframe not in TIMEFRAME_MAP:
        return None, {"error": f"Invalid timeframe: {timeframe}. Valid options: {list(TIMEFRAME_MAP.keys())}"}

    mt5_gateway = gateway or _get_mt5_gateway()
    mt5_tf = TIMEFRAME_MAP[timeframe]
    _info = mt5_gateway.symbol_info(symbol)
    _was_visible = bool(_info.visible) if _info is not None else None
    try:
        if _was_visible is False:
            mt5_gateway.symbol_select(symbol, True)
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


def _elliott_timeframe_suggestion(timeframe: Optional[str]) -> str:
    tf = str(timeframe or "").upper()
    suggestion_map: Dict[str, List[str]] = {
        "M1": ["H1", "H4"],
        "M5": ["H1", "H4"],
        "M15": ["H1", "H4"],
        "M30": ["H4", "D1"],
        "H1": ["H4", "D1"],
        "H4": ["D1", "W1"],
        "D1": ["H4", "W1"],
        "W1": ["D1", "MN1"],
        "MN1": ["W1", "D1"],
    }
    raw_suggestions = suggestion_map.get(tf, ["H4", "D1"])
    suggestions = [s for s in raw_suggestions if s != tf]
    if not suggestions:
        suggestions = ["H4"] if tf != "H4" else ["D1"]
    if len(suggestions) == 1:
        return f"Try --timeframe {suggestions[0]} or increase --limit."
    return f"Try --timeframe {suggestions[0]} or --timeframe {suggestions[1]}."


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
    detail: Literal["compact", "full"] = "full",  # type: ignore
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
    if str(mode).lower() == "elliott" and int(len(filtered)) == 0:
        resp["diagnostic"] = (
            f"No valid Elliott Wave structures detected in {int(limit)} {timeframe} bars. "
            f"{_elliott_timeframe_suggestion(timeframe)} "
            "You can also increase lookback or focus on a clearer trending segment."
        )
    
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

    if str(detail).lower().strip() == "compact":
        return _compact_patterns_payload(resp)
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


def _register_classic_engine(name: str) -> Callable[[ClassicEngineRunner], ClassicEngineRunner]:
    norm_name = _normalize_engine_name(name)

    def _decorator(func: ClassicEngineRunner) -> ClassicEngineRunner:
        _CLASSIC_ENGINE_REGISTRY[norm_name] = func
        return func

    return _decorator


def _available_classic_engines() -> Tuple[str, ...]:
    ordered = [name for name in _CLASSIC_ENGINE_ORDER if name in _CLASSIC_ENGINE_REGISTRY]
    ordered.extend(name for name in _CLASSIC_ENGINE_REGISTRY.keys() if name not in ordered)
    return tuple(ordered)


def _select_classic_engines(engine: str, ensemble: bool) -> Tuple[List[str], List[str]]:
    available = _available_classic_engines()
    requested = _parse_engine_list(engine)
    if not requested:
        requested = ["native"]
    if ensemble and requested == ["native"]:
        requested = list(available)
    if ensemble and "native" not in requested:
        requested = ["native"] + requested
    unique: List[str] = []
    invalid: List[str] = []
    for e in requested:
        if e in unique:
            continue
        if e in available:
            unique.append(e)
        else:
            invalid.append(e)
    if not unique:
        unique = ["native"]
    return unique, invalid


@_register_classic_engine("native")
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


@_register_classic_engine("stock_pattern")
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


@_register_classic_engine("precise_patterns")
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
    runner = _CLASSIC_ENGINE_REGISTRY.get(_normalize_engine_name(engine))
    if runner is None:
        return [], f"Unsupported classic engine: {engine}"
    return runner(symbol, df, cfg, config)


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
def patterns_detect(
    request: PatternsDetectRequest,
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
        - "chart": Alias for classic chart patterns
        - "elliott": Elliott Wave patterns

    detail : str, optional (default="compact")
        Output verbosity:
        - "compact": trader-focused summary with recent patterns and pattern mix.
        - "full": complete pattern rows suitable for research/debugging.
    
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

    last_n_bars : int, optional
        Candlestick mode only. Restrict detections to patterns that occur in the
        most recent N bars.
    
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
    patterns_detect(symbol="BTCUSD", mode="elliott", timeframe="H4", detail="full")
    """
    def _run() -> Dict[str, Any]:
        connection_error = _patterns_connection_error()
        if connection_error is not None:
            return connection_error
        return run_patterns_detect(
            request,
            timeframe_map=TIMEFRAME_MAP,
            compact_patterns_payload=_compact_patterns_payload,
            fetch_pattern_data=_fetch_pattern_data,
            classic_cfg_cls=_ClassicCfg,
            elliott_cfg_cls=_ElliottCfg,
            apply_config_to_obj=_apply_config_to_obj,
            select_classic_engines=_select_classic_engines,
            available_classic_engines=_available_classic_engines,
            run_classic_engine=_run_classic_engine,
            resolve_engine_weights=_resolve_engine_weights,
            merge_classic_ensemble=_merge_classic_ensemble,
            enrich_classic_patterns=_enrich_classic_patterns,
            summarize_engine_findings=_summarize_engine_findings,
            summarize_pattern_bias=_summarize_pattern_bias,
            build_pattern_response=_build_pattern_response,
            format_elliott_patterns=_format_elliott_patterns,
            detect_candlestick_patterns=_detect_candlestick_patterns,
            elliott_timeframe_suggestion=_elliott_timeframe_suggestion,
            format_time_minimal=_format_time_minimal,
            to_float_np=__to_float_np,
        )

    return run_logged_operation(
        logger,
        operation="patterns_detect",
        symbol=request.symbol,
        timeframe=request.timeframe,
        mode=request.mode,
        detail=request.detail,
        func=_run,
    )
