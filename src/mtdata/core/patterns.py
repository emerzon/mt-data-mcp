import copy
import logging
import warnings
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

from ..patterns.candlestick import (
    detect_candlestick_patterns as _detect_candlestick_patterns,
)
from ..patterns.classic import ClassicDetectorConfig as _ClassicCfg
from ..patterns.classic import detect_classic_patterns as _detect_classic_patterns
from ..patterns.classic_impl.config import (
    fatal_classic_detector_config_errors as _fatal_classic_detector_config_errors,
)
from ..patterns.common import data_quality_warnings, should_drop_last_live_bar
from ..patterns.elliott import ElliottWaveConfig as _ElliottCfg
from ..patterns.elliott import detect_elliott_waves as _detect_elliott_waves
from ..patterns.fractal import FractalDetectorConfig as _FractalCfg
from ..patterns.fractal import detect_fractal_patterns as _detect_fractal_patterns
from ..patterns.fractal import (
    validate_fractal_detector_config as _validate_fractal_detector_config,
)
from ..shared.validators import invalid_timeframe_error
from ..utils.denoise import _apply_denoise as _apply_denoise_util
from ..utils.denoise import normalize_denoise_spec as _normalize_denoise_spec
from ..utils.mt5 import _mt5_copy_rates_from, ensure_mt5_connection_or_raise, mt5
from ..utils.ohlcv import validate_and_clean_ohlcv_frame
from ..utils.utils import _UNPARSED_BOOL, _format_time_minimal, _parse_bool_like
from ..utils.utils import to_float_np as __to_float_np
from ._mcp_instance import mcp
from .constants import TIMEFRAME_MAP, TIMEFRAME_SECONDS
from .execution_logging import run_logged_operation
from .mt5_gateway import get_mt5_gateway, mt5_connection_error
from .patterns_requests import PatternsDetectRequest, PatternsDetailLiteral
from .patterns_support import (
    _STOCK_PATTERN_UTILS_CACHE,  # noqa: F401
    _build_stock_pattern_frame,
    _compact_patterns_payload,
    _count_patterns_with_status,
    _elliott_completed_preview,
    _elliott_hidden_completed_note,
    _enrich_classic_patterns,
    _enrich_elliott_patterns,
    _estimate_classic_bars_to_completion,
    _format_pattern_dates,
    _index_pos_for_timestamp,
    _infer_stock_pattern_confidence,
    _interval_overlap_ratio,  # noqa: F401
    _load_stock_pattern_utils,
    _map_stock_pattern_name,
    _merge_classic_ensemble,
    _normalize_engine_name,
    _parse_engine_list,
    _parse_native_scale_factors,
    _resolve_elliott_pattern_status,
    _resolve_engine_weights,
    _round_value,
    _summarize_engine_findings,
    _summarize_pattern_bias,
    _timestamp_to_label,
    _to_float_safe,  # noqa: F401
    _to_jsonable,
    _visible_pattern_rows,
)
from .patterns_use_cases import PatternsDetectDeps, run_patterns_detect

logger = logging.getLogger(__name__)

_CLASSIC_ENGINE_ORDER = ("native", "stock_pattern")
_DEFAULT_ELLIOTT_SCAN_TIMEFRAMES = ("H1", "H4", "D1")
ClassicEngineRunner = Callable[
    [str, pd.DataFrame, _ClassicCfg, Optional[Dict[str, Any]]],
    Tuple[List[Dict[str, Any]], Optional[str]],
]
_CLASSIC_ENGINE_REGISTRY: Dict[str, ClassicEngineRunner] = {}


def _filter_non_actionable_elliott_warnings(
    warnings_in: Any,
    *,
    mode: str,
    diagnostic: Any,
    n_patterns: int,
) -> List[str]:
    if not isinstance(warnings_in, list):
        return []
    warnings_clean = [str(w) for w in warnings_in if str(w)]
    if str(mode).strip().lower() != "elliott" or not diagnostic or int(n_patterns) != 0:
        return warnings_clean
    return [
        warning_text
        for warning_text in warnings_clean
        if not warning_text.startswith("Data quality warning:")
    ]


def _patterns_connection_error() -> Optional[Dict[str, Any]]:
    return mt5_connection_error(
        get_mt5_gateway(
            adapter=mt5,
            ensure_connection_impl=ensure_mt5_connection_or_raise,
        )
    )


def _should_drop_last_pattern_bar(
    df: pd.DataFrame,
    timeframe: str,
    *,
    now_utc: Optional[datetime] = None,
    current_time_epoch: Optional[float] = None,
) -> bool:
    return should_drop_last_live_bar(
        df,
        timeframe,
        now_utc=now_utc,
        current_time_epoch=current_time_epoch,
    )


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
        return None, {"error": invalid_timeframe_error(timeframe, TIMEFRAME_MAP)}

    mt5_gateway = gateway or get_mt5_gateway(
        adapter=mt5,
        ensure_connection_impl=ensure_mt5_connection_or_raise,
    )
    mt5_tf = TIMEFRAME_MAP[timeframe]
    _info = mt5_gateway.symbol_info(symbol)
    if _info is None:
        return None, {"error": f"Symbol '{symbol}' not found or is not available in MT5."}
    _was_visible = bool(_info.visible) if _info is not None else None
    try:
        if _was_visible is False:
            if not mt5_gateway.symbol_select(symbol, True):
                return None, {"error": f"Symbol '{symbol}' is not visible and could not be selected in MT5."}
    except Exception as exc:
        return None, {"error": f"Failed to enable symbol '{symbol}' in MT5: {exc}"}
    
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

    warnings_out: List[str] = []
    try:
        df, quality_warnings = validate_and_clean_ohlcv_frame(df, epoch_col="time")
    except ValueError as exc:
        return None, {"error": str(exc)}
    if len(df) == 0:
        return None, {"error": f"No valid bars available for {symbol}"}
    warnings_out.extend(quality_warnings)

    # Drop the last bar only when it is still open or cannot be validated.
    from ..services.data_service import _resolve_live_bar_reference_epoch

    live_bar_reference_epoch = _resolve_live_bar_reference_epoch(symbol, timeframe)
    if _should_drop_last_pattern_bar(
        df,
        timeframe,
        now_utc=utc_now,
        current_time_epoch=live_bar_reference_epoch,
    ):
        df = df.iloc[:-1].copy()
    
    # Apply denoising if requested
    if denoise:
        try:
            dn = _normalize_denoise_spec(denoise, default_when='pre_ti')
            if dn:
                _apply_denoise_util(df, dn, default_when='pre_ti')
        except Exception as exc:
            warning = f"Denoise failed for pattern detection on {symbol} {timeframe}; raw prices were used."
            logger.warning(warning, exc_info=True)
            warnings_out.append(f"{warning} {exc}")
    
    # Trim to requested limit
    if len(df) > int(limit):
        df = df.iloc[-int(limit):].copy()

    # Freshness warning: flag when the most recent bar is unusually old.
    # Uses a generous threshold (7 days) to tolerate weekend/holiday closures.
    tf_secs = float(TIMEFRAME_SECONDS.get(timeframe, 0) or 0)
    if tf_secs > 0 and len(df) > 0:
        try:
            last_epoch = float(df["time"].iloc[-1])
            staleness = utc_now.timestamp() - last_epoch
            if staleness > 7 * 86400:
                warnings_out.append(
                    f"Data may be stale for {symbol} {timeframe}: "
                    f"latest bar is {staleness / 86400:.1f} days old."
                )
        except Exception:
            pass

    warnings_out.extend(
        data_quality_warnings(
            df,
            symbol=symbol,
            timeframe_seconds=float(TIMEFRAME_SECONDS.get(timeframe, 0) or 0),
        )
    )
    if warnings_out:
        df.attrs["warnings"] = list(warnings_out)
    
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


def _resolve_elliott_scan_timeframes(cfg: _ElliottCfg) -> List[str]:
    raw_scan = getattr(cfg, "scan_timeframes", None)
    requested: List[str] = []
    if isinstance(raw_scan, str):
        requested = [part.strip().upper() for part in raw_scan.replace(";", ",").split(",") if part.strip()]
    elif isinstance(raw_scan, (list, tuple, set)):
        requested = [str(part).strip().upper() for part in raw_scan if str(part).strip()]

    if not requested:
        requested = [tf for tf in _DEFAULT_ELLIOTT_SCAN_TIMEFRAMES if tf in TIMEFRAME_MAP]
    if not requested:
        requested = [str(tf).strip().upper() for tf in TIMEFRAME_MAP.keys()]

    try:
        max_scan = int(getattr(cfg, "max_scan_timeframes", 3))
    except Exception:
        max_scan = 3
    if max_scan <= 0:
        max_scan = max(1, len(TIMEFRAME_MAP))

    out: List[str] = []
    seen = set()
    for timeframe in requested:
        if timeframe not in TIMEFRAME_MAP or timeframe in seen:
            continue
        seen.add(timeframe)
        out.append(timeframe)
        if len(out) >= max_scan:
            break
    if out:
        return out

    fallback: List[str] = []
    for timeframe in TIMEFRAME_MAP.keys():
        tf = str(timeframe).strip().upper()
        if tf in seen:
            continue
        seen.add(tf)
        fallback.append(tf)
        if len(fallback) >= max_scan:
            break
    return fallback


def _patterns_detect_deps() -> PatternsDetectDeps:
    return PatternsDetectDeps(
        compact_patterns_payload=_compact_patterns_payload,
        fetch_pattern_data=_fetch_pattern_data,
        classic_cfg_cls=_ClassicCfg,
        elliott_cfg_cls=_ElliottCfg,
        fractal_cfg_cls=_FractalCfg,
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
        format_fractal_patterns=_format_fractal_patterns,
        detect_candlestick_patterns=_detect_candlestick_patterns,
        elliott_timeframe_suggestion=_elliott_timeframe_suggestion,
        resolve_elliott_scan_timeframes=_resolve_elliott_scan_timeframes,
        validate_classic_config_errors=_fatal_classic_detector_config_errors,
        validate_fractal_config=_validate_fractal_detector_config,
        summarize_fractal_context=_summarize_fractal_context,
        format_time_minimal=_format_time_minimal,
        to_float_np=__to_float_np,
    )


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
    detail: PatternsDetailLiteral = "full",
) -> Dict[str, Any]:
    """Build the response dict for pattern detection results."""
    # Filter patterns based on include_completed
    filtered = _visible_pattern_rows(patterns, include_completed=include_completed)
    completed_hidden = 0 if include_completed else _count_patterns_with_status(patterns, "completed")
    elliott_preview = (
        _elliott_completed_preview(patterns, timeframe=timeframe)
        if str(mode).lower() == "elliott" and completed_hidden > 0
        else []
    )
    
    resp: Dict[str, Any] = {
        "success": True,
        "symbol": symbol,
        "timeframe": timeframe,
        "lookback": int(limit),
        "mode": mode,
        "patterns": filtered,
        "n_patterns": int(len(filtered)),
    }
    if completed_hidden > 0:
        resp["completed_patterns_hidden"] = int(completed_hidden)
        if elliott_preview:
            resp["completed_patterns_preview"] = elliott_preview
        resp["note"] = (
            _elliott_hidden_completed_note(completed_hidden, elliott_preview)
            if str(mode).lower() == "elliott"
            else (
                f"{int(completed_hidden)} completed pattern(s) hidden; "
                "set include_completed=true to include them."
            )
        )
    if str(mode).lower() == "elliott" and int(len(filtered)) == 0:
        if completed_hidden > 0:
            resp["diagnostic"] = (
                f"No forming Elliott Wave structures detected in {int(limit)} {timeframe} bars. "
                f"{int(completed_hidden)} completed structure(s) were detected but hidden by default. "
                f"{_elliott_timeframe_suggestion(timeframe)} "
                "You can also increase lookback or focus on a clearer trending segment."
            )
        else:
            resp["diagnostic"] = (
                f"No valid Elliott Wave structures detected in {int(limit)} {timeframe} bars. "
                f"{_elliott_timeframe_suggestion(timeframe)} "
                "You can also increase lookback or focus on a clearer trending segment."
            )
    
    # Add data freshness metadata for high timeframes with ancient patterns
    # For MN1 (monthly) and W1 (weekly), check if patterns are very old
    tf_upper = str(timeframe).upper()
    if tf_upper in ("MN1", "W1") and filtered:
        oldest_pattern_time = None
        newest_pattern_time = None
        
        # Extract pattern times from all detected patterns
        for pattern in filtered:
            try:
                if "end_date" in pattern:
                    # Pattern end_date is a string like "2011-08-31 21:00"
                    end_date_str = str(pattern.get("end_date", ""))
                    if end_date_str and oldest_pattern_time is None:
                        oldest_pattern_time = end_date_str
                        newest_pattern_time = end_date_str
                    elif end_date_str:
                        # Keep track of oldest and newest
                        if end_date_str < oldest_pattern_time:
                            oldest_pattern_time = end_date_str
                        if end_date_str > newest_pattern_time:
                            newest_pattern_time = end_date_str
                elif "time" in pattern:
                    time_val = pattern.get("time")
                    if time_val and oldest_pattern_time is None:
                        oldest_pattern_time = str(time_val)
                        newest_pattern_time = str(time_val)
                    elif time_val:
                        time_str = str(time_val)
                        if time_str < oldest_pattern_time:
                            oldest_pattern_time = time_str
                        if time_str > newest_pattern_time:
                            newest_pattern_time = time_str
            except Exception:
                continue
        
        # Add data freshness info if we found pattern times
        if oldest_pattern_time and newest_pattern_time:
            resp["data_freshness"] = {
                "oldest_pattern": oldest_pattern_time,
                "newest_pattern": newest_pattern_time,
            }
            
            # Calculate years spanned
            try:
                # Try to extract years from date strings (format: "YYYY-MM-DD HH:MM")
                oldest_year = int(oldest_pattern_time.split("-")[0]) if oldest_pattern_time else None
                newest_year = int(newest_pattern_time.split("-")[0]) if newest_pattern_time else None
                if oldest_year and newest_year:
                    years_spanned = newest_year - oldest_year
                    if years_spanned >= 10:
                        resp["data_freshness"]["years_spanned"] = years_spanned
                        # Add warning if patterns are very old or span many years
                        if not "warnings" in resp:
                            resp["warnings"] = []
                        if isinstance(resp["warnings"], list):
                            warning_msg = (
                                f"Caution: Monthly timeframe patterns span {years_spanned}+ years "
                                f"(from {oldest_year} to {newest_year}). "
                                "Older patterns may not reflect current market structure. "
                                "Consider using W1 or D1 for more recent patterns, or increase "
                                "the limit to see more recent bars."
                            )
                            resp["warnings"].append(warning_msg)
            except Exception:
                pass
    
    warnings_out = df.attrs.get("warnings")
    if isinstance(warnings_out, list) and warnings_out:
        filtered_warnings = _filter_non_actionable_elliott_warnings(
            warnings_out,
            mode=mode,
            diagnostic=resp.get("diagnostic"),
            n_patterns=int(len(filtered)),
        )
        if filtered_warnings:
            if "warnings" not in resp:
                resp["warnings"] = []
            if isinstance(resp["warnings"], list):
                for warning_text in filtered_warnings:
                    if warning_text not in resp["warnings"]:
                        resp["warnings"].append(warning_text)
    
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

    detail_value = str(detail).lower().strip()
    # Keep this helper's implicit default as full for direct callers that still
    # rely on the legacy shape. The public patterns_detect contract defaults to
    # compact via PatternsDetectRequest.
    if detail_value in ("compact", "highlights"):
        compact_resp = _compact_patterns_payload(resp)
        if detail_value == "highlights":
            return {
                key: value
                for key, value in compact_resp.items()
                if key
                in {
                    "success",
                    "symbol",
                    "timeframe",
                    "lookback",
                    "mode",
                    "n_patterns",
                    "summary",
                    "recent_patterns",
                    "show_all_hint",
                    "warnings",
                    "note",
                    "failed_timeframes",
                }
            }
        return compact_resp
    return resp


def _format_elliott_patterns(df: pd.DataFrame, cfg: _ElliottCfg) -> List[Dict[str, Any]]:
    """Run Elliott detection on prepared data and normalize result rows."""
    pats = _detect_elliott_waves(df, cfg)
    out_list: List[Dict[str, Any]] = []
    n_bars = len(df)

    for p in pats:
        try:
            start_date, end_date = _format_pattern_dates(p.start_time, p.end_time)
            raw_recent_bars = getattr(cfg, "recent_bars", 3)
            if isinstance(raw_recent_bars, (int, float, np.integer, np.floating)) and not isinstance(raw_recent_bars, bool):
                recent_bars = max(1, int(raw_recent_bars))
            else:
                recent_bars = 3
            status = _resolve_elliott_pattern_status(
                p.end_index,
                n_bars=n_bars,
                recent_bars=recent_bars,
            )

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
    return _enrich_elliott_patterns(out_list, df, cfg)


def _format_pattern_timestamp(epoch_value: Any) -> Optional[str]:
    try:
        epoch = float(epoch_value)
    except Exception:
        return None
    if not np.isfinite(epoch):
        return None
    return _format_time_minimal(epoch)


def _format_fractal_patterns(
    df: pd.DataFrame,
    cfg: _FractalCfg,
) -> List[Dict[str, Any]]:
    pats = _detect_fractal_patterns(df, cfg)
    out_list: List[Dict[str, Any]] = []
    close_arr = __to_float_np(df.get("close"))
    current_close: Optional[float] = None
    if close_arr.size > 0:
        last_close = float(close_arr[-1])
        if np.isfinite(last_close):
            current_close = float(last_close)

    for p in pats:
        try:
            start_date, end_date = _format_pattern_dates(p.start_time, p.end_time)
            details = {k: _round_value(v) for k, v in (p.details or {}).items()}
            bias = str(details.get("bias") or p.direction or "").strip().lower()
            if bias not in {"bullish", "bearish", "neutral", "mixed"}:
                bias = str(p.direction).strip().lower()
            row: Dict[str, Any] = {
                "name": p.name,
                "status": p.status,
                "confidence": float(max(0.0, min(1.0, p.confidence))),
                "start_index": int(p.start_index),
                "end_index": int(p.end_index),
                "start_date": start_date,
                "end_date": end_date,
                "direction": str(p.direction),
                "bias": bias,
                "price": float(p.price),
                "level_price": float(p.price),
                "details": details,
            }
            if current_close is not None:
                row["reference_price"] = float(current_close)
            for key in (
                "level_state",
                "level_role",
                "confirmation_index",
                "bars_since_confirmation",
                "breakout_direction",
                "breakout_index",
                "breakout_price",
                "breakout_bars_after_confirmation",
                "breakout_basis",
                "prominence_pct",
            ):
                value = details.get(key)
                if value not in (None, ""):
                    row[key] = value
            confirmation_date = _format_pattern_timestamp(details.get("confirmation_time"))
            if confirmation_date:
                row["confirmation_date"] = confirmation_date
            breakout_date = _format_pattern_timestamp(details.get("breakout_time"))
            if breakout_date:
                row["breakout_date"] = breakout_date
            out_list.append(row)
        except Exception:
            continue
    return out_list


def _summarize_fractal_context(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    active_levels: Dict[str, Dict[str, Any]] = {}
    latest_breakouts: Dict[str, Dict[str, Any]] = {}

    def _latest_row(
        candidates: List[Dict[str, Any]],
        key_name: str,
    ) -> Optional[Dict[str, Any]]:
        latest: Optional[Dict[str, Any]] = None
        latest_value = float("-inf")
        for row in candidates:
            try:
                value = float(row.get(key_name))
            except Exception:
                value = float("-inf")
            if value >= latest_value:
                latest = row
                latest_value = value
        return latest

    for direction in ("bullish", "bearish"):
        active_candidates = [
            row
            for row in rows
            if str(row.get("direction", "")).strip().lower() == direction
            and str(row.get("level_state", "")).strip().lower() == "active"
        ]
        latest_active = _latest_row(active_candidates, "confirmation_index")
        if latest_active is not None:
            item: Dict[str, Any] = {
                "pattern": latest_active.get("name"),
                "level_price": latest_active.get("level_price", latest_active.get("price")),
                "status": latest_active.get("status"),
                "bias": latest_active.get("bias"),
            }
            for key in ("confirmation_date", "bars_since_confirmation", "reference_price"):
                value = latest_active.get(key)
                if value not in (None, ""):
                    item[key] = value
            active_levels[direction] = item

    for breakout_direction in ("bullish", "bearish"):
        breakout_candidates = [
            row
            for row in rows
            if str(row.get("breakout_direction", "")).strip().lower() == breakout_direction
        ]
        latest_breakout = _latest_row(breakout_candidates, "breakout_index")
        if latest_breakout is not None:
            item = {
                "pattern": latest_breakout.get("name"),
                "breakout_direction": latest_breakout.get("breakout_direction"),
                "level_price": latest_breakout.get("level_price", latest_breakout.get("price")),
                "breakout_price": latest_breakout.get("breakout_price"),
                "status": latest_breakout.get("status"),
                "bias": latest_breakout.get("bias"),
            }
            for key in ("breakout_date", "confirmation_date", "reference_price"):
                value = latest_breakout.get(key)
                if value not in (None, ""):
                    item[key] = value
            latest_breakouts[breakout_direction] = item

    out: Dict[str, Any] = {}
    if active_levels:
        out["active_levels"] = active_levels
    if latest_breakouts:
        out["latest_breakouts"] = latest_breakouts
    return out


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
    stock_pattern_bias = {
        "find_double_top": "bearish",
        "find_double_bottom": "bullish",
        "find_triangles": "neutral",
        "find_hns": "bearish",
        "find_reverse_hns": "bullish",
        "find_bullish_flag": "bullish",
        "find_bearish_flag": "bearish",
        "find_uptrend_line": "bullish",
        "find_downtrend_line": "bearish",
    }
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
        if isinstance(details, dict):
            details = dict(details)
            details.setdefault("bias", stock_pattern_bias.get(fn_name, "neutral"))
        d: Dict[str, Any] = {
            "name": name,
            "status": "forming",
            "confidence": float(max(0.0, min(0.95, _infer_stock_pattern_confidence(res)))),
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


def _apply_config_to_obj(cfg: Any, config: Optional[Dict[str, Any]]) -> List[str]:
    """Apply config dict values to a config object's attributes.

    Returns keys that were not applied to the target object, including both
    unknown keys and keys whose values were invalid or could not be coerced.
    """

    if not isinstance(config, dict):
        return []
    invalid_keys: List[str] = []
    for k, v in config.items():
        if not hasattr(cfg, k):
            invalid_keys.append(str(k))
            continue
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
                coerced = _parse_bool_like(v)
                if coerced is _UNPARSED_BOOL:
                    invalid_keys.append(str(k))
                    continue
                setattr(cfg, k, bool(coerced))
            elif current is None:
                setattr(cfg, k, v)
            else:
                setattr(cfg, k, type(current)(v))
        except Exception:
            invalid_keys.append(str(k))
    deduped: List[str] = []
    for key in invalid_keys:
        if key not in deduped:
            deduped.append(key)
    return deduped

@mcp.tool()
def patterns_detect(
    request: PatternsDetectRequest,
) -> Dict[str, Any]:
    """Detect chart patterns (candlestick, classic chart patterns, fractals, or Elliott Wave).
    
    **REQUIRED**: symbol parameter must be provided (e.g., "EURUSD", "BTCUSD")
    
    By default (mode="all"), runs all pattern types across a default
    multi-timeframe set (`M30`, `H1`, `H4`, `D1`, `W1`)
    and returns a sectioned response with candlestick, classic, fractal, and Elliott results.
    
    Parameters:
    -----------
    symbol : str (REQUIRED)
        Trading symbol to analyze (e.g., "EURUSD", "GBPUSD", "BTCUSD")
    
    timeframe : str, optional
        Chart timeframe: "M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1", "MN1"
        For `mode="all"`, when omitted, a default multi-timeframe set
        (`M30`, `H1`, `H4`, `D1`, `W1`) is scanned automatically.
        For `mode="elliott"`, when omitted, a higher-structure subset
        (`H1`, `H4`, `D1`) is scanned automatically.
    
    mode : str, optional (default="all")
        Pattern detection method:
        - "all": Comprehensive scan — candlestick + classic + fractal + Elliott across
          multiple timeframes. Returns sectioned output.
        - "candlestick": Japanese candlestick patterns (Doji, Hammer, Engulfing, etc.)
        - "classic": Chart patterns (Head & Shoulders, Triangles, Flags, etc.)
        - "chart": Alias for classic chart patterns
        - "fractal": Bill Williams-style bullish/bearish fractal levels with breakout context
        - "elliott": Elliott Wave patterns

    detail : str, optional (default="compact")
        Output verbosity:
        - "highlights": quick-read highlights and aggregate bias/counts only.
        - "compact": trader-focused summary with recent patterns and pattern mix.
        - "standard": sectioned all-mode output with trimmed pattern rows.
        - "full": complete pattern rows suitable for research/debugging.
    
    limit : int, optional (default=500)
        Number of historical bars to analyze
    
    Candlestick Mode Parameters:
    ----------------------------
    min_strength : float, optional (default=0.90)
        Minimum semantic conviction threshold (0.0 to 1.0). This filters on a
        normalized candlestick strength score that combines pattern reliability,
        multi-bar span, and any raw detector bonus rather than raw pandas_ta
        signal magnitude alone.
    
    min_gap : int, optional (default=3)
        Minimum gap between patterns (in bars)
    
    robust_only : bool, optional (default=False)
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
        Useful fractal options include:
        - left_bars, right_bars
        - breakout_basis: "close" or "high_low"
        - min_prominence_pct, confidence_prominence_cap_pct

    engine : str, optional (default="native")
        Classic engine selection: "native", "stock_pattern",
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
    # Comprehensive scan across all pattern types and timeframes (default)
    patterns_detect(symbol="EURUSD")
    
    # Comprehensive scan on a single timeframe
    patterns_detect(symbol="EURUSD", timeframe="H4")
    
    # Detect candlestick patterns only
    patterns_detect(symbol="EURUSD", mode="candlestick", timeframe="M15", min_strength=0.90, top_k=3)
    
    # Detect classic chart patterns
    patterns_detect(symbol="GBPUSD", mode="classic", limit=500)

    # Detect fractal levels and breakouts
    patterns_detect(symbol="EURUSD", mode="fractal", timeframe="H1", config={"breakout_basis": "high_low"})

    # Detect Elliott Wave patterns
    patterns_detect(symbol="BTCUSD", mode="elliott", timeframe="H4", detail="full")
    """
    def _run() -> Dict[str, Any]:
        connection_error = _patterns_connection_error()
        if connection_error is not None:
            return connection_error
        return run_patterns_detect(request, _patterns_detect_deps())

    return run_logged_operation(
        logger,
        operation="patterns_detect",
        symbol=request.symbol,
        timeframe=request.timeframe,
        mode=request.mode,
        detail=request.detail,
        func=_run,
    )
