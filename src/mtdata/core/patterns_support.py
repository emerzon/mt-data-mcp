import importlib
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..utils.utils import _format_time_minimal, _safe_float, to_float_np as __to_float_np

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


def _pattern_label(row: Dict[str, Any]) -> Optional[str]:
    for key in ("pattern", "name", "wave_type"):
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return None


def _normalize_pattern_bias(value: Any) -> Optional[str]:
    s = str(value or "").strip().lower()
    if not s:
        return None
    if s in {"bull", "bullish", "buy", "long", "up", "positive"}:
        return "bullish"
    if s in {"bear", "bearish", "sell", "short", "down", "negative"}:
        return "bearish"
    if s in {"neutral", "mixed", "flat", "sideways"}:
        return "neutral"
    return None


def _row_pattern_bias(row: Dict[str, Any]) -> Optional[str]:
    bias = _normalize_pattern_bias(row.get("bias"))
    if bias:
        return bias
    direction = _normalize_pattern_bias(row.get("direction"))
    if direction:
        return direction
    details = row.get("details")
    if isinstance(details, dict):
        for key in ("breakout_direction", "breakout_expected"):
            bias = _normalize_pattern_bias(details.get(key))
            if bias:
                return bias
    return None


def _row_confidence_weight(row: Dict[str, Any]) -> float:
    conf = _safe_float(row.get("confidence"))
    if conf is None:
        conf = _safe_float(row.get("strength"))
    if conf is None or not np.isfinite(conf):
        conf = 0.5
    return float(max(0.0, min(1.0, conf)))


def _summarize_pattern_bias(rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    bullish_score = 0.0
    bearish_score = 0.0
    bullish_count = 0
    bearish_count = 0
    neutral_count = 0
    strongest_bullish: Optional[Dict[str, Any]] = None
    strongest_bearish: Optional[Dict[str, Any]] = None

    for row in rows:
        if not isinstance(row, dict):
            continue
        bias = _row_pattern_bias(row)
        if bias is None:
            continue
        weight = _row_confidence_weight(row)
        label = _pattern_label(row)
        if bias == "bullish":
            bullish_count += 1
            bullish_score += weight
            if strongest_bullish is None or weight > float(strongest_bullish.get("confidence", 0.0)):
                strongest_bullish = {"pattern": label, "confidence": float(weight)}
        elif bias == "bearish":
            bearish_count += 1
            bearish_score += weight
            if strongest_bearish is None or weight > float(strongest_bearish.get("confidence", 0.0)):
                strongest_bearish = {"pattern": label, "confidence": float(weight)}
        else:
            neutral_count += 1

    if bullish_count == 0 and bearish_count == 0 and neutral_count == 0:
        return None

    directional_total = bullish_score + bearish_score
    net_score = bullish_score - bearish_score
    net_conf = float(abs(net_score) / directional_total) if directional_total > 1e-9 else 0.0
    conflict = bool(bullish_count > 0 and bearish_count > 0)
    if directional_total <= 1e-9:
        net_bias = "neutral"
    elif conflict and net_conf < 0.2:
        net_bias = "mixed"
    else:
        net_bias = "bullish" if net_score > 0 else "bearish"

    out: Dict[str, Any] = {
        "net_bias": net_bias,
        "net_confidence": float(max(0.0, min(1.0, net_conf))),
        "net_score": _round_value(net_score),
        "bullish_score": _round_value(bullish_score),
        "bearish_score": _round_value(bearish_score),
        "bullish_patterns": int(bullish_count),
        "bearish_patterns": int(bearish_count),
        "neutral_patterns": int(neutral_count),
        "conflict": conflict,
    }
    if strongest_bullish:
        out["strongest_bullish"] = strongest_bullish
    if strongest_bearish:
        out["strongest_bearish"] = strongest_bearish
    return out


def _compact_patterns_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict) or payload.get("error"):
        return payload

    rows: List[Dict[str, Any]] = []
    if isinstance(payload.get("patterns"), list):
        rows = [row for row in payload.get("patterns", []) if isinstance(row, dict)]
    elif isinstance(payload.get("data"), list):
        rows = [row for row in payload.get("data", []) if isinstance(row, dict)]
    if not rows:
        return payload

    counts: Dict[str, int] = {}
    status_counts: Dict[str, int] = {}
    tf_counts: Dict[str, int] = {}
    indexed_rows: List[Tuple[int, Dict[str, Any]]] = []
    for idx, row in enumerate(rows):
        indexed_rows.append((idx, row))
        label = _pattern_label(row)
        if label:
            counts[label] = counts.get(label, 0) + 1
        status = row.get("status")
        if status not in (None, ""):
            status_key = str(status)
            status_counts[status_key] = status_counts.get(status_key, 0) + 1
        timeframe = row.get("timeframe")
        if timeframe not in (None, ""):
            timeframe_key = str(timeframe)
            tf_counts[timeframe_key] = tf_counts.get(timeframe_key, 0) + 1

    def _sort_key(item: Tuple[int, Dict[str, Any]]) -> Tuple[float, float, int]:
        idx, row = item
        end_idx = _safe_float(row.get("end_index"))
        conf = _safe_float(row.get("confidence")) or 0.0
        return (end_idx if end_idx is not None else float(idx), conf, idx)

    preview_limit = 8
    preview_rows = [row for _, row in sorted(indexed_rows, key=_sort_key, reverse=True)[:preview_limit]]
    recent_rows: List[Dict[str, Any]] = []
    for row in preview_rows:
        item: Dict[str, Any] = {}
        if row.get("timeframe") not in (None, ""):
            item["timeframe"] = row.get("timeframe")
        label = _pattern_label(row)
        if label:
            item["pattern"] = label
        for key in (
            "time",
            "start_date",
            "end_date",
            "status",
            "confidence",
            "strength",
            "direction",
            "bias",
            "price",
            "reference_price",
            "target_price",
            "invalidation_price",
            "bars_to_completion",
        ):
            value = row.get(key)
            if value not in (None, ""):
                item[key] = value
        if not item:
            item = dict(row)
        recent_rows.append(item)

    strongest_pattern: Optional[Dict[str, Any]] = None
    if preview_rows:
        best = max(preview_rows, key=lambda row: (_safe_float(row.get("confidence")) or 0.0))
        best_label = _pattern_label(best)
        strongest_pattern = {}
        if best_label:
            strongest_pattern["pattern"] = best_label
        for key in (
            "timeframe",
            "time",
            "end_date",
            "status",
            "confidence",
            "strength",
            "direction",
            "bias",
            "price",
            "reference_price",
            "target_price",
            "invalidation_price",
        ):
            value = best.get(key)
            if value not in (None, ""):
                strongest_pattern[key] = value
        if not strongest_pattern:
            strongest_pattern = None

    total = payload.get("n_patterns")
    if total in (None, ""):
        total = payload.get("count")
    try:
        total_i = int(total)
    except Exception:
        total_i = len(rows)

    summary: Dict[str, Any] = {
        "unique_patterns": len(counts),
        "showing_recent": len(recent_rows),
        "more_patterns": max(0, total_i - len(recent_rows)),
    }
    if counts:
        summary["pattern_mix"] = [
            {"pattern": name, "count": count}
            for name, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:5]
        ]
    if status_counts:
        summary["status_counts"] = status_counts
    if tf_counts:
        summary["timeframe_mix"] = [
            {"timeframe": timeframe, "count": count}
            for timeframe, count in sorted(tf_counts.items(), key=lambda item: (-item[1], item[0]))
        ]
    signal_bias = _summarize_pattern_bias(rows)
    if signal_bias:
        summary["signal_bias"] = signal_bias
    if strongest_pattern:
        summary["strongest_pattern"] = strongest_pattern

    compact: Dict[str, Any] = {
        "success": bool(payload.get("success", True)),
        "symbol": payload.get("symbol"),
        "timeframe": payload.get("timeframe"),
        "lookback": payload.get("lookback"),
        "mode": payload.get("mode"),
        "n_patterns": total_i,
        "summary": summary,
    }
    if summary["more_patterns"] > 0:
        compact["show_all_hint"] = "Use --detail full to show all detected patterns."
    compact["recent_patterns"] = recent_rows

    for key in ("engine", "engines_run", "engine_findings", "engine_errors", "scanned_timeframes"):
        value = payload.get(key)
        if value not in (None, "", [], {}):
            compact[key] = value

    findings = payload.get("findings")
    if isinstance(findings, list):
        compact["findings"] = findings
        tf_summary: List[Dict[str, Any]] = []
        for item in findings:
            if not isinstance(item, dict):
                continue
            tf_summary.append({"timeframe": item.get("timeframe"), "n_patterns": item.get("n_patterns")})
        if tf_summary:
            compact["timeframe_findings"] = tf_summary

    if isinstance(payload.get("failed_timeframes"), dict) and payload.get("failed_timeframes"):
        compact["failed_timeframes"] = payload.get("failed_timeframes")

    if "series_close" in payload:
        compact["series_close"] = payload.get("series_close")
    if "series_time" in payload:
        compact["series_time"] = payload.get("series_time")
    if "series_epoch" in payload:
        compact["series_epoch"] = payload.get("series_epoch")
    if "series_by_timeframe" in payload:
        compact["series_by_timeframe"] = payload.get("series_by_timeframe")

    return compact


def _detail_float(details: Dict[str, Any], key: str) -> Optional[float]:
    try:
        value = float(details.get(key))
        if np.isfinite(value):
            return value
    except Exception:
        return None
    return None


def _close_price_at_index(df: pd.DataFrame, end_index: Any) -> Optional[float]:
    close = __to_float_np(df.get("close"))
    if close.size <= 0:
        return None
    try:
        idx = int(end_index)
    except Exception:
        idx = int(close.size - 1)
    idx = max(0, min(idx, int(close.size - 1)))
    price = _safe_float(close[idx])
    if price is not None and np.isfinite(price):
        return float(price)
    last = _safe_float(close[-1])
    return float(last) if last is not None and np.isfinite(last) else None


def _infer_classic_bias(name: Any, details: Dict[str, Any]) -> str:
    for key in ("breakout_direction", "breakout_expected"):
        bias = _normalize_pattern_bias(details.get(key))
        if bias in ("bullish", "bearish"):
            return bias

    name_text = str(name or "").strip().lower()
    if not name_text:
        return "neutral"
    if "inverse head and shoulders" in name_text:
        return "bullish"
    if "head and shoulders" in name_text:
        return "bearish"
    if "falling wedge" in name_text:
        return "bullish"
    if "rising wedge" in name_text:
        return "bearish"
    if "double bottom" in name_text or "triple bottom" in name_text:
        return "bullish"
    if "double top" in name_text or "triple top" in name_text:
        return "bearish"
    if "rounding bottom" in name_text:
        return "bullish"
    if "rounding top" in name_text:
        return "bearish"
    if "cup and handle" in name_text:
        return "bullish"
    if "ascending triangle" in name_text:
        return "bullish"
    if "descending triangle" in name_text:
        return "bearish"
    if "bull" in name_text or "ascending" in name_text or "uptrend" in name_text:
        return "bullish"
    if "bear" in name_text or "descending" in name_text or "downtrend" in name_text:
        return "bearish"
    return "neutral"


def _classic_price_levels(details: Dict[str, Any], end_index: Any) -> Dict[str, float]:
    levels: Dict[str, float] = {}
    for key in (
        "support",
        "resistance",
        "level",
        "neckline",
        "breakout_level",
        "left_rim",
        "right_rim",
        "bottom",
        "line_level_recent",
    ):
        value = _detail_float(details, key)
        if value is not None:
            levels[key] = value

    try:
        idx_float = float(end_index)
    except Exception:
        idx_float = 0.0

    top_slope = _detail_float(details, "top_slope")
    top_intercept = _detail_float(details, "top_intercept")
    bottom_slope = _detail_float(details, "bottom_slope")
    bottom_intercept = _detail_float(details, "bottom_intercept")
    if top_slope is not None and top_intercept is not None:
        levels.setdefault("resistance", float(top_slope * idx_float + top_intercept))
    if bottom_slope is not None and bottom_intercept is not None:
        levels.setdefault("support", float(bottom_slope * idx_float + bottom_intercept))

    upper_slope = _detail_float(details, "upper_slope")
    upper_intercept = _detail_float(details, "upper_intercept")
    lower_slope = _detail_float(details, "lower_slope")
    lower_intercept = _detail_float(details, "lower_intercept")
    if upper_slope is not None and upper_intercept is not None:
        levels.setdefault("resistance", float(upper_slope * idx_float + upper_intercept))
    if lower_slope is not None and lower_intercept is not None:
        levels.setdefault("support", float(lower_slope * idx_float + lower_intercept))

    return {key: float(value) for key, value in levels.items() if np.isfinite(value)}


def _classic_pattern_height(
    levels: Dict[str, float],
    details: Dict[str, Any],
    reference_price: Optional[float],
) -> Optional[float]:
    support = levels.get("support")
    resistance = levels.get("resistance")
    if support is not None and resistance is not None and resistance > support:
        return float(resistance - support)

    head = _detail_float(details, "head")
    neckline = levels.get("neckline")
    if head is not None and neckline is not None:
        return float(abs(head - neckline))

    left_rim = levels.get("left_rim")
    bottom = levels.get("bottom")
    if left_rim is not None and bottom is not None and left_rim > bottom:
        return float(left_rim - bottom)

    pole_return_pct = _detail_float(details, "pole_return_pct")
    if pole_return_pct is not None and reference_price is not None:
        return float(abs(reference_price * pole_return_pct / 100.0))

    amplitude_pct = _detail_float(details, "amplitude_pct")
    if amplitude_pct is not None and reference_price is not None:
        return float(abs(reference_price * amplitude_pct / 100.0))
    return None


def _enrich_classic_pattern_row(row: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
    out = dict(row)
    details = out.get("details")
    if not isinstance(details, dict):
        details = {}
    name = out.get("name") or out.get("pattern")
    bias = _infer_classic_bias(name, details)
    reference_price = _close_price_at_index(df, out.get("end_index"))
    levels = _classic_price_levels(details, out.get("end_index"))
    height = _classic_pattern_height(levels, details, reference_price)

    support = levels.get("support")
    resistance = levels.get("resistance")
    neckline = levels.get("neckline")
    target: Optional[float] = None
    invalidation: Optional[float] = None
    if reference_price is not None:
        if bias == "bullish":
            if resistance is not None and resistance > reference_price:
                target = resistance
            elif height is not None and height > 0:
                target = reference_price + height
            if support is not None and support < reference_price:
                invalidation = support
            elif neckline is not None and neckline < reference_price:
                invalidation = neckline
            elif height is not None and height > 0:
                invalidation = reference_price - 0.5 * height
        elif bias == "bearish":
            if support is not None and support < reference_price:
                target = support
            elif height is not None and height > 0:
                target = reference_price - height
            if resistance is not None and resistance > reference_price:
                invalidation = resistance
            elif neckline is not None and neckline > reference_price:
                invalidation = neckline
            elif height is not None and height > 0:
                invalidation = reference_price + 0.5 * height

    out["bias"] = bias
    if reference_price is not None:
        out["reference_price"] = _round_value(reference_price)
    if target is not None:
        out["target_price"] = _round_value(target)
    if invalidation is not None:
        out["invalidation_price"] = _round_value(invalidation)
    if levels:
        out["price_levels"] = {key: _round_value(value) for key, value in levels.items()}
    return out


def _enrich_classic_patterns(rows: List[Dict[str, Any]], df: pd.DataFrame) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        out.append(_enrich_classic_pattern_row(row, df))
    return out


def _summarize_engine_findings(
    per_engine: Dict[str, List[Dict[str, Any]]],
    engines: List[str],
    include_completed: bool,
) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []
    for engine in engines:
        rows = [row for row in per_engine.get(engine, []) if isinstance(row, dict)]
        n_total = int(len(rows))
        n_forming = int(sum(1 for row in rows if str(row.get("status", "")).lower() == "forming"))
        n_completed = int(sum(1 for row in rows if str(row.get("status", "")).lower() == "completed"))
        n_shown = n_total if include_completed else n_forming
        item: Dict[str, Any] = {
            "engine": engine,
            "n_patterns": n_shown,
            "n_forming": n_forming,
            "n_completed": n_completed if include_completed else 0,
            "n_patterns_total": n_total,
        }
        if not include_completed and n_completed > 0:
            item["n_completed_hidden"] = n_completed
        findings.append(item)
    return findings


def _estimate_classic_bars_to_completion(
    name: str,
    details: Dict[str, Any],
    start_idx: int,
    end_idx: int,
    n_bars: int,
) -> Optional[int]:
    try:
        length = max(1, int(end_idx) - int(start_idx) + 1)
        name_text = str(name).lower()
        if all(key in details for key in ("top_slope", "top_intercept", "bottom_slope", "bottom_intercept")):
            top_slope = float(details.get("top_slope"))
            top_intercept = float(details.get("top_intercept"))
            bottom_slope = float(details.get("bottom_slope"))
            bottom_intercept = float(details.get("bottom_intercept"))
            denom = top_slope - bottom_slope
            if abs(denom) <= 1e-12:
                return None
            t_star = (bottom_intercept - top_intercept) / denom
            bars = int(max(0, int(round(t_star - (n_bars - 1)))))
            return int(min(max(0, bars), 3 * length))
        if all(key in details for key in ("upper_slope", "upper_intercept", "lower_slope", "lower_intercept")):
            top_slope = float(details.get("upper_slope"))
            top_intercept = float(details.get("upper_intercept"))
            bottom_slope = float(details.get("lower_slope"))
            bottom_intercept = float(details.get("lower_intercept"))
            denom = top_slope - bottom_slope
            if abs(denom) <= 1e-12:
                return None
            t_star = (bottom_intercept - top_intercept) / denom
            bars = int(max(0, int(round(t_star - (n_bars - 1)))))
            return int(min(max(0, bars), 3 * length))
        if name_text in ("pennants", "flag", "bull pennants", "bear pennants", "bull flag", "bear flag"):
            return int(max(1, min(2 * length, int(round(0.3 * length)))))
        if "pennant" in name_text or "flag" in name_text:
            return int(max(1, min(2 * length, int(round(0.3 * length)))))
    except Exception:
        return None
    return None


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d %H:%M")
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(item) for item in value]
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
    candidate_modules = (
        "stock_pattern.utils",
        "stockpattern.utils",
    )
    required = ("get_max_min", "find_double_top", "find_double_bottom")
    last_err: Optional[str] = None
    for module_name in candidate_modules:
        if module_name in _STOCK_PATTERN_UTILS_CACHE:
            return _STOCK_PATTERN_UTILS_CACHE[module_name], None
        try:
            module = importlib.import_module(module_name)
        except Exception as ex:
            last_err = str(ex)
            continue
        if not all(callable(getattr(module, fn, None)) for fn in required):
            last_err = f"module '{module_name}' missing required stock-pattern functions"
            continue
        _STOCK_PATTERN_UTILS_CACHE[module_name] = module
        return module, None

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
    volume_col = "volume" if "volume" in src.columns else ("tick_volume" if "tick_volume" in src.columns else "Volume")

    out = pd.DataFrame(
        {
            "Open": pd.to_numeric(src.get(open_col), errors="coerce"),
            "High": pd.to_numeric(src.get(high_col), errors="coerce"),
            "Low": pd.to_numeric(src.get(low_col), errors="coerce"),
            "Close": pd.to_numeric(src.get(close_col), errors="coerce"),
            "Volume": pd.to_numeric(src.get(volume_col), errors="coerce"),
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
        touch_value = _to_float_safe(touches, 0.0)
        return float(max(0.35, min(0.95, 0.5 + 0.05 * touch_value)))
    return 0.6


def _normalize_engine_name(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_")


def _parse_engine_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        parts = [part.strip() for part in value.replace(";", ",").split(",") if part.strip()]
        return [_normalize_engine_name(part) for part in parts]
    if isinstance(value, (list, tuple, set)):
        return [_normalize_engine_name(part) for part in value if str(part).strip()]
    return [_normalize_engine_name(value)]


def _parse_native_scale_factors(config: Optional[Dict[str, Any]]) -> List[float]:
    cfg_map = config if isinstance(config, dict) else {}
    raw = cfg_map.get("native_scale_factors", cfg_map.get("native_scales"))
    vals: List[float] = []
    if isinstance(raw, str):
        parts = [part.strip() for part in raw.replace(";", ",").split(",") if part.strip()]
        for part in parts:
            try:
                vals.append(float(part))
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
    for value in vals:
        if not np.isfinite(value) or value <= 0:
            continue
        clamped = float(max(0.3, min(3.0, value)))
        key = round(clamped, 4)
        if key in seen:
            continue
        seen.add(key)
        out.append(clamped)
    if 1.0 not in [round(value, 4) for value in out]:
        out.insert(0, 1.0)
    return out


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
    out = {engine: 1.0 for engine in engines}
    if not isinstance(ensemble_weights, dict):
        return out
    for key, value in ensemble_weights.items():
        engine_key = _normalize_engine_name(key)
        if engine_key not in out:
            continue
        try:
            weight = float(value)
            if np.isfinite(weight) and weight > 0:
                out[engine_key] = weight
        except Exception:
            continue
    return out


def _merge_classic_ensemble(
    engine_patterns: Dict[str, List[Dict[str, Any]]],
    weights: Dict[str, float],
    overlap_threshold: float = 0.5,
) -> List[Dict[str, Any]]:
    groups: List[Dict[str, Any]] = []
    groups_by_name: Dict[str, List[Dict[str, Any]]] = {}
    for engine, patterns in engine_patterns.items():
        for pattern in patterns:
            try:
                name_norm = str(pattern.get("name", "")).strip().lower()
                start_idx = int(pattern.get("start_index", 0))
                end_idx = int(pattern.get("end_index", start_idx))
            except Exception:
                continue
            target: Optional[Dict[str, Any]] = None
            same_name_groups = groups_by_name.setdefault(name_norm, [])
            for group in same_name_groups:
                if _interval_overlap_ratio(
                    start_idx,
                    end_idx,
                    int(group["start_index"]),
                    int(group["end_index"]),
                ) >= float(overlap_threshold):
                    target = group
                    break
            if target is None:
                target = {
                    "name_norm": name_norm,
                    "start_index": start_idx,
                    "end_index": end_idx,
                    "items": [],
                }
                groups.append(target)
                same_name_groups.append(target)
            target["start_index"] = min(int(target["start_index"]), start_idx)
            target["end_index"] = max(int(target["end_index"]), end_idx)
            target["items"].append((engine, pattern))

    merged: List[Dict[str, Any]] = []
    for group in groups:
        items: List[Tuple[str, Dict[str, Any]]] = group.get("items", [])
        if not items:
            continue
        by_engine: Dict[str, Dict[str, Any]] = {}
        for engine, pattern in items:
            prev = by_engine.get(engine)
            if prev is None or float(pattern.get("confidence", 0.0)) > float(prev.get("confidence", 0.0)):
                by_engine[engine] = pattern
        engines = list(by_engine.keys())
        total_w = float(sum(weights.get(engine, 1.0) for engine in engines)) or 1.0
        conf = float(
            sum(
                float(by_engine[engine].get("confidence", 0.0)) * float(weights.get(engine, 1.0))
                for engine in engines
            )
            / total_w
        )
        anchor_engine = max(engines, key=lambda engine: float(by_engine[engine].get("confidence", 0.0)))
        anchor = dict(by_engine[anchor_engine])
        statuses = [str(by_engine[engine].get("status", "forming")).lower() for engine in engines]
        anchor["status"] = "completed" if statuses and all(status == "completed" for status in statuses) else "forming"
        anchor["confidence"] = float(max(0.0, min(1.0, conf)))
        anchor["support_count"] = int(len(engines))
        anchor["source_engines"] = engines
        details = anchor.get("details")
        if not isinstance(details, dict):
            details = {}
        details = dict(details)
        details["engine_confidences"] = {
            engine: float(max(0.0, min(1.0, float(by_engine[engine].get("confidence", 0.0)))))
            for engine in engines
        }
        details["consensus_support"] = int(len(engines))
        anchor["details"] = details
        merged.append(anchor)

    merged.sort(
        key=lambda pattern: (
            int(pattern.get("support_count", 1)),
            float(pattern.get("confidence", 0.0)),
            int(pattern.get("end_index", -1)),
        ),
        reverse=True,
    )
    return merged


def _format_pattern_dates(start_time: Optional[float], end_time: Optional[float]) -> Tuple[Optional[str], Optional[str]]:
    """Format epoch times to date strings."""
    start_epoch = float(start_time) if start_time is not None else None
    end_epoch = float(end_time) if end_time is not None else None

    try:
        start_date = _format_time_minimal(start_epoch) if start_epoch is not None else None
    except Exception:
        start_date = None

    try:
        end_date = _format_time_minimal(end_epoch) if end_epoch is not None else None
    except Exception:
        end_date = None

    return start_date, end_date
