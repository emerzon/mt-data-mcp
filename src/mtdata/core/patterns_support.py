import importlib
import threading
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..patterns.common import interval_overlap_ratio as _interval_overlap_ratio
from ..utils.utils import _format_time_minimal, _safe_float
from ..utils.utils import to_float_np as __to_float_np

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
_STOCK_PATTERN_UTILS_CACHE_LOCK = threading.Lock()


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


def _pattern_status_value(row: Any) -> str:
    try:
        return str(row.get("status", "")).strip().lower()
    except Exception:
        return ""


def _pattern_has_status(row: Any, status: str) -> bool:
    return _pattern_status_value(row) == str(status).strip().lower()


def _count_patterns_with_status(rows: List[Dict[str, Any]], status: str) -> int:
    return int(sum(1 for row in rows if _pattern_has_status(row, status)))


def _visible_pattern_rows(
    rows: List[Dict[str, Any]],
    *,
    include_completed: bool,
) -> List[Dict[str, Any]]:
    if include_completed:
        return rows
    return [row for row in rows if _pattern_has_status(row, "forming")]


def _resolve_elliott_pattern_status(
    end_index: Any,
    *,
    n_bars: int,
    recent_bars: int,
) -> str:
    try:
        recent = max(1, int(recent_bars))
    except Exception:
        recent = 1
    try:
        end_idx = int(end_index)
    except Exception:
        end_idx = -1
    return "forming" if end_idx >= int(max(0, n_bars - recent)) else "completed"


def _elliott_preview_sort_key(row: Dict[str, Any]) -> Tuple[float, float, str]:
    conf = _safe_float(row.get("confidence")) or 0.0
    end_idx = _safe_float(row.get("end_index"))
    label = _pattern_label(row) or ""
    return (
        float(conf),
        float(end_idx) if end_idx is not None else float("-inf"),
        label,
    )


def _elliott_completed_preview(
    patterns: List[Dict[str, Any]],
    *,
    timeframe: Optional[str] = None,
    limit: int = 3,
) -> List[Dict[str, Any]]:
    if int(limit) <= 0:
        return []
    completed_rows = [
        row
        for row in patterns
        if isinstance(row, dict) and _pattern_has_status(row, "completed")
    ]
    preview: List[Dict[str, Any]] = []
    for row in sorted(completed_rows, key=_elliott_preview_sort_key, reverse=True)[
        : int(limit)
    ]:
        item: Dict[str, Any] = {}
        tf_value = timeframe if timeframe not in (None, "") else row.get("timeframe")
        if tf_value not in (None, ""):
            item["timeframe"] = tf_value
        label = _pattern_label(row)
        if label:
            item["pattern"] = label
        for key in (
            "status",
            "confidence",
            "start_date",
            "end_date",
            "start_index",
            "end_index",
        ):
            value = row.get(key)
            if value not in (None, ""):
                item[key] = value
        details = row.get("details")
        if isinstance(details, dict):
            direction = details.get("sequence_direction")
            if direction in (None, ""):
                direction = details.get("trend")
            if direction not in (None, ""):
                item["direction"] = direction
            if "pattern_confirmed" in details:
                item["pattern_confirmed"] = bool(details.get("pattern_confirmed"))
            if "has_unconfirmed_terminal_pivot" in details:
                item["has_unconfirmed_terminal_pivot"] = bool(
                    details.get("has_unconfirmed_terminal_pivot")
                )
        if not item:
            item = dict(row)
        preview.append(item)
    return preview


def _elliott_hidden_completed_note(
    completed_hidden: int,
    preview: Optional[List[Dict[str, Any]]] = None,
) -> str:
    strongest_text = ""
    if isinstance(preview, list) and preview:
        strongest = preview[0]
        parts: List[str] = []
        timeframe = strongest.get("timeframe")
        if timeframe not in (None, ""):
            parts.append(str(timeframe))
        pattern = strongest.get("pattern")
        if pattern not in (None, ""):
            parts.append(str(pattern))
        direction = strongest.get("direction")
        if direction not in (None, ""):
            parts.append(str(direction))
        strongest_text = " ".join(parts).strip()
        start_date = strongest.get("start_date")
        end_date = strongest.get("end_date")
        if start_date not in (None, "") and end_date not in (None, ""):
            strongest_text = f"{strongest_text} {start_date} -> {end_date}".strip()
        conf = _safe_float(strongest.get("confidence"))
        if conf is not None:
            strongest_text = f"{strongest_text} (confidence {float(conf):.2f})".strip()
    note = f"{int(completed_hidden)} completed pattern(s) hidden; "
    if strongest_text:
        note += f"strongest hidden count: {strongest_text}; "
    note += "set include_completed=true to include them."
    return note


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
        for key in ("bias", "pattern_bias", "breakout_direction", "breakout_expected"):
            bias = _normalize_pattern_bias(details.get(key))
            if bias:
                return bias
        pattern_family = str(details.get("pattern_family") or "").strip().lower()
        trend_context = str(details.get("trend_context") or "").strip().lower()
        if pattern_family != "correction" and trend_context != "counter_trend":
            for key in ("sequence_direction", "trend"):
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
            if strongest_bullish is None or weight > float(
                strongest_bullish.get("confidence", 0.0)
            ):
                strongest_bullish = {"pattern": label, "confidence": float(weight)}
        elif bias == "bearish":
            bearish_count += 1
            bearish_score += weight
            if strongest_bearish is None or weight > float(
                strongest_bearish.get("confidence", 0.0)
            ):
                strongest_bearish = {"pattern": label, "confidence": float(weight)}
        else:
            neutral_count += 1

    if bullish_count == 0 and bearish_count == 0 and neutral_count == 0:
        return None

    directional_total = bullish_score + bearish_score
    net_score = bullish_score - bearish_score
    net_conf = (
        float(abs(net_score) / directional_total) if directional_total > 1e-9 else 0.0
    )
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


def _compact_patterns_payload(payload: Dict[str, Any]) -> Dict[str, Any]:  # noqa: C901
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
    preview_rows = [
        row
        for _, row in sorted(indexed_rows, key=_sort_key, reverse=True)[:preview_limit]
    ]
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
            "confirmation_date",
            "breakout_date",
            "status",
            "confidence",
            "strength",
            "direction",
            "bias",
            "price",
            "level_price",
            "level_state",
            "reference_price",
            "breakout_direction",
            "breakout_price",
            "target_price",
            "target_stale",
            "target_reference_age_bars",
            "invalidation_price",
            "bars_to_completion",
            "bars_since_confirmation",
        ):
            value = row.get(key)
            if value not in (None, ""):
                item[key] = value
        if not item:
            item = dict(row)
        recent_rows.append(item)

    strongest_pattern: Optional[Dict[str, Any]] = None
    if preview_rows:
        best = max(
            preview_rows, key=lambda row: _safe_float(row.get("confidence")) or 0.0
        )
        best_label = _pattern_label(best)
        strongest_pattern = {}
        if best_label:
            strongest_pattern["pattern"] = best_label
        for key in (
            "timeframe",
            "time",
            "end_date",
            "confirmation_date",
            "breakout_date",
            "status",
            "confidence",
            "strength",
            "direction",
            "bias",
            "price",
            "level_price",
            "level_state",
            "reference_price",
            "breakout_direction",
            "breakout_price",
            "target_price",
            "target_stale",
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
        "more_patterns": max(0, total_i - len(recent_rows)),
    }
    if counts:
        summary["pattern_mix"] = [
            {"pattern": name, "count": count}
            for name, count in sorted(
                counts.items(), key=lambda item: (-item[1], item[0])
            )[:5]
        ]
    if status_counts:
        summary["status_counts"] = status_counts
    if tf_counts:
        summary["timeframe_mix"] = [
            {"timeframe": timeframe, "count": count}
            for timeframe, count in sorted(
                tf_counts.items(), key=lambda item: (-item[1], item[0])
            )
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
        compact["show_all_hint"] = "Set detail='full' to show all detected patterns."
    compact["recent_patterns"] = recent_rows

    for key in (
        "engine",
        "engines_run",
        "engine_findings",
        "engine_errors",
        "scanned_timeframes",
    ):
        value = payload.get(key)
        if value not in (None, "", [], {}):
            compact[key] = value
    for key in (
        "warnings",
        "note",
        "completed_patterns_hidden",
        "completed_patterns_preview",
    ):
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
            tf_summary.append(
                {
                    "timeframe": item.get("timeframe"),
                    "n_patterns": item.get("n_patterns"),
                }
            )
        if tf_summary:
            compact["timeframe_findings"] = tf_summary

    if isinstance(payload.get("failed_timeframes"), dict) and payload.get(
        "failed_timeframes"
    ):
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


_ALL_COMPACT_CLASSIC_KEYS = (
    "timeframe", "name", "status", "confidence", "bias",
    "reference_price", "target_price", "invalidation_price",
    "bars_to_completion", "start_date", "end_date",
)
_ALL_COMPACT_ELLIOTT_KEYS = (
    "timeframe", "wave_type", "status", "confidence",
    "start_date", "end_date",
)
_ALL_COMPACT_FRACTAL_KEYS = (
    "timeframe", "name", "status", "confidence", "direction", "bias",
    "level_price", "reference_price", "level_state",
    "confirmation_date", "breakout_direction", "breakout_date", "breakout_price",
)

_HIGHLIGHT_KEYS = (
    "section", "timeframe", "name", "direction", "status",
    "confidence", "time", "bar_index", "price", "target_price", "invalidation_price",
)

# Section weight multipliers for highlight ranking
_SECTION_WEIGHT = {"classic": 1.0, "elliott": 1.0, "fractal": 0.75, "candlestick": 0.5}

# Weights for relevance = w_conf * confidence + w_rec * recency
_W_CONFIDENCE = 0.6
_W_RECENCY = 0.4


def _bar_age_recency(row: Dict[str, Any], limit: int) -> float:
    """Compute a 0‑1 recency score from bar‑age.

    ``end_index`` close to the data length means the pattern is recent.
    Uses an exponential decay so the score drops quickly for older patterns.
    When available, ``_data_length`` (set per-timeframe) is preferred over
    the global *limit* so that patterns from shorter TF windows are scored
    relative to their own data.
    """
    if "end_index" not in row:
        return 0.0
    try:
        end_idx = int(row["end_index"])
    except (TypeError, ValueError):
        return 0.0
    effective_limit = int(row.get("_data_length", limit))
    if effective_limit <= 0:
        return 0.0
    bars_ago = max(effective_limit - 1 - end_idx, 0)
    half_life = max(effective_limit * 0.20, 1.0)
    import math
    return math.exp(-0.693 * bars_ago / half_life)


def _compute_relevance(row: Dict[str, Any], limit: int) -> float:
    conf = 0.0
    try:
        conf = float(row.get("confidence", 0))
    except (TypeError, ValueError):
        pass
    rec = _bar_age_recency(row, limit)
    return round(_W_CONFIDENCE * conf + _W_RECENCY * rec, 4)


def score_all_mode_patterns(
    patterns: List[Dict[str, Any]],
    limit: int,
) -> None:
    """Attach ``relevance`` and ``recency`` scores in-place, sort descending."""
    for row in patterns:
        rec = _bar_age_recency(row, limit)
        row["recency"] = round(rec, 4)
        row["relevance"] = _compute_relevance(row, limit)
    patterns.sort(key=lambda r: r.get("relevance", 0), reverse=True)


# ── Candlestick per-timeframe summary ───────────────────────────────────

def _summarize_candlestick_by_tf(
    patterns: List[Dict[str, Any]],
    top_n: int = 3,
) -> Dict[str, Any]:
    """Aggregate candlestick patterns into a per-timeframe summary.

    Returns ``{"by_timeframe": {TF: {bullish, bearish, net, top: [...]}}, ...}``.
    """
    from collections import defaultdict

    by_tf: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for p in patterns:
        tf = p.get("timeframe", "?")
        by_tf[tf].append(p)

    summary: Dict[str, Any] = {}
    total_bullish = 0
    total_bearish = 0

    for tf, rows in by_tf.items():
        bullish = sum(1 for r in rows if str(r.get("direction", "")).lower() == "bullish")
        bearish = sum(1 for r in rows if str(r.get("direction", "")).lower() == "bearish")
        total_bullish += bullish
        total_bearish += bearish

        if bullish > bearish:
            net = "bullish"
        elif bearish > bullish:
            net = "bearish"
        else:
            net = "mixed"

        # Top N most recent (already sorted by relevance)
        top = []
        for r in rows[:top_n]:
            item: Dict[str, Any] = {}
            for k in ("pattern", "direction", "confidence", "time", "price"):
                v = r.get(k)
                if v not in (None, ""):
                    item[k] = v
            if item:
                top.append(item)

        tf_entry: Dict[str, Any] = {
            "bullish": bullish,
            "bearish": bearish,
            "net": net,
        }
        if top:
            tf_entry["top"] = top
        summary[tf] = tf_entry

    result: Dict[str, Any] = {
        "bullish_total": total_bullish,
        "bearish_total": total_bearish,
        "by_timeframe": summary,
    }
    # Expose n_patterns for internal use (tests, full detail) but the compact
    # formatter can choose to omit it.
    result["n_patterns"] = len(patterns)
    return result


# ── Highlights builder ──────────────────────────────────────────────────

def _build_highlights(
    payload: Dict[str, Any],
    limit: int = 5,
) -> List[Dict[str, Any]]:
    """Build a merged top-N list across all sections, ranked by weighted relevance.

    Applies section weights (classic/elliott > candlestick) and enforces
    diversity (max 2 entries from the same section+timeframe).
    """
    candidates: List[Dict[str, Any]] = []

    for row in payload.get("candlestick", {}).get("patterns", []):
        candidates.append({
            "section": "candlestick",
            "timeframe": row.get("timeframe"),
            "name": row.get("pattern"),
            "direction": row.get("direction"),
            "status": "trigger",
            "confidence": row.get("confidence"),
            "time": row.get("time"),
            "bar_index": row.get("bar_index", row.get("end_index")),
            "price": row.get("price"),
            "_relevance": (row.get("relevance", 0) or 0) * _SECTION_WEIGHT["candlestick"],
        })

    for row in payload.get("classic", {}).get("patterns", []):
        candidates.append({
            "section": "classic",
            "timeframe": row.get("timeframe"),
            "name": row.get("name"),
            "direction": row.get("bias"),
            "status": row.get("status"),
            "confidence": row.get("confidence"),
            "time": row.get("end_date", row.get("start_date")),
            "bar_index": row.get("end_index", row.get("start_index")),
            "price": row.get("reference_price"),
            "target_price": row.get("target_price"),
            "invalidation_price": row.get("invalidation_price"),
            "_relevance": (row.get("relevance", 0) or 0) * _SECTION_WEIGHT["classic"],
        })

    for row in payload.get("elliott", {}).get("patterns", []):
        candidates.append({
            "section": "elliott",
            "timeframe": row.get("timeframe"),
            "name": row.get("wave_type"),
            "direction": None,
            "status": row.get("status"),
            "confidence": row.get("confidence"),
            "time": row.get("end_date", row.get("start_date")),
            "bar_index": row.get("end_index", row.get("start_index")),
            "price": None,
            "_relevance": (row.get("relevance", 0) or 0) * _SECTION_WEIGHT["elliott"],
        })

    for row in payload.get("fractal", {}).get("patterns", []):
        candidates.append({
            "section": "fractal",
            "timeframe": row.get("timeframe"),
            "name": row.get("name"),
            "direction": row.get("bias") or row.get("direction"),
            "status": row.get("level_state") or row.get("status"),
            "confidence": row.get("confidence"),
            "time": row.get("breakout_date", row.get("confirmation_date", row.get("time"))),
            "bar_index": row.get("breakout_index", row.get("confirmation_index", row.get("end_index"))),
            "price": row.get("level_price", row.get("price")),
            "_relevance": (row.get("relevance", 0) or 0) * _SECTION_WEIGHT["fractal"],
        })

    candidates.sort(key=lambda r: r.get("_relevance", 0), reverse=True)

    # Diversity: max 2 per (section, timeframe) combo
    from collections import defaultdict
    seen: Dict[str, int] = defaultdict(int)
    highlights: List[Dict[str, Any]] = []
    for c in candidates:
        if len(highlights) >= limit:
            break
        key = f"{c.get('section')}:{c.get('timeframe')}"
        if seen[key] >= 2:
            continue
        seen[key] += 1
        item = {k: c[k] for k in _HIGHLIGHT_KEYS if c.get(k) is not None}
        highlights.append(item)
    return highlights


# ── Section trimmer and compact formatter ───────────────────────────────

def _trim_section_rows(
    rows: List[Dict[str, Any]],
    keys: Tuple[str, ...],
    limit: int = 8,
) -> List[Dict[str, Any]]:
    """Keep top rows by relevance (falls back to confidence) and project *keys*."""
    sorted_rows = sorted(
        rows,
        key=lambda r: (r.get("relevance") or _safe_float(r.get("confidence")) or 0.0),
        reverse=True,
    )
    trimmed: List[Dict[str, Any]] = []
    for row in sorted_rows[:limit]:
        item = {k: row[k] for k in keys if row.get(k) not in (None, "")}
        if item:
            trimmed.append(item)
    return trimmed


def _compact_all_mode_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Compact the all-mode response for trader-focused quick analysis.
    
    Emphasizes highlights while keeping backward-compatible section structure.
    - Highlights: Top actionable patterns (ranked by relevance)
    - Section details: Trimmed pattern arrays (only top patterns, key fields only)
    - Bias: Keep original signal_bias format for compatibility
    
    Pattern arrays are trimmed to top 3 patterns per section to reduce output size.
    """
    if not isinstance(payload, dict) or payload.get("error"):
        return payload

    compact: Dict[str, Any] = {
        "success": True,
        "symbol": payload.get("symbol"),
        "mode": "all",
        "timeframes": payload.get("timeframes"),
    }

    # Top-level highlights — the trader's quick-read (most important)
    highlights = payload.get("highlights")
    if highlights:
        compact["highlights"] = highlights

    # Candlestick: per-TF summary instead of raw pattern list
    candle_section = payload.get("candlestick", {})
    candle_patterns = candle_section.get("patterns", [])
    if candle_patterns:
        candle_summary = _summarize_candlestick_by_tf(candle_patterns)
        candle_summary.pop("n_patterns", None)
        compact["candlestick"] = candle_summary
    else:
        compact["candlestick"] = {"by_timeframe": {}}

    # Classic + Elliott + Fractal: trimmed pattern lists (top 3 per section, key fields only)
    for section_name, keys in (
        ("classic", _ALL_COMPACT_CLASSIC_KEYS),
        ("elliott", _ALL_COMPACT_ELLIOTT_KEYS),
        ("fractal", _ALL_COMPACT_FRACTAL_KEYS),
    ):
        section = payload.get(section_name, {})
        rows = section.get("patterns", [])
        # Trim to top 3 patterns by relevance/confidence
        trimmed = _trim_section_rows(rows, keys, limit=3) if rows else []
        result: Dict[str, Any] = {
            "patterns": trimmed,
        }
        bias = section.get("signal_bias")
        if bias:
            result["signal_bias"] = bias
        for key in ("active_levels", "latest_breakouts"):
            value = section.get(key)
            if value not in (None, "", [], {}):
                result[key] = value
        compact[section_name] = result

    errors = payload.get("errors")
    if errors:
        compact["errors"] = errors

    return compact


def _highlights_all_mode_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Return only the all-mode quick-read pattern summary."""
    if not isinstance(payload, dict) or payload.get("error"):
        return payload

    compact: Dict[str, Any] = {
        "success": True,
        "symbol": payload.get("symbol"),
        "mode": "all",
        "timeframes": payload.get("timeframes"),
        "total_patterns": int(payload.get("total_patterns") or 0),
    }

    highlights = payload.get("highlights")
    if highlights:
        compact["highlights"] = highlights

    section_counts: Dict[str, int] = {}
    signal_inputs: List[Dict[str, Any]] = []
    for section_name in ("candlestick", "classic", "elliott", "fractal"):
        section = payload.get(section_name)
        if not isinstance(section, dict):
            continue
        rows = section.get("patterns")
        if isinstance(rows, list):
            section_counts[section_name] = len(rows)
            signal_inputs.extend(row for row in rows if isinstance(row, dict))

    if section_counts:
        compact["section_counts"] = section_counts
    signal_bias = _summarize_pattern_bias(signal_inputs)
    if signal_bias:
        compact["signal_bias"] = signal_bias

    errors = payload.get("errors")
    if errors:
        compact["errors"] = errors

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


def _config_value(config: Any, key: str) -> tuple[bool, Any]:
    if isinstance(config, dict):
        if key in config:
            return True, config.get(key)
        return False, None
    try:
        return True, getattr(config, key)
    except Exception:
        return False, None


def _config_bool(config: Any, key: str, default: bool) -> bool:
    found, value = _config_value(config, key)
    if not found:
        return bool(default)
    if value is None:
        return bool(default)
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
    return bool(default)


def _config_int(config: Any, key: str, default: int, *, minimum: int = 0) -> int:
    found, value_raw = _config_value(config, key)
    if not found:
        value = int(default)
    else:
        try:
            value = int(value_raw)
        except Exception:
            value = int(default)
    return max(int(minimum), int(value))


def _config_float(
    config: Any, key: str, default: float, *, minimum: float = 0.0
) -> float:
    found, value_raw = _config_value(config, key)
    if not found:
        value = float(default)
    else:
        try:
            value = float(value_raw)
        except Exception:
            value = float(default)
    if not np.isfinite(value):
        value = float(default)
    return float(max(float(minimum), value))


def _resolve_volume_series(
    df: pd.DataFrame,
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    if not isinstance(df, pd.DataFrame) or len(df) <= 0:
        return None, None

    if "real_volume" in df.columns:
        try:
            real_volume = pd.to_numeric(df["real_volume"], errors="coerce").to_numpy(
                dtype=float, copy=False
            )
        except Exception:
            real_volume = np.asarray([], dtype=float)
        finite_real = real_volume[np.isfinite(real_volume)]
        if finite_real.size > 0 and np.nanmax(finite_real) > 0:
            return real_volume, "real_volume"

    for col in ("volume", "tick_volume", "Volume"):
        if col not in df.columns:
            continue
        try:
            volume = pd.to_numeric(df[col], errors="coerce").to_numpy(
                dtype=float, copy=False
            )
        except Exception:
            continue
        if volume.size <= 0:
            continue
        if np.isfinite(volume).any():
            return volume, str(col)
    return None, None


def _volume_window_mean(
    volume: Optional[np.ndarray], start_index: Any, end_index: Any
) -> Optional[float]:
    if volume is None or len(volume) <= 0:
        return None
    try:
        start_i = int(start_index)
        end_i = int(end_index)
    except Exception:
        return None
    start_i = max(0, start_i)
    end_i = min(int(len(volume) - 1), end_i)
    if end_i < start_i:
        return None
    window = np.asarray(volume[start_i : end_i + 1], dtype=float)
    window = window[np.isfinite(window)]
    window = window[window >= 0]
    if window.size <= 0:
        return None
    return float(np.mean(window))


def _apply_confidence_delta(row: Dict[str, Any], delta: float) -> None:
    if not np.isfinite(delta) or abs(float(delta)) <= 1e-12:
        return
    conf = _row_confidence_weight(row)
    cap = 0.95 if str(row.get("status", "")).lower() == "forming" else 1.0
    row["confidence"] = float(max(0.0, min(cap, conf + float(delta))))


def _infer_market_regime(df: pd.DataFrame, config: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(df, pd.DataFrame) or len(df) <= 0:
        return None

    try:
        close = pd.to_numeric(df.get("close"), errors="coerce").to_numpy(
            dtype=float, copy=False
        )
    except Exception:
        return None
    if close.size < 20:
        return None

    close = close[np.isfinite(close)]
    if close.size < 20:
        return None

    window_bars = min(
        int(close.size),
        _config_int(config, "regime_window_bars", 160, minimum=20),
    )
    segment = np.asarray(close[-window_bars:], dtype=float)
    if segment.size < 20:
        return None

    diffs = np.diff(segment)
    finite_diffs = diffs[np.isfinite(diffs)]
    path_length = float(np.sum(np.abs(finite_diffs))) if finite_diffs.size else 0.0
    move = float(segment[-1] - segment[0])
    base_price = float(segment[0]) if abs(float(segment[0])) > 1e-9 else 1e-9
    trend_strength = float(abs(move) / max(float(np.nanstd(segment)), 1e-9))
    efficiency_ratio = float(abs(move) / max(path_length, 1e-9))
    trend_threshold = _config_float(
        config, "regime_trend_strength_threshold", 1.25, minimum=0.1
    )
    efficiency_threshold = _config_float(
        config,
        "regime_efficiency_trending_threshold",
        0.35,
        minimum=0.05,
    )

    if efficiency_ratio >= efficiency_threshold and trend_strength >= trend_threshold:
        state = "trending"
    elif efficiency_ratio <= max(0.1, 0.55 * efficiency_threshold):
        state = "ranging"
    else:
        state = "transition"

    direction = "neutral"
    if move > 1e-9:
        direction = "bullish"
    elif move < -1e-9:
        direction = "bearish"

    return {
        "state": state,
        "direction": direction,
        "window_bars": int(window_bars),
        "trend_strength": _round_value(trend_strength),
        "efficiency_ratio": _round_value(efficiency_ratio),
        "window_move_pct": _round_value((move / base_price) * 100.0),
    }


def _attach_regime_context(
    row: Dict[str, Any],
    regime_context: Optional[Dict[str, Any]],
    config: Any,
) -> Dict[str, Any]:
    out = dict(row)
    details = out.get("details")
    if not isinstance(details, dict):
        details = {}
    else:
        details = dict(details)

    payload: Dict[str, Any] = {
        "status": "disabled"
        if not _config_bool(config, "use_regime_context", True)
        else "unavailable",
    }
    if payload["status"] == "disabled":
        details["regime_context"] = payload
        out["details"] = details
        return out
    if not isinstance(regime_context, dict):
        details["regime_context"] = payload
        out["details"] = details
        return out

    payload.update(regime_context)
    bias = _row_pattern_bias(out)
    if bias:
        payload["pattern_bias"] = bias

    bonus = _config_float(config, "regime_alignment_bonus", 0.05, minimum=0.0)
    penalty = _config_float(config, "regime_countertrend_penalty", 0.05, minimum=0.0)
    confidence_delta = 0.0

    if bias in {"bullish", "bearish"}:
        if payload.get("state") == "trending" and payload.get("direction") in {
            "bullish",
            "bearish",
        }:
            if bias == payload.get("direction"):
                payload["status"] = "aligned"
                payload["alignment"] = "aligned"
                confidence_delta = float(bonus)
            else:
                payload["status"] = "countertrend"
                payload["alignment"] = "countertrend"
                confidence_delta = -float(penalty)
        else:
            payload["status"] = "context_only"
            payload["alignment"] = "neutral"
    elif bias == "neutral":
        if payload.get("state") == "ranging":
            payload["status"] = "range_aligned"
            payload["alignment"] = "range_aligned"
            confidence_delta = float(0.5 * bonus)
        else:
            payload["status"] = "context_only"
            payload["alignment"] = "neutral"
    else:
        payload["status"] = "not_directional"

    if abs(confidence_delta) > 1e-12:
        payload["confidence_delta"] = _round_value(confidence_delta)
        _apply_confidence_delta(out, confidence_delta)

    details["regime_context"] = payload
    out["details"] = details
    return out


def _attach_classic_volume_confirmation(
    row: Dict[str, Any],
    df: pd.DataFrame,
    config: Any,
) -> Dict[str, Any]:
    out = dict(row)
    details = out.get("details")
    if not isinstance(details, dict):
        details = {}
    else:
        details = dict(details)

    payload: Dict[str, Any] = {
        "mode": "breakout",
        "status": "disabled"
        if not _config_bool(config, "use_volume_confirmation", True)
        else "unavailable",
        "volume_source": None,
    }
    if payload["status"] == "disabled":
        details["volume_confirmation"] = payload
        out["details"] = details
        return out

    volume, source = _resolve_volume_series(df)
    payload["volume_source"] = source
    if volume is None or source is None:
        details["volume_confirmation"] = payload
        out["details"] = details
        return out

    breakout_bars = _config_int(config, "volume_confirm_breakout_bars", 2, minimum=1)
    lookback_bars = _config_int(
        config, "volume_confirm_lookback_bars", 20, minimum=breakout_bars + 1
    )
    min_ratio = _config_float(config, "volume_confirm_min_ratio", 1.10, minimum=1.0)
    bonus = _config_float(config, "volume_confirm_bonus", 0.08, minimum=0.0)
    penalty = _config_float(config, "volume_confirm_penalty", 0.06, minimum=0.0)

    last_index = max(int(len(volume) - 1), 0)
    raw_end_index = _safe_float(out.get("end_index"))
    end_index = max(
        0,
        min(
            last_index, int(raw_end_index) if raw_end_index is not None else last_index
        ),
    )
    signal_start = max(0, int(end_index - breakout_bars + 1))
    baseline_end = int(signal_start - 1)
    baseline_start = max(0, int(baseline_end - lookback_bars + 1))
    signal_avg = _volume_window_mean(volume, signal_start, end_index)
    baseline_avg = _volume_window_mean(volume, baseline_start, baseline_end)
    ratio = (
        float(signal_avg) / float(baseline_avg)
        if signal_avg is not None and baseline_avg is not None and baseline_avg > 0
        else None
    )

    payload["lookback_bars"] = int(lookback_bars)
    payload["breakout_bars"] = int(breakout_bars)
    if baseline_avg is not None:
        payload["baseline_avg_volume"] = _round_value(baseline_avg)
    if signal_avg is not None:
        payload["breakout_avg_volume"] = _round_value(signal_avg)
    if ratio is not None and np.isfinite(ratio):
        payload["breakout_to_baseline_ratio"] = _round_value(ratio)

    status = str(out.get("status", "")).strip().lower()
    confidence_delta = 0.0
    if ratio is None:
        payload["status"] = "unavailable"
    elif status == "completed":
        reject_ratio = (1.0 / float(min_ratio)) if min_ratio > 0 else 0.0
        if ratio >= float(min_ratio):
            payload["status"] = "confirmed"
            confidence_delta = float(bonus)
        elif ratio <= float(reject_ratio):
            payload["status"] = "rejected"
            confidence_delta = -float(penalty)
        else:
            payload["status"] = "neutral"
    else:
        payload["status"] = "pending"

    if abs(confidence_delta) > 1e-12:
        payload["confidence_delta"] = _round_value(confidence_delta)
        _apply_confidence_delta(out, confidence_delta)

    details["volume_confirmation"] = payload
    out["details"] = details
    return out


def _elliott_wave_indices(details: Dict[str, Any]) -> List[int]:
    points = details.get("wave_points_labeled")
    if not isinstance(points, list):
        return []
    out: List[int] = []
    for item in points:
        if not isinstance(item, dict):
            continue
        idx = _safe_float(item.get("index"))
        if idx is None:
            continue
        out.append(int(idx))
    return out


def _attach_elliott_volume_confirmation(
    row: Dict[str, Any],
    df: pd.DataFrame,
    config: Any,
) -> Dict[str, Any]:
    out = dict(row)
    details = out.get("details")
    if not isinstance(details, dict):
        details = {}
    else:
        details = dict(details)

    payload: Dict[str, Any] = {
        "mode": "wave_segments",
        "status": "disabled"
        if not _config_bool(config, "use_volume_confirmation", True)
        else "unavailable",
        "volume_source": None,
    }
    if payload["status"] == "disabled":
        details["volume_confirmation"] = payload
        out["details"] = details
        return out

    wave_type = (
        str(out.get("wave_type") or details.get("pattern_family") or "").strip().lower()
    )
    if wave_type == "candidate":
        payload["status"] = "candidate"
        details["volume_confirmation"] = payload
        out["details"] = details
        return out

    family = str(details.get("pattern_family") or wave_type).strip().lower()
    if family not in {"impulse", "correction"}:
        details["volume_confirmation"] = payload
        out["details"] = details
        return out

    volume, source = _resolve_volume_series(df)
    payload["volume_source"] = source
    if volume is None or source is None:
        details["volume_confirmation"] = payload
        out["details"] = details
        return out

    pivots = _elliott_wave_indices(details)
    if len(pivots) < 4:
        details["volume_confirmation"] = payload
        out["details"] = details
        return out

    segment_averages: Dict[int, float] = {}
    for segment_index in range(len(pivots) - 1):
        seg_avg = _volume_window_mean(
            volume, pivots[segment_index], pivots[segment_index + 1]
        )
        if seg_avg is not None:
            segment_averages[segment_index] = float(seg_avg)

    trend_slots = [0, 2, 4] if family == "impulse" else [0, 2]
    counter_slots = [1, 3] if family == "impulse" else [1]
    trend_values = [segment_averages[i] for i in trend_slots if i in segment_averages]
    counter_values = [
        segment_averages[i] for i in counter_slots if i in segment_averages
    ]
    if not trend_values or not counter_values:
        details["volume_confirmation"] = payload
        out["details"] = details
        return out

    trend_avg = float(np.mean(np.asarray(trend_values, dtype=float)))
    counter_avg = float(np.mean(np.asarray(counter_values, dtype=float)))
    ratio = (trend_avg / counter_avg) if counter_avg > 0 else None

    payload["trend_segment_avg_volume"] = _round_value(trend_avg)
    payload["counter_segment_avg_volume"] = _round_value(counter_avg)
    payload["trend_segments_used"] = int(len(trend_values))
    payload["counter_segments_used"] = int(len(counter_values))
    if ratio is not None and np.isfinite(ratio):
        payload["trend_to_counter_ratio"] = _round_value(ratio)

    min_ratio = _config_float(config, "volume_confirm_min_ratio", 1.05, minimum=1.0)
    bonus = _config_float(config, "volume_confirm_bonus", 0.06, minimum=0.0)
    penalty = _config_float(config, "volume_confirm_penalty", 0.04, minimum=0.0)
    confidence_delta = 0.0
    if ratio is None:
        payload["status"] = "unavailable"
    else:
        reject_ratio = (1.0 / float(min_ratio)) if min_ratio > 0 else 0.0
        if ratio >= float(min_ratio):
            payload["status"] = "confirmed"
            confidence_delta = float(bonus)
        elif ratio <= float(reject_ratio):
            payload["status"] = "rejected"
            confidence_delta = -float(penalty)
        else:
            payload["status"] = "neutral"

    if abs(confidence_delta) > 1e-12:
        payload["confidence_delta"] = _round_value(confidence_delta)
        _apply_confidence_delta(out, confidence_delta)

    details["volume_confirmation"] = payload
    out["details"] = details
    return out


_CLASSIC_BIAS_KEYWORDS = (
    ("inverse head and shoulders", "bullish"),
    ("head and shoulders", "bearish"),
    ("falling wedge", "bullish"),
    ("rising wedge", "bearish"),
    ("double bottom", "bullish"),
    ("triple bottom", "bullish"),
    ("double top", "bearish"),
    ("triple top", "bearish"),
    ("rounding bottom", "bullish"),
    ("rounding top", "bearish"),
    ("cup and handle", "bullish"),
    ("ascending triangle", "bullish"),
    ("descending triangle", "bearish"),
    ("bull", "bullish"),
    ("ascending", "bullish"),
    ("uptrend", "bullish"),
    ("bear", "bearish"),
    ("descending", "bearish"),
    ("downtrend", "bearish"),
)


def _infer_classic_bias(name: Any, details: Dict[str, Any]) -> str:
    for key in ("bias", "pattern_bias", "breakout_direction", "breakout_expected"):
        bias = _normalize_pattern_bias(details.get(key))
        if bias in ("bullish", "bearish"):
            return bias

    name_text = str(name or "").strip().lower()
    if not name_text:
        return "neutral"
    for keyword, bias in _CLASSIC_BIAS_KEYWORDS:
        if keyword in name_text:
            return bias
    return "neutral"


def _classic_price_levels(
    details: Dict[str, Any],
    end_index: Any,
    *,
    eval_index: Any = None,
) -> Dict[str, float]:
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
        idx_float = float(end_index if eval_index is None else eval_index)
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
        levels.setdefault(
            "resistance", float(upper_slope * idx_float + upper_intercept)
        )
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
    cup_extreme = _detail_float(details, "cup_extreme")
    if left_rim is not None and cup_extreme is not None:
        return float(abs(left_rim - cup_extreme))

    pole_return_pct = _detail_float(details, "pole_return_pct")
    if pole_return_pct is not None and reference_price is not None:
        return float(abs(reference_price * pole_return_pct / 100.0))

    amplitude_pct = _detail_float(details, "amplitude_pct")
    if amplitude_pct is not None and reference_price is not None:
        return float(abs(reference_price * amplitude_pct / 100.0))
    return None


def _enrich_classic_pattern_row(  # noqa: C901
    row: Dict[str, Any],
    df: pd.DataFrame,
    config: Any = None,
    regime_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    out = dict(row)
    details = out.get("details")
    if not isinstance(details, dict):
        details = {}
    name = out.get("name") or out.get("pattern")
    bias = _infer_classic_bias(name, details)
    reference_price = _close_price_at_index(df, out.get("end_index"))
    status = str(out.get("status", "")).strip().lower()
    current_bar_index = int(max(0, len(df) - 1)) if len(df) > 0 else 0
    try:
        end_index_i = int(out.get("end_index"))
    except Exception:
        end_index_i = current_bar_index
    reference_age_bars = max(0, current_bar_index - end_index_i)
    level_eval_index = out.get("end_index")
    if status == "forming" and len(df) > 0:
        level_eval_index = current_bar_index
    levels = _classic_price_levels(
        details, out.get("end_index"), eval_index=level_eval_index
    )
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
            elif (
                _detail_float(details, "breakout_level") is not None
                and float(_detail_float(details, "breakout_level")) < reference_price
            ):
                invalidation = float(_detail_float(details, "breakout_level"))
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
            elif (
                _detail_float(details, "breakout_level") is not None
                and float(_detail_float(details, "breakout_level")) > reference_price
            ):
                invalidation = float(_detail_float(details, "breakout_level"))
            elif height is not None and height > 0:
                invalidation = reference_price + 0.5 * height

    out["bias"] = bias
    if reference_price is not None:
        out["reference_price"] = _round_value(reference_price)
    if target is not None:
        out["target_price"] = _round_value(target)
        out["target_reference_age_bars"] = int(reference_age_bars)
        if status == "completed" and reference_age_bars > 0:
            out["target_stale"] = True
    if invalidation is not None:
        out["invalidation_price"] = _round_value(invalidation)
    if levels:
        out["price_levels"] = {
            key: _round_value(value) for key, value in levels.items()
        }
    if status == "forming" and reference_price is not None and len(df) > 1:
        structural_est = _estimate_classic_bars_to_completion(
            str(name or ""),
            details,
            int(out.get("start_index", 0)),
            int(
                out.get(
                    "end_index", level_eval_index if level_eval_index is not None else 0
                )
            ),
            len(df),
        )
        price_est: Optional[int] = None
        try:
            close_arr = pd.to_numeric(df.get("close"), errors="coerce").to_numpy(
                dtype=float, copy=False
            )
            recent_steps = np.abs(np.diff(close_arr[-min(len(close_arr), 20) :]))
            finite_steps = recent_steps[np.isfinite(recent_steps)]
            step_size = float(np.median(finite_steps)) if finite_steps.size else 0.0
            if step_size > 1e-9:
                level_targets: List[float] = []
                breakout_level = _detail_float(details, "breakout_level")
                if bias == "bullish":
                    if resistance is not None and resistance > reference_price:
                        level_targets.append(float(resistance))
                    if breakout_level is not None and breakout_level > reference_price:
                        level_targets.append(float(breakout_level))
                elif bias == "bearish":
                    if support is not None and support < reference_price:
                        level_targets.append(float(support))
                    if breakout_level is not None and breakout_level < reference_price:
                        level_targets.append(float(breakout_level))
                else:
                    for candidate in (support, resistance, breakout_level):
                        if candidate is not None:
                            level_targets.append(float(candidate))
                distances = [
                    abs(float(level) - reference_price)
                    for level in level_targets
                    if np.isfinite(level)
                ]
                if distances:
                    price_est = int(max(0, int(np.ceil(min(distances) / step_size))))
        except Exception:
            price_est = None
        if structural_est is not None and price_est is not None:
            out["bars_to_completion"] = int(max(structural_est, price_est))
            out["bars_to_completion_basis"] = "structure_and_price"
        elif structural_est is not None:
            out["bars_to_completion"] = int(structural_est)
            out["bars_to_completion_basis"] = "structure"
        elif price_est is not None:
            out["bars_to_completion"] = int(price_est)
            out["bars_to_completion_basis"] = "price_proximity"
    out = _attach_classic_volume_confirmation(out, df, config)
    return _attach_regime_context(out, regime_context, config)


def _enrich_classic_patterns(
    rows: List[Dict[str, Any]], df: pd.DataFrame, config: Any = None
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    regime_context = _infer_market_regime(df, config)
    for row in rows:
        if not isinstance(row, dict):
            continue
        out.append(_enrich_classic_pattern_row(row, df, config, regime_context))
    return out


def _enrich_elliott_patterns(
    rows: List[Dict[str, Any]], df: pd.DataFrame, config: Any = None
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    regime_context = _infer_market_regime(df, config)
    for row in rows:
        if not isinstance(row, dict):
            continue
        enriched = _attach_elliott_volume_confirmation(row, df, config)
        out.append(_attach_regime_context(enriched, regime_context, config))
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
        n_forming = _count_patterns_with_status(rows, "forming")
        n_completed = _count_patterns_with_status(rows, "completed")
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
        if all(
            details.get(key) is not None
            for key in (
                "top_slope",
                "top_intercept",
                "bottom_slope",
                "bottom_intercept",
            )
        ):
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
        if all(
            details.get(key) is not None
            for key in (
                "upper_slope",
                "upper_intercept",
                "lower_slope",
                "lower_intercept",
            )
        ):
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
        if "pennant" in name_text or "flag" in name_text:
            return int(max(1, int(round(0.3 * length))))
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


def _load_stock_pattern_utils(
    config: Optional[Dict[str, Any]],
) -> Tuple[Optional[Any], Optional[str]]:
    _ = config
    # Try known module paths for stock_pattern (underscore and no-separator variants)
    candidate_modules = (
        "stock_pattern.utils",
        "stockpattern.utils",
    )
    required = ("get_max_min", "find_double_top", "find_double_bottom")
    last_err: Optional[str] = None
    for module_name in candidate_modules:
        module = _STOCK_PATTERN_UTILS_CACHE.get(module_name)
        if module is not None:
            return module, None

        with _STOCK_PATTERN_UTILS_CACHE_LOCK:
            module = _STOCK_PATTERN_UTILS_CACHE.get(module_name)
            if module is not None:
                return module, None
            try:
                module = importlib.import_module(module_name)
            except Exception as ex:
                last_err = str(ex)
                continue
            if not all(callable(getattr(module, fn, None)) for fn in required):
                last_err = (
                    f"module '{module_name}' missing required stock-pattern functions"
                )
                continue
            _STOCK_PATTERN_UTILS_CACHE[module_name] = module
            return module, None

    tail = f" Last import error: {last_err}" if last_err else ""
    return None, (
        "stock-pattern engine unavailable in current environment; "
        "install stock-pattern from https://github.com/BennyThadikaran/stock-pattern"
        + tail
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
    volume_col = (
        "volume"
        if "volume" in src.columns
        else ("tick_volume" if "tick_volume" in src.columns else "Volume")
    )

    out = pd.DataFrame(
        {
            "Open": pd.to_numeric(src.get(open_col), errors="coerce"),
            "High": pd.to_numeric(src.get(high_col), errors="coerce"),
            "Low": pd.to_numeric(src.get(low_col), errors="coerce"),
            "Close": pd.to_numeric(src.get(close_col), errors="coerce"),
            "Volume": pd.to_numeric(src.get(volume_col), errors="coerce"),
        }
    )

    # Build a UTC-normalized DatetimeIndex from the 'time' column when available;
    # fall back to an existing DatetimeIndex (after stripping timezone) or a RangeIndex.
    if "time" in src.columns:
        try:
            idx = pd.to_datetime(
                pd.to_numeric(src["time"], errors="coerce"), unit="s", utc=True
            )
            out.index = pd.DatetimeIndex(idx).tz_localize(None)
        except Exception:
            out.index = pd.RangeIndex(start=0, stop=len(out), step=1)
    elif isinstance(src.index, pd.DatetimeIndex):
        out.index = src.index.tz_localize(None) if src.index.tz is not None else src.index
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
        parts = [
            part.strip() for part in value.replace(";", ",").split(",") if part.strip()
        ]
        return [_normalize_engine_name(part) for part in parts]
    if isinstance(value, (list, tuple, set)):
        return [_normalize_engine_name(part) for part in value if str(part).strip()]
    return [_normalize_engine_name(value)]


def _parse_native_scale_factors(config: Optional[Dict[str, Any]]) -> List[float]:
    cfg_map = config if isinstance(config, dict) else {}
    raw = cfg_map.get("native_scale_factors", cfg_map.get("native_scales"))
    vals: List[float] = []
    if isinstance(raw, str):
        parts = [
            part.strip() for part in raw.replace(";", ",").split(",") if part.strip()
        ]
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
            if prev is None or float(pattern.get("confidence", 0.0)) > float(
                prev.get("confidence", 0.0)
            ):
                by_engine[engine] = pattern
        engines = list(by_engine.keys())
        total_w = float(sum(weights.get(engine, 1.0) for engine in engines)) or 1.0
        conf = float(
            sum(
                float(by_engine[engine].get("confidence", 0.0))
                * float(weights.get(engine, 1.0))
                for engine in engines
            )
            / total_w
        )
        anchor_engine = max(
            engines, key=lambda engine: float(by_engine[engine].get("confidence", 0.0))
        )
        anchor = dict(by_engine[anchor_engine])
        statuses = [
            str(by_engine[engine].get("status", "forming")).lower()
            for engine in engines
        ]
        anchor["status"] = (
            "completed"
            if any(status == "completed" for status in statuses)
            else "forming"
        )
        anchor["confidence"] = float(max(0.0, min(1.0, conf)))
        anchor["support_count"] = int(len(engines))
        anchor["source_engines"] = engines
        details = anchor.get("details")
        if not isinstance(details, dict):
            details = {}
        details = dict(details)
        details["engine_confidences"] = {
            engine: float(
                max(0.0, min(1.0, float(by_engine[engine].get("confidence", 0.0))))
            )
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


def _format_pattern_dates(
    start_time: Optional[float], end_time: Optional[float]
) -> Tuple[Optional[str], Optional[str]]:
    """Format epoch times to date strings."""
    start_epoch = float(start_time) if start_time is not None else None
    end_epoch = float(end_time) if end_time is not None else None

    try:
        start_date = (
            _format_time_minimal(start_epoch) if start_epoch is not None else None
        )
    except Exception:
        start_date = None

    try:
        end_date = _format_time_minimal(end_epoch) if end_epoch is not None else None
    except Exception:
        end_date = None

    return start_date, end_date
