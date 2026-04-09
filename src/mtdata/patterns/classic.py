from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..utils.utils import to_float_np
from .classic_impl.config import ClassicDetectorConfig, ClassicPatternResult
from .classic_impl.continuation import detect_cup_handle, detect_flags_pennants
from .classic_impl.reversal import (
    detect_head_shoulders,
    detect_rounding,
    detect_tops_bottoms,
)
from .classic_impl.shapes import (
    detect_broadening,
    detect_diamonds,
    detect_rectangles,
    detect_triangles,
    detect_wedges,
)
from .classic_impl.trend import detect_channels, detect_trend_lines
from .classic_impl.utils import (
    _alias,
    _build_time_array,
    _calibrate_confidence,
    _collect_calibration_points,
    _compute_atr,
    _conf,
    _count_recent_touches,
    _count_touches,
    _detect_pivots_close,
    _dtw_distance,
    _find_forward_level_breakout,
    _find_recent_breakout,
    _fit_line,
    _fit_line_robust,
    _fit_lines_and_arrays,
    _is_converging,
    _last_touch_indexes,
    _level_close,
    _paa,
    _pivot_thresholds,
    _result,
    _template_hs,
    _tol_abs_from_close,
    _znorm,
)

# Re-export for backward compatibility
__all__ = [
    "ClassicDetectorConfig",
    "ClassicPatternResult",
    "detect_classic_patterns",
]

def _prepare_classic_inputs(
    df: pd.DataFrame,
    cfg: ClassicDetectorConfig,
) -> Optional[tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]]:
    if not isinstance(df, pd.DataFrame) or 'close' not in df.columns:
        return None
    if len(df) > cfg.max_bars:
        df = df.iloc[-cfg.max_bars:].copy()

    t = _build_time_array(df)
    c = to_float_np(df['close'])
    h = to_float_np(df['high']) if 'high' in df.columns else c
    l = to_float_np(df['low']) if 'low' in df.columns else c
    if h.size != c.size:
        h = c
    if l.size != c.size:
        l = c
    n = c.size
    min_input_bars = max(20, int(getattr(cfg, "min_input_bars", 100)))
    if n < min_input_bars:
        return None
    return df, t, c, h, l, n


def _detect_classic_patterns_once(
    t: np.ndarray,
    c: np.ndarray,
    h: np.ndarray,
    l: np.ndarray,
    n: int,
    cfg: ClassicDetectorConfig,
    *,
    peaks: Optional[np.ndarray] = None,
    troughs: Optional[np.ndarray] = None,
) -> List[ClassicPatternResult]:
    if peaks is None or troughs is None:
        try:
            peaks, troughs = _detect_pivots_close(c, cfg, h, l)
        except TypeError:
            # Keep compatibility with monkeypatched tests that still use the old 2-arg signature.
            peaks, troughs = _detect_pivots_close(c, cfg)
    else:
        peaks = np.asarray(peaks, dtype=int)
        troughs = np.asarray(troughs, dtype=int)

    results: List[ClassicPatternResult] = []
    results.extend(detect_trend_lines(c, peaks, troughs, t, cfg))
    results.extend(detect_channels(c, peaks, troughs, t, cfg))
    results.extend(detect_rectangles(c, peaks, troughs, t, cfg))
    results.extend(detect_triangles(c, peaks, troughs, t, cfg))
    results.extend(detect_wedges(c, peaks, troughs, t, cfg))
    results.extend(detect_broadening(c, peaks, troughs, t, cfg))
    results.extend(detect_diamonds(c, t, cfg, h, l, peaks=peaks, troughs=troughs))
    results.extend(detect_tops_bottoms(c, peaks, troughs, t, cfg))
    results.extend(detect_head_shoulders(c, peaks, troughs, t, cfg))
    results.extend(detect_rounding(c, t, cfg))
    results.extend(detect_flags_pennants(c, h, l, t, n, cfg, peaks=peaks, troughs=troughs))
    results.extend(detect_cup_handle(c, t, cfg))
    return results


def _pattern_overlap_ratio(a: ClassicPatternResult, b: ClassicPatternResult) -> float:
    lo = max(int(a.start_index), int(b.start_index))
    hi = min(int(a.end_index), int(b.end_index))
    if hi < lo:
        return 0.0
    inter = hi - lo + 1
    union = max(int(a.end_index), int(b.end_index)) - min(int(a.start_index), int(b.start_index)) + 1
    if union <= 0:
        return 0.0
    return float(inter) / float(union)


def _prefer_pattern_candidate(
    current: ClassicPatternResult,
    candidate: ClassicPatternResult,
) -> ClassicPatternResult:
    current_rank = (
        1 if str(current.status).lower() == "completed" else 0,
        int(current.end_index),
        float(current.confidence),
    )
    candidate_rank = (
        1 if str(candidate.status).lower() == "completed" else 0,
        int(candidate.end_index),
        float(candidate.confidence),
    )
    return candidate if candidate_rank > current_rank else current


def _merge_scanned_patterns(
    existing: List[ClassicPatternResult],
    new_results: List[ClassicPatternResult],
    cfg: ClassicDetectorConfig,
) -> List[ClassicPatternResult]:
    overlap_min = float(max(0.0, min(1.0, getattr(cfg, "scan_dedupe_overlap", 0.8))))
    merged = list(existing)
    for candidate in new_results:
        match_i: Optional[int] = None
        for i, prior in enumerate(merged):
            if prior.name != candidate.name:
                continue
            if _pattern_overlap_ratio(prior, candidate) < overlap_min:
                continue
            match_i = i
            break
        if match_i is None:
            merged.append(candidate)
            continue
        merged[match_i] = _prefer_pattern_candidate(merged[match_i], candidate)
    return merged


def _scan_classic_patterns(
    t: np.ndarray,
    c: np.ndarray,
    h: np.ndarray,
    l: np.ndarray,
    cfg: ClassicDetectorConfig,
) -> List[ClassicPatternResult]:
    n_total = int(c.size)
    step = max(1, int(getattr(cfg, "scan_step_bars", 10)))
    min_prefix = max(int(getattr(cfg, "min_input_bars", 100)), int(getattr(cfg, "scan_min_prefix_bars", 120)))
    prefix_ends = list(range(min_prefix, n_total + 1, step))
    if not prefix_ends or prefix_ends[-1] != n_total:
        prefix_ends.append(n_total)

    scan_cfg = replace(cfg, scan_historical=False)
    try:
        full_peaks, full_troughs = _detect_pivots_close(c, scan_cfg, h, l)
    except TypeError:
        full_peaks, full_troughs = _detect_pivots_close(c, scan_cfg)
    full_peaks = np.asarray(full_peaks, dtype=int)
    full_troughs = np.asarray(full_troughs, dtype=int)
    pivot_confirm_gap = max(2, int(getattr(scan_cfg, "min_distance", 5)))

    merged: List[ClassicPatternResult] = []
    for end in prefix_ends:
        pivot_cutoff = max(0, int(end) - pivot_confirm_gap)
        batch = _detect_classic_patterns_once(
            t[:end],
            c[:end],
            h[:end],
            l[:end],
            int(end),
            scan_cfg,
            peaks=full_peaks[full_peaks < pivot_cutoff],
            troughs=full_troughs[full_troughs < pivot_cutoff],
        )
        merged = _merge_scanned_patterns(merged, batch, cfg)
    return merged


def _postprocess_classic_results(
    results: List[ClassicPatternResult],
    cfg: ClassicDetectorConfig,
    n: int,
) -> List[ClassicPatternResult]:
    if bool(cfg.auto_complete_stale_forming):
        recent_bars = max(1, int(getattr(cfg, "stale_completion_recent_bars", 3)))
        for i, r in enumerate(results):
            if r.status == 'forming' and r.end_index < (n - recent_bars):
                results[i] = replace(r, status='completed')

    for r in results:
        raw_conf = float(r.confidence)
        cal_conf = _calibrate_confidence(raw_conf, r.name, cfg)
        if not isinstance(r.details, dict):
            r.details = {}
        if bool(getattr(cfg, "calibrate_confidence", False)):
            r.details["raw_confidence"] = float(raw_conf)
            r.details["calibrated_confidence"] = float(cal_conf)
        r.confidence = float(cal_conf)

    min_conf = float(max(0.0, min(1.0, getattr(cfg, "min_confidence", 0.0))))
    if min_conf > 0.0:
        results = [r for r in results if float(r.confidence) >= min_conf]

    if bool(getattr(cfg, "include_lifecycle_metadata", True)):
        for r in results:
            if not isinstance(r.details, dict):
                r.details = {}
            if r.status == "completed":
                r.details.setdefault("lifecycle_state", "confirmed")
            else:
                r.details.setdefault("lifecycle_state", "forming")

    results.sort(key=lambda r: (r.end_index, r.confidence), reverse=True)
    return results


def detect_classic_patterns(df: pd.DataFrame, cfg: Optional[ClassicDetectorConfig] = None) -> List[ClassicPatternResult]:
    """Detect classic chart patterns on OHLCV DataFrame with 'time' and 'close' columns."""
    if cfg is None:
        cfg = ClassicDetectorConfig()
    prepared = _prepare_classic_inputs(df, cfg)
    if prepared is None:
        return []

    _, t, c, h, l, n = prepared
    if bool(getattr(cfg, "scan_historical", False)):
        results = _scan_classic_patterns(t, c, h, l, cfg)
    else:
        results = _detect_classic_patterns_once(t, c, h, l, n, cfg)

    return _postprocess_classic_results(results, cfg, n)
