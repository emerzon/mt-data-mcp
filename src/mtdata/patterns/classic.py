from __future__ import annotations
from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd

from ..utils.utils import to_float_np
from .classic_impl.config import ClassicDetectorConfig, ClassicPatternResult
from .classic_impl.utils import (
    _build_time_array, _detect_pivots_close, _calibrate_confidence,
    _fit_lines_and_arrays, _count_recent_touches, _fit_line_robust,
    _fit_line, _tol_abs_from_close, _level_close, _is_converging,
    _find_recent_breakout, _find_forward_level_breakout, _result, _alias,
    _count_touches, _conf, _last_touch_indexes, _compute_atr, _pivot_thresholds,
    _znorm, _paa, _dtw_distance, _template_hs, _collect_calibration_points
)
from .classic_impl.trend import detect_trend_lines, detect_channels
from .classic_impl.shapes import (
    detect_rectangles, detect_triangles, detect_wedges, 
    detect_broadening, detect_diamonds
)
from .classic_impl.reversal import (
    detect_tops_bottoms, detect_head_shoulders, detect_rounding
)
from .classic_impl.continuation import (
    detect_flags_pennants, detect_cup_handle
)

# Re-export for backward compatibility
__all__ = [
    "ClassicDetectorConfig",
    "ClassicPatternResult",
    "detect_classic_patterns",
]

def detect_classic_patterns(df: pd.DataFrame, cfg: Optional[ClassicDetectorConfig] = None) -> List[ClassicPatternResult]:
    """Detect classic chart patterns on OHLCV DataFrame with 'time' and 'close' columns.

    Returns a list of ClassicPatternResult with details and confidence.
    """
    if cfg is None:
        cfg = ClassicDetectorConfig()
    if not isinstance(df, pd.DataFrame) or 'close' not in df.columns:
        return []
    
    # Enforce max bars limit
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
    if n < 100:
        return []

    try:
        peaks, troughs = _detect_pivots_close(c, cfg, h, l)
    except TypeError:
        # Keep compatibility with monkeypatched tests that still use the old 2-arg signature.
        peaks, troughs = _detect_pivots_close(c, cfg)

    results: List[ClassicPatternResult] = []

    # 1. Trend Patterns
    results.extend(detect_trend_lines(c, peaks, troughs, t, cfg))
    results.extend(detect_channels(c, peaks, troughs, t, cfg))

    # 2. Shape Patterns
    results.extend(detect_rectangles(c, peaks, troughs, t, cfg))
    results.extend(detect_triangles(c, peaks, troughs, t, cfg))
    results.extend(detect_wedges(c, peaks, troughs, t, cfg))
    results.extend(detect_broadening(c, peaks, troughs, t, cfg))
    results.extend(detect_diamonds(c, t, cfg))

    # 3. Reversal Patterns
    results.extend(detect_tops_bottoms(c, peaks, troughs, t, cfg))
    results.extend(detect_head_shoulders(c, peaks, troughs, t, cfg))
    results.extend(detect_rounding(c, t, cfg))

    # 4. Continuation Patterns
    results.extend(detect_flags_pennants(c, h, l, t, n, cfg))
    results.extend(detect_cup_handle(c, t, cfg))

    # Post-processing
    
    # Optional backward-compatible status aging.
    if bool(cfg.auto_complete_stale_forming):
        try:
            recent_bars = 3
            for i, r in enumerate(results):
                try:
                    if r.status == 'forming' and r.end_index < (n - recent_bars):
                        results[i] = ClassicPatternResult(
                            name=r.name,
                            status='completed',
                            confidence=r.confidence,
                            start_index=r.start_index,
                            end_index=r.end_index,
                            start_time=r.start_time,
                            end_time=r.end_time,
                            details=r.details,
                        )
                except Exception:
                    continue
        except Exception:
            pass

    # Optional confidence calibration map from raw->empirical likelihood.
    for r in results:
        try:
            raw_conf = float(r.confidence)
            cal_conf = _calibrate_confidence(raw_conf, r.name, cfg)
            if bool(getattr(cfg, "calibrate_confidence", False)):
                if not isinstance(r.details, dict):
                    r.details = {}
                r.details["raw_confidence"] = float(raw_conf)
                r.details["calibrated_confidence"] = float(cal_conf)
            r.confidence = float(cal_conf)
        except Exception:
            continue

    # Lifecycle metadata used by downstream consumers.
    if bool(getattr(cfg, "include_lifecycle_metadata", True)):
        for r in results:
            try:
                if not isinstance(r.details, dict):
                    r.details = {}
                if r.status == "completed":
                    r.details.setdefault("lifecycle_state", "confirmed")
                else:
                    r.details.setdefault("lifecycle_state", "forming")
            except Exception:
                continue

    # Sort results by end_index (recency) then confidence
    results.sort(key=lambda r: (r.end_index, r.confidence), reverse=True)
    return results
