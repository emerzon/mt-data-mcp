import numpy as np
from typing import List, Optional
from .config import ClassicDetectorConfig, ClassicPatternResult
from .utils import (
    _fit_line, _fit_line_robust, _tol_abs_from_close, _last_touch_indexes,
    _count_recent_touches, _result, _alias, _fit_lines_and_arrays,
    _is_converging, _count_touches, _conf
)

def detect_trend_lines(
    c: np.ndarray,
    peaks: np.ndarray,
    troughs: np.ndarray,
    t: np.ndarray,
    cfg: ClassicDetectorConfig
) -> List[ClassicPatternResult]:
    results: List[ClassicPatternResult] = []
    n = c.size
    confirm_needed = max(1, int(cfg.completion_confirm_bars))
    confirm_lookback = max(confirm_needed, int(cfg.completion_lookback_bars))

    for side, piv in (("high", peaks), ("low", troughs)):
        if piv.size >= max(3, cfg.min_touches):
            k = min(8, piv.size)
            idxs = piv[-k:]
            xs = idxs.astype(float)
            ys = c[idxs]
            slope, intercept, r2 = _fit_line_robust(xs, ys, cfg) if cfg.use_robust_fit else _fit_line(xs, ys)
            
            line_vals = slope * np.arange(n, dtype=float) + intercept
            tol_abs = _tol_abs_from_close(c, cfg.same_level_tol_pct)
            touches = len(_last_touch_indexes(line_vals, idxs, c, tol_abs))
            geom_ok = 1.0
            conf = _conf(touches, r2, geom_ok, cfg)
            status = "forming"
            tl_dir = 'Ascending' if slope > cfg.max_flat_slope else ('Descending' if slope < -cfg.max_flat_slope else 'Horizontal')
            name = f"{tl_dir} Trend Line"
            
            recent_i = n - 1
            recent_touches = _count_recent_touches(line_vals, c, tol_abs, confirm_lookback)
            if recent_touches >= confirm_needed:
                status = "completed"
            
            details = {
                "side": side,
                "slope": float(slope),
                "intercept": float(intercept),
                "r2": float(r2),
                "touches": int(touches),
                "line_level_recent": float(line_vals[recent_i]),
                "completion_touches_recent": int(recent_touches),
            }
            base_item = _result(name, status, conf, int(idxs[0]), int(idxs[-1]), t, details)
            results.append(base_item)
            
            if tl_dir != 'Horizontal' and bool(cfg.include_aliases):
                results.append(_alias(base_item, "Trend Line", 0.95))
    return results

def detect_channels(
    c: np.ndarray,
    peaks: np.ndarray,
    troughs: np.ndarray,
    t: np.ndarray,
    cfg: ClassicDetectorConfig
) -> List[ClassicPatternResult]:
    ch_results: List[ClassicPatternResult] = []
    if peaks.size < 3 or troughs.size < 3:
        return ch_results
        
    n = c.size
    k = min(8, peaks.size, troughs.size)
    ih = peaks[-k:]
    il = troughs[-k:]
    
    sh, bh, r2h, sl, bl, r2l, upper, lower = _fit_lines_and_arrays(ih, il, c, n, cfg)
    
    slope_diff = abs(sh - sl)
    approx_parallel = slope_diff <= max(
        1e-4,
        float(cfg.channel_parallel_slope_ratio) * max(abs(sh), abs(sl), cfg.max_flat_slope),
    )
    converging = _is_converging(upper, lower, k, n, cfg)
    width = upper - lower
    width_ok = float(np.std(width[-k:]) / (np.mean(width[-k:]) + 1e-9))
    geom_ok = 1.0 - min(1.0, max(0.0, width_ok))
    
    touches = 0
    tol_abs = _tol_abs_from_close(c, cfg.same_level_tol_pct)
    touches += _count_touches(upper, lower, peaks[-k:], troughs[-k:], c, tol_abs)
    
    name = "Trend Channel"
    if sh > cfg.max_flat_slope and sl > cfg.max_flat_slope:
        name = "Ascending Channel"
    elif sh < -cfg.max_flat_slope and sl < -cfg.max_flat_slope:
        name = "Descending Channel"
    elif abs(sh) <= cfg.max_flat_slope and abs(sl) <= cfg.max_flat_slope:
        name = "Horizontal Channel"
        
    if approx_parallel and not converging and touches >= cfg.min_channel_touches:
        conf = _conf(touches, min(r2h, r2l), geom_ok, cfg)
        status = "forming"
        recent_i = n - 1
        confirm_needed = max(1, int(cfg.completion_confirm_bars))
        confirm_lookback = max(confirm_needed, int(cfg.completion_lookback_bars))
        
        recent_hits = _count_recent_touches(upper, c, tol_abs, confirm_lookback) + _count_recent_touches(lower, c, tol_abs, confirm_lookback)
        if recent_hits >= confirm_needed:
            status = "completed"
            
        details = {
            "upper_slope": float(sh),
            "upper_intercept": float(bh),
            "lower_slope": float(sl),
            "lower_intercept": float(bl),
            "r2_upper": float(r2h),
            "r2_lower": float(r2l),
            "channel_width_recent": float(width[recent_i]),
            "completion_touches_recent": int(recent_hits),
        }
        base = _result(name, status, conf, int(min(ih[0], il[0])), int(max(ih[-1], il[-1])), t, details)
        ch_results.append(base)
        
        if bool(cfg.include_aliases):
            ch_results.append(_alias(base, "Trend Channel", 0.95))
            
    return ch_results
