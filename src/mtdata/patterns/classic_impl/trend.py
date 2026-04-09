from typing import List

import numpy as np

from .config import ClassicDetectorConfig, ClassicPatternResult
from .utils import (
    _alias,
    _boundaries_are_ordered,
    _conf,
    _count_recent_touches,
    _count_touches,
    _find_recent_breakout,
    _fit_line,
    _fit_line_robust,
    _fit_lines_and_arrays,
    _is_converging,
    _last_touch_indexes,
    _result,
    _tol_abs_from_close,
)


def _channel_width_expansion_ratio(
    width: np.ndarray,
    k: int,
    n: int,
) -> float:
    if width.size == 0:
        return 0.0
    last = max(5, int(k))
    recent = float(np.mean(width[-last:])) if width.size >= last else float(np.mean(width))
    past_win = max(20, 2 * int(k))
    prev_win = width[-past_win:-last] if n > past_win else width[:max(1, width.size // 2)]
    past = float(np.mean(prev_win)) if prev_win.size > 0 else recent
    if not np.isfinite(recent) or not np.isfinite(past) or abs(past) <= 1e-9:
        return 0.0
    return float((recent - past) / abs(past))

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
            k = min(int(cfg.max_pattern_pivots), piv.size)
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
            breakout_direction = None
            breakout_index = None
            if side == "high":
                breakout_direction, breakout_index = _find_recent_breakout(
                    c,
                    upper=line_vals,
                    tol_abs=tol_abs,
                    tol_pct=float(cfg.same_level_tol_pct),
                    lookback_bars=confirm_lookback,
                )
                if breakout_direction == "up":
                    status = "completed"
            else:
                breakout_direction, breakout_index = _find_recent_breakout(
                    c,
                    lower=line_vals,
                    tol_abs=tol_abs,
                    tol_pct=float(cfg.same_level_tol_pct),
                    lookback_bars=confirm_lookback,
                )
                if breakout_direction == "down":
                    status = "completed"
            end_i = int(breakout_index if breakout_index is not None else (n - 1))
            if recent_touches >= confirm_needed:
                conf = min(1.0, float(conf) + 0.02)
            if breakout_index is not None:
                conf = min(1.0, float(conf) + 0.06)

            details = {
                "side": side,
                "slope": float(slope),
                "intercept": float(intercept),
                "r2": float(r2),
                "touches": int(touches),
                "line_level_recent": float(line_vals[recent_i]),
                "touches_recent": int(recent_touches),
                "breakout_direction": breakout_direction,
                "breakout_index": int(breakout_index) if breakout_index is not None else None,
                "bias": "bullish" if tl_dir == "Ascending" else "bearish" if tl_dir == "Descending" else "neutral",
            }
            base_item = _result(name, status, conf, int(idxs[0]), end_i, t, details)
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
    k = min(int(cfg.max_pattern_pivots), peaks.size, troughs.size)
    ih = peaks[-k:]
    il = troughs[-k:]
    
    sh, bh, r2h, sl, bl, r2l, upper, lower = _fit_lines_and_arrays(ih, il, c, n, cfg)
    channel_start = int(min(ih[0], il[0]))
    if not _boundaries_are_ordered(upper, lower, start_idx=channel_start, end_idx=n - 1):
        return ch_results
    
    slope_diff = abs(sh - sl)
    approx_parallel = slope_diff <= max(
        float(cfg.channel_parallel_min_abs_tol),
        float(cfg.channel_parallel_slope_ratio) * max(abs(sh), abs(sl)),
    )
    converging = _is_converging(upper, lower, k, n, cfg)
    width = upper - lower
    width_ok = float(np.std(width[-k:]) / (np.mean(width[-k:]) + 1e-9))
    geom_ok = 1.0 - min(1.0, max(0.0, width_ok))
    width_expansion_ratio = _channel_width_expansion_ratio(width, k, n)
    widening = width_expansion_ratio > float(cfg.channel_max_width_expansion_ratio)
    
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
        
    if approx_parallel and not converging and not widening and touches >= cfg.min_channel_touches:
        conf = _conf(touches, min(r2h, r2l), geom_ok, cfg)
        status = "forming"
        recent_i = n - 1
        confirm_needed = max(1, int(cfg.completion_confirm_bars))
        confirm_lookback = max(confirm_needed, int(cfg.completion_lookback_bars))
        
        recent_hits = _count_recent_touches(upper, c, tol_abs, confirm_lookback) + _count_recent_touches(lower, c, tol_abs, confirm_lookback)
        breakout_direction, breakout_index = _find_recent_breakout(
            c,
            upper=upper,
            lower=lower,
            tol_abs=tol_abs,
            tol_pct=float(cfg.same_level_tol_pct),
            lookback_bars=confirm_lookback,
        )
        if breakout_index is not None:
            status = "completed"
        if recent_hits >= confirm_needed:
            conf = min(1.0, float(conf) + 0.02)
        if breakout_index is not None:
            conf = min(1.0, float(conf) + 0.06)
        end_i = int(breakout_index if breakout_index is not None else max(ih[-1], il[-1]))
            
        details = {
            "upper_slope": float(sh),
            "upper_intercept": float(bh),
            "lower_slope": float(sl),
            "lower_intercept": float(bl),
            "r2_upper": float(r2h),
            "r2_lower": float(r2l),
            "channel_width_recent": float(width[recent_i]),
            "channel_width_expansion_ratio": float(width_expansion_ratio),
            "touches_recent": int(recent_hits),
            "breakout_direction": breakout_direction,
            "breakout_index": int(breakout_index) if breakout_index is not None else None,
            "bias": "bullish" if name == "Ascending Channel" else "bearish" if name == "Descending Channel" else "neutral",
        }
        base = _result(name, status, conf, int(min(ih[0], il[0])), end_i, t, details)
        ch_results.append(base)
        
        if bool(cfg.include_aliases):
            ch_results.append(_alias(base, "Trend Channel", 0.95))
            
    return ch_results
