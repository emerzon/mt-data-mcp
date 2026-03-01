import numpy as np
from typing import List, Optional
from .config import ClassicDetectorConfig, ClassicPatternResult
from .utils import (
    _fit_line, _fit_line_robust, _fit_lines_and_arrays, 
    _tol_abs_from_close, _level_close, _count_touches, _count_recent_touches,
    _is_converging, _find_recent_breakout, _find_forward_level_breakout, 
    _result, _alias, _conf
)

def detect_rectangles(
    c: np.ndarray,
    peaks: np.ndarray,
    troughs: np.ndarray,
    t: np.ndarray,
    cfg: ClassicDetectorConfig
) -> List[ClassicPatternResult]:
    out: List[ClassicPatternResult] = []
    n = c.size
    k = min(8, peaks.size, troughs.size)
    if k < 3:
        return out
        
    ph = c[peaks[-k:]]; pl = c[troughs[-k:]]
    top = float(np.median(ph))
    bot = float(np.median(pl))
    if top <= bot:
        return out
        
    equal_highs = np.mean([_level_close(v, top, cfg.same_level_tol_pct) for v in ph])
    equal_lows = np.mean([_level_close(v, bot, cfg.same_level_tol_pct) for v in pl])
    touches = int(np.round(equal_highs * ph.size + equal_lows * pl.size))
    
    if equal_highs > 0.6 and equal_lows > 0.6 and touches >= cfg.min_channel_touches - 1:
        geom_ok = 1.0
        conf = _conf(touches, 1.0, geom_ok, cfg)
        status = "forming"
        
        recent_i = n - 1
        top_line = np.full(n, top, dtype=float)
        bot_line = np.full(n, bot, dtype=float)
        confirm_needed = max(1, int(cfg.completion_confirm_bars))
        confirm_lookback = max(confirm_needed, int(cfg.completion_lookback_bars))
        
        recent_hits = _count_recent_touches(top_line, c, _tol_abs_from_close(c, cfg.same_level_tol_pct), confirm_lookback)
        recent_hits += _count_recent_touches(bot_line, c, _tol_abs_from_close(c, cfg.same_level_tol_pct), confirm_lookback)
        
        if recent_hits >= confirm_needed:
            status = "completed"
            
        out.append(ClassicPatternResult(
            name="Rectangle",
            status=status,
            confidence=conf,
            start_index=int(min(peaks[-k], troughs[-k])),
            end_index=int(max(peaks[-1], troughs[-1])),
            start_time=float(t[int(min(peaks[-k], troughs[-k]))]) if t.size else None,
            end_time=float(t[int(max(peaks[-1], troughs[-1]))]) if t.size else None,
            details={"resistance": top, "support": bot, "touches": touches, "completion_touches_recent": int(recent_hits)},
        ))
    return out

def detect_triangles(
    c: np.ndarray,
    peaks: np.ndarray,
    troughs: np.ndarray,
    t: np.ndarray,
    cfg: ClassicDetectorConfig
) -> List[ClassicPatternResult]:
    out: List[ClassicPatternResult] = []
    n = c.size
    k = min(8, peaks.size, troughs.size)
    if k < 4:
        return out
        
    ih = peaks[-k:]; il = troughs[-k:]
    sh, bh, r2h, sl, bl, r2l, top, bot = _fit_lines_and_arrays(ih, il, c, n, cfg)
    
    converging = _is_converging(top, bot, k, n, cfg)
    tol_abs = _tol_abs_from_close(c, cfg.same_level_tol_pct)
    touches = _count_touches(top, bot, ih, il, c, tol_abs)
    
    if converging and touches >= cfg.min_channel_touches - 1:
        if abs(sh) <= cfg.max_flat_slope and sl > cfg.max_flat_slope:
            name = "Ascending Triangle"
        elif abs(sl) <= cfg.max_flat_slope and sh < -cfg.max_flat_slope:
            name = "Descending Triangle"
        else:
            name = "Symmetrical Triangle"
            
        conf = _conf(touches, min(r2h, r2l), 1.0, cfg)
        status = "forming"
        breakout_look = max(int(cfg.completion_lookback_bars), int(max(1, cfg.breakout_lookahead)))
        bdir, bidx = _find_recent_breakout(c, upper=top, lower=bot, tol_abs=tol_abs, lookback_bars=breakout_look)
        
        if bdir is not None and bidx is not None:
            status = "completed"
            conf = min(1.0, conf + 0.08)
            
        out.append(_result(
            name,
            status,
            conf,
            int(min(ih[0], il[0])),
            int(max(int(max(ih[-1], il[-1])), int(bidx) if bidx is not None else int(max(ih[-1], il[-1])))),
            t,
            {
                "top_slope": float(sh),
                "top_intercept": float(bh),
                "bottom_slope": float(sl),
                "bottom_intercept": float(bl),
                "breakout_direction": bdir,
                "breakout_index": int(bidx) if bidx is not None else None,
            },
        ))
    return out

def detect_wedges(
    c: np.ndarray,
    peaks: np.ndarray,
    troughs: np.ndarray,
    t: np.ndarray,
    cfg: ClassicDetectorConfig
) -> List[ClassicPatternResult]:
    out: List[ClassicPatternResult] = []
    n = c.size
    k = min(8, peaks.size, troughs.size)
    if k < 4:
        return out
        
    ih = peaks[-k:]; il = troughs[-k:]
    sh, bh, r2h, sl, bl, r2l, top, bot = _fit_lines_and_arrays(ih, il, c, n, cfg)
    
    converging = _is_converging(top, bot, k, n, cfg)
    same_sign = (sh > 0 and sl > 0) or (sh < 0 and sl < 0)
    tol_abs = _tol_abs_from_close(c, cfg.same_level_tol_pct)
    touches = _count_touches(top, bot, ih, il, c, tol_abs)
    
    if converging and same_sign and touches >= cfg.min_channel_touches - 1:
        name = "Rising Wedge" if sh > 0 and sl > 0 else "Falling Wedge"
        conf = _conf(touches, min(r2h, r2l), 1.0, cfg)
        status = "forming"
        breakout_look = max(int(cfg.completion_lookback_bars), int(max(1, cfg.breakout_lookahead)))
        bdir, bidx = _find_recent_breakout(c, upper=top, lower=bot, tol_abs=tol_abs, lookback_bars=breakout_look)
        
        if bdir is not None and bidx is not None:
            status = "completed"
            conf = min(1.0, conf + 0.08)
            
        base = _result(
            name,
            status,
            conf,
            int(min(ih[0], il[0])),
            int(max(int(max(ih[-1], il[-1])), int(bidx) if bidx is not None else int(max(ih[-1], il[-1])))),
            t,
            {
                "top_slope": float(sh),
                "bottom_slope": float(sl),
                "top_intercept": float(bh),
                "bottom_intercept": float(bl),
                "breakout_direction": bdir,
                "breakout_index": int(bidx) if bidx is not None else None,
            },
        )
        out.append(base)
        out.append(_alias(base, "Wedge", 0.95))
    return out

def detect_broadening(
    c: np.ndarray,
    peaks: np.ndarray,
    troughs: np.ndarray,
    t: np.ndarray,
    cfg: ClassicDetectorConfig
) -> List[ClassicPatternResult]:
    out: List[ClassicPatternResult] = []
    n = c.size
    k = min(10, peaks.size, troughs.size)
    if k < 4:
        return out
        
    ih = peaks[-k:]; il = troughs[-k:]
    sh, bh, r2h = _fit_line(ih.astype(float), c[ih])
    sl, bl, r2l = _fit_line(il.astype(float), c[il])
    
    diverging = (sh > cfg.max_flat_slope and sl < -cfg.max_flat_slope)
    if diverging:
        conf = _conf(4, min(r2h, r2l), 1.0, cfg)
        x = np.arange(n, dtype=float)
        top = sh * x + bh
        bot = sl * x + bl
        tol_abs = _tol_abs_from_close(c, cfg.same_level_tol_pct)
        status = "forming"
        breakout_look = max(int(cfg.completion_lookback_bars), int(max(1, cfg.breakout_lookahead)))
        bdir, bidx = _find_recent_breakout(c, upper=top, lower=bot, tol_abs=tol_abs, lookback_bars=breakout_look)
        
        if bdir is not None and bidx is not None:
            status = "completed"
            conf = min(1.0, conf + 0.08)
            
        out.append(_result(
            "Broadening Formation",
            status,
            conf,
            int(min(ih[0], il[0])),
            int(max(int(max(ih[-1], il[-1])), int(bidx) if bidx is not None else int(max(ih[-1], il[-1])))),
            t,
            {"top_slope": float(sh), "bottom_slope": float(sl),
             "top_intercept": float(bh), "bottom_intercept": float(bl),
             "breakout_direction": bdir,
             "breakout_index": int(bidx) if bidx is not None else None},
        ))
    return out

def detect_diamonds(
    c: np.ndarray,
    t: np.ndarray,
    cfg: ClassicDetectorConfig
) -> List[ClassicPatternResult]:
    out: List[ClassicPatternResult] = []
    n = c.size
    W = min(240, n)
    if W < 120:
        return out
        
    seg = c[-W:]
    half = W // 2
    left = seg[:half]; right = seg[half:]
    
    def width(a: np.ndarray) -> float:
        return float(np.max(a) - np.min(a))
        
    expanding = width(left[:half//2]) < width(left[half//2:])
    contracting = width(right[:half//2]) > width(right[half//2:])
    
    if expanding and contracting:
        pole = min(60, max(10, W // 3))
        if n > 2 * pole + 1:
            ret = (c[-1 - pole] - c[-1 - 2*pole]) / max(1e-9, c[-1 - 2*pole]) * 100.0
        else:
            ret = 0.0
            
        name = "Continuation Diamond" if abs(ret) >= 2.0 else "Diamond"
        status = "forming"
        conf = 0.6
        
        right_hi = float(np.max(right)) if right.size else float(np.max(seg))
        right_lo = float(np.min(right)) if right.size else float(np.min(seg))
        tol_abs = _tol_abs_from_close(c, cfg.same_level_tol_pct)
        breakout_look = max(int(cfg.completion_lookback_bars), int(max(1, cfg.breakout_lookahead)))
        
        break_up = _find_forward_level_breakout(c, int(n - W + half), right_hi, "up", breakout_look, tol_abs)
        break_dn = _find_forward_level_breakout(c, int(n - W + half), right_lo, "down", breakout_look, tol_abs)
        
        break_i: Optional[int] = None
        break_dir: Optional[str] = None
        
        if break_up is not None and (break_dn is None or break_up <= break_dn):
            break_i = int(break_up)
            break_dir = "up"
        elif break_dn is not None:
            break_i = int(break_dn)
            break_dir = "down"
            
        if break_i is not None:
            status = "completed"
            conf = min(1.0, conf + 0.08)
            
        out.append(_result(
            name,
            status,
            conf,
            int(n - W),
            int(break_i if break_i is not None else (n - 1)),
            t,
            {
                "prior_pole_return_pct": float(ret),
                "breakout_direction": break_dir,
                "breakout_index": int(break_i) if break_i is not None else None,
            },
        ))
    return out
