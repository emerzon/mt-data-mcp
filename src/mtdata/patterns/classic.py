from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from ..utils.utils import to_float_np
from scipy.signal import find_peaks


@dataclass
class ClassicDetectorConfig:
    # General
    max_bars: int = 1500
    # Pivot/zigzag parameters
    min_prominence_pct: float = 0.5  # peak/trough prominence in percent of price
    min_distance: int = 5            # minimum distance between pivots (bars)
    # Trendline/line-fit
    max_flat_slope: float = 1e-4     # absolute slope to consider line flat (price units per bar)
    min_r2: float = 0.6              # minimum R^2 for line fit confidence
    # Levels tolerance (same-level checks)
    same_level_tol_pct: float = 0.4  # peaks/lows considered equal if within this percent
    # Pattern-specific
    min_touches: int = 2             # minimum touches to validate support/resistance boundary
    min_channel_touches: int = 4     # across both bounds
    max_consolidation_bars: int = 60 # for flags/pennants after pole
    min_pole_return_pct: float = 2.0 # minimum pole size (percent) before a flag/pennant
    breakout_lookahead: int = 8      # bars to consider breakout confirmation
    # Confidence blending
    touch_weight: float = 0.35
    r2_weight: float = 0.35
    geometry_weight: float = 0.30
    # Robust fitting and shape checks
    use_robust_fit: bool = False     # use RANSAC for line fits when available
    ransac_residual_pct: float = 0.15  # residual threshold as fraction of median price
    ransac_min_samples: int = 2
    ransac_max_trials: int = 50
    use_dtw_check: bool = False      # optional DTW shape confirmation for select patterns
    dtw_paa_len: int = 80            # PAA downsampling length for DTW
    dtw_max_dist: float = 0.6        # acceptance threshold after z-norm


@dataclass
class ClassicPatternResult:
    name: str
    status: str  # "completed" | "forming"
    confidence: float
    start_index: int
    end_index: int
    start_time: Optional[float]
    end_time: Optional[float]
    details: Dict[str, Any]


def _level_close(a: float, b: float, tol_pct: float) -> bool:
    if a == 0 or b == 0:
        return abs(a - b) <= 1e-12
    return abs((a - b) / ((abs(a) + abs(b)) / 2.0)) * 100.0 <= tol_pct


def _fit_line(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    # slope, intercept, r2
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2:
        return 0.0, float(y.mean() if y.size else 0.0), 0.0
    p = np.polyfit(x, y, 1)
    y_hat = p[0] * x + p[1]
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2)) if y.size else 0.0
    r2 = 0.0 if ss_tot == 0 else max(0.0, 1.0 - ss_res / ss_tot)
    return float(p[0]), float(p[1]), float(r2)


def _fit_line_robust(x: np.ndarray, y: np.ndarray, cfg: ClassicDetectorConfig) -> Tuple[float, float, float]:
    """Optionally fit a robust line via RANSAC; fallback to ordinary fit.

    Returns (slope, intercept, r2_like). r2_like is computed vs. fitted values.
    """
    try:
        if not cfg.use_robust_fit:
            return _fit_line(x, y)
        from sklearn.linear_model import RANSACRegressor  # type: ignore
        X = x.reshape(-1, 1).astype(float)
        yv = y.astype(float)
        if X.shape[0] < max(2, int(cfg.ransac_min_samples)):
            return _fit_line(x, y)
        med = float(np.median(np.abs(yv))) if yv.size else 1.0
        resid = max(1e-9, float(cfg.ransac_residual_pct) * max(1.0, med))
        model = RANSACRegressor(min_samples=max(2, int(cfg.ransac_min_samples)),
                                max_trials=max(10, int(cfg.ransac_max_trials)),
                                residual_threshold=resid,
                                random_state=0)
        model.fit(X, yv)
        # Extract slope/intercept from linear estimator
        est = model.estimator_
        slope = float(getattr(est, 'coef_', [0.0])[0])
        intercept = float(getattr(est, 'intercept_', 0.0))
        y_hat = slope * x + intercept
        ss_res = float(np.sum((yv - y_hat) ** 2))
        ss_tot = float(np.sum((yv - yv.mean()) ** 2)) if yv.size else 0.0
        r2 = 0.0 if ss_tot == 0 else max(0.0, 1.0 - ss_res / ss_tot)
        return slope, intercept, r2
    except Exception:
        return _fit_line(x, y)


def _znorm(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    if a.size == 0:
        return a
    mu = float(np.mean(a))
    sd = float(np.std(a))
    if not np.isfinite(sd) or sd <= 1e-12:
        return np.zeros_like(a, dtype=float)
    return (a - mu) / sd


def _paa(a: np.ndarray, m: int) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    n = a.size
    if n == 0 or m <= 0:
        return np.asarray([], dtype=float)
    if n == m:
        return a.copy()
    idx = np.linspace(0, n, num=m + 1, dtype=float)
    out = []
    for i in range(m):
        s = int(idx[i]); e = int(idx[i + 1])
        if e <= s:
            e = min(n, s + 1)
        out.append(float(np.mean(a[s:e])))
    return np.asarray(out, dtype=float)


def _dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    try:
        from tslearn.metrics import dtw as _ts_dtw  # type: ignore
        return float(_ts_dtw(a.astype(float), b.astype(float)))
    except Exception:
        try:
            from dtaidistance import dtw as _dd_dtw  # type: ignore
            return float(_dd_dtw.distance_fast(a.astype(float), b.astype(float)))
        except Exception:
            # Simple DP fallback
            n, m = len(a), len(b)
            if n == 0 or m == 0:
                return float('inf')
            dp = np.full((n + 1, m + 1), np.inf, dtype=float)
            dp[0, 0] = 0.0
            for i in range(1, n + 1):
                for j in range(1, m + 1):
                    cost = abs(a[i - 1] - b[j - 1])
                    dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
            return float(dp[n, m])


def _template_hs(L: int, inverse: bool = False) -> np.ndarray:
    """Simple H&S template over L points (z-norm target)."""
    L = max(20, int(L))
    x = np.linspace(0, 1, L)
    # Shoulders around 0.7, head at 1.0 (or inverted)
    y = 0.0 * x
    y += 0.7 * np.exp(-((x - 0.25) ** 2) / 0.01)
    y += 1.0 * np.exp(-((x - 0.5) ** 2) / 0.008)
    y += 0.7 * np.exp(-((x - 0.75) ** 2) / 0.01)
    if inverse:
        y = -y
    return _znorm(y)


def _detect_pivots_close(close: np.ndarray, cfg: ClassicDetectorConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Return arrays of peak and trough indices based on prominence in percent of price."""
    x = np.asarray(close, dtype=float)
    if x.size < max(50, cfg.min_distance * 3):
        return np.array([], dtype=int), np.array([], dtype=int)
    # Use percent of median price as absolute prominence threshold
    base = float(np.median(x)) if np.isfinite(np.median(x)) and np.median(x) > 1e-9 else float(np.mean(x))
    prom_abs = abs(base) * (cfg.min_prominence_pct / 100.0)
    try:
        peaks, _ = find_peaks(x, prominence=prom_abs, distance=cfg.min_distance)
        troughs, _ = find_peaks(-x, prominence=prom_abs, distance=cfg.min_distance)
    except Exception:
        return np.array([], dtype=int), np.array([], dtype=int)
    return peaks.astype(int), troughs.astype(int)


    


def _last_touch_indexes(bound_y: np.ndarray, idxs: np.ndarray, y: np.ndarray, tol: float) -> List[int]:
    out: List[int] = []
    for i in idxs.tolist():
        if i < 0 or i >= y.size:
            continue
        if abs(bound_y[i] - y[i]) <= tol:
            out.append(int(i))
    return out


def _build_time_array(df: pd.DataFrame) -> np.ndarray:
    t = df.get('time')
    if t is None:
        return np.arange(len(df), dtype=float)
    try:
        return to_float_np(t)
    except Exception:
        return np.arange(len(df), dtype=float)


def _tol_abs_from_close(close: np.ndarray, tol_pct: float) -> float:
    """Absolute tolerance based on median close and percent threshold."""
    med = float(np.median(close)) if close.size else 0.0
    return abs(med) * (float(tol_pct) / 100.0)


def _result(name: str, status: str, confidence: float,
            start_index: int, end_index: int,
            t: np.ndarray, details: Dict[str, Any]) -> ClassicPatternResult:
    return ClassicPatternResult(
        name=name,
        status=status,
        confidence=float(confidence),
        start_index=int(start_index),
        end_index=int(end_index),
        start_time=float(t[int(start_index)]) if t.size else None,
        end_time=float(t[int(end_index)]) if t.size else None,
        details=details,
    )


def _alias(base: ClassicPatternResult, name: str, conf_scale: float = 0.95) -> ClassicPatternResult:
    return ClassicPatternResult(
        name=name,
        status=base.status,
        confidence=min(1.0, float(base.confidence) * float(conf_scale)),
        start_index=base.start_index,
        end_index=base.end_index,
        start_time=base.start_time,
        end_time=base.end_time,
        details=base.details,
    )


def _fit_lines_and_arrays(ih: np.ndarray, il: np.ndarray, c: np.ndarray, n: int):
    # Capture cfg via closure in detect_classic_patterns
    try:
        sh, bh, r2h = _fit_line_robust(ih.astype(float), c[ih], cfg)
    except Exception:
        sh, bh, r2h = _fit_line(ih.astype(float), c[ih])
    try:
        sl, bl, r2l = _fit_line_robust(il.astype(float), c[il], cfg)
    except Exception:
        sl, bl, r2l = _fit_line(il.astype(float), c[il])
    x = np.arange(n, dtype=float)
    upper = sh * x + bh
    lower = sl * x + bl
    return sh, bh, r2h, sl, bl, r2l, upper, lower


def _count_touches(upper: np.ndarray, lower: np.ndarray,
                   peak_idxs: np.ndarray, trough_idxs: np.ndarray,
                   c: np.ndarray, tol_abs: float) -> int:
    touches = 0
    touches += len(_last_touch_indexes(upper, peak_idxs, c, tol_abs))
    touches += len(_last_touch_indexes(lower, trough_idxs, c, tol_abs))
    return touches


def _is_converging(upper: np.ndarray, lower: np.ndarray, k: int, n: int) -> bool:
    """Heuristic to detect converging lines over recent window vs past window."""
    span = upper - lower
    last = max(5, int(k))
    recent = float(np.mean(span[-last:])) if span.size >= last else float(np.mean(span))
    past_win = max(20, 2 * int(k))
    prev_win = span[-past_win:-last] if n > past_win else span[:max(1, span.size // 2)]
    past = float(np.mean(prev_win)) if prev_win.size > 0 else recent * 1.2
    return bool(recent < past)


def detect_classic_patterns(df: pd.DataFrame, cfg: Optional[ClassicDetectorConfig] = None) -> List[ClassicPatternResult]:
    """Detect classic chart patterns on OHLCV DataFrame with 'time' and 'close' columns.

    Returns a list of ClassicPatternResult with details and confidence.
    """
    if cfg is None:
        cfg = ClassicDetectorConfig()
    if not isinstance(df, pd.DataFrame) or 'close' not in df.columns:
        return []
    if len(df) > cfg.max_bars:
        df = df.iloc[-cfg.max_bars:].copy()
    t = _build_time_array(df)
    c = to_float_np(df['close'])
    n = c.size
    if n < 100:
        return []
    peaks, troughs = _detect_pivots_close(c, cfg)
    results: List[ClassicPatternResult] = []

    # Helper for confidence calculation
    def _conf(touches: int, r2: float, geom_ok: float) -> float:
        a = min(1.0, touches / max(1, cfg.min_touches)) * cfg.touch_weight
        b = max(0.0, min(1.0, r2)) * cfg.r2_weight
        g = max(0.0, min(1.0, geom_ok)) * cfg.geometry_weight
        return float(min(1.0, a + b + g))

    # 1) Trend Lines (Ascending/Descending/Horizontal) and Trend Channels
    # Fit on last K pivot highs and lows
    for side, piv in (("high", peaks), ("low", troughs)):
        if piv.size >= max(3, cfg.min_touches):
            k = min(8, piv.size)
            idxs = piv[-k:]
            xs = idxs.astype(float)
            ys = c[idxs]
            slope, intercept, r2 = _fit_line_robust(xs, ys, cfg) if cfg.use_robust_fit else _fit_line(xs, ys)
            # Assess touches: distance to line relative to price
            line_vals = slope * np.arange(n, dtype=float) + intercept
            tol_abs = _tol_abs_from_close(c, cfg.same_level_tol_pct)
            touches = len(_last_touch_indexes(line_vals, idxs, c, tol_abs))
            geom_ok = 1.0
            conf = _conf(touches, r2, geom_ok)
            status = "forming"
            tl_dir = 'Ascending' if slope > cfg.max_flat_slope else ('Descending' if slope < -cfg.max_flat_slope else 'Horizontal')
            name = f"{tl_dir} Trend Line"
            # Completion heuristic: recent close touched line within tolerance
            recent_i = n - 1
            if abs(line_vals[recent_i] - c[recent_i]) <= tol_abs:
                status = "completed"
            details = {
                "side": side,
                "slope": float(slope),
                "intercept": float(intercept),
                "r2": float(r2),
                "touches": int(touches),
                "line_level_recent": float(line_vals[recent_i]),
            }
            base_item = _result(name, status, conf, int(idxs[0]), int(idxs[-1]), t, details)
            results.append(base_item)
            # Also add a generic Trend Line alias (except perfectly horizontal)
            if tl_dir != 'Horizontal':
                results.append(_alias(base_item, "Trend Line", 0.95))

    # Channels: parallel lines from highs and lows
    def _try_channel():
        ch_results: List[ClassicPatternResult] = []
        if peaks.size < 3 or troughs.size < 3:
            return ch_results
        # Fit lines on last K highs and lows
        k = min(8, peaks.size, troughs.size)
        ih = peaks[-k:]; il = troughs[-k:]
        sh, bh, r2h, sl, bl, r2l, upper, lower = _fit_lines_and_arrays(ih, il, c, n)
        # Slopes near equal for channels; width relatively stable
        slope_diff = abs(sh - sl)
        approx_parallel = slope_diff <= max(1e-4, 0.15 * max(abs(sh), abs(sl), cfg.max_flat_slope))
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
        if approx_parallel and touches >= cfg.min_channel_touches:
            conf = _conf(touches, min(r2h, r2l), geom_ok)
            status = "forming"
            recent_i = n - 1
            hit_upper = abs(upper[recent_i] - c[recent_i]) <= tol_abs
            hit_lower = abs(lower[recent_i] - c[recent_i]) <= tol_abs
            if hit_upper or hit_lower:
                status = "completed"
            details = {
                "upper_slope": float(sh),
                "upper_intercept": float(bh),
                "lower_slope": float(sl),
                "lower_intercept": float(bl),
                "r2_upper": float(r2h),
                "r2_lower": float(r2l),
                "channel_width_recent": float(width[recent_i]),
            }
            base = _result(name, status, conf, int(min(ih[0], il[0])), int(max(ih[-1], il[-1])), t, details)
            ch_results.append(base)
            # Generic channel alias
            ch_results.append(_alias(base, "Trend Channel", 0.95))
        return ch_results

    results.extend(_try_channel())

    # Rectangles (Range/Rectangle)
    def _try_rectangle():
        out: List[ClassicPatternResult] = []
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
            conf = _conf(touches, 1.0, geom_ok)
            status = "forming"
            recent_i = n - 1
            if _level_close(c[recent_i], top, cfg.same_level_tol_pct) or _level_close(c[recent_i], bot, cfg.same_level_tol_pct):
                status = "completed"
            out.append(ClassicPatternResult(
                name="Rectangle",
                status=status,
                confidence=conf,
                start_index=int(min(peaks[-k], troughs[-k])),
                end_index=int(max(peaks[-1], troughs[-1])),
                start_time=float(t[int(min(peaks[-k], troughs[-k]))]) if t.size else None,
                end_time=float(t[int(max(peaks[-1], troughs[-1]))]) if t.size else None,
                details={"resistance": top, "support": bot, "touches": touches},
            ))
        return out

    results.extend(_try_rectangle())

    # Triangles (Ascending, Descending, Symmetrical)
    def _try_triangles():
        out: List[ClassicPatternResult] = []
        k = min(8, peaks.size, troughs.size)
        if k < 4:
            return out
        ih = peaks[-k:]; il = troughs[-k:]
        sh, bh, r2h, sl, bl, r2l, top, bot = _fit_lines_and_arrays(ih, il, c, n)
        # Lines must converge: distance decreasing toward recent bars
        converging = _is_converging(top, bot, k, n)
        tol_abs = _tol_abs_from_close(c, cfg.same_level_tol_pct)
        touches = _count_touches(top, bot, ih, il, c, tol_abs)
        if converging and touches >= cfg.min_channel_touches - 1:
            if abs(sh) <= cfg.max_flat_slope and sl > cfg.max_flat_slope:
                name = "Ascending Triangle"
            elif abs(sl) <= cfg.max_flat_slope and sh < -cfg.max_flat_slope:
                name = "Descending Triangle"
            else:
                name = "Symmetrical Triangle"
            conf = _conf(touches, min(r2h, r2l), 1.0)
            status = "forming"
            out.append(_result(
                name,
                status,
                conf,
                int(min(ih[0], il[0])),
                int(max(ih[-1], il[-1])),
                t,
                {
                    "top_slope": float(sh),
                    "top_intercept": float(bh),
                    "bottom_slope": float(sl),
                    "bottom_intercept": float(bl),
                },
            ))
        return out

    results.extend(_try_triangles())

    # Wedges (Rising/Falling): converging with both sloped same sign
    def _try_wedges():
        out: List[ClassicPatternResult] = []
        k = min(8, peaks.size, troughs.size)
        if k < 4:
            return out
        ih = peaks[-k:]; il = troughs[-k:]
        sh, bh, r2h, sl, bl, r2l, top, bot = _fit_lines_and_arrays(ih, il, c, n)
        converging = _is_converging(top, bot, k, n)
        same_sign = (sh > 0 and sl > 0) or (sh < 0 and sl < 0)
        tol_abs = _tol_abs_from_close(c, cfg.same_level_tol_pct)
        touches = _count_touches(top, bot, ih, il, c, tol_abs)
        if converging and same_sign and touches >= cfg.min_channel_touches - 1:
            name = "Rising Wedge" if sh > 0 and sl > 0 else "Falling Wedge"
            conf = _conf(touches, min(r2h, r2l), 1.0)
            base = _result(name, "forming", conf, int(min(ih[0], il[0])), int(max(ih[-1], il[-1])), t,
                           {"top_slope": float(sh), "bottom_slope": float(sl)})
            out.append(base)
            out.append(_alias(base, "Wedge", 0.95))
        return out

    results.extend(_try_wedges())

    # Double/Triple Tops & Bottoms
    def _tops_bottoms():
        out: List[ClassicPatternResult] = []
        def group_levels(idxs: np.ndarray, name_top: str, name_triple: str):
            if idxs.size < 2:
                return
            vals = c[idxs]
            # cluster by approximate equal level
            used = np.zeros(idxs.size, dtype=bool)
            for i in range(idxs.size):
                if used[i]:
                    continue
                level = vals[i]
                cluster = [i]
                for j in range(i+1, idxs.size):
                    if used[j]:
                        continue
                    if _level_close(vals[j], level, cfg.same_level_tol_pct):
                        cluster.append(j)
                if len(cluster) >= 2:
                    used[cluster] = True
                    name = name_triple if len(cluster) >= 3 else name_top
                    ii = idxs[cluster]
                    start_i, end_i = int(ii[0]), int(ii[-1])
                    status = "forming"
                    out.append(ClassicPatternResult(
                        name=name,
                        status=status,
                        confidence=min(1.0, 0.5 + 0.1 * (len(cluster) - 2 + 2)),
                        start_index=start_i,
                        end_index=end_i,
                        start_time=float(t[start_i]) if t.size else None,
                        end_time=float(t[end_i]) if t.size else None,
                        details={"level": float(np.median(vals[cluster])), "touches": int(len(cluster))},
                    ))
        group_levels(peaks[-10:], "Double Top", "Triple Top")
        group_levels(troughs[-10:], "Double Bottom", "Triple Bottom")
        return out

    results.extend(_tops_bottoms())

    # Head and Shoulders / Inverse (flexible)
    def _head_shoulders():
        out: List[ClassicPatternResult] = []
        if peaks.size < 3 or troughs.size < 2:
            return out
        tol_pct = float(cfg.same_level_tol_pct)
        for head_idx in peaks.tolist():
            try:
                head_price = float(c[head_idx])
                ls_candidates = [pi for pi in peaks.tolist() if pi < head_idx]
                rs_candidates = [pi for pi in peaks.tolist() if pi > head_idx]
                if not ls_candidates or not rs_candidates:
                    continue
                lsh = int(ls_candidates[-1]); rsh = int(rs_candidates[0])
                ls_p = float(c[lsh]); rs_p = float(c[rsh])
                regular = (ls_p < head_price) and (rs_p < head_price)
                inverse = (ls_p > head_price) and (rs_p > head_price)
                if not (regular or inverse):
                    continue
                if not _level_close(ls_p, rs_p, tol_pct * 1.5):
                    continue
                nl1_candidates = [ti for ti in troughs.tolist() if lsh < ti < head_idx]
                nl2_candidates = [ti for ti in troughs.tolist() if head_idx < ti < rsh]
                if not nl1_candidates or not nl2_candidates:
                    continue
                nl1 = int(nl1_candidates[-1]); nl2 = int(nl2_candidates[0])
                slope, intercept, r2 = _fit_line(np.array([nl1, nl2], dtype=float), np.array([c[nl1], c[nl2]], dtype=float))
                left_span = head_idx - lsh; right_span = rsh - head_idx
                span_ratio = left_span / float(max(1, right_span))
                if not (0.5 <= span_ratio <= 2.0):
                    continue
                sh_avg = (ls_p + rs_p) / 2.0
                head_prom = (head_price - sh_avg) / abs(sh_avg) * 100.0 if sh_avg != 0 else 0.0
                if regular and head_prom < max(1.0, tol_pct):
                    continue
                if inverse and head_prom > -max(1.0, tol_pct):
                    continue
                look = int(max(1, int(getattr(cfg, 'breakout_lookahead', 8))))
                status = 'forming'
                name = 'Head and Shoulders' if regular else 'Inverse Head and Shoulders'
                broke = False
                end_i = int(rsh)
                for k in range(1, look + 1):
                    i = rsh + k
                    if i >= n:
                        break
                    neck_i = slope * i + intercept
                    px = float(c[i])
                    if regular and px < neck_i:
                        status = 'completed'; broke = True; end_i = int(i); break
                    if inverse and px > neck_i:
                        status = 'completed'; broke = True; end_i = int(i); break
                sym_conf = max(0.0, 1.0 - abs(span_ratio - 1.0))
                sh_sim_conf = max(0.0, 1.0 - (abs(ls_p - rs_p) / max(1e-9, abs(sh_avg))))
                neck_penalty = max(0.0, 1.0 - min(1.0, abs(slope) / max(1e-6, cfg.max_flat_slope * 5.0)))
                prom_conf = min(1.0, abs(head_prom) / (tol_pct * 2.0))
                base_conf = 0.25 * sym_conf + 0.35 * sh_sim_conf + 0.2 * neck_penalty + 0.2 * prom_conf
                if broke:
                    base_conf = min(1.0, base_conf + 0.1)
                details = {
                    'left_shoulder': float(ls_p),
                    'right_shoulder': float(rs_p),
                    'head': float(head_price),
                    'neck_slope': float(slope),
                    'neck_intercept': float(intercept),
                    'neck_r2': float(r2),
                }
                out.append(ClassicPatternResult(
                    name=name,
                    status=status,
                    confidence=float(base_conf),
                    start_index=int(lsh),
                    end_index=end_i,
                    start_time=float(t[int(lsh)]) if t.size else None,
                    end_time=float(t[int(end_i)]) if t.size else None,
                    details=details,
                ))
            except Exception:
                continue
        return out

    results.extend(_head_shoulders())

    # Flags and Pennants
    def _flags_pennants():
        out: List[ClassicPatternResult] = []
        # Identify a recent impulse (pole)
        max_len = min(cfg.max_consolidation_bars, n // 3)
        if max_len < 10:
            return out
        pole_len = max(10, max_len // 2)
        ret = (c[-1] - c[-1 - pole_len]) / max(1e-9, c[-1 - pole_len]) * 100.0
        if abs(ret) < cfg.min_pole_return_pct:
            return out
        window = cfg.max_consolidation_bars
        seg = c[-window:]
        idx0 = n - window
        peaks2, troughs2 = _detect_pivots_close(seg, cfg)
        if peaks2.size < 2 or troughs2.size < 2:
            return out
        # build local arrays for consolidation region
        sh, bh, r2h, sl, bl, r2l, top, bot = _fit_lines_and_arrays(peaks2, troughs2, seg, seg.size)
        dist_recent = float(np.mean((top - bot)[-max(5, seg.size//4):]))
        dist_past = float(np.mean((top - bot)[:max(5, seg.size//4)]))
        converging = dist_recent < dist_past
        parallel = abs(sh - sl) <= max(1e-4, 0.2 * max(abs(sh), abs(sl), cfg.max_flat_slope))
        name = None
        if converging:
            name = "Pennants"
        elif parallel:
            name = "Flag"
        if name:
            conf = _conf(4, min(r2h, r2l), 1.0)
            titled = ("Bull " + name) if ret > 0 else ("Bear " + name)
            base = _result(
                titled,
                "forming",
                conf,
                int(idx0 + (peaks2[0] if peaks2.size else 0)),
                n - 1,
                t,
                {"pole_return_pct": float(ret), "top_slope": float(sh), "bottom_slope": float(sl)},
            )
            out.append(base)
            # Generic names for matching
            out.append(_alias(base, name, 0.95))
            out.append(_alias(base, "Continuation Pattern", 0.9))
        return out

    results.extend(_flags_pennants())

    # Cup and Handle (approximate)
    def _cup_handle():
        out: List[ClassicPatternResult] = []
        W = min(300, n)
        if W < 120:
            return out
        seg = c[-W:]
        i_min = int(np.argmin(seg))
        i_max_left = int(np.argmax(seg[:i_min])) if i_min > 5 else 0
        i_max_right = int(np.argmax(seg[i_min:])) + i_min if i_min < W - 5 else W - 1
        left = seg[i_max_left]; bottom = seg[i_min]; right = seg[i_max_right]
        if left <= 0 or right <= 0:
            return out
        near_equal_rim = _level_close(left, right, cfg.same_level_tol_pct)
        depth_pct = (left - bottom) / left * 100.0 if left != 0 else 0.0
        # Handle: small pullback after right rim (last 20% segment)
        tail = seg[int(W*0.8):]
        handle_pullback = float(np.max(tail) - tail[-1]) / max(1e-9, np.max(tail)) * 100.0 if tail.size > 2 else 0.0
        if near_equal_rim and depth_pct > 2.0:
            conf = min(1.0, 0.6 + 0.4 * (depth_pct / 20.0))
            out.append(_result(
                "Cup and Handle",
                "forming",
                conf,
                int(n - W + i_max_left),
                int(n - W + i_max_right),
                t,
                {
                    "left_rim": float(left),
                    "bottom": float(bottom),
                    "right_rim": float(right),
                    "handle_pullback_pct": float(handle_pullback),
                },
            ))
        return out

    results.extend(_cup_handle())

    # Broadening Formation: diverging highs and lows
    def _broadening():
        out: List[ClassicPatternResult] = []
        k = min(10, peaks.size, troughs.size)
        if k < 4:
            return out
        ih = peaks[-k:]; il = troughs[-k:]
        sh, bh, r2h = _fit_line(ih.astype(float), c[ih])
        sl, bl, r2l = _fit_line(il.astype(float), c[il])
        diverging = (sh > cfg.max_flat_slope and sl < -cfg.max_flat_slope)
        if diverging:
            conf = _conf(4, min(r2h, r2l), 1.0)
            out.append(_result(
                "Broadening Formation",
                "forming",
                conf,
                int(min(ih[0], il[0])),
                int(max(ih[-1], il[-1])),
                t,
                {"top_slope": float(sh), "bottom_slope": float(sl)},
            ))
        return out

    results.extend(_broadening())

    # Diamonds (approximate): expansion then contraction
    def _diamond():
        out: List[ClassicPatternResult] = []
        W = min(240, n)
        if W < 120:
            return out
        seg = c[-W:]
        half = W // 2
        left = seg[:half]; right = seg[half:]
        # Measure width via rolling (max-min)
        def width(a: np.ndarray) -> float:
            return float(np.max(a) - np.min(a))
        expanding = width(left[:half//2]) < width(left[half//2:])
        contracting = width(right[:half//2]) > width(right[half//2:])
        if expanding and contracting:
            # Check for continuation context (prior pole)
            pole = min(60, max(10, W // 3))
            if n > 2 * pole + 1:
                ret = (c[-1 - pole] - c[-1 - 2*pole]) / max(1e-9, c[-1 - 2*pole]) * 100.0
            else:
                ret = 0.0
            name = "Continuation Diamond" if abs(ret) >= 2.0 else "Diamond"
            out.append(_result(
                name,
                "forming",
                0.6,
                int(n - W),
                int(n - 1),
                t,
                {"prior_pole_return_pct": float(ret)},
            ))
        return out

    results.extend(_diamond())

    # Map to user-requested aliases
    # Many patterns already covered: Trend Line/Channel, Triangles, Flags, Pennants, Wedges,
    # Head and Shoulders, Rectangle, Double/Triple Tops/Bottoms, Cup and Handle, Broadening, Diamond.
    # Continuation Diamond/Continuation Pattern are subsumed by Diamond/Flags/Pennants contexts.

    # Normalize statuses: mark older patterns as 'completed' unless very recent
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

    # Sort results by end_index (recency) then confidence
    results.sort(key=lambda r: (r.end_index, r.confidence), reverse=True)
    return results
