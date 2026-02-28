from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from ..utils.utils import to_float_np
from .common import PatternResultBase
from scipy.signal import find_peaks
from tslearn.metrics import dtw as _ts_dtw


@dataclass
class ClassicDetectorConfig:
    # General
    max_bars: int = 1500
    # Pivot/zigzag parameters
    min_prominence_pct: float = 0.5  # peak/trough prominence in percent of price
    min_distance: int = 5            # minimum distance between pivots (bars)
    pivot_use_hl: bool = True        # use high/low (when available) for pivot extraction
    pivot_use_atr_adaptive_prominence: bool = True
    pivot_use_atr_adaptive_distance: bool = True
    pivot_atr_period: int = 14
    pivot_atr_prominence_mult: float = 1.0
    pivot_atr_distance_mult: float = 0.2
    pivot_max_distance_scale: float = 3.0
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
    use_robust_fit: bool = True     # use RANSAC for line fits when available
    ransac_residual_pct: float = 0.15  # residual threshold as fraction of median price
    ransac_min_samples: int = 2
    ransac_max_trials: int = 50
    use_dtw_check: bool = True      # optional DTW shape confirmation for select patterns
    dtw_paa_len: int = 80            # PAA downsampling length for DTW
    dtw_max_dist: float = 0.6        # acceptance threshold after z-norm
    # Output/completion controls
    include_aliases: bool = False    # include generic aliases like "Trend Line"/"Trend Channel"
    completion_confirm_bars: int = 2 # touches needed near the right edge to mark completed
    completion_lookback_bars: int = 5  # lookback window for completion confirmation
    auto_complete_stale_forming: bool = False  # backward-compat aging of old forming patterns
    include_lifecycle_metadata: bool = True
    # Optional confidence calibration map:
    # { "default": {"0.40": 0.35, "0.70": 0.62, "0.90": 0.82},
    #   "head and shoulders": {"0.50": 0.45, "0.80": 0.76} }
    calibrate_confidence: bool = False
    confidence_calibration_map: Dict[str, Any] = field(default_factory=dict)
    confidence_calibration_blend: float = 1.0


@dataclass
class ClassicPatternResult(PatternResultBase):
    name: str
    status: str  # "completed" | "forming"
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
        return float(_ts_dtw(a.astype(float), b.astype(float)))
    except Exception:
        return float('inf')


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


def _compute_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    h = np.asarray(high, dtype=float)
    l = np.asarray(low, dtype=float)
    c = np.asarray(close, dtype=float)
    n = min(h.size, l.size, c.size)
    if n <= 0:
        return np.asarray([], dtype=float)
    h = h[:n]
    l = l[:n]
    c = c[:n]
    prev_c = np.concatenate(([c[0]], c[:-1]))
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    win = max(2, int(period))
    try:
        atr = pd.Series(tr).rolling(win, min_periods=max(2, win // 2)).mean().to_numpy(dtype=float)
    except Exception:
        atr = tr.astype(float)
    return atr


def _pivot_thresholds(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    cfg: ClassicDetectorConfig,
) -> Tuple[float, int]:
    x = np.asarray(close, dtype=float)
    base = float(np.median(x)) if np.isfinite(np.median(x)) and np.median(x) > 1e-9 else float(np.mean(x))
    prom_abs = abs(base) * (cfg.min_prominence_pct / 100.0)
    min_dist = max(2, int(cfg.min_distance))

    if bool(cfg.pivot_use_atr_adaptive_prominence) or bool(cfg.pivot_use_atr_adaptive_distance):
        atr = _compute_atr(high, low, x, int(cfg.pivot_atr_period))
        finite = atr[np.isfinite(atr)]
        if finite.size > 0:
            atr_med = float(np.median(finite))
            if bool(cfg.pivot_use_atr_adaptive_prominence):
                prom_abs = max(prom_abs, float(cfg.pivot_atr_prominence_mult) * atr_med)
            if bool(cfg.pivot_use_atr_adaptive_distance) and base > 1e-12:
                atr_pct = abs(atr_med / base) * 100.0
                scale = 1.0 + max(0.0, float(cfg.pivot_atr_distance_mult)) * atr_pct
                scale = min(float(max(1.0, cfg.pivot_max_distance_scale)), max(1.0, scale))
                min_dist = max(2, int(round(float(cfg.min_distance) * scale)))
    return float(max(1e-12, prom_abs)), int(min_dist)


def _detect_pivots_close(
    close: np.ndarray,
    cfg: ClassicDetectorConfig,
    high: Optional[np.ndarray] = None,
    low: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return arrays of peak and trough indices using close or optional high/low arrays."""
    x = np.asarray(close, dtype=float)
    if x.size < max(50, cfg.min_distance * 3):
        return np.array([], dtype=int), np.array([], dtype=int)

    hi = np.asarray(high, dtype=float) if high is not None else x
    lo = np.asarray(low, dtype=float) if low is not None else x
    if hi.size != x.size:
        hi = x
    if lo.size != x.size:
        lo = x
    if not np.isfinite(hi).all():
        hi = x
    if not np.isfinite(lo).all():
        lo = x

    prom_abs, min_dist = _pivot_thresholds(x, hi, lo, cfg)
    src_hi = hi if bool(cfg.pivot_use_hl) else x
    src_lo = lo if bool(cfg.pivot_use_hl) else x
    try:
        peaks, _ = find_peaks(src_hi, prominence=prom_abs, distance=min_dist)
        troughs, _ = find_peaks(-src_lo, prominence=prom_abs, distance=min_dist)
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


def _fit_lines_and_arrays(
    ih: np.ndarray,
    il: np.ndarray,
    c: np.ndarray,
    n: int,
    cfg: ClassicDetectorConfig,
):
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


def _count_recent_touches(
    series: np.ndarray,
    close: np.ndarray,
    tol_abs: float,
    lookback_bars: int,
) -> int:
    if series.size == 0 or close.size == 0:
        return 0
    n = min(series.size, close.size)
    lb = max(1, int(lookback_bars))
    start = max(0, n - lb)
    recent_s = series[start:n]
    recent_c = close[start:n]
    if recent_s.size == 0 or recent_c.size == 0:
        return 0
    return int(np.sum(np.abs(recent_s - recent_c) <= tol_abs))


def _is_converging(upper: np.ndarray, lower: np.ndarray, k: int, n: int) -> bool:
    """Heuristic to detect converging lines over recent window vs past window."""
    span = upper - lower
    last = max(5, int(k))
    recent = float(np.mean(span[-last:])) if span.size >= last else float(np.mean(span))
    past_win = max(20, 2 * int(k))
    prev_win = span[-past_win:-last] if n > past_win else span[:max(1, span.size // 2)]
    past = float(np.mean(prev_win)) if prev_win.size > 0 else recent * 1.2
    return bool(recent < past)


def _find_recent_breakout(
    close: np.ndarray,
    *,
    upper: Optional[np.ndarray] = None,
    lower: Optional[np.ndarray] = None,
    tol_abs: float = 0.0,
    lookback_bars: int = 8,
) -> Tuple[Optional[str], Optional[int]]:
    n = int(close.size)
    if n <= 0:
        return None, None
    lb = max(1, int(lookback_bars))
    start = max(0, n - lb)
    for i in range(start, n):
        px = float(close[i])
        if upper is not None and i < int(upper.size):
            up = float(upper[i])
            if np.isfinite(up) and px > (up + tol_abs):
                return "up", int(i)
        if lower is not None and i < int(lower.size):
            lo = float(lower[i])
            if np.isfinite(lo) and px < (lo - tol_abs):
                return "down", int(i)
    return None, None


def _find_forward_level_breakout(
    close: np.ndarray,
    start_idx: int,
    level: float,
    direction: str,
    lookahead: int,
    tol_abs: float,
) -> Optional[int]:
    n = int(close.size)
    if n <= 0:
        return None
    lo = max(0, int(start_idx) + 1)
    hi = min(n, lo + max(1, int(lookahead)))
    for i in range(lo, hi):
        px = float(close[i])
        if direction == "up" and px > (float(level) + tol_abs):
            return int(i)
        if direction == "down" and px < (float(level) - tol_abs):
            return int(i)
    return None


def _collect_calibration_points(cal_map: Any, name: str) -> List[Tuple[float, float]]:
    points_src: Any = cal_map
    if isinstance(cal_map, dict):
        key = str(name or "").strip().lower()
        if key in cal_map and isinstance(cal_map.get(key), dict):
            points_src = cal_map.get(key)
        elif "default" in cal_map and isinstance(cal_map.get("default"), dict):
            points_src = cal_map.get("default")
    points: List[Tuple[float, float]] = []
    if isinstance(points_src, dict):
        for k, v in points_src.items():
            try:
                x = float(k)
                y = float(v)
            except Exception:
                continue
            if np.isfinite(x) and np.isfinite(y):
                points.append((max(0.0, min(1.0, x)), max(0.0, min(1.0, y))))
    elif isinstance(points_src, list):
        for item in points_src:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            try:
                x = float(item[0])
                y = float(item[1])
            except Exception:
                continue
            if np.isfinite(x) and np.isfinite(y):
                points.append((max(0.0, min(1.0, x)), max(0.0, min(1.0, y))))
    points.sort(key=lambda p: p[0])
    return points


def _calibrate_confidence(raw: float, name: str, cfg: ClassicDetectorConfig) -> float:
    conf = float(max(0.0, min(1.0, raw)))
    if not bool(getattr(cfg, "calibrate_confidence", False)):
        return conf
    points = _collect_calibration_points(getattr(cfg, "confidence_calibration_map", {}), name)
    if len(points) < 2:
        return conf

    if conf <= points[0][0]:
        cal = points[0][1]
    elif conf >= points[-1][0]:
        cal = points[-1][1]
    else:
        cal = conf
        for i in range(1, len(points)):
            x0, y0 = points[i - 1]
            x1, y1 = points[i]
            if x0 <= conf <= x1:
                frac = 0.0 if abs(x1 - x0) <= 1e-12 else (conf - x0) / (x1 - x0)
                cal = y0 + frac * (y1 - y0)
                break
    blend = float(max(0.0, min(1.0, getattr(cfg, "confidence_calibration_blend", 1.0))))
    out = (1.0 - blend) * conf + blend * float(cal)
    return float(max(0.0, min(1.0, out)))


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
    confirm_needed = max(1, int(cfg.completion_confirm_bars))
    confirm_lookback = max(confirm_needed, int(cfg.completion_lookback_bars))
    breakout_look = max(confirm_lookback, int(max(1, cfg.breakout_lookahead)))

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
            # Completion heuristic: require repeated recent touches near the fitted line.
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
            # Optional generic alias for compatibility.
            if tl_dir != 'Horizontal' and bool(cfg.include_aliases):
                results.append(_alias(base_item, "Trend Line", 0.95))

    # Channels: parallel lines from highs and lows
    def _try_channel():
        ch_results: List[ClassicPatternResult] = []
        if peaks.size < 3 or troughs.size < 3:
            return ch_results
        # Fit lines on last K highs and lows
        k = min(8, peaks.size, troughs.size)
        ih = peaks[-k:]; il = troughs[-k:]
        sh, bh, r2h, sl, bl, r2l, upper, lower = _fit_lines_and_arrays(ih, il, c, n, cfg)
        # Slopes near equal for channels; width relatively stable
        slope_diff = abs(sh - sl)
        approx_parallel = slope_diff <= max(1e-4, 0.15 * max(abs(sh), abs(sl), cfg.max_flat_slope))
        # Avoid double-reporting mildly converging structures as both channels and triangles.
        converging = _is_converging(upper, lower, k, n)
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
            conf = _conf(touches, min(r2h, r2l), geom_ok)
            status = "forming"
            recent_i = n - 1
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
            # Optional generic alias for compatibility.
            if bool(cfg.include_aliases):
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
            top_line = np.full(n, top, dtype=float)
            bot_line = np.full(n, bot, dtype=float)
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

    results.extend(_try_rectangle())

    # Triangles (Ascending, Descending, Symmetrical)
    def _try_triangles():
        out: List[ClassicPatternResult] = []
        k = min(8, peaks.size, troughs.size)
        if k < 4:
            return out
        ih = peaks[-k:]; il = troughs[-k:]
        sh, bh, r2h, sl, bl, r2l, top, bot = _fit_lines_and_arrays(ih, il, c, n, cfg)
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

    results.extend(_try_triangles())

    # Wedges (Rising/Falling): converging with both sloped same sign
    def _try_wedges():
        out: List[ClassicPatternResult] = []
        k = min(8, peaks.size, troughs.size)
        if k < 4:
            return out
        ih = peaks[-k:]; il = troughs[-k:]
        sh, bh, r2h, sl, bl, r2l, top, bot = _fit_lines_and_arrays(ih, il, c, n, cfg)
        converging = _is_converging(top, bot, k, n)
        same_sign = (sh > 0 and sl > 0) or (sh < 0 and sl < 0)
        tol_abs = _tol_abs_from_close(c, cfg.same_level_tol_pct)
        touches = _count_touches(top, bot, ih, il, c, tol_abs)
        if converging and same_sign and touches >= cfg.min_channel_touches - 1:
            name = "Rising Wedge" if sh > 0 and sl > 0 else "Falling Wedge"
            conf = _conf(touches, min(r2h, r2l), 1.0)
            status = "forming"
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

    results.extend(_try_wedges())

    # Double/Triple Tops & Bottoms
    def _tops_bottoms():
        out: List[ClassicPatternResult] = []
        def group_levels(idxs: np.ndarray, name_top: str, name_triple: str, kind: str):
            if idxs.size < 2:
                return
            vals = c[idxs]
            tol_abs = _tol_abs_from_close(c, cfg.same_level_tol_pct)
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
                    level = float(np.median(vals[cluster]))
                    neckline = float(np.min(c[start_i:end_i + 1])) if kind == "top" else float(np.max(c[start_i:end_i + 1]))
                    direction = "down" if kind == "top" else "up"
                    break_i = _find_forward_level_breakout(c, end_i, neckline, direction, breakout_look, tol_abs)
                    if break_i is not None:
                        status = "completed"
                    out.append(ClassicPatternResult(
                        name=name,
                        status=status,
                        confidence=min(1.0, (0.5 + 0.1 * (len(cluster) - 2 + 2)) + (0.08 if break_i is not None else 0.0)),
                        start_index=start_i,
                        end_index=int(break_i if break_i is not None else end_i),
                        start_time=float(t[start_i]) if t.size else None,
                        end_time=float(t[int(break_i if break_i is not None else end_i)]) if t.size else None,
                        details={
                            "level": level,
                            "touches": int(len(cluster)),
                            "neckline": neckline,
                            "breakout_direction": direction if break_i is not None else None,
                            "breakout_index": int(break_i) if break_i is not None else None,
                        },
                    ))
        group_levels(peaks[-10:], "Double Top", "Triple Top", "top")
        group_levels(troughs[-10:], "Double Bottom", "Triple Bottom", "bottom")
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
                if getattr(cfg, 'use_robust_fit', False):
                    slope, intercept, r2 = _fit_line_robust(
                        np.array([nl1, nl2], dtype=float), np.array([c[nl1], c[nl2]], dtype=float), cfg
                    )
                else:
                    slope, intercept, r2 = _fit_line(
                        np.array([nl1, nl2], dtype=float), np.array([c[nl1], c[nl2]], dtype=float)
                    )
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
                # Optional DTW shape confirmation
                if getattr(cfg, 'use_dtw_check', False):
                    seg_start = max(0, int(lsh))
                    seg_end = int(end_i if broke else rsh)
                    seg = c[seg_start: seg_end + 1].astype(float)
                    seg_n = _znorm(_paa(seg, int(getattr(cfg, 'dtw_paa_len', 80))))
                    tpl = _template_hs(len(seg_n), inverse=bool(inverse))
                    dist = _dtw_distance(seg_n, tpl)
                    maxd = float(getattr(cfg, 'dtw_max_dist', 0.6))
                    if not np.isfinite(dist):
                        dist = maxd * 10.0
                    if dist > (2.0 * maxd):
                        # too dissimilar; skip candidate
                        continue
                    elif dist > maxd:
                        base_conf *= 0.7
                    else:
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
        seg_h = h[-window:] if h.size >= window else seg
        seg_l = l[-window:] if l.size >= window else seg
        idx0 = n - window
        try:
            peaks2, troughs2 = _detect_pivots_close(seg, cfg, seg_h, seg_l)
        except TypeError:
            peaks2, troughs2 = _detect_pivots_close(seg, cfg)
        if peaks2.size < 2 or troughs2.size < 2:
            return out
        # build local arrays for consolidation region
        sh, bh, r2h, sl, bl, r2l, top, bot = _fit_lines_and_arrays(peaks2, troughs2, seg, seg.size, cfg)
        dist_recent = float(np.mean((top - bot)[-max(5, seg.size//4):]))
        dist_past = float(np.mean((top - bot)[:max(5, seg.size//4)]))
        converging = dist_recent < dist_past
        parallel = abs(sh - sl) <= max(1e-4, 0.2 * max(abs(sh), abs(sl), cfg.max_flat_slope))
        name = None
        if converging:
            name = "Pennant"
        elif parallel:
            name = "Flag"
        if name:
            conf = _conf(4, min(r2h, r2l), 1.0)
            titled = ("Bull " + name) if ret > 0 else ("Bear " + name)
            status = "forming"
            tol_abs = _tol_abs_from_close(seg, cfg.same_level_tol_pct)
            bdir, bidx_local = _find_recent_breakout(seg, upper=top, lower=bot, tol_abs=tol_abs, lookback_bars=breakout_look)
            expected = "up" if ret > 0 else "down"
            if bdir == expected and bidx_local is not None:
                status = "completed"
                conf = min(1.0, conf + 0.08)
            base = _result(
                titled,
                status,
                conf,
                int(idx0 + (peaks2[0] if peaks2.size else 0)),
                int(idx0 + bidx_local) if bidx_local is not None else n - 1,
                t,
                {
                    "pole_return_pct": float(ret),
                    "top_slope": float(sh),
                    "bottom_slope": float(sl),
                    "breakout_direction": bdir,
                    "breakout_index": int(idx0 + bidx_local) if bidx_local is not None else None,
                    "breakout_expected": expected,
                },
            )
            out.append(base)
            # Optional generic aliases for compatibility.
            if bool(cfg.include_aliases):
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
            status = "forming"
            rim = float(max(left, right))
            tol_abs = _tol_abs_from_close(c, cfg.same_level_tol_pct)
            break_i = _find_forward_level_breakout(c, int(n - W + i_max_right), rim, "up", breakout_look, tol_abs)
            if break_i is not None:
                status = "completed"
                conf = min(1.0, conf + 0.08)
            out.append(_result(
                "Cup and Handle",
                status,
                conf,
                int(n - W + i_max_left),
                int(break_i if break_i is not None else (n - W + i_max_right)),
                t,
                {
                    "left_rim": float(left),
                    "bottom": float(bottom),
                    "right_rim": float(right),
                    "handle_pullback_pct": float(handle_pullback),
                    "breakout_level": rim,
                    "breakout_index": int(break_i) if break_i is not None else None,
                },
            ))
        return out

    results.extend(_cup_handle())

    # Rounding Bottom/Top (saucer-like)
    def _rounding():
        out: List[ClassicPatternResult] = []
        W = min(220, n)
        if W < 100:
            return out
        seg = c[-W:]
        x = np.linspace(-1.0, 1.0, W)
        try:
            qa, qb, qc = np.polyfit(x, seg.astype(float), 2)
        except Exception:
            return out
        if not (np.isfinite(qa) and np.isfinite(qb) and np.isfinite(qc)):
            return out
        # Vertex in central region.
        if abs(float(qa)) <= 1e-12:
            return out
        xv = -float(qb) / (2.0 * float(qa))
        if not (-0.55 <= xv <= 0.55):
            return out
        edge_n = max(6, W // 10)
        left_edge = float(np.mean(seg[:edge_n]))
        right_edge = float(np.mean(seg[-edge_n:]))
        if not _level_close(left_edge, right_edge, cfg.same_level_tol_pct * 2.0):
            return out

        peak = float(np.max(seg))
        trough = float(np.min(seg))
        amp_pct = abs(peak - trough) / max(1e-9, abs((peak + trough) / 2.0)) * 100.0
        if amp_pct < 2.0:
            return out

        tol_abs = _tol_abs_from_close(c, cfg.same_level_tol_pct)
        if qa > 0:
            name = "Rounding Bottom"
            status = "completed" if float(c[-1]) > (max(left_edge, right_edge) + tol_abs) else "forming"
        else:
            name = "Rounding Top"
            status = "completed" if float(c[-1]) < (min(left_edge, right_edge) - tol_abs) else "forming"
        conf = min(1.0, 0.5 + 0.3 * min(1.0, amp_pct / 12.0))
        out.append(
            _result(
                name,
                status,
                conf,
                int(n - W),
                int(n - 1),
                t,
                {
                    "quad_a": float(qa),
                    "quad_b": float(qb),
                    "vertex_x_norm": float(xv),
                    "left_edge": left_edge,
                    "right_edge": right_edge,
                    "amplitude_pct": float(amp_pct),
                },
            )
        )
        return out

    results.extend(_rounding())

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
            x = np.arange(n, dtype=float)
            top = sh * x + bh
            bot = sl * x + bl
            tol_abs = _tol_abs_from_close(c, cfg.same_level_tol_pct)
            status = "forming"
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
            status = "forming"
            conf = 0.6
            right_hi = float(np.max(right)) if right.size else float(np.max(seg))
            right_lo = float(np.min(right)) if right.size else float(np.min(seg))
            tol_abs = _tol_abs_from_close(c, cfg.same_level_tol_pct)
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

    results.extend(_diamond())

    # Map to user-requested aliases
    # Many patterns already covered: Trend Line/Channel, Triangles, Flags, Pennants, Wedges,
    # Head and Shoulders, Rectangle, Double/Triple Tops/Bottoms, Cup and Handle, Broadening, Diamond.
    # Continuation Diamond/Continuation Pattern are subsumed by Diamond/Flags/Pennants contexts.

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
