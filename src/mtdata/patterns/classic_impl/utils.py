from __future__ import annotations
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from tslearn.metrics import dtw as _ts_dtw

from ...utils.utils import to_float_np
from .config import ClassicDetectorConfig, ClassicPatternResult


def _level_close(a: float, b: float, tol_pct: float) -> bool:
    if a == 0 or b == 0:
        return abs(a - b) <= 1e-12
    return abs((a - b) / ((abs(a) + abs(b)) / 2.0)) * 100.0 <= tol_pct


@lru_cache(maxsize=1)
def _get_ransac_regressor_cls():
    from sklearn.linear_model import RANSACRegressor  # type: ignore
    return RANSACRegressor


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
    """Optionally fit a robust line via RANSAC; fallback to ordinary fit."""
    if not cfg.use_robust_fit:
        return _fit_line(x, y)
    X = x.reshape(-1, 1).astype(float)
    yv = y.astype(float)
    if X.shape[0] < max(2, int(cfg.ransac_min_samples)):
        return _fit_line(x, y)
    med = float(np.median(np.abs(yv))) if yv.size else 1.0
    resid = max(1e-9, float(cfg.ransac_residual_pct) * max(1.0, med))
    try:
        ransac_cls = _get_ransac_regressor_cls()
    except ImportError:
        return _fit_line(x, y)
    try:
        model = ransac_cls(
            min_samples=max(2, int(cfg.ransac_min_samples)),
            max_trials=max(10, int(cfg.ransac_max_trials)),
            residual_threshold=resid,
            random_state=0,
        )
        model.fit(X, yv)
        est = model.estimator_
        slope = float(getattr(est, 'coef_', [0.0])[0])
        intercept = float(getattr(est, 'intercept_', 0.0))
        y_hat = slope * x + intercept
        ss_res = float(np.sum((yv - y_hat) ** 2))
        ss_tot = float(np.sum((yv - yv.mean()) ** 2)) if yv.size else 0.0
        r2 = 0.0 if ss_tot == 0 else max(0.0, 1.0 - ss_res / ss_tot)
        return slope, intercept, r2
    except (AttributeError, TypeError, ValueError, RuntimeError):
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
    except (TypeError, ValueError, RuntimeError):
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
    except (TypeError, ValueError):
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
    except ValueError:
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
    except (TypeError, ValueError):
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
    fit_func = _fit_line_robust if bool(cfg.use_robust_fit) else _fit_line
    if fit_func is _fit_line_robust:
        sh, bh, r2h = fit_func(ih.astype(float), c[ih], cfg)
        sl, bl, r2l = fit_func(il.astype(float), c[il], cfg)
    else:
        sh, bh, r2h = fit_func(ih.astype(float), c[ih])
        sl, bl, r2l = fit_func(il.astype(float), c[il])
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


def _is_converging(
    upper: np.ndarray,
    lower: np.ndarray,
    k: int,
    n: int,
    cfg: ClassicDetectorConfig,
) -> bool:
    """Heuristic to detect converging lines over recent window vs past window."""
    span = upper - lower
    last = max(5, int(k))
    recent = float(np.mean(span[-last:])) if span.size >= last else float(np.mean(span))
    past_win = max(20, 2 * int(k))
    prev_win = span[-past_win:-last] if n > past_win else span[:max(1, span.size // 2)]
    past = float(np.mean(prev_win)) if prev_win.size > 0 else recent * float(cfg.convergence_fallback_scale)
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
    last_dir: Optional[str] = None
    last_idx: Optional[int] = None
    for i in range(start, n):
        px = float(close[i])
        if upper is not None and i < int(upper.size):
            up = float(upper[i])
            if np.isfinite(up) and px > (up + tol_abs):
                last_dir = "up"
                last_idx = int(i)
        if lower is not None and i < int(lower.size):
            lo = float(lower[i])
            if np.isfinite(lo) and px < (lo - tol_abs):
                last_dir = "down"
                last_idx = int(i)
    return last_dir, last_idx


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
            except (TypeError, ValueError):
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
            except (TypeError, ValueError):
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

def _conf(touches: int, r2: float, geom_ok: float, cfg: ClassicDetectorConfig) -> float:
    a = min(1.0, touches / max(1, cfg.min_touches)) * cfg.touch_weight
    b = max(0.0, min(1.0, r2)) * cfg.r2_weight
    g = max(0.0, min(1.0, geom_ok)) * cfg.geometry_weight
    return float(min(1.0, a + b + g))
