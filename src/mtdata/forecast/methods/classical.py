from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import math
import numpy as np


def forecast_naive(series: np.ndarray, fh: int) -> Tuple[np.ndarray, Dict[str, Any]]:
    last_val = float(series[-1])
    f_vals = np.full(int(fh), last_val, dtype=float)
    return f_vals, {}


def forecast_drift(series: np.ndarray, fh: int, n: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
    if n is None:
        n = int(series.size)
    slope = (float(series[-1]) - float(series[0])) / float(max(1, n - 1))
    f_vals = float(series[-1]) + slope * np.arange(1, int(fh) + 1, dtype=float)
    return f_vals, {"slope": slope}


def forecast_seasonal_naive(series: np.ndarray, fh: int, m: int) -> Tuple[np.ndarray, Dict[str, Any]]:
    if m is None or int(m) <= 0 or series.size < int(m):
        raise ValueError("Insufficient data for seasonal_naive")
    m = int(m)
    last_season = series[-m:]
    reps = int(math.ceil(int(fh) / float(m)))
    f_vals = np.tile(last_season, reps)[: int(fh)]
    return f_vals, {"m": m}


def forecast_theta(series: np.ndarray, fh: int, alpha: float = 0.2) -> Tuple[np.ndarray, Dict[str, Any]]:
    n = int(series.size)
    tt = np.arange(1, n + 1, dtype=float)
    A = np.vstack([np.ones(n), tt]).T
    coef, _, _, _ = np.linalg.lstsq(A, series, rcond=None)
    a, b = float(coef[0]), float(coef[1])
    trend_future = a + b * (tt[-1] + np.arange(1, int(fh) + 1, dtype=float))
    level = float(series[0])
    for v in series[1:]:
        level = float(alpha) * float(v) + (1.0 - float(alpha)) * level
    ses_future = np.full(int(fh), level, dtype=float)
    f_vals = 0.5 * (trend_future + ses_future)
    return f_vals, {"alpha": float(alpha), "trend_slope": b}


def forecast_fourier_ols(series: np.ndarray, fh: int, m: Optional[int], K: Optional[int], trend: bool = True) -> Tuple[np.ndarray, Dict[str, Any]]:
    n = int(series.size)
    m_eff = int(m) if (m is not None and int(m) > 0) else 0
    if K is None:
        K_eff = min(3, max(1, (m_eff // 2) if m_eff else 2))
    else:
        K_eff = int(K)
    tt = np.arange(1, n + 1, dtype=float)
    X_list = [np.ones(n)]
    if trend:
        X_list.append(tt)
    for k in range(1, K_eff + 1):
        w = 2.0 * math.pi * k / float(m_eff if m_eff else max(2, n))
        X_list.append(np.sin(w * tt))
        X_list.append(np.cos(w * tt))
    X = np.vstack(X_list).T
    coef, _, _, _ = np.linalg.lstsq(X, series, rcond=None)
    tt_f = tt[-1] + np.arange(1, int(fh) + 1, dtype=float)
    Xf_list = [np.ones(int(fh))]
    if trend:
        Xf_list.append(tt_f)
    for k in range(1, K_eff + 1):
        w = 2.0 * math.pi * k / float(m_eff if m_eff else max(2, n))
        Xf_list.append(np.sin(w * tt_f))
        Xf_list.append(np.cos(w * tt_f))
    Xf = np.vstack(Xf_list).T
    f_vals = Xf @ coef
    return f_vals.astype(float, copy=False), {"m": m_eff, "K": K_eff, "trend": bool(trend)}

