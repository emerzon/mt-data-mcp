"""Polynomial and regression-based filters: Savgol, LOESS, STL."""
import logging
from typing import Any, Dict

import numpy as np
import pandas as pd

_logger = logging.getLogger(__name__)

try:
    from scipy.signal import savgol_filter as _savgol_filter
except Exception:
    _savgol_filter = None  # type: ignore

try:
    from statsmodels.nonparametric.smoothers_lowess import lowess as _lowess
except Exception:
    _lowess = None  # type: ignore

try:
    from statsmodels.tsa.seasonal import STL as _STL
except Exception:
    _STL = None  # type: ignore

from ..base import _series_like, register_filter


def _savgol_causal_1d(x: np.ndarray, window: int, polyorder: int) -> np.ndarray:
    """Trailing-window Savitzky–Golay: fit on x[t-w+1:t+1], evaluate at t."""
    n = len(x)
    if n == 0:
        return np.asarray(x, dtype=float)
    values = np.asarray(x, dtype=float)
    max_poly = max(0, n - 1)
    poly = max(0, min(int(polyorder), max_poly))
    width = max(poly + 1, int(window))
    y = np.empty(n, dtype=float)
    coeff_cache: Dict[int, np.ndarray] = {}

    def _coeffs(length: int) -> np.ndarray:
        cached = coeff_cache.get(length)
        if cached is not None:
            return cached
        degree = min(poly, length - 1)
        time = np.arange(length, dtype=float)
        design = np.vander(time, N=degree + 1, increasing=True)
        weights = design[-1] @ np.linalg.pinv(design)
        coeff_cache[length] = weights
        return weights

    warmup = min(n, width - 1)
    for index in range(warmup):
        length = index + 1
        if length <= poly:
            y[index] = values[index]
            continue
        y[index] = float(np.dot(_coeffs(length), values[:length]))
    if n >= width:
        y[width - 1 :] = np.lib.stride_tricks.sliding_window_view(values, width) @ _coeffs(width)
    return y


@register_filter('savgol')
def _denoise_savgol_series(
    s: pd.Series,
    x: np.ndarray,
    params: Dict[str, Any],
    causality: str,
) -> pd.Series:
    window = int(params.get('window', 11))
    if window < 3:
        return s
    polyorder = int(params.get('polyorder', 2))
    if causality != 'zero_phase':
        polyorder = max(0, min(polyorder, window - 1))
        return _series_like(s, _savgol_causal_1d(x, window=window, polyorder=polyorder))
    if _savgol_filter is None:
        return s
    if window % 2 == 0:
        window += 1
    if window > len(x):
        _logger.warning(
            "savgol window_length (%d) must not exceed series length (%d)",
            window,
            len(x),
        )
        raise ValueError(
            f"savgol window_length ({window}) must not exceed series length ({len(x)})"
        )
    polyorder = max(0, min(polyorder, window - 1))
    mode = str(params.get('mode', 'interp'))
    try:
        y = _savgol_filter(x, window_length=window, polyorder=polyorder, mode=mode)
    except Exception as exc:
        _logger.warning("savgol_filter failed (window=%d, polyorder=%d): %s", window, polyorder, exc)
        return s
    return _series_like(s, y)


@register_filter('loess')
def _denoise_loess_series(
    s: pd.Series,
    x: np.ndarray,
    params: Dict[str, Any],
    causality: str,
) -> pd.Series:
    del causality
    if _lowess is None:
        return s
    frac = float(params.get('frac', 0.2))
    it = int(params.get('it', 0))
    delta = float(params.get('delta', 0.0))
    exog = np.arange(len(x), dtype=float)
    y = _lowess(x, exog, frac=frac, it=it, delta=delta, return_sorted=False)
    return _series_like(s, y)


@register_filter('stl')
def _denoise_stl_series(
    s: pd.Series,
    x: np.ndarray,
    params: Dict[str, Any],
    causality: str,
) -> pd.Series:
    del causality
    if _STL is None:
        return s
    period = params.get('period')
    if period is None:
        raise ValueError("STL denoise requires a 'period' parameter of at least 2.")
    try:
        period_val = int(period)
    except (TypeError, ValueError) as exc:
        raise ValueError("STL denoise parameter 'period' must be an integer.") from exc
    if period_val < 2 or period_val >= len(x):
        raise ValueError(
            "STL denoise parameter 'period' must be at least 2 and shorter "
            "than the input series."
        )
    seasonal = params.get('seasonal')
    trend = params.get('trend')
    low_pass = params.get('low_pass')
    robust = bool(params.get('robust', False))
    stl_kwargs: Dict[str, Any] = {"period": period_val, "robust": robust}
    if seasonal is not None:
        stl_kwargs["seasonal"] = int(seasonal)
    if trend is not None:
        stl_kwargs["trend"] = int(trend)
    if low_pass is not None:
        stl_kwargs["low_pass"] = int(low_pass)
    stl = _STL(x, **stl_kwargs)
    res = stl.fit()
    component = str(params.get('component', 'trend')).lower().strip()
    if component == 'seasonal':
        y = res.seasonal
    elif component == 'resid':
        y = res.resid
    elif component in ('trend+seasonal', 'trend_seasonal'):
        y = res.trend + res.seasonal
    elif component in ('trend+resid', 'trend_resid'):
        y = res.trend + res.resid
    else:
        y = res.trend
    return _series_like(s, y)
