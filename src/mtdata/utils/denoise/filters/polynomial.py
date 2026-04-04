"""Polynomial and regression-based filters: Savgol, LOESS, STL."""
from typing import Any, Dict
import pandas as pd
import numpy as np
import logging

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

from ..base import register_filter, _series_like


@register_filter('savgol')
def _denoise_savgol_series(
    s: pd.Series,
    x: np.ndarray,
    params: Dict[str, Any],
    causality: str,
) -> pd.Series:
    del causality
    if _savgol_filter is None:
        return s
    window = int(params.get('window', 11))
    if window < 3:
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
    polyorder = int(params.get('polyorder', 2))
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
        return s
    period_val = int(period)
    if period_val < 2 or period_val >= len(x):
        return s
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
