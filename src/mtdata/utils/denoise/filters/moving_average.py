"""Moving average filters: EMA, SMA, median."""
from typing import Any, Dict

import numpy as np
import pandas as pd

from ..base import _series_like, register_filter


@register_filter('ema')
def _denoise_ema_series(
    s: pd.Series,
    x: np.ndarray,
    params: Dict[str, Any],
    causality: str,
) -> pd.Series:
    alpha = params.get('alpha')
    span = params.get('span', 10)
    if alpha is not None:
        y = pd.Series(x).ewm(alpha=float(alpha), adjust=False).mean().values
    else:
        y = pd.Series(x).ewm(span=int(span), adjust=False).mean().values
    if causality == 'zero_phase':
        if alpha is not None:
            y2 = pd.Series(x[::-1]).ewm(alpha=float(alpha), adjust=False).mean().values[::-1]
        else:
            y2 = pd.Series(x[::-1]).ewm(span=int(span), adjust=False).mean().values[::-1]
        y = 0.5 * (y + y2)
    return _series_like(s, y)


@register_filter('sma')
def _denoise_sma_series(
    s: pd.Series,
    x: np.ndarray,
    params: Dict[str, Any],
    causality: str,
) -> pd.Series:
    window = max(1, int(params.get('window', 10)))
    series = pd.Series(x)
    rolling_kwargs: Dict[str, Any] = {"window": window, "min_periods": 1}
    if causality == 'zero_phase':
        rolling_kwargs["center"] = True
    y = series.rolling(**rolling_kwargs).mean().values
    return _series_like(s, y)


def _kama_1d(x: np.ndarray, window: int, fast: float, slow: float) -> np.ndarray:
    n = len(x)
    y = np.empty(n, dtype=float)
    if n == 0:
        return y
    er_period = max(1, int(window))
    fastest = 2.0 / (float(fast) + 1.0)
    slowest = 2.0 / (float(slow) + 1.0)
    values = np.asarray(x, dtype=float)
    y[: min(n, er_period)] = values[: min(n, er_period)]
    if n <= er_period:
        return y
    for t in range(er_period, n):
        change = abs(values[t] - values[t - er_period])
        volatility = float(np.sum(np.abs(np.diff(values[t - er_period : t + 1]))))
        efficiency = change / volatility if volatility > 0.0 else 0.0
        smoothing = (efficiency * (fastest - slowest) + slowest) ** 2
        y[t] = y[t - 1] + smoothing * (values[t] - y[t - 1])
    return y


@register_filter('kama')
def _denoise_kama_series(
    s: pd.Series,
    x: np.ndarray,
    params: Dict[str, Any],
    causality: str,
) -> pd.Series:
    window = max(1, int(params.get('window', 10)))
    fast = float(params.get('fast', 2))
    slow = float(params.get('slow', 30))
    if slow < fast:
        fast, slow = slow, fast
    y = _kama_1d(x, window=window, fast=fast, slow=slow)
    if causality == 'zero_phase':
        y = 0.5 * (y + _kama_1d(x[::-1], window=window, fast=fast, slow=slow)[::-1])
    return _series_like(s, y)


@register_filter('median')
def _denoise_median_series(
    s: pd.Series,
    x: np.ndarray,
    params: Dict[str, Any],
    causality: str,
) -> pd.Series:
    window = max(1, int(params.get('window', 7)))
    series = pd.Series(x)
    rolling_kwargs: Dict[str, Any] = {"window": window, "min_periods": 1}
    if causality == 'zero_phase':
        rolling_kwargs["center"] = True
    y = series.rolling(**rolling_kwargs).median().values
    return _series_like(s, y)
