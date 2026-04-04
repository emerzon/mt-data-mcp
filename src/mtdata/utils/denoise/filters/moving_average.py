"""Moving average filters: EMA, SMA, median."""
from typing import Any, Dict
import pandas as pd
import numpy as np
from ..base import register_filter, _series_like


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
            y2 = pd.Series(y[::-1]).ewm(alpha=float(alpha), adjust=False).mean().values[::-1]
        else:
            y2 = pd.Series(y[::-1]).ewm(span=int(span), adjust=False).mean().values[::-1]
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
