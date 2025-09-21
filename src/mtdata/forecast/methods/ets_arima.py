from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import warnings
import numpy as np

try:
    from statsmodels.tsa.holtwinters import SimpleExpSmoothing as _SES, ExponentialSmoothing as _ETS  # type: ignore
    _SM_ETS_AVAILABLE = True
except Exception:
    _SM_ETS_AVAILABLE = False

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX as _SARIMAX  # type: ignore
    _SM_SARIMAX_AVAILABLE = True
except Exception:
    _SM_SARIMAX_AVAILABLE = False


def forecast_ses(series: np.ndarray, fh: int, alpha: Optional[float] = None) -> Tuple[np.ndarray, Dict[str, Any], Optional[np.ndarray]]:
    if not _SM_ETS_AVAILABLE:
        raise RuntimeError("SES requires statsmodels")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if alpha is None:
            res = _SES(series, initialization_method='heuristic').fit(optimized=True)
        else:
            res = _SES(series, initialization_method='heuristic').fit(smoothing_level=float(alpha), optimized=False)
    f_vals = np.asarray(res.forecast(int(fh)), dtype=float)
    try:
        fitted = np.asarray(res.fittedvalues, dtype=float)
    except Exception:
        fitted = None
    # Try to recover effective alpha used
    alpha_used = None
    try:
        par = getattr(res, 'params', None)
        if par is not None and hasattr(par, 'get'):
            alpha_used = par.get('smoothing_level', alpha)
    except Exception:
        alpha_used = alpha
    return f_vals, {"alpha": alpha_used}, fitted


def forecast_holt(series: np.ndarray, fh: int, damped: bool = True) -> Tuple[np.ndarray, Dict[str, Any], Optional[np.ndarray]]:
    if not _SM_ETS_AVAILABLE:
        raise RuntimeError("Holt requires statsmodels")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = _ETS(series, trend='add', damped_trend=bool(damped), initialization_method='heuristic')
        res = model.fit(optimized=True)
    f_vals = np.asarray(res.forecast(int(fh)), dtype=float)
    try:
        fitted = np.asarray(res.fittedvalues, dtype=float)
    except Exception:
        fitted = None
    return f_vals, {"damped": bool(damped)}, fitted


def forecast_holt_winters(series: np.ndarray, fh: int, m: int, seasonal: str = 'add') -> Tuple[np.ndarray, Dict[str, Any], Optional[np.ndarray]]:
    if not _SM_ETS_AVAILABLE:
        raise RuntimeError("Holt-Winters requires statsmodels")
    if m is None or int(m) <= 0:
        raise ValueError("Holt-Winters requires positive seasonality")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = _ETS(series, trend='add', seasonal=('add' if seasonal == 'add' else 'mul'), seasonal_periods=int(m), initialization_method='heuristic')
        res = model.fit(optimized=True)
    f_vals = np.asarray(res.forecast(int(fh)), dtype=float)
    try:
        fitted = np.asarray(res.fittedvalues, dtype=float)
    except Exception:
        fitted = None
    return f_vals, {"seasonal": 'add' if seasonal == 'add' else 'mul', "m": int(m)}, fitted


def forecast_sarimax(
    series: np.ndarray,
    fh: int,
    order: Tuple[int, int, int],
    seasonal_order: Tuple[int, int, int, int] = (0, 0, 0, 0),
    trend: str = 'c',
    exog_used: Optional[np.ndarray] = None,
    exog_future: Optional[np.ndarray] = None,
    ci_alpha: Optional[float] = 0.05,
) -> Tuple[np.ndarray, Dict[str, Any], Optional[Tuple[np.ndarray, np.ndarray]]]:
    if not _SM_SARIMAX_AVAILABLE:
        raise RuntimeError("SARIMAX requires statsmodels")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = _SARIMAX(
            series.astype(float),
            order=order,
            seasonal_order=seasonal_order,
            trend=str(trend),
            enforce_stationarity=True,
            enforce_invertibility=True,
            exog=exog_used if exog_used is not None else None,
        )
        res = model.fit(method='lbfgs', disp=False, maxiter=100)
        if exog_future is not None:
            pred = res.get_forecast(steps=int(fh), exog=exog_future)
        else:
            pred = res.get_forecast(steps=int(fh))
    pm = pred.predicted_mean
    if hasattr(pm, 'to_numpy'):
        f_vals = pm.to_numpy()
    else:
        f_vals = np.asarray(pm, dtype=float)
    ci = None
    try:
        _alpha = float(ci_alpha) if ci_alpha is not None else 0.05
        ci_df = pred.conf_int(alpha=_alpha)
        if hasattr(ci_df, 'iloc') and hasattr(ci_df.iloc[:, 0], 'to_numpy'):
            ci = (ci_df.iloc[:, 0].to_numpy(), ci_df.iloc[:, 1].to_numpy())
        else:
            ci_arr = np.asarray(ci_df)
            if ci_arr.ndim == 2 and ci_arr.shape[1] >= 2:
                ci = (ci_arr[:, 0], ci_arr[:, 1])
    except Exception:
        ci = None
    params_used: Dict[str, Any] = {"order": tuple(order), "seasonal_order": tuple(seasonal_order), "trend": str(trend)}
    if exog_used is not None:
        params_used["exog"] = {"n_features": int(exog_used.shape[1])}
    return f_vals.astype(float, copy=False), params_used, ci

