from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import warnings
import numpy as np


def forecast_statsforecast(
    *,
    method: str,
    series: np.ndarray,
    fh: int,
    timeframe: str,
    m_eff: int,
    exog_used: Optional[np.ndarray] = None,
    exog_future: Optional[np.ndarray] = None,
    future_times: Optional[list] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """StatsForecast models: sf_autoarima, sf_theta, sf_autoets, sf_seasonalnaive.

    Returns forecast array and params_used.
    """
    try:
        from statsforecast import StatsForecast as _StatsForecast  # type: ignore
        from statsforecast.models import AutoARIMA as _SF_AutoARIMA, Theta as _SF_Theta, AutoETS as _SF_AutoETS, SeasonalNaive as _SF_SeasonalNaive  # type: ignore
        import pandas as _pd
    except Exception as ex:
        raise RuntimeError(f"Failed to import statsforecast: {ex}")

    # Build single-series training dataframe
    # We do not have timestamps here; require caller to align if needed. Accept plain index.
    Y_df = _pd.DataFrame({
        'unique_id': ['ts'] * int(len(series)),
        'ds': _pd.RangeIndex(start=0, stop=int(len(series))),
        'y': series.astype(float),
    })
    X_df = None
    Xf_df = None
    if exog_used is not None and isinstance(exog_used, np.ndarray) and exog_used.size:
        cols = [f'x{i}' for i in range(exog_used.shape[1])]
        X_df = _pd.DataFrame({'unique_id': ['ts'] * int(len(series)), 'ds': _pd.RangeIndex(start=0, stop=int(len(series)))})
        for j, cname in enumerate(cols):
            X_df[cname] = exog_used[:, j]
        if exog_future is not None and isinstance(exog_future, np.ndarray) and exog_future.size:
            Xf_df = _pd.DataFrame({'unique_id': ['ts'] * int(fh), 'ds': _pd.RangeIndex(start=int(len(series)), stop=int(len(series))+int(fh))})
            for j, cname in enumerate(cols):
                Xf_df[cname] = exog_future[:, j]

    # Choose model
    method_l = str(method).lower().strip()
    if method_l == 'sf_autoarima':
        model = _SF_AutoARIMA(season_length=max(1, m_eff or 1))
        params_used: Dict[str, Any] = {"seasonality": m_eff}
    elif method_l == 'sf_theta':
        model = _SF_Theta(season_length=max(1, m_eff or 1))
        params_used = {"seasonality": m_eff}
    elif method_l == 'sf_autoets':
        model = _SF_AutoETS(season_length=max(1, m_eff or 1))
        params_used = {"seasonality": m_eff}
    elif method_l == 'sf_seasonalnaive':
        model = _SF_SeasonalNaive(season_length=max(1, m_eff or 1))
        params_used = {"seasonality": m_eff}
    else:
        raise ValueError(f"Unsupported StatsForecast method: {method}")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sf = _StatsForecast(models=[model], freq='D')  # freq unused for single series index
            if X_df is not None:
                sf.fit(Y_df, X_df=X_df)
            else:
                sf.fit(Y_df)
            if Xf_df is not None:
                Yf = sf.predict(h=int(fh), X_df=Xf_df)
            else:
                Yf = sf.predict(h=int(fh))
        try:
            Yf = Yf[Yf['unique_id'] == 'ts']
        except Exception:
            pass
        pred_col = None
        for c in list(Yf.columns):
            if c not in ('unique_id', 'ds', 'y'):
                pred_col = c
                if c == 'y':
                    break
        if pred_col is None:
            pred_col = 'y' if 'y' in Yf.columns else None
        if pred_col is None:
            raise RuntimeError("StatsForecast prediction columns not found")
        vals = np.asarray(Yf[pred_col].to_numpy(), dtype=float)
        f_vals = vals[:fh] if vals.size >= fh else np.pad(vals, (0, fh - vals.size), mode='edge')
        return f_vals.astype(float, copy=False), params_used
    except Exception as ex:
        raise RuntimeError(f"StatsForecast {method_l} error: {ex}")

