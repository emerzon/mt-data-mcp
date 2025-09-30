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

    # Build single-series training dataframe using common helper
    from ..common import _create_training_dataframes
    Y_df, X_df, Xf_df = _create_training_dataframes(series, fh, exog_used, exog_future)

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
            # RangeIndex timeline -> use integer frequency to satisfy StatsForecast
            sf = _StatsForecast(models=[model], freq=1)
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
        
        # Use common forecast value extraction
        from ..common import _extract_forecast_values
        f_vals = _extract_forecast_values(Yf, fh, f"StatsForecast {method_l}")
        
        return f_vals, params_used
    except Exception as ex:
        raise RuntimeError(f"StatsForecast {method_l} error: {ex}")

