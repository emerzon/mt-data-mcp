from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import warnings
import numpy as np


def forecast_mlf_rf(
    *,
    series: np.ndarray,
    fh: int,
    timeframe: str,
    lags: Optional[list] = None,
    rolling_agg: Optional[str] = None,
    exog_used: Optional[np.ndarray] = None,
    exog_future: Optional[np.ndarray] = None,
    future_times: Optional[list] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    try:
        from mlforecast import MLForecast as _MLForecast  # type: ignore
        from sklearn.ensemble import RandomForestRegressor as _RF  # type: ignore
        import pandas as _pd
    except Exception as ex:
        raise RuntimeError(f"Failed to import mlforecast/sklearn: {ex}")

    # Use common DataFrame creation helper
    from ..common import _create_training_dataframes
    Y_df, X_df, Xf_df = _create_training_dataframes(series, fh, exog_used, exog_future)
    
    rf = _RF(n_estimators=200, max_depth=None, random_state=42)
    try:
        mlf = _MLForecast(models=[rf], freq='D')
        if lags:
            mlf = mlf.add_lags(lags)
        if rolling_agg in {'mean', 'min', 'max', 'std'} and lags:
            for w in sorted(set([x for x in lags if x and x > 1])):
                mlf = mlf.add_rolling_windows(rolling_features={rolling_agg: [w]})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mlf.fit(Y_df)
        if Xf_df is not None:
            Yf = mlf.predict(h=int(fh), X_df=Xf_df)
        else:
            Yf = mlf.predict(h=int(fh))
        try:
            Yf = Yf[Yf['unique_id'] == 'ts']
        except Exception:
            pass
        
        # Use common forecast value extraction
        from ..common import _extract_forecast_values
        f_vals = _extract_forecast_values(Yf, fh, "mlf_rf")
        
        params_used = {
            'lags': lags or [],
            'rolling_agg': rolling_agg,
        }
        return f_vals, params_used
    except Exception as ex:
        raise RuntimeError(f"mlf_rf error: {ex}")


def forecast_mlf_lightgbm(
    *,
    series: np.ndarray,
    fh: int,
    timeframe: str,
    lags: Optional[list] = None,
    rolling_agg: Optional[str] = None,
    n_estimators: int = 200,
    learning_rate: float = 0.05,
    num_leaves: int = 31,
    max_depth: int = -1,
    exog_used: Optional[np.ndarray] = None,
    exog_future: Optional[np.ndarray] = None,
    future_times: Optional[list] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    try:
        from mlforecast import MLForecast as _MLForecast  # type: ignore
        from lightgbm import LGBMRegressor as _LGBM  # type: ignore
        import pandas as _pd
    except Exception as ex:
        raise RuntimeError(f"Failed to import mlforecast/lightgbm: {ex}")

    # Use common DataFrame creation helper
    from ..common import _create_training_dataframes
    Y_df, X_df, Xf_df = _create_training_dataframes(series, fh, exog_used, exog_future)
    
    lgbm = _LGBM(n_estimators=int(n_estimators), learning_rate=float(learning_rate), num_leaves=int(num_leaves), max_depth=int(max_depth), random_state=42)
    try:
        mlf = _MLForecast(models=[lgbm], freq='D')
        if lags:
            mlf = mlf.add_lags(lags)
        if rolling_agg in {'mean', 'min', 'max', 'std'} and lags:
            for w in sorted(set([x for x in lags if x and x > 1])):
                mlf = mlf.add_rolling_windows(rolling_features={rolling_agg: [w]})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mlf.fit(Y_df)
        if Xf_df is not None:
            Yf = mlf.predict(h=int(fh), X_df=Xf_df)
        else:
            Yf = mlf.predict(h=int(fh))
        try:
            Yf = Yf[Yf['unique_id'] == 'ts']
        except Exception:
            pass
        
        # Use common forecast value extraction
        from ..common import _extract_forecast_values
        f_vals = _extract_forecast_values(Yf, fh, "mlf_lightgbm")
        
        params_used = {
            'lags': lags or [],
            'rolling_agg': rolling_agg,
            'n_estimators': int(n_estimators),
            'learning_rate': float(learning_rate),
            'num_leaves': int(num_leaves),
            'max_depth': int(max_depth),
        }
        return f_vals, params_used
    except Exception as ex:
        raise RuntimeError(f"mlf_lightgbm error: {ex}")

