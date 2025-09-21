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

    Y_df = _pd.DataFrame({'unique_id': ['ts'] * int(len(series)), 'ds': _pd.RangeIndex(0, int(len(series))), 'y': series.astype(float)})
    rf = _RF(n_estimators=200, max_depth=None, random_state=42)
    try:
        Xf_df = None
        if exog_used is not None and isinstance(exog_used, np.ndarray) and exog_used.size:
            cols = [f'x{i}' for i in range(exog_used.shape[1])]
            for j, cname in enumerate(cols):
                Y_df[cname] = exog_used[:, j]
            if exog_future is not None and isinstance(exog_future, np.ndarray) and exog_future.size:
                Xf_df = _pd.DataFrame({'unique_id': ['ts'] * int(fh), 'ds': _pd.RangeIndex(int(len(series)), int(len(series))+int(fh))})
                for j, cname in enumerate(cols):
                    Xf_df[cname] = exog_future[:, j]
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
        pred_col = 'y' if 'y' in Yf.columns else None
        if pred_col is None:
            for c in list(Yf.columns):
                if c not in ('unique_id', 'ds'):
                    pred_col = c; break
        if pred_col is None:
            raise RuntimeError("mlf_rf prediction columns not found")
        vals = np.asarray(Yf[pred_col].to_numpy(), dtype=float)
        f_vals = vals[:fh] if vals.size >= fh else np.pad(vals, (0, fh - vals.size), mode='edge')
        params_used = {
            'lags': lags or [],
            'rolling_agg': rolling_agg,
        }
        return f_vals.astype(float, copy=False), params_used
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

    Y_df = _pd.DataFrame({'unique_id': ['ts'] * int(len(series)), 'ds': _pd.RangeIndex(0, int(len(series))), 'y': series.astype(float)})
    lgbm = _LGBM(n_estimators=int(n_estimators), learning_rate=float(learning_rate), num_leaves=int(num_leaves), max_depth=int(max_depth), random_state=42)
    try:
        Xf_df = None
        if exog_used is not None and isinstance(exog_used, np.ndarray) and exog_used.size:
            cols = [f'x{i}' for i in range(exog_used.shape[1])]
            for j, cname in enumerate(cols):
                Y_df[cname] = exog_used[:, j]
            if exog_future is not None and isinstance(exog_future, np.ndarray) and exog_future.size:
                Xf_df = _pd.DataFrame({'unique_id': ['ts'] * int(fh), 'ds': _pd.RangeIndex(int(len(series)), int(len(series))+int(fh))})
                for j, cname in enumerate(cols):
                    Xf_df[cname] = exog_future[:, j]
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
        pred_col = 'y' if 'y' in Yf.columns else None
        if pred_col is None:
            for c in list(Yf.columns):
                if c not in ('unique_id', 'ds'):
                    pred_col = c; break
        if pred_col is None:
            raise RuntimeError("mlf_lightgbm prediction columns not found")
        vals = np.asarray(Yf[pred_col].to_numpy(), dtype=float)
        f_vals = vals[:fh] if vals.size >= fh else np.pad(vals, (0, fh - vals.size), mode='edge')
        params_used = {
            'lags': lags or [],
            'rolling_agg': rolling_agg,
            'n_estimators': int(n_estimators),
            'learning_rate': float(learning_rate),
            'num_leaves': int(num_leaves),
            'max_depth': int(max_depth),
        }
        return f_vals.astype(float, copy=False), params_used
    except Exception as ex:
        raise RuntimeError(f"mlf_lightgbm error: {ex}")

