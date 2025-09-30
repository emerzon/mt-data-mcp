"""
Forecast engine core logic and orchestration.
"""

from typing import Any, Dict, Optional, List, Literal, Tuple
from datetime import datetime
import numpy as np
import pandas as pd
import math
import warnings
import sys
import os

# Add the src directory to path for relative imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from mtdata.core.constants import TIMEFRAME_MAP, TIMEFRAME_SECONDS
from mtdata.utils.mt5 import _mt5_epoch_to_utc, _mt5_copy_rates_from, _ensure_symbol_ready
from mtdata.utils.utils import (
    _parse_start_datetime as _parse_start_datetime_util,
    _format_time_minimal as _format_time_minimal_util,
    _format_time_minimal_local as _format_time_minimal_local_util,
    _use_client_tz as _use_client_tz_util,
)
from mtdata.utils.indicators import _parse_ti_specs as _parse_ti_specs_util, _apply_ta_indicators as _apply_ta_indicators_util
from mtdata.utils.denoise import _apply_denoise, normalize_denoise_spec as _normalize_denoise_spec
from mtdata.forecast.common import (
    parse_kv_or_json as _parse_kv_or_json,
    fetch_history as _fetch_history,
)

# Import individual forecast methods
from mtdata.forecast.methods.transformers import (
    forecast_chronos_bolt as _chronos_bolt_impl,
    forecast_timesfm as _timesfm_impl,
)
from mtdata.forecast.methods.classical import (
    forecast_naive as _naive_impl,
    forecast_drift as _drift_impl,
    forecast_seasonal_naive as _snaive_impl,
    forecast_theta as _theta_impl,
    forecast_fourier_ols as _fourier_impl,
)
from mtdata.forecast.methods.ets_arima import (
    forecast_ses as _ses_impl,
    forecast_holt as _holt_impl,
    forecast_holt_winters as _hw_impl,
    forecast_sarimax as _sarimax_impl,
)
from mtdata.forecast.methods.neural import forecast_neural as _neural_impl

_ENSEMBLE_BASE_METHODS = (
    'naive',
    'drift',
    'seasonal_naive',
    'theta',
    'fourier_ols',
    'ses',
    'holt',
    'holt_winters_add',
    'holt_winters_mul',
    'arima',
    'sarima',
)


def _normalize_weights(weights: Any, size: int) -> Optional[np.ndarray]:
    if weights is None:
        return None
    vals: List[float] = []
    if isinstance(weights, (list, tuple)):
        vals = [float(v) for v in list(weights)[:size]]
    elif isinstance(weights, str):
        parts = [p.strip() for p in weights.split(',') if p.strip()]
        vals = [float(p) for p in parts[:size]]
    else:
        return None
    if len(vals) != size:
        return None
    arr = np.asarray(vals, dtype=float)
    if not np.all(np.isfinite(arr)):
        return None
    arr = np.clip(arr, a_min=0.0, a_max=None)
    total = float(np.sum(arr))
    if total <= 0:
        return None
    return arr / total


def _ensemble_dispatch_method(
    method_name: str,
    series: pd.Series,
    horizon: int,
    seasonality: Optional[int],
    params: Optional[Dict[str, Any]],
) -> Optional[np.ndarray]:
    """Run a supported ensemble base method with safe fallbacks."""

    m = str(method_name).lower().strip()
    if m not in _ENSEMBLE_BASE_METHODS:
        return None
    kwargs = dict(params or {})
    try:
        if m == 'naive':
            out = _naive_impl(series, horizon=horizon, **kwargs)
        elif m == 'drift':
            out = _drift_impl(series, horizon=horizon, **kwargs)
        elif m == 'seasonal_naive':
            out = _snaive_impl(series, horizon=horizon, seasonality=seasonality, **kwargs)
        elif m == 'theta':
            out, _ = _theta_impl(series, horizon=horizon, seasonality=seasonality, ci_alpha=None, **kwargs)
        elif m == 'fourier_ols':
            out = _fourier_impl(series, horizon=horizon, seasonality=seasonality, **kwargs)
        elif m == 'ses':
            out = _ses_impl(series, horizon=horizon, **kwargs)
        elif m == 'holt':
            out = _holt_impl(series, horizon=horizon, **kwargs)
        elif m in ('holt_winters_add', 'holt_winters_mul'):
            seasonal = 'add' if m.endswith('add') else 'mul'
            out = _hw_impl(series, horizon=horizon, seasonal=seasonal, seasonality=seasonality, **kwargs)
        elif m == 'arima':
            out, _ = _sarimax_impl(series, horizon=horizon, seasonal=False, ci_alpha=None, **kwargs)
        elif m == 'sarima':
            out, _ = _sarimax_impl(series, horizon=horizon, seasonal=True, seasonality=seasonality, ci_alpha=None, **kwargs)
        else:
            return None
        if out is None:
            return None
        return np.asarray(out, dtype=float)
    except Exception:
        return None


def _prepare_ensemble_cv(
    series: pd.Series,
    methods: List[str],
    horizon: int,
    seasonality: Optional[int],
    params_map: Dict[str, Dict[str, Any]],
    cv_points: int,
    min_train: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Collect walk-forward one-step predictions for ensemble weighting."""

    n = len(series)
    if n <= max(min_train, horizon + 2):
        return np.empty((0, len(methods))), np.empty((0,))

    end = n - horizon
    candidate_idx = list(range(max(min_train, 3), end))
    if not candidate_idx:
        return np.empty((0, len(methods))), np.empty((0,))
    if cv_points and len(candidate_idx) > cv_points:
        candidate_idx = candidate_idx[-cv_points:]

    rows: List[List[float]] = []
    targets: List[float] = []
    for idx in candidate_idx:
        train = series.iloc[:idx]
        if len(train) < min_train:
            continue
        row: List[float] = []
        success = True
        for m in methods:
            fc = _ensemble_dispatch_method(m, train, horizon, seasonality, params_map.get(m, {}))
            if fc is None or fc.size == 0 or not math.isfinite(float(fc[0])):
                success = False
                break
            row.append(float(fc[0]))
        if not success:
            continue
        rows.append(row)
        targets.append(float(series.iloc[idx]))

    if not rows:
        return np.empty((0, len(methods))), np.empty((0,))

    return np.asarray(rows, dtype=float), np.asarray(targets, dtype=float)


# Local fallbacks for typing aliases used in signatures (avoid import cycle)
try:
    from ..core.server import ForecastMethodLiteral, TimeframeLiteral, DenoiseSpec  # type: ignore
except Exception:  # runtime fallback
    ForecastMethodLiteral = str
    TimeframeLiteral = str
    DenoiseSpec = Dict[str, Any]

# Supported forecast methods
_FORECAST_METHODS = (
    "naive",
    "seasonal_naive",
    "drift",
    "theta",
    "fourier_ols",
    "ses",
    "holt",
    "holt_winters_add",
    "holt_winters_mul",
    "arima",
    "sarima",
    "mc_gbm",
    "hmm_mc",
    "nhits",
    "nbeatsx",
    "tft",
    "patchtst",
    "sf_autoarima",
    "sf_theta",
    "sf_autoets",
    "sf_seasonalnaive",
    "mlf_rf",
    "mlf_lightgbm",
    "chronos_bolt",
    "timesfm",
    "lag_llama",
    "ensemble",
)


def _default_seasonality_period(timeframe: str) -> int:
    """Return default seasonality period based on timeframe."""
    tf_map = {
        'M1': 1440, 'M5': 288, 'M15': 96, 'M30': 48,
        'H1': 24, 'H4': 6, 'H12': 2,
        'D1': 30, 'W1': 52, 'MN1': 12
    }
    return tf_map.get(timeframe, 12)


def _next_times_from_last(last_epoch: float, tf_secs: int, horizon: int) -> List[float]:
    """Generate future time epochs from last timestamp."""
    return [last_epoch + tf_secs * (i + 1) for i in range(horizon)]


def _pd_freq_from_timeframe(tf: str) -> str:
    """Convert timeframe to pandas frequency string."""
    mapping = {
        'M1': '1min', 'M5': '5min', 'M15': '15min', 'M30': '30min',
        'H1': '1H', 'H2': '2H', 'H3': '3H', 'H4': '4H',
        'H6': '6H', 'H8': '8H', 'H12': '12H',
        'D1': '1D', 'W1': '1W', 'MN1': '1M'
    }
    return mapping.get(tf, '1H')


def _calculate_lookback_bars(method_l: str, horizon: int, lookback: Optional[int],
                             seasonality: int, timeframe: str) -> int:
    """Calculate the number of bars needed for forecasting."""
    if lookback is not None and lookback > 0:
        return int(lookback) + 2

    if method_l == 'seasonal_naive':
        return max(3 * seasonality, int(horizon) + seasonality + 2)
    elif method_l in ('theta', 'fourier_ols'):
        return max(300, int(horizon) + (2 * seasonality if seasonality else 50))
    else:  # naive, drift and others
        return max(100, int(horizon) + 10)


def _prepare_base_data(df: pd.DataFrame, quantity: str, target: str) -> str:
    """Prepare base data column for forecasting."""
    base_col = 'close'

    if quantity_l == 'return':
        df['__log_return'] = np.log(df['close'] / df['close'].shift(1))
        base_col = '__log_return'
    elif quantity_l == 'volatility':
        if target_l == 'price':
            df['__log_return'] = np.log(df['close'] / df['close'].shift(1))
            df['__squared_return'] = df['__log_return'] ** 2
            base_col = '__squared_return'
        else:  # return
            df['__squared_return'] = df['__log_return'] ** 2
            base_col = '__squared_return'

    return base_col


def _apply_features_and_target_spec(df: pd.DataFrame, features: Optional[Dict[str, Any]],
                                   target_spec: Optional[Dict[str, Any]], base_col: str) -> str:
    """Apply features and target specification to the dataframe."""
    # Apply technical indicators if specified in features
    if features:
        ti_spec = features.get('ti')
        if ti_spec:
            ti_list = _parse_ti_specs_util(ti_spec)
            if ti_list:
                ti_cols = _apply_ta_indicators_util(df, ti_spec)
                # Update base_col if TI column is specified as target
                if target_spec and target_spec.get('column') in ti_cols:
                    base_col = target_spec.get('column')

    # Apply target column transformations
    if target_spec:
        target_col = target_spec.get('column', base_col)
        transform = target_spec.get('transform')

        if transform == 'log':
            df[f'__target_{target_col}'] = np.log(df[target_col])
            base_col = f'__target_{target_col}'
        elif transform == 'diff':
            df[f'__target_{target_col}'] = df[target_col].diff()
            base_col = f'__target_{target_col}'
        elif transform == 'pct':
            df[f'__target_{target_col}'] = df[target_col].pct_change()
            base_col = f'__target_{target_col}'
        elif target_col != base_col:
            base_col = target_col

    return base_col


def _apply_dimensionality_reduction(X: pd.DataFrame, dimred_method: Optional[str],
                                   dimred_params: Optional[Dict[str, Any]]) -> pd.DataFrame:
    """Apply dimensionality reduction to feature matrix."""
    if not dimred_method or len(X.columns) <= 1:
        return X

    try:
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        from sklearn.feature_selection import SelectKBest, f_regression

        params = dimred_params or {}

        if dimred_method.lower() == 'pca':
            n_components = params.get('n_components', min(5, X.shape[1]))
            reducer = PCA(n_components=n_components)
            X_reduced = reducer.fit_transform(X)
            return pd.DataFrame(X_reduced, columns=[f'pca_{i}' for i in range(X_reduced.shape[1])])

        elif dimred_method.lower() == 'tsne':
            n_components = params.get('n_components', 2)
            reducer = TSNE(n_components=n_components, random_state=42)
            X_reduced = reducer.fit_transform(X)
            return pd.DataFrame(X_reduced, columns=[f'tsne_{i}' for i in range(X_reduced.shape[1])])

        elif dimred_method.lower() == 'selectkbest':
            k = params.get('k', min(5, X.shape[1]))
            selector = SelectKBest(score_func=f_regression, k=k)
            X_reduced = selector.fit_transform(X, y=None)  # unsupervised selection
            return pd.DataFrame(X_reduced, columns=[f'select_{i}' for i in range(X_reduced.shape[1])])

    except Exception:
        # Fall back to original features if dimensionality reduction fails
        pass

    return X


def _format_forecast_output(forecast_values: np.ndarray, last_epoch: float, tf_secs: int,
                            horizon: int, base_col: str, df: pd.DataFrame,
                            ci_alpha: Optional[float], ci_values: Optional[np.ndarray],
                            method: str, quantity: str, denoise_used: bool) -> Dict[str, Any]:
    """Format forecast output with proper structure."""
    # Generate future time indices
    future_epochs = _next_times_from_last(last_epoch, tf_secs, horizon)

    # Time formatting
    _use_ctz = _use_client_tz_util()
    if _use_ctz:
        future_times = [_format_time_minimal_local_util(e) for e in future_epochs]
    else:
        future_times = [_format_time_minimal_util(e) for e in future_epochs]

    # Build base result
    result = {
        "success": True,
        "method": method,
        "horizon": horizon,
        "base_col": base_col,
        "forecast_price": [float(v) for v in forecast_values],
        "forecast_time": future_times,
        "forecast_epoch": future_epochs,
    }

    # Add confidence intervals if available
    if ci_alpha is not None and ci_values is not None:
        result["ci_alpha"] = float(ci_alpha)
        if len(ci_values) == 2:  # [lower, upper]
            result["lower_price"] = [float(v) for v in ci_values[0]]
            result["upper_price"] = [float(v) for v in ci_values[1]]

    # Add metadata
    result.update({
        "last_epoch": float(last_epoch),
        "quantity": quantity,
        "denoise_applied": denoise_used,
    })

    return result


def forecast_engine(
    symbol: str,
    timeframe: TimeframeLiteral = "H1",
    method: ForecastMethodLiteral = "theta",
    horizon: int = 12,
    lookback: Optional[int] = None,
    as_of: Optional[str] = None,
    params: Optional[Dict[str, Any]] = None,
    ci_alpha: Optional[float] = 0.05,
    quantity: Literal['price','return','volatility'] = 'price',
    target: Literal['price','return'] = 'price',
    denoise: Optional[DenoiseSpec] = None,
    features: Optional[Dict[str, Any]] = None,
    dimred_method: Optional[str] = None,
    dimred_params: Optional[Dict[str, Any]] = None,
    target_spec: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Core forecast engine implementation.

    This is the main orchestration function that coordinates all forecasting operations.
    """
    try:
        # Validation
        if timeframe not in TIMEFRAME_MAP:
            return {"error": f"Invalid timeframe: {timeframe}. Valid options: {list(TIMEFRAME_MAP.keys())}"}
        mt5_tf = TIMEFRAME_MAP[timeframe]
        tf_secs = TIMEFRAME_SECONDS.get(timeframe)
        if not tf_secs:
            return {"error": f"Unsupported timeframe seconds for {timeframe}"}

        method_l = str(method).lower().strip()
        quantity_l = str(quantity).lower().strip()
        if method_l not in _FORECAST_METHODS:
            return {"error": f"Invalid method: {method}. Valid options: {list(_FORECAST_METHODS)}"}

        # Volatility models have a dedicated endpoint
        if quantity_l == 'volatility' or method_l.startswith('vol_'):
            return {"error": "Use forecast_volatility for volatility models"}

        # Parse method params
        p = _parse_kv_or_json(params)
        seasonality = int(p.get('seasonality')) if p.get('seasonality') is not None else _default_seasonality_period(timeframe)

        if method_l == 'seasonal_naive' and (not seasonality or seasonality <= 0):
            return {"error": "seasonal_naive requires a positive 'seasonality' in params or auto period"}

        # Calculate lookback bars
        need = _calculate_lookback_bars(method_l, horizon, lookback, seasonality, timeframe)

        # Fetch data
        try:
            df = _fetch_history(symbol, timeframe, int(need), as_of)
        except Exception as ex:
            return {"error": str(ex)}
        if len(df) < 3:
            return {"error": "Not enough closed bars to compute forecast"}

        # Apply denoising
        base_col = 'close'
        dn_spec_used = None
        if denoise:
            try:
                _dn = _normalize_denoise_spec(denoise, default_when='pre_ti')
            except Exception:
                _dn = None
            added = _apply_denoise(df, _dn, default_when='pre_ti') if _dn else []
            dn_spec_used = _dn
            if len(added) > 0 and f"{base_col}_dn" in added:
                base_col = f"{base_col}_dn"

        # Prepare base data
        base_col = _prepare_base_data(df, quantity, target)

        # Apply features and target specification
        base_col = _apply_features_and_target_spec(df, features, target_spec, base_col)

        # Prepare target series
        target_series = df[base_col].dropna()
        if len(target_series) < 3:
            return {"error": f"Not enough valid data points in column '{base_col}'"}

        # Prepare feature matrix if applicable
        X = None
        if features and features.get('exog'):
            exog_cols = features['exog']
            if isinstance(exog_cols, str):
                exog_cols = [c.strip() for c in exog_cols.split(',')]

            # Filter to available columns
            available_exog = [col for col in exog_cols if col in df.columns and col != base_col]
            if available_exog:
                X = df[available_exog].loc[target_series.index]
                # Apply dimensionality reduction if specified
                X = _apply_dimensionality_reduction(X, dimred_method, dimred_params)

        # Get last timestamp and values
        last_epoch = float(df['__epoch'].iloc[-1])
        last_value = float(target_series.iloc[-1])

        # Dispatch to appropriate forecast method
        forecast_values = None
        ci_values = None
        ensemble_meta: Optional[Dict[str, Any]] = None

        try:
            if method_l == 'naive':
                forecast_values = _naive_impl(target_series, horizon=horizon, **p)
            elif method_l == 'drift':
                forecast_values = _drift_impl(target_series, horizon=horizon, **p)
            elif method_l == 'seasonal_naive':
                forecast_values = _snaive_impl(target_series, horizon=horizon, seasonality=seasonality, **p)
            elif method_l == 'theta':
                forecast_values, ci_values = _theta_impl(target_series, horizon=horizon, seasonality=seasonality, ci_alpha=ci_alpha, **p)
            elif method_l == 'fourier_ols':
                forecast_values = _fourier_impl(target_series, horizon=horizon, seasonality=seasonality, **p)
            elif method_l == 'ses':
                forecast_values = _ses_impl(target_series, horizon=horizon, **p)
            elif method_l == 'holt':
                forecast_values = _holt_impl(target_series, horizon=horizon, **p)
            elif method_l == 'holt_winters_add' or method_l == 'holt_winters_mul':
                method = method_l.split('_')[-1]  # 'add' or 'mul'
                forecast_values = _hw_impl(target_series, horizon=horizon, seasonal=method, seasonality=seasonality, **p)
            elif method_l == 'arima':
                forecast_values, ci_values = _sarimax_impl(target_series, horizon=horizon, seasonal=False, ci_alpha=ci_alpha, **p)
            elif method_l == 'sarima':
                forecast_values, ci_values = _sarimax_impl(target_series, horizon=horizon, seasonal=True, seasonality=seasonality, ci_alpha=ci_alpha, **p)
            elif method_l in ('chronos_bolt', 'timesfm'):
                if method_l == 'chronos_bolt':
                    forecast_values = _chronos_bolt_impl(target_series, horizon=horizon, **p)
                elif method_l == 'timesfm':
                    forecast_values = _timesfm_impl(target_series, horizon=horizon, **p)
            elif method_l == 'ensemble':
                ensemble_meta = {}
                base_methods_in = p.get('methods')
                if isinstance(base_methods_in, str):
                    base_methods = [m.strip().lower() for m in base_methods_in.split(',') if m.strip()]
                elif isinstance(base_methods_in, (list, tuple)):
                    base_methods = [str(m).lower().strip() for m in base_methods_in if str(m).strip()]
                else:
                    base_methods = ['naive', 'theta', 'fourier_ols']
                base_methods = [m for m in base_methods if m in _ENSEMBLE_BASE_METHODS]
                seen: set[str] = set()
                base_methods = [m for m in base_methods if not (m in seen or seen.add(m))]
                if not base_methods:
                    base_methods = ['naive', 'theta']
                params_in = p.get('method_params') if isinstance(p.get('method_params'), dict) else {}
                params_map = {str(k).lower(): (v if isinstance(v, dict) else {}) for k, v in params_in.items()}
                mode = str(p.get('mode', 'average')).lower()
                cv_points = int(p.get('cv_points', max(6, len(base_methods) * 2)))
                min_train = int(p.get('min_train_size', max(30, horizon * 3)))
                expose_components = bool(p.get('expose_components', True))
                weights_vec = _normalize_weights(p.get('weights'), len(base_methods))
                ensemble_meta = {
                    'mode_requested': mode,
                    'methods': list(base_methods),
                    'cv_points_requested': cv_points,
                }
                effective_mode = mode
                rmse = None
                ensemble_intercept = 0.0
                coeffs = None
                cv_rows = 0
                if mode in ('bma', 'stacking'):
                    X_cv, y_cv = _prepare_ensemble_cv(target_series, base_methods, horizon, seasonality, params_map, cv_points, min_train)
                    cv_rows = int(len(y_cv))
                    if X_cv.shape[0] >= max(3, len(base_methods)):
                        if mode == 'bma':
                            errors = X_cv - y_cv[:, None]
                            rmse = np.sqrt(np.mean(np.square(errors), axis=0))
                            min_rmse = float(np.min(rmse))
                            weights_vec = np.exp(-0.5 * (rmse - min_rmse) / (min_rmse + 1e-12))
                            total = float(np.sum(weights_vec))
                            if total > 0:
                                weights_vec = weights_vec / total
                            else:
                                weights_vec = None
                        else:
                            X_aug = np.column_stack([np.ones(X_cv.shape[0]), X_cv])
                            beta, *_ = np.linalg.lstsq(X_aug, y_cv, rcond=None)
                            ensemble_intercept = float(beta[0])
                            coeffs = beta[1:]
                            effective_mode = 'stacking'
                    else:
                        effective_mode = 'average'
                if effective_mode != 'stacking':
                    if weights_vec is None:
                        weights_vec = np.full(len(base_methods), 1.0 / max(1, len(base_methods)))
                    else:
                        total = float(np.sum(weights_vec))
                        weights_vec = (weights_vec / total) if total > 0 else np.full(len(base_methods), 1.0 / max(1, len(base_methods)))
                component_methods: List[str] = []
                component_forecasts: List[np.ndarray] = []
                for m in base_methods:
                    fc = _ensemble_dispatch_method(m, target_series, horizon, seasonality, params_map.get(m, {}))
                    if fc is None or fc.size == 0:
                        continue
                    component_methods.append(m)
                    component_forecasts.append(fc)
                if not component_forecasts:
                    return {'error': 'Ensemble failed: no component forecasts'}
                if len(component_methods) != len(base_methods):
                    keep_idx = [base_methods.index(m) for m in component_methods]
                    if effective_mode == 'stacking' and coeffs is not None:
                        coeffs = coeffs[keep_idx]
                    elif weights_vec is not None:
                        weights_vec = weights_vec[keep_idx]
                    base_methods = component_methods
                if effective_mode != 'stacking':
                    total = float(np.sum(weights_vec)) if weights_vec is not None else 0.0
                    if weights_vec is None or total <= 0:
                        weights_vec = np.full(len(base_methods), 1.0 / len(base_methods))
                    else:
                        weights_vec = weights_vec / total
                    combined = np.zeros_like(component_forecasts[0], dtype=float)
                    for w, fc in zip(weights_vec, component_forecasts):
                        combined = combined + float(w) * fc
                else:
                    if coeffs is None or coeffs.size != len(base_methods):
                        coeffs = np.full(len(base_methods), 1.0 / len(base_methods))
                        ensemble_intercept = 0.0
                    combined = np.full_like(component_forecasts[0], ensemble_intercept, dtype=float)
                    for w, fc in zip(coeffs, component_forecasts):
                        combined = combined + float(w) * fc
                    weights_vec = coeffs
                forecast_values = combined
                ensemble_meta.update({
                    'mode_used': effective_mode,
                    'methods': list(base_methods),
                    'cv_points_used': cv_rows,
                    'weights': [float(w) for w in (weights_vec.tolist() if isinstance(weights_vec, np.ndarray) else weights_vec)],
                })
                if rmse is not None:
                    ensemble_meta['cv_rmse'] = [float(v) for v in rmse.tolist()]
                if effective_mode == 'stacking':
                    ensemble_meta['intercept'] = float(ensemble_intercept)
                if expose_components:
                    ensemble_meta['components'] = {m: [float(v) for v in fc.tolist()] for m, fc in zip(base_methods, component_forecasts)}

        except Exception as e:
            return {"error": f"Forecast method '{method}' failed: {str(e)}"}

        if forecast_values is None:
            return {"error": f"Method '{method}' returned no forecast values"}

        # Format and return output
        denoise_used = dn_spec_used is not None
        result = _format_forecast_output(
            forecast_values, last_epoch, tf_secs, horizon, base_col, df,
            ci_alpha, ci_values, method, quantity, denoise_used
        )
        if method_l == 'ensemble' and ensemble_meta:
            result['ensemble'] = ensemble_meta
        return result

    except Exception as e:
        return {"error": f"Forecast engine failed: {str(e)}"}