"""
Forecast engine core logic and orchestration.
"""

from typing import Any, Dict, Optional, List, Literal, Tuple
import logging
import numpy as np
import pandas as pd
import math

from mtdata.shared.constants import TIMEFRAME_MAP, TIMEFRAME_SECONDS
from mtdata.shared.schema import ForecastMethodLiteral, TimeframeLiteral, DenoiseSpec
from mtdata.utils.mt5 import get_symbol_info_cached, mt5
from mtdata.utils.utils import (
    _format_time_minimal,
    _format_time_minimal_local,
    _use_client_tz,
    parse_kv_or_json as _parse_kv_or_json,
)
from mtdata.utils.denoise import _apply_denoise, normalize_denoise_spec as _normalize_denoise_spec
from mtdata.forecast import forecast_preprocessing as _forecast_preprocessing
from mtdata.forecast.common import (
    fetch_history as _fetch_history,
    default_seasonality as _default_seasonality_period,
    next_times_from_last as _next_times_from_last,
)
from mtdata.forecast.target_builder import build_target_series
from mtdata.forecast.registry import ForecastRegistry
# Import all method modules to ensure registration
import mtdata.forecast.methods.classical
import mtdata.forecast.methods.ets_arima
import mtdata.forecast.methods.statsforecast
import mtdata.forecast.methods.mlforecast
import mtdata.forecast.methods.pretrained
import mtdata.forecast.methods.neural
import mtdata.forecast.methods.sktime
import mtdata.forecast.methods.gluonts_extra
import mtdata.forecast.methods.analog
import mtdata.forecast.methods.monte_carlo  # noqa: F401 (method registration side effects)

# Backward-compatibility surface for tests/monkeypatching.
_PATCHABLE_GLOBALS = (mt5,)

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

logger = logging.getLogger(__name__)


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
    # Allow any registered method in ensemble if it supports what we need
    # But for safety/speed, we might restrict to fast methods or check registry
    
    method_params = dict(params or {})
    try:
        forecaster = ForecastRegistry.get(m)
        res = forecaster.forecast(series, horizon, seasonality or 1, method_params)
        return res.forecast
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


# Supported forecast methods - dynamically fetch from registry
def _get_available_methods():
    return tuple(ForecastRegistry.get_all_method_names())

_FORECAST_METHODS = _get_available_methods()



def _calculate_lookback_bars(method_l: str, horizon: int, lookback: Optional[int],
                             seasonality: int, timeframe: str) -> int:
    """Calculate the number of bars needed for forecasting."""
    if lookback is not None and lookback > 0:
        return int(lookback) + 2

    if method_l == 'analog':
        # Default search_depth=5000, window=64.
        # But we don't have params here. Assume reasonable default.
        # If the user provides params['search_depth'], we can't see it here easily without changing signature.
        # So we return a large enough default for fetch. 
        # Actually AnalogMethod re-fetches, so this 'need' is just for the 'target_series' passed to it,
        # which it ignores (via Option A).
        # EXCEPT: the engine checks len(df) < 3. 
        # So we just need something small like 100 to pass checks.
        return max(100, int(horizon) + 10)

    if method_l == 'seasonal_naive':
        return max(3 * seasonality, int(horizon) + seasonality + 2)
    elif method_l in ('theta', 'fourier_ols'):
        return max(300, int(horizon) + (2 * seasonality if seasonality else 50))
    else:  # naive, drift and others
        return max(100, int(horizon) + 10)
def _format_forecast_output(
    forecast_values: np.ndarray,
    last_epoch: float,
    tf_secs: int,
    horizon: int,
    base_col: str,
    df: pd.DataFrame,
    ci_alpha: Optional[float],
    ci_values: Optional[np.ndarray],
    method: str,
    quantity: str,
    denoise_used: bool,
    metadata: Optional[Dict[str, Any]] = None,
    digits: Optional[int] = None,
    forecast_return_values: Optional[np.ndarray] = None,
    reconstructed_prices: Optional[np.ndarray] = None,
    symbol: Optional[str] = None,
    timeframe: Optional[str] = None,
) -> Dict[str, Any]:
    """Format forecast output with proper structure."""
    # Generate future time indices
    future_epochs = _next_times_from_last(last_epoch, tf_secs, horizon)

    # Time formatting
    _use_ctz = _use_client_tz()
    if _use_ctz:
        future_times = [_format_time_minimal_local(e) for e in future_epochs]
    else:
        future_times = [_format_time_minimal(e) for e in future_epochs]

    # Build base result
    if _use_ctz:
        last_observation_time = _format_time_minimal_local(float(last_epoch))
    else:
        last_observation_time = _format_time_minimal(float(last_epoch))
    forecast_start_epoch = float(future_epochs[0]) if future_epochs else None
    forecast_start_time = future_times[0] if future_times else None
    forecast_start_gap_seconds = (
        float(forecast_start_epoch - float(last_epoch))
        if forecast_start_epoch is not None
        else None
    )
    forecast_start_gap_bars = (
        float(forecast_start_gap_seconds) / float(tf_secs)
        if forecast_start_gap_seconds is not None and tf_secs
        else None
    )
    result: Dict[str, Any] = {
        "success": True,
        "method": method,
        "horizon": horizon,
        "base_col": base_col,
        "last_observation_time": last_observation_time,
        "last_observation_epoch": float(last_epoch),
        "forecast_start_time": forecast_start_time,
        "forecast_start_epoch": forecast_start_epoch,
        "forecast_start_gap_seconds": forecast_start_gap_seconds,
        "forecast_start_gap_bars": forecast_start_gap_bars,
        "forecast_anchor": "next_timeframe_bar_after_last_observation",
        "forecast_step_seconds": int(tf_secs),
        "forecast_time": future_times,
        "forecast_epoch": future_epochs,
    }

    # Choose which arrays to expose
    if quantity == 'return':
        if forecast_return_values is None:
            forecast_return_values = forecast_values
        result["forecast_return"] = [float(v) for v in forecast_return_values]
        result["forecast"] = [float(v) for v in forecast_return_values]
        if reconstructed_prices is not None:
            result["forecast_price"] = [float(v) for v in reconstructed_prices]
    else:
        result["forecast_price"] = [float(v) for v in forecast_values]
        result["forecast"] = [float(v) for v in forecast_values]
    
    if digits is not None:
        result["digits"] = int(digits)

    # Add confidence intervals if available. If they are requested but missing,
    # surface an explicit warning to avoid misleading point-only interpretation.
    if ci_alpha is not None:
        ci_alpha_value: Optional[float] = None
        try:
            ci_alpha_value = float(ci_alpha)
        except Exception:
            ci_alpha_value = None
        result["ci_requested"] = True
        if ci_alpha_value is not None:
            result["ci_alpha_requested"] = ci_alpha_value

        if ci_values is not None and len(ci_values) == 2:  # [lower, upper]
            result["ci_available"] = True
            result["ci_status"] = "available"
            if ci_alpha_value is not None:
                result["ci_alpha"] = ci_alpha_value
            lower_vals = [float(v) for v in ci_values[0]]
            upper_vals = [float(v) for v in ci_values[1]]
            if quantity == 'return':
                result["lower_return"] = lower_vals
                result["upper_return"] = upper_vals
                # Keep generic keys for lightweight renderers expecting non-price intervals.
                result["lower"] = lower_vals
                result["upper"] = upper_vals
            else:
                result["lower_price"] = lower_vals
                result["upper_price"] = upper_vals
        else:
            symbol_arg = str(symbol).strip() if symbol is not None else ""
            timeframe_arg = str(timeframe).strip() if timeframe is not None else ""
            symbol_token = symbol_arg if symbol_arg else "SYMBOL"
            timeframe_token = timeframe_arg if timeframe_arg else None
            cmd_parts = [
                "mtdata-cli forecast_conformal_intervals",
                symbol_token,
            ]
            if timeframe_token:
                cmd_parts.extend(["--timeframe", timeframe_token])
            cmd_parts.extend(["--method", str(method), "--horizon", str(horizon)])
            conformal_cmd = " ".join(cmd_parts)
            warning_text = (
                f"Point forecast only for method '{method}'; confidence intervals are unavailable. "
                f"Use forecast_conformal_intervals for uncertainty bands. Example: {conformal_cmd}"
            )
            warnings = result.get("warnings")
            if not isinstance(warnings, list):
                warnings = []
            warnings.append(warning_text)
            result["warnings"] = warnings
            result["ci_available"] = False
            result["ci_status"] = "unavailable"
            result["ci_unavailable"] = True

    # Add metadata
    result.update({
        "last_epoch": float(last_epoch),
        "quantity": quantity,
        "denoise_applied": denoise_used,
    })
    
    if metadata:
        result.update(metadata)

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
    exog_used: Optional[np.ndarray] = None,
    exog_future: Optional[np.ndarray] = None,
    prefetched_df: Optional[pd.DataFrame] = None,
    prefetched_base_col: Optional[str] = None,
    prefetched_denoise_spec: Optional[Any] = None,
) -> Dict[str, Any]:
    """Core forecast engine implementation.

    This is the main orchestration function that coordinates all forecasting operations.
    """
    try:
        ci_values = None
        # Coerce CLI string inputs to proper types
        try:
            horizon = int(horizon) if horizon is not None else 12
        except (ValueError, TypeError):
            horizon = 12
            
        try:
            lookback = int(lookback) if lookback is not None else None
        except (ValueError, TypeError):
            lookback = None
        
        # Validation
        if timeframe not in TIMEFRAME_MAP:
            return {"error": f"Invalid timeframe: {timeframe}. Valid options: {list(TIMEFRAME_MAP.keys())}"}
        tf_secs = TIMEFRAME_SECONDS.get(timeframe)
        if not tf_secs:
            return {"error": f"Unsupported timeframe seconds for {timeframe}"}

        method_l = str(method).lower().strip()
        quantity_l = str(quantity).lower().strip()
        
        # Refresh available methods
        available_methods = _get_available_methods()
        if method_l not in available_methods:
            return {"error": f"Invalid method: {method}. Valid options: {list(available_methods)}"}

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

        # Fetch data (or reuse prefetched) and optional denoise
        if prefetched_df is not None:
            df = prefetched_df
            base_col = prefetched_base_col or ('close_dn' if 'close_dn' in df.columns else 'close')
            dn_spec_used = prefetched_denoise_spec
        else:
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

        # Track last close for potential price reconstruction
        try:
            last_close = float(df['close'].iloc[-1])
        except Exception:
            last_close = float('nan')

        # Prepare base data
        base_col_initial = base_col
        base_col_prepared = _forecast_preprocessing._prepare_base_data(df, quantity_l, target, base_col)

        # Apply features and target specification
        base_col_prepared = _forecast_preprocessing._apply_features_and_target_spec(
            df,
            features,
            target_spec,
            base_col_prepared,
            parse_kv_or_json=_parse_kv_or_json,
        )

        # Prepare target series, honoring target_spec if provided
        target_series = df[base_col_prepared].dropna()
        target_info: Dict[str, Any] = {}
        if target_spec:
            try:
                y_arr, target_info = build_target_series(df, base_col_initial, target_spec, legacy_target=str(target))
                target_series = pd.Series(y_arr, index=df.index)
                base_col = target_info.get('base', base_col_initial)
            except Exception as ex:
                return {"error": f"Invalid target_spec: {ex}"}
        else:
            base_col = base_col_prepared
            if quantity_l == 'return' or str(target).lower() == 'return':
                target_series = df[base_col].dropna()
            else:
                target_series = df[base_col]

        target_series = target_series.dropna()
        if len(target_series) < 3:
            return {"error": f"Not enough valid data points in column '{base_col}'"}

        # Prepare feature matrices if applicable (only if exog_used not provided).
        X = exog_used
        future_exog = exog_future
        if X is None and features:
            future_times = _next_times_from_last(float(df['time'].iloc[-1]), int(tf_secs), int(horizon))
            try:
                X, built_future_exog, _feat_info = _forecast_preprocessing.prepare_features(
                    df,
                    features,
                    future_times,
                    horizon,
                    training_index=target_series.index,
                    dimred_method=dimred_method,
                    dimred_params=dimred_params,
                    parse_kv_or_json=_parse_kv_or_json,
                    reducer_factory=_forecast_preprocessing._create_dimred_reducer,
                )
            except Exception as exc:
                logger.debug("Feature preparation failed: %s", exc)
                X, built_future_exog, _feat_info = None, None, {'error': f"feature_build_error: {str(exc)}"}
            if future_exog is None:
                future_exog = built_future_exog

        # Get last timestamp and values
        last_epoch = float(df['time'].iloc[-1])

        # Core run diagnostics to make model context explicit for users.
        history_start_epoch: Optional[float]
        history_end_epoch: Optional[float]
        try:
            history_start_epoch = float(df['time'].iloc[0])
        except Exception:
            history_start_epoch = None
        try:
            history_end_epoch = float(df['time'].iloc[-1])
        except Exception:
            history_end_epoch = None
        if _use_client_tz():
            fmt_time = _format_time_minimal_local
        else:
            fmt_time = _format_time_minimal
        engine_diagnostics: Dict[str, Any] = {
            "lookback_bars_requested": int(lookback) if lookback is not None else None,
            "lookback_bars_fetched": int(need),
            "history_bars_used": int(len(df)),
            "target_points_used": int(len(target_series)),
            "seasonality_used": int(seasonality),
            "quantity": quantity_l,
            "base_col_used": str(base_col),
        }
        if history_start_epoch is not None:
            engine_diagnostics["history_start_epoch"] = history_start_epoch
            engine_diagnostics["history_start_time"] = fmt_time(history_start_epoch)
        if history_end_epoch is not None:
            engine_diagnostics["history_end_epoch"] = history_end_epoch
            engine_diagnostics["history_end_time"] = fmt_time(history_end_epoch)
        
        # Get symbol info for digits
        digits = None
        try:
            s_info = get_symbol_info_cached(symbol)
            if s_info:
                digits = s_info.digits
        except Exception:
            pass

        # Call engine
        metadata: Dict[str, Any] = {}
        try:
            if method_l == 'ensemble':
                ensemble_meta = {}
                base_methods_in = p.get('methods')
                if isinstance(base_methods_in, str):
                    base_methods = [m.strip().lower() for m in base_methods_in.split(',') if m.strip()]
                elif isinstance(base_methods_in, (list, tuple)):
                    base_methods = [str(m).lower().strip() for m in base_methods_in if str(m).strip()]
                else:
                    base_methods = ['naive', 'theta', 'fourier_ols']
                
                # Filter to available methods
                avail = _get_available_methods()
                base_methods = [m for m in base_methods if m in avail and m != 'ensemble']
                
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
                metadata = ensemble_meta
                if rmse is not None:
                    ensemble_meta['cv_rmse'] = [float(v) for v in rmse.tolist()]
                if effective_mode == 'stacking':
                    ensemble_meta['intercept'] = float(ensemble_intercept)
                if expose_components:
                    ensemble_meta['components'] = {m: [float(v) for v in fc.tolist()] for m, fc in zip(base_methods, component_forecasts)}
            
            else:
                # Use Registry for all other methods
                forecaster = ForecastRegistry.get(method_l)
                
                # Prepare exog variables if supported and available
                # Note: X is the feature matrix for the training period. 
                # For future exog, we would need to generate/fetch it. 
                # Currently the engine doesn't support generating future exog automatically 
                # unless provided in params or features.
                # But some methods (like ML) might use X (training exog) during training.
                
                # Call forecast
                method_params = dict(p)
                if method_l == 'analog':
                    method_params.setdefault('symbol', symbol)
                    method_params.setdefault('timeframe', timeframe)
                    if as_of is not None:
                        method_params.setdefault('as_of', as_of)
                if ci_alpha is not None and 'ci_alpha' not in method_params:
                    method_params['ci_alpha'] = ci_alpha
                call_kwargs: Dict[str, Any] = {
                    'ci_alpha': ci_alpha,
                    'as_of': as_of,
                    'quantity': quantity_l,
                    'target': target,
                }
                if X is not None:
                    call_kwargs['exog_used'] = X

                res = forecaster.forecast(
                    target_series,
                    horizon,
                    seasonality,
                    method_params,
                    exog_future=future_exog,
                    **call_kwargs,
                )
                forecast_values = res.forecast
                ci_values = res.ci_values
                metadata = res.metadata or {}
                
                # Add params used to metadata
                metadata['params_used'] = res.params_used

        except Exception as e:
            return {"error": f"Forecast method '{method}' failed: {str(e)}"}

        if forecast_values is None:
            return {"error": f"Method '{method}' returned no forecast values"}

        if not isinstance(metadata, dict):
            metadata = {}
        existing_diagnostics = metadata.get("diagnostics")
        if not isinstance(existing_diagnostics, dict):
            existing_diagnostics = {}
        merged_diagnostics = dict(existing_diagnostics)
        for k, v in engine_diagnostics.items():
            if k not in merged_diagnostics:
                merged_diagnostics[k] = v
        metadata["diagnostics"] = merged_diagnostics

        # Prepare output arrays
        forecast_return_vals = None
        reconstructed_prices = None
        if quantity_l == 'return':
            forecast_return_vals = np.asarray(forecast_values, dtype=float)
            if np.isfinite(last_close):
                reconstructed_prices = last_close * np.exp(np.cumsum(forecast_return_vals))

        # Format and return output
        denoise_used = dn_spec_used is not None
        result = _format_forecast_output(
            forecast_values,
            last_epoch,
            tf_secs,
            horizon,
            base_col,
            df,
            ci_alpha,
            ci_values,
            method,
            quantity_l,
            denoise_used,
            metadata,
            digits=digits,
            forecast_return_values=forecast_return_vals,
            reconstructed_prices=reconstructed_prices,
            symbol=symbol,
            timeframe=timeframe,
        )
        if method_l == 'ensemble' and ensemble_meta:
            result['ensemble'] = ensemble_meta
        return result

    except Exception as e:
        return {"error": f"Forecast engine failed: {str(e)}"}
