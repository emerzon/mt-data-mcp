"""
Forecast engine core logic and orchestration.
"""

from typing import Any, Dict, Optional, List, Literal, Tuple
import logging
import numpy as np
import pandas as pd
import math

from ..bootstrap.settings import mt5_config
from ..shared.constants import TIMEFRAME_MAP, TIMEFRAME_SECONDS
from ..shared.schema import ForecastMethodLiteral, TimeframeLiteral, DenoiseSpec
from ..shared.validators import invalid_timeframe_error, unsupported_timeframe_seconds_error
from ..utils.denoise import _apply_denoise, normalize_denoise_spec as _normalize_denoise_spec
from ..utils.mt5 import get_cached_mt5_time_alignment, get_symbol_info_cached
from ..utils.utils import (
    _format_time_minimal,
    _format_time_minimal_local,
    _use_client_tz,
    parse_kv_or_json as _parse_kv_or_json,
)
from . import forecast_preprocessing as _forecast_preprocessing
from .common import (
    default_seasonality,
    fetch_history as _fetch_history,
    next_times_from_last,
)
from .forecast_validation import format_invalid_method_error
from .interface import ForecastCallContext
from .registry import ForecastRegistry
from .target_builder import build_target_series

# Import all method modules to ensure registration
from .methods import analog as _analog_methods
from .methods import classical as _classical_methods
from .methods import ensemble as _ensemble_methods
from .methods import ets_arima as _ets_arima_methods
from .methods import gluonts_extra as _gluonts_extra_methods
from .methods import mlforecast as _mlforecast_methods
from .methods import monte_carlo as _monte_carlo_methods
from .methods import neural as _neural_methods
from .methods import pretrained as _pretrained_methods
from .methods import sktime as _sktime_methods
from .methods import statsforecast as _statsforecast_methods

_REGISTERED_METHOD_MODULES = (
    _analog_methods,
    _classical_methods,
    _ensemble_methods,
    _ets_arima_methods,
    _gluonts_extra_methods,
    _mlforecast_methods,
    _monte_carlo_methods,
    _neural_methods,
    _pretrained_methods,
    _sktime_methods,
    _statsforecast_methods,
)

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


def _clear_ensemble_dispatch_error() -> None:
    try:
        setattr(_ensemble_dispatch_method, "_last_error", None)
    except Exception:
        pass


def _record_ensemble_dispatch_error(method_name: str, exc: BaseException) -> None:
    try:
        setattr(
            _ensemble_dispatch_method,
            "_last_error",
            {
                "method": str(method_name),
                "error": str(exc),
                "error_type": type(exc).__name__,
            },
        )
    except Exception:
        pass


def _consume_ensemble_dispatch_error() -> Optional[Dict[str, Any]]:
    try:
        error = getattr(_ensemble_dispatch_method, "_last_error", None)
    except Exception:
        return None
    _clear_ensemble_dispatch_error()
    return dict(error) if isinstance(error, dict) else None


def _append_ensemble_failure(
    failures: Optional[List[Dict[str, Any]]],
    *,
    method_name: str,
    anchor_index: int,
    error_detail: Optional[Dict[str, Any]],
) -> None:
    if failures is None or len(failures) >= 12:
        return
    payload: Dict[str, Any] = {
        "stage": "cv",
        "method": str(method_name),
        "anchor_index": int(anchor_index),
    }
    if isinstance(error_detail, dict):
        for key, value in error_detail.items():
            if value is not None:
                payload[str(key)] = value
    failures.append(payload)


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
    _clear_ensemble_dispatch_error()
    method_params = dict(params or {})
    try:
        forecaster = ForecastRegistry.get(m)
        res = forecaster.forecast(series, horizon, seasonality or 1, method_params)
        return res.forecast
    except Exception as ex:
        _record_ensemble_dispatch_error(m, ex)
        return None


def _prepare_ensemble_cv(
    series: pd.Series,
    methods: List[str],
    horizon: int,
    seasonality: Optional[int],
    params_map: Dict[str, Dict[str, Any]],
    cv_points: int,
    min_train: int,
    failure_sink: Optional[List[Dict[str, Any]]] = None,
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
    horizon_i = int(horizon)
    for idx in candidate_idx:
        train = series.iloc[:idx]
        if len(train) < min_train:
            continue
        row_forecasts: List[np.ndarray] = []
        success = True
        for m in methods:
            fc = _ensemble_dispatch_method(m, train, horizon, seasonality, params_map.get(m, {}))
            if fc is None:
                _append_ensemble_failure(
                    failure_sink,
                    method_name=m,
                    anchor_index=idx,
                    error_detail=_consume_ensemble_dispatch_error()
                    or {"error": "Component forecast unavailable", "error_type": "forecast_unavailable"},
                )
                success = False
                break
            try:
                fc_arr = np.asarray(fc, dtype=float).reshape(-1)
            except Exception as ex:
                _append_ensemble_failure(
                    failure_sink,
                    method_name=m,
                    anchor_index=idx,
                    error_detail={"error": str(ex), "error_type": type(ex).__name__},
                )
                success = False
                break
            if fc_arr.size < horizon_i or not np.all(np.isfinite(fc_arr[:horizon_i])):
                _append_ensemble_failure(
                    failure_sink,
                    method_name=m,
                    anchor_index=idx,
                    error_detail={"error": "Forecast output was too short or non-finite", "error_type": "invalid_forecast"},
                )
                success = False
                break
            row_forecasts.append(fc_arr[:horizon_i])
        if not success:
            continue
        target_slice = np.asarray(series.iloc[idx: idx + horizon_i], dtype=float).reshape(-1)
        if target_slice.size < horizon_i or not np.all(np.isfinite(target_slice)):
            continue
        for step_idx in range(horizon_i):
            rows.append([float(forecast[step_idx]) for forecast in row_forecasts])
            targets.append(float(target_slice[step_idx]))

    if not rows:
        return np.empty((0, len(methods))), np.empty((0,))

    return np.asarray(rows, dtype=float), np.asarray(targets, dtype=float)


# Supported forecast methods - dynamically fetch from registry
def _get_available_methods():
    return tuple(ForecastRegistry.get_all_method_names())



def _calculate_lookback_bars(method_l: str, horizon: int, lookback: Optional[int],
                             seasonality: int, timeframe: str,
                             params: Optional[Dict[str, Any]] = None) -> int:
    """Calculate the number of bars needed for forecasting."""
    if method_l == 'analog':
        p = dict(params or {})
        try:
            window_size = int(p.get('window_size', 64))
        except Exception:
            window_size = 64
        try:
            search_depth = int(p.get('search_depth', 5000))
        except Exception:
            search_depth = 5000
        search_depth = max(1, search_depth)
        # Analog search needs enough history for:
        # 1. `search_depth` disjoint candidate starts
        # 2. each candidate's full window plus forecast future
        # 3. the active query window at the end of the series
        analog_history_bars = search_depth + (2 * window_size) + int(horizon) - 1
        if lookback is not None and lookback > 0:
            return max(int(lookback) + 2, analog_history_bars)
        return max(100, analog_history_bars)

    if lookback is not None and lookback > 0:
        return int(lookback) + 2

    if method_l == 'seasonal_naive':
        return max(3 * seasonality, int(horizon) + seasonality + 2)
    elif method_l in ('theta', 'fourier_ols'):
        return max(300, int(horizon) + (2 * seasonality if seasonality else 50))
    else:  # naive, drift and others
        return max(100, int(horizon) + 10)


def _resolve_history_context(
    *,
    symbol: str,
    timeframe: TimeframeLiteral,
    need: int,
    as_of: Optional[str],
    prefetched_df: Optional[pd.DataFrame],
    prefetched_base_col: Optional[str],
    prefetched_denoise_spec: Optional[Any],
    denoise: Optional[DenoiseSpec],
) -> Tuple[pd.DataFrame, str, Optional[Any]]:
    """Return the source DataFrame, active base column, and denoise spec used."""
    if prefetched_df is not None:
        base_col = prefetched_base_col or ('close_dn' if 'close_dn' in prefetched_df.columns else 'close')
        if prefetched_denoise_spec:
            try:
                prefetched_denoise_spec = _normalize_denoise_spec(prefetched_denoise_spec, default_when='pre_ti')
            except Exception:
                prefetched_denoise_spec = None
        return prefetched_df, base_col, prefetched_denoise_spec

    df = _fetch_history(symbol, timeframe, int(need), as_of)
    if len(df) < 3:
        raise ValueError("Not enough closed bars to compute forecast")

    base_col = 'close'
    dn_spec_used = None
    if denoise:
        try:
            normalized = _normalize_denoise_spec(denoise, default_when='pre_ti')
        except Exception:
            normalized = None
        added = _apply_denoise(df, normalized, default_when='pre_ti') if normalized else []
        dn_spec_used = normalized
        if len(added) > 0 and f"{base_col}_dn" in added:
            base_col = f"{base_col}_dn"
    return df, base_col, dn_spec_used


def _prepare_target_series_context(
    *,
    df: pd.DataFrame,
    quantity_l: str,
    base_col: str,
    features: Optional[Dict[str, Any]],
    target_spec: Optional[Dict[str, Any]],
) -> Tuple[pd.Series, str, str]:
    """Prepare the effective base column and target series consumed by forecasters."""
    base_col_initial = base_col
    base_col_prepared = _forecast_preprocessing._prepare_base_data(df, quantity_l, base_col)
    base_col_prepared = _forecast_preprocessing._apply_features_and_target_spec(
        df,
        features,
        target_spec,
        base_col_prepared,
        parse_kv_or_json=_parse_kv_or_json,
    )

    target_series = df[base_col_prepared].dropna()
    if target_spec:
        y_arr, target_info = build_target_series(df, base_col_initial, target_spec, quantity=quantity_l)
        target_series = pd.Series(y_arr, index=df.index)
        base_col_final = target_info.get('base', base_col_initial)
    else:
        base_col_final = base_col_prepared
        if quantity_l == 'return':
            target_series = df[base_col_final].dropna()
        else:
            target_series = df[base_col_final]

    target_series = target_series.dropna()
    return target_series, base_col_initial, base_col_final


def _prepare_feature_context(
    *,
    df: pd.DataFrame,
    features: Optional[Dict[str, Any]],
    exog_used: Optional[np.ndarray],
    exog_future: Optional[np.ndarray],
    tf_secs: int,
    horizon: int,
    target_series: pd.Series,
    dimred_method: Optional[str],
    dimred_params: Optional[Dict[str, Any]],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]:
    """Prepare training and future exogenous features if requested."""
    X = exog_used
    future_exog = exog_future
    feat_info: Dict[str, Any] = {}
    if X is None and features:
        future_times = next_times_from_last(float(df['time'].iloc[-1]), int(tf_secs), int(horizon))
        try:
            X, built_future_exog, feat_info = _forecast_preprocessing.prepare_features(
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
            X, built_future_exog, feat_info = None, None, {'error': f"feature_build_error: {str(exc)}"}
        if future_exog is None:
            future_exog = built_future_exog
    return X, future_exog, feat_info


def _build_engine_diagnostics(
    *,
    df: pd.DataFrame,
    need: int,
    lookback: Optional[int],
    seasonality: int,
    quantity_l: str,
    base_col: str,
    target_series: pd.Series,
) -> Dict[str, Any]:
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

    fmt_time = _format_time_minimal_local if _use_client_tz() else _format_time_minimal
    diagnostics: Dict[str, Any] = {
        "lookback_bars_requested": int(lookback) if lookback is not None else None,
        "lookback_bars_fetched": int(need),
        "history_bars_used": int(len(df)),
        "target_points_used": int(len(target_series)),
        "seasonality_used": int(seasonality),
        "quantity": quantity_l,
        "base_col_used": str(base_col),
    }
    if history_start_epoch is not None:
        diagnostics["history_start_epoch"] = history_start_epoch
        diagnostics["history_start_time"] = fmt_time(history_start_epoch)
    if history_end_epoch is not None:
        diagnostics["history_end_epoch"] = history_end_epoch
        diagnostics["history_end_time"] = fmt_time(history_end_epoch)
    return diagnostics


def _run_registered_forecast_method(
    *,
    method_l: str,
    method: ForecastMethodLiteral,
    df: pd.DataFrame,
    target_series: pd.Series,
    horizon: int,
    seasonality: int,
    params: Dict[str, Any],
    ci_alpha: Optional[float],
    as_of: Optional[str],
    quantity_l: str,
    symbol: str,
    timeframe: TimeframeLiteral,
    base_col: str,
    denoise_spec_used: Optional[Any],
    X: Optional[np.ndarray],
    future_exog: Optional[np.ndarray],
) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
    forecaster = ForecastRegistry.get(method_l)
    method_params = dict(params)
    if ci_alpha is not None and 'ci_alpha' not in method_params:
        method_params['ci_alpha'] = ci_alpha

    call_kwargs: Dict[str, Any] = {
        'ci_alpha': ci_alpha,
        'as_of': as_of,
        'quantity': quantity_l,
        'timeframe': timeframe,
    }
    if X is not None:
        call_kwargs['exog_used'] = X
    call_context = ForecastCallContext(
        method=method_l,
        symbol=symbol,
        timeframe=str(timeframe),
        quantity=quantity_l,
        horizon=int(horizon),
        seasonality=int(seasonality),
        base_col=str(base_col),
        ci_alpha=ci_alpha,
        as_of=as_of,
        denoise_spec_used=denoise_spec_used,
        history_df=df,
        target_series=target_series,
        exog_used=X,
        future_exog=future_exog,
    )
    prepare_call = getattr(forecaster, "prepare_forecast_call", None)
    if callable(prepare_call):
        method_params, call_kwargs = prepare_call(
            method_params,
            call_kwargs,
            call_context,
        )

    res = forecaster.forecast(
        target_series,
        horizon,
        seasonality,
        method_params,
        exog_future=future_exog,
        **call_kwargs,
    )
    metadata = res.metadata or {}
    metadata['params_used'] = res.params_used
    return res.forecast, res.ci_values, metadata


def _merge_engine_diagnostics(metadata: Dict[str, Any], diagnostics: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(metadata, dict):
        metadata = {}
    existing_diagnostics = metadata.get("diagnostics")
    if not isinstance(existing_diagnostics, dict):
        existing_diagnostics = {}
    merged_diagnostics = dict(existing_diagnostics)
    for key, value in diagnostics.items():
        if key not in merged_diagnostics:
            merged_diagnostics[key] = value
    metadata["diagnostics"] = merged_diagnostics
    return metadata


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
    future_epochs = next_times_from_last(last_epoch, tf_secs, horizon)
    fmt_time = _format_time_minimal_local if _use_client_tz() else _format_time_minimal
    forecast_times = [fmt_time(float(epoch)) for epoch in future_epochs]
    last_observation_time = fmt_time(float(last_epoch))

    # Build base result
    forecast_start_epoch = float(future_epochs[0]) if future_epochs else None
    forecast_start_gap_bars = (
        float(forecast_start_epoch - float(last_epoch)) / float(tf_secs)
        if forecast_start_epoch is not None and tf_secs
        else None
    )
    result: Dict[str, Any] = {
        "success": True,
        "method": method,
        "horizon": horizon,
        "base_col": base_col,
        "last_observation_epoch": float(last_epoch),
        "last_observation_time": last_observation_time,
        "forecast_start_epoch": forecast_start_epoch,
        "forecast_start_time": forecast_times[0] if forecast_times else None,
        "forecast_start_gap_bars": forecast_start_gap_bars,
        "forecast_anchor": "next_timeframe_bar_after_last_observation",
        "forecast_step_seconds": int(tf_secs),
        "forecast_epoch": future_epochs,
        "forecast_time": forecast_times,
    }

    # Choose which arrays to expose
    if quantity == 'return':
        if forecast_return_values is None:
            forecast_return_values = forecast_values
        result["forecast_return"] = [float(v) for v in forecast_return_values]
        if reconstructed_prices is not None:
            result["forecast_price"] = [float(v) for v in reconstructed_prices]
    else:
        result["forecast_price"] = [float(v) for v in forecast_values]
    
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
        if ci_alpha_value is not None:
            result["ci_alpha"] = ci_alpha_value

        if ci_values is not None and len(ci_values) == 2:  # [lower, upper]
            result["ci_status"] = "available"
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
            result["ci_status"] = "unavailable"

    # Add metadata
    result.update({
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
        ensemble_meta: Dict[str, Any] = {}
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
            return {"error": invalid_timeframe_error(timeframe, TIMEFRAME_MAP)}
        tf_secs = TIMEFRAME_SECONDS.get(timeframe)
        if not tf_secs:
            return {"error": unsupported_timeframe_seconds_error(timeframe)}

        method_l = str(method).lower().strip()
        quantity_l = str(quantity).lower().strip()
        
        # Refresh available methods
        available_methods = _get_available_methods()
        if method_l not in available_methods:
            return {"error": format_invalid_method_error(method, list(available_methods))}

        # Volatility models have a dedicated endpoint
        if quantity_l == 'volatility' or method_l.startswith('vol_'):
            return {"error": "Use forecast_volatility for volatility models"}

        # Parse method params
        p = _parse_kv_or_json(params)
        seasonality = int(p.get('seasonality')) if p.get('seasonality') is not None else default_seasonality(timeframe)

        if method_l == 'seasonal_naive' and (not seasonality or seasonality <= 0):
            return {"error": "seasonal_naive requires a positive 'seasonality' in params or auto period"}

        # Calculate lookback bars
        need = _calculate_lookback_bars(method_l, horizon, lookback, seasonality, timeframe, params=p)

        # Fetch data (or reuse prefetched) and optional denoise
        try:
            df, base_col, dn_spec_used = _resolve_history_context(
                symbol=symbol,
                timeframe=timeframe,
                need=need,
                as_of=as_of,
                prefetched_df=prefetched_df,
                prefetched_base_col=prefetched_base_col,
                prefetched_denoise_spec=prefetched_denoise_spec,
                denoise=denoise,
            )
        except ValueError as ex:
            return {"error": str(ex)}
        except Exception as ex:
            return {"error": str(ex)}

        # Track last close for potential price reconstruction
        try:
            last_close = float(df['close'].iloc[-1])
        except Exception:
            last_close = float('nan')

        # Prepare target series, honoring target_spec if provided
        try:
            target_series, base_col_initial, base_col = _prepare_target_series_context(
                df=df,
                quantity_l=quantity_l,
                base_col=base_col,
                features=features,
                target_spec=target_spec,
            )
        except Exception as ex:
            return {"error": f"Invalid target_spec: {ex}"}

        if len(target_series) < 3:
            return {"error": f"Not enough valid data points in column '{base_col}'"}

        # Prepare feature matrices if applicable (only if exog_used not provided).
        X, future_exog, feature_info = _prepare_feature_context(
            df=df,
            features=features,
            exog_used=exog_used,
            exog_future=exog_future,
            tf_secs=tf_secs,
            horizon=horizon,
            target_series=target_series,
            dimred_method=dimred_method,
            dimred_params=dimred_params,
        )

        # Get last timestamp and values
        last_epoch = float(df['time'].iloc[-1])

        # Core run diagnostics to make model context explicit for users.
        engine_diagnostics = _build_engine_diagnostics(
            df=df,
            need=need,
            lookback=lookback,
            seasonality=seasonality,
            quantity_l=quantity_l,
            base_col=base_col,
            target_series=target_series,
        )
        if feature_info:
            engine_diagnostics["feature_preparation"] = feature_info
        broker_time_check_result: Optional[Dict[str, Any]] = None
        broker_time_check_enabled = bool(getattr(mt5_config, "broker_time_check_enabled", False))
        broker_time_check_ttl_seconds = int(getattr(mt5_config, "broker_time_check_ttl_seconds", 60))
        if broker_time_check_enabled and prefetched_df is None and as_of is None:
            try:
                broker_time_check_result = get_cached_mt5_time_alignment(
                    symbol=symbol,
                    probe_timeframe='M1',
                    ttl_seconds=broker_time_check_ttl_seconds,
                )
            except Exception as exc:
                broker_time_check_result = {
                    "symbol": str(symbol),
                    "probe_timeframe": "M1",
                    "status": "unavailable",
                    "reason": "inspection_failed",
                    "error": str(exc),
                }
            engine_diagnostics["broker_time_check"] = broker_time_check_result

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
            forecast_values, ci_values, metadata = _run_registered_forecast_method(
                method_l=method_l,
                method=method,
                df=df,
                target_series=target_series,
                horizon=horizon,
                seasonality=seasonality,
                params=p,
                ci_alpha=ci_alpha,
                as_of=as_of,
                quantity_l=quantity_l,
                symbol=symbol,
                timeframe=timeframe,
                base_col=base_col,
                denoise_spec_used=dn_spec_used,
                X=X,
                future_exog=future_exog,
            )
        except ValueError as e:
            if method_l == 'ensemble':
                return {"error": str(e)}
            return {"error": f"Forecast method '{method}' failed: {str(e)}"}
        except Exception as e:
            return {"error": f"Forecast method '{method}' failed: {str(e)}"}

        if forecast_values is None:
            return {"error": f"Method '{method}' returned no forecast values"}

        metadata = _merge_engine_diagnostics(metadata, engine_diagnostics)

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
        if broker_time_check_result and broker_time_check_result.get("status") == "misaligned":
            warning_text = str(broker_time_check_result.get("warning") or "").strip()
            if warning_text:
                warnings = result.get("warnings")
                if not isinstance(warnings, list):
                    warnings = []
                if warning_text not in warnings:
                    warnings.append(warning_text)
                if warnings:
                    result["warnings"] = warnings
        if method_l == 'ensemble' and metadata:
            result['ensemble'] = metadata
        return result

    except Exception as e:
        return {"error": f"Forecast engine failed: {str(e)}"}
