from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple
import inspect
import math
import numpy as np
import pandas as pd

from ..interface import ForecastMethod, ForecastResult
from ..registry import ForecastRegistry

_FAILURE_DETAIL_LIMIT = 12


def _normalize_weights_default(weights: Any, size: int) -> Optional[np.ndarray]:
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


def _clear_dispatch_error(dispatch_method: Any) -> None:
    try:
        setattr(dispatch_method, "_last_error", None)
    except Exception:
        pass


def _record_dispatch_error(dispatch_method: Any, method_name: str, exc: BaseException) -> None:
    try:
        setattr(
            dispatch_method,
            "_last_error",
            {
                "method": str(method_name),
                "error": str(exc),
                "error_type": type(exc).__name__,
            },
        )
    except Exception:
        pass


def _consume_dispatch_error(dispatch_method: Any) -> Optional[Dict[str, Any]]:
    try:
        error = getattr(dispatch_method, "_last_error", None)
    except Exception:
        return None
    _clear_dispatch_error(dispatch_method)
    return dict(error) if isinstance(error, dict) else None


def _append_failure(
    failures: Optional[List[Dict[str, Any]]],
    *,
    stage: str,
    method_name: str,
    error_detail: Optional[Dict[str, Any]] = None,
    anchor_index: Optional[int] = None,
) -> None:
    if failures is None or len(failures) >= _FAILURE_DETAIL_LIMIT:
        return
    payload: Dict[str, Any] = {
        "stage": str(stage),
        "method": str(method_name),
    }
    if anchor_index is not None:
        payload["anchor_index"] = int(anchor_index)
    if isinstance(error_detail, dict):
        for key, value in error_detail.items():
            if value is not None:
                payload[str(key)] = value
    failures.append(payload)


def _stabilized_bma_weights(rmse: np.ndarray) -> Optional[np.ndarray]:
    rmse_arr = np.asarray(rmse, dtype=float).reshape(-1)
    if rmse_arr.size == 0 or not np.all(np.isfinite(rmse_arr)):
        return None
    rmse_safe = np.maximum(rmse_arr, 1e-8)
    scale = float(np.median(rmse_safe))
    if not math.isfinite(scale) or scale <= 0.0:
        scale = float(np.mean(rmse_safe))
    scale = max(scale, 1e-8)
    log_weights = -0.5 * np.square(rmse_safe / scale)
    log_weights = log_weights - float(np.max(log_weights))
    weights = np.exp(log_weights)
    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 0.0:
        return None
    return weights / total


def _ensemble_dispatch_method_default(
    method_name: str,
    series: pd.Series,
    horizon: int,
    seasonality: Optional[int],
    params: Optional[Dict[str, Any]],
) -> Optional[np.ndarray]:
    method_l = str(method_name).lower().strip()
    _clear_dispatch_error(_ensemble_dispatch_method_default)
    try:
        forecaster = ForecastRegistry.get(method_l)
        res = forecaster.forecast(series, horizon, seasonality or 1, dict(params or {}))
        return res.forecast
    except Exception as ex:
        _record_dispatch_error(_ensemble_dispatch_method_default, method_l, ex)
        return None


def _prepare_ensemble_cv_default(
    series: pd.Series,
    methods: List[str],
    horizon: int,
    seasonality: Optional[int],
    params_map: Dict[str, Dict[str, Any]],
    cv_points: int,
    min_train: int,
    dispatch_method: Callable[[str, pd.Series, int, Optional[int], Optional[Dict[str, Any]]], Optional[np.ndarray]],
    failure_sink: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
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
        for method_name in methods:
            fc = dispatch_method(method_name, train, horizon, seasonality, params_map.get(method_name, {}))
            if fc is None:
                _append_failure(
                    failure_sink,
                    stage="cv",
                    method_name=method_name,
                    anchor_index=idx,
                    error_detail=_consume_dispatch_error(dispatch_method)
                    or {"error": "Component forecast unavailable", "error_type": "forecast_unavailable"},
                )
                success = False
                break
            try:
                fc_arr = np.asarray(fc, dtype=float).reshape(-1)
            except Exception as ex:
                _append_failure(
                    failure_sink,
                    stage="cv",
                    method_name=method_name,
                    anchor_index=idx,
                    error_detail={"error": str(ex), "error_type": type(ex).__name__},
                )
                success = False
                break
            if fc_arr.size < horizon_i or not np.all(np.isfinite(fc_arr[:horizon_i])):
                _append_failure(
                    failure_sink,
                    stage="cv",
                    method_name=method_name,
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


@ForecastRegistry.register("ensemble")
class EnsembleMethod(ForecastMethod):
    """Adaptive ensemble with averaging, Bayesian model averaging, or stacking."""

    PARAMS: List[Dict[str, Any]] = [
        {"name": "methods", "type": "list", "description": "Methods to ensemble (default: naive,theta,fourier_ols)"},
        {"name": "mode", "type": "str", "description": "average|bma|stacking (default: average)"},
        {"name": "weights", "type": "list", "description": "Manual weights when mode=average"},
        {"name": "cv_points", "type": "int", "description": "Walk-forward anchors for weighting (default: 2*len(methods))"},
        {"name": "min_train_size", "type": "int", "description": "Minimum history per CV anchor (default: max(30, horizon*3))"},
        {"name": "method_params", "type": "dict", "description": "Per-method parameter overrides"},
        {"name": "expose_components", "type": "bool", "description": "Include component forecasts in response (default: True)"},
    ]

    @property
    def name(self) -> str:
        return "ensemble"

    @property
    def category(self) -> str:
        return "ensemble"

    @property
    def supports_features(self) -> Dict[str, bool]:
        return {"price": True, "return": True, "volatility": True, "ci": False}

    def forecast(
        self,
        series: pd.Series,
        horizon: int,
        seasonality: int,
        params: Dict[str, Any],
        exog_future: Optional[pd.DataFrame] = None,
        **kwargs
    ) -> ForecastResult:
        dispatch_method = kwargs.get("ensemble_dispatch_method")
        if not callable(dispatch_method):
            dispatch_method = _ensemble_dispatch_method_default

        cv_failures: List[Dict[str, Any]] = []
        component_failures: List[Dict[str, Any]] = []

        prepare_cv = kwargs.get("prepare_ensemble_cv")
        if not callable(prepare_cv):
            def _prepare(series_in, methods, horizon_in, seasonality_in, params_map, cv_points, min_train):
                return _prepare_ensemble_cv_default(
                    series_in,
                    methods,
                    horizon_in,
                    seasonality_in,
                    params_map,
                    cv_points,
                    min_train,
                    dispatch_method,
                    failure_sink=cv_failures,
                )
            prepare_cv = _prepare

        normalize_weights = kwargs.get("normalize_weights")
        if not callable(normalize_weights):
            normalize_weights = _normalize_weights_default

        get_available_methods = kwargs.get("get_available_methods")
        if not callable(get_available_methods):
            get_available_methods = ForecastRegistry.get_all_method_names

        base_methods_in = params.get('methods')
        if isinstance(base_methods_in, str):
            base_methods = [m.strip().lower() for m in base_methods_in.split(',') if m.strip()]
        elif isinstance(base_methods_in, (list, tuple)):
            base_methods = [str(m).lower().strip() for m in base_methods_in if str(m).strip()]
        else:
            base_methods = ['naive', 'theta', 'fourier_ols']

        available_methods = get_available_methods()
        base_methods = [m for m in base_methods if m in available_methods and m != 'ensemble']

        seen: set[str] = set()
        base_methods = [m for m in base_methods if not (m in seen or seen.add(m))]
        if not base_methods:
            base_methods = ['naive', 'theta']

        params_in = params.get('method_params') if isinstance(params.get('method_params'), dict) else {}
        params_map = {str(k).lower(): (v if isinstance(v, dict) else {}) for k, v in params_in.items()}
        mode = str(params.get('mode', 'average')).lower()
        cv_points = int(params.get('cv_points', max(6, len(base_methods) * 2)))
        min_train = int(params.get('min_train_size', max(30, horizon * 3)))
        expose_components = bool(params.get('expose_components', True))
        weights_vec = normalize_weights(params.get('weights'), len(base_methods))

        ensemble_meta: Dict[str, Any] = {
            'mode_requested': mode,
            'methods': list(base_methods),
            'cv_points_requested': cv_points,
        }
        params_used: Dict[str, Any] = {
            'methods': list(base_methods),
            'mode': mode,
            'cv_points': cv_points,
            'min_train_size': min_train,
            'expose_components': expose_components,
        }
        if params.get('weights') is not None:
            params_used['weights'] = params.get('weights')
        if params_map:
            params_used['method_params'] = params_map

        effective_mode = mode
        rmse = None
        ensemble_intercept = 0.0
        coeffs = None
        cv_rows = 0
        if mode in ('bma', 'stacking'):
            prepare_kwargs: Dict[str, Any] = {}
            try:
                if "failure_sink" in inspect.signature(prepare_cv).parameters:
                    prepare_kwargs["failure_sink"] = cv_failures
            except (TypeError, ValueError):
                pass
            X_cv, y_cv = prepare_cv(
                series,
                base_methods,
                horizon,
                seasonality,
                params_map,
                cv_points,
                min_train,
                **prepare_kwargs,
            )
            cv_rows = int(len(y_cv))
            if X_cv.shape[0] >= max(3, len(base_methods)):
                if mode == 'bma':
                    errors = X_cv - y_cv[:, None]
                    rmse = np.sqrt(np.mean(np.square(errors), axis=0))
                    weights_vec = _stabilized_bma_weights(rmse)
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
        for method_name in base_methods:
            fc = dispatch_method(method_name, series, horizon, seasonality, params_map.get(method_name, {}))
            if fc is None:
                _append_failure(
                    component_failures,
                    stage="component",
                    method_name=method_name,
                    error_detail=_consume_dispatch_error(dispatch_method)
                    or {"error": "Component forecast unavailable", "error_type": "forecast_unavailable"},
                )
                continue
            try:
                fc_arr = np.asarray(fc, dtype=float).reshape(-1)
            except Exception as ex:
                _append_failure(
                    component_failures,
                    stage="component",
                    method_name=method_name,
                    error_detail={"error": str(ex), "error_type": type(ex).__name__},
                )
                continue
            if fc_arr.size < int(horizon) or not np.all(np.isfinite(fc_arr[: int(horizon)])):
                _append_failure(
                    component_failures,
                    stage="component",
                    method_name=method_name,
                    error_detail={"error": "Forecast output was too short or non-finite", "error_type": "invalid_forecast"},
                )
                continue
            component_methods.append(method_name)
            component_forecasts.append(fc_arr[: int(horizon)])

        if not component_forecasts:
            raise ValueError("Ensemble failed: no component forecasts")

        if len(component_methods) != len(base_methods):
            keep_idx = [base_methods.index(method_name) for method_name in component_methods]
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
            for weight, forecast in zip(weights_vec, component_forecasts):
                combined = combined + float(weight) * forecast
        else:
            if coeffs is None or coeffs.size != len(base_methods):
                coeffs = np.full(len(base_methods), 1.0 / len(base_methods))
                ensemble_intercept = 0.0
            combined = np.full_like(component_forecasts[0], ensemble_intercept, dtype=float)
            for weight, forecast in zip(coeffs, component_forecasts):
                combined = combined + float(weight) * forecast
            weights_vec = coeffs

        ensemble_meta.update({
            'mode_used': effective_mode,
            'methods': list(base_methods),
            'cv_points_used': cv_rows,
            'weights': [float(w) for w in (weights_vec.tolist() if isinstance(weights_vec, np.ndarray) else weights_vec)],
        })
        params_used['mode_used'] = effective_mode
        if rmse is not None:
            ensemble_meta['cv_rmse'] = [float(value) for value in rmse.tolist()]
        if effective_mode == 'stacking':
            ensemble_meta['intercept'] = float(ensemble_intercept)
        if cv_failures:
            ensemble_meta['cv_failures'] = cv_failures
        if component_failures:
            ensemble_meta['component_failures'] = component_failures
        if expose_components:
            ensemble_meta['components'] = {
                method_name: [float(value) for value in forecast.tolist()]
                for method_name, forecast in zip(base_methods, component_forecasts)
            }

        return ForecastResult(
            forecast=combined,
            params_used=params_used,
            metadata=ensemble_meta,
        )
