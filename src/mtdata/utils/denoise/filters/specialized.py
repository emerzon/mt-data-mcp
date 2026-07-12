"""Specialized filters: Kalman, Hampel, bilateral, TV denoising."""
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

try:
    from skimage.restoration import denoise_tv_chambolle as _denoise_tv_chambolle
except Exception:
    _denoise_tv_chambolle = None  # type: ignore[assignment]

from ..base import _series_like, register_filter


def _kalman_filter_1d(
    x: np.ndarray,
    process_var: float,
    measurement_var: float,
    initial_state: Optional[float] = None,
    initial_cov: Optional[float] = None,
) -> np.ndarray:
    n = len(x)
    xhat = np.zeros(n, dtype=float)
    p = np.zeros(n, dtype=float)
    meas = max(float(measurement_var), 1e-12)
    proc = max(float(process_var), 1e-12)
    xhat[0] = float(initial_state) if initial_state is not None else float(x[0])
    p[0] = float(initial_cov) if initial_cov is not None else meas
    for t in range(1, n):
        x_pred = xhat[t - 1]
        p_pred = p[t - 1] + proc
        k = p_pred / (p_pred + meas)
        xhat[t] = x_pred + k * (x[t] - x_pred)
        p[t] = (1 - k) * p_pred
    return xhat


def _kalman_filter_causal_auto_1d(
    x: np.ndarray,
    *,
    process_var: Optional[float],
    measurement_var: Optional[float],
    initial_state: Optional[float] = None,
    initial_cov: Optional[float] = None,
) -> np.ndarray:
    """Run a causal Kalman filter with expanding, prefix-only auto variances."""
    n = len(x)
    xhat = np.zeros(n, dtype=float)
    covariance = np.zeros(n, dtype=float)
    xhat[0] = float(initial_state) if initial_state is not None else float(x[0])
    initial_measurement = max(float(measurement_var or 1.0), 1e-12)
    covariance[0] = (
        float(initial_cov) if initial_cov is not None else initial_measurement
    )
    for t in range(1, n):
        prefix_variance = float(np.var(x[: t + 1]))
        measurement = max(
            float(measurement_var)
            if measurement_var is not None
            else (prefix_variance if prefix_variance > 0.0 else 1.0),
            1e-12,
        )
        process = max(
            float(process_var)
            if process_var is not None
            else measurement * 0.01,
            1e-12,
        )
        predicted_covariance = covariance[t - 1] + process
        gain = predicted_covariance / (predicted_covariance + measurement)
        xhat[t] = xhat[t - 1] + gain * (float(x[t]) - xhat[t - 1])
        covariance[t] = (1.0 - gain) * predicted_covariance
    return xhat


def _kalman_rts_smoother_1d(
    x: np.ndarray,
    process_var: float,
    measurement_var: float,
    initial_state: Optional[float] = None,
    initial_cov: Optional[float] = None,
) -> np.ndarray:
    """Run a scalar random-walk Kalman filter followed by RTS smoothing."""
    n = len(x)
    if n == 0:
        return np.asarray([], dtype=float)
    meas = max(float(measurement_var), 1e-12)
    proc = max(float(process_var), 1e-12)
    filtered = np.zeros(n, dtype=float)
    covariance = np.zeros(n, dtype=float)
    predicted = np.zeros(n, dtype=float)
    predicted_covariance = np.zeros(n, dtype=float)
    filtered[0] = float(initial_state) if initial_state is not None else float(x[0])
    covariance[0] = float(initial_cov) if initial_cov is not None else meas
    predicted[0] = filtered[0]
    predicted_covariance[0] = covariance[0]
    for t in range(1, n):
        predicted[t] = filtered[t - 1]
        predicted_covariance[t] = covariance[t - 1] + proc
        gain = predicted_covariance[t] / (predicted_covariance[t] + meas)
        filtered[t] = predicted[t] + gain * (float(x[t]) - predicted[t])
        covariance[t] = (1.0 - gain) * predicted_covariance[t]

    smoothed = filtered.copy()
    for t in range(n - 2, -1, -1):
        smoothing_gain = covariance[t] / max(predicted_covariance[t + 1], 1e-12)
        smoothed[t] = filtered[t] + smoothing_gain * (
            smoothed[t + 1] - predicted[t + 1]
        )
    return smoothed


@register_filter('kalman')
def _denoise_kalman_series(
    s: pd.Series,
    x: np.ndarray,
    params: Dict[str, Any],
    causality: str,
) -> pd.Series:
    measurement_var = params.get('measurement_var', params.get('r', 'auto'))
    process_var = params.get('process_var', params.get('q', 'auto'))
    measurement_auto = measurement_var == 'auto' or measurement_var is None
    process_auto = process_var == 'auto' or process_var is None
    if causality == 'causal' and (measurement_auto or process_auto):
        y = _kalman_filter_causal_auto_1d(
            x,
            process_var=None if process_auto else float(process_var),
            measurement_var=None if measurement_auto else float(measurement_var),
            initial_state=params.get('initial_state'),
            initial_cov=params.get('initial_cov'),
        )
        return _series_like(s, y)

    series_var = float(np.var(x))
    if measurement_auto:
        measurement_val = series_var if series_var > 0 else 1.0
    else:
        measurement_val = float(measurement_var)
    if process_auto:
        process_val = measurement_val * 0.01
    else:
        process_val = float(process_var)
    init_state = params.get('initial_state')
    init_cov = params.get('initial_cov')
    y_fwd = _kalman_filter_1d(
        x,
        process_var=process_val,
        measurement_var=measurement_val,
        initial_state=init_state,
        initial_cov=init_cov,
    )
    if causality == 'zero_phase':
        y = _kalman_rts_smoother_1d(
            x,
            process_var=process_val,
            measurement_var=measurement_val,
            initial_state=init_state,
            initial_cov=init_cov,
        )
    else:
        y = y_fwd
    return _series_like(s, y)


def _hampel_filter(
    x: np.ndarray,
    window: int,
    n_sigmas: float,
    causality: str,
) -> np.ndarray:
    n = len(x)
    if n < 3:
        return x
    win = max(3, int(window))
    half = win // 2
    y = x.copy()
    for i in range(n):
        if causality == 'causal':
            start = max(0, i - win + 1)
            end = i + 1
        else:
            start = max(0, i - half)
            end = min(n, i + half + 1)
        vals = x[start:end]
        if len(vals) == 0:
            continue
        med = float(np.median(vals))
        mad = float(np.median(np.abs(vals - med)))
        scale = 1.4826 * mad if mad > 0 else 0.0
        if scale > 0 and abs(x[i] - med) > float(n_sigmas) * scale:
            y[i] = med
    return y


@register_filter('hampel')
def _denoise_hampel_series(
    s: pd.Series,
    x: np.ndarray,
    params: Dict[str, Any],
    causality: str,
) -> pd.Series:
    window = int(params.get('window', 7))
    n_sigmas = float(params.get('n_sigmas', 3.0))
    y = _hampel_filter(x, window=window, n_sigmas=n_sigmas, causality=causality)
    return _series_like(s, y)


def _bilateral_filter_1d(
    x: np.ndarray,
    sigma_s: float,
    sigma_r: float,
    truncate: float,
    causality: str,
) -> np.ndarray:
    n = len(x)
    if n < 3:
        return x
    if sigma_s <= 0 or sigma_r <= 0:
        return x
    radius = max(1, int(round(float(truncate) * float(sigma_s))))
    y = np.zeros_like(x)
    for i in range(n):
        if causality == 'causal':
            start = max(0, i - radius)
            end = i + 1
        else:
            start = max(0, i - radius)
            end = min(n, i + radius + 1)
        idx = np.arange(start, end)
        if idx.size == 0:
            y[i] = x[i]
            continue
        dist = idx - i
        w_s = np.exp(-0.5 * (dist / float(sigma_s)) ** 2)
        w_r = np.exp(-0.5 * ((x[idx] - x[i]) / float(sigma_r)) ** 2)
        w = w_s * w_r
        denom = np.sum(w)
        y[i] = np.sum(w * x[idx]) / denom if denom > 0 else x[i]
    return y


@register_filter('bilateral')
def _denoise_bilateral_series(
    s: pd.Series,
    x: np.ndarray,
    params: Dict[str, Any],
    causality: str,
) -> pd.Series:
    sigma_s = float(params.get('sigma_s', 2.0))
    sigma_r = float(params.get('sigma_r', 0.5))
    truncate = float(params.get('truncate', 3.0))
    y = _bilateral_filter_1d(x, sigma_s=sigma_s, sigma_r=sigma_r, truncate=truncate, causality=causality)
    return _series_like(s, y)


def _tv_denoise_1d(
    x: np.ndarray,
    weight: float,
    n_iter: int = 50,
    tol: float = 1e-4,
) -> np.ndarray:
    if weight <= 0:
        return x
    n = len(x)
    if n < 3:
        return x
    if _denoise_tv_chambolle is None:
        raise RuntimeError("TV denoise requires scikit-image")
    try:
        y = _denoise_tv_chambolle(
            x,
            weight=float(weight),
            eps=float(max(tol, 1e-12)),
            max_num_iter=max(1, int(n_iter)),
            channel_axis=None,
        )
    except TypeError:
        y = _denoise_tv_chambolle(
            x,
            weight=float(weight),
            eps=float(max(tol, 1e-12)),
            n_iter_max=max(1, int(n_iter)),
        )
    return np.asarray(y, dtype=float)


@register_filter('tv')
def _denoise_tv_series(
    s: pd.Series,
    x: np.ndarray,
    params: Dict[str, Any],
    causality: str,
) -> pd.Series:
    del causality
    weight = params.get('weight', params.get('lambda', 'auto'))
    if weight == 'auto' or weight is None:
        scale = float(np.std(x))
        weight_val = 0.1 * scale if scale > 0 else 1.0
    else:
        weight_val = float(weight)
    n_iter = int(params.get('n_iter', 50))
    tol = float(params.get('tol', 1e-4))
    y = _tv_denoise_1d(x, weight=weight_val, n_iter=n_iter, tol=tol)
    return _series_like(s, y)
