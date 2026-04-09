"""Adaptive filters: LMS and RLS."""
from typing import Any, Dict

import numpy as np
import pandas as pd

from ..base import _series_like, register_filter


def _adaptive_lms_filter(
    x: np.ndarray,
    order: int,
    mu: float,
    eps: float = 1e-6,
    leak: float = 0.0,
    use_bias: bool = True,
) -> np.ndarray:
    n = len(x)
    k = max(1, int(order))
    mu_val = float(mu)
    if n < k + 1 or mu_val <= 0:
        return x
    leak_val = max(0.0, float(leak))
    if use_bias:
        w = np.zeros(k + 1, dtype=float)
        w[1:] = 1.0 / float(k)
    else:
        w = np.full(k, 1.0 / float(k), dtype=float)
    y = x.copy()
    for t in range(k, n):
        if use_bias:
            x_vec = np.concatenate(([1.0], x[t - k : t][::-1]))
        else:
            x_vec = x[t - k : t][::-1]
        y_hat = float(np.dot(w, x_vec))
        y[t] = y_hat
        err = x[t] - y_hat
        denom = float(np.dot(x_vec, x_vec)) + float(eps)
        step = mu_val / denom
        w = (1.0 - leak_val) * w + step * err * x_vec
    return y


@register_filter('lms')
def _denoise_lms_series(
    s: pd.Series,
    x: np.ndarray,
    params: Dict[str, Any],
    causality: str,
) -> pd.Series:
    order = int(params.get('order', 5))
    mu_param = params.get('mu', 'auto')
    mu = float(mu_param) if mu_param not in ('auto', None) else 0.5
    mu = max(1e-4, min(mu, 1.5))
    eps = float(params.get('eps', 1e-6))
    leak = float(params.get('leak', 0.0))
    bias = bool(params.get('bias', True))
    y_fwd = _adaptive_lms_filter(x, order=order, mu=mu, eps=eps, leak=leak, use_bias=bias)
    if causality == 'zero_phase':
        y_bwd = _adaptive_lms_filter(x[::-1], order=order, mu=mu, eps=eps, leak=leak, use_bias=bias)[::-1]
        y = 0.5 * (y_fwd + y_bwd)
    else:
        y = y_fwd
    return _series_like(s, y)


def _adaptive_rls_filter(
    x: np.ndarray,
    order: int,
    lam: float,
    delta: float,
    use_bias: bool = True,
) -> np.ndarray:
    n = len(x)
    k = max(1, int(order))
    lam_val = float(lam)
    if n < k + 1 or lam_val <= 0 or lam_val > 1:
        return x
    delta_val = max(float(delta), 1e-6)
    if use_bias:
        w = np.zeros(k + 1, dtype=float)
        w[1:] = 1.0 / float(k)
        p = (1.0 / delta_val) * np.eye(k + 1, dtype=float)
    else:
        w = np.full(k, 1.0 / float(k), dtype=float)
        p = (1.0 / delta_val) * np.eye(k, dtype=float)
    y = x.copy()
    for t in range(k, n):
        if use_bias:
            x_vec = np.concatenate(([1.0], x[t - k : t][::-1]))
        else:
            x_vec = x[t - k : t][::-1]
        px = p @ x_vec
        denom = lam_val + float(np.dot(x_vec, px))
        if denom <= 0:
            y[t] = x[t]
            continue
        k_gain = px / denom
        y_hat = float(np.dot(w, x_vec))
        y[t] = y_hat
        err = x[t] - y_hat
        w = w + k_gain * err
        p = (p - np.outer(k_gain, x_vec) @ p) / lam_val
    return y


@register_filter('rls')
def _denoise_rls_series(
    s: pd.Series,
    x: np.ndarray,
    params: Dict[str, Any],
    causality: str,
) -> pd.Series:
    order = int(params.get('order', 5))
    lam_param = params.get('lambda_', params.get('lam', 'auto'))
    lam = float(lam_param) if lam_param not in ('auto', None) else 0.99
    lam = max(0.9, min(lam, 0.999))
    delta = float(params.get('delta', 1.0))
    bias = bool(params.get('bias', True))
    y_fwd = _adaptive_rls_filter(x, order=order, lam=lam, delta=delta, use_bias=bias)
    if causality == 'zero_phase':
        y_bwd = _adaptive_rls_filter(x[::-1], order=order, lam=lam, delta=delta, use_bias=bias)[::-1]
        y = 0.5 * (y_fwd + y_bwd)
    else:
        y = y_fwd
    return _series_like(s, y)
