from __future__ import annotations

import numpy as np

from mtdata.utils.denoise.filters.adaptive import (
    _adaptive_lms_filter,
    _adaptive_rls_filter,
)


def _reference_lms(
    x: np.ndarray,
    *,
    order: int,
    mu: float,
    eps: float,
    leak: float,
    use_bias: bool,
) -> np.ndarray:
    k = max(1, int(order))
    if use_bias:
        weights = np.zeros(k + 1)
        weights[1:] = 1.0 / k
    else:
        weights = np.full(k, 1.0 / k)
    out = x.copy()
    for index in range(k, len(x)):
        taps = x[index - k : index][::-1]
        vector = np.concatenate(([1.0], taps)) if use_bias else taps
        estimate = float(weights @ vector)
        out[index] = estimate
        error = x[index] - estimate
        step = mu / (float(vector @ vector) + eps)
        weights = (1.0 - leak) * weights + step * error * vector
    return out


def _reference_rls(
    x: np.ndarray,
    *,
    order: int,
    lam: float,
    delta: float,
    use_bias: bool,
) -> np.ndarray:
    k = max(1, int(order))
    if use_bias:
        weights = np.zeros(k + 1)
        weights[1:] = 1.0 / k
        covariance = (1.0 / delta) * np.eye(k + 1)
    else:
        weights = np.full(k, 1.0 / k)
        covariance = (1.0 / delta) * np.eye(k)
    out = x.copy()
    for index in range(k, len(x)):
        taps = x[index - k : index][::-1]
        vector = np.concatenate(([1.0], taps)) if use_bias else taps
        projected = covariance @ vector
        gain = projected / (lam + float(vector @ projected))
        estimate = float(weights @ vector)
        out[index] = estimate
        weights = weights + gain * (x[index] - estimate)
        covariance = (
            covariance - np.outer(gain, vector) @ covariance
        ) / lam
    return out


def test_lms_preallocated_regressor_matches_reference() -> None:
    values = np.random.default_rng(10).normal(size=200)

    actual = _adaptive_lms_filter(
        values,
        order=5,
        mu=0.4,
        eps=1e-6,
        leak=0.01,
        use_bias=True,
    )
    expected = _reference_lms(
        values,
        order=5,
        mu=0.4,
        eps=1e-6,
        leak=0.01,
        use_bias=True,
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)


def test_rls_rank_one_update_matches_reference() -> None:
    values = np.random.default_rng(11).normal(size=200)

    actual = _adaptive_rls_filter(
        values,
        order=5,
        lam=0.99,
        delta=1.0,
        use_bias=True,
    )
    expected = _reference_rls(
        values,
        order=5,
        lam=0.99,
        delta=1.0,
        use_bias=True,
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-11)
