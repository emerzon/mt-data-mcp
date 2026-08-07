from __future__ import annotations

import numpy as np

from mtdata.utils.denoise.filters import decomposition
from mtdata.utils.denoise.filters.adaptive import (
    _adaptive_lms_filter,
    _adaptive_rls_filter,
)
from mtdata.utils.denoise.filters.specialized import (
    _bilateral_filter_1d,
    _hampel_filter,
    _kalman_filter_causal_auto_1d,
)
from mtdata.utils.denoise.filters.trend import _beta_smooth


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


def _reference_hampel(
    values: np.ndarray,
    *,
    window: int,
    n_sigmas: float,
    causality: str,
) -> np.ndarray:
    out = values.copy()
    half = window // 2
    for index in range(len(values)):
        if causality == "causal":
            start, stop = max(0, index - window + 1), index + 1
        else:
            start, stop = max(0, index - half), min(len(values), index + half + 1)
        sample = values[start:stop]
        median = float(np.median(sample))
        mad = float(np.median(np.abs(sample - median)))
        scale = 1.4826 * mad if mad > 0.0 else 0.0
        if scale > 0.0 and abs(values[index] - median) > n_sigmas * scale:
            out[index] = median
    return out


def _reference_bilateral(
    values: np.ndarray,
    *,
    sigma_s: float,
    sigma_r: float,
    truncate: float,
    causality: str,
) -> np.ndarray:
    radius = max(1, int(round(truncate * sigma_s)))
    out = np.zeros_like(values)
    for index in range(len(values)):
        start = max(0, index - radius)
        stop = (
            index + 1
            if causality == "causal"
            else min(len(values), index + radius + 1)
        )
        positions = np.arange(start, stop)
        spatial = np.exp(-0.5 * ((positions - index) / sigma_s) ** 2)
        ranges = np.exp(-0.5 * ((values[positions] - values[index]) / sigma_r) ** 2)
        weights = spatial * ranges
        out[index] = np.sum(weights * values[positions]) / np.sum(weights)
    return out


def _reference_beta(
    values: np.ndarray,
    *,
    window: int,
    beta: float,
    n_iter: int,
    eps: float,
    causality: str,
) -> np.ndarray:
    from mtdata.utils.denoise.filters.trend import _beta_irls_mean

    out = values.copy()
    half = window // 2
    for index in range(len(values)):
        if causality == "causal":
            start, stop = max(0, index - window + 1), index + 1
        else:
            start, stop = max(0, index - half), min(len(values), index + half + 1)
        out[index] = _beta_irls_mean(
            values[start:stop],
            beta=beta,
            n_iter=n_iter,
            eps=eps,
        )
    return out


def test_vectorized_hampel_matches_reference_for_both_causalities() -> None:
    values = np.random.default_rng(12).normal(size=100)
    values[[20, 70]] = [15.0, -12.0]
    for causality in ("causal", "zero_phase"):
        actual = _hampel_filter(values, 7, 3.0, causality)
        expected = _reference_hampel(
            values,
            window=7,
            n_sigmas=3.0,
            causality=causality,
        )
        np.testing.assert_allclose(actual, expected)


def test_vectorized_bilateral_matches_reference_for_both_causalities() -> None:
    values = np.random.default_rng(13).normal(size=100)
    for causality in ("causal", "zero_phase"):
        actual = _bilateral_filter_1d(values, 2.0, 0.5, 3.0, causality)
        expected = _reference_bilateral(
            values,
            sigma_s=2.0,
            sigma_r=0.5,
            truncate=3.0,
            causality=causality,
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)


def test_running_variance_kalman_matches_prefix_variance_reference() -> None:
    values = np.random.default_rng(14).normal(size=200)
    actual = _kalman_filter_causal_auto_1d(
        values,
        process_var=None,
        measurement_var=None,
    )

    expected = np.zeros_like(values)
    covariance = np.zeros_like(values)
    expected[0] = values[0]
    covariance[0] = 1.0
    for index in range(1, len(values)):
        measurement = max(float(np.var(values[: index + 1])), 1e-12)
        process = max(measurement * 0.01, 1e-12)
        predicted = covariance[index - 1] + process
        gain = predicted / (predicted + measurement)
        expected[index] = expected[index - 1] + gain * (
            values[index] - expected[index - 1]
        )
        covariance[index] = (1.0 - gain) * predicted

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_vectorized_beta_smoother_matches_windowed_irls() -> None:
    values = np.random.default_rng(15).normal(size=100)
    for causality in ("causal", "zero_phase"):
        actual = _beta_smooth(values, 9, 1.3, 20, 1e-6, causality)
        expected = _reference_beta(
            values,
            window=9,
            beta=1.3,
            n_iter=20,
            eps=1e-6,
            causality=causality,
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_truncated_ssa_matches_full_svd_reconstruction() -> None:
    values = np.random.default_rng(16).normal(size=120)
    window = 30
    components = 2
    trajectory = np.column_stack(
        [values[index : index + window] for index in range(len(values) - window + 1)]
    )
    left, singular, right = np.linalg.svd(trajectory, full_matrices=False)
    reconstructed = (
        left[:, :components] * singular[:components]
    ) @ right[:components, :]
    expected = np.zeros(len(values))
    counts = np.zeros(len(values))
    for row in range(reconstructed.shape[0]):
        for column in range(reconstructed.shape[1]):
            expected[row + column] += reconstructed[row, column]
            counts[row + column] += 1.0
    expected /= counts

    actual = decomposition._ssa_denoise(values, window, components)

    np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-8)


def test_ssa_auto_window_is_bounded_but_explicit_window_is_preserved(monkeypatch) -> None:
    observed = []

    def fake_ssa(values, window, components):
        observed.append(window)
        return values

    monkeypatch.setattr(decomposition, "_ssa_denoise", fake_ssa)
    series = decomposition.pd.Series(np.arange(1200, dtype=float))

    decomposition._denoise_ssa_series(series, series.to_numpy(), {}, "zero_phase")
    decomposition._denoise_ssa_series(
        series,
        series.to_numpy(),
        {"window": 400},
        "zero_phase",
    )

    assert observed == [decomposition._SSA_DEFAULT_MAX_WINDOW, 400]
