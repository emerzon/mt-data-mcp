"""Tests for price reconstruction from forecast target transforms."""

from __future__ import annotations

import numpy as np
import pytest

from mtdata.forecast.forecast_engine import (
    _RECONSTRUCTION_MODES,
    _inverse_diff,
    _inverse_log_return,
    _inverse_pct,
    _inverse_return,
    _reconstruct_prices_from_target,
)


class TestInverseFunctions:
    """Unit tests for each inverse-transform function."""

    def test_inverse_log_return(self):
        # anchor=100, value=ln(1.05) → 100 * exp(ln(1.05)) = 105
        val = np.log(1.05)
        assert pytest.approx(_inverse_log_return(100.0, val), rel=1e-6) == 105.0

    def test_inverse_return(self):
        # anchor=100, value=0.05 → 100 * 1.05 = 105
        assert pytest.approx(_inverse_return(100.0, 0.05)) == 105.0

    def test_inverse_pct(self):
        # anchor=100, value=5.0 → 100 * (1 + 5/100) = 105
        assert pytest.approx(_inverse_pct(100.0, 5.0)) == 105.0

    def test_inverse_diff(self):
        # anchor=100, value=5 → 105
        assert pytest.approx(_inverse_diff(100.0, 5.0)) == 105.0

    def test_reconstruction_modes_registry(self):
        assert set(_RECONSTRUCTION_MODES) == {
            "log_return", "return", "pct_change", "pct", "diff",
        }


class TestReconstructPricesFromTarget:
    """Tests for _reconstruct_prices_from_target."""

    def test_none_transform_passthrough(self):
        history = np.array([100.0])
        forecast = np.array([101.0, 102.0])
        result = _reconstruct_prices_from_target(forecast, history, {"transform": "none"})
        np.testing.assert_array_equal(result, [101.0, 102.0])

    def test_log_transform(self):
        forecast = np.array([np.log(100.0), np.log(105.0)])
        result = _reconstruct_prices_from_target(forecast, np.array([1.0]), {"transform": "log"})
        np.testing.assert_allclose(result, [100.0, 105.0], rtol=1e-6)

    def test_log_return_reconstruction(self):
        # Two log-return steps: ln(105/100), ln(110/105)
        history = np.array([100.0])
        lr1 = np.log(105.0 / 100.0)
        lr2 = np.log(110.0 / 105.0)
        result = _reconstruct_prices_from_target(
            np.array([lr1, lr2]), history, {"transform": "log_return(k=1)"},
        )
        np.testing.assert_allclose(result, [105.0, 110.0], rtol=1e-6)

    def test_return_reconstruction(self):
        history = np.array([100.0])
        result = _reconstruct_prices_from_target(
            np.array([0.05, 0.10]), history, {"transform": "return(k=1)"},
        )
        np.testing.assert_allclose(result, [105.0, 115.5], rtol=1e-6)

    def test_pct_reconstruction(self):
        history = np.array([100.0])
        result = _reconstruct_prices_from_target(
            np.array([5.0, -2.0]), history, {"transform": "pct(k=1)"},
        )
        expected_1 = 105.0
        expected_2 = 105.0 * (1.0 - 2.0 / 100.0)
        np.testing.assert_allclose(result, [expected_1, expected_2], rtol=1e-6)

    def test_diff_reconstruction(self):
        history = np.array([100.0])
        result = _reconstruct_prices_from_target(
            np.array([5.0, -3.0]), history, {"transform": "diff(k=1)"},
        )
        np.testing.assert_allclose(result, [105.0, 102.0], rtol=1e-6)

    def test_nan_propagation_stops_chain(self):
        """Once NaN enters, all subsequent values should be NaN."""
        history = np.array([100.0])
        forecast = np.array([0.01, float("nan"), 0.02])
        result = _reconstruct_prices_from_target(
            forecast, history, {"transform": "log_return(k=1)"},
        )
        assert result is not None
        assert np.isfinite(result[0])
        assert np.isnan(result[1])
        assert np.isnan(result[2])  # propagated

    def test_empty_history_returns_none(self):
        result = _reconstruct_prices_from_target(np.array([0.01]), None, None)
        assert result is None

    def test_default_transform_is_log_return(self):
        history = np.array([100.0])
        lr = np.log(1.01)
        result = _reconstruct_prices_from_target(
            np.array([lr]), history, {},
        )
        np.testing.assert_allclose(result, [100.0 * np.exp(lr)], rtol=1e-6)

    def test_lag_greater_than_1(self):
        """With k=2, each step uses anchor from 2 steps back."""
        history = np.array([100.0, 105.0])
        result = _reconstruct_prices_from_target(
            np.array([0.10, 0.05]), history, {"transform": "return(k=2)"},
        )
        # step 0: anchor = history[-2] = 100, price = 100 * 1.10 = 110
        # step 1: anchor = history[-1] = 105, price = 105 * 1.05 = 110.25
        np.testing.assert_allclose(result, [110.0, 110.25], rtol=1e-6)

    def test_lagged_reconstruction_recovers_when_nan_breaks_unrelated_branch(self):
        history = np.array([100.0, 105.0])
        result = _reconstruct_prices_from_target(
            np.array([0.10, float("nan"), 0.05]),
            history,
            {"transform": "return(k=2)"},
        )

        assert result is not None
        assert result[0] == pytest.approx(110.0)
        assert np.isnan(result[1])
        assert result[2] == pytest.approx(115.5)

    def test_inf_from_reconstruction_propagates_nan(self):
        """If a reconstruction step produces inf, subsequent steps get NaN."""
        history = np.array([100.0])
        # Huge value that produces inf via exp
        forecast = np.array([1000.0, 0.01])
        result = _reconstruct_prices_from_target(
            forecast, history, {"transform": "log_return(k=1)"},
        )
        assert result is not None
        assert np.isinf(result[0]) or np.isnan(result[0])
        assert np.isnan(result[1])  # propagated
