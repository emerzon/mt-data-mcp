"""Tests for utils/regime.py — BOCPD and Student-t helpers."""
import numpy as np
import pytest

from mtdata.utils.regime import bocpd_gaussian, _student_t_logpdf


class TestStudentTLogpdf:
    def test_basic_shape(self):
        x = np.array([0.0, 1.0, -1.0])
        mu = np.array([0.0, 0.0, 0.0])
        lam = np.array([1.0, 1.0, 1.0])
        alpha = np.array([1.0, 1.0, 1.0])
        beta = np.array([1.0, 1.0, 1.0])
        result = _student_t_logpdf(x, mu, lam, alpha, beta)
        assert result.shape == (3,)
        assert all(np.isfinite(result))

    def test_symmetric(self):
        mu = np.array([0.0])
        lam = np.array([1.0])
        alpha = np.array([2.0])
        beta = np.array([1.0])
        lp_pos = _student_t_logpdf(np.array([1.0]), mu, lam, alpha, beta)
        lp_neg = _student_t_logpdf(np.array([-1.0]), mu, lam, alpha, beta)
        np.testing.assert_allclose(lp_pos, lp_neg, atol=1e-10)

    def test_peak_at_mean(self):
        mu = np.array([5.0])
        lam = np.array([2.0])
        alpha = np.array([3.0])
        beta = np.array([1.0])
        lp_at_mu = _student_t_logpdf(np.array([5.0]), mu, lam, alpha, beta)
        lp_off = _student_t_logpdf(np.array([8.0]), mu, lam, alpha, beta)
        assert lp_at_mu > lp_off

    def test_scalar_input(self):
        result = _student_t_logpdf(0.0, 0.0, 1.0, 1.0, 1.0)
        assert np.isfinite(result)

    def test_zero_alpha_or_lambda_stays_finite(self):
        result = _student_t_logpdf(
            np.array([0.0, 0.0]),
            np.array([0.0, 0.0]),
            np.array([0.0, 1.0]),
            np.array([1.0, 0.0]),
            np.array([1.0, 1.0]),
        )
        assert np.all(np.isfinite(result))


class TestBocpdGaussian:
    def test_empty_input(self):
        result = bocpd_gaussian(np.array([]))
        assert result["cp_prob"].shape == (0,)
        assert result["run_length_map"].shape == (0,)

    def test_constant_series(self):
        x = np.ones(50)
        result = bocpd_gaussian(x)
        assert result["cp_prob"].shape == (50,)
        assert result["run_length_map"].shape == (50,)
        assert all(np.isfinite(result["cp_prob"]))
        # Constant series should have low CP probability after initial transient
        assert np.mean(result["cp_prob"][10:]) < 0.5

    def test_single_changepoint(self):
        rng = np.random.RandomState(42)
        seg1 = rng.normal(0.0, 0.1, 100)
        seg2 = rng.normal(5.0, 0.1, 100)
        x = np.concatenate([seg1, seg2])
        result = bocpd_gaussian(x, hazard_lambda=50)
        cp = result["cp_prob"]
        # Around index 100 there should be elevated CP probability
        peak_region = cp[95:110]
        baseline = np.mean(cp[20:80])
        assert np.max(peak_region) > baseline

    def test_nan_filtered(self):
        x = np.array([1.0, np.nan, 2.0, np.inf, 3.0])
        result = bocpd_gaussian(x)
        # NaN and inf are filtered: only 3 finite values remain
        assert result["cp_prob"].shape == (3,)

    def test_custom_params(self):
        x = np.random.RandomState(7).normal(0, 1, 30)
        result = bocpd_gaussian(
            x, hazard_lambda=10, max_run_length=50,
            mu0=0.0, kappa0=0.5, alpha0=2.0, beta0=2.0,
        )
        assert result["cp_prob"].shape == (30,)
        assert all(result["cp_prob"] >= 0)
        assert all(result["cp_prob"] <= 1)

    def test_run_length_map_nonneg(self):
        x = np.random.RandomState(99).normal(0, 1, 40)
        result = bocpd_gaussian(x)
        assert all(result["run_length_map"] >= 0)

    def test_hazard_lambda_zero_safe(self):
        """hazard_lambda=0 should not crash (clamped to 1)."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = bocpd_gaussian(x, hazard_lambda=0)
        assert result["cp_prob"].shape == (5,)

    def test_hazard_lambda_one_stays_finite(self):
        x = np.array([1.0, 1.1, 1.2, 1.3, 1.4])
        result = bocpd_gaussian(x, hazard_lambda=1)
        assert result["cp_prob"].shape == (5,)
        assert np.all(np.isfinite(result["cp_prob"]))
        assert np.all(np.isfinite(result["run_length_map"]))
