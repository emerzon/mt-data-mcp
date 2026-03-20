"""Tests for forecast/monte_carlo.py — pure NumPy simulation functions."""
import numpy as np
import pytest
import warnings

from mtdata.forecast.monte_carlo import (
    _normalize_probability_vector,
    _normalize_transition_matrix,
    _safe_log,
    fit_gaussian_mixture_1d,
    estimate_transition_matrix_from_gamma,
    simulate_markov_chain,
    simulate_gbm_mc,
    simulate_hmm_mc,
    simulate_heston_mc,
    simulate_jump_diffusion_mc,
    simulate_garch_mc,
    simulate_bootstrap_mc,
    summarize_paths,
    gbm_single_barrier_upcross_prob,
)


class TestNormalizeProbabilityVector:
    def test_basic(self):
        result = _normalize_probability_vector(np.array([1.0, 3.0]))
        np.testing.assert_allclose(result, [0.25, 0.75])

    def test_all_zero(self):
        result = _normalize_probability_vector(np.array([0.0, 0.0, 0.0]))
        np.testing.assert_allclose(result, [1/3, 1/3, 1/3])

    def test_all_zero_with_fallback(self):
        result = _normalize_probability_vector(np.array([0.0, 0.0, 0.0]), fallback_index=1)
        np.testing.assert_allclose(result, [0.0, 1.0, 0.0])

    def test_negative_clipped(self):
        result = _normalize_probability_vector(np.array([-1.0, 2.0, 3.0]))
        assert result[0] == 0.0
        assert np.isclose(np.sum(result), 1.0)

    def test_nan_inf_clipped(self):
        result = _normalize_probability_vector(np.array([np.nan, np.inf, 2.0]))
        assert np.isclose(np.sum(result), 1.0)
        assert result[2] == 1.0


class TestNormalizeTransitionMatrix:
    def test_identity(self):
        A = np.eye(3)
        result = _normalize_transition_matrix(A)
        np.testing.assert_allclose(result, np.eye(3))

    def test_rows_sum_to_one(self):
        A = np.array([[1, 2], [3, 1]], dtype=float)
        result = _normalize_transition_matrix(A)
        for row in result:
            assert np.isclose(np.sum(row), 1.0)

    def test_degenerate_row(self):
        A = np.array([[0, 0], [1, 1]], dtype=float)
        result = _normalize_transition_matrix(A)
        # First row: zero → self-loop fallback
        assert result[0, 0] == 1.0
        assert np.isclose(np.sum(result[1]), 1.0)

    def test_non_square_fallback(self):
        A = np.array([[1, 2, 3]], dtype=float)
        result = _normalize_transition_matrix(A)
        assert result.shape[0] == result.shape[1]  # returns identity


class TestSafeLog:
    def test_positive(self):
        result = _safe_log(np.array([1.0, np.e]))
        np.testing.assert_allclose(result, [0.0, 1.0], atol=1e-10)

    def test_zero_clipped(self):
        result = _safe_log(np.array([0.0]))
        assert np.isfinite(result[0])


class TestFitGaussianMixture1d:
    def test_two_clusters(self):
        rng = np.random.RandomState(42)
        x = np.concatenate([rng.normal(-2, 0.3, 200), rng.normal(2, 0.3, 200)])
        w, mu, sigma, gamma, ll = fit_gaussian_mixture_1d(x, n_states=2, seed=42)
        assert len(w) == 2
        assert len(mu) == 2
        assert len(sigma) == 2
        assert gamma.shape == (400, 2)
        # Means should be roughly separated
        assert abs(mu[1] - mu[0]) > 2.0

    def test_degenerate_few_points(self):
        x = np.array([1.0])
        w, mu, sigma, gamma, ll = fit_gaussian_mixture_1d(x, n_states=3)
        assert len(w) == 1  # falls back to single Gaussian
        assert gamma.shape == (1, 1)

    def test_empty_input(self):
        x = np.array([])
        w, mu, sigma, gamma, ll = fit_gaussian_mixture_1d(x, n_states=2)
        assert len(w) == 1
        assert mu[0] == 0.0

    def test_convergence_warning_is_suppressed(self, monkeypatch):
        from sklearn.exceptions import ConvergenceWarning

        class DummyGaussianMixture:
            def __init__(self, n_components, covariance_type, reg_covar, max_iter, tol, n_init, random_state):
                self.n_components = int(n_components)

            def fit(self, x):
                warnings.warn("Best performing initialization did not converge.", ConvergenceWarning)
                return self

            def predict_proba(self, x):
                n = int(x.shape[0])
                out = np.zeros((n, self.n_components), dtype=float)
                out[:, 0] = 1.0
                return out

            @property
            def weights_(self):
                out = np.zeros(self.n_components, dtype=float)
                out[0] = 1.0
                return out

            @property
            def means_(self):
                return np.arange(self.n_components, dtype=float).reshape(-1, 1)

            @property
            def covariances_(self):
                return np.ones(self.n_components, dtype=float)

            def score(self, x):
                return -0.5

        monkeypatch.setattr("mtdata.forecast.monte_carlo.GaussianMixture", DummyGaussianMixture)

        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            _ = fit_gaussian_mixture_1d(np.linspace(-1.0, 1.0, 50), n_states=2, seed=1)

        assert not any(isinstance(w.message, ConvergenceWarning) for w in records)


class TestEstimateTransitionMatrix:
    def test_basic(self):
        gamma = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
        A = estimate_transition_matrix_from_gamma(gamma)
        assert A.shape == (2, 2)
        for row in A:
            assert np.isclose(np.sum(row), 1.0)

    def test_single_row(self):
        gamma = np.array([[0.5, 0.5]])
        A = estimate_transition_matrix_from_gamma(gamma)
        assert A.shape == (2, 2)


class TestSimulateMarkovChain:
    def test_basic_shape(self):
        A = np.array([[0.9, 0.1], [0.2, 0.8]])
        init = np.array([1.0, 0.0])
        states = simulate_markov_chain(A, init, steps=20, sims=10)
        assert states.shape == (10, 20)
        assert set(np.unique(states)).issubset({0, 1})

    def test_deterministic_seed(self):
        A = np.array([[0.5, 0.5], [0.5, 0.5]])
        init = np.array([1.0, 0.0])
        r1 = simulate_markov_chain(A, init, 10, 5, rng=np.random.RandomState(1))
        r2 = simulate_markov_chain(A, init, 10, 5, rng=np.random.RandomState(1))
        np.testing.assert_array_equal(r1, r2)


class TestSimulateGbmMc:
    def _prices(self, n=200, seed=42):
        rng = np.random.RandomState(seed)
        return 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))

    def test_output_keys(self):
        result = simulate_gbm_mc(self._prices(), horizon=10, n_sims=50, seed=1)
        assert "price_paths" in result
        assert "return_paths" in result
        assert "mu" in result
        assert "sigma" in result
        assert result["price_paths"].shape == (50, 10)

    def test_too_few_prices(self):
        with pytest.raises(ValueError, match="Not enough"):
            simulate_gbm_mc(np.array([1.0, 2.0]), horizon=5)

    def test_positive_prices(self):
        result = simulate_gbm_mc(self._prices(), horizon=20, n_sims=30, seed=7)
        assert np.all(result["price_paths"] > 0)

    def test_sigma_uses_sample_standard_deviation(self):
        prices = np.array([100.0, 101.0, 103.0, 102.0, 104.0], dtype=float)
        result = simulate_gbm_mc(prices, horizon=2, n_sims=4, seed=1)
        expected = np.std(np.diff(np.log(prices)), ddof=1) + 1e-12
        assert result["sigma"] == pytest.approx(expected)


class TestSimulateHmmMc:
    def _prices(self, n=200, seed=42):
        rng = np.random.RandomState(seed)
        return 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))

    def test_output_keys(self):
        result = simulate_hmm_mc(self._prices(), horizon=10, n_sims=30, seed=1)
        assert "price_paths" in result
        assert "state_paths" in result
        assert "trans" in result
        assert result["model_type"] == "gaussian_hmm_baum_welch"
        assert result["price_paths"].shape == (30, 10)

    def test_too_few(self):
        with pytest.raises(ValueError):
            simulate_hmm_mc(np.array([1.0, 2.0]), horizon=5)


class TestSimulateHestonMc:
    def _prices(self, n=200, seed=42):
        rng = np.random.RandomState(seed)
        return 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))

    def test_output_shape(self):
        result = simulate_heston_mc(self._prices(), horizon=10, n_sims=20, seed=1)
        assert result["price_paths"].shape == (20, 10)
        assert "vol_paths" in result
        assert "params" in result

    def test_custom_params(self):
        result = simulate_heston_mc(
            self._prices(), horizon=5, n_sims=10, seed=1,
            kappa=3.0, theta=0.02, xi=0.3, rho=-0.5, v0=0.01,
        )
        assert result["params"]["kappa"] == 3.0
        assert result["params"]["rho"] == -0.5

    def test_too_few(self):
        with pytest.raises(ValueError):
            simulate_heston_mc(np.array([1.0, 2.0, 3.0]), horizon=5)


class TestSimulateJumpDiffusionMc:
    def _prices(self, n=200, seed=42):
        rng = np.random.RandomState(seed)
        return 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))

    def test_output_shape(self):
        result = simulate_jump_diffusion_mc(self._prices(), horizon=10, n_sims=20, seed=1)
        assert result["price_paths"].shape == (20, 10)
        assert "params" in result

    def test_custom_jump_params(self):
        result = simulate_jump_diffusion_mc(
            self._prices(), horizon=5, n_sims=10, seed=1,
            jump_lambda=0.1, jump_mu=-0.02, jump_sigma=0.05,
        )
        assert result["params"]["jump_lambda"] == 0.1

    def test_too_few(self):
        with pytest.raises(ValueError):
            simulate_jump_diffusion_mc(np.array([1.0, 2.0, 3.0]), horizon=5)


class TestSimulateGarchMc:
    def _prices(self, n=300, seed=42):
        rng = np.random.RandomState(seed)
        return 100.0 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))

    def test_output_shape(self):
        result = simulate_garch_mc(self._prices(), horizon=10, n_sims=50, seed=1)
        assert result["price_paths"].shape == (50, 10)
        assert "return_paths" in result

    def test_too_few(self):
        with pytest.raises(ValueError, match="Not enough"):
            simulate_garch_mc(np.array([1.0] * 10), horizon=5)


class TestSimulateBootstrapMc:
    def _prices(self, n=200, seed=42):
        rng = np.random.RandomState(seed)
        return 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))

    def test_output_shape(self):
        result = simulate_bootstrap_mc(self._prices(), horizon=10, n_sims=30, seed=1)
        assert result["price_paths"].shape == (30, 10)
        assert "block_size" in result

    def test_custom_block_size(self):
        result = simulate_bootstrap_mc(self._prices(), horizon=5, n_sims=10, seed=1, block_size=5)
        assert int(result["block_size"]) == 5

    def test_short_bootstrap_generator_backfills_from_real_samples(self, monkeypatch):
        class _ShortBootstrap:
            def __init__(self, block_size, rets, seed=None):
                self._rets = np.asarray(rets, dtype=float)

            def bootstrap(self, n_sims):
                for _ in range(2):
                    yield ((self._rets[:3],),)

        monkeypatch.setattr("mtdata.forecast.monte_carlo._load_circular_block_bootstrap", lambda: _ShortBootstrap)

        result = simulate_bootstrap_mc(self._prices(), horizon=4, n_sims=5, seed=7, block_size=3)

        assert result["return_paths"].shape == (5, 4)
        assert not np.allclose(result["return_paths"][2:], 0.0)

    def test_too_few(self):
        with pytest.raises(ValueError):
            simulate_bootstrap_mc(np.array([1.0, 2.0, 3.0]), horizon=5)


class TestSummarizePaths:
    def test_price_only(self):
        paths = np.random.RandomState(1).uniform(90, 110, size=(100, 10))
        result = summarize_paths(paths)
        assert "price_mean" in result
        assert "price_lower" in result
        assert "price_upper" in result
        assert "return_mean" not in result
        assert result["price_mean"].shape == (10,)

    def test_with_returns(self):
        p = np.random.RandomState(1).uniform(90, 110, size=(50, 5))
        r = np.random.RandomState(2).normal(0, 0.01, size=(50, 5))
        result = summarize_paths(p, return_paths=r)
        assert "return_mean" in result
        assert result["return_lower"].shape == (5,)

    def test_ci_ordering(self):
        paths = np.random.RandomState(1).uniform(90, 110, size=(200, 10))
        result = summarize_paths(paths, alpha=0.1)
        assert np.all(result["price_lower"] <= result["price_mean"])
        assert np.all(result["price_mean"] <= result["price_upper"])


class TestGbmBarrierUpcrossProb:
    def test_at_barrier(self):
        p = gbm_single_barrier_upcross_prob(100.0, 100.0, 0.05, 0.2, 1.0)
        assert p == 1.0  # already at/above barrier

    def test_far_above(self):
        p = gbm_single_barrier_upcross_prob(100.0, 1e6, 0.0, 0.01, 1.0)
        assert p < 0.01

    def test_zero_sigma(self):
        # Deterministic: S0 * exp(mu * T) = 100 * exp(0.5) ≈ 164.87
        p = gbm_single_barrier_upcross_prob(100.0, 150.0, 0.5, 0.0, 1.0)
        assert p == 1.0

    def test_zero_sigma_no_reach(self):
        p = gbm_single_barrier_upcross_prob(100.0, 200.0, 0.1, 0.0, 1.0)
        assert p == 0.0

    def test_invalid_inputs(self):
        assert gbm_single_barrier_upcross_prob(0.0, 100.0, 0.05, 0.2, 1.0) == 0.0
        assert gbm_single_barrier_upcross_prob(100.0, 110.0, 0.05, 0.2, 0.0) == 0.0
        assert gbm_single_barrier_upcross_prob(float("nan"), 110.0, 0.05, 0.2, 1.0) == 0.0

    def test_reasonable_prob(self):
        p = gbm_single_barrier_upcross_prob(100.0, 110.0, 0.05, 0.2, 1.0)
        assert 0.0 < p < 1.0

    def test_extreme_drift_overflow_saturates_high_probability(self):
        p = gbm_single_barrier_upcross_prob(100.0, 101.0, 25.0, 0.01, 1.0)
        assert 0.9 <= p <= 1.0
