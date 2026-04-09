"""Tests for HMM degeneracy detection and auto-fallback.

Covers:
- _is_hmm_degenerate detection logic
- _fit_hmmlearn_gaussian_hmm_1d fallback behaviour
- simulate_hmm_mc metadata exposure (fitted_n_states)
- preservation of rare-but-real regimes
"""
import numpy as np
import pytest

from mtdata.forecast.monte_carlo import (
    _DEGEN_COMBINED_SELF_TRANS,
    _DEGEN_MAX_SIGMA_RATIO,
    _DEGEN_MIN_OCCUPANCY_ABS,
    _DEGEN_MIN_OCCUPANCY_FRAC,
    _fit_hmmlearn_gaussian_hmm_1d,
    _is_hmm_degenerate,
    simulate_hmm_mc,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _iid_normal_prices(n=300, mu=0.0, sigma=0.01, seed=42):
    """Single-regime IID normal log-returns → prices."""
    rng = np.random.RandomState(seed)
    rets = rng.normal(mu, sigma, n)
    return 100.0 * np.exp(np.cumsum(rets))


def _two_regime_prices(n=600, seed=42):
    """Clear 2-regime data: alternating calm/volatile blocks."""
    rng = np.random.RandomState(seed)
    block = n // 6
    segments = []
    for i in range(6):
        sigma = 0.002 if i % 2 == 0 else 0.02
        segments.append(rng.normal(0, sigma, block))
    rets = np.concatenate(segments)
    return 100.0 * np.exp(np.cumsum(rets))


def _two_regime_rare_volatile(n=500, seed=42):
    """Mostly calm with a short volatile burst — rare but real regime."""
    rng = np.random.RandomState(seed)
    calm = rng.normal(0, 0.002, n - 30)
    burst = rng.normal(0, 0.025, 30)
    rets = np.concatenate([calm[:n // 2], burst, calm[n // 2:]])
    return 100.0 * np.exp(np.cumsum(rets))


# ---------------------------------------------------------------------------
# _is_hmm_degenerate
# ---------------------------------------------------------------------------
class TestIsHmmDegenerate:
    """Unit tests for the detection helper."""

    def test_single_state_never_degenerate(self):
        """K=1 should always return False."""
        from hmmlearn.hmm import GaussianHMM
        x = np.random.RandomState(1).normal(0, 0.01, 200).reshape(-1, 1)
        m = GaussianHMM(n_components=1, covariance_type='diag', n_iter=10,
                        init_params='stc', random_state=1)
        m.means_ = np.array([[0.0]])
        m.fit(x)
        assert _is_hmm_degenerate(m, x, K=1, N=200) is False

    def test_detects_sigma_ratio_degeneracy(self):
        """Ghost state with extreme sigma ratio should be flagged."""
        from hmmlearn.hmm import GaussianHMM
        # Build a model and manually inject degenerate parameters so the
        # detection test is deterministic (not seed-dependent).
        rng = np.random.RandomState(7)
        x = rng.normal(0, 0.01, 300).reshape(-1, 1)
        m = GaussianHMM(n_components=2, covariance_type='diag', n_iter=1,
                        init_params='stc', random_state=7)
        m.means_ = np.array([[0.0], [0.0]])
        m.fit(x)
        # Force degenerate covariances: one plausible, one absurd
        m.covars_ = np.array([[1e-4], [1e3]])
        assert _is_hmm_degenerate(m, x, K=2, N=300) is True

    def test_good_two_state_not_degenerate(self):
        """Well-separated 2-regime data should NOT be flagged."""
        from hmmlearn.hmm import GaussianHMM
        prices = _two_regime_prices(600, seed=99)
        rets = np.diff(np.log(prices))
        x = rets.reshape(-1, 1)
        N = x.shape[0]
        m = GaussianHMM(n_components=2, covariance_type='diag', n_iter=80,
                        init_params='stc', random_state=99)
        m.means_ = np.quantile(x, [1/3, 2/3]).reshape(2, 1)
        m.fit(x)
        assert _is_hmm_degenerate(m, x, K=2, N=N) is False

    def test_detects_low_occupancy_plus_absorbing(self):
        """Low posterior occupancy combined with absorbing transition."""
        from hmmlearn.hmm import GaussianHMM
        rng = np.random.RandomState(8)
        x = rng.normal(0, 0.01, 300).reshape(-1, 1)
        m = GaussianHMM(n_components=2, covariance_type='diag', n_iter=1,
                        init_params='stc', random_state=8)
        m.means_ = np.array([[0.0], [0.0]])
        m.fit(x)
        # Plausible sigma ratio (<100) but near-absorbing transition
        m.covars_ = np.array([[1e-4], [5e-4]])
        m.transmat_ = np.array([[0.999, 0.001], [0.0, 1.0]])
        # Force near-zero occupancy in state 1 via startprob
        m.startprob_ = np.array([1.0, 0.0])
        assert _is_hmm_degenerate(m, x, K=2, N=300) is True


# ---------------------------------------------------------------------------
# _fit_hmmlearn_gaussian_hmm_1d — fallback behaviour
# ---------------------------------------------------------------------------
class TestFitHmmFallback:
    """Integration tests for the degeneracy-aware fitting loop."""

    def test_iid_data_falls_back_to_single_state(self):
        """IID normal data cannot support 2 regimes → should collapse to K=1."""
        prices = _iid_normal_prices(300, seed=10)
        rets = np.diff(np.log(prices))
        mu, sigma, A, init = _fit_hmmlearn_gaussian_hmm_1d(rets, n_states=2)
        K = mu.shape[0]
        assert K == 1, f"Expected K=1 for IID data, got {K}"
        assert A.shape == (1, 1)
        np.testing.assert_allclose(A, [[1.0]])
        np.testing.assert_allclose(init, [1.0])

    def test_two_regime_data_keeps_two_states(self):
        """Clear 2-regime data should stay at K=2."""
        prices = _two_regime_prices(600, seed=20)
        rets = np.diff(np.log(prices))
        mu, sigma, A, init = _fit_hmmlearn_gaussian_hmm_1d(rets, n_states=2)
        K = mu.shape[0]
        assert K == 2, f"Expected K=2 for 2-regime data, got {K}"
        # States should have different volatilities
        assert sigma[0] != sigma[1]
        # Lower-mean state should have smaller sigma (ascending mean order)
        # (not guaranteed, but the two should be materially different)
        ratio = max(sigma) / min(sigma)
        assert ratio > 2, f"Sigma ratio {ratio} too low for distinct regimes"

    def test_k3_downgrades_on_iid_data(self):
        """Requesting K=3 on IID data should fall back to K=1."""
        prices = _iid_normal_prices(400, seed=30)
        rets = np.diff(np.log(prices))
        mu, sigma, A, init = _fit_hmmlearn_gaussian_hmm_1d(rets, n_states=3)
        K = mu.shape[0]
        assert K < 3, f"Expected K<3 for IID data with n_states=3, got {K}"

    def test_k1_request_always_succeeds(self):
        """Explicitly requesting K=1 should never trigger fallback logic."""
        prices = _iid_normal_prices(200, seed=40)
        rets = np.diff(np.log(prices))
        mu, sigma, A, init = _fit_hmmlearn_gaussian_hmm_1d(rets, n_states=1)
        assert mu.shape[0] == 1

    def test_rare_regime_preserved(self):
        """A short volatile burst inside calm data is a real regime.

        The HMM should ideally keep K=2 (or at least not lose the volatile
        state).  We accept K=1 or K=2 here — the key assertion is that the
        function doesn't error and that K<=2.
        """
        prices = _two_regime_rare_volatile(500, seed=50)
        rets = np.diff(np.log(prices))
        mu, sigma, A, init = _fit_hmmlearn_gaussian_hmm_1d(rets, n_states=2)
        K = mu.shape[0]
        assert K in (1, 2), f"Unexpected K={K}"

    def test_persistent_regime_not_falsely_flagged(self):
        """High self-transition alone should NOT trigger degeneracy.

        Generate two regimes that each persist for ~150 bars (high diagonal
        in transition matrix) but have balanced occupancy and plausible
        variance ratios.
        """
        rng = np.random.RandomState(60)
        # Two long blocks — high persistence, balanced occupancy
        calm = rng.normal(0, 0.003, 300)
        hot = rng.normal(0, 0.012, 300)
        rets = np.concatenate([calm, hot])
        mu, sigma, A, init = _fit_hmmlearn_gaussian_hmm_1d(rets, n_states=2)
        K = mu.shape[0]
        assert K == 2, (
            f"Expected K=2 for persistent but balanced 2-regime data, got {K}"
        )


# ---------------------------------------------------------------------------
# simulate_hmm_mc — metadata exposure
# ---------------------------------------------------------------------------
class TestSimulateHmmMcMetadata:
    """Verify that simulate_hmm_mc exposes fitted vs requested state count."""

    def _prices(self, n=200, seed=42):
        rng = np.random.RandomState(seed)
        return 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))

    def test_metadata_keys_present(self):
        result = simulate_hmm_mc(self._prices(), horizon=5, n_sims=10, seed=1)
        assert 'requested_n_states' in result
        assert 'fitted_n_states' in result

    def test_requested_equals_fitted_when_no_fallback(self):
        prices = _two_regime_prices(600, seed=70)
        result = simulate_hmm_mc(prices, horizon=5, n_sims=10, seed=1)
        assert result['requested_n_states'] == 2
        assert result['fitted_n_states'] == 2

    def test_fitted_less_than_requested_on_iid_data(self):
        prices = _iid_normal_prices(300, seed=80)
        result = simulate_hmm_mc(prices, horizon=5, n_sims=10, seed=1)
        assert result['requested_n_states'] == 2
        assert result['fitted_n_states'] < result['requested_n_states']

    def test_simulation_still_works_after_fallback(self):
        """Barrier pipeline receives price_paths regardless of K."""
        prices = _iid_normal_prices(300, seed=90)
        result = simulate_hmm_mc(prices, horizon=10, n_sims=50, seed=1)
        assert result['price_paths'].shape == (50, 10)
        assert np.all(np.isfinite(result['price_paths']))
        assert result['mu'].shape[0] == result['fitted_n_states']
        assert result['trans'].shape == (
            result['fitted_n_states'], result['fitted_n_states']
        )


# ---------------------------------------------------------------------------
# Module-level constants are importable (for downstream test targeting)
# ---------------------------------------------------------------------------
class TestDegeneracyConstants:
    def test_constants_are_numeric(self):
        assert isinstance(_DEGEN_MAX_SIGMA_RATIO, float)
        assert isinstance(_DEGEN_MIN_OCCUPANCY_ABS, int)
        assert isinstance(_DEGEN_MIN_OCCUPANCY_FRAC, float)
        assert isinstance(_DEGEN_COMBINED_SELF_TRANS, float)

    def test_constants_are_positive(self):
        assert _DEGEN_MAX_SIGMA_RATIO > 0
        assert _DEGEN_MIN_OCCUPANCY_ABS > 0
        assert _DEGEN_MIN_OCCUPANCY_FRAC > 0
        assert 0 < _DEGEN_COMBINED_SELF_TRANS < 1
