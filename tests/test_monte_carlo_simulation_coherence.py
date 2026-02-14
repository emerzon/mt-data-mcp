from __future__ import annotations

import os
import sys
import unittest

import numpy as np

# Add src to path to ensure local package is found
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from mtdata.forecast.method_adapters import MonteCarloMethodAdapter
from mtdata.forecast.monte_carlo import (
    estimate_transition_matrix_from_gamma,
    simulate_garch_mc,
    simulate_gbm_mc,
    simulate_markov_chain,
)


class TestMonteCarloSimulationCoherence(unittest.TestCase):
    def test_estimate_transition_matrix_handles_degenerate_rows(self) -> None:
        gamma = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], dtype=float)
        A = estimate_transition_matrix_from_gamma(gamma)
        self.assertTrue(np.isfinite(A).all())
        self.assertTrue(np.allclose(A.sum(axis=1), np.ones(A.shape[0])))
        self.assertTrue(np.allclose(A[1], np.array([0.0, 1.0])))

    def test_simulate_markov_chain_tolerates_nan_transition_rows(self) -> None:
        A = np.array([[1.0, 0.0], [np.nan, np.nan]], dtype=float)
        init = np.array([0.0, 1.0], dtype=float)
        states = simulate_markov_chain(A, init, steps=4, sims=3, rng=np.random.RandomState(1))
        self.assertEqual(states.shape, (3, 4))
        self.assertTrue(np.isin(states, [0, 1]).all())

    def test_simulate_gbm_mc_exposes_calibration_metadata(self) -> None:
        prices = np.linspace(100.0, 120.0, 200)
        sim = simulate_gbm_mc(prices=prices, horizon=5, n_sims=20, seed=2)
        self.assertIn("mu", sim)
        self.assertIn("sigma", sim)
        self.assertTrue(np.isfinite(float(sim["mu"])))
        self.assertTrue(np.isfinite(float(sim["sigma"])))

    def test_monte_carlo_method_adapter_matches_dict_simulator_contract(self) -> None:
        series = np.linspace(100.0, 120.0, 200)
        adapter = MonteCarloMethodAdapter("mc_gbm")
        forecast, meta = adapter.execute(series=series, fh=6, params={"n_sims": 25, "seed": 3})
        self.assertEqual(forecast.shape[0], 6)
        self.assertIn("mu", meta)
        self.assertIn("sigma", meta)

    def test_simulate_garch_mc_is_reproducible_with_seed(self) -> None:
        try:
            import arch  # noqa: F401
        except Exception:
            self.skipTest("arch package not installed")

        prices = np.exp(np.cumsum(np.random.RandomState(7).normal(0.0, 0.01, 500))) * 100.0
        a = simulate_garch_mc(prices=prices, horizon=5, n_sims=120, seed=42)
        b = simulate_garch_mc(prices=prices, horizon=5, n_sims=120, seed=42)
        self.assertTrue(np.allclose(a["price_paths"], b["price_paths"]))


if __name__ == "__main__":
    unittest.main()
