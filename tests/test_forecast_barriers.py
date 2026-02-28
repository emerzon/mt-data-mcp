import importlib.util
import unittest
from unittest.mock import patch
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from mtdata.forecast.barriers import forecast_barrier_hit_probabilities, forecast_barrier_closed_form, forecast_barrier_optimize
from mtdata.forecast.monte_carlo import gbm_single_barrier_upcross_prob

class TestForecastBarriers(unittest.TestCase):

    def setUp(self):
        # Patch pip size resolver in barriers module
        self.pip_size_patcher = patch('mtdata.forecast.barriers._get_pip_size', return_value=0.0001)
        self.pip_size_patcher.start()
        
        # Mock fetch_history
        self.fetch_history_patcher = patch('mtdata.forecast.barriers._fetch_history')
        self.mock_fetch_history = self.fetch_history_patcher.start()
        
        # Create dummy dataframe
        dates = pd.date_range(start='2023-01-01', periods=500, freq='h')
        prices = np.linspace(1.0, 1.1, 500) + np.random.normal(0, 0.001, 500)
        self.df = pd.DataFrame({'time': dates, 'close': prices})
        self.mock_fetch_history.return_value = self.df

    def _set_flat_history(self, price: float = 1.0, bars: int = 200):
        dates = pd.date_range(start='2023-01-01', periods=bars, freq='h')
        closes = np.full(bars, float(price))
        self.mock_fetch_history.return_value = pd.DataFrame({'time': dates, 'close': closes})

    def _sample_paths(self):
        return np.array([
            [1.0, 1.01, 1.02, 1.03],
            [1.0, 0.99, 0.98, 0.97],
            [1.0, 1.002, 0.998, 1.006],
        ])

    def tearDown(self):
        self.pip_size_patcher.stop()
        self.fetch_history_patcher.stop()

    def test_forecast_barrier_hit_probabilities(self):
        result = forecast_barrier_hit_probabilities(
            symbol="EURUSD",
            timeframe="H1",
            horizon=10,
            method="mc_gbm",
            direction="long",
            tp_pct=0.5,
            sl_pct=0.5
        )
        self.assertIn("success", result)
        self.assertTrue(result["success"])
        self.assertIn("prob_tp_first", result)
        self.assertIn("prob_sl_first", result)

    def test_forecast_barrier_bootstrap(self):
        result = forecast_barrier_hit_probabilities(
            symbol="EURUSD",
            timeframe="H1",
            horizon=10,
            method="bootstrap",
            direction="long",
            tp_pct=0.5,
            sl_pct=0.5,
            params={"block_size": 5}
        )
        self.assertIn("success", result)
        self.assertTrue(result["success"])
        self.assertEqual(result["method"], "bootstrap")

    def test_forecast_barrier_garch(self):
        if importlib.util.find_spec("arch") is None:
            self.skipTest("arch package not installed")
            
        result = forecast_barrier_hit_probabilities(
            symbol="EURUSD",
            timeframe="H1",
            horizon=10,
            method="garch",
            direction="long",
            tp_pct=0.5,
            sl_pct=0.5
        )
        self.assertIn("success", result)
        self.assertTrue(result["success"])
        self.assertEqual(result["method"], "garch")

    def test_forecast_barrier_closed_form(self):
        result = forecast_barrier_closed_form(
            symbol="EURUSD",
            timeframe="H1",
            horizon=10,
            direction="long",
            barrier=1.2
        )
        self.assertIn("success", result)
        self.assertTrue(result["success"])
        self.assertIn("prob_hit", result)

    def test_gbm_single_barrier_upcross_prob_returns_one_when_barrier_below_start(self):
        self.assertAlmostEqual(
            gbm_single_barrier_upcross_prob(
                s0=1.0,
                barrier=0.5,
                mu=0.0,
                sigma=0.2,
                T=1.0,
            ),
            1.0,
            places=12,
        )

    def test_forecast_barrier_closed_form_returns_one_when_barrier_already_hit(self):
        up = forecast_barrier_closed_form(
            symbol="EURUSD",
            timeframe="H1",
            horizon=10,
            direction="long",
            barrier=0.5,
        )
        self.assertTrue(up["success"])
        self.assertAlmostEqual(up["prob_hit"], 1.0, places=12)

        down = forecast_barrier_closed_form(
            symbol="EURUSD",
            timeframe="H1",
            horizon=10,
            direction="short",
            barrier=10.0,
        )
        self.assertTrue(down["success"])
        self.assertAlmostEqual(down["prob_hit"], 1.0, places=12)

    def test_forecast_barrier_optimize(self):
        result = forecast_barrier_optimize(
            symbol="EURUSD",
            timeframe="H1",
            horizon=10,
            method="mc_gbm",
            direction="long",
            mode="pct",
            tp_min=0.1, tp_max=0.5, tp_steps=3,
            sl_min=0.1, sl_max=0.5, sl_steps=3,
            objective="edge"
        )
        self.assertIn("success", result)
        self.assertTrue(result["success"])
        self.assertIn("best", result)
        self.assertIn("grid", result)

    def test_forecast_barrier_optimize_refine_and_metrics(self):
        # Deterministic paths to verify refine pass and ranking
        self._set_flat_history(1.0)
        paths = np.array([
            [1.0, 1.01, 1.02, 1.03],
            [1.0, 0.99, 0.98, 0.97],
            [1.0, 1.002, 0.998, 1.006],
        ])
        with patch('mtdata.forecast.barriers._simulate_gbm_mc') as mock_sim:
            mock_sim.return_value = {"price_paths": paths}
            result = forecast_barrier_optimize(
                symbol="EURUSD",
                timeframe="H1",
                horizon=4,
                method="mc_gbm",
                direction="long",
                mode="pct",
                tp_min=0.5, tp_max=1.0, tp_steps=2,
                sl_min=0.5, sl_max=1.0, sl_steps=2,
                objective="kelly",
                refine=True,
                refine_radius=0.4,
                refine_steps=3,
                return_grid=True,
            )
        self.assertTrue(result["success"])
        grid = result.get("grid")
        self.assertTrue(grid)
        # Base grid has 4 combos; refine should add more unique candidates
        self.assertGreater(len(grid), 4)
        self.assertIn("prob_no_hit", grid[0])
        self.assertIn("t_hit_tp_median", grid[0])
        self.assertIn("prob_resolve", grid[0])
        self.assertIn("ev_cond", grid[0])
        self.assertIn("kelly_cond", grid[0])
        self.assertIn("ev_per_bar", grid[0])
        self.assertIn("profit_factor", grid[0])
        self.assertIn("utility", grid[0])
        self.assertIn("t_hit_resolve_median", grid[0])
        kelly_vals = [g["kelly"] for g in grid]
        self.assertEqual(kelly_vals, sorted(kelly_vals, reverse=True))

    def test_forecast_barrier_optimize_summary_truncates_grid(self):
        self._set_flat_history(1.0)
        paths = np.array([
            [1.0, 1.01, 1.02, 1.03],
            [1.0, 0.99, 0.98, 0.97],
            [1.0, 1.002, 0.998, 1.006],
        ])
        with patch('mtdata.forecast.barriers._simulate_gbm_mc') as mock_sim:
            mock_sim.return_value = {"price_paths": paths}
            result = forecast_barrier_optimize(
                symbol="EURUSD",
                timeframe="H1",
                horizon=4,
                method="mc_gbm",
                direction="long",
                mode="pct",
                tp_min=0.5, tp_max=1.0, tp_steps=2,
                sl_min=0.5, sl_max=1.0, sl_steps=2,
                objective="edge",
                refine=False,
                output="summary",
                top_k=2,
                return_grid=True,
            )
        self.assertTrue(result["success"])
        grid = result.get("grid")
        self.assertTrue(grid)
        self.assertEqual(len(grid), 2)
        self.assertEqual(result["best"], grid[0])

    def test_forecast_barrier_optimize_volatility_grid(self):
        self._set_flat_history(1.0)
        paths = self._sample_paths()
        with patch('mtdata.forecast.barriers._simulate_gbm_mc') as mock_sim:
            mock_sim.return_value = {"price_paths": paths}
            result = forecast_barrier_optimize(
                symbol="EURUSD",
                timeframe="H1",
                horizon=4,
                method="mc_gbm",
                direction="long",
                mode="pct",
                grid_style="volatility",
                vol_steps=2,
                return_grid=True,
            )
        self.assertTrue(result["success"])
        grid = result.get("grid")
        self.assertTrue(grid)
        self.assertIn("tp", grid[0])
        self.assertIn("sl", grid[0])

    def test_forecast_barrier_optimize_ratio_grid(self):
        self._set_flat_history(1.0)
        paths = self._sample_paths()
        with patch('mtdata.forecast.barriers._simulate_gbm_mc') as mock_sim:
            mock_sim.return_value = {"price_paths": paths}
            result = forecast_barrier_optimize(
                symbol="EURUSD",
                timeframe="H1",
                horizon=4,
                method="mc_gbm",
                direction="long",
                mode="pct",
                grid_style="ratio",
                ratio_min=1.0,
                ratio_max=2.0,
                ratio_steps=2,
                sl_min=0.5,
                sl_max=1.0,
                sl_steps=2,
                return_grid=True,
            )
        self.assertTrue(result["success"])
        grid = result.get("grid")
        self.assertTrue(grid)
        for entry in grid:
            self.assertGreaterEqual(entry["rr"], 1.0)
            self.assertLessEqual(entry["rr"], 2.0)

    def test_forecast_barrier_optimize_constraints(self):
        self._set_flat_history(1.0)
        paths = self._sample_paths()
        with patch('mtdata.forecast.barriers._simulate_gbm_mc') as mock_sim:
            mock_sim.return_value = {"price_paths": paths}
            result = forecast_barrier_optimize(
                symbol="EURUSD",
                timeframe="H1",
                horizon=4,
                method="mc_gbm",
                direction="long",
                mode="pct",
                tp_min=0.2, tp_max=0.2, tp_steps=1,
                sl_min=2.0, sl_max=2.0, sl_steps=1,
                objective="prob_resolve",
                min_prob_win=0.5,
                max_prob_no_hit=0.2,
                max_median_time=2,
                return_grid=True,
            )
        self.assertTrue(result["success"])
        grid = result.get("grid")
        self.assertTrue(grid)
        for entry in grid:
            self.assertGreaterEqual(entry["prob_win"], 0.5)
            self.assertLessEqual(entry["prob_no_hit"], 0.2)
            if entry.get("t_hit_resolve_median") is not None:
                self.assertLessEqual(entry["t_hit_resolve_median"], 2)

    def test_forecast_barrier_optimize_preset_grid(self):
        self._set_flat_history(1.0)
        paths = self._sample_paths()
        with patch('mtdata.forecast.barriers._simulate_gbm_mc') as mock_sim:
            mock_sim.return_value = {"price_paths": paths}
            result = forecast_barrier_optimize(
                symbol="EURUSD",
                timeframe="H1",
                horizon=4,
                method="mc_gbm",
                direction="long",
                mode="pct",
                grid_style="preset",
                preset="scalp",
                return_grid=True,
            )
        self.assertTrue(result["success"])
        grid = result.get("grid")
        self.assertTrue(grid)

    def test_forecast_barrier_optimize_pips_mode(self):
        self._set_flat_history(1.0)
        paths = self._sample_paths()
        with patch('mtdata.forecast.barriers._simulate_gbm_mc') as mock_sim:
            mock_sim.return_value = {"price_paths": paths}
            result = forecast_barrier_optimize(
                symbol="EURUSD",
                timeframe="H1",
                horizon=4,
                method="mc_gbm",
                direction="long",
                mode="pips",
                tp_min=5.0, tp_max=10.0, tp_steps=2,
                sl_min=5.0, sl_max=10.0, sl_steps=2,
                return_grid=True,
            )
        self.assertTrue(result["success"])
        grid = result.get("grid")
        self.assertTrue(grid)

    def test_forecast_barrier_optimize_tie_probabilities_sum_to_one(self):
        # Force TP and SL to coincide at zero so every path ties.
        self._set_flat_history(0.0)
        paths = np.zeros((3, 4))
        with patch('mtdata.forecast.barriers._simulate_gbm_mc') as mock_sim:
            mock_sim.return_value = {"price_paths": paths}
            result = forecast_barrier_optimize(
                symbol="EURUSD",
                timeframe="H1",
                horizon=4,
                method="mc_gbm",
                direction="long",
                mode="pct",
                tp_min=0.5, tp_max=1.0, tp_steps=2,
                sl_min=0.5, sl_max=1.0, sl_steps=2,
                objective="edge",
                return_grid=True,
            )
        self.assertTrue(result["success"])
        grid = result.get("grid")
        self.assertTrue(grid)
        for entry in grid:
            total = entry["prob_win"] + entry["prob_loss"] + entry["prob_tie"] + entry["prob_no_hit"]
            self.assertAlmostEqual(total, 1.0, places=7)
            self.assertAlmostEqual(entry["prob_tie"], 1.0, places=7)
            self.assertAlmostEqual(entry["prob_no_hit"], 0.0, places=7)
            self.assertAlmostEqual(entry["prob_tp_first"], 0.5, places=7)
            self.assertAlmostEqual(entry["prob_sl_first"], 0.5, places=7)

    def test_forecast_barrier_hit_probabilities_rejects_non_positive_horizon(self):
        result = forecast_barrier_hit_probabilities(
            symbol="EURUSD",
            timeframe="H1",
            horizon=0,
            method="mc_gbm",
            direction="long",
            tp_pct=0.5,
            sl_pct=0.5,
        )
        self.assertIn("error", result)
        self.assertIn("Invalid horizon", result["error"])

    def test_forecast_barrier_hit_probabilities_normalizes_direction_aliases(self):
        result = forecast_barrier_hit_probabilities(
            symbol="EURUSD",
            timeframe="H1",
            horizon=5,
            method="mc_gbm",
            direction="up",
            tp_pct=0.5,
            sl_pct=0.5,
        )
        self.assertTrue(result.get("success"))
        self.assertEqual(result.get("direction"), "long")

    def test_forecast_barrier_hit_probabilities_rejects_invalid_direction(self):
        result = forecast_barrier_hit_probabilities(
            symbol="EURUSD",
            timeframe="H1",
            horizon=5,
            method="mc_gbm",
            direction="sideways",
            tp_pct=0.5,
            sl_pct=0.5,
        )
        self.assertIn("error", result)
        self.assertIn("Invalid direction", result["error"])

    def test_forecast_barrier_optimize_rejects_invalid_mode(self):
        result = forecast_barrier_optimize(
            symbol="EURUSD",
            timeframe="H1",
            horizon=4,
            method="mc_gbm",
            direction="long",
            mode="invalid",
        )
        self.assertIn("error", result)
        self.assertIn("Invalid mode", result["error"])

    def test_forecast_barrier_optimize_rejects_invalid_top_k(self):
        result = forecast_barrier_optimize(
            symbol="EURUSD",
            timeframe="H1",
            horizon=4,
            method="mc_gbm",
            direction="long",
            mode="pct",
            top_k=0,
        )
        self.assertIn("error", result)
        self.assertIn("Invalid top_k", result["error"])

    def test_forecast_barrier_optimize_reports_objective_fallback(self):
        result = forecast_barrier_optimize(
            symbol="EURUSD",
            timeframe="H1",
            horizon=4,
            method="mc_gbm",
            direction="long",
            mode="pct",
            objective="not_a_real_objective",
        )
        self.assertTrue(result.get("success"))
        self.assertEqual(result.get("objective"), "edge")
        self.assertEqual(result.get("objective_requested"), "not_a_real_objective")
        self.assertEqual(result.get("objective_used"), "edge")

    def test_forecast_barrier_optimize_flags_no_candidates(self):
        result = forecast_barrier_optimize(
            symbol="EURUSD",
            timeframe="H1",
            horizon=4,
            method="mc_gbm",
            direction="long",
            mode="pct",
            tp_min=0.1,
            tp_max=0.2,
            tp_steps=2,
            sl_min=0.1,
            sl_max=0.2,
            sl_steps=2,
            params={"rr_min": 1000},
            return_grid=True,
        )
        self.assertTrue(result.get("success"))
        self.assertTrue(result.get("no_candidates"))
        self.assertEqual(result.get("results"), [])
        self.assertEqual(result.get("grid"), [])
        self.assertIn("warning", result)

if __name__ == '__main__':
    unittest.main()
