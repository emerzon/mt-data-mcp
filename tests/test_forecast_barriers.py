import unittest
from unittest.mock import patch
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from mtdata.forecast.barriers import forecast_barrier_hit_probabilities, forecast_barrier_closed_form, forecast_barrier_optimize

class TestForecastBarriers(unittest.TestCase):

    def setUp(self):
        # Patch pip size resolver in barriers module
        self.pip_size_patcher = patch('mtdata.forecast.barriers._get_pip_size', return_value=0.0001)
        self.pip_size_patcher.start()
        
        # Mock fetch_history
        self.fetch_history_patcher = patch('mtdata.forecast.barriers._fetch_history')
        self.mock_fetch_history = self.fetch_history_patcher.start()
        
        # Create dummy dataframe
        dates = pd.date_range(start='2023-01-01', periods=500, freq='H')
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
        try:
            import arch
        except ImportError:
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
            direction="up",
            barrier=1.2
        )
        self.assertIn("success", result)
        self.assertTrue(result["success"])
        self.assertIn("prob_hit", result)

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

if __name__ == '__main__':
    unittest.main()
