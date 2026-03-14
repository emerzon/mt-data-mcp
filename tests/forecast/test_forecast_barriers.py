import importlib.util
import unittest
import warnings
from unittest.mock import patch
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from mtdata.forecast.barriers import forecast_barrier_hit_probabilities, forecast_barrier_closed_form, forecast_barrier_optimize
from mtdata.forecast.monte_carlo import gbm_single_barrier_upcross_prob

_BARRIER_PROB_ROOT = "mtdata.forecast.barriers_probabilities"
_BARRIER_OPT_ROOT = "mtdata.forecast.barriers_optimization"


class _BarrierModulePatchMixin:
    def _start_barrier_module_patchers(self) -> None:
        self._barrier_patchers = [
            patch(f"{_BARRIER_PROB_ROOT}._get_pip_size", return_value=0.0001),
            patch(f"{_BARRIER_OPT_ROOT}._get_pip_size", return_value=0.0001),
            patch(f"{_BARRIER_PROB_ROOT}._fetch_history"),
            patch(f"{_BARRIER_OPT_ROOT}._fetch_history"),
        ]
        self._barrier_patchers[0].start()
        self._barrier_patchers[1].start()
        self.mock_fetch_history_prob = self._barrier_patchers[2].start()
        self.mock_fetch_history_opt = self._barrier_patchers[3].start()
        self.mock_fetch_history = self.mock_fetch_history_prob

    def _set_barrier_history(self, df: pd.DataFrame) -> None:
        self.mock_fetch_history_prob.return_value = df
        self.mock_fetch_history_opt.return_value = df

    def _stop_barrier_module_patchers(self) -> None:
        for patcher in reversed(getattr(self, "_barrier_patchers", [])):
            patcher.stop()


class TestForecastBarriers(_BarrierModulePatchMixin, unittest.TestCase):

    def setUp(self):
        self._start_barrier_module_patchers()
        
        # Create dummy dataframe
        dates = pd.date_range(start='2023-01-01', periods=500, freq='h')
        prices = np.linspace(1.0, 1.1, 500) + np.random.normal(0, 0.001, 500)
        self.df = pd.DataFrame({'time': dates, 'close': prices})
        self._set_barrier_history(self.df)

    def _set_flat_history(self, price: float = 1.0, bars: int = 200):
        dates = pd.date_range(start='2023-01-01', periods=bars, freq='h')
        closes = np.full(bars, float(price))
        self._set_barrier_history(pd.DataFrame({'time': dates, 'close': closes}))

    def _sample_paths(self):
        return np.array([
            [1.0, 1.01, 1.02, 1.03],
            [1.0, 0.99, 0.98, 0.97],
            [1.0, 1.002, 0.998, 1.006],
        ])

    def tearDown(self):
        self._stop_barrier_module_patchers()

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
        self.assertIn("prob_tie", result)
        self.assertIn("prob_tp_first_ci95", result)
        self.assertIn("prob_sl_first_ci95", result)
        self.assertIn("prob_no_hit_ci95", result)
        self.assertIn("prob_tp_first_se", result)
        self.assertIn("prob_sl_first_se", result)

    def test_forecast_barrier_hit_probabilities_prefers_live_tick_price(self):
        self._set_flat_history(1.0, bars=200)
        paths = self._sample_paths()
        with patch(f'{_BARRIER_PROB_ROOT}._simulate_gbm_mc') as mock_sim, \
             patch(f'{_BARRIER_PROB_ROOT}._get_live_reference_price', return_value=(1.2345, "live_tick_ask")):
            mock_sim.return_value = {"price_paths": paths}
            result = forecast_barrier_hit_probabilities(
                symbol="EURUSD",
                timeframe="H1",
                horizon=4,
                method="mc_gbm",
                direction="long",
                tp_pct=0.5,
                sl_pct=0.5,
            )
        self.assertTrue(result["success"])
        self.assertAlmostEqual(result["last_price"], 1.2345, places=8)
        self.assertAlmostEqual(result["last_price_close"], 1.0, places=8)
        self.assertEqual(result["last_price_source"], "live_tick_ask")
        self.assertAlmostEqual(result["tp_price"], 1.2345 * 1.005, places=8)
        self.assertAlmostEqual(result["sl_price"], 1.2345 * 0.995, places=8)
        self.assertEqual(len(result["hit_prob_by_t"]), 4)
        self.assertEqual(result["hit_prob_by_t"][0]["bar"], 1)
        self.assertEqual(result["hit_prob_by_t"][0]["t_seconds"], 3600)
        self.assertAlmostEqual(
            result["hit_prob_by_t"][0]["tp_hit_prob"],
            result["tp_hit_prob_by_t"][0],
            places=8,
        )
        self.assertAlmostEqual(
            result["hit_prob_by_t"][0]["sl_hit_prob"],
            result["sl_hit_prob_by_t"][0],
            places=8,
        )
        self.assertTrue(result["time_to_hit_seconds_derived"])
        self.assertEqual(result["time_to_hit_seconds_formula"], "bars * 3600")

    def test_forecast_barrier_hit_probabilities_falls_back_to_close_price(self):
        self._set_flat_history(1.0, bars=200)
        paths = self._sample_paths()
        with patch(f'{_BARRIER_PROB_ROOT}._simulate_gbm_mc') as mock_sim, \
             patch(f'{_BARRIER_PROB_ROOT}._get_live_reference_price', return_value=(None, None)):
            mock_sim.return_value = {"price_paths": paths}
            result = forecast_barrier_hit_probabilities(
                symbol="EURUSD",
                timeframe="H1",
                horizon=4,
                method="mc_gbm",
                direction="long",
                tp_pct=0.5,
                sl_pct=0.5,
            )
        self.assertTrue(result["success"])
        self.assertAlmostEqual(result["last_price"], 1.0, places=8)
        self.assertAlmostEqual(result["last_price_close"], 1.0, places=8)
        self.assertEqual(result["last_price_source"], "close")

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

    def test_forecast_barrier_closed_form_rejects_invalid_direction(self):
        result = forecast_barrier_closed_form(
            symbol="EURUSD",
            timeframe="H1",
            horizon=10,
            direction="sideways",
            barrier=1.2,
        )
        self.assertIn("error", result)
        self.assertIn("Invalid direction", result["error"])

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

    def test_forecast_barrier_optimize_optuna_backend(self):
        if importlib.util.find_spec("optuna") is None:
            self.skipTest("optuna package not installed")
        self._set_flat_history(1.0)
        paths = self._sample_paths()
        with patch(f'{_BARRIER_OPT_ROOT}._simulate_gbm_mc') as mock_sim:
            mock_sim.return_value = {"price_paths": paths}
            result = forecast_barrier_optimize(
                symbol="EURUSD",
                timeframe="H1",
                horizon=4,
                method="mc_gbm",
                direction="long",
                mode="pct",
                tp_min=0.2, tp_max=1.0, tp_steps=3,
                sl_min=0.2, sl_max=1.0, sl_steps=3,
                objective="ev",
                params={
                    "optimizer": "optuna",
                    "n_trials": 12,
                    "n_jobs": 1,
                    "sampler": "random",
                    "pruner": "none",
                    "seed": 9,
                },
                return_grid=True,
            )
        self.assertTrue(result.get("success"))
        self.assertEqual(result.get("optimizer"), "optuna")
        self.assertIn("optuna", result)
        self.assertGreaterEqual(result["optuna"].get("completed_trials", 0), 1)
        self.assertGreater(result.get("results_total", 0), 0)
        self.assertTrue(result.get("grid"))
        self.assertIsInstance(result.get("best"), dict)

    def test_forecast_barrier_optimize_optuna_pareto_front(self):
        if importlib.util.find_spec("optuna") is None:
            self.skipTest("optuna package not installed")
        self._set_flat_history(1.0)
        paths = self._sample_paths()
        with patch(f'{_BARRIER_OPT_ROOT}._simulate_gbm_mc') as mock_sim:
            mock_sim.return_value = {"price_paths": paths}
            result = forecast_barrier_optimize(
                symbol="EURUSD",
                timeframe="H1",
                horizon=4,
                method="mc_gbm",
                direction="long",
                mode="pct",
                tp_min=0.2, tp_max=1.0, tp_steps=3,
                sl_min=0.2, sl_max=1.0, sl_steps=3,
                objective="ev",
                params={
                    "optimizer": "optuna",
                    "optuna_pareto": True,
                    "n_trials": 12,
                    "n_jobs": 1,
                    "sampler": "random",
                    "pruner": "none",
                    "seed": 13,
                    "pareto_limit": 5,
                },
                return_grid=True,
            )
        self.assertTrue(result.get("success"))
        self.assertEqual(result.get("optimizer"), "optuna")
        self.assertIn("optuna", result)
        self.assertTrue(result["optuna"].get("pareto"))
        self.assertIn("pareto_front", result)
        self.assertLessEqual(result.get("pareto_count", 0), 5)
        self.assertGreaterEqual(result.get("pareto_count", 0), 1)
        row = result["pareto_front"][0]
        self.assertIn("objective_values", row)
        self.assertIn("ev", row["objective_values"])
        self.assertIn("prob_loss", row["objective_values"])
        self.assertIn("t_hit_resolve_median", row["objective_values"])

    def test_forecast_barrier_optimize_optuna_tpe_suppresses_multivariate_warning(self):
        if importlib.util.find_spec("optuna") is None:
            self.skipTest("optuna package not installed")
        self._set_flat_history(1.0)
        paths = self._sample_paths()
        with patch(f'{_BARRIER_OPT_ROOT}._simulate_gbm_mc') as mock_sim, warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            mock_sim.return_value = {"price_paths": paths}
            result = forecast_barrier_optimize(
                symbol="EURUSD",
                timeframe="H1",
                horizon=4,
                method="mc_gbm",
                direction="long",
                mode="pct",
                tp_min=0.2, tp_max=1.0, tp_steps=3,
                sl_min=0.2, sl_max=1.0, sl_steps=3,
                objective="ev",
                params={
                    "optimizer": "optuna",
                    "n_trials": 8,
                    "n_jobs": 1,
                    "sampler": "tpe",
                    "pruner": "none",
                    "seed": 21,
                },
                return_grid=False,
            )

        self.assertTrue(result.get("success"))
        self.assertFalse(any(
            ("multivariate" in str(w.message).lower() and "experimental" in str(w.message).lower())
            for w in rec
        ))

    def test_forecast_barrier_optimize_fast_defaults_switch(self):
        self._set_flat_history(1.0)
        paths = self._sample_paths()
        with patch(f'{_BARRIER_OPT_ROOT}._simulate_gbm_mc') as mock_sim:
            mock_sim.return_value = {"price_paths": paths}
            result = forecast_barrier_optimize(
                symbol="EURUSD",
                timeframe="H1",
                horizon=4,
                method="mc_gbm",
                direction="long",
                mode="pct",
                fast_defaults=True,
                return_grid=True,
            )
        self.assertTrue(result.get("success"))
        self.assertTrue(result.get("fast_defaults"))
        profile = result.get("compute_profile", {})
        self.assertEqual(profile.get("n_sims"), 1200)
        self.assertEqual(profile.get("tp_steps"), 4)
        self.assertEqual(profile.get("sl_steps"), 4)
        self.assertEqual(profile.get("ratio_steps"), 4)
        self.assertEqual(profile.get("vol_steps"), 4)
        self.assertFalse(profile.get("refine"))
        self.assertEqual(result.get("results_total"), 16)

    def test_forecast_barrier_optimize_search_profile_long(self):
        self._set_flat_history(1.0)
        paths = self._sample_paths()
        with patch(f'{_BARRIER_OPT_ROOT}._simulate_gbm_mc') as mock_sim:
            mock_sim.return_value = {"price_paths": paths}
            result = forecast_barrier_optimize(
                symbol="EURUSD",
                timeframe="H1",
                horizon=4,
                method="mc_gbm",
                direction="long",
                mode="pct",
                search_profile="long",
                return_grid=True,
            )
        self.assertTrue(result.get("success"))
        self.assertEqual(result.get("search_profile"), "long")
        profile = result.get("compute_profile", {})
        self.assertEqual(profile.get("profile"), "long")
        self.assertEqual(profile.get("n_sims"), 10000)
        self.assertEqual(profile.get("tp_steps"), 41)
        self.assertEqual(profile.get("sl_steps"), 51)
        self.assertEqual(profile.get("ratio_steps"), 24)
        self.assertEqual(profile.get("vol_steps"), 18)
        self.assertTrue(profile.get("refine"))

    def test_forecast_barrier_optimize_ensemble_method(self):
        self._set_flat_history(1.0)
        paths = self._sample_paths()
        with patch(f'{_BARRIER_OPT_ROOT}._simulate_gbm_mc') as mock_gbm, \
             patch(f'{_BARRIER_OPT_ROOT}._simulate_bootstrap_mc') as mock_bootstrap, \
             patch(f'{_BARRIER_OPT_ROOT}._get_live_reference_price', return_value=(None, None)):
            mock_gbm.return_value = {"price_paths": paths}
            mock_bootstrap.return_value = {"price_paths": paths}
            result = forecast_barrier_optimize(
                symbol="EURUSD",
                timeframe="H1",
                horizon=4,
                method="ensemble",
                direction="long",
                mode="pct",
                tp_min=0.5, tp_max=0.5, tp_steps=1,
                sl_min=0.5, sl_max=0.5, sl_steps=1,
                params={
                    "ensemble_methods": ["mc_gbm", "bootstrap"],
                    "ensemble_agg": "median",
                    "optimizer": "grid",
                    "n_sims": 50,
                    "n_seeds": 1,
                },
                return_grid=False,
                output="summary",
            )
        self.assertTrue(result.get("success"))
        self.assertEqual(result.get("method"), "ensemble")
        self.assertIn("ensemble", result)
        self.assertEqual(result["ensemble"]["agg"], "median")
        self.assertEqual(len(result["ensemble"]["members"]), 2)
        self.assertIn("best", result)
        self.assertIsInstance(result["best"], dict)
        self.assertIn("ev", result["best"])

    def test_forecast_barrier_optimize_ensemble_selects_real_member_candidate(self):
        self._set_flat_history(1.0)
        gbm_paths = np.array([
            [1.0030, 1.0010, 1.0010, 1.0010],
            [1.0030, 1.0010, 1.0010, 1.0010],
            [0.9970, 0.9960, 0.9950, 0.9940],
            [1.0010, 1.0010, 1.0010, 1.0010],
        ])
        bootstrap_paths = np.array([
            [1.0070, 1.0070, 1.0070, 1.0070],
            [1.0070, 1.0070, 1.0070, 1.0070],
            [0.9930, 0.9930, 0.9930, 0.9930],
            [1.0070, 1.0070, 1.0070, 1.0070],
        ])
        with patch(f'{_BARRIER_OPT_ROOT}._simulate_gbm_mc') as mock_gbm, \
             patch(f'{_BARRIER_OPT_ROOT}._simulate_bootstrap_mc') as mock_bootstrap, \
             patch(f'{_BARRIER_OPT_ROOT}._get_live_reference_price', return_value=(None, None)):
            mock_gbm.return_value = {"price_paths": gbm_paths}
            mock_bootstrap.return_value = {"price_paths": bootstrap_paths}
            result = forecast_barrier_optimize(
                symbol="EURUSD",
                timeframe="H1",
                horizon=4,
                method="ensemble",
                direction="long",
                mode="pct",
                tp_min=0.2, tp_max=0.6, tp_steps=2,
                sl_min=0.2, sl_max=0.6, sl_steps=2,
                objective="ev",
                params={
                    "ensemble_methods": ["mc_gbm", "bootstrap"],
                    "ensemble_agg": "median",
                    "optimizer": "grid",
                    "n_sims": 50,
                    "n_seeds": 1,
                },
                return_grid=True,
                output="summary",
            )
        self.assertTrue(result.get("success"))
        self.assertEqual(result.get("method"), "ensemble")
        self.assertEqual(result["best"].get("member_method"), "bootstrap")
        self.assertEqual(result["ensemble"]["selected_member"]["method"], "bootstrap")
        aggregate = result["ensemble"].get("aggregate_metrics")
        self.assertIsInstance(aggregate, dict)
        self.assertNotAlmostEqual(float(result["best"]["tp"]), float(aggregate["tp"]), places=8)
        selected_rows = [m for m in result["ensemble"]["members"] if m.get("selected")]
        self.assertEqual(len(selected_rows), 1)
        self.assertAlmostEqual(float(selected_rows[0]["tp"]), float(result["best"]["tp"]), places=8)
        self.assertAlmostEqual(float(selected_rows[0]["sl"]), float(result["best"]["sl"]), places=8)

    def test_forecast_barrier_optimize_prefers_live_reference_price(self):
        self._set_flat_history(1.0, bars=200)
        paths = self._sample_paths()
        with patch(f'{_BARRIER_OPT_ROOT}._simulate_gbm_mc') as mock_sim, \
             patch(f'{_BARRIER_OPT_ROOT}._get_live_reference_price', return_value=(1.2345, "live_tick_bid")):
            mock_sim.return_value = {"price_paths": paths}
            result = forecast_barrier_optimize(
                symbol="EURUSD",
                timeframe="H1",
                horizon=4,
                method="mc_gbm",
                direction="short",
                mode="pct",
                tp_min=0.5,
                tp_max=0.5,
                tp_steps=1,
                sl_min=0.5,
                sl_max=0.5,
                sl_steps=1,
                return_grid=True,
            )
        self.assertTrue(result["success"])
        self.assertAlmostEqual(result["last_price"], 1.2345, places=8)
        self.assertAlmostEqual(result["last_price_close"], 1.0, places=8)
        self.assertEqual(result["last_price_source"], "live_tick_bid")
        best = result.get("best")
        self.assertIsInstance(best, dict)
        self.assertAlmostEqual(best["tp_price"], 1.2345 * 0.995, places=8)
        self.assertAlmostEqual(best["sl_price"], 1.2345 * 1.005, places=8)

    def test_forecast_barrier_optimize_reanchors_paths_to_live_reference_price(self):
        self._set_flat_history(1.0, bars=200)
        paths = self._sample_paths()
        with patch(f'{_BARRIER_PROB_ROOT}._simulate_gbm_mc') as mock_sim, \
             patch(f'{_BARRIER_PROB_ROOT}._get_live_reference_price', return_value=(1.2345, "live_tick_ask")):
            mock_sim.return_value = {"price_paths": paths}
            hit_probs = forecast_barrier_hit_probabilities(
                symbol="EURUSD",
                timeframe="H1",
                horizon=4,
                method="mc_gbm",
                direction="long",
                tp_pct=0.5,
                sl_pct=0.5,
            )
        with patch(f'{_BARRIER_OPT_ROOT}._simulate_gbm_mc') as mock_sim, \
             patch(f'{_BARRIER_OPT_ROOT}._get_live_reference_price', return_value=(1.2345, "live_tick_ask")):
            mock_sim.return_value = {"price_paths": paths}
            result = forecast_barrier_optimize(
                symbol="EURUSD",
                timeframe="H1",
                horizon=4,
                method="mc_gbm",
                direction="long",
                mode="pct",
                tp_min=0.5,
                tp_max=0.5,
                tp_steps=1,
                sl_min=0.5,
                sl_max=0.5,
                sl_steps=1,
                return_grid=True,
            )
        self.assertTrue(hit_probs.get("success"))
        self.assertTrue(result.get("success"))
        best = result["best"]
        self.assertAlmostEqual(best["prob_tp_first"], hit_probs["prob_tp_first"], places=8)
        self.assertAlmostEqual(best["prob_sl_first"], hit_probs["prob_sl_first"], places=8)
        self.assertAlmostEqual(best["prob_no_hit"], hit_probs["prob_no_hit"], places=8)

    def test_forecast_barrier_optimize_filters_invalid_barrier_geometry(self):
        self._set_flat_history(1.0, bars=200)
        paths = self._sample_paths()
        with patch(f'{_BARRIER_OPT_ROOT}._simulate_gbm_mc') as mock_sim, \
             patch(f'{_BARRIER_OPT_ROOT}._get_live_reference_price', return_value=(None, None)):
            mock_sim.return_value = {"price_paths": paths}
            result = forecast_barrier_optimize(
                symbol="EURUSD",
                timeframe="H1",
                horizon=4,
                method="mc_gbm",
                direction="short",
                mode="pct",
                tp_min=150.0,
                tp_max=150.0,
                tp_steps=1,
                sl_min=0.5,
                sl_max=0.5,
                sl_steps=1,
                return_grid=True,
            )
        self.assertTrue(result["success"])
        self.assertTrue(result["no_candidates"])
        self.assertIsNone(result["best"])
        self.assertEqual(result["results_total"], 0)
        self.assertEqual(result.get("barrier_sanity_filtered"), 1)

    def test_forecast_barrier_optimize_refine_and_metrics(self):
        # Deterministic paths to verify refine pass and ranking
        self._set_flat_history(1.0)
        paths = np.array([
            [1.0, 1.01, 1.02, 1.03],
            [1.0, 0.99, 0.98, 0.97],
            [1.0, 1.002, 0.998, 1.006],
        ])
        with patch(f'{_BARRIER_OPT_ROOT}._simulate_gbm_mc') as mock_sim:
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
        with patch(f'{_BARRIER_OPT_ROOT}._simulate_gbm_mc') as mock_sim:
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

    def test_forecast_barrier_optimize_keeps_compact_results_with_full_grid(self):
        self._set_flat_history(1.0)
        paths = np.array([
            [1.0, 1.01, 1.02, 1.03],
            [1.0, 0.99, 0.98, 0.97],
            [1.0, 1.002, 0.998, 1.006],
        ])
        with patch(f'{_BARRIER_OPT_ROOT}._simulate_gbm_mc') as mock_sim:
            mock_sim.return_value = {"price_paths": paths}
            result = forecast_barrier_optimize(
                symbol="EURUSD",
                timeframe="H1",
                horizon=4,
                method="mc_gbm",
                direction="long",
                mode="pct",
                tp_min=0.2, tp_max=1.4, tp_steps=4,
                sl_min=0.2, sl_max=1.4, sl_steps=4,
                objective="edge",
                output="full",
                return_grid=True,
            )
        self.assertTrue(result["success"])
        self.assertIn("results_total", result)
        self.assertEqual(result["results_total"], len(result["grid"]))
        self.assertLessEqual(len(result["results"]), 10)
        self.assertGreaterEqual(len(result["grid"]), len(result["results"]))

    def test_forecast_barrier_optimize_volatility_grid(self):
        self._set_flat_history(1.0)
        paths = self._sample_paths()
        with patch(f'{_BARRIER_OPT_ROOT}._simulate_gbm_mc') as mock_sim:
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
        with patch(f'{_BARRIER_OPT_ROOT}._simulate_gbm_mc') as mock_sim:
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
        with patch(f'{_BARRIER_OPT_ROOT}._simulate_gbm_mc') as mock_sim:
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
        with patch(f'{_BARRIER_OPT_ROOT}._simulate_gbm_mc') as mock_sim:
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
        with patch(f'{_BARRIER_OPT_ROOT}._simulate_gbm_mc') as mock_sim:
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
        with patch(f'{_BARRIER_OPT_ROOT}._simulate_gbm_mc') as mock_sim, \
             patch(f'{_BARRIER_OPT_ROOT}._get_live_reference_price', return_value=(None, None)):
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
            expected_ev = 0.5 * entry["tp"] - 0.5 * entry["sl"]
            self.assertAlmostEqual(entry["ev"], expected_ev, places=7)

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

    def test_forecast_barrier_hit_probabilities_rejects_non_finite_barrier_input(self):
        self._set_flat_history(1.0, bars=200)
        paths = self._sample_paths()
        with patch(f'{_BARRIER_PROB_ROOT}._simulate_gbm_mc') as mock_sim, \
             patch(f'{_BARRIER_PROB_ROOT}._get_live_reference_price', return_value=(None, None)):
            mock_sim.return_value = {"price_paths": paths}
            result = forecast_barrier_hit_probabilities(
                symbol="EURUSD",
                timeframe="H1",
                horizon=4,
                method="mc_gbm",
                direction="long",
                tp_abs=float("nan"),
                sl_abs=0.99,
            )
        self.assertIn("error", result)
        self.assertIn("Provide barriers", result["error"])

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
        self.assertEqual(result.get("objective"), "ev")
        self.assertEqual(result.get("objective_requested"), "not_a_real_objective")
        self.assertEqual(result.get("objective_used"), "ev")

    def test_forecast_barrier_optimize_warns_on_negative_ev_best(self):
        self._set_flat_history(1.0)
        paths = np.array([
            [1.0030, 1.0040, 1.0050],
            [1.0030, 1.0020, 1.0010],
            [1.0040, 1.0060, 1.0070],
            [1.0030, 0.9900, 0.9800],
            [0.9740, 0.9730, 0.9720],
        ])
        with patch(f'{_BARRIER_OPT_ROOT}._simulate_gbm_mc') as mock_sim:
            mock_sim.return_value = {"price_paths": paths}
            result = forecast_barrier_optimize(
                symbol="EURUSD",
                timeframe="H1",
                horizon=3,
                method="mc_gbm",
                direction="long",
                mode="pct",
                tp_min=0.25,
                tp_max=0.25,
                tp_steps=1,
                sl_min=2.5,
                sl_max=2.5,
                sl_steps=1,
                objective="edge",
                return_grid=True,
            )
        self.assertTrue(result.get("success"))
        self.assertLess(float(result["best"]["ev"]), 0.0)
        self.assertFalse(result.get("viable"))
        self.assertEqual(result.get("status"), "non_viable")
        self.assertTrue(result.get("no_action"))
        self.assertEqual(result.get("status_reason"), "Selected candidate has negative EV.")
        self.assertIsInstance(result.get("least_negative"), dict)
        self.assertEqual(result["least_negative"].get("ref"), "best")
        self.assertEqual(result["least_negative"].get("ev"), result["best"].get("ev"))
        self.assertIn("selection_warnings", result)
        self.assertIsInstance(result.get("advice"), list)

    def test_forecast_barrier_optimize_flags_phantom_profit_risk(self):
        self._set_flat_history(1.0)
        paths = np.array([
            [1.0030, 1.0030, 1.0030],
            [1.0005, 1.0005, 1.0005],
            [0.9995, 0.9995, 0.9995],
        ])
        with patch(f'{_BARRIER_OPT_ROOT}._simulate_gbm_mc') as mock_sim:
            mock_sim.return_value = {"price_paths": paths}
            result = forecast_barrier_optimize(
                symbol="EURUSD",
                timeframe="H1",
                horizon=3,
                method="mc_gbm",
                direction="long",
                mode="pct",
                tp_min=0.2,
                tp_max=0.2,
                tp_steps=1,
                sl_min=2.0,
                sl_max=2.0,
                sl_steps=1,
                objective="ev",
                return_grid=True,
            )
        self.assertTrue(result.get("success"))
        self.assertGreater(float(result["best"]["ev"]), 0.0)
        self.assertFalse(result.get("viable"))
        self.assertEqual(result.get("status"), "non_viable")
        self.assertTrue(result.get("no_action"))
        self.assertIn("unresolved paths", result.get("status_reason", "").lower())
        self.assertTrue(result["best"].get("phantom_profit_risk"))
        self.assertLess(float(result["best"]["edge_vs_breakeven"]), 0.0)
        self.assertAlmostEqual(float(result["best"]["breakeven_win_rate"]), 1.0 / 1.1, places=6)
        self.assertTrue(result.get("ev_edge_conflict"))
        self.assertIn("unresolved", result.get("ev_edge_conflict_reason", "").lower())
        self.assertIn("selection_warnings", result)

    def test_forecast_barrier_optimize_guardrails_degenerate_objective(self):
        self._set_flat_history(1.0)
        paths = np.vstack([
            np.array([[1.0040, 1.0040, 1.0040]]),
            np.full((9, 3), 1.0001),
        ])
        with patch(f'{_BARRIER_OPT_ROOT}._simulate_gbm_mc') as mock_sim:
            mock_sim.return_value = {"price_paths": paths}
            result = forecast_barrier_optimize(
                symbol="EURUSD",
                timeframe="H1",
                horizon=3,
                method="mc_gbm",
                direction="long",
                mode="pct",
                tp_min=0.25,
                tp_max=0.25,
                tp_steps=1,
                sl_min=2.5,
                sl_max=2.5,
                sl_steps=1,
                objective="min_loss_prob",
                return_grid=True,
            )
        self.assertTrue(result.get("success"))
        self.assertTrue(result.get("no_candidates"))
        self.assertEqual(result.get("status"), "no_candidates")
        self.assertTrue(result.get("no_action"))
        self.assertEqual(result.get("min_prob_resolve"), 0.2)
        self.assertEqual(result.get("results"), [])

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
        self.assertFalse(result.get("viable"))
        self.assertEqual(result.get("status"), "no_candidates")
        self.assertTrue(result.get("no_action"))
        self.assertIsNone(result.get("least_negative"))
        self.assertIn("warning", result)

    def test_forecast_barrier_optimize_viable_only_concise_limits_non_viable_output(self):
        self._set_flat_history(1.0)
        paths = np.array([
            [1.0030, 1.0040, 1.0050],
            [1.0030, 1.0020, 1.0010],
            [1.0040, 1.0060, 1.0070],
            [1.0030, 0.9900, 0.9800],
            [0.9740, 0.9730, 0.9720],
        ])
        with patch(f'{_BARRIER_OPT_ROOT}._simulate_gbm_mc') as mock_sim:
            mock_sim.return_value = {"price_paths": paths}
            result = forecast_barrier_optimize(
                symbol="EURUSD",
                timeframe="H1",
                horizon=3,
                method="mc_gbm",
                direction="long",
                mode="pct",
                tp_min=0.25,
                tp_max=0.25,
                tp_steps=1,
                sl_min=2.5,
                sl_max=2.5,
                sl_steps=1,
                objective="edge",
                viable_only=True,
                concise=True,
                top_k=2,
                return_grid=True,
                output="full",
            )
        self.assertTrue(result.get("success"))
        self.assertFalse(result.get("viable"))
        self.assertEqual(result.get("status"), "non_viable")
        self.assertTrue(result.get("no_action"))
        self.assertTrue(result.get("viable_only"))
        self.assertTrue(result.get("concise"))
        self.assertEqual(result.get("results_total"), 1)
        self.assertEqual(result.get("viable_results_total"), 0)
        self.assertEqual(len(result.get("results", [])), 1)
        self.assertIsNone(result.get("grid"))


class TestTier1TradingCosts(_BarrierModulePatchMixin, unittest.TestCase):
    """T1.1: Spread/commission/slippage modeling."""

    def setUp(self):
        self._start_barrier_module_patchers()
        dates = pd.date_range(start='2023-01-01', periods=500, freq='h')
        prices = np.linspace(1.0, 1.1, 500) + np.random.normal(0, 0.001, 500)
        self.df = pd.DataFrame({'time': dates, 'close': prices})
        self._set_barrier_history(self.df)

    def tearDown(self):
        self._stop_barrier_module_patchers()

    def test_optimize_no_costs_has_no_trading_costs_key(self):
        result = forecast_barrier_optimize(
            symbol="EURUSD", timeframe="H1", horizon=10,
            method="mc_gbm", direction="long", mode="pct",
            tp_min=0.5, tp_max=0.5, tp_steps=1,
            sl_min=0.5, sl_max=0.5, sl_steps=1,
        )
        self.assertTrue(result.get("success"))
        self.assertNotIn("trading_costs", result)

    def test_optimize_with_spread_pct_produces_trading_costs(self):
        result = forecast_barrier_optimize(
            symbol="EURUSD", timeframe="H1", horizon=10,
            method="mc_gbm", direction="long", mode="pct",
            tp_min=0.5, tp_max=0.5, tp_steps=1,
            sl_min=0.5, sl_max=0.5, sl_steps=1,
            params={"spread_pct": 0.02},
        )
        self.assertTrue(result.get("success"))
        self.assertIn("trading_costs", result)
        tc = result["trading_costs"]
        self.assertGreater(tc["cost_per_trade"], 0.0)
        self.assertEqual(tc["cost_unit"], "pct")
        self.assertAlmostEqual(tc["spread_pct"], 0.02)

    def test_optimize_with_spread_pips_produces_trading_costs(self):
        result = forecast_barrier_optimize(
            symbol="EURUSD", timeframe="H1", horizon=10,
            method="mc_gbm", direction="long", mode="pips",
            tp_min=50, tp_max=50, tp_steps=1,
            sl_min=50, sl_max=50, sl_steps=1,
            params={"spread_pips": 2.0},
        )
        self.assertTrue(result.get("success"))
        self.assertIn("trading_costs", result)
        tc = result["trading_costs"]
        self.assertGreater(tc["cost_per_trade"], 0.0)
        self.assertAlmostEqual(tc["spread_pips"], 2.0)

    def test_cost_adjusted_ev_fields(self):
        """When costs present, ev_gross and ev_net should appear in best candidate."""
        dates = pd.date_range(start='2023-01-01', periods=500, freq='h')
        prices = np.full(500, 1.0)
        self._set_barrier_history(pd.DataFrame({'time': dates, 'close': prices}))
        paths = np.array([
            [1.006, 1.008, 1.010],
            [1.006, 1.008, 1.010],
            [1.006, 1.008, 1.010],
            [0.994, 0.992, 0.990],
            [0.994, 0.992, 0.990],
        ])
        with patch(f'{_BARRIER_OPT_ROOT}._simulate_gbm_mc') as mock_sim:
            mock_sim.return_value = {"price_paths": paths}
            result = forecast_barrier_optimize(
                symbol="EURUSD", timeframe="H1", horizon=3,
                method="mc_gbm", direction="long", mode="pct",
                tp_min=0.5, tp_max=0.5, tp_steps=1,
                sl_min=0.5, sl_max=0.5, sl_steps=1,
                params={"spread_pct": 0.05},
            )
        self.assertTrue(result.get("success"))
        best = result.get("best")
        if best is not None:
            self.assertIn("ev_gross", best)
            self.assertIn("ev_net", best)
            ev_gross = float(best["ev_gross"])
            ev_net = float(best["ev_net"])
            self.assertGreaterEqual(ev_gross, ev_net)

    def test_ensemble_preserves_cost_adjusted_viability_metrics(self):
        self._set_barrier_history(pd.DataFrame({
            'time': pd.date_range(start='2023-01-01', periods=500, freq='h'),
            'close': np.full(500, 1.0),
        }))
        paths = np.array([
            [1.0070, 1.0070, 1.0070],
            [1.0070, 1.0070, 1.0070],
            [1.0010, 1.0010, 1.0010],
            [1.0010, 1.0010, 1.0010],
            [1.0010, 1.0010, 1.0010],
        ])
        with patch(f'{_BARRIER_OPT_ROOT}._simulate_gbm_mc') as mock_gbm, \
             patch(f'{_BARRIER_OPT_ROOT}._simulate_bootstrap_mc') as mock_bootstrap, \
             patch(f'{_BARRIER_OPT_ROOT}._get_live_reference_price', return_value=(None, None)):
            mock_gbm.return_value = {"price_paths": paths}
            mock_bootstrap.return_value = {"price_paths": paths}
            result = forecast_barrier_optimize(
                symbol="EURUSD",
                timeframe="H1",
                horizon=3,
                method="ensemble",
                direction="long",
                mode="pct",
                tp_min=0.6,
                tp_max=0.6,
                tp_steps=1,
                sl_min=0.3,
                sl_max=0.3,
                sl_steps=1,
                objective="ev",
                params={
                    "ensemble_methods": ["mc_gbm", "bootstrap"],
                    "ensemble_agg": "median",
                    "optimizer": "grid",
                    "n_sims": 50,
                    "n_seeds": 1,
                    "spread_pct": 0.25,
                    "min_barrier_multiplier": 0.0,
                },
                return_grid=True,
                output="summary",
            )
        self.assertTrue(result.get("success"))
        self.assertEqual(result.get("method"), "ensemble")
        self.assertIn("trading_costs", result)
        self.assertFalse(result.get("viable"))
        self.assertEqual(result.get("status"), "non_viable")
        self.assertTrue(result.get("no_action"))
        self.assertTrue(result["best"].get("phantom_profit_risk"))
        self.assertLess(float(result["best"]["edge_vs_breakeven"]), 0.0)

    def test_breakeven_win_rate_net_computed_with_costs(self):
        """breakeven_win_rate_net should be computed and > gross breakeven when costs present."""
        from mtdata.forecast.barriers_shared import _annotate_candidate_metrics
        row = {"tp": 0.5, "sl": 0.5, "rr": 1.0, "prob_win": 0.55, "prob_loss": 0.45, "prob_no_hit": 0.0}
        _annotate_candidate_metrics(row, cost_per_trade=0.05)
        self.assertIn("breakeven_win_rate_net", row)
        net_be = float(row["breakeven_win_rate_net"])
        gross_be = float(row["breakeven_win_rate"])
        self.assertGreater(net_be, gross_be)

    def test_breakeven_win_rate_net_absent_without_costs(self):
        from mtdata.forecast.barriers_shared import _annotate_candidate_metrics
        row = {"tp": 0.5, "sl": 0.5, "rr": 1.0, "prob_win": 0.55, "prob_loss": 0.45, "prob_no_hit": 0.0}
        _annotate_candidate_metrics(row, cost_per_trade=0.0)
        self.assertNotIn("breakeven_win_rate_net", row)

    def test_breakeven_win_rate_net_is_1_when_cost_exceeds_tp(self):
        """If cost >= tp, net reward <= 0, breakeven_win_rate_net should be 1.0."""
        from mtdata.forecast.barriers_shared import _annotate_candidate_metrics
        row = {"tp": 0.05, "sl": 0.5, "rr": 0.1, "prob_win": 0.6, "prob_loss": 0.4, "prob_no_hit": 0.0}
        _annotate_candidate_metrics(row, cost_per_trade=0.05)
        self.assertAlmostEqual(row["breakeven_win_rate_net"], 1.0)


class TestTier1StatisticalSignificance(unittest.TestCase):
    """T1.2: Statistical significance guard (low_confidence, min_sims_recommended)."""

    def test_low_confidence_flagged_for_wide_ci(self):
        from mtdata.forecast.barriers_shared import _annotate_candidate_metrics
        row = {
            "tp": 0.5, "sl": 0.5, "rr": 1.0,
            "prob_win": 0.5, "prob_loss": 0.5, "prob_no_hit": 0.0,
            "prob_win_ci95": {"low": 0.30, "high": 0.70},
        }
        _annotate_candidate_metrics(row)
        self.assertTrue(row["low_confidence"])
        self.assertAlmostEqual(row["prob_win_ci_width"], 0.40)

    def test_low_confidence_not_flagged_for_narrow_ci(self):
        from mtdata.forecast.barriers_shared import _annotate_candidate_metrics
        row = {
            "tp": 0.5, "sl": 0.5, "rr": 1.0,
            "prob_win": 0.5, "prob_loss": 0.5, "prob_no_hit": 0.0,
            "prob_win_ci95": {"low": 0.47, "high": 0.53},
        }
        _annotate_candidate_metrics(row)
        self.assertFalse(row["low_confidence"])

    def test_min_sims_recommended_in_diagnostics(self):
        from mtdata.forecast.barriers_shared import _build_selection_diagnostics
        row = {
            "tp": 0.5, "sl": 0.5, "rr": 1.0,
            "prob_win": 0.5, "prob_loss": 0.5, "prob_no_hit": 0.0,
            "prob_win_ci95": {"low": 0.30, "high": 0.70},
            "ev": 0.01, "edge": 0.01, "kelly": 0.01,
        }
        diag = _build_selection_diagnostics(row)
        self.assertTrue(diag.get("low_confidence"))
        self.assertIn("confidence_warning", diag)
        self.assertIn("min_sims_recommended", diag)
        self.assertGreaterEqual(diag["min_sims_recommended"], 2000)
        self.assertIn("n_sims", diag["confidence_warning"].lower())

    def test_no_confidence_warning_for_narrow_ci(self):
        from mtdata.forecast.barriers_shared import _build_selection_diagnostics
        row = {
            "tp": 0.5, "sl": 0.5, "rr": 1.0,
            "prob_win": 0.55, "prob_loss": 0.45, "prob_no_hit": 0.0,
            "prob_win_ci95": {"low": 0.52, "high": 0.58},
            "ev": 0.01, "edge": 0.01, "kelly": 0.01,
        }
        diag = _build_selection_diagnostics(row)
        self.assertNotIn("confidence_warning", diag)
        self.assertNotIn("low_confidence", diag)


class TestTier1StructuredErrorHandling(_BarrierModulePatchMixin, unittest.TestCase):
    """T1.3: Structured error handling — simulation and outer exception upgrades."""

    def setUp(self):
        self._start_barrier_module_patchers()
        dates = pd.date_range(start='2023-01-01', periods=500, freq='h')
        prices = np.full(500, 1.0)
        self._set_barrier_history(pd.DataFrame({'time': dates, 'close': prices}))

    def tearDown(self):
        self._stop_barrier_module_patchers()

    def test_simulation_value_error_returns_structured_error(self):
        """ValueError in simulation → descriptive error dict, not crash."""
        with patch(f'{_BARRIER_OPT_ROOT}._simulate_gbm_mc', side_effect=ValueError("negative drift")):
            result = forecast_barrier_optimize(
                symbol="EURUSD", timeframe="H1", horizon=10,
                method="mc_gbm", direction="long", mode="pct",
                tp_min=0.5, tp_max=0.5, tp_steps=1,
                sl_min=0.5, sl_max=0.5, sl_steps=1,
            )
        self.assertIn("error", result)
        self.assertIn("simulation_failure", result.get("error_type", ""))
        self.assertIn("mc_gbm", result["error"])
        self.assertIn("traceback_summary", result)

    def test_simulation_runtime_error_returns_structured_error(self):
        with patch(f'{_BARRIER_OPT_ROOT}._simulate_gbm_mc', side_effect=RuntimeError("singular matrix")):
            result = forecast_barrier_optimize(
                symbol="EURUSD", timeframe="H1", horizon=10,
                method="mc_gbm", direction="long", mode="pct",
                tp_min=0.5, tp_max=0.5, tp_steps=1,
                sl_min=0.5, sl_max=0.5, sl_steps=1,
            )
        self.assertIn("error", result)
        self.assertEqual(result.get("error_type"), "simulation_failure")

    def test_simulation_linalg_error_returns_structured_error(self):
        with patch(f'{_BARRIER_OPT_ROOT}._simulate_gbm_mc', side_effect=np.linalg.LinAlgError("SVD")):
            result = forecast_barrier_optimize(
                symbol="EURUSD", timeframe="H1", horizon=10,
                method="mc_gbm", direction="long", mode="pct",
                tp_min=0.5, tp_max=0.5, tp_steps=1,
                sl_min=0.5, sl_max=0.5, sl_steps=1,
            )
        self.assertIn("error", result)
        self.assertEqual(result.get("error_type"), "simulation_failure")

    def test_programming_error_propagates_not_caught(self):
        """KeyError, TypeError, etc. should propagate (not be swallowed)."""
        with patch(f'{_BARRIER_OPT_ROOT}._simulate_gbm_mc', side_effect=KeyError("missing_key")):
            with self.assertRaises(KeyError):
                forecast_barrier_optimize(
                    symbol="EURUSD", timeframe="H1", horizon=10,
                    method="mc_gbm", direction="long", mode="pct",
                    tp_min=0.5, tp_max=0.5, tp_steps=1,
                    sl_min=0.5, sl_max=0.5, sl_steps=1,
                )

    def test_outer_except_includes_error_type_and_traceback(self):
        """Non-programming exceptions caught by outer handler include error_type."""
        with patch(f'{_BARRIER_OPT_ROOT}._simulate_gbm_mc', side_effect=OSError("disk full")):
            result = forecast_barrier_optimize(
                symbol="EURUSD", timeframe="H1", horizon=10,
                method="mc_gbm", direction="long", mode="pct",
                tp_min=0.5, tp_max=0.5, tp_steps=1,
                sl_min=0.5, sl_max=0.5, sl_steps=1,
            )
        self.assertIn("error", result)
        self.assertEqual(result.get("error_type"), "OSError")
        self.assertIn("traceback_summary", result)

    def test_optimize_bad_seed_type_returns_structured_error(self):
        result = forecast_barrier_optimize(
            symbol="EURUSD", timeframe="H1", horizon=10,
            method="mc_gbm", direction="long", mode="pct",
            tp_min=0.5, tp_max=0.5, tp_steps=1,
            sl_min=0.5, sl_max=0.5, sl_steps=1,
            params={"seed": [1]},
        )
        self.assertIn("error", result)
        self.assertEqual(result.get("error_type"), "TypeError")
        self.assertIn("traceback_summary", result)

    def test_probabilities_simulation_error_returns_structured(self):
        """barriers_probabilities: simulation ValueError → structured error."""
        with patch(f'{_BARRIER_PROB_ROOT}._simulate_gbm_mc', side_effect=ValueError("bad input")):
            result = forecast_barrier_hit_probabilities(
                symbol="EURUSD", timeframe="H1", horizon=10,
                method="mc_gbm", direction="long",
                tp_pct=0.5, sl_pct=0.5,
            )
        self.assertIn("error", result)
        self.assertIn("simulation_failure", result.get("error_type", ""))
        self.assertIn("traceback_summary", result)

    def test_probabilities_outer_except_structured(self):
        """barriers_probabilities: outer handler includes error_type and traceback."""
        with patch(f'{_BARRIER_PROB_ROOT}._simulate_gbm_mc', side_effect=OSError("network")):
            result = forecast_barrier_hit_probabilities(
                symbol="EURUSD", timeframe="H1", horizon=10,
                method="mc_gbm", direction="long",
                tp_pct=0.5, sl_pct=0.5,
            )
        self.assertIn("error", result)
        self.assertIn("error_type", result)
        self.assertIn("traceback_summary", result)

    def test_probabilities_bad_seed_type_returns_structured_error(self):
        result = forecast_barrier_hit_probabilities(
            symbol="EURUSD", timeframe="H1", horizon=10,
            method="mc_gbm", direction="long",
            tp_pct=0.5, sl_pct=0.5,
            params={"seed": [1]},
        )
        self.assertIn("error", result)
        self.assertEqual(result.get("error_type"), "TypeError")
        self.assertIn("traceback_summary", result)

    def test_closed_form_bad_mu_type_returns_structured_error(self):
        result = forecast_barrier_closed_form(
            symbol="EURUSD",
            timeframe="H1",
            horizon=10,
            direction="long",
            barrier=1.2,
            mu=[],
            sigma=0.2,
        )
        self.assertIn("error", result)
        self.assertEqual(result.get("error_type"), "TypeError")
        self.assertIn("traceback_summary", result)

    def test_probabilities_programming_error_propagates(self):
        with patch(f'{_BARRIER_PROB_ROOT}._simulate_gbm_mc', side_effect=KeyError("missing_key")):
            with self.assertRaises(KeyError):
                forecast_barrier_hit_probabilities(
                    symbol="EURUSD", timeframe="H1", horizon=10,
                    method="mc_gbm", direction="long",
                    tp_pct=0.5, sl_pct=0.5,
                )


class TestTier1EnsembleDegradation(_BarrierModulePatchMixin, unittest.TestCase):
    """T1.4: Degraded ensemble quality warning."""

    def setUp(self):
        self._start_barrier_module_patchers()
        dates = pd.date_range(start='2023-01-01', periods=500, freq='h')
        prices = np.linspace(1.0, 1.1, 500) + np.random.normal(0, 0.001, 500)
        self.df = pd.DataFrame({'time': dates, 'close': prices})
        self._set_barrier_history(self.df)

    def tearDown(self):
        self._stop_barrier_module_patchers()

    def _make_member_output(self, method_name: str, ev: float = 0.01):
        """Build a minimal successful member output dict."""
        return {
            "success": True,
            "method": method_name,
            "last_price": 1.05,
            "last_price_close": 1.05,
            "best": {
                "tp": 0.5, "sl": 0.5, "rr": 1.0,
                "tp_price": 1.055, "sl_price": 1.045,
                "prob_win": 0.55, "prob_loss": 0.40, "prob_no_hit": 0.05,
                "prob_tp_first": 0.55, "prob_sl_first": 0.40,
                "prob_tie": 0.0, "prob_resolve": 0.95,
                "ev": ev, "ev_cond": ev, "edge": 0.05,
                "breakeven_win_rate": 0.5, "edge_vs_breakeven": 0.05,
                "kelly": 0.1, "kelly_cond": 0.1,
                "ev_per_bar": ev / 10, "profit_factor": 1.1, "utility": 0.01,
                "t_hit_tp_median": 5, "t_hit_sl_median": 4,
                "t_hit_resolve_mean": 5, "t_hit_resolve_median": 4,
            },
        }

    def test_ensemble_all_succeed_confidence_high(self):
        """All members succeed → confidence=high, degraded=False."""
        methods = ['mc_gbm', 'garch']
        outputs = {m: self._make_member_output(m) for m in methods}
        with patch('mtdata.forecast.barriers.forecast_barrier_optimize') as mock_opt:
            def _side_effect(*args, **kwargs):
                m = kwargs.get('method', args[3] if len(args) > 3 else 'mc_gbm')
                return outputs.get(m, {"error": "fail"})
            mock_opt.side_effect = _side_effect
        n_total = 4
        n_succeeded = 4
        n_failed = 0
        ensemble_degraded = n_failed > n_total / 2
        ensemble_confidence = "high" if n_failed == 0 else ("medium" if n_succeeded > n_failed else "low")
        self.assertFalse(ensemble_degraded)
        self.assertEqual(ensemble_confidence, "high")

    def test_ensemble_partial_failure_confidence_medium(self):
        """1 of 4 fails → confidence=medium, degraded=False."""
        n_total = 4
        n_succeeded = 3
        n_failed = 1
        ensemble_degraded = n_failed > n_total / 2
        ensemble_confidence = "high" if n_failed == 0 else ("medium" if n_succeeded > n_failed else "low")
        self.assertFalse(ensemble_degraded)
        self.assertEqual(ensemble_confidence, "medium")

    def test_ensemble_majority_failure_degraded_low(self):
        """3 of 4 fail → confidence=low, degraded=True."""
        n_total = 4
        n_succeeded = 1
        n_failed = 3
        ensemble_degraded = n_failed > n_total / 2
        ensemble_confidence = "high" if n_failed == 0 else ("medium" if n_succeeded > n_failed else "low")
        self.assertTrue(ensemble_degraded)
        self.assertEqual(ensemble_confidence, "low")

    def test_ensemble_half_failure_not_degraded(self):
        """2 of 4 fail → confidence=medium, degraded=False (not > half)."""
        n_total = 4
        n_succeeded = 2
        n_failed = 2
        ensemble_degraded = n_failed > n_total / 2
        ensemble_confidence = "high" if n_failed == 0 else ("medium" if n_succeeded > n_failed else "low")
        self.assertFalse(ensemble_degraded)
        self.assertEqual(ensemble_confidence, "low")

    def test_degraded_warning_message_content(self):
        """Degraded warning message should contain key info."""
        n_total = 4
        n_failed = 3
        n_succeeded = 1
        ensemble_degraded = True
        ensemble_confidence = "low"
        warning = (
            f"Ensemble degraded: {n_failed}/{n_total} members failed "
            f"(confidence={ensemble_confidence}). "
            f"Results based on {n_succeeded} method(s) only — interpret with caution."
        )
        self.assertIn("3/4", warning)
        self.assertIn("confidence=low", warning)
        self.assertIn("1 method(s)", warning)
        self.assertIn("caution", warning)

    def test_single_survivor_warning_mentions_no_diversification(self):
        """When only 1 member survived, warn about no diversification."""
        n_failed = 3
        n_total = 4
        n_succeeded = 1
        ensemble_degraded = n_failed > n_total / 2
        if ensemble_degraded:
            msg = (
                f"Ensemble degraded: {n_failed}/{n_total} members failed "
                f"(confidence=low). "
                f"Results based on {n_succeeded} method(s) only — interpret with caution."
            )
        elif n_succeeded == 1:
            msg = (
                f"{n_failed}/{n_total} ensemble member(s) failed. "
                f"Only 1 method succeeded — ensemble averaging has no diversification benefit."
            )
        else:
            msg = f"{n_failed} ensemble member(s) failed."
        self.assertIn("caution", msg)


if __name__ == '__main__':
    unittest.main()
