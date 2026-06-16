"""Tests for optimize output modes, grid styles, constraints, refinement,
tie-probability accounting, and the trade-gate / actionability fields.
"""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from ._helpers import _BarrierTestBase, _BARRIER_PROB_ROOT, _BARRIER_OPT_ROOT
from mtdata.forecast.barriers_optimization import forecast_barrier_optimize
from mtdata.forecast.barriers_shared import _build_actionability_payload, _build_selection_diagnostics


class TestBarrierOptimizeOutputGrid(_BarrierTestBase):
    """Output modes, grids (volatility/ratio/preset), constraints, refinement,
    tie probabilities, bridge hits, and trade-gate/actionability behaviour."""

    def test_forecast_barrier_optimize_refine_and_metrics(self):
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
                output_mode="summary",
                top_k=2,
                return_grid=True,
            )
        self.assertTrue(result["success"])
        grid = result.get("grid")
        self.assertTrue(grid)
        self.assertEqual(len(grid), 2)
        self.assertEqual(result["best"], grid[0])
        self.assertEqual(result.get("output_mode"), "summary")
        self.assertNotIn("diagnostics", result)
        self.assertIn("compute_profile", result)

    def test_forecast_barrier_optimize_uses_null_profit_factor_when_lossless(self):
        self._set_flat_history(1.0)
        paths = np.repeat(np.array([[1.0, 1.01]]), 20, axis=0)
        with patch(f'{_BARRIER_OPT_ROOT}._simulate_gbm_mc') as mock_sim:
            mock_sim.return_value = {"price_paths": paths}
            result = forecast_barrier_optimize(
                symbol="EURUSD",
                timeframe="H1",
                horizon=2,
                method="mc_gbm",
                direction="long",
                mode="pct",
                tp_min=0.5,
                tp_max=0.5,
                tp_steps=1,
                sl_min=0.5,
                sl_max=0.5,
                sl_steps=1,
                objective="profit_factor",
                return_grid=True,
            )

        self.assertTrue(result["success"])
        best = result["best"]
        self.assertIsNone(best["profit_factor"])
        self.assertEqual(
            best["profit_factor_note"],
            "Undefined: no simulated losses for this barrier pair.",
        )
        self.assertNotEqual(best["profit_factor"], 1e9)

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
                output_mode="full",
                return_grid=True,
            )
        self.assertTrue(result["success"])
        self.assertIn("results_total", result)
        self.assertEqual(result["results_total"], len(result["grid"]))
        self.assertLessEqual(len(result["results"]), 10)
        self.assertGreaterEqual(len(result["grid"]), len(result["results"]))
        self.assertEqual(result.get("output_mode"), "full")
        self.assertIn("diagnostics", result)
        self.assertEqual(result["diagnostics"]["candidate_counts"]["returned_total"], result["results_total"])
        self.assertEqual(result["diagnostics"]["candidate_counts"]["grid_total"], len(result["grid"]))

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
                viable_only=False,
            )
        self.assertTrue(result["success"])
        grid = result.get("grid")
        self.assertTrue(grid)
        for entry in grid:
            self.assertGreaterEqual(entry["prob_tp_first"], 0.5)
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
                mode="ticks",
                tp_min=5.0, tp_max=10.0, tp_steps=2,
                sl_min=5.0, sl_max=10.0, sl_steps=2,
                return_grid=True,
            )
        self.assertTrue(result["success"])
        grid = result.get("grid")
        self.assertTrue(grid)

    def test_forecast_barrier_optimize_tie_probabilities_sum_to_one(self):
        paths = np.array([
            [1.0, 1.01, 1.02, 1.03],
            [1.0, 0.99, 0.98, 0.97],
            [1.0, 1.002, 0.998, 1.006],
        ])

        def _force_bridge_ties(*args, **kwargs):
            uniform = kwargs.get("uniform")
            return np.ones_like(uniform, dtype=bool)

        with patch(f'{_BARRIER_OPT_ROOT}._simulate_gbm_mc') as mock_sim, \
             patch(f'{_BARRIER_OPT_ROOT}._get_live_reference_price', return_value=(None, None)), \
             patch(f'{_BARRIER_OPT_ROOT}._brownian_bridge_hits', side_effect=_force_bridge_ties):
            mock_sim.return_value = {"price_paths": paths}
            result = forecast_barrier_optimize(
                symbol="EURUSD",
                timeframe="H1",
                horizon=4,
                method="mc_gbm_bb",
                direction="long",
                mode="pct",
                tp_min=0.5, tp_max=1.0, tp_steps=2,
                sl_min=0.5, sl_max=1.0, sl_steps=2,
                objective="edge",
                return_grid=True,
                viable_only=False,
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
            expected_kelly = entry["prob_tp_first"] - (entry["prob_sl_first"] / entry["rr"])
            self.assertAlmostEqual(entry["kelly"], expected_kelly, places=7)
            self.assertAlmostEqual(entry["ev_cond"], expected_ev, places=7)
            self.assertAlmostEqual(entry["kelly_cond"], expected_kelly, places=7)
            expected_profit_factor = (
                (entry["prob_tp_first"] * entry["tp"])
                / (entry["prob_sl_first"] * entry["sl"])
            )
            self.assertAlmostEqual(entry["profit_factor"], expected_profit_factor, places=7)
            reward_frac = entry["tp"] / 100.0
            risk_frac = entry["sl"] / 100.0
            expected_utility = (
                entry["prob_tp_first"] * np.log1p(reward_frac)
                + entry["prob_sl_first"] * np.log1p(-risk_frac)
            )
            self.assertAlmostEqual(entry["utility"], expected_utility, places=7)

    def test_forecast_barrier_optimize_min_prob_win_uses_tie_adjusted_probability(self):
        paths = np.repeat(np.array([[1.0, 1.01, 1.02, 1.03]]), 100, axis=0)

        def _force_bridge_ties(*args, **kwargs):
            uniform = kwargs.get("uniform")
            return np.ones_like(uniform, dtype=bool)

        with patch(f'{_BARRIER_OPT_ROOT}._simulate_gbm_mc') as mock_sim, \
             patch(f'{_BARRIER_OPT_ROOT}._get_live_reference_price', return_value=(None, None)), \
             patch(f'{_BARRIER_OPT_ROOT}._brownian_bridge_hits', side_effect=_force_bridge_ties):
            mock_sim.return_value = {"price_paths": paths}
            result = forecast_barrier_optimize(
                symbol="EURUSD",
                timeframe="H1",
                horizon=4,
                method="mc_gbm_bb",
                direction="long",
                mode="pct",
                tp_min=0.75,
                tp_max=0.75,
                tp_steps=1,
                sl_min=0.5,
                sl_max=0.5,
                sl_steps=1,
                min_prob_win=0.5,
                objective="ev",
                return_grid=True,
                viable_only=False,
            )

        self.assertTrue(result.get("success"))
        self.assertEqual(result.get("status"), "ok")
        self.assertFalse(result.get("no_candidates"))
        self.assertTrue(result.get("viable"))
        self.assertFalse(result.get("no_action"))
        self.assertTrue(result.get("trade_gate_passed"))
        self.assertEqual(result.get("actionability"), "actionable")
        self.assertNotIn("ev_edge_conflict", result)
        best = result["best"]
        self.assertAlmostEqual(best["prob_win"], 0.0, places=7)
        self.assertAlmostEqual(best["prob_tp_first"], 0.5, places=7)
        self.assertGreater(float(best["edge_vs_breakeven"]), 0.0)

    def test_forecast_barrier_optimize_no_action_follows_trade_gate_when_status_ok(self):
        self._set_flat_history(1.0)
        wins = np.repeat(np.array([[1.0, 1.01]]), 20, axis=0)
        losses = np.repeat(np.array([[1.0, 0.995]]), 10, axis=0)
        unresolved = np.repeat(np.array([[1.0, 1.002]]), 70, axis=0)
        paths = np.vstack([wins, losses, unresolved])

        with patch(f'{_BARRIER_OPT_ROOT}._simulate_gbm_mc') as mock_sim:
            mock_sim.return_value = {"price_paths": paths}
            result = forecast_barrier_optimize(
                symbol="EURUSD",
                timeframe="H1",
                horizon=2,
                method="mc_gbm",
                direction="long",
                mode="pct",
                tp_min=1.0,
                tp_max=1.0,
                tp_steps=1,
                sl_min=0.5,
                sl_max=0.5,
                sl_steps=1,
                objective="ev",
                return_grid=True,
            )

        self.assertTrue(result.get("success"))
        self.assertEqual(result.get("status"), "ok")
        self.assertTrue(result.get("viable"))
        self.assertTrue(result.get("ev_edge_conflict"))
        self.assertEqual(result.get("actionability"), "blocked")
        self.assertFalse(result.get("trade_gate_passed"))
        self.assertFalse(result.get("tradable"))
        self.assertTrue(result.get("mathematically_viable"))
        self.assertIn("EV screen", result.get("viability_note", ""))
        self.assertIn("metric_interpretation", result)
        self.assertIn(
            "edge_vs_breakeven=prob_tp_first-breakeven_win_rate",
            result["metric_interpretation"],
        )
        self.assertTrue(result.get("no_action"))
        self.assertIn("ev_edge_conflict", result.get("actionability_flags", []))

    def test_barrier_diagnostics_block_low_practical_win_probability(self):
        row = {
            "ev": 0.02,
            "edge": -0.35,
            "prob_win": 0.004,
            "prob_tp_first": 0.004,
            "prob_loss": 0.35,
            "prob_sl_first": 0.35,
            "prob_no_hit": 0.646,
            "rr": 6.0,
            "tp": 1.5,
            "sl": 0.25,
            "profit_factor": 0.014,
        }

        diagnostics = _build_selection_diagnostics(row)
        actionability = _build_actionability_payload(
            status="ok",
            row=row,
            diagnostics=diagnostics,
        )

        self.assertTrue(diagnostics["low_practical_win_probability"])
        self.assertEqual(diagnostics["low_practical_win_probability_threshold"], 0.05)
        self.assertIn("unresolved/no-hit paths", diagnostics["metric_interpretation"])
        self.assertTrue(
            any("raw win/loss edge" in msg for msg in diagnostics["selection_warnings"])
        )
        self.assertEqual(actionability["actionability"], "blocked")
        self.assertFalse(actionability["trade_gate_passed"])
        self.assertIn(
            "low_practical_win_probability",
            actionability["actionability_flags"],
        )


if __name__ == '__main__':
    unittest.main()
