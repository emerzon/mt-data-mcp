"""Tests for forecast_barrier_hit_probabilities, forecast_barrier_closed_form,
and related probability-computation paths (GBM, HMM, bootstrap, GARCH).
"""

import importlib.util
import unittest
from datetime import datetime, timezone
from unittest.mock import patch

import numpy as np
import pandas as pd

from mtdata.forecast.barriers_optimization import forecast_barrier_optimize
from mtdata.forecast.barriers_probabilities import (
    _history_freshness_context,
    forecast_barrier_closed_form,
    forecast_barrier_hit_probabilities,
)
from mtdata.forecast.monte_carlo import gbm_single_barrier_upcross_prob

from ._helpers import _BARRIER_OPT_ROOT, _BARRIER_PROB_ROOT, _BarrierTestBase

# ---------------------------------------------------------------------------
# Standalone tests (no mock history needed)
# ---------------------------------------------------------------------------

def test_barrier_history_freshness_relaxes_for_closed_weekend():
    saturday = datetime(2026, 6, 6, 12, tzinfo=timezone.utc).timestamp()
    friday_close = datetime(2026, 6, 5, 20, tzinfo=timezone.utc).timestamp()
    frame = pd.DataFrame({"time": [friday_close]})

    result = _history_freshness_context(
        frame,
        "H1",
        symbol="EURUSD",
        now_epoch=saturday,
    )

    assert result["data_stale"] is False
    assert result["market_status_reason"] == "weekend"
    assert result["freshness"].startswith("closed weekend, data ")


def test_barrier_optimize_rejects_removed_profile_alias():
    result = forecast_barrier_optimize(
        symbol="EURUSD",
        timeframe="H1",
        params={"profile": "long"},
    )

    assert result == {
        "error": (
            "params.profile is not supported. Use search_profile either as "
            "the tool parameter or inside params."
        )
    }


# ---------------------------------------------------------------------------
# Main test class
# ---------------------------------------------------------------------------

class TestBarrierHitProbabilities(_BarrierTestBase):
    """Tests for forecast_barrier_hit_probabilities and forecast_barrier_closed_form."""

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
        self.assertIn("prob_same_bar", result)
        self.assertEqual(result["same_bar_policy"], "sl_first")
        self.assertIn("prob_tp_first_ci95", result)
        self.assertIn("prob_sl_first_ci95", result)
        self.assertIn("prob_no_hit_ci95", result)
        self.assertIn("prob_tp_first_se", result)
        self.assertIn("prob_sl_first_se", result)
        self.assertEqual(result["intra_bar_hit_detection"], "simulated_bar_close")
        self.assertTrue(any("intra-bar touches" in item for item in result["warnings"]))

    def test_forecast_barrier_hit_probabilities_default_seed_is_deterministic(self):
        dates = pd.date_range(start='2023-01-01', periods=500, freq='h')
        prices = np.linspace(1.0, 1.05, 500)
        self._set_barrier_history(pd.DataFrame({'time': dates, 'close': prices}))

        kwargs = {
            "symbol": "EURUSD",
            "timeframe": "H1",
            "horizon": 5,
            "method": "mc_gbm",
            "direction": "long",
            "tp_pct": 0.5,
            "sl_pct": 0.5,
            "params": {"n_sims": 50},
        }
        with patch(f'{_BARRIER_PROB_ROOT}._get_live_reference_price', return_value=(None, None)):
            first = forecast_barrier_hit_probabilities(**kwargs)
            second = forecast_barrier_hit_probabilities(**kwargs)

        self.assertTrue(first["success"])
        self.assertEqual(first["seed"], second["seed"])
        self.assertEqual(first["seed_source"], "request")
        self.assertEqual(first["n_sims"], 50)
        self.assertEqual(first["prob_tp_first"], second["prob_tp_first"])
        self.assertEqual(first["prob_sl_first"], second["prob_sl_first"])
        self.assertEqual(first["prob_no_hit"], second["prob_no_hit"])

    def test_forecast_barrier_hit_probabilities_normalizes_oversized_seed(self):
        self._set_flat_history(1.0)
        seen_seeds = []

        def _fake_sim(*args, seed=None, **kwargs):
            seen_seeds.append(seed)
            return {"price_paths": self._sample_paths()}

        with patch(f'{_BARRIER_PROB_ROOT}._simulate_gbm_mc', side_effect=_fake_sim), \
             patch(f'{_BARRIER_PROB_ROOT}._get_live_reference_price', return_value=(None, None)):
            result = forecast_barrier_hit_probabilities(
                symbol="EURUSD",
                timeframe="H1",
                horizon=4,
                method="mc_gbm",
                direction="long",
                tp_pct=0.5,
                sl_pct=0.5,
                params={"seed": 2**32 + 5, "n_sims": 10},
            )

        self.assertTrue(result["success"])
        self.assertEqual(result["seed"], 5)
        self.assertEqual(seen_seeds, [5])

    def test_forecast_barrier_hit_probabilities_accepts_tick_aliases(self):
        result = forecast_barrier_hit_probabilities(
            symbol="EURUSD",
            timeframe="H1",
            horizon=10,
            method="mc_gbm",
            direction="long",
            tp_ticks=5,
            sl_ticks=5,
        )
        self.assertTrue(result["success"])

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
        self.assertEqual(len(result["tp_hit_prob_by_t"]), 4)
        self.assertEqual(len(result["sl_hit_prob_by_t"]), 4)
        self.assertAlmostEqual(result["tp_hit_prob_by_t"][0], 0.0, places=8)
        self.assertAlmostEqual(result["sl_hit_prob_by_t"][0], 0.0, places=8)
        self.assertNotIn("hit_prob_by_t", result)
        self.assertNotIn("time_to_tp_seconds", result)
        self.assertNotIn("time_to_sl_seconds", result)
        self.assertNotIn("time_to_hit_seconds_derived", result)
        self.assertNotIn("time_to_hit_seconds_formula", result)

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

    def test_forecast_barrier_hmm_warns_when_states_collapse(self):
        self._set_flat_history(1.0, bars=200)
        paths = self._sample_paths()
        with patch(f'{_BARRIER_PROB_ROOT}._simulate_hmm_mc') as mock_sim:
            mock_sim.return_value = {
                "price_paths": paths,
                "requested_n_states": 2,
                "fitted_n_states": 1,
                "model_type": "gaussian_hmm_baum_welch",
            }
            result = forecast_barrier_hit_probabilities(
                symbol="EURUSD",
                timeframe="H1",
                horizon=4,
                method="hmm_mc",
                direction="long",
                tp_pct=0.5,
                sl_pct=0.5,
            )

        self.assertTrue(result["success"])
        self.assertEqual(result["sim_meta"]["requested_n_states"], 2)
        self.assertEqual(result["sim_meta"]["fitted_n_states"], 1)
        self.assertIn(
            "HMM state collapse: requested 2 states but fitted 1; "
            "probabilities use the reduced-state model.",
            result["warnings"],
        )

    def test_forecast_barrier_hit_probabilities_surfaces_denoise_warning(self):
        self._set_flat_history(1.0, bars=200)
        paths = self._sample_paths()
        with patch(f'{_BARRIER_PROB_ROOT}._simulate_gbm_mc') as mock_sim, \
             patch("mtdata.utils.denoise._apply_denoise", side_effect=RuntimeError("bad denoise")):
            mock_sim.return_value = {"price_paths": paths}
            result = forecast_barrier_hit_probabilities(
                symbol="EURUSD",
                timeframe="H1",
                horizon=4,
                method="mc_gbm",
                direction="long",
                tp_pct=0.5,
                sl_pct=0.5,
                denoise={"method": "wavelet"},
            )
        self.assertTrue(result["success"])
        self.assertIn("warnings", result)
        self.assertIn("using raw close prices instead", result["warnings"][0])

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
        self.assertEqual(result["last_price_source"], "candle_close")

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


if __name__ == '__main__':
    unittest.main()
