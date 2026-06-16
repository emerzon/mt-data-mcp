"""Tests for low-level candidate evaluation: viability check, unresolved
terminal PnL accounting, and barrier geometry validation.
"""

import unittest

import numpy as np

from mtdata.forecast.barriers_shared import _candidate_is_viable


class TestCandidateViability(unittest.TestCase):
    """Tests for _candidate_is_viable from barriers_shared."""

    def test_low_win_probability_candidate_is_not_viable(self):
        candidate = {
            "ev": 0.017,
            "tp": 1.5,
            "sl": 0.25,
            "rr": 6.0,
            "prob_win": 0.001,
            "prob_loss": 0.001,
            "prob_tp_first": 0.001,
            "prob_sl_first": 0.001,
            "prob_no_hit": 0.998,
        }

        self.assertFalse(_candidate_is_viable(candidate))


class TestUnresolvedTerminalPnl(unittest.TestCase):
    """Tests for unresolved-path terminal PnL contribution to barrier EV."""

    def _make_context(self, *, mode="ticks", dir_long=True, last_price=1.1000, pip_size=0.0001):
        from mtdata.forecast.barriers_optimization import _BarrierEvaluationContext
        return _BarrierEvaluationContext(
            mode_val=mode,
            dir_long=dir_long,
            last_price=last_price,
            pip_size=pip_size,
            rr_min_val=None,
            rr_max_val=None,
            has_trading_costs=False,
            ev_deduct_cost=0.0,
            cost_per_trade=0.0,
            min_prob_win_val=None,
            max_prob_no_hit_val=None,
            min_prob_resolve_val=None,
            max_median_time_val=None,
        )

    def test_unresolved_terminal_pnl_long_pips(self):
        """Unresolved long paths that drift up give positive PnL in pips."""
        from mtdata.forecast.barriers_optimization import _unresolved_terminal_pnl
        ctx = self._make_context(last_price=1.1000, pip_size=0.0001, dir_long=True)
        paths = np.full((5, 10), 1.1010)
        mask = np.ones(5, dtype=bool)
        pnl = _unresolved_terminal_pnl(paths, mask, context=ctx)
        self.assertAlmostEqual(pnl, 10.0, places=1)

    def test_unresolved_terminal_pnl_short_pips(self):
        """Unresolved short paths that drift down give positive PnL in pips."""
        from mtdata.forecast.barriers_optimization import _unresolved_terminal_pnl
        ctx = self._make_context(last_price=1.1000, pip_size=0.0001, dir_long=False)
        paths = np.full((5, 10), 1.0990)  # Price dropped 10 pips
        mask = np.ones(5, dtype=bool)
        pnl = _unresolved_terminal_pnl(paths, mask, context=ctx)
        self.assertAlmostEqual(pnl, 10.0, places=1)

    def test_unresolved_terminal_pnl_pct_mode(self):
        """Pct mode: long path ending 1% up gives +1.0 pct."""
        from mtdata.forecast.barriers_optimization import _unresolved_terminal_pnl
        ctx = self._make_context(mode="pct", last_price=100.0, dir_long=True)
        paths = np.full((4, 5), 101.0)  # +1%
        mask = np.ones(4, dtype=bool)
        pnl = _unresolved_terminal_pnl(paths, mask, context=ctx)
        self.assertAlmostEqual(pnl, 1.0, places=4)

    def test_unresolved_terminal_pnl_no_unresolved(self):
        """No unresolved paths → returns 0."""
        from mtdata.forecast.barriers_optimization import _unresolved_terminal_pnl
        ctx = self._make_context()
        paths = np.full((3, 5), 1.1010)
        mask = np.zeros(3, dtype=bool)
        pnl = _unresolved_terminal_pnl(paths, mask, context=ctx)
        self.assertEqual(pnl, 0.0)

    def test_ev_unresolved_appears_in_candidate_result(self):
        """_evaluate_barrier_candidate includes ev_unresolved in output."""
        from mtdata.forecast.barriers_optimization import (
            _BarrierBridgeInputs,
            _evaluate_barrier_candidate,
        )
        ctx = self._make_context(last_price=1.1000, pip_size=0.0001, dir_long=True)
        bridge = _BarrierBridgeInputs(enabled=False, sigma=0.0, log_paths=None, uniform_tp=None, uniform_sl=None)
        paths = np.full((100, 10), 1.1000)
        result, is_invalid = _evaluate_barrier_candidate(
            50.0, 50.0, paths, context=ctx, bridge_inputs=bridge,
        )
        self.assertIsNotNone(result)
        self.assertIn("ev_unresolved", result)
        self.assertTrue(result["zero_win_probability"])
        self.assertEqual(
            result["warning"],
            "prob_win is 0: no simulated paths reached TP within horizon.",
        )

    def test_evaluate_barrier_candidate_rejects_empty_paths(self):
        from mtdata.forecast.barriers_optimization import (
            _BarrierBridgeInputs,
            _evaluate_barrier_candidate,
        )

        ctx = self._make_context(last_price=1.1000, pip_size=0.0001, dir_long=True)
        bridge = _BarrierBridgeInputs(enabled=False, sigma=0.0, log_paths=None, uniform_tp=None, uniform_sl=None)

        result, is_invalid = _evaluate_barrier_candidate(
            50.0,
            50.0,
            np.empty((0, 10)),
            context=ctx,
            bridge_inputs=bridge,
        )

        self.assertIsNone(result)
        self.assertTrue(is_invalid)


class TestCandidateBarrierGeometry(unittest.TestCase):
    """Tests for _candidate_barrier_geometry_is_valid."""

    def _make_context(self, *, dir_long=True, last_price=1.1000):
        from mtdata.forecast.barriers_optimization import _BarrierEvaluationContext
        return _BarrierEvaluationContext(
            mode_val="pct",
            dir_long=dir_long,
            last_price=last_price,
            pip_size=0.0001,
            rr_min_val=None,
            rr_max_val=None,
            has_trading_costs=False,
            ev_deduct_cost=0.0,
            cost_per_trade=0.0,
            min_prob_win_val=None,
            max_prob_no_hit_val=None,
            min_prob_resolve_val=None,
            max_median_time_val=None,
        )

    def test_rejects_non_positive_or_non_finite_anchor_price(self):
        from mtdata.forecast.barriers_optimization import _candidate_barrier_geometry_is_valid

        for last_price in (0.0, -1.0, float("nan"), float("inf")):
            ctx = self._make_context(last_price=last_price)
            assert _candidate_barrier_geometry_is_valid(101.0, 99.0, context=ctx) is False


if __name__ == '__main__':
    unittest.main()
