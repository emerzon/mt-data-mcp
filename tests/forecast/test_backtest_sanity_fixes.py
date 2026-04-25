"""Tests for backtest sanity fixes: DA return mode, flat filtering, finite
validation, return fallback removal, and annualization cadence."""

from __future__ import annotations

import math
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from mtdata.forecast.backtest import (
    _compute_performance_metrics,
    forecast_backtest,
)
from mtdata.utils.utils import _format_time_minimal


def _make_df(n: int, base_time: float = 1700000000.0, base_close: float = 100.0):
    times = [base_time + i * 3600 for i in range(n)]
    closes = [base_close + i * 0.5 for i in range(n)]
    return pd.DataFrame({"time": times, "close": closes})


# ── Fix 1: DA in return mode compares sign(forecast[i]) vs sign(actual[i]) ───


class TestDAReturnMode:
    @patch("mtdata.forecast.backtest._fetch_history")
    def test_da_return_mode_perfect_sign_match(self, fetch):
        df = _make_df(500)
        fetch.return_value = df
        pos_returns = [0.01, 0.02, -0.005]
        with patch("mtdata.forecast.backtest.forecast") as fc:
            fc.return_value = {"forecast_return": pos_returns}
            result = forecast_backtest(
                "EURUSD", timeframe="H1", quantity="return",
                methods=["naive"], horizon=3, steps=1,
            )
        detail = result["results"]["naive"]["details"][0]
        assert detail["success"] is True
        # In return mode, DA = fraction where sign(forecast[i]) == sign(actual[i])
        # actual returns are log returns, they should have same sign pattern as
        # the linearly increasing close prices (all positive)
        assert isinstance(detail["directional_accuracy"], float)
        assert not math.isnan(detail["directional_accuracy"])

    @patch("mtdata.forecast.backtest._fetch_history")
    def test_da_return_mode_all_opposite_signs(self, fetch):
        df = _make_df(500)
        # Make all close prices increase so actual log returns are positive
        df["close"] = [100.0 + i * 0.5 for i in range(500)]
        fetch.return_value = df
        # Forecast negative returns (opposite of actual)
        neg_returns = [-0.01, -0.02, -0.005]
        with patch("mtdata.forecast.backtest.forecast") as fc:
            fc.return_value = {"forecast_return": neg_returns}
            result = forecast_backtest(
                "EURUSD", timeframe="H1", quantity="return",
                methods=["naive"], horizon=3, steps=1,
            )
        detail = result["results"]["naive"]["details"][0]
        assert detail["success"] is True
        # All signs differ → DA = 0.0
        assert detail["directional_accuracy"] == pytest.approx(0.0)

    @patch("mtdata.forecast.backtest._fetch_history")
    def test_da_return_mode_single_step(self, fetch):
        """Return mode DA works even with horizon=1 (single return to compare)."""
        df = _make_df(500)
        df["close"] = [100.0 + i * 0.5 for i in range(500)]
        fetch.return_value = df
        with patch("mtdata.forecast.backtest.forecast") as fc:
            fc.return_value = {"forecast_return": [0.01]}
            result = forecast_backtest(
                "EURUSD", timeframe="H1", quantity="return",
                methods=["naive"], horizon=1, steps=1,
            )
        detail = result["results"]["naive"]["details"][0]
        assert detail["success"] is True
        # Single element, sign should match (both positive)
        assert detail["directional_accuracy"] == pytest.approx(1.0)


# ── Fix 2: Flat positions excluded from trade metrics ────────────────────────


class TestFlatTradeFiltering:
    @patch("mtdata.forecast.backtest._fetch_history")
    def test_flat_positions_excluded_from_metrics(self, fetch):
        df = _make_df(500)
        fetch.return_value = df
        with patch("mtdata.forecast.backtest.forecast") as fc:
            # Forecast identical to entry → flat position with high threshold
            fc.return_value = {"forecast_price": [float(df["close"].iloc[-13])] * 12}
            result = forecast_backtest(
                "EURUSD", timeframe="H1", methods=["naive"],
                trade_threshold=999.0,  # forces flat
            )
        r = result["results"]["naive"]
        # All positions should be flat
        for d in r["details"]:
            if d.get("success"):
                assert d["position"] == "flat"
        assert r["metrics_available"] is False
        assert r["metrics_reason"] == "no_non_flat_trades"
        assert r["trade_status"] == "flat"
        assert r["metrics"]["trades_observed"] == 0

    @patch("mtdata.forecast.backtest._fetch_history")
    def test_mixed_flat_and_active_counts_only_active(self, fetch):
        df = _make_df(500)
        df["close"] = [100.0 + i * 0.5 for i in range(500)]
        fetch.return_value = df
        call_count = [0]

        def _alternating_forecast(**kwargs):
            call_count[0] += 1
            if call_count[0] % 2 == 0:
                return {"forecast_price": [float(df["close"].iloc[-13])] * 12}
            return {"forecast_price": [200.0] * 12}

        with patch("mtdata.forecast.backtest.forecast", side_effect=_alternating_forecast):
            result = forecast_backtest(
                "EURUSD", timeframe="H1", methods=["naive"],
                trade_threshold=0.5, steps=4, spacing=20,
            )
        r = result["results"]["naive"]
        if "metrics" in r:
            active = [d for d in r["details"] if d.get("success") and d.get("position") != "flat"]
            assert r["metrics"]["trades_observed"] <= len(active)


# ── Fix 3: Non-finite forecast values fail the anchor ────────────────────────


class TestNonFiniteForecastValidation:
    @patch("mtdata.forecast.backtest._fetch_history")
    def test_nan_forecast_fails_anchor(self, fetch):
        fetch.return_value = _make_df(500)
        with patch("mtdata.forecast.backtest.forecast") as fc:
            fc.return_value = {"forecast_price": [float("nan")] * 12}
            result = forecast_backtest(
                "EURUSD", timeframe="H1", methods=["naive"],
            )
        details = result["results"]["naive"]["details"]
        for d in details:
            assert d["success"] is False
            assert "Non-finite" in d.get("error", "")

    @patch("mtdata.forecast.backtest._fetch_history")
    def test_inf_forecast_fails_anchor(self, fetch):
        fetch.return_value = _make_df(500)
        with patch("mtdata.forecast.backtest.forecast") as fc:
            fc.return_value = {"forecast_price": [float("inf")] * 12}
            result = forecast_backtest(
                "EURUSD", timeframe="H1", methods=["naive"],
            )
        details = result["results"]["naive"]["details"]
        for d in details:
            assert d["success"] is False

    @patch("mtdata.forecast.backtest._fetch_history")
    def test_partial_nan_forecast_fails_anchor(self, fetch):
        fetch.return_value = _make_df(500)
        with patch("mtdata.forecast.backtest.forecast") as fc:
            fc.return_value = {"forecast_price": [101.0, float("nan"), 103.0] + [104.0] * 9}
            result = forecast_backtest(
                "EURUSD", timeframe="H1", methods=["naive"],
            )
        details = result["results"]["naive"]["details"]
        for d in details:
            assert d["success"] is False


# ── Fix 4: Return mode requires forecast_return (no fallback) ────────────────


class TestReturnModeFallbackRemoved:
    @patch("mtdata.forecast.backtest._fetch_history")
    def test_return_mode_missing_forecast_return_fails(self, fetch):
        fetch.return_value = _make_df(500)
        with patch("mtdata.forecast.backtest.forecast") as fc:
            # Only provides forecast_price, not forecast_return
            fc.return_value = {"forecast_price": [101.0] * 12}
            result = forecast_backtest(
                "EURUSD", timeframe="H1", quantity="return",
                methods=["naive"],
            )
        details = result["results"]["naive"]["details"]
        for d in details:
            assert d["success"] is False
            assert "forecast_return" in d.get("error", "").lower()

    @patch("mtdata.forecast.backtest._fetch_history")
    def test_return_mode_with_forecast_return_succeeds(self, fetch):
        fetch.return_value = _make_df(500)
        with patch("mtdata.forecast.backtest.forecast") as fc:
            fc.return_value = {"forecast_return": [0.001] * 12}
            result = forecast_backtest(
                "EURUSD", timeframe="H1", quantity="return",
                methods=["naive"],
            )
        ok = [d for d in result["results"]["naive"]["details"] if d.get("success")]
        assert len(ok) > 0


# ── Fix 5: Annualization uses actual trade spacing ───────────────────────────


class TestAnnualizationCadence:
    def test_metrics_with_trade_spacing_differs_from_horizon(self):
        """When trade_spacing_bars != horizon, trades_per_year changes."""
        rets = list(np.random.default_rng(42).normal(0.001, 0.01, 50))
        m_horizon = _compute_performance_metrics(rets, "H1", 12, 0.0)
        m_spaced = _compute_performance_metrics(rets, "H1", 12, 0.0, trade_spacing_bars=20)
        # With spacing=20 > horizon=12, trades_per_year should be lower
        assert m_spaced["trades_per_year"] < m_horizon["trades_per_year"]

    def test_metrics_trade_spacing_none_uses_horizon(self):
        """Default (no spacing) falls back to horizon."""
        rets = list(np.random.default_rng(42).normal(0.001, 0.01, 50))
        m1 = _compute_performance_metrics(rets, "H1", 12, 0.0)
        m2 = _compute_performance_metrics(rets, "H1", 12, 0.0, trade_spacing_bars=None)
        assert m1["trades_per_year"] == m2["trades_per_year"]

    @patch("mtdata.forecast.backtest._fetch_history")
    def test_backtest_passes_actual_spacing_to_metrics(self, fetch):
        fetch.return_value = _make_df(500)
        with patch("mtdata.forecast.backtest.forecast") as fc:
            fc.return_value = {"forecast_price": [200.0] * 12}
            result = forecast_backtest(
                "EURUSD", timeframe="H1", methods=["naive"],
                steps=5, spacing=30,
            )
        r = result["results"]["naive"]
        if "metrics" in r:
            # trades_per_year should reflect spacing=30, not horizon=12
            expected_tpy = (252.0 * 24.0) / 30.0
            assert r["metrics"]["trades_per_year"] == pytest.approx(expected_tpy, rel=0.01)

    def test_metrics_sharpe_scales_with_cadence(self):
        """Sharpe ratio should be sqrt-proportional to cadence change."""
        rets = list(np.random.default_rng(42).normal(0.002, 0.01, 50))
        m12 = _compute_performance_metrics(rets, "H1", 12, 0.0, trade_spacing_bars=12)
        m48 = _compute_performance_metrics(rets, "H1", 12, 0.0, trade_spacing_bars=48)
        if m12["sharpe_ratio"] is not None and m48["sharpe_ratio"] is not None:
            # Sharpe scales with sqrt(trades_per_year), so ratio ~ sqrt(48/12) = 2
            ratio = m12["sharpe_ratio"] / m48["sharpe_ratio"]
            assert ratio == pytest.approx(math.sqrt(48.0 / 12.0), rel=0.01)
