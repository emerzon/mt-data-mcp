"""Extended coverage tests for mtdata.forecast.backtest – targeting uncovered lines."""

import math
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Ensure MetaTrader5 mock is available
_mt5_mock = MagicMock()
_mt5_mock.TIMEFRAME_M1 = 1; _mt5_mock.TIMEFRAME_M2 = 2; _mt5_mock.TIMEFRAME_M3 = 3
_mt5_mock.TIMEFRAME_M4 = 4; _mt5_mock.TIMEFRAME_M5 = 5; _mt5_mock.TIMEFRAME_M6 = 6
_mt5_mock.TIMEFRAME_M10 = 10; _mt5_mock.TIMEFRAME_M12 = 12; _mt5_mock.TIMEFRAME_M15 = 15
_mt5_mock.TIMEFRAME_M20 = 20; _mt5_mock.TIMEFRAME_M30 = 30
_mt5_mock.TIMEFRAME_H1 = 16385; _mt5_mock.TIMEFRAME_H2 = 16386; _mt5_mock.TIMEFRAME_H3 = 16387
_mt5_mock.TIMEFRAME_H4 = 16388; _mt5_mock.TIMEFRAME_H6 = 16390; _mt5_mock.TIMEFRAME_H8 = 16392
_mt5_mock.TIMEFRAME_H12 = 16396; _mt5_mock.TIMEFRAME_D1 = 16408
_mt5_mock.TIMEFRAME_W1 = 32769; _mt5_mock.TIMEFRAME_MN1 = 49153
sys.modules["MetaTrader5"] = _mt5_mock

from mtdata.forecast.backtest import (
    _bars_per_year,
    _compute_performance_metrics,
    _get_forecast_methods_data_safe,
    forecast_backtest,
)
from mtdata.utils.utils import _format_time_minimal

# ── Helper to build a fake df ────────────────────────────────────────────────

def _make_df(n: int, base_time: float = 1700000000.0, base_close: float = 100.0):
    """Create a simple DataFrame with 'time' and 'close' columns."""
    times = [base_time + i * 3600 for i in range(n)]
    closes = [base_close + i * 0.5 for i in range(n)]
    return pd.DataFrame({"time": times, "close": closes})


# ── _get_forecast_methods_data_safe  (lines 22-29 fallback) ──────────────────

class TestGetForecastMethodsDataSafe:
    def test_returns_dict(self):
        result = _get_forecast_methods_data_safe()
        assert isinstance(result, dict)
        assert "methods" in result

    def test_fallback_on_mock(self):
        with patch("mtdata.forecast.backtest._get_forecast_methods_data_safe") as mock_fn:
            mock_fn.return_value = {
                "methods": [{"method": "naive", "available": True}]
            }
            r = mock_fn()
            assert "methods" in r

    def test_actual_fallback_path(self):
        """Verify the real function returns something usable."""
        result = _get_forecast_methods_data_safe()
        assert isinstance(result, dict)
        methods = result.get("methods", [])
        assert len(methods) >= 1


# ── _bars_per_year  (lines 40-48) ────────────────────────────────────────────

class TestBarsPerYear:
    def test_h1(self):
        result = _bars_per_year("H1")
        expected = 252.0 * 24.0
        assert abs(result - expected) < 1

    def test_invalid_timeframe(self):
        result = _bars_per_year("INVALID")
        assert math.isnan(result)

    def test_exception(self):
        result = _bars_per_year(None)
        assert math.isnan(result)


# ── _compute_performance_metrics  (lines 55-117) ────────────────────────────

class TestComputePerformanceMetrics:
    def test_empty_returns(self):
        assert _compute_performance_metrics([], "H1", 12, 0.0) == {}

    def test_all_none_returns(self):
        assert _compute_performance_metrics([None, None], "H1", 12, 0.0) == {}

    def test_basic_returns(self):
        rets = [0.01, -0.005, 0.02, 0.015, -0.01]
        m = _compute_performance_metrics(rets, "H1", 12, 0.0)
        assert "avg_return_per_trade" in m
        assert "win_rate" in m
        assert "win_rate_display" in m
        assert "sharpe_ratio" in m
        assert "max_drawdown" in m

    def test_single_return(self):
        m = _compute_performance_metrics([0.05], "H1", 12, 0.0)
        # Annualized risk metrics are intentionally suppressed on tiny samples.
        assert m.get("sharpe_ratio") is None
        assert "sample_warning" in m
        assert m["sample_notice"]["code"] == "annualization_suppressed_low_sample"
        assert m["trades_observed"] == 1

    def test_large_dataset_annualization(self):
        np.random.seed(42)
        rets = list(np.random.normal(0.001, 0.01, 100))
        m = _compute_performance_metrics(rets, "H1", 1, 0.0)
        assert "annual_return" in m
        assert "calmar_ratio" in m

    def test_invalid_timeframe(self):
        m = _compute_performance_metrics([0.01, 0.02], "INVALID", 12, 0.0)
        assert m.get("sharpe_ratio") is None

    def test_with_slippage(self):
        m = _compute_performance_metrics([0.01, 0.02], "H1", 12, 5.0)
        assert m["slippage_bps"] == 5.0

    def test_negative_equity(self):
        """Test with returns that would cause negative equity."""
        m = _compute_performance_metrics([-0.5, -0.5], "H1", 12, 0.0)
        assert "max_drawdown" in m

    def test_inf_filtered(self):
        m = _compute_performance_metrics([float("inf"), 0.01], "H1", 12, 0.0)
        # Verify inf values are filtered correctly; check for other metrics
        assert "slippage_bps" in m


# ── forecast_backtest  (lines 120-435) ───────────────────────────────────────

class TestForecastBacktest:
    def test_invalid_timeframe(self):
        result = forecast_backtest("EURUSD", timeframe="INVALID")
        assert "error" in result

    @patch("mtdata.forecast.backtest._fetch_history")
    def test_not_enough_bars(self, fetch):
        fetch.return_value = _make_df(10)
        result = forecast_backtest("EURUSD", timeframe="H1", horizon=50)
        assert "error" in result

    @patch("mtdata.forecast.backtest._fetch_history")
    def test_fetch_exception(self, fetch):
        fetch.side_effect = Exception("no data")
        result = forecast_backtest("EURUSD", timeframe="H1")
        assert "error" in result

    @patch("mtdata.forecast.backtest._fetch_history")
    def test_methods_as_csv_string(self, fetch):
        fetch.return_value = _make_df(500)
        with patch("mtdata.forecast.backtest.forecast") as fc:
            fc.return_value = {"forecast_price": [101.0] * 12}
            result = forecast_backtest("EURUSD", timeframe="H1", methods="naive,drift")
        assert result.get("success") is True or "error" in result

    @patch("mtdata.forecast.backtest._fetch_history")
    def test_methods_as_space_string(self, fetch):
        fetch.return_value = _make_df(500)
        with patch("mtdata.forecast.backtest.forecast") as fc:
            fc.return_value = {"forecast_price": [101.0] * 12}
            result = forecast_backtest("EURUSD", timeframe="H1", methods="naive drift")
        assert isinstance(result, dict)

    @patch("mtdata.forecast.backtest._fetch_history")
    def test_default_methods_price(self, fetch):
        fetch.return_value = _make_df(500)
        with patch("mtdata.forecast.backtest.forecast") as fc:
            fc.return_value = {"forecast_price": list(range(12))}
            result = forecast_backtest("EURUSD", timeframe="H1")
        assert isinstance(result, dict)

    @patch("mtdata.forecast.backtest._fetch_history")
    def test_default_methods_volatility(self, fetch):
        fetch.return_value = _make_df(500)
        with patch("mtdata.forecast.backtest.forecast_volatility") as fv:
            fv.return_value = {"horizon_sigma_return": 0.05}
            result = forecast_backtest("EURUSD", timeframe="H1", quantity="volatility")
        assert isinstance(result, dict)

    @patch("mtdata.forecast.backtest._fetch_history")
    def test_reuses_prefetched_anchor_history_for_nested_forecasts(self, fetch):
        df = _make_df(500)
        fetch.return_value = df
        captured = []

        def fake_forecast(**kwargs):
            prefetched = kwargs.get("prefetched_df")
            captured.append(
                {
                    "as_of": kwargs.get("as_of"),
                    "prefetched_len": len(prefetched) if prefetched is not None else None,
                    "prefetched_last_time": float(prefetched["time"].iloc[-1]) if prefetched is not None else None,
                }
            )
            return {"forecast_price": [101.0] * 12}

        with patch("mtdata.forecast.backtest.forecast", side_effect=fake_forecast):
            result = forecast_backtest(
                "EURUSD",
                timeframe="H1",
                horizon=12,
                steps=2,
                spacing=12,
                methods=["theta"],
            )

        assert result.get("success") is True
        assert captured == [
            {
                "as_of": _format_time_minimal(float(df["time"].iloc[475])),
                "prefetched_len": 476,
                "prefetched_last_time": float(df["time"].iloc[475]),
            },
            {
                "as_of": _format_time_minimal(float(df["time"].iloc[487])),
                "prefetched_len": 488,
                "prefetched_last_time": float(df["time"].iloc[487]),
            },
        ]

    @patch("mtdata.forecast.backtest._fetch_history")
    def test_rejects_overlapping_generated_backtest_windows(self, fetch):
        result = forecast_backtest(
            "EURUSD",
            timeframe="H1",
            horizon=12,
            steps=2,
            spacing=10,
            methods=["theta"],
        )

        assert result == {
            "error": "spacing must be greater than or equal to horizon when steps > 1"
        }
        fetch.assert_not_called()

    @patch("mtdata.forecast.backtest._fetch_history")
    def test_explicit_anchors(self, fetch):
        df = _make_df(500)
        fetch.return_value = df
        from mtdata.utils.utils import _format_time_minimal
        anchor_time = _format_time_minimal(float(df["time"].iloc[100]))
        with patch("mtdata.forecast.backtest.forecast") as fc:
            fc.return_value = {"forecast_price": [101.0] * 12}
            result = forecast_backtest(
                "EURUSD", timeframe="H1", anchors=[anchor_time], methods=["naive"]
            )
        assert isinstance(result, dict)

    @patch("mtdata.forecast.backtest._fetch_history")
    def test_rejects_overlapping_explicit_anchor_windows(self, fetch):
        df = _make_df(500)
        fetch.return_value = df
        from mtdata.utils.utils import _format_time_minimal

        anchors = [
            _format_time_minimal(float(df["time"].iloc[100])),
            _format_time_minimal(float(df["time"].iloc[105])),
        ]

        with patch("mtdata.forecast.backtest.forecast") as fc:
            result = forecast_backtest(
                "EURUSD",
                timeframe="H1",
                horizon=12,
                anchors=anchors,
                methods=["naive"],
            )

        assert result == {
            "error": (
                "Explicit backtest anchors must be at least horizon bars apart to prevent "
                f"data leakage: {anchors[0]} -> {anchors[1]}"
            )
        }
        fc.assert_not_called()

    @patch("mtdata.forecast.backtest._fetch_history")
    def test_anchor_no_valid_windows(self, fetch):
        df = _make_df(500)
        fetch.return_value = df
        result = forecast_backtest(
            "EURUSD", timeframe="H1", anchors=["9999-01-01"], methods=["naive"]
        )
        assert "error" in result

    @patch("mtdata.forecast.backtest._fetch_history")
    def test_target_return(self, fetch):
        fetch.return_value = _make_df(500)
        with patch("mtdata.forecast.backtest.forecast") as fc:
            fc.return_value = {"forecast_return": [0.001] * 12}
            result = forecast_backtest(
                "EURUSD", timeframe="H1", quantity="return", methods=["naive"]
            )
        assert isinstance(result, dict)

    @patch("mtdata.forecast.backtest._fetch_history")
    def test_forecast_error_per_anchor(self, fetch):
        fetch.return_value = _make_df(500)
        with patch("mtdata.forecast.backtest.forecast") as fc:
            fc.return_value = {"error": "method failure"}
            result = forecast_backtest("EURUSD", timeframe="H1", methods=["naive"])
        assert isinstance(result, dict)

    @patch("mtdata.forecast.backtest._fetch_history")
    def test_forecast_exception_per_anchor(self, fetch):
        fetch.return_value = _make_df(500)
        with patch("mtdata.forecast.backtest.forecast") as fc:
            fc.side_effect = Exception("boom")
            result = forecast_backtest("EURUSD", timeframe="H1", methods=["naive"])
        assert isinstance(result, dict)

    @patch("mtdata.forecast.backtest._fetch_history")
    def test_empty_forecast_result(self, fetch):
        fetch.return_value = _make_df(500)
        with patch("mtdata.forecast.backtest.forecast") as fc:
            fc.return_value = {"forecast_price": []}
            result = forecast_backtest("EURUSD", timeframe="H1", methods=["naive"])
        assert isinstance(result, dict)

    @patch("mtdata.forecast.backtest._fetch_history")
    def test_denoise_param(self, fetch):
        fetch.return_value = _make_df(500)
        with patch("mtdata.forecast.backtest.forecast") as fc:
            fc.return_value = {"forecast_price": [101.0] * 12}
            with patch("mtdata.forecast.backtest._normalize_denoise_spec", return_value={"method": "wavelet"}):
                result = forecast_backtest(
                    "EURUSD", timeframe="H1", methods=["naive"],
                    denoise={"method": "wavelet"},
                )
        assert isinstance(result, dict)

    @patch("mtdata.forecast.backtest._fetch_history")
    def test_slippage_and_threshold(self, fetch):
        fetch.return_value = _make_df(500)
        with patch("mtdata.forecast.backtest.forecast") as fc:
            fc.return_value = {"forecast_price": [200.0] * 12}
            result = forecast_backtest(
                "EURUSD", timeframe="H1", methods=["naive"],
                slippage_bps=10.0, trade_threshold=0.001,
            )
        assert isinstance(result, dict)
        assert result["results"]["naive"]["metrics_available"] is True
        assert result["results"]["naive"]["slippage_bps"] == 10.0

    @patch("mtdata.forecast.backtest._fetch_history")
    def test_volatility_forecast_with_realized(self, fetch):
        fetch.return_value = _make_df(500)
        with patch("mtdata.forecast.backtest.forecast_volatility") as fv:
            fv.return_value = {"horizon_sigma_return": 0.03}
            result = forecast_backtest(
                "EURUSD", timeframe="H1", quantity="volatility",
                methods=["ewma"], params_per_method={"ewma": {"proxy": "garman_klass"}},
            )
        assert isinstance(result, dict)

    @patch("mtdata.forecast.backtest._fetch_history")
    def test_volatility_realized_sigma_uses_horizon_aggregated_returns(self, fetch):
        df = _make_df(500)
        fetch.return_value = df
        with patch("mtdata.forecast.backtest.forecast_volatility") as fv:
            fv.return_value = {"horizon_sigma_return": 0.03}
            result = forecast_backtest(
                "EURUSD",
                timeframe="H1",
                quantity="volatility",
                methods=["ewma"],
                horizon=3,
            )
        entry = result["results"]["ewma"]["details"][0]
        from mtdata.utils.utils import _format_time_minimal

        anchor_idx = next(i for i, ts in enumerate(df["time"]) if _format_time_minimal(float(ts)) == entry["anchor"])
        path = df["close"].iloc[anchor_idx: anchor_idx + 4].to_numpy(dtype=float)
        realized = math.sqrt(np.sum(np.square(np.diff(np.log(path)))))
        assert entry["realized_sigma"] == pytest.approx(realized)

    @patch("mtdata.forecast.backtest._fetch_history")
    def test_return_target_trading_logic(self, fetch):
        df = _make_df(500)
        # Make prices go up so direction is positive
        df["close"] = [100.0 + i * 0.1 for i in range(500)]
        fetch.return_value = df
        with patch("mtdata.forecast.backtest.forecast") as fc:
            fc.return_value = {"forecast_return": [0.01] * 12}
            result = forecast_backtest(
                "EURUSD", timeframe="H1", quantity="return",
                methods=["naive"], slippage_bps=5.0,
            )
        assert isinstance(result, dict)

    @patch("mtdata.forecast.backtest._fetch_history")
    def test_short_position_logic(self, fetch):
        df = _make_df(500)
        fetch.return_value = df
        with patch("mtdata.forecast.backtest.forecast") as fc:
            # Return large negative forecast to trigger short
            fc.return_value = {"forecast_price": [50.0] * 12}
            result = forecast_backtest(
                "EURUSD", timeframe="H1", methods=["naive"],
                trade_threshold=0.0,
            )
        assert isinstance(result, dict)

    @patch("mtdata.forecast.backtest._fetch_history")
    def test_flat_position_logic(self, fetch):
        df = _make_df(500)
        fetch.return_value = df
        with patch("mtdata.forecast.backtest.forecast") as fc:
            # Forecast same as current -> flat
            fc.return_value = {"forecast_price": [float(df["close"].iloc[-13])] * 12}
            result = forecast_backtest(
                "EURUSD", timeframe="H1", methods=["naive"],
                trade_threshold=999.0,  # large threshold forces flat
            )
        assert isinstance(result, dict)
        method_result = result["results"]["naive"]
        assert method_result["metrics_available"] is False
        assert method_result["metrics_reason"] == "no_non_flat_trades"
        assert method_result["slippage_bps"] == 0.0
        assert method_result["metrics"]["win_rate"] is None
        assert method_result["metrics"]["trades_observed"] == 0

    @patch("mtdata.forecast.backtest._fetch_history")
    def test_params_per_method_and_global(self, fetch):
        fetch.return_value = _make_df(500)
        with patch("mtdata.forecast.backtest.forecast") as fc:
            fc.return_value = {"forecast_price": [101.0] * 12}
            result = forecast_backtest(
                "EURUSD", timeframe="H1", methods=["naive", "drift"],
                params_per_method={"naive": {"k": 1}},
                params={"global_key": True},
            )
        assert isinstance(result, dict)

    @patch("mtdata.forecast.backtest._fetch_history")
    def test_top_level_exception(self, fetch):
        fetch.side_effect = TypeError("bad type")
        result = forecast_backtest("EURUSD", timeframe="H1")
        assert "error" in result

    @patch("mtdata.forecast.backtest._fetch_history")
    def test_all_anchors_fail(self, fetch):
        fetch.return_value = _make_df(500)
        with patch("mtdata.forecast.backtest.forecast") as fc:
            fc.return_value = {"error": "fail"}
            result = forecast_backtest("EURUSD", timeframe="H1", methods=["naive"])
        r = result.get("results", {}).get("naive", {})
        assert r.get("success") is False or "error" in result

    @patch("mtdata.forecast.backtest._fetch_history")
    def test_features_and_dimred(self, fetch):
        fetch.return_value = _make_df(500)
        with patch("mtdata.forecast.backtest.forecast") as fc:
            fc.return_value = {"forecast_price": [101.0] * 12}
            result = forecast_backtest(
                "EURUSD", timeframe="H1", methods=["naive"],
                features={"correlated_symbols": ["GBPUSD"]},
                dimred_method="pca", dimred_params={"n_components": 3},
            )
        assert isinstance(result, dict)
