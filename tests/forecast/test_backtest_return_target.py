from __future__ import annotations

import os
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SRC = os.path.join(_ROOT, "src")
for _p in (_SRC, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from mtdata.forecast.backtest import _compute_performance_metrics, forecast_backtest
from mtdata.utils.utils import _format_time_minimal


def test_backtest_return_target_scores_against_returns() -> None:
    times = np.arange(1700000000, 1700000000 + 70 * 3600, 3600, dtype=float)
    close = np.linspace(100.0, 120.0, 70, dtype=float)
    df = pd.DataFrame({"time": times, "close": close})

    idx = 60
    horizon = 2
    anchor = _format_time_minimal(float(times[idx]))
    actual_returns = np.log(close[idx + 1 : idx + 1 + horizon] / close[idx : idx + horizon])

    with patch("mtdata.forecast.backtest._fetch_history", return_value=df), patch(
        "mtdata.forecast.backtest.forecast",
        return_value={"forecast_return": [float(v) for v in actual_returns.tolist()]},
    ):
        res = forecast_backtest(
            symbol="EURUSD",
            timeframe="H1",
            horizon=horizon,
            methods=["naive"],
            anchors=[anchor],
            quantity="return",
        )

    detail = res["results"]["naive"]["details"][0]
    assert detail["success"] is True
    assert abs(float(detail["mae"])) < 1e-12
    assert abs(float(detail["rmse"])) < 1e-12


def test_backtest_volatility_with_return_target_uses_price_truth_windows() -> None:
    times = np.arange(1700000000, 1700000000 + 70 * 3600, 3600, dtype=float)
    close = np.linspace(100.0, 120.0, 70, dtype=float)
    df = pd.DataFrame({"time": times, "close": close})

    idx = 60
    horizon = 3
    anchor = _format_time_minimal(float(times[idx]))
    truth_path = close[idx : idx + 1 + horizon]
    expected_sigma = float(np.sqrt(np.sum(np.diff(np.log(truth_path)) ** 2)))

    with patch("mtdata.forecast.backtest._fetch_history", return_value=df), patch(
        "mtdata.forecast.backtest.forecast_volatility",
        return_value={"horizon_sigma_return": expected_sigma},
    ):
        res = forecast_backtest(
            symbol="EURUSD",
            timeframe="H1",
            horizon=horizon,
            methods=["ewma"],
            anchors=[anchor],
            quantity="volatility",
        )

    detail = res["results"]["ewma"]["details"][0]
    assert bool(detail["success"]) is True
    assert abs(float(detail["realized_sigma"]) - expected_sigma) < 1e-12


def test_backtest_return_target_converts_log_returns_to_simple_trade_returns() -> None:
    times = np.arange(1700000000, 1700000000 + 80 * 3600, 3600, dtype=float)
    close = np.linspace(100.0, 140.0, 80, dtype=float)
    df = pd.DataFrame({"time": times, "close": close})

    idx = 70
    horizon = 2
    anchor = _format_time_minimal(float(times[idx]))
    actual_returns = np.log(close[idx + 1 : idx + 1 + horizon] / close[idx : idx + horizon])
    expected_simple = float(np.exp(np.sum(actual_returns)) - 1.0)

    with patch("mtdata.forecast.backtest._fetch_history", return_value=df), patch(
        "mtdata.forecast.backtest.forecast",
        return_value={"forecast_return": [float(v) for v in actual_returns.tolist()]},
    ):
        res = forecast_backtest(
            symbol="EURUSD",
            timeframe="H1",
            horizon=horizon,
            methods=["naive"],
            anchors=[anchor],
            quantity="return",
        )

    detail = res["results"]["naive"]["details"][0]
    assert detail["success"] is True
    assert abs(float(detail["trade_return"]) - expected_simple) < 1e-12


def test_performance_metrics_skip_annualization_for_short_samples() -> None:
    metrics = _compute_performance_metrics(
        returns=[0.01, -0.02, 0.015, -0.005, 0.01, 0.0],
        timeframe="M15",
        horizon=12,
        slippage_bps=0.0,
    )
    assert metrics["sharpe_ratio"] is None
    assert metrics["annual_return"] is None
    assert metrics["calmar_ratio"] is None
    assert "sample_warning" in metrics
    assert int(metrics["min_trades_for_annualization"]) == 30


def test_backtest_price_target_trade_returns_vary_by_forecast_implied_exit() -> None:
    times = np.arange(1700000000, 1700000000 + 90 * 3600, 3600, dtype=float)
    close = np.linspace(100.0, 145.0, 90, dtype=float)
    df = pd.DataFrame({"time": times, "close": close})

    idx = 80
    horizon = 3
    anchor = _format_time_minimal(float(times[idx]))

    def _fake_forecast(**kwargs):
        method = str(kwargs.get("method", ""))
        if method == "slow":
            return {"forecast_price": [140.8, 141.0, 141.2]}
        return {"forecast_price": [143.5, 144.0, 144.5]}

    with patch("mtdata.forecast.backtest._fetch_history", return_value=df), patch(
        "mtdata.forecast.backtest.forecast",
        side_effect=_fake_forecast,
    ):
        res = forecast_backtest(
            symbol="EURUSD",
            timeframe="H1",
            horizon=horizon,
            methods=["slow", "aggressive"],
            anchors=[anchor],
        )

    slow_detail = res["results"]["slow"]["details"][0]
    aggressive_detail = res["results"]["aggressive"]["details"][0]

    assert slow_detail["success"] is True
    assert aggressive_detail["success"] is True
    assert slow_detail["position"] == "long"
    assert aggressive_detail["position"] == "long"
    assert float(slow_detail["trade_return"]) != float(aggressive_detail["trade_return"])
    assert int(slow_detail["exit_step"]) < int(aggressive_detail["exit_step"])


def test_backtest_default_detail_is_compact_without_full_series_arrays() -> None:
    times = np.arange(1700000000, 1700000000 + 70 * 3600, 3600, dtype=float)
    close = np.linspace(100.0, 120.0, 70, dtype=float)
    df = pd.DataFrame({"time": times, "close": close})

    idx = 60
    anchor = _format_time_minimal(float(times[idx]))
    with patch("mtdata.forecast.backtest._fetch_history", return_value=df), patch(
        "mtdata.forecast.backtest.forecast",
        return_value={"forecast_price": [110.0, 111.0, 112.0]},
    ):
        res = forecast_backtest(
            symbol="EURUSD",
            timeframe="H1",
            horizon=3,
            methods=["naive"],
            anchors=[anchor],
        )

    detail = res["results"]["naive"]["details"][0]
    assert res["detail"] == "compact"
    assert "forecast" not in detail
    assert "actual" not in detail
    assert "forecast_end" in detail
    assert "actual_end" in detail


def test_backtest_full_detail_includes_series_arrays() -> None:
    times = np.arange(1700000000, 1700000000 + 70 * 3600, 3600, dtype=float)
    close = np.linspace(100.0, 120.0, 70, dtype=float)
    df = pd.DataFrame({"time": times, "close": close})

    idx = 60
    anchor = _format_time_minimal(float(times[idx]))
    with patch("mtdata.forecast.backtest._fetch_history", return_value=df), patch(
        "mtdata.forecast.backtest.forecast",
        return_value={"forecast_price": [110.0, 111.0, 112.0]},
    ):
        res = forecast_backtest(
            symbol="EURUSD",
            timeframe="H1",
            horizon=3,
            methods=["naive"],
            anchors=[anchor],
            detail="full",
        )

    detail = res["results"]["naive"]["details"][0]
    assert res["detail"] == "full"
    assert isinstance(detail["forecast"], list)
    assert isinstance(detail["actual"], list)


def test_backtest_exposes_request_metadata_blocks() -> None:
    times = np.arange(1700000000, 1700000000 + 70 * 3600, 3600, dtype=float)
    close = np.linspace(100.0, 120.0, 70, dtype=float)
    df = pd.DataFrame({"time": times, "close": close})

    idx = 60
    anchor = _format_time_minimal(float(times[idx]))
    with patch("mtdata.forecast.backtest._fetch_history", return_value=df), patch(
        "mtdata.forecast.backtest.forecast",
        return_value={"forecast_price": [110.0, 111.0, 112.0]},
    ):
        res = forecast_backtest(
            symbol="EURUSD",
            timeframe="H1",
            horizon="3",  # type: ignore[arg-type]
            methods="naive drift",  # type: ignore[arg-type]
            anchors=[anchor],
            detail="FULL",  # type: ignore[arg-type]
            slippage_bps=2.5,
            trade_threshold=0.01,
        )

    assert res["request"]["detail"] == "FULL"
    assert res["request"]["methods"] == "naive drift"
    assert res["request"]["slippage_bps"] == 2.5
    assert res["resolved_request"]["detail"] == "full"
    assert res["resolved_request"]["methods"] == ["naive", "drift"]
    assert res["resolved_request"]["horizon"] == 3
    assert res["resolved_request"]["slippage_bps"] == 2.5
    assert res["resolved_request"]["trade_threshold"] == 0.01
