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
            target="return",
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
    truth_prices = close[idx + 1 : idx + 1 + horizon]
    expected_sigma = float(np.sqrt(np.sum(np.diff(np.log(truth_prices)) ** 2)))

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
            target="return",
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
            target="return",
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
    assert np.isnan(float(metrics["annual_return"]))
    assert np.isnan(float(metrics["calmar_ratio"]))
