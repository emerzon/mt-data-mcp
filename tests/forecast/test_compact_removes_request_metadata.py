"""Test that compact detail mode removes request metadata from responses."""
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

from mtdata.forecast.backtest import forecast_backtest, strategy_backtest
from mtdata.utils.utils import _format_time_minimal


def test_forecast_backtest_compact_excludes_request_metadata() -> None:
    """Test that forecast_backtest with detail='compact' doesn't echo request/resolved_request."""
    times = np.arange(1700000000, 1700000000 + 70 * 3600, 3600, dtype=float)
    close = np.linspace(100.0, 120.0, 70, dtype=float)
    df = pd.DataFrame({"time": times, "close": close})

    idx = 60
    anchor = _format_time_minimal(float(times[idx]))
    with patch("mtdata.forecast.backtest._fetch_history", return_value=df), patch(
        "mtdata.forecast.backtest.forecast",
        return_value={"forecast_price": [110.0, 111.0, 112.0]},
    ):
        res_compact = forecast_backtest(
            symbol="EURUSD",
            timeframe="H1",
            horizon=3,
            methods=["naive"],
            anchors=[anchor],
            detail="compact",
            slippage_bps=2.5,
        )

    # In compact mode, request and resolved_request should NOT be present
    assert "request" not in res_compact, "compact mode should not include 'request'"
    assert "resolved_request" not in res_compact, "compact mode should not include 'resolved_request'"
    # But the response should still contain results
    assert res_compact["success"] is True
    assert "results" in res_compact


def test_forecast_backtest_full_includes_request_metadata() -> None:
    """Test that forecast_backtest with detail='full' includes request/resolved_request."""
    times = np.arange(1700000000, 1700000000 + 70 * 3600, 3600, dtype=float)
    close = np.linspace(100.0, 120.0, 70, dtype=float)
    df = pd.DataFrame({"time": times, "close": close})

    idx = 60
    anchor = _format_time_minimal(float(times[idx]))
    with patch("mtdata.forecast.backtest._fetch_history", return_value=df), patch(
        "mtdata.forecast.backtest.forecast",
        return_value={"forecast_price": [110.0, 111.0, 112.0]},
    ):
        res_full = forecast_backtest(
            symbol="EURUSD",
            timeframe="H1",
            horizon=3,
            methods=["naive"],
            anchors=[anchor],
            detail="full",
            slippage_bps=2.5,
        )

    # In full mode, request and resolved_request SHOULD be present
    assert "request" in res_full, "full mode should include 'request'"
    assert res_full["request"]["symbol"] == "EURUSD"
    assert res_full["request"]["slippage_bps"] == 2.5


def test_strategy_backtest_compact_excludes_request_metadata() -> None:
    """Test that strategy_backtest with detail='compact' doesn't echo request/resolved_request."""
    times = np.arange(1700000000, 1700000000 + 100 * 3600, 3600, dtype=float)
    close = np.linspace(100.0, 120.0, 100, dtype=float)
    high = close + 0.5
    low = close - 0.5
    open_ = np.roll(close, 1)
    df = pd.DataFrame(
        {"time": times, "open": open_, "high": high, "low": low, "close": close}
    )

    with patch("mtdata.forecast.backtest._fetch_history", return_value=df):
        res_compact = strategy_backtest(
            symbol="EURUSD",
            timeframe="H1",
            strategy="sma_cross",
            lookback=50,
            detail="compact",
        )

    # In compact mode, request should NOT be present
    assert "request" not in res_compact, "compact mode should not include 'request'"
    assert "resolved_request" not in res_compact, "compact mode should not include 'resolved_request'"
    # But the response should still contain results
    assert res_compact["success"] is True
    assert "summary" in res_compact


def test_strategy_backtest_full_includes_request_metadata() -> None:
    """Test that strategy_backtest with detail='full' includes request/resolved_request."""
    times = np.arange(1700000000, 1700000000 + 100 * 3600, 3600, dtype=float)
    close = np.linspace(100.0, 120.0, 100, dtype=float)
    high = close + 0.5
    low = close - 0.5
    open_ = np.roll(close, 1)
    df = pd.DataFrame(
        {"time": times, "open": open_, "high": high, "low": low, "close": close}
    )

    with patch("mtdata.forecast.backtest._fetch_history", return_value=df):
        res_full = strategy_backtest(
            symbol="EURUSD",
            timeframe="H1",
            strategy="sma_cross",
            lookback=50,
            detail="full",
        )

    # In full mode, request SHOULD be present
    assert "request" in res_full, "full mode should include 'request'"
    assert res_full["request"]["symbol"] == "EURUSD"
    assert res_full["request"]["strategy"] == "sma_cross"
    assert res_full["request"]["lookback"] == 50
