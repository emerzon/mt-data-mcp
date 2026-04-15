from __future__ import annotations

import pandas as pd
import pytest

from mtdata.core import forecast as core_forecast
from mtdata.forecast import backtest as forecast_backtest
from mtdata.forecast.requests import StrategyBacktestRequest


def _unwrap(fn):
    current = fn
    while hasattr(current, "__wrapped__"):
        current = current.__wrapped__
    return current


def _history_from_closes(closes: list[float]) -> pd.DataFrame:
    rows = []
    for index, close in enumerate(closes):
        open_price = closes[index - 1] if index > 0 else close
        rows.append(
            {
                "time": 1700000000.0 + (index * 3600.0),
                "open": float(open_price),
                "high": float(max(open_price, close)),
                "low": float(min(open_price, close)),
                "close": float(close),
            }
        )
    return pd.DataFrame(rows)


def test_strategy_backtest_sma_cross_generates_long_trade(monkeypatch):
    monkeypatch.setattr(
        forecast_backtest,
        "_fetch_history",
        lambda symbol, timeframe, need, as_of=None: _history_from_closes(
            [1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        ),
    )

    out = forecast_backtest.strategy_backtest(
        symbol="EURUSD",
        timeframe="H1",
        strategy="sma_cross",
        lookback=8,
        fast_period=2,
        slow_period=3,
        detail="full",
    )

    assert out["success"] is True
    assert out["summary"]["trade_count"] == 1
    assert out["summary"]["num_trades"] == 1
    assert out["summary"]["long_trades"] == 1
    assert out["trades"][0]["direction"] == "long"
    assert out["summary"]["net_return"] > 0.0


def test_strategy_backtest_compact_mode_keeps_stable_trades_key(monkeypatch):
    monkeypatch.setattr(
        forecast_backtest,
        "_fetch_history",
        lambda symbol, timeframe, need, as_of=None: _history_from_closes(
            [1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        ),
    )

    out = forecast_backtest.strategy_backtest(
        symbol="EURUSD",
        timeframe="H1",
        strategy="sma_cross",
        lookback=8,
        fast_period=2,
        slow_period=3,
        detail="compact",
    )

    assert out["success"] is True
    assert out["summary"]["trade_count"] == 1
    assert out["summary"]["num_trades"] == 1
    assert out["trades"][0]["direction"] == "long"
    assert out["trade_sample"] == out["trades"]


def test_strategy_backtest_returns_no_action_on_flat_history(monkeypatch):
    monkeypatch.setattr(
        forecast_backtest,
        "_fetch_history",
        lambda symbol, timeframe, need, as_of=None: _history_from_closes([1.0] * 40),
    )

    out = forecast_backtest.strategy_backtest(
        symbol="EURUSD",
        timeframe="H1",
        strategy="sma_cross",
        lookback=30,
        fast_period=2,
        slow_period=5,
    )

    assert out["success"] is True
    assert out["no_action"] is True
    assert out["summary"]["trade_count"] == 0
    assert out["summary"]["num_trades"] == 0
    assert out["message"] == "The strategy generated no trades on the requested history."


def test_strategy_backtest_request_allows_rsi_reversion_without_ma_constraint():
    request = StrategyBacktestRequest(
        symbol="EURUSD",
        strategy="rsi_reversion",
        fast_period=30,
        slow_period=10,
    )

    assert request.strategy == "rsi_reversion"


def test_core_strategy_backtest_wrapper_routes_request(monkeypatch):
    raw = _unwrap(core_forecast.strategy_backtest)
    monkeypatch.setattr(core_forecast, "ensure_mt5_connection_or_raise", lambda: None)
    monkeypatch.setattr(
        core_forecast,
        "_strategy_backtest_impl",
        lambda **kwargs: {"ok": True, "strategy": kwargs["strategy"], "symbol": kwargs["symbol"]},
    )

    out = raw(
        request=StrategyBacktestRequest(
            symbol="EURUSD",
            strategy="ema_cross",
            lookback=50,
        )
    )

    assert out["ok"] is True
    assert out["strategy"] == "ema_cross"
    assert out["symbol"] == "EURUSD"


def test_strategy_backtest_request_rejects_invalid_ma_periods():
    with pytest.raises(ValueError, match="fast_period must be less than slow_period"):
        StrategyBacktestRequest(
            symbol="EURUSD",
            strategy="sma_cross",
            fast_period=20,
            slow_period=10,
        )
