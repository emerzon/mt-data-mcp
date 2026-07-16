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
    assert out["summary"]["num_trades"] == 1
    assert out["summary"]["long_trades"] == 1
    assert out["units"]["returns"] == "return_fraction"
    assert out["units"]["net_return"] == "return_fraction"
    assert out["units"]["net_return_pct"] == "percentage_points"
    assert out["units"]["drawdown"] == "return_fraction"
    assert out["units"]["win_rate"] == "fraction"
    assert out["trades"][0]["direction"] == "long"
    assert out["summary"]["net_return"] > 0.0
    assert out["summary"]["net_return_pct"] == pytest.approx(
        out["summary"]["net_return"] * 100.0
    )


def test_strategy_backtest_discloses_current_spread_proxy(monkeypatch):
    history = _history_from_closes([1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    monkeypatch.setattr(forecast_backtest, "_fetch_history", lambda *args, **kwargs: history)
    monkeypatch.setattr(
        forecast_backtest.mt5,
        "symbol_info_tick",
        lambda _symbol: type("Tick", (), {"bid": 0.9995, "ask": 1.0005})(),
    )

    proxied = forecast_backtest.strategy_backtest(
        symbol="EURUSD", lookback=8, fast_period=2, slow_period=3, detail="full"
    )
    fixed = forecast_backtest.strategy_backtest(
        symbol="EURUSD", lookback=8, fast_period=2, slow_period=3, detail="full",
        cost_model="fixed", spread_bps=0.0,
    )

    assert proxied["cost_model"]["type"] == "current_spread_proxy"
    assert proxied["cost_model"]["spread_bps_round_trip"] == pytest.approx(10.0)
    assert proxied["cost_model"]["complete"] is False
    assert "not historical observed spreads" in proxied["warnings"][0]
    assert proxied["summary"]["net_return"] < fixed["summary"]["net_return"]


def test_strategy_backtest_includes_first_valid_warmup_signal(monkeypatch):
    monkeypatch.setattr(
        forecast_backtest,
        "_fetch_history",
        lambda *args, **kwargs: _history_from_closes(
            [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        ),
    )

    out = forecast_backtest.strategy_backtest(
        symbol="EURUSD",
        timeframe="H1",
        strategy="sma_cross",
        lookback=10,
        fast_period=2,
        slow_period=3,
        detail="full",
        position_mode="long_short",
    )

    assert out["success"] is True
    assert out["trades"][0]["direction"] == "long"


def test_strategy_backtest_compact_mode_excludes_trades(monkeypatch):
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
    assert out["summary"]["num_trades"] == 1
    assert out["summary"]["sample_status"] == "insufficient_trades"
    assert out["summary"]["minimum_trades"] == 30
    assert out["is_signal"] is False
    assert out["usage"] == "research_only"
    assert "usable_for_live_trading" not in out
    assert out["price_basis"] == "mt5_bid_ohlc"
    assert out["cost_model"] == {
        "type": "current_spread_proxy",
        "spread_bps_round_trip": 0.0,
        "spread_source": "unavailable",
        "slippage_bps_per_side": 1.0,
        "round_trip_cost_bps": 2.0,
        "complete": False,
    }
    assert "not historical observed spreads" in out["warnings"][0]
    assert StrategyBacktestRequest(symbol="EURUSD").slippage_bps == 1.0
    assert out["signal_status"] == "not_actionable"
    assert "last_signal" not in out
    assert out["last_historical_signal"]["signal_status"] == "historical_observation_only"
    assert out["last_historical_signal"]["direction"] == "long"
    assert "signal" not in out["last_historical_signal"]
    assert "metrics_reliability" not in out["summary"]
    assert "trades_observed" not in out["summary"]
    assert out["metrics"]["metrics_reliability"] == "low"
    assert out["metrics"]["trades_observed"] == 1
    assert "sample_notice" not in out["metrics"]
    assert "warning" not in out
    assert out["units"]["returns"] == "return_fraction"
    assert "avg_directional_accuracy" not in out["units"]
    assert len(out["units"]) < len(forecast_backtest._backtest_units())
    assert "trades" not in out, "compact mode should not include trades array"
    assert "trade_sample" not in out


def test_strategy_backtest_uses_date_range_when_provided(monkeypatch):
    captured = {}

    def fake_fetch_history(symbol, timeframe, need, **kwargs):
        captured.update(kwargs)
        return _history_from_closes(
            [1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        )

    monkeypatch.setattr(forecast_backtest, "_fetch_history", fake_fetch_history)

    out = forecast_backtest.strategy_backtest(
        symbol="EURUSD",
        timeframe="H1",
        strategy="sma_cross",
        lookback=5,
        start="2023-01-01",
        end="2023-12-31",
        fast_period=2,
        slow_period=3,
        detail="full",
    )

    assert captured["start"] == "2023-01-01"
    assert captured["end"] == "2023-12-31"
    assert out["success"] is True
    assert out["summary"]["bars_used"] == 10
    assert out["parameters"]["start"] == "2023-01-01"
    assert out["parameters"]["end"] == "2023-12-31"


def test_strategy_backtest_exposes_request_metadata_blocks(monkeypatch):
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
        strategy="SMA_CROSS",  # type: ignore[arg-type]
        lookback="8",  # type: ignore[arg-type]
        fast_period="2",  # type: ignore[arg-type]
        slow_period="3",  # type: ignore[arg-type]
        detail="FULL",  # type: ignore[arg-type]
        position_mode="LONG_SHORT",  # type: ignore[arg-type]
        slippage_bps=1.5,
    )

    assert out["request"]["detail"] == "FULL"
    assert out["request"]["strategy"] == "SMA_CROSS"
    assert out["request"]["slippage_bps"] == 1.5
    assert out["resolved_request"]["detail"] == "full"
    assert out["resolved_request"]["strategy"] == "sma_cross"
    assert out["resolved_request"]["position_mode"] == "long_short"
    assert out["resolved_request"]["lookback"] == 8
    assert out["resolved_request"]["slippage_bps"] == 1.5
    assert out["parameters"]["slippage_bps"] == 1.5
    strategy_params = out["contracts"]["strategy"]["parameters"]
    assert strategy_params["fast_period"] == 2
    assert strategy_params["slow_period"] == 3
    assert "rsi_length" not in strategy_params
    assert "oversold" not in strategy_params
    assert "overbought" not in strategy_params
    assert out["contracts"]["data_preparation"]["symbol"] == "EURUSD"
    assert out["contracts"]["evaluation"]["detail"] == "full"
    assert out["contracts"]["strategy"]["kind"] == "indicator_strategy"
    assert out["contracts"]["strategy"]["position_mode"] == "long_short"


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
    assert out["summary"]["num_trades"] == 0
    assert out["message"] == "The strategy generated no trades on the requested history."


def test_strategy_backtest_long_only_signal_suppresses_shorts_and_warmup_nan():
    df = _history_from_closes([5.0, 4.0, 3.0, 2.0, 1.0, 2.0, 3.0, 4.0])

    long_short_signal, _diagnostics, _warmup = forecast_backtest._build_strategy_signal_series(
        df,
        strategy="sma_cross",
        position_mode="long_short",
        fast_period=2,
        slow_period=3,
        rsi_length=14,
        oversold=30.0,
        overbought=70.0,
    )
    long_only_signal, _diagnostics, warmup = forecast_backtest._build_strategy_signal_series(
        df,
        strategy="sma_cross",
        position_mode="long_only",
        fast_period=2,
        slow_period=3,
        rsi_length=14,
        oversold=30.0,
        overbought=70.0,
    )

    assert long_short_signal.isna().any()
    assert (long_short_signal == -1.0).any()
    assert not long_only_signal.isna().any()
    assert (long_only_signal >= 0.0).all()
    assert long_only_signal.iloc[:warmup].eq(0.0).all()


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
        lambda **kwargs: {
            "ok": True,
            "strategy": kwargs["strategy"],
            "symbol": kwargs["symbol"],
            "start": kwargs["start"],
            "end": kwargs["end"],
        },
    )

    out = raw(
        request=StrategyBacktestRequest(
            symbol="EURUSD",
            strategy="ema_cross",
            lookback=50,
            start="2023-01-01",
            end="2023-12-31",
        )
    )

    assert out["ok"] is True
    assert out["strategy"] == "ema_cross"
    assert out["symbol"] == "EURUSD"
    assert out["start"] == "2023-01-01"
    assert out["end"] == "2023-12-31"


def test_strategy_backtest_request_rejects_invalid_ma_periods():
    with pytest.raises(ValueError, match="fast_period must be less than slow_period"):
        StrategyBacktestRequest(
            symbol="EURUSD",
            strategy="sma_cross",
            fast_period=20,
            slow_period=10,
        )
