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


def _history_from_closes(
    closes: list[float],
    *,
    spread_points: float | None = None,
) -> pd.DataFrame:
    rows = []
    for index, close in enumerate(closes):
        open_price = closes[index - 1] if index > 0 else close
        row = {
            "time": 1700000000.0 + (index * 3600.0),
            "open": float(open_price),
            "high": float(max(open_price, close)),
            "low": float(min(open_price, close)),
            "close": float(close),
        }
        if spread_points is not None:
            row["spread"] = float(spread_points)
        rows.append(row)
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
    assert "return_after_known_costs" not in out["units"]
    assert "return_after_known_costs_pct" not in out["units"]
    assert "net_return" not in out["units"]
    assert out["units"]["drawdown"] == "return_fraction"
    assert out["units"]["win_rate"] == "fraction"
    assert out["trades"][0]["direction"] == "long"
    assert out["trades"][0]["spread_cost_status"] == "missing"
    assert "return_net" not in out["trades"][0]
    assert "return_after_known_costs" not in out["trades"][0]
    assert "return_after_known_costs" not in out["summary"]
    assert "return_after_known_costs_pct" not in out["summary"]
    assert out["summary"]["return_status"] == "unavailable_transaction_costs"
    assert "net_return" not in out["summary"]
    assert out["metrics"] == {
        "metrics_available": False,
        "metrics_reason": "incomplete_transaction_costs",
        "metrics_reliability": "unavailable",
        "trades_observed": 1,
    }
    assert "equity_curve" not in out
    assert all("_entry_idx" not in trade for trade in out["trades"])
    assert all("_exit_idx" not in trade for trade in out["trades"])
    assert "drawdown_periods" not in out
    assert "monthly_breakdown" not in out
    assert "trade_distribution" not in out


def test_strategy_backtest_uses_historical_bar_spread_by_default(monkeypatch):
    history = _history_from_closes(
        [1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        spread_points=10.0,
    )
    monkeypatch.setattr(forecast_backtest, "_fetch_history", lambda *args, **kwargs: history)
    monkeypatch.setattr(
        forecast_backtest.mt5,
        "symbol_info",
        lambda _symbol: type("Info", (), {"point": 0.0001})(),
    )

    historical = forecast_backtest.strategy_backtest(
        symbol="EURUSD", lookback=8, fast_period=2, slow_period=3, detail="full"
    )
    fixed = forecast_backtest.strategy_backtest(
        symbol="EURUSD", lookback=8, fast_period=2, slow_period=3, detail="full",
        cost_model="fixed", spread_bps=0.0,
    )

    assert historical["cost_model"]["type"] == "historical_bar_spread"
    assert historical["cost_model"]["spread_source"] == "mt5_historical_bar_spread"
    assert historical["cost_model"]["historical_spread_coverage_pct"] == 100.0
    assert historical["cost_model"]["spread_observations"] == 1
    assert historical["cost_model"]["spread_bps_round_trip"] == pytest.approx(
        historical["trades"][0]["spread_cost_bps"]
    )
    assert historical["cost_model"]["complete"] is True
    assert "warnings" not in historical
    assert historical["summary"]["net_return"] < fixed["summary"]["net_return"]
    assert historical["trades"][0]["spread_cost_status"] == "included"
    assert "return_net" in historical["trades"][0]
    assert "return_after_known_costs" not in historical["trades"][0]


def test_strategy_backtest_rejects_zero_historical_spread_samples(monkeypatch):
    history = _history_from_closes(
        [1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        spread_points=0.0,
    )
    monkeypatch.setattr(forecast_backtest, "_fetch_history", lambda *args, **kwargs: history)
    monkeypatch.setattr(
        forecast_backtest.mt5,
        "symbol_info",
        lambda _symbol: type("Info", (), {"point": 0.0001})(),
    )

    out = forecast_backtest.strategy_backtest(
        symbol="EURUSD", lookback=8, fast_period=2, slow_period=3, detail="full"
    )

    assert out["cost_model"]["spread_source"] == "unavailable"
    assert out["cost_model"]["historical_spread_coverage_pct"] == 0.0
    assert out["cost_model"]["historical_spread_status"] == (
        "unavailable_zero_or_missing_samples"
    )
    assert out["cost_model"]["complete"] is False
    assert "zero spread samples are treated as unavailable" in out["warnings"][0]
    assert "net_return" not in out["summary"]
    assert out["summary"]["return_status"] == "unavailable_transaction_costs"
    assert out["summary"]["costs_complete"] is False
    assert out["summary"]["cost_coverage_pct"] == 0.0
    assert out["metrics"]["metrics_available"] is False


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
        "type": "historical_bar_spread",
        "spread_bps_round_trip": None,
        "spread_source": "unavailable",
        "spread_observations": 0,
        "unpriced_trades": 1,
        "priced_trade_coverage_pct": 0.0,
        "slippage_bps_per_side": 1.0,
        "round_trip_cost_bps": None,
        "complete": False,
        "historical_spread_coverage_pct": 0.0,
        "historical_bar_spread_coverage_pct": 0.0,
    }
    assert "costs are unavailable" in out["warnings"][0]
    assert StrategyBacktestRequest(symbol="EURUSD").slippage_bps == 1.0
    assert StrategyBacktestRequest(symbol="EURUSD").cost_model == "historical_bar_spread"
    assert out["signal_status"] == "not_actionable"
    assert "last_signal" not in out
    assert out["last_historical_signal"]["signal_status"] == "historical_observation_only"
    assert out["last_historical_signal"]["direction"] == "long"
    assert "signal" not in out["last_historical_signal"]
    assert out["summary"]["metrics_reliability"] == "unavailable"
    assert out["summary"]["metrics_reliability_reasons"] == [
        "incomplete_transaction_costs"
    ]
    assert "trades_observed" not in out["summary"]
    assert out["summary"]["return_status"] == "unavailable_transaction_costs"
    assert out["summary"]["costs_complete"] is False
    assert out["summary"]["cost_coverage_pct"] == 0.0
    assert "net_return" not in out["summary"]
    assert out["metrics"]["metrics_available"] is False
    assert out["metrics"]["metrics_reason"] == "incomplete_transaction_costs"
    assert out["metrics"]["metrics_reliability"] == "unavailable"
    assert out["metrics"]["trades_observed"] == 1
    assert "sample_notice" not in out["metrics"]
    assert "warning" not in out
    assert out["units"]["returns"] == "return_fraction"
    assert "avg_directional_accuracy" not in out["units"]
    assert len(out["units"]) < len(forecast_backtest._backtest_units())
    assert "trades" not in out, "compact mode should not include trades array"
    assert "trade_sample" not in out


def test_strategy_backtest_uses_date_range_when_provided(monkeypatch):
    calls = []
    history = _history_from_closes(
        [1.0 + value / 10.0 for value in range(15)],
        spread_points=10.0,
    )

    def fake_fetch_history(symbol, timeframe, need, **kwargs):
        calls.append((need, kwargs))
        if kwargs.get("as_of"):
            return history.iloc[:5].reset_index(drop=True)
        return history.iloc[5:].reset_index(drop=True)

    monkeypatch.setattr(forecast_backtest, "_fetch_history", fake_fetch_history)
    monkeypatch.setattr(
        forecast_backtest.mt5,
        "symbol_info",
        lambda _symbol: type("Info", (), {"point": 0.0001})(),
    )

    out = forecast_backtest.strategy_backtest(
        symbol="EURUSD",
        timeframe="H1",
        strategy="sma_cross",
        lookback=5,
        start="2023-01-01",
        end="2023-12-31",
        fast_period=2,
        slow_period=5,
        detail="full",
    )

    assert out["success"] is True
    assert out["summary"]["bars_used"] == 10
    assert out["summary"]["warmup_history_bars"] == 5
    assert out["summary"]["signal_bars"] == 10
    assert out["summary"]["evaluation_start"] == (
        forecast_backtest._format_time_minimal(float(history["time"].iloc[5]))
    )
    assert calls[0][1]["start"] == "2023-01-01"
    assert calls[0][1]["end"] == "2023-12-31"
    assert calls[1] == (5, {"as_of": "2023-01-01"})
    assert all(
        trade["entry_time"] >= out["summary"]["evaluation_start"]
        for trade in out.get("trades", [])
    )
    assert out["parameters"]["start"] == "2023-01-01"
    assert out["parameters"]["end"] == "2023-12-31"


def test_strategy_backtest_rejects_range_without_prestart_warmup(monkeypatch):
    history = _history_from_closes([float(value) for value in range(1, 16)])

    def fake_fetch_history(symbol, timeframe, need, **kwargs):
        if kwargs.get("as_of"):
            return history.iloc[:2].reset_index(drop=True)
        return history.iloc[5:].reset_index(drop=True)

    monkeypatch.setattr(forecast_backtest, "_fetch_history", fake_fetch_history)

    out = forecast_backtest.strategy_backtest(
        symbol="EURUSD",
        start="2023-01-01",
        end="2023-12-31",
        lookback=5,
        fast_period=2,
        slow_period=5,
        cost_model="fixed",
        spread_bps=1.0,
    )

    assert out["success"] is False
    assert out["error_code"] == "insufficient_warmup_history"
    assert out["warmup_bars_required"] == 5
    assert out["warmup_bars_available"] == 2


@pytest.mark.parametrize(
    ("strategy", "strategy_kwargs"),
    [
        ("sma_cross", {"fast_period": 2, "slow_period": 5}),
        ("ema_cross", {"fast_period": 2, "slow_period": 5}),
        ("rsi_reversion", {"rsi_length": 4, "oversold": 40.0, "overbought": 60.0}),
    ],
)
def test_range_results_match_same_prefetched_history(
    monkeypatch,
    strategy,
    strategy_kwargs,
):
    closes = [100.0, 90.0, 80.0, 90.0, 100.0] * 5
    history = _history_from_closes(closes)
    warmup_bars = 5
    evaluation = history.iloc[warmup_bars:].reset_index(drop=True)
    prehistory = history.iloc[:warmup_bars].reset_index(drop=True)

    def ranged_fetch(symbol, timeframe, need, **kwargs):
        return prehistory if kwargs.get("as_of") else evaluation

    monkeypatch.setattr(forecast_backtest, "_fetch_history", ranged_fetch)
    ranged = forecast_backtest.strategy_backtest(
        symbol="EURUSD",
        strategy=strategy,
        lookback=20,
        start="2023-01-01",
        end="2023-12-31",
        cost_model="fixed",
        spread_bps=1.0,
        slippage_bps=0.0,
        detail="full",
        **strategy_kwargs,
    )

    monkeypatch.setattr(
        forecast_backtest,
        "_fetch_history",
        lambda *args, **kwargs: history,
    )
    prefetched = forecast_backtest.strategy_backtest(
        symbol="EURUSD",
        strategy=strategy,
        lookback=20,
        cost_model="fixed",
        spread_bps=1.0,
        slippage_bps=0.0,
        detail="full",
        **strategy_kwargs,
    )

    trade_fields = ("direction", "entry_time", "exit_time", "exit_reason")
    assert ranged["summary"]["num_trades"] > 0
    assert [
        tuple(trade[field] for field in trade_fields) for trade in ranged["trades"]
    ] == [
        tuple(trade[field] for field in trade_fields)
        for trade in prefetched["trades"]
    ]
    assert ranged["summary"]["gross_return"] == pytest.approx(
        prefetched["summary"]["gross_return"]
    )
    assert ranged["summary"]["net_return"] == pytest.approx(
        prefetched["summary"]["net_return"]
    )


def test_max_hold_waits_for_fresh_signal_before_same_direction_reentry(monkeypatch):
    history = _history_from_closes([1.0 + value / 100.0 for value in range(15)])
    persistent = pd.Series([1.0] * len(history))
    monkeypatch.setattr(
        forecast_backtest,
        "_fetch_history",
        lambda *args, **kwargs: history,
    )
    monkeypatch.setattr(
        forecast_backtest,
        "_build_strategy_signal_series",
        lambda *args, **kwargs: (persistent, {}, 1),
    )

    out = forecast_backtest.strategy_backtest(
        symbol="EURUSD",
        lookback=12,
        fast_period=2,
        slow_period=3,
        max_hold_bars=3,
        cost_model="fixed",
        spread_bps=1.0,
        slippage_bps=1.0,
        detail="full",
    )

    assert out["summary"]["num_trades"] == 1
    assert out["trades"][0]["exit_reason"] == "max_hold"
    assert out["trades"][0]["bars_held"] == 3
    assert out["summary"]["max_hold_reentry_policy"] == "fresh_signal_required"
    assert out["summary"]["longest_continuous_exposure_bars"] == 3
    assert out["cost_model"]["spread_observations"] == 1


def test_max_hold_allows_opposite_signal_at_boundary(monkeypatch):
    history = _history_from_closes([1.0 + value / 100.0 for value in range(12)])
    signals = pd.Series([1.0, 1.0, 1.0, -1.0] + [-1.0] * 8)
    monkeypatch.setattr(
        forecast_backtest,
        "_fetch_history",
        lambda *args, **kwargs: history,
    )
    monkeypatch.setattr(
        forecast_backtest,
        "_build_strategy_signal_series",
        lambda *args, **kwargs: (signals, {}, 1),
    )

    out = forecast_backtest.strategy_backtest(
        symbol="EURUSD",
        lookback=12,
        fast_period=2,
        slow_period=3,
        max_hold_bars=3,
        cost_model="fixed",
        spread_bps=0.0,
        slippage_bps=0.0,
        detail="full",
    )

    assert [trade["direction"] for trade in out["trades"]] == ["long", "short"]
    assert out["trades"][0]["exit_reason"] == "signal_reversal"
    assert out["trades"][0]["exit_time"] == out["trades"][1]["entry_time"]


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


def test_strategy_backtest_request_rejects_spread_with_historical_model():
    with pytest.raises(ValueError, match="spread_bps is only valid"):
        StrategyBacktestRequest(symbol="EURUSD", spread_bps=1.0)


def test_strategy_backtest_request_requires_spread_with_fixed_model():
    with pytest.raises(ValueError, match="spread_bps is required"):
        StrategyBacktestRequest(symbol="EURUSD", cost_model="fixed")
