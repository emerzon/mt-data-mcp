from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from mtdata.analytics.engines import (
    _barrier_returns,
    _filtered_historical_returns,
    _tick_frame,
    analyze_execution_quality,
    analyze_microstructure,
    decompose_portfolio_risk,
    rank_relative_strength,
    validate_strategies,
)
from mtdata.core.analytics_requests import (
    MarketMicrostructureRequest,
    MarketRelativeStrengthRequest,
    PortfolioRiskDecomposeRequest,
    StrategyValidateRequest,
    TradeExecutionQualityRequest,
)


def _now() -> int:
    import time

    return int(time.time())


def _ticks(count: int = 200, *, start: int | None = None, real_volume: bool = False):
    start = start or (_now() - count)
    rows = []
    for idx in range(count):
        mid = 1.1 + idx * 0.000001
        rows.append(
            {
                "time": start + idx,
                "time_msc": (start + idx) * 1000,
                "bid": mid - 0.00005,
                "ask": mid + 0.00005,
                "last": mid if real_volume else 0.0,
                "volume": 1,
                "volume_real": 2.0 if real_volume else 0.0,
                "flags": 1054 if real_volume else 6,
            }
        )
    return rows


def _bars(count: int = 500, *, drift: float = 0.0002):
    end = _now() - 7200
    start = end - count * 3600
    rows = []
    price = 1.0
    for idx in range(count):
        change = drift + np.sin(idx / 8.0) * 0.0004
        opened = price
        price = max(0.1, price * (1.0 + change))
        rows.append(
            {
                "time": start + idx * 3600,
                "open": opened,
                "high": max(opened, price) * 1.002,
                "low": min(opened, price) * 0.998,
                "close": price,
                "tick_volume": 1000 + idx,
                "real_volume": 0,
                "spread": 10,
            }
        )
    return rows


class FakeGateway:
    COPY_TICKS_ALL = 0
    DEAL_TYPE_BUY = 0
    DEAL_TYPE_SELL = 1
    POSITION_TYPE_BUY = 0
    POSITION_TYPE_SELL = 1
    ORDER_TYPE_BUY = 0
    ORDER_TYPE_SELL = 1

    def __init__(self):
        self.tick_rows = _ticks()
        self.bar_rows = {"EURUSD": _bars(), "GBPUSD": _bars(drift=0.0001), "USDJPY": _bars(drift=-0.00005)}
        self.deals = []
        self.orders = []
        self.positions = []

    def copy_ticks_range(self, symbol, start, end, flags):
        lo = start.timestamp()
        hi = end.timestamp()
        return [row for row in self.tick_rows if lo <= row["time"] <= hi]

    def copy_rates_from_pos(self, symbol, timeframe, start, count):
        return self.bar_rows[symbol][-count:]

    def copy_rates_range(self, symbol, timeframe, start, end):
        return [row for row in self.bar_rows[symbol] if start.timestamp() <= row["time"] <= end.timestamp()]

    def history_deals_get(self, start, end, **kwargs):
        return self.deals

    def history_orders_get(self, start, end, **kwargs):
        return self.orders

    def positions_get(self):
        return self.positions

    def symbols_get(self):
        return [SimpleNamespace(name=name, path="Forex\\Majors", visible=True) for name in self.bar_rows]

    def symbol_info_tick(self, symbol):
        return SimpleNamespace(bid=1.0999, ask=1.1001)

    def symbol_info(self, symbol):
        return SimpleNamespace(point=0.00001, digits=5)

    def order_calc_profit(self, action, symbol, volume, opened, closed):
        sign = 1.0 if action == self.ORDER_TYPE_BUY else -1.0
        return sign * (closed - opened) * 100_000 * volume

    def order_calc_margin(self, action, symbol, volume, price):
        return volume * 1000.0


def test_microstructure_distinguishes_trade_volume_from_quote_proxy() -> None:
    gateway = FakeGateway()
    gateway.tick_rows = _ticks(real_volume=True)
    result = analyze_microstructure(
        MarketMicrostructureRequest(symbol="EURUSD", minutes_back=60), gateway
    )
    assert result["success"] is True
    assert result["summary"]["feed_tier"] == "trade_volume"
    assert result["method_applicability"]["volume_impact_metrics"] is True
    assert "signed_volume_imbalance" in result["summary"]
    assert "kyle_lambda" not in result["summary"]
    assert "amihud_impact" not in result["summary"]
    assert result["estimator_scope"]["market_scope"] == "connected_broker_tick_feed"
    assert result["timezone"] == "UTC"
    assert result["summary"]["spread_points"]["median"] == pytest.approx(10.0)
    assert result["summary"]["spread_pips"]["median"] == pytest.approx(1.0)
    assert result["units"]["spread_points"] == "broker_points"
    assert all("start" in item and "end" in item for item in result["liquidity_events"])
    assert all("start_epoch" not in item for item in result["liquidity_events"])
    assert any("broker's tick feed" in warning for warning in result["warnings"])


def test_microstructure_does_not_recount_last_trade_snapshots() -> None:
    gateway = FakeGateway()
    gateway.tick_rows = _ticks(real_volume=True)
    for row in gateway.tick_rows:
        row["last"] = 1.101
        row["volume_real"] = 5.0
        row["flags"] = 6
    gateway.tick_rows[0]["flags"] = 1032

    result = analyze_microstructure(
        MarketMicrostructureRequest(symbol="EURUSD", minutes_back=60), gateway
    )

    assert result["success"] is True
    assert result["summary"]["trade_count"] == 1
    assert result["data_quality"]["trade_tick_coverage"] == pytest.approx(1 / 200)


def test_tick_frame_keeps_distinct_same_timestamp_events() -> None:
    gateway = FakeGateway()
    epoch = _now() - 10
    gateway.tick_rows = [
        {
            "time": epoch,
            "time_msc": epoch * 1000,
            "bid": 1.1,
            "ask": 1.2,
            "last": 1.15,
            "volume": 5,
            "volume_real": 5.0,
            "flags": 1032,
        },
        {
            "time": epoch,
            "time_msc": epoch * 1000,
            "bid": 1.1,
            "ask": 1.2,
            "last": 1.15,
            "volume": 5,
            "volume_real": 5.0,
            "flags": 6,
        },
    ]

    frame, truncated = _tick_frame(
        gateway,
        "EURUSD",
        datetime.fromtimestamp(epoch - 1, tz=timezone.utc),
        datetime.fromtimestamp(epoch + 1, tz=timezone.utc),
        10,
    )

    assert truncated is False
    assert len(frame) == 2


def test_tick_frame_empty_result_retains_analysis_schema() -> None:
    gateway = FakeGateway()
    gateway.tick_rows = []
    now = datetime.now(timezone.utc)

    frame, truncated = _tick_frame(gateway, "EURUSD", now, now, 10)

    assert truncated is False
    assert frame.empty
    assert {"epoch", "bid", "ask", "mid", "spread"}.issubset(frame.columns)


def test_microstructure_reports_closed_session_for_short_tick_stream(monkeypatch) -> None:
    gateway = FakeGateway()
    gateway.tick_rows = _ticks(3)
    monkeypatch.setattr(
        "mtdata.analytics.engines.closed_session_context",
        lambda *args, **kwargs: {
            "market_status": "closed",
            "market_status_reason": "weekend",
        },
    )

    result = analyze_microstructure(
        MarketMicrostructureRequest(symbol="EURUSD", minutes_back=60), gateway
    )

    assert result["error_code"] == "market_closed"
    assert result["ticks_available"] == 3
    assert result["market_status_reason"] == "weekend"
    assert "reopen" in result["remediation"]


def test_tick_frame_nulls_derived_quotes_for_one_sided_updates() -> None:
    gateway = FakeGateway()
    gateway.tick_rows = [
        {"time": 1, "bid": 1.1, "ask": 1.1, "flags": 2},
        {"time": 2, "bid": 1.1, "ask": 1.1002, "flags": 6},
    ]

    frame, _ = _tick_frame(
        gateway,
        "EURUSD",
        datetime.fromtimestamp(0, tz=timezone.utc),
        datetime.fromtimestamp(3, tz=timezone.utc),
        100,
    )

    assert np.isnan(frame.iloc[0]["mid"])
    assert np.isnan(frame.iloc[0]["spread"])
    assert frame.iloc[1]["spread"] == pytest.approx(0.0002)


def test_execution_quality_matches_order_and_computes_markout() -> None:
    gateway = FakeGateway()
    start = _now() - 100
    gateway.tick_rows = _ticks(100, start=start)
    gateway.orders = [
        {"ticket": 10, "type": 0, "price_open": 1.10005, "volume_initial": 1.0, "time_setup_msc": (start + 9) * 1000}
    ]
    gateway.deals = [
        {"ticket": 20, "order": 10, "position_id": 30, "symbol": "EURUSD", "type": 0, "volume": 1.0, "price": 1.10008, "time": start + 10, "time_msc": (start + 10) * 1000}
    ]
    result = analyze_execution_quality(
        TradeExecutionQualityRequest(minutes_back=60, markout_seconds=[1, 5], detail="full"),
        gateway,
    )
    assert result["summary"]["fills"] == 1
    assert result["items"][0]["benchmark_source"] == "arrival_quote"
    assert result["items"][0]["order_to_fill_ms"] == 1000.0
    assert result["summary"]["market_order_latency_ms"]["mean"] == 1000.0
    assert result["latency_definition"]["order_to_fill_ms"].endswith(
        "including_pending_wait"
    )
    assert result["items"][0]["markout_bps"]["5"] is not None


def test_execution_quality_handles_empty_tick_history() -> None:
    gateway = FakeGateway()
    gateway.tick_rows = []
    fill_epoch = _now() - 10
    gateway.orders = [{"ticket": 10, "price_open": 1.1, "volume_initial": 1.0}]
    gateway.deals = [
        {
            "ticket": 20,
            "order": 10,
            "position_id": 30,
            "symbol": "EURUSD",
            "type": 0,
            "volume": 1.0,
            "price": 1.1001,
            "time": fill_epoch,
            "time_msc": fill_epoch * 1000,
        }
    ]

    result = analyze_execution_quality(
        TradeExecutionQualityRequest(minutes_back=60, markout_seconds=[1, 5]),
        gateway,
    )

    assert result["success"] is True
    assert result["summary"]["fills"] == 1
    assert result["data_quality"]["skipped"]["missing_markout"] == 2


def test_strategy_validation_returns_walk_forward_oos_metrics() -> None:
    gateway = FakeGateway()
    request = StrategyValidateRequest(
        symbol="EURUSD",
        lookback=400,
        candidates=[
            {"id": "cross", "type": "builtin_strategy", "strategy": "sma_cross", "params": {"fast_period": 5, "slow_period": 20}}
        ],
        barrier={"horizon": 5, "tp_pct": 0.15, "sl_pct": 0.15},
        n_splits=3,
        cost_model="fixed",
        spread_bps=1.0,
        bootstrap_samples=100,
        detail="full",
    )
    result = validate_strategies(request, gateway)
    assert result["success"] is True
    assert result["validation"]["outcome_horizon_bars"] == 5
    assert result["validation"]["extra_purge_bars"] == 0
    assert result["validation"]["protocol"] == "anchored_expanding_fixed_candidate_oos"
    assert result["validation"]["execution_timing"] == "next_bar_open"
    assert result["rankings"][0]["id"] == "cross"
    assert result["rankings"][0]["trades"] > 0
    assert result["rankings"][0]["evaluation_status"] == "complete"
    assert result["rankings"][0]["evidence"]["classification"] in {
        "positive", "negative", "inconclusive"
    }
    for fold in result["rankings"][0]["folds"]:
        assert fold["test_end_bar"] + request.barrier.horizon <= fold["test_window_end_bar"]


def test_strategy_barrier_entry_uses_next_bar_open_after_gap() -> None:
    frame = pd.DataFrame(
        {
            "open": [100.0, 110.0, 110.0],
            "high": [101.0, 111.0, 111.0],
            "low": [99.0, 109.0, 109.0],
            "close": [100.0, 110.0, 110.0],
        }
    )
    signal = pd.Series([1.0, np.nan, np.nan])

    indices, outcomes = _barrier_returns(
        frame,
        signal,
        horizon=1,
        tp_pct=5.0,
        sl_pct=5.0,
    )

    assert indices.tolist() == [0]
    assert outcomes.tolist() == pytest.approx([0.0])


def test_forecast_strategy_folds_cover_computed_signal_window(monkeypatch) -> None:
    gateway = FakeGateway()

    def fake_execute_forecast(**kwargs):
        history_length = len(kwargs["prefetched_df"])
        return {"expected_return": 0.01 if history_length % 2 else -0.01}

    monkeypatch.setattr(
        "mtdata.forecast.forecast.execute_forecast",
        fake_execute_forecast,
    )
    request = StrategyValidateRequest(
        symbol="EURUSD",
        lookback=400,
        candidates=[
            {
                "id": "forecast",
                "type": "forecast_threshold",
                "method": "naive",
                "params": {"lookback": 20},
                "horizon": 1,
                "long_above": 0.0,
                "short_below": 0.0,
            }
        ],
        barrier={"horizon": 1, "tp_pct": 0.15, "sl_pct": 0.15},
        n_splits=3,
        cost_model="fixed",
        spread_bps=1.0,
        bootstrap_samples=100,
        detail="full",
    )

    result = validate_strategies(request, gateway)
    candidate = result["rankings"][0]

    assert candidate["evaluation_status"] == "complete"
    assert candidate["signal_coverage"]["anchors_computed"] == 200
    assert candidate["signal_coverage"]["anchor_limit"] == 200
    assert candidate["folds_requested"] == 3
    assert candidate["folds_evaluated"] == 3
    assert candidate["fold_coverage"] == 1.0
    assert candidate["evidence"]["criteria"]["all_requested_folds_evaluated"] is True
    assert result["validation"]["forecast_signal_anchor_limit"] == 200


def test_portfolio_risk_reconciles_component_expected_shortfall() -> None:
    gateway = FakeGateway()
    gateway.account_info = lambda: SimpleNamespace(currency="USD", equity=25000.0)
    gateway.positions = [
        {"ticket": 1, "symbol": "EURUSD", "type": 0, "volume": 1.0, "price_current": 1.1},
        {"ticket": 2, "symbol": "GBPUSD", "type": 1, "volume": 0.5, "price_current": 1.1},
    ]
    result = decompose_portfolio_risk(
        PortfolioRiskDecomposeRequest(lookback=300, horizon_bars=[1], confidence=[0.95], simulations=500),
        gateway,
    )
    assert result["success"] is True
    assert result["currency"] == "USD"
    assert result["equity"] == 25000.0
    row = result["risk"][0]
    component_total = sum(item["value"] for item in row["component_expected_shortfall"])
    assert component_total == pytest.approx(row["expected_shortfall"])
    assert "correlation_to_one_loss_proxy" not in result["stresses"]
    assert result["stresses"]["perfect_positive_correlation_1sigma"][0]["horizon_bars"] == 1


def test_filtered_historical_shock_uses_pre_shock_volatility() -> None:
    baseline = np.tile(np.array([-0.01, 0.01]), 60)
    values = np.concatenate([baseline, np.array([0.20])])
    returns = pd.DataFrame({"EURUSD": values})
    alpha = 0.1

    standardized, _ = _filtered_historical_returns(returns, alpha=alpha)
    concurrent_vol = returns.ewm(alpha=alpha, adjust=False).std().iloc[-1, 0]

    assert standardized.iloc[-1, 0] == pytest.approx(
        values[-1] / returns.ewm(alpha=alpha, adjust=False).std().shift(1).iloc[-1, 0]
    )
    assert standardized.iloc[-1, 0] > values[-1] / concurrent_vol * 2.0


def test_portfolio_risk_fails_closed_when_symbol_history_is_missing() -> None:
    gateway = FakeGateway()
    gateway.positions = [
        {"ticket": 1, "symbol": "EURUSD", "type": 0, "volume": 1.0, "price_current": 1.1},
        {"ticket": 2, "symbol": "GBPUSD", "type": 1, "volume": 0.5, "price_current": 1.1},
    ]
    gateway.bar_rows["GBPUSD"] = _bars(50)

    result = decompose_portfolio_risk(
        PortfolioRiskDecomposeRequest(
            lookback=300,
            horizon_bars=[1],
            confidence=[0.95],
            simulations=500,
        ),
        gateway,
    )

    assert result["error_code"] == "portfolio_pricing_incomplete"
    assert result["failures"] == [
        {
            "symbol": "GBPUSD",
            "stage": "return_history",
            "bars_available": 50,
            "bars_required": 100,
            "reason": "insufficient completed return history",
        }
    ]


def test_portfolio_risk_discloses_history_omissions_in_partial_mode() -> None:
    gateway = FakeGateway()
    gateway.positions = [
        {"ticket": 1, "symbol": "EURUSD", "type": 0, "volume": 1.0, "price_current": 1.1},
        {"ticket": 2, "symbol": "GBPUSD", "type": 1, "volume": 0.5, "price_current": 1.1},
    ]
    gateway.bar_rows["GBPUSD"] = _bars(50)

    result = decompose_portfolio_risk(
        PortfolioRiskDecomposeRequest(
            lookback=300,
            horizon_bars=[1],
            confidence=[0.95],
            simulations=500,
            allow_partial=True,
        ),
        gateway,
    )

    assert result["success"] is True
    assert result["summary"]["symbols"] == 1
    assert result["summary"]["symbols_requested"] == 2
    assert result["data_quality"]["symbols_modeled"] == ["EURUSD"]
    assert result["data_quality"]["symbols_omitted"] == ["GBPUSD"]
    assert result["data_quality"]["history_failures"][0]["symbol"] == "GBPUSD"
    assert any("allow_partial=true" in warning for warning in result["warnings"])


def test_relative_strength_ranks_and_reports_breadth() -> None:
    gateway = FakeGateway()
    request = MarketRelativeStrengthRequest(
        symbols="EURUSD,GBPUSD,USDJPY",
        horizons=[5, 20],
        weights=[0.4, 0.6],
        volatility_lookback=30,
        limit=3,
    )
    result = rank_relative_strength(request, gateway)
    assert result["success"] is True
    assert len(result["leaders"]) == 2
    assert len(result["laggards"]) == 1
    assert {
        row["symbol"] for row in result["leaders"]
    }.isdisjoint(row["symbol"] for row in result["laggards"])
    assert result["leaders"][0]["score"] >= result["leaders"][-1]["score"]
    assert set(result["breadth"]["positive_by_horizon"]) == {"5", "20"}
    assert result["universe_size"] == 3
    assert result["rank_quality"] == "illustrative_small_universe"
    assert result["score_definition"]["weights"] == [0.4, 0.6]
    assert all("rank_percentile" not in row for row in result["leaders"])
