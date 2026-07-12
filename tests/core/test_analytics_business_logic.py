from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from mtdata.analytics.engines import (
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
                "flags": 6,
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


def test_execution_quality_matches_order_and_computes_markout() -> None:
    gateway = FakeGateway()
    start = _now() - 100
    gateway.tick_rows = _ticks(100, start=start)
    gateway.orders = [
        {"ticket": 10, "price_open": 1.10005, "volume_initial": 1.0, "time_setup_msc": (start + 9) * 1000}
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
    assert result["items"][0]["latency_ms"] == 1000.0
    assert result["items"][0]["markout_bps"]["5"] is not None


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
    )
    result = validate_strategies(request, gateway)
    assert result["success"] is True
    assert result["validation"]["purge_bars"] == 5
    assert result["rankings"][0]["id"] == "cross"
    assert result["rankings"][0]["trades"] > 0


def test_portfolio_risk_reconciles_component_expected_shortfall() -> None:
    gateway = FakeGateway()
    gateway.positions = [
        {"ticket": 1, "symbol": "EURUSD", "type": 0, "volume": 1.0, "price_current": 1.1},
        {"ticket": 2, "symbol": "GBPUSD", "type": 1, "volume": 0.5, "price_current": 1.1},
    ]
    result = decompose_portfolio_risk(
        PortfolioRiskDecomposeRequest(lookback=300, horizon_bars=[1], confidence=[0.95], simulations=500),
        gateway,
    )
    assert result["success"] is True
    row = result["risk"][0]
    component_total = sum(item["value"] for item in row["component_expected_shortfall"])
    assert component_total == pytest.approx(row["expected_shortfall"])


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
    assert len(result["leaders"]) == 3
    assert result["leaders"][0]["score"] >= result["leaders"][-1]["score"]
    assert set(result["breadth"]["positive_by_horizon"]) == {"5", "20"}
