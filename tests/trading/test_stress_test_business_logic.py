from __future__ import annotations

from types import SimpleNamespace

from mtdata.core.trading.requests import TradeStressTestRequest
from mtdata.core.trading.use_cases import run_trade_stress_test


class _Gateway:
    POSITION_TYPE_BUY = 0
    POSITION_TYPE_SELL = 1

    def ensure_connection(self) -> None:
        return None

    def account_info(self):
        return SimpleNamespace(equity=10_000.0, currency="USD")

    def positions_get(self):
        return [
            SimpleNamespace(
                ticket=1,
                symbol="EURUSD",
                type=0,
                volume=1.0,
                price_current=1.1000,
                price_open=1.0900,
            ),
            SimpleNamespace(
                ticket=2,
                symbol="EURUSD",
                type=1,
                volume=0.5,
                price_current=1.1000,
                price_open=1.1200,
            ),
        ]

    def symbol_info(self, symbol):
        return SimpleNamespace(
            trade_tick_size=0.0001,
            trade_tick_value=10.0,
            trade_tick_value_profit=10.0,
            trade_tick_value_loss=10.0,
            point=0.0001,
        )


def test_trade_stress_test_offsets_long_and_short_positions():
    result = run_trade_stress_test(
        TradeStressTestRequest(shocks={"EURUSD": -1.0}, detail="full"),
        gateway=_Gateway(),
    )

    assert result["success"] is True
    assert result["positions_evaluated"] == 2
    assert result["total_pnl_impact"] == -550.0
    assert result["equity_after"] == 9450.0
