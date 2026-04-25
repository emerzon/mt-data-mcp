from __future__ import annotations

from types import SimpleNamespace

from mtdata.core.trading.requests import TradeVarCvarRequest
from mtdata.core.trading.use_cases import (
    _calculate_var_cvar_from_pnl,
    run_trade_var_cvar_calculate,
)


def test_calculate_var_cvar_from_pnl_uses_conservative_historical_tail() -> None:
    var_value, cvar_value, threshold = _calculate_var_cvar_from_pnl(
        [10.0, -5.0, -15.0, 5.0],
        confidence=0.75,
        method="historical",
    )

    assert var_value == 15.0
    assert cvar_value == 15.0
    assert threshold == -15.0


def test_calculate_var_cvar_from_pnl_gaussian_cvar_exceeds_var() -> None:
    var_value, cvar_value, threshold = _calculate_var_cvar_from_pnl(
        [12.0, -8.0, 5.0, -3.0, -15.0, 9.0],
        confidence=0.95,
        method="gaussian",
    )

    assert threshold < 0.0
    assert var_value > 0.0
    assert cvar_value >= var_value


def test_run_trade_var_cvar_calculate_summarizes_open_position_portfolio() -> None:
    position = SimpleNamespace(
        ticket=11,
        symbol="EURUSD",
        type=0,
        volume=1.0,
        price_current=100.0,
        price_open=99.0,
        profit=1.0,
    )
    gateway = SimpleNamespace(
        ensure_connection=lambda: None,
        account_info=lambda: SimpleNamespace(equity=1000.0, currency="USD"),
        positions_get=lambda symbol=None: [position],
        symbol_info=lambda symbol: SimpleNamespace(trade_contract_size=1.0),
        copy_rates_from_pos=lambda symbol, timeframe, start, count: [
            {"time": 1, "close": 100.0},
            {"time": 2, "close": 95.0},
            {"time": 3, "close": 105.0},
            {"time": 4, "close": 90.0},
            {"time": 5, "close": 110.0},
        ],
        POSITION_TYPE_BUY=0,
        POSITION_TYPE_SELL=1,
        ORDER_TYPE_BUY=0,
        ORDER_TYPE_SELL=1,
    )

    out = run_trade_var_cvar_calculate(
        TradeVarCvarRequest(
            timeframe="H1",
            lookback=5,
            confidence=75,
            method="historical",
            transform="pct",
            min_observations=4,
        ),
        gateway=gateway,
    )

    assert out["success"] is True
    assert out["summary"]["positions"] == 1
    assert out["summary"]["symbols"] == 1
    assert out["summary"]["observations"] == 4
    assert out["summary"]["var"] == 14.29
    assert out["summary"]["cvar"] == 14.29
    assert out["summary"]["var_pct_of_equity"] == 1.4286
    assert out["symbol_exposures"][0]["symbol"] == "EURUSD"
    assert out["positions"][0]["signed_notional"] == 100.0
    assert out["worst_observations"][0]["simulated_pnl"] == -14.29


def test_run_trade_var_cvar_calculate_returns_empty_when_no_open_positions() -> None:
    gateway = SimpleNamespace(
        ensure_connection=lambda: None,
        account_info=lambda: SimpleNamespace(equity=1000.0, currency="USD"),
        positions_get=lambda symbol=None: [],
    )

    out = run_trade_var_cvar_calculate(
        TradeVarCvarRequest(),
        gateway=gateway,
    )

    assert out["success"] is True
    assert out["empty"] is True
    assert out["reason"] == "No open positions found for VaR/CVaR calculation."
    assert "no_action" not in out
    assert out["summary"]["positions"] == 0
    assert out["summary"]["equity"] == 1000.0
    assert out["summary"]["currency"] == "USD"
    assert "var" not in out["summary"]
    assert "symbol_exposures" not in out
    assert "positions" not in out
    assert "worst_observations" not in out
    assert out["message"] == "No open positions found for VaR/CVaR calculation."


def test_run_trade_var_cvar_calculate_full_detail_keeps_empty_shape() -> None:
    gateway = SimpleNamespace(
        ensure_connection=lambda: None,
        account_info=lambda: SimpleNamespace(equity=1000.0, currency="USD"),
        positions_get=lambda symbol=None: [],
    )

    out = run_trade_var_cvar_calculate(
        TradeVarCvarRequest(detail="full"),
        gateway=gateway,
    )

    assert out["success"] is True
    assert out["empty"] is True
    assert out["reason"] == "No open positions found for VaR/CVaR calculation."
    assert "no_action" not in out
    assert out["summary"]["positions"] == 0
    assert out["summary"]["var"] == 0.0
    assert out["symbol_exposures"] == []
    assert out["positions"] == []
    assert out["worst_observations"] == []
    assert out["message"] == "No open positions found for VaR/CVaR calculation."
