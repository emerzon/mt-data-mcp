from __future__ import annotations

from types import SimpleNamespace

from mtdata.core.trading.requests import TradeVarCvarRequest
from mtdata.core.trading.use_cases import (
    _calculate_var_cvar_from_pnl,
    run_trade_var_cvar_calculate,
)


def _symbol_info(**overrides):
    values = {
        "trade_contract_size": 1.0,
        "trade_tick_size": 1.0,
        "trade_tick_value": 1.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


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
        symbol_info=lambda symbol: _symbol_info(),
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
            detail="full",
        ),
        gateway=gateway,
    )

    assert out["success"] is True
    assert out["summary"]["positions"] == 1
    assert out["summary"]["symbols"] == 1
    assert out["summary"]["observations"] == 4
    assert out["summary"]["horizon_bars"] == 1
    assert out["summary"]["holding_period"] == "1 H1 bar"
    assert out["summary"]["var_interpretation"] == (
        "One H1 bar loss on the current position snapshot."
    )
    assert out["summary"]["var"] == 14.29
    assert out["summary"]["cvar"] == 14.29
    assert out["summary"]["var_pct_of_equity"] == 1.4286
    assert out["symbol_exposures"][0]["symbol"] == "EURUSD"
    assert out["positions"][0]["signed_notional"] == 100.0
    assert out["worst_observations"][0]["simulated_pnl"] == -14.29


def test_run_trade_var_cvar_uses_account_currency_tick_sensitivity() -> None:
    position = SimpleNamespace(
        ticket=12,
        symbol="USDJPY",
        type=0,
        volume=1.0,
        price_current=110.0,
        price_open=109.0,
        profit=0.0,
    )
    gateway = SimpleNamespace(
        ensure_connection=lambda: None,
        account_info=lambda: SimpleNamespace(equity=200_000.0, currency="USD"),
        positions_get=lambda symbol=None: [position],
        symbol_info=lambda symbol: _symbol_info(
            trade_contract_size=100_000.0,
            trade_tick_size=0.001,
            trade_tick_value=0.91,
        ),
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
            detail="full",
        ),
        gateway=gateway,
    )

    assert out["positions"][0]["signed_notional"] == 100_100.0
    assert out["positions"][0]["contract_price_product"] == 11_000_000.0
    assert out["summary"]["var"] == 14_300.0
    assert out["summary"]["pnl_model"] == "tick_value_linear_sensitivity"
    assert out["summary"]["pnl_unit"] == "account_currency"


def test_run_trade_var_cvar_rejects_position_without_tick_economics() -> None:
    position = SimpleNamespace(
        ticket=13,
        symbol="USDJPY",
        type=0,
        volume=1.0,
        price_current=110.0,
        price_open=109.0,
        profit=0.0,
    )
    gateway = SimpleNamespace(
        ensure_connection=lambda: None,
        account_info=lambda: SimpleNamespace(equity=200_000.0, currency="USD"),
        positions_get=lambda symbol=None: [position],
        symbol_info=lambda symbol: _symbol_info(
            trade_tick_size=0.0,
            trade_tick_value=0.0,
        ),
        POSITION_TYPE_BUY=0,
        POSITION_TYPE_SELL=1,
        ORDER_TYPE_BUY=0,
        ORDER_TYPE_SELL=1,
    )

    out = run_trade_var_cvar_calculate(
        TradeVarCvarRequest(),
        gateway=gateway,
    )

    assert out["error"] == (
        "No usable open positions available for VaR/CVaR calculation."
    )
    assert "cannot be included" in out["warnings"][0]["warning"]


def test_run_trade_var_cvar_calculate_converts_log_returns_to_price_changes() -> None:
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
        symbol_info=lambda symbol: _symbol_info(),
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
    request_args = {
        "timeframe": "H1",
        "lookback": 5,
        "confidence": 75,
        "method": "historical",
        "min_observations": 4,
        "detail": "full",
    }

    log_out = run_trade_var_cvar_calculate(
        TradeVarCvarRequest(**request_args),
        gateway=gateway,
    )
    pct_out = run_trade_var_cvar_calculate(
        TradeVarCvarRequest(**request_args, transform="pct"),
        gateway=gateway,
    )

    assert log_out["summary"]["transform"] == "log_return"
    assert log_out["summary"]["var"] == pct_out["summary"]["var"] == 14.29
    assert log_out["summary"]["cvar"] == pct_out["summary"]["cvar"] == 14.29
    assert log_out["worst_observations"] == pct_out["worst_observations"]


def test_run_trade_var_cvar_calculate_compacts_non_empty_portfolio() -> None:
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
        symbol_info=lambda symbol: _symbol_info(),
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
    assert out["scope"] == "portfolio"
    assert out["summary"]["var"] == 14.29
    assert "symbol_exposures" not in out
    assert "positions" not in out
    assert "worst_observations" not in out


def test_run_trade_var_cvar_calculate_low_sample_error_mentions_override() -> None:
    position = SimpleNamespace(
        ticket=11,
        symbol="BTCUSD",
        type=0,
        volume=1.0,
        price_current=100.0,
        price_open=99.0,
        profit=1.0,
    )
    rates = [
        {"time": index + 1, "close": 100.0 + index}
        for index in range(49)
    ]
    gateway = SimpleNamespace(
        ensure_connection=lambda: None,
        account_info=lambda: SimpleNamespace(equity=1000.0, currency="USD"),
        positions_get=lambda symbol=None: [position],
        symbol_info=lambda symbol: _symbol_info(),
        copy_rates_from_pos=lambda symbol, timeframe, start, count: rates[:count],
        POSITION_TYPE_BUY=0,
        POSITION_TYPE_SELL=1,
        ORDER_TYPE_BUY=0,
        ORDER_TYPE_SELL=1,
    )

    out = run_trade_var_cvar_calculate(
        TradeVarCvarRequest(
            symbol="BTCUSD",
            timeframe="H1",
            lookback=49,
            transform="pct",
        ),
        gateway=gateway,
    )

    assert out["error"] == (
        "Not enough aligned return observations for VaR/CVaR calculation: "
        "lookback=49 yielded 48, need 50. Increase lookback or lower "
        "min_observations to <= 48."
    )
    assert out["available_observations"] == 48
    assert out["min_observations"] == 50
    assert out["lookback"] == 49


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
    assert out["status"] == "no_open_positions"
    assert out["message"] == "No open positions found for VaR/CVaR calculation."
    assert out["positions"] == 0
    assert "summary" not in out
    assert out["equity"] == 1000.0
    assert out["currency"] == "USD"
    assert "symbol_exposures" not in out
    assert "worst_observations" not in out


def test_run_trade_var_cvar_calculate_reports_account_info_failure() -> None:
    def account_info():
        raise RuntimeError("MT5 disconnected")

    gateway = SimpleNamespace(
        ensure_connection=lambda: None,
        account_info=account_info,
        positions_get=lambda symbol=None: [],
    )

    out = run_trade_var_cvar_calculate(
        TradeVarCvarRequest(),
        gateway=gateway,
    )

    assert out["error"] == (
        "Failed to get account info for VaR/CVaR calculation: MT5 disconnected"
    )


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
    assert out["message"] == "No open positions found for VaR/CVaR calculation."
    assert "reason" not in out
    assert "no_action" not in out
    assert out["summary"]["positions"] == 0
    assert out["summary"]["horizon_bars"] == 1
    assert out["summary"]["holding_period"] == "1 H1 bar"
    assert out["summary"]["var"] == 0.0
    assert out["symbol_exposures"] == []
    assert out["positions"] == []
    assert out["worst_observations"] == []
    assert out["message"] == "No open positions found for VaR/CVaR calculation."
