from __future__ import annotations

import copy
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from mtdata.core.config import trade_guardrails_config
from mtdata.core.trading.requests import TradePlaceRequest
from mtdata.core.trading.safety import (
    TradeGuardrailsConfig,
    WalletRiskLimits,
    evaluate_trade_guardrails,
    preview_trade_guardrails,
)
from mtdata.core.trading.use_cases import run_trade_place


@pytest.fixture
def restore_trade_guardrails():
    snapshot = copy.deepcopy(trade_guardrails_config.model_dump())
    yield
    for name, value in snapshot.items():
        setattr(trade_guardrails_config, name, value)


def test_trade_guardrails_config_reloads_from_env(monkeypatch, restore_trade_guardrails):
    monkeypatch.setenv("MTDATA_TRADE_ALLOWED_SYMBOLS", "EURUSD, btcusd")
    monkeypatch.setenv("MTDATA_TRADE_BLOCKED_SYMBOLS", "XAUUSD")
    monkeypatch.setenv("MTDATA_TRADE_MAX_VOLUME_BY_SYMBOL", "EURUSD:0.5, BTCUSD=0.03")
    monkeypatch.setenv("MTDATA_TRADE_MAX_RISK_PCT_OF_EQUITY", "1.25")

    trade_guardrails_config.reload_from_env()

    assert trade_guardrails_config.allowed_symbols == ["EURUSD", "BTCUSD"]
    assert trade_guardrails_config.blocked_symbols == ["XAUUSD"]
    assert trade_guardrails_config.max_volume_by_symbol == {
        "EURUSD": 0.5,
        "BTCUSD": 0.03,
    }
    assert trade_guardrails_config.wallet_risk_limits.max_risk_pct_of_equity == 1.25
    assert trade_guardrails_config.is_enabled() is True


def test_preview_trade_guardrails_reports_dynamic_checks(restore_trade_guardrails):
    trade_guardrails_config.enabled = True
    trade_guardrails_config.wallet_risk_limits.max_risk_pct_of_equity = 1.0

    preview = preview_trade_guardrails(
        trade_guardrails_config,
        symbol="EURUSD",
        volume=0.1,
        stop_loss=1.09,
        deviation=10,
        side="BUY",
    )

    assert preview["enabled"] is True
    assert preview["blocked"] is False
    assert "wallet_risk" in preview["checks_not_performed"]


def test_evaluate_trade_guardrails_blocks_wallet_risk_threshold():
    config = TradeGuardrailsConfig(
        enabled=True,
        wallet_risk_limits=WalletRiskLimits(max_risk_pct_of_equity=1.0),
    )
    account = SimpleNamespace(equity=10000.0, balance=10000.0, margin_free=8000.0)
    symbol_info = SimpleNamespace(
        trade_tick_size=1.0,
        trade_tick_value=1.0,
        trade_tick_value_loss=1.0,
    )

    result = evaluate_trade_guardrails(
        config,
        symbol="BTCUSD",
        volume=200.0,
        stop_loss=90.0,
        side="BUY",
        entry_price=100.0,
        account_info=account,
        existing_positions=[],
        symbol_info=symbol_info,
        symbol_info_resolver=lambda _symbol: symbol_info,
    )

    assert result is not None
    assert result["guardrail_blocked"] is True
    assert result["guardrail_rule"] == "wallet_risk"


def test_run_trade_place_dry_run_reports_guardrail_block(restore_trade_guardrails):
    trade_guardrails_config.enabled = True
    trade_guardrails_config.blocked_symbols = ["BTCUSD"]
    place_market_order = MagicMock()
    place_pending_order = MagicMock()

    result = run_trade_place(
        TradePlaceRequest(
            symbol="BTCUSD",
            volume=0.03,
            order_type="BUY",
            require_sl_tp=False,
            dry_run=True,
        ),
        normalize_order_type_input=lambda value: ("BUY", None),
        normalize_pending_expiration=lambda value: (value, False),
        prevalidate_trade_place_market_input=lambda symbol, volume: None,
        place_market_order=place_market_order,
        place_pending_order=place_pending_order,
        close_positions=lambda **kwargs: {"closed_count": 1},
        safe_int_ticket=lambda value: value,
    )

    assert result["guardrail_blocked"] is True
    assert result["dry_run"] is True
    assert result["actionability"] == "blocked_by_guardrails"
    place_market_order.assert_not_called()
    place_pending_order.assert_not_called()


def test_run_trade_place_blocks_static_guardrail_before_send(restore_trade_guardrails):
    trade_guardrails_config.enabled = True
    trade_guardrails_config.max_volume_by_symbol = {"BTCUSD": 0.01}
    place_market_order = MagicMock()

    result = run_trade_place(
        TradePlaceRequest(
            symbol="BTCUSD",
            volume=0.03,
            order_type="BUY",
            require_sl_tp=False,
        ),
        normalize_order_type_input=lambda value: ("BUY", None),
        normalize_pending_expiration=lambda value: (value, False),
        prevalidate_trade_place_market_input=lambda symbol, volume: None,
        place_market_order=place_market_order,
        place_pending_order=lambda **kwargs: {"ok": True},
        close_positions=lambda **kwargs: {"closed_count": 1},
        safe_int_ticket=lambda value: value,
    )

    assert result["guardrail_blocked"] is True
    assert result["guardrail_rule"] == "symbol_policy"
    place_market_order.assert_not_called()
