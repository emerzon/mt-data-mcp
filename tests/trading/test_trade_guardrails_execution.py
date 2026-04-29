from __future__ import annotations

import copy
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from mtdata.bootstrap.settings import trade_guardrails_config
from mtdata.core.trading.execution import _modify_pending_order
from mtdata.core.trading.gateway import create_trading_gateway as create_real_trading_gateway


@pytest.fixture
def restore_trade_guardrails():
    snapshot = copy.deepcopy(trade_guardrails_config.model_dump())
    yield
    for name, value in snapshot.items():
        setattr(trade_guardrails_config, name, value)


@pytest.fixture
def mock_mt5():
    mt5 = SimpleNamespace()
    mt5.ORDER_TYPE_BUY_LIMIT = 2
    mt5.ORDER_TYPE_SELL_LIMIT = 3
    mt5.ORDER_TYPE_BUY_STOP = 4
    mt5.ORDER_TYPE_SELL_STOP = 5
    mt5.ORDER_TIME_GTC = 0
    mt5.ORDER_TIME_SPECIFIED = 1
    mt5.TRADE_ACTION_MODIFY = 7
    mt5.TRADE_RETCODE_DONE = 10009
    mt5.retcode_name = lambda retcode: {10009: "TRADE_RETCODE_DONE"}.get(retcode, str(retcode))
    mt5.account_info = lambda: SimpleNamespace(
        equity=10000.0,
        balance=10000.0,
        margin_free=9000.0,
        profit=0.0,
        margin_level=500.0,
    )
    mt5.positions_get = lambda *args, **kwargs: []
    mt5.symbol_info = lambda symbol: SimpleNamespace(
        visible=True,
        point=0.0001,
        digits=4,
        trade_stops_level=0,
        trade_freeze_level=0,
        trade_tick_size=0.0001,
        trade_tick_value=10.0,
        trade_tick_value_loss=10.0,
    )
    mt5.symbol_info_tick = lambda symbol: SimpleNamespace(bid=1.1002, ask=1.1004)
    mt5.orders_get = lambda *args, **kwargs: [
        SimpleNamespace(
            ticket=100,
            symbol="EURUSD",
            price_open=1.1000,
            sl=1.0990,
            tp=1.1200,
            type=mt5.ORDER_TYPE_BUY_LIMIT,
            volume=1.0,
            volume_current=1.0,
            volume_initial=1.0,
            type_time=mt5.ORDER_TIME_GTC,
            time_expiration=0,
            magic=123,
        )
    ]
    mt5.order_send = lambda request: SimpleNamespace(
        retcode=10009,
        deal=0,
        order=request["order"],
        comment="ok",
        request_id=1,
    )
    return mt5


@pytest.fixture
def patch_gateway(mock_mt5):
    def _build_gateway(*, gateway=None, include_retcode_name=False, **_kwargs):
        if gateway is not None:
            return gateway
        return create_real_trading_gateway(
            adapter=mock_mt5,
            include_retcode_name=include_retcode_name,
            ensure_connection_impl=lambda: None,
        )

    with patch("mtdata.core.trading.execution.create_trading_gateway", side_effect=_build_gateway):
        yield mock_mt5


def test_modify_pending_order_blocks_risk_increase(
    restore_trade_guardrails,
    patch_gateway,
):
    trade_guardrails_config.enabled = True
    trade_guardrails_config.wallet_risk_limits.max_risk_pct_of_equity = 5.0

    result = _modify_pending_order(ticket=100, price=1.1000, stop_loss=1.0940)

    assert result["guardrail_blocked"] is True
    assert result["guardrail_rule"] == "wallet_risk"


def test_modify_pending_order_allows_tighter_stop_loss(
    restore_trade_guardrails,
    patch_gateway,
):
    trade_guardrails_config.enabled = True
    trade_guardrails_config.wallet_risk_limits.max_risk_pct_of_equity = 1.0

    result = _modify_pending_order(ticket=100, price=1.1000, stop_loss=1.0995)

    assert result["success"] is True
    assert result["pending_order_ticket"] == 100
