import math
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.mtdata.core.trading import (
    _cancel_pending,
    _close_positions,
    _modify_pending_order,
    _modify_position,
    _place_market_order,
    _place_pending_order,
)
from src.mtdata.core.trading.gateway import MT5TradingGateway
from src.mtdata.core.trading.gateway import (
    create_trading_gateway as create_real_trading_gateway,
)


@pytest.fixture
def mock_mt5():
    prev_mt5 = sys.modules.get("MetaTrader5")
    mock_mt5 = MagicMock()
    sys.modules["MetaTrader5"] = mock_mt5
    
    # Defaults
    mock_mt5.symbol_info.return_value = MagicMock(
        visible=True,
        point=0.00001,
        digits=5,
        trade_calc_mode=0, # Forex
        volume_min=0.01,
        volume_max=100.0,
        volume_step=0.01,
    )
    mock_mt5.symbol_info_tick.return_value = MagicMock(
        bid=1.05000,
        ask=1.05010,
    )
    mock_mt5.symbol_select.return_value = True
    
    # Order constants
    mock_mt5.ORDER_TYPE_BUY = 0
    mock_mt5.ORDER_TYPE_SELL = 1
    mock_mt5.ORDER_TYPE_BUY_LIMIT = 2
    mock_mt5.ORDER_TYPE_SELL_LIMIT = 3
    mock_mt5.ORDER_TYPE_BUY_STOP = 4
    mock_mt5.ORDER_TYPE_SELL_STOP = 5
    mock_mt5.TRADE_ACTION_DEAL = 1
    mock_mt5.TRADE_ACTION_PENDING = 5
    mock_mt5.TRADE_RETCODE_DONE = 10009
    
    # Default success
    mock_mt5.order_send.return_value = MagicMock(
        retcode=10009,
        deal=123,
        order=456,
        volume=0.1,
        price=1.05010,
        bid=1.05000,
        ask=1.05010,
        comment="",
        request_id=789,
    )
    mock_mt5.positions_get.return_value = [MagicMock(symbol="EURUSD", sl=0.0, tp=0.0)]
    
    # Guard
    def _build_gateway(*, gateway=None, include_trade_preflight=False, include_retcode_name=False, **_):
        if gateway is not None:
            return gateway
        return create_real_trading_gateway(
            adapter=mock_mt5,
            include_trade_preflight=include_trade_preflight,
            include_retcode_name=include_retcode_name,
            ensure_connection_impl=lambda: None,
        )

    with patch("src.mtdata.core.trading.orders.create_trading_gateway", side_effect=_build_gateway), patch(
        "src.mtdata.core.trading.execution.create_trading_gateway", side_effect=_build_gateway
    ), patch("src.mtdata.core.trading.orders.trade_guardrails_config") as mock_guard_config:
        mock_guard_config.enabled = False
        mock_guard_config.trading_enabled = True
        mock_guard_config.max_volume_by_symbol = {}
        mock_guard_config.allowed_symbols = []
        mock_guard_config.blocked_symbols = []
        mock_guard_config.max_volume = None
        mock_guard_config.is_enabled.return_value = False
        yield mock_mt5
    if prev_mt5 is not None:
        sys.modules["MetaTrader5"] = prev_mt5


def test_place_market_order_success(mock_mt5):
    res = _place_market_order(
        symbol="EURUSD",
        volume=0.1,
        order_type="BUY",
        stop_loss=1.04000,
        take_profit=1.06000,
        comment="Test Buy",
        deviation=10
    )

    assert "error" not in res
    assert res["retcode"] == mock_mt5.TRADE_RETCODE_DONE
    assert res["sl_tp_result"]["status"] == "applied"
    
    # The protected market order is submitted atomically with SL/TP.
    assert mock_mt5.order_send.call_count == 1
    
    deal_req = mock_mt5.order_send.call_args.args[0]
    assert deal_req["action"] == mock_mt5.TRADE_ACTION_DEAL
    assert deal_req["symbol"] == "EURUSD"
    assert deal_req["type"] == mock_mt5.ORDER_TYPE_BUY
    assert deal_req["volume"] == 0.1
    assert math.isclose(deal_req["sl"], 1.04000)
    assert math.isclose(deal_req["tp"], 1.06000)
    assert "comment_fallback" not in res
    assert "fallback_used" not in res["sl_tp_result"]


def test_place_orders_use_configured_magic_number(mock_mt5, monkeypatch):
    monkeypatch.setattr("src.mtdata.core.trading.orders.mt5_config.order_magic", 345678)

    market_res = _place_market_order(
        symbol="EURUSD",
        volume=0.1,
        order_type="BUY",
    )
    assert "error" not in market_res
    market_req = mock_mt5.order_send.call_args_list[0].args[0]
    assert market_req["magic"] == 345678

    mock_mt5.order_send.reset_mock()

    pending_res = _place_pending_order(
        symbol="EURUSD",
        volume=0.1,
        order_type="BUY_LIMIT",
        price=1.04000,
    )
    assert "error" not in pending_res
    pending_req = mock_mt5.order_send.call_args[0][0]
    assert pending_req["magic"] == 345678


def test_place_orders_allow_explicit_magic_override(mock_mt5, monkeypatch):
    monkeypatch.setattr("src.mtdata.core.trading.orders.mt5_config.order_magic", 345678)

    market_res = _place_market_order(
        symbol="EURUSD",
        volume=0.1,
        order_type="BUY",
        magic=111222,
    )
    assert "error" not in market_res
    market_req = mock_mt5.order_send.call_args_list[0].args[0]
    assert market_req["magic"] == 111222

    mock_mt5.order_send.reset_mock()

    pending_res = _place_pending_order(
        symbol="EURUSD",
        volume=0.1,
        order_type="BUY_LIMIT",
        price=1.04000,
        magic=333444,
    )
    assert "error" not in pending_res
    pending_req = mock_mt5.order_send.call_args[0][0]
    assert pending_req["magic"] == 333444


def test_place_market_order_treats_tiny_zero_stop_loss_as_omitted(mock_mt5):
    res = _place_market_order(
        symbol="EURUSD",
        volume=0.1,
        order_type="BUY",
        stop_loss=1e-12,
    )

    assert "error" not in res
    assert res["sl_tp_result"]["status"] == "not_requested"
    assert mock_mt5.order_send.call_count == 1


def test_place_market_order_treats_tiny_zero_take_profit_as_omitted(mock_mt5):
    res = _place_market_order(
        symbol="EURUSD",
        volume=0.1,
        order_type="BUY",
        take_profit=1e-12,
    )

    assert "error" not in res
    assert res["sl_tp_result"]["status"] == "not_requested"
    assert mock_mt5.order_send.call_count == 1


def test_place_market_order_invalid_type(mock_mt5):
    res = _place_market_order(
        symbol="EURUSD",
        volume=0.1,
        order_type="BUY_LIMIT",  # Invalid for market order
    )
    assert "error" in res
    assert "Use BUY or SELL for market orders" in res["error"]


def test_place_market_order_sl_tp_logic_violations(mock_mt5):
    # BUY with SL above price (invalid)
    res = _place_market_order(
        symbol="EURUSD",
        volume=0.1,
        order_type="BUY",
        stop_loss=1.06000,  # Ask is 1.05010
    )
    assert "error" in res
    assert "stop_loss must be below the live bid" in res["error"]

    # SELL with SL below price (invalid)
    res = _place_market_order(
        symbol="EURUSD",
        volume=0.1,
        order_type="SELL",
        stop_loss=1.04000,  # Bid is 1.05000
    )
    assert "error" in res
    assert "stop_loss must be above the live ask" in res["error"]

    # BUY with TP below price (invalid)
    res = _place_market_order(
        symbol="EURUSD",
        volume=0.1,
        order_type="BUY",
        take_profit=1.04000,  # Ask is 1.05010
    )
    assert "error" in res
    assert "take_profit must be above the live bid" in res["error"]

    # SELL with TP above price (invalid)
    res = _place_market_order(
        symbol="EURUSD",
        volume=0.1,
        order_type="SELL",
        take_profit=1.06000,  # Bid is 1.05000
    )
    assert "error" in res
    assert "take_profit must be below the live ask" in res["error"]


def test_place_market_order_rejects_buy_stop_loss_inside_live_spread(mock_mt5):
    res = _place_market_order(
        symbol="EURUSD",
        volume=0.1,
        order_type="BUY",
        stop_loss=1.05005,  # Below ask, but still above bid and invalid for a BUY position
        take_profit=1.06000,
    )

    assert "error" in res
    assert "live bid" in res["error"]
    mock_mt5.order_send.assert_not_called()


def test_place_market_order_accepts_buy_take_profit_inside_live_spread(mock_mt5):
    res = _place_market_order(
        symbol="EURUSD",
        volume=0.1,
        order_type="BUY",
        stop_loss=1.04000,
        take_profit=1.05005,
    )

    assert "error" not in res
    assert res["retcode"] == mock_mt5.TRADE_RETCODE_DONE
    assert res["sl_tp_result"]["status"] == "applied"


def test_place_market_order_accepts_sell_take_profit_inside_live_spread(mock_mt5):
    res = _place_market_order(
        symbol="EURUSD",
        volume=0.1,
        order_type="SELL",
        stop_loss=1.06000,
        take_profit=1.05005,
    )

    assert "error" not in res
    assert res["retcode"] == mock_mt5.TRADE_RETCODE_DONE
    assert res["sl_tp_result"]["status"] == "applied"


def test_place_market_order_rejects_levels_inside_broker_stop_distance(mock_mt5):
    mock_mt5.symbol_info.return_value = MagicMock(
        visible=True,
        point=0.00001,
        digits=5,
        trade_calc_mode=0,
        volume_min=0.01,
        volume_max=100.0,
        volume_step=0.01,
        trade_stops_level=30,
        trade_freeze_level=0,
    )

    res = _place_market_order(
        symbol="EURUSD",
        volume=0.1,
        order_type="BUY",
        stop_loss=1.04990,  # 10 points below bid; broker requires 30
        take_profit=1.06000,
    )

    assert "error" in res
    assert "min_distance_points=30" in res["error"]
    mock_mt5.order_send.assert_not_called()


def test_place_market_order_volume_validation(mock_mt5):
    res = _place_market_order(
        symbol="EURUSD",
        volume=0.001, # Below min 0.01
        order_type="BUY",
    )
    assert "error" in res
    assert "volume must be >=" in res["error"]
    
    res = _place_market_order(
        symbol="EURUSD",
        volume=0.015, # Not aligned with step 0.01
        order_type="BUY",
    )
    assert "error" in res
    assert "volume must align to step" in res["error"]


def test_place_pending_order_success(mock_mt5):
    res = _place_pending_order(
        symbol="EURUSD",
        volume=0.1,
        order_type="BUY_LIMIT",
        price=1.04000, # Below ask 1.05010 (valid)
        stop_loss=1.03000,
        take_profit=1.06000,
    )
    
    assert "error" not in res
    assert res["success"] is True
    assert mock_mt5.order_send.call_count >= 1
    
    req = mock_mt5.order_send.call_args[0][0]
    assert req["action"] == mock_mt5.TRADE_ACTION_PENDING
    assert req["type"] == mock_mt5.ORDER_TYPE_BUY_LIMIT
    assert math.isclose(req["price"], 1.04000)
    assert math.isclose(req["sl"], 1.03000)
    assert math.isclose(req["tp"], 1.06000)


def test_place_pending_order_treats_tiny_zero_stop_loss_as_omitted(mock_mt5):
    res = _place_pending_order(
        symbol="EURUSD",
        volume=0.1,
        order_type="BUY_LIMIT",
        price=1.04000,
        stop_loss=1e-12,
    )

    assert "error" not in res
    req = mock_mt5.order_send.call_args[0][0]
    assert math.isclose(req["sl"], 0.0)
    assert math.isclose(req["tp"], 0.0)


def test_place_pending_order_treats_tiny_zero_take_profit_as_omitted(mock_mt5):
    res = _place_pending_order(
        symbol="EURUSD",
        volume=0.1,
        order_type="BUY_LIMIT",
        price=1.04000,
        take_profit=1e-12,
    )

    assert "error" not in res
    req = mock_mt5.order_send.call_args[0][0]
    assert math.isclose(req["sl"], 0.0)
    assert math.isclose(req["tp"], 0.0)


def test_place_pending_order_bad_side(mock_mt5):
    # BUY_LIMIT above ask
    res = _place_pending_order(
        symbol="EURUSD",
        volume=0.1,
        order_type="BUY_LIMIT",
        price=1.06000, # Ask is 1.05010
    )
    assert "error" in res
    assert "Price must be below ask for BUY_LIMIT" in res["error"]

    # SELL_STOP above bid
    res = _place_pending_order(
        symbol="EURUSD",
        volume=0.1,
        order_type="SELL_STOP",
        price=1.06000, # Bid is 1.05000
    )
    assert "error" in res
    assert "Price must be below bid for SELL_STOP" in res["error"]

def test_place_pending_order_sl_tp_violations(mock_mt5):
    # Pending BUY with SL > Entry
    res = _place_pending_order(
        symbol="EURUSD",
        volume=0.1,
        order_type="BUY_LIMIT",
        price=1.04000,
        stop_loss=1.04500, # Invalid
    )
    assert "error" in res
    assert "stop_loss must be below entry" in res["error"]

    # Pending SELL with TP > Entry
    res = _place_pending_order(
        symbol="EURUSD",
        volume=0.1,
        order_type="SELL_LIMIT",
        price=1.06000,
        take_profit=1.07000, # Invalid
    )
    assert "error" in res
    assert "take_profit must be below entry" in res["error"]

def test_place_pending_order_implicit_types(mock_mt5):
    # Passing 'BUY' with a price below ask should yield a BUY_LIMIT
    res = _place_pending_order(
        symbol="EURUSD", volume=0.1, order_type="BUY", price=1.04000
    )
    assert "error" not in res
    req = mock_mt5.order_send.call_args[0][0]
    assert req["type"] == mock_mt5.ORDER_TYPE_BUY_LIMIT

    # Passing 'BUY' with a price above ask should yield a BUY_STOP
    res = _place_pending_order(
        symbol="EURUSD", volume=0.1, order_type="BUY", price=1.06000
    )
    assert "error" not in res
    req = mock_mt5.order_send.call_args[0][0]
    assert req["type"] == mock_mt5.ORDER_TYPE_BUY_STOP

    # Passing 'SELL' with a price above bid should yield a SELL_LIMIT
    res = _place_pending_order(
        symbol="EURUSD", volume=0.1, order_type="SELL", price=1.06000
    )
    assert "error" not in res
    req = mock_mt5.order_send.call_args[0][0]
    assert req["type"] == mock_mt5.ORDER_TYPE_SELL_LIMIT

    # Passing 'SELL' with a price below bid should yield a SELL_STOP
    res = _place_pending_order(
        symbol="EURUSD", volume=0.1, order_type="SELL", price=1.04000
    )
    assert "error" not in res
    req = mock_mt5.order_send.call_args[0][0]
    assert req["type"] == mock_mt5.ORDER_TYPE_SELL_STOP


def test_place_pending_order_implicit_type_uses_refreshed_tick_before_send(mock_mt5):
    mock_mt5.symbol_info_tick.side_effect = [
        MagicMock(bid=1.05000, ask=1.05010),
        MagicMock(bid=1.03980, ask=1.03990),
    ]

    res = _place_pending_order(
        symbol="EURUSD",
        volume=0.1,
        order_type="BUY",
        price=1.04000,
    )

    assert "error" not in res
    req = mock_mt5.order_send.call_args[0][0]
    assert req["type"] == mock_mt5.ORDER_TYPE_BUY_STOP
    assert mock_mt5.symbol_info_tick.call_count == 2


def test_place_pending_order_rejects_price_at_market_for_auto_side(mock_mt5):
    res = _place_pending_order(
        symbol="EURUSD",
        volume=0.1,
        order_type="BUY",
        price=1.05010,
    )

    assert "error" in res
    assert "price is at market for BUY pending order" in res["error"]
    mock_mt5.order_send.assert_not_called()


def test_place_pending_order_rejects_entry_inside_broker_stop_distance(mock_mt5):
    mock_mt5.symbol_info.return_value = MagicMock(
        visible=True,
        point=0.00001,
        digits=5,
        trade_calc_mode=0,
        volume_min=0.01,
        volume_max=100.0,
        volume_step=0.01,
        trade_stops_level=30,
        trade_freeze_level=0,
    )

    res = _place_pending_order(
        symbol="EURUSD",
        volume=0.1,
        order_type="BUY_LIMIT",
        price=1.04995,
    )

    assert "error" in res
    assert "pending entry is too close" in res["error"]
    assert "min_distance_points=30" in res["error"]
    mock_mt5.order_send.assert_not_called()


def test_place_pending_order_rejects_stop_loss_inside_broker_stop_distance(mock_mt5):
    mock_mt5.symbol_info.return_value = MagicMock(
        visible=True,
        point=0.00001,
        digits=5,
        trade_calc_mode=0,
        volume_min=0.01,
        volume_max=100.0,
        volume_step=0.01,
        trade_stops_level=30,
        trade_freeze_level=0,
    )

    res = _place_pending_order(
        symbol="EURUSD",
        volume=0.1,
        order_type="BUY_LIMIT",
        price=1.04000,
        stop_loss=1.03990,
    )

    assert "error" in res
    assert "stop_loss is too close to entry for BUY pending orders" in res["error"]
    assert "min_distance_points=30" in res["error"]
    mock_mt5.order_send.assert_not_called()


def test_place_pending_order_blocks_on_trade_preflight(mock_mt5):
    mock_mt5.account_info.return_value = MagicMock(
        trade_allowed=True,
        trade_expert=True,
        server="Demo",
        company="Broker",
        trade_mode=0,
    )
    mock_mt5.terminal_info.return_value = MagicMock(
        trade_allowed=False,
        tradeapi_disabled=False,
        connected=True,
        community_account=True,
    )

    res = _place_pending_order(
        symbol="EURUSD",
        volume=0.1,
        order_type="BUY_LIMIT",
        price=1.04000,
    )

    assert "error" in res
    assert "Trading not ready" in res["error"]
    assert res["preflight"]["execution_ready"] is True
    assert res["preflight"]["execution_ready_strict"] is False
    assert "Terminal AutoTrading is disabled." in res["preflight"]["execution_blockers"]
    assert "Enable AutoTrading in the MT5 terminal toolbar" in res["hint"]
    assert any("AutoTrading" in step for step in res["next_steps"])
    mock_mt5.order_send.assert_not_called()


def test_place_market_order_blocks_on_trade_preflight(mock_mt5):
    mock_mt5.account_info.return_value = MagicMock(
        trade_allowed=True,
        trade_expert=True,
        server="Demo",
        company="Broker",
        trade_mode=0,
    )
    mock_mt5.terminal_info.return_value = MagicMock(
        trade_allowed=False,
        tradeapi_disabled=False,
        connected=True,
        community_account=True,
    )

    res = _place_market_order(
        symbol="EURUSD",
        volume=0.1,
        order_type="BUY",
    )

    assert "error" in res
    assert "Trading not ready" in res["error"]
    assert res["preflight"]["execution_ready"] is True
    assert res["preflight"]["execution_ready_strict"] is False
    assert "Terminal AutoTrading is disabled." in res["preflight"]["execution_blockers"]
    assert "Enable AutoTrading in the MT5 terminal toolbar" in res["hint"]
    assert any("AutoTrading" in step for step in res["next_steps"])
    mock_mt5.order_send.assert_not_called()


def test_place_market_order_accepts_injected_gateway():
    adapter = MagicMock()
    adapter.symbol_info.return_value = MagicMock(
        visible=True,
        point=0.00001,
        digits=5,
        volume_min=0.01,
        volume_max=100.0,
        volume_step=0.01,
    )
    adapter.symbol_info_tick.side_effect = [
        MagicMock(bid=1.05000, ask=1.05010),
        MagicMock(bid=1.05000, ask=1.05010),
    ]
    adapter.positions_get.return_value = [MagicMock(sl=1.04000, tp=1.06000)]
    adapter.order_send.side_effect = [
        MagicMock(
            retcode=10009,
            deal=123,
            order=456,
            volume=0.1,
            price=1.05010,
            bid=1.05000,
            ask=1.05010,
            comment="",
            request_id=789,
        ),
        MagicMock(retcode=10009, comment="", request_id=790),
    ]
    adapter.ORDER_TYPE_BUY = 0
    adapter.ORDER_TYPE_SELL = 1
    adapter.TRADE_ACTION_DEAL = 1
    adapter.TRADE_ACTION_SLTP = 6
    adapter.TRADE_RETCODE_DONE = 10009
    adapter.ORDER_TIME_GTC = 0
    adapter.ORDER_FILLING_IOC = 1
    ensure_connection = MagicMock()
    gateway = MT5TradingGateway(
        adapter=adapter,
        ensure_connection_impl=ensure_connection,
        build_trade_preflight_impl=lambda mt5, **_: {
            "execution_ready": True,
            "execution_ready_strict": True,
        },
        retcode_name_impl=lambda mt5, retcode: "TRADE_RETCODE_DONE",
    )

    with patch(
        "src.mtdata.core.trading.orders._resolve_open_position",
        return_value=(MagicMock(sl=0.0, tp=0.0), 456, {}),
    ), patch("src.mtdata.core.trading.orders.trade_guardrails_config") as mock_guard_config:
        mock_guard_config.enabled = False
        mock_guard_config.trading_enabled = True
        mock_guard_config.max_volume_by_symbol = {}
        mock_guard_config.is_enabled.return_value = False
        
        res = _place_market_order(
            symbol="EURUSD",
            volume=0.1,
            order_type="BUY",
            stop_loss=1.04000,
            take_profit=1.06000,
            gateway=gateway,
        )

    assert "error" not in res
    ensure_connection.assert_called_once_with()
    assert adapter.order_send.call_count == 1
    request = adapter.order_send.call_args.args[0]
    assert math.isclose(request["sl"], 1.04000)
    assert math.isclose(request["tp"], 1.06000)
    assert "comment_fallback" not in res


def test_place_market_order_retries_fill_modes_when_first_mode_fails(mock_mt5):
    mock_mt5.ORDER_FILLING_IOC = 1
    mock_mt5.ORDER_FILLING_FOK = 0
    mock_mt5.ORDER_FILLING_RETURN = 2
    mock_mt5.ORDER_TIME_GTC = 0
    mock_mt5.order_send.side_effect = [
        MagicMock(
            retcode=10030,
            deal=0,
            order=0,
            volume=0.1,
            price=1.05010,
            bid=1.05000,
            ask=1.05010,
            comment="IOC rejected",
            request_id=700,
        ),
        MagicMock(
            retcode=10009,
            deal=123,
            order=456,
            volume=0.1,
            price=1.05010,
            bid=1.05000,
            ask=1.05010,
            comment="",
            request_id=701,
        ),
    ]

    res = _place_market_order(
        symbol="EURUSD",
        volume=0.1,
        order_type="BUY",
    )

    assert "error" not in res
    assert res["type_filling_used"] == mock_mt5.ORDER_FILLING_FOK
    assert len(res["fill_mode_attempts"]) == 2
    first_req = mock_mt5.order_send.call_args_list[0].args[0]
    second_req = mock_mt5.order_send.call_args_list[1].args[0]
    assert first_req["type_filling"] == mock_mt5.ORDER_FILLING_IOC
    assert second_req["type_filling"] == mock_mt5.ORDER_FILLING_FOK


def test_place_market_order_retries_same_fill_mode_after_price_change(mock_mt5):
    mock_mt5.ORDER_FILLING_IOC = 1
    mock_mt5.ORDER_FILLING_FOK = 0
    mock_mt5.ORDER_FILLING_RETURN = 2
    mock_mt5.SYMBOL_FILLING_FOK = 1
    mock_mt5.SYMBOL_FILLING_IOC = 2
    mock_mt5.SYMBOL_FILLING_RETURN = 4
    mock_mt5.TRADE_RETCODE_PRICE_CHANGED = 10020
    mock_mt5.ORDER_TIME_GTC = 0
    mock_mt5.symbol_info.return_value.filling_mode = mock_mt5.SYMBOL_FILLING_FOK
    mock_mt5.symbol_info_tick.side_effect = [
        MagicMock(bid=1.05000, ask=1.05010),
        MagicMock(bid=1.05000, ask=1.05010),
        MagicMock(bid=1.05020, ask=1.05030),
    ]
    mock_mt5.order_send.side_effect = [
        MagicMock(
            retcode=10020,
            deal=0,
            order=0,
            volume=0.1,
            price=1.05010,
            bid=1.05000,
            ask=1.05010,
            comment="price changed",
            request_id=703,
        ),
        MagicMock(
            retcode=10009,
            deal=123,
            order=456,
            volume=0.1,
            price=1.05030,
            bid=1.05020,
            ask=1.05030,
            comment="",
            request_id=704,
        ),
    ]

    res = _place_market_order(
        symbol="EURUSD",
        volume=0.1,
        order_type="BUY",
    )

    assert "error" not in res
    assert mock_mt5.order_send.call_count == 2
    first_req = mock_mt5.order_send.call_args_list[0].args[0]
    second_req = mock_mt5.order_send.call_args_list[1].args[0]
    assert first_req["type_filling"] == mock_mt5.ORDER_FILLING_FOK
    assert second_req["type_filling"] == mock_mt5.ORDER_FILLING_FOK
    assert first_req["price"] == pytest.approx(1.05010)
    assert second_req["price"] == pytest.approx(1.05030)
    assert res["type_filling_used"] == mock_mt5.ORDER_FILLING_FOK


def test_place_market_order_accepts_done_partial_without_retry(mock_mt5):
    mock_mt5.ORDER_FILLING_IOC = 1
    mock_mt5.ORDER_FILLING_FOK = 0
    mock_mt5.ORDER_FILLING_RETURN = 2
    mock_mt5.ORDER_TIME_GTC = 0
    mock_mt5.TRADE_RETCODE_DONE_PARTIAL = 10010
    mock_mt5.order_send.return_value = MagicMock(
        retcode=10010,
        deal=123,
        order=456,
        volume=0.05,
        price=1.05010,
        bid=1.05000,
        ask=1.05010,
        comment="partial fill",
        request_id=702,
    )

    res = _place_market_order(
        symbol="EURUSD",
        volume=0.1,
        order_type="BUY",
    )

    assert "error" not in res
    assert res["retcode"] == 10010
    assert len(res["fill_mode_attempts"]) == 1
    assert res["type_filling_used"] == mock_mt5.ORDER_FILLING_IOC
    assert mock_mt5.order_send.call_count == 1


def test_place_market_order_does_not_retry_fill_modes_after_terminal_reject(mock_mt5):
    mock_mt5.ORDER_FILLING_IOC = 1
    mock_mt5.ORDER_FILLING_FOK = 0
    mock_mt5.ORDER_FILLING_RETURN = 2
    mock_mt5.TRADE_RETCODE_NO_MONEY = 10019
    mock_mt5.ORDER_TIME_GTC = 0
    mock_mt5.order_send.side_effect = [
        MagicMock(
            retcode=10019,
            deal=0,
            order=0,
            volume=0.1,
            price=1.05010,
            bid=1.05000,
            ask=1.05010,
            comment="No money",
            request_id=703,
        ),
        MagicMock(
            retcode=10009,
            deal=123,
            order=456,
            volume=0.1,
            price=1.05010,
            bid=1.05000,
            ask=1.05010,
            comment="",
            request_id=704,
        ),
    ]

    res = _place_market_order(
        symbol="EURUSD",
        volume=0.1,
        order_type="BUY",
    )

    assert res["error"] == "Failed to send order"
    assert res["retcode"] == 10019
    assert len(res["fill_mode_attempts"]) == 1
    assert mock_mt5.order_send.call_count == 1


def test_place_market_order_prefers_symbol_fill_mode(mock_mt5):
    mock_mt5.ORDER_FILLING_IOC = 1
    mock_mt5.ORDER_FILLING_FOK = 0
    mock_mt5.ORDER_FILLING_RETURN = 2
    mock_mt5.SYMBOL_FILLING_FOK = 1
    mock_mt5.SYMBOL_FILLING_IOC = 2
    mock_mt5.SYMBOL_FILLING_RETURN = 4
    mock_mt5.ORDER_TIME_GTC = 0
    mock_mt5.symbol_info.return_value.filling_mode = mock_mt5.SYMBOL_FILLING_FOK

    res = _place_market_order(
        symbol="EURUSD",
        volume=0.1,
        order_type="BUY",
    )

    assert "error" not in res
    assert res["type_filling_used"] == mock_mt5.ORDER_FILLING_FOK
    assert mock_mt5.order_send.call_count == 1
    first_req = mock_mt5.order_send.call_args.args[0]
    assert first_req["type_filling"] == mock_mt5.ORDER_FILLING_FOK


def test_place_pending_order_retries_fill_modes_when_first_mode_fails(mock_mt5):
    mock_mt5.ORDER_FILLING_IOC = 1
    mock_mt5.ORDER_FILLING_FOK = 0
    mock_mt5.ORDER_FILLING_RETURN = 2
    mock_mt5.ORDER_TIME_GTC = 0
    mock_mt5.order_send.side_effect = [
        MagicMock(
            retcode=10030,
            deal=0,
            order=0,
            volume=0.1,
            price=1.04000,
            bid=1.05000,
            ask=1.05010,
            comment="IOC rejected",
            request_id=800,
        ),
        MagicMock(
            retcode=10009,
            deal=0,
            order=456,
            volume=0.1,
            price=1.04000,
            bid=1.05000,
            ask=1.05010,
            comment="",
            request_id=801,
        ),
    ]

    res = _place_pending_order(
        symbol="EURUSD",
        volume=0.1,
        order_type="BUY_LIMIT",
        price=1.04000,
    )

    assert "error" not in res
    assert res["type_filling_used"] == mock_mt5.ORDER_FILLING_FOK
    assert len(res["fill_mode_attempts"]) == 2
    first_req = mock_mt5.order_send.call_args_list[0].args[0]
    second_req = mock_mt5.order_send.call_args_list[1].args[0]
    assert first_req["type_filling"] == mock_mt5.ORDER_FILLING_IOC
    assert second_req["type_filling"] == mock_mt5.ORDER_FILLING_FOK


def test_place_pending_order_prefers_symbol_fill_mode(mock_mt5):
    mock_mt5.ORDER_FILLING_IOC = 1
    mock_mt5.ORDER_FILLING_FOK = 0
    mock_mt5.ORDER_FILLING_RETURN = 2
    mock_mt5.SYMBOL_FILLING_FOK = 1
    mock_mt5.SYMBOL_FILLING_IOC = 2
    mock_mt5.SYMBOL_FILLING_RETURN = 4
    mock_mt5.ORDER_TIME_GTC = 0
    mock_mt5.symbol_info.return_value.filling_mode = mock_mt5.SYMBOL_FILLING_FOK

    res = _place_pending_order(
        symbol="EURUSD",
        volume=0.1,
        order_type="BUY_LIMIT",
        price=1.04000,
    )

    assert "error" not in res
    assert res["type_filling_used"] == mock_mt5.ORDER_FILLING_FOK
    assert mock_mt5.order_send.call_count == 1
    first_req = mock_mt5.order_send.call_args.args[0]
    assert first_req["type_filling"] == mock_mt5.ORDER_FILLING_FOK


def test_place_pending_order_accepts_done_partial_without_retry(mock_mt5):
    mock_mt5.ORDER_FILLING_IOC = 1
    mock_mt5.ORDER_FILLING_FOK = 0
    mock_mt5.ORDER_FILLING_RETURN = 2
    mock_mt5.ORDER_TIME_GTC = 0
    mock_mt5.TRADE_RETCODE_DONE_PARTIAL = 10010
    mock_mt5.order_send.return_value = MagicMock(
        retcode=10010,
        deal=0,
        order=456,
        volume=0.05,
        price=1.04000,
        bid=1.05000,
        ask=1.05010,
        comment="partial fill",
        request_id=802,
    )

    res = _place_pending_order(
        symbol="EURUSD",
        volume=0.1,
        order_type="BUY_LIMIT",
        price=1.04000,
    )

    assert "error" not in res
    assert res["success"] is True
    assert res["retcode"] == 10010
    assert len(res["fill_mode_attempts"]) == 1
    assert res["type_filling_used"] == mock_mt5.ORDER_FILLING_IOC
    assert mock_mt5.order_send.call_count == 1


def test_place_market_order_submits_protection_atomically(mock_mt5):
    mock_mt5.TRADE_ACTION_SLTP = 6
    position = SimpleNamespace(symbol="EURUSD", sl=0.0, tp=0.0, type=0, magic=24680)

    with patch(
        "src.mtdata.core.trading.orders._resolve_open_position",
        return_value=(position, 456, {}),
    ):
        res = _place_market_order(
            symbol="EURUSD",
            volume=0.1,
            order_type="BUY",
            stop_loss=1.04000,
            take_profit=1.06000,
        )

    assert "error" not in res
    assert mock_mt5.order_send.call_count == 1
    request = mock_mt5.order_send.call_args.args[0]
    assert request["action"] == mock_mt5.TRADE_ACTION_DEAL
    assert math.isclose(request["sl"], 1.04000)
    assert math.isclose(request["tp"], 1.06000)
    assert "fallback_used" not in res["sl_tp_result"]


def test_place_market_order_records_unmatched_position_resolution_without_retry(mock_mt5):
    mock_mt5.TRADE_ACTION_SLTP = 6
    unresolved = (None, None, {"method": "positions_get", "matched": False})

    with patch(
        "src.mtdata.core.trading.orders._resolve_open_position",
        return_value=unresolved,
    ) as mock_resolve, patch("mtdata.core.trading.orders._stdlib_time.sleep") as mock_sleep:
        res = _place_market_order(
            symbol="EURUSD",
            volume=0.1,
            order_type="BUY",
            stop_loss=1.04000,
            take_profit=1.06000,
        )

    assert "error" not in res
    assert res["sl_tp_result"]["status"] == "applied"
    assert res["position_ticket"] is None
    assert res["position_ticket_resolution"]["matched"] is False
    assert mock_resolve.call_count == 1
    mock_sleep.assert_not_called()


def test_place_market_order_returns_invalid_comment_error_without_fallback(mock_mt5):
    mock_mt5.TRADE_ACTION_SLTP = 6
    mock_mt5.last_error.side_effect = [
        (-2, 'Invalid "comment" argument'),
    ]
    mock_mt5.order_send.side_effect = [
        MagicMock(
            retcode=10013,
            deal=0,
            order=0,
            volume=0.1,
            price=1.05010,
            bid=1.05000,
            ask=1.05010,
            comment='Invalid "comment" argument',
            request_id=789,
        ),
    ]

    res = _place_market_order(
        symbol="EURUSD",
        volume=0.1,
        order_type="BUY",
        stop_loss=1.04000,
        take_profit=1.06000,
        comment="Breakout scalp comment that will be truncated anyway",
    )

    assert "broker rejected the comment field" in res["error"]
    assert mock_mt5.order_send.call_count == 1
    assert "comment_fallback" not in res


def test_place_market_order_does_not_attempt_sltp_follow_up(mock_mt5):
    mock_mt5.TRADE_ACTION_SLTP = 6
    position = SimpleNamespace(symbol="EURUSD", sl=0.0, tp=0.0, type=0, magic=24680)

    with patch(
        "src.mtdata.core.trading.orders._resolve_open_position",
        return_value=(position, 456, {}),
    ), patch(
        "src.mtdata.core.trading.orders._stdlib_time.sleep",
    ) as sleep_mock:
        res = _place_market_order(
            symbol="EURUSD",
            volume=0.1,
            order_type="BUY",
            stop_loss=1.04000,
            take_profit=1.06000,
        )

    assert "error" not in res
    assert res["sl_tp_result"]["status"] == "applied"
    assert res["sl_tp_result"]["attempts"] == 1
    assert mock_mt5.order_send.call_count == 1
    sleep_mock.assert_not_called()


def test_place_market_order_does_not_verify_protection_with_follow_up(mock_mt5, caplog):
    mock_mt5.TRADE_ACTION_SLTP = 6
    position = SimpleNamespace(symbol="EURUSD", sl=0.0, tp=0.0, type=0, magic=24680)
    mock_mt5.positions_get.side_effect = RuntimeError("readback lost")

    with patch(
        "src.mtdata.core.trading.orders._resolve_open_position",
        return_value=(position, 456, {}),
    ):
        res = _place_market_order(
            symbol="EURUSD",
            volume=0.1,
            order_type="BUY",
            stop_loss=1.04000,
            take_profit=1.06000,
        )

    assert "error" not in res
    assert res["sl_tp_result"]["status"] == "applied"
    assert "verification_failed" not in res["sl_tp_result"]
    assert not any("verification readback failed" in str(w) for w in res.get("warnings", []))
    assert not any("SL/TP verification failed for ticket 456" in record.message for record in caplog.records)


def test_modify_position_preserves_existing_magic(mock_mt5):
    mock_mt5.TRADE_ACTION_SLTP = 6
    position = SimpleNamespace(symbol="EURUSD", sl=1.03000, tp=1.05000, type=0, magic=98765)

    with patch(
        "src.mtdata.core.trading.execution._resolve_open_position",
        return_value=(position, 456, {}),
    ):
        res = _modify_position(
            ticket=456,
            stop_loss=1.04000,
            take_profit=1.06000,
        )

    assert "error" not in res
    req = mock_mt5.order_send.call_args[0][0]
    assert req["magic"] == 98765


def test_modify_position_treats_exact_matching_protection_as_no_change(mock_mt5):
    mock_mt5.TRADE_ACTION_SLTP = 6
    mock_mt5.TRADE_RETCODE_NO_CHANGES = 10025
    position = SimpleNamespace(symbol="EURUSD", sl=1.04000, tp=1.06000, type=0, magic=98765)

    with patch(
        "src.mtdata.core.trading.execution._resolve_open_position",
        return_value=(position, 456, {}),
    ):
        res = _modify_position(
            ticket=456,
            stop_loss=1.04000,
            take_profit=1.06000,
        )

    assert res.get("success") is True
    assert res.get("no_change") is True
    assert res.get("retcode") == 10025
    mock_mt5.order_send.assert_not_called()


def test_modify_position_applies_one_point_protection_change(mock_mt5):
    mock_mt5.TRADE_ACTION_SLTP = 6
    position = SimpleNamespace(symbol="EURUSD", sl=1.04000, tp=1.06000, type=0, magic=98765)

    with patch(
        "src.mtdata.core.trading.execution._resolve_open_position",
        return_value=(position, 456, {}),
    ):
        res = _modify_position(
            ticket=456,
            stop_loss=1.04001,
            take_profit=1.06000,
        )

    assert res.get("success") is True
    assert res.get("no_change") is not True
    req = mock_mt5.order_send.call_args[0][0]
    assert req["position"] == 456
    assert math.isclose(req["sl"], 1.04001)
    assert math.isclose(req["tp"], 1.06000)


def test_modify_position_retries_without_comment_when_comment_is_invalid(mock_mt5):
    mock_mt5.TRADE_ACTION_SLTP = 6
    position = SimpleNamespace(symbol="EURUSD", sl=1.03000, tp=1.05000, type=0, magic=98765)
    mock_mt5.last_error.side_effect = [
        (-2, 'Invalid "comment" argument'),
        (-2, 'Invalid "comment" argument'),
        (1, "Success"),
    ]
    mock_mt5.order_send.side_effect = [
        None,
        None,
        MagicMock(
            retcode=10009,
            deal=0,
            order=456,
            comment="",
            request_id=790,
        ),
    ]

    with patch(
        "src.mtdata.core.trading.execution._resolve_open_position",
        return_value=(position, 456, {}),
    ):
        res = _modify_position(
            ticket=456,
            stop_loss=1.04000,
            take_profit=1.06000,
            comment="Breakout scalp comment that will be truncated anyway",
        )

    assert res.get("success") is True
    assert res.get("comment_fallback", {}).get("used") is True
    assert res.get("comment_fallback", {}).get("strategy") == "none"
    final_req = mock_mt5.order_send.call_args_list[-1].args[0]
    assert "comment" not in final_req


def test_modify_pending_order_preserves_existing_magic(mock_mt5):
    mock_mt5.TRADE_ACTION_MODIFY = 7
    mock_mt5.ORDER_TIME_GTC = 0
    mock_mt5.orders_get.return_value = [
        SimpleNamespace(
            ticket=123,
            symbol="EURUSD",
            volume=0.1,
            type=mock_mt5.ORDER_TYPE_BUY_LIMIT,
            price_open=1.04000,
            sl=1.03000,
            tp=1.06000,
            magic=54321,
            type_time=0,
            time_expiration=0,
        )
    ]

    res = _modify_pending_order(ticket=123, price=1.04000)

    assert "error" not in res
    req = mock_mt5.order_send.call_args[0][0]
    assert req["magic"] == 54321


def test_modify_pending_order_retries_without_comment_when_comment_is_invalid(mock_mt5):
    mock_mt5.TRADE_ACTION_MODIFY = 7
    mock_mt5.ORDER_TIME_GTC = 0
    mock_mt5.last_error.side_effect = [
        (-2, 'Invalid "comment" argument'),
        (-2, 'Invalid "comment" argument'),
        (1, "Success"),
    ]
    mock_mt5.orders_get.return_value = [
        SimpleNamespace(
            ticket=123,
            symbol="EURUSD",
            volume=0.1,
            type=mock_mt5.ORDER_TYPE_BUY_LIMIT,
            price_open=1.04000,
            sl=1.03000,
            tp=1.06000,
            magic=54321,
            type_time=0,
            time_expiration=0,
        )
    ]
    mock_mt5.order_send.side_effect = [
        MagicMock(retcode=10013, comment='Invalid "comment" argument', request_id=790),
        MagicMock(retcode=10013, comment='Invalid "comment" argument', request_id=791),
        MagicMock(retcode=10009, deal=0, order=123, comment="", request_id=792),
    ]

    res = _modify_pending_order(
        ticket=123,
        price=1.04000,
        comment="Breakout scalp comment that will be truncated anyway",
    )

    assert res.get("success") is True
    assert res.get("comment_fallback", {}).get("used") is True
    assert res.get("comment_fallback", {}).get("strategy") == "none"
    assert any("pending order was retried" in str(w) for w in res.get("warnings", []))
    final_req = mock_mt5.order_send.call_args_list[-1].args[0]
    assert "comment" not in final_req


def test_close_positions_preserves_existing_magic(mock_mt5):
    mock_mt5.ORDER_FILLING_IOC = 1
    mock_mt5.ORDER_FILLING_FOK = 0
    mock_mt5.ORDER_FILLING_RETURN = 2
    mock_mt5.ORDER_TIME_GTC = 0
    mock_mt5.positions_get.return_value = [
        SimpleNamespace(ticket=123, symbol="EURUSD", volume=0.1, type=0, profit=10.0, magic=67890)
    ]

    res = _close_positions(ticket=123)

    assert "error" not in res
    req = mock_mt5.order_send.call_args[0][0]
    assert req["magic"] == 67890


def test_close_positions_uses_history_deal_profit_when_result_profit_is_missing(mock_mt5):
    mock_mt5.ORDER_FILLING_IOC = 1
    mock_mt5.ORDER_FILLING_FOK = 0
    mock_mt5.ORDER_FILLING_RETURN = 2
    mock_mt5.ORDER_TIME_GTC = 0
    mock_mt5.positions_get.return_value = [
        SimpleNamespace(
            ticket=123,
            symbol="EURUSD",
            volume=0.1,
            type=0,
            profit=10.0,
            magic=67890,
            price_open=1.04000,
            time=1700000000,
        )
    ]
    mock_mt5.order_send.return_value = SimpleNamespace(
        retcode=10009,
        deal=987,
        order=456,
        volume=0.1,
        price=1.05010,
        bid=1.05000,
        ask=1.05010,
        comment="",
        profit=None,
    )
    mock_mt5.history_deals_get.return_value = [
        SimpleNamespace(ticket=111, order=111, position_id=123, profit=1.0, time=1700000100),
        SimpleNamespace(ticket=987, order=456, position_id=123, profit=42.5, time=1700000200),
    ]

    res = _close_positions(ticket=123)

    assert "error" not in res
    assert res["pnl"] == pytest.approx(42.5)
    assert res["pnl"] != pytest.approx(10.0)
    mock_mt5.history_deals_get.assert_called_once()


def test_close_positions_converts_history_lookup_window_to_utc(mock_mt5):
    mock_mt5.positions_get.return_value = [
        SimpleNamespace(
            ticket=123,
            symbol="EURUSD",
            volume=0.1,
            type=0,
            profit=10.0,
            magic=67890,
            price_open=1.04000,
            time=1700000000,
        )
    ]
    mock_mt5.order_send.return_value = SimpleNamespace(
        retcode=10009,
        deal=987,
        order=456,
        volume=0.1,
        price=1.05010,
        bid=1.05000,
        ask=1.05010,
        comment="",
        profit=None,
    )
    mock_mt5.history_deals_get.return_value = [
        SimpleNamespace(ticket=987, order=456, position_id=123, profit=42.5, time=1700000200),
    ]
    utc_from = object()
    utc_to = object()

    with patch(
        "src.mtdata.core.trading.execution._to_utc_history_query_dt",
        side_effect=[utc_from, utc_to],
    ) as to_utc:
        res = _close_positions(ticket=123)

    assert "error" not in res
    assert res["pnl"] == pytest.approx(42.5)
    assert to_utc.call_count == 2
    mock_mt5.history_deals_get.assert_called_once_with(utc_from, utc_to)


def test_close_positions_retries_without_comment_when_result_retcode_rejects_comment(mock_mt5):
    mock_mt5.ORDER_FILLING_IOC = 1
    mock_mt5.ORDER_FILLING_FOK = 0
    mock_mt5.ORDER_FILLING_RETURN = 2
    mock_mt5.ORDER_TIME_GTC = 0
    mock_mt5.last_error.side_effect = [
        (-2, 'Invalid "comment" argument'),
        (-2, 'Invalid "comment" argument'),
        (1, "Success"),
    ]
    mock_mt5.positions_get.return_value = [
        SimpleNamespace(ticket=123, symbol="EURUSD", volume=0.1, type=0, profit=10.0, magic=67890)
    ]
    mock_mt5.order_send.side_effect = [
        MagicMock(
            retcode=10013,
            deal=0,
            order=0,
            volume=0.1,
            price=1.05010,
            bid=1.05000,
            ask=1.05010,
            comment='Invalid "comment" argument',
        ),
        MagicMock(
            retcode=10013,
            deal=0,
            order=0,
            volume=0.1,
            price=1.05010,
            bid=1.05000,
            ask=1.05010,
            comment='Invalid "comment" argument',
        ),
        MagicMock(
            retcode=10009,
            deal=123,
            order=456,
            volume=0.1,
            price=1.05010,
            bid=1.05000,
            ask=1.05010,
            comment="",
        ),
    ]

    res = _close_positions(ticket=123, comment="Breakout scalp comment that will be truncated anyway")

    assert res.get("retcode") == 10009
    assert res.get("comment_fallback", {}).get("used") is True
    assert res.get("comment_fallback", {}).get("strategy") == "none"
    assert any("close order was retried" in str(w) for w in res.get("warnings", []))
    final_req = mock_mt5.order_send.call_args_list[-1].args[0]
    assert "comment" not in final_req


def test_close_positions_prefers_symbol_fill_mode(mock_mt5):
    mock_mt5.ORDER_FILLING_IOC = 1
    mock_mt5.ORDER_FILLING_FOK = 0
    mock_mt5.ORDER_FILLING_RETURN = 2
    mock_mt5.SYMBOL_FILLING_FOK = 1
    mock_mt5.SYMBOL_FILLING_IOC = 2
    mock_mt5.SYMBOL_FILLING_RETURN = 4
    mock_mt5.ORDER_TIME_GTC = 0
    mock_mt5.symbol_info.return_value.filling_mode = mock_mt5.SYMBOL_FILLING_FOK
    mock_mt5.positions_get.return_value = [
        SimpleNamespace(ticket=123, symbol="EURUSD", volume=0.1, type=0, profit=10.0, magic=67890)
    ]

    res = _close_positions(symbol="EURUSD")

    assert res["closed_count"] == 1
    first_req = mock_mt5.order_send.call_args.args[0]
    assert first_req["type_filling"] == mock_mt5.ORDER_FILLING_FOK


def test_close_positions_retries_default_fill_mode_when_fok_constant_missing(mock_mt5):
    mock_mt5.ORDER_FILLING_IOC = 1
    mock_mt5.ORDER_FILLING_FOK = None
    mock_mt5.ORDER_FILLING_RETURN = 2
    mock_mt5.ORDER_TIME_GTC = 0
    mock_mt5.positions_get.return_value = [
        SimpleNamespace(ticket=123, symbol="EURUSD", volume=0.1, type=0, profit=10.0, magic=67890)
    ]
    mock_mt5.order_send.side_effect = [
        MagicMock(
            retcode=10030,
            deal=0,
            order=0,
            volume=0.1,
            price=1.05010,
            bid=1.05000,
            ask=1.05010,
            comment="IOC rejected",
        ),
        MagicMock(
            retcode=10009,
            deal=123,
            order=456,
            volume=0.1,
            price=1.05010,
            bid=1.05000,
            ask=1.05010,
            comment="",
        ),
    ]

    res = _close_positions(symbol="EURUSD")

    assert res["closed_count"] == 1
    first_req = mock_mt5.order_send.call_args_list[0].args[0]
    second_req = mock_mt5.order_send.call_args_list[1].args[0]
    assert first_req["type_filling"] == 1
    assert second_req["type_filling"] == 0


def test_close_positions_counts_done_partial_as_success(mock_mt5):
    mock_mt5.ORDER_FILLING_IOC = 1
    mock_mt5.ORDER_FILLING_FOK = 0
    mock_mt5.ORDER_FILLING_RETURN = 2
    mock_mt5.ORDER_TIME_GTC = 0
    mock_mt5.TRADE_RETCODE_DONE_PARTIAL = 10010
    mock_mt5.positions_get.return_value = [
        SimpleNamespace(ticket=123, symbol="EURUSD", volume=0.1, type=0, profit=10.0, magic=67890)
    ]
    mock_mt5.order_send.return_value = MagicMock(
        retcode=10010,
        deal=123,
        order=456,
        volume=0.1,
        price=1.05010,
        bid=1.05000,
        ask=1.05010,
        comment="",
    )

    res = _close_positions(symbol="EURUSD")

    assert res["closed_count"] == 1


def test_close_positions_preserves_partial_results_when_later_position_raises(mock_mt5):
    mock_mt5.ORDER_FILLING_IOC = 1
    mock_mt5.ORDER_FILLING_FOK = 0
    mock_mt5.ORDER_FILLING_RETURN = 2
    mock_mt5.ORDER_TIME_GTC = 0
    position_one = SimpleNamespace(
        ticket=123,
        symbol="EURUSD",
        volume=0.1,
        type=0,
        profit=10.0,
        magic=67890,
        price_open=1.04000,
        time=0,
    )
    position_two = SimpleNamespace(
        ticket=456,
        symbol="GBPUSD",
        volume=0.1,
        type=0,
        profit=5.0,
        magic=67890,
        price_open=1.24000,
        time=0,
    )

    def _positions_get(*args, **kwargs):
        ticket = kwargs.get("ticket")
        if ticket == 123:
            return [position_one]
        if ticket == 456:
            return [position_two]
        return [position_one, position_two]

    def _symbol_info_tick(symbol):
        if symbol == "EURUSD":
            return MagicMock(bid=1.05000, ask=1.05010)
        raise RuntimeError("tick feed exploded")

    mock_mt5.positions_get.side_effect = _positions_get
    mock_mt5.symbol_info_tick.side_effect = _symbol_info_tick
    mock_mt5.order_send.return_value = MagicMock(
        retcode=10009,
        deal=123,
        order=456,
        volume=0.1,
        price=1.05010,
        bid=1.05000,
        ask=1.05010,
        comment="",
    )

    res = _close_positions()

    assert res["closed_count"] == 1
    assert res["attempted_count"] == 2
    assert res["partial_failure"] is True
    assert res["results"][0]["ticket"] == 123
    assert res["results"][0]["retcode"] == 10009
    assert res["results"][1]["ticket"] == 456
    assert res["results"][1]["error"] == "tick feed exploded"


def test_close_positions_ticket_requires_exact_live_ticket_match(mock_mt5):
    mock_mt5.ORDER_FILLING_IOC = 1
    mock_mt5.ORDER_FILLING_FOK = 0
    mock_mt5.ORDER_FILLING_RETURN = 2
    mock_mt5.ORDER_TIME_GTC = 0
    position = SimpleNamespace(
        ticket=999,
        order=123,
        symbol="EURUSD",
        volume=0.1,
        type=0,
        profit=10.0,
        magic=67890,
        price_open=1.04000,
        time=0,
    )

    def _positions_get(*args, **kwargs):
        if kwargs.get("ticket") == 123:
            return []
        return [position]

    mock_mt5.positions_get.side_effect = _positions_get

    res = _close_positions(ticket=123)

    assert res["error"] == "Position 123 not found"
    assert res["checked_scopes"] == ["positions"]
    assert res["ticket_resolution"]["exact_ticket_required"] is True
    mock_mt5.order_send.assert_not_called()


def test_close_positions_fetches_tick_once_before_fill_mode_retries(mock_mt5):
    mock_mt5.ORDER_FILLING_IOC = 1
    mock_mt5.ORDER_FILLING_FOK = 0
    mock_mt5.ORDER_FILLING_RETURN = 2
    mock_mt5.ORDER_TIME_GTC = 0
    mock_mt5.positions_get.return_value = [
        SimpleNamespace(ticket=123, symbol="EURUSD", volume=0.1, type=0, profit=10.0, magic=67890)
    ]
    mock_mt5.order_send.side_effect = [
        MagicMock(
            retcode=10030,
            deal=0,
            order=0,
            volume=0.1,
            price=1.05010,
            bid=1.05000,
            ask=1.05010,
            comment="IOC rejected",
        ),
        MagicMock(
            retcode=10009,
            deal=123,
            order=456,
            volume=0.1,
            price=1.05010,
            bid=1.05000,
            ask=1.05010,
            comment="",
        ),
    ]

    res = _close_positions(symbol="EURUSD")

    assert res["closed_count"] == 1
    assert mock_mt5.symbol_info_tick.call_count == 1


def test_close_positions_refreshes_live_volume_before_partial_close_validation(mock_mt5):
    mock_mt5.ORDER_FILLING_IOC = 1
    mock_mt5.ORDER_FILLING_FOK = 0
    mock_mt5.ORDER_FILLING_RETURN = 2
    mock_mt5.ORDER_TIME_GTC = 0

    stale_position = SimpleNamespace(ticket=123, symbol="EURUSD", volume=0.2, type=0, profit=10.0, magic=67890)
    fresh_position = SimpleNamespace(ticket=123, symbol="EURUSD", volume=0.1, type=0, profit=10.0, magic=67890)

    def _positions_get(*args, **kwargs):
        if kwargs.get("ticket") == 123:
            return [fresh_position]
        if kwargs.get("symbol") == "EURUSD":
            return [stale_position]
        return [stale_position]

    mock_mt5.positions_get.side_effect = _positions_get

    res = _close_positions(symbol="EURUSD", volume=0.15)

    assert res["closed_count"] == 0
    assert res["results"][0]["ticket"] == 123
    assert "volume must be <= open position volume (0.1)" in res["results"][0]["error"]
    assert res["results"][0]["position_volume"] == 0.1
    assert mock_mt5.order_send.call_count == 0


# ---------------------------------------------------------------------------
# _execute_single_close unit tests
# ---------------------------------------------------------------------------

from src.mtdata.core.trading.execution import _execute_single_close


def test_execute_single_close_success(mock_mt5):
    """Successful close returns result with PnL metadata."""
    mock_mt5.ORDER_FILLING_IOC = 1
    mock_mt5.ORDER_TIME_GTC = 0
    mock_mt5.TRADE_RETCODE_DONE = 10009

    position = SimpleNamespace(
        ticket=42, symbol="EURUSD", volume=0.1, type=0,
        price_open=1.04000, time=1700000000, magic=100,
    )
    mock_mt5.order_send.return_value = SimpleNamespace(
        retcode=10009, deal=500, order=600, volume=0.1,
        price=1.05000, comment="close", profit=10.0,
    )
    mock_mt5.symbol_info_tick.return_value = SimpleNamespace(bid=1.05000, ask=1.05010)

    from src.mtdata.core.trading.gateway import create_trading_gateway
    gw = create_trading_gateway(
        adapter=mock_mt5, include_retcode_name=True,
        ensure_connection_impl=lambda: None,
    )
    fill_modes = [getattr(mock_mt5, "ORDER_FILLING_IOC", 1)]

    result = _execute_single_close(
        gw, position,
        requested_volume=None, position_volume_before=0.1,
        remaining_volume_estimate=None, deviation=20,
        comment="test", fill_modes=fill_modes,
    )
    assert result["ticket"] == 42
    assert result["retcode"] == 10009
    assert result.get("error") is None
    assert result["pnl"] == 10.0


def test_execute_single_close_retries_same_fill_mode_after_price_change(mock_mt5):
    mock_mt5.ORDER_FILLING_FOK = 0
    mock_mt5.ORDER_TIME_GTC = 0
    mock_mt5.TRADE_RETCODE_DONE = 10009
    mock_mt5.TRADE_RETCODE_PRICE_CHANGED = 10020

    position = SimpleNamespace(
        ticket=42, symbol="EURUSD", volume=0.1, type=0,
        price_open=1.04000, time=1700000000, magic=100,
    )
    mock_mt5.order_send.side_effect = [
        SimpleNamespace(
            retcode=10020, deal=0, order=0, volume=0.1,
            price=1.05000, comment="price changed", profit=None,
        ),
        SimpleNamespace(
            retcode=10009, deal=500, order=600, volume=0.1,
            price=1.04990, comment="close", profit=9.0,
        ),
    ]
    mock_mt5.symbol_info_tick.side_effect = [
        SimpleNamespace(bid=1.05000, ask=1.05010),
        SimpleNamespace(bid=1.04990, ask=1.05000),
    ]

    from src.mtdata.core.trading.gateway import create_trading_gateway
    gw = create_trading_gateway(
        adapter=mock_mt5, include_retcode_name=True,
        ensure_connection_impl=lambda: None,
    )

    result = _execute_single_close(
        gw, position,
        requested_volume=None, position_volume_before=0.1,
        remaining_volume_estimate=None, deviation=20,
        comment="test", fill_modes=[mock_mt5.ORDER_FILLING_FOK],
    )

    assert result["ticket"] == 42
    assert result["retcode"] == 10009
    assert mock_mt5.order_send.call_count == 2
    first_req = mock_mt5.order_send.call_args_list[0].args[0]
    second_req = mock_mt5.order_send.call_args_list[1].args[0]
    assert first_req["type_filling"] == mock_mt5.ORDER_FILLING_FOK
    assert second_req["type_filling"] == mock_mt5.ORDER_FILLING_FOK
    assert first_req["price"] == pytest.approx(1.05000)
    assert second_req["price"] == pytest.approx(1.04990)


def test_execute_single_close_no_tick(mock_mt5):
    """Returns error when tick data unavailable."""
    mock_mt5.ORDER_FILLING_IOC = 1
    mock_mt5.ORDER_TIME_GTC = 0

    position = SimpleNamespace(
        ticket=42, symbol="XYZUSD", volume=0.1, type=0,
        price_open=1.0, time=1700000000, magic=100,
    )
    mock_mt5.symbol_info_tick.return_value = None
    mock_mt5.last_error.return_value = (10006, "No tick")

    from src.mtdata.core.trading.gateway import create_trading_gateway
    gw = create_trading_gateway(
        adapter=mock_mt5, include_retcode_name=True,
        ensure_connection_impl=lambda: None,
    )
    fill_modes = [1]

    result = _execute_single_close(
        gw, position,
        requested_volume=None, position_volume_before=0.1,
        remaining_volume_estimate=None, deviation=20,
        comment=None, fill_modes=fill_modes,
    )
    assert result["ticket"] == 42
    assert "tick data" in result["error"].lower()


def test_execute_single_close_unknown_side(mock_mt5):
    """Returns error when position side cannot be determined."""
    mock_mt5.ORDER_FILLING_IOC = 1
    mock_mt5.ORDER_TIME_GTC = 0

    # Position with no type attribute at all
    position = SimpleNamespace(ticket=42, symbol="EURUSD", volume=0.1, magic=100)

    from src.mtdata.core.trading.gateway import create_trading_gateway
    gw = create_trading_gateway(
        adapter=mock_mt5, include_retcode_name=True,
        ensure_connection_impl=lambda: None,
    )
    fill_modes = [1]

    result = _execute_single_close(
        gw, position,
        requested_volume=None, position_volume_before=0.1,
        remaining_volume_estimate=None, deviation=20,
        comment=None, fill_modes=fill_modes,
    )
    assert result["ticket"] == 42
    assert "side" in result["error"].lower()


# ---------------------------------------------------------------------------
# Abort policy tests
# ---------------------------------------------------------------------------

def test_close_positions_aborts_after_consecutive_failures(mock_mt5):
    """Bulk close skips remaining positions after 3 consecutive failures."""
    mock_mt5.ORDER_FILLING_IOC = 1
    mock_mt5.ORDER_FILLING_FOK = 0
    mock_mt5.ORDER_FILLING_RETURN = 2
    mock_mt5.ORDER_TIME_GTC = 0

    positions = [
        SimpleNamespace(ticket=i, symbol="EURUSD", volume=0.1, type=0, profit=1.0, magic=0)
        for i in range(1, 6)
    ]
    mock_mt5.positions_get.return_value = positions

    # All order_sends fail
    mock_mt5.order_send.return_value = None
    mock_mt5.last_error.return_value = (10004, "Connection lost")
    mock_mt5.symbol_info_tick.return_value = SimpleNamespace(bid=1.05, ask=1.0501)

    # positions_get with ticket= returns the matching position
    def _positions_get(*args, **kwargs):
        t = kwargs.get("ticket")
        if t is not None:
            return [p for p in positions if p.ticket == t] or None
        return positions
    mock_mt5.positions_get.side_effect = _positions_get

    res = _close_positions(symbol="EURUSD")

    results = res.get("results", [res])
    aborted = [r for r in results if r.get("aborted")]
    # Positions 4 and 5 should be aborted (after 3 consecutive failures on 1,2,3)
    assert len(aborted) == 2
    assert all("consecutive" in r["error"].lower() for r in aborted)


def test_close_positions_resets_abort_counter_on_success(mock_mt5):
    """A successful close resets the consecutive failure counter."""
    mock_mt5.ORDER_FILLING_IOC = 1
    mock_mt5.ORDER_FILLING_FOK = 0
    mock_mt5.ORDER_FILLING_RETURN = 2
    mock_mt5.ORDER_TIME_GTC = 0
    mock_mt5.TRADE_RETCODE_DONE = 10009

    positions = [
        SimpleNamespace(ticket=i, symbol="EURUSD", volume=0.1, type=0, profit=1.0, magic=0, price_open=1.04, time=1700000000)
        for i in range(1, 7)
    ]

    # Position 3 succeeds, all others fail
    def _order_send(request):
        if request.get("position") == 3:
            return SimpleNamespace(
                retcode=10009, deal=500, order=600, volume=0.1,
                price=1.05, comment="ok", profit=5.0,
            )
        return None

    mock_mt5.order_send.side_effect = _order_send
    mock_mt5.last_error.return_value = (10004, "err")
    mock_mt5.symbol_info_tick.return_value = SimpleNamespace(bid=1.05, ask=1.0501)

    def _positions_get(*args, **kwargs):
        t = kwargs.get("ticket")
        if t is not None:
            return [p for p in positions if p.ticket == t] or None
        return positions
    mock_mt5.positions_get.side_effect = _positions_get

    res = _close_positions(symbol="EURUSD")

    results = res.get("results", [res])
    aborted = [r for r in results if r.get("aborted")]
    # Pos 1 fail, 2 fail, 3 success (resets), 4 fail, 5 fail, 6 fail → abort on remaining
    # After pos 6, consecutive=3 but loop ends. No positions after 6.
    # Actually: check happens at top of loop, so
    # Pos 1: consec=0 → execute → fail → consec=1
    # Pos 2: consec=1 → execute → fail → consec=2
    # Pos 3: consec=2 → execute → success → consec=0
    # Pos 4: consec=0 → execute → fail → consec=1
    # Pos 5: consec=1 → execute → fail → consec=2
    # Pos 6: consec=2 → execute → fail → consec=3
    # No aborts since threshold is never reached at loop top.
    assert len(aborted) == 0
    # But the counter was reset — verify pos 6 was attempted (not aborted)
    assert results[-1]["ticket"] == 6
    assert results[-1].get("aborted") is None


# ---------------------------------------------------------------------------
# _sort_close_positions and close_priority integration tests
# ---------------------------------------------------------------------------

from src.mtdata.core.trading.execution import _sort_close_positions


def test_sort_close_positions_loss_first():
    """loss_first sorts by ascending profit (most negative first)."""
    positions = [
        SimpleNamespace(ticket=1, profit=50.0, volume=0.1),
        SimpleNamespace(ticket=2, profit=-100.0, volume=0.2),
        SimpleNamespace(ticket=3, profit=-20.0, volume=0.3),
    ]
    result = _sort_close_positions(positions, "loss_first")
    assert [p.ticket for p in result] == [2, 3, 1]


def test_sort_close_positions_profit_first():
    """profit_first sorts by descending profit (most positive first)."""
    positions = [
        SimpleNamespace(ticket=1, profit=-10.0, volume=0.1),
        SimpleNamespace(ticket=2, profit=100.0, volume=0.2),
        SimpleNamespace(ticket=3, profit=50.0, volume=0.3),
    ]
    result = _sort_close_positions(positions, "profit_first")
    assert [p.ticket for p in result] == [2, 3, 1]


def test_sort_close_positions_largest_first():
    """largest_first sorts by descending volume."""
    positions = [
        SimpleNamespace(ticket=1, profit=10.0, volume=0.01),
        SimpleNamespace(ticket=2, profit=20.0, volume=1.0),
        SimpleNamespace(ticket=3, profit=-5.0, volume=0.5),
    ]
    result = _sort_close_positions(positions, "largest_first")
    assert [p.ticket for p in result] == [2, 3, 1]


def test_sort_close_positions_none_preserves_order():
    """None priority preserves discovery order."""
    positions = [
        SimpleNamespace(ticket=3, profit=10.0, volume=0.1),
        SimpleNamespace(ticket=1, profit=-10.0, volume=0.5),
    ]
    result = _sort_close_positions(positions, None)
    assert [p.ticket for p in result] == [3, 1]


def test_sort_close_positions_missing_profit():
    """Handles positions with missing profit attribute gracefully."""
    positions = [
        SimpleNamespace(ticket=1, volume=0.1),
        SimpleNamespace(ticket=2, profit=-50.0, volume=0.2),
    ]
    result = _sort_close_positions(positions, "loss_first")
    # Missing profit defaults to 0.0, so -50 sorts first
    assert [p.ticket for p in result] == [2, 1]


def test_close_positions_loss_first_integration(mock_mt5):
    """close_priority=loss_first closes losers before winners."""
    mock_mt5.ORDER_FILLING_IOC = 1
    mock_mt5.ORDER_FILLING_FOK = 0
    mock_mt5.ORDER_FILLING_RETURN = 2
    mock_mt5.ORDER_TIME_GTC = 0
    mock_mt5.TRADE_RETCODE_DONE = 10009

    positions = [
        SimpleNamespace(ticket=1, symbol="EURUSD", volume=0.1, type=0, profit=100.0, magic=0, price_open=1.04, time=1700000000),
        SimpleNamespace(ticket=2, symbol="EURUSD", volume=0.1, type=0, profit=-50.0, magic=0, price_open=1.06, time=1700000000),
        SimpleNamespace(ticket=3, symbol="EURUSD", volume=0.1, type=0, profit=-200.0, magic=0, price_open=1.07, time=1700000000),
    ]

    mock_mt5.order_send.return_value = SimpleNamespace(
        retcode=10009, deal=500, order=600, volume=0.1,
        price=1.05, comment="ok", profit=5.0,
    )
    mock_mt5.symbol_info_tick.return_value = SimpleNamespace(bid=1.05, ask=1.0501)

    def _positions_get(*args, **kwargs):
        t = kwargs.get("ticket")
        if t is not None:
            return [p for p in positions if p.ticket == t] or None
        return positions
    mock_mt5.positions_get.side_effect = _positions_get

    res = _close_positions(symbol="EURUSD", close_priority="loss_first")

    assert res["close_priority"] == "loss_first"
    tickets_in_order = [r["ticket"] for r in res["results"]]
    # Should close biggest loser first: 3 (-200), then 2 (-50), then 1 (+100)
    assert tickets_in_order == [3, 2, 1]


def test_cancel_pending_counts_done_partial_as_success(mock_mt5):
    mock_mt5.TRADE_ACTION_REMOVE = 8
    mock_mt5.TRADE_RETCODE_DONE_PARTIAL = 10010
    mock_mt5.orders_get.return_value = [SimpleNamespace(ticket=123)]
    mock_mt5.order_send.return_value = MagicMock(
        retcode=10010,
        deal=0,
        order=123,
        comment="",
    )

    res = _cancel_pending(symbol="EURUSD")

    assert res["cancelled_count"] == 1


def test_cancel_pending_ticket_requires_exact_live_ticket_match(mock_mt5):
    mock_mt5.TRADE_ACTION_REMOVE = 8
    order = SimpleNamespace(ticket=999, position=123, symbol="EURUSD", magic=54321)

    def _orders_get(*args, **kwargs):
        if kwargs.get("ticket") == 123:
            return []
        return [order]

    mock_mt5.orders_get.side_effect = _orders_get

    res = _cancel_pending(ticket=123)

    assert res["error"] == "Pending order 123 not found"
    assert res["checked_scopes"] == ["pending_orders"]
    assert res["ticket_resolution"]["exact_ticket_required"] is True
    mock_mt5.order_send.assert_not_called()


def test_cancel_pending_preserves_existing_magic(mock_mt5):
    mock_mt5.TRADE_ACTION_REMOVE = 8
    mock_mt5.orders_get.return_value = [SimpleNamespace(ticket=123, magic=54321)]

    res = _cancel_pending(ticket=123)

    assert "error" not in res
    req = mock_mt5.order_send.call_args[0][0]
    assert req["magic"] == 54321


def test_cancel_pending_retries_without_comment_when_comment_is_invalid(mock_mt5):
    mock_mt5.TRADE_ACTION_REMOVE = 8
    mock_mt5.orders_get.return_value = [SimpleNamespace(ticket=123, magic=54321)]
    mock_mt5.last_error.side_effect = [
        (-2, 'Invalid "comment" argument'),
        (-2, 'Invalid "comment" argument'),
        (1, "Success"),
    ]
    mock_mt5.order_send.side_effect = [
        MagicMock(retcode=10013, deal=0, order=123, comment='Invalid "comment" argument'),
        MagicMock(retcode=10013, deal=0, order=123, comment='Invalid "comment" argument'),
        MagicMock(retcode=10009, deal=0, order=123, comment=""),
    ]

    res = _cancel_pending(
        ticket=123,
        comment="Breakout scalp comment that will be truncated anyway",
    )

    assert "error" not in res
    assert res.get("comment_fallback", {}).get("used") is True
    assert res.get("comment_fallback", {}).get("strategy") == "none"
    final_req = mock_mt5.order_send.call_args_list[-1].args[0]
    assert "comment" not in final_req


# ---------------------------------------------------------------------------
# _attach_post_fill_protection unit tests
# ---------------------------------------------------------------------------

from src.mtdata.core.trading.orders import _attach_post_fill_protection


def _make_protection_gateway(mock_mt5):
    """Build a gateway for _attach_post_fill_protection tests."""
    from src.mtdata.core.trading.gateway import create_trading_gateway as _cg

    return _cg(
        adapter=mock_mt5,
        include_trade_preflight=False,
        include_retcode_name=True,
        ensure_connection_impl=lambda: None,
    )


def test_attach_protection_not_requested(mock_mt5):
    gw = _make_protection_gateway(mock_mt5)
    symbol_info = mock_mt5.symbol_info.return_value

    outcome = _attach_post_fill_protection(
        gw,
        symbol="EURUSD",
        side="BUY",
        volume=0.1,
        position_ticket_candidates=[456],
        stop_loss=None,
        take_profit=None,
        symbol_info=symbol_info,
        comment=None,
        request_comment="MCP order",
    )

    assert outcome["sl_tp_result"]["status"] == "not_requested"
    assert "protection_status" not in outcome
    assert outcome["warnings"] == []


def test_attach_protection_success(mock_mt5):
    gw = _make_protection_gateway(mock_mt5)
    symbol_info = mock_mt5.symbol_info.return_value
    mock_mt5.TRADE_ACTION_SLTP = 6
    mock_mt5.positions_get.return_value = [
        SimpleNamespace(
            ticket=456, symbol="EURUSD", type=0, sl=1.04, tp=1.06,
            volume=0.1, magic=234000, time_update_msc=1000,
        )
    ]

    outcome = _attach_post_fill_protection(
        gw,
        symbol="EURUSD",
        side="BUY",
        volume=0.1,
        position_ticket_candidates=[456],
        stop_loss=1.04,
        take_profit=1.06,
        symbol_info=symbol_info,
        comment=None,
        request_comment="MCP order",
    )

    assert outcome["sl_tp_result"]["status"] == "applied"
    assert outcome["protection_status"] == "protected"
    assert outcome["position_ticket"] == 456
    assert outcome["warnings"] == []


def test_attach_protection_direct_sltp_failure_has_no_fallback(mock_mt5):
    """Direct SLTP failure is returned without a fallback modify call."""
    gw = _make_protection_gateway(mock_mt5)
    symbol_info = mock_mt5.symbol_info.return_value
    mock_mt5.TRADE_ACTION_SLTP = 6
    mock_mt5.positions_get.return_value = [
        SimpleNamespace(
            ticket=456, symbol="EURUSD", type=0, sl=0.0, tp=0.0,
            volume=0.1, magic=234000, time_update_msc=1000,
        )
    ]
    # Direct SLTP always fails
    failed_result = SimpleNamespace(retcode=10006, comment="Rejected")
    mock_mt5.order_send.return_value = failed_result

    outcome = _attach_post_fill_protection(
        gw,
        symbol="EURUSD",
        side="BUY",
        volume=0.1,
        position_ticket_candidates=[456],
        stop_loss=1.04,
        take_profit=1.06,
        symbol_info=symbol_info,
        comment=None,
        request_comment="MCP order",
    )

    assert outcome["sl_tp_result"]["status"] == "failed"
    assert outcome["protection_status"] == "unprotected_position"
    assert "fallback_used" not in outcome["sl_tp_result"]
    assert any("CRITICAL" in w for w in outcome["warnings"])


def test_attach_protection_failure_reports_unprotected(mock_mt5):
    """Direct SLTP failure reports the position as unprotected."""
    gw = _make_protection_gateway(mock_mt5)
    symbol_info = mock_mt5.symbol_info.return_value
    mock_mt5.TRADE_ACTION_SLTP = 6
    mock_mt5.positions_get.return_value = [
        SimpleNamespace(
            ticket=456, symbol="EURUSD", type=0, sl=0.0, tp=0.0,
            volume=0.1, magic=234000, time_update_msc=1000,
        )
    ]
    failed_result = SimpleNamespace(retcode=10006, comment="Rejected")
    mock_mt5.order_send.return_value = failed_result

    outcome = _attach_post_fill_protection(
        gw,
        symbol="EURUSD",
        side="BUY",
        volume=0.1,
        position_ticket_candidates=[456],
        stop_loss=1.04,
        take_profit=1.06,
        symbol_info=symbol_info,
        comment=None,
        request_comment="MCP order",
    )

    assert outcome["sl_tp_result"]["status"] == "failed"
    assert outcome["protection_status"] == "unprotected_position"
    assert any("CRITICAL" in w for w in outcome["warnings"])


def test_attach_protection_invalid_stops_fails_fast_without_fallback(mock_mt5):
    gw = _make_protection_gateway(mock_mt5)
    symbol_info = mock_mt5.symbol_info.return_value
    mock_mt5.TRADE_ACTION_SLTP = 6
    mock_mt5.TRADE_RETCODE_INVALID_STOPS = 10016
    mock_mt5.positions_get.return_value = [
        SimpleNamespace(
            ticket=456, symbol="EURUSD", type=0, sl=0.0, tp=0.0,
            volume=0.1, magic=234000, time_update_msc=1000,
        )
    ]
    mock_mt5.order_send.return_value = SimpleNamespace(retcode=10016, comment="Invalid stops")

    outcome = _attach_post_fill_protection(
        gw,
        symbol="EURUSD",
        side="BUY",
        volume=0.1,
        position_ticket_candidates=[456],
        stop_loss=1.04,
        take_profit=1.06,
        symbol_info=symbol_info,
        comment=None,
        request_comment="MCP order",
    )

    assert mock_mt5.order_send.call_count == 1
    assert outcome["sl_tp_result"]["status"] == "failed"
    assert outcome["sl_tp_result"]["attempts"] == 1
    assert outcome["sl_tp_result"]["last_retcode"] == 10016
    assert "fallback_used" not in outcome["sl_tp_result"]


def test_attach_protection_position_not_found(mock_mt5):
    """Position resolution never succeeds → failed with candidates info."""
    gw = _make_protection_gateway(mock_mt5)
    symbol_info = mock_mt5.symbol_info.return_value
    mock_mt5.TRADE_ACTION_SLTP = 6
    mock_mt5.positions_get.return_value = []

    outcome = _attach_post_fill_protection(
        gw,
        symbol="EURUSD",
        side="BUY",
        volume=0.1,
        position_ticket_candidates=[456, 123],
        stop_loss=1.04,
        take_profit=None,
        symbol_info=symbol_info,
        comment=None,
        request_comment="MCP order",
    )

    assert outcome["sl_tp_result"]["status"] == "failed"
    assert outcome["protection_status"] == "unprotected_position"
    assert outcome["position_ticket"] is None
    resolution = outcome["position_ticket_resolution"]
    assert resolution is not None
    assert resolution.get("matched") is False


# ---------------------------------------------------------------------------
# Tick freshness validator
# ---------------------------------------------------------------------------
import time as _time_module

from src.mtdata.core.trading.validation import (
    _tick_age_seconds,
    _validate_tick_freshness,
    _DEFAULT_TICK_MAX_AGE_SECONDS,
)


def test_tick_age_seconds_from_time_msc():
    """Prefers time_msc (millisecond epoch) when available."""
    now_ms = _time_module.time() * 1000.0
    tick = SimpleNamespace(time_msc=now_ms - 5000, time=0)
    age = _tick_age_seconds(tick)
    assert age is not None
    assert 4.5 <= age <= 6.0


def test_tick_age_seconds_from_time_seconds():
    """Falls back to time (seconds) when time_msc missing."""
    now_s = _time_module.time()
    tick = SimpleNamespace(time=int(now_s) - 10)
    age = _tick_age_seconds(tick)
    assert age is not None
    assert 9.0 <= age <= 12.0


def test_tick_age_seconds_no_timestamp():
    """Returns None when tick carries no usable timestamp."""
    tick = SimpleNamespace()
    assert _tick_age_seconds(tick) is None
    tick2 = SimpleNamespace(time_msc=None, time=None)
    assert _tick_age_seconds(tick2) is None


def test_validate_tick_freshness_fresh():
    """Fresh tick passes validation."""
    now_ms = _time_module.time() * 1000.0
    tick = SimpleNamespace(time_msc=now_ms - 1000)
    result = _validate_tick_freshness(tick, symbol="EURUSD")
    assert result is None


def test_validate_tick_freshness_stale():
    """Stale tick returns error dict with metadata."""
    now_ms = _time_module.time() * 1000.0
    tick = SimpleNamespace(time_msc=now_ms - 60_000)  # 60s old
    result = _validate_tick_freshness(tick, symbol="EURUSD")
    assert result is not None
    assert "stale" in result["error"].lower()
    assert result["tick_age_seconds"] > 50
    assert result["tick_max_age_seconds"] == _DEFAULT_TICK_MAX_AGE_SECONDS


def test_validate_tick_freshness_custom_threshold():
    """Custom threshold narrows acceptance window."""
    now_ms = _time_module.time() * 1000.0
    tick = SimpleNamespace(time_msc=now_ms - 3000)  # 3s old
    assert _validate_tick_freshness(tick, symbol="GOLD", max_age_seconds=2.0) is not None
    assert _validate_tick_freshness(tick, symbol="GOLD", max_age_seconds=5.0) is None


def test_validate_tick_freshness_no_timestamp():
    """Tick without timestamp passes (preserves existing null-tick semantics)."""
    tick = SimpleNamespace()
    assert _validate_tick_freshness(tick, symbol="EURUSD") is None

