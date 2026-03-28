import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import sys
import math

from src.mtdata.core.trading import (
    _cancel_pending,
    _close_positions,
    _modify_pending_order,
    _modify_position,
    _place_market_order,
    _place_pending_order,
)
from src.mtdata.core.trading_gateway import MT5TradingGateway
from src.mtdata.core.trading_gateway import create_trading_gateway as create_real_trading_gateway


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

    with patch("src.mtdata.core.trading_orders.create_trading_gateway", side_effect=_build_gateway), patch(
        "src.mtdata.core.trading_execution.create_trading_gateway", side_effect=_build_gateway
    ):
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
    
    # Check that order_send was called twice (once for deal, once for modifying SL/TP since TRADE_ACTION_DEAL doesn't take SL/TP natively)
    assert mock_mt5.order_send.call_count == 2
    
    deal_req = mock_mt5.order_send.call_args_list[0].args[0]
    assert deal_req["action"] == mock_mt5.TRADE_ACTION_DEAL
    assert deal_req["symbol"] == "EURUSD"
    assert deal_req["type"] == mock_mt5.ORDER_TYPE_BUY
    assert deal_req["volume"] == 0.1
    
    sltp_req = mock_mt5.order_send.call_args_list[1].args[0]
    assert sltp_req["action"] == mock_mt5.TRADE_ACTION_SLTP
    assert sltp_req["symbol"] == "EURUSD"
    assert math.isclose(sltp_req["sl"], 1.04000)
    assert math.isclose(sltp_req["tp"], 1.06000)
    assert sltp_req["position"] == 456  # Matching the 'order' ticket from the mock result


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
    assert "stop_loss must be below entry" in res["error"]

    # SELL with SL below price (invalid)
    res = _place_market_order(
        symbol="EURUSD",
        volume=0.1,
        order_type="SELL",
        stop_loss=1.04000,  # Bid is 1.05000
    )
    assert "error" in res
    assert "stop_loss must be above entry" in res["error"]

    # BUY with TP below price (invalid)
    res = _place_market_order(
        symbol="EURUSD",
        volume=0.1,
        order_type="BUY",
        take_profit=1.04000,  # Ask is 1.05010
    )
    assert "error" in res
    assert "take_profit must be above entry" in res["error"]

    # SELL with TP above price (invalid)
    res = _place_market_order(
        symbol="EURUSD",
        volume=0.1,
        order_type="SELL",
        take_profit=1.06000,  # Bid is 1.05000
    )
    assert "error" in res
    assert "take_profit must be below entry" in res["error"]


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
        "src.mtdata.core.trading_orders._resolve_open_position",
        return_value=(MagicMock(sl=0.0, tp=0.0), 456, {}),
    ):
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
    assert adapter.order_send.call_count == 2


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


def test_place_market_order_preserves_existing_position_magic_on_sltp_follow_up(mock_mt5):
    mock_mt5.TRADE_ACTION_SLTP = 6
    position = SimpleNamespace(symbol="EURUSD", sl=0.0, tp=0.0, type=0, magic=24680)

    with patch(
        "src.mtdata.core.trading_orders._resolve_open_position",
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
    sltp_req = mock_mt5.order_send.call_args_list[1].args[0]
    assert sltp_req["magic"] == 24680


def test_place_market_order_retries_sltp_without_comment_when_comment_is_invalid(mock_mt5):
    mock_mt5.TRADE_ACTION_SLTP = 6
    position = SimpleNamespace(symbol="EURUSD", sl=0.0, tp=0.0, type=0, magic=24680)
    mock_mt5.last_error.side_effect = [
        (1, "Success"),
        (-2, 'Invalid "comment" argument'),
        (-2, 'Invalid "comment" argument'),
        (1, "Success"),
    ]
    mock_mt5.order_send.side_effect = [
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
        "src.mtdata.core.trading_orders._resolve_open_position",
        return_value=(position, 456, {}),
    ):
        res = _place_market_order(
            symbol="EURUSD",
            volume=0.1,
            order_type="BUY",
            stop_loss=1.04000,
            take_profit=1.06000,
            comment="Breakout scalp comment that will be truncated anyway",
        )

    assert "error" not in res
    assert res["sl_tp_result"]["status"] == "applied"
    assert res["sl_tp_result"]["comment_fallback"]["used"] is True
    assert res["sl_tp_result"]["comment_fallback"]["strategy"] == "none"
    assert any("TP/SL modification" in str(w) for w in res.get("warnings", []))
    final_req = mock_mt5.order_send.call_args_list[-1].args[0]
    assert "comment" not in final_req


def test_modify_position_preserves_existing_magic(mock_mt5):
    mock_mt5.TRADE_ACTION_SLTP = 6
    position = SimpleNamespace(symbol="EURUSD", sl=1.03000, tp=1.05000, type=0, magic=98765)

    with patch(
        "src.mtdata.core.trading_execution._resolve_open_position",
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
        "src.mtdata.core.trading_execution._resolve_open_position",
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


def test_cancel_pending_preserves_existing_magic(mock_mt5):
    mock_mt5.TRADE_ACTION_REMOVE = 8
    mock_mt5.orders_get.return_value = [SimpleNamespace(ticket=123, magic=54321)]

    res = _cancel_pending(ticket=123)

    assert "error" not in res
    req = mock_mt5.order_send.call_args[0][0]
    assert req["magic"] == 54321


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
