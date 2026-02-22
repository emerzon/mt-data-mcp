import pytest
from unittest.mock import MagicMock, patch
import sys
import math

sys.modules.setdefault("MetaTrader5", MagicMock())

from src.mtdata.core.trading import _place_market_order, _place_pending_order


@pytest.fixture
def mock_mt5():
    mock_mt5 = sys.modules["MetaTrader5"]
    
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
    with patch("src.mtdata.core.trading._auto_connect_wrapper", lambda f: f):
        yield mock_mt5


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
    assert res["sl_tp_modified"] is True
    
    # Check that order_send was called twice (once for deal, once for modifying SL/TP since TRADE_ACTION_DEAL doesn't take SL/TP natively)
    assert mock_mt5.order_send.call_count == 2
    
    deal_req = mock_mt5.order_send.call_args_list[0].args[0]
    assert deal_req["action"] == mock_mt5.TRADE_ACTION_DEAL
    assert deal_req["symbol"] == "EURUSD"
    assert deal_req["type"] == mock_mt5.ORDER_TYPE_BUY
    assert deal_req["volume"] == 0.1
    
    sltp_req = mock_mt5.order_send.call_args_list[1].args[0]
    assert sltp_req["action"] == mock_mt5.TRADE_ACTION_SLTP
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
