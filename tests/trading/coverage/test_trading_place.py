"""Tests for trade place operations in mtdata.core.trading.

Covers:
- trade_place (dispatch logic)
- _place_market_order (MT5)
- _place_pending_order (MT5)
"""

import importlib
import math
import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Ensure src/ is importable and stub MetaTrader5 before importing the module
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

_mt5_stub = MagicMock()
_mt5_stub.ORDER_TYPE_BUY = 0
_mt5_stub.ORDER_TYPE_SELL = 1
_mt5_stub.ORDER_TYPE_BUY_LIMIT = 2
_mt5_stub.ORDER_TYPE_SELL_LIMIT = 3
_mt5_stub.ORDER_TYPE_BUY_STOP = 4
_mt5_stub.ORDER_TYPE_SELL_STOP = 5
_mt5_stub.TRADE_ACTION_DEAL = 1
_mt5_stub.TRADE_ACTION_PENDING = 5
_mt5_stub.TRADE_ACTION_SLTP = 6
_mt5_stub.TRADE_ACTION_MODIFY = 7
_mt5_stub.TRADE_ACTION_REMOVE = 8
_mt5_stub.TRADE_RETCODE_DONE = 10009
_mt5_stub.ORDER_TIME_GTC = 0
_mt5_stub.ORDER_TIME_SPECIFIED = 2
_mt5_stub.ORDER_FILLING_IOC = 1
_mt5_stub.POSITION_TYPE_BUY = 0
_mt5_stub.POSITION_TYPE_SELL = 1
sys.modules["MetaTrader5"] = _mt5_stub

from mtdata.core.trading import (
    trade_place as _trade_place_tool,
)
from mtdata.core.trading.requests import (
    TradePlaceRequest,
)


# ===================================================================
# Helpers
# ===================================================================

def trade_place(**kwargs):
    raw_output = bool(kwargs.pop("__cli_raw", False))
    request = kwargs.pop("request", None)
    if request is None:
        request = TradePlaceRequest(**kwargs)
    return _trade_place_tool(request=request, __cli_raw=raw_output)


def _unwrap_mcp(result):
    """MCP tool decorator serialises dicts to strings; return the raw dict
    when the underlying function is patched to return one, otherwise return
    the string for assertion with ``in``."""
    if isinstance(result, dict):
        return result
    return str(result)


def _sym(
    volume_min=0.01,
    volume_max=100.0,
    volume_step=0.01,
    point=0.00001,
    digits=5,
    visible=True,
    trade_contract_size=100000,
    trade_tick_value=1.0,
    trade_tick_size=0.00001,
):
    """Create a mock symbol_info object."""
    return SimpleNamespace(
        volume_min=volume_min,
        volume_max=volume_max,
        volume_step=volume_step,
        point=point,
        digits=digits,
        visible=visible,
        trade_contract_size=trade_contract_size,
        trade_tick_value=trade_tick_value,
        trade_tick_size=trade_tick_size,
        symbol="EURUSD",
    )


def _tick(bid=1.10000, ask=1.10020):
    return SimpleNamespace(bid=bid, ask=ask)


def _order_result(retcode=10009, deal=1, order=1, volume=0.01, price=1.1,
                  bid=1.1, ask=1.1002, comment="done", request_id=1):
    return SimpleNamespace(
        retcode=retcode, deal=deal, order=order, volume=volume,
        price=price, bid=bid, ask=ask, comment=comment, request_id=request_id,
    )


def _position(ticket=1, symbol="EURUSD", type_=0, volume=0.01,
              price_open=1.1, sl=1.09, tp=1.12, profit=50.0,
              price_current=1.105, comment="", magic=234000,
              time=1700000000, time_update=1700000000, swap=0.0,
              identifier=None, position_id=None, order=None, deal=None):
    return SimpleNamespace(
        ticket=ticket, symbol=symbol, type=type_, volume=volume,
        price_open=price_open, sl=sl, tp=tp, profit=profit,
        price_current=price_current, comment=comment, magic=magic,
        time=time, time_update=time_update, swap=swap,
        identifier=identifier, position_id=position_id, order=order, deal=deal,
        _asdict=lambda: {
            "ticket": ticket, "symbol": symbol, "type": type_, "volume": volume,
            "price_open": price_open, "sl": sl, "tp": tp, "profit": profit,
            "price_current": price_current, "comment": comment, "magic": magic,
            "time": time, "time_update": time_update, "swap": swap,
            "identifier": identifier, "position_id": position_id, "order": order, "deal": deal,
        },
    )


# Patch MT5 connection helpers to avoid real terminal access in tests
@pytest.fixture(autouse=True)
def _bypass_auto_connect(monkeypatch):
    """Neutralize MT5 connection guards so no real terminal access is needed."""
    passthrough = lambda fn=None, **kw: fn if fn else (lambda f: f)
    monkeypatch.setattr("mtdata.core.trading.gateway.ensure_mt5_connection_or_raise", lambda: None)
    monkeypatch.setattr("src.mtdata.core.trading.gateway.ensure_mt5_connection_or_raise", lambda: None)
    for module_name in [
        "mtdata.core.trading.account",
        "mtdata.core.trading.execution",
        "mtdata.core.trading.orders",
        "mtdata.core.trading.positions",
        "mtdata.core.trading.risk",
        "mtdata.core.trading.validation",
    ]:
        module = importlib.import_module(module_name)
        if hasattr(module, "ensure_mt5_connection_or_raise"):
            monkeypatch.setattr(f"{module_name}.ensure_mt5_connection_or_raise", lambda: None)
    return passthrough


# ===================================================================
#  trade_place  (dispatch / validation – no MT5 calls for validation errors)
# ===================================================================

class TestTradePlace:

    def test_missing_symbol(self):
        result = _unwrap_mcp(trade_place(symbol=None, volume=0.01, order_type="BUY"))
        assert "symbol" in result

    def test_missing_volume(self):
        result = _unwrap_mcp(trade_place(symbol="EURUSD", volume=None, order_type="BUY"))
        assert "volume" in result

    def test_missing_order_type(self):
        result = _unwrap_mcp(trade_place(symbol="EURUSD", volume=0.01, order_type=None))
        assert "order_type" in result

    def test_all_missing(self):
        result = _unwrap_mcp(trade_place())
        for field in ("symbol", "volume", "order_type"):
            assert field in result

    def test_empty_order_type_string(self):
        result = _unwrap_mcp(trade_place(symbol="EURUSD", volume=0.01, order_type="  "))
        assert "order_type" in result

    def test_invalid_order_type(self):
        result = _unwrap_mcp(trade_place(symbol="EURUSD", volume=0.01, order_type="GARBAGE"))
        assert "Unsupported" in result

    def test_pending_without_price_returns_error(self):
        result = _unwrap_mcp(trade_place(symbol="EURUSD", volume=0.01, order_type="BUY_LIMIT", price=None))
        assert "price" in result.lower()

    def test_invalid_expiration_returns_error(self):
        result = _unwrap_mcp(trade_place(
            symbol="EURUSD", volume=0.01, order_type="BUY_LIMIT",
            price=1.09, expiration="not-a-date-at-all-xyz",
        ))
        assert "error" in result.lower() or "unsupported" in result.lower()

    @patch("mtdata.core.trading._place_market_order")
    def test_dispatches_to_market(self, mock_market):
        mock_market.return_value = {"retcode": 10009}
        trade_place(symbol="EURUSD", volume=0.01, order_type="BUY", require_sl_tp=False)
        mock_market.assert_called_once()

    @patch("mtdata.core.trading._place_pending_order")
    def test_dispatches_to_pending_explicit_type(self, mock_pending):
        mock_pending.return_value = {"success": True}
        trade_place(symbol="EURUSD", volume=0.01, order_type="BUY_LIMIT", price=1.09)
        mock_pending.assert_called_once()

    def test_rejects_market_order_with_price(self):
        with patch("mtdata.core.trading._place_market_order") as mock_market, patch(
            "mtdata.core.trading._place_pending_order"
        ) as mock_pending:
            result = trade_place(
                symbol="EURUSD",
                volume=0.01,
                order_type="BUY",
                price=1.09,
                __cli_raw=True,
            )
        assert "Conflicting arguments" in result["error"]
        assert "BUY_LIMIT/BUY_STOP" in result["error"]
        mock_market.assert_not_called()
        mock_pending.assert_not_called()

    def test_rejects_market_order_with_price_and_expiration(self):
        with patch("mtdata.core.trading._place_market_order") as mock_market, patch(
            "mtdata.core.trading._place_pending_order"
        ) as mock_pending:
            result = trade_place(
                symbol="EURUSD",
                volume=0.01,
                order_type="BUY",
                price=1.09,
                expiration="GTC",
                __cli_raw=True,
            )
        assert "Conflicting arguments" in result["error"]
        mock_market.assert_not_called()
        mock_pending.assert_not_called()


# ===================================================================
#  _place_market_order (MT5 mocked)
# ===================================================================

class TestPlaceMarketOrder:

    def _setup_mt5(self, mock_mt5):
        mock_mt5.symbol_info.return_value = _sym()
        mock_mt5.symbol_info_tick.return_value = _tick()
        mock_mt5.symbol_select.return_value = True
        mock_mt5.ORDER_TYPE_BUY = 0
        mock_mt5.ORDER_TYPE_SELL = 1
        mock_mt5.TRADE_ACTION_DEAL = 1
        mock_mt5.TRADE_ACTION_SLTP = 6
        mock_mt5.TRADE_RETCODE_DONE = 10009
        mock_mt5.ORDER_TIME_GTC = 0
        mock_mt5.ORDER_FILLING_IOC = 1

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_symbol_not_found(self):
        import sys
        mt5 = sys.modules["MetaTrader5"]
        mt5.symbol_info.return_value = None
        from mtdata.core.trading import _place_market_order
        result = _place_market_order("INVALID", 0.01, "BUY")
        assert "error" in result and "not found" in result["error"]

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_order_send_returns_none(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.order_send.return_value = None
        mt5.last_error.return_value = (10006, "No connection")
        from mtdata.core.trading import _place_market_order
        result = _place_market_order("EURUSD", 0.01, "BUY")
        assert "error" in result

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_order_send_retcode_fail(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.order_send.return_value = _order_result(retcode=10004)
        from mtdata.core.trading import _place_market_order
        result = _place_market_order("EURUSD", 0.01, "BUY")
        assert "error" in result and result["retcode"] == 10004

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_invisible_symbol_selected(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.symbol_info.return_value = _sym(visible=False)
        mt5.symbol_select.return_value = True
        mt5.order_send.return_value = _order_result()
        from mtdata.core.trading import _place_market_order
        result = _place_market_order("EURUSD", 0.01, "BUY")
        mt5.symbol_select.assert_called_with("EURUSD", True)

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_invisible_symbol_select_fails(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.symbol_info.return_value = _sym(visible=False)
        mt5.symbol_select.return_value = False
        from mtdata.core.trading import _place_market_order
        result = _place_market_order("EURUSD", 0.01, "BUY")
        assert "error" in result and "Failed to select" in result["error"]

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_buy_sl_above_price_rejected(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        from mtdata.core.trading import _place_market_order
        result = _place_market_order("EURUSD", 0.01, "BUY", stop_loss=1.2)
        assert "error" in result and "below" in result["error"]

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_sell_sl_below_price_rejected(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        from mtdata.core.trading import _place_market_order
        result = _place_market_order("EURUSD", 0.01, "SELL", stop_loss=1.0)
        assert "error" in result and "above" in result["error"]

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_buy_tp_below_price_rejected(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        from mtdata.core.trading import _place_market_order
        result = _place_market_order("EURUSD", 0.01, "BUY", take_profit=1.0)
        assert "error" in result and "above" in result["error"]

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_sell_tp_above_price_rejected(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        from mtdata.core.trading import _place_market_order
        result = _place_market_order("EURUSD", 0.01, "SELL", take_profit=1.2)
        assert "error" in result and "below" in result["error"]

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_success_with_sl_tp_modification(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.order_send.side_effect = [_order_result(), _order_result()]
        mt5.positions_get.return_value = [_position()]
        from mtdata.core.trading import _place_market_order
        result = _place_market_order("EURUSD", 0.01, "BUY", stop_loss=1.09, take_profit=1.12)
        sl_tp_result = result.get("sl_tp_result") or {}
        assert sl_tp_result.get("status") == "applied"
        assert sl_tp_result.get("requested") == {"sl": pytest.approx(1.09), "tp": pytest.approx(1.12)}
        assert sl_tp_result.get("applied") == {"sl": pytest.approx(1.09), "tp": pytest.approx(1.12)}
        assert sl_tp_result.get("broker_adjusted") is None
        assert sl_tp_result.get("adjustment") is None
        assert sl_tp_result.get("error") is None

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_sl_tp_uses_resolved_position_ticket_without_follow_up(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.order_send.side_effect = [
            _order_result(order=1001, deal=2002),
        ]
        mt5.positions_get.side_effect = [
            [],  # candidate order ticket lookup
            [_position(ticket=3003, deal=2002, sl=1.09, tp=1.12)],  # candidate deal ticket lookup
        ]
        from mtdata.core.trading import _place_market_order
        result = _place_market_order("EURUSD", 0.01, "BUY", stop_loss=1.09, take_profit=1.12)
        assert (result.get("sl_tp_result") or {}).get("status") == "applied"
        assert result.get("position_ticket") == 3003
        assert mt5.order_send.call_count == 1
        request = mt5.order_send.call_args.args[0]
        assert request.get("sl") == pytest.approx(1.09)
        assert request.get("tp") == pytest.approx(1.12)

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_initial_protected_order_failure_returns_error(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.order_send.return_value = _order_result(retcode=10016, comment="Invalid stops")
        mt5.positions_get.return_value = [_position()]
        from mtdata.core.trading import _place_market_order
        result = _place_market_order("EURUSD", 0.01, "BUY", stop_loss=1.09, take_profit=1.12)
        assert result.get("error") == "Failed to send order"
        assert result.get("retcode") == 10016
        assert "sl_tp_result" not in result
        assert "fallback_used" not in result
        assert mt5.order_send.call_count == 1

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_sl_tp_position_not_found_does_not_mark_protected_order_failed(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.order_send.return_value = _order_result()
        mt5.positions_get.return_value = []
        from mtdata.core.trading import _place_market_order
        result = _place_market_order("EURUSD", 0.01, "BUY", stop_loss=1.09)
        sl_tp_result = result.get("sl_tp_result") or {}
        assert sl_tp_result.get("requested") == {"sl": pytest.approx(1.09)}
        assert sl_tp_result.get("status") == "applied"
        assert sl_tp_result.get("applied") == {"sl": pytest.approx(1.09)}
        assert result.get("position_ticket") is None
        assert result.get("position_ticket_candidates") == [1]
        assert not any("trade_modify 1" in str(w) for w in result.get("warnings", []))

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_sl_tp_not_requested_state_is_explicit(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.order_send.return_value = _order_result()
        from mtdata.core.trading import _place_market_order
        result = _place_market_order("EURUSD", 0.01, "BUY")
        assert result.get("sl_tp_result") == {"status": "not_requested"}

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_comment_truncation_is_reported(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.order_send.return_value = _order_result()
        from mtdata.core.trading import _place_market_order
        long_comment = "X" * 60
        result = _place_market_order("EURUSD", 0.01, "BUY", comment=long_comment)
        trunc = result.get("comment_truncation")
        assert isinstance(trunc, dict)
        assert trunc.get("requested") == long_comment
        assert trunc.get("max_length") == 31
        assert len(str(trunc.get("applied") or "")) <= 31
        assert any("Comment truncated" in str(w) for w in result.get("warnings", []))

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_comment_rejection_returns_error_without_fallback(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.order_send.side_effect = [
            _order_result(retcode=10013, comment='Invalid "comment" argument'),
        ]
        from mtdata.core.trading import _place_market_order

        result = _place_market_order(
            "BTCUSD",
            0.01,
            "SELL",
            comment="Short: ETS bearish, barrier EV+, R:R 9.65",
        )

        assert "broker rejected the comment field" in result["error"]
        assert "comment_fallback" not in result
        assert mt5.order_send.call_count == 1
        first_request = mt5.order_send.call_args_list[0].args[0]
        assert first_request["comment"] == "Short ETS bearish barrier EV R"

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_atomic_sl_tp_result_does_not_report_readback_adjustment(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.order_send.side_effect = [_order_result(), _order_result()]
        mt5.positions_get.side_effect = [
            [_position(sl=1.09, tp=1.12)],
            [_position(sl=1.0895, tp=1.1203)],
        ]
        from mtdata.core.trading import _place_market_order
        result = _place_market_order("EURUSD", 0.01, "BUY", stop_loss=1.09, take_profit=1.12)
        sl_tp_result = result.get("sl_tp_result") or {}
        assert sl_tp_result.get("status") == "applied"
        assert sl_tp_result.get("broker_adjusted") is None
        assert sl_tp_result.get("adjustment") is None
        assert mt5.order_send.call_count == 1

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_pending_type_rejected_for_market(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        from mtdata.core.trading import _place_market_order
        result = _place_market_order("EURUSD", 0.01, "BUY_LIMIT")
        assert "error" in result

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_tick_none_returns_error(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.symbol_info_tick.return_value = None
        from mtdata.core.trading import _place_market_order
        result = _place_market_order("EURUSD", 0.01, "BUY")
        assert "error" in result and "current price" in result["error"]

    def test_accepts_injected_gateway(self):
        """Market order placement accepts an injected MT5TradingGateway."""
        from mtdata.core.trading.gateway import MT5TradingGateway
        from mtdata.core.trading import _place_market_order

        adapter = MagicMock()
        adapter.symbol_info.return_value = _sym()
        adapter.symbol_info_tick.side_effect = [
            _tick(bid=1.05000, ask=1.05010),
            _tick(bid=1.05000, ask=1.05010),
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
            "mtdata.core.trading.orders._resolve_open_position",
            return_value=(MagicMock(sl=0.0, tp=0.0), 456, {}),
        ), patch("mtdata.core.trading.orders.trade_guardrails_config") as mock_guard_config:
            mock_guard_config.enabled = False
            mock_guard_config.trading_enabled = True
            mock_guard_config.max_volume_by_symbol = {}
            mock_guard_config.is_enabled.return_value = False

            result = _place_market_order(
                symbol="EURUSD",
                volume=0.1,
                order_type="BUY",
                stop_loss=1.04000,
                take_profit=1.06000,
                gateway=gateway,
            )

        assert "error" not in result
        ensure_connection.assert_called_once_with()
        assert adapter.order_send.call_count == 1
        request = adapter.order_send.call_args.args[0]
        assert math.isclose(request["sl"], 1.04000)
        assert math.isclose(request["tp"], 1.06000)
        assert "comment_fallback" not in result


# ===================================================================
#  _place_pending_order (MT5 mocked)
# ===================================================================

class TestPlacePendingOrder:

    def _setup_mt5(self, mock_mt5):
        mock_mt5.symbol_info.return_value = _sym()
        mock_mt5.symbol_info_tick.return_value = _tick(bid=1.10000, ask=1.10020)
        mock_mt5.symbol_select.return_value = True
        mock_mt5.ORDER_TYPE_BUY = 0
        mock_mt5.ORDER_TYPE_SELL = 1
        mock_mt5.ORDER_TYPE_BUY_LIMIT = 2
        mock_mt5.ORDER_TYPE_SELL_LIMIT = 3
        mock_mt5.ORDER_TYPE_BUY_STOP = 4
        mock_mt5.ORDER_TYPE_SELL_STOP = 5
        mock_mt5.TRADE_ACTION_PENDING = 5
        mock_mt5.TRADE_RETCODE_DONE = 10009
        mock_mt5.ORDER_TIME_GTC = 0
        mock_mt5.ORDER_TIME_SPECIFIED = 2
        mock_mt5.ORDER_FILLING_IOC = 1

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_buy_limit_success(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.order_send.return_value = _order_result()
        from mtdata.core.trading import _place_pending_order
        result = _place_pending_order("EURUSD", 0.01, "BUY_LIMIT", price=1.09)
        assert result.get("success") is True
        assert result.get("requested_price") == 1.09
        assert result.get("requested_sl") is None
        assert result.get("requested_tp") is None

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_sell_stop_success(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.order_send.return_value = _order_result()
        from mtdata.core.trading import _place_pending_order
        result = _place_pending_order("EURUSD", 0.01, "SELL_STOP", price=1.09)
        assert result.get("success") is True

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_buy_limit_price_above_ask_rejected(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        from mtdata.core.trading import _place_pending_order
        result = _place_pending_order("EURUSD", 0.01, "BUY_LIMIT", price=1.20)
        assert "error" in result and "below ask" in result["error"]

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_sell_limit_price_below_bid_rejected(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        from mtdata.core.trading import _place_pending_order
        result = _place_pending_order("EURUSD", 0.01, "SELL_LIMIT", price=1.05)
        assert "error" in result and "above bid" in result["error"]

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_buy_stop_price_below_ask_rejected(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        from mtdata.core.trading import _place_pending_order
        result = _place_pending_order("EURUSD", 0.01, "BUY_STOP", price=1.05)
        assert "error" in result and "above ask" in result["error"]

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_sell_stop_price_above_bid_rejected(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        from mtdata.core.trading import _place_pending_order
        result = _place_pending_order("EURUSD", 0.01, "SELL_STOP", price=1.20)
        assert "error" in result and "below bid" in result["error"]

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_pending_sl_tp_buy_validation(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        from mtdata.core.trading import _place_pending_order
        # SL above entry for BUY_LIMIT should fail
        result = _place_pending_order("EURUSD", 0.01, "BUY_LIMIT", price=1.09, stop_loss=1.10)
        assert "error" in result and "below entry" in result["error"]

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_pending_tp_buy_validation(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        from mtdata.core.trading import _place_pending_order
        # TP below entry for BUY_LIMIT should fail
        result = _place_pending_order("EURUSD", 0.01, "BUY_LIMIT", price=1.09, take_profit=1.05)
        assert "error" in result and "above entry" in result["error"]

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_pending_sl_sell_validation(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        from mtdata.core.trading import _place_pending_order
        # SL below entry for SELL_LIMIT should fail
        result = _place_pending_order("EURUSD", 0.01, "SELL_LIMIT", price=1.12, stop_loss=1.10)
        assert "error" in result and "above entry" in result["error"]

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_pending_tp_sell_validation(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        from mtdata.core.trading import _place_pending_order
        # TP above entry for SELL_LIMIT should fail
        result = _place_pending_order("EURUSD", 0.01, "SELL_LIMIT", price=1.12, take_profit=1.15)
        assert "error" in result and "below entry" in result["error"]

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_implicit_buy_below_ask_becomes_buy_limit(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.order_send.return_value = _order_result()
        from mtdata.core.trading import _place_pending_order
        result = _place_pending_order("EURUSD", 0.01, "BUY", price=1.09)
        # Verify the request used BUY_LIMIT (type=2)
        call_args = mt5.order_send.call_args[0][0]
        assert call_args["type"] == mt5.ORDER_TYPE_BUY_LIMIT

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_implicit_sell_above_bid_becomes_sell_limit(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.order_send.return_value = _order_result()
        from mtdata.core.trading import _place_pending_order
        result = _place_pending_order("EURUSD", 0.01, "SELL", price=1.12)
        call_args = mt5.order_send.call_args[0][0]
        assert call_args["type"] == mt5.ORDER_TYPE_SELL_LIMIT

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_order_send_none_returns_error(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.order_send.return_value = None
        mt5.last_error.return_value = (10006, "err")
        from mtdata.core.trading import _place_pending_order
        result = _place_pending_order("EURUSD", 0.01, "BUY_LIMIT", price=1.09)
        assert "error" in result and "Failed" in result["error"]

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_order_send_retcode_fail(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.order_send.return_value = _order_result(retcode=10004)
        from mtdata.core.trading import _place_pending_order
        result = _place_pending_order("EURUSD", 0.01, "BUY_LIMIT", price=1.09)
        assert "error" in result

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_non_finite_price_rejected(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        from mtdata.core.trading import _place_pending_order
        result = _place_pending_order("EURUSD", 0.01, "BUY_LIMIT", price=float("inf"))
        assert "error" in result and "finite" in result["error"]

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_with_expiration_sets_type_time_specified(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.order_send.return_value = _order_result()
        from mtdata.core.trading import _place_pending_order
        # Use a numeric expiration (epoch seconds)
        with patch("mtdata.core.trading.time._normalize_pending_expiration", return_value=(1717200000, True)):
            result = _place_pending_order("EURUSD", 0.01, "BUY_LIMIT", price=1.09, expiration=1717200000)
        call_args = mt5.order_send.call_args[0][0]
        assert call_args["type_time"] == 2  # ORDER_TIME_SPECIFIED
        assert call_args["expiration"] == 1717200000
