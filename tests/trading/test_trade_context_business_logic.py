from unittest.mock import patch

from mtdata.core.trading.context import trade_session_context
from mtdata.core.trading.requests import TradeSessionContextRequest


def _raw_trade_session_context(symbol: str, *, detail: str = "compact"):
    return trade_session_context.__wrapped__(
        TradeSessionContextRequest(symbol=symbol, detail=detail)
    )


def test_trade_session_context_compacts_nested_sections_by_default() -> None:
    timezone_meta = {"used": {"tz": "UTC"}}
    ticker_compact = {
        "success": True,
        "symbol": "EURUSD",
        "bid": 1.1,
        "ask": 1.1002,
        "spread": 0.0002,
        "spread_pips": 2.0,
        "time": "2023-11-14 22:13",
        "timezone": "UTC",
    }
    open_positions = {
        "success": True,
        "kind": "open_positions",
        "count": 1,
        "items": [
            {
                "Symbol": "EURUSD",
                "Ticket": 123456,
                "Time": "2023-11-14 22:13",
                "Type": "BUY",
                "Volume": 0.1,
                "Open Price": 1.1,
                "Current Price": 1.1004,
                "SL": 1.095,
                "TP": 1.11,
                "Profit": 4.2,
                "Comments": "agent-open",
                "Magic": 77,
            }
        ],
    }

    with patch(
        "mtdata.core.output_contract.build_runtime_timezone_meta",
        return_value=timezone_meta,
    ), patch(
        "mtdata.core.trading.context.trade_account_info",
        new=lambda: {
            "success": True,
            "balance": 10000.0,
            "equity": 10010.0,
            "margin_level": 250.0,
            "terminal_connected": True,
            "execution_ready": True,
        },
    ), patch(
        "mtdata.core.trading.context.market_ticker",
        new=lambda symbol, detail="compact": ticker_compact,
    ), patch(
        "mtdata.core.trading.context.trade_get_open",
        new=lambda request: open_positions,
    ), patch(
        "mtdata.core.trading.context.trade_get_pending",
        new=lambda request: {
            "success": True,
            "kind": "pending_orders",
            "count": 0,
            "message": "No pending orders for EURUSD",
            "no_action": True,
        },
    ):
        out = _raw_trade_session_context("EURUSD")

    assert out["state"] == "open_position"
    assert out["account"] == {
        "balance": 10000.0,
        "equity": 10010.0,
        "margin_level": 250.0,
    }
    assert out["ticker"] == {
        "bid": 1.1,
        "ask": 1.1002,
        "spread": 0.0002,
        "spread_pips": 2.0,
        "time": "2023-11-14 22:13",
        "timezone": "UTC",
    }
    assert out["open_positions"] == [
        {
            "symbol": "EURUSD",
            "ticket": 123456,
            "time": "2023-11-14 22:13",
            "type": "BUY",
            "volume": 0.1,
            "open_price": 1.1,
            "current_price": 1.1004,
            "sl": 1.095,
            "tp": 1.11,
            "profit": 4.2,
            "comment": "agent-open",
            "magic": 77,
        }
    ]
    assert "pending_orders" not in out
    assert out["meta"]["tool"] == "trade_session_context"
    assert out["meta"]["runtime"]["timezone"] == timezone_meta


def test_trade_session_context_full_detail_keeps_nested_full_payloads() -> None:
    timezone_meta = {"used": {"tz": "UTC"}}
    ticker = {
        "success": True,
        "symbol": "EURUSD",
        "bid": 1.1,
        "ask": 1.1002,
        "time": 1700000000,
        "time_display": "2023-11-14 22:13",
        "meta": {
            "tool": "market_ticker",
            "runtime": {"timezone": timezone_meta},
        },
        "timezone": "UTC",
    }

    with patch(
        "mtdata.core.output_contract.build_runtime_timezone_meta",
        return_value=timezone_meta,
    ), patch(
        "mtdata.core.trading.context.trade_account_info",
        new=lambda: {"success": True, "balance": 10000.0},
    ), patch(
        "mtdata.core.trading.context.market_ticker",
        new=lambda symbol, detail="compact": ticker,
    ), patch(
        "mtdata.core.trading.context.trade_get_open",
        new=lambda request: {"success": True, "count": 0},
    ), patch(
        "mtdata.core.trading.context.trade_get_pending",
        new=lambda request: {"success": True, "count": 1},
    ):
        out = _raw_trade_session_context("EURUSD", detail="full")

    assert out["state"] == "pending_only"
    # With the fix, nested success and meta are stripped
    assert "success" not in out["ticker"]
    assert "meta" not in out["ticker"]
    assert out["ticker"]["bid"] == 1.1  # data is preserved
    assert out["ticker"]["time"] == "2023-11-14 22:13"
    assert out["ticker"]["time_epoch"] == 1700000000
    assert "time_display" not in out["ticker"]
    assert out["account"]["balance"] == 10000.0
    assert "success" not in out["account"]
    assert "meta" not in out["account"]
    assert out["pending_orders"]["count"] == 1
    assert "success" not in out["pending_orders"]
    assert "meta" not in out["pending_orders"]
    assert out["meta"]["tool"] == "trade_session_context"
    assert out["meta"]["runtime"]["timezone"] == timezone_meta


def test_trade_session_context_compact_sanitizes_nested_tool_errors() -> None:
    with patch(
        "mtdata.core.trading.context.trade_account_info",
        new=lambda: {"success": True, "balance": 10000.0, "equity": 10010.0},
    ), patch(
        "mtdata.core.trading.context.market_ticker",
        new=lambda symbol, detail="compact": {"success": True, "bid": 1.1, "ask": 1.1002},
    ), patch(
        "mtdata.core.trading.context.trade_get_open",
        new=lambda request: {
            "success": False,
            "error": "'types.SimpleNamespace' object has no attribute '_asdict'",
        },
    ), patch(
        "mtdata.core.trading.context.trade_get_pending",
        new=lambda request: {"success": True, "count": 0, "message": "No pending orders"},
    ):
        out = _raw_trade_session_context("EURUSD")

    assert out["success"] is True
    assert out["partial_failure"] is True
    assert out["state"] == "flat"
    assert out["open_positions"] == {
        "error": "Unable to fetch open positions.",
        "count": 0,
    }
    assert "SimpleNamespace" not in str(out["open_positions"])


def test_trade_session_context_compact_keeps_order_attribution_fields() -> None:
    with patch(
        "mtdata.core.trading.context.trade_account_info",
        new=lambda: {"success": True, "balance": 10000.0, "equity": 10010.0},
    ), patch(
        "mtdata.core.trading.context.market_ticker",
        new=lambda symbol, detail="compact": {"success": True, "bid": 1.1, "ask": 1.1002},
    ), patch(
        "mtdata.core.trading.context.trade_get_open",
        new=lambda request: {
            "success": True,
            "count": 1,
            "items": [
                {
                    "Symbol": "EURUSD",
                    "Ticket": 123,
                    "Type": "BUY",
                    "Volume": 0.1,
                    "Open Price": 1.1,
                    "Comments": "open-agent",
                    "Magic": 7001,
                }
            ],
        },
    ), patch(
        "mtdata.core.trading.context.trade_get_pending",
        new=lambda request: {
            "success": True,
            "count": 1,
            "items": [
                {
                    "Symbol": "EURUSD",
                    "Ticket": 456,
                    "Type": "BUY_LIMIT",
                    "Volume": 0.1,
                    "Open Price": 1.095,
                    "Comments": "pending-agent",
                    "Magic": 7002,
                }
            ],
        },
    ):
        out = _raw_trade_session_context("EURUSD")

    assert out["state"] == "mixed"
    assert out["open_positions"] == [
        {
            "symbol": "EURUSD",
            "ticket": 123,
            "type": "BUY",
            "volume": 0.1,
            "open_price": 1.1,
            "comment": "open-agent",
            "magic": 7001,
        }
    ]
    assert out["pending_orders"] == [
        {
            "symbol": "EURUSD",
            "ticket": 456,
            "type": "BUY_LIMIT",
            "volume": 0.1,
            "open_price": 1.095,
            "comment": "pending-agent",
            "magic": 7002,
        }
    ]
