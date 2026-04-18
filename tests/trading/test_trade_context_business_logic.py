from unittest.mock import patch

from mtdata.core.trading.context import trade_session_context


def _raw_trade_session_context(symbol: str):
    return trade_session_context.__wrapped__(symbol)


def test_trade_session_context_includes_shared_meta_and_keeps_nested_ticker() -> None:
    timezone_meta = {"used": {"tz": "UTC"}}
    ticker = {
        "success": True,
        "symbol": "EURUSD",
        "bid": 1.1,
        "ask": 1.1002,
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
        new=lambda symbol: ticker,
    ), patch(
        "mtdata.core.trading.context.trade_get_open",
        new=lambda request: {"success": True, "count": 0},
    ), patch(
        "mtdata.core.trading.context.trade_get_pending",
        new=lambda request: {"success": True, "count": 1},
    ):
        out = _raw_trade_session_context("EURUSD")

    assert out["state"] == "pending_only"
    assert out["ticker"]["meta"]["tool"] == "market_ticker"
    assert out["meta"]["tool"] == "trade_session_context"
    assert out["meta"]["runtime"]["timezone"] == timezone_meta
