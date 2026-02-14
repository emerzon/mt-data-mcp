from __future__ import annotations

import os
import sys
from unittest.mock import patch

# Add src to path to ensure local package is found
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from mtdata.core.trading import _normalize_order_type_input, trade_place


def test_normalize_order_type_accepts_mt5_integer() -> None:
    normalized, error = _normalize_order_type_input(2)
    assert error is None
    assert normalized == "BUY_LIMIT"


def test_normalize_order_type_accepts_prefixed_symbolic_name() -> None:
    normalized, error = _normalize_order_type_input("ORDER_TYPE_BUY_STOP")
    assert error is None
    assert normalized == "BUY_STOP"


def test_normalize_order_type_accepts_numeric_string() -> None:
    normalized, error = _normalize_order_type_input("4")
    assert error is None
    assert normalized == "BUY_STOP"


def test_trade_place_routes_numeric_order_type_to_pending() -> None:
    with patch("mtdata.core.trading._place_pending_order", return_value={"ok": True}) as mock_pending:
        out = trade_place(symbol="BTCUSD", volume=0.03, order_type=2, price=68750, __cli_raw=True)
        assert out == {"ok": True}
        assert mock_pending.call_args.kwargs["order_type"] == "BUY_LIMIT"


def test_trade_place_routes_prefixed_order_type_to_pending() -> None:
    with patch("mtdata.core.trading._place_pending_order", return_value={"ok": True}) as mock_pending:
        out = trade_place(symbol="BTCUSD", volume=0.03, order_type="ORDER_TYPE_BUY_STOP", price=70650, __cli_raw=True)
        assert out == {"ok": True}
        assert mock_pending.call_args.kwargs["order_type"] == "BUY_STOP"


def test_trade_place_routes_prefixed_market_order_type() -> None:
    with patch("mtdata.core.trading._place_market_order", return_value={"ok": True}) as mock_market:
        out = trade_place(symbol="BTCUSD", volume=0.03, order_type="ORDER_TYPE_BUY", __cli_raw=True)
        assert out == {"ok": True}
        assert mock_market.call_args.kwargs["order_type"] == "BUY"


def test_trade_place_rejects_unknown_numeric_order_type() -> None:
    out = trade_place(symbol="BTCUSD", volume=0.03, order_type=99, price=68750, __cli_raw=True)
    assert isinstance(out, dict)
    assert "error" in out
    assert "Numeric values must match MT5 constants 0..5" in str(out["error"])


def test_trade_place_missing_required_fields_returns_friendly_error() -> None:
    out = trade_place(__cli_raw=True)
    assert isinstance(out, dict)
    assert "error" in out
    assert "Missing required field(s): symbol, volume, order_type" in str(out["error"])
    assert out.get("required") == ["symbol", "volume", "order_type"]
