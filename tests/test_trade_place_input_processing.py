from __future__ import annotations

import os
import sys
from unittest.mock import patch

# Add src to path to ensure local package is found
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from mtdata.core.trading import _normalize_order_type_input, trade_place, trade_modify


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
        out = trade_place(
            symbol="BTCUSD",
            volume=0.03,
            order_type="ORDER_TYPE_BUY",
            require_sl_tp=False,
            __cli_raw=True,
        )
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


def test_trade_place_blank_expiration_keeps_market_routing() -> None:
    with patch("mtdata.core.trading._place_market_order", return_value={"ok": True}) as mock_market, patch(
        "mtdata.core.trading._place_pending_order", return_value={"pending": True}
    ) as mock_pending:
        out = trade_place(
            symbol="BTCUSD",
            volume=0.03,
            order_type="BUY",
            expiration="",
            require_sl_tp=False,
            __cli_raw=True,
        )
        assert out == {"ok": True}
        mock_market.assert_called_once()
        mock_pending.assert_not_called()


def test_trade_place_require_sl_tp_needs_inputs_before_market_send() -> None:
    with patch(
        "mtdata.core.trading._prevalidate_trade_place_market_input",
        return_value=None,
    ) as mock_prevalidate, patch("mtdata.core.trading._place_market_order") as mock_market:
        out = trade_place(
            symbol="BTCUSD",
            volume=0.03,
            order_type="BUY",
            require_sl_tp=True,
            __cli_raw=True,
        )
    assert "error" in out
    assert out.get("require_sl_tp") is True
    assert set(out.get("missing", [])) == {"stop_loss", "take_profit"}
    mock_prevalidate.assert_called_once_with("BTCUSD", 0.03)
    mock_market.assert_not_called()


def test_trade_place_reports_symbol_error_before_sl_tp_requirement() -> None:
    with patch(
        "mtdata.core.trading._prevalidate_trade_place_market_input",
        return_value={"error": "Symbol FAKESYM not found"},
    ) as mock_prevalidate, patch("mtdata.core.trading._place_market_order") as mock_market:
        out = trade_place(
            symbol="FAKESYM",
            volume=0.03,
            order_type="BUY",
            require_sl_tp=True,
            __cli_raw=True,
        )
    assert out.get("error") == "Symbol FAKESYM not found"
    mock_prevalidate.assert_called_once_with("FAKESYM", 0.03)
    mock_market.assert_not_called()


def test_trade_place_require_sl_tp_false_allows_market_without_sl_tp() -> None:
    with patch("mtdata.core.trading._place_market_order", return_value={"retcode": 10009}) as mock_market:
        out = trade_place(
            symbol="BTCUSD",
            volume=0.03,
            order_type="BUY",
            require_sl_tp=False,
            __cli_raw=True,
        )
    assert out.get("retcode") == 10009
    mock_market.assert_called_once()


def test_trade_place_require_sl_tp_flags_unprotected_market_fill() -> None:
    with patch(
        "mtdata.core.trading._place_market_order",
        return_value={
            "retcode": 10009,
            "sl_tp_requested": True,
            "sl_tp_apply_status": "failed",
        },
    ):
        out = trade_place(
            symbol="BTCUSD",
            volume=0.03,
            order_type="BUY",
            stop_loss=64000,
            take_profit=68000,
            require_sl_tp=True,
            __cli_raw=True,
        )
    assert "error" in out
    assert out.get("require_sl_tp") is True
    assert out.get("protection_status") == "unprotected_position"
    assert any("CRITICAL" in str(w) for w in out.get("warnings", []))


def test_trade_place_defaults_to_failing_unprotected_market_fill() -> None:
    with patch(
        "mtdata.core.trading._place_market_order",
        return_value={
            "retcode": 10009,
            "sl_tp_requested": True,
            "sl_tp_apply_status": "failed",
            "position_ticket": 456,
        },
    ):
        out = trade_place(
            symbol="BTCUSD",
            volume=0.03,
            order_type="BUY",
            stop_loss=64000,
            take_profit=68000,
            __cli_raw=True,
        )
    assert "error" in out
    assert out.get("require_sl_tp") is True
    assert "TP/SL protection could not be applied" in str(out.get("error"))
    assert any("trade_modify 456" in str(w) for w in out.get("warnings", []))


def test_trade_place_auto_close_attempts_recovery_on_sl_tp_fail() -> None:
    with patch(
        "mtdata.core.trading._place_market_order",
        return_value={
            "retcode": 10009,
            "sl_tp_requested": True,
            "sl_tp_apply_status": "failed",
            "position_ticket": 789,
        },
    ), patch(
        "mtdata.core.trading._close_positions",
        return_value={"retcode": 10009, "ticket": 789},
    ) as mock_close:
        out = trade_place(
            symbol="BTCUSD",
            volume=0.03,
            order_type="BUY",
            stop_loss=64000,
            take_profit=68000,
            auto_close_on_sl_tp_fail=True,
            __cli_raw=True,
        )
    mock_close.assert_called_once()
    assert out.get("auto_close_on_sl_tp_fail") is True
    assert out.get("protection_status") == "auto_closed_after_sl_tp_fail"
    assert out.get("auto_close_result", {}).get("retcode") == 10009


def test_trade_place_preserves_fallback_protection_status_and_warning() -> None:
    with patch(
        "mtdata.core.trading._place_market_order",
        return_value={
            "retcode": 10009,
            "sl_tp_requested": True,
            "sl_tp_apply_status": "applied",
            "sl_tp_fallback_used": True,
            "protection_status": "protected_after_fallback",
            "warnings": [
                "TP/SL protection required a post-fill fallback modification. Verify the live position is protected."
            ],
        },
    ):
        out = trade_place(
            symbol="BTCUSD",
            volume=0.03,
            order_type="BUY",
            stop_loss=64000,
            take_profit=68000,
            __cli_raw=True,
        )
    assert out.get("protection_status") == "protected_after_fallback"
    assert any("fallback modification" in str(w) for w in out.get("warnings", []))


def test_trade_modify_blank_expiration_keeps_position_path() -> None:
    with patch("mtdata.core.trading._modify_position", return_value={"success": True}) as mock_pos, patch(
        "mtdata.core.trading._modify_pending_order", return_value={"success": True}
    ) as mock_pending:
        out = trade_modify(ticket=123, stop_loss=1.0, expiration="", __cli_raw=True)
        assert out.get("success") is True
        mock_pos.assert_called_once()
        mock_pending.assert_not_called()


def test_trade_modify_pending_not_found_reports_checked_scope() -> None:
    with patch(
        "mtdata.core.trading._modify_pending_order",
        return_value={"error": "Pending order 123 not found"},
    ):
        out = trade_modify(ticket=123, price=1.2, __cli_raw=True)
    assert "error" in out
    assert out.get("checked_scopes") == ["pending_orders"]


def test_trade_modify_missing_ticket_reports_both_checked_scopes() -> None:
    with patch(
        "mtdata.core.trading._modify_position",
        return_value={"error": "Position 123 not found"},
    ), patch(
        "mtdata.core.trading._modify_pending_order",
        return_value={"error": "Pending order 123 not found"},
    ):
        out = trade_modify(ticket=123, stop_loss=1.0, __cli_raw=True)
    assert "error" in out
    assert out.get("checked_scopes") == ["positions", "pending_orders"]
