from __future__ import annotations

import logging
import os
import sys
from unittest.mock import patch

import pytest
from pydantic import ValidationError

# Add src to path to ensure local package is found
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from mtdata.core.trading import trade_modify as _trade_modify_tool
from mtdata.core.trading import trade_place as _trade_place_tool
from mtdata.core.trading.requests import TradeModifyRequest, TradePlaceRequest
from mtdata.core.trading.validation import _normalize_order_type_input


def trade_place(**kwargs):
    raw_output = bool(kwargs.pop("__cli_raw", False))
    request = kwargs.pop("request", None)
    if request is None:
        request = TradePlaceRequest(**kwargs)
    return _trade_place_tool(request=request, __cli_raw=raw_output)


def trade_modify(**kwargs):
    raw_output = bool(kwargs.pop("__cli_raw", False))
    request = kwargs.pop("request", None)
    if request is None:
        request = TradeModifyRequest(**kwargs)
    return _trade_modify_tool(request=request, __cli_raw=raw_output)


def test_trade_place_request_uses_preview_detail_canonical_field() -> None:
    fields = TradePlaceRequest.model_fields

    assert "preview_detail" in fields
    assert "detail" not in fields
    assert TradePlaceRequest(detail="full").preview_detail == "full"
    assert TradePlaceRequest(detail="compact").preview_detail == "compact"
    assert TradePlaceRequest(detail="standard").preview_detail == "compact"
    assert TradePlaceRequest(preview_detail="basic").preview_detail == "compact"


def test_normalize_order_type_rejects_mt5_integer() -> None:
    normalized, error = _normalize_order_type_input(2)
    assert normalized is None
    assert "canonical string" in error


def test_normalize_order_type_rejects_prefixed_symbolic_name() -> None:
    normalized, error = _normalize_order_type_input("ORDER_TYPE_BUY_STOP")
    assert normalized is None
    assert "Unsupported" in error


def test_normalize_order_type_rejects_numeric_string() -> None:
    normalized, error = _normalize_order_type_input("4")
    assert normalized is None
    assert "canonical string" in error


def test_trade_place_rejects_numeric_order_type() -> None:
    with pytest.raises(ValidationError):
        TradePlaceRequest(symbol="BTCUSD", volume=0.03, order_type=2, price=68750)


def test_trade_place_rejects_prefixed_order_type() -> None:
    with patch("mtdata.core.trading._place_pending_order", return_value={"ok": True}) as mock_pending:
        out = trade_place(symbol="BTCUSD", volume=0.03, order_type="ORDER_TYPE_BUY_STOP", price=70650, __cli_raw=True)

    assert "Unsupported order_type" in out["error"]
    mock_pending.assert_not_called()


def test_trade_place_routes_prefixed_market_order_type() -> None:
    with patch("mtdata.core.trading._place_market_order", return_value={"ok": True}) as mock_market:
        out = trade_place(
            symbol="BTCUSD",
            volume=0.03,
            order_type="BUY",
            require_sl_tp=False,
            __cli_raw=True,
        )
        assert out == {"ok": True}
        assert mock_market.call_args.kwargs["order_type"] == "BUY"


def test_trade_place_rejects_unknown_numeric_order_type() -> None:
    with pytest.raises(ValidationError):
        TradePlaceRequest(symbol="BTCUSD", volume=0.03, order_type=99, price=68750)


def test_trade_place_logs_finish_event(caplog) -> None:
    with patch("mtdata.core.trading._place_market_order", return_value={"success": True}), caplog.at_level(logging.DEBUG,
        logger="mtdata.core.trading",
    ):
        out = trade_place(
            symbol="BTCUSD",
            volume=0.03,
            order_type="BUY",
            require_sl_tp=False,
            __cli_raw=True,
        )

    assert out["success"] is True
    assert any(
        "event=finish operation=trade_place success=True" in record.message
        for record in caplog.records
    )


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
        "mtdata.core.trading.validation._prevalidate_trade_place_market_input",
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
        "mtdata.core.trading.validation._prevalidate_trade_place_market_input",
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


def test_trade_place_dry_run_market_preview_skips_order_send() -> None:
    with patch("mtdata.core.trading._place_market_order") as mock_market:
        out = trade_place(
            symbol="BTCUSD",
            volume=0.03,
            order_type="BUY",
            stop_loss=64000,
            take_profit=68000,
            dry_run=True,
            __cli_raw=True,
        )

    assert out.get("success") is True
    assert out.get("dry_run") is True
    assert out.get("no_action") is True
    assert out.get("pending") is False
    assert out.get("action") == "place_market_order"
    assert out.get("actionability") == "preview_only"
    assert out.get("preview_scope_summary") == "Routing and local request checks only."
    assert "validation_not_performed" not in out
    assert "warnings" not in out
    assert "validation_scope" not in out
    assert "trade_gate_passed" not in out
    assert out.get("message") == "Dry run only. No order was sent to MT5."
    mock_market.assert_not_called()


def test_trade_place_dry_run_preview_detail_omits_safety_lists() -> None:
    with patch("mtdata.core.trading._place_market_order") as mock_market:
        out = trade_place(
            symbol="BTCUSD",
            volume=0.03,
            order_type="BUY",
            stop_loss=64000,
            take_profit=68000,
            dry_run=True,
            preview_detail="compact",
            __cli_raw=True,
        )

    assert out.get("success") is True
    assert out.get("dry_run") is True
    assert out.get("actionability") == "preview_only"
    assert out.get("preview_scope_summary") == "Routing and local request checks only."
    assert "warnings" not in out
    assert "validation_not_performed" not in out
    assert "guardrails_preview" not in out
    assert "validation_scope" not in out
    assert "trade_gate_passed" not in out
    mock_market.assert_not_called()


def test_trade_place_dry_run_pending_preview_skips_order_send() -> None:
    with patch("mtdata.core.trading._place_pending_order") as mock_pending:
        out = trade_place(
            symbol="BTCUSD",
            volume=0.03,
            order_type="BUY_LIMIT",
            price=64500,
            expiration="GTC",
            dry_run=True,
            __cli_raw=True,
        )

    assert out.get("success") is True
    assert out.get("dry_run") is True
    assert out.get("no_action") is True
    assert out.get("pending") is True
    assert out.get("action") == "place_pending_order"
    assert out.get("actionability") == "preview_only"
    assert out.get("preview_scope_summary") == "Routing and local request checks only."
    assert "warnings" not in out
    assert "trade_gate_passed" not in out
    assert out.get("requested_price") == 64500
    assert out.get("expiration") == "GTC"
    mock_pending.assert_not_called()


def test_trade_place_require_sl_tp_flags_unprotected_market_fill() -> None:
    with patch(
        "mtdata.core.trading._place_market_order",
        return_value={
            "retcode": 10009,
            "sl_tp_result": {"status": "failed", "requested": {"sl": 64000.0, "tp": 68000.0}},
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


def test_trade_place_defaults_to_auto_closing_unprotected_market_fill() -> None:
    with patch(
        "mtdata.core.trading._place_market_order",
        return_value={
            "retcode": 10009,
            "sl_tp_result": {"status": "failed", "requested": {"sl": 64000.0, "tp": 68000.0}},
            "position_ticket": 456,
        },
    ), patch(
        "mtdata.core.trading._close_positions",
        return_value={"closed_count": 1},
    ) as mock_close:
        out = trade_place(
            symbol="BTCUSD",
            volume=0.03,
            order_type="BUY",
            stop_loss=64000,
            take_profit=68000,
            __cli_raw=True,
        )
    mock_close.assert_called_once()
    assert "error" in out
    assert out.get("require_sl_tp") is True
    assert out.get("auto_close_on_sl_tp_fail") is True
    assert out.get("protection_status") == "auto_closed_after_sl_tp_fail"
    assert "TP/SL protection could not be applied" in str(out.get("error"))


def test_trade_place_auto_close_attempts_recovery_on_sl_tp_fail() -> None:
    with patch(
        "mtdata.core.trading._place_market_order",
        return_value={
            "retcode": 10009,
            "sl_tp_result": {"status": "failed", "requested": {"sl": 64000.0, "tp": 68000.0}},
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
            "sl_tp_result": {
                "status": "applied",
                "requested": {"sl": 64000.0, "tp": 68000.0},
                "fallback_used": True,
            },
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
    assert out.get("error_code") == "ticket_not_found"
    assert out.get("ticket") == 123
    assert out.get("checked_scopes") == ["pending_orders"]
    assert "trade_get_pending" in str(out.get("suggestion"))


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
    assert out.get("error_code") == "ticket_not_found"
    assert out.get("ticket") == 123
    assert out.get("checked_scopes") == ["positions", "pending_orders"]
    assert "trade_get_open" in str(out.get("suggestion"))
