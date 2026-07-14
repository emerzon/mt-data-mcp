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
from mtdata.core.trading.requests import (
    TradeCloseRequest,
    TradeModifyRequest,
    TradePlaceRequest,
)
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


def test_trading_order_requests_expose_canonical_detail_field() -> None:
    fields = TradePlaceRequest.model_fields

    assert "detail" in fields
    assert "preview_detail" not in fields
    assert TradePlaceRequest(detail="full").detail == "full"
    assert TradePlaceRequest(detail="compact").detail == "compact"
    assert TradePlaceRequest(detail="standard").detail == "standard"
    assert TradePlaceRequest(detail="summary").detail == "summary"
    assert TradeModifyRequest(ticket=100, detail="summary").detail == "summary"
    assert TradeCloseRequest(detail="summary").detail == "summary"


def test_execution_request_dry_run_defaults() -> None:
    # All trade tools execute by default; pass dry_run=true to preview.
    assert TradePlaceRequest().dry_run is True
    assert TradeModifyRequest(ticket=100).dry_run is True
    assert TradeCloseRequest(ticket=100).dry_run is True


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
    assert out["error_code"] == "invalid_order_type"
    assert "BUY_STOP" in out["valid_values"]["order_type"]
    assert out["example"].startswith("mtdata-cli trade_place")
    mock_pending.assert_not_called()


def test_trade_place_routes_prefixed_market_order_type() -> None:
    with patch("mtdata.core.trading._place_market_order", return_value={"ok": True}) as mock_market:
        out = trade_place(
            symbol="BTCUSD",
            volume=0.03,
            order_type="BUY",
            require_sl_tp=False,
            dry_run=False,
            __cli_raw=True,
        )
        assert out == {"ok": True, "success": True}
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
            dry_run=False,
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
    assert out.get("success") is False
    assert out.get("error_code") == "trade_place_error"
    assert out.get("operation") == "trade_place"
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
            dry_run=False,
            __cli_raw=True,
        )
        assert out == {"ok": True, "success": True}
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
            dry_run=False,
            __cli_raw=True,
        )
    assert "error" in out
    assert out.get("require_sl_tp") is True
    assert set(out.get("missing", [])) == {"stop_loss", "take_profit"}
    assert "trade_risk_analyze" in out.get("hint", "")
    assert out.get("related_tools") == [
        "trade_risk_analyze",
        "forecast_barrier_optimize",
    ]
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
            dry_run=False,
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
            dry_run=False,
            __cli_raw=True,
        )
    assert out.get("retcode") == 10009
    mock_market.assert_called_once()


def test_trade_place_dry_run_market_preview_skips_order_send() -> None:
    with patch("mtdata.core.trading._place_market_order") as mock_market, patch(
        "mtdata.core.trading.build_trade_place_dry_run_preview",
        return_value={"bid": 64999.0, "ask": 65001.0, "estimated_fill_price": 65001.0},
    ) as mock_preview:
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
    assert out.get("would_send_order") is False
    assert out.get("pending") is False
    assert out.get("action") == "place_market_order"
    assert out["actionability"] == "preview_only"
    assert "preview_scope_summary" not in out
    assert "validation_not_performed" not in out
    assert "warnings" not in out
    assert out["validation_scope"] == "local_preview_plus_estimates"
    assert out["validation_passed"] is True
    assert out["preview_ok"] is True
    assert "broker_acceptance" in out["broker_validation_not_performed"]
    assert "trade_gate_passed" not in out
    assert out.get("message") == "Dry run only. No order was sent to MT5."
    assert out.get("bid") == 64999.0
    assert out.get("ask") == 65001.0
    assert out.get("estimated_fill_price") == 65001.0
    mock_preview.assert_called_once()
    mock_market.assert_not_called()


def test_trade_place_dry_run_market_preview_allows_missing_sl_tp() -> None:
    with patch("mtdata.core.trading._place_market_order") as mock_market, patch(
        "mtdata.core.trading.build_trade_place_dry_run_preview",
        return_value={"bid": 64999.0, "ask": 65001.0, "estimated_fill_price": 65001.0},
    ) as mock_preview:
        out = trade_place(
            symbol="BTCUSD",
            volume=0.03,
            order_type="BUY",
            dry_run=True,
            __cli_raw=True,
        )

    assert out.get("success") is True
    assert out.get("dry_run") is True
    assert out.get("require_sl_tp") is True
    assert "live submission with require_sl_tp=true would be rejected" in out.get(
        "dry_run_note", ""
    )
    assert out["validation"] == {
        "local_requirements_passed": False,
        "live_submission_eligible": False,
        "blockers": ["missing_stop_loss", "missing_take_profit"],
        "broker_validation_performed": False,
    }
    assert out.get("would_send_order") is False
    assert out["preview_ok"] is False
    assert out["validation_passed"] is False
    assert out.get("action") == "place_market_order"
    assert "requested_sl" not in out
    assert "requested_tp" not in out
    mock_preview.assert_called_once()
    assert mock_preview.call_args.kwargs["stop_loss"] is None
    assert mock_preview.call_args.kwargs["take_profit"] is None
    mock_market.assert_not_called()


def test_trade_place_dry_run_preview_detail_omits_safety_lists() -> None:
    with patch("mtdata.core.trading._place_market_order") as mock_market, patch(
        "mtdata.core.trading.build_trade_place_dry_run_preview",
        return_value={"bid": 64999.0, "ask": 65001.0, "estimated_fill_price": 65001.0},
    ):
        out = trade_place(
            symbol="BTCUSD",
            volume=0.03,
            order_type="BUY",
            stop_loss=64000,
            take_profit=68000,
            dry_run=True,
            detail="compact",
            __cli_raw=True,
        )

    assert out.get("success") is True
    assert out.get("dry_run") is True
    assert out.get("would_send_order") is False
    assert out["actionability"] == "preview_only"
    assert "preview_scope_summary" not in out
    assert "warnings" not in out
    assert "validation_not_performed" not in out
    assert "guardrails_preview" not in out
    assert out["guardrails_enabled"] is False
    assert out["validation_scope"] == "local_preview_plus_estimates"
    assert out["preview_ok"] is True
    assert out["validation_passed"] is True
    assert "trade_gate_passed" not in out
    mock_market.assert_not_called()


def test_trade_place_dry_run_standard_detail_keeps_validation_context() -> None:
    with patch("mtdata.core.trading._place_market_order") as mock_market, patch(
        "mtdata.core.trading.build_trade_place_dry_run_preview",
        return_value={"bid": 64999.0, "ask": 65001.0, "estimated_fill_price": 65001.0},
    ):
        out = trade_place(
            symbol="BTCUSD",
            volume=0.03,
            order_type="BUY",
            stop_loss=64000,
            take_profit=68000,
            dry_run=True,
            detail="standard",
            __cli_raw=True,
        )

    assert out.get("success") is True
    assert out.get("dry_run") is True
    assert out.get("actionability") == "preview_only"
    assert "preview_scope_summary" in out
    assert "warnings" in out
    assert "guardrails_preview" in out
    assert out["guardrails_enabled"] is False
    assert out["validation_scope"] == "local_preview_plus_estimates"
    assert "trade_gate_passed" not in out
    mock_market.assert_not_called()


def test_trade_place_dry_run_preview_error_uses_standard_error_shape() -> None:
    with patch("mtdata.core.trading._place_market_order") as mock_market, patch(
        "mtdata.core.trading.build_trade_place_dry_run_preview",
        return_value={"preview_error": "Failed to get current price for BTCUSD"},
    ):
        out = trade_place(
            symbol="BTCUSD",
            volume=0.03,
            order_type="BUY",
            stop_loss=64000,
            take_profit=68000,
            dry_run=True,
            __cli_raw=True,
        )

    assert out.get("success") is False
    assert out.get("preview_ok") is False
    assert out.get("error") == "Failed to get current price for BTCUSD"
    assert out.get("error_code") == "trade_preview_error"
    assert out.get("operation") == "trade_place"
    assert out.get("preview_error") == out.get("error")
    assert out.get("no_action") is True
    assert out.get("no_action_reason") == "dry_run_preview_error"
    assert out.get("would_send_order") is False
    mock_market.assert_not_called()


def test_trade_place_dry_run_rejects_invalid_live_protection_preview() -> None:
    with patch("mtdata.core.trading._place_market_order") as mock_market, patch(
        "mtdata.core.trading.build_trade_place_dry_run_preview",
        return_value={
            "bid": 65000.0,
            "ask": 65002.0,
            "estimated_fill_price": 65002.0,
            "sl_tp_valid": False,
            "sl_tp_error": "stop_loss must be below the live bid for BUY orders. sl=65100.0",
        },
    ):
        out = trade_place(
            symbol="BTCUSD",
            volume=0.03,
            order_type="BUY",
            stop_loss=65100,
            take_profit=68000,
            dry_run=True,
            __cli_raw=True,
        )

    assert out.get("success") is False
    assert out.get("preview_ok") is False
    assert out.get("dry_run") is True
    assert out.get("no_action") is True
    assert out.get("error_code") == "invalid_protection_levels"
    assert out.get("error") == out.get("sl_tp_error")
    assert "stop_loss must be below the live bid" in out.get("error", "")
    assert out.get("no_action_reason") == "dry_run_preview_error"
    mock_market.assert_not_called()


def test_trade_place_dry_run_rejects_identical_protection_before_mt5() -> None:
    with patch("mtdata.core.trading._place_market_order") as mock_market, patch(
        "mtdata.core.trading.build_trade_place_dry_run_preview"
    ) as mock_preview:
        out = trade_place(
            symbol="EURUSD",
            volume=0.01,
            order_type="BUY",
            stop_loss=1.0,
            take_profit=1.0,
            dry_run=True,
            detail="standard",
            __cli_raw=True,
        )

    assert out["success"] is False
    assert out["error_code"] == "invalid_protection_levels"
    assert out["error"] == "stop_loss and take_profit must be different prices."
    assert out["validation"]["local_requirements_passed"] is False
    assert out["validation"]["live_submission_eligible"] is False
    assert "invalid_protection_levels" in out["validation"]["blockers"]
    mock_preview.assert_not_called()
    mock_market.assert_not_called()


def test_trade_place_dry_run_preserves_mt5_connection_error() -> None:
    with patch("mtdata.core.trading._place_market_order") as mock_market, patch(
        "mtdata.core.trading.build_trade_place_dry_run_preview",
        return_value={
            "preview_error": "Failed to connect to MetaTrader5.",
            "preview_error_code": "mt5_connection_error",
        },
    ):
        out = trade_place(
            symbol="EURUSD",
            volume=0.01,
            order_type="BUY",
            stop_loss=1.08,
            take_profit=1.12,
            dry_run=True,
            detail="standard",
            __cli_raw=True,
        )

    assert out["success"] is False
    assert out["error_code"] == "mt5_connection_error"
    assert out["validation"]["local_requirements_passed"] is True
    assert out["validation"]["live_submission_eligible"] is False
    assert "mt5_connection_error" in out["validation"]["blockers"]
    mock_market.assert_not_called()


def test_trade_place_dry_run_rejects_bool_like_invalid_protection_preview() -> None:
    class BoolLikeFalse:
        def __bool__(self) -> bool:
            return False

    with patch("mtdata.core.trading._place_market_order") as mock_market, patch(
        "mtdata.core.trading.build_trade_place_dry_run_preview",
        return_value={
            "sl_tp_valid": BoolLikeFalse(),
            "sl_tp_error": "take_profit must be above the live ask for BUY orders.",
        },
    ):
        out = trade_place(
            symbol="BTCUSD",
            volume=0.03,
            order_type="BUY",
            stop_loss=64000,
            take_profit=63000,
            dry_run=True,
            __cli_raw=True,
        )

    assert out.get("success") is False
    assert out.get("preview_ok") is False
    assert out.get("dry_run") is True
    assert out.get("no_action") is True
    assert out.get("error_code") == "invalid_protection_levels"
    assert "take_profit must be above the live ask" in out.get("error", "")
    mock_market.assert_not_called()


def test_trade_place_dry_run_pending_preview_skips_order_send() -> None:
    with patch("mtdata.core.trading._place_pending_order") as mock_pending, patch(
        "mtdata.core.trading.build_trade_place_dry_run_preview",
        return_value={"bid": 64999.0, "ask": 65001.0, "entry_price": 64500.0},
    ):
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
    assert out["actionability"] == "preview_only"
    assert out["preview_ok"] is True
    assert out["validation_passed"] is True
    assert "preview_scope_summary" not in out
    assert "warnings" not in out
    assert "trade_gate_passed" not in out
    assert out.get("requested_price") == 64500
    assert out.get("expiration") == "GTC"
    mock_pending.assert_not_called()


def test_trade_place_rejects_market_order_with_price() -> None:
    with patch("mtdata.core.trading._place_market_order") as mock_market, patch(
        "mtdata.core.trading._place_pending_order"
    ) as mock_pending:
        out = trade_place(
            symbol="BTCUSD",
            volume=0.03,
            order_type="BUY",
            price=64500,
            stop_loss=64000,
            take_profit=68000,
            __cli_raw=True,
        )

    assert "Conflicting arguments" in out["error"]
    assert "order_type=BUY is a market order" in out["error"]
    assert "BUY_LIMIT/BUY_STOP" in out["error"]
    assert out.get("pending") is None
    assert out.get("order_type") == "BUY"
    assert out.get("price") == 64500
    mock_market.assert_not_called()
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
            dry_run=False,
            __cli_raw=True,
        )
    assert "error" in out
    assert out.get("require_sl_tp") is True
    assert out.get("protection_status") == "unprotected_position"
    assert any("CRITICAL" in str(w) for w in out.get("warnings", []))


def test_trade_place_preserves_scalar_warning_on_unprotected_market_fill() -> None:
    with patch(
        "mtdata.core.trading._place_market_order",
        return_value={
            "retcode": 10009,
            "warnings": "broker warning",
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
            dry_run=False,
            __cli_raw=True,
        )
    assert "broker warning" in out.get("warnings", [])
    assert "b" not in out.get("warnings", [])
    assert any("CRITICAL" in str(w) for w in out.get("warnings", []))


def test_trade_place_require_sl_tp_flags_unverified_market_fill() -> None:
    with patch(
        "mtdata.core.trading._place_market_order",
        return_value={
            "retcode": 10009,
            "warnings": ["verify protection"],
            "sl_tp_result": {"status": "unverified", "requested": {"sl": 64000.0, "tp": 68000.0}},
            "protection_status": "protection_unverified",
            "position_ticket_candidates": [456],
        },
    ), patch("mtdata.core.trading._close_positions") as mock_close:
        out = trade_place(
            symbol="BTCUSD",
            volume=0.03,
            order_type="BUY",
            stop_loss=64000,
            take_profit=68000,
            require_sl_tp=True,
            dry_run=False,
            __cli_raw=True,
        )
    mock_close.assert_called_once_with(
        ticket=456,
        comment="AUTO-CLOSE: TP/SL protection unresolved",
        deviation=20,
    )
    assert out.get("error") == "Order was executed, but TP/SL protection could not be verified."
    assert out.get("protection_status") == "protection_unverified"
    assert "verify protection" in out.get("warnings", [])
    assert any("AUTO-CLOSE FAILED" in warning for warning in out.get("warnings", [])), out


def test_trade_place_does_not_treat_auto_close_not_found_as_closed() -> None:
    with patch(
        "mtdata.core.trading._place_market_order",
        return_value={
            "retcode": 10009,
            "sl_tp_result": {"status": "failed", "requested": {"sl": 64000.0}},
            "position_ticket": 456,
        },
    ), patch(
        "mtdata.core.trading._close_positions",
        return_value={"error": "Position 456 not found"},
    ):
        out = trade_place(
            symbol="EURUSD",
            volume=1.0,
            order_type="BUY",
            stop_loss=1.0,
            take_profit=1.2,
            dry_run=False,
            __cli_raw=True,
        )

    assert out.get("protection_status") != "auto_closed_after_sl_tp_fail"
    assert out.get("auto_close_result", {}).get("already_closed") is not True
    assert any("AUTO-CLOSE FAILED" in warning for warning in out.get("warnings", [])), out


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
            dry_run=False,
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
            dry_run=False,
            __cli_raw=True,
        )
    mock_close.assert_called_once()
    assert out.get("auto_close_on_sl_tp_fail") is True
    assert out.get("protection_status") == "auto_closed_after_sl_tp_fail"
    assert out.get("auto_close_result", {}).get("retcode") == 10009


def test_trade_place_preserves_atomic_protection_status_without_fallback_fields() -> None:
    with patch(
        "mtdata.core.trading._place_market_order",
        return_value={
            "retcode": 10009,
            "sl_tp_result": {
                "status": "applied",
                "requested": {"sl": 64000.0, "tp": 68000.0},
            },
            "protection_status": "protected",
        },
    ):
        out = trade_place(
            symbol="BTCUSD",
            volume=0.03,
            order_type="BUY",
            stop_loss=64000,
            take_profit=68000,
            dry_run=False,
            __cli_raw=True,
        )
    assert out.get("protection_status") == "protected"
    assert "fallback_used" not in out.get("sl_tp_result", {})
    assert not any("fallback" in str(w).lower() for w in out.get("warnings", []))


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
