from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

from mtdata.core.trading.comments import (
    _comment_sanitization_info,
    _normalize_trade_comment,
)
from mtdata.core.trading.requests import TradeCloseRequest, TradePlaceRequest
from mtdata.core.trading.time import _server_time_naive_to_mt5_timestamp
from mtdata.core.trading.use_cases import run_trade_close, run_trade_place
from mtdata.core.trading.validation import (
    _normalize_order_type_input,
    _normalize_price_for_symbol,
    _retcode_is_done,
    _safe_float_attr,
    _trade_done_codes,
    _validate_deviation,
    _validate_live_protection_levels,
    _validate_volume,
)


def test_normalize_order_type_rejects_bool_and_fractional_numeric():
    normalized, error = _normalize_order_type_input(True)
    assert normalized is None
    assert "Unsupported order_type" in error

    normalized, error = _normalize_order_type_input(2.5)
    assert normalized is None
    assert "Unsupported order_type" in error


def test_normalize_order_type_rejects_alias_and_prefixed_names():
    normalized, error = _normalize_order_type_input("long")
    assert normalized is None
    assert "Unsupported order_type" in error

    normalized, error = _normalize_order_type_input("mt5.order_type_sell_limit")
    assert normalized is None
    assert "Unsupported order_type" in error


def test_validate_volume_handles_bad_symbol_constraints_gracefully():
    symbol_info = SimpleNamespace(volume_min=-1, volume_max="bad", volume_step=0)

    volume, error = _validate_volume(0.13, symbol_info)

    assert error is None
    assert volume == 0.13


def test_validate_deviation_rejects_negative_and_non_numeric():
    value, error = _validate_deviation(-1)
    assert value is None
    assert error == "deviation must be >= 0"

    value, error = _validate_deviation("abc")
    assert value is None
    assert error == "deviation must be numeric"


def test_trade_done_helpers_use_safe_int_attr_and_cached_codes():
    mt5 = SimpleNamespace(
        TRADE_RETCODE_DONE=True,
        TRADE_RETCODE_DONE_PARTIAL="10010",
    )

    done_codes = _trade_done_codes(mt5)

    assert done_codes == {10009, 10010}
    assert _retcode_is_done(mt5, "10010", done_codes) is True
    assert _retcode_is_done(mt5, 1, done_codes) is False


def test_normalize_price_for_symbol_accepts_negative_non_zero_values():
    normalized = _normalize_price_for_symbol(-37.634, point=0.01, digits=2)
    assert normalized == -37.63


def test_validate_live_protection_levels_accepts_negative_quotes():
    symbol_info = SimpleNamespace(point=0.01, trade_stops_level=0, trade_freeze_level=0)
    tick = SimpleNamespace(bid=-37.64, ask=-37.63)

    result = _validate_live_protection_levels(
        symbol_info=symbol_info,
        tick=tick,
        side="BUY",
        stop_loss=-37.80,
        take_profit=-37.20,
    )

    assert result is None


def test_run_trade_close_rejects_conflicting_profit_and_loss_filters():
    request = TradeCloseRequest(close_all=True, profit_only=True, loss_only=True)
    close_positions = MagicMock()
    cancel_pending = MagicMock()

    result = run_trade_close(
        request,
        close_positions=close_positions,
        cancel_pending=cancel_pending,
    )

    assert result["error"] == "profit_only and loss_only cannot both be true."
    close_positions.assert_not_called()
    cancel_pending.assert_not_called()


def test_run_trade_close_uses_history_lookup_when_ticket_is_already_closed():
    request = TradeCloseRequest(ticket=123)
    close_positions = MagicMock(return_value={"error": "Position 123 not found"})
    cancel_pending = MagicMock(return_value={"error": "Pending order 123 not found"})
    lookup_ticket_history = MagicMock(
        return_value={
            "message": "Ticket 123 was a Buy position that has already been closed at 2026-03-29 10:00 UTC. No action taken.",
            "no_action": True,
            "checked_scopes": ["positions", "pending_orders", "history_deals"],
        }
    )

    result = run_trade_close(
        request,
        close_positions=close_positions,
        cancel_pending=cancel_pending,
        lookup_ticket_history=lookup_ticket_history,
    )

    assert result["no_action"] is True
    assert "already been closed" in result["message"]
    assert result["checked_scopes"] == ["positions", "pending_orders", "history_deals"]
    lookup_ticket_history.assert_called_once_with(123)


def test_run_trade_close_passes_magic_filter_to_close_and_cancel_paths():
    request = TradeCloseRequest(ticket=123, magic=987)
    close_positions = MagicMock(return_value={"error": "Position 123 not found"})
    cancel_pending = MagicMock(return_value={"cancelled_count": 1})

    result = run_trade_close(
        request,
        close_positions=close_positions,
        cancel_pending=cancel_pending,
    )

    assert result["cancelled_count"] == 1
    close_positions.assert_called_once()
    cancel_pending.assert_called_once()
    assert close_positions.call_args.kwargs["magic"] == 987
    assert cancel_pending.call_args.kwargs["magic"] == 987


def test_normalize_trade_comment_applies_default_and_suffix_length_caps():
    comment = _normalize_trade_comment(None, default="DefaultComment", suffix="-MKT")
    assert comment == "DefaultComment-MKT"

    long_comment = _normalize_trade_comment("x" * 50, default="ignored", suffix="-TP")
    assert len(long_comment) == 31
    assert long_comment.endswith("-TP")


def test_normalize_trade_comment_sanitizes_special_characters():
    comment = _normalize_trade_comment(
        "Short: EV+, R:R 9.65",
        default="ignored",
    )
    assert ":" not in comment
    assert "," not in comment
    assert "+" not in comment
    assert comment == "Short EV R R 9.65"


def test_comment_sanitization_info_reports_changes():
    info = _comment_sanitization_info(
        "Short: ETS bearish, barrier EV+, R:R 9.65",
        "Short ETS bearish barrier EV R R 9.65",
    )
    assert info == {
        "requested": "Short: ETS bearish, barrier EV+, R:R 9.65",
        "applied": "Short ETS bearish barrier EV R R 9.65",
    }


def test_server_time_naive_to_mt5_timestamp_strips_timezone():
    ts = _server_time_naive_to_mt5_timestamp(datetime(1970, 1, 1, 0, 1, 0, tzinfo=timezone.utc))
    assert ts == 60


def test_run_trade_place_logs_finish_event(caplog):
    request = TradePlaceRequest(
        symbol="EURUSD",
        volume=0.1,
        order_type="BUY",
        require_sl_tp=False,
    )

    with caplog.at_level("DEBUG", logger="mtdata.core.trading.use_cases"):
        result = run_trade_place(
            request,
            normalize_order_type_input=lambda value: ("BUY", None),
            normalize_pending_expiration=lambda value: (value, False),
            prevalidate_trade_place_market_input=lambda symbol, volume: None,
            place_market_order=lambda **kwargs: {"success": True, "order_id": 7},
            place_pending_order=lambda **kwargs: {"success": True, "order_id": 8},
            close_positions=lambda **kwargs: {"closed_count": 1},
            safe_int_ticket=lambda value: value,
        )

    assert result == {"success": True, "order_id": 7}
    assert any(
        "event=finish operation=trade_place success=True" in record.message
        for record in caplog.records
    )


def test_run_trade_place_ignores_gtc_for_market_buy_sell_without_price():
    request = TradePlaceRequest(
        symbol="EURUSD",
        volume=0.1,
        order_type="BUY",
        expiration="GTC",
        require_sl_tp=False,
    )
    place_market_order = MagicMock(return_value={"success": True, "path": "market"})
    place_pending_order = MagicMock(return_value={"success": True, "path": "pending"})

    result = run_trade_place(
        request,
        normalize_order_type_input=lambda value: ("BUY", None),
        normalize_pending_expiration=lambda value: (None, True),
        prevalidate_trade_place_market_input=lambda symbol, volume: None,
        place_market_order=place_market_order,
        place_pending_order=place_pending_order,
        close_positions=lambda **kwargs: {"closed_count": 1},
        safe_int_ticket=lambda value: value,
    )

    assert result == {"success": True, "path": "market"}
    place_market_order.assert_called_once()
    place_pending_order.assert_not_called()


def test_run_trade_place_rejects_dated_market_expiration_without_price():
    request = TradePlaceRequest(
        symbol="EURUSD",
        volume=0.1,
        order_type="BUY",
        expiration="2026-04-01 12:00",
        require_sl_tp=False,
    )
    place_market_order = MagicMock(return_value={"success": True, "path": "market"})
    place_pending_order = MagicMock(return_value={"success": True, "path": "pending"})

    result = run_trade_place(
        request,
        normalize_order_type_input=lambda value: ("BUY", None),
        normalize_pending_expiration=lambda value: (1711972800, True),
        prevalidate_trade_place_market_input=lambda symbol, volume: None,
        place_market_order=place_market_order,
        place_pending_order=place_pending_order,
        close_positions=lambda **kwargs: {"closed_count": 1},
        safe_int_ticket=lambda value: value,
    )

    assert "error" in result
    assert "expiration only applies to pending orders placed with a price" in result["error"]
    place_market_order.assert_not_called()
    place_pending_order.assert_not_called()


def test_run_trade_place_dry_run_returns_preview_without_execution():
    request = TradePlaceRequest(
        symbol="EURUSD",
        volume=0.1,
        order_type="BUY",
        stop_loss=1.08,
        take_profit=1.12,
        dry_run=True,
        preview_detail="full",
    )
    place_market_order = MagicMock(return_value={"success": True, "path": "market"})
    place_pending_order = MagicMock(return_value={"success": True, "path": "pending"})

    result = run_trade_place(
        request,
        normalize_order_type_input=lambda value: ("BUY", None),
        normalize_pending_expiration=lambda value: (value, False),
        prevalidate_trade_place_market_input=lambda symbol, volume: {"error": "should not run"},
        place_market_order=place_market_order,
        place_pending_order=place_pending_order,
        close_positions=lambda **kwargs: {"closed_count": 1},
        safe_int_ticket=lambda value: value,
    )

    assert result["dry_run"] is True
    assert result["no_action"] is True
    assert result["pending"] is False
    assert result["action"] == "place_market_order"
    assert result["validation_scope"] == "request_routing_only"
    assert result["trade_gate_passed"] is False
    assert result["actionability"] == "preview_only"
    assert "validation_not_performed" in result
    assert "broker_acceptance" in result["validation_not_performed"]
    assert result["requested_sl"] == 1.08
    assert result["requested_tp"] == 1.12
    place_market_order.assert_not_called()
    place_pending_order.assert_not_called()


def test_run_trade_place_uses_candidate_tickets_when_position_ticket_is_missing():
    request = TradePlaceRequest(
        symbol="EURUSD",
        volume=0.1,
        order_type="BUY",
        stop_loss=1.08,
        take_profit=1.12,
        auto_close_on_sl_tp_fail=False,
    )

    result = run_trade_place(
        request,
        normalize_order_type_input=lambda value: ("BUY", None),
        normalize_pending_expiration=lambda value: (value, False),
        prevalidate_trade_place_market_input=lambda symbol, volume: None,
        place_market_order=lambda **kwargs: {
            "retcode": 10009,
            "sl_tp_result": {"status": "failed", "requested": {"sl": 1.08, "tp": 1.12}},
            "position_ticket_candidates": [111, 222],
        },
        place_pending_order=lambda **kwargs: {"success": True, "path": "pending"},
        close_positions=lambda **kwargs: {"closed_count": 1},
        safe_int_ticket=lambda value: value,
    )

    assert result["protection_status"] == "unprotected_position"
    assert any("trade_modify 111" in str(w) for w in result.get("warnings", []))
    assert any("trade_get_open" in str(w) for w in result.get("warnings", []))


def test_run_trade_place_auto_close_uses_candidate_ticket_when_primary_is_missing():
    request = TradePlaceRequest(
        symbol="EURUSD",
        volume=0.1,
        order_type="BUY",
        stop_loss=1.08,
        take_profit=1.12,
        auto_close_on_sl_tp_fail=True,
    )
    close_calls = []

    def close_positions(**kwargs):
        close_calls.append(kwargs)
        return {"closed_count": 1}

    result = run_trade_place(
        request,
        normalize_order_type_input=lambda value: ("BUY", None),
        normalize_pending_expiration=lambda value: (value, False),
        prevalidate_trade_place_market_input=lambda symbol, volume: None,
        place_market_order=lambda **kwargs: {
            "retcode": 10009,
            "sl_tp_result": {"status": "failed", "requested": {"sl": 1.08, "tp": 1.12}},
            "position_ticket_candidates": [111, 222],
        },
        place_pending_order=lambda **kwargs: {"success": True, "path": "pending"},
        close_positions=close_positions,
        safe_int_ticket=lambda value: value,
    )

    assert close_calls and close_calls[0]["ticket"] == 111
    assert result["auto_close_on_sl_tp_fail"] is True
    assert result["protection_status"] == "auto_closed_after_sl_tp_fail"
    assert result["auto_close_result"]["closed_count"] == 1


def test_run_trade_place_without_any_ticket_guides_to_trade_get_open():
    request = TradePlaceRequest(
        symbol="EURUSD",
        volume=0.1,
        order_type="BUY",
        stop_loss=1.08,
        take_profit=1.12,
    )

    result = run_trade_place(
        request,
        normalize_order_type_input=lambda value: ("BUY", None),
        normalize_pending_expiration=lambda value: (value, False),
        prevalidate_trade_place_market_input=lambda symbol, volume: None,
        place_market_order=lambda **kwargs: {
            "retcode": 10009,
            "sl_tp_result": {"status": "failed", "requested": {"sl": 1.08, "tp": 1.12}},
        },
        place_pending_order=lambda **kwargs: {"success": True, "path": "pending"},
        close_positions=lambda **kwargs: {"closed_count": 1},
        safe_int_ticket=lambda value: value,
    )

    assert result["protection_status"] == "unprotected_position"
    assert any("trade_get_open" in str(w) for w in result.get("warnings", []))
    assert not any("trade_modify now" in str(w) for w in result.get("warnings", []))


# ---------------------------------------------------------------------------
# _safe_float_attr tests
# ---------------------------------------------------------------------------

class TestSafeFloatAttr:
    def test_normal_float_attribute(self):
        obj = SimpleNamespace(price=1.2345)
        assert _safe_float_attr(obj, "price") == 1.2345

    def test_int_attribute_coerced(self):
        obj = SimpleNamespace(volume=100)
        assert _safe_float_attr(obj, "volume") == 100.0

    def test_string_numeric_coerced(self):
        obj = SimpleNamespace(value="3.14")
        assert _safe_float_attr(obj, "value") == 3.14

    def test_missing_attribute_returns_default(self):
        obj = SimpleNamespace()
        assert _safe_float_attr(obj, "price") == 0.0
        assert _safe_float_attr(obj, "price", -1.0) == -1.0

    def test_none_attribute_returns_default(self):
        obj = SimpleNamespace(profit=None)
        assert _safe_float_attr(obj, "profit") == 0.0

    def test_bool_attribute_returns_default(self):
        obj = SimpleNamespace(flag=True)
        assert _safe_float_attr(obj, "flag") == 0.0

    def test_nan_returns_default(self):
        obj = SimpleNamespace(price=float("nan"))
        assert _safe_float_attr(obj, "price") == 0.0

    def test_inf_returns_default(self):
        obj = SimpleNamespace(price=float("inf"))
        assert _safe_float_attr(obj, "price") == 0.0

    def test_negative_inf_returns_default(self):
        obj = SimpleNamespace(price=float("-inf"))
        assert _safe_float_attr(obj, "price") == 0.0

    def test_non_numeric_string_returns_default(self):
        obj = SimpleNamespace(price="not_a_number")
        assert _safe_float_attr(obj, "price") == 0.0

    def test_zero_is_valid(self):
        obj = SimpleNamespace(profit=0.0)
        assert _safe_float_attr(obj, "profit") == 0.0

    def test_negative_is_valid(self):
        obj = SimpleNamespace(profit=-42.5)
        assert _safe_float_attr(obj, "profit") == -42.5

    def test_getattr_exception_returns_default(self):
        class Broken:
            def __getattr__(self, name):
                raise RuntimeError("boom")
        assert _safe_float_attr(Broken(), "price") == 0.0

    def test_custom_default(self):
        obj = SimpleNamespace()
        assert _safe_float_attr(obj, "bid", 99.0) == 99.0
