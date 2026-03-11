from datetime import datetime, timezone
from types import SimpleNamespace

from mtdata.core.trading_requests import TradePlaceRequest
from mtdata.core.trading_use_cases import run_trade_place
from mtdata.core.trading_comments import _comment_sanitization_info, _normalize_trade_comment
from mtdata.core.trading_time import _server_time_naive_to_mt5_timestamp
from mtdata.core.trading_validation import (
    _normalize_order_type_input,
    _validate_deviation,
    _validate_volume,
)


def test_normalize_order_type_rejects_bool_and_fractional_numeric():
    normalized, error = _normalize_order_type_input(True)
    assert normalized is None
    assert "Unsupported order_type" in error

    normalized, error = _normalize_order_type_input(2.5)
    assert normalized is None
    assert "Unsupported order_type" in error


def test_normalize_order_type_accepts_alias_and_prefixed_names():
    normalized, error = _normalize_order_type_input("long")
    assert error is None
    assert normalized == "BUY"

    normalized, error = _normalize_order_type_input("mt5.order_type_sell_limit")
    assert error is None
    assert normalized == "SELL_LIMIT"


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

    with caplog.at_level("INFO", logger="mtdata.core.trading_use_cases"):
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
