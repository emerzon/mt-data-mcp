from datetime import datetime, timezone
from types import SimpleNamespace

from mtdata.core.trading import (
    _normalize_order_type_input,
    _normalize_trade_comment,
    _server_time_naive_to_mt5_timestamp,
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


def test_server_time_naive_to_mt5_timestamp_strips_timezone():
    ts = _server_time_naive_to_mt5_timestamp(datetime(1970, 1, 1, 0, 1, 0, tzinfo=timezone.utc))
    assert ts == 60
