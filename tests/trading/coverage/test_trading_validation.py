"""Tests for input validation and normalization in mtdata.core.trading.

Covers:
- _normalize_order_type_input (pure)
- _server_time_naive_to_mt5_timestamp (pure)
- _to_server_time_naive (config-dependent)
- _validate_volume (pure with symbol_info)
- _validate_deviation (pure)
"""

import importlib
import os
import sys
from datetime import datetime, timezone
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

from mtdata.core.trading.time import (
    _server_time_naive_to_mt5_timestamp,
    _to_server_time_naive,
)
from mtdata.core.trading.validation import (
    _normalize_order_type_input,
    _validate_deviation,
    _validate_volume,
)


# ===================================================================
# Helpers
# ===================================================================

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
#  _normalize_order_type_input  (pure)
# ===================================================================

class TestNormalizeOrderTypeInput:
    """Exhaustive tests for _normalize_order_type_input."""

    def test_none_returns_error(self):
        t, err = _normalize_order_type_input(None)
        assert t is None and "required" in err

    @pytest.mark.parametrize("val", [
        0, 1, 2, 3, 4, 5,
    ])
    def test_numeric_ints_rejected(self, val):
        t, err = _normalize_order_type_input(val)
        assert t is None and "canonical string" in err

    @pytest.mark.parametrize("val", [
        0.0, 1.0, 5.0,
    ])
    def test_numeric_floats_whole_rejected(self, val):
        t, err = _normalize_order_type_input(val)
        assert t is None and "canonical string" in err

    def test_numeric_float_non_integer_rejected(self):
        t, err = _normalize_order_type_input(1.5)
        assert t is None and "Unsupported" in err

    def test_numeric_out_of_range(self):
        t, err = _normalize_order_type_input(99)
        assert t is None and "canonical string" in err

    def test_numeric_negative(self):
        t, err = _normalize_order_type_input(-1)
        assert t is None

    def test_inf_rejected(self):
        t, err = _normalize_order_type_input(float("inf"))
        assert t is None

    def test_nan_rejected(self):
        t, err = _normalize_order_type_input(float("nan"))
        assert t is None

    def test_bool_rejected(self):
        t, err = _normalize_order_type_input(True)
        assert t is None and "Unsupported" in err

    @pytest.mark.parametrize("text,expected", [
        ("BUY", "BUY"),
        ("sell", "SELL"),
        ("Buy_Limit", "BUY_LIMIT"),
        ("SELL-STOP", "SELL_STOP"),
        ("buy limit", "BUY_LIMIT"),
        ("sell  stop", "SELL_STOP"),
    ])
    def test_text_canonical_and_variations(self, text, expected):
        t, err = _normalize_order_type_input(text)
        assert t == expected and err is None

    def test_text_alias_long_rejected(self):
        t, err = _normalize_order_type_input("LONG")
        assert t is None and "Use BUY/SELL" in err

    def test_text_alias_short_rejected(self):
        t, err = _normalize_order_type_input("SHORT")
        assert t is None and "Use BUY/SELL" in err

    def test_text_mt5_prefix_rejected(self):
        t, err = _normalize_order_type_input("MT5.ORDER_TYPE_BUY_LIMIT")
        assert t is None and "Unsupported" in err

    def test_text_order_type_prefix_rejected(self):
        t, err = _normalize_order_type_input("ORDER_TYPE_SELL_STOP")
        assert t is None and "Unsupported" in err

    def test_empty_string(self):
        t, err = _normalize_order_type_input("")
        assert t is None and "required" in err

    def test_whitespace_only(self):
        t, err = _normalize_order_type_input("   ")
        assert t is None and "required" in err

    def test_unsupported_text(self):
        t, err = _normalize_order_type_input("MARKET_BUY")
        assert t is None and "Unsupported" in err

    def test_string_numeric_rejected(self):
        t, err = _normalize_order_type_input("0")
        assert t is None and "canonical string" in err

    def test_string_numeric_float_rejected(self):
        t, err = _normalize_order_type_input("3.0")
        assert t is None and "canonical string" in err

    def test_string_numeric_out_of_range(self):
        t, err = _normalize_order_type_input("99")
        assert t is None

    def test_string_inf_rejected(self):
        t, err = _normalize_order_type_input("inf")
        assert t is None

    def test_negative_float(self):
        t, err = _normalize_order_type_input(-0.5)
        assert t is None


# ===================================================================
#  _server_time_naive_to_mt5_timestamp (pure)
# ===================================================================

class TestServerTimeNaiveToMT5Timestamp:

    def test_epoch_zero(self):
        assert _server_time_naive_to_mt5_timestamp(datetime(1970, 1, 1)) == 0

    def test_known_timestamp(self):
        dt = datetime(2024, 1, 1, 0, 0, 0)
        expected = int((dt - datetime(1970, 1, 1)).total_seconds())
        assert _server_time_naive_to_mt5_timestamp(dt) == expected

    def test_strips_tzinfo(self):
        dt = datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        result = _server_time_naive_to_mt5_timestamp(dt)
        expected = int((datetime(2024, 6, 15, 12, 0, 0) - datetime(1970, 1, 1)).total_seconds())
        assert result == expected

    def test_truncates_microseconds(self):
        dt = datetime(2024, 1, 1, 0, 0, 0, 500000)
        result = _server_time_naive_to_mt5_timestamp(dt)
        assert result == int((datetime(2024, 1, 1) - datetime(1970, 1, 1)).total_seconds())


# ===================================================================
#  _to_server_time_naive (config-dependent)
# ===================================================================

class TestToServerTimeNaive:

    @patch("mtdata.core.trading.time.mt5_config")
    def test_naive_utc_no_tz_config(self, mock_config):
        mock_config.get_server_tz.side_effect = Exception("no tz")
        mock_config.get_client_tz.side_effect = Exception("no tz")
        mock_config.get_time_offset_seconds.return_value = 0
        dt = datetime(2024, 1, 1, 12, 0, 0)
        result = _to_server_time_naive(dt)
        assert result.tzinfo is None
        assert result == datetime(2024, 1, 1, 12, 0, 0)

    @patch("mtdata.core.trading.time.mt5_config")
    def test_applies_offset_seconds(self, mock_config):
        mock_config.get_server_tz.return_value = None
        mock_config.get_client_tz.return_value = None
        mock_config.get_time_offset_seconds.return_value = 3600  # +1h
        dt = datetime(2024, 1, 1, 12, 0, 0)
        result = _to_server_time_naive(dt)
        assert result.tzinfo is None
        assert result == datetime(2024, 1, 1, 13, 0, 0)

    @patch("mtdata.core.trading.time.mt5_config")
    def test_aware_input_without_tz_config(self, mock_config):
        mock_config.get_server_tz.return_value = None
        mock_config.get_client_tz.return_value = None
        mock_config.get_time_offset_seconds.return_value = 0
        dt = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        result = _to_server_time_naive(dt)
        assert result.tzinfo is None
        # Should convert to UTC first (12:00 +5 = 07:00 UTC), then add offset 0
        assert result == datetime(2024, 1, 1, 7, 0, 0)


# ===================================================================
#  _validate_volume (pure with symbol_info mock)
# ===================================================================

class TestValidateVolume:

    def test_valid_volume(self):
        vol, err = _validate_volume(0.05, _sym())
        assert err is None and vol == 0.05

    def test_non_numeric(self):
        vol, err = _validate_volume("abc", _sym())
        assert vol is None and "numeric" in err

    def test_zero_volume(self):
        vol, err = _validate_volume(0, _sym())
        assert vol is None and "positive" in err

    def test_negative_volume(self):
        vol, err = _validate_volume(-1, _sym())
        assert vol is None and "positive" in err

    def test_inf_volume(self):
        vol, err = _validate_volume(float("inf"), _sym())
        assert vol is None and "positive" in err or "finite" in err

    def test_nan_volume(self):
        vol, err = _validate_volume(float("nan"), _sym())
        assert vol is None

    def test_below_min(self):
        vol, err = _validate_volume(0.001, _sym(volume_min=0.01))
        assert vol is None and ">= 0.01" in err

    def test_above_max(self):
        vol, err = _validate_volume(200, _sym(volume_max=100))
        assert vol is None and "<= 100" in err

    def test_misaligned_step(self):
        vol, err = _validate_volume(0.015, _sym(volume_step=0.01))
        assert vol is None
        assert err == "volume must align to step 0.01. Try 0.01"

    def test_step_rounding_accepted(self):
        vol, err = _validate_volume(0.10, _sym(volume_step=0.01))
        assert err is None

    def test_step_alignment_without_lower_aligned_value_reports_minimum(self):
        vol, err = _validate_volume(0.005, _sym(volume_min=None, volume_step=0.01))
        assert vol is None
        assert err == "volume must align to step 0.01. Minimum aligned volume is 0.01"

    def test_none_constraints_accepted(self):
        si = SimpleNamespace(volume_min=None, volume_max=None, volume_step=None)
        vol, err = _validate_volume(5.0, si)
        assert err is None and vol == 5.0

    def test_zero_min_ignored(self):
        si = _sym(volume_min=0)
        vol, err = _validate_volume(0.01, si)
        assert err is None

    def test_zero_max_ignored(self):
        si = _sym(volume_max=0)
        vol, err = _validate_volume(0.01, si)
        assert err is None

    def test_zero_step_ignored(self):
        si = _sym(volume_step=0)
        vol, err = _validate_volume(0.05, si)
        assert err is None

    def test_non_numeric_constraints_handled(self):
        si = SimpleNamespace(volume_min="bad", volume_max="bad", volume_step="bad")
        vol, err = _validate_volume(1.0, si)
        assert err is None and vol == 1.0


# ===================================================================
#  _validate_deviation (pure)
# ===================================================================

class TestValidateDeviation:

    def test_valid(self):
        dev, err = _validate_deviation(20)
        assert dev == 20 and err is None

    def test_zero(self):
        dev, err = _validate_deviation(0)
        assert dev == 0 and err is None

    def test_negative(self):
        dev, err = _validate_deviation(-5)
        assert dev is None and ">= 0" in err

    def test_float_truncated(self):
        dev, err = _validate_deviation(10.9)
        assert dev == 10 and err is None

    def test_non_numeric(self):
        dev, err = _validate_deviation("abc")
        assert dev is None and "numeric" in err

    def test_none(self):
        dev, err = _validate_deviation(None)
        assert dev is None
