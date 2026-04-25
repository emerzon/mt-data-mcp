"""Comprehensive tests for mtdata.core.trading – targeting ~418 uncovered lines.

Covers:
- _normalize_order_type_input (pure)
- _to_server_time_naive (config-dependent)
- _server_time_naive_to_mt5_timestamp (pure)
- _normalize_pending_expiration (mixed)
- _validate_volume (pure with symbol_info)
- _validate_deviation (pure)
- _normalize_trade_comment (pure)
- trade_place (dispatch logic)
- trade_modify (dispatch logic)
- trade_close (dispatch logic)
- trade_account_info / trade_history / trade_get_open / trade_get_pending (MT5)
- _place_market_order / _place_pending_order (MT5)
- _modify_position / _modify_pending_order (MT5)
- _close_positions / _cancel_pending (MT5)
- trade_risk_analyze (MT5)
"""

import importlib
import math
import os
import sys
from collections import namedtuple
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Ensure src/ is importable and stub MetaTrader5 before importing the module
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

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

from mtdata.core.trading import (
    trade_account_info,
)
from mtdata.core.trading import (
    trade_close as _trade_close_tool,
)
from mtdata.core.trading import (
    trade_modify as _trade_modify_tool,
)
from mtdata.core.trading import (
    trade_place as _trade_place_tool,
)
from mtdata.core.trading import (
    trade_risk_analyze as _trade_risk_analyze_tool,
)
from mtdata.core.trading.comments import (
    _comment_row_metadata,
    _normalize_close_trade_comment,
    _normalize_trade_comment,
)
from mtdata.core.trading.execution import (
    _deal_history_sort_key,
    _resolve_closed_deal_from_history,
)
from mtdata.core.trading.requests import (
    TradeCloseRequest,
    TradeModifyRequest,
    TradePlaceRequest,
    TradeRiskAnalyzeRequest,
)
from mtdata.core.trading.time import (
    _GTC_EXPIRATION_TOKENS,
    _normalize_pending_expiration,
    _server_time_naive_to_mt5_timestamp,
    _to_server_time_naive,
)
from mtdata.core.trading.validation import (
    _ORDER_TYPE_NUMERIC_MAP,
    _SUPPORTED_ORDER_TYPES,
    _normalize_order_type_input,
    _validate_deviation,
    _validate_volume,
)


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


def trade_close(**kwargs):
    raw_output = bool(kwargs.pop("__cli_raw", False))
    request = kwargs.pop("request", None)
    if request is None:
        request = TradeCloseRequest(**kwargs)
    return _trade_close_tool(request=request, __cli_raw=raw_output)


def trade_risk_analyze(**kwargs):
    raw_output = bool(kwargs.pop("__cli_raw", False))
    request = kwargs.pop("request", None)
    if request is None:
        request = TradeRiskAnalyzeRequest(**kwargs)
    return _trade_risk_analyze_tool(request=request, __cli_raw=raw_output)


def _unwrap_mcp(result):
    """MCP tool decorator serialises dicts to strings; return the raw dict
    when the underlying function is patched to return one, otherwise return
    the string for assertion with ``in``."""
    if isinstance(result, dict):
        return result
    return str(result)


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


def _tick(bid=1.10000, ask=1.10020):
    return SimpleNamespace(bid=bid, ask=ask)


def _order_result(retcode=10009, deal=1, order=1, volume=0.01, price=1.1,
                  bid=1.1, ask=1.1002, comment="done", request_id=1):
    return SimpleNamespace(
        retcode=retcode, deal=deal, order=order, volume=volume,
        price=price, bid=bid, ask=ask, comment=comment, request_id=request_id,
    )


def test_deal_history_sort_key_normalizes_millisecond_timestamps():
    earlier = SimpleNamespace(time_msc=1744876700000)
    later = SimpleNamespace(time=1744876800)

    assert _deal_history_sort_key(earlier) == 1744876700.0
    assert _deal_history_sort_key(later) == 1744876800.0
    assert max([earlier, later], key=_deal_history_sort_key) is later


def test_resolve_closed_deal_from_history_queries_recent_window_in_utc():
    matched_row = SimpleNamespace(
        ticket=101,
        order=202,
        position_id=303,
        position_by_id=None,
        position=None,
        time=1700000000,
    )
    mt5 = SimpleNamespace(history_deals_get=MagicMock(return_value=[matched_row]))

    result = _resolve_closed_deal_from_history(
        mt5,
        result=SimpleNamespace(deal=101, order=202),
        position=SimpleNamespace(ticket=303),
        closed_at_utc=datetime(2026, 3, 1, 11, 0, tzinfo=timezone.utc),
    )

    called_from, called_to = mt5.history_deals_get.call_args.args
    assert called_from == datetime(2026, 3, 1, 10, 55, tzinfo=timezone.utc)
    assert called_to == datetime(2026, 3, 1, 11, 1, tzinfo=timezone.utc)
    assert result is matched_row


def _position(ticket=1, symbol="EURUSD", type_=0, volume=0.01,
              price_open=1.1, sl=1.09, tp=1.12, profit=50.0,
              price_current=1.105, comment="", magic=234000,
              time=1700000000, time_update=1700000000, swap=0.0,
              identifier=None, position_id=None, order=None, deal=None):
    return SimpleNamespace(
        ticket=ticket, symbol=symbol, type=type_, volume=volume,
        price_open=price_open, sl=sl, tp=tp, profit=profit,
        price_current=price_current, comment=comment, magic=magic,
        time=time, time_update=time_update, swap=swap,
        identifier=identifier, position_id=position_id, order=order, deal=deal,
        _asdict=lambda: {
            "ticket": ticket, "symbol": symbol, "type": type_, "volume": volume,
            "price_open": price_open, "sl": sl, "tp": tp, "profit": profit,
            "price_current": price_current, "comment": comment, "magic": magic,
            "time": time, "time_update": time_update, "swap": swap,
            "identifier": identifier, "position_id": position_id, "order": order, "deal": deal,
        },
    )


def _pending_order(ticket=100, symbol="EURUSD", type_=2, volume=0.01,
                   price_open=1.09, sl=1.08, tp=1.11, time_setup=1700000000,
                   time_expiration=0, type_time=0, comment="", magic=234000):
    return SimpleNamespace(
        ticket=ticket, symbol=symbol, type=type_, volume=volume,
        price_open=price_open, sl=sl, tp=tp, time_setup=time_setup,
        time_expiration=time_expiration, type_time=type_time,
        comment=comment, magic=magic,
        _asdict=lambda: {
            "ticket": ticket, "symbol": symbol, "type": type_, "volume": volume,
            "price_open": price_open, "sl": sl, "tp": tp,
            "time_setup": time_setup, "time_expiration": time_expiration,
            "comment": comment, "magic": magic,
        },
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

    @pytest.mark.parametrize("val,expected", [
        (0, "BUY"), (1, "SELL"), (2, "BUY_LIMIT"),
        (3, "SELL_LIMIT"), (4, "BUY_STOP"), (5, "SELL_STOP"),
    ])
    def test_numeric_ints(self, val, expected):
        t, err = _normalize_order_type_input(val)
        assert t == expected and err is None

    @pytest.mark.parametrize("val,expected", [
        (0.0, "BUY"), (1.0, "SELL"), (5.0, "SELL_STOP"),
    ])
    def test_numeric_floats_whole(self, val, expected):
        t, err = _normalize_order_type_input(val)
        assert t == expected and err is None

    def test_numeric_float_non_integer_rejected(self):
        t, err = _normalize_order_type_input(1.5)
        assert t is None and "Unsupported" in err

    def test_numeric_out_of_range(self):
        t, err = _normalize_order_type_input(99)
        assert t is None and "0..5" in err

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

    def test_text_alias_long(self):
        t, err = _normalize_order_type_input("LONG")
        assert t == "BUY"

    def test_text_alias_short(self):
        t, err = _normalize_order_type_input("SHORT")
        assert t == "SELL"

    def test_text_mt5_prefix(self):
        t, err = _normalize_order_type_input("MT5.ORDER_TYPE_BUY_LIMIT")
        assert t == "BUY_LIMIT"

    def test_text_order_type_prefix(self):
        t, err = _normalize_order_type_input("ORDER_TYPE_SELL_STOP")
        assert t == "SELL_STOP"

    def test_empty_string(self):
        t, err = _normalize_order_type_input("")
        assert t is None and "required" in err

    def test_whitespace_only(self):
        t, err = _normalize_order_type_input("   ")
        assert t is None and "required" in err

    def test_unsupported_text(self):
        t, err = _normalize_order_type_input("MARKET_BUY")
        assert t is None and "Unsupported" in err

    def test_string_numeric_coerced(self):
        t, err = _normalize_order_type_input("0")
        assert t == "BUY"

    def test_string_numeric_float(self):
        t, err = _normalize_order_type_input("3.0")
        assert t == "SELL_LIMIT"

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
        dt = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone(timedelta(hours=5)))
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


# ===================================================================
#  _normalize_trade_comment (pure)
# ===================================================================

class TestNormalizeTradeComment:

    def test_default_used_when_empty(self):
        assert _normalize_trade_comment("", default="MyDef") == "MyDef"

    def test_none_uses_default(self):
        assert _normalize_trade_comment(None, default="MyDef") == "MyDef"

    def test_custom_comment(self):
        assert _normalize_trade_comment("hello", default="x") == "hello"

    def test_suffix_appended(self):
        assert _normalize_trade_comment("base", default="x", suffix="!") == "base!"

    def test_truncation_at_31(self):
        long = "a" * 50
        result = _normalize_trade_comment(long, default="x")
        assert len(result) == 31

    def test_suffix_truncation(self):
        base = "a" * 30
        result = _normalize_trade_comment(base, default="x", suffix="XX")
        assert len(result) <= 31

    def test_empty_default_falls_back_to_mcp(self):
        result = _normalize_trade_comment(None, default="")
        assert result == "MCP"


class TestNormalizeCloseTradeComment:

    def test_truncation_at_close_limit(self):
        result = _normalize_close_trade_comment("a" * 50)
        assert len(result) == 24

    def test_very_long_suffix(self):
        result = _normalize_trade_comment("b", default="x", suffix="S" * 40)
        assert result == "b" + ("S" * 30)

    def test_comment_row_metadata_marks_only_limit_length_as_truncated(self):
        short_meta = _comment_row_metadata("audit short")
        exact_limit_meta = _comment_row_metadata("x" * 31)

        assert short_meta["comment_may_be_truncated"] is False
        assert exact_limit_meta["comment_may_be_truncated"] is True


# ===================================================================
#  _normalize_pending_expiration (mixed: some paths need config mock)
# ===================================================================

class TestNormalizePendingExpiration:

    def test_none_not_explicit(self):
        val, explicit = _normalize_pending_expiration(None)
        assert val is None and explicit is False

    def test_empty_string_not_explicit(self):
        val, explicit = _normalize_pending_expiration("")
        assert val is None and explicit is False

    def test_whitespace_string_not_explicit(self):
        val, explicit = _normalize_pending_expiration("   ")
        assert val is None and explicit is False

    @pytest.mark.parametrize("token", list(_GTC_EXPIRATION_TOKENS))
    def test_gtc_tokens(self, token):
        val, explicit = _normalize_pending_expiration(token)
        assert val is None and explicit is True

    def test_gtc_case_insensitive(self):
        val, explicit = _normalize_pending_expiration("gtc")
        assert val is None and explicit is True

    def test_negative_number_clears(self):
        val, explicit = _normalize_pending_expiration(-1)
        assert val is None and explicit is True

    def test_zero_clears(self):
        val, explicit = _normalize_pending_expiration(0)
        assert val is None and explicit is True

    def test_inf_clears(self):
        val, explicit = _normalize_pending_expiration(float("inf"))
        assert val is None and explicit is True

    @patch("mtdata.core.trading.time._to_server_time_naive", side_effect=lambda dt: dt.replace(tzinfo=None) if dt.tzinfo else dt)
    def test_datetime_returns_timestamp(self, _mock):
        dt = datetime(2025, 6, 1, 12, 0, 0)
        val, explicit = _normalize_pending_expiration(dt)
        assert isinstance(val, int) and explicit is True

    @patch("mtdata.core.trading.time._to_server_time_naive", side_effect=lambda dt: dt.replace(tzinfo=None) if dt.tzinfo else dt)
    def test_positive_numeric_epoch(self, _mock):
        ts = 1717200000.0  # some future epoch
        val, explicit = _normalize_pending_expiration(ts)
        assert isinstance(val, int) and explicit is True

    @patch("mtdata.core.trading.time._to_server_time_naive", side_effect=lambda dt: dt.replace(tzinfo=None) if dt.tzinfo else dt)
    def test_string_numeric_epoch(self, _mock):
        val, explicit = _normalize_pending_expiration("1717200000")
        assert isinstance(val, int) and explicit is True

    @patch("mtdata.core.trading.time._to_server_time_naive", side_effect=lambda dt: dt.replace(tzinfo=None) if dt.tzinfo else dt)
    def test_iso8601_string(self, _mock):
        val, explicit = _normalize_pending_expiration("2025-06-01T12:00:00")
        assert isinstance(val, int) and explicit is True

    def test_string_negative_numeric(self):
        # "-1" may be parsed by dateparser as a relative date; the key point
        # is that a negative numeric string is handled without raising.
        val, explicit = _normalize_pending_expiration("-1")
        assert explicit is True

    def test_unsupported_type_raises(self):
        with pytest.raises(TypeError, match="Unsupported expiration type"):
            _normalize_pending_expiration([1, 2, 3])

    @patch("mtdata.core.trading.time._to_server_time_naive", side_effect=lambda dt: dt.replace(tzinfo=None) if dt.tzinfo else dt)
    def test_relative_time_30m(self, _mock):
        val, explicit = _normalize_pending_expiration("30m")
        assert isinstance(val, int) and explicit is True

    @patch("mtdata.core.trading.time._to_server_time_naive", side_effect=lambda dt: dt.replace(tzinfo=None) if dt.tzinfo else dt)
    def test_relative_time_in_2_hours(self, _mock):
        val, explicit = _normalize_pending_expiration("in 2 hours")
        assert isinstance(val, int) and explicit is True

    @patch("mtdata.core.trading.time._to_server_time_naive", side_effect=lambda dt: dt.replace(tzinfo=None) if dt.tzinfo else dt)
    def test_relative_time_1d(self, _mock):
        val, explicit = _normalize_pending_expiration("1d")
        assert isinstance(val, int) and explicit is True

    @patch("mtdata.core.trading.time._to_server_time_naive", side_effect=lambda dt: dt.replace(tzinfo=None) if dt.tzinfo else dt)
    def test_relative_time_seconds(self, _mock):
        val, explicit = _normalize_pending_expiration("60s")
        assert isinstance(val, int) and explicit is True

    @patch("mtdata.core.trading.time._to_server_time_naive", side_effect=lambda dt: dt.replace(tzinfo=None) if dt.tzinfo else dt)
    def test_relative_time_weeks(self, _mock):
        val, explicit = _normalize_pending_expiration("2w")
        assert isinstance(val, int) and explicit is True

    def test_string_inf_clears(self):
        val, explicit = _normalize_pending_expiration("inf")
        assert val is None and explicit is True

    def test_quoted_gtc(self):
        val, explicit = _normalize_pending_expiration('"GTC"')
        assert val is None and explicit is True


# ===================================================================
#  trade_place  (dispatch / validation – no MT5 calls for validation errors)
# ===================================================================

class TestTradePlace:

    def test_missing_symbol(self):
        result = _unwrap_mcp(trade_place(symbol=None, volume=0.01, order_type="BUY"))
        assert "symbol" in result

    def test_missing_volume(self):
        result = _unwrap_mcp(trade_place(symbol="EURUSD", volume=None, order_type="BUY"))
        assert "volume" in result

    def test_missing_order_type(self):
        result = _unwrap_mcp(trade_place(symbol="EURUSD", volume=0.01, order_type=None))
        assert "order_type" in result

    def test_all_missing(self):
        result = _unwrap_mcp(trade_place())
        for field in ("symbol", "volume", "order_type"):
            assert field in result

    def test_empty_order_type_string(self):
        result = _unwrap_mcp(trade_place(symbol="EURUSD", volume=0.01, order_type="  "))
        assert "order_type" in result

    def test_invalid_order_type(self):
        result = _unwrap_mcp(trade_place(symbol="EURUSD", volume=0.01, order_type="GARBAGE"))
        assert "Unsupported" in result

    def test_pending_without_price_returns_error(self):
        result = _unwrap_mcp(trade_place(symbol="EURUSD", volume=0.01, order_type="BUY_LIMIT", price=None))
        assert "price" in result.lower()

    def test_invalid_expiration_returns_error(self):
        result = _unwrap_mcp(trade_place(
            symbol="EURUSD", volume=0.01, order_type="BUY_LIMIT",
            price=1.09, expiration="not-a-date-at-all-xyz",
        ))
        assert "error" in result.lower() or "unsupported" in result.lower()

    @patch("mtdata.core.trading._place_market_order")
    def test_dispatches_to_market(self, mock_market):
        mock_market.return_value = {"retcode": 10009}
        trade_place(symbol="EURUSD", volume=0.01, order_type="BUY", require_sl_tp=False)
        mock_market.assert_called_once()

    @patch("mtdata.core.trading._place_pending_order")
    def test_dispatches_to_pending_explicit_type(self, mock_pending):
        mock_pending.return_value = {"success": True}
        trade_place(symbol="EURUSD", volume=0.01, order_type="BUY_LIMIT", price=1.09)
        mock_pending.assert_called_once()

    @patch("mtdata.core.trading._place_pending_order")
    def test_dispatches_to_pending_with_price(self, mock_pending):
        mock_pending.return_value = {"success": True}
        trade_place(symbol="EURUSD", volume=0.01, order_type="BUY", price=1.09)
        mock_pending.assert_called_once()

    @patch("mtdata.core.trading._place_pending_order")
    def test_dispatches_to_pending_with_expiration(self, mock_pending):
        mock_pending.return_value = {"success": True}
        trade_place(
            symbol="EURUSD", volume=0.01, order_type="BUY",
            price=1.09, expiration="GTC",
        )
        mock_pending.assert_called_once()


# ===================================================================
#  trade_close (dispatch)
# ===================================================================

class TestTradeClose:

    @patch("mtdata.core.trading._cancel_pending")
    @patch("mtdata.core.trading._close_positions")
    def test_profit_only_routes_to_positions_only(self, mock_close, mock_cancel):
        mock_close.return_value = {"closed_count": 1}
        trade_close(ticket=123, profit_only=True)
        mock_close.assert_called_once()
        mock_cancel.assert_not_called()

    @patch("mtdata.core.trading._cancel_pending")
    @patch("mtdata.core.trading._close_positions")
    def test_loss_only_routes_to_positions_only(self, mock_close, mock_cancel):
        mock_close.return_value = {"closed_count": 1}
        trade_close(ticket=123, loss_only=True)
        mock_close.assert_called_once()
        mock_cancel.assert_not_called()

    @patch("mtdata.core.trading._cancel_pending")
    @patch("mtdata.core.trading._close_positions")
    def test_ticket_falls_back_to_cancel_pending(self, mock_close, mock_cancel):
        mock_close.return_value = {"error": "Position 123 not found"}
        mock_cancel.return_value = {"cancelled_count": 1}
        out = trade_close(ticket=123, __cli_raw=True)
        mock_close.assert_called_once()
        mock_cancel.assert_called_once_with(ticket=123, symbol=None, comment=None)
        assert out["cancelled_count"] == 1

    @patch("mtdata.core.trading._cancel_pending")
    @patch("mtdata.core.trading._close_positions")
    def test_ticket_missing_reports_both_scopes(self, mock_close, mock_cancel):
        mock_close.return_value = {"error": "Position 123 not found"}
        mock_cancel.return_value = {"error": "Pending order 123 not found"}
        result = _unwrap_mcp(trade_close(ticket=123))
        if isinstance(result, dict):
            assert "position or pending order" in str(result.get("error", "")).lower()
            assert result.get("checked_scopes") == ["positions", "pending_orders"]
        else:
            assert "position or pending order" in result.lower()
            assert "pending_orders" in result.lower()

    @patch("mtdata.core.trading._cancel_pending")
    @patch("mtdata.core.trading._close_positions")
    def test_dispatches_to_close_positions(self, mock_close, mock_cancel):
        mock_close.return_value = {"closed_count": 1}
        trade_close(ticket=123)
        mock_close.assert_called_once()
        mock_cancel.assert_not_called()

    @patch("mtdata.core.trading._cancel_pending")
    @patch("mtdata.core.trading._close_positions")
    def test_partial_close_by_ticket_passes_volume(self, mock_close, mock_cancel):
        mock_close.return_value = {"closed_count": 1}
        trade_close(ticket=123, volume=0.05)
        mock_close.assert_called_once_with(
            ticket=123,
            symbol=None,
            volume=0.05,
            profit_only=False,
            loss_only=False,
            close_priority=None,
            comment=None,
            deviation=20,
        )
        mock_cancel.assert_not_called()

    @patch("mtdata.core.trading._cancel_pending")
    @patch("mtdata.core.trading._close_positions")
    def test_dry_run_ticket_preview_skips_execution(self, mock_close, mock_cancel):
        out = trade_close(ticket=123, volume=0.05, dry_run=True, __cli_raw=True)

        assert out["success"] is True
        assert out["dry_run"] is True
        assert out["actionability"] == "preview_only"
        assert out["operation"] == "partial_close_position"
        assert out["ticket"] == 123
        assert out["volume"] == 0.05
        assert out["would_send_order"] is False
        assert "realized_pnl" in out["not_estimated"]
        mock_close.assert_not_called()
        mock_cancel.assert_not_called()

    @patch("mtdata.core.trading._cancel_pending")
    @patch("mtdata.core.trading._close_positions")
    def test_partial_close_requires_ticket(self, mock_close, mock_cancel):
        out = _unwrap_mcp(trade_close(symbol="EURUSD", volume=0.05))
        if isinstance(out, dict):
            assert "volume is only supported" in str(out.get("error", "")).lower()
        else:
            assert "volume is only supported" in out.lower()
        mock_close.assert_not_called()
        mock_cancel.assert_not_called()

    @patch("mtdata.core.trading._cancel_pending")
    @patch("mtdata.core.trading._close_positions")
    def test_partial_close_ticket_not_found_does_not_cancel_pending(self, mock_close, mock_cancel):
        mock_close.return_value = {"error": "Position 123 not found"}
        out = _unwrap_mcp(trade_close(ticket=123, volume=0.05))
        if isinstance(out, dict):
            assert "partial close volume only applies to open positions" in str(out.get("error", "")).lower()
            assert out.get("checked_scopes") == ["positions"]
        else:
            assert "partial close volume only applies to open positions" in out.lower()
        mock_cancel.assert_not_called()

    @patch("mtdata.core.trading._cancel_pending")
    @patch("mtdata.core.trading._close_positions")
    def test_symbol_close_requires_close_all_confirmation(self, mock_close, mock_cancel):
        out = _unwrap_mcp(trade_close(symbol="EURUSD"))
        if isinstance(out, dict):
            assert "bulk close requires explicit confirmation" in str(out.get("error", "")).lower()
        else:
            assert "bulk close requires explicit confirmation" in out.lower()
        mock_close.assert_not_called()
        mock_cancel.assert_not_called()

    @patch("mtdata.core.trading._cancel_pending")
    @patch("mtdata.core.trading._close_positions")
    def test_global_close_requires_close_all_confirmation(self, mock_close, mock_cancel):
        out = _unwrap_mcp(trade_close())
        if isinstance(out, dict):
            assert "bulk close requires explicit confirmation" in str(out.get("error", "")).lower()
        else:
            assert "bulk close requires explicit confirmation" in out.lower()
        mock_close.assert_not_called()
        mock_cancel.assert_not_called()

    @patch("mtdata.core.trading._cancel_pending")
    @patch("mtdata.core.trading._close_positions")
    def test_close_all_cannot_be_combined_with_ticket(self, mock_close, mock_cancel):
        out = _unwrap_mcp(trade_close(ticket=123, close_all=True))
        if isinstance(out, dict):
            assert "close_all cannot be combined with ticket" in str(out.get("error", "")).lower()
        else:
            assert "close_all cannot be combined with ticket" in out.lower()
        mock_close.assert_not_called()
        mock_cancel.assert_not_called()

    @patch("mtdata.core.trading._cancel_pending")
    @patch("mtdata.core.trading._close_positions")
    def test_symbol_no_open_or_pending_marks_no_action(self, mock_close, mock_cancel):
        mock_close.return_value = {"message": "No open positions for EURUSD"}
        mock_cancel.return_value = {"message": "No pending orders for EURUSD"}
        out = _unwrap_mcp(trade_close(symbol="EURUSD", close_all=True))
        if isinstance(out, dict):
            assert out.get("message") == "No open positions or pending orders for EURUSD"
            assert out.get("no_action") is True
        else:
            assert "No open positions or pending orders for EURUSD" in out
            assert "no_action" in out.lower()
        mock_close.assert_called_once()
        mock_cancel.assert_called_once_with(symbol="EURUSD", comment=None)

    @patch("mtdata.core.trading._cancel_pending")
    @patch("mtdata.core.trading._close_positions")
    def test_global_no_open_or_pending_marks_no_action(self, mock_close, mock_cancel):
        mock_close.return_value = {"message": "No open positions"}
        mock_cancel.return_value = {"message": "No pending orders"}
        out = _unwrap_mcp(trade_close(close_all=True))
        if isinstance(out, dict):
            assert out.get("message") == "No open positions or pending orders"
            assert out.get("no_action") is True
        else:
            assert "No open positions or pending orders" in out
            assert "no_action" in out.lower()
        mock_close.assert_called_once()
        mock_cancel.assert_called_once_with(comment=None)


# ===================================================================
#  trade_modify (dispatch)
# ===================================================================

class TestTradeModify:

    @patch("mtdata.core.trading._modify_pending_order")
    def test_price_routes_to_pending(self, mock_pending):
        mock_pending.return_value = {"success": True}
        trade_modify(ticket=100, price=1.09)
        mock_pending.assert_called_once()

    @patch("mtdata.core.trading._modify_pending_order")
    def test_expiration_routes_to_pending(self, mock_pending):
        mock_pending.return_value = {"success": True}
        trade_modify(ticket=100, expiration="GTC")
        mock_pending.assert_called_once()

    @patch("mtdata.core.trading._modify_position")
    def test_sl_tp_only_tries_position_first(self, mock_pos):
        mock_pos.return_value = {"success": True}
        trade_modify(ticket=100, stop_loss=1.08)
        mock_pos.assert_called_once()

    @patch("mtdata.core.trading._modify_pending_order")
    @patch("mtdata.core.trading._modify_position")
    def test_position_not_found_falls_back_to_pending(self, mock_pos, mock_pend):
        mock_pos.return_value = {"error": "Position 100 not found"}
        mock_pend.return_value = {"success": True}
        result = _unwrap_mcp(trade_modify(ticket=100, stop_loss=1.08))
        assert "success" in result or "True" in result

    @patch("mtdata.core.trading._modify_pending_order")
    @patch("mtdata.core.trading._modify_position")
    def test_both_not_found(self, mock_pos, mock_pend):
        mock_pos.return_value = {"error": "Position 100 not found"}
        mock_pend.return_value = {"error": "Pending order 100 not found"}
        result = _unwrap_mcp(trade_modify(ticket=100, stop_loss=1.08))
        assert "not found" in result

    @patch("mtdata.core.trading._modify_pending_order")
    def test_pending_not_found_with_price(self, mock_pend):
        mock_pend.return_value = {"error": "Pending order 100 not found"}
        result = _unwrap_mcp(trade_modify(ticket=100, price=1.09))
        assert "Pending order 100 not found" in result

    def test_bad_expiration_returns_error(self):
        result = _unwrap_mcp(trade_modify(ticket=100, expiration="not-a-date-xyz-abc"))
        assert "error" in result.lower() or "unsupported" in result.lower()


# ===================================================================
#  trade_account_info (MT5 mocked)
# ===================================================================

class TestTradeAccountInfo:

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_account_info_success(self):
        mt5 = sys.modules["MetaTrader5"]
        mt5.account_info.return_value = SimpleNamespace(
            balance=10000, equity=10000, profit=0, margin=100,
            margin_free=9900, margin_level=10000, currency="USD",
            leverage=100, trade_allowed=True, trade_expert=True,
        )
        result = _unwrap_mcp(trade_account_info())
        assert "10000" in result or "balance" in result

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_account_info_none(self):
        mt5 = sys.modules["MetaTrader5"]
        mt5.account_info.return_value = None
        result = _unwrap_mcp(trade_account_info())
        assert "error" in result.lower()

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_margin_level_is_null_when_no_open_margin(self):
        mt5 = sys.modules["MetaTrader5"]
        mt5.account_info.return_value = SimpleNamespace(
            balance=10000, equity=10000, profit=0, margin=0,
            margin_free=10000, margin_level=0, currency="USD",
            leverage=100, trade_allowed=True, trade_expert=True,
        )
        # Call the raw wrapped function so assertions are not formatter-dependent.
        result = trade_account_info.__wrapped__()
        assert result.get("margin_level") is None
        assert "N/A" in str(result.get("margin_level_note", ""))


# ===================================================================
#  _place_market_order (MT5 mocked)
# ===================================================================

class TestPlaceMarketOrder:

    def _setup_mt5(self, mock_mt5):
        mock_mt5.symbol_info.return_value = _sym()
        mock_mt5.symbol_info_tick.return_value = _tick()
        mock_mt5.symbol_select.return_value = True
        mock_mt5.ORDER_TYPE_BUY = 0
        mock_mt5.ORDER_TYPE_SELL = 1
        mock_mt5.TRADE_ACTION_DEAL = 1
        mock_mt5.TRADE_ACTION_SLTP = 6
        mock_mt5.TRADE_RETCODE_DONE = 10009
        mock_mt5.ORDER_TIME_GTC = 0
        mock_mt5.ORDER_FILLING_IOC = 1

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_symbol_not_found(self):
        import sys
        mt5 = sys.modules["MetaTrader5"]
        mt5.symbol_info.return_value = None
        from mtdata.core.trading import _place_market_order
        result = _place_market_order("INVALID", 0.01, "BUY")
        assert "error" in result and "not found" in result["error"]

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_order_send_returns_none(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.order_send.return_value = None
        mt5.last_error.return_value = (10006, "No connection")
        from mtdata.core.trading import _place_market_order
        result = _place_market_order("EURUSD", 0.01, "BUY")
        assert "error" in result

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_order_send_retcode_fail(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.order_send.return_value = _order_result(retcode=10004)
        from mtdata.core.trading import _place_market_order
        result = _place_market_order("EURUSD", 0.01, "BUY")
        assert "error" in result and result["retcode"] == 10004

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_invisible_symbol_selected(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.symbol_info.return_value = _sym(visible=False)
        mt5.symbol_select.return_value = True
        mt5.order_send.return_value = _order_result()
        from mtdata.core.trading import _place_market_order
        result = _place_market_order("EURUSD", 0.01, "BUY")
        mt5.symbol_select.assert_called_with("EURUSD", True)

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_invisible_symbol_select_fails(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.symbol_info.return_value = _sym(visible=False)
        mt5.symbol_select.return_value = False
        from mtdata.core.trading import _place_market_order
        result = _place_market_order("EURUSD", 0.01, "BUY")
        assert "error" in result and "Failed to select" in result["error"]

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_buy_sl_above_price_rejected(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        from mtdata.core.trading import _place_market_order
        result = _place_market_order("EURUSD", 0.01, "BUY", stop_loss=1.2)
        assert "error" in result and "below" in result["error"]

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_sell_sl_below_price_rejected(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        from mtdata.core.trading import _place_market_order
        result = _place_market_order("EURUSD", 0.01, "SELL", stop_loss=1.0)
        assert "error" in result and "above" in result["error"]

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_buy_tp_below_price_rejected(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        from mtdata.core.trading import _place_market_order
        result = _place_market_order("EURUSD", 0.01, "BUY", take_profit=1.0)
        assert "error" in result and "above" in result["error"]

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_sell_tp_above_price_rejected(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        from mtdata.core.trading import _place_market_order
        result = _place_market_order("EURUSD", 0.01, "SELL", take_profit=1.2)
        assert "error" in result and "below" in result["error"]

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_success_with_sl_tp_modification(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.order_send.side_effect = [_order_result(), _order_result()]
        mt5.positions_get.return_value = [_position()]
        from mtdata.core.trading import _place_market_order
        result = _place_market_order("EURUSD", 0.01, "BUY", stop_loss=1.09, take_profit=1.12)
        sl_tp_result = result.get("sl_tp_result") or {}
        assert sl_tp_result.get("status") == "applied"
        assert sl_tp_result.get("requested") == {"sl": pytest.approx(1.09), "tp": pytest.approx(1.12)}
        assert sl_tp_result.get("applied") == {"sl": pytest.approx(1.09), "tp": pytest.approx(1.12)}
        assert sl_tp_result.get("broker_adjusted") is None
        assert sl_tp_result.get("adjustment") is None
        assert sl_tp_result.get("error") is None

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_sl_tp_uses_resolved_position_ticket_from_deal_fallback(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.order_send.side_effect = [
            _order_result(order=1001, deal=2002),
            _order_result(),
        ]
        mt5.positions_get.side_effect = [
            [],  # candidate order ticket lookup
            [_position(ticket=3003, deal=2002, sl=1.09, tp=1.12)],  # candidate deal ticket lookup
            [_position(ticket=3003, deal=2002, sl=1.09, tp=1.12)],  # post-modify verification
        ]
        from mtdata.core.trading import _place_market_order
        result = _place_market_order("EURUSD", 0.01, "BUY", stop_loss=1.09, take_profit=1.12)
        assert (result.get("sl_tp_result") or {}).get("status") == "applied"
        assert result.get("position_ticket") == 3003
        modify_req = mt5.order_send.call_args_list[1].args[0]
        assert modify_req.get("position") == 3003

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_sl_tp_modification_fails_gracefully(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.order_send.side_effect = [_order_result(), _order_result(retcode=10004)]
        mt5.positions_get.return_value = [_position()]
        from mtdata.core.trading import _place_market_order
        result = _place_market_order("EURUSD", 0.01, "BUY", stop_loss=1.09, take_profit=1.12)
        sl_tp_result = result.get("sl_tp_result") or {}
        assert sl_tp_result.get("error") is None
        assert sl_tp_result.get("requested") == {"sl": pytest.approx(1.09), "tp": pytest.approx(1.12)}
        assert sl_tp_result.get("status") == "applied"
        assert sl_tp_result.get("applied") == {"sl": pytest.approx(1.09), "tp": pytest.approx(1.12)}
        assert sl_tp_result.get("fallback_used") is True
        assert result.get("protection_status") == "protected_after_fallback"

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_sl_tp_position_not_found(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.order_send.return_value = _order_result()
        mt5.positions_get.return_value = []
        from mtdata.core.trading import _place_market_order
        result = _place_market_order("EURUSD", 0.01, "BUY", stop_loss=1.09)
        sl_tp_result = result.get("sl_tp_result") or {}
        assert "Position not found" in str(sl_tp_result.get("error") or "")
        assert sl_tp_result.get("requested") == {"sl": pytest.approx(1.09)}
        assert sl_tp_result.get("status") == "failed"
        assert sl_tp_result.get("applied") is None
        assert result.get("position_ticket") is None
        assert result.get("position_ticket_candidates") == [1]
        assert any("trade_modify 1" in str(w) for w in result.get("warnings", []))
        assert not any("<position_ticket>" in str(w) for w in result.get("warnings", []))

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_sl_tp_not_requested_state_is_explicit(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.order_send.return_value = _order_result()
        from mtdata.core.trading import _place_market_order
        result = _place_market_order("EURUSD", 0.01, "BUY")
        assert result.get("sl_tp_result") == {"status": "not_requested"}

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_comment_truncation_is_reported(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.order_send.return_value = _order_result()
        from mtdata.core.trading import _place_market_order
        long_comment = "X" * 60
        result = _place_market_order("EURUSD", 0.01, "BUY", comment=long_comment)
        trunc = result.get("comment_truncation")
        assert isinstance(trunc, dict)
        assert trunc.get("requested") == long_comment
        assert trunc.get("max_length") == 31
        assert len(str(trunc.get("applied") or "")) <= 31
        assert any("Comment truncated" in str(w) for w in result.get("warnings", []))

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_comment_is_sanitized_and_retried_when_broker_rejects_field(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.order_send.side_effect = [
            _order_result(retcode=10013, comment='Invalid "comment" argument'),
            _order_result(comment=""),
        ]
        from mtdata.core.trading import _place_market_order

        result = _place_market_order(
            "BTCUSD",
            0.01,
            "SELL",
            comment="Short: ETS bearish, barrier EV+, R:R 9.65",
        )

        assert "error" not in result
        assert result["comment_sanitization"]["requested"] == "Short: ETS bearish, barrier EV+, R:R 9.65"
        assert result["comment_fallback"]["used"] is True
        assert result["comment_fallback"]["strategy"] == "minimal"
        assert any("sanitized for broker compatibility" in str(w) for w in result.get("warnings", []))
        fallback_warnings = [
            str(w)
            for w in result.get("warnings", [])
            if "retried with a minimal MT5-safe comment" in str(w)
        ]
        assert len(fallback_warnings) == 1
        first_request = mt5.order_send.call_args_list[0].args[0]
        second_request = mt5.order_send.call_args_list[1].args[0]
        assert first_request["comment"] == "Short ETS bearish barrier EV R"
        assert second_request["comment"] == "MCP"

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_sl_tp_broker_adjustment_is_reported(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.order_send.side_effect = [_order_result(), _order_result()]
        mt5.positions_get.side_effect = [
            [_position(sl=1.09, tp=1.12)],
            [_position(sl=1.0895, tp=1.1203)],
        ]
        from mtdata.core.trading import _place_market_order
        result = _place_market_order("EURUSD", 0.01, "BUY", stop_loss=1.09, take_profit=1.12)
        sl_tp_result = result.get("sl_tp_result") or {}
        assert sl_tp_result.get("status") == "applied"
        assert sl_tp_result.get("broker_adjusted") is True
        assert sl_tp_result.get("adjustment") == {
            "sl": {"requested": pytest.approx(1.09), "applied": pytest.approx(1.0895)},
            "tp": {"requested": pytest.approx(1.12), "applied": pytest.approx(1.1203)},
        }

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_pending_type_rejected_for_market(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        from mtdata.core.trading import _place_market_order
        result = _place_market_order("EURUSD", 0.01, "BUY_LIMIT")
        assert "error" in result

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_tick_none_returns_error(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.symbol_info_tick.return_value = None
        from mtdata.core.trading import _place_market_order
        result = _place_market_order("EURUSD", 0.01, "BUY")
        assert "error" in result and "current price" in result["error"]


# ===================================================================
#  _place_pending_order (MT5 mocked)
# ===================================================================

class TestPlacePendingOrder:

    def _setup_mt5(self, mock_mt5):
        mock_mt5.symbol_info.return_value = _sym()
        mock_mt5.symbol_info_tick.return_value = _tick(bid=1.10000, ask=1.10020)
        mock_mt5.symbol_select.return_value = True
        mock_mt5.ORDER_TYPE_BUY = 0
        mock_mt5.ORDER_TYPE_SELL = 1
        mock_mt5.ORDER_TYPE_BUY_LIMIT = 2
        mock_mt5.ORDER_TYPE_SELL_LIMIT = 3
        mock_mt5.ORDER_TYPE_BUY_STOP = 4
        mock_mt5.ORDER_TYPE_SELL_STOP = 5
        mock_mt5.TRADE_ACTION_PENDING = 5
        mock_mt5.TRADE_RETCODE_DONE = 10009
        mock_mt5.ORDER_TIME_GTC = 0
        mock_mt5.ORDER_TIME_SPECIFIED = 2
        mock_mt5.ORDER_FILLING_IOC = 1

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_buy_limit_success(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.order_send.return_value = _order_result()
        from mtdata.core.trading import _place_pending_order
        result = _place_pending_order("EURUSD", 0.01, "BUY_LIMIT", price=1.09)
        assert result.get("success") is True
        assert result.get("requested_price") == 1.09
        assert result.get("requested_sl") is None
        assert result.get("requested_tp") is None

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_sell_stop_success(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.order_send.return_value = _order_result()
        from mtdata.core.trading import _place_pending_order
        result = _place_pending_order("EURUSD", 0.01, "SELL_STOP", price=1.09)
        assert result.get("success") is True

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_buy_limit_price_above_ask_rejected(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        from mtdata.core.trading import _place_pending_order
        result = _place_pending_order("EURUSD", 0.01, "BUY_LIMIT", price=1.20)
        assert "error" in result and "below ask" in result["error"]

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_sell_limit_price_below_bid_rejected(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        from mtdata.core.trading import _place_pending_order
        result = _place_pending_order("EURUSD", 0.01, "SELL_LIMIT", price=1.05)
        assert "error" in result and "above bid" in result["error"]

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_buy_stop_price_below_ask_rejected(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        from mtdata.core.trading import _place_pending_order
        result = _place_pending_order("EURUSD", 0.01, "BUY_STOP", price=1.05)
        assert "error" in result and "above ask" in result["error"]

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_sell_stop_price_above_bid_rejected(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        from mtdata.core.trading import _place_pending_order
        result = _place_pending_order("EURUSD", 0.01, "SELL_STOP", price=1.20)
        assert "error" in result and "below bid" in result["error"]

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_pending_sl_tp_buy_validation(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        from mtdata.core.trading import _place_pending_order
        # SL above entry for BUY_LIMIT should fail
        result = _place_pending_order("EURUSD", 0.01, "BUY_LIMIT", price=1.09, stop_loss=1.10)
        assert "error" in result and "below entry" in result["error"]

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_pending_tp_buy_validation(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        from mtdata.core.trading import _place_pending_order
        # TP below entry for BUY_LIMIT should fail
        result = _place_pending_order("EURUSD", 0.01, "BUY_LIMIT", price=1.09, take_profit=1.05)
        assert "error" in result and "above entry" in result["error"]

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_pending_sl_sell_validation(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        from mtdata.core.trading import _place_pending_order
        # SL below entry for SELL_LIMIT should fail
        result = _place_pending_order("EURUSD", 0.01, "SELL_LIMIT", price=1.12, stop_loss=1.10)
        assert "error" in result and "above entry" in result["error"]

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_pending_tp_sell_validation(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        from mtdata.core.trading import _place_pending_order
        # TP above entry for SELL_LIMIT should fail
        result = _place_pending_order("EURUSD", 0.01, "SELL_LIMIT", price=1.12, take_profit=1.15)
        assert "error" in result and "below entry" in result["error"]

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_implicit_buy_below_ask_becomes_buy_limit(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.order_send.return_value = _order_result()
        from mtdata.core.trading import _place_pending_order
        result = _place_pending_order("EURUSD", 0.01, "BUY", price=1.09)
        # Verify the request used BUY_LIMIT (type=2)
        call_args = mt5.order_send.call_args[0][0]
        assert call_args["type"] == mt5.ORDER_TYPE_BUY_LIMIT

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_implicit_sell_above_bid_becomes_sell_limit(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.order_send.return_value = _order_result()
        from mtdata.core.trading import _place_pending_order
        result = _place_pending_order("EURUSD", 0.01, "SELL", price=1.12)
        call_args = mt5.order_send.call_args[0][0]
        assert call_args["type"] == mt5.ORDER_TYPE_SELL_LIMIT

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_order_send_none_returns_error(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.order_send.return_value = None
        mt5.last_error.return_value = (10006, "err")
        from mtdata.core.trading import _place_pending_order
        result = _place_pending_order("EURUSD", 0.01, "BUY_LIMIT", price=1.09)
        assert "error" in result and "Failed" in result["error"]

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_order_send_retcode_fail(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.order_send.return_value = _order_result(retcode=10004)
        from mtdata.core.trading import _place_pending_order
        result = _place_pending_order("EURUSD", 0.01, "BUY_LIMIT", price=1.09)
        assert "error" in result

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_non_finite_price_rejected(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        from mtdata.core.trading import _place_pending_order
        result = _place_pending_order("EURUSD", 0.01, "BUY_LIMIT", price=float("inf"))
        assert "error" in result and "finite" in result["error"]

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_with_expiration_sets_type_time_specified(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.order_send.return_value = _order_result()
        from mtdata.core.trading import _place_pending_order
        # Use a numeric expiration (epoch seconds)
        with patch("mtdata.core.trading.time._normalize_pending_expiration", return_value=(1717200000, True)):
            result = _place_pending_order("EURUSD", 0.01, "BUY_LIMIT", price=1.09, expiration=1717200000)
        call_args = mt5.order_send.call_args[0][0]
        assert call_args["type_time"] == 2  # ORDER_TIME_SPECIFIED
        assert call_args["expiration"] == 1717200000


# ===================================================================
#  _close_positions (MT5 mocked)
# ===================================================================

class TestClosePositions:

    def _setup_mt5(self, mt5):
        mt5.ORDER_TYPE_BUY = 0
        mt5.ORDER_TYPE_SELL = 1
        mt5.POSITION_TYPE_BUY = 0
        mt5.POSITION_TYPE_SELL = 1
        mt5.TRADE_ACTION_DEAL = 1
        mt5.TRADE_RETCODE_DONE = 10009
        mt5.TRADE_RETCODE_DONE_PARTIAL = 10010
        mt5.ORDER_TIME_GTC = 0
        mt5.ORDER_FILLING_IOC = 1
        mt5.ORDER_FILLING_FOK = 0
        mt5.ORDER_FILLING_RETURN = 2

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_ticket_not_found(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.positions_get.return_value = None
        from mtdata.core.trading import _close_positions
        result = _close_positions(ticket=999)
        assert "error" in result

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_no_open_positions(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.positions_get.return_value = []
        from mtdata.core.trading import _close_positions
        result = _close_positions()
        assert "message" in result

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_profit_only_filter(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        pos_profit = _position(ticket=1, profit=50)
        pos_loss = _position(ticket=2, profit=-30)
        mt5.positions_get.return_value = [pos_profit, pos_loss]
        mt5.symbol_info_tick.return_value = _tick()
        mt5.order_send.return_value = _order_result()
        from mtdata.core.trading import _close_positions
        result = _close_positions(profit_only=True)
        assert result["attempted_count"] == 1

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_loss_only_filter(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        pos_profit = _position(ticket=1, profit=50)
        pos_loss = _position(ticket=2, profit=-30)
        mt5.positions_get.return_value = [pos_profit, pos_loss]
        mt5.symbol_info_tick.return_value = _tick()
        mt5.order_send.return_value = _order_result()
        from mtdata.core.trading import _close_positions
        result = _close_positions(loss_only=True)
        assert result["attempted_count"] == 1

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_profit_only_excludes_breakeven_positions(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.positions_get.return_value = [_position(ticket=3, profit=0.0)]
        mt5.symbol_info_tick.return_value = _tick()
        mt5.order_send.return_value = _order_result()
        from mtdata.core.trading import _close_positions
        result = _close_positions(profit_only=True)
        assert result["message"] == "No positions matched criteria"

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_loss_only_excludes_breakeven_positions(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.positions_get.return_value = [_position(ticket=4, profit=0.0)]
        mt5.symbol_info_tick.return_value = _tick()
        mt5.order_send.return_value = _order_result()
        from mtdata.core.trading import _close_positions
        result = _close_positions(loss_only=True)
        assert result["message"] == "No positions matched criteria"

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_profit_only_skips_positions_with_unreadable_profit(self):
        class _BrokenProfitPosition:
            ticket = 7
            symbol = "EURUSD"
            type = 0
            volume = 0.01
            price_current = 1.105
            price_open = 1.1
            comment = ""
            magic = 234000
            time = 1700000000
            time_update = 1700000000
            swap = 0.0

            @property
            def profit(self):
                raise RuntimeError("bad profit")

        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.positions_get.return_value = [_BrokenProfitPosition(), _position(ticket=8, profit=50.0)]
        mt5.symbol_info_tick.return_value = _tick()
        mt5.order_send.return_value = _order_result()
        from mtdata.core.trading import _close_positions
        result = _close_positions(profit_only=True)
        assert result["attempted_count"] == 1
        assert mt5.order_send.call_count == 1

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_conflicting_profit_and_loss_filters_rejected(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.positions_get.return_value = [_position(ticket=5, profit=10.0)]
        from mtdata.core.trading import _close_positions
        result = _close_positions(profit_only=True, loss_only=True)
        assert result["error"] == "profit_only and loss_only cannot both be true."

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_tick_failure_for_position(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.positions_get.return_value = [_position()]
        mt5.symbol_info_tick.return_value = None
        from mtdata.core.trading import _close_positions
        result = _close_positions(ticket=1)
        assert "error" in result and "tick" in result["error"].lower()

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_close_rejects_position_with_unreadable_side(self):
        class _BrokenTypePosition:
            ticket = 11
            symbol = "EURUSD"
            volume = 0.01
            price_open = 1.1
            sl = 1.09
            tp = 1.12
            profit = 50.0
            price_current = 1.105
            comment = ""
            magic = 234000
            time = 1700000000
            time_update = 1700000000
            swap = 0.0

            @property
            def type(self):
                raise RuntimeError("bad type")

        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.positions_get.return_value = [_BrokenTypePosition()]
        from mtdata.core.trading import _close_positions
        result = _close_positions(ticket=11)
        assert result["error"] == "Unable to determine position side for close request."
        mt5.order_send.assert_not_called()

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_order_send_none(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.positions_get.return_value = [_position()]
        mt5.symbol_info_tick.return_value = _tick()
        mt5.order_send.return_value = None
        mt5.last_error.return_value = (10006, "No connection")
        from mtdata.core.trading import _close_positions
        result = _close_positions(ticket=1)
        assert "error" in result
        assert "attempts" in result

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_close_retries_fill_modes_when_first_attempt_fails(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.positions_get.return_value = [_position(ticket=77)]
        mt5.symbol_info_tick.return_value = _tick()
        mt5.order_send.side_effect = [None, _order_result()]
        mt5.last_error.return_value = (10006, "No connection")
        from mtdata.core.trading import _close_positions
        result = _close_positions(ticket=77)
        assert result["ticket"] == 77
        assert int(result["retcode"]) == 10009
        assert isinstance(result.get("attempts"), list)
        assert len(result.get("attempts") or []) >= 2

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_no_positions_matched_criteria(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        # All profitable, but loss_only filter
        mt5.positions_get.return_value = [_position(profit=50)]
        from mtdata.core.trading import _close_positions
        result = _close_positions(loss_only=True)
        assert "message" in result and "No positions matched" in result["message"]

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_single_ticket_returns_single_result(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.positions_get.return_value = [_position(ticket=42)]
        mt5.symbol_info_tick.return_value = _tick()
        mt5.order_send.return_value = _order_result()
        from mtdata.core.trading import _close_positions
        result = _close_positions(ticket=42)
        assert result["ticket"] == 42
        assert result["open_price"] == 1.1
        assert result["close_price"] == 1.1
        assert "pnl" in result
        assert "duration_seconds" in result

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_partial_close_uses_requested_volume(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.positions_get.return_value = [_position(ticket=42, volume=0.10)]
        mt5.symbol_info.return_value = _sym(volume_min=0.01, volume_step=0.01)
        mt5.symbol_info_tick.return_value = _tick()
        mt5.order_send.return_value = _order_result(volume=0.05)
        from mtdata.core.trading import _close_positions
        result = _close_positions(ticket=42, volume=0.05)
        request = mt5.order_send.call_args[0][0]
        assert request["volume"] == 0.05
        assert result["requested_volume"] == 0.05
        assert result["position_volume_before"] == 0.10
        assert result["position_volume_remaining_estimate"] == 0.05

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_partial_close_rejects_volume_greater_than_position(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.positions_get.return_value = [_position(ticket=42, volume=0.10)]
        mt5.symbol_info.return_value = _sym(volume_min=0.01, volume_step=0.01)
        from mtdata.core.trading import _close_positions
        result = _close_positions(ticket=42, volume=0.20)
        assert "volume must be <=" in result["error"]
        assert result["position_volume"] == 0.10
        mt5.order_send.assert_not_called()

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_partial_close_rejects_invalid_remaining_volume(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.positions_get.return_value = [_position(ticket=42, volume=0.03)]
        mt5.symbol_info.return_value = _sym(volume_min=0.02, volume_step=0.01)
        from mtdata.core.trading import _close_positions
        result = _close_positions(ticket=42, volume=0.02)
        assert "remaining position volume would be invalid" in result["error"]
        assert result["remaining_volume"] == pytest.approx(0.01)
        mt5.order_send.assert_not_called()

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_symbol_no_positions(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.positions_get.return_value = []
        from mtdata.core.trading import _close_positions
        result = _close_positions(symbol="GBPUSD")
        assert "message" in result


# ===================================================================
#  _cancel_pending (MT5 mocked)
# ===================================================================

class TestCancelPending:

    def _setup_mt5(self, mt5):
        mt5.TRADE_ACTION_REMOVE = 8
        mt5.TRADE_RETCODE_DONE = 10009
        mt5.TRADE_RETCODE_DONE_PARTIAL = 10010

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_ticket_not_found(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.orders_get.return_value = None
        from mtdata.core.trading import _cancel_pending
        result = _cancel_pending(ticket=999)
        assert "error" in result

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_no_pending_orders(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.orders_get.return_value = []
        from mtdata.core.trading import _cancel_pending
        result = _cancel_pending()
        assert "message" in result

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_single_cancel_success(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.orders_get.return_value = [_pending_order(ticket=100)]
        mt5.order_send.return_value = _order_result()
        from mtdata.core.trading import _cancel_pending
        result = _cancel_pending(ticket=100)
        assert result["ticket"] == 100

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_cancel_order_send_none(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.orders_get.return_value = [_pending_order(ticket=100)]
        mt5.order_send.return_value = None
        from mtdata.core.trading import _cancel_pending
        result = _cancel_pending(ticket=100)
        assert "error" in result

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_bulk_cancel(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.orders_get.return_value = [
            _pending_order(ticket=100),
            _pending_order(ticket=101),
        ]
        mt5.order_send.return_value = _order_result()
        from mtdata.core.trading import _cancel_pending
        result = _cancel_pending()
        assert result["attempted_count"] == 2

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_symbol_no_pending_orders(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.orders_get.return_value = []
        from mtdata.core.trading import _cancel_pending
        result = _cancel_pending(symbol="GBPUSD")
        assert "message" in result


# ===================================================================
#  _modify_position (MT5 mocked)
# ===================================================================

class TestModifyPosition:

    def _setup_mt5(self, mt5):
        mt5.TRADE_ACTION_SLTP = 6
        mt5.TRADE_RETCODE_DONE = 10009
        mt5.symbol_info_tick.return_value = _tick(bid=1.1050, ask=1.1052)

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_position_not_found(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.positions_get.return_value = []
        from mtdata.core.trading import _modify_position
        result = _modify_position(ticket=999)
        assert "error" in result and "not found" in result["error"]

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_success(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.positions_get.return_value = [_position()]
        mt5.symbol_info.return_value = _sym()
        mt5.order_send.return_value = _order_result()
        from mtdata.core.trading import _modify_position
        result = _modify_position(ticket=1, stop_loss=1.08, take_profit=1.13)
        assert result.get("success") is True

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_no_changes_returns_success_when_levels_already_match(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.TRADE_RETCODE_NO_CHANGES = 10025
        mt5.positions_get.return_value = [_position(sl=1.08, tp=1.13)]
        mt5.symbol_info.return_value = _sym()
        from mtdata.core.trading import _modify_position
        result = _modify_position(ticket=1, stop_loss=1.08, take_profit=1.13)
        assert result.get("success") is True
        assert result.get("no_change") is True
        assert result.get("retcode") == 10025
        mt5.order_send.assert_not_called()

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_resolves_position_by_identifier_when_ticket_lookup_misses(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.positions_get.side_effect = [
            [],
            [_position(ticket=777, identifier=123)],
        ]
        mt5.symbol_info.return_value = _sym()
        mt5.order_send.return_value = _order_result()
        from mtdata.core.trading import _modify_position
        result = _modify_position(ticket=123, stop_loss=1.08)
        assert result.get("success") is True
        req = mt5.order_send.call_args.args[0]
        assert req.get("symbol") == "EURUSD"
        assert req.get("position") == 777
        assert result.get("position_ticket") == 777

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_order_send_none(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.positions_get.return_value = [_position()]
        mt5.symbol_info.return_value = _sym()
        mt5.order_send.return_value = None
        mt5.last_error.return_value = (10006, "err")
        from mtdata.core.trading import _modify_position
        result = _modify_position(ticket=1, stop_loss=1.08)
        assert "error" in result

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_order_send_retcode_fail(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.positions_get.return_value = [_position()]
        mt5.symbol_info.return_value = _sym()
        mt5.order_send.return_value = _order_result(retcode=10004)
        from mtdata.core.trading import _modify_position
        result = _modify_position(ticket=1, stop_loss=1.08)
        assert "error" in result

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_broker_no_changes_retcode_is_success_when_live_position_matches(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.TRADE_RETCODE_NO_CHANGES = 10025
        mt5.positions_get.side_effect = [
            [_position(sl=1.09, tp=1.12)],
            [_position(sl=1.08, tp=1.12)],
        ]
        mt5.symbol_info.return_value = _sym()
        mt5.order_send.return_value = _order_result(retcode=10025, comment="No changes")
        from mtdata.core.trading import _modify_position
        result = _modify_position(ticket=1, stop_loss=1.08)
        assert result.get("success") is True
        assert result.get("no_change") is True
        assert result.get("retcode") == 10025

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_broker_no_changes_retcode_errors_when_live_position_still_differs(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.TRADE_RETCODE_NO_CHANGES = 10025
        mt5.positions_get.side_effect = [
            [_position(sl=1.09, tp=1.12)],
            [_position(sl=1.09, tp=1.12)],
        ]
        mt5.symbol_info.return_value = _sym()
        mt5.order_send.return_value = _order_result(retcode=10025, comment="No changes")
        from mtdata.core.trading import _modify_position
        result = _modify_position(ticket=1, stop_loss=1.08)
        assert result.get("error")
        assert result.get("no_change") is True
        assert result.get("desired_sl") == 1.08
        assert result.get("actual_sl") == 1.09

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_symbol_info_none(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.positions_get.return_value = [_position()]
        mt5.symbol_info.return_value = None
        from mtdata.core.trading import _modify_position
        result = _modify_position(ticket=1, stop_loss=1.08)
        assert "error" in result

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_only_tp_change_revalidates_existing_sl(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.positions_get.return_value = [_position(sl=1.10495, tp=1.12)]
        symbol_info = _sym()
        symbol_info.trade_stops_level = 30
        symbol_info.trade_freeze_level = 0
        mt5.symbol_info.return_value = symbol_info
        from mtdata.core.trading import _modify_position
        result = _modify_position(ticket=1, take_profit=1.13)
        assert "error" in result
        assert "stop_loss is too close" in result["error"]
        mt5.order_send.assert_not_called()

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_digits_none_uses_safe_default_when_modifying_position(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.positions_get.return_value = [_position()]
        mt5.symbol_info.return_value = _sym(point=0.0, digits=None)
        mt5.order_send.return_value = _order_result()
        from mtdata.core.trading import _modify_position
        result = _modify_position(ticket=1, stop_loss=1.08001)
        assert result.get("success") is True

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_returns_error_when_position_side_cannot_be_resolved(self):
        class _BrokenTypePosition:
            ticket = 1
            symbol = "EURUSD"
            sl = 1.09
            tp = 1.12

            @property
            def type(self):
                raise RuntimeError("boom")

        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.positions_get.return_value = [_BrokenTypePosition()]
        mt5.symbol_info.return_value = _sym()
        from mtdata.core.trading import _modify_position
        result = _modify_position(ticket=1, stop_loss=1.08)
        assert result["error"] == "Unable to determine position side for protection validation."
        mt5.order_send.assert_not_called()

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_tiny_stop_loss_is_treated_as_explicit_remove(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.positions_get.return_value = [_position(sl=1.09, tp=1.12)]
        mt5.symbol_info.return_value = _sym()
        mt5.order_send.return_value = _order_result()
        from mtdata.core.trading import _modify_position
        result = _modify_position(ticket=1, stop_loss=1e-12)
        assert result.get("success") is True
        request = mt5.order_send.call_args.args[0]
        assert request["sl"] == 0.0
        assert request["tp"] == pytest.approx(1.12)


# ===================================================================
#  _modify_pending_order (MT5 mocked)
# ===================================================================

class TestModifyPendingOrder:

    def _setup_mt5(self, mt5):
        mt5.TRADE_ACTION_MODIFY = 7
        mt5.TRADE_RETCODE_DONE = 10009
        mt5.ORDER_TIME_GTC = 0
        mt5.ORDER_TIME_SPECIFIED = 2
        mt5.ORDER_TYPE_BUY_LIMIT = 2
        mt5.ORDER_TYPE_SELL_LIMIT = 3
        mt5.ORDER_TYPE_BUY_STOP = 4
        mt5.ORDER_TYPE_SELL_STOP = 5
        mt5.symbol_info.return_value = _sym()
        mt5.symbol_info_tick.return_value = _tick()

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_order_not_found(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.orders_get.return_value = []
        from mtdata.core.trading import _modify_pending_order
        result = _modify_pending_order(ticket=999)
        assert "error" in result and "not found" in result["error"]

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_success(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.orders_get.return_value = [_pending_order()]
        mt5.order_send.return_value = _order_result()
        from mtdata.core.trading import _modify_pending_order
        result = _modify_pending_order(ticket=100, price=1.095)
        assert result.get("success") is True

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_gtc_expiration_sets_type_time(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.orders_get.return_value = [_pending_order()]
        mt5.order_send.return_value = _order_result()
        from mtdata.core.trading import _modify_pending_order
        result = _modify_pending_order(ticket=100, expiration="GTC")
        call_args = mt5.order_send.call_args[0][0]
        assert call_args["type_time"] == 0  # GTC

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_order_send_none(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.orders_get.return_value = [_pending_order()]
        mt5.order_send.return_value = None
        mt5.last_error.return_value = (10006, "err")
        from mtdata.core.trading import _modify_pending_order
        result = _modify_pending_order(ticket=100)
        assert "error" in result

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_preserves_existing_expiration(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        order = _pending_order(type_time=2, time_expiration=1717200000)
        mt5.orders_get.return_value = [order]
        mt5.order_send.return_value = _order_result()
        from mtdata.core.trading import _modify_pending_order
        _modify_pending_order(ticket=100, price=1.095)
        call_args = mt5.order_send.call_args[0][0]
        assert call_args.get("expiration") == 1717200000

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_modify_request_includes_symbol_type_and_volume(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.orders_get.return_value = [_pending_order(symbol="EURUSD", type_=2, volume=0.07)]
        mt5.order_send.return_value = _order_result()
        from mtdata.core.trading import _modify_pending_order
        result = _modify_pending_order(ticket=100, price=1.095)
        assert result.get("success") is True
        call_args = mt5.order_send.call_args[0][0]
        assert call_args["symbol"] == "EURUSD"
        assert call_args["type"] == mt5.ORDER_TYPE_BUY_LIMIT
        assert call_args["volume"] == pytest.approx(0.07)

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_normalizes_pending_modify_prices(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.orders_get.return_value = [_pending_order(price_open=1.0945, sl=1.0845, tp=1.1145)]
        mt5.symbol_info.return_value = _sym(point=0.0001, digits=4)
        mt5.symbol_info_tick.return_value = _tick(bid=1.1000, ask=1.1002)
        mt5.order_send.return_value = _order_result()
        from mtdata.core.trading import _modify_pending_order
        result = _modify_pending_order(ticket=100, price=1.09456, stop_loss=1.08456, take_profit=1.11456)
        assert result.get("success") is True
        call_args = mt5.order_send.call_args[0][0]
        assert call_args["price"] == pytest.approx(1.0946)
        assert call_args["sl"] == pytest.approx(1.0846)
        assert call_args["tp"] == pytest.approx(1.1146)

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_pending_modify_rejects_invalid_take_profit_side(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.orders_get.return_value = [_pending_order(type_=3, price_open=1.1050, sl=1.1100, tp=1.0950)]
        mt5.symbol_info.return_value = _sym()
        mt5.symbol_info_tick.return_value = _tick(bid=1.1000, ask=1.1002)
        from mtdata.core.trading import _modify_pending_order
        result = _modify_pending_order(ticket=100, take_profit=1.1100)
        assert "error" in result
        assert "take_profit must be below entry for SELL orders" in result["error"]

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_pending_modify_rejects_entry_inside_broker_stop_distance(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.orders_get.return_value = [_pending_order(type_=2, price_open=1.0950, sl=1.0850, tp=1.1150)]
        mt5.symbol_info.return_value = _sym(point=0.00001, digits=5)
        mt5.symbol_info.return_value.trade_stops_level = 30
        mt5.symbol_info.return_value.trade_freeze_level = 0
        mt5.symbol_info_tick.return_value = _tick(bid=1.1000, ask=1.1002)
        from mtdata.core.trading import _modify_pending_order
        result = _modify_pending_order(ticket=100, price=1.1000)
        assert "error" in result
        assert "pending entry is too close" in result["error"]
        assert "min_distance_points=30" in result["error"]


# ===================================================================
#  trade_risk_analyze (MT5 mocked)
# ===================================================================

class TestTradeRiskAnalyze:

    def _setup_mt5(self, mt5):
        mt5.TRADE_RETCODE_DONE = 10009

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_account_info_none(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.account_info.return_value = None
        result = _unwrap_mcp(trade_risk_analyze())
        assert "error" in result.lower()

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_no_positions(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.account_info.return_value = SimpleNamespace(equity=10000, currency="USD")
        mt5.positions_get.return_value = None
        result = _unwrap_mcp(trade_risk_analyze())
        assert "success" in result and "positions_count" in result

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_positions_with_sl(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.account_info.return_value = SimpleNamespace(equity=10000, currency="USD")
        pos = SimpleNamespace(
            ticket=1, symbol="EURUSD", type=0, volume=0.1,
            price_open=1.1, sl=1.09, tp=1.12, profit=50,
        )
        mt5.positions_get.return_value = [pos]
        mt5.symbol_info.return_value = _sym()
        result = _unwrap_mcp(trade_risk_analyze())
        assert "defined" in result

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_positions_without_sl(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.account_info.return_value = SimpleNamespace(equity=10000, currency="USD")
        pos = SimpleNamespace(
            ticket=1, symbol="EURUSD", type=0, volume=0.1,
            price_open=1.1, sl=0, tp=0, profit=50,
        )
        mt5.positions_get.return_value = [pos]
        mt5.symbol_info.return_value = _sym()
        result = _unwrap_mcp(trade_risk_analyze())
        assert "unlimited" in result.lower()

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_position_sizing(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.account_info.return_value = SimpleNamespace(equity=10000, currency="USD")
        mt5.positions_get.return_value = []
        mt5.symbol_info.return_value = _sym()
        result = _unwrap_mcp(trade_risk_analyze(
            symbol="EURUSD",
            desired_risk_pct=2.0,
            entry=1.1000,
            stop_loss=1.0950,
            take_profit=1.1100,
        ))
        assert "position_sizing" in result or "suggested_volume" in result

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_position_sizing_no_symbol(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.account_info.return_value = SimpleNamespace(equity=10000, currency="USD")
        mt5.positions_get.return_value = []
        result = _unwrap_mcp(trade_risk_analyze(
            desired_risk_pct=2.0, entry=1.1, stop_loss=1.09,
        ))
        assert "error" in result.lower()

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_position_sizing_symbol_not_found(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.account_info.return_value = SimpleNamespace(equity=10000, currency="USD")
        mt5.positions_get.return_value = []
        mt5.symbol_info.return_value = None
        result = _unwrap_mcp(trade_risk_analyze(
            symbol="INVALID", desired_risk_pct=2.0,
            entry=1.1, stop_loss=1.09,
        ))
        assert "error" in result.lower() or "not found" in result.lower()

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_risk_level_high(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.account_info.return_value = SimpleNamespace(equity=1000, currency="USD")
        pos = SimpleNamespace(
            ticket=1, symbol="EURUSD", type=0, volume=10.0,
            price_open=1.1, sl=1.0, tp=1.2, profit=0,
        )
        mt5.positions_get.return_value = [pos]
        mt5.symbol_info.return_value = _sym()
        result = _unwrap_mcp(trade_risk_analyze())
        assert "high" in result

    @patch.dict("sys.modules", {"MetaTrader5": MagicMock()})
    def test_position_sizing_zero_sl_distance(self):
        mt5 = sys.modules["MetaTrader5"]
        self._setup_mt5(mt5)
        mt5.account_info.return_value = SimpleNamespace(equity=10000, currency="USD")
        mt5.positions_get.return_value = []
        mt5.symbol_info.return_value = _sym()
        result = _unwrap_mcp(trade_risk_analyze(
            symbol="EURUSD", desired_risk_pct=2.0,
            entry=1.1, stop_loss=1.1,
        ))
        assert "position_sizing_error" in result or "SL distance" in result


# ===================================================================
#  Edge cases / integration
# ===================================================================

class TestEdgeCases:

    def test_order_type_numeric_map_completeness(self):
        """All values 0-5 should be in the numeric map."""
        for i in range(6):
            assert i in _ORDER_TYPE_NUMERIC_MAP

    def test_supported_order_types_match_map(self):
        """All mapped names should be in SUPPORTED_ORDER_TYPES."""
        for name in _ORDER_TYPE_NUMERIC_MAP.values():
            assert name in _SUPPORTED_ORDER_TYPES

    def test_gtc_tokens_all_uppercase(self):
        for token in _GTC_EXPIRATION_TOKENS:
            assert token == token.upper()

    def test_normalize_order_type_with_double_underscores(self):
        t, err = _normalize_order_type_input("BUY__LIMIT")
        assert t == "BUY_LIMIT"

    def test_normalize_order_type_with_dashes(self):
        t, err = _normalize_order_type_input("sell-limit")
        assert t == "SELL_LIMIT"

    def test_validate_volume_exactly_at_min(self):
        vol, err = _validate_volume(0.01, _sym(volume_min=0.01))
        assert err is None and vol == 0.01

    def test_validate_volume_exactly_at_max(self):
        vol, err = _validate_volume(100.0, _sym(volume_max=100.0))
        assert err is None

    def test_trade_comment_with_non_string_input(self):
        result = _normalize_trade_comment(123, default="x")
        assert result == "123"

    def test_server_time_naive_large_timestamp(self):
        dt = datetime(2099, 12, 31, 23, 59, 59)
        result = _server_time_naive_to_mt5_timestamp(dt)
        assert result > 0

    @patch("mtdata.core.trading.time._to_server_time_naive", side_effect=lambda dt: dt.replace(tzinfo=None) if dt.tzinfo else dt)
    def test_expiration_relative_minutes_variants(self, _mock):
        for variant in ("30min", "30 mins", "30 minute", "30 minutes"):
            val, explicit = _normalize_pending_expiration(variant)
            assert isinstance(val, int) and explicit is True, f"Failed for: {variant}"

    @patch("mtdata.core.trading.time._to_server_time_naive", side_effect=lambda dt: dt.replace(tzinfo=None) if dt.tzinfo else dt)
    def test_expiration_relative_hours_variants(self, _mock):
        for variant in ("2h", "2 hr", "2 hrs", "2 hour", "2 hours"):
            val, explicit = _normalize_pending_expiration(variant)
            assert isinstance(val, int) and explicit is True, f"Failed for: {variant}"

    @patch("mtdata.core.trading.time._to_server_time_naive", side_effect=lambda dt: dt.replace(tzinfo=None) if dt.tzinfo else dt)
    def test_expiration_relative_seconds_variants(self, _mock):
        for variant in ("60s", "60 sec", "60 secs", "60 second", "60 seconds"):
            val, explicit = _normalize_pending_expiration(variant)
            assert isinstance(val, int) and explicit is True, f"Failed for: {variant}"

    @patch("mtdata.core.trading.time._to_server_time_naive", side_effect=lambda dt: dt.replace(tzinfo=None) if dt.tzinfo else dt)
    def test_expiration_relative_days_variants(self, _mock):
        for variant in ("1d", "1 day", "1 days"):
            val, explicit = _normalize_pending_expiration(variant)
            assert isinstance(val, int) and explicit is True, f"Failed for: {variant}"

    @patch("mtdata.core.trading.time._to_server_time_naive", side_effect=lambda dt: dt.replace(tzinfo=None) if dt.tzinfo else dt)
    def test_expiration_relative_weeks_variants(self, _mock):
        for variant in ("1w", "1 wk", "1 weeks"):
            val, explicit = _normalize_pending_expiration(variant)
            assert isinstance(val, int) and explicit is True, f"Failed for: {variant}"

    @patch("mtdata.core.trading.time._to_server_time_naive", side_effect=lambda dt: dt.replace(tzinfo=None) if dt.tzinfo else dt)
    def test_expiration_in_prefix(self, _mock):
        val, explicit = _normalize_pending_expiration("in 5 minutes")
        assert isinstance(val, int) and explicit is True

