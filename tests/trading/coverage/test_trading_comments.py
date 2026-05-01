"""Tests for comments and expiration handling in mtdata.core.trading.

Covers:
- _normalize_trade_comment (pure)
- _normalize_close_trade_comment (pure)
- _normalize_pending_expiration (mixed)
"""

import importlib
import os
import sys
from datetime import datetime
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

from mtdata.core.trading.comments import (
    _comment_row_metadata,
    _normalize_close_trade_comment,
    _normalize_trade_comment,
)
from mtdata.core.trading.time import (
    _GTC_EXPIRATION_TOKENS,
    _normalize_pending_expiration,
)


# ===================================================================
# Helpers
# ===================================================================

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
