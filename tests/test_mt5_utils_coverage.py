"""Coverage tests for mtdata.utils.mt5 – targeting uncovered lines."""

import sys
import types
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import patch, MagicMock, PropertyMock

import numpy as np
import pandas as pd
import pytest

# ── Provide a fake MetaTrader5 module before any project imports ──────────
_mt5_mock = MagicMock()
_mt5_mock.TIMEFRAME_M1 = 1
_mt5_mock.TIMEFRAME_M2 = 2
_mt5_mock.TIMEFRAME_M3 = 3
_mt5_mock.TIMEFRAME_M4 = 4
_mt5_mock.TIMEFRAME_M5 = 5
_mt5_mock.TIMEFRAME_M6 = 6
_mt5_mock.TIMEFRAME_M10 = 10
_mt5_mock.TIMEFRAME_M12 = 12
_mt5_mock.TIMEFRAME_M15 = 15
_mt5_mock.TIMEFRAME_M20 = 20
_mt5_mock.TIMEFRAME_M30 = 30
_mt5_mock.TIMEFRAME_H1 = 16385
_mt5_mock.TIMEFRAME_H2 = 16386
_mt5_mock.TIMEFRAME_H3 = 16387
_mt5_mock.TIMEFRAME_H4 = 16388
_mt5_mock.TIMEFRAME_H6 = 16390
_mt5_mock.TIMEFRAME_H8 = 16392
_mt5_mock.TIMEFRAME_H12 = 16396
_mt5_mock.TIMEFRAME_D1 = 16408
_mt5_mock.TIMEFRAME_W1 = 32769
_mt5_mock.TIMEFRAME_MN1 = 49153
sys.modules["MetaTrader5"] = _mt5_mock

import mtdata.utils.mt5 as _mt5_mod
_mt5_mod.mt5 = _mt5_mock

from mtdata.utils.mt5 import (
    get_symbol_info_cached,
    clear_symbol_info_cache,
    _mt5_epoch_to_utc,
    _rates_to_df,
    _to_server_naive_dt,
    _normalize_times_in_struct,
    _mt5_copy_rates_from,
    _mt5_copy_rates_range,
    _mt5_copy_ticks_from,
    _mt5_copy_rates_from_pos,
    _mt5_copy_ticks_range,
    MT5Connection,
    MT5Adapter,
    MT5Service,
    _auto_connect_wrapper,
    _ensure_symbol_ready,
    _symbol_ready_guard,
    estimate_server_offset,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_tick(**kw):
    t = MagicMock()
    for k, v in kw.items():
        setattr(t, k, v)
    return t


# ── get_symbol_info_cached  (lines 23-32) ────────────────────────────────────

class TestGetSymbolInfoCached:
    def setup_method(self):
        clear_symbol_info_cache()

    def test_positive_ttl(self):
        _mt5_mock.symbol_info.return_value = MagicMock(bid=1.1)
        result = get_symbol_info_cached("EURUSD", ttl_seconds=5)
        assert result is not None

    def test_zero_ttl_calls_raw(self):
        _mt5_mock.symbol_info.return_value = MagicMock(bid=1.2)
        result = get_symbol_info_cached("EURUSD", ttl_seconds=0)
        _mt5_mock.symbol_info.assert_called_with("EURUSD")
        assert result is not None

    def test_negative_ttl_calls_raw(self):
        _mt5_mock.symbol_info.return_value = MagicMock(bid=1.3)
        result = get_symbol_info_cached("EURUSD", ttl_seconds=-1)
        assert result is not None

    def test_non_numeric_ttl_falls_back(self):
        _mt5_mock.symbol_info.return_value = MagicMock(bid=1.4)
        result = get_symbol_info_cached("EURUSD", ttl_seconds="bad")
        assert result is not None


class TestClearSymbolInfoCache:
    def test_clears_without_error(self):
        clear_symbol_info_cache()


class TestMt5Adapter:
    def test_adapter_reads_live_sys_modules_binding(self):
        adapter = MT5Adapter()
        original = sys.modules.get("MetaTrader5")
        first = types.SimpleNamespace(VERSION="first")
        second = types.SimpleNamespace(VERSION="second")
        try:
            sys.modules["MetaTrader5"] = first
            assert adapter.VERSION == "first"
            sys.modules["MetaTrader5"] = second
            assert adapter.VERSION == "second"
        finally:
            if original is not None:
                sys.modules["MetaTrader5"] = original


# ── _mt5_epoch_to_utc  (lines 40-59) ─────────────────────────────────────────

class TestMt5EpochToUtc:
    @patch("mtdata.utils.mt5.mt5_config")
    def test_no_tz_with_offset(self, cfg):
        cfg.get_server_tz.return_value = None
        cfg.get_time_offset_seconds.return_value = 7200
        result = _mt5_epoch_to_utc(1000000.0)
        assert result == 1000000.0 - 7200

    @patch("mtdata.utils.mt5.mt5_config")
    def test_with_tz_localize_success(self, cfg):
        tz = MagicMock()
        cfg.get_server_tz.return_value = tz
        dt_local = MagicMock()
        dt_local.astimezone.return_value.timestamp.return_value = 999.0
        tz.localize.return_value = dt_local
        assert _mt5_epoch_to_utc(100000.0) == 999.0

    @patch("mtdata.utils.mt5.mt5_config")
    def test_with_tz_localize_fails_fallback(self, cfg):
        tz = MagicMock()
        cfg.get_server_tz.return_value = tz
        tz.localize.side_effect = Exception("ambiguous")
        dt_replaced = MagicMock()
        dt_replaced.astimezone.return_value.timestamp.return_value = 888.0
        # replace(tzinfo=...) is called on the naive datetime
        with patch("mtdata.utils.mt5.datetime") as dt_cls:
            dt_cls.side_effect = lambda *a, **kw: datetime(*a, **kw)
            result = _mt5_epoch_to_utc(100000.0)
        assert isinstance(result, float)

    @patch("mtdata.utils.mt5.mt5_config")
    def test_exception_returns_raw(self, cfg):
        cfg.get_server_tz.side_effect = Exception("boom")
        assert _mt5_epoch_to_utc(42.0) == 42.0


# ── _rates_to_df  (lines 62-72) ──────────────────────────────────────────────

class TestRatesToDf:
    def test_with_time_column(self):
        rates = [{"time": 1000.0, "close": 1.1}, {"time": 2000.0, "close": 1.2}]
        df = _rates_to_df(rates)
        assert "time" in df.columns
        assert len(df) == 2

    def test_without_time_column(self):
        rates = [{"close": 1.1}]
        df = _rates_to_df(rates)
        assert "close" in df.columns

    def test_empty_rates(self):
        df = _rates_to_df([])
        assert len(df) == 0


# ── _to_server_naive_dt  (lines 75-85) ───────────────────────────────────────

class TestToServerNaiveDt:
    @patch("mtdata.utils.mt5.mt5_config")
    def test_no_tz_returns_same(self, cfg):
        cfg.get_server_tz.return_value = None
        dt = datetime(2024, 1, 1)
        assert _to_server_naive_dt(dt) == dt

    @patch("mtdata.utils.mt5.mt5_config")
    def test_with_tz(self, cfg):
        tz = MagicMock()
        aware_srv = MagicMock()
        aware_srv.replace.return_value = datetime(2024, 1, 1, 3, 0)
        tz.__class__ = type(tz)  # keep mock
        # Simulate: aware_utc.astimezone(tz) -> aware_srv
        with patch("mtdata.utils.mt5.datetime", wraps=datetime) as dt_cls:
            cfg.get_server_tz.return_value = tz
            result = _to_server_naive_dt(datetime(2024, 1, 1))
        assert isinstance(result, datetime)

    @patch("mtdata.utils.mt5.mt5_config")
    def test_exception_returns_original(self, cfg):
        cfg.get_server_tz.side_effect = Exception("fail")
        dt = datetime(2024, 6, 15)
        assert _to_server_naive_dt(dt) == dt


# ── _normalize_times_in_struct  (lines 88-102) ───────────────────────────────

class TestNormalizeTimesInStruct:
    def test_none_input(self):
        assert _normalize_times_in_struct(None) is None

    def test_no_dtype(self):
        result = _normalize_times_in_struct([1, 2, 3])
        assert result == [1, 2, 3]

    def test_no_time_in_names(self):
        dt = np.dtype([("close", float)])
        arr = np.array([(1.1,)], dtype=dt)
        result = _normalize_times_in_struct(arr)
        assert result is not None

    def test_with_time_field(self):
        dt = np.dtype([("time", float), ("close", float)])
        arr = np.array([(1000.0, 1.1), (2000.0, 1.2)], dtype=dt)
        with patch("mtdata.utils.mt5._mt5_epoch_to_utc", side_effect=lambda x: x + 1):
            result = _normalize_times_in_struct(arr)
        assert float(result[0]["time"]) == 1001.0

    def test_exception_returns_input(self):
        obj = MagicMock()
        obj.dtype = MagicMock()
        obj.dtype.names = ("time",)
        obj.__len__ = MagicMock(side_effect=TypeError)
        result = _normalize_times_in_struct(obj)
        assert result is obj


# ── MT5 copy wrappers  (lines 105-133) ───────────────────────────────────────

class TestCopyWrappers:
    def test_copy_rates_from(self):
        _mt5_mock.copy_rates_from.return_value = None
        with patch("mtdata.utils.mt5._to_server_naive_dt", return_value=datetime(2024, 1, 1)):
            with patch("mtdata.utils.mt5._normalize_times_in_struct", return_value=None):
                result = _mt5_copy_rates_from("EURUSD", 1, datetime(2024, 1, 1), 100)
        assert result is None

    def test_copy_rates_range(self):
        _mt5_mock.copy_rates_range.return_value = None
        with patch("mtdata.utils.mt5._to_server_naive_dt", return_value=datetime(2024, 1, 1)):
            with patch("mtdata.utils.mt5._normalize_times_in_struct", return_value=None):
                result = _mt5_copy_rates_range("EURUSD", 1, datetime(2024, 1, 1), datetime(2024, 1, 2))
        assert result is None

    def test_copy_ticks_from(self):
        _mt5_mock.copy_ticks_from.return_value = None
        with patch("mtdata.utils.mt5._to_server_naive_dt", return_value=datetime(2024, 1, 1)):
            with patch("mtdata.utils.mt5._normalize_times_in_struct", return_value=None):
                result = _mt5_copy_ticks_from("EURUSD", datetime(2024, 1, 1), 100, 0)
        assert result is None

    def test_copy_rates_from_pos(self):
        _mt5_mock.copy_rates_from_pos.return_value = None
        with patch("mtdata.utils.mt5._normalize_times_in_struct", return_value=None):
            result = _mt5_copy_rates_from_pos("EURUSD", 1, 0, 100)
        assert result is None

    def test_copy_ticks_range(self):
        _mt5_mock.copy_ticks_range.return_value = None
        with patch("mtdata.utils.mt5._to_server_naive_dt", return_value=datetime(2024, 1, 1)):
            with patch("mtdata.utils.mt5._normalize_times_in_struct", return_value=None):
                result = _mt5_copy_ticks_range("EURUSD", datetime(2024, 1, 1), datetime(2024, 1, 2), 0)
        assert result is None


# ── MT5Connection  (lines 136-178) ───────────────────────────────────────────

class TestMT5Connection:
    def test_initial_state(self):
        conn = MT5Connection()
        assert conn.connected is False

    def test_is_connected_when_not_connected(self):
        conn = MT5Connection()
        assert conn.is_connected() is False

    def test_is_connected_true(self):
        conn = MT5Connection()
        conn.connected = True
        ti = MagicMock()
        ti.connected = True
        _mt5_mock.terminal_info.return_value = ti
        assert conn.is_connected() is True

    def test_is_connected_terminal_none(self):
        conn = MT5Connection()
        conn.connected = True
        _mt5_mock.terminal_info.return_value = None
        assert conn.is_connected() is False

    @patch("mtdata.utils.mt5.mt5_config")
    def test_ensure_connection_with_credentials_success(self, cfg):
        conn = MT5Connection()
        _mt5_mock.terminal_info.return_value = None
        cfg.has_credentials.return_value = True
        cfg.get_login.return_value = 12345
        cfg.get_password.return_value = "pass"
        cfg.get_server.return_value = "Demo"
        _mt5_mock.initialize.return_value = True
        assert conn._ensure_connection() is True
        assert conn.connected is True

    @patch("mtdata.utils.mt5.mt5_config")
    def test_ensure_connection_cred_fail_fallback_success(self, cfg):
        conn = MT5Connection()
        _mt5_mock.terminal_info.return_value = None
        cfg.has_credentials.return_value = True
        cfg.get_login.return_value = 12345
        cfg.get_password.return_value = "pass"
        cfg.get_server.return_value = "Demo"
        _mt5_mock.initialize.side_effect = [False, True]
        _mt5_mock.last_error.return_value = (-1, "err")
        assert conn._ensure_connection() is True

    @patch("mtdata.utils.mt5.mt5_config")
    def test_ensure_connection_cred_fail_fallback_fail(self, cfg):
        conn = MT5Connection()
        _mt5_mock.terminal_info.return_value = None
        cfg.has_credentials.return_value = True
        cfg.get_login.return_value = 12345
        cfg.get_password.return_value = "pass"
        cfg.get_server.return_value = "Demo"
        _mt5_mock.initialize.side_effect = [False, False]
        _mt5_mock.last_error.return_value = (-1, "err")
        assert conn._ensure_connection() is False

    @patch("mtdata.utils.mt5.mt5_config")
    def test_ensure_connection_no_cred_success(self, cfg):
        conn = MT5Connection()
        _mt5_mock.terminal_info.return_value = None
        cfg.has_credentials.return_value = False
        _mt5_mock.initialize.reset_mock(side_effect=True)
        _mt5_mock.initialize.return_value = True
        assert conn._ensure_connection() is True

    @patch("mtdata.utils.mt5.mt5_config")
    def test_ensure_connection_no_cred_fail(self, cfg):
        conn = MT5Connection()
        _mt5_mock.terminal_info.return_value = None
        cfg.has_credentials.return_value = False
        _mt5_mock.initialize.return_value = False
        _mt5_mock.last_error.return_value = (-1, "err")
        assert conn._ensure_connection() is False

    @patch("mtdata.utils.mt5.mt5_config")
    def test_ensure_connection_exception(self, cfg):
        conn = MT5Connection()
        _mt5_mock.terminal_info.return_value = None
        cfg.has_credentials.side_effect = Exception("boom")
        assert conn._ensure_connection() is False

    def test_disconnect_when_connected(self):
        conn = MT5Connection()
        conn.connected = True
        conn.disconnect()
        assert conn.connected is False
        _mt5_mock.shutdown.assert_called()

    def test_disconnect_when_not_connected(self):
        conn = MT5Connection()
        conn.disconnect()
        assert conn.connected is False


# ── MT5Service  (lines 183-196) ──────────────────────────────────────────────

class TestMT5Service:
    def test_default_connection(self):
        svc = MT5Service()
        assert svc.connection is not None

    def test_custom_connection(self):
        c = MT5Connection()
        svc = MT5Service(connection=c)
        assert svc.connection is c

    def test_ensure_connected(self):
        c = MagicMock()
        c._ensure_connection.return_value = True
        svc = MT5Service(connection=c)
        assert svc.ensure_connected() is True

    def test_disconnect(self):
        c = MagicMock()
        svc = MT5Service(connection=c)
        svc.disconnect()
        c.disconnect.assert_called_once()


# ── _auto_connect_wrapper  (lines 199-218) ──────────────────────────────────

class TestAutoConnectWrapper:
    def test_as_decorator_connected(self):
        svc = MagicMock()
        svc.ensure_connected.return_value = True

        @_auto_connect_wrapper(service=svc)
        def my_func():
            return "ok"

        assert my_func() == "ok"

    def test_as_decorator_not_connected(self):
        svc = MagicMock()
        svc.ensure_connected.return_value = False

        @_auto_connect_wrapper(service=svc)
        def my_func():
            return "ok"

        result = my_func()
        assert "error" in result

    def test_as_bare_decorator(self):
        # Use the global mt5_service; patch its ensure_connected
        with patch("mtdata.utils.mt5.mt5_service") as svc:
            svc.ensure_connected.return_value = True

            @_auto_connect_wrapper
            def my_func():
                return "bare"

            assert my_func() == "bare"


# ── _ensure_symbol_ready  (lines 221-245) ────────────────────────────────────

class TestEnsureSymbolReady:
    def test_select_fails(self):
        _mt5_mock.symbol_info.return_value = MagicMock(visible=True)
        _mt5_mock.symbol_select.return_value = False
        _mt5_mock.last_error.return_value = (-1, "err")
        err = _ensure_symbol_ready("BAD")
        assert err is not None and "Failed to select" in err

    def test_was_visible_false_waits_for_tick(self):
        info = MagicMock(visible=False)
        _mt5_mock.symbol_info.return_value = info
        _mt5_mock.symbol_select.return_value = True
        tick = MagicMock(time=1000, bid=1.1, ask=1.2)
        _mt5_mock.symbol_info_tick.return_value = tick
        err = _ensure_symbol_ready("EURUSD")
        assert err is None

    def test_tick_none_returns_error(self):
        _mt5_mock.symbol_info.return_value = MagicMock(visible=True)
        _mt5_mock.symbol_select.return_value = True
        _mt5_mock.symbol_info_tick.return_value = None
        _mt5_mock.last_error.return_value = (-1, "err")
        err = _ensure_symbol_ready("EURUSD")
        assert err is not None

    def test_info_none(self):
        _mt5_mock.symbol_info.return_value = None
        _mt5_mock.symbol_select.return_value = True
        _mt5_mock.symbol_info_tick.return_value = MagicMock(time=1, bid=1, ask=1)
        err = _ensure_symbol_ready("EURUSD")
        assert err is None

    def test_exception(self):
        _mt5_mock.symbol_info.side_effect = Exception("boom")
        err = _ensure_symbol_ready("EURUSD")
        assert "Error ensuring" in err
        _mt5_mock.symbol_info.side_effect = None


# ── _symbol_ready_guard  (lines 248-264) ─────────────────────────────────────

class TestSymbolReadyGuard:
    def test_guard_restores_visibility(self):
        info = MagicMock(visible=False)
        _mt5_mock.symbol_info.return_value = info
        _mt5_mock.symbol_select.return_value = True
        _mt5_mock.symbol_info_tick.return_value = MagicMock(time=1, bid=1, ask=1)
        with _symbol_ready_guard("EURUSD") as (err, inf):
            pass
        _mt5_mock.symbol_select.assert_called_with("EURUSD", False)

    def test_guard_no_restore_when_visible(self):
        info = MagicMock(visible=True)
        _mt5_mock.symbol_info.return_value = info
        _mt5_mock.symbol_select.return_value = True
        _mt5_mock.symbol_info_tick.return_value = MagicMock(time=1, bid=1, ask=1)
        _mt5_mock.symbol_select.reset_mock()
        with _symbol_ready_guard("EURUSD", info_before=info) as (err, inf):
            pass
        # symbol_select called with True for selection, but NOT False for restore
        calls = [c for c in _mt5_mock.symbol_select.call_args_list if c == (("EURUSD", False),)]
        assert len(calls) == 0


# ── estimate_server_offset  (lines 267-307) ──────────────────────────────────

class TestEstimateServerOffset:
    @patch("mtdata.utils.mt5.mt5_connection")
    def test_connection_fails(self, conn):
        conn._ensure_connection.return_value = False
        assert estimate_server_offset() == 0

    @patch("mtdata.utils.mt5.time")
    @patch("mtdata.utils.mt5.mt5_connection")
    def test_success(self, conn, mock_time):
        conn._ensure_connection.return_value = True
        _mt5_mock.symbol_select.return_value = True
        now = 1700000000.0
        mock_time.time.return_value = now
        mock_time.sleep = MagicMock()
        tick = MagicMock()
        tick.time = now + 7200  # server is UTC+2
        _mt5_mock.symbol_info_tick.return_value = tick
        result = estimate_server_offset("EURUSD", samples=3)
        assert result == 7200

    @patch("mtdata.utils.mt5.time")
    @patch("mtdata.utils.mt5.mt5_connection")
    def test_no_ticks(self, conn, mock_time):
        conn._ensure_connection.return_value = True
        _mt5_mock.symbol_select.return_value = True
        mock_time.time.return_value = 1700000000.0
        mock_time.sleep = MagicMock()
        _mt5_mock.symbol_info_tick.return_value = None
        assert estimate_server_offset("EURUSD", samples=2) == 0

    @patch("mtdata.utils.mt5.time")
    @patch("mtdata.utils.mt5.mt5_connection")
    def test_fallback_symbol(self, conn, mock_time):
        conn._ensure_connection.return_value = True
        # First symbol_select fails, then GBPUSD succeeds
        _mt5_mock.symbol_select.side_effect = [False, True]
        mock_time.time.return_value = 1700000000.0
        mock_time.sleep = MagicMock()
        tick = MagicMock()
        tick.time = 1700000000.0
        _mt5_mock.symbol_info_tick.return_value = tick
        result = estimate_server_offset("EURUSD", samples=1)
        assert isinstance(result, int)
        _mt5_mock.symbol_select.side_effect = None

    @patch("mtdata.utils.mt5.mt5_connection")
    def test_exception(self, conn):
        conn._ensure_connection.side_effect = Exception("boom")
        assert estimate_server_offset() == 0
        conn._ensure_connection.side_effect = None
