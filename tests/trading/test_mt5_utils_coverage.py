"""Coverage tests for mtdata.utils.mt5 – targeting uncovered lines."""

import sys
import time
import types
from collections import namedtuple
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, PropertyMock, patch

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
    MT5Adapter,
    MT5Connection,
    MT5ConnectionError,
    MT5Service,
    _ensure_symbol_ready,
    _mt5_copy_rates_from,
    _mt5_copy_rates_from_pos,
    _mt5_copy_rates_range,
    _mt5_copy_ticks_from,
    _mt5_copy_ticks_range,
    _mt5_epoch_to_utc,
    _normalize_object_time_rows,
    _normalize_object_times,
    _normalize_times_in_struct,
    _rates_to_df,
    _symbol_ready_guard,
    _to_utc_history_query_dt,
    _to_server_naive_dt,
    clear_mt5_time_alignment_cache,
    clear_symbol_info_cache,
    ensure_mt5_connection_or_raise,
    estimate_server_offset,
    get_cached_mt5_time_alignment,
    get_symbol_info_cached,
    inspect_mt5_time_alignment,
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
        _mt5_mock.symbol_info.reset_mock()
        _mt5_mock.symbol_info.side_effect = None

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

    def test_non_numeric_ttl_falls_back_to_raw_no_cache(self):
        _mt5_mock.symbol_info.return_value = MagicMock(bid=1.4)
        first = get_symbol_info_cached("EURUSD", ttl_seconds="bad")
        second = get_symbol_info_cached("EURUSD", ttl_seconds="bad")
        assert first is not None
        assert second is not None
        assert _mt5_mock.symbol_info.call_count == 2

    @pytest.mark.parametrize("ttl_seconds", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_ttl_falls_back_to_raw_no_cache(self, ttl_seconds):
        _mt5_mock.symbol_info.return_value = MagicMock(bid=1.5)
        first = get_symbol_info_cached("EURUSD", ttl_seconds=ttl_seconds)
        second = get_symbol_info_cached("EURUSD", ttl_seconds=ttl_seconds)
        assert first is not None
        assert second is not None
        assert _mt5_mock.symbol_info.call_count == 2

    def test_fractional_ttl_uses_cache_bucket(self, monkeypatch):
        _mt5_mock.symbol_info.return_value = MagicMock(bid=1.6)
        monkeypatch.setattr(_mt5_mod.time, "monotonic_ns", lambda: 10_000_000_000)

        first = get_symbol_info_cached("EURUSD", ttl_seconds=0.5)
        second = get_symbol_info_cached("EURUSD", ttl_seconds=0.5)

        assert first is second
        assert _mt5_mock.symbol_info.call_count == 1

    def test_tiny_positive_ttl_uses_nanosecond_bucket_without_overflow(self, monkeypatch):
        _mt5_mock.symbol_info.return_value = MagicMock(bid=1.7)
        monkeypatch.setattr(_mt5_mod.time, "monotonic_ns", lambda: 100)

        first = get_symbol_info_cached("EURUSD", ttl_seconds=1e-300)
        second = get_symbol_info_cached("EURUSD", ttl_seconds=1e-300)

        assert first is second
        assert _mt5_mock.symbol_info.call_count == 1

    def test_large_ttl_is_capped_to_short_lived_window(self, monkeypatch):
        current_time = {"value": 10_000_000_000_000}
        _mt5_mock.symbol_info.return_value = MagicMock(bid=1.8)
        monkeypatch.setattr(_mt5_mod.time, "monotonic_ns", lambda: current_time["value"])

        first = get_symbol_info_cached("EURUSD", ttl_seconds=10_000_000_000)
        current_time["value"] += 4_000_000_000_000
        second = get_symbol_info_cached("EURUSD", ttl_seconds=10_000_000_000)

        assert first is not None
        assert second is not None
        assert _mt5_mock.symbol_info.call_count == 2


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

    def test_adapter_normalizes_symbol_tick_time(self):
        Tick = namedtuple("Tick", ["time", "time_msc", "bid"])
        _mt5_mock.symbol_info_tick.return_value = Tick(time=7200, time_msc=7_200_000, bid=1.1)

        with patch.dict(sys.modules, {"MetaTrader5": _mt5_mock}), patch(
            "mtdata.utils.mt5._mt5_epoch_to_utc",
            side_effect=lambda value: value - 3600,
        ):
            result = MT5Adapter().symbol_info_tick("EURUSD")

        assert hasattr(result, "_asdict")
        assert result.time == 3600
        assert result.time_msc == 3_600_000
        assert result.bid == 1.1

    def test_adapter_normalizes_position_order_and_history_rows(self):
        Position = namedtuple("Position", ["ticket", "time_update"])
        Order = namedtuple("Order", ["ticket", "time_setup", "time_expiration"])
        Deal = namedtuple("Deal", ["ticket", "time", "time_msc"])
        _mt5_mock.positions_get.return_value = (Position(1, 7200),)
        _mt5_mock.orders_get.return_value = [Order(2, 7200, 0)]
        _mt5_mock.history_deals_get.return_value = (Deal(3, 7200, 7_200_000),)

        with patch.dict(sys.modules, {"MetaTrader5": _mt5_mock}), patch(
            "mtdata.utils.mt5._mt5_epoch_to_utc",
            side_effect=lambda value: value - 3600,
        ):
            adapter = MT5Adapter()
            positions = adapter.positions_get()
            orders = adapter.orders_get()
            deals = adapter.history_deals_get(datetime(2024, 1, 1), datetime(2024, 1, 2))

        assert positions[0].time_update == 3600
        assert orders[0].time_setup == 3600
        assert orders[0].time_expiration == 0
        assert deals[0].time == 3600
        assert deals[0].time_msc == 3_600_000


class TestNormalizeObjectTimes:
    def test_preserves_namedtuple_shape_and_zero_expiration(self):
        Order = namedtuple("Order", ["ticket", "time_setup", "time_expiration"])
        order = Order(ticket=1, time_setup=7200, time_expiration=0)

        with patch("mtdata.utils.mt5._mt5_epoch_to_utc", side_effect=lambda value: value - 3600):
            result = _normalize_object_times(order)

        assert hasattr(result, "_asdict")
        assert result.time_setup == 3600
        assert result.time_expiration == 0

    def test_normalizes_list_rows(self):
        Position = namedtuple("Position", ["ticket", "time_update"])
        rows = [Position(ticket=1, time_update=7200)]

        with patch("mtdata.utils.mt5._mt5_epoch_to_utc", side_effect=lambda value: value - 3600):
            result = _normalize_object_time_rows(rows)

        assert isinstance(result, list)
        assert result[0].time_update == 3600


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
    def test_with_tz_localize_resolves_ambiguous_time(self, cfg, caplog):
        tz = MagicMock()
        cfg.get_server_tz.return_value = tz
        dt_local = MagicMock()
        dt_local.astimezone.return_value.timestamp.return_value = 888.0
        tz.localize.side_effect = [_mt5_mod.AmbiguousTimeError("ambiguous"), dt_local]

        result = _mt5_epoch_to_utc(100000.0)

        assert result == 888.0
        assert tz.localize.call_args_list[0].args[0] == datetime(1970, 1, 2, 3, 46, 40)
        assert tz.localize.call_args_list[0].kwargs == {"is_dst": None}
        assert tz.localize.call_args_list[1].args[0] == datetime(1970, 1, 2, 3, 46, 40)
        assert tz.localize.call_args_list[1].kwargs == {"is_dst": False}
        assert any("Ambiguous MT5 server-local time" in record.message for record in caplog.records)

    @patch("mtdata.utils.mt5.mt5_config")
    def test_with_tz_localize_shifts_nonexistent_time(self, cfg, caplog):
        tz = MagicMock()
        cfg.get_server_tz.return_value = tz
        dt_local = MagicMock()
        dt_local.astimezone.return_value.timestamp.return_value = 777.0
        tz.localize.side_effect = [_mt5_mod.NonExistentTimeError("missing"), dt_local]

        result = _mt5_epoch_to_utc(100000.0)

        assert result == 777.0
        assert tz.localize.call_args_list[0].args[0] == datetime(1970, 1, 2, 3, 46, 40)
        assert tz.localize.call_args_list[0].kwargs == {"is_dst": None}
        assert tz.localize.call_args_list[1].args[0] == datetime(1970, 1, 2, 4, 46, 40)
        assert tz.localize.call_args_list[1].kwargs == {"is_dst": False}
        assert any("Non-existent MT5 server-local time" in record.message for record in caplog.records)

    @patch("mtdata.utils.mt5.mt5_config")
    def test_exception_returns_raw(self, cfg, caplog):
        cfg.get_server_tz.side_effect = Exception("boom")
        result = _mt5_epoch_to_utc(42.0)
        assert result == 42.0
        assert any("Failed to convert MT5 epoch 42.0 to UTC" in record.message for record in caplog.records)


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
        cfg.get_time_offset_seconds.return_value = 0
        dt = datetime(2024, 1, 1)
        assert _to_server_naive_dt(dt) == dt

    @patch("mtdata.utils.mt5.mt5_config")
    def test_static_offset_is_applied_when_timezone_is_unset(self, cfg):
        cfg.get_server_tz.return_value = None
        cfg.get_time_offset_seconds.return_value = 7200
        dt = datetime(2024, 1, 1, 12, 0)

        assert _to_server_naive_dt(dt) == datetime(2024, 1, 1, 14, 0)

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
    def test_exception_returns_original(self, cfg, caplog):
        cfg.get_server_tz.side_effect = Exception("fail")
        dt = datetime(2024, 6, 15)
        assert _to_server_naive_dt(dt) == dt
        assert any("Failed to convert UTC datetime" in record.message for record in caplog.records)


class TestToUtcHistoryQueryDt:
    def test_naive_datetime_is_normalized_to_utc(self):
        dt = datetime(2024, 1, 1, 12, 0)

        assert _to_utc_history_query_dt(dt) == datetime(
            2024, 1, 1, 12, 0, tzinfo=timezone.utc
        )

    def test_aware_datetime_is_converted_to_utc(self):
        dt = datetime(2024, 1, 1, 12, 0, tzinfo=timezone(timedelta(hours=2)))

        assert _to_utc_history_query_dt(dt) == datetime(
            2024, 1, 1, 10, 0, tzinfo=timezone.utc
        )


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

    def test_with_read_only_time_field_returns_normalized_copy(self):
        dt = np.dtype([("time", int), ("close", float)])
        arr = np.array([(1000, 1.1), (2000, 1.2)], dtype=dt)
        arr.flags.writeable = False
        with patch("mtdata.utils.mt5._mt5_epoch_to_utc", side_effect=lambda x: x - 10):
            result = _normalize_times_in_struct(arr)
        assert result is not arr
        assert float(result[0]["time"]) == 990.0
        assert float(arr[0]["time"]) == 1000.0

    @patch("mtdata.utils.mt5.mt5_config")
    def test_with_time_field_uses_static_offset_fast_path(self, cfg):
        dt = np.dtype([("time", float), ("close", float)])
        arr = np.array([(1000.0, 1.1), (2000.0, 1.2)], dtype=dt)
        cfg.get_server_tz.return_value = None
        cfg.get_time_offset_seconds.return_value = 60
        sentinel = MagicMock(side_effect=AssertionError("slow path should not run"))

        with patch("mtdata.utils.mt5._mt5_epoch_to_utc", new=sentinel), patch(
            "mtdata.utils.mt5._DEFAULT_MT5_EPOCH_TO_UTC",
            new=sentinel,
        ):
            result = _normalize_times_in_struct(arr)

        assert float(result[0]["time"]) == 940.0
        assert float(result[1]["time"]) == 1940.0
        sentinel.assert_not_called()

    @patch("mtdata.utils.mt5.mt5_config")
    def test_static_offset_fast_path_leaves_zero_expiration_unchanged(self, cfg):
        dt = np.dtype([("time_expiration", float), ("price", float)])
        arr = np.array([(0.0, 1.1), (7200.0, 1.2)], dtype=dt)
        cfg.get_server_tz.return_value = None
        cfg.get_time_offset_seconds.return_value = 3600

        result = _normalize_times_in_struct(arr)

        assert float(result[0]["time_expiration"]) == 0.0
        assert float(result[1]["time_expiration"]) == 3600.0

    def test_exception_returns_input(self):
        obj = MagicMock()
        obj.dtype = MagicMock()
        obj.dtype.names = ("time",)
        obj.__len__ = MagicMock(side_effect=TypeError)
        result = _normalize_times_in_struct(obj)
        assert result is obj

    def test_element_conversion_exception_logs_and_keeps_raw_time(self, caplog):
        dt = np.dtype([("time", float), ("close", float)])
        arr = np.array([(1000.0, 1.1), (2000.0, 1.2)], dtype=dt)

        def _convert(value):
            if value == 1000.0:
                raise ValueError("bad element")
            return value + 1.0

        with patch("mtdata.utils.mt5._mt5_epoch_to_utc", side_effect=_convert):
            result = _normalize_times_in_struct(arr)

        assert float(result[0]["time"]) == 1000.0
        assert float(result[1]["time"]) == 2001.0
        assert any("Failed to normalize MT5 timestamp at index 0" in record.message for record in caplog.records)


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

    @patch("mtdata.utils.mt5.clear_mt5_time_alignment_cache")
    @patch("mtdata.utils.mt5.clear_symbol_info_cache")
    @patch("mtdata.utils.mt5.mt5_config")
    def test_ensure_connection_success_clears_mt5_caches(self, cfg, clear_symbol_cache, clear_alignment_cache):
        conn = MT5Connection()
        _mt5_mock.terminal_info.return_value = None
        _mt5_mock.account_info.return_value = MagicMock(login=12345, server="Demo")
        cfg.has_credentials.return_value = False
        _mt5_mock.initialize.return_value = True

        assert conn._ensure_connection() is True
        clear_symbol_cache.assert_called_once()
        clear_alignment_cache.assert_called_once()

    @patch("mtdata.utils.mt5.clear_mt5_time_alignment_cache")
    @patch("mtdata.utils.mt5.clear_symbol_info_cache")
    def test_connected_session_refresh_clears_caches_when_identity_changes(self, clear_symbol_cache, clear_alignment_cache):
        conn = MT5Connection()
        conn.connected = True
        conn._connection_identity = (12345, "Demo-A")
        _mt5_mock.terminal_info.return_value = MagicMock(connected=True, server="Demo-B")
        _mt5_mock.account_info.return_value = MagicMock(login=67890, server="Demo-B")

        assert conn._ensure_connection() is True
        clear_symbol_cache.assert_called_once()
        clear_alignment_cache.assert_called_once()
        assert conn._connection_identity == (67890, "Demo-B")

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


class TestEnsureMt5ConnectionOrRaise:
    def test_returns_when_service_connects(self):
        svc = MagicMock()
        svc.ensure_connected.return_value = True
        assert ensure_mt5_connection_or_raise(service=svc) is None

    def test_raises_when_service_returns_false(self):
        svc = MagicMock()
        svc.ensure_connected.return_value = False
        with pytest.raises(MT5ConnectionError, match="Failed to connect to MetaTrader5"):
            ensure_mt5_connection_or_raise(service=svc)

    def test_wraps_unexpected_service_errors(self):
        svc = MagicMock()
        svc.ensure_connected.side_effect = RuntimeError("boom")
        with pytest.raises(MT5ConnectionError, match="Failed to connect to MetaTrader5"):
            ensure_mt5_connection_or_raise(service=svc)


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
    @patch("mtdata.utils.mt5.ensure_mt5_connection_or_raise",
           side_effect=MT5ConnectionError("no connection"))
    def test_connection_fails(self, _mock_conn):
        assert estimate_server_offset() == 0

    @patch("mtdata.utils.mt5.time")
    @patch("mtdata.utils.mt5.ensure_mt5_connection_or_raise")
    def test_success(self, _mock_conn, mock_time):
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
    @patch("mtdata.utils.mt5.ensure_mt5_connection_or_raise")
    def test_no_ticks(self, _mock_conn, mock_time):
        _mt5_mock.symbol_select.return_value = True
        mock_time.time.return_value = 1700000000.0
        mock_time.sleep = MagicMock()
        _mt5_mock.symbol_info_tick.return_value = None
        assert estimate_server_offset("EURUSD", samples=2) == 0

    @patch("mtdata.utils.mt5.time")
    @patch("mtdata.utils.mt5.ensure_mt5_connection_or_raise")
    def test_fallback_symbol(self, _mock_conn, mock_time):
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

    @patch("mtdata.utils.mt5.ensure_mt5_connection_or_raise",
           side_effect=Exception("boom"))
    def test_exception(self, _mock_conn):
        assert estimate_server_offset() == 0


class TestCachedMt5TimeAlignment:
    def setup_method(self):
        clear_mt5_time_alignment_cache()

    def teardown_method(self):
        clear_mt5_time_alignment_cache()

    def test_uses_cache_within_ttl_bucket(self, monkeypatch):
        calls = {"count": 0}

        def fake_inspect(**kwargs):
            calls["count"] += 1
            return {"status": "ok", "sequence": calls["count"], "symbol": kwargs["symbol"]}

        monkeypatch.setattr(_mt5_mod, "inspect_mt5_time_alignment", fake_inspect)
        monkeypatch.setattr(_mt5_mod.time, "time", lambda: 120.0)

        first = get_cached_mt5_time_alignment("EURUSD", ttl_seconds=60)
        second = get_cached_mt5_time_alignment("EURUSD", ttl_seconds=60)

        assert calls["count"] == 1
        assert first["sequence"] == 1
        assert second["sequence"] == 1
        assert first is not second

    def test_ttl_zero_bypasses_cache(self, monkeypatch):
        calls = {"count": 0}

        def fake_inspect(**kwargs):
            calls["count"] += 1
            return {"status": "ok", "sequence": calls["count"]}

        monkeypatch.setattr(_mt5_mod, "inspect_mt5_time_alignment", fake_inspect)

        first = get_cached_mt5_time_alignment("EURUSD", ttl_seconds=0)
        second = get_cached_mt5_time_alignment("EURUSD", ttl_seconds=0)

        assert calls["count"] == 2
        assert first["sequence"] == 1
        assert second["sequence"] == 2


class TestInspectMt5TimeAlignment:
    @patch("mtdata.utils.mt5.mt5_config")
    def test_ok(self, cfg, monkeypatch):
        now = 1_700_000_045.0
        current_bar = float((int(now) // 60) * 60)
        last_closed_bar = current_bar - 60.0
        cfg.get_server_tz.return_value = None
        cfg.get_time_offset_seconds.return_value = 7200
        cfg.server_tz_name = "Europe/Nicosia"

        monkeypatch.setattr(_mt5_mod, "ensure_mt5_connection_or_raise", lambda **kwargs: None)
        monkeypatch.setattr(_mt5_mod, "_ensure_symbol_ready", lambda symbol: None)
        monkeypatch.setattr(_mt5_mod.time, "time", lambda: now)
        monkeypatch.setattr(_mt5_mod.mt5, "symbol_info_tick", lambda symbol: MagicMock(time=now + 7200.0))
        monkeypatch.setattr(
            _mt5_mod,
            "_mt5_copy_rates_from_pos",
            lambda symbol, timeframe, start_pos, count: [
                {"time": last_closed_bar - 60.0},
                {"time": last_closed_bar},
                {"time": current_bar},
            ],
        )

        result = inspect_mt5_time_alignment("EURUSD")

        assert result["status"] == "ok"
        assert result["reason"] is None
        assert result["configured_offset_seconds"] == 7200
        assert result["inferred_offset_seconds"] == 7200
        assert result["configured_server_tz"] == "Europe/Nicosia"
        assert abs(result["current_bar_delta_seconds"]) < 1e-9
        assert abs(result["last_closed_bar_delta_seconds"]) < 1e-9

    @patch("mtdata.utils.mt5.mt5_config")
    def test_reports_misalignment(self, cfg, monkeypatch):
        now = 1_700_000_045.0
        current_bar = float((int(now) // 60) * 60)
        last_closed_bar = current_bar - 60.0
        cfg.get_server_tz.return_value = None
        cfg.get_time_offset_seconds.return_value = 7200

        monkeypatch.setattr(_mt5_mod, "ensure_mt5_connection_or_raise", lambda **kwargs: None)
        monkeypatch.setattr(_mt5_mod, "_ensure_symbol_ready", lambda symbol: None)
        monkeypatch.setattr(_mt5_mod.time, "time", lambda: now)
        monkeypatch.setattr(_mt5_mod.mt5, "symbol_info_tick", lambda symbol: MagicMock(time=now + 10800.0))
        monkeypatch.setattr(
            _mt5_mod,
            "_mt5_copy_rates_from_pos",
            lambda symbol, timeframe, start_pos, count: [
                {"time": last_closed_bar - 60.0},
                {"time": last_closed_bar},
                {"time": current_bar},
            ],
        )

        result = inspect_mt5_time_alignment("EURUSD")

        assert result["status"] == "misaligned"
        assert result["reason"] == "timezone_mismatch"
        assert result["offset_mismatch_seconds"] == 3600
        assert "inferred broker offset is 10800s" in result["warning"]

    @patch("mtdata.utils.mt5.mt5_config")
    def test_reports_stale_data(self, cfg, monkeypatch):
        now = 1_700_000_045.0
        current_bar = float((int(now) // 60) * 60)
        cfg.get_server_tz.return_value = None
        cfg.get_time_offset_seconds.return_value = 7200

        monkeypatch.setattr(_mt5_mod, "ensure_mt5_connection_or_raise", lambda **kwargs: None)
        monkeypatch.setattr(_mt5_mod, "_ensure_symbol_ready", lambda symbol: None)
        monkeypatch.setattr(_mt5_mod.time, "time", lambda: now)
        monkeypatch.setattr(_mt5_mod.mt5, "symbol_info_tick", lambda symbol: MagicMock(time=now + 7200.0))
        monkeypatch.setattr(
            _mt5_mod,
            "_mt5_copy_rates_from_pos",
            lambda symbol, timeframe, start_pos, count: [
                {"time": current_bar - 360.0},
                {"time": current_bar - 300.0},
                {"time": current_bar - 240.0},
            ],
        )

        result = inspect_mt5_time_alignment("EURUSD")

        assert result["status"] == "stale"
        assert result["reason"] == "market_data_stale"
        assert "lags expected current bar" in result["warning"]
        assert result["offset_inference_reliable"] is True

    @patch("mtdata.utils.mt5.mt5_config")
    def test_closed_market_tick_is_not_treated_as_timezone_mismatch(self, cfg, monkeypatch):
        now = 1_700_000_045.0
        current_bar = float((int(now) // 60) * 60)
        cfg.get_server_tz.return_value = None
        cfg.get_time_offset_seconds.return_value = 7200

        monkeypatch.setattr(_mt5_mod, "ensure_mt5_connection_or_raise", lambda **kwargs: None)
        monkeypatch.setattr(_mt5_mod, "_ensure_symbol_ready", lambda symbol: None)
        monkeypatch.setattr(_mt5_mod.time, "time", lambda: now)
        # Simulate a Friday tick sampled on Sunday: unusable for offset inference.
        monkeypatch.setattr(_mt5_mod.mt5, "symbol_info_tick", lambda symbol: MagicMock(time=now - 147600.0))
        monkeypatch.setattr(
            _mt5_mod,
            "_mt5_copy_rates_from_pos",
            lambda symbol, timeframe, start_pos, count: [
                {"time": current_bar - 3600.0},
                {"time": current_bar - 3540.0},
                {"time": current_bar - 3480.0},
            ],
        )

        result = inspect_mt5_time_alignment("EURUSD")

        assert result["status"] == "stale"
        assert result["reason"] == "market_data_stale"
        assert result["offset_inference_reliable"] is False
        assert result["inferred_offset_seconds"] == -147600
        assert "not a plausible live broker offset" in result["warning"]
        assert "timezone_mismatch" != result["reason"]

    def test_connection_failure(self, monkeypatch):
        monkeypatch.setattr(
            _mt5_mod,
            "ensure_mt5_connection_or_raise",
            lambda **kwargs: (_ for _ in ()).throw(MT5ConnectionError("boom")),
        )

        result = inspect_mt5_time_alignment("EURUSD")

        assert result["status"] == "unavailable"
        assert result["reason"] == "connection_failed"
        assert result["error"] == "boom"
