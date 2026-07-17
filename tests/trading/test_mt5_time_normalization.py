from __future__ import annotations

import sys
from collections import namedtuple
from datetime import datetime, timezone
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import numpy as np
import pytest

import mtdata.utils.mt5 as mt5_mod


@pytest.fixture(autouse=True)
def _clear_timestamp_mode_cache() -> None:
    mt5_mod.clear_mt5_timestamp_mode_cache()
    yield
    mt5_mod.clear_mt5_timestamp_mode_cache()


def test_describe_mt5_time_normalization_reports_native_utc(monkeypatch) -> None:
    monkeypatch.setattr(mt5_mod.mt5_config, "server_tz_name", "Europe/Nicosia", raising=False)
    monkeypatch.setattr(mt5_mod.mt5_config, "time_offset_minutes", 0, raising=False)

    meta = mt5_mod.describe_mt5_time_normalization()

    assert meta["raw_time_basis"] == "mt5_utc_epoch"
    assert meta["time_basis"] == "utc"
    assert meta["time_normalization"] == "mt5_utc_native"
    assert meta["broker_server_tz"] == "Europe/Nicosia"
    assert "request bounds and returned epochs use native UTC" in meta["timezone_note"]
    assert "session/calendar calculations use Europe/Nicosia" in meta["timezone_note"]


def test_describe_mt5_time_normalization_reports_utc_session_default(monkeypatch) -> None:
    monkeypatch.setattr(mt5_mod.mt5_config, "server_tz_name", None, raising=False)
    monkeypatch.setattr(mt5_mod.mt5_config, "time_offset_minutes", 0, raising=False)

    meta = mt5_mod.describe_mt5_time_normalization()

    assert meta["raw_time_basis"] == "mt5_utc_epoch"
    assert meta["time_basis"] == "utc"
    assert meta["time_normalization"] == "mt5_utc_native"
    assert "session/calendar calculations use UTC" in meta["timezone_note"]


def test_describe_mt5_time_normalization_reports_session_offset(monkeypatch) -> None:
    monkeypatch.setattr(mt5_mod.mt5_config, "server_tz_name", None, raising=False)
    monkeypatch.setattr(mt5_mod.mt5_config, "time_offset_minutes", 120, raising=False)

    meta = mt5_mod.describe_mt5_time_normalization()

    assert meta["session_utc_offset_seconds"] == 7200
    assert meta["time_normalization"] == "mt5_utc_native"


def test_normalize_times_in_struct_preserves_native_utc_epochs(monkeypatch) -> None:
    arr = np.array(
        [(1_768_478_400.0, 1_768_478_400_000.0), (0.0, 0.0)],
        dtype=[("time", float), ("time_msc", float)],
    )
    monkeypatch.setattr(mt5_mod.mt5_config, "server_tz_name", "Europe/Nicosia", raising=False)
    monkeypatch.setattr(mt5_mod.mt5_config, "time_offset_minutes", 120, raising=False)

    result = mt5_mod._normalize_times_in_struct(arr)

    assert result is arr
    assert result.tolist() == arr.tolist()


def test_adapter_aligns_server_clock_tick_history_to_utc(monkeypatch) -> None:
    now = datetime(2026, 7, 14, 14, 45, tzinfo=timezone.utc)
    now_epoch = now.timestamp()
    Tick = namedtuple("Tick", ["time", "time_msc", "bid", "ask"])
    raw_tick = Tick(
        time=int(now_epoch + 3 * 60 * 60),
        time_msc=int((now_epoch + 3 * 60 * 60) * 1000),
        bid=397.4,
        ask=397.5,
    )
    Deal = namedtuple("Deal", ["time", "ticket"])
    raw_deal = Deal(time=int(now_epoch + 3 * 60 * 60), ticket=123)
    position_probe_calls = []
    rows = np.array(
        [(now_epoch + 3 * 60 * 60, 397.4, 397.5)],
        dtype=[("time", float), ("bid", float), ("ask", float)],
    )
    observed_bounds = {}

    def copy_ticks_range(symbol, dt_from, dt_to, flags):
        observed_bounds["from"] = dt_from
        observed_bounds["to"] = dt_to
        return rows

    def history_deals_get(dt_from, dt_to, **kwargs):
        observed_bounds["deals_from"] = dt_from
        observed_bounds["deals_to"] = dt_to
        return (raw_deal,)

    def positions_get():
        position_probe_calls.append(True)
        return ()

    module = SimpleNamespace(
        symbol_info_tick=lambda symbol: raw_tick,
        copy_ticks_range=copy_ticks_range,
        history_deals_get=history_deals_get,
        positions_get=positions_get,
    )
    monkeypatch.setitem(sys.modules, "MetaTrader5", module)
    monkeypatch.setattr(mt5_mod.time, "time", lambda: now_epoch)
    monkeypatch.setattr(
        mt5_mod.mt5_config,
        "get_time_offset_seconds",
        lambda at_time=None: 3 * 60 * 60,
    )
    monkeypatch.setattr(
        mt5_mod.mt5_config,
        "get_server_tz",
        lambda: ZoneInfo("Europe/Nicosia"),
    )
    monkeypatch.setattr(mt5_mod.mt5_config, "time_offset_minutes", 0)

    adapter = mt5_mod.MT5Adapter()
    normalized_tick = adapter.symbol_info_tick("TSLA.NAS-24")
    result = adapter.copy_ticks_range(
        "TSLA.NAS-24",
        now.replace(minute=35),
        now,
        0,
    )
    deals = adapter.history_deals_get(now_epoch - 60, now_epoch)

    assert observed_bounds["from"] == now.replace(minute=35, hour=17)
    assert observed_bounds["to"] == now.replace(hour=17)
    assert float(normalized_tick.time) == now_epoch
    assert float(normalized_tick.time_msc) == now_epoch * 1000
    assert float(result[0]["time"]) == now_epoch
    assert observed_bounds["deals_from"] == now_epoch - 60 + 3 * 60 * 60
    assert observed_bounds["deals_to"] == now_epoch + 3 * 60 * 60
    assert position_probe_calls == []
    assert float(deals[0].time) == now_epoch
    assert mt5_mod.get_mt5_timestamp_mode("TSLA.NAS-24") == "server_clock"


def test_standalone_history_probes_open_position_clock_mode(monkeypatch) -> None:
    now = datetime(2026, 7, 14, 14, 45, tzinfo=timezone.utc)
    now_epoch = now.timestamp()
    Tick = namedtuple("Tick", ["time", "time_msc", "bid", "ask"])
    Position = namedtuple("Position", ["ticket", "symbol", "time"])
    Deal = namedtuple("Deal", ["ticket", "symbol", "time"])
    raw_epoch = int(now_epoch + 3 * 60 * 60)
    observed_bounds = {}

    def history_deals_get(dt_from, dt_to, **kwargs):
        observed_bounds["from"] = dt_from
        observed_bounds["to"] = dt_to
        return (Deal(2, "TSLA.NAS-24", raw_epoch),)

    module = SimpleNamespace(
        positions_get=lambda: (Position(1, "TSLA.NAS-24", raw_epoch),),
        symbol_info_tick=lambda symbol: Tick(
            raw_epoch,
            raw_epoch * 1000,
            397.4,
            397.5,
        ),
        history_deals_get=history_deals_get,
    )
    monkeypatch.setitem(sys.modules, "MetaTrader5", module)
    monkeypatch.setattr(mt5_mod.time, "time", lambda: now_epoch)
    monkeypatch.setattr(
        mt5_mod.mt5_config,
        "get_time_offset_seconds",
        lambda at_time=None: 3 * 60 * 60,
    )
    monkeypatch.setattr(
        mt5_mod.mt5_config,
        "get_server_tz",
        lambda: ZoneInfo("Europe/Nicosia"),
    )
    monkeypatch.setattr(mt5_mod.mt5_config, "time_offset_minutes", 0)

    deals = mt5_mod.MT5Adapter().history_deals_get(
        now_epoch - 60,
        now_epoch,
    )

    assert observed_bounds["from"] == now_epoch - 60 + 3 * 60 * 60
    assert observed_bounds["to"] == now_epoch + 3 * 60 * 60
    assert deals[0].time == now_epoch
    assert mt5_mod.get_mt5_timestamp_mode("TSLA.NAS-24") == "server_clock"


def test_adapter_keeps_native_utc_terminal_unchanged(monkeypatch) -> None:
    now = datetime(2026, 7, 14, 14, 45, tzinfo=timezone.utc)
    now_epoch = now.timestamp()
    Tick = namedtuple("Tick", ["time", "time_msc", "bid", "ask"])
    raw_tick = Tick(
        time=int(now_epoch),
        time_msc=int(now_epoch * 1000),
        bid=397.4,
        ask=397.5,
    )
    rows = np.array(
        [(now_epoch, 397.4, 397.5)],
        dtype=[("time", float), ("bid", float), ("ask", float)],
    )
    observed_bounds = {}

    def copy_ticks_range(symbol, dt_from, dt_to, flags):
        observed_bounds["to"] = dt_to
        return rows

    module = SimpleNamespace(
        symbol_info_tick=lambda symbol: raw_tick,
        copy_ticks_range=copy_ticks_range,
    )
    monkeypatch.setitem(sys.modules, "MetaTrader5", module)
    monkeypatch.setattr(mt5_mod.time, "time", lambda: now_epoch)
    monkeypatch.setattr(
        mt5_mod.mt5_config,
        "get_time_offset_seconds",
        lambda at_time=None: 3 * 60 * 60,
    )
    monkeypatch.setattr(
        mt5_mod.mt5_config,
        "get_server_tz",
        lambda: ZoneInfo("Europe/Nicosia"),
    )
    monkeypatch.setattr(mt5_mod.mt5_config, "time_offset_minutes", 0)

    result = mt5_mod.MT5Adapter().copy_ticks_range(
        "TSLA.NAS-24",
        now.replace(minute=35),
        now,
        0,
    )

    assert observed_bounds["to"] == now
    assert result is rows
    assert mt5_mod.get_mt5_timestamp_mode("TSLA.NAS-24") == "native_utc"


def test_adapter_detects_clock_before_normalizing_positions_and_symbol_info(monkeypatch) -> None:
    now = datetime(2026, 7, 14, 15, 45, tzinfo=timezone.utc)
    now_epoch = now.timestamp()
    Tick = namedtuple("Tick", ["time", "time_msc", "bid", "ask"])
    Position = namedtuple("Position", ["time", "time_msc", "symbol", "ticket"])
    SymbolInfo = namedtuple("SymbolInfo", ["time", "name", "digits"])
    raw_tick = Tick(
        time=int(now_epoch + 3 * 60 * 60),
        time_msc=int((now_epoch + 3 * 60 * 60) * 1000),
        bid=397.4,
        ask=397.5,
    )
    raw_position = Position(
        time=int(now_epoch - 30 * 60 + 3 * 60 * 60),
        time_msc=int((now_epoch - 30 * 60 + 3 * 60 * 60) * 1000),
        symbol="TSLA.NAS-24",
        ticket=123,
    )
    raw_info = SymbolInfo(
        time=int(now_epoch + 3 * 60 * 60),
        name="TSLA.NAS-24",
        digits=2,
    )
    module = SimpleNamespace(
        symbol_info_tick=lambda symbol: raw_tick,
        symbol_info=lambda symbol: raw_info,
        positions_get=lambda **kwargs: (raw_position,),
    )
    monkeypatch.setitem(sys.modules, "MetaTrader5", module)
    monkeypatch.setattr(mt5_mod.time, "time", lambda: now_epoch)
    monkeypatch.setattr(
        mt5_mod.mt5_config,
        "get_time_offset_seconds",
        lambda at_time=None: 3 * 60 * 60,
    )
    monkeypatch.setattr(
        mt5_mod.mt5_config,
        "get_server_tz",
        lambda: ZoneInfo("Europe/Nicosia"),
    )
    monkeypatch.setattr(mt5_mod.mt5_config, "time_offset_minutes", 0)

    adapter = mt5_mod.MT5Adapter()
    positions = adapter.positions_get()
    info = adapter.symbol_info("TSLA.NAS-24")

    assert float(positions[0].time) == now_epoch - 30 * 60
    assert float(positions[0].time_msc) == (now_epoch - 30 * 60) * 1000
    assert float(info.time) == now_epoch
