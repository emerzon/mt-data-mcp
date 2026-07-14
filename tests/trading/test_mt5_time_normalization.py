from __future__ import annotations

import numpy as np

import mtdata.utils.mt5 as mt5_mod


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
