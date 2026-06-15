from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

import mtdata.utils.mt5 as mt5_mod

try:
    import pytz
except Exception:  # pragma: no cover - optional dependency in some environments
    pytz = None


def _server_local_epoch(local_dt: datetime) -> float:
    return (local_dt - datetime(1970, 1, 1)).total_seconds()


def test_describe_mt5_time_normalization_reports_server_timezone(monkeypatch) -> None:
    monkeypatch.setattr(mt5_mod.mt5_config, "server_tz_name", "Europe/Nicosia", raising=False)
    monkeypatch.setattr(mt5_mod.mt5_config, "time_offset_minutes", 0, raising=False)

    meta = mt5_mod.describe_mt5_time_normalization()

    assert meta == {
        "raw_time_basis": "mt5_server_epoch",
        "time_basis": "utc_normalized",
        "time_normalization": "dst_aware_server_timezone",
        "broker_server_tz": "Europe/Nicosia",
    }


def test_describe_mt5_time_normalization_reports_unconfigured_mode(monkeypatch) -> None:
    monkeypatch.setattr(mt5_mod.mt5_config, "server_tz_name", None, raising=False)
    monkeypatch.setattr(mt5_mod.mt5_config, "time_offset_minutes", 0, raising=False)

    meta = mt5_mod.describe_mt5_time_normalization()

    assert meta == {
        "raw_time_basis": "mt5_server_epoch",
        "time_basis": "raw_mt5_server_epoch",
        "time_normalization": "unconfigured",
    }


@pytest.mark.skipif(pytz is None, reason="pytz is required for timezone normalization")
def test_normalize_times_in_struct_vectorizes_server_timezone(monkeypatch) -> None:
    tz = pytz.timezone("Europe/Nicosia")
    winter_local = datetime(2026, 1, 15, 12, 0, 0)
    summer_local = datetime(2026, 6, 15, 12, 0, 0)
    winter_raw = _server_local_epoch(winter_local)
    summer_raw = _server_local_epoch(summer_local)
    arr = np.array(
        [
            (winter_raw, winter_raw * 1000.0),
            (summer_raw, summer_raw * 1000.0),
            (0.0, 0.0),
        ],
        dtype=[("time", float), ("time_msc", float)],
    )

    monkeypatch.setattr(mt5_mod.mt5_config, "get_server_tz", lambda: tz)
    monkeypatch.setattr(mt5_mod.mt5_config, "get_time_offset_seconds", lambda *args, **kwargs: 0)
    monkeypatch.setattr(mt5_mod, "_mt5_epoch_to_utc", mt5_mod._DEFAULT_MT5_EPOCH_TO_UTC)

    result = mt5_mod._normalize_times_in_struct(arr.copy())

    expected_winter = tz.localize(winter_local, is_dst=None).astimezone(timezone.utc).timestamp()
    expected_summer = tz.localize(summer_local, is_dst=None).astimezone(timezone.utc).timestamp()

    assert float(result[0]["time"]) == pytest.approx(expected_winter)
    assert float(result[0]["time_msc"]) == pytest.approx(expected_winter * 1000.0)
    assert float(result[1]["time"]) == pytest.approx(expected_summer)
    assert float(result[1]["time_msc"]) == pytest.approx(expected_summer * 1000.0)
    assert float(result[2]["time"]) == 0.0
    assert float(result[2]["time_msc"]) == 0.0
