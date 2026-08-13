from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from mtdata.forecast import common as fc


def test_future_as_of_is_rejected_against_wall_clock():
    now = datetime(2026, 8, 12, 14, 0).timestamp()

    assert fc.future_as_of_error("2030-01-01T00:00:00Z", now_epoch=now) == (
        "as_of must not be in the future."
    )
    assert fc.future_as_of_error("2024-01-01T00:00:00Z", now_epoch=now) is None


def test_extract_forecast_values_requires_exact_horizon():
    yf_standard = pd.DataFrame({"pred": [1.0, 2.0]})
    out = fc._extract_forecast_values(yf_standard, fh=2, method_name="m")
    assert out.tolist() == [1.0, 2.0]

    yf_alt = pd.DataFrame({"unique_id": ["ts"], "ds": [0], "pred": [9.0]})
    with pytest.raises(ValueError, match="requested 3, received 1"):
        fc._extract_forecast_values(yf_alt, fh=3, method_name="m")

    yf_with_actuals = pd.DataFrame(
        {
            "unique_id": ["ts"] * 2,
            "ds": [0, 1],
            "y": [1.0, 2.0],
            "pred": [9.0, 10.0],
        }
    )
    out = fc._extract_forecast_values(yf_with_actuals, fh=2, method_name="m")
    assert out.tolist() == [9.0, 10.0]

    with pytest.raises(RuntimeError, match="refusing to use actuals column 'y'"):
        fc._extract_forecast_values(pd.DataFrame({"y": [1.0, 2.0]}), fh=2, method_name="m")

    yf_with_auxiliary = pd.DataFrame(
        {
            "unique_id": ["ts"] * 2,
            "ds": [0, 1],
            "cutoff": ["2026-04-09T00:00:00Z"] * 2,
            "NHITS": [7.5, 8.0],
        }
    )
    out = fc._extract_forecast_values(yf_with_auxiliary, fh=2, method_name="nhits")
    assert out.tolist() == [7.5, 8.0]

    with pytest.raises(RuntimeError, match="prediction columns not found"):
        fc._extract_forecast_values(pd.DataFrame({"unique_id": ["ts"], "ds": [0]}), fh=1, method_name="demo")


def test_create_training_dataframes_with_and_without_exog():
    series = np.array([1.0, 2.0, 3.0], dtype=float)
    exog = np.array([[10.0, 20.0], [11.0, 21.0], [12.0, 22.0]], dtype=float)
    exog_future = np.array([[13.0, 23.0], [14.0, 24.0]], dtype=float)

    y_df, x_df, xf_df = fc._create_training_dataframes(series, fh=2, exog_used=exog, exog_future=exog_future)
    assert list(y_df.columns) == ["unique_id", "ds", "y"]
    assert y_df["y"].tolist() == [1.0, 2.0, 3.0]
    assert x_df is not None and list(x_df.columns) == ["unique_id", "ds", "x0", "x1"]
    assert xf_df is not None and list(xf_df.columns) == ["unique_id", "ds", "x0", "x1"]
    assert xf_df["x0"].tolist() == [13.0, 14.0]

    y_df, x_df, xf_df = fc._create_training_dataframes(series, fh=2, exog_used=None, exog_future=None)
    assert x_df is None
    assert xf_df is None
    assert len(y_df) == 3


def test_create_training_dataframes_accepts_1d_single_column_exog():
    series = np.array([1.0, 2.0, 3.0], dtype=float)
    exog = np.array([10.0, 11.0, 12.0], dtype=float)
    exog_future = np.array([13.0, 14.0], dtype=float)

    _, x_df, xf_df = fc._create_training_dataframes(
        series,
        fh=2,
        exog_used=exog,
        exog_future=exog_future,
    )

    assert x_df is not None
    assert list(x_df.columns) == ["unique_id", "ds", "x0"]
    assert x_df["x0"].tolist() == [10.0, 11.0, 12.0]
    assert xf_df is not None
    assert list(xf_df.columns) == ["unique_id", "ds", "x0"]
    assert xf_df["x0"].tolist() == [13.0, 14.0]


def test_timeframe_helpers_cover_key_branches():
    assert fc.default_seasonality("H1") == 24
    assert fc.default_seasonality("D1") == 5
    assert fc.default_seasonality("W1") == 52
    assert fc.default_seasonality("MN1") == 12
    assert fc.default_seasonality("NOPE") == 0


def test_default_seasonality_uses_observed_session_bar_count() -> None:
    sessions = pd.DatetimeIndex(
        [
            *pd.date_range("2026-01-05 14:30", periods=7, freq="h", tz="UTC"),
            *pd.date_range("2026-01-06 14:30", periods=7, freq="h", tz="UTC"),
            *pd.date_range("2026-01-07 14:30", periods=7, freq="h", tz="UTC"),
        ]
    )

    assert fc.default_seasonality("H1", sessions) == 7

    assert fc.next_times_from_last(100.0, 60, 3) == [160.0, 220.0, 280.0]
    assert fc.pd_freq_from_timeframe("H4") == "4h"
    assert fc.pd_freq_from_timeframe("x") == "D"


def test_fetch_history_validates_inputs_and_symbol_readiness(monkeypatch):
    monkeypatch.setattr(fc, "TIMEFRAME_MAP", {"H1": 1})
    monkeypatch.setattr(fc, "get_symbol_info_cached", lambda _symbol: None)
    monkeypatch.setattr(fc, "_ensure_symbol_ready", lambda _symbol: "symbol error")

    with pytest.raises(RuntimeError, match="symbol error"):
        fc.fetch_history("EURUSD", "H1", need=5)

    with pytest.raises(RuntimeError, match="Invalid timeframe"):
        fc.fetch_history("EURUSD", "BAD", need=5)


def test_fetch_history_rejects_materially_coarser_provider_cadence(monkeypatch):
    monkeypatch.setattr(fc, "TIMEFRAME_MAP", {"H1": 1})
    monkeypatch.setattr(fc, "_ensure_symbol_ready", lambda _symbol: None)
    monkeypatch.setattr(
        fc,
        "get_symbol_info_cached",
        lambda _symbol: SimpleNamespace(visible=True),
    )
    monkeypatch.setattr(fc.mt5, "last_error", lambda: (1, "err"))
    monkeypatch.setattr(
        fc,
        "_mt5_copy_rates_from_pos",
        lambda symbol, tf, start, count: [
            {"time": index * 86400.0, "open": float(index)}
            for index in range(8)
        ],
    )

    with pytest.raises(RuntimeError, match="observed_median_bar_seconds=86400.0"):
        fc.fetch_history("EURUSD", "H1", need=8, drop_last_live=False)


def test_fetch_history_as_of_and_drop_last_live_paths(monkeypatch):
    monkeypatch.setattr(fc, "TIMEFRAME_MAP", {"H1": 1})
    monkeypatch.setattr(fc, "_ensure_symbol_ready", lambda _symbol: None)
    monkeypatch.setattr(fc, "get_symbol_info_cached", lambda _symbol: SimpleNamespace(visible=False))

    symbol_select_calls = []
    monkeypatch.setattr(fc.mt5, "symbol_select", lambda symbol, visible: symbol_select_calls.append((symbol, visible)) or True)
    monkeypatch.setattr(fc.mt5, "last_error", lambda: (1, "err"))

    rates = [
        {"time": 100.0, "open": 1.0},
        {"time": 3700.0, "open": 2.0},
        {"time": 7300.0, "open": 3.0},
        {"time": 10900.0, "open": 4.0},
    ]

    monkeypatch.setattr(fc, "_mt5_copy_rates_from", lambda symbol, tf, to_dt, count: rates)
    monkeypatch.setattr(fc, "_mt5_copy_rates_from_pos", lambda symbol, tf, start, count: rates)
    monkeypatch.setattr(fc, "_parse_start_datetime", lambda _as_of: datetime(2024, 1, 1))
    monkeypatch.setattr(fc, "_utc_epoch_seconds", lambda _dt: 7300.0)

    out = fc.fetch_history("EURUSD", "H1", need=2, as_of="2024-01-01")
    assert out["time"].tolist() == [100.0, 3700.0]
    assert ("EURUSD", False) in symbol_select_calls

    monkeypatch.setattr(fc, "_is_last_bar_forming", lambda rates, timeframe: True)
    out = fc.fetch_history("EURUSD", "H1", need=4, as_of=None, drop_last_live=True)
    assert out["time"].tolist() == [100.0, 3700.0, 7300.0]


def test_fetch_history_as_of_anchors_directly_not_latest_window(monkeypatch):
    monkeypatch.setattr(fc, "TIMEFRAME_MAP", {"H1": 1})
    monkeypatch.setattr(fc, "_ensure_symbol_ready", lambda _symbol: None)
    monkeypatch.setattr(fc, "get_symbol_info_cached", lambda _symbol: SimpleNamespace(visible=True))
    monkeypatch.setattr(fc.mt5, "last_error", lambda: (1, "err"))

    # Would be returned by a latest-window fetch, but should not be used for old as_of.
    newest_rates = [{"time": 9000.0, "open": 9.0}, {"time": 9100.0, "open": 9.1}]
    asof_rates = [{"time": 100.0, "open": 1.0}, {"time": 200.0, "open": 2.0}]

    monkeypatch.setattr(fc, "_mt5_copy_rates_from_pos", lambda symbol, tf, start, count: newest_rates)
    monkeypatch.setattr(fc, "_mt5_copy_rates_from", lambda symbol, tf, to_dt, count: asof_rates)
    monkeypatch.setattr(fc, "_parse_start_datetime", lambda _as_of: datetime(2020, 1, 1))
    monkeypatch.setattr(fc, "_utc_epoch_seconds", lambda _dt: 3800.0)

    out = fc.fetch_history("EURUSD", "H1", need=2, as_of="2020-01-01")
    assert out["time"].tolist() == [100.0, 200.0]


def test_fetch_history_start_end_uses_range_without_lookback_trim(monkeypatch):
    monkeypatch.setattr(fc, "TIMEFRAME_MAP", {"H1": 1})
    monkeypatch.setattr(fc, "_ensure_symbol_ready", lambda _symbol: None)
    monkeypatch.setattr(fc, "get_symbol_info_cached", lambda _symbol: SimpleNamespace(visible=True))
    monkeypatch.setattr(fc.mt5, "last_error", lambda: (1, "err"))

    rates = [
        {"time": 100.0, "open": 1.0},
        {"time": 200.0, "open": 2.0},
        {"time": 300.0, "open": 3.0},
        {"time": 400.0, "open": 4.0},
    ]
    captured = {}

    def fake_range(symbol, tf, from_dt, to_dt):
        captured.update({"symbol": symbol, "tf": tf, "from_dt": from_dt, "to_dt": to_dt})
        return rates

    monkeypatch.setattr(fc, "_mt5_copy_rates_range", fake_range)
    monkeypatch.setattr(
        fc,
        "_parse_start_datetime",
        lambda value: datetime(2024, 1, 1) if value == "2024-01-01" else datetime(2024, 1, 2),
    )
    monkeypatch.setattr(fc, "_utc_epoch_seconds", lambda _dt: 3950.0)

    out = fc.fetch_history(
        "EURUSD",
        "H1",
        need=2,
        start="2024-01-01",
        end="2024-01-02",
    )

    assert captured["symbol"] == "EURUSD"
    assert captured["tf"] == 1
    assert out["time"].tolist() == [100.0, 200.0, 300.0]


def test_fetch_history_end_bound_excludes_bar_not_closed_by_cutoff(monkeypatch):
    monkeypatch.setattr(fc, "TIMEFRAME_MAP", {"H1": 1})
    monkeypatch.setattr(fc, "_ensure_symbol_ready", lambda _symbol: None)
    monkeypatch.setattr(
        fc,
        "get_symbol_info_cached",
        lambda _symbol: SimpleNamespace(visible=True),
    )
    monkeypatch.setattr(fc.mt5, "last_error", lambda: (1, "err"))
    rates = [
        {"time": 100.0, "open": 1.0},
        {"time": 3700.0, "open": 2.0},
        {"time": 7300.0, "open": 3.0},
    ]
    monkeypatch.setattr(
        fc,
        "_mt5_copy_rates_from",
        lambda symbol, tf, to_dt, count: rates,
    )
    monkeypatch.setattr(fc, "_parse_end_datetime", lambda _end: datetime(2024, 1, 1))
    monkeypatch.setattr(fc, "_utc_epoch_seconds", lambda _dt: 7300.0)

    closed = fc.fetch_history("EURUSD", "H1", need=3, end="2024-01-01T02:00:00Z")
    including_live = fc.fetch_history(
        "EURUSD",
        "H1",
        need=3,
        end="2024-01-01T02:00:00Z",
        drop_last_live=False,
    )

    assert closed["time"].tolist() == [100.0, 3700.0]
    assert including_live["time"].tolist() == [100.0, 3700.0, 7300.0]


def test_fetch_history_daily_dates_use_broker_sessions_and_wall_clock(monkeypatch):
    fixed_now = datetime(2026, 8, 13, 14, 0, tzinfo=timezone.utc)

    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed_now.replace(tzinfo=None) if tz is None else fixed_now.astimezone(tz)

    rates = [
        {
            "time": pd.Timestamp(value, tz="UTC").timestamp(),
            "close": close,
        }
        for value, close in (
            ("2026-08-09 21:00", 1.10),
            ("2026-08-10 21:00", 1.11),
            ("2026-08-11 21:00", 1.12),
            ("2026-08-12 21:00", 1.13),
        )
    ]
    captured = {}

    def fake_range(_symbol, _tf, start_dt, end_dt):
        captured.update(start=start_dt, end=end_dt)
        return rates

    monkeypatch.setattr(fc, "datetime", FixedDateTime)
    monkeypatch.setattr(fc, "_ensure_symbol_ready", lambda _symbol: None)
    monkeypatch.setattr(fc, "get_symbol_info_cached", lambda _symbol: SimpleNamespace(visible=True))
    monkeypatch.setattr(fc, "_mt5_copy_rates_range", fake_range)
    monkeypatch.setattr(fc.mt5, "last_error", lambda: (0, "ok"))
    monkeypatch.setattr(
        "mtdata.services.data_service.mt5_config.get_server_tz",
        lambda: __import__("zoneinfo").ZoneInfo("Europe/Nicosia"),
    )

    out = fc.fetch_history(
        "EURUSD",
        "D1",
        need=10,
        start="2026-08-10",
        end="2026-08-13",
    )

    assert captured["start"] == datetime(2026, 8, 9, 21, tzinfo=timezone.utc)
    assert captured["end"] == datetime(
        2026, 8, 13, 20, 59, 59, 999999, tzinfo=timezone.utc
    )
    assert out["close"].tolist() == [1.10, 1.11, 1.12]


def test_fetch_history_preserves_live_native_utc_epochs(monkeypatch):
    monkeypatch.setattr(fc, "TIMEFRAME_MAP", {"H1": 1})
    monkeypatch.setattr(fc, "_ensure_symbol_ready", lambda _symbol: None)
    monkeypatch.setattr(fc, "get_symbol_info_cached", lambda _symbol: SimpleNamespace(visible=True))
    monkeypatch.setattr(fc.mt5, "last_error", lambda: (1, "err"))
    monkeypatch.setattr(
        fc,
        "_mt5_copy_rates_from_pos",
        lambda symbol, tf, start, count: [
            {"time": 100.0, "open": 1.0},
            {"time": 200.0, "open": 2.0},
            {"time": 300.0, "open": 3.0},
        ],
    )
    out = fc.fetch_history("EURUSD", "H1", need=3, as_of=None, drop_last_live=False)
    assert out["time"].tolist() == [100.0, 200.0, 300.0]


def test_fetch_history_handles_invalid_as_of_and_empty_rates(monkeypatch):
    monkeypatch.setattr(fc, "TIMEFRAME_MAP", {"H1": 1})
    monkeypatch.setattr(fc, "_ensure_symbol_ready", lambda _symbol: None)
    monkeypatch.setattr(fc, "get_symbol_info_cached", lambda _symbol: SimpleNamespace(visible=True))
    monkeypatch.setattr(fc.mt5, "last_error", lambda: (500, "no data"))

    monkeypatch.setattr(fc, "_parse_start_datetime", lambda _as_of: None)
    with pytest.raises(RuntimeError, match="Invalid as_of time"):
        fc.fetch_history("EURUSD", "H1", need=2, as_of="bad")

    monkeypatch.setattr(fc, "_parse_start_datetime", lambda _as_of: datetime(2024, 1, 1))
    monkeypatch.setattr(fc, "_mt5_copy_rates_from_pos", lambda symbol, tf, start, count: [])
    with pytest.raises(RuntimeError, match="Failed to get rates"):
        fc.fetch_history("EURUSD", "H1", need=2)
