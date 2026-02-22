from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from mtdata.forecast import common as fc


def test_extract_forecast_values_handles_standard_alt_and_padding():
    yf_standard = pd.DataFrame({"y": [1.0, 2.0, 3.0]})
    out = fc._extract_forecast_values(yf_standard, fh=2, method_name="m")
    assert out.tolist() == [1.0, 2.0]

    yf_alt = pd.DataFrame({"unique_id": ["ts"], "ds": [0], "pred": [9.0]})
    out = fc._extract_forecast_values(yf_alt, fh=3, method_name="m")
    assert out.tolist() == [9.0, 9.0, 9.0]

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


def test_timeframe_helpers_cover_key_branches():
    assert fc.default_seasonality("H1") == 24
    assert fc.default_seasonality("D1") == 5
    assert fc.default_seasonality("W1") == 52
    assert fc.default_seasonality("MN1") == 12
    assert fc.default_seasonality("NOPE") == 0

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


def test_fetch_history_as_of_and_drop_last_live_paths(monkeypatch):
    monkeypatch.setattr(fc, "TIMEFRAME_MAP", {"H1": 1})
    monkeypatch.setattr(fc, "_ensure_symbol_ready", lambda _symbol: None)
    monkeypatch.setattr(fc, "get_symbol_info_cached", lambda _symbol: SimpleNamespace(visible=False))

    symbol_select_calls = []
    monkeypatch.setattr(fc.mt5, "symbol_select", lambda symbol, visible: symbol_select_calls.append((symbol, visible)) or True)
    monkeypatch.setattr(fc.mt5, "last_error", lambda: (1, "err"))

    rates = [
        {"time": 100.0, "open": 1.0},
        {"time": 200.0, "open": 2.0},
        {"time": 300.0, "open": 3.0},
        {"time": 400.0, "open": 4.0},
    ]

    monkeypatch.setattr(fc, "_mt5_copy_rates_from_pos", lambda symbol, tf, start, count: rates)
    monkeypatch.setattr(fc, "_parse_start_datetime", lambda _as_of: datetime(2024, 1, 1))
    monkeypatch.setattr(fc, "_utc_epoch_seconds", lambda _dt: 300.0)

    out = fc.fetch_history("EURUSD", "H1", need=2, as_of="2024-01-01")
    assert out["time"].tolist() == [200.0, 300.0]
    assert ("EURUSD", False) in symbol_select_calls

    out = fc.fetch_history("EURUSD", "H1", need=4, as_of=None, drop_last_live=True)
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
