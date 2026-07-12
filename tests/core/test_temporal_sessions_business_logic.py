from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import numpy as np

import mtdata.core.temporal as temporal_mod

_raw_temporal_analyze = temporal_mod.temporal_analyze.__wrapped__
_P = "mtdata.core.temporal."


def _make_rates_from_epochs(times: list[int]) -> np.ndarray:
    dtype = np.dtype([
        ("time", "<i8"),
        ("open", "<f8"),
        ("high", "<f8"),
        ("low", "<f8"),
        ("close", "<f8"),
        ("tick_volume", "<i8"),
        ("spread", "<i4"),
        ("real_volume", "<i8"),
    ])
    rates = np.empty(len(times), dtype=dtype)
    close = np.linspace(1.1, 1.1 + 0.0005 * (len(times) - 1), len(times), dtype=float)
    rates["time"] = np.asarray(times, dtype=np.int64)
    rates["open"] = close - 0.0001
    rates["high"] = close + 0.0002
    rates["low"] = close - 0.0002
    rates["close"] = close
    rates["tick_volume"] = np.arange(100, 100 + len(times), dtype=np.int64)
    rates["spread"] = np.full(len(times), 10, dtype=np.int32)
    rates["real_volume"] = np.zeros(len(times), dtype=np.int64)
    return rates


@contextmanager
def _mock_guard_ok():
    yield (None, MagicMock())


def _guard_stub(*_args, **_kwargs):
    return _mock_guard_ok()


def _info_stub(*_args, **_kwargs):
    return MagicMock()


def test_market_session_label_tracks_dst_boundaries() -> None:
    assert temporal_mod._market_session_label(
        datetime(2026, 1, 15, 7, 30, tzinfo=timezone.utc)
    ) == "asia"
    assert temporal_mod._market_session_label(
        datetime(2026, 7, 15, 7, 30, tzinfo=timezone.utc)
    ) == "london"
    assert temporal_mod._market_session_label(
        datetime(2026, 1, 15, 15, 30, tzinfo=timezone.utc)
    ) == "london_ny_overlap"
    assert temporal_mod._market_session_label(
        datetime(2026, 7, 15, 15, 30, tzinfo=timezone.utc)
    ) == "ny"


def test_temporal_analyze_session_groups_use_analysis_timezone_clock() -> None:
    rates = _make_rates_from_epochs(
        [
            int(datetime(2026, 7, 15, 14, 30, tzinfo=timezone.utc).timestamp()),
            int(datetime(2026, 7, 15, 15, 0, tzinfo=timezone.utc).timestamp()),
            int(datetime(2026, 7, 15, 15, 30, tzinfo=timezone.utc).timestamp()),
        ]
    )

    with patch(_P + "_fetch_rates", return_value=(rates, None)), patch(
        _P + "_symbol_ready_guard",
        new=_guard_stub,
    ), patch(
        _P + "ensure_mt5_connection_or_raise",
        new=lambda: None,
    ), patch(
        _P + "get_symbol_info_cached",
        new=_info_stub,
    ), patch(
        _P + "_resolve_client_tz",
        return_value=ZoneInfo("Europe/London"),
    ):
        result = _raw_temporal_analyze(
            symbol="EURUSD",
            timeframe="M30",
            lookback=100,
            group_by="session",
            time_range="16:00-17:00",
            detail="full",
        )

    assert result["success"] is True
    assert result["timezone"] == "Europe/London"
    assert result["bars"] == 2
    assert result["session_definition"]["clock"] == "Europe/London"
    assert [group["group"] for group in result["groups"]] == ["ny"]
    assert [group["group_label"] for group in result["groups"]] == ["ny"]


def test_temporal_analyze_compact_keeps_session_clock_definition() -> None:
    rates = _make_rates_from_epochs(
        [
            int(datetime(2026, 7, 15, 14, 30, tzinfo=timezone.utc).timestamp()),
            int(datetime(2026, 7, 15, 15, 0, tzinfo=timezone.utc).timestamp()),
            int(datetime(2026, 7, 15, 15, 30, tzinfo=timezone.utc).timestamp()),
        ]
    )
    with patch(_P + "_fetch_rates", return_value=(rates, None)), patch(
        _P + "_symbol_ready_guard", new=_guard_stub
    ), patch(_P + "ensure_mt5_connection_or_raise", new=lambda: None), patch(
        _P + "get_symbol_info_cached", new=_info_stub
    ):
        result = _raw_temporal_analyze(
            symbol="EURUSD",
            timeframe="M30",
            lookback=100,
            group_by="session",
            detail="compact",
        )

    assert result["session_definition"]["basis"] == "dst_aware_market_sessions"
    assert result["session_definition"]["clock"] == result["timezone"]
