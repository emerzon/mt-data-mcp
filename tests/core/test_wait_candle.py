from datetime import datetime, timezone

import pytest

from mtdata.core.data.requests import WaitCandleRequest
from mtdata.core.data.use_cases import run_wait_candle
from mtdata.core.trading import time
from mtdata.core.trading.time import (
    _next_candle_close_server_time,
    _next_candle_wait_payload,
    _sleep_until_next_candle,
)


@pytest.fixture()
def utc_server_clock(monkeypatch):
    monkeypatch.setattr(time.mt5_config, "get_server_tz", lambda: None)
    monkeypatch.setattr(time.mt5_config, "get_time_offset_seconds", lambda: 0)
    monkeypatch.setattr(time.mt5_config, "server_tz_name", None)


def test_wait_candle_request_rejects_negative_buffer() -> None:
    with pytest.raises(ValueError, match="Input should be greater than or equal to 0"):
        WaitCandleRequest(timeframe="M5", buffer_seconds=-0.1)


def test_wait_candle_request_defaults_max_wait_to_one_hour() -> None:
    request = WaitCandleRequest(timeframe="M5")

    assert request.max_wait_seconds == 3600.0


def test_wait_candle_request_rejects_negative_max_wait() -> None:
    with pytest.raises(ValueError, match="Input should be greater than or equal to 0"):
        WaitCandleRequest(timeframe="M5", max_wait_seconds=-0.1)


def test_next_candle_close_server_time_rounds_intraday_frame(utc_server_clock) -> None:
    now_utc = datetime(2026, 3, 13, 10, 2, 10, tzinfo=timezone.utc)

    result = _next_candle_close_server_time("M5", now_utc=now_utc)

    assert result == datetime(2026, 3, 13, 10, 5, 0)


def test_next_candle_close_server_time_handles_weekly_boundary(utc_server_clock) -> None:
    now_utc = datetime(2026, 3, 13, 10, 2, 10, tzinfo=timezone.utc)

    result = _next_candle_close_server_time("W1", now_utc=now_utc)

    assert result == datetime(2026, 3, 16, 0, 0, 0)


def test_next_candle_close_server_time_uses_shared_unsupported_timeframe_error(
    utc_server_clock,
    monkeypatch,
) -> None:
    monkeypatch.setitem(time.TIMEFRAME_SECONDS, "M5", 0)
    monkeypatch.setattr(
        time,
        "unsupported_timeframe_seconds_error",
        lambda timeframe: f"custom unsupported {timeframe}",
    )

    with pytest.raises(ValueError, match="custom unsupported M5"):
        _next_candle_close_server_time("M5", now_utc=datetime(2026, 3, 13, 10, 2, 10, tzinfo=timezone.utc))


def test_sleep_until_next_candle_returns_expected_wait(utc_server_clock) -> None:
    slept = []

    payload = _sleep_until_next_candle(
        "M5",
        buffer_seconds=1.0,
        sleep_impl=lambda seconds: slept.append(seconds),
        now_utc=datetime(2026, 3, 13, 10, 2, 10, tzinfo=timezone.utc),
    )

    assert slept == [171.0]
    assert payload["sleep_seconds"] == 171.0
    assert payload["slept"] is True
    assert payload["status"] == "completed"
    assert payload["next_candle_close_utc"] == "2026-03-13T10:05:00+00:00"


def test_next_candle_wait_payload_handles_pytz_dst_gap(monkeypatch) -> None:
    pytz = pytest.importorskip("pytz")

    monkeypatch.setattr(time.mt5_config, "get_server_tz", lambda: pytz.timezone("Europe/Nicosia"))
    monkeypatch.setattr(time.mt5_config, "get_time_offset_seconds", lambda: 7200)
    monkeypatch.setattr(time.mt5_config, "server_tz_name", "Europe/Nicosia")

    payload = _next_candle_wait_payload(
        "M15",
        buffer_seconds=1.0,
        now_utc=datetime(2026, 3, 29, 0, 54, 0, tzinfo=timezone.utc),
    )

    assert payload["next_candle_close_server"] == "2026-03-29T03:00:00+02:00"
    assert datetime.fromisoformat(payload["next_candle_close_server"]).astimezone(
        timezone.utc
    ) == datetime.fromisoformat(payload["next_candle_close_utc"])
    assert payload["next_candle_close_utc"] == "2026-03-29T01:00:00+00:00"
    assert payload["sleep_seconds"] == 361.0


def test_run_wait_candle_returns_success(monkeypatch) -> None:
    monkeypatch.setattr(
        "mtdata.core.data.use_cases._sleep_until_next_candle",
        lambda timeframe, buffer_seconds, sleep_impl: {
            "timeframe": timeframe,
            "buffer_seconds": buffer_seconds,
            "sleep_seconds": 12.5,
            "status": "completed",
            "slept": True,
            "slept_seconds": 12.5,
            "remaining_seconds": 0.0,
            "started_at_utc": "2026-03-13T10:00:00+00:00",
            "next_candle_close_utc": "2026-03-13T10:05:00+00:00",
            "next_candle_close_server": "2026-03-13T10:05:00",
            "server_timezone": "UTC",
        },
    )
    monkeypatch.setattr(
        "mtdata.core.data.use_cases._next_candle_wait_payload",
        lambda timeframe, buffer_seconds: {
            "timeframe": timeframe,
            "buffer_seconds": buffer_seconds,
            "sleep_seconds": 12.5,
            "started_at_utc": "2026-03-13T10:00:00+00:00",
            "next_candle_close_utc": "2026-03-13T10:05:00+00:00",
            "next_candle_close_server": "2026-03-13T10:05:00",
            "server_timezone": "UTC",
        },
    )

    result = run_wait_candle(WaitCandleRequest(timeframe="M5", buffer_seconds=0.5))

    assert result["success"] is True
    assert result["sleep_seconds"] == 12.5
    assert result["status"] == "completed"


def test_run_wait_candle_defers_when_wait_exceeds_cap(monkeypatch) -> None:
    monkeypatch.setattr(
        "mtdata.core.data.use_cases._next_candle_wait_payload",
        lambda timeframe, buffer_seconds: {
            "timeframe": timeframe,
            "buffer_seconds": buffer_seconds,
            "sleep_seconds": 171.0,
            "started_at_utc": "2026-03-13T10:02:10+00:00",
            "next_candle_close_utc": "2026-03-13T10:05:00+00:00",
            "next_candle_close_server": "2026-03-13T10:05:00",
            "server_timezone": "UTC",
        },
    )

    result = run_wait_candle(WaitCandleRequest(timeframe="M5", max_wait_seconds=25.0))

    assert result["success"] is False
    assert result["slept"] is False
    assert result["status"] == "wait_budget_exceeded"
    assert result["error_code"] == "wait_budget_exceeded"
    assert result["not_waited"] is True
    assert result["remaining_seconds"] == 171.0
