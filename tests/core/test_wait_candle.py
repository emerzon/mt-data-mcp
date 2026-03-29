from datetime import datetime, timezone

import pytest

from mtdata.core.data_requests import WaitCandleRequest
from mtdata.core import trading_time
from mtdata.core.trading_time import _next_candle_close_server_time, _next_candle_wait_payload, _sleep_until_next_candle


@pytest.fixture()
def utc_server_clock(monkeypatch):
    monkeypatch.setattr(trading_time.mt5_config, "get_server_tz", lambda: None)
    monkeypatch.setattr(trading_time.mt5_config, "get_time_offset_seconds", lambda: 0)
    monkeypatch.setattr(trading_time.mt5_config, "server_tz_name", None)


def test_wait_candle_request_rejects_negative_buffer() -> None:
    with pytest.raises(ValueError, match="buffer_seconds must be greater than or equal to 0"):
        WaitCandleRequest(timeframe="M5", buffer_seconds=-0.1)


def test_wait_candle_request_defaults_max_wait_to_one_hour() -> None:
    request = WaitCandleRequest(timeframe="M5")

    assert request.max_wait_seconds == 3600.0


def test_wait_candle_request_rejects_negative_max_wait() -> None:
    with pytest.raises(ValueError, match="max_wait_seconds must be greater than or equal to 0"):
        WaitCandleRequest(timeframe="M5", max_wait_seconds=-0.1)


def test_next_candle_close_server_time_rounds_intraday_frame(utc_server_clock) -> None:
    now_utc = datetime(2026, 3, 13, 10, 2, 10, tzinfo=timezone.utc)

    result = _next_candle_close_server_time("M5", now_utc=now_utc)

    assert result == datetime(2026, 3, 13, 10, 5, 0)


def test_next_candle_close_server_time_handles_weekly_boundary(utc_server_clock) -> None:
    now_utc = datetime(2026, 3, 13, 10, 2, 10, tzinfo=timezone.utc)

    result = _next_candle_close_server_time("W1", now_utc=now_utc)

    assert result == datetime(2026, 3, 16, 0, 0, 0)


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

    monkeypatch.setattr(trading_time.mt5_config, "get_server_tz", lambda: pytz.timezone("Europe/Nicosia"))
    monkeypatch.setattr(trading_time.mt5_config, "get_time_offset_seconds", lambda: 7200)
    monkeypatch.setattr(trading_time.mt5_config, "server_tz_name", "Europe/Nicosia")

    payload = _next_candle_wait_payload(
        "M15",
        buffer_seconds=1.0,
        now_utc=datetime(2026, 3, 29, 0, 54, 0, tzinfo=timezone.utc),
    )

    assert payload["next_candle_close_server"] == "2026-03-29T03:00:00"
    assert payload["next_candle_close_utc"] == "2026-03-29T01:00:00+00:00"
    assert payload["sleep_seconds"] == 361.0
