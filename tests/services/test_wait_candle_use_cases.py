from mtdata.core.data_requests import WaitCandleRequest
from mtdata.core.data_use_cases import run_wait_candle


def test_run_wait_candle_returns_success(monkeypatch) -> None:
    monkeypatch.setattr(
        "mtdata.core.data_use_cases._sleep_until_next_candle",
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
        "mtdata.core.data_use_cases._next_candle_wait_payload",
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
        "mtdata.core.data_use_cases._next_candle_wait_payload",
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

    assert result["success"] is True
    assert result["slept"] is False
    assert result["status"] == "deferred_timeout_risk"
    assert result["remaining_seconds"] == 171.0
