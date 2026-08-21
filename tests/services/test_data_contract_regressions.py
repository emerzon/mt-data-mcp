from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from mtdata.core.data.requests import DataFetchCandlesRequest, DataFetchTicksRequest
from mtdata.core.data.use_cases import (
    _normalize_range_limit_contract,
    run_data_fetch_candles,
    run_data_fetch_ticks,
)
from mtdata.utils.utils import _calendar_period_bounds


def _gateway() -> SimpleNamespace:
    return SimpleNamespace(ensure_connection=lambda: None)


def test_implicit_range_limit_does_not_become_excluded_candles() -> None:
    payload = {
        "candles_excluded": 99_996,
        "query_applied": {"mode": "range", "limit": 100_000},
        "candle_counts": {
            "requested": 100_000,
            "excluded": {"window_or_source_shortfall": 99_996, "total": 99_996},
        },
    }

    _normalize_range_limit_contract(
        payload,
        effective_limit=100_000,
        limit_explicit=False,
    )

    assert payload["candles_excluded"] == 0
    assert payload["candle_counts"]["excluded"]["total"] == 0


@pytest.mark.parametrize(
    ("value", "expected_start", "expected_end", "kind"),
    [
        ("this month", datetime(2026, 8, 1), datetime(2026, 8, 31, 23, 59, 59, 999999), "month"),
        ("last month", datetime(2026, 7, 1), datetime(2026, 7, 31, 23, 59, 59, 999999), "month"),
        ("this year", datetime(2026, 1, 1), datetime(2026, 12, 31, 23, 59, 59, 999999), "year"),
    ],
)
def test_calendar_month_and_year_phrases_expand_to_periods(
    value: str,
    expected_start: datetime,
    expected_end: datetime,
    kind: str,
) -> None:
    assert _calendar_period_bounds(
        value,
        now=datetime(2026, 8, 19, 12, tzinfo=timezone.utc),
    ) == (expected_start, expected_end, kind)


@pytest.mark.parametrize(
    ("timestamp_format", "time_value", "timezone_name", "expected", "mode"),
    [
        ("epoch", 1_787_098_197.256, "America/New_York", "epoch_seconds", "utc"),
        ("iso", "2026-08-19T00:00:00Z", "UTC", "iso_utc", "utc"),
        (
            "iso",
            "2026-01-15T09:30:00-05:00",
            "America/New_York",
            "iso_offset",
            "client_timezone",
        ),
        (
            "iso",
            "2026-07-15T09:30:00-04:00",
            "America/New_York",
            "iso_offset",
            "client_timezone",
        ),
    ],
)
def test_tick_results_publish_timestamp_format(
    timestamp_format: str,
    time_value: float | str,
    timezone_name: str,
    expected: str,
    mode: str,
) -> None:
    request = DataFetchTicksRequest(
        symbol="EURUSD",
        limit=1,
        detail="full",
        timestamp_format=timestamp_format,
    )
    result = run_data_fetch_ticks(
        request,
        gateway=_gateway(),
        fetch_ticks_impl=lambda **_: {
            "success": True,
            "count": 1,
            "tick_count": 1,
            "timezone": timezone_name,
            "data": [{"time": time_value, "bid": 1.1, "ask": 1.2}],
        },
    )

    assert result["timestamp_format"] == expected
    assert result["timestamp_mode"] == mode
    assert result["public_timestamp_mode"] == mode
    assert result["timestamp_timezone"] == (
        "UTC" if mode == "utc" else "America/New_York"
    )
    if timestamp_format == "epoch":
        assert result["timezone"] == "UTC"


@pytest.mark.parametrize(
    "time_value",
    ["2026-01-15T09:30:00-05:00", "2026-07-15T09:30:00-04:00"],
)
def test_candle_metadata_labels_client_offset_representation(time_value: str) -> None:
    result = run_data_fetch_candles(
        DataFetchCandlesRequest(symbol="AAPL.NAS", limit=1, detail="full"),
        gateway=_gateway(),
        fetch_candles_impl=lambda **_: {
            "success": True,
            "candles": 1,
            "time_basis": "utc",
            "timestamp_mode": "server_clock",
            "timezone": "America/New_York",
            "data": [{"time": time_value, "close": 200.0}],
        },
    )

    assert result["time_basis"] == "utc"
    assert result["timestamp_format"] == "iso_offset"
    assert result["timestamp_mode"] == "client_timezone"
    assert result["public_timestamp_mode"] == "client_timezone"
    assert result["timestamp_timezone"] == "America/New_York"
    assert result["raw_timestamp_mode"] == "server_clock"


def test_compact_candle_metadata_keeps_only_client_timezone_distinction() -> None:
    result = run_data_fetch_candles(
        DataFetchCandlesRequest(symbol="AAPL.NAS", limit=1),
        gateway=_gateway(),
        fetch_candles_impl=lambda **_: {
            "success": True,
            "candles": 1,
            "time_basis": "utc",
            "timestamp_mode": "server_clock",
            "timezone": "America/New_York",
            "data": [{"time": "2026-07-15T09:30:00-04:00", "close": 200.0}],
        },
    )

    assert result["timestamp_format"] == "iso_offset"
    assert result["timezone"] == "America/New_York"
    assert result["time_basis"] == "utc"
    assert "timestamp_mode" not in result
    assert "public_timestamp_mode" not in result
    assert "timestamp_timezone" not in result


def test_compact_utc_candle_metadata_has_one_representation_contract() -> None:
    result = run_data_fetch_candles(
        DataFetchCandlesRequest(symbol="EURUSD", limit=1),
        gateway=_gateway(),
        fetch_candles_impl=lambda **_: {
            "success": True,
            "candles": 1,
            "time_basis": "utc",
            "timestamp_mode": "native_utc",
            "timezone": "UTC",
            "data": [{"time": "2026-08-19T00:00:00Z", "close": 1.1}],
        },
    )

    timestamp_keys = {
        "timezone",
        "time_basis",
        "timestamp_format",
        "timestamp_mode",
        "public_timestamp_mode",
        "timestamp_timezone",
    }
    assert {
        key: result[key] for key in timestamp_keys if key in result
    } == {"timezone": "UTC", "timestamp_format": "iso_utc"}


def test_compact_empty_tick_metadata_has_one_representation_contract() -> None:
    result = run_data_fetch_ticks(
        DataFetchTicksRequest(symbol="EURUSD", limit=1),
        gateway=_gateway(),
        fetch_ticks_impl=lambda **_: {
            "success": True,
            "count": 0,
            "timezone": "UTC",
            "data": [],
            "empty": True,
            "empty_reason": "no_ticks_in_range",
        },
    )

    timestamp_keys = {
        "timezone",
        "time_basis",
        "timestamp_format",
        "timestamp_mode",
        "public_timestamp_mode",
        "timestamp_timezone",
    }
    assert {
        key: result[key] for key in timestamp_keys if key in result
    } == {"timezone": "UTC", "timestamp_format": "iso_utc"}


def test_compact_candles_omit_operator_diagnostics() -> None:
    request = DataFetchCandlesRequest(symbol="EURUSD", limit=2)
    result = run_data_fetch_candles(
        request,
        gateway=_gateway(),
        fetch_candles_impl=lambda **_: {
            "success": True,
            "candles": 2,
            "candles_requested": 2,
            "volume_type": "tick_count",
            "volume_unit": "broker_tick_count",
            "bar_spacing": {"status": "ok"},
            "source_bar_spacing": {"status": "ok"},
            "time_normalization": {"source": "mt5"},
            "timestamp_mode": "server_shifted_to_utc",
            "time_basis": "utc",
            "data": [{"time": 1.0, "close": 1.1, "tick_volume": 10}],
        },
    )

    assert result["volume_semantics"] == "tick_volume_is_broker_tick_count_not_lots"
    for key in (
        "bar_spacing",
        "source_bar_spacing",
        "time_normalization",
        "tick_volume_event_basis",
        "tick_volume_tape_equivalent",
        "tick_volume_comparison_note",
    ):
        assert key not in result
