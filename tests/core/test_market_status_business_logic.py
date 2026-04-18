from __future__ import annotations

from datetime import date, datetime, timezone
from inspect import signature

import mtdata.core.market_status as market_status_mod


def _unwrap(fn):
    while hasattr(fn, "__wrapped__"):
        fn = fn.__wrapped__
    return fn


def test_market_status_tool_supports_detail_contract() -> None:
    raw = _unwrap(market_status_mod.market_status)
    params = list(signature(raw).parameters.values())

    assert [param.name for param in params] == ["region", "timezone_display", "detail"]
    assert params[0].default == "all"
    assert params[1].default == "local"
    assert params[2].default == "compact"


def test_is_holiday_loads_the_requested_year(monkeypatch) -> None:
    calls: list[tuple[str, tuple[int, ...]]] = []

    def fake_country_holidays(country: str, years):
        year_tuple = tuple(int(value) for value in years)
        calls.append((country, year_tuple))
        year = year_tuple[0]
        return {date(year, 1, 1): f"{country}-{year}"}

    market_status_mod._get_holidays.cache_clear()
    monkeypatch.setattr(market_status_mod.holidays, "country_holidays", fake_country_holidays)

    is_holiday_result, holiday_name = market_status_mod._is_holiday(
        "US",
        datetime(2031, 1, 1, tzinfo=timezone.utc),
    )

    assert is_holiday_result is True
    assert holiday_name == "US-2031"
    assert calls == [("US", (2031,))]


def test_upcoming_holidays_crosses_into_the_next_year(monkeypatch) -> None:
    calls: list[tuple[str, tuple[int, ...]]] = []

    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2030, 12, 30, 12, 0, tzinfo=tz or timezone.utc)

    def fake_country_holidays(country: str, years):
        year_tuple = tuple(int(value) for value in years)
        calls.append((country, year_tuple))
        year = year_tuple[0]
        if year == 2031:
            return {date(2031, 1, 1): "New Year's Day"}
        return {}

    market_status_mod._get_holidays.cache_clear()
    monkeypatch.setattr(market_status_mod, "datetime", FixedDateTime)
    monkeypatch.setattr(market_status_mod.holidays, "country_holidays", fake_country_holidays)

    upcoming = market_status_mod._get_upcoming_holidays(["NYSE"], days_ahead=3)

    assert upcoming == [
        {
            "date": "2031-01-01",
            "holiday": "New Year's Day",
            "country": "US",
            "markets_affected": ["NYSE"],
            "impact": "closed",
            "early_close_time": None,
            "days_away": 2,
        }
    ]
    assert calls == [("US", (2030,)), ("US", (2031,))]


def test_normalize_market_status_output_compact_summarizes_holidays() -> None:
    payload = {
        "success": True,
        "upcoming_holidays": [
            {
                "date": "2031-01-01",
                "holiday": "New Year's Day",
                "country": "US",
                "markets_affected": ["NYSE", "NASDAQ"],
                "impact": "closed",
                "early_close_time": None,
                "days_away": 2,
            },
            {
                "date": "2031-01-02",
                "holiday": "Day after New Year's Day",
                "country": "US",
                "markets_affected": ["NYSE"],
                "impact": "early_close",
                "early_close_time": "13:00",
                "days_away": 3,
            },
        ],
    }

    compact = market_status_mod.normalize_market_status_output(payload, detail="compact")
    full = market_status_mod.normalize_market_status_output(payload, detail="full")

    assert "upcoming_holidays" not in compact
    assert compact["upcoming_holidays_count"] == 2
    assert compact["upcoming_holidays_summary"] == [
        {
            "date": "2031-01-01",
            "holiday": "New Year's Day",
            "impact": "closed",
            "days_away": 2,
            "markets_affected": ["NYSE", "NASDAQ"],
        },
        {
            "date": "2031-01-02",
            "holiday": "Day after New Year's Day",
            "impact": "early_close",
            "days_away": 3,
            "markets_affected": ["NYSE"],
            "early_close_time": "13:00",
        },
    ]
    assert compact["show_all_hint"] == "Use --detail full / --verbose for the upcoming_holidays list."

    assert full["upcoming_holidays"] == payload["upcoming_holidays"]
