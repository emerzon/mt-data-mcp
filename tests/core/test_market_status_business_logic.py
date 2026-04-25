from __future__ import annotations

from datetime import date, datetime, timezone
from inspect import signature
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import mtdata.core.market_status as market_status_mod


def _unwrap(fn):
    while hasattr(fn, "__wrapped__"):
        fn = fn.__wrapped__
    return fn


def test_market_status_tool_supports_detail_contract() -> None:
    raw = _unwrap(market_status_mod.market_status)
    params = list(signature(raw).parameters.values())

    assert [param.name for param in params] == ["symbol", "region", "timezone_display", "detail"]
    assert params[0].default is None
    assert params[1].default == "all"
    assert params[2].default == "local"
    assert params[3].default == "compact"


def test_market_status_timezone_display_utc_converts_market_times(monkeypatch) -> None:
    raw = _unwrap(market_status_mod.market_status)
    fixed_now = datetime(2024, 1, 2, 10, 0, tzinfo=ZoneInfo("America/New_York"))

    monkeypatch.setattr(market_status_mod, "_get_local_time", lambda _tz_name: fixed_now)
    monkeypatch.setattr(market_status_mod, "_get_upcoming_holidays", lambda _markets: [])

    result = raw(region="us", timezone_display="utc", detail="full")

    assert result["success"] is True
    assert {market["symbol"] for market in result["markets"]} == {"NYSE", "NASDAQ"}
    for market in result["markets"]:
        assert market["local_time"] == "15:00"
        assert market["next_close"] == "2024-01-02T21:00:00+00:00"


def test_market_status_uses_utc_weekend_for_closed_reason(monkeypatch) -> None:
    raw = _unwrap(market_status_mod.market_status)
    fixed_utc = datetime(2026, 4, 25, 3, 18, tzinfo=timezone.utc)

    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            if tz is None:
                return fixed_utc.replace(tzinfo=None)
            return fixed_utc.astimezone(tz)

    monkeypatch.setattr(market_status_mod, "datetime", FixedDateTime)
    monkeypatch.setattr(
        market_status_mod,
        "_get_local_time",
        lambda tz_name: fixed_utc.astimezone(ZoneInfo(tz_name)),
    )
    monkeypatch.setattr(market_status_mod, "_get_upcoming_holidays", lambda _markets: [])

    result = raw(region="all", detail="full")

    assert result["success"] is True
    assert result["global_status"] == "weekend"
    assert result["closed_reason_counts"] == {"weekend": 9}
    reasons_by_symbol = {
        market["symbol"]: market.get("reason") for market in result["markets"]
    }
    assert reasons_by_symbol["NYSE"] == "weekend"
    assert reasons_by_symbol["NASDAQ"] == "weekend"


def test_market_status_rejects_invalid_timezone_display() -> None:
    raw = _unwrap(market_status_mod.market_status)

    result = raw(timezone_display="broker")

    assert result == {"error": "Invalid timezone_display. Use 'local', 'utc', or 'auto'."}


def test_market_status_symbol_mode_reports_heuristic_status(monkeypatch) -> None:
    raw = _unwrap(market_status_mod.market_status)
    now_epoch = datetime.now(timezone.utc).timestamp()

    class Gateway:
        SYMBOL_TRADE_MODE_FULL = 4
        SYMBOL_TRADE_MODE_DISABLED = 0
        SYMBOL_TRADE_MODE_CLOSEONLY = 3
        SYMBOL_TRADE_MODE_LONGONLY = 1
        SYMBOL_TRADE_MODE_SHORTONLY = 2

        def ensure_connection(self) -> None:
            return None

        def symbol_info(self, symbol: str):
            assert symbol == "EURUSD"
            return SimpleNamespace(
                name="EURUSD",
                description="Euro vs US Dollar",
                visible=True,
                trade_mode=4,
            )

        def symbol_info_tick(self, symbol: str):
            assert symbol == "EURUSD"
            return SimpleNamespace(time=now_epoch, bid=1.1, ask=1.2)

    monkeypatch.setattr(market_status_mod, "get_mt5_gateway", lambda **kwargs: Gateway())

    result = raw(symbol="eurusd")

    assert result["mode"] == "symbol"
    assert result["symbol"] == "EURUSD"
    assert result["status"] == "probably_open"
    assert result["status_source"] == "trade_mode_and_tick_freshness"
    assert result["status_confidence"] == "heuristic"
    assert result["can_open_new_positions"] is True
    assert result["tick_freshness"] == "fresh"
    assert result["tick_available"] is True


def test_market_status_symbol_mode_full_includes_diagnostics(monkeypatch) -> None:
    raw = _unwrap(market_status_mod.market_status)

    class Gateway:
        SYMBOL_TRADE_MODE_FULL = 4
        SYMBOL_TRADE_MODE_DISABLED = 0
        SYMBOL_TRADE_MODE_CLOSEONLY = 3
        SYMBOL_TRADE_MODE_LONGONLY = 1
        SYMBOL_TRADE_MODE_SHORTONLY = 2

        def ensure_connection(self) -> None:
            return None

        def symbol_info(self, symbol: str):
            return SimpleNamespace(name=symbol, visible=False, trade_mode=0)

        def symbol_info_tick(self, symbol: str):
            return None

    monkeypatch.setattr(market_status_mod, "get_mt5_gateway", lambda **kwargs: Gateway())

    result = raw(symbol="BTCUSD", detail="full")

    assert result["status"] == "disabled"
    assert result["can_open_new_positions"] is False
    assert result["trade_mode"] == 0
    assert result["symbol_info"]["name"] == "BTCUSD"
    assert result["tick"]["tick_available"] is False


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
    assert "show_all_hint" not in compact

    assert full["upcoming_holidays"] == payload["upcoming_holidays"]
