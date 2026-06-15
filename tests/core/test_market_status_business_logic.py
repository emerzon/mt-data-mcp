from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
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

    assert [param.name for param in params] == ["symbol", "region", "timezone_display", "detail", "extras"]
    assert params[0].default is None
    assert params[1].default == "all"
    assert params[2].default == "auto"
    assert params[3].default == "compact"
    assert params[4].default is None


def test_market_status_timezone_display_utc_converts_market_times(monkeypatch) -> None:
    raw = _unwrap(market_status_mod.market_status)
    fixed_now = datetime(2024, 1, 2, 10, 0, tzinfo=ZoneInfo("America/New_York"))

    monkeypatch.setattr(market_status_mod, "_get_local_time", lambda _tz_name: fixed_now)
    monkeypatch.setattr(market_status_mod, "_get_upcoming_holidays", lambda _markets: [])

    result = raw(region="us", timezone_display="utc", detail="full")

    assert result["success"] is True
    assert result["mode"] == "global"
    assert result["timezone"] == "UTC"
    assert {market["symbol"] for market in result["markets"]} == {"NYSE", "NASDAQ"}
    for market in result["markets"]:
        assert market["local_time"] == "2024-01-02T10:00:00-05:00"
        assert market["display_time"] == "2024-01-02T15:00:00Z"
        assert market["next_close"] == "2024-01-02T21:00:00Z"


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
    assert result["mode"] == "global"
    assert result["data_fetched_at"] == "2026-04-25T03:18:00Z"
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

    assert result == {
        "error": "Invalid timezone_display. Use 'local', 'utc', 'server', or 'auto'."
    }


def test_market_status_symbol_mode_reports_heuristic_status(monkeypatch) -> None:
    raw = _unwrap(market_status_mod.market_status)
    fixed_now = datetime(2024, 1, 2, 12, 0, tzinfo=timezone.utc)
    now_epoch = fixed_now.timestamp()

    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            if tz is None:
                return fixed_now.replace(tzinfo=None)
            return fixed_now.astimezone(tz)

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

    monkeypatch.setattr(market_status_mod, "datetime", FixedDateTime)
    monkeypatch.setattr(market_status_mod, "create_mt5_gateway", lambda **kwargs: Gateway())

    result = raw(symbol="eurusd")

    assert result["mode"] == "symbol"
    assert result["symbol"] == "EURUSD"
    assert result["timezone"] == "UTC"
    assert result["status"] == "probably_open"
    assert result["status_source"] == "trade_mode_and_tick_freshness"
    assert result["status_confidence"] == "heuristic"
    assert result["can_open_new_positions"] is True
    assert result["trade_mode_allows_opening"] is True
    assert result["tick_freshness"] == "fresh"
    assert result["tick_available"] is True
    assert result["data_fetched_at"] == "2024-01-02T12:00:00Z"
    assert result["last_tick_time"] == "2024-01-02T12:00:00Z"
    assert result["is_tradable"] is True
    assert result["market_clock"] == "2024-01-02T12:00:00Z"
    assert result["market_clock_timezone"] == "UTC"
    assert result["authoritative_clock"] in {"server", "utc"}
    assert "timezone_context" not in result


def test_market_status_symbol_timezone_context_labels_server_clock(monkeypatch) -> None:
    monkeypatch.setattr(
        market_status_mod,
        "build_runtime_timezone_meta",
        lambda _result, include_now=True: {
            "server": {"tz": "Europe/Nicosia", "offset_seconds": 7200},
            "client": {"tz": "UTC", "now": "2024-01-02T12:00:00+00:00"},
        },
    )

    context = market_status_mod._symbol_market_status_timezone_context(
        "server",
        now_utc=datetime(2024, 1, 2, 12, 0, tzinfo=timezone.utc),
    )

    assert context["timezone_display"] == "server"
    assert context["authoritative_clock"] == "server"
    assert context["status_timezone"] == "Europe/Nicosia"
    assert context["market_now"] == "2024-01-02T14:00:00+02:00"


def test_market_status_symbol_timezone_context_honors_local_and_utc_display(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        market_status_mod,
        "build_runtime_timezone_meta",
        lambda _result, include_now=True: {
            "server": {"tz": "Europe/Nicosia", "offset_seconds": 7200},
            "client": {"tz": "America/New_York", "now": "2024-01-02T07:00:00-05:00"},
        },
    )

    local = market_status_mod._symbol_market_status_timezone_context(
        "local",
        now_utc=datetime(2024, 1, 2, 12, 0, tzinfo=timezone.utc),
    )
    utc = market_status_mod._symbol_market_status_timezone_context(
        "utc",
        now_utc=datetime(2024, 1, 2, 12, 0, tzinfo=timezone.utc),
    )

    assert local["authoritative_clock"] == "client"
    assert local["status_timezone"] == "America/New_York"
    assert local["market_now"] == "2024-01-02T07:00:00-05:00"
    assert utc["authoritative_clock"] == "utc"
    assert utc["status_timezone"] == "UTC"
    assert utc["market_now"] == "2024-01-02T12:00:00Z"


def test_market_status_symbol_mode_honors_timezone_display(monkeypatch) -> None:
    raw = _unwrap(market_status_mod.market_status)
    fixed_now = datetime(2024, 1, 2, 12, 0, tzinfo=timezone.utc)
    now_epoch = fixed_now.timestamp()

    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            if tz is None:
                return fixed_now.replace(tzinfo=None)
            return fixed_now.astimezone(tz)

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

    monkeypatch.setattr(market_status_mod, "datetime", FixedDateTime)
    monkeypatch.setattr(market_status_mod, "create_mt5_gateway", lambda **kwargs: Gateway())
    monkeypatch.setattr(
        market_status_mod,
        "build_runtime_timezone_meta",
        lambda _result, include_now=True: {
            "server": {"tz": "Europe/Nicosia", "offset_seconds": 7200},
            "client": {"tz": "America/New_York", "now": "2024-01-02T07:00:00-05:00"},
        },
    )

    expected = {
        "server": ("2024-01-02T14:00:00+02:00", "Europe/Nicosia", "server"),
        "local": ("2024-01-02T07:00:00-05:00", "America/New_York", "client"),
        "utc": ("2024-01-02T12:00:00Z", "UTC", "utc"),
    }
    for display, (clock, tz_name, authority) in expected.items():
        result = raw(symbol="EURUSD", timezone_display=display)
        assert result["market_clock"] == clock
        assert result["market_clock_timezone"] == tz_name
        assert result["authoritative_clock"] == authority


def test_market_status_symbol_mode_handles_bool_like_trade_and_schedule(monkeypatch) -> None:
    fixed_now = datetime(2024, 1, 2, 12, 0, tzinfo=timezone.utc)

    class BoolLike:
        def __bool__(self) -> bool:
            return True

    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            if tz is None:
                return fixed_now.replace(tzinfo=None)
            return fixed_now.astimezone(tz)

    class Gateway:
        def ensure_connection(self) -> None:
            return None

        def symbol_info(self, symbol: str):
            return SimpleNamespace(name=symbol, visible=True, trade_mode=4)

        def symbol_info_tick(self, symbol: str):
            return SimpleNamespace(time=fixed_now.timestamp(), bid=1.1, ask=1.2)

    monkeypatch.setattr(market_status_mod, "datetime", FixedDateTime)
    monkeypatch.setattr(
        market_status_mod,
        "_symbol_trade_mode_status",
        lambda gateway, trade_mode: {
            "can_open_new_positions": BoolLike(),
            "status": "open",
            "trade_mode_label": "Full",
        },
    )
    monkeypatch.setattr(
        market_status_mod,
        "_infer_symbol_schedule_from_recent_candles",
        lambda symbol, gateway, now_utc=None: {
            "source": "recent_candles",
            "confidence": "high",
            "current_time_in_active_session": BoolLike(),
            "trades_on_weekends": False,
            "inferred_24_7": False,
        },
    )

    result = market_status_mod._check_symbol_market_status(
        "EURUSD",
        detail="summary",
        gateway=Gateway(),
    )

    assert result["status"] == "probably_open"
    assert result["can_open_new_positions"] is True
    assert result["trade_mode_allows_opening"] is True


def test_market_status_symbol_mode_blocks_weekend_opening(monkeypatch) -> None:
    raw = _unwrap(market_status_mod.market_status)
    fixed_now = datetime(2026, 4, 25, 3, 14, tzinfo=timezone.utc)
    now_epoch = fixed_now.timestamp()

    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            if tz is None:
                return fixed_now.replace(tzinfo=None)
            return fixed_now.astimezone(tz)

    class Gateway:
        SYMBOL_TRADE_MODE_FULL = 4
        SYMBOL_TRADE_MODE_DISABLED = 0
        SYMBOL_TRADE_MODE_CLOSEONLY = 3
        SYMBOL_TRADE_MODE_LONGONLY = 1
        SYMBOL_TRADE_MODE_SHORTONLY = 2

        def ensure_connection(self) -> None:
            return None

        def symbol_info(self, symbol: str):
            return SimpleNamespace(name=symbol, visible=True, trade_mode=4)

        def symbol_info_tick(self, symbol: str):
            return SimpleNamespace(time=now_epoch - 60, bid=1.1, ask=1.2)

    monkeypatch.setattr(market_status_mod, "datetime", FixedDateTime)
    monkeypatch.setattr(market_status_mod, "create_mt5_gateway", lambda **kwargs: Gateway())

    result = raw(symbol="EURUSD")

    assert result["status"] == "weekend_closed"
    assert result["reason"] == "weekend"
    assert result["can_open_new_positions"] is False
    assert result["trade_mode_allows_opening"] is True
    assert "message" not in result


def test_market_status_symbol_mode_allows_crypto_on_weekend(monkeypatch) -> None:
    raw = _unwrap(market_status_mod.market_status)
    fixed_now = datetime(2026, 4, 25, 3, 14, tzinfo=timezone.utc)
    now_epoch = fixed_now.timestamp()

    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            if tz is None:
                return fixed_now.replace(tzinfo=None)
            return fixed_now.astimezone(tz)

    class Gateway:
        SYMBOL_TRADE_MODE_FULL = 4
        SYMBOL_TRADE_MODE_DISABLED = 0
        SYMBOL_TRADE_MODE_CLOSEONLY = 3
        SYMBOL_TRADE_MODE_LONGONLY = 1
        SYMBOL_TRADE_MODE_SHORTONLY = 2

        def ensure_connection(self) -> None:
            return None

        def symbol_info(self, symbol: str):
            assert symbol == "BTCUSD"
            return SimpleNamespace(name=symbol, visible=True, trade_mode=4)

        def symbol_info_tick(self, symbol: str):
            assert symbol == "BTCUSD"
            return SimpleNamespace(time=now_epoch - 60, bid=65000.0, ask=65001.0)

    monkeypatch.setattr(market_status_mod, "datetime", FixedDateTime)
    monkeypatch.setattr(market_status_mod, "create_mt5_gateway", lambda **kwargs: Gateway())

    result = raw(symbol="BTCUSD")

    assert result["status"] == "probably_open"
    assert result["can_open_new_positions"] is True
    assert result["trade_mode_allows_opening"] is True
    assert "reason" not in result
    assert result["tick_freshness"] == "fresh"


def test_market_status_symbol_mode_uses_recent_candles_for_weekend_session(
    monkeypatch,
) -> None:
    raw = _unwrap(market_status_mod.market_status)
    fixed_now = datetime(2026, 4, 25, 3, 14, tzinfo=timezone.utc)
    now_epoch = fixed_now.timestamp()
    previous_week_same_hour = fixed_now - timedelta(days=7)

    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            if tz is None:
                return fixed_now.replace(tzinfo=None)
            return fixed_now.astimezone(tz)

    class Gateway:
        TIMEFRAME_M1 = 1
        SYMBOL_TRADE_MODE_FULL = 4
        SYMBOL_TRADE_MODE_DISABLED = 0
        SYMBOL_TRADE_MODE_CLOSEONLY = 3
        SYMBOL_TRADE_MODE_LONGONLY = 1
        SYMBOL_TRADE_MODE_SHORTONLY = 2

        def ensure_connection(self) -> None:
            return None

        def symbol_info(self, symbol: str):
            assert symbol == "XAUUSD"
            return SimpleNamespace(name=symbol, visible=True, trade_mode=4)

        def symbol_info_tick(self, symbol: str):
            assert symbol == "XAUUSD"
            return SimpleNamespace(time=now_epoch - 60, bid=2400.0, ask=2400.5)

        def copy_rates_range(self, symbol: str, timeframe: int, start, end):
            assert symbol == "XAUUSD"
            assert timeframe == self.TIMEFRAME_M1
            assert start < end
            return [{"time": previous_week_same_hour.timestamp()}]

    monkeypatch.setattr(market_status_mod, "datetime", FixedDateTime)
    monkeypatch.setattr(market_status_mod, "create_mt5_gateway", lambda **kwargs: Gateway())

    result = raw(symbol="XAUUSD", detail="full")

    assert result["status"] == "probably_open"
    assert result["can_open_new_positions"] is True
    assert result["current_time_in_recent_session"] is True
    assert result["trades_on_weekends"] is True
    assert result["schedule_source"] == "recent_m1_candles"
    assert result["inferred_schedule"]["active_hours_utc"] == {
        "saturday": ["03:00-04:00"]
    }
    assert "reason" not in result


def test_market_status_symbol_mode_marks_weekend_snapshot_freshness(monkeypatch) -> None:
    raw = _unwrap(market_status_mod.market_status)
    fixed_now = datetime(2026, 4, 25, 12, 0, tzinfo=timezone.utc)
    now_epoch = fixed_now.timestamp()

    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            if tz is None:
                return fixed_now.replace(tzinfo=None)
            return fixed_now.astimezone(tz)

    class Gateway:
        TIMEFRAME_M1 = 1
        SYMBOL_TRADE_MODE_FULL = 4
        SYMBOL_TRADE_MODE_DISABLED = 0
        SYMBOL_TRADE_MODE_CLOSEONLY = 3
        SYMBOL_TRADE_MODE_LONGONLY = 1
        SYMBOL_TRADE_MODE_SHORTONLY = 2

        def ensure_connection(self) -> None:
            return None

        def symbol_info(self, symbol: str):
            assert symbol == "EURUSD"
            return SimpleNamespace(name=symbol, visible=True, trade_mode=4)

        def symbol_info_tick(self, symbol: str):
            assert symbol == "EURUSD"
            return SimpleNamespace(time=now_epoch - (36 * 60 * 60), bid=1.1, ask=1.2)

        def copy_rates_range(self, symbol: str, timeframe: int, start, end):
            return []

    monkeypatch.setattr(market_status_mod, "datetime", FixedDateTime)
    monkeypatch.setattr(market_status_mod, "create_mt5_gateway", lambda **kwargs: Gateway())

    result = raw(symbol="EURUSD", detail="full")

    assert result["status"] == "weekend_closed"
    assert result["tick_freshness"] == "closed_weekend_snapshot"
    assert result["tick"]["market_status"] == "closed"
    assert result["tick"]["market_status_reason"] == "weekend"
    assert result["tick"]["freshness_policy_relaxed"] is True


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

    monkeypatch.setattr(market_status_mod, "create_mt5_gateway", lambda **kwargs: Gateway())

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


def test_normalize_market_status_output_compact_hides_messages_and_holidays() -> None:
    payload = {
        "success": True,
        "message": "human summary",
        "markets": [
            {"symbol": "NYSE", "status": "open", "message": "NYSE: Open"},
            {"symbol": "NASDAQ", "status": "closed", "reason": "weekend"},
        ],
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

    assert "message" not in compact
    assert "message" not in compact["markets"][0]
    assert "upcoming_holidays" not in compact
    assert "upcoming_holidays_count" not in compact
    assert "upcoming_holidays_summary" not in compact
    assert "show_all_hint" not in compact

    assert full["upcoming_holidays"] == payload["upcoming_holidays"]
    assert full["markets"][0]["message"] == "NYSE: Open"


def test_normalize_market_status_output_metadata_extra_keeps_holidays() -> None:
    payload = {
        "success": True,
        "message": "human summary",
        "markets": [{"symbol": "NYSE", "status": "open", "message": "NYSE: Open"}],
        "upcoming_holidays": [{"date": "2031-01-01", "holiday": "New Year's Day"}],
        "upcoming_holidays_count": 1,
    }

    compact = market_status_mod.normalize_market_status_output(
        payload,
        detail="compact",
        extras="metadata",
    )

    assert "message" not in compact
    assert "message" not in compact["markets"][0]
    assert compact["upcoming_holidays"] == payload["upcoming_holidays"]
    assert compact["upcoming_holidays_count"] == 1


def test_normalize_market_status_output_handles_payload_without_markets() -> None:
    payload = {"success": True, "message": "human summary"}

    compact = market_status_mod.normalize_market_status_output(payload, detail="compact")

    assert compact == {"success": True}
