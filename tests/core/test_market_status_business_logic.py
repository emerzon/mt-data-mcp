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

    assert [param.name for param in params] == [
        "symbol",
        "region",
        "timezone_display",
        "detail",
    ]
    assert params[0].default is None
    assert params[1].default == "all"
    assert params[2].default == "auto"
    assert params[3].default == "compact"


def test_market_status_timezone_display_utc_converts_market_times(monkeypatch) -> None:
    raw = _unwrap(market_status_mod.market_status)
    fixed_now = datetime(2024, 1, 2, 10, 0, tzinfo=ZoneInfo("America/New_York"))

    monkeypatch.setattr(market_status_mod, "_get_local_time", lambda _tz_name: fixed_now)
    monkeypatch.setattr(market_status_mod, "_get_upcoming_holidays", lambda _markets: [])

    result = raw(region="us", timezone_display="utc", detail="full")

    assert result["success"] is True
    assert result["mode"] == "equity_exchanges"
    assert result["market_scope"] == "major_equity_exchanges"
    assert "pass a broker symbol" in result["scope_note"].lower()
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
    assert result["mode"] == "equity_exchanges"
    assert result["market_scope"] == "major_equity_exchanges"
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

        def symbols_get(self):
            return [SimpleNamespace(name="EURUSD")]

        def symbol_info_tick(self, symbol: str):
            assert symbol == "EURUSD"
            return SimpleNamespace(time=now_epoch, bid=1.1, ask=1.2)

    class GatewayWithEmptySchedule(Gateway):
        TIMEFRAME_M1 = 1

        def copy_rates_range(self, symbol: str, timeframe: int, start, end):
            return []

    monkeypatch.setattr(market_status_mod, "datetime", FixedDateTime)
    monkeypatch.setattr(
        market_status_mod,
        "create_mt5_gateway",
        lambda **kwargs: GatewayWithEmptySchedule(),
    )

    result = raw(symbol="EUR/USD", timezone_display="utc")

    assert result["mode"] == "symbol"
    assert result["symbol"] == "EURUSD"
    assert result["symbol_input"] == "EUR/USD"
    assert result["timezone"] == "UTC"
    assert result["status"] == "probably_open"
    assert result["status_source"] == "trade_mode_and_tick_freshness"
    assert result["status_confidence"] == "heuristic"
    assert result["heuristic_note"].startswith(
        "Symbol status is inferred from MT5 trade_mode, tick freshness"
    )
    assert "FX weekly sessions typically run Sun 17:00-Fri 17:00" in result[
        "heuristic_note"
    ]
    assert result["can_open_new_positions"] is True
    assert result["trade_mode_allows_opening"] is True
    assert result["tick_freshness"] == "live"
    assert result["tick_available"] is True
    assert result["data_fetched_at"] == "2024-01-02T12:00:00Z"
    assert result["last_tick_time"] == "2024-01-02T12:00:00Z"
    assert result["is_tradable"] is True
    assert result["is_tradable_confidence"] == "broker_trade_mode"
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


def test_symbol_tick_snapshot_prefers_millisecond_timestamp() -> None:
    now_utc = datetime.fromtimestamp(1_001.0, tz=timezone.utc)

    result = market_status_mod._symbol_tick_snapshot(
        "EURUSD",
        {
            "time": 1_000.0,
            "time_msc": 1_000_750,
            "bid": 1.1,
            "ask": 1.1002,
        },
        now_utc=now_utc,
    )

    assert result["last_tick_time"] == "1970-01-01T00:16:40Z"
    assert result["last_tick_age_seconds"] == 0.25
    assert result["tick_freshness"] == "live"


def test_market_status_blocks_new_entries_when_tick_timestamp_is_unsafe(monkeypatch) -> None:
    raw = _unwrap(market_status_mod.market_status)
    fixed_now = datetime(2026, 7, 16, 12, 0, tzinfo=timezone.utc)

    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed_now.replace(tzinfo=None) if tz is None else fixed_now.astimezone(tz)

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
            return SimpleNamespace(
                time=fixed_now.timestamp() + 15.0,
                bid=1.1,
                ask=1.2,
            )

    monkeypatch.setattr(market_status_mod, "datetime", FixedDateTime)
    monkeypatch.setattr(market_status_mod, "create_mt5_gateway", lambda **kwargs: Gateway())

    result = raw(symbol="EURUSD")

    assert result["status"] == "quote_not_live_ready"
    assert result["trade_mode_allows_opening"] is True
    assert result["can_open_new_positions"] is False
    assert result["is_tradable"] is True
    assert result["tick_freshness"] == "clock_skew"
    assert result["freshness_reason"] == "future_timestamp"
    assert result["timestamp_in_future"] is True


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
                "is_tradable": BoolLike(),
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
    assert "exchange-calendar guarantee" in result["heuristic_note"]


def test_inferred_symbol_schedule_normalizes_server_epochs_to_utc(monkeypatch) -> None:
    now_utc = datetime(2026, 4, 20, 12, 0, tzinfo=timezone.utc)
    server_epoch = (now_utc + timedelta(hours=3)).timestamp()

    class Gateway:
        TIMEFRAME_M1 = 1

        def copy_rates_range(self, symbol, timeframe, start, end):
            return [{"time": server_epoch}]

    monkeypatch.setattr(
        market_status_mod,
        "_normalize_times_in_struct",
        lambda rows: [{**row, "time": row["time"] - 3 * 3600} for row in rows],
    )

    result = market_status_mod._infer_symbol_schedule_from_recent_candles(
        "TEST",
        Gateway(),
        now_utc=now_utc,
    )

    assert result["active_hours_utc"] == {"monday": ["12:00-13:00"]}
    assert result["current_time_in_active_session"] is True


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
    assert result["is_tradable"] is True
    assert "message" not in result


def test_market_status_uses_standard_weekend_boundary_for_index_cfd(monkeypatch) -> None:
    raw = _unwrap(market_status_mod.market_status)
    fixed_now = datetime(2026, 7, 17, 22, 30, tzinfo=timezone.utc)

    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed_now.replace(tzinfo=None) if tz is None else fixed_now.astimezone(tz)

    class Gateway:
        SYMBOL_TRADE_MODE_FULL = 4
        SYMBOL_TRADE_MODE_DISABLED = 0
        SYMBOL_TRADE_MODE_CLOSEONLY = 3
        SYMBOL_TRADE_MODE_LONGONLY = 1
        SYMBOL_TRADE_MODE_SHORTONLY = 2
        TIMEFRAME_M1 = 1

        def ensure_connection(self):
            return None

        def symbol_info(self, symbol):
            return SimpleNamespace(name=symbol, visible=True, trade_mode=4)

        def symbol_info_tick(self, symbol):
            return SimpleNamespace(time=fixed_now.timestamp() - 60, bid=45000.0, ask=45001.0)

        def copy_rates_range(self, symbol, timeframe, start, end):
            return []

    monkeypatch.setattr(market_status_mod, "datetime", FixedDateTime)
    monkeypatch.setattr(market_status_mod, "create_mt5_gateway", lambda **kwargs: Gateway())

    result = raw(symbol="US30")

    assert result["status"] == "weekend_closed"
    assert result["can_open_new_positions"] is False
    assert result["is_tradable"] is True


def test_close_only_symbol_remains_tradable_but_cannot_open() -> None:
    gateway = SimpleNamespace(
        SYMBOL_TRADE_MODE_FULL=4,
        SYMBOL_TRADE_MODE_DISABLED=0,
        SYMBOL_TRADE_MODE_CLOSEONLY=3,
        SYMBOL_TRADE_MODE_LONGONLY=1,
        SYMBOL_TRADE_MODE_SHORTONLY=2,
    )

    result = market_status_mod._symbol_trade_mode_status(gateway, 3)

    assert result["status"] == "close_only"
    assert result["can_open_new_positions"] is False
    assert result["is_tradable"] is True


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

    assert result["status"] == "quote_not_live_ready"
    assert result["can_open_new_positions"] is False
    assert result["trade_mode_allows_opening"] is True
    assert result["usable_for_live_trading"] is False
    assert result["tick_freshness"] == "recent"
    assert "FX weekly sessions" not in result["heuristic_note"]


def test_market_status_symbol_mode_allows_fx_after_sunday_open(monkeypatch) -> None:
    raw = _unwrap(market_status_mod.market_status)
    fixed_now = datetime(2026, 4, 26, 22, 15, tzinfo=timezone.utc)

    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed_now.replace(tzinfo=None) if tz is None else fixed_now.astimezone(tz)

    class Gateway:
        SYMBOL_TRADE_MODE_FULL = 4
        SYMBOL_TRADE_MODE_DISABLED = 0
        SYMBOL_TRADE_MODE_CLOSEONLY = 3
        SYMBOL_TRADE_MODE_LONGONLY = 1
        SYMBOL_TRADE_MODE_SHORTONLY = 2

        def ensure_connection(self):
            return None

        def symbol_info(self, symbol):
            return SimpleNamespace(name=symbol, visible=True, trade_mode=4)

        def symbol_info_tick(self, symbol):
            return SimpleNamespace(time=fixed_now.timestamp() - 60, bid=1.1, ask=1.2)

    monkeypatch.setattr(market_status_mod, "datetime", FixedDateTime)
    monkeypatch.setattr(market_status_mod, "create_mt5_gateway", lambda **kwargs: Gateway())

    result = raw(symbol="EURUSD")

    assert result["status"] == "quote_not_live_ready"
    assert result["can_open_new_positions"] is False
    assert result["trade_mode_allows_opening"] is True


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

    assert result["status"] == "quote_not_live_ready"
    assert result["can_open_new_positions"] is False
    assert result["trade_mode_allows_opening"] is True
    assert result["current_time_in_recent_session"] is True
    assert result["trades_on_weekends"] is True
    assert result["schedule_source"] == "recent_m1_candles"
    assert result["inferred_schedule"]["active_hours_utc"] == {
        "saturday": ["03:00-04:00"]
    }
    assert result["reason"] == "market_closed"


def test_recent_sunday_reopen_is_not_classified_as_weekend_trading() -> None:
    sunday_open = datetime(2026, 7, 12, 21, 0, tzinfo=timezone.utc)
    gateway = SimpleNamespace(
        TIMEFRAME_M1=1,
        copy_rates_range=lambda *args: [{"time": sunday_open.timestamp()}],
    )

    result = market_status_mod._infer_symbol_schedule_from_recent_candles(
        "EURUSD",
        gateway,
        now_utc=datetime(2026, 7, 14, 12, 0, tzinfo=timezone.utc),
    )

    assert result["trades_on_weekends"] is False
    assert result["saturday_candles"] == 0
    assert result["sunday_candles"] == 1


def test_market_status_reconciles_future_cached_tick_with_live_stream(monkeypatch) -> None:
    fixed_now = datetime(2026, 7, 30, 12, 0, tzinfo=timezone.utc)
    now_epoch = fixed_now.timestamp()

    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed_now.replace(tzinfo=None) if tz is None else fixed_now.astimezone(tz)

    class Gateway:
        COPY_TICKS_ALL = 0
        TIMEFRAME_M1 = 1
        SYMBOL_TRADE_MODE_FULL = 4
        SYMBOL_TRADE_MODE_DISABLED = 0
        SYMBOL_TRADE_MODE_CLOSEONLY = 3
        SYMBOL_TRADE_MODE_LONGONLY = 1
        SYMBOL_TRADE_MODE_SHORTONLY = 2

        def ensure_connection(self) -> None:
            return None

        def symbol_info(self, symbol: str):
            return SimpleNamespace(name=symbol, visible=True, trade_mode=4)

        def symbol_info_tick(self, _symbol: str):
            return SimpleNamespace(time=now_epoch + 45, bid=100.0, ask=101.0)

        def copy_ticks_range(self, _symbol, _start, _end, _flags):
            return [
                {
                    "time": now_epoch - 1,
                    "time_msc": (now_epoch - 1) * 1000,
                    "bid": 100.1,
                    "ask": 100.2,
                }
            ]

        def copy_rates_range(self, _symbol, _timeframe, _start, _end):
            return []

    monkeypatch.setattr(market_status_mod, "datetime", FixedDateTime)
    monkeypatch.setattr(market_status_mod, "create_mt5_gateway", lambda **kwargs: Gateway())

    result = _unwrap(market_status_mod.market_status)(symbol="BTCUSD", detail="full")

    assert result["status"] == "probably_open"
    assert result["can_open_new_positions"] is True
    assert result["tick"]["quote_source"] == "mt5.copy_ticks_range"
    assert result["tick"]["quote_source_state"] == "refreshed_from_tick_stream"
    assert result["tick"]["last_tick_age_seconds"] == 1.0

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

    def fake_financial_holidays(exchange: str, years):
        year_tuple = tuple(int(value) for value in years)
        calls.append((exchange, year_tuple))
        year = year_tuple[0]
        if year == 2031:
            return {date(2031, 1, 1): "New Year's Day"}
        return {}

    market_status_mod._get_exchange_holidays.cache_clear()
    monkeypatch.setattr(market_status_mod, "datetime", FixedDateTime)
    monkeypatch.setattr(
        market_status_mod.holidays,
        "financial_holidays",
        fake_financial_holidays,
    )

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
            "calendar_source": "exchange_calendar",
        }
    ]
    assert calls == [("XNYS", (2030,)), ("XNYS", (2031,))]


def test_upcoming_holidays_use_each_venue_local_date(monkeypatch) -> None:
    checked_dates: list[date] = []

    def fake_is_holiday(_country: str, dt: datetime, _exchange=None):
        checked_dates.append(dt.date())
        if dt.date() == date(2026, 1, 2):
            return True, "Local closure"
        return False, None

    monkeypatch.setattr(market_status_mod, "_is_holiday", fake_is_holiday)

    upcoming = market_status_mod._get_upcoming_holidays(
        ["TSE"],
        days_ahead=1,
        now_utc=datetime(2026, 1, 1, 15, 30, tzinfo=timezone.utc),
    )

    assert checked_dates[0] == date(2026, 1, 2)
    assert upcoming[0]["date"] == "2026-01-02"
    assert upcoming[0]["days_away"] == 0


def test_exchange_calendar_differs_from_country_calendar() -> None:
    market_status_mod._get_exchange_holidays.cache_clear()

    good_friday = datetime(2026, 4, 3, 12, tzinfo=timezone.utc)
    columbus_day = datetime(2026, 10, 12, 12, tzinfo=timezone.utc)
    veterans_day = datetime(2026, 11, 11, 12, tzinfo=timezone.utc)

    assert market_status_mod._is_holiday("US", good_friday, "XNYS")[0] is True
    assert market_status_mod._is_holiday("US", columbus_day, "XNYS")[0] is False
    assert market_status_mod._is_holiday("US", veterans_day, "XNYS")[0] is False


def test_tokyo_session_uses_current_1530_close(monkeypatch) -> None:
    monkeypatch.setattr(
        market_status_mod,
        "_is_holiday",
        lambda _country, _dt, _exchange=None: (False, None),
    )

    result = market_status_mod._check_market_status(
        "TSE",
        datetime(2026, 8, 6, 15, 15, tzinfo=ZoneInfo("Asia/Tokyo")),
    )

    assert result["status"] == "open"
    assert result["next_close"].endswith("T15:30:00+09:00")


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


def test_normalize_market_status_output_full_detail_keeps_holidays() -> None:
    payload = {
        "success": True,
        "message": "human summary",
        "markets": [{"symbol": "NYSE", "status": "open", "message": "NYSE: Open"}],
        "upcoming_holidays": [{"date": "2031-01-01", "holiday": "New Year's Day"}],
        "upcoming_holidays_count": 1,
    }

    full = market_status_mod.normalize_market_status_output(
        payload,
        detail="full",
    )

    assert full["message"] == "human summary"
    assert full["markets"][0]["message"] == "NYSE: Open"
    assert full["upcoming_holidays"] == payload["upcoming_holidays"]
    assert full["upcoming_holidays_count"] == 1


def test_normalize_market_status_output_handles_payload_without_markets() -> None:
    payload = {"success": True, "message": "human summary"}

    compact = market_status_mod.normalize_market_status_output(payload, detail="compact")

    assert compact == {"success": True}
