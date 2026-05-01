from __future__ import annotations

import inspect
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from mtdata.core import data as core_data
from mtdata.core.data import wait_events as wait_events_mod
from mtdata.core.data.requests import WaitEventRequest
from mtdata.core.data.use_cases import _wait_event_needs_gateway, run_wait_event


class FakeClock:
    def __init__(self, start: datetime) -> None:
        self.current = start
        self.monotonic_value = 0.0

    def now_utc(self) -> datetime:
        return self.current

    def monotonic(self) -> float:
        return self.monotonic_value

    def sleep(self, seconds: float) -> None:
        self.monotonic_value += float(seconds)
        self.current = self.current + timedelta(seconds=float(seconds))


class OversleepClock(FakeClock):
    def __init__(self, start: datetime, *, extra_sleep_seconds: float) -> None:
        super().__init__(start)
        self.extra_sleep_seconds = float(extra_sleep_seconds)

    def sleep(self, seconds: float) -> None:
        super().sleep(float(seconds) + self.extra_sleep_seconds)


class SequenceGateway:
    COPY_TICKS_ALL = 0
    POSITION_TYPE_BUY = 0
    POSITION_TYPE_SELL = 1
    ORDER_TYPE_BUY = 0
    ORDER_TYPE_SELL = 1
    DEAL_TYPE_BUY = 0
    DEAL_TYPE_SELL = 1
    DEAL_ENTRY_IN = 0
    DEAL_ENTRY_OUT = 1
    DEAL_ENTRY_INOUT = 2
    DEAL_ENTRY_OUT_BY = 3
    DEAL_REASON_TP = 5
    DEAL_REASON_SL = 6
    ORDER_STATE_CANCELED = 4

    def __init__(
        self,
        *,
        orders_seq=None,
        positions_seq=None,
        history_orders_seq=None,
        history_deals_seq=None,
        ticks_by_symbol=None,
        rates_by_symbol=None,
        rates_by_symbol_timeframe=None,
    ) -> None:
        self.orders_seq = list(orders_seq or [[]])
        self.positions_seq = list(positions_seq or [[]])
        self.history_orders_seq = list(history_orders_seq or [[]])
        self.history_deals_seq = list(history_deals_seq or [[]])
        self.ticks_by_symbol = dict(ticks_by_symbol or {})
        self.rates_by_symbol = dict(rates_by_symbol or {})
        self.rates_by_symbol_timeframe = dict(rates_by_symbol_timeframe or {})
        self._orders_calls = 0
        self._positions_calls = 0
        self._history_orders_calls = 0
        self._history_deals_calls = 0

    def ensure_connection(self) -> None:
        return None

    def symbol_select(self, symbol: str, visible: bool = True) -> bool:
        return True

    def orders_get(self, **kwargs):
        return self._next("orders")

    def positions_get(self, **kwargs):
        return self._next("positions")

    def history_orders_get(self, dt_from, dt_to, **kwargs):
        return self._next("history_orders")

    def history_deals_get(self, dt_from, dt_to, **kwargs):
        return self._next("history_deals")

    def copy_ticks_range(self, symbol, dt_from, dt_to, flags):
        rows = list(self.ticks_by_symbol.get(str(symbol).upper(), []))
        out = []
        from_epoch = float(dt_from.timestamp())
        to_epoch = float(dt_to.timestamp())
        for row in rows:
            epoch = float(row["time"])
            if from_epoch <= epoch <= to_epoch:
                out.append(row)
        return out

    def copy_rates_from(self, symbol, timeframe, dt_from, count):
        rows = list(
            self.rates_by_symbol_timeframe.get(
                (str(symbol).upper(), timeframe),
                self.rates_by_symbol.get(str(symbol).upper(), []),
            )
        )
        if not rows:
            return []
        if getattr(dt_from, "tzinfo", None) is None:
            dt_from = dt_from.replace(tzinfo=timezone.utc)
        to_epoch = float(dt_from.timestamp())
        filtered = [
            row for row in rows if float(row.get("time", 0.0)) <= to_epoch + 1e-6
        ]
        return filtered[-int(count):]

    def symbol_info_tick(self, symbol):
        rows = list(self.ticks_by_symbol.get(str(symbol).upper(), []))
        if not rows:
            return None
        row = rows[-1]
        return SimpleNamespace(
            time=row.get("time"),
            time_msc=row.get("time_msc"),
            bid=row.get("bid"),
            ask=row.get("ask"),
            last=row.get("last"),
            volume=row.get("volume"),
            volume_real=row.get("volume_real"),
            flags=row.get("flags"),
        )

    def _next(self, kind: str):
        seq = getattr(self, f"{kind}_seq")
        counter_name = f"_{kind}_calls"
        idx = getattr(self, counter_name)
        setattr(self, counter_name, idx + 1)
        if idx >= len(seq):
            return seq[-1]
        return seq[idx]


class ReplayHistoryGateway(SequenceGateway):
    def __init__(self, *, replay_deals=None, replay_orders=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.replay_deals = list(replay_deals or [])
        self.replay_orders = list(replay_orders or [])

    def history_orders_get(self, dt_from, dt_to, **kwargs):
        return list(self.replay_orders)

    def history_deals_get(self, dt_from, dt_to, **kwargs):
        return list(self.replay_deals)


class TrackingHistoryWindowGateway(SequenceGateway):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.history_order_calls = []
        self.history_deal_calls = []

    def history_orders_get(self, dt_from, dt_to, **kwargs):
        self.history_order_calls.append((dt_from, dt_to))
        return super().history_orders_get(dt_from, dt_to, **kwargs)

    def history_deals_get(self, dt_from, dt_to, **kwargs):
        self.history_deal_calls.append((dt_from, dt_to))
        return super().history_deals_get(dt_from, dt_to, **kwargs)


class DisconnectingGateway(SequenceGateway):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.ensure_calls = 0

    def ensure_connection(self) -> None:
        self.ensure_calls += 1
        if self.ensure_calls >= 2:
            raise RuntimeError("lost connection")


def test_wait_event_tool_exposes_minimal_public_contract(monkeypatch) -> None:
    def _mock_run_wait_event(request, gateway):
        return {
            "success": True,
            "symbol": request.symbol,
            "timeframe": request.timeframe,
            "status": "boundary_reached",
            "matched": False,
            "event": None,
            "boundary_event": {
                "type": "candle_close",
                "timeframe": request.timeframe,
                "buffer_seconds": 1.0,
                "next_candle_close_utc": "2026-04-06T02:01:00+00:00",
                "next_candle_close_server": "2026-04-06T05:01:00",
                "server_timezone": "Europe/Nicosia",
            },
            "bid": 1.2345,
            "ask": 1.2347,
            "started_at_utc": "2026-04-06T02:00:29.017205+00:00",
            "observed_at_utc": "2026-04-06T02:01:01+00:00",
            "elapsed_seconds": 32.0,
            "polls": 65,
            "poll_interval_seconds": request.poll_interval_seconds,
            "criteria": {
                "watch_for": list(request.watch_for or []),
                "watch_for_inferred": False,
                "end_on": list(
                    request.end_on
                    or [{"type": "candle_close", "timeframe": request.timeframe}]
                ),
                "end_on_inferred": False,
                "accept_preexisting": bool(request.accept_preexisting),
            },
            "max_wait_seconds": request.max_wait_seconds,
        }

    monkeypatch.setattr(
        core_data,
        "run_wait_event",
        _mock_run_wait_event,
    )
    monkeypatch.setattr(core_data, "create_mt5_gateway", lambda ensure_connection_impl=None: object())
    monkeypatch.setattr(
        core_data,
        "_build_default_wait_event_watchers",
        lambda symbol, timeframe, watch_tick_count_spike: [
            {"type": "position_opened", "symbol": symbol},
            {"type": "tick_count_spike", "symbol": symbol},
            {"type": "price_touch_level", "symbol": symbol, "level": 100.0},
        ] if watch_tick_count_spike else [
            {"type": "position_opened", "symbol": symbol},
            {"type": "price_touch_level", "symbol": symbol, "level": 100.0},
        ],
    )

    sig = inspect.signature(core_data.wait_event)
    assert tuple(sig.parameters.keys()) == (
        "symbol",
        "timeframe",
        "watch_tick_count_spike",
        "watch_for",
        "end_on",
        "detail",
        "json",
        "extras",
    )

    raw = getattr(core_data.wait_event, "__wrapped__", core_data.wait_event)
    result = raw(symbol="BTCUSD", timeframe="M1")

    assert result["success"] is True
    assert result["symbol"] == "BTCUSD"
    assert result["boundary_event"] == {
        "type": "candle_close",
        "timeframe": "M1",
    }
    assert result["bid"] == 1.2345
    assert result["ask"] == 1.2347
    assert result["observed_at_utc"] == "2026-04-06T02:01:01+00:00"
    assert "matched" not in result
    assert "event" not in result
    assert "criteria" not in result
    assert "started_at_utc" not in result
    assert "elapsed_seconds" not in result
    assert "polls" not in result
    assert "poll_interval_seconds" not in result
    assert "max_wait_seconds" not in result
    assert "timeframe" not in result
    assert "watched_for" not in result
    assert "ending_on" not in result
    assert "reason" not in result

    without_tick_count = raw(symbol="BTCUSD", timeframe="M1", watch_tick_count_spike=False)
    assert "criteria" not in without_tick_count

    explicit = raw(
        symbol="BTCUSD",
        timeframe="M1",
        watch_tick_count_spike=True,
        watch_for=[{"type": "price_touch_level", "symbol": "BTCUSD", "level": 100.0}],
        end_on=[{"type": "candle_close", "timeframe": "M5"}],
        detail="full",
    )
    assert [item.type for item in explicit["criteria"]["watch_for"]] == ["price_touch_level"]
    assert [item.type for item in explicit["criteria"]["end_on"]] == ["candle_close"]
    assert explicit["criteria"]["watch_for_inferred"] is False
    assert explicit["criteria"]["end_on_inferred"] is False
    assert explicit["boundary_event"]["buffer_seconds"] == 1.0
    assert explicit["matched"] is False
    assert explicit["event"] is None
    assert explicit["started_at_utc"] == "2026-04-06T02:00:29.017205+00:00"

def test_wait_event_tool_compact_result_preserves_boundary_closed_candle() -> None:
    result = core_data._compact_wait_event_public_result(
        {
            "success": True,
            "status": "boundary_reached",
            "matched": False,
            "event": None,
            "boundary_event": {
                "type": "candle_close",
                "timeframe": "M1",
                "buffer_seconds": 1.0,
                "closed_candle": {
                    "symbol": "EURUSD",
                    "timeframe": "M1",
                    "open": 1.1,
                    "high": 1.2,
                    "low": 1.0,
                    "close": 1.15,
                    "volume": 42,
                    "direction": "bullish",
                    "range": 0.2,
                },
            },
            "criteria": {
                "watch_for": [{"type": "order_created", "symbol": "EURUSD"}],
                "watch_for_inferred": False,
                "end_on": [{"type": "candle_close", "timeframe": "M1"}],
                "end_on_inferred": False,
                "accept_preexisting": False,
            },
        },
        explicit_watch_for=True,
        explicit_end_on=True,
    )

    assert result["boundary_event"] == {
        "type": "candle_close",
        "timeframe": "M1",
        "closed_candle": {
            "symbol": "EURUSD",
            "timeframe": "M1",
            "open": 1.1,
            "high": 1.2,
            "low": 1.0,
            "close": 1.15,
            "volume": 42,
            "direction": "bullish",
            "range": 0.2,
        },
    }

def test_wait_event_tool_compacts_matched_event_by_default(monkeypatch) -> None:
    def _mock_run_wait_event(request, gateway):
        return {
            "success": True,
            "status": "matched",
            "matched": True,
            "event": "price_touch_level",
            "symbol": request.symbol,
            "matched_event": {
                "type": "price_touch_level",
                "symbol": request.symbol,
                "criteria": {
                    "symbol": request.symbol,
                    "level": 100.0,
                    "tolerance": 0.1,
                    "direction": "up",
                },
                "observed": {
                    "symbol": request.symbol,
                    "current_price": 100.02,
                    "distance": 0.02,
                },
            },
            "criteria": {
                "watch_for": list(request.watch_for or []),
                "watch_for_inferred": False,
                "end_on": list(request.end_on or []),
                "end_on_inferred": False,
                "accept_preexisting": bool(request.accept_preexisting),
            },
            "bid": 100.01,
            "ask": 100.03,
            "started_at_utc": "2026-04-06T02:00:29.017205+00:00",
            "observed_at_utc": "2026-04-06T02:00:30+00:00",
            "elapsed_seconds": 1.0,
            "polls": 2,
            "poll_interval_seconds": request.poll_interval_seconds,
        }

    monkeypatch.setattr(core_data, "run_wait_event", _mock_run_wait_event)
    monkeypatch.setattr(core_data, "create_mt5_gateway", lambda ensure_connection_impl=None: object())

    raw = getattr(core_data.wait_event, "__wrapped__", core_data.wait_event)
    result = raw(
        symbol="BTCUSD",
        timeframe="M1",
        watch_tick_count_spike=True,
        watch_for=[{"type": "price_touch_level", "symbol": "BTCUSD", "level": 100.0}],
        end_on=None,
        detail="compact",
    )

    assert result["matched_event"] == {
        "type": "price_touch_level",
        "symbol": "BTCUSD",
        "observed": {
            "symbol": "BTCUSD",
            "current_price": 100.02,
            "distance": 0.02,
        },
    }
    assert result["symbol"] == "BTCUSD"
    assert result["bid"] == 100.01
    assert result["ask"] == 100.03
    assert result["observed_at_utc"] == "2026-04-06T02:00:30+00:00"
    assert "matched" not in result
    assert "event" not in result
    assert "criteria" not in result
    assert "started_at_utc" not in result
    assert "polls" not in result
    assert "watched_for" not in result
    assert "ending_on" not in result

def test_wait_event_tool_preserves_shared_account_identity_fields(monkeypatch) -> None:
    def _mock_run_wait_event(request, gateway):
        return {
            "success": True,
            "status": "matched",
            "matched": True,
            "event": "order_created",
            "symbol": request.symbol,
            "ticket": 7001,
            "order_ticket": 7001,
            "matched_event": {
                "type": "order_created",
                "symbol": request.symbol,
                "ticket": 7001,
                "order_ticket": 7001,
                "observed": {
                    "symbol": request.symbol,
                    "ticket": 7001,
                    "order_ticket": 7001,
                    "side": "buy",
                },
            },
            "started_at_utc": "2026-04-06T02:00:29.017205+00:00",
            "observed_at_utc": "2026-04-06T02:00:30+00:00",
            "elapsed_seconds": 1.0,
            "polls": 2,
            "poll_interval_seconds": request.poll_interval_seconds,
        }

    monkeypatch.setattr(core_data, "run_wait_event", _mock_run_wait_event)
    monkeypatch.setattr(core_data, "create_mt5_gateway", lambda ensure_connection_impl=None: object())

    raw = getattr(core_data.wait_event, "__wrapped__", core_data.wait_event)
    result = raw(
        symbol="EURUSD",
        timeframe="M1",
        watch_tick_count_spike=True,
        watch_for=[{"type": "order_created", "symbol": "EURUSD"}],
        end_on=None,
        detail="compact",
    )

    assert result["symbol"] == "EURUSD"
    assert result["ticket"] == 7001
    assert result["order_ticket"] == 7001
    assert result["matched_event"] == {
        "type": "order_created",
        "symbol": "EURUSD",
        "ticket": 7001,
        "order_ticket": 7001,
        "observed": {
            "symbol": "EURUSD",
            "ticket": 7001,
            "order_ticket": 7001,
            "side": "buy",
        },
    }

def test_wait_event_request_rejects_explicit_empty_watchers_without_boundary() -> None:
    with pytest.raises(
        ValidationError,
        match="watch_for cannot be an explicit empty list unless end_on or timeframe is provided",
    ):
        WaitEventRequest(watch_for=[])

def test_wait_event_gateway_check_treats_constructed_none_boundary_as_empty() -> None:
    request = WaitEventRequest.model_construct(watch_for=[], end_on=None, timeframe="M1")

    assert _wait_event_needs_gateway(request) is False

def test_run_wait_event_uses_all_default_watchers_when_omitted() -> None:
    gateway = SequenceGateway(
        orders_seq=[
            [],
            [{"ticket": 456, "symbol": "EURUSD", "type": "buy"}],
        ]
    )
    clock = FakeClock(datetime(2026, 3, 15, 12, 0, 0, tzinfo=timezone.utc))

    result = run_wait_event(
        WaitEventRequest(
            symbol="EURUSD",
            poll_interval_seconds=0.5,
            max_wait_seconds=5.0,
        ),
        gateway=gateway,
        sleep_impl=clock.sleep,
        monotonic_impl=clock.monotonic,
        now_utc_impl=clock.now_utc,
    )

    assert result["status"] == "matched"
    assert result["matched_event"]["type"] == "order_created"
    assert result["criteria"]["watch_for_inferred"] is True
    assert any(item["type"] == "price_change" for item in result["criteria"]["watch_for"])
    assert result["criteria"]["end_on_inferred"] is False

def test_run_wait_event_infers_candle_boundary_from_request_timeframe(monkeypatch) -> None:
    monkeypatch.setattr(
        "mtdata.core.data.wait_events._next_candle_wait_payload",
        lambda timeframe, buffer_seconds, now_utc: {
            "timeframe": timeframe,
            "buffer_seconds": buffer_seconds,
            "sleep_seconds": 2.0,
            "started_at_utc": now_utc.isoformat(),
            "next_candle_close_utc": (now_utc + timedelta(seconds=2)).isoformat(),
            "next_candle_close_server": "2026-03-15T12:00:02",
            "server_timezone": "UTC",
        },
    )
    gateway = SequenceGateway(orders_seq=[[], [], []], positions_seq=[[], [], []])
    clock = FakeClock(datetime(2026, 3, 15, 12, 0, 0, tzinfo=timezone.utc))

    result = run_wait_event(
        WaitEventRequest(
            watch_for=[],
            symbol="EURUSD",
            timeframe="M1",
            poll_interval_seconds=1.0,
            max_wait_seconds=10.0,
        ),
        gateway=gateway,
        sleep_impl=clock.sleep,
        monotonic_impl=clock.monotonic,
        now_utc_impl=clock.now_utc,
    )

    assert result["status"] == "completed"
    assert result["event"] == "candle_close"
    assert result["symbol"] == "EURUSD"
    assert result["boundary_event"]["type"] == "candle_close"
    assert result["boundary_event"]["timeframe"] == "M1"

def test_run_wait_event_defers_boundary_only_when_cap_is_short(monkeypatch) -> None:
    monkeypatch.setattr(
        "mtdata.core.data.wait_events._next_candle_wait_payload",
        lambda timeframe, buffer_seconds, now_utc: {
            "timeframe": timeframe,
            "buffer_seconds": buffer_seconds,
            "sleep_seconds": 120.0,
            "started_at_utc": "2026-03-15T12:00:00+00:00",
            "next_candle_close_utc": "2026-03-15T12:02:00+00:00",
            "next_candle_close_server": "2026-03-15T12:02:00",
            "server_timezone": "UTC",
        },
    )

    result = run_wait_event(
        WaitEventRequest(
            watch_for=[],
            end_on=[{"type": "candle_close", "timeframe": "M1"}],
            max_wait_seconds=30.0,
        ),
        gateway=None,
    )

    assert result["success"] is True
    assert result["status"] == "deferred_timeout_risk"
    assert result["event"] == "candle_close"

def test_run_wait_event_uses_timeframe_as_boundary_when_watchers_are_inferred(monkeypatch) -> None:
    monkeypatch.setattr(
        "mtdata.core.data.wait_events._next_candle_wait_payload",
        lambda timeframe, buffer_seconds, now_utc: {
            "timeframe": timeframe,
            "buffer_seconds": buffer_seconds,
            "sleep_seconds": 2.0,
            "started_at_utc": now_utc.isoformat(),
            "next_candle_close_utc": (now_utc + timedelta(seconds=2)).isoformat(),
            "next_candle_close_server": "2026-03-15T12:00:02",
            "server_timezone": "UTC",
        },
    )
    gateway = SequenceGateway(
        orders_seq=[[], [], []],
        positions_seq=[[], [], []],
        history_orders_seq=[[], [], []],
        history_deals_seq=[[], [], []],
        ticks_by_symbol={
            "EURUSD": [
                {
                    "time": 1_773_942_001.0,
                    "time_msc": 1_773_942_001_000,
                    "bid": 1.205,
                    "ask": 1.2053,
                    "last": 1.20515,
                }
            ]
        },
    )
    clock = FakeClock(datetime(2026, 3, 15, 12, 0, 0, tzinfo=timezone.utc))

    result = run_wait_event(
        WaitEventRequest(
            symbol="EURUSD",
            timeframe="M1",
            poll_interval_seconds=1.0,
            max_wait_seconds=10.0,
        ),
        gateway=gateway,
        sleep_impl=clock.sleep,
        monotonic_impl=clock.monotonic,
        now_utc_impl=clock.now_utc,
    )

    assert result["status"] == "boundary_reached"
    assert result["boundary_event"]["type"] == "candle_close"
    assert result["boundary_event"]["timeframe"] == "M1"
    assert result["bid"] == 1.205
    assert result["ask"] == 1.2053
    assert result["criteria"]["watch_for_inferred"] is True
    assert result["criteria"]["end_on_inferred"] is True

def test_run_wait_event_boundary_only_includes_gateway_quote_when_symbol_is_set(monkeypatch) -> None:
    monkeypatch.setattr(
        "mtdata.core.data.wait_events._next_candle_wait_payload",
        lambda timeframe, buffer_seconds, now_utc: {
            "timeframe": timeframe,
            "buffer_seconds": buffer_seconds,
            "sleep_seconds": 1.0,
            "started_at_utc": now_utc.isoformat(),
            "next_candle_close_utc": (now_utc + timedelta(seconds=1)).isoformat(),
            "next_candle_close_server": "2026-03-15T12:00:01",
            "server_timezone": "UTC",
        },
    )
    monkeypatch.setattr(
        "mtdata.core.data.wait_events._sleep_until_next_candle",
        lambda timeframe, buffer_seconds, sleep_impl, now_utc: {
            "timeframe": timeframe,
            "buffer_seconds": buffer_seconds,
            "sleep_seconds": 1.0,
            "started_at_utc": now_utc.isoformat(),
            "next_candle_close_utc": (now_utc + timedelta(seconds=1)).isoformat(),
            "next_candle_close_server": "2026-03-15T12:00:01",
            "server_timezone": "UTC",
            "status": "completed",
            "slept": True,
            "slept_seconds": 1.0,
            "remaining_seconds": 0.0,
        },
    )
    gateway = SequenceGateway(
        ticks_by_symbol={
            "EURUSD": [
                {
                    "time": 1_773_942_001.0,
                    "time_msc": 1_773_942_001_000,
                    "bid": 1.305,
                    "ask": 1.3054,
                    "last": 1.3052,
                }
            ]
        }
    )
    clock = FakeClock(datetime(2026, 3, 15, 12, 0, 0, tzinfo=timezone.utc))

    result = run_wait_event(
        WaitEventRequest(
            symbol="EURUSD",
            watch_for=[],
            end_on=[{"type": "candle_close", "timeframe": "M1"}],
            poll_interval_seconds=1.0,
            max_wait_seconds=10.0,
        ),
        gateway=gateway,
        sleep_impl=clock.sleep,
        monotonic_impl=clock.monotonic,
        now_utc_impl=clock.now_utc,
    )

    assert result["status"] == "completed"
    assert result["event"] == "candle_close"
    assert result["bid"] == 1.305
    assert result["ask"] == 1.3054
    assert result["observed_at_utc"] == "2026-03-15T12:00:01+00:00"

def test_run_wait_event_boundary_only_includes_closed_candle_stats(monkeypatch) -> None:
    monkeypatch.setattr(
        "mtdata.core.data.wait_events._next_candle_wait_payload",
        lambda timeframe, buffer_seconds, now_utc: {
            "timeframe": timeframe,
            "buffer_seconds": buffer_seconds,
            "sleep_seconds": 60.0,
            "started_at_utc": now_utc.isoformat(),
            "next_candle_close_utc": (now_utc + timedelta(seconds=60)).isoformat(),
            "next_candle_close_server": "2026-03-15T12:01:00",
            "server_timezone": "UTC",
        },
    )
    monkeypatch.setattr(
        "mtdata.core.data.wait_events._sleep_until_next_candle",
        lambda timeframe, buffer_seconds, sleep_impl, now_utc: {
            "timeframe": timeframe,
            "buffer_seconds": buffer_seconds,
            "sleep_seconds": 60.0,
            "started_at_utc": now_utc.isoformat(),
            "next_candle_close_utc": (now_utc + timedelta(seconds=60)).isoformat(),
            "next_candle_close_server": "2026-03-15T12:01:00",
            "server_timezone": "UTC",
            "status": "completed",
            "slept": True,
            "slept_seconds": 60.0,
            "remaining_seconds": 0.0,
        },
    )
    started = datetime(2026, 3, 15, 12, 0, 0, tzinfo=timezone.utc)
    gateway = SequenceGateway(
        rates_by_symbol={
            "EURUSD": [
                {
                    "time": started.timestamp(),
                    "open": 1.1000,
                    "high": 1.2000,
                    "low": 1.0000,
                    "close": 1.1500,
                    "tick_volume": 42,
                    "real_volume": 0,
                    "spread": 7,
                }
            ]
        }
    )
    clock = FakeClock(started)

    result = run_wait_event(
        WaitEventRequest(
            symbol="EURUSD",
            watch_for=[],
            end_on=[{"type": "candle_close", "timeframe": "M1"}],
            poll_interval_seconds=1.0,
            max_wait_seconds=120.0,
        ),
        gateway=gateway,
        sleep_impl=clock.sleep,
        monotonic_impl=clock.monotonic,
        now_utc_impl=clock.now_utc,
    )

    closed_candle = result["boundary_event"]["closed_candle"]
    assert closed_candle["symbol"] == "EURUSD"
    assert closed_candle["timeframe"] == "M1"
    assert closed_candle["open_time_utc"] == "2026-03-15T12:00:00+00:00"
    assert closed_candle["close_time_utc"] == "2026-03-15T12:01:00+00:00"
    assert closed_candle["open"] == 1.1
    assert closed_candle["high"] == 1.2
    assert closed_candle["low"] == 1.0
    assert closed_candle["close"] == 1.15
    assert closed_candle["volume"] == 42
    assert closed_candle["tick_volume"] == 42
    assert closed_candle["spread"] == 7
    assert closed_candle["direction"] == "bullish"
    assert closed_candle["change"] == 0.05
    assert closed_candle["range"] == 0.2
    assert closed_candle["body"] == 0.05
    assert closed_candle["close_position"] == 0.75

def test_run_wait_event_still_matches_pre_boundary_market_event_after_oversleep(monkeypatch) -> None:
    monkeypatch.setattr(
        "mtdata.core.data.wait_events._next_candle_wait_payload",
        lambda timeframe, buffer_seconds, now_utc: {
            "timeframe": timeframe,
            "buffer_seconds": buffer_seconds,
            "sleep_seconds": 1.0,
            "started_at_utc": now_utc.isoformat(),
            "next_candle_close_utc": (now_utc + timedelta(seconds=1)).isoformat(),
            "next_candle_close_server": "2026-03-15T12:00:01",
            "server_timezone": "UTC",
        },
    )
    started = datetime(2026, 3, 15, 12, 0, 0, tzinfo=timezone.utc)
    base_epoch = started.timestamp()
    gateway = SequenceGateway(
        ticks_by_symbol={
            "EURUSD": [
                {
                    "time": base_epoch - 0.2,
                    "time_msc": int((base_epoch - 0.2) * 1000),
                    "bid": 99.7,
                    "ask": 99.9,
                    "last": 99.8,
                },
                {
                    "time": base_epoch + 0.8,
                    "time_msc": int((base_epoch + 0.8) * 1000),
                    "bid": 99.95,
                    "ask": 100.05,
                    "last": 100.0,
                },
            ]
        }
    )
    clock = OversleepClock(started, extra_sleep_seconds=0.5)

    result = run_wait_event(
        WaitEventRequest(
            watch_for=[
                {
                    "type": "price_touch_level",
                    "symbol": "EURUSD",
                    "level": 100.0,
                    "price_source": "mid",
                    "direction": "up",
                    "tolerance": 0.05,
                }
            ],
            end_on=[{"type": "candle_close", "timeframe": "M1", "buffer_seconds": 0.0}],
            poll_interval_seconds=10.0,
            max_wait_seconds=30.0,
        ),
        gateway=gateway,
        sleep_impl=clock.sleep,
        monotonic_impl=clock.monotonic,
        now_utc_impl=clock.now_utc,
    )

    assert result["status"] == "matched"
    assert result["symbol"] == "EURUSD"
    assert result["matched_event"]["type"] == "price_touch_level"
    assert result["matched_event"]["symbol"] == "EURUSD"

def test_run_wait_event_stops_on_candle_boundary_when_no_watch_event(monkeypatch) -> None:
    monkeypatch.setattr(
        "mtdata.core.data.wait_events._next_candle_wait_payload",
        lambda timeframe, buffer_seconds, now_utc: {
            "timeframe": timeframe,
            "buffer_seconds": buffer_seconds,
            "sleep_seconds": 2.0,
            "started_at_utc": now_utc.isoformat(),
            "next_candle_close_utc": (now_utc + timedelta(seconds=2)).isoformat(),
            "next_candle_close_server": "2026-03-15T12:00:02",
            "server_timezone": "UTC",
        },
    )
    started = datetime(2026, 3, 15, 12, 0, 0, tzinfo=timezone.utc)
    gateway = SequenceGateway(
        orders_seq=[[], [], []],
        rates_by_symbol={
            "EURUSD": [
                {
                    "time": started.timestamp() - 58.0,
                    "open": 1.2000,
                    "high": 1.2050,
                    "low": 1.1975,
                    "close": 1.1980,
                    "tick_volume": 18,
                }
            ]
        },
    )
    clock = FakeClock(started)

    result = run_wait_event(
        WaitEventRequest(
            watch_for=[{"type": "order_created", "symbol": "EURUSD"}],
            end_on=[{"type": "candle_close", "timeframe": "M1", "buffer_seconds": 0.0}],
            poll_interval_seconds=1.0,
            max_wait_seconds=10.0,
        ),
        gateway=gateway,
        sleep_impl=clock.sleep,
        monotonic_impl=clock.monotonic,
        now_utc_impl=clock.now_utc,
    )

    assert result["status"] == "boundary_reached"
    assert result["boundary_event"]["type"] == "candle_close"
    assert result["boundary_event"]["closed_candle"]["symbol"] == "EURUSD"
    assert result["boundary_event"]["closed_candle"]["direction"] == "bearish"
    assert result["boundary_event"]["closed_candle"]["volume"] == 18

def test_run_wait_event_respects_boundary_when_live_state_changes_after_oversleep(monkeypatch) -> None:
    monkeypatch.setattr(
        "mtdata.core.data.wait_events._next_candle_wait_payload",
        lambda timeframe, buffer_seconds, now_utc: {
            "timeframe": timeframe,
            "buffer_seconds": buffer_seconds,
            "sleep_seconds": 1.0,
            "started_at_utc": now_utc.isoformat(),
            "next_candle_close_utc": (now_utc + timedelta(seconds=1)).isoformat(),
            "next_candle_close_server": "2026-03-15T12:00:01",
            "server_timezone": "UTC",
        },
    )
    started = datetime(2026, 3, 15, 12, 0, 0, tzinfo=timezone.utc)
    gateway = SequenceGateway(
        orders_seq=[
            [],
            [],
            [
                {
                    "ticket": 123,
                    "symbol": "EURUSD",
                    "type": "buy",
                    "time_setup": int(started.timestamp()) + 2,
                }
            ],
        ],
        history_orders_seq=[[], [], []],
    )
    clock = OversleepClock(started, extra_sleep_seconds=0.5)

    result = run_wait_event(
        WaitEventRequest(
            watch_for=[{"type": "order_created", "symbol": "EURUSD"}],
            end_on=[{"type": "candle_close", "timeframe": "M1", "buffer_seconds": 0.0}],
            poll_interval_seconds=10.0,
            max_wait_seconds=30.0,
        ),
        gateway=gateway,
        sleep_impl=clock.sleep,
        monotonic_impl=clock.monotonic,
        now_utc_impl=clock.now_utc,
    )

    assert result["status"] == "boundary_reached"
    assert result["matched_event"] is None
    assert result["boundary_event"]["type"] == "candle_close"

def test_run_wait_event_waits_across_pytz_dst_gap(monkeypatch) -> None:
    pytz = pytest.importorskip("pytz")
    from mtdata.core.trading import time

    monkeypatch.setattr(time.mt5_config, "get_server_tz", lambda: pytz.timezone("Europe/Nicosia"))
    monkeypatch.setattr(time.mt5_config, "get_time_offset_seconds", lambda: 7200)
    monkeypatch.setattr(time.mt5_config, "server_tz_name", "Europe/Nicosia")

    gateway = SequenceGateway(orders_seq=[[], [], [], []])
    clock = FakeClock(datetime(2026, 3, 29, 0, 54, 0, tzinfo=timezone.utc))

    result = run_wait_event(
        WaitEventRequest(
            watch_for=[{"type": "order_created", "symbol": "BTCUSD"}],
            end_on=[{"type": "candle_close", "timeframe": "M15", "buffer_seconds": 1.0}],
            poll_interval_seconds=120.0,
            max_wait_seconds=600.0,
        ),
        gateway=gateway,
        sleep_impl=clock.sleep,
        monotonic_impl=clock.monotonic,
        now_utc_impl=clock.now_utc,
    )

    assert result["status"] == "boundary_reached"
    assert result["boundary_event"]["type"] == "candle_close"
    assert result["boundary_event"]["next_candle_close_utc"] == "2026-03-29T01:00:00+00:00"
    assert result["elapsed_seconds"] == 361.0
    assert result["polls"] > 1

def test_run_wait_event_returns_error_when_tick_retention_cap_is_exceeded(monkeypatch) -> None:
    monkeypatch.setattr(wait_events_mod, "_MARKET_TICK_RETENTION_MAX_TICKS", 4)

    started = datetime(2026, 3, 15, 12, 0, 0, tzinfo=timezone.utc)
    base_epoch = int(started.timestamp()) - 3
    ticks = []
    for idx in range(5):
        epoch = base_epoch + idx
        mid = 100.0 + idx * 0.01
        ticks.append(
            {
                "time": epoch,
                "time_msc": epoch * 1000,
                "bid": mid - 0.0005,
                "ask": mid + 0.0005,
                "last": mid,
            }
        )
    gateway = SequenceGateway(ticks_by_symbol={"EURUSD": ticks})
    clock = FakeClock(started)

    result = run_wait_event(
        WaitEventRequest(
            watch_for=[
                {
                    "type": "price_touch_level",
                    "symbol": "EURUSD",
                    "level": 200.0,
                    "price_source": "mid",
                    "direction": "up",
                    "tolerance": 0.01,
                }
            ],
            poll_interval_seconds=1.0,
            max_wait_seconds=5.0,
        ),
        gateway=gateway,
        sleep_impl=clock.sleep,
        monotonic_impl=clock.monotonic,
        now_utc_impl=clock.now_utc,
    )

    assert "error" in result
    assert "tick retention" in result["error"]
    assert "EURUSD" in result["error"]
    assert "5 retained ticks > 4" in result["error"]
    assert result["error_code"] == "WAIT_EVENT_TICK_RETENTION_CAP"
    assert result["diagnostics"]["retention_guardrail"]["symbol"] == "EURUSD"
    assert result["diagnostics"]["retention_guardrail"]["retained_tick_count"] == 5
    assert result["diagnostics"]["retention_guardrail"]["retention_cap_ticks"] == 4
    assert result["diagnostics"]["retention_guardrail"]["first_retained_epoch"] == float(base_epoch)
    assert result["diagnostics"]["retention_guardrail"]["last_retained_epoch"] == float(base_epoch + 4)

def test_run_wait_event_returns_connection_error_when_gateway_disconnects_mid_loop() -> None:
    gateway = DisconnectingGateway(
        positions_seq=[[]],
        history_deals_seq=[[]],
    )
    clock = FakeClock(datetime(2026, 3, 15, 12, 0, 0, tzinfo=timezone.utc))

    result = run_wait_event(
        WaitEventRequest(
            watch_for=[{"type": "position_closed", "symbol": "BTCUSD"}],
            poll_interval_seconds=1.0,
            max_wait_seconds=5.0,
        ),
        gateway=gateway,
        sleep_impl=clock.sleep,
        monotonic_impl=clock.monotonic,
        now_utc_impl=clock.now_utc,
    )

    assert "error" in result
    assert "lost connection" in result["error"]
