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
    ) -> None:
        self.orders_seq = list(orders_seq or [[]])
        self.positions_seq = list(positions_seq or [[]])
        self.history_orders_seq = list(history_orders_seq or [[]])
        self.history_deals_seq = list(history_deals_seq or [[]])
        self.ticks_by_symbol = dict(ticks_by_symbol or {})
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
                "end_on": list(request.end_on or []),
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
    monkeypatch.setattr(core_data, "get_mt5_gateway", lambda ensure_connection_impl=None: object())
    monkeypatch.setattr(
        core_data,
        "_build_default_wait_event_watchers",
        lambda instrument, timeframe, watch_tick_count_spike: [
            {"type": "position_opened", "symbol": instrument},
            {"type": "tick_count_spike", "symbol": instrument},
            {"type": "price_touch_level", "symbol": instrument, "level": 100.0},
        ] if watch_tick_count_spike else [
            {"type": "position_opened", "symbol": instrument},
            {"type": "price_touch_level", "symbol": instrument, "level": 100.0},
        ],
    )

    sig = inspect.signature(core_data.wait_event)
    assert tuple(sig.parameters.keys()) == (
        "symbol",
        "instrument",
        "timeframe",
        "watch_tick_count_spike",
        "watch_for",
        "end_on",
        "verbose",
    )

    raw = getattr(core_data.wait_event, "__wrapped__", core_data.wait_event)
    result = raw("BTCUSD", "M1")

    assert result["success"] is True
    assert result["symbol"] == "BTCUSD"
    assert result["timeframe"] == "M1"
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

    without_tick_count = raw("BTCUSD", "M1", False)
    assert "criteria" not in without_tick_count

    explicit = raw(
        "BTCUSD",
        "M1",
        True,
        [{"type": "price_touch_level", "symbol": "BTCUSD", "level": 100.0}],
        [{"type": "candle_close", "timeframe": "M5"}],
        True,
    )
    assert [item.type for item in explicit["criteria"]["watch_for"]] == ["price_touch_level"]
    assert [item.type for item in explicit["criteria"]["end_on"]] == ["candle_close"]
    assert explicit["criteria"]["watch_for_inferred"] is False
    assert explicit["criteria"]["end_on_inferred"] is False
    assert explicit["boundary_event"]["buffer_seconds"] == 1.0
    assert explicit["matched"] is False
    assert explicit["event"] is None
    assert explicit["started_at_utc"] == "2026-04-06T02:00:29.017205+00:00"


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
    monkeypatch.setattr(core_data, "get_mt5_gateway", lambda ensure_connection_impl=None: object())

    raw = getattr(core_data.wait_event, "__wrapped__", core_data.wait_event)
    result = raw(
        "BTCUSD",
        "M1",
        True,
        [{"type": "price_touch_level", "symbol": "BTCUSD", "level": 100.0}],
        None,
        False,
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
    monkeypatch.setattr(core_data, "get_mt5_gateway", lambda ensure_connection_impl=None: object())

    raw = getattr(core_data.wait_event, "__wrapped__", core_data.wait_event)
    result = raw(
        "EURUSD",
        "M1",
        True,
        [{"type": "order_created", "symbol": "EURUSD"}],
        None,
        False,
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


def test_support_resistance_watchers_use_compact_levels(monkeypatch) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        core_data,
        "support_resistance_levels",
        lambda symbol, timeframe="auto", detail="compact": (
            captured.update({
                "symbol": symbol,
                "timeframe": timeframe,
                "detail": detail,
            })
            or {
                "success": True,
                "levels": [
                    {"type": "support", "value": 99.5},
                    {"type": "resistance", "value": 101.0},
                ],
            }
        ),
    )

    watchers = core_data._support_resistance_watchers(
        instrument="BTCUSD",
        timeframe="M15",
    )

    assert watchers == [
        {"type": "price_touch_level", "symbol": "BTCUSD", "level": 99.5, "direction": "either"},
        {"type": "price_break_level", "symbol": "BTCUSD", "level": 99.5, "direction": "down"},
        {"type": "price_touch_level", "symbol": "BTCUSD", "level": 101.0, "direction": "either"},
        {"type": "price_break_level", "symbol": "BTCUSD", "level": 101.0, "direction": "up"},
    ]
    assert captured == {"symbol": "BTCUSD", "timeframe": "M15", "detail": "compact"}


def test_support_resistance_watchers_ignore_non_finite_levels(monkeypatch) -> None:
    monkeypatch.setattr(
        core_data,
        "support_resistance_levels",
        lambda symbol, timeframe="auto", detail="compact": {
            "success": True,
            "levels": [
                {"type": "support", "value": float("inf")},
                {"type": "resistance", "value": 101.0},
            ],
        },
    )

    watchers = core_data._support_resistance_watchers(
        instrument="BTCUSD",
        timeframe="H1",
    )

    assert watchers == [
        {"type": "price_touch_level", "symbol": "BTCUSD", "level": 101.0, "direction": "either"},
        {"type": "price_break_level", "symbol": "BTCUSD", "level": 101.0, "direction": "up"},
    ]


def test_pivot_zone_watchers_use_adjacent_pivot_bands(monkeypatch) -> None:
    monkeypatch.setattr(
        core_data,
        "pivot_compute_points",
        lambda symbol, timeframe="D1": {
            "success": True,
            "levels": [
                {"level": "S1", "traditional": 99.0, "fibonacci": 99.0},
                {"level": "PP", "traditional": 100.0, "fibonacci": 100.0},
                {"level": "R1", "traditional": 101.0, "fibonacci": 101.0},
            ],
        },
    )

    watchers = core_data._pivot_zone_watchers(instrument="BTCUSD", timeframe="M15")

    assert watchers == [
        {"type": "price_enter_zone", "symbol": "BTCUSD", "lower": 99.0, "upper": 100.0, "direction": "either"},
        {"type": "price_enter_zone", "symbol": "BTCUSD", "lower": 100.0, "upper": 101.0, "direction": "either"},
    ]


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


def test_run_wait_event_matches_new_order() -> None:
    gateway = SequenceGateway(
        orders_seq=[
            [],
            [{"ticket": 123, "symbol": "EURUSD", "type": "buy"}],
        ],
        ticks_by_symbol={
            "EURUSD": [
                {
                    "time": 1_773_942_001.0,
                    "time_msc": 1_773_942_001_000,
                    "bid": 1.101,
                    "ask": 1.1012,
                    "last": 1.1011,
                }
            ]
        },
    )
    clock = FakeClock(datetime(2026, 3, 15, 12, 0, 0, tzinfo=timezone.utc))

    result = run_wait_event(
        WaitEventRequest(
            watch_for=[{"type": "order_created", "symbol": "EURUSD"}],
            poll_interval_seconds=1.0,
            max_wait_seconds=10.0,
        ),
        gateway=gateway,
        sleep_impl=clock.sleep,
        monotonic_impl=clock.monotonic,
        now_utc_impl=clock.now_utc,
    )

    assert result["status"] == "matched"
    assert result["symbol"] == "EURUSD"
    assert result["ticket"] == 123
    assert result["order_ticket"] == 123
    assert result["matched_event"]["type"] == "order_created"
    assert result["matched_event"]["symbol"] == "EURUSD"
    assert result["matched_event"]["ticket"] == 123
    assert result["matched_event"]["order_ticket"] == 123
    assert result["matched_event"]["observed"]["ticket"] == 123
    assert result["bid"] == 1.101
    assert result["ask"] == 1.1012


def test_run_wait_event_matches_short_lived_order_from_history() -> None:
    started = datetime(2026, 3, 15, 12, 0, 0, tzinfo=timezone.utc)
    gateway = SequenceGateway(
        orders_seq=[[], [], []],
        history_orders_seq=[
            [],
            [],
            [
                {
                    "ticket": 7001,
                    "symbol": "EURUSD",
                    "type": "buy_limit",
                    "state": 4,
                    "time_setup": int(started.timestamp()) + 1,
                }
            ],
        ],
    )
    clock = FakeClock(started)

    result = run_wait_event(
        WaitEventRequest(
            watch_for=[{"type": "order_created", "symbol": "EURUSD"}],
            poll_interval_seconds=1.0,
            max_wait_seconds=5.0,
        ),
        gateway=gateway,
        sleep_impl=clock.sleep,
        monotonic_impl=clock.monotonic,
        now_utc_impl=clock.now_utc,
    )

    assert result["status"] == "matched"
    assert result["matched_event"]["type"] == "order_created"
    assert result["matched_event"]["observed"]["ticket"] == 7001


def test_run_wait_event_accumulates_partial_order_fills_until_target_volume() -> None:
    started = datetime(2026, 3, 15, 12, 0, 0, tzinfo=timezone.utc)
    gateway = SequenceGateway(
        orders_seq=[
            [
                {
                    "ticket": 7001,
                    "symbol": "EURUSD",
                    "type": "buy_limit",
                    "volume_initial": 1.0,
                    "volume_current": 1.0,
                }
            ],
            [
                {
                    "ticket": 7001,
                    "symbol": "EURUSD",
                    "type": "buy_limit",
                    "volume_initial": 1.0,
                    "volume_current": 0.6,
                }
            ],
            [],
        ],
        history_deals_seq=[
            [],
            [
                {
                    "ticket": 3001,
                    "order": 7001,
                    "symbol": "EURUSD",
                    "entry": 0,
                    "type": "buy",
                    "volume": 0.4,
                    "time": int(started.timestamp()) + 1,
                }
            ],
            [
                {
                    "ticket": 3002,
                    "order": 7001,
                    "symbol": "EURUSD",
                    "entry": 0,
                    "type": "buy",
                    "volume": 0.6,
                    "time": int(started.timestamp()) + 2,
                }
            ],
        ],
    )
    clock = FakeClock(started)

    result = run_wait_event(
        WaitEventRequest(
            watch_for=[{"type": "order_filled", "symbol": "EURUSD", "order_ticket": 7001}],
            poll_interval_seconds=1.0,
            max_wait_seconds=5.0,
        ),
        gateway=gateway,
        sleep_impl=clock.sleep,
        monotonic_impl=clock.monotonic,
        now_utc_impl=clock.now_utc,
    )

    assert result["status"] == "matched"
    assert result["polls"] == 2
    assert result["order_ticket"] == 7001
    assert result["matched_event"]["type"] == "order_filled"
    assert result["matched_event"]["order_ticket"] == 7001
    assert result["matched_event"]["observed"]["ticket"] == 3002
    assert result["matched_event"]["observed"]["order_ticket"] == 7001


def test_run_wait_event_order_filled_falls_back_when_order_volume_is_unknown() -> None:
    started = datetime(2026, 3, 15, 12, 0, 0, tzinfo=timezone.utc)
    gateway = SequenceGateway(
        orders_seq=[[], []],
        history_deals_seq=[
            [],
            [
                {
                    "ticket": 3001,
                    "order": 7001,
                    "symbol": "EURUSD",
                    "entry": 0,
                    "type": "buy",
                    "volume": 0.4,
                    "time": int(started.timestamp()) + 1,
                }
            ],
        ],
    )
    clock = FakeClock(started)

    result = run_wait_event(
        WaitEventRequest(
            watch_for=[{"type": "order_filled", "symbol": "EURUSD", "order_ticket": 7001}],
            poll_interval_seconds=1.0,
            max_wait_seconds=5.0,
        ),
        gateway=gateway,
        sleep_impl=clock.sleep,
        monotonic_impl=clock.monotonic,
        now_utc_impl=clock.now_utc,
    )

    assert result["status"] == "matched"
    assert result["polls"] == 1
    assert result["matched_event"]["type"] == "order_filled"
    assert result["matched_event"]["observed"]["ticket"] == 3001


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


def test_collect_new_account_history_rows_keeps_same_second_coarse_rows() -> None:
    started = datetime(2026, 3, 15, 12, 0, 0, 500000, tzinfo=timezone.utc)

    rows = wait_events_mod._collect_new_account_history_rows(
        fetch_impl=lambda dt_from, dt_to: [
            {
                "ticket": 7001,
                "symbol": "EURUSD",
                "state": 4,
                "type": "buy_limit",
                "time_setup": int(started.timestamp()),
            }
        ],
        started_at_utc=started,
        observed_at_utc=started + timedelta(seconds=1),
        state={},
        row_kind="order",
        label="order history",
    )

    assert isinstance(rows, list)
    assert len(rows) == 1
    assert rows[0]["ticket"] == 7001


def test_collect_new_account_history_rows_advances_poll_cursor(monkeypatch) -> None:
    monkeypatch.setattr(
        wait_events_mod,
        "_to_server_naive_dt",
        lambda dt: wait_events_mod._normalize_utc_datetime(dt).replace(tzinfo=None),
    )

    started = datetime(2026, 3, 15, 12, 0, 0, 500000, tzinfo=timezone.utc)
    state = {"cursor_from_utc": started}
    calls = []

    def fetch_impl(dt_from, dt_to):
        calls.append((dt_from, dt_to))
        return []

    first_observed = started + timedelta(seconds=0.6)
    second_observed = started + timedelta(seconds=1.2)

    first_rows = wait_events_mod._collect_new_account_history_rows(
        fetch_impl=fetch_impl,
        started_at_utc=started,
        observed_at_utc=first_observed,
        state=state,
        row_kind="order",
        label="order history",
    )
    second_rows = wait_events_mod._collect_new_account_history_rows(
        fetch_impl=fetch_impl,
        started_at_utc=started,
        observed_at_utc=second_observed,
        state=state,
        row_kind="order",
        label="order history",
    )

    assert first_rows == []
    assert second_rows == []
    assert calls == [
        (
            datetime(2026, 3, 15, 12, 0, 0),
            datetime(2026, 3, 15, 12, 0, 1, 100000),
        ),
        (
            datetime(2026, 3, 15, 12, 0, 1),
            datetime(2026, 3, 15, 12, 0, 1, 700000),
        ),
    ]


def test_seed_account_history_keys_converts_window_to_server_naive(monkeypatch) -> None:
    calls = []

    monkeypatch.setattr(
        wait_events_mod,
        "_to_server_naive_dt",
        lambda dt: (dt - timedelta(hours=2)).replace(tzinfo=None),
    )

    def fetch_impl(dt_from, dt_to):
        calls.append((dt_from, dt_to))
        return []

    started = datetime(2026, 3, 15, 12, 0, 0, tzinfo=timezone.utc)
    seen = wait_events_mod._seed_account_history_keys(
        fetch_impl=fetch_impl,
        started_at_utc=started,
        row_kind="deal",
        label="deal history",
    )

    assert seen == set()
    assert calls == [
        (
            datetime(2026, 3, 15, 9, 59, 55),
            datetime(2026, 3, 15, 10, 0, 0),
        )
    ]


def test_fetch_market_ticks_range_converts_window_to_server_naive(monkeypatch) -> None:
    captured = {}

    monkeypatch.setattr(
        wait_events_mod,
        "_to_server_naive_dt",
        lambda dt: (dt - timedelta(hours=3)).replace(tzinfo=None),
    )

    class Gateway:
        COPY_TICKS_ALL = 0

        def symbol_select(self, symbol, visible=True):
            return True

        def copy_ticks_range(self, symbol, dt_from, dt_to, flags):
            captured["args"] = (symbol, dt_from, dt_to, flags)
            return []

    out = wait_events_mod._fetch_market_ticks_range(
        gateway=Gateway(),
        symbol="EURUSD",
        from_dt_utc=datetime(2026, 3, 15, 12, 0, 0, tzinfo=timezone.utc),
        to_dt_utc=datetime(2026, 3, 15, 12, 5, 0, tzinfo=timezone.utc),
    )

    assert out == []
    assert captured["args"] == (
        "EURUSD",
        datetime(2026, 3, 15, 9, 0, 0),
        datetime(2026, 3, 15, 9, 5, 0),
        0,
    )


def test_normalize_tick_rows_keeps_epoch_and_time_msc_on_same_utc_basis(monkeypatch) -> None:
    monkeypatch.setattr(wait_events_mod, "_mt5_epoch_to_utc", lambda value: float(value) - 7200.0)

    rows = [
        {
            "time": 7201.0,
            "time_msc": 7201500,
            "bid": 1.1,
            "ask": 1.2,
            "last": 1.15,
            "volume": 0.0,
            "volume_real": 0.0,
            "flags": 0,
        }
    ]

    normalized = wait_events_mod._normalize_tick_rows(rows)

    assert normalized == [
        {
            "epoch": 1.0,
            "time_msc": 1500,
            "bid": 1.1,
            "ask": 1.2,
            "last": 1.15,
            "volume": 0.0,
            "volume_real": 0.0,
            "flags": 0,
            "key": (1500, 1.1, 1.2, 1.15, 0.0, 0.0, 0),
        }
    ]


def test_merge_market_ticks_dedupes_rows_with_missing_volume_fields() -> None:
    existing = wait_events_mod._normalize_tick_rows(
        [
            {
                "time": 100.0,
                "time_msc": 100000,
                "bid": 1.1,
                "ask": 1.2,
                "last": 1.15,
            }
        ]
    )
    incoming = wait_events_mod._normalize_tick_rows(
        [
            {
                "time": 100.0,
                "time_msc": 100000,
                "bid": 1.1,
                "ask": 1.2,
                "last": 1.15,
            }
        ]
    )

    merged = wait_events_mod._merge_market_ticks(existing, incoming)

    assert len(merged) == 1


def test_merge_market_ticks_keeps_older_existing_keys_in_seen_set(monkeypatch) -> None:
    monkeypatch.setattr(wait_events_mod, "_MARKET_BUFFER_EXTRA_TICKS", 1)

    existing = wait_events_mod._normalize_tick_rows(
        [
            {
                "time": 100.0,
                "time_msc": 100000,
                "bid": 1.1,
                "ask": 1.2,
                "last": 1.15,
            },
            {
                "time": 101.0,
                "time_msc": 101000,
                "bid": 1.11,
                "ask": 1.21,
                "last": 1.16,
            },
            {
                "time": 102.0,
                "time_msc": 102000,
                "bid": 1.12,
                "ask": 1.22,
                "last": 1.17,
            },
        ]
    )
    incoming = wait_events_mod._normalize_tick_rows(
        [
            {
                "time": 100.0,
                "time_msc": 100000,
                "bid": 1.1,
                "ask": 1.2,
                "last": 1.15,
            },
            {
                "time": 103.0,
                "time_msc": 103000,
                "bid": 1.13,
                "ask": 1.23,
                "last": 1.18,
            },
        ]
    )

    merged = wait_events_mod._merge_market_ticks(existing, incoming)

    assert [tick["time_msc"] for tick in merged] == [100000, 101000, 102000, 103000]


def test_trim_market_ticks_keeps_rows_at_or_after_time_cutoff(monkeypatch) -> None:
    monkeypatch.setattr(wait_events_mod, "_MARKET_BUFFER_EXTRA_TICKS", 0)
    monkeypatch.setattr(wait_events_mod, "_MARKET_ESTIMATED_SECONDS_PER_TICK", 2.0)

    observed_at = datetime(2026, 3, 15, 12, 0, 10, tzinfo=timezone.utc)
    base_epoch = int(observed_at.timestamp()) - 9
    ticks = [{"epoch": float(base_epoch + idx)} for idx in range(10)]

    trimmed = wait_events_mod._trim_market_ticks(
        ticks=ticks,
        specs=[{"required_history_seconds": 3.0, "required_tick_count": 0}],
        observed_at_utc=observed_at,
    )

    assert [tick["epoch"] for tick in trimmed] == [float(base_epoch + idx) for idx in range(4, 10)]


def test_trim_market_ticks_still_honors_keep_tick_floor(monkeypatch) -> None:
    monkeypatch.setattr(wait_events_mod, "_MARKET_BUFFER_EXTRA_TICKS", 1)
    monkeypatch.setattr(wait_events_mod, "_MARKET_ESTIMATED_SECONDS_PER_TICK", 2.0)

    observed_at = datetime(2026, 3, 15, 12, 0, 10, tzinfo=timezone.utc)
    base_epoch = int(observed_at.timestamp()) - 9
    ticks = [{"epoch": float(base_epoch + idx)} for idx in range(10)]

    trimmed = wait_events_mod._trim_market_ticks(
        ticks=ticks,
        specs=[{"required_history_seconds": 1.0, "required_tick_count": 4}],
        observed_at_utc=observed_at,
    )

    assert [tick["epoch"] for tick in trimmed] == [float(base_epoch + idx) for idx in range(5, 10)]


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
    gateway = SequenceGateway(orders_seq=[[], [], []])
    clock = FakeClock(datetime(2026, 3, 15, 12, 0, 0, tzinfo=timezone.utc))

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


def test_run_wait_event_matches_adaptive_price_change() -> None:
    clock = FakeClock(datetime(2026, 3, 15, 12, 0, 0, tzinfo=timezone.utc))
    base_epoch = int(clock.now_utc().timestamp()) - 26
    ticks = []
    mid = 100.0
    for idx in range(26):
        if idx < 20:
            mid += 0.03
        elif idx < 25:
            mid += 0.01
        else:
            mid += 3.0
        ticks.append(
            {
                "time": base_epoch + idx,
                "time_msc": (base_epoch + idx) * 1000,
                "bid": mid - 0.0005,
                "ask": mid + 0.0005,
                "last": mid,
                "volume": 1.0,
            }
        )
    gateway = SequenceGateway(ticks_by_symbol={"EURUSD": ticks})

    result = run_wait_event(
        WaitEventRequest(
            watch_for=[
                {
                    "type": "price_change",
                    "symbol": "EURUSD",
                    "window": {"kind": "ticks", "value": 5},
                    "baseline_window": {"kind": "ticks", "value": 20},
                    "threshold_mode": "ratio_to_baseline",
                    "threshold_value": 4.0,
                }
            ],
            poll_interval_seconds=0.5,
            max_wait_seconds=5.0,
        ),
        gateway=gateway,
        sleep_impl=clock.sleep,
        monotonic_impl=clock.monotonic,
        now_utc_impl=clock.now_utc,
    )

    assert result["status"] == "matched"
    assert result["matched_event"]["type"] == "price_change"
    assert result["matched_event"]["observed"]["ratio"] > 4.0


def test_run_wait_event_matches_volume_spike() -> None:
    clock = FakeClock(datetime(2026, 3, 15, 12, 0, 0, tzinfo=timezone.utc))
    ticks = []
    base_epoch = int(clock.now_utc().timestamp()) - 20 * 30
    for idx in range(20):
        ticks.append(
            {
                "time": base_epoch + idx * 30,
                "time_msc": (base_epoch + idx * 30) * 1000,
                "bid": 100.0 + idx * 0.01,
                "ask": 100.01 + idx * 0.01,
                "last": 100.005 + idx * 0.01,
                "volume": 2.0 if idx < 17 else 40.0,
            }
        )
    gateway = SequenceGateway(ticks_by_symbol={"EURUSD": ticks})

    result = run_wait_event(
        WaitEventRequest(
            watch_for=[
                {
                    "type": "volume_spike",
                    "symbol": "EURUSD",
                    "window": {"kind": "ticks", "value": 3},
                    "baseline_window": {"kind": "ticks", "value": 12},
                    "source": "volume",
                    "threshold_mode": "ratio_to_baseline",
                    "threshold_value": 4.0,
                }
            ],
            poll_interval_seconds=0.5,
            max_wait_seconds=5.0,
        ),
        gateway=gateway,
        sleep_impl=clock.sleep,
        monotonic_impl=clock.monotonic,
        now_utc_impl=clock.now_utc,
    )

    assert result["status"] == "matched"
    assert result["matched_event"]["type"] == "volume_spike"
    assert result["matched_event"]["observed"]["ratio"] > 4.0


def test_run_wait_event_matches_tick_count_spike() -> None:
    clock = FakeClock(datetime(2026, 3, 15, 12, 0, 0, tzinfo=timezone.utc))
    now_epoch = int(clock.now_utc().timestamp())
    ticks = []

    for offset_seconds in (300, 270, 240, 210, 180, 150, 120, 90):
        ticks.append(
            {
                "time": now_epoch - offset_seconds,
                "time_msc": (now_epoch - offset_seconds) * 1000,
                "bid": 100.0,
                "ask": 100.01,
                "last": 100.005,
            }
        )

    for offset_seconds in (55, 50, 45, 40, 35, 30, 25, 20, 15, 10, 5, 0):
        ticks.append(
            {
                "time": now_epoch - offset_seconds,
                "time_msc": (now_epoch - offset_seconds) * 1000,
                "bid": 100.1,
                "ask": 100.11,
                "last": 100.105,
            }
        )

    gateway = SequenceGateway(ticks_by_symbol={"EURUSD": ticks})

    result = run_wait_event(
        WaitEventRequest(
            watch_for=[
                {
                    "type": "tick_count_spike",
                    "symbol": "EURUSD",
                    "window": {"kind": "minutes", "value": 1},
                    "baseline_window": {"kind": "minutes", "value": 4},
                    "threshold_mode": "ratio_to_baseline",
                    "threshold_value": 2.0,
                }
            ],
            poll_interval_seconds=0.5,
            max_wait_seconds=5.0,
        ),
        gateway=gateway,
        sleep_impl=clock.sleep,
        monotonic_impl=clock.monotonic,
        now_utc_impl=clock.now_utc,
    )

    assert result["status"] == "matched"
    assert result["matched_event"]["type"] == "tick_count_spike"
    assert result["matched_event"]["observed"]["volume_source"] == "tick_count"
    assert result["matched_event"]["observed"]["ratio"] > 2.0


def test_run_wait_event_matches_spread_spike() -> None:
    clock = FakeClock(datetime(2026, 3, 15, 12, 0, 0, tzinfo=timezone.utc))
    base_epoch = int(clock.now_utc().timestamp()) - 20
    ticks = []
    for idx in range(20):
        bid = 100.0 + idx * 0.01
        spread = 0.01 if idx < 17 else 0.20
        ticks.append(
            {
                "time": base_epoch + idx,
                "time_msc": (base_epoch + idx) * 1000,
                "bid": bid,
                "ask": bid + spread,
                "last": bid + spread / 2.0,
            }
        )
    gateway = SequenceGateway(ticks_by_symbol={"EURUSD": ticks})

    result = run_wait_event(
        WaitEventRequest(
            watch_for=[
                {
                    "type": "spread_spike",
                    "symbol": "EURUSD",
                    "window": {"kind": "ticks", "value": 3},
                    "baseline_window": {"kind": "ticks", "value": 12},
                    "threshold_mode": "ratio_to_baseline",
                    "threshold_value": 4.0,
                }
            ],
            poll_interval_seconds=0.5,
            max_wait_seconds=5.0,
        ),
        gateway=gateway,
        sleep_impl=clock.sleep,
        monotonic_impl=clock.monotonic,
        now_utc_impl=clock.now_utc,
    )

    assert result["status"] == "matched"
    assert result["matched_event"]["type"] == "spread_spike"
    assert result["matched_event"]["observed"]["ratio"] > 4.0


def test_run_wait_event_matches_tick_count_drought() -> None:
    clock = FakeClock(datetime(2026, 3, 15, 12, 0, 0, tzinfo=timezone.utc))
    now_epoch = int(clock.now_utc().timestamp())
    ticks = []
    for offset_seconds in (300, 270, 240, 210, 180, 150, 120, 90, 75, 70):
        ticks.append(
            {
                "time": now_epoch - offset_seconds,
                "time_msc": (now_epoch - offset_seconds) * 1000,
                "bid": 100.0,
                "ask": 100.02,
                "last": 100.01,
            }
        )
    ticks.append(
        {
            "time": now_epoch - 5,
            "time_msc": (now_epoch - 5) * 1000,
            "bid": 100.1,
            "ask": 100.12,
            "last": 100.11,
        }
    )
    gateway = SequenceGateway(ticks_by_symbol={"EURUSD": ticks})

    result = run_wait_event(
        WaitEventRequest(
            watch_for=[
                {
                    "type": "tick_count_drought",
                    "symbol": "EURUSD",
                    "window": {"kind": "minutes", "value": 1},
                    "baseline_window": {"kind": "minutes", "value": 4},
                    "threshold_mode": "ratio_to_baseline",
                    "threshold_value": 0.5,
                }
            ],
            poll_interval_seconds=0.5,
            max_wait_seconds=5.0,
        ),
        gateway=gateway,
        sleep_impl=clock.sleep,
        monotonic_impl=clock.monotonic,
        now_utc_impl=clock.now_utc,
    )

    assert result["status"] == "matched"
    assert result["matched_event"]["type"] == "tick_count_drought"
    assert result["matched_event"]["observed"]["ratio"] <= 0.5


def test_run_wait_event_tick_count_drought_handles_empty_ticks() -> None:
    clock = FakeClock(datetime(2026, 3, 15, 12, 0, 0, tzinfo=timezone.utc))
    gateway = SequenceGateway(ticks_by_symbol={"EURUSD": []})

    result = run_wait_event(
        WaitEventRequest(
            watch_for=[
                {
                    "type": "tick_count_drought",
                    "symbol": "EURUSD",
                    "window": {"kind": "minutes", "value": 1},
                    "baseline_window": {"kind": "minutes", "value": 4},
                    "threshold_mode": "ratio_to_baseline",
                    "threshold_value": 0.5,
                }
            ],
            poll_interval_seconds=0.5,
            max_wait_seconds=0.0,
        ),
        gateway=gateway,
        sleep_impl=clock.sleep,
        monotonic_impl=clock.monotonic,
        now_utc_impl=clock.now_utc,
    )

    assert result["status"] == "timeout"
    assert result["matched_event"] is None


def test_run_wait_event_matches_range_expansion() -> None:
    clock = FakeClock(datetime(2026, 3, 15, 12, 0, 0, tzinfo=timezone.utc))
    base_epoch = int(clock.now_utc().timestamp()) - 20
    ticks = []
    mid = 100.0
    for idx in range(20):
        if idx < 16:
            mid += 0.02 if idx % 2 == 0 else -0.01
        else:
            mid = 100.0 + [0.0, 0.1, 2.0, -1.5][idx - 16]
        ticks.append(
            {
                "time": base_epoch + idx,
                "time_msc": (base_epoch + idx) * 1000,
                "bid": mid - 0.01,
                "ask": mid + 0.01,
                "last": mid,
            }
        )
    gateway = SequenceGateway(ticks_by_symbol={"EURUSD": ticks})

    result = run_wait_event(
        WaitEventRequest(
            watch_for=[
                {
                    "type": "range_expansion",
                    "symbol": "EURUSD",
                    "window": {"kind": "ticks", "value": 4},
                    "baseline_window": {"kind": "ticks", "value": 12},
                    "price_source": "mid",
                    "threshold_mode": "ratio_to_baseline",
                    "threshold_value": 5.0,
                }
            ],
            poll_interval_seconds=0.5,
            max_wait_seconds=5.0,
        ),
        gateway=gateway,
        sleep_impl=clock.sleep,
        monotonic_impl=clock.monotonic,
        now_utc_impl=clock.now_utc,
    )

    assert result["status"] == "matched"
    assert result["matched_event"]["type"] == "range_expansion"
    assert result["matched_event"]["observed"]["ratio"] > 5.0


def test_run_wait_event_matches_price_touch_level() -> None:
    clock = FakeClock(datetime(2026, 3, 15, 12, 0, 0, tzinfo=timezone.utc))
    base_epoch = int(clock.now_utc().timestamp()) - 2
    ticks = [
        {"time": base_epoch, "time_msc": base_epoch * 1000, "bid": 99.7, "ask": 99.9, "last": 99.8},
        {"time": base_epoch + 1, "time_msc": (base_epoch + 1) * 1000, "bid": 99.95, "ask": 100.05, "last": 100.0},
    ]
    gateway = SequenceGateway(ticks_by_symbol={"EURUSD": ticks})

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
            poll_interval_seconds=0.5,
            max_wait_seconds=5.0,
        ),
        gateway=gateway,
        sleep_impl=clock.sleep,
        monotonic_impl=clock.monotonic,
        now_utc_impl=clock.now_utc,
    )

    assert result["status"] == "matched"
    assert result["matched_event"]["type"] == "price_touch_level"


def test_run_wait_event_matches_price_touch_level_when_price_gaps_over_band() -> None:
    clock = FakeClock(datetime(2026, 3, 15, 12, 0, 0, tzinfo=timezone.utc))
    base_epoch = int(clock.now_utc().timestamp()) - 2
    ticks = [
        {"time": base_epoch, "time_msc": base_epoch * 1000, "bid": 99.95, "ask": 100.05, "last": 100.0},
        {"time": base_epoch + 1, "time_msc": (base_epoch + 1) * 1000, "bid": 100.95, "ask": 101.05, "last": 101.0},
    ]
    gateway = SequenceGateway(ticks_by_symbol={"EURUSD": ticks})

    result = run_wait_event(
        WaitEventRequest(
            watch_for=[
                {
                    "type": "price_touch_level",
                    "symbol": "EURUSD",
                    "level": 100.5,
                    "price_source": "mid",
                    "direction": "up",
                    "tolerance": 0.05,
                }
            ],
            poll_interval_seconds=0.5,
            max_wait_seconds=5.0,
        ),
        gateway=gateway,
        sleep_impl=clock.sleep,
        monotonic_impl=clock.monotonic,
        now_utc_impl=clock.now_utc,
    )

    assert result["status"] == "matched"
    assert result["matched_event"]["type"] == "price_touch_level"


def test_run_wait_event_matches_price_break_level() -> None:
    clock = FakeClock(datetime(2026, 3, 15, 12, 0, 0, tzinfo=timezone.utc))
    base_epoch = int(clock.now_utc().timestamp()) - 3
    ticks = [
        {"time": base_epoch, "time_msc": base_epoch * 1000, "bid": 99.8, "ask": 99.9, "last": 99.85},
        {"time": base_epoch + 1, "time_msc": (base_epoch + 1) * 1000, "bid": 100.1, "ask": 100.2, "last": 100.15},
        {"time": base_epoch + 2, "time_msc": (base_epoch + 2) * 1000, "bid": 100.2, "ask": 100.3, "last": 100.25},
    ]
    gateway = SequenceGateway(ticks_by_symbol={"EURUSD": ticks})

    result = run_wait_event(
        WaitEventRequest(
            watch_for=[
                {
                    "type": "price_break_level",
                    "symbol": "EURUSD",
                    "level": 100.0,
                    "price_source": "mid",
                    "direction": "up",
                    "confirm_ticks": 2,
                }
            ],
            poll_interval_seconds=0.5,
            max_wait_seconds=5.0,
        ),
        gateway=gateway,
        sleep_impl=clock.sleep,
        monotonic_impl=clock.monotonic,
        now_utc_impl=clock.now_utc,
    )

    assert result["status"] == "matched"
    assert result["matched_event"]["type"] == "price_break_level"


def test_run_wait_event_matches_price_enter_zone() -> None:
    clock = FakeClock(datetime(2026, 3, 15, 12, 0, 0, tzinfo=timezone.utc))
    base_epoch = int(clock.now_utc().timestamp()) - 2
    ticks = [
        {"time": base_epoch, "time_msc": base_epoch * 1000, "bid": 101.2, "ask": 101.4, "last": 101.3},
        {"time": base_epoch + 1, "time_msc": (base_epoch + 1) * 1000, "bid": 100.4, "ask": 100.6, "last": 100.5},
    ]
    gateway = SequenceGateway(ticks_by_symbol={"EURUSD": ticks})

    result = run_wait_event(
        WaitEventRequest(
            watch_for=[
                {
                    "type": "price_enter_zone",
                    "symbol": "EURUSD",
                    "lower": 100.0,
                    "upper": 101.0,
                    "price_source": "mid",
                    "direction": "down",
                }
            ],
            poll_interval_seconds=0.5,
            max_wait_seconds=5.0,
        ),
        gateway=gateway,
        sleep_impl=clock.sleep,
        monotonic_impl=clock.monotonic,
        now_utc_impl=clock.now_utc,
    )

    assert result["status"] == "matched"
    assert result["matched_event"]["type"] == "price_enter_zone"


def test_run_wait_event_matches_price_enter_zone_when_price_gaps_over_zone() -> None:
    clock = FakeClock(datetime(2026, 3, 15, 12, 0, 0, tzinfo=timezone.utc))
    base_epoch = int(clock.now_utc().timestamp()) - 2
    ticks = [
        {"time": base_epoch, "time_msc": base_epoch * 1000, "bid": 99.9, "ask": 100.1, "last": 100.0},
        {"time": base_epoch + 1, "time_msc": (base_epoch + 1) * 1000, "bid": 100.9, "ask": 101.1, "last": 101.0},
    ]
    gateway = SequenceGateway(ticks_by_symbol={"EURUSD": ticks})

    result = run_wait_event(
        WaitEventRequest(
            watch_for=[
                {
                    "type": "price_enter_zone",
                    "symbol": "EURUSD",
                    "lower": 100.4,
                    "upper": 100.6,
                    "price_source": "mid",
                    "direction": "up",
                }
            ],
            poll_interval_seconds=0.5,
            max_wait_seconds=5.0,
        ),
        gateway=gateway,
        sleep_impl=clock.sleep,
        monotonic_impl=clock.monotonic,
        now_utc_impl=clock.now_utc,
    )

    assert result["status"] == "matched"
    assert result["matched_event"]["type"] == "price_enter_zone"


def test_run_wait_event_matches_pending_near_fill() -> None:
    clock = FakeClock(datetime(2026, 3, 15, 12, 0, 0, tzinfo=timezone.utc))
    now_epoch = int(clock.now_utc().timestamp())
    gateway = SequenceGateway(
        orders_seq=[
            [{"ticket": 7001, "symbol": "EURUSD", "type": "buy_limit", "price_open": 100.0}],
            [{"ticket": 7001, "symbol": "EURUSD", "type": "buy_limit", "price_open": 100.0}],
        ],
        ticks_by_symbol={
            "EURUSD": [
                {
                    "time": now_epoch - 1,
                    "time_msc": (now_epoch - 1) * 1000,
                    "bid": 99.96,
                    "ask": 100.04,
                    "last": 100.0,
                }
            ]
        },
    )

    result = run_wait_event(
        WaitEventRequest(
            watch_for=[
                {
                    "type": "pending_near_fill",
                    "symbol": "EURUSD",
                    "distance": 0.05,
                }
            ],
            poll_interval_seconds=0.5,
            max_wait_seconds=5.0,
        ),
        gateway=gateway,
        sleep_impl=clock.sleep,
        monotonic_impl=clock.monotonic,
        now_utc_impl=clock.now_utc,
    )

    assert result["status"] == "matched"
    assert result["matched_event"]["type"] == "pending_near_fill"
    assert result["matched_event"]["observed"]["ticket"] == 7001


def test_run_wait_event_matches_stop_threat() -> None:
    clock = FakeClock(datetime(2026, 3, 15, 12, 0, 0, tzinfo=timezone.utc))
    now_epoch = int(clock.now_utc().timestamp())
    gateway = SequenceGateway(
        positions_seq=[
            [{"ticket": 9001, "symbol": "EURUSD", "type": "buy", "sl": 99.9, "price_current": 99.95}],
            [{"ticket": 9001, "symbol": "EURUSD", "type": "buy", "sl": 99.9, "price_current": 99.95}],
        ],
        ticks_by_symbol={
            "EURUSD": [
                {
                    "time": now_epoch - 1,
                    "time_msc": (now_epoch - 1) * 1000,
                    "bid": 99.94,
                    "ask": 99.96,
                    "last": 99.95,
                }
            ]
        },
    )

    result = run_wait_event(
        WaitEventRequest(
            watch_for=[
                {
                    "type": "stop_threat",
                    "symbol": "EURUSD",
                    "distance": 0.05,
                }
            ],
            poll_interval_seconds=0.5,
            max_wait_seconds=5.0,
        ),
        gateway=gateway,
        sleep_impl=clock.sleep,
        monotonic_impl=clock.monotonic,
        now_utc_impl=clock.now_utc,
    )

    assert result["status"] == "matched"
    assert result["matched_event"]["type"] == "stop_threat"
    assert result["matched_event"]["observed"]["ticket"] == 9001


def test_run_wait_event_matches_position_closed_from_history_out_by() -> None:
    gateway = SequenceGateway(
        positions_seq=[
            [{"ticket": 9001, "symbol": "BTCUSD", "type": "buy"}],
            [{"ticket": 9001, "symbol": "BTCUSD", "type": "buy"}],
        ],
        history_deals_seq=[
            [],
            [{"ticket": 3001, "position_id": 9001, "symbol": "BTCUSD", "entry": 3, "type": "sell"}],
        ],
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

    assert result["status"] == "matched"
    assert result["matched_event"]["type"] == "position_closed"
    assert result["matched_event"]["observed"]["position_ticket"] == 9001


def test_run_wait_event_ignores_replayed_preexisting_deal_history() -> None:
    clock = FakeClock(datetime(2026, 3, 15, 12, 0, 0, tzinfo=timezone.utc))
    gateway = ReplayHistoryGateway(
        positions_seq=[[], [], []],
        replay_deals=[
            {
                "ticket": 3001,
                "position_id": 9001,
                "symbol": "BTCUSD",
                "entry": 3,
                "type": "sell",
                "time": int(clock.now_utc().timestamp()),
            }
        ],
    )

    result = run_wait_event(
        WaitEventRequest(
            watch_for=[{"type": "position_closed", "symbol": "BTCUSD"}],
            poll_interval_seconds=0.1,
            max_wait_seconds=0.2,
        ),
        gateway=gateway,
        sleep_impl=clock.sleep,
        monotonic_impl=clock.monotonic,
        now_utc_impl=clock.now_utc,
    )

    assert result["status"] == "timeout"
    assert result["matched_event"] is None


def test_run_wait_event_ignores_replayed_preexisting_order_history() -> None:
    clock = FakeClock(datetime(2026, 3, 15, 12, 0, 0, tzinfo=timezone.utc))
    gateway = ReplayHistoryGateway(
        replay_orders=[
            {
                "ticket": 7001,
                "symbol": "EURUSD",
                "state": 4,
                "type": "buy",
                "time_setup": int(clock.now_utc().timestamp()),
            }
        ],
    )

    result = run_wait_event(
        WaitEventRequest(
            watch_for=[{"type": "order_cancelled", "symbol": "EURUSD"}],
            poll_interval_seconds=0.1,
            max_wait_seconds=0.2,
        ),
        gateway=gateway,
        sleep_impl=clock.sleep,
        monotonic_impl=clock.monotonic,
        now_utc_impl=clock.now_utc,
    )

    assert result["status"] == "timeout"
    assert result["matched_event"] is None


def test_run_wait_event_advances_history_poll_cursors_per_stream(monkeypatch) -> None:
    monkeypatch.setattr(
        wait_events_mod,
        "_to_server_naive_dt",
        lambda dt: wait_events_mod._normalize_utc_datetime(dt).replace(tzinfo=None),
    )

    clock = FakeClock(datetime(2026, 3, 15, 12, 0, 0, 500000, tzinfo=timezone.utc))
    gateway = TrackingHistoryWindowGateway(
        history_orders_seq=[[], [], [], []],
        history_deals_seq=[[], [], [], []],
    )

    result = run_wait_event(
        WaitEventRequest(
            watch_for=[
                {"type": "order_cancelled", "symbol": "EURUSD"},
                {"type": "order_filled", "symbol": "EURUSD"},
            ],
            poll_interval_seconds=0.6,
            max_wait_seconds=1.2,
        ),
        gateway=gateway,
        sleep_impl=clock.sleep,
        monotonic_impl=clock.monotonic,
        now_utc_impl=clock.now_utc,
    )

    assert result["status"] == "timeout"
    assert result["matched_event"] is None
    assert gateway.history_order_calls == [
        (
            datetime(2026, 3, 15, 11, 59, 55, 500000),
            datetime(2026, 3, 15, 12, 0, 0, 500000),
        ),
        (
            datetime(2026, 3, 15, 12, 0, 0),
            datetime(2026, 3, 15, 12, 0, 0, 500000),
        ),
        (
            datetime(2026, 3, 15, 12, 0),
            datetime(2026, 3, 15, 12, 0, 1, 100000),
        ),
        (
            datetime(2026, 3, 15, 12, 0, 1),
            datetime(2026, 3, 15, 12, 0, 1, 700000),
        ),
    ]
    assert gateway.history_deal_calls == gateway.history_order_calls


def test_run_wait_event_ignores_same_second_order_history_at_startup_watermark() -> None:
    clock = FakeClock(datetime(2026, 3, 15, 12, 0, 0, 500000, tzinfo=timezone.utc))
    gateway = SequenceGateway(
        history_orders_seq=[
            [
                {
                    "ticket": 7002,
                    "symbol": "EURUSD",
                    "state": 4,
                    "type": "buy",
                    "time_setup": int(clock.now_utc().timestamp()),
                }
            ],
            [
                {
                    "ticket": 7001,
                    "symbol": "EURUSD",
                    "state": 4,
                    "type": "buy",
                    "time_setup": int(clock.now_utc().timestamp()),
                }
            ],
        ]
    )

    result = run_wait_event(
        WaitEventRequest(
            watch_for=[{"type": "order_cancelled", "symbol": "EURUSD"}],
            poll_interval_seconds=0.1,
            max_wait_seconds=0.2,
        ),
        gateway=gateway,
        sleep_impl=clock.sleep,
        monotonic_impl=clock.monotonic,
        now_utc_impl=clock.now_utc,
    )

    assert result["status"] == "timeout"
    assert result["matched_event"] is None


def test_run_wait_event_ignores_same_second_deal_history_at_startup_watermark() -> None:
    clock = FakeClock(datetime(2026, 3, 15, 12, 0, 0, 500000, tzinfo=timezone.utc))
    gateway = SequenceGateway(
        positions_seq=[[], [], []],
        history_deals_seq=[
            [
                {
                    "ticket": 3002,
                    "position_id": 9002,
                    "symbol": "BTCUSD",
                    "entry": 3,
                    "type": "sell",
                    "time": int(clock.now_utc().timestamp()),
                }
            ],
            [
                {
                    "ticket": 3001,
                    "position_id": 9001,
                    "symbol": "BTCUSD",
                    "entry": 3,
                    "type": "sell",
                    "time": int(clock.now_utc().timestamp()),
                }
            ],
        ],
    )

    result = run_wait_event(
        WaitEventRequest(
            watch_for=[{"type": "position_closed", "symbol": "BTCUSD"}],
            poll_interval_seconds=0.1,
            max_wait_seconds=0.2,
        ),
        gateway=gateway,
        sleep_impl=clock.sleep,
        monotonic_impl=clock.monotonic,
        now_utc_impl=clock.now_utc,
    )

    assert result["status"] == "timeout"
    assert result["matched_event"] is None


def test_run_wait_event_matches_position_closed_when_position_disappears() -> None:
    gateway = SequenceGateway(
        positions_seq=[
            [{"ticket": 9002, "symbol": "BTCUSD", "type": "buy"}],
            [],
        ],
        history_deals_seq=[[], []],
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

    assert result["status"] == "matched"
    assert result["symbol"] == "BTCUSD"
    assert result["position_ticket"] == 9002
    assert result["matched_event"]["type"] == "position_closed"
    assert result["matched_event"]["symbol"] == "BTCUSD"
    assert result["matched_event"]["position_ticket"] == 9002
    assert "ticket" not in result["matched_event"]
    assert result["matched_event"]["observed"]["ticket"] is None
    assert result["matched_event"]["observed"]["position_ticket"] == 9002
    assert result["matched_event"]["observed"]["inferred"] is True
    assert result["matched_event"]["observed"]["source"] == "position_disappeared"


def test_exit_trigger_text_matching_ignores_non_hit_mentions() -> None:
    gateway = SequenceGateway()
    row = {"comment": "stop loss hit, no tp set", "reason": "manual"}

    assert wait_events_mod._is_exit_trigger(row, gateway=gateway, trigger="tp") is False
    assert wait_events_mod._is_exit_trigger(row, gateway=gateway, trigger="sl") is True


def test_exit_trigger_text_matching_accepts_explicit_tp_markers() -> None:
    gateway = SequenceGateway()
    row = {"comment": "tp", "reason": ""}

    assert wait_events_mod._is_exit_trigger(row, gateway=gateway, trigger="tp") is True


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
