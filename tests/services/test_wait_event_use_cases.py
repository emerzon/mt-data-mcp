from __future__ import annotations

import inspect
from datetime import datetime, timedelta, timezone

from mtdata.core import data as core_data
from mtdata.core.data_requests import WaitEventRequest
from mtdata.core.data_use_cases import run_wait_event


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

    def _next(self, kind: str):
        seq = getattr(self, f"{kind}_seq")
        counter_name = f"_{kind}_calls"
        idx = getattr(self, counter_name)
        setattr(self, counter_name, idx + 1)
        if idx >= len(seq):
            return seq[-1]
        return seq[idx]


def test_wait_event_tool_exposes_minimal_public_contract(monkeypatch) -> None:
    monkeypatch.setattr(
        core_data,
        "run_wait_event",
        lambda request, gateway: {
            "success": True,
            "symbol": request.symbol,
            "timeframe": request.timeframe,
            "watch_for_inferred": request.watch_for is None,
            "watch_for": list(request.watch_for or []),
            "max_wait_seconds": request.max_wait_seconds,
        },
    )
    monkeypatch.setattr(core_data, "get_mt5_gateway", lambda ensure_connection_impl=None: object())

    sig = inspect.signature(core_data.wait_event)
    assert tuple(sig.parameters.keys()) == ("instrument", "timeframe")

    raw = getattr(core_data.wait_event, "__wrapped__", core_data.wait_event)
    result = raw("BTCUSD", "M1")

    assert result["success"] is True
    assert result["symbol"] == "BTCUSD"
    assert result["timeframe"] == "M1"
    assert result["watch_for_inferred"] is False
    assert [item.type for item in result["watch_for"]] == [
        "position_opened",
        "position_closed",
        "tp_hit",
        "sl_hit",
    ]
    assert "max_wait_seconds" not in result


def test_run_wait_event_defers_boundary_only_when_cap_is_short(monkeypatch) -> None:
    monkeypatch.setattr(
        "mtdata.core.wait_events._next_candle_wait_payload",
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
        ]
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
    assert result["matched_event"]["type"] == "order_created"
    assert result["matched_event"]["observed"]["ticket"] == 123


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
        "mtdata.core.wait_events._next_candle_wait_payload",
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
    assert result["boundary_event"]["type"] == "candle_close"
    assert result["boundary_event"]["timeframe"] == "M1"


def test_run_wait_event_uses_timeframe_as_boundary_when_watchers_are_inferred(monkeypatch) -> None:
    monkeypatch.setattr(
        "mtdata.core.wait_events._next_candle_wait_payload",
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
        ticks_by_symbol={"EURUSD": []},
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
    assert result["criteria"]["watch_for_inferred"] is True
    assert result["criteria"]["end_on_inferred"] is True


def test_run_wait_event_stops_on_candle_boundary_when_no_watch_event(monkeypatch) -> None:
    monkeypatch.setattr(
        "mtdata.core.wait_events._next_candle_wait_payload",
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
    assert result["matched_event"]["type"] == "position_closed"
    assert result["matched_event"]["observed"]["ticket"] == 9002
