from __future__ import annotations

from datetime import datetime, timezone

from mtdata.core.data.requests import WaitEventRequest
from mtdata.core.data import wait_events


def test_compile_request_precomputes_watcher_requirements() -> None:
    request = WaitEventRequest.model_validate(
        {
            "symbol": "EURUSD",
            "watch_for": [
                {"type": "pending_near_fill", "distance": 0.2},
                {"type": "order_cancelled"},
                {"type": "position_closed"},
                {
                    "type": "price_change",
                    "window": {"kind": "ticks", "value": 5},
                    "baseline_window": {"kind": "ticks", "value": 20},
                    "threshold_mode": "ratio_to_baseline",
                    "threshold_value": 3.0,
                },
            ],
        }
    )

    compiled = wait_events._compile_request(
        request,
        started_at_utc=datetime(2026, 4, 5, tzinfo=timezone.utc),
    )

    assert compiled["needs_orders"] is True
    assert compiled["needs_positions"] is True
    assert compiled["needs_current_state"] is True
    assert compiled["needs_history_deals"] is True
    assert compiled["needs_history_orders"] is True
    assert [item["type"] for item in compiled["market_specs"]] == [
        "pending_near_fill",
        "price_change",
    ]


def test_collect_snapshot_uses_precomputed_market_specs(monkeypatch) -> None:
    captured = {}

    class Gateway:
        def orders_get(self):  # pragma: no cover - should not be called
            raise AssertionError("orders_get should not be called")

        def positions_get(self):  # pragma: no cover - should not be called
            raise AssertionError("positions_get should not be called")

    def fake_refresh_market_state(*, market_state, gateway, market_specs, observed_at_utc):
        captured["market_specs"] = market_specs
        return {"EURUSD": {"last_epoch": observed_at_utc.timestamp(), "ticks": []}}

    monkeypatch.setattr(wait_events, "_refresh_market_state", fake_refresh_market_state)

    market_specs = [{"type": "price_change", "symbol": "EURUSD"}]
    observed_at_utc = datetime(2026, 4, 5, 13, 0, tzinfo=timezone.utc)
    snapshot = wait_events._collect_snapshot(
        gateway=Gateway(),
        baseline={},
        history_state={},
        market_state={"EURUSD": {"last_epoch": observed_at_utc.timestamp(), "ticks": []}},
        started_at_utc=observed_at_utc,
        observed_at_utc=observed_at_utc,
        needs_orders=False,
        needs_positions=False,
        needs_history_deals=False,
        needs_history_orders=False,
        market_specs=market_specs,
    )

    assert snapshot["market_data"] == {
        "EURUSD": {"last_epoch": observed_at_utc.timestamp(), "ticks": []}
    }
    assert captured["market_specs"] == market_specs
