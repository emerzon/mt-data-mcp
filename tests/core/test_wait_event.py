from __future__ import annotations

import pytest

from mtdata.core.data_requests import WaitEventRequest


def test_wait_event_request_defaults_watch_for_to_inferred_set() -> None:
    request = WaitEventRequest()

    assert request.watch_for is None
    assert request.end_on == []
    assert request.max_wait_seconds == 86400.0


def test_wait_event_request_rejects_non_positive_poll_interval() -> None:
    with pytest.raises(
        ValueError, match="poll_interval_seconds must be greater than 0"
    ):
        WaitEventRequest(
            watch_for=[{"type": "order_created"}],
            poll_interval_seconds=0.0,
        )


def test_wait_event_request_parses_market_event_specs() -> None:
    request = WaitEventRequest.model_validate(
        {
            "symbol": "EURUSD",
            "watch_for": [
                {
                    "type": "price_change",
                    "window": {"kind": "ticks", "value": 5},
                    "baseline_window": {"kind": "ticks", "value": 20},
                    "threshold_mode": "ratio_to_baseline",
                    "threshold_value": 3.0,
                },
                {
                    "type": "volume_spike",
                    "window": {"kind": "minutes", "value": 3},
                    "baseline_window": {"kind": "minutes", "value": 30},
                    "source": "tick_count",
                    "threshold_mode": "zscore",
                    "threshold_value": 2.5,
                },
            ],
            "end_on": [{"type": "candle_close", "timeframe": "M5"}],
        }
    )

    assert len(request.watch_for) == 2
    assert request.watch_for[0].type == "price_change"
    assert request.watch_for[1].type == "volume_spike"
    assert request.end_on[0].type == "candle_close"
