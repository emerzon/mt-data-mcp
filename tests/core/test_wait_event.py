from __future__ import annotations

import pytest
from unittest.mock import patch

from mtdata.core.data.requests import WaitEventRequest


def test_wait_event_request_defaults_watch_for_to_inferred_set() -> None:
    request = WaitEventRequest()

    assert request.watch_for is None
    assert request.end_on == []
    assert request.max_wait_seconds == 86400.0


def test_wait_event_request_rejects_non_positive_poll_interval() -> None:
    with pytest.raises(ValueError, match="poll_interval_seconds must be greater than 0"):
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
                {
                    "type": "tick_count_spike",
                    "window": {"kind": "minutes", "value": 2},
                    "baseline_window": {"kind": "minutes", "value": 20},
                    "threshold_mode": "ratio_to_baseline",
                    "threshold_value": 3.0,
                },
                {
                    "type": "spread_spike",
                    "window": {"kind": "ticks", "value": 3},
                    "baseline_window": {"kind": "ticks", "value": 12},
                    "threshold_mode": "ratio_to_baseline",
                    "threshold_value": 2.0,
                },
                {
                    "type": "tick_count_drought",
                    "window": {"kind": "minutes", "value": 2},
                    "baseline_window": {"kind": "minutes", "value": 20},
                    "threshold_mode": "ratio_to_baseline",
                    "threshold_value": 0.5,
                },
                {
                    "type": "range_expansion",
                    "window": {"kind": "ticks", "value": 4},
                    "baseline_window": {"kind": "ticks", "value": 16},
                    "price_source": "mid",
                    "threshold_mode": "zscore",
                    "threshold_value": 2.0,
                },
                {
                    "type": "price_touch_level",
                    "level": 100.5,
                    "direction": "up",
                    "tolerance": 0.1,
                },
                {
                    "type": "price_break_level",
                    "level": 101.0,
                    "direction": "up",
                    "tolerance": 0.05,
                    "confirm_ticks": 2,
                },
                {
                    "type": "price_enter_zone",
                    "lower": 99.5,
                    "upper": 100.5,
                    "direction": "either",
                },
                {
                    "type": "pending_near_fill",
                    "distance": 0.2,
                },
                {
                    "type": "stop_threat",
                    "distance": 0.15,
                },
            ],
            "end_on": [{"type": "candle_close", "timeframe": "M5"}],
        }
    )

    assert len(request.watch_for) == 11
    assert request.watch_for[0].type == "price_change"
    assert request.watch_for[1].type == "volume_spike"
    assert request.watch_for[2].type == "tick_count_spike"
    assert request.watch_for[3].type == "spread_spike"
    assert request.watch_for[4].type == "tick_count_drought"
    assert request.watch_for[5].type == "range_expansion"
    assert request.watch_for[6].type == "price_touch_level"
    assert request.watch_for[7].type == "price_break_level"
    assert request.watch_for[8].type == "price_enter_zone"
    assert request.watch_for[9].type == "pending_near_fill"
    assert request.watch_for[10].type == "stop_threat"
    assert request.end_on[0].type == "candle_close"


def test_wait_event_request_normalizes_legacy_event_names() -> None:
    request = WaitEventRequest.model_validate(
        {
            "symbol": "EURUSD",
            "watch_for": [
                {"type": "candle_close", "timeframe": "M15"},
                {"type": "price_level_touch", "level": 1.1454, "tolerance": 0.0002},
            ],
        }
    )

    assert [item.type for item in request.watch_for] == ["price_touch_level"]
    assert [item.type for item in request.end_on] == ["candle_close"]
    assert request.end_on[0].timeframe == "M15"


def test_wait_event_request_normalizes_price_direction_aliases() -> None:
    request = WaitEventRequest.model_validate(
        {
            "symbol": "EURUSD",
            "watch_for": [
                {"type": "price_touch_level", "level": 1.1454, "direction": "above"},
                {"type": "price_break_level", "level": 1.1460, "direction": "below"},
                {
                    "type": "price_enter_zone",
                    "lower": 1.1440,
                    "upper": 1.1460,
                    "direction": "above",
                },
            ],
        }
    )

    assert [item.direction for item in request.watch_for] == ["up", "down", "up"]


def test_wait_event_request_normalizes_account_side_aliases() -> None:
    request = WaitEventRequest.model_validate(
        {
            "symbol": "EURUSD",
            "watch_for": [
                {"type": "position_opened", "symbol": "EURUSD", "side": "long"},
                {"type": "position_closed", "symbol": "EURUSD", "side": "short"},
            ],
        }
    )

    assert [item.side for item in request.watch_for] == ["buy", "sell"]


def test_wait_event_request_deduplicates_identical_candle_close_events() -> None:
    request = WaitEventRequest.model_validate(
        {
            "symbol": "EURUSD",
            "watch_for": [
                {"type": "candle_close", "timeframe": "M15"},
                {"type": "price_level_touch", "level": 1.1454, "tolerance": 0.0002},
            ],
            "end_on": [{"type": "candle_close", "timeframe": "M15"}],
        }
    )

    assert [item.type for item in request.watch_for] == ["price_touch_level"]
    assert [(item.type, item.timeframe) for item in request.end_on] == [("candle_close", "M15")]


def test_wait_event_request_deduplicates_candle_close_events_with_null_optionals() -> None:
    request = WaitEventRequest.model_validate(
        {
            "symbol": "EURUSD",
            "watch_for": [{"type": "candle_close", "timeframe": "M15", "buffer_seconds": None}],
            "end_on": [{"type": "candle_close", "timeframe": "M15"}],
        }
    )

    assert request.watch_for == []
    assert [(item.type, item.timeframe, item.buffer_seconds) for item in request.end_on] == [
        ("candle_close", "M15", None),
    ]


def test_wait_event_request_preserves_distinct_candle_close_boundaries() -> None:
    request = WaitEventRequest.model_validate(
        {
            "symbol": "EURUSD",
            "watch_for": [{"type": "candle_close", "timeframe": "M15"}],
            "end_on": [{"type": "candle_close", "timeframe": "M5"}],
        }
    )

    assert request.watch_for == []
    assert [(item.type, item.timeframe) for item in request.end_on] == [
        ("candle_close", "M15"),
        ("candle_close", "M5"),
    ]


def _raw_wait_event():
    from mtdata.core.data import wait_event

    return wait_event.__wrapped__


@patch("mtdata.core.data.get_mt5_gateway", return_value=object())
@patch("mtdata.core.data._compact_wait_event_public_result", side_effect=lambda result, **_: result)
@patch("mtdata.core.data.run_wait_event", return_value={"success": True})
def test_wait_event_prefers_public_symbol_name(mock_run_wait, _mock_compact, _mock_gateway) -> None:
    result = _raw_wait_event()(symbol="EURUSD", watch_for=[])

    assert result == {"success": True}
    request = mock_run_wait.call_args.args[0]
    assert request.symbol == "EURUSD"


@patch("mtdata.core.data.get_mt5_gateway", return_value=object())
@patch("mtdata.core.data._compact_wait_event_public_result", side_effect=lambda result, **_: result)
@patch("mtdata.core.data.run_wait_event", return_value={"success": True})
def test_wait_event_keeps_instrument_alias_for_compatibility(mock_run_wait, _mock_compact, _mock_gateway) -> None:
    result = _raw_wait_event()(instrument="EURUSD", watch_for=[])

    assert result == {"success": True}
    request = mock_run_wait.call_args.args[0]
    assert request.symbol == "EURUSD"


def test_wait_event_rejects_conflicting_symbol_and_instrument() -> None:
    result = _raw_wait_event()(symbol="EURUSD", instrument="GBPUSD", watch_for=[])

    assert result == {
        "error": "Provide either symbol or instrument, not both with different values."
    }
