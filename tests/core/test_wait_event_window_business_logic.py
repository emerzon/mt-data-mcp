from __future__ import annotations

import pytest

from mtdata.core import wait_events


def test_window_ticks_uses_time_window_cutoff() -> None:
    ticks = [
        {"epoch": 10.0},
        {"epoch": 40.0},
        {"epoch": 80.0},
        {"epoch": 100.0},
    ]

    out = wait_events._window_ticks(ticks, {"kind": "minutes", "value": 0.5})

    assert [tick["epoch"] for tick in out] == [80.0, 100.0]


def test_window_prices_uses_time_window_cutoff() -> None:
    prices = [
        (10.0, 100.0),
        (40.0, 101.0),
        (80.0, 102.0),
        (100.0, 103.0),
    ]

    out = wait_events._window_prices(prices, {"kind": "minutes", "value": 0.5})

    assert out == [(80.0, 102.0), (100.0, 103.0)]


def test_duration_price_change_baseline_samples_preserve_time_window_boundaries() -> None:
    spec = {
        "window": {"kind": "minutes", "value": 1.0},
        "baseline_window": {"kind": "minutes", "value": 2.0},
    }
    prices = [
        (0.0, 100.0),
        (30.0, 101.0),
        (60.0, 102.0),
        (90.0, 103.0),
        (120.0, 104.0),
        (150.0, 105.0),
        (180.0, 106.0),
    ]

    out = wait_events._duration_price_change_baseline_samples(spec, prices)

    assert out[0] == pytest.approx(2.0)
    assert out[1] == pytest.approx(((104.0 - 102.0) / 102.0) * 100.0)
