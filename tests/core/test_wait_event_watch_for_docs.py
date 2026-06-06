"""Tests for WaitEventRequest.watch_for discoverability docs."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from mtdata.core.data.requests import WaitEventRequest


def test_watch_for_description_enumerates_event_types():
    desc = WaitEventRequest.model_fields["watch_for"].description or ""
    for event_type in ("price_change", "volume_spike", "tp_hit", "sl_hit", "stop_threat"):
        assert event_type in desc
    assert "Example" in desc


def test_watch_for_documented_example_is_valid():
    example = [
        {
            "type": "price_change",
            "direction": "up",
            "threshold_mode": "fixed_pct",
            "threshold_value": 0.1,
        }
    ]
    req = WaitEventRequest(symbol="EURUSD", watch_for=example)
    assert req.watch_for[0].type == "price_change"
