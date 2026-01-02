from __future__ import annotations

from datetime import datetime, timezone

import os
import sys

# Add src to path to ensure local package is found
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from mtdata.utils.utils import _parse_start_datetime, _utc_epoch_seconds


def test_utc_epoch_seconds_treats_naive_as_utc() -> None:
    dt = datetime(2020, 1, 1, 0, 0, 0)
    assert int(_utc_epoch_seconds(dt)) == 1577836800


def test_utc_epoch_seconds_handles_aware() -> None:
    dt = datetime(2020, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    assert int(_utc_epoch_seconds(dt)) == 1577836800


def test_parse_start_datetime_epoch_is_utc() -> None:
    dt = _parse_start_datetime("2020-01-01 00:00:00")
    assert dt is not None
    assert dt.tzinfo is None
    assert int(_utc_epoch_seconds(dt)) == 1577836800
