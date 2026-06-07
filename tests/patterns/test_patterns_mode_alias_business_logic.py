from __future__ import annotations

from unittest.mock import patch

import pandas as pd

from mtdata.core.patterns import patterns_detect
from mtdata.core.patterns_requests import PatternsDetectRequest


def _unwrap(fn):
    while hasattr(fn, "__wrapped__"):
        fn = fn.__wrapped__
    return fn


def _call_patterns_detect(**kwargs):
    raw = _unwrap(patterns_detect)
    with patch("mtdata.core.patterns.ensure_mt5_connection_or_raise", return_value=None):
        return raw(request=PatternsDetectRequest(**kwargs))


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": [1.0, 2.0, 3.0, 4.0, 5.0],
            "open": [1.0, 1.1, 1.2, 1.3, 1.4],
            "high": [1.1, 1.2, 1.3, 1.4, 1.5],
            "low": [0.9, 1.0, 1.1, 1.2, 1.3],
            "close": [1.05, 1.15, 1.25, 1.35, 1.45],
            "tick_volume": [10, 11, 12, 13, 14],
        }
    )


def test_patterns_detect_rejects_removed_chart_alias() -> None:
    out = _call_patterns_detect(symbol="EURUSD", mode="chart", timeframe="H1")

    assert out["error"].startswith("Unknown mode: chart.")


def test_patterns_detect_rejects_engine_for_non_classic_mode() -> None:
    out = _call_patterns_detect(
        symbol="EURUSD",
        mode="candlestick",
        timeframe="H1",
        engine="stock_pattern",
    )

    assert out == {"error": "engine applies only to mode='classic'."}
