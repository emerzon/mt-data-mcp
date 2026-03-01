from __future__ import annotations

from unittest.mock import patch

import pandas as pd

from mtdata.core.patterns import patterns_detect


def _unwrap(fn):
    while hasattr(fn, "__wrapped__"):
        fn = fn.__wrapped__
    return fn


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


def test_patterns_detect_chart_alias_routes_to_classic_mode() -> None:
    raw = _unwrap(patterns_detect)
    with patch("mtdata.core.patterns._fetch_pattern_data", return_value=(_sample_df(), None)), patch(
        "mtdata.core.patterns._select_classic_engines",
        return_value=(["native"], []),
    ), patch(
        "mtdata.core.patterns._run_classic_engine",
        return_value=([{"pattern": "double_top", "status": "forming"}], None),
    ):
        out = raw(symbol="EURUSD", mode="chart", timeframe="H1")

    assert out.get("success") is True
    assert out.get("mode") == "classic"
