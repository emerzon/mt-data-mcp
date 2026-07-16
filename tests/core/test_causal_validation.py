from datetime import datetime, timezone
from unittest.mock import patch

import pandas as pd
import pytest

from mtdata.core import causal


@pytest.mark.parametrize("significance", [0.0, 1.0, -0.1, 2.0, float("nan"), float("inf")])
def test_causal_discovery_rejects_invalid_significance_before_connecting(significance):
    with patch.object(causal, "_causal_connection_error") as connect:
        result = causal.causal_discover_signals.__wrapped__(
            symbols="EURUSD,GBPUSD",
            significance=significance,
        )

    assert result["success"] is False
    assert result["error_code"] == "invalid_input"
    assert "strictly between 0 and 1" in result["error"]
    connect.assert_not_called()


def test_causal_discovery_fails_when_requested_lag_prevents_all_tests():
    index = pd.date_range("2026-01-01", periods=50, freq="h")
    left = pd.Series(range(1, 51), index=index, dtype=float)
    right = pd.Series([value**2 + 1 for value in range(1, 51)], index=index, dtype=float)

    with (
        patch.object(causal, "_causal_connection_error", return_value=None),
        patch.object(
            causal,
            "_fetch_series_for_window",
            side_effect=[(left, None), (right, None)],
        ),
    ):
        result = causal.causal_discover_signals(
            symbols="EURUSD,GBPUSD",
            window_bars=50,
            max_lag=20,
            json=True,
        )

    assert result["success"] is False
    assert result["error_code"] == "no_tests_completed"
    assert result["details"]
    assert all("maximum_allowable_lag" in item for item in result["details"])


def test_fetch_series_excludes_forming_bar_by_default():
    now_epoch = datetime.now(timezone.utc).timestamp()
    completed_open = now_epoch - 7200
    forming_open = now_epoch - 1800
    rates = [
        {"time": completed_open, "close": 1.1},
        {"time": forming_open, "close": 1.2},
    ]

    with (
        patch.object(causal, "_ensure_symbol_ready", return_value=None),
        patch.object(causal, "_mt5_copy_rates_from", return_value=rates),
    ):
        closed, error = causal._fetch_series(
            "EURUSD",
            causal.TIMEFRAME_MAP["H1"],
            2,
            timeframe_key="H1",
        )
        including_forming, include_error = causal._fetch_series(
            "EURUSD",
            causal.TIMEFRAME_MAP["H1"],
            2,
            timeframe_key="H1",
            include_incomplete=True,
        )

    assert error is None
    assert include_error is None
    assert closed.tolist() == [1.1]
    assert closed.attrs["forming_candle_skipped"] is True
    assert including_forming.tolist() == [1.1, 1.2]
