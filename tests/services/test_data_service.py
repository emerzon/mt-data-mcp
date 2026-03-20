# ruff: noqa: E402
import unittest
from unittest.mock import patch, MagicMock
from contextlib import contextmanager
from typing import Any, Iterator, Tuple
from datetime import datetime, timedelta, timezone
import sys
import os

# Add src to path to ensure local package is found
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

# Mock mt5 before importing data_service because it imports mt5 at top level
sys.modules["MetaTrader5"] = MagicMock()

from mtdata.services.data_service import fetch_candles, fetch_ticks


@contextmanager
def _mock_symbol_ready_guard(
    *args: Any, **kwargs: Any
) -> Iterator[Tuple[None, MagicMock]]:
    yield None, MagicMock()


class TestDataService(unittest.TestCase):
    @patch("mtdata.services.data_service._mt5_copy_rates_from")
    @patch(
        "mtdata.services.data_service._mt5_epoch_to_utc",
        side_effect=AssertionError("unexpected extra UTC conversion"),
    )
    @patch("mtdata.services.data_service._symbol_ready_guard", _mock_symbol_ready_guard)
    def test_fetch_candles_basic(self, _mock_epoch_to_utc, mock_copy_rates):
        # Mock rates data
        now = datetime.now(timezone.utc)
        rates = []
        for i in range(10):
            t = now - timedelta(minutes=10 - i)
            rates.append(
                {
                    "time": t.timestamp(),
                    "open": 1.1,
                    "high": 1.2,
                    "low": 1.0,
                    "close": 1.15,
                    "tick_volume": 100,
                    "real_volume": 0,
                    "spread": 1,
                }
            )
        # mt5 returns a numpy structured array; for tests, a list of dicts is sufficient and
        # is directly supported by pd.DataFrame.
        mock_copy_rates.return_value = rates

        # Execute
        result = fetch_candles(symbol="EURUSD", limit=5)

        # Verify
        self.assertIsInstance(result, dict)
        self.assertTrue(result.get("success"))
        self.assertEqual(result.get("candles"), 5)
        # Check tabular rows
        data = result.get("data")
        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 5)
        self.assertTrue(all(isinstance(row, dict) for row in data))
        self.assertTrue(
            {"time", "open", "high", "low", "close"}.issubset(set(data[0].keys()))
        )
        self.assertIsInstance(data[0]["open"], (int, float))
        self.assertIsInstance(data[0]["close"], (int, float))

    @patch("mtdata.services.data_service._mt5_copy_rates_from")
    @patch("mtdata.services.data_service._symbol_ready_guard", _mock_symbol_ready_guard)
    def test_fetch_candles_time_as_epoch(self, mock_copy_rates):
        now = datetime.now(timezone.utc)
        rates = []
        for i in range(10):
            t = now - timedelta(minutes=10 - i)
            rates.append(
                {
                    "time": t.timestamp(),
                    "open": 1.1,
                    "high": 1.2,
                    "low": 1.0,
                    "close": 1.15,
                    "tick_volume": 100,
                    "real_volume": 0,
                    "spread": 1,
                }
            )
        mock_copy_rates.return_value = rates

        result = fetch_candles(symbol="EURUSD", limit=5, time_as_epoch=True)

        self.assertIsInstance(result, dict)
        self.assertTrue(result.get("success"))
        data = result.get("data") or []
        self.assertTrue(all(isinstance(row.get("time"), (int, float)) for row in data))

    @patch("mtdata.services.data_service._mt5_copy_rates_from")
    @patch(
        "mtdata.services.data_service._mt5_epoch_to_utc",
        side_effect=AssertionError("unexpected extra UTC conversion"),
    )
    @patch("mtdata.services.data_service._symbol_ready_guard", _mock_symbol_ready_guard)
    def test_fetch_candles_does_not_double_convert_normalized_times(
        self, _mock_epoch_to_utc, mock_copy_rates
    ):
        rates = [
            {
                "time": 1704067200.0,
                "open": 1.1,
                "high": 1.2,
                "low": 1.0,
                "close": 1.15,
                "tick_volume": 100,
                "real_volume": 0,
                "spread": 1,
            },
            {
                "time": 1704067260.0,
                "open": 1.15,
                "high": 1.25,
                "low": 1.05,
                "close": 1.2,
                "tick_volume": 120,
                "real_volume": 0,
                "spread": 1,
            },
        ]
        mock_copy_rates.return_value = rates

        result = fetch_candles(symbol="EURUSD", limit=2, time_as_epoch=True)

        self.assertTrue(result.get("success"))
        self.assertEqual(
            [row["time"] for row in result.get("data", [])],
            [1704067200.0, 1704067260.0],
        )

    @patch("mtdata.services.data_service._mt5_copy_ticks_range")
    @patch(
        "mtdata.services.data_service._mt5_epoch_to_utc",
        side_effect=AssertionError("unexpected extra UTC conversion"),
    )
    @patch("mtdata.services.data_service._symbol_ready_guard", _mock_symbol_ready_guard)
    def test_fetch_ticks_basic(self, _mock_epoch_to_utc, mock_copy_ticks):
        # Mock ticks data
        now = datetime.now(timezone.utc)
        ticks = []
        for i in range(10):
            t = now - timedelta(seconds=10 - i)
            ticks.append(
                {
                    "time": t.timestamp(),
                    "bid": 1.1,
                    "ask": 1.1001,
                    "last": 1.1,
                    "volume": 1.0,
                    "time_msc": t.timestamp() * 1000,
                    "flags": 0,
                    "volume_real": 0.0,
                }
            )
        mock_copy_ticks.return_value = ticks

        # Execute
        result = fetch_ticks(symbol="EURUSD", limit=5)

        # Verify
        self.assertIsInstance(result, dict)
        self.assertTrue(result.get("success"))
        self.assertEqual(result.get("count"), 5)

    @patch("mtdata.services.data_service._mt5_copy_ticks_range")
    @patch(
        "mtdata.services.data_service._mt5_epoch_to_utc",
        side_effect=AssertionError("unexpected extra UTC conversion"),
    )
    @patch("mtdata.services.data_service._symbol_ready_guard", _mock_symbol_ready_guard)
    def test_fetch_ticks_does_not_double_convert_normalized_times(
        self, _mock_epoch_to_utc, mock_copy_ticks
    ):
        ticks = [
            {
                "time": 1704067200.0,
                "bid": 1.1,
                "ask": 1.1001,
                "last": 1.1,
                "volume": 1.0,
                "time_msc": 1704067200000.0,
                "flags": 0,
                "volume_real": 0.0,
            },
            {
                "time": 1704067201.0,
                "bid": 1.1001,
                "ask": 1.1002,
                "last": 1.10015,
                "volume": 1.0,
                "time_msc": 1704067201000.0,
                "flags": 0,
                "volume_real": 0.0,
            },
        ]
        mock_copy_ticks.return_value = ticks

        result = fetch_ticks(symbol="EURUSD", limit=2, output="summary")

        self.assertTrue(result.get("success"))
        self.assertEqual(result.get("count"), 2)

    @patch("mtdata.services.data_service._mt5_copy_ticks_range")
    @patch("mtdata.services.data_service._symbol_ready_guard", _mock_symbol_ready_guard)
    def test_fetch_ticks_zero_duration_uses_note_instead_of_nan_rate(
        self, mock_copy_ticks
    ):
        base_ts = datetime.now(timezone.utc).timestamp()
        ticks = []
        for _ in range(5):
            ticks.append(
                {
                    "time": base_ts,
                    "bid": 1.1,
                    "ask": 1.1001,
                    "last": 1.1,
                    "volume": 1.0,
                    "time_msc": base_ts * 1000,
                    "flags": 0,
                    "volume_real": 0.0,
                }
            )
        mock_copy_ticks.return_value = ticks

        result = fetch_ticks(symbol="EURUSD", limit=5)

        self.assertIsInstance(result, dict)
        self.assertTrue(result.get("success"))
        self.assertEqual(result.get("duration_seconds"), 0.0)
        self.assertIsNone(result.get("tick_rate_per_second"))
        self.assertEqual(result.get("tick_rate_note"), "< 1s window")

    @patch("mtdata.services.data_service._mt5_copy_ticks_range")
    @patch("mtdata.services.data_service._symbol_ready_guard", _mock_symbol_ready_guard)
    def test_fetch_ticks_spread_change_pct_zero_first_uses_note(self, mock_copy_ticks):
        now = datetime.now(timezone.utc).timestamp()
        ticks = [
            {
                "time": now - 2,
                "bid": 1.1000,
                "ask": 1.1000,
                "last": 1.1000,
                "volume": 1.0,
                "time_msc": (now - 2) * 1000,
                "flags": 0,
                "volume_real": 0.0,
            },
            {
                "time": now - 1,
                "bid": 1.1000,
                "ask": 1.1001,
                "last": 1.10005,
                "volume": 1.0,
                "time_msc": (now - 1) * 1000,
                "flags": 0,
                "volume_real": 0.0,
            },
            {
                "time": now,
                "bid": 1.1000,
                "ask": 1.1002,
                "last": 1.1001,
                "volume": 1.0,
                "time_msc": now * 1000,
                "flags": 0,
                "volume_real": 0.0,
            },
        ]
        mock_copy_ticks.return_value = ticks

        result = fetch_ticks(symbol="EURUSD", limit=3, output="summary")

        self.assertTrue(result.get("success"))
        self.assertEqual(result.get("spread_change_pct_note"), "first spread was zero")
        self.assertIsNone(result.get("stats", {}).get("spread", {}).get("change_pct"))


if __name__ == "__main__":
    try:
        unittest.main(exit=False)
        with open("test_service_results.txt", "w") as f:
            f.write("SUCCESS")
    except Exception as e:
        with open("test_service_results.txt", "w") as f:
            f.write(f"FAILURE: {e}")
