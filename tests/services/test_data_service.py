

import math
import os
import sys
import unittest
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Iterator, Tuple
from unittest.mock import MagicMock, patch

# Add src to path to ensure local package is found
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

# Mock mt5 before importing data_service because it imports mt5 at top level
sys.modules['MetaTrader5'] = MagicMock()

from mtdata.services.data_service import fetch_candles, fetch_ticks


@contextmanager
def _mock_symbol_ready_guard(*args: Any, **kwargs: Any) -> Iterator[Tuple[None, MagicMock]]:
    yield None, MagicMock()


class TestDataService(unittest.TestCase):

    def test_tick_field_value_returns_none_for_none_tick(self):
        from mtdata.services import data_service as data_service_mod

        self.assertIsNone(data_service_mod._tick_field_value(None, "bid"))
    @patch('mtdata.services.data_service._mt5_copy_rates_from')
    @patch('mtdata.services.data_service._mt5_epoch_to_utc', side_effect=AssertionError("unexpected extra UTC conversion"))
    @patch('mtdata.services.data_service._symbol_ready_guard', _mock_symbol_ready_guard)
    def test_fetch_candles_basic(self, _mock_epoch_to_utc, mock_copy_rates):
        
        # Mock rates data — use H1-spaced bars so the last bar is clearly closed
        now = datetime.now(timezone.utc)
        rates = []
        for i in range(10):
            t = now - timedelta(hours=10-i)
            rates.append({
                'time': t.timestamp(),
                'open': 1.1, 'high': 1.2, 'low': 1.0, 'close': 1.15,
                'tick_volume': 100, 'real_volume': 0, 'spread': 1
            })
        # mt5 returns a numpy structured array; for tests, a list of dicts is sufficient and
        # is directly supported by pd.DataFrame.
        mock_copy_rates.return_value = rates
        
        # Execute
        result = fetch_candles(symbol="EURUSD", limit=5)
        
        # Verify
        self.assertIsInstance(result, dict)
        self.assertTrue(result.get('success'))
        self.assertEqual(result.get('candles'), 5)
        self.assertNotIn('count', result)
        # Check tabular rows
        data = result.get('data')
        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 5)
        self.assertTrue(all(isinstance(row, dict) for row in data))
        self.assertTrue({'time', 'open', 'high', 'low', 'close'}.issubset(set(data[0].keys())))
        self.assertIsInstance(data[0]['open'], (int, float))
        self.assertIsInstance(data[0]['close'], (int, float))

    @patch('mtdata.services.data_service._mt5_copy_rates_from')
    @patch('mtdata.services.data_service._symbol_ready_guard', _mock_symbol_ready_guard)
    def test_fetch_candles_time_as_epoch(self, mock_copy_rates):
        now = datetime.now(timezone.utc)
        rates = []
        for i in range(10):
            t = now - timedelta(minutes=10 - i)
            rates.append({
                'time': t.timestamp(),
                'open': 1.1, 'high': 1.2, 'low': 1.0, 'close': 1.15,
                'tick_volume': 100, 'real_volume': 0, 'spread': 1
            })
        mock_copy_rates.return_value = rates

        result = fetch_candles(symbol="EURUSD", limit=5, time_as_epoch=True)

        self.assertIsInstance(result, dict)
        self.assertTrue(result.get('success'))
        data = result.get('data') or []
        self.assertTrue(all(isinstance(row.get('time'), (int, float)) for row in data))

    @patch('mtdata.services.data_service._mt5_copy_rates_from')
    @patch('mtdata.services.data_service._mt5_epoch_to_utc', side_effect=AssertionError("unexpected extra UTC conversion"))
    @patch('mtdata.services.data_service._symbol_ready_guard', _mock_symbol_ready_guard)
    def test_fetch_candles_does_not_double_convert_normalized_times(self, _mock_epoch_to_utc, mock_copy_rates):
        now = datetime.now(timezone.utc)
        rates = [
            {
                'time': (now - timedelta(hours=2)).timestamp(),
                'open': 1.1, 'high': 1.2, 'low': 1.0, 'close': 1.15,
                'tick_volume': 100, 'real_volume': 0, 'spread': 1,
            },
            {
                'time': (now - timedelta(hours=1)).timestamp(),
                'open': 1.15, 'high': 1.25, 'low': 1.05, 'close': 1.2,
                'tick_volume': 120, 'real_volume': 0, 'spread': 1,
            },
        ]
        mock_copy_rates.return_value = rates

        result = fetch_candles(symbol="EURUSD", limit=2, time_as_epoch=True)

        self.assertTrue(result.get("success"))
        self.assertEqual(
            [row["time"] for row in result.get("data", [])],
            [rates[0]["time"], rates[1]["time"]],
        )
        
    @patch('mtdata.services.data_service._mt5_copy_ticks_range')
    @patch('mtdata.services.data_service._mt5_epoch_to_utc', side_effect=AssertionError("unexpected extra UTC conversion"))
    @patch('mtdata.services.data_service._symbol_ready_guard', _mock_symbol_ready_guard)
    def test_fetch_ticks_basic(self, _mock_epoch_to_utc, mock_copy_ticks):
        
        # Mock ticks data
        now = datetime.now(timezone.utc)
        ticks = []
        for i in range(10):
            t = now - timedelta(seconds=10-i)
            ticks.append({
                'time': t.timestamp(),
                'bid': 1.1, 'ask': 1.1001, 'last': 1.1, 'volume': 1.0, 'time_msc': t.timestamp()*1000, 'flags': 0, 'volume_real': 0.0
            })
        mock_copy_ticks.return_value = ticks
        
        # Execute
        result = fetch_ticks(symbol="EURUSD", limit=5)
        
        # Verify
        self.assertIsInstance(result, dict)
        self.assertTrue(result.get('success'))
        self.assertEqual(result.get('count'), 5)

    @patch('mtdata.services.data_service._mt5_copy_ticks_range')
    @patch('mtdata.services.data_service._resolve_client_tz', return_value=None)
    @patch('mtdata.services.data_service._symbol_ready_guard', _mock_symbol_ready_guard)
    def test_fetch_ticks_select_simplify_reuses_cached_tick_fields(self, mock_ctz, mock_copy_ticks):
        now = datetime.now(timezone.utc)
        ticks = []
        for i in range(20):
            t = now - timedelta(seconds=20 - i)
            ticks.append({
                'time': t.timestamp(),
                'bid': 1.1 + i * 0.0001,
                'ask': 1.1001 + i * 0.0001,
                'last': 1.10005 + i * 0.0001,
                'volume': 1.0,
                'time_msc': t.timestamp() * 1000,
                'flags': 0,
                'volume_real': 0.0,
            })
        mock_copy_ticks.return_value = ticks

        from mtdata.services import data_service as data_service_mod

        original_tick_field_value = data_service_mod._tick_field_value
        call_count = {"value": 0}

        def counting_tick_field_value(tick, name):
            call_count["value"] += 1
            return original_tick_field_value(tick, name)

        with patch(
            'mtdata.services.data_service._select_indices_for_timeseries',
            return_value=([0, 19], 'lttb', {}),
        ), patch(
            'mtdata.services.data_service._lttb_select_indices',
            return_value=[0, 5, 10, 15, 19],
        ), patch(
            'mtdata.services.data_service._tick_field_value',
            side_effect=counting_tick_field_value,
        ):
            result = fetch_ticks(
                symbol="EURUSD",
                limit=20,
                format="rows",
                simplify={'mode': 'select', 'points': 5},
            )

        self.assertTrue(result.get('success'))
        self.assertEqual(result.get('count'), 5)
        scanned_fields = ("time", "bid", "ask", "last", "flags", "volume", "volume_real")
        # The shared extraction pass should read each relevant tick field once,
        # even when simplification and row rendering are both enabled.
        expected_max_field_reads = len(ticks) * len(scanned_fields)
        self.assertLessEqual(call_count["value"], expected_max_field_reads)

    @patch('mtdata.services.data_service._mt5_copy_ticks_range')
    @patch('mtdata.services.data_service._mt5_epoch_to_utc', side_effect=AssertionError("unexpected extra UTC conversion"))
    @patch('mtdata.services.data_service._symbol_ready_guard', _mock_symbol_ready_guard)
    def test_fetch_ticks_does_not_double_convert_normalized_times(self, _mock_epoch_to_utc, mock_copy_ticks):
        ticks = [
            {
                'time': 1704067200.0,
                'bid': 1.1,
                'ask': 1.1001,
                'last': 1.1,
                'volume': 1.0,
                'time_msc': 1704067200000.0,
                'flags': 0,
                'volume_real': 0.0,
            },
            {
                'time': 1704067201.0,
                'bid': 1.1001,
                'ask': 1.1002,
                'last': 1.10015,
                'volume': 1.0,
                'time_msc': 1704067201000.0,
                'flags': 0,
                'volume_real': 0.0,
            },
        ]
        mock_copy_ticks.return_value = ticks

        result = fetch_ticks(symbol="EURUSD", limit=2, format="summary")

        self.assertTrue(result.get("success"))
        self.assertEqual(result.get("count"), 2)

    @patch('mtdata.services.data_service._mt5_copy_ticks_range')
    @patch('mtdata.services.data_service._symbol_ready_guard', _mock_symbol_ready_guard)
    def test_fetch_ticks_zero_duration_uses_note_instead_of_nan_rate(self, mock_copy_ticks):
        base_ts = datetime.now(timezone.utc).timestamp()
        ticks = []
        for _ in range(5):
            ticks.append(
                {
                    'time': base_ts,
                    'bid': 1.1,
                    'ask': 1.1001,
                    'last': 1.1,
                    'volume': 1.0,
                    'time_msc': base_ts * 1000,
                    'flags': 0,
                    'volume_real': 0.0,
                }
            )
        mock_copy_ticks.return_value = ticks

        result = fetch_ticks(symbol="EURUSD", limit=5)

        self.assertIsInstance(result, dict)
        self.assertTrue(result.get('success'))
        self.assertEqual(result.get('duration_seconds'), 0.0)
        self.assertIsNone(result.get('tick_rate_per_second'))
        self.assertEqual(result.get('tick_rate_note'), "< 1s window")

    @patch('mtdata.services.data_service._mt5_copy_ticks_range')
    @patch('mtdata.services.data_service._symbol_ready_guard', _mock_symbol_ready_guard)
    def test_fetch_ticks_spread_change_pct_zero_first_uses_note(self, mock_copy_ticks):
        now = datetime.now(timezone.utc).timestamp()
        ticks = [
            {
                'time': now - 2,
                'bid': 1.1000,
                'ask': 1.1000,
                'last': 1.1000,
                'volume': 1.0,
                'time_msc': (now - 2) * 1000,
                'flags': 0,
                'volume_real': 0.0,
            },
            {
                'time': now - 1,
                'bid': 1.1000,
                'ask': 1.1001,
                'last': 1.10005,
                'volume': 1.0,
                'time_msc': (now - 1) * 1000,
                'flags': 0,
                'volume_real': 0.0,
            },
            {
                'time': now,
                'bid': 1.1000,
                'ask': 1.1002,
                'last': 1.1001,
                'volume': 1.0,
                'time_msc': now * 1000,
                'flags': 0,
                'volume_real': 0.0,
            },
        ]
        mock_copy_ticks.return_value = ticks

        result = fetch_ticks(symbol="EURUSD", limit=3, format="summary")

        self.assertTrue(result.get("success"))
        self.assertEqual(result.get("spread_change_pct_note"), "first spread was zero")
        self.assertIsNone(result.get("stats", {}).get("spread", {}).get("change_pct"))

    @patch('mtdata.services.data_service._mt5_copy_ticks_range')
    @patch('mtdata.services.data_service._symbol_ready_guard', _mock_symbol_ready_guard)
    def test_fetch_ticks_summary_keeps_empty_series_stats_shape(self, mock_copy_ticks):
        now = datetime.now(timezone.utc).timestamp()
        ticks = [
            {
                'time': now - 2,
                'bid': float("nan"),
                'ask': 1.1001,
                'last': 1.10005,
                'volume': 1.0,
                'time_msc': (now - 2) * 1000,
                'flags': 0,
                'volume_real': 0.0,
            },
            {
                'time': now - 1,
                'bid': float("nan"),
                'ask': 1.1002,
                'last': 1.10015,
                'volume': 1.0,
                'time_msc': (now - 1) * 1000,
                'flags': 0,
                'volume_real': 0.0,
            },
            {
                'time': now,
                'bid': float("nan"),
                'ask': 1.1003,
                'last': 1.10025,
                'volume': 1.0,
                'time_msc': now * 1000,
                'flags': 0,
                'volume_real': 0.0,
            },
        ]
        mock_copy_ticks.return_value = ticks

        result = fetch_ticks(symbol="EURUSD", limit=3, format="summary")

        self.assertTrue(result.get("success"))
        bid_stats = result.get("stats", {}).get("bid", {})
        self.assertFalse(bid_stats.get("available"))
        for key in ("first", "last", "low", "high", "mean", "std", "stderr", "kurtosis", "change", "change_pct"):
            self.assertIn(key, bid_stats)
        self.assertEqual(bid_stats.get("count"), 0)
        self.assertTrue(math.isnan(bid_stats["first"]))
        self.assertTrue(math.isnan(bid_stats["mean"]))

    def test_format_candle_times_vectorizes_utc_formatting(self):
        import pandas as pd

        from mtdata.services import data_service as data_service_mod

        df = pd.DataFrame({
            '__epoch': [1704067200.0, 1704067260.0, 1704067320.0],
            'time': [None, None, None],
        })

        with patch('mtdata.services.data_service.datetime') as mock_datetime:
            mock_datetime.fromtimestamp.side_effect = AssertionError("should not format rows one by one")
            data_service_mod._format_candle_times(
                df,
                ['time'],
                time_as_epoch=False,
                use_client_tz=False,
                client_tz=None,
            )

        self.assertEqual(df.attrs['_tz_used_name'], 'UTC')
        self.assertEqual(list(df['time']), ['2024-01-01 00:00', '2024-01-01 00:01', '2024-01-01 00:02'])

    def test_format_candle_times_vectorizes_client_tz_formatting(self):
        import pandas as pd

        from mtdata.services import data_service as data_service_mod

        df = pd.DataFrame({
            '__epoch': [1704067200.0, 1704067260.0, 1704067320.0],
            'time': [None, None, None],
        })
        client_tz = timezone(timedelta(hours=-6))

        with patch('mtdata.services.data_service.datetime') as mock_datetime:
            mock_datetime.fromtimestamp.side_effect = AssertionError("should not format rows one by one")
            data_service_mod._format_candle_times(
                df,
                ['time'],
                time_as_epoch=False,
                use_client_tz=True,
                client_tz=client_tz,
            )

        self.assertEqual(df.attrs['_tz_used_name'], str(client_tz))
        self.assertEqual(list(df['time']), ['2023-12-31 18:00', '2023-12-31 18:01', '2023-12-31 18:02'])
 
if __name__ == '__main__':
    try:
        unittest.main(exit=False)
        with open("test_service_results.txt", "w") as f:
            f.write("SUCCESS")
    except Exception as e:
        with open("test_service_results.txt", "w") as f:
            f.write(f"FAILURE: {e}")
