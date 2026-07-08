"""Tests for fetch_candles: core success paths, error handling, and datetime queries.

Covers:
  - Basic success (OHLCV, spread, volume, timezone, metadata)
  - Forming candle behaviour (include_incomplete, live-tail trim)
  - Session gap detection and quality filtering
  - Error paths (invalid timeframe/limit, symbol error, no data, stale data)
  - Start / end datetime queries
  - Top-level exception handling
"""

import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

from ._helpers import (
    _UTC, _NOW_TS,
    _mt5_mock,
    _mock_symbol_guard, _mock_symbol_guard_error,
    _make_rates, _make_rates_array,
    _DS, _GUARD, _RATES_FROM, _RATES_RANGE,
    _CACHED_INFO, _RESOLVE_CTZ, _PARSE_START,
    _ESTIMATE_WARMUP, _MT5_CONFIG,
)

from mtdata.services.data_service import fetch_candles


class TestFetchCandlesCore(unittest.TestCase):
    """Core success paths and error handling for fetch_candles."""

    # ------------------------------------------------------------------ #
    # Success paths                                                        #
    # ------------------------------------------------------------------ #

    @patch(_MT5_CONFIG)
    @patch(_RATES_FROM)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=0)
    @patch(_GUARD, _mock_symbol_guard)
    def test_basic_success(self, mock_warmup, mock_ctz, mock_info, mock_from, mock_cfg):
        mock_cfg.get_time_offset_seconds.return_value = 0
        mock_from.return_value = _make_rates(10, step=3600)
        result = fetch_candles('EURUSD', limit=5)
        self.assertTrue(result.get('success'))
        self.assertEqual(result['candles'], 5)
        self.assertEqual(result['symbol'], 'EURUSD')
        self.assertEqual(result['timeframe'], 'H1')
        self.assertEqual(result['volume_type'], 'tick_count')
        self.assertRegex(result["data"][0]["time"], r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}Z$")

    @patch(_MT5_CONFIG)
    @patch(_RATES_FROM)
    @patch(
        _CACHED_INFO,
        return_value=SimpleNamespace(
            digits=5,
            currency_profit="USD",
            chart_mode=0,
        ),
    )
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=0)
    @patch(_GUARD, _mock_symbol_guard)
    def test_candles_include_price_currency(self, mock_warmup, mock_ctz, mock_info, mock_from, mock_cfg):
        mock_cfg.get_time_offset_seconds.return_value = 0
        mock_from.return_value = _make_rates(10, step=3600)
        result = fetch_candles('EURUSD', limit=5)
        self.assertTrue(result.get('success'))
        self.assertEqual(result["price_currency"], "USD")
        self.assertEqual(result["price_basis"], "bid")

    @patch(_MT5_CONFIG)
    @patch(_RATES_FROM)
    @patch(_CACHED_INFO, return_value=SimpleNamespace(digits=5))
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=0)
    @patch(_GUARD, _mock_symbol_guard)
    def test_candle_prices_round_to_symbol_digits(
        self,
        mock_warmup,
        mock_ctz,
        mock_info,
        mock_from,
        mock_cfg,
    ):
        mock_cfg.get_time_offset_seconds.return_value = 0
        rates = _make_rates(10, step=3600)
        for row in rates:
            row.update(
                {
                    "open": 1.1736900000000001,
                    "high": 1.1745600000000002,
                    "low": 1.1723400000000002,
                    "close": 1.1736900000000001,
                }
            )
        mock_from.return_value = rates

        result = fetch_candles('EURUSD', limit=5, ohlcv='OHLC')

        self.assertTrue(result.get('success'))
        row = result['data'][0]
        self.assertEqual(row['open'], 1.17369)
        self.assertEqual(row['high'], 1.17456)
        self.assertEqual(row['low'], 1.17234)
        self.assertEqual(row['close'], 1.17369)

    @patch(_MT5_CONFIG)
    @patch(_RATES_FROM)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=0)
    @patch(_GUARD, _mock_symbol_guard)
    def test_time_as_epoch(self, mock_warmup, mock_ctz, mock_info, mock_from, mock_cfg):
        mock_cfg.get_time_offset_seconds.return_value = 0
        mock_from.return_value = _make_rates(10)
        result = fetch_candles('EURUSD', limit=5, time_as_epoch=True)
        self.assertTrue(result.get('success'))
        for row in result.get('data', []):
            self.assertIsInstance(row.get('time'), (int, float))

    @patch(_MT5_CONFIG)
    @patch(_RATES_FROM)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=0)
    @patch(_GUARD, _mock_symbol_guard)
    def test_ohlcv_filter_close_only(self, mock_warmup, mock_ctz, mock_info, mock_from, mock_cfg):
        mock_cfg.get_time_offset_seconds.return_value = 0
        mock_from.return_value = _make_rates(10)
        result = fetch_candles('EURUSD', limit=5, ohlcv='C')
        self.assertTrue(result.get('success'))
        row_keys = set(result['data'][0].keys())
        self.assertIn('close', row_keys)
        self.assertNotIn('open', row_keys)
        self.assertNotIn('high', row_keys)

    @patch(_MT5_CONFIG)
    @patch(_RATES_FROM)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=0)
    @patch(_GUARD, _mock_symbol_guard)
    def test_ohlcv_filter_oh(self, mock_warmup, mock_ctz, mock_info, mock_from, mock_cfg):
        mock_cfg.get_time_offset_seconds.return_value = 0
        mock_from.return_value = _make_rates(10)
        result = fetch_candles('EURUSD', limit=5, ohlcv='OH')
        self.assertTrue(result.get('success'))
        row_keys = set(result['data'][0].keys())
        self.assertIn('open', row_keys)
        self.assertIn('high', row_keys)
        self.assertNotIn('low', row_keys)

    def test_ohlcv_filter_rejects_invalid_value(self):
        result = fetch_candles('EURUSD', limit=5, ohlcv='invalid')

        self.assertIn('Invalid ohlcv value', result.get('error', ''))

    @patch(_MT5_CONFIG)
    @patch(_RATES_FROM)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=0)
    @patch(_GUARD, _mock_symbol_guard)
    def test_default_candle_payload_excludes_spread(
        self,
        mock_warmup,
        mock_ctz,
        mock_info,
        mock_from,
        mock_cfg,
    ):
        mock_cfg.get_time_offset_seconds.return_value = 0
        mock_from.return_value = _make_rates(10)
        result = fetch_candles('EURUSD', limit=5)
        self.assertTrue(result.get('success'))
        row_keys = set(result['data'][0].keys())
        self.assertNotIn('spread', row_keys)

    @patch(_MT5_CONFIG)
    @patch(_RATES_FROM)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=0)
    @patch(_GUARD, _mock_symbol_guard)
    def test_include_spread_appends_spread_column(
        self,
        mock_warmup,
        mock_ctz,
        mock_info,
        mock_from,
        mock_cfg,
    ):
        mock_cfg.get_time_offset_seconds.return_value = 0
        mock_from.return_value = _make_rates(10)
        result = fetch_candles('EURUSD', limit=5, ohlcv='C', include_spread=True)
        self.assertTrue(result.get('success'))
        row_keys = list(result['data'][0].keys())
        self.assertEqual(row_keys, ['time', 'close', 'spread'])

    @patch(f'{_DS}.fetch_ticks')
    @patch(_MT5_CONFIG)
    @patch(_RATES_FROM)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=0)
    @patch(_GUARD, _mock_symbol_guard)
    def test_estimated_spread_marks_row_and_payload_source(
        self,
        mock_warmup,
        mock_ctz,
        mock_info,
        mock_from,
        mock_cfg,
        mock_fetch_ticks,
    ):
        mock_cfg.get_time_offset_seconds.return_value = 0
        mock_from.return_value = _make_rates(10, spread=0)
        mock_fetch_ticks.return_value = {
            'stats': {
                'spread': {
                    'mean': 0.00009,
                },
            },
        }

        result = fetch_candles('EURUSD', limit=5, ohlcv='C', include_spread=True)

        self.assertTrue(result.get('success'))
        row = result['data'][0]
        self.assertEqual(row['spread'], 0.00009)
        self.assertEqual(row['spread_source'], 'tick_stats')
        self.assertTrue(result['spread_estimated'])
        self.assertEqual(result['spread_source'], 'tick_stats')
        self.assertEqual(result['spread_unit'], 'price')

    @patch(_MT5_CONFIG)
    @patch(_RATES_FROM)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=0)
    @patch(_GUARD, _mock_symbol_guard)
    def test_real_volume_included(self, mock_warmup, mock_ctz, mock_info, mock_from, mock_cfg):
        mock_cfg.get_time_offset_seconds.return_value = 0
        mock_from.return_value = _make_rates(10, real_vol=500)
        result = fetch_candles('EURUSD', limit=5)
        self.assertTrue(result.get('success'))
        row_keys = set(result['data'][0].keys())
        self.assertIn('real_volume', row_keys)

    @patch(_MT5_CONFIG)
    @patch(_RATES_FROM)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=0)
    @patch(_GUARD, _mock_symbol_guard)
    def test_zero_tick_volume_excluded(self, mock_warmup, mock_ctz, mock_info, mock_from, mock_cfg):
        mock_cfg.get_time_offset_seconds.return_value = 0
        mock_from.return_value = _make_rates(10, tick_vol=0)
        result = fetch_candles('EURUSD', limit=5)
        self.assertTrue(result.get('success'))
        row_keys = set(result['data'][0].keys())
        self.assertNotIn('tick_volume', row_keys)

    @patch(_MT5_CONFIG)
    @patch(_RATES_FROM)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=0)
    @patch(_GUARD, _mock_symbol_guard)
    def test_omits_legacy_last_candle_open_flag(self, mock_warmup, mock_ctz, mock_info, mock_from, mock_cfg):
        mock_cfg.get_time_offset_seconds.return_value = 0
        mock_from.return_value = _make_rates(10, step=3600)
        result = fetch_candles('EURUSD', timeframe='H1', limit=5)
        self.assertTrue(result.get('success'))
        self.assertNotIn('last_candle_open', result)
        self.assertTrue(result['has_forming_candle'])

    @patch(_MT5_CONFIG)
    @patch(_RATES_FROM)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=0)
    @patch(_GUARD, _mock_symbol_guard)
    def test_default_excludes_incomplete_last_candle(self, mock_warmup, mock_ctz, mock_info, mock_from, mock_cfg):
        mock_cfg.get_time_offset_seconds.return_value = 0
        mock_from.return_value = _make_rates(10, step=3600)
        result = fetch_candles('EURUSD', timeframe='H1', limit=5)
        self.assertTrue(result.get('success'))
        self.assertEqual(result['candles'], 5)
        self.assertEqual(result['candles_requested'], 5)
        self.assertEqual(result['candles_excluded'], 0)
        self.assertNotIn('last_candle_open', result)
        self.assertTrue(result['has_forming_candle'])
        self.assertEqual(result['forming_candle_status'], 'skipped')
        self.assertFalse(result['forming_candle_included'])
        self.assertTrue(result['forming_candle_skipped'])
        self.assertEqual(len(result['data']), 5)

    @patch(_MT5_CONFIG)
    @patch(_RATES_FROM)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=0)
    @patch(_GUARD, _mock_symbol_guard)
    def test_include_incomplete_keeps_open_last_candle(self, mock_warmup, mock_ctz, mock_info, mock_from, mock_cfg):
        mock_cfg.get_time_offset_seconds.return_value = 0
        mock_from.return_value = _make_rates(10, step=3600)
        result = fetch_candles('EURUSD', timeframe='H1', limit=5, include_incomplete=True)
        self.assertTrue(result.get('success'))
        self.assertEqual(result['candles'], 5)
        self.assertEqual(result['candles_requested'], 5)
        self.assertEqual(result['candles_excluded'], 0)
        self.assertNotIn('last_candle_open', result)
        self.assertTrue(result['has_forming_candle'])
        self.assertEqual(result['forming_candle_status'], 'included')
        self.assertTrue(result['forming_candle_included'])
        self.assertFalse(result['forming_candle_skipped'])
        self.assertEqual(len(result['data']), 5)
        self.assertNotIn('is_forming', result['data'][-1])

    @patch(_MT5_CONFIG)
    @patch(_RATES_FROM)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=0)
    @patch(_GUARD, _mock_symbol_guard)
    def test_recent_broker_tick_drives_incomplete_bar_trim(self, mock_warmup, mock_ctz, mock_info, mock_from, mock_cfg):
        base_ts = _NOW_TS
        mock_cfg.get_server_tz.return_value = None
        mock_cfg.get_time_offset_seconds.return_value = 0
        mock_from.return_value = _make_rates(6, base_ts=base_ts, step=3600)
        with patch(f'{_DS}._utc_epoch_seconds', return_value=base_ts - 60), \
             patch(f'{_DS}.mt5.symbol_info_tick', return_value=SimpleNamespace(time=base_ts + 120)):
            result = fetch_candles('EURUSD', timeframe='H1', limit=5, time_as_epoch=True)
        self.assertTrue(result.get('success'))
        returned_times = [row['time'] for row in result.get('data', [])]
        self.assertNotIn(base_ts, returned_times)
        self.assertEqual(returned_times[-1], base_ts - 3600)
        self.assertNotIn('last_candle_open', result)
        self.assertTrue(result['has_forming_candle'])
        self.assertEqual(result['forming_candle_status'], 'skipped')
        self.assertFalse(result['forming_candle_included'])
        self.assertTrue(result['forming_candle_skipped'])

    @patch(_MT5_CONFIG)
    @patch(_RATES_FROM)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=0)
    @patch(_GUARD, _mock_symbol_guard)
    def test_numpy_rates_trim_live_tail_before_limit_window(self, mock_warmup, mock_ctz, mock_info, mock_from, mock_cfg):
        base_ts = _NOW_TS
        mock_cfg.get_server_tz.return_value = None
        mock_cfg.get_time_offset_seconds.return_value = 0
        mock_from.return_value = _make_rates_array(6, base_ts=base_ts, step=3600)
        with patch(f'{_DS}._utc_epoch_seconds', return_value=base_ts - 60), \
             patch(f'{_DS}.mt5.symbol_info_tick', return_value=SimpleNamespace(time=base_ts + 120)):
            result = fetch_candles('EURUSD', timeframe='H1', limit=5, time_as_epoch=True)
        self.assertTrue(result.get('success'))
        returned_times = [row['time'] for row in result.get('data', [])]
        self.assertEqual(len(returned_times), 5)
        self.assertEqual(returned_times[0], base_ts - (5 * 3600))
        self.assertEqual(returned_times[-1], base_ts - 3600)
        self.assertEqual(result['candles'], 5)
        self.assertEqual(result['candles_excluded'], 0)
        self.assertEqual(result['incomplete_candles_skipped'], 1)
        self.assertTrue(result['has_forming_candle'])
        self.assertEqual(result['forming_candle_status'], 'skipped')
        self.assertFalse(result['forming_candle_included'])
        self.assertTrue(result['forming_candle_skipped'])

    @patch(_MT5_CONFIG)
    @patch(_RATES_FROM)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=0)
    @patch(_GUARD, _mock_symbol_guard)
    def test_exposes_requested_and_excluded_counts_when_live_tail_is_trimmed(self, mock_warmup, mock_ctz, mock_info, mock_from, mock_cfg):
        mock_cfg.get_time_offset_seconds.return_value = 0
        mock_from.return_value = _make_rates(5, step=3600)
        result = fetch_candles('EURUSD', timeframe='H1', limit=5)
        self.assertTrue(result.get('success'))
        self.assertEqual(result['candles_requested'], 5)
        self.assertEqual(result['candles'], 4)
        self.assertEqual(result['candles_excluded'], 1)
        self.assertEqual(result['incomplete_candles_skipped'], 1)
        self.assertEqual(
            result['candle_counts']['excluded'],
            {
                'forming_bar': 1,
                'indicator_warmup': 0,
                'quality_filtered': 0,
                'window_or_source_shortfall': 0,
                'total': 1,
            },
        )
        self.assertTrue(result['has_forming_candle'])
        self.assertEqual(result['forming_candle_status'], 'skipped')
        self.assertFalse(result['forming_candle_included'])
        self.assertTrue(result['forming_candle_skipped'])
        self.assertIn('include_incomplete=true', result['hint'])
        self.assertNotIn('last_candle_open', result)

    @patch(_MT5_CONFIG)
    @patch(_RATES_FROM)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=0)
    @patch(_GUARD, _mock_symbol_guard)
    def test_stale_broker_tick_does_not_mark_closed_bar_as_live(self, mock_warmup, mock_ctz, mock_info, mock_from, mock_cfg):
        base_ts = _NOW_TS
        mock_cfg.get_server_tz.return_value = None
        mock_cfg.get_time_offset_seconds.return_value = 0
        mock_from.return_value = _make_rates(6, base_ts=base_ts, step=3600)
        with patch(f'{_DS}._utc_epoch_seconds', return_value=base_ts + 7200), \
             patch(f'{_DS}.mt5.symbol_info_tick', return_value=SimpleNamespace(time=base_ts + 120)):
            result = fetch_candles('EURUSD', timeframe='H1', limit=5, time_as_epoch=True)
        self.assertTrue(result.get('success'))
        returned_times = [row['time'] for row in result.get('data', [])]
        self.assertIn(base_ts, returned_times)
        self.assertNotIn('last_candle_open', result)
        self.assertFalse(result['has_forming_candle'])

    @patch(_MT5_CONFIG)
    @patch(_RATES_FROM)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=0)
    @patch(_GUARD, _mock_symbol_guard)
    def test_meta_runtime_timezone_not_in_raw_output(self, mock_warmup, mock_ctz, mock_info, mock_from, mock_cfg):
        mock_cfg.server_tz_name = "Europe/Nicosia"
        mock_cfg.client_tz_name = None
        mock_cfg.get_server_tz.return_value = ZoneInfo("Europe/Nicosia")
        mock_cfg.get_client_tz.return_value = None
        mock_cfg.get_time_offset_seconds.return_value = 7200
        mock_from.return_value = _make_rates(10)
        result = fetch_candles('EURUSD', limit=5)
        self.assertIsInstance(result.get('meta'), dict)
        self.assertIsNone(result['meta'].get('runtime'))
        self.assertEqual(result.get('timezone'), "UTC")

    @patch(_MT5_CONFIG)
    @patch(_RATES_FROM)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=ZoneInfo("America/Chicago"))
    @patch(_ESTIMATE_WARMUP, return_value=0)
    @patch(_GUARD, _mock_symbol_guard)
    def test_meta_runtime_not_in_raw_with_client_tz(self, mock_warmup, mock_ctz, mock_info, mock_from, mock_cfg):
        mock_cfg.server_tz_name = "Europe/Nicosia"
        mock_cfg.client_tz_name = "America/Chicago"
        mock_cfg.get_server_tz.return_value = ZoneInfo("Europe/Nicosia")
        mock_cfg.get_client_tz.return_value = ZoneInfo("America/Chicago")
        mock_cfg.get_time_offset_seconds.return_value = 0
        mock_from.return_value = _make_rates(10)
        result = fetch_candles('EURUSD', limit=5)
        self.assertIsInstance(result.get('meta'), dict)
        self.assertIsNone(result['meta'].get('runtime'))
        self.assertEqual(result.get('timezone'), "America/Chicago")

    @patch(_MT5_CONFIG)
    @patch(_RATES_FROM)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=0)
    @patch(_GUARD, _mock_symbol_guard)
    def test_meta_includes_query_diagnostics(self, mock_warmup, mock_ctz, mock_info, mock_from, mock_cfg):
        mock_cfg.get_time_offset_seconds.return_value = 0
        mock_from.return_value = _make_rates(10, step=3600)
        result = fetch_candles('EURUSD', limit=5)
        self.assertTrue(result.get('success'))
        self.assertEqual(result['candles_requested'], 5)
        self.assertEqual(result['requested_limit'], 5)
        self.assertEqual(result['returned_count'], 5)
        self.assertEqual(result['candles_excluded'], 0)
        self.assertIn("as_of", result)
        self.assertEqual(result["data_window"]["requested_limit"], 5)
        self.assertEqual(result["data_window"]["returned_count"], 5)
        self.assertTrue(result["data_window"]["latest_bar_complete"])
        self.assertIn("latest_bar_age_seconds", result["data_window"])
        self.assertEqual(
            result["data_window"]["latest_bar_age_metric"],
            "latest_completed_bar_close_age_seconds",
        )
        self.assertLess(result["data_window"]["latest_bar_age_seconds"], 120.0)
        diagnostics = result['meta']['diagnostics']
        query = diagnostics['query']
        self.assertEqual(query['requested_bars'], 5)
        self.assertEqual(query['raw_bars_fetched'], 10)
        self.assertEqual(result['candles'], 5)
        self.assertIn('latency_ms', query)
        self.assertIn('warmup_retry', query)
        freshness = diagnostics['freshness']
        self.assertEqual(
            set(freshness.keys()),
            {
                'last_bar_epoch',
                'expected_end_epoch',
                'freshness_cutoff_epoch',
                'data_freshness_seconds',
                'data_freshness_anchor',
                'data_freshness_metric',
                'last_bar_within_policy_window',
            },
        )
        self.assertTrue(freshness['last_bar_within_policy_window'])
        self.assertEqual(freshness['data_freshness_anchor'], 'wall_clock')
        self.assertEqual(
            freshness['data_freshness_metric'],
            'last_completed_bar_age_seconds',
        )
        self.assertEqual(diagnostics['indicators']['requested'], False)
        self.assertEqual(diagnostics['session_gaps']['expected_bar_seconds'], 3600.0)

    @patch(_MT5_CONFIG)
    @patch(_RATES_FROM)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=0)
    @patch(_GUARD, _mock_symbol_guard)
    def test_timezone_utc_in_payload(self, mock_warmup, mock_ctz, mock_info, mock_from, mock_cfg):
        mock_cfg.get_time_offset_seconds.return_value = 0
        mock_from.return_value = _make_rates(10)
        result = fetch_candles('EURUSD', limit=5)
        self.assertEqual(result.get('timezone'), 'UTC')

    @patch(_MT5_CONFIG)
    @patch(_RATES_FROM)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=0)
    @patch(_GUARD, _mock_symbol_guard)
    def test_session_gap_annotation(self, mock_warmup, mock_ctz, mock_info, mock_from, mock_cfg):
        mock_cfg.get_time_offset_seconds.return_value = 0
        t0 = _NOW_TS
        rates = [
            {
                'time': t0,
                'open': 1.1000,
                'high': 1.2000,
                'low': 1.0000,
                'close': 1.1500,
                'tick_volume': 100,
                'real_volume': 0,
                'spread': 1,
            },
            {
                'time': t0 + 3600,
                'open': 1.1001,
                'high': 1.2001,
                'low': 1.0001,
                'close': 1.1501,
                'tick_volume': 101,
                'real_volume': 0,
                'spread': 1,
            },
            {
                'time': t0 + 7200,
                'open': 1.1002,
                'high': 1.2002,
                'low': 1.0002,
                'close': 1.1502,
                'tick_volume': 102,
                'real_volume': 0,
                'spread': 1,
            },
            {
                # 9-hour session jump from prior bar (missing 8 H1 bars)
                'time': t0 + 39600,
                'open': 1.1003,
                'high': 1.2003,
                'low': 1.0003,
                'close': 1.1503,
                'tick_volume': 103,
                'real_volume': 0,
                'spread': 1,
            },
            {
                'time': t0 + 43200,
                'open': 1.1004,
                'high': 1.2004,
                'low': 1.0004,
                'close': 1.1504,
                'tick_volume': 104,
                'real_volume': 0,
                'spread': 1,
            },
        ]
        mock_from.return_value = rates

        result = fetch_candles('EURUSD', timeframe='H1', limit=5)
        self.assertTrue(result.get('success'))
        self.assertIn('session_gaps', result)
        self.assertEqual(len(result['session_gaps']), 1)
        gap = result['session_gaps'][0]
        self.assertEqual(gap['gap_seconds'], 32400.0)
        self.assertEqual(gap['expected_bar_seconds'], 3600.0)
        self.assertEqual(gap['missing_bars_est'], 8)
        self.assertIn('context', gap)
        self.assertIn('warnings', result)
        self.assertTrue(any('session gap' in w.lower() for w in result['warnings']))
        self.assertTrue(any('example gap' in w.lower() for w in result['warnings']))

    @patch(_MT5_CONFIG)
    @patch(_RATES_FROM)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=0)
    @patch("mtdata.services.data_service._collect_session_gaps", return_value=([], "Session gap diagnostics unavailable."))
    @patch(_GUARD, _mock_symbol_guard)
    def test_session_gap_diagnostic_failure_is_surfaced(
        self,
        mock_collect,
        mock_warmup,
        mock_ctz,
        mock_info,
        mock_from,
        mock_cfg,
    ):
        mock_cfg.get_time_offset_seconds.return_value = 0
        mock_from.return_value = _make_rates(10)

        result = fetch_candles('EURUSD', limit=5)

        self.assertTrue(result.get('success'))
        self.assertIn('warnings', result)
        self.assertIn("Session gap diagnostics unavailable.", result['warnings'])
        self.assertEqual(
            result['meta']['diagnostics']['session_gaps']['warning'],
            "Session gap diagnostics unavailable.",
        )

    @patch(_MT5_CONFIG)
    @patch(_RATES_FROM)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=0)
    @patch(_GUARD, _mock_symbol_guard)
    def test_quality_filter_removes_malformed_rows_and_warns(
        self,
        mock_warmup,
        mock_ctz,
        mock_info,
        mock_from,
        mock_cfg,
    ):
        mock_cfg.get_time_offset_seconds.return_value = 0
        rates = _make_rates(8, step=3600)
        rates[1]['close'] = float('nan')
        rates[2]['low'] = rates[2]['high'] + 0.0001
        rates[4]['time'] = rates[3]['time']
        rates[6]['time'] = rates[5]['time'] - 7200
        mock_from.return_value = rates

        result = fetch_candles(
            'EURUSD',
            timeframe='H1',
            limit=8,
            time_as_epoch=True,
            include_incomplete=True,
        )

        self.assertTrue(result.get('success'))
        self.assertEqual(result['candles'], 4)
        times = [float(row['time']) for row in result['data']]
        self.assertEqual(times, sorted(times))
        self.assertEqual(len(times), len(set(times)))
        self.assertEqual(result['meta']['diagnostics']['query']['quality_rows_removed'], 4)
        self.assertTrue(any('non-finite time/OHLC values' in str(w) for w in result.get('warnings', [])))
        self.assertTrue(any('inconsistent OHLC ranges' in str(w) for w in result.get('warnings', [])))
        self.assertTrue(any('duplicate candle timestamp' in str(w) for w in result.get('warnings', [])))
        self.assertTrue(any('Sorted candle rows by timestamp' in str(w) for w in result.get('warnings', [])))

    # ------------------------------------------------------------------ #
    # Error paths                                                         #
    # ------------------------------------------------------------------ #

    def test_invalid_timeframe(self):
        result = fetch_candles('EURUSD', timeframe='INVALID')
        self.assertIn('error', result)
        self.assertIn('Invalid timeframe', result['error'])

    def test_negative_limit_returns_friendly_error(self):
        result = fetch_candles('EURUSD', limit=-5)
        self.assertEqual(result, {'error': 'limit must be greater than 0.'})

    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_GUARD, _mock_symbol_guard_error)
    def test_symbol_guard_error(self, mock_info):
        result = fetch_candles('INVALID')
        self.assertIn('error', result)

    @patch(_MT5_CONFIG)
    @patch(_RATES_FROM, return_value=None)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=0)
    @patch(_GUARD, _mock_symbol_guard)
    def test_rates_none(self, mock_warmup, mock_ctz, mock_info, mock_from, mock_cfg):
        mock_cfg.get_time_offset_seconds.return_value = 0
        _mt5_mock.last_error.return_value = (-1, 'No data')
        result = fetch_candles('EURUSD', limit=5)
        self.assertIn('error', result)
        self.assertIn('Failed to get rates', result['error'])

    @patch(_MT5_CONFIG)
    @patch(_RATES_FROM, return_value=None)
    @patch(_CACHED_INFO, return_value=None)
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=0)
    @patch(_GUARD, _mock_symbol_guard)
    def test_generic_symbol_failure_is_rewritten(self, mock_warmup, mock_ctz, mock_info, mock_from, mock_cfg):
        mock_cfg.get_time_offset_seconds.return_value = 0
        _mt5_mock.last_error.return_value = (-1, 'Terminal: Call failed')
        result = fetch_candles('NOTASYMBOL', limit=5)
        self.assertEqual(
            result['error'],
            "Symbol 'NOTASYMBOL' was not found or is not available in MT5. "
            "Use symbols_list(search_term='NOTASYMBOL') to find broker-specific names and suffixes.",
        )

    @patch(_MT5_CONFIG)
    @patch(_RATES_FROM, return_value=[])
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=0)
    @patch(_GUARD, _mock_symbol_guard)
    def test_rates_empty(self, mock_warmup, mock_ctz, mock_info, mock_from, mock_cfg):
        mock_cfg.get_time_offset_seconds.return_value = 0
        result = fetch_candles('EURUSD', limit=5)
        self.assertIn('error', result)
        self.assertIn('No data', result['error'])

    @patch(_MT5_CONFIG)
    @patch(_PARSE_START)
    @patch(_RATES_FROM)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=0)
    @patch(_GUARD, _mock_symbol_guard)
    def test_stale_rates_are_rejected(self, mock_warmup, mock_ctz, mock_info, mock_from, mock_parse, mock_cfg):
        to_date = datetime(2025, 1, 2, tzinfo=_UTC)
        mock_parse.return_value = to_date
        mock_cfg.get_time_offset_seconds.return_value = 0
        mock_from.return_value = _make_rates(5, base_ts=to_date.timestamp() - (10 * 60 * 60), step=60 * 60)
        with patch(f'{_DS}.FETCH_RETRY_ATTEMPTS', 2), patch(f'{_DS}.FETCH_RETRY_DELAY', 0):
            result = fetch_candles('EURUSD', limit=5, end='2025-01-02')
        self.assertIn('error', result)
        self.assertIn('Data appears stale for EURUSD H1', result['error'])
        self.assertIn('allow_stale=true', result['error'])
        self.assertEqual(
            result['details']['diagnostics']['freshness'],
            {
                'last_bar_epoch': float(to_date.timestamp() - (10 * 60 * 60)),
                'expected_end_epoch': float(to_date.timestamp()),
                'freshness_cutoff_epoch': float(to_date.timestamp() - (4 * 60 * 60)),
                'data_freshness_seconds': float(10 * 60 * 60),
                'last_bar_within_policy_window': False,
            },
        )

    @patch(_MT5_CONFIG)
    @patch(_PARSE_START)
    @patch(_RATES_FROM)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=0)
    @patch(_GUARD, _mock_symbol_guard)
    def test_allow_stale_returns_latest_available_rates(
        self,
        mock_warmup,
        mock_ctz,
        mock_info,
        mock_from,
        mock_parse,
        mock_cfg,
    ):
        to_date = datetime(2025, 1, 2, tzinfo=_UTC)
        mock_parse.return_value = to_date
        mock_cfg.get_time_offset_seconds.return_value = 0
        mock_from.return_value = _make_rates(5, base_ts=to_date.timestamp() - (10 * 60 * 60), step=60 * 60)

        result = fetch_candles(
            'EURUSD',
            limit=5,
            end='2025-01-02',
            allow_stale=True,
        )

        self.assertTrue(result['success'])
        self.assertEqual(result['candles'], 5)
        self.assertNotIn('error', result)

    # ------------------------------------------------------------------ #
    # Start / End datetime queries                                         #
    # ------------------------------------------------------------------ #

    @patch(_MT5_CONFIG)
    @patch(_RATES_RANGE)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=0)
    @patch(_GUARD, _mock_symbol_guard)
    def test_start_and_end_datetime(self, mock_warmup, mock_ctz, mock_info, mock_range, mock_cfg):
        mock_cfg.get_time_offset_seconds.return_value = 0
        base = datetime(2025, 1, 1, tzinfo=_UTC).timestamp()
        mock_range.return_value = _make_rates(20, base_ts=base + 20 * 60, step=60)
        result = fetch_candles('EURUSD', limit=100, start='2025-01-01', end='2025-01-01 00:20')
        self.assertTrue(result.get('success'))

    @patch(_MT5_CONFIG)
    @patch(_RATES_RANGE)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=0)
    @patch(_GUARD, _mock_symbol_guard)
    def test_start_only(self, mock_warmup, mock_ctz, mock_info, mock_range, mock_cfg):
        mock_cfg.get_time_offset_seconds.return_value = 0
        mock_range.return_value = _make_rates(20)
        result = fetch_candles('EURUSD', limit=5, start='2025-01-01')
        self.assertTrue(result.get('success'))

    @patch(_MT5_CONFIG)
    @patch(_RATES_FROM)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=0)
    @patch(_GUARD, _mock_symbol_guard)
    def test_end_only(self, mock_warmup, mock_ctz, mock_info, mock_from, mock_cfg):
        mock_cfg.get_time_offset_seconds.return_value = 0
        mock_from.return_value = _make_rates(20)
        result = fetch_candles('EURUSD', limit=5, end='2025-01-02')
        self.assertTrue(result.get('success'))

    # ------------------------------------------------------------------ #
    # Exception handling                                                   #
    # ------------------------------------------------------------------ #

    @patch(_CACHED_INFO, side_effect=RuntimeError("boom"))
    def test_exception_returns_error(self, mock_info):
        result = fetch_candles('EURUSD', 'H1')
        self.assertIn('error', result)
        self.assertIn('Error getting rates', result['error'])
        self.assertIn('RuntimeError', result['error'])
        detail = result.get('error_detail') or {}
        self.assertEqual(detail.get('operation'), 'fetch_candles')
        self.assertEqual(detail.get('symbol'), 'EURUSD')
        self.assertEqual(detail.get('timeframe'), 'H1')


if __name__ == '__main__':
    unittest.main()
