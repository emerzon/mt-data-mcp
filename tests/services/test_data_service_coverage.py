"""Comprehensive tests for mtdata.services.data_service.

Targets ~294 uncovered lines with 50+ tests covering public methods,
private helpers, data transformation, validation, and error handling.
"""

import os
import sys
import unittest
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any, Iterator, Tuple
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import numpy as np

# Ensure src is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

# Mock mt5 module before importing data_service
_mt5_mock = MagicMock()
sys.modules['MetaTrader5'] = _mt5_mock

import pandas as pd  # noqa: E402

from mtdata.services.data_service import (  # noqa: E402
    _build_rates_df,
    _fetch_rates_with_warmup,
    _shift_rate_times,
    _trim_df_to_target,
    fetch_candles,
    fetch_ticks,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_UTC = timezone.utc
# Keep default fixture data close to the test run so freshness checks remain stable.
_NOW = datetime.now(_UTC).replace(second=0, microsecond=0)
_NOW_TS = _NOW.timestamp()


@contextmanager
def _mock_symbol_guard(*args: Any, **kwargs: Any) -> Iterator[Tuple[None, MagicMock]]:
    """Context manager stub that always succeeds."""
    yield None, MagicMock()


@contextmanager
def _mock_symbol_guard_error(*args: Any, **kwargs: Any) -> Iterator[Tuple[str, None]]:
    """Context manager stub that returns an error."""
    yield "Symbol INVALID not found", None


def _make_rates(n: int, *, base_ts: float = _NOW_TS, step: int = 60,
                tick_vol: int = 100, real_vol: int = 0) -> list:
    """Generate a list of rate dicts mimicking MT5 structured array rows."""
    rates = []
    for i in range(n):
        rates.append({
            'time': base_ts - (n - 1 - i) * step,
            'open': 1.1000 + i * 0.001,
            'high': 1.2000 + i * 0.001,
            'low': 1.0000 + i * 0.001,
            'close': 1.1500 + i * 0.001,
            'tick_volume': tick_vol,
            'real_volume': real_vol,
            'spread': 1,
        })
    return rates


def _make_ticks(n: int, *, base_ts: float = _NOW_TS, step: float = 1.0) -> list:
    """Generate a list of tick dicts."""
    ticks = []
    for i in range(n):
        ticks.append({
            'time': base_ts - (n - 1 - i) * step,
            'bid': 1.1000 + i * 0.0001,
            'ask': 1.1002 + i * 0.0001,
            'last': 1.1001 + i * 0.0001,
            'volume': 1.0,
            'time_msc': (base_ts - (n - 1 - i) * step) * 1000,
            'flags': 0,
            'volume_real': 0.0,
        })
    return ticks


# Patch targets (module-level strings)
_DS = 'mtdata.services.data_service'
_GUARD = f'{_DS}._symbol_ready_guard'
_RATES_FROM = f'{_DS}._mt5_copy_rates_from'
_RATES_RANGE = f'{_DS}._mt5_copy_rates_range'
_TICKS_FROM = f'{_DS}._mt5_copy_ticks_from'
_TICKS_RANGE = f'{_DS}._mt5_copy_ticks_range'
_CACHED_INFO = f'{_DS}.get_symbol_info_cached'
_RESOLVE_CTZ = f'{_DS}._resolve_client_tz'
_PARSE_START = f'{_DS}._parse_start_datetime'
_ESTIMATE_WARMUP = f'{_DS}._estimate_warmup_bars_util'
_APPLY_TI = f'{_DS}._apply_ta_indicators_util'
_SIMPLIFY_EXT = f'{_DS}._simplify_dataframe_rows_ext'
_MT5_CONFIG = f'{_DS}.mt5_config'


# ============================================================================
# _shift_rate_times
# ============================================================================

class TestShiftRateTimes(unittest.TestCase):
    def test_returns_shifted_list_copy_without_mutating_input(self):
        rates = _make_rates(3)
        original_times = [float(row['time']) for row in rates]

        shifted = _shift_rate_times(rates, 3600)

        self.assertIsInstance(shifted, list)
        self.assertIsNot(shifted, rates)
        self.assertIsNot(shifted[0], rates[0])
        self.assertEqual([float(row['time']) for row in rates], original_times)
        self.assertEqual(
            [float(row['time']) for row in shifted],
            [ts + 3600.0 for ts in original_times],
        )

    def test_returns_shifted_structured_array_copy_without_mutating_input(self):
        rates = np.array(
            [(1704067200.0, 1.1), (1704067260.0, 1.2)],
            dtype=[('time', 'f8'), ('close', 'f8')],
        )

        shifted = _shift_rate_times(rates, 120)

        self.assertIsNot(shifted, rates)
        np.testing.assert_array_equal(rates['time'], np.array([1704067200.0, 1704067260.0]))
        np.testing.assert_array_equal(
            shifted['time'],
            np.array([1704067320.0, 1704067380.0]),
        )
        np.testing.assert_array_equal(shifted['close'], rates['close'])

    def test_list_without_time_keys_returns_original_input(self):
        rates = [{'close': 1.1}, {'open': 1.2}, 'passthrough']

        shifted = _shift_rate_times(rates, 60)

        self.assertIs(shifted, rates)

    def test_list_with_non_numeric_time_returns_shifted_copy_and_leaves_failing_row_unshifted(self):
        rates = [
            {'time': 1704067200.0, 'close': 1.1},
            {'time': 'not-a-number', 'close': 1.2},
            {'time': 1704067320.0, 'close': 1.3},
        ]
        original = [row.copy() for row in rates]

        shifted = _shift_rate_times(rates, 60)

        self.assertIsInstance(shifted, list)
        self.assertIsNot(shifted, rates)
        self.assertEqual(rates, original)
        self.assertEqual(shifted[0]['time'], 1704067260.0)
        self.assertEqual(shifted[1]['time'], 'not-a-number')
        self.assertEqual(shifted[2]['time'], 1704067380.0)
        self.assertEqual(shifted[0]['close'], rates[0]['close'])
        self.assertEqual(shifted[1]['close'], rates[1]['close'])
        self.assertEqual(shifted[2]['close'], rates[2]['close'])

    def test_structured_array_with_non_numeric_time_returns_original_untouched_input(self):
        rates = np.array(
            [('1704067200.0', 1.1), ('not-a-number', 1.2)],
            dtype=[('time', 'U32'), ('close', 'f8')],
        )
        original = rates.copy()

        shifted = _shift_rate_times(rates, 60)

        self.assertIs(shifted, rates)
        np.testing.assert_array_equal(rates, original)
        np.testing.assert_array_equal(shifted, original)


# ============================================================================
# _fetch_rates_with_warmup
# ============================================================================

class TestFetchRatesWithWarmup(unittest.TestCase):
    """Tests for the _fetch_rates_with_warmup helper."""

    @patch(_RATES_FROM)
    def test_no_datetime_uses_copy_rates_from(self, mock_from):
        """Default path: no start/end datetime."""
        rates = _make_rates(10)
        mock_from.return_value = rates
        result, err = _fetch_rates_with_warmup(
            'EURUSD', 16385, 'H1', 5, 0, None, None, retry=False, sanity_check=False,
        )
        self.assertIsNone(err)
        self.assertEqual(result, rates)
        mock_from.assert_called_once()

    @patch(_RATES_RANGE)
    @patch(_PARSE_START)
    def test_start_and_end_datetime(self, mock_parse, mock_range):
        """Both start and end provided — uses copy_rates_range."""
        t1 = datetime(2025, 1, 1, tzinfo=_UTC)
        t2 = datetime(2025, 1, 2, tzinfo=_UTC)
        mock_parse.side_effect = [t1, t2]
        rates = _make_rates(5, base_ts=t2.timestamp())
        mock_range.return_value = rates
        result, err = _fetch_rates_with_warmup(
            'EURUSD', 16385, 'H1', 5, 0, '2025-01-01', '2025-01-02',
            retry=False, sanity_check=False,
        )
        self.assertIsNone(err)
        self.assertIsNotNone(result)

    @patch(_RATES_RANGE)
    @patch(_PARSE_START)
    def test_start_and_end_invalid_from(self, mock_parse, mock_range):
        """start_datetime fails to parse."""
        mock_parse.side_effect = [None, datetime(2025, 1, 2, tzinfo=_UTC)]
        result, err = _fetch_rates_with_warmup(
            'EURUSD', 16385, 'H1', 5, 0, 'bad', '2025-01-02',
            retry=False, sanity_check=False,
        )
        self.assertIsNone(result)
        self.assertIn('Invalid date', err)

    @patch(_RATES_RANGE)
    @patch(_PARSE_START)
    def test_start_and_end_invalid_to(self, mock_parse, mock_range):
        """end_datetime fails to parse."""
        mock_parse.side_effect = [datetime(2025, 1, 1, tzinfo=_UTC), None]
        result, err = _fetch_rates_with_warmup(
            'EURUSD', 16385, 'H1', 5, 0, '2025-01-01', 'bad',
            retry=False, sanity_check=False,
        )
        self.assertIsNone(result)
        self.assertIn('Invalid date', err)

    @patch(_RATES_RANGE)
    @patch(_PARSE_START)
    def test_start_after_end_returns_error(self, mock_parse, mock_range):
        """start > end should error."""
        mock_parse.side_effect = [
            datetime(2025, 2, 1, tzinfo=_UTC),
            datetime(2025, 1, 1, tzinfo=_UTC),
        ]
        result, err = _fetch_rates_with_warmup(
            'EURUSD', 16385, 'H1', 5, 0, '2025-02-01', '2025-01-01',
            retry=False, sanity_check=False,
        )
        self.assertIsNone(result)
        self.assertIn('before', err)

    @patch(_RATES_RANGE)
    @patch(_PARSE_START)
    def test_start_only(self, mock_parse, mock_range):
        """Only start_datetime provided."""
        t1 = datetime(2025, 1, 1, tzinfo=_UTC)
        mock_parse.return_value = t1
        rates = _make_rates(5, base_ts=t1.timestamp() + 600)
        mock_range.return_value = rates
        result, err = _fetch_rates_with_warmup(
            'EURUSD', 16385, 'H1', 5, 0, '2025-01-01', None,
            retry=False, sanity_check=False,
        )
        self.assertIsNone(err)
        self.assertEqual(result, rates)

    @patch(_PARSE_START)
    def test_start_only_invalid(self, mock_parse):
        """start_datetime fails to parse (no end)."""
        mock_parse.return_value = None
        result, err = _fetch_rates_with_warmup(
            'EURUSD', 16385, 'H1', 5, 0, 'bad', None,
            retry=False, sanity_check=False,
        )
        self.assertIsNone(result)
        self.assertIn('Invalid date', err)

    @patch(_RATES_RANGE)
    @patch(_PARSE_START)
    def test_start_only_unknown_timeframe_seconds(self, mock_parse, mock_range):
        """start only with a timeframe whose seconds can't be resolved."""
        mock_parse.return_value = datetime(2025, 1, 1, tzinfo=_UTC)
        with patch(f'{_DS}.TIMEFRAME_SECONDS', {}):
            result, err = _fetch_rates_with_warmup(
                'EURUSD', 16385, 'H1', 5, 0, '2025-01-01', None,
                retry=False, sanity_check=False,
            )
        self.assertIsNone(result)
        self.assertIn('Unable to determine', err)

    @patch(_RATES_RANGE)
    @patch(_PARSE_START)
    def test_start_and_end_unknown_timeframe_seconds(self, mock_parse, mock_range):
        """start/end fetches should fail fast when timeframe seconds are unavailable."""
        mock_parse.side_effect = [
            datetime(2025, 1, 1, tzinfo=_UTC),
            datetime(2025, 1, 2, tzinfo=_UTC),
        ]
        with patch(f'{_DS}.TIMEFRAME_SECONDS', {}):
            result, err = _fetch_rates_with_warmup(
                'EURUSD', 16385, 'H1', 5, 0, '2025-01-01', '2025-01-02',
                retry=False, sanity_check=False,
            )
        self.assertIsNone(result)
        self.assertIn('Unable to determine', err)

    @patch(_RATES_FROM)
    @patch(_PARSE_START)
    def test_end_only(self, mock_parse, mock_from):
        """Only end_datetime provided."""
        t2 = datetime(2025, 1, 2, tzinfo=_UTC)
        mock_parse.return_value = t2
        rates = _make_rates(5, base_ts=t2.timestamp())
        mock_from.return_value = rates
        result, err = _fetch_rates_with_warmup(
            'EURUSD', 16385, 'H1', 5, 0, None, '2025-01-02',
            include_incomplete=True, retry=False, sanity_check=False,
        )
        self.assertIsNone(err)
        self.assertEqual(result, rates)
        mock_from.assert_called_once_with('EURUSD', 16385, t2, 5)

    @patch(_PARSE_START)
    def test_end_only_invalid(self, mock_parse):
        """end_datetime fails to parse (no start)."""
        mock_parse.return_value = None
        result, err = _fetch_rates_with_warmup(
            'EURUSD', 16385, 'H1', 5, 0, None, 'bad',
            retry=False, sanity_check=False,
        )
        self.assertIsNone(result)
        self.assertIn('Invalid date', err)

    @patch(_RATES_FROM)
    def test_no_datetime_unknown_timeframe_seconds(self, mock_from):
        """Default fetches should fail fast when timeframe seconds are unavailable."""
        with patch(f'{_DS}.TIMEFRAME_SECONDS', {}):
            result, err = _fetch_rates_with_warmup(
                'EURUSD', 16385, 'H1', 5, 0, None, None,
                retry=False, sanity_check=False,
            )
        self.assertIsNone(result)
        self.assertIn('Unable to determine', err)

    @patch(_RATES_FROM)
    def test_retry_logic(self, mock_from):
        """Retry returns data on second attempt."""
        mock_from.side_effect = [None, _make_rates(5)]
        with patch(f'{_DS}.FETCH_RETRY_DELAY', 0):
            result, err = _fetch_rates_with_warmup(
                'EURUSD', 16385, 'H1', 5, 0, None, None,
                retry=True, sanity_check=False,
            )
        self.assertIsNone(err)
        self.assertIsNotNone(result)
        self.assertEqual(mock_from.call_count, 2)

    @patch(_RATES_FROM)
    def test_sanity_check_pass(self, mock_from):
        """Sanity check passes when last bar is recent."""
        rates = _make_rates(5)
        mock_from.return_value = rates
        result, err = _fetch_rates_with_warmup(
            'EURUSD', 16385, 'H1', 5, 0, None, None,
            retry=False, sanity_check=True,
        )
        self.assertIsNone(err)
        self.assertIsNotNone(result)

    @patch(_RATES_FROM)
    def test_sanity_check_rejects_stale_rates_after_retries(self, mock_from):
        """Stale bars should fail instead of being returned after retry exhaustion."""
        stale_rates = _make_rates(5, base_ts=60 * 60 * 5, step=60 * 60)
        mock_from.return_value = stale_rates
        diagnostics = {}
        with (
            patch(f'{_DS}.FETCH_RETRY_ATTEMPTS', 2),
            patch(f'{_DS}.FETCH_RETRY_DELAY', 0),
            patch(f'{_DS}._utc_epoch_seconds', return_value=12 * 60 * 60),
        ):
            result, err = _fetch_rates_with_warmup(
                'EURUSD', 16385, 'H1', 5, 0, None, None,
                include_incomplete=True,
                retry=True, sanity_check=True, diagnostics=diagnostics,
            )
        self.assertIsNone(result)
        self.assertIn('Data appears stale for EURUSD H1', err)
        self.assertIn('allow_stale=true', err)
        self.assertEqual(mock_from.call_count, 2)
        self.assertEqual(
            diagnostics['freshness'],
            {
                'last_bar_epoch': float(stale_rates[-1]['time']),
                'expected_end_epoch': float(12 * 60 * 60),
                'freshness_cutoff_epoch': float((12 * 60 * 60) - (3 * 60 * 60)),
                'data_freshness_seconds': float((12 * 60 * 60) - stale_rates[-1]['time']),
                'last_bar_within_policy_window': False,
            },
        )

    @patch(_RATES_FROM)
    def test_live_completed_bars_relax_stale_policy(self, mock_from):
        """Live requests can return the latest completed bar when markets are idle."""
        stale_rates = _make_rates(5, base_ts=60 * 60 * 5, step=60 * 60)
        mock_from.return_value = stale_rates
        diagnostics = {}
        with (
            patch(f'{_DS}.FETCH_RETRY_ATTEMPTS', 2),
            patch(f'{_DS}.FETCH_RETRY_DELAY', 0),
            patch(f'{_DS}._utc_epoch_seconds', return_value=12 * 60 * 60),
        ):
            result, err = _fetch_rates_with_warmup(
                'EURUSD', 16385, 'H1', 5, 0, None, None,
                retry=True, sanity_check=True, diagnostics=diagnostics,
            )

        self.assertIsNone(err)
        self.assertEqual(result, stale_rates)
        self.assertEqual(mock_from.call_count, 1)
        self.assertEqual(
            diagnostics['freshness']['freshness_policy_relaxed'],
            'latest_completed_bar_for_live_request',
        )

    @patch(_RATES_FROM)
    @patch(_PARSE_START)
    def test_sanity_check_accepts_fresh_retry_after_initial_stale_result(self, mock_parse, mock_from):
        """A fresh retry should clear stale state instead of tripping the post-loop guard."""
        to_date = datetime(2025, 1, 2, tzinfo=_UTC)
        mock_parse.return_value = to_date
        stale_rates = _make_rates(5, base_ts=to_date.timestamp() - (10 * 60 * 60), step=60 * 60)
        fresh_rates = _make_rates(5, base_ts=to_date.timestamp(), step=60 * 60)
        mock_from.side_effect = [stale_rates, fresh_rates]
        diagnostics = {}

        with patch(f'{_DS}.FETCH_RETRY_ATTEMPTS', 2), patch(f'{_DS}.FETCH_RETRY_DELAY', 0):
            result, err = _fetch_rates_with_warmup(
                'EURUSD', 16385, 'H1', 5, 0, None, '2025-01-02',
                retry=True, sanity_check=True, diagnostics=diagnostics,
            )

        self.assertIsNone(err)
        self.assertEqual(result, fresh_rates)
        self.assertEqual(mock_from.call_count, 2)
        self.assertEqual(
            diagnostics['freshness'],
            {
                'last_bar_epoch': float(fresh_rates[-1]['time']),
                'expected_end_epoch': float(to_date.timestamp()),
                'freshness_cutoff_epoch': float(to_date.timestamp() - (4 * 60 * 60)),
                'data_freshness_seconds': 0.0,
                'last_bar_within_policy_window': True,
            },
        )


# ============================================================================
# _build_rates_df
# ============================================================================

class TestBuildRatesDf(unittest.TestCase):
    """Tests for _build_rates_df."""

    @patch(f'{_DS}._rates_to_df')
    def test_basic_utc(self, mock_to_df):
        """UTC mode: stores __epoch and formats time column."""
        raw_df = pd.DataFrame({
            'time': [1000.0, 2000.0],
            'open': [1.1, 1.2],
            'tick_volume': [50, 60],
        })
        mock_to_df.return_value = raw_df
        df = _build_rates_df([{}, {}], use_client_tz=False)
        self.assertIn('__epoch', df.columns)
        self.assertEqual(list(df['__epoch']), [1000.0, 2000.0])
        # time column should be string-formatted (not raw float)
        self.assertIsInstance(df['time'].iloc[0], str)

    @patch(f'{_DS}._rates_to_df')
    def test_client_tz(self, mock_to_df):
        """Client-tz mode: applies local time formatting."""
        raw_df = pd.DataFrame({
            'time': [1000.0, 2000.0],
            'open': [1.1, 1.2],
            'tick_volume': [50, 60],
        })
        mock_to_df.return_value = raw_df
        df = _build_rates_df([{}, {}], use_client_tz=True)
        self.assertIn('__epoch', df.columns)
        self.assertIsInstance(df['time'].iloc[0], str)

    @patch(f'{_DS}._rates_to_df')
    def test_volume_alias(self, mock_to_df):
        """If 'volume' is absent but 'tick_volume' exists, it gets aliased."""
        raw_df = pd.DataFrame({
            'time': [1000.0],
            'tick_volume': [123],
        })
        mock_to_df.return_value = raw_df
        df = _build_rates_df([{}], use_client_tz=False)
        self.assertIn('volume', df.columns)
        self.assertEqual(df['volume'].iloc[0], 123)

    @patch(f'{_DS}._rates_to_df')
    def test_volume_already_present(self, mock_to_df):
        """If 'volume' already exists, tick_volume is not aliased."""
        raw_df = pd.DataFrame({
            'time': [1000.0],
            'volume': [999],
            'tick_volume': [123],
        })
        mock_to_df.return_value = raw_df
        df = _build_rates_df([{}], use_client_tz=False)
        self.assertEqual(df['volume'].iloc[0], 999)


# ============================================================================
# _trim_df_to_target
# ============================================================================

class TestTrimDfToTarget(unittest.TestCase):
    """Tests for _trim_df_to_target."""

    def _make_df(self, n: int = 20) -> pd.DataFrame:
        base = 1_000_000.0
        return pd.DataFrame({
            '__epoch': [base + i * 60 for i in range(n)],
            'time': [f"t{i}" for i in range(n)],
            'close': [1.1 + i * 0.01 for i in range(n)],
        })

    def test_no_datetime_trims_to_candles(self):
        df = self._make_df(20)
        out = _trim_df_to_target(df, None, None, 5)
        self.assertEqual(len(out), 5)
        # Should be the last 5 rows
        self.assertEqual(list(out['time']), [f"t{i}" for i in range(15, 20)])

    def test_no_datetime_no_trim_needed(self):
        df = self._make_df(3)
        out = _trim_df_to_target(df, None, None, 10)
        self.assertEqual(len(out), 3)

    @patch(_PARSE_START)
    def test_start_and_end(self, mock_parse):
        df = self._make_df(20)
        epoch_5 = df['__epoch'].iloc[5]
        epoch_14 = df['__epoch'].iloc[14]
        mock_parse.side_effect = [
            datetime.fromtimestamp(epoch_5, tz=_UTC),
            datetime.fromtimestamp(epoch_14, tz=_UTC),
        ]
        with patch(f'{_DS}._utc_epoch_seconds', side_effect=lambda d: d.timestamp()):
            out = _trim_df_to_target(df, '2025-01-01', '2025-01-02', 100)
        self.assertEqual(len(out), 10)  # rows 5..14 inclusive

    @patch(_PARSE_START)
    def test_start_and_end_invalid_parse(self, mock_parse):
        df = self._make_df(10)
        mock_parse.side_effect = [None, None]
        out = _trim_df_to_target(df, 'bad', 'bad', 5)
        self.assertEqual(len(out), 0)

    @patch(_PARSE_START)
    def test_start_only_trims_from_start(self, mock_parse):
        df = self._make_df(20)
        epoch_10 = df['__epoch'].iloc[10]
        mock_parse.return_value = datetime.fromtimestamp(epoch_10, tz=_UTC)
        with patch(f'{_DS}._utc_epoch_seconds', side_effect=lambda d: d.timestamp()):
            out = _trim_df_to_target(df, '2025-01-01', None, 5)
        # 10 rows from index 10 onward, but capped at candles=5
        self.assertEqual(len(out), 5)

    @patch(_PARSE_START)
    def test_start_only_invalid(self, mock_parse):
        df = self._make_df(10)
        mock_parse.return_value = None
        out = _trim_df_to_target(df, 'bad', None, 5)
        self.assertEqual(len(out), 0)

    def test_end_only_trims_tail(self):
        df = self._make_df(20)
        out = _trim_df_to_target(df, None, '2025-01-02', 5)
        self.assertEqual(len(out), 5)

    def test_copy_rows_false(self):
        df = self._make_df(10)
        out = _trim_df_to_target(df, None, None, 5, copy_rows=False)
        self.assertEqual(len(out), 5)


# ============================================================================
# fetch_candles — integration-level tests
# ============================================================================

class TestFetchCandles(unittest.TestCase):
    """Tests for the public fetch_candles function."""

    def _default_patches(self):
        """Return a dict of common patches for fetch_candles tests."""
        return {
            _GUARD: _mock_symbol_guard,
            _CACHED_INFO: MagicMock(return_value=MagicMock()),
            _RESOLVE_CTZ: MagicMock(return_value=None),
            _ESTIMATE_WARMUP: MagicMock(return_value=0),
        }

    # -- Success paths -------------------------------------------------------

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
    def test_last_candle_open_flag(self, mock_warmup, mock_ctz, mock_info, mock_from, mock_cfg):
        mock_cfg.get_time_offset_seconds.return_value = 0
        # Make the last bar's epoch == now so it should be "open"
        mock_from.return_value = _make_rates(10, step=3600)
        result = fetch_candles('EURUSD', timeframe='H1', limit=5)
        self.assertTrue(result.get('success'))
        self.assertIn('last_candle_open', result)

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
        self.assertFalse(result['last_candle_open'])
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
        self.assertTrue(result['last_candle_open'])
        self.assertEqual(len(result['data']), 5)

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
        self.assertFalse(result['last_candle_open'])

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
        self.assertTrue(result['has_forming_candle'])
        self.assertIn('include_incomplete=true', result['hint'])
        self.assertFalse(result['last_candle_open'])

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
        self.assertFalse(result['last_candle_open'])

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
        self.assertEqual(result['candles_excluded'], 0)
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
                'last_bar_within_policy_window',
            },
        )
        self.assertTrue(freshness['last_bar_within_policy_window'])
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

    # -- Error paths ---------------------------------------------------------

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
        self.assertEqual(result['error'], "Symbol 'NOTASYMBOL' was not found or is not available in MT5.")

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

    # -- Indicators ----------------------------------------------------------

    @patch(_MT5_CONFIG)
    @patch(_APPLY_TI, return_value=['rsi_14'])
    @patch(_RATES_FROM)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=14)
    @patch(_GUARD, _mock_symbol_guard)
    def test_indicators_list_of_dicts(self, mock_warmup, mock_ctz, mock_info, mock_from, mock_ti, mock_cfg):
        mock_cfg.get_time_offset_seconds.return_value = 0
        rates = _make_rates(30)
        mock_from.return_value = rates
        # _apply_ta_indicators_util is called; it adds column to df in-place
        def add_col(df, spec):
            df['rsi_14'] = 50.0
            return ['rsi_14']
        mock_ti.side_effect = add_col
        result = fetch_candles(
            'EURUSD', limit=10,
            indicators=[{'name': 'rsi', 'params': [14]}],
        )
        self.assertTrue(result.get('success'))

    @patch(_MT5_CONFIG)
    @patch(_APPLY_TI, return_value=['ema_20'])
    @patch(_RATES_FROM)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=20)
    @patch(_GUARD, _mock_symbol_guard)
    def test_indicators_string_spec(self, mock_warmup, mock_ctz, mock_info, mock_from, mock_ti, mock_cfg):
        mock_cfg.get_time_offset_seconds.return_value = 0
        mock_from.return_value = _make_rates(30)
        def add_col(df, spec):
            df['ema_20'] = 1.1
            return ['ema_20']
        mock_ti.side_effect = add_col
        result = fetch_candles('EURUSD', limit=10, indicators='ema(20)')
        self.assertTrue(result.get('success'))

    @patch(_MT5_CONFIG)
    @patch(_APPLY_TI, return_value=['rsi_14'])
    @patch(_RATES_FROM)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=14)
    @patch(_GUARD, _mock_symbol_guard)
    def test_indicators_named_string_spec(self, mock_warmup, mock_ctz, mock_info, mock_from, mock_ti, mock_cfg):
        mock_cfg.get_time_offset_seconds.return_value = 0
        mock_from.return_value = _make_rates(30)

        def add_col(df, spec):
            self.assertEqual(spec, 'rsi(length=14)')
            df['rsi_14'] = 50.0
            return ['rsi_14']

        mock_ti.side_effect = add_col
        result = fetch_candles('EURUSD', limit=10, indicators='rsi(length=14)')
        self.assertTrue(result.get('success'))

    @patch(_MT5_CONFIG)
    @patch(_APPLY_TI, return_value=['RSI_14', 'EMA_20', 'ATRr_14'])
    @patch(_RATES_FROM)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=20)
    @patch(_GUARD, _mock_symbol_guard)
    def test_indicator_display_names_and_spec_are_normalized(
        self,
        mock_warmup,
        mock_ctz,
        mock_info,
        mock_from,
        mock_ti,
        mock_cfg,
    ):
        mock_cfg.get_time_offset_seconds.return_value = 0
        mock_from.return_value = _make_rates(30)

        def add_cols(df, spec):
            df['RSI_14'] = 50.0
            df['EMA_20'] = 1.1
            df['ATRr_14'] = 2.2
            return ['RSI_14', 'EMA_20', 'ATRr_14']

        mock_ti.side_effect = add_cols

        result = fetch_candles('EURUSD', limit=10, indicators='RSI(14.0),EMA(20.0),ATR(14.0)')

        self.assertTrue(result.get('success'))
        self.assertEqual(
            result['meta']['diagnostics']['indicators']['spec'],
            'RSI(14),EMA(20),ATR(14)',
        )
        self.assertEqual(
            result['meta']['diagnostics']['indicators']['added_columns'],
            ['RSI_14', 'EMA_20', 'ATR_14'],
        )
        self.assertIn('ATR_14', result['data'][0])
        self.assertNotIn('ATRr_14', result['data'][0])

    @patch(_MT5_CONFIG)
    @patch(_APPLY_TI, return_value=['ATRr_14'])
    @patch(_RATES_FROM)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=20)
    @patch(_GUARD, _mock_symbol_guard)
    def test_indicator_display_name_conflict_preserves_actual_column_name(
        self,
        mock_warmup,
        mock_ctz,
        mock_info,
        mock_from,
        mock_ti,
        mock_cfg,
    ):
        mock_cfg.get_time_offset_seconds.return_value = 0
        mock_from.return_value = _make_rates(30)

        def add_cols(df, spec):
            df['ATR_14'] = 99.0
            df['ATRr_14'] = 2.2
            return ['ATRr_14']

        mock_ti.side_effect = add_cols

        result = fetch_candles('EURUSD', limit=10, indicators='ATR(14.0)')

        self.assertTrue(result.get('success'))
        self.assertEqual(
            result['meta']['diagnostics']['indicators']['added_columns'],
            ['ATRr_14'],
        )
        self.assertIn('ATRr_14', result['data'][0])
        self.assertEqual(result['data'][0]['ATRr_14'], 2.2)
        self.assertNotIn('ATR_14', result['data'][0])

    @patch(_MT5_CONFIG)
    @patch(_RATES_FROM)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=0)
    @patch(_GUARD, _mock_symbol_guard)
    def test_unknown_indicator_returns_error(self, mock_warmup, mock_ctz, mock_info, mock_from, mock_cfg):
        mock_cfg.get_time_offset_seconds.return_value = 0
        mock_from.return_value = _make_rates(30)
        result = fetch_candles('EURUSD', limit=10, indicators='nonexistent_indicator')
        self.assertIn('error', result)
        self.assertIn('Unknown indicator', result['error'])
        mock_from.assert_not_called()

    @patch(_MT5_CONFIG)
    @patch(_RATES_FROM)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=0)
    @patch(_GUARD, _mock_symbol_guard)
    def test_indicator_param_syntax_error_is_friendly(self, mock_warmup, mock_ctz, mock_info, mock_from, mock_cfg):
        mock_cfg.get_time_offset_seconds.return_value = 0
        result = fetch_candles('EURUSD', limit=10, indicators='sma,20')
        self.assertEqual(result['error'], 'Indicator params must use parentheses, e.g. sma(20), not sma,20.')
        mock_from.assert_not_called()

    @patch(_MT5_CONFIG)
    @patch(_RATES_FROM)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=0)
    @patch(_GUARD, _mock_symbol_guard)
    def test_invalid_indicator_json_returns_parse_error(
        self,
        mock_warmup,
        mock_ctz,
        mock_info,
        mock_from,
        mock_cfg,
    ):
        mock_cfg.get_time_offset_seconds.return_value = 0
        result = fetch_candles('EURUSD', limit=10, indicators='[{"name":"rsi","params":[14]}')
        self.assertTrue(result['error'].startswith('Invalid indicator JSON:'))
        mock_from.assert_not_called()

    @patch(_MT5_CONFIG)
    @patch(_APPLY_TI, return_value=['rsi_14'])
    @patch(_RATES_FROM)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=14)
    @patch(_GUARD, _mock_symbol_guard)
    def test_indicators_json_string(self, mock_warmup, mock_ctz, mock_info, mock_from, mock_ti, mock_cfg):
        mock_cfg.get_time_offset_seconds.return_value = 0
        mock_from.return_value = _make_rates(30)
        def add_col(df, spec):
            df['rsi_14'] = 50.0
            return ['rsi_14']
        mock_ti.side_effect = add_col
        result = fetch_candles(
            'EURUSD', limit=10,
            indicators='[{"name":"rsi","params":[14]}]',
        )
        self.assertTrue(result.get('success'))

    @patch(_MT5_CONFIG)
    @patch(_APPLY_TI, return_value=['rsi_14'])
    @patch(_RATES_FROM)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=14)
    @patch(_GUARD, _mock_symbol_guard)
    def test_indicators_json_named_params(self, mock_warmup, mock_ctz, mock_info, mock_from, mock_ti, mock_cfg):
        mock_cfg.get_time_offset_seconds.return_value = 0
        mock_from.return_value = _make_rates(30)

        def add_col(df, spec):
            self.assertEqual(spec, 'rsi(length=14)')
            df['rsi_14'] = 50.0
            return ['rsi_14']

        mock_ti.side_effect = add_col
        result = fetch_candles(
            'EURUSD',
            limit=10,
            indicators='[{"name":"rsi","params":{"length":14}}]',
        )
        self.assertTrue(result.get('success'))

    @patch(_MT5_CONFIG)
    @patch(_APPLY_TI, return_value=['BBL_20_2.0'])
    @patch(_RATES_FROM)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=20)
    @patch(_GUARD, _mock_symbol_guard)
    def test_bollinger_alias_indicator_string_spec(self, mock_warmup, mock_ctz, mock_info, mock_from, mock_ti, mock_cfg):
        mock_cfg.get_time_offset_seconds.return_value = 0
        mock_from.return_value = _make_rates(30)

        def add_col(df, spec):
            self.assertEqual(spec, 'bb(20)')
            df['BBL_20_2.0'] = 1.1
            return ['BBL_20_2.0']

        mock_ti.side_effect = add_col
        result = fetch_candles('EURUSD', limit=10, indicators='bb(20)')
        self.assertTrue(result.get('success'))

    @patch(_MT5_CONFIG)
    @patch(_APPLY_TI)
    @patch(_RATES_FROM)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=14)
    @patch(_GUARD, _mock_symbol_guard)
    def test_nan_warmup_retry(self, mock_warmup, mock_ctz, mock_info, mock_from, mock_ti, mock_cfg):
        """When TI columns have NaN, fetch_candles retries with more warmup."""
        mock_cfg.get_time_offset_seconds.return_value = 0
        rates = _make_rates(30)
        mock_from.return_value = rates

        call_count = [0]
        def add_col(df, spec):
            call_count[0] += 1
            if call_count[0] == 1:
                df['rsi_14'] = float('nan')
            else:
                df['rsi_14'] = 50.0
            return ['rsi_14']
        mock_ti.side_effect = add_col

        result = fetch_candles('EURUSD', limit=5, indicators=[{'name': 'rsi', 'params': [14]}])
        self.assertTrue(result.get('success'))
        # _apply_ta_indicators_util should have been called twice (original + retry)
        self.assertGreaterEqual(mock_ti.call_count, 2)

    @patch(_MT5_CONFIG)
    @patch(_APPLY_TI)
    @patch(f'{_DS}.FETCH_RETRY_DELAY', 0)
    @patch(f'{_DS}.FETCH_RETRY_ATTEMPTS', 1)
    @patch(_RATES_FROM)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=14)
    @patch(_GUARD, _mock_symbol_guard)
    def test_nan_warmup_retry_metadata_stays_unapplied_when_refetch_is_empty(
        self,
        mock_warmup,
        mock_ctz,
        mock_info,
        mock_from,
        mock_ti,
        mock_cfg,
    ):
        mock_cfg.get_time_offset_seconds.return_value = 0
        mock_from.side_effect = [_make_rates(30), []]

        def add_nan_col(df, spec):
            df['rsi_14'] = float('nan')
            return ['rsi_14']

        mock_ti.side_effect = add_nan_col

        result = fetch_candles('EURUSD', limit=5, indicators=[{'name': 'rsi', 'params': [14]}])

        self.assertTrue(result.get('success'))
        warmup_retry = result['meta']['diagnostics']['query']['warmup_retry']
        self.assertFalse(warmup_retry['applied'])
        self.assertEqual(warmup_retry['raw_bars_fetched'], 0)

    @patch(_MT5_CONFIG)
    @patch(_APPLY_TI)
    @patch(f'{_DS}.FETCH_RETRY_DELAY', 0)
    @patch(f'{_DS}.FETCH_RETRY_ATTEMPTS', 2)
    @patch(_RATES_FROM)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=14)
    @patch(_GUARD, _mock_symbol_guard)
    def test_nan_warmup_retry_refetch_retries_transient_empty_fetch(
        self,
        mock_warmup,
        mock_ctz,
        mock_info,
        mock_from,
        mock_ti,
        mock_cfg,
    ):
        mock_cfg.get_time_offset_seconds.return_value = 0
        mock_from.side_effect = [_make_rates(30), None, _make_rates(60)]

        call_count = [0]

        def add_col(df, spec):
            call_count[0] += 1
            if call_count[0] == 1:
                df['rsi_14'] = float('nan')
            else:
                df['rsi_14'] = 50.0
            return ['rsi_14']

        mock_ti.side_effect = add_col

        result = fetch_candles('EURUSD', limit=5, indicators=[{'name': 'rsi', 'params': [14]}])

        self.assertTrue(result.get('success'))
        warmup_retry = result['meta']['diagnostics']['query']['warmup_retry']
        self.assertTrue(warmup_retry['applied'])
        self.assertEqual(warmup_retry['raw_bars_fetched'], 60)
        self.assertGreaterEqual(mock_from.call_count, 3)

    @patch(_MT5_CONFIG)
    @patch(f"{_DS}._rebuild_candle_indicator_window", side_effect=RuntimeError("retry boom"))
    @patch(_APPLY_TI)
    @patch(_RATES_FROM)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=14)
    @patch(_GUARD, _mock_symbol_guard)
    def test_nan_warmup_retry_failure_surfaces_warning_and_meta_error(
        self,
        mock_warmup,
        mock_ctz,
        mock_info,
        mock_from,
        mock_ti,
        mock_rebuild,
        mock_cfg,
    ):
        mock_cfg.get_time_offset_seconds.return_value = 0
        mock_from.side_effect = [_make_rates(30), _make_rates(60)]

        def add_nan_col(df, spec):
            df['rsi_14'] = float('nan')
            return ['rsi_14']

        mock_ti.side_effect = add_nan_col

        result = fetch_candles('EURUSD', limit=5, indicators=[{'name': 'rsi', 'params': [14]}])

        self.assertTrue(result.get('success'))
        warmup_retry = result['meta']['diagnostics']['query']['warmup_retry']
        self.assertTrue(warmup_retry['applied'])
        self.assertEqual(warmup_retry['error'], 'retry boom')
        self.assertTrue(any('Indicator warmup retry failed: retry boom' in str(w) for w in result.get('warnings', [])))
        self.assertEqual(mock_rebuild.call_count, 1)

    @patch(_MT5_CONFIG)
    @patch(_PARSE_START)
    @patch(f'{_DS}.FETCH_RETRY_DELAY', 0)
    @patch(f'{_DS}.FETCH_RETRY_ATTEMPTS', 1)
    @patch(_APPLY_TI)
    @patch(_RATES_FROM)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=14)
    @patch(_GUARD, _mock_symbol_guard)
    def test_nan_warmup_retry_preserves_freshness_policy(
        self,
        mock_warmup,
        mock_ctz,
        mock_info,
        mock_from,
        mock_ti,
        mock_parse,
        mock_cfg,
    ):
        mock_cfg.get_time_offset_seconds.return_value = 0
        to_date = datetime(2025, 1, 2, tzinfo=_UTC)
        mock_parse.return_value = to_date
        fresh_rates = _make_rates(30, base_ts=to_date.timestamp(), step=60 * 60)
        stale_rates = _make_rates(60, base_ts=to_date.timestamp() - (10 * 60 * 60), step=60 * 60)
        mock_from.side_effect = [fresh_rates, stale_rates]

        call_count = [0]

        def add_col(df, spec):
            call_count[0] += 1
            if call_count[0] == 1:
                df['rsi_14'] = float('nan')
            else:
                df['rsi_14'] = 50.0
            return ['rsi_14']

        mock_ti.side_effect = add_col

        result = fetch_candles(
            'EURUSD',
            limit=5,
            end='2025-01-02',
            indicators=[{'name': 'rsi', 'params': [14]}],
            time_as_epoch=True,
        )

        self.assertTrue(result.get('success'))
        self.assertEqual(mock_from.call_count, 2)
        self.assertEqual(mock_ti.call_count, 1)
        self.assertEqual(float(result['data'][-1]['time']), float(fresh_rates[-1]['time']))
        warmup_retry = result['meta']['diagnostics']['query']['warmup_retry']
        self.assertFalse(warmup_retry['applied'])
        self.assertIn('Data appears stale for EURUSD H1', warmup_retry['error'])
        self.assertTrue(any('Indicator warmup retry failed:' in str(w) for w in result.get('warnings', [])))

    # -- Simplify ------------------------------------------------------------

    @patch(_MT5_CONFIG)
    @patch(_SIMPLIFY_EXT)
    @patch(_RATES_FROM)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=0)
    @patch(_GUARD, _mock_symbol_guard)
    def test_simplify_basic(self, mock_warmup, mock_ctz, mock_info, mock_from, mock_simp, mock_cfg):
        mock_cfg.get_time_offset_seconds.return_value = 0
        rates = _make_rates(20)
        mock_from.return_value = rates

        # simplify returns the df unchanged with metadata
        def passthrough(df, hdrs, spec):
            meta = {'method': 'lttb', 'original_rows': len(df), 'returned_rows': len(df), 'headers': hdrs}
            return df, meta
        mock_simp.side_effect = passthrough

        result = fetch_candles('EURUSD', limit=10, simplify={'mode': 'select', 'points': 5})
        self.assertTrue(result.get('success'))

    @patch(_MT5_CONFIG)
    @patch(_SIMPLIFY_EXT)
    @patch(_RATES_FROM)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=0)
    @patch(_GUARD, _mock_symbol_guard)
    def test_simplify_reduced_rows(self, mock_warmup, mock_ctz, mock_info, mock_from, mock_simp, mock_cfg):
        mock_cfg.get_time_offset_seconds.return_value = 0
        rates = _make_rates(20)
        mock_from.return_value = rates

        def reduce_rows(df, hdrs, spec):
            reduced = df.iloc[:3].copy()
            meta = {'method': 'lttb', 'original_rows': len(df), 'returned_rows': 3}
            return reduced, meta
        mock_simp.side_effect = reduce_rows

        result = fetch_candles('EURUSD', limit=10, simplify={'mode': 'select', 'points': 3})
        self.assertTrue(result.get('success'))
        self.assertTrue(result.get('simplified'))

    @patch(_MT5_CONFIG)
    @patch(_SIMPLIFY_EXT)
    @patch(_RATES_FROM)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=0)
    @patch(_GUARD, _mock_symbol_guard)
    def test_simplify_no_explicit_points(self, mock_warmup, mock_ctz, mock_info, mock_from, mock_simp, mock_cfg):
        """When no points/ratio specified, default ratio is used."""
        mock_cfg.get_time_offset_seconds.return_value = 0
        mock_from.return_value = _make_rates(20)
        mock_simp.side_effect = lambda df, h, s: (df, None)
        result = fetch_candles('EURUSD', limit=10, simplify={'mode': 'select'})
        self.assertTrue(result.get('success'))

    # -- Denoise -------------------------------------------------------------

    @patch(_MT5_CONFIG)
    @patch(f'{_DS}._normalize_denoise_spec')
    @patch(f'{_DS}._apply_denoise_util', return_value=[])
    @patch(_SIMPLIFY_EXT, side_effect=lambda df, h, s: (df, None))
    @patch(_RATES_FROM)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=0)
    @patch(_GUARD, _mock_symbol_guard)
    def test_denoise_pre_ti(self, mock_warmup, mock_ctz, mock_info, mock_from, mock_simp,
                            mock_apply_dn, mock_norm_dn, mock_cfg):
        mock_cfg.get_time_offset_seconds.return_value = 0
        mock_from.return_value = _make_rates(10)
        mock_norm_dn.return_value = {'method': 'ema', 'when': 'pre_ti', 'params': {}}
        result = fetch_candles('EURUSD', limit=5, denoise={'method': 'ema', 'when': 'pre_ti'})
        self.assertTrue(result.get('success'))

    @patch(_MT5_CONFIG)
    @patch(f'{_DS}._normalize_denoise_spec')
    @patch(f'{_DS}._apply_denoise_util', return_value=['close_dn'])
    @patch(_SIMPLIFY_EXT, side_effect=lambda df, h, s: (df, None))
    @patch(_RATES_FROM)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=0)
    @patch(_GUARD, _mock_symbol_guard)
    def test_denoise_post_ti(self, mock_warmup, mock_ctz, mock_info, mock_from, mock_simp,
                             mock_apply_dn, mock_norm_dn, mock_cfg):
        mock_cfg.get_time_offset_seconds.return_value = 0
        mock_from.return_value = _make_rates(10)
        mock_norm_dn.return_value = {'method': 'ema', 'when': 'post_ti', 'params': {}}
        mock_apply_dn.side_effect = lambda df, spec, **kw: (
            df.__setitem__('close_dn', 1.0) or ['close_dn']
        )
        result = fetch_candles('EURUSD', limit=5, denoise={'method': 'ema', 'when': 'post_ti'})
        self.assertTrue(result.get('success'))
        if result.get('denoise'):
            self.assertTrue(result['denoise']['applications'])

    @patch(_MT5_CONFIG)
    @patch(f'{_DS}._normalize_denoise_spec')
    @patch(_SIMPLIFY_EXT, side_effect=lambda df, h, s: (df, None))
    @patch(_RATES_FROM)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=0)
    @patch(_GUARD, _mock_symbol_guard)
    def test_denoise_warning_is_surfaced(self, mock_warmup, mock_ctz, mock_info, mock_from, mock_simp,
                                         mock_norm_dn, mock_cfg):
        mock_cfg.get_time_offset_seconds.return_value = 0
        mock_from.return_value = _make_rates(10)
        mock_norm_dn.return_value = {'method': 'wavelet', 'when': 'pre_ti', 'params': {}}

        def add_warning(df, spec, **kwargs):
            df.attrs["denoise_warnings"] = [
                "Denoise method 'wavelet' requires PyWavelets, but it is not installed."
            ]
            return []

        with patch(f'{_DS}._apply_denoise_util', side_effect=add_warning):
            result = fetch_candles('EURUSD', limit=5, denoise={'method': 'wavelet'})

        self.assertTrue(result.get('success'))
        self.assertIn('warnings', result)
        self.assertIn("requires PyWavelets", result['warnings'][0])

    # -- Start / End datetime ------------------------------------------------

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

    # -- Exception handling --------------------------------------------------

    @patch(_CACHED_INFO, side_effect=RuntimeError("boom"))
    def test_exception_returns_error(self, mock_info):
        result = fetch_candles('EURUSD')
        self.assertIn('error', result)
        self.assertIn('Error getting rates', result['error'])


# ============================================================================
# fetch_ticks
# ============================================================================

class TestFetchTicks(unittest.TestCase):
    """Tests for the public fetch_ticks function."""

    # -- Success paths -------------------------------------------------------

    @patch(_TICKS_RANGE)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_GUARD, _mock_symbol_guard)
    def test_basic_rows(self, mock_ctz, mock_info, mock_ticks):
        mock_ticks.return_value = _make_ticks(10)
        result = fetch_ticks('EURUSD', limit=5, format='rows')
        self.assertTrue(result.get('success'))
        self.assertEqual(result['count'], 5)
        self.assertEqual(result.get('timezone'), 'UTC')

    @patch(f'{_DS}.FETCH_RETRY_DELAY', 0)
    @patch(f'{_DS}.FETCH_RETRY_ATTEMPTS', 1)
    @patch(_TICKS_RANGE)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_GUARD, _mock_symbol_guard)
    def test_recent_ticks_expand_lookback_until_limit_is_met(self, mock_ctz, mock_info, mock_ticks):
        mock_ticks.side_effect = [_make_ticks(2), _make_ticks(10)]
        result = fetch_ticks('EURUSD', limit=5, format='rows')

        self.assertTrue(result.get('success'))
        self.assertEqual(result['count'], 5)
        self.assertEqual(mock_ticks.call_count, 2)

    @patch(_TICKS_RANGE)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_GUARD, _mock_symbol_guard)
    def test_summary_output(self, mock_ctz, mock_info, mock_ticks):
        mock_ticks.return_value = _make_ticks(20)
        result = fetch_ticks('EURUSD', limit=20, format='summary')
        self.assertTrue(result.get('success'))
        self.assertNotIn('output', result)
        self.assertIn('stats', result)
        stats = result['stats']
        self.assertEqual(set(stats), {'spread'})
        self.assertEqual(set(stats['spread']), {'low', 'high', 'mean'})

    @patch(_TICKS_RANGE)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_GUARD, _mock_symbol_guard)
    def test_compact_output_alias_is_rejected(self, mock_ctz, mock_info, mock_ticks):
        mock_ticks.return_value = _make_ticks(20)
        result = fetch_ticks('EURUSD', limit=10, format='compact')
        self.assertIn("Invalid format", result.get("error", ""))

    @patch(_TICKS_RANGE)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_GUARD, _mock_symbol_guard)
    def test_stats_output(self, mock_ctz, mock_info, mock_ticks):
        mock_ticks.return_value = _make_ticks(20)
        result = fetch_ticks('EURUSD', limit=10, format='stats')
        self.assertTrue(result.get('success'))
        self.assertEqual(result['output'], 'stats')
        bid_stats = result['stats']['bid']
        # detailed_stats should include median, skew, quantiles
        self.assertIn('median', bid_stats)
        self.assertIn('skew', bid_stats)
        self.assertIn('q25', bid_stats)
        self.assertIn('q75', bid_stats)

    @patch(_TICKS_RANGE)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_GUARD, _mock_symbol_guard)
    def test_full_output_alias_is_rejected(self, mock_ctz, mock_info, mock_ticks):
        mock_ticks.return_value = _make_ticks(20)
        result = fetch_ticks('EURUSD', limit=10, format='full')
        self.assertIn("Invalid format", result.get("error", ""))

    @patch(_TICKS_RANGE)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_GUARD, _mock_symbol_guard)
    def test_summary_stats_structure(self, mock_ctz, mock_info, mock_ticks):
        ticks = _make_ticks(20)
        mock_ticks.return_value = ticks
        result = fetch_ticks('EURUSD', limit=20, format='summary')
        spread = result['stats']['spread']
        for key in ('low', 'high', 'mean'):
            self.assertIn(key, spread)
            self.assertIsInstance(spread[key], float)

    @patch(_TICKS_RANGE)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_GUARD, _mock_symbol_guard)
    def test_small_summary_uses_spread_only_stats(self, mock_ctz, mock_info, mock_ticks):
        mock_ticks.return_value = _make_ticks(5)
        result = fetch_ticks('EURUSD', limit=5, format='summary')
        self.assertTrue(result.get('success'))
        self.assertNotIn('sample_adequacy', result)
        self.assertNotIn('sample_min_ticks', result)
        self.assertNotIn('sample_adequacy_note', result)
        self.assertEqual(set(result['stats']), {'spread'})

    @patch(_TICKS_RANGE)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_GUARD, _mock_symbol_guard)
    def test_summary_duration_and_tick_rate(self, mock_ctz, mock_info, mock_ticks):
        mock_ticks.return_value = _make_ticks(10, step=2.0)
        result = fetch_ticks('EURUSD', limit=10, format='summary')
        self.assertGreater(result['duration_seconds'], 0)
        self.assertIsInstance(result['tick_rate_per_second'], float)

    @patch(_TICKS_RANGE)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_GUARD, _mock_symbol_guard)
    def test_last_column_included(self, mock_ctz, mock_info, mock_ticks):
        ticks = _make_ticks(5)
        # Ensure 'last' values are non-zero/distinct
        for i, t in enumerate(ticks):
            t['last'] = 1.1 + i * 0.01
        mock_ticks.return_value = ticks
        result = fetch_ticks('EURUSD', limit=5, format='rows')
        self.assertTrue(result.get('success'))

    @patch(_TICKS_RANGE)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_GUARD, _mock_symbol_guard)
    def test_zero_last_excluded(self, mock_ctz, mock_info, mock_ticks):
        ticks = _make_ticks(5)
        for t in ticks:
            t['last'] = 0.0
        mock_ticks.return_value = ticks
        result = fetch_ticks('EURUSD', limit=5, format='rows')
        self.assertTrue(result.get('success'))

    @patch(_TICKS_RANGE)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_GUARD, _mock_symbol_guard)
    def test_namedtuple_like_ticks_are_supported_for_row_output(self, mock_ctz, mock_info, mock_ticks):
        ticks = [
            SimpleNamespace(
                time=_NOW_TS + i,
                bid=1.1000 + i * 0.0001,
                ask=1.1002 + i * 0.0001,
                last=1.1001 + i * 0.0001,
                volume=1.0,
                flags=0,
                volume_real=0.0,
            )
            for i in range(5)
        ]
        mock_ticks.return_value = ticks

        result = fetch_ticks('EURUSD', limit=5, format='rows')

        self.assertTrue(result.get('success'))
        self.assertEqual(result['count'], 5)

    # -- With start/end dates ------------------------------------------------

    @patch(_TICKS_RANGE)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_GUARD, _mock_symbol_guard)
    def test_start_only(self, mock_ctz, mock_info, mock_range):
        mock_range.return_value = _make_ticks(10)
        result = fetch_ticks('EURUSD', limit=5, start='2025-01-01', format='rows')
        self.assertTrue(result.get('success'))
        self.assertEqual(result['count'], 5)

    @patch(_TICKS_RANGE)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_GUARD, _mock_symbol_guard)
    def test_start_and_end(self, mock_ctz, mock_info, mock_range):
        mock_range.return_value = _make_ticks(10)
        result = fetch_ticks('EURUSD', limit=5, start='2025-01-01', end='2025-01-02', format='rows')
        self.assertTrue(result.get('success'))

    # -- Error paths ---------------------------------------------------------

    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_GUARD, _mock_symbol_guard_error)
    def test_symbol_guard_error(self, mock_info):
        result = fetch_ticks('INVALID')
        self.assertIn('error', result)

    @patch(_TICKS_RANGE, return_value=None)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_GUARD, _mock_symbol_guard)
    def test_ticks_none(self, mock_ctz, mock_info, mock_ticks):
        _mt5_mock.last_error.return_value = (-1, 'No ticks')
        result = fetch_ticks('EURUSD', limit=5)
        self.assertIn('error', result)
        self.assertIn('Failed to get ticks', result['error'])

    @patch(_TICKS_RANGE, return_value=[])
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_GUARD, _mock_symbol_guard)
    def test_ticks_empty(self, mock_ctz, mock_info, mock_ticks):
        result = fetch_ticks('EURUSD', limit=5)
        self.assertIn('error', result)
        self.assertIn('No tick data', result['error'])

    @patch(_TICKS_RANGE)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_GUARD, _mock_symbol_guard)
    def test_invalid_output_mode(self, mock_ctz, mock_info, mock_ticks):
        mock_ticks.return_value = _make_ticks(5)
        result = fetch_ticks('EURUSD', limit=5, format='BADMODE')
        self.assertIn('error', result)
        self.assertIn('Invalid format', result['error'])
        self.assertIn('summary', result['error'])
        self.assertIn('stats', result['error'])
        self.assertIn('rows', result['error'])

    @patch(_TICKS_RANGE)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_GUARD, _mock_symbol_guard)
    @patch(_PARSE_START)
    def test_start_only_returns_latest_ticks_since_start(self, mock_parse, mock_ctz, mock_info, mock_range):
        now = datetime.now(timezone.utc)
        mock_parse.return_value = now - timedelta(days=2)

        older_ticks = _make_ticks(3, base_ts=(now - timedelta(days=1, hours=12)).timestamp())
        newer_ticks = _make_ticks(4, base_ts=now.timestamp())
        for i, tick in enumerate(older_ticks):
            tick['bid'] = 1.2000 + i * 0.0001
            tick['ask'] = tick['bid'] + 0.0002
            tick['last'] = tick['bid'] + 0.0001
        for i, tick in enumerate(newer_ticks):
            tick['bid'] = 1.3000 + i * 0.0001
            tick['ask'] = tick['bid'] + 0.0002
            tick['last'] = tick['bid'] + 0.0001

        mock_range.side_effect = [newer_ticks, older_ticks]

        result = fetch_ticks('EURUSD', limit=5, start='ignored', format='rows')

        self.assertTrue(result.get('success'))
        self.assertEqual(result['count'], 5)
        bids = [row['bid'] for row in result['data']]
        self.assertEqual(bids, [1.2002, 1.3, 1.3001, 1.3002, 1.3003])

    @patch(_TICKS_RANGE)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_GUARD, _mock_symbol_guard)
    def test_start_after_end_is_rejected(self, mock_ctz, mock_info, mock_range):
        result = fetch_ticks('EURUSD', limit=5, start='2025-01-02', end='2025-01-01', format='rows')

        self.assertEqual(result['error'], 'start must be before or equal to end.')
        mock_range.assert_not_called()

    @patch(_TICKS_RANGE)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_GUARD, _mock_symbol_guard)
    def test_invalid_start_date(self, mock_ctz, mock_info, mock_ticks):
        result = fetch_ticks('EURUSD', limit=5, start='not-a-date')
        self.assertIn('error', result)

    @patch(_TICKS_RANGE)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_GUARD, _mock_symbol_guard)
    def test_invalid_end_date(self, mock_ctz, mock_info, mock_ticks):
        result = fetch_ticks('EURUSD', limit=5, start='2025-01-01', end='not-a-date')
        self.assertIn('error', result)

    @patch(_CACHED_INFO, side_effect=RuntimeError("boom"))
    def test_exception_returns_error(self, mock_info):
        result = fetch_ticks('EURUSD')
        self.assertIn('error', result)
        self.assertIn('Error getting ticks', result['error'])

    # -- Volume stats --------------------------------------------------------

    @patch(_TICKS_RANGE)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_GUARD, _mock_symbol_guard)
    def test_summary_with_real_volume(self, mock_ctz, mock_info, mock_ticks):
        ticks = _make_ticks(20)
        for i, t in enumerate(ticks):
            t['volume_real'] = float(100 + i * 10)
        mock_ticks.return_value = ticks
        result = fetch_ticks('EURUSD', limit=20, format='stats')
        self.assertTrue(result.get('success'))
        vol = result['stats']['volume']
        self.assertEqual(vol['kind'], 'real_volume')
        self.assertIn('sum', vol)
        self.assertIn('per_second', vol)

    @patch(_TICKS_RANGE)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_GUARD, _mock_symbol_guard)
    def test_stats_with_real_volume_detail(self, mock_ctz, mock_info, mock_ticks):
        ticks = _make_ticks(20)
        for i, t in enumerate(ticks):
            t['volume_real'] = float(100 + i * 10)
        mock_ticks.return_value = ticks
        result = fetch_ticks('EURUSD', limit=20, format='stats')
        vol = result['stats']['volume']
        self.assertIn('dist', vol)

    @patch(_TICKS_RANGE)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_GUARD, _mock_symbol_guard)
    def test_summary_tick_volume_fallback(self, mock_ctz, mock_info, mock_ticks):
        ticks = _make_ticks(20)
        # All volume_real == 0 → fallback to tick_volume
        mock_ticks.return_value = ticks
        result = fetch_ticks('EURUSD', limit=20, format='stats')
        vol = result['stats']['volume']
        self.assertEqual(vol['kind'], 'tick_volume')

    @patch(_TICKS_RANGE)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_GUARD, _mock_symbol_guard)
    def test_detailed_tick_volume_stats(self, mock_ctz, mock_info, mock_ticks):
        ticks = _make_ticks(10)
        mock_ticks.return_value = ticks
        result = fetch_ticks('EURUSD', limit=10, format='stats')
        vol = result['stats']['volume']
        self.assertEqual(vol['kind'], 'tick_volume')
        self.assertIn('sum', vol)

    # -- Simplify for ticks --------------------------------------------------

    @patch(_SIMPLIFY_EXT)
    @patch(_TICKS_RANGE)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_GUARD, _mock_symbol_guard)
    def test_simplify_approximate_mode(self, mock_ctz, mock_info, mock_ticks, mock_simp):
        ticks = _make_ticks(20)
        mock_ticks.return_value = ticks
        # The approximate mode uses _simplify_dataframe_rows_ext
        def passthrough(df, hdrs, spec):
            return df.iloc[:5].copy(), {'method': 'approximate', 'returned_rows': 5}
        mock_simp.side_effect = passthrough
        result = fetch_ticks('EURUSD', limit=20, format='rows',
                             simplify={'mode': 'approximate', 'points': 5})
        self.assertTrue(result.get('success'))

    @patch(_TICKS_RANGE)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_GUARD, _mock_symbol_guard)
    def test_simplify_select_mode(self, mock_ctz, mock_info, mock_ticks):
        ticks = _make_ticks(20)
        mock_ticks.return_value = ticks
        result = fetch_ticks('EURUSD', limit=20, format='rows',
                             simplify={'mode': 'select', 'points': 5})
        self.assertTrue(result.get('success'))
        # Should have fewer rows than original
        self.assertLessEqual(result['count'], 20)

    @patch(_TICKS_RANGE)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_GUARD, _mock_symbol_guard)
    def test_simplify_no_points_uses_default(self, mock_ctz, mock_info, mock_ticks):
        ticks = _make_ticks(20)
        mock_ticks.return_value = ticks
        result = fetch_ticks('EURUSD', limit=20, format='rows',
                             simplify={'mode': 'select'})
        self.assertTrue(result.get('success'))


# ============================================================================
# Edge cases and integration
# ============================================================================

class TestEdgeCases(unittest.TestCase):
    """Edge cases and misc coverage."""

    @patch(_MT5_CONFIG)
    @patch(_RATES_FROM)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=0)
    @patch(_GUARD, _mock_symbol_guard)
    def test_single_candle(self, mock_warmup, mock_ctz, mock_info, mock_from, mock_cfg):
        mock_cfg.get_time_offset_seconds.return_value = 0
        mock_from.return_value = _make_rates(1, base_ts=_NOW_TS - 7200, step=3600)
        result = fetch_candles('EURUSD', limit=1)
        self.assertTrue(result.get('success'))
        self.assertEqual(result['candles'], 1)

    @patch(_MT5_CONFIG)
    @patch(_RATES_FROM)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=0)
    @patch(_GUARD, _mock_symbol_guard)
    def test_limit_larger_than_data(self, mock_warmup, mock_ctz, mock_info, mock_from, mock_cfg):
        mock_cfg.get_time_offset_seconds.return_value = 0
        mock_from.return_value = _make_rates(3, base_ts=_NOW_TS - 7200, step=3600)
        result = fetch_candles('EURUSD', limit=100)
        self.assertTrue(result.get('success'))
        self.assertEqual(result['candles'], 3)

    @patch(_MT5_CONFIG)
    @patch(_RATES_FROM)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=0)
    @patch(_GUARD, _mock_symbol_guard)
    def test_ohlcv_v_includes_tick_volume(self, mock_warmup, mock_ctz, mock_info, mock_from, mock_cfg):
        mock_cfg.get_time_offset_seconds.return_value = 0
        mock_from.return_value = _make_rates(10)
        result = fetch_candles('EURUSD', limit=5, ohlcv='V')
        self.assertTrue(result.get('success'))
        row_keys = set(result['data'][0].keys())
        self.assertIn('tick_volume', row_keys)
        self.assertNotIn('open', row_keys)

    def test_build_rates_df_no_tick_volume(self):
        """DataFrame without tick_volume column doesn't crash."""
        raw_df = pd.DataFrame({
            'time': [1000.0],
            'open': [1.1],
            'volume': [10],
        })
        with patch(f'{_DS}._rates_to_df', return_value=raw_df):
            df = _build_rates_df([{}], use_client_tz=False)
        self.assertIn('volume', df.columns)
        self.assertEqual(df['volume'].iloc[0], 10)

    def test_trim_df_start_only_fewer_than_candles(self):
        """start_only with fewer matching rows than candles requested."""
        base = 1_000_000.0
        df = pd.DataFrame({
            '__epoch': [base + i * 60 for i in range(5)],
            'time': [f"t{i}" for i in range(5)],
        })
        with patch(_PARSE_START, return_value=datetime.fromtimestamp(base, tz=_UTC)), \
             patch(f'{_DS}._utc_epoch_seconds', side_effect=lambda d: d.timestamp()):
            out = _trim_df_to_target(df, '2025-01-01', None, 100)
        self.assertEqual(len(out), 5)

    @patch(_TICKS_RANGE)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_GUARD, _mock_symbol_guard)
    def test_single_tick_summary(self, mock_ctz, mock_info, mock_ticks):
        mock_ticks.return_value = _make_ticks(1)
        result = fetch_ticks('EURUSD', limit=1, format='summary')
        self.assertTrue(result.get('success'))
        self.assertEqual(result['count'], 1)

    @patch(_TICKS_RANGE)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_GUARD, _mock_symbol_guard)
    def test_ticks_flags_included(self, mock_ctz, mock_info, mock_ticks):
        ticks = _make_ticks(5)
        for i, t in enumerate(ticks):
            t['flags'] = i + 1  # nonzero flags
        mock_ticks.return_value = ticks
        result = fetch_ticks('EURUSD', limit=5, format='rows')
        self.assertTrue(result.get('success'))
        self.assertEqual(result["flags_decoded"]["1"], ["unknown_1"])
        self.assertEqual(result["flags_decoded"]["2"], ["bid"])
        self.assertEqual(result["flags_decoded"]["3"], ["bid", "unknown_1"])
        self.assertNotIn("flags_legend", result)

    @patch(_TICKS_RANGE)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_GUARD, _mock_symbol_guard)
    def test_ticks_volume_real_cv_computed(self, mock_ctz, mock_info, mock_ticks):
        """Real volume stats include coefficient of variation."""
        ticks = _make_ticks(20)
        for i, t in enumerate(ticks):
            t['volume_real'] = float(50 + i * 5)
        mock_ticks.return_value = ticks
        result = fetch_ticks('EURUSD', limit=20, format='stats')
        vol = result['stats']['volume']
        self.assertIn('cv', vol)

    @patch(_TICKS_RANGE)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_GUARD, _mock_symbol_guard)
    def test_ticks_volume_top10_share(self, mock_ctz, mock_info, mock_ticks):
        ticks = _make_ticks(20)
        for i, t in enumerate(ticks):
            t['volume_real'] = float(100 + i * 10)
        mock_ticks.return_value = ticks
        result = fetch_ticks('EURUSD', limit=20, format='stats')
        vol = result['stats']['volume']
        self.assertIn('top10_share', vol)

    @patch(_TICKS_RANGE)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_GUARD, _mock_symbol_guard)
    def test_ticks_volume_half_ratio(self, mock_ctz, mock_info, mock_ticks):
        ticks = _make_ticks(20)
        for i, t in enumerate(ticks):
            t['volume_real'] = float(100 + i * 10)
        mock_ticks.return_value = ticks
        result = fetch_ticks('EURUSD', limit=20, format='stats')
        vol = result['stats']['volume']
        self.assertIn('half_ratio', vol)

    @patch(_TICKS_RANGE)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_GUARD, _mock_symbol_guard)
    def test_ticks_vwap_mid(self, mock_ctz, mock_info, mock_ticks):
        ticks = _make_ticks(20)
        for i, t in enumerate(ticks):
            t['volume_real'] = float(100 + i * 10)
        mock_ticks.return_value = ticks
        result = fetch_ticks('EURUSD', limit=20, format='stats')
        vol = result['stats']['volume']
        self.assertIn('vwap_mid', vol)

    @patch(_TICKS_RANGE)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_GUARD, _mock_symbol_guard)
    def test_ticks_spike95(self, mock_ctz, mock_info, mock_ticks):
        ticks = _make_ticks(20)
        for i, t in enumerate(ticks):
            t['volume_real'] = float(100 + i * 10)
        mock_ticks.return_value = ticks
        result = fetch_ticks('EURUSD', limit=20, format='stats')
        vol = result['stats']['volume']
        self.assertIn('spike95_count', vol)
        self.assertIn('spike95_share', vol)

    @patch(_TICKS_RANGE)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_GUARD, _mock_symbol_guard)
    def test_ticks_corr_abs_mid_change(self, mock_ctz, mock_info, mock_ticks):
        ticks = _make_ticks(20, step=2.0)
        for i, t in enumerate(ticks):
            t['volume_real'] = float(100 + i * 10)
            t['bid'] = 1.1 + i * 0.001
            t['ask'] = 1.1002 + i * 0.001
        mock_ticks.return_value = ticks
        result = fetch_ticks('EURUSD', limit=20, format='stats')
        vol = result['stats']['volume']
        self.assertIn('corr_abs_mid_change', vol)


if __name__ == '__main__':
    unittest.main()
