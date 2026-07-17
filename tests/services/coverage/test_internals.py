"""Tests for data_service private helpers and low-level building blocks.

Covers:
  - Standalone helper function tests (build_candle_headers, freshness diagnostics, etc.)
  - TestFetchRatesWithWarmup
  - TestBuildRatesDf
  - TestTrimDfToTarget
"""

import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from mtdata.services.data_service import (
    _build_candle_freshness_diagnostics,
    _build_candle_headers,
    _build_no_data_error_with_context,
    _build_rates_df,
    _compact_tick_summary,
    _fetch_rates_with_warmup,
    _trim_df_to_target,
)

from ._helpers import (
    _DS,
    _NOW_TS,
    _PARSE_START,
    _RATES_FROM,
    _RATES_RANGE,
    _UTC,
    _make_rates,
    _make_rates_array,
)

# ============================================================================
# Standalone helper function tests
# ============================================================================

def test_build_candle_headers_tolerates_missing_volume_fields() -> None:
    rates = [{
        "time": _NOW_TS,
        "open": 1.1,
        "high": 1.2,
        "low": 1.0,
        "close": 1.15,
    }]

    headers = _build_candle_headers(rates, "OHLC")

    assert headers == ["time", "open", "high", "low", "close"]


def test_candle_freshness_diagnostics_never_reports_negative_freshness() -> None:
    diagnostics = _build_candle_freshness_diagnostics(
        last_bar_epoch=200.0,
        expected_end_epoch=100.0,
        freshness_cutoff_epoch=50.0,
    )

    assert diagnostics["data_freshness_seconds"] == 0.0
    assert diagnostics["last_bar_within_policy_window"] is True


def test_candle_freshness_diagnostics_rounds_machine_age() -> None:
    diagnostics = _build_candle_freshness_diagnostics(
        last_bar_epoch=100.0,
        expected_end_epoch=2495.7353789806366,
        freshness_cutoff_epoch=50.0,
    )

    assert diagnostics["data_freshness_seconds"] == 2395.735


def test_no_data_context_uses_non_negative_history_position(monkeypatch) -> None:
    calls = []

    def fake_copy_rates_from_pos(symbol, timeframe, start_pos, count):
        calls.append((symbol, timeframe, start_pos, count))
        return [{"time": 100.0}, {"time": 200.0}]

    monkeypatch.setattr(
        "mtdata.services.data_service._mt5_copy_rates_from_pos",
        fake_copy_rates_from_pos,
    )

    result = _build_no_data_error_with_context(
        "EURUSD",
        "H1",
        1,
        "1970-01-01 00:00:01",
        None,
    )

    assert calls == [("EURUSD", 1, 0, 100_000)]
    assert result["success"] is False
    assert result["error_code"] == "data_fetch_candles_no_data"
    assert result["operation"] == "data_fetch_candles"
    assert result["request_id"]
    assert result["details"]["available_range"] == {
        "earliest": "1970-01-01T00:01Z",
        "latest": "1970-01-01T00:03Z",
    }
    assert "before earliest available data" in result["error"]


def test_no_data_context_explains_bounded_weekend_closure(monkeypatch) -> None:
    monkeypatch.setattr(
        "mtdata.services.data_service._mt5_copy_rates_from_pos",
        lambda *args, **kwargs: None,
    )

    result = _build_no_data_error_with_context(
        "EURUSD",
        "H1",
        1,
        "2026-07-11",
        "2026-07-12",
    )

    assert result["details"]["no_data_reason"] == "market_closed_weekend"
    assert result["details"]["market_status_reason"] == "weekend"
    assert "no candles are expected" in result["details"]["note"]


def test_no_data_context_does_not_label_continuous_crypto_weekend(monkeypatch) -> None:
    monkeypatch.setattr(
        "mtdata.services.data_service._mt5_copy_rates_from_pos",
        lambda *args, **kwargs: None,
    )

    result = _build_no_data_error_with_context(
        "BTCUSD",
        "H1",
        1,
        "2026-07-11",
        "2026-07-12",
    )

    assert "no_data_reason" not in result["details"]


def test_compact_tick_summary_preserves_false_like_spread_availability() -> None:
    class FalseLike:
        def __bool__(self):
            return False

    payload = {
        "success": True,
        "stats": {"spread": {"available": FalseLike(), "low": 1.0, "high": 2.0}},
    }

    result = _compact_tick_summary(payload)

    assert result["stats"]["spread"] == {"available": False}


# ============================================================================
# TestFetchRatesWithWarmup
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
        self.assertIn('Could not parse date', err)

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
        self.assertIn('Could not parse date', err)

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
    def test_equal_start_and_end_is_allowed(self, mock_parse, mock_range):
        """Inclusive MT5 ranges allow a single timestamp boundary."""
        instant = datetime(2025, 1, 1, tzinfo=_UTC)
        mock_parse.side_effect = [instant, instant]
        mock_range.return_value = _make_rates(1, base_ts=instant.timestamp())

        result, err = _fetch_rates_with_warmup(
            'EURUSD', 16385, 'H1', 5, 0, '2025-01-01', '2025-01-01',
            retry=False, sanity_check=False,
        )

        self.assertIsNone(err)
        self.assertIsNotNone(result)
        mock_range.assert_called_once()

    @patch(_RATES_RANGE)
    @patch(_PARSE_START)
    def test_future_start_with_end_returns_error(self, mock_parse, mock_range):
        """A start in the future yields no historical data and must error."""
        mock_parse.side_effect = [
            datetime(2099, 1, 1, tzinfo=_UTC),
            datetime(2099, 2, 1, tzinfo=_UTC),
        ]
        result, err = _fetch_rates_with_warmup(
            'EURUSD', 16385, 'H1', 5, 0, '2099-01-01', '2099-02-01',
            retry=False, sanity_check=False,
        )
        self.assertIsNone(result)
        self.assertIn('future', err)
        mock_range.assert_not_called()

    @patch(_RATES_RANGE)
    @patch(_PARSE_START)
    def test_future_start_only_returns_error(self, mock_parse, mock_range):
        """A future start without end must error rather than silently empty."""
        mock_parse.return_value = datetime(2099, 1, 1, tzinfo=_UTC)
        result, err = _fetch_rates_with_warmup(
            'EURUSD', 16385, 'H1', 5, 0, '2099-01-01', None,
            retry=False, sanity_check=False,
        )
        self.assertIsNone(result)
        self.assertIn('future', err)
        mock_range.assert_not_called()

    @patch(_RATES_FROM)
    @patch(_PARSE_START)
    def test_start_only(self, mock_parse, mock_from):
        """Only start_datetime provided."""
        t1 = datetime(2025, 1, 1, tzinfo=_UTC)
        mock_parse.return_value = t1
        rates = _make_rates(5, base_ts=t1.timestamp() + 600)
        mock_from.return_value = rates
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
        self.assertIn('Could not parse date', err)

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
        self.assertIn('Could not parse date', err)

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
    def test_include_incomplete_does_not_relax_unverified_stale_policy(self, mock_from):
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
        self.assertIn('allow_stale=true', err)
        self.assertEqual(mock_from.call_count, 2)
        self.assertNotIn('freshness_policy_relaxed', diagnostics['freshness'])

    @patch(_RATES_FROM)
    def test_live_completed_bars_require_verified_closed_session(self, mock_from):
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

        self.assertIsNone(result)
        self.assertIn('allow_stale=true', err)
        self.assertEqual(mock_from.call_count, 2)
        self.assertNotIn('freshness_policy_relaxed', diagnostics['freshness'])

    @patch(_RATES_FROM)
    def test_weekend_completed_bars_report_closed_weekend(self, mock_from):
        now = datetime(2026, 6, 13, 12, 0, tzinfo=_UTC)
        latest = datetime(2026, 6, 12, 20, 0, tzinfo=_UTC)
        stale_rates = _make_rates(
            5,
            base_ts=latest.timestamp(),
            step=60 * 60,
        )
        mock_from.return_value = stale_rates
        diagnostics = {}

        with patch(f'{_DS}._utc_epoch_seconds', return_value=now.timestamp()):
            result, err = _fetch_rates_with_warmup(
                'EURUSD', 16385, 'H1', 5, 0, None, None,
                retry=False, sanity_check=True, diagnostics=diagnostics,
            )

        self.assertIsNone(err)
        self.assertEqual(result, stale_rates)
        freshness = diagnostics['freshness']
        self.assertTrue(freshness['freshness_policy_relaxed'])
        self.assertEqual(freshness['market_session_status'], 'closed')
        self.assertEqual(freshness['market_session_reason'], 'weekend')

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
                'data_freshness_anchor': 'query_expected_end',
                'data_freshness_metric': 'requested_range_end_gap_seconds',
                'last_bar_within_policy_window': True,
            },
        )


# ============================================================================
# TestBuildRatesDf
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
# TestTrimDfToTarget
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
        self.assertEqual(list(out['time']), [f"t{i}" for i in range(15, 20)])

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


if __name__ == '__main__':
    unittest.main()
def test_live_bar_reference_uses_wall_clock_when_tick_is_stale(monkeypatch):
    from mtdata.services import data_service

    monkeypatch.setattr(data_service, "_utc_epoch_seconds", lambda _value: 1_000.0)
    monkeypatch.setattr(
        data_service.mt5,
        "symbol_info_tick",
        lambda _symbol: type("Tick", (), {"time": 900.0})(),
    )

    assert data_service._resolve_live_bar_reference_epoch("EURUSD", "M1") == 1_000.0
