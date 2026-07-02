"""Tests for fetch_candles indicator processing and NaN warmup-retry logic.

Covers:
  - Indicator specs: list-of-dicts, string, named-string, JSON
  - Normalisation of indicator display names
  - Unknown / malformed indicator errors
  - NaN warmup retry: success, empty-refetch, transient-empty, rebuild error,
    freshness-policy preservation
"""

import unittest
from datetime import datetime
from unittest.mock import MagicMock, patch

from ._helpers import (
    _UTC,
    _mock_symbol_guard,
    _make_rates,
    _DS, _GUARD, _RATES_FROM,
    _CACHED_INFO, _RESOLVE_CTZ,
    _ESTIMATE_WARMUP, _APPLY_TI, _MT5_CONFIG,
    _PARSE_START,
)

from mtdata.services.data_service import fetch_candles


class TestFetchCandlesIndicators(unittest.TestCase):
    """Indicator spec parsing, application, and NaN-warmup retry for fetch_candles."""

    # ------------------------------------------------------------------ #
    # Valid indicator specs                                               #
    # ------------------------------------------------------------------ #

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
            ['rsi_14', 'ema_20', 'atr_14'],
        )
        self.assertIn('rsi_14', result['data'][0])
        self.assertIn('ema_20', result['data'][0])
        self.assertIn('atr_14', result['data'][0])
        self.assertNotIn('RSI_14', result['data'][0])
        self.assertNotIn('EMA_20', result['data'][0])
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
            df['atr_14'] = 99.0
            df['ATRr_14'] = 2.2
            return ['ATRr_14']

        mock_ti.side_effect = add_cols

        result = fetch_candles('EURUSD', limit=10, indicators='ATR(14.0)')

        self.assertTrue(result.get('success'))
        self.assertEqual(
            result['meta']['diagnostics']['indicators']['added_columns'],
            ['ATRr_14', 'atr_14'],
        )
        self.assertIn('ATRr_14', result['data'][0])
        self.assertEqual(result['data'][0]['ATRr_14'], 2.2)
        self.assertEqual(result['data'][0]['atr_14'], 99.0)

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

    # ------------------------------------------------------------------ #
    # Invalid indicator specs                                             #
    # ------------------------------------------------------------------ #

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

    # ------------------------------------------------------------------ #
    # NaN warmup retry                                                    #
    # ------------------------------------------------------------------ #

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
        # _apply_ta_indicators should have been called twice (original + retry)
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
    def test_nan_warmup_retry_drops_incomplete_rows_when_refetch_is_empty(
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
            df['rsi_14'] = [float('nan')] * max(0, len(df) - 3) + [50.0, 51.0, 52.0]
            return ['rsi_14']

        mock_ti.side_effect = add_nan_col

        result = fetch_candles('EURUSD', limit=5, indicators=[{'name': 'rsi', 'params': [14]}])

        self.assertTrue(result.get('success'))
        self.assertEqual(result['returned_count'], 2)
        self.assertTrue(all(row['rsi_14'] is not None for row in result['data']))
        self.assertEqual(result['candle_counts']['excluded']['indicator_warmup'], 1)
        warmup_retry = result['meta']['diagnostics']['query']['warmup_retry']
        self.assertFalse(warmup_retry['applied'])
        self.assertEqual(warmup_retry['raw_bars_fetched'], 0)
        self.assertEqual(warmup_retry['incomplete_rows_dropped'], 2)
        self.assertEqual(result['meta']['diagnostics']['query']['indicator_rows_dropped'], 2)

    @patch(_MT5_CONFIG)
    @patch(_APPLY_TI)
    @patch(f'{_DS}.FETCH_RETRY_DELAY', 0)
    @patch(f'{_DS}.FETCH_RETRY_ATTEMPTS', 1)
    @patch(_RATES_FROM)
    @patch(_CACHED_INFO, return_value=MagicMock())
    @patch(_RESOLVE_CTZ, return_value=None)
    @patch(_ESTIMATE_WARMUP, return_value=14)
    @patch(_GUARD, _mock_symbol_guard)
    def test_nan_warmup_retry_fails_when_no_complete_rows_remain(
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

        self.assertFalse(result.get('success'))
        self.assertEqual(result['error_code'], 'data_fetch_candles_incomplete_indicators')
        self.assertEqual(result['indicator_columns'], ['rsi_14'])
        warmup_retry = result['meta']['diagnostics']['query']['warmup_retry']
        self.assertFalse(warmup_retry['applied'])
        self.assertEqual(warmup_retry['raw_bars_fetched'], 0)
        self.assertEqual(warmup_retry['incomplete_rows_dropped'], 5)

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

        self.assertFalse(result.get('success'))
        self.assertEqual(result['error_code'], 'data_fetch_candles_incomplete_indicators')
        warmup_retry = result['meta']['diagnostics']['query']['warmup_retry']
        self.assertTrue(warmup_retry['applied'])
        self.assertEqual(warmup_retry['error'], 'retry boom')
        self.assertTrue(any('Indicator warmup retry failed: retry boom' in str(w) for w in result.get('warnings', [])))
        self.assertEqual(warmup_retry['incomplete_rows_dropped'], 5)
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
    def test_nan_warmup_retry_allows_historical_end_retry_window(
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
        self.assertEqual(mock_ti.call_count, 2)
        self.assertEqual(float(result['data'][-1]['time']), float(stale_rates[-1]['time']))
        warmup_retry = result['meta']['diagnostics']['query']['warmup_retry']
        self.assertTrue(warmup_retry['applied'])
        self.assertEqual(warmup_retry['raw_bars_fetched'], 60)


if __name__ == '__main__':
    unittest.main()
