"""Tests for fetch_candles simplify and denoise features.

Covers:
  - Simplify: basic passthrough, row reduction, no explicit points
  - Denoise: pre-TI, post-TI, warning surfacing
"""

import unittest
from unittest.mock import MagicMock, patch

from ._helpers import (
    _mock_symbol_guard,
    _make_rates,
    _DS, _GUARD, _RATES_FROM,
    _CACHED_INFO, _RESOLVE_CTZ,
    _ESTIMATE_WARMUP, _SIMPLIFY_EXT, _MT5_CONFIG,
)

from mtdata.services.data_service import fetch_candles


class TestFetchCandlesAdvanced(unittest.TestCase):
    """Simplify and denoise tests for fetch_candles."""

    # ------------------------------------------------------------------ #
    # Simplify                                                            #
    # ------------------------------------------------------------------ #

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
        self.assertEqual(result['series_type'], 'downsampled_visualization')
        self.assertFalse(result['equal_interval'])
        self.assertFalse(result['analysis_compatible'])
        self.assertIn('irregular time gaps', result['warnings'][0])

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

    # ------------------------------------------------------------------ #
    # Denoise                                                             #
    # ------------------------------------------------------------------ #

    @patch(_MT5_CONFIG)
    @patch(f'{_DS}._normalize_denoise_spec')
    @patch(f'{_DS}.apply_denoise_util', return_value=[])
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
    @patch(f'{_DS}.apply_denoise_util', return_value=['close_dn'])
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
            self.assertEqual(
                result['denoise']['applications'][0]['causality'],
                'causal',
            )

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

        with patch(f'{_DS}.apply_denoise_util', side_effect=add_warning):
            result = fetch_candles('EURUSD', limit=5, denoise={'method': 'wavelet'})

        self.assertTrue(result.get('success'))
        self.assertIn('warnings', result)
        self.assertIn("requires PyWavelets", result['warnings'][0])


if __name__ == '__main__':
    unittest.main()

