import math
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import types

if 'MetaTrader5' not in sys.modules:
    mt5_mock = types.ModuleType('MetaTrader5')
    mt5_mock.symbol_info = lambda *args, **kwargs: types.SimpleNamespace(visible=True)
    mt5_mock.symbol_select = lambda *args, **kwargs: True
    mt5_mock.symbol_info_tick = lambda *args, **kwargs: types.SimpleNamespace(time=0)
    mt5_mock.copy_rates_from = lambda *args, **kwargs: []
    mt5_mock.last_error = lambda: (0, '')
    # Provide timeframe constants consumed by core.constants
    mt5_mock.TIMEFRAME_M1 = 1
    mt5_mock.TIMEFRAME_M2 = 2
    mt5_mock.TIMEFRAME_M3 = 3
    mt5_mock.TIMEFRAME_M4 = 4
    mt5_mock.TIMEFRAME_M5 = 5
    mt5_mock.TIMEFRAME_M6 = 6
    mt5_mock.TIMEFRAME_M10 = 10
    mt5_mock.TIMEFRAME_M12 = 12
    mt5_mock.TIMEFRAME_M15 = 15
    mt5_mock.TIMEFRAME_M20 = 20
    mt5_mock.TIMEFRAME_M30 = 30
    mt5_mock.TIMEFRAME_H1 = 60
    mt5_mock.TIMEFRAME_H2 = 120
    mt5_mock.TIMEFRAME_H3 = 180
    mt5_mock.TIMEFRAME_H4 = 240
    mt5_mock.TIMEFRAME_H6 = 360
    mt5_mock.TIMEFRAME_H8 = 480
    mt5_mock.TIMEFRAME_H12 = 720
    mt5_mock.TIMEFRAME_D1 = 1440
    mt5_mock.TIMEFRAME_W1 = 10080
    mt5_mock.TIMEFRAME_MN1 = 43200
    sys.modules['MetaTrader5'] = mt5_mock

from mtdata.forecast.backtest import forecast_backtest, _compute_performance_metrics
from mtdata.forecast.volatility import _realized_kernel_variance
from mtdata.forecast.forecast_engine import forecast_engine


class BacktestEnhancementsTest(unittest.TestCase):
    @patch('mtdata.forecast.backtest.forecast')
    @patch('mtdata.forecast.backtest._fetch_history')
    @patch('mtdata.forecast.backtest._ensure_symbol_ready', return_value=None)
    def test_backtest_metrics_include_slippage(self, mock_ensure_ready, mock_fetch_history, mock_forecast):
        closes = np.linspace(100.0, 110.0, 40)
        times = np.arange(len(closes), dtype=float)
        df = pd.DataFrame({'time': times, 'close': closes})
        mock_fetch_history.return_value = df

        mock_forecast.return_value = {'forecast_price': [200.0, 201.0]}

        result = forecast_backtest(
            symbol='TEST',
            timeframe='H1',
            horizon=2,
            steps=4,
            spacing=3,
            methods=['naive'],
            slippage_bps=10.0,
        )

        self.assertTrue(result.get('success'))
        method_result = result['results']['naive']
        self.assertIn('metrics', method_result)
        metrics = method_result['metrics']
        self.assertIn('sharpe_ratio', metrics)
        self.assertIn('max_drawdown', metrics)
        self.assertIn('calmar_ratio', metrics)
        self.assertAlmostEqual(method_result['slippage_bps'], 10.0)
        self.assertAlmostEqual(metrics['slippage_bps'], 10.0)
        self.assertAlmostEqual(method_result['avg_trade_return'], metrics['avg_return_per_trade'])
        self.assertGreaterEqual(metrics['win_rate'], 0.0)

    def test_compute_performance_metrics_handles_positive_returns(self):
        returns = [0.02, 0.015, -0.005, 0.01]
        metrics = _compute_performance_metrics(returns, timeframe='H1', horizon=2, slippage_bps=5.0)
        self.assertAlmostEqual(metrics['avg_return_per_trade'], np.mean(returns))
        self.assertIn('sharpe_ratio', metrics)
        self.assertTrue(math.isfinite(metrics['sharpe_ratio']))


class VolatilityEnhancementsTest(unittest.TestCase):
    def test_realized_kernel_variance_positive(self):
        returns = np.array([0.01, -0.015, 0.02, -0.005, 0.012, -0.003, 0.008], dtype=float)
        var = _realized_kernel_variance(returns, bandwidth=3)
        self.assertTrue(var >= 0.0)
        self.assertTrue(math.isfinite(var))


class EnsembleEnhancementsTest(unittest.TestCase):
    @patch('mtdata.forecast.forecast_engine._fetch_history')
    def test_ensemble_bma_metadata(self, mock_fetch_history):
        closes = np.linspace(100.0, 120.0, 80)
        epochs = np.arange(len(closes), dtype=float)
        df = pd.DataFrame({'close': closes, '__epoch': epochs, 'time': epochs})
        mock_fetch_history.return_value = df

        response = forecast_engine(
            symbol='TEST',
            timeframe='H1',
            method='ensemble',
            horizon=3,
            params={
                'methods': ['naive', 'drift'],
                'mode': 'bma',
                'cv_points': 10,
                'min_train_size': 5,
                'expose_components': True,
            },
        )
        self.assertTrue(response.get('success'))
        ensemble_meta = response.get('ensemble')
        self.assertIsNotNone(ensemble_meta)
        self.assertEqual(ensemble_meta.get('mode_used'), 'bma')
        self.assertEqual(len(ensemble_meta.get('weights', [])), 2)
        self.assertIn('components', ensemble_meta)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
