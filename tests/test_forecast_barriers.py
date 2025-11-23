
import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from mtdata.forecast.barriers import forecast_barrier_hit_probabilities, forecast_barrier_closed_form, forecast_barrier_optimize

class TestForecastBarriers(unittest.TestCase):

    def setUp(self):
        self.mock_mt5 = MagicMock()
        self.mock_mt5.symbol_info.return_value = MagicMock(digits=5, point=0.00001)
        
        # Patch mt5 in barriers module
        self.mt5_patcher = patch('mtdata.forecast.barriers.mt5', self.mock_mt5)
        self.mt5_patcher.start()
        
        # Mock fetch_history
        self.fetch_history_patcher = patch('mtdata.forecast.barriers._fetch_history')
        self.mock_fetch_history = self.fetch_history_patcher.start()
        
        # Create dummy dataframe
        dates = pd.date_range(start='2023-01-01', periods=500, freq='H')
        prices = np.linspace(1.0, 1.1, 500) + np.random.normal(0, 0.001, 500)
        self.df = pd.DataFrame({'time': dates, 'close': prices})
        self.mock_fetch_history.return_value = self.df

    def tearDown(self):
        self.mt5_patcher.stop()
        self.fetch_history_patcher.stop()

    def test_forecast_barrier_hit_probabilities(self):
        result = forecast_barrier_hit_probabilities(
            symbol="EURUSD",
            timeframe="H1",
            horizon=10,
            method="mc_gbm",
            direction="long",
            tp_pct=0.5,
            sl_pct=0.5
        )
        self.assertIn("success", result)
        self.assertTrue(result["success"])
        self.assertIn("prob_tp_first", result)
        self.assertIn("prob_sl_first", result)

    def test_forecast_barrier_closed_form(self):
        result = forecast_barrier_closed_form(
            symbol="EURUSD",
            timeframe="H1",
            horizon=10,
            direction="up",
            barrier=1.2
        )
        self.assertIn("success", result)
        self.assertTrue(result["success"])
        self.assertIn("prob_hit", result)

    def test_forecast_barrier_optimize(self):
        result = forecast_barrier_optimize(
            symbol="EURUSD",
            timeframe="H1",
            horizon=10,
            method="mc_gbm",
            direction="long",
            mode="pct",
            tp_min=0.1, tp_max=0.5, tp_steps=3,
            sl_min=0.1, sl_max=0.5, sl_steps=3,
            objective="edge"
        )
        self.assertIn("success", result)
        self.assertTrue(result["success"])
        self.assertIn("best", result)
        self.assertIn("grid", result)

if __name__ == '__main__':
    unittest.main()
