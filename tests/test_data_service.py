

import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add src to path to ensure local package is found
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

# Mock mt5 before importing data_service because it imports mt5 at top level
sys.modules['MetaTrader5'] = MagicMock()
import MetaTrader5 as mt5

from mtdata.services.data_service import fetch_candles, fetch_ticks

class TestDataService(unittest.TestCase):
    
    @patch('mtdata.utils.mt5._mt5_copy_rates_from')
    @patch('mtdata.services.data_service._ensure_symbol_ready')
    def test_fetch_candles_basic(self, mock_ensure, mock_copy_rates):
        # Setup
        mock_ensure.return_value = None
        
        # Mock rates data
        now = datetime.utcnow()
        rates = []
        for i in range(10):
            t = now - timedelta(minutes=10-i)
            rates.append({
                'time': t.timestamp(),
                'open': 1.1, 'high': 1.2, 'low': 1.0, 'close': 1.15,
                'tick_volume': 100, 'real_volume': 0, 'spread': 1
            })
        mock_copy_rates.return_value = np.array(rates) # usually returns numpy array of void, but list of dicts might work for DataFrame if structured correctly?
        # Actually mt5 returns numpy structured array.
        # Let's just return a list of dicts, pd.DataFrame(rates) handles it.
        
        # Execute
        result = fetch_candles(symbol="EURUSD", limit=5)
        
        # Verify
        self.assertIsInstance(result, dict)
        self.assertTrue(result.get('success'))
        self.assertEqual(result.get('candles'), 5)
        # Check CSV content
        csv_data = result.get('data')
        self.assertIn('time,open,high,low,close', csv_data)
        
    @patch('mtdata.utils.mt5._mt5_copy_ticks_from')
    @patch('mtdata.services.data_service._ensure_symbol_ready')
    def test_fetch_ticks_basic(self, mock_ensure, mock_copy_ticks):
        # Setup
        mock_ensure.return_value = None
        
        # Mock ticks data
        now = datetime.utcnow()
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

if __name__ == '__main__':
    try:
        unittest.main(exit=False)
        with open("test_service_results.txt", "w") as f:
            f.write("SUCCESS")
    except Exception as e:
        with open("test_service_results.txt", "w") as f:
            f.write(f"FAILURE: {e}")
