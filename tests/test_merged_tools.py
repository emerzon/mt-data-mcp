import unittest
from unittest.mock import MagicMock, patch
import sys
import pandas as pd

# Mock mt5 before importing the module
sys.modules['MetaTrader5'] = MagicMock()
import MetaTrader5 as mt5

# Mock pandas_ta
pd.DataFrame.ta = MagicMock()

# Now import the tools
# We need to make sure the path is in sys.path
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.mtdata.core.trading import trading_open_get
from src.mtdata.core.patterns import patterns_detect
from src.mtdata.core.forecast import forecast_barrier_prob

class TestMergedTools(unittest.TestCase):
    def test_trading_open_get_positions(self):
        # Setup mock
        mt5.positions_get.return_value = None # Simulate empty

        # Test default
        res = trading_open_get(__cli_raw=True)
        self.assertIsInstance(res, list)

        # Test with symbol
        trading_open_get(symbol="EURUSD", __cli_raw=True)
        mt5.positions_get.assert_called_with(symbol="EURUSD")

        # Test with ticket
        trading_open_get(ticket=123, __cli_raw=True)
        mt5.positions_get.assert_called_with(ticket=123)

    def test_trading_open_get_pending(self):
        mt5.orders_get.return_value = None

        trading_open_get(open_kind="pending", __cli_raw=True)

        trading_open_get(open_kind="pending", symbol="EURUSD", __cli_raw=True)
        mt5.orders_get.assert_called_with(symbol="EURUSD")

        trading_open_get(open_kind="pending", ticket=123, __cli_raw=True)
        mt5.orders_get.assert_called_with(ticket=123)

    def test_patterns_detect(self):
        # Mock symbol info
        mt5.symbol_info.return_value = MagicMock(visible=True)
        # Mock copy rates to return None or empty to trigger error handling, which is fine for signature check
        mt5.copy_rates_from.return_value = None
        
        res = patterns_detect(symbol="EURUSD", mode="candlestick", __cli_raw=True)
        # It might return error because of mocked mt5 returning None for rates
        self.assertTrue("error" in res or "success" in res)
        
        res = patterns_detect(symbol="EURUSD", mode="classic", __cli_raw=True)
        self.assertTrue("error" in res or "success" in res)

    def test_forecast_barrier_prob(self):
        # Mock the internal implementations which are imported inside the function
        
        with patch('src.mtdata.forecast.barriers.forecast_barrier_hit_probabilities') as mock_mc:
            mock_mc.return_value = {"success": True}
            res = forecast_barrier_prob(symbol="EURUSD", method="mc", __cli_raw=True)
            self.assertEqual(res, {"success": True})
            
        with patch('src.mtdata.forecast.barriers.forecast_barrier_closed_form') as mock_cf:
            mock_cf.return_value = {"success": True}
            res = forecast_barrier_prob(symbol="EURUSD", method="closed_form", __cli_raw=True)
            self.assertEqual(res, {"success": True})

    def test_forecast_barrier_prob_direction_normalization(self):
        with patch('src.mtdata.forecast.barriers.forecast_barrier_hit_probabilities') as mock_mc:
            mock_mc.return_value = {"success": True}
            forecast_barrier_prob(symbol="EURUSD", method="mc", direction="LONG", __cli_raw=True)
            self.assertEqual(mock_mc.call_args.kwargs.get("direction"), "long")

        with patch('src.mtdata.forecast.barriers.forecast_barrier_closed_form') as mock_cf:
            mock_cf.return_value = {"success": True}
            forecast_barrier_prob(symbol="EURUSD", method="closed_form", direction="UP", __cli_raw=True)
            self.assertEqual(mock_cf.call_args.kwargs.get("direction"), "up")

    def test_trading_close_positions(self):
        # Mock positions
        mock_pos = MagicMock()
        mock_pos.ticket = 123
        mock_pos.symbol = "EURUSD"
        mock_pos.type = 0 # BUY
        mock_pos.volume = 1.0
        mock_pos.profit = 10.0
        
        mt5.positions_get.return_value = [mock_pos]
        mt5.symbol_info_tick.return_value = MagicMock(bid=1.0, ask=1.0)
        mt5.order_send.return_value = MagicMock(retcode=mt5.TRADE_RETCODE_DONE)
        
        from src.mtdata.core.trading import trading_close

        # Test close by ticket
        trading_close(ticket=123, __cli_raw=True)
        mt5.positions_get.assert_called_with(ticket=123)

        # Test close by symbol
        trading_close(symbol="EURUSD", __cli_raw=True)
        mt5.positions_get.assert_called_with(symbol="EURUSD")

        # Test close all
        trading_close(__cli_raw=True)
        mt5.positions_get.assert_called_with()

    def test_trading_close_pending(self):
        # Mock orders
        mock_order = MagicMock()
        mock_order.ticket = 456

        mt5.orders_get.return_value = [mock_order]
        mt5.order_send.return_value = MagicMock(retcode=mt5.TRADE_RETCODE_DONE) 

        from src.mtdata.core.trading import trading_close

        # Test cancel by ticket
        trading_close(close_kind="pending", ticket=456, __cli_raw=True)
        mt5.orders_get.assert_called_with(ticket=456)

        # Test cancel by symbol
        trading_close(close_kind="pending", symbol="EURUSD", __cli_raw=True)
        mt5.orders_get.assert_called_with(symbol="EURUSD")

        # Test cancel all
        trading_close(close_kind="pending", __cli_raw=True)
        mt5.orders_get.assert_called_with()

if __name__ == '__main__':
    unittest.main()
