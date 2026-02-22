import unittest
from unittest.mock import MagicMock, patch
import sys
import pandas as pd
from collections import namedtuple

from src.mtdata.utils.mt5 import _mt5_epoch_to_utc
from src.mtdata.utils.utils import _format_time_minimal, _format_time_minimal_local, _use_client_tz

# Mock mt5 before importing the module
sys.modules['MetaTrader5'] = MagicMock()
import MetaTrader5 as mt5


# Now import the tools
# We need to make sure the path is in sys.path
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.mtdata.core.trading import trade_get_open
from src.mtdata.core.trading import trade_get_pending
from src.mtdata.core.patterns import patterns_detect
from src.mtdata.core.forecast import forecast_barrier_prob

class TestMergedTools(unittest.TestCase):
    def test_trading_open_get_positions(self):
        # Setup mock
        mt5.positions_get.return_value = None # Simulate empty

        # Test default
        res = trade_get_open(__cli_raw=True)
        self.assertIsInstance(res, list)

        # Test with symbol
        trade_get_open(symbol="EURUSD", __cli_raw=True)
        mt5.positions_get.assert_called_with(symbol="EURUSD")

        # Test with ticket
        trade_get_open(ticket=123, __cli_raw=True)
        mt5.positions_get.assert_called_with(ticket=123)

    def test_trading_open_get_positions_type_translation(self):
        Pos = namedtuple("Pos", ["ticket", "time", "time_msc", "time_update", "time_update_msc", "type", "symbol"])
        mt5.positions_get.return_value = [
            Pos(
                ticket=1,
                time=1700000000,
                time_msc=1700000000000,
                time_update=1700000001,
                time_update_msc=1700000001000,
                type=0,
                symbol="EURUSD",
            )
        ]

        res = trade_get_open(__cli_raw=True)
        self.assertIsInstance(res, list)
        self.assertGreaterEqual(len(res), 1)
        self.assertEqual(res[0].get("Type"), "BUY")
        self.assertEqual(res[0].get("Symbol"), "EURUSD")
        self.assertEqual(res[0].get("Ticket"), 1)
        fmt_time = _format_time_minimal_local if _use_client_tz() else _format_time_minimal
        expected_time = fmt_time(_mt5_epoch_to_utc(1700000001))
        self.assertEqual(res[0].get("Time"), expected_time)
        self.assertIsNone(res[0].get("time_msc"))
        self.assertIsNone(res[0].get("time_update"))
        self.assertIsNone(res[0].get("time_update_msc"))
        self.assertIsNone(res[0].get("direction"))
        self.assertIsNone(res[0].get("type_code"))

    def test_trading_open_get_pending(self):
        mt5.orders_get.return_value = None

        trade_get_pending(__cli_raw=True)

        trade_get_pending(symbol="EURUSD", __cli_raw=True)
        mt5.orders_get.assert_called_with(symbol="EURUSD")

        trade_get_pending(ticket=123, __cli_raw=True)
        mt5.orders_get.assert_called_with(ticket=123)

    def test_trading_open_get_pending_type_translation(self):
        Order = namedtuple("Order", ["ticket", "time_setup", "time_setup_msc", "time_expiration", "type", "symbol"])
        mt5.orders_get.return_value = [
            Order(
                ticket=1,
                time_setup=1700000000,
                time_setup_msc=1700000000000,
                time_expiration=1700003600,
                type=3,
                symbol="EURUSD",
            )
        ]

        res = trade_get_pending(__cli_raw=True)
        self.assertIsInstance(res, list)
        self.assertGreaterEqual(len(res), 1)
        self.assertEqual(res[0].get("Type"), "SELL_LIMIT")
        self.assertEqual(res[0].get("Symbol"), "EURUSD")
        self.assertEqual(res[0].get("Ticket"), 1)
        fmt_time = _format_time_minimal_local if _use_client_tz() else _format_time_minimal
        expected_time = fmt_time(_mt5_epoch_to_utc(1700000000))
        expected_exp = fmt_time(_mt5_epoch_to_utc(1700003600))
        self.assertEqual(res[0].get("Time"), expected_time)
        self.assertIn("Expiration", res[0])
        self.assertEqual(res[0].get("Expiration"), expected_exp)
        self.assertNotIn("Profit", res[0])
        self.assertNotIn("Swap", res[0])
        self.assertIsNone(res[0].get("time_setup"))
        self.assertIsNone(res[0].get("time_setup_msc"))
        self.assertIsNone(res[0].get("direction"))
        self.assertIsNone(res[0].get("type_code"))

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
            forecast_barrier_prob(symbol="EURUSD", method="closed_form", direction="SHORT", __cli_raw=True)
            self.assertEqual(mock_cf.call_args.kwargs.get("direction"), "short")

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
        
        from src.mtdata.core.trading import trade_close

        # Test close by ticket
        trade_close(ticket=123, __cli_raw=True)
        mt5.positions_get.assert_called_with(ticket=123)

        # Test close by symbol
        trade_close(symbol="EURUSD", __cli_raw=True)
        mt5.positions_get.assert_called_with(symbol="EURUSD")

        # Test close all
        trade_close(__cli_raw=True)
        mt5.positions_get.assert_called_with()

    def test_trading_close_pending(self):
        # Mock orders
        mock_order = MagicMock()
        mock_order.ticket = 456

        mt5.orders_get.return_value = [mock_order]
        mt5.order_send.return_value = MagicMock(retcode=mt5.TRADE_RETCODE_DONE) 

        from src.mtdata.core.trading import trade_close

        # Test cancel by ticket
        trade_close(close_kind="pending", ticket=456, __cli_raw=True)
        mt5.orders_get.assert_called_with(ticket=456)

        # Test cancel by symbol
        trade_close(close_kind="pending", symbol="EURUSD", __cli_raw=True)
        mt5.orders_get.assert_called_with(symbol="EURUSD")

        # Test cancel all
        trade_close(close_kind="pending", __cli_raw=True)
        mt5.orders_get.assert_called_with()

if __name__ == '__main__':
    unittest.main()
