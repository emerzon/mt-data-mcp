import unittest
from unittest.mock import MagicMock, patch
import sys
from collections import namedtuple

from src.mtdata.utils.mt5 import _mt5_epoch_to_utc
from src.mtdata.utils.utils import _format_time_minimal, _format_time_minimal_local, _use_client_tz

# Mock mt5 before importing the module
sys.modules['MetaTrader5'] = MagicMock()


# Now import the tools
# We need to make sure the path is in sys.path
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.mtdata.core.trading import trade_get_open
from src.mtdata.core.trading import trade_get_pending
from src.mtdata.core.patterns import patterns_detect
from src.mtdata.core.forecast import forecast_barrier_prob
from src.mtdata.core.trading.requests import TradeGetOpenRequest, TradeGetPendingRequest
from src.mtdata.core.patterns_requests import PatternsDetectRequest
from src.mtdata.forecast.requests import ForecastBarrierProbRequest


def get_open(**kwargs):
    raw_output = bool(kwargs.pop("__cli_raw", False))
    request = kwargs.pop("request", None)
    if request is None:
        request = TradeGetOpenRequest(**kwargs)
    return trade_get_open(request=request, __cli_raw=raw_output)


def get_pending(**kwargs):
    raw_output = bool(kwargs.pop("__cli_raw", False))
    request = kwargs.pop("request", None)
    if request is None:
        request = TradeGetPendingRequest(**kwargs)
    return trade_get_pending(request=request, __cli_raw=raw_output)


def barrier_prob(**kwargs):
    raw_output = bool(kwargs.pop("__cli_raw", False))
    request = kwargs.pop("request", None)
    if request is None:
        request = ForecastBarrierProbRequest(**kwargs)
    return forecast_barrier_prob(request=request, __cli_raw=raw_output)


def detect_patterns(**kwargs):
    raw_output = bool(kwargs.pop("__cli_raw", False))
    request = kwargs.pop("request", None)
    if request is None:
        request = PatternsDetectRequest(**kwargs)
    return patterns_detect(request=request, __cli_raw=raw_output)

class TestMergedTools(unittest.TestCase):
    def setUp(self):
        # Create a fresh mock and install it so that trading functions
        # (which do ``import MetaTrader5 as mt5`` at call time) pick it up.
        self.mt5 = MagicMock()
        sys.modules['MetaTrader5'] = self.mt5

    def test_trading_open_get_positions(self):
        # Setup mock
        self.mt5.positions_get.return_value = None # Simulate empty

        # Test default
        res = get_open(__cli_raw=True)
        self.assertIsInstance(res, dict)
        self.assertTrue(res.get("success"))
        self.assertEqual(res.get("count"), 0)
        self.assertTrue(res.get("no_action"))

        # Test with symbol
        get_open(symbol="EURUSD", __cli_raw=True)
        self.mt5.positions_get.assert_called_with(symbol="EURUSD")

        # Test with ticket
        get_open(ticket=123, __cli_raw=True)
        self.mt5.positions_get.assert_called_with(ticket=123)

    def test_trading_open_get_positions_type_translation(self):
        Pos = namedtuple("Pos", ["ticket", "time", "time_msc", "time_update", "time_update_msc", "type", "symbol"])
        self.mt5.positions_get.return_value = [
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

        res = get_open(__cli_raw=True)
        self.assertIsInstance(res, dict)
        self.assertGreaterEqual(res.get("count", 0), 1)
        row = res["items"][0]
        self.assertEqual(row.get("Type"), "BUY")
        self.assertEqual(row.get("Symbol"), "EURUSD")
        self.assertEqual(row.get("Ticket"), 1)
        fmt_time = _format_time_minimal_local if _use_client_tz() else _format_time_minimal
        expected_time = fmt_time(_mt5_epoch_to_utc(1700000001))
        self.assertEqual(row.get("Time"), expected_time)
        self.assertIsNone(row.get("time_msc"))
        self.assertIsNone(row.get("time_update"))
        self.assertIsNone(row.get("time_update_msc"))
        self.assertIsNone(row.get("direction"))
        self.assertIsNone(row.get("type_code"))

    def test_trading_open_comment_metadata_is_exposed(self):
        Pos = namedtuple(
            "Pos",
            [
                "ticket",
                "time",
                "time_msc",
                "time_update",
                "time_update_msc",
                "type",
                "symbol",
                "comment",
            ],
        )
        self.mt5.positions_get.return_value = [
            Pos(
                ticket=1,
                time=1700000000,
                time_msc=1700000000000,
                time_update=1700000001,
                time_update_msc=1700000001000,
                type=0,
                symbol="EURUSD",
                comment="audit short",
            )
        ]

        res = get_open(__cli_raw=True)
        row = res["items"][0]
        self.assertEqual(row.get("Comment Limit"), 31)
        self.assertEqual(row.get("Comment Length"), len("audit short"))
        self.assertFalse(row.get("Comment May Be Truncated"))

    def test_trading_open_get_pending(self):
        self.mt5.orders_get.return_value = None

        res = get_pending(__cli_raw=True)
        self.assertIsInstance(res, dict)
        self.assertTrue(res.get("success"))
        self.assertEqual(res.get("count"), 0)
        self.assertTrue(res.get("no_action"))

        get_pending(symbol="EURUSD", __cli_raw=True)
        self.mt5.orders_get.assert_called_with(symbol="EURUSD")

        get_pending(ticket=123, __cli_raw=True)
        self.mt5.orders_get.assert_called_with(ticket=123)

    def test_trading_open_get_pending_type_translation(self):
        Order = namedtuple("Order", ["ticket", "time_setup", "time_setup_msc", "time_expiration", "type", "symbol"])
        self.mt5.orders_get.return_value = [
            Order(
                ticket=1,
                time_setup=1700000000,
                time_setup_msc=1700000000000,
                time_expiration=1700003600,
                type=3,
                symbol="EURUSD",
            )
        ]

        res = get_pending(__cli_raw=True)
        self.assertIsInstance(res, dict)
        self.assertGreaterEqual(res.get("count", 0), 1)
        row = res["items"][0]
        self.assertEqual(row.get("Type"), "SELL_LIMIT")
        self.assertEqual(row.get("Symbol"), "EURUSD")
        self.assertEqual(row.get("Ticket"), 1)
        fmt_time = _format_time_minimal_local if _use_client_tz() else _format_time_minimal
        expected_time = fmt_time(_mt5_epoch_to_utc(1700000000))
        expected_exp = fmt_time(_mt5_epoch_to_utc(1700003600))
        self.assertEqual(row.get("Time"), expected_time)
        self.assertIn("Expiration", row)
        self.assertEqual(row.get("Expiration"), expected_exp)
        self.assertNotIn("Profit", row)
        self.assertNotIn("Swap", row)
        self.assertIsNone(row.get("time_setup"))
        self.assertIsNone(row.get("time_setup_msc"))
        self.assertIsNone(row.get("direction"))
        self.assertIsNone(row.get("type_code"))

    def test_patterns_detect(self):
        # Mock symbol info
        self.mt5.symbol_info.return_value = MagicMock(visible=True)
        # Mock copy rates to return None or empty to trigger error handling, which is fine for signature check
        self.mt5.copy_rates_from.return_value = None
        
        res = detect_patterns(symbol="EURUSD", mode="candlestick", __cli_raw=True)
        # It might return error because of mocked mt5 returning None for rates
        self.assertTrue("error" in res or "success" in res)
        
        res = detect_patterns(symbol="EURUSD", mode="classic", __cli_raw=True)
        self.assertTrue("error" in res or "success" in res)

    def test_forecast_barrier_prob(self):
        with patch('src.mtdata.forecast.barriers.forecast_barrier_hit_probabilities') as mock_mc:
            mock_mc.return_value = {"success": True}
            res = barrier_prob(symbol="EURUSD", method="mc", __cli_raw=True)
            self.assertEqual(res, {"success": True})
            
        with patch('src.mtdata.forecast.barriers.forecast_barrier_closed_form') as mock_cf:
            mock_cf.return_value = {"success": True}
            res = barrier_prob(symbol="EURUSD", method="closed_form", __cli_raw=True)
            self.assertEqual(res, {"success": True})

    def test_forecast_barrier_prob_direction_normalization(self):
        with patch('src.mtdata.forecast.barriers.forecast_barrier_hit_probabilities') as mock_mc:
            mock_mc.return_value = {"success": True}
            barrier_prob(symbol="EURUSD", method="mc", direction="LONG", __cli_raw=True)
            self.assertEqual(mock_mc.call_args.kwargs.get("direction"), "long")

        with patch('src.mtdata.forecast.barriers.forecast_barrier_closed_form') as mock_cf:
            mock_cf.return_value = {"success": True}
            barrier_prob(symbol="EURUSD", method="closed_form", direction="SHORT", __cli_raw=True)
            self.assertEqual(mock_cf.call_args.kwargs.get("direction"), "short")

    def test_forecast_barrier_prob_rejects_invalid_direction(self):
        with patch('src.mtdata.forecast.barriers.forecast_barrier_hit_probabilities') as mock_mc:
            res = barrier_prob(symbol="EURUSD", method="mc", direction="SIDEWAYS", __cli_raw=True)
            self.assertIn("error", res)
            self.assertIn("Invalid direction", res["error"])
            mock_mc.assert_not_called()

        with patch('src.mtdata.forecast.barriers.forecast_barrier_closed_form') as mock_cf:
            res = barrier_prob(symbol="EURUSD", method="closed_form", direction="SIDEWAYS", __cli_raw=True)
            self.assertIn("error", res)
            self.assertIn("Invalid direction", res["error"])
            mock_cf.assert_not_called()

    def test_trading_close_positions(self):
        # Mock positions
        mock_pos = MagicMock()
        mock_pos.ticket = 123
        mock_pos.symbol = "EURUSD"
        mock_pos.type = 0 # BUY
        mock_pos.volume = 1.0
        mock_pos.profit = 10.0
        
        self.mt5.positions_get.return_value = [mock_pos]
        self.mt5.symbol_info_tick.return_value = MagicMock(bid=1.0, ask=1.0)
        self.mt5.order_send.return_value = MagicMock(retcode=self.mt5.TRADE_RETCODE_DONE)
        
        from src.mtdata.core.trading import trade_close
        from src.mtdata.core.trading.requests import TradeCloseRequest

        # Test close by ticket
        trade_close(request=TradeCloseRequest(ticket=123), __cli_raw=True)
        self.mt5.positions_get.assert_called_with(ticket=123)

        # Test close by symbol
        trade_close(request=TradeCloseRequest(symbol="EURUSD", close_all=True), __cli_raw=True)
        self.mt5.positions_get.assert_called_with(symbol="EURUSD")

        # Test close all
        trade_close(request=TradeCloseRequest(close_all=True), __cli_raw=True)
        self.mt5.positions_get.assert_called_with()

    def test_trading_close_pending(self):
        # Mock orders
        mock_order = MagicMock()
        mock_order.ticket = 456

        self.mt5.positions_get.return_value = []
        self.mt5.orders_get.return_value = [mock_order]
        self.mt5.order_send.return_value = MagicMock(retcode=self.mt5.TRADE_RETCODE_DONE) 

        from src.mtdata.core.trading import trade_close
        from src.mtdata.core.trading.requests import TradeCloseRequest

        # Test cancel by ticket
        trade_close(request=TradeCloseRequest(ticket=456), __cli_raw=True)
        self.mt5.orders_get.assert_called_with(ticket=456)

        # Test cancel by symbol
        trade_close(request=TradeCloseRequest(symbol="EURUSD", close_all=True), __cli_raw=True)
        self.mt5.orders_get.assert_called_with(symbol="EURUSD")

        # Test cancel all
        trade_close(request=TradeCloseRequest(close_all=True), __cli_raw=True)
        self.mt5.orders_get.assert_called_with()

if __name__ == '__main__':
    unittest.main()
