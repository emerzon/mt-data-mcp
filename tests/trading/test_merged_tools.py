import sys
import time
import unittest
from collections import namedtuple
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.mtdata.utils.time import (
    format_epoch_utc,
)

# Mock mt5 before importing the module
sys.modules['MetaTrader5'] = MagicMock()


# Now import the tools
# We need to make sure the path is in sys.path
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.mtdata.core.forecast import forecast_barrier_prob
from src.mtdata.core.patterns import patterns_detect
from src.mtdata.core.patterns_requests import PatternsDetectRequest
from src.mtdata.core.trading import trade_get_open, trade_get_pending
from src.mtdata.core.trading.requests import TradeGetOpenRequest, TradeGetPendingRequest
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
        # These tests assert raw UTC epochs. Isolate them from broker timezone
        # configuration mutated by other full-suite tests.
        offset_patch = patch(
            "src.mtdata.utils.mt5._configured_static_offset_seconds",
            return_value=0,
        )
        timezone_patch = patch(
            "src.mtdata.utils.mt5.mt5_config.get_server_tz",
            return_value=None,
        )
        offset_patch.start()
        timezone_patch.start()
        self.addCleanup(offset_patch.stop)
        self.addCleanup(timezone_patch.stop)

    def test_trading_open_get_positions(self):
        # Setup mock
        self.mt5.positions_get.return_value = None # Simulate empty

        # Test default
        res = get_open(__cli_raw=True)
        self.assertIsInstance(res, dict)
        self.assertTrue(res.get("success"))
        self.assertEqual(res.get("kind"), "open_positions")
        self.assertEqual(res.get("count"), 0)
        self.assertEqual(res.get("items"), [])
        self.assertTrue(res.get("empty"))
        self.assertIn("message", res)
        self.assertIn("hint", res)
        self.assertNotIn("reason", res)

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
        # Open positions expose side (BUY/SELL), not raw MT5 type codes.
        self.assertEqual(row.get("side"), "BUY")
        self.assertEqual(row.get("symbol"), "EURUSD")
        self.assertEqual(row.get("ticket"), 1)
        expected_time = format_epoch_utc(1700000001)
        self.assertEqual(row.get("time"), expected_time)
        self.assertIsNone(row.get("time_msc"))
        self.assertIsNone(row.get("time_update"))
        self.assertIsNone(row.get("time_update_msc"))
        self.assertIsNone(row.get("direction"))
        self.assertIsNone(row.get("type_code"))
        self.assertIsNone(row.get("type"))

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

        res = get_open(detail="full", __cli_raw=True)
        row = res["items"][0]
        self.assertEqual(row.get("comment_max_length"), 31)
        self.assertEqual(row.get("comment_visible_length"), len("audit short"))
        self.assertFalse(row.get("comment_may_be_truncated"))

    def test_trading_open_compact_omits_comment_metadata(self):
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

        self.assertEqual(row.get("comment"), "audit short")
        self.assertNotIn("comment_max_length", row)
        self.assertNotIn("comment_visible_length", row)
        self.assertNotIn("comment_may_be_truncated", row)

    def test_trading_open_compact_detail_omits_echoed_request_metadata(self):
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

        res = get_open(symbol="EURUSD", limit=5, detail="compact", __cli_raw=True)
        self.assertIsInstance(res, dict)
        self.assertNotIn("kind", res)
        self.assertNotIn("scope", res)
        self.assertNotIn("symbol", res)
        self.assertNotIn("limit", res)

    def test_trading_open_get_pending(self):
        self.mt5.orders_get.return_value = None

        res = get_pending(__cli_raw=True)
        self.assertIsInstance(res, dict)
        self.assertTrue(res.get("success"))
        self.assertEqual(res.get("kind"), "pending_orders")
        self.assertEqual(res.get("count"), 0)
        self.assertEqual(res.get("items"), [])
        self.assertTrue(res.get("empty"))
        self.assertIn("message", res)
        self.assertIn("hint", res)
        self.assertNotIn("reason", res)

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
        # Pending orders expose order_type + side (not raw MT5 type codes).
        self.assertEqual(row.get("order_type"), "SELL_LIMIT")
        self.assertEqual(row.get("side"), "SELL")
        self.assertEqual(row.get("symbol"), "EURUSD")
        self.assertEqual(row.get("ticket"), 1)
        expected_time = format_epoch_utc(1700000000)
        expected_exp = format_epoch_utc(1700003600)
        self.assertEqual(row.get("time"), expected_time)
        self.assertIn("expiration", row)
        self.assertEqual(row.get("expiration"), expected_exp)
        self.assertNotIn("profit", row)
        self.assertNotIn("swap", row)
        self.assertIsNone(row.get("time_setup"))
        self.assertIsNone(row.get("time_setup_msc"))
        self.assertIsNone(row.get("direction"))
        self.assertIsNone(row.get("type_code"))
        self.assertIsNone(row.get("type"))

    def test_trading_pending_compact_detail_omits_echoed_request_metadata(self):
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

        res = get_pending(symbol="EURUSD", limit=5, detail="compact", __cli_raw=True)
        self.assertIsInstance(res, dict)
        self.assertNotIn("kind", res)
        self.assertNotIn("scope", res)
        self.assertNotIn("symbol", res)
        self.assertNotIn("limit", res)

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
        with patch('src.mtdata.forecast.barriers_probabilities.forecast_barrier_hit_probabilities') as mock_mc:
            mock_mc.return_value = {"success": True}
            res = barrier_prob(symbol="EURUSD", method="hmm_mc", __cli_raw=True)
            self.assertTrue(res.get("success"))
            self.assertEqual(res.get("detail"), "compact")
            self.assertEqual(res.get("symbol"), "EURUSD")
            self.assertEqual(res.get("timeframe"), "H1")
            self.assertEqual(res.get("direction"), "long")
            self.assertEqual(res.get("horizon"), 12)
            self.assertEqual(
                res.get("warnings"),
                [
                    "Default 1% symmetrical barriers applied; pass tp_pct/sl_pct, "
                    "tp_abs/sl_abs, or tp_ticks/sl_ticks to customize."
                ],
            )
            self.assertEqual(res.get("tp_pct"), 1.0)
            self.assertEqual(res.get("sl_pct"), 1.0)
            self.assertEqual(res.get("barrier_unit"), "percent")
            self.assertEqual(res.get("probability_unit"), "fraction")
            # Product uses probability_edge_definition (edge_definition is legacy).
            self.assertEqual(
                res.get("probability_edge_definition"),
                "prob_tp_first - prob_sl_first",
            )
            self.assertEqual(mock_mc.call_args.kwargs.get("tp_pct"), 1.0)
            self.assertEqual(mock_mc.call_args.kwargs.get("sl_pct"), 1.0)
            
        with patch('src.mtdata.forecast.barriers_probabilities.forecast_barrier_closed_form') as mock_cf:
            mock_cf.return_value = {"success": True}
            res = barrier_prob(symbol="EURUSD", method="closed_form", __cli_raw=True)
            self.assertTrue(res.get("success"))
            self.assertEqual(res.get("detail"), "compact")
            self.assertEqual(res.get("probability_unit"), "fraction")
            self.assertEqual(
                res.get("probability_edge_definition"),
                "prob_tp_first - prob_sl_first",
            )

    def test_forecast_barrier_prob_direction_normalization(self):
        with patch('src.mtdata.forecast.barriers_probabilities.forecast_barrier_hit_probabilities') as mock_mc:
            mock_mc.return_value = {"success": True}
            barrier_prob(symbol="EURUSD", method="hmm_mc", direction="LONG", __cli_raw=True)
            self.assertEqual(mock_mc.call_args.kwargs.get("direction"), "long")

        with patch('src.mtdata.forecast.barriers_probabilities.forecast_barrier_closed_form') as mock_cf:
            mock_cf.return_value = {"success": True}
            barrier_prob(symbol="EURUSD", method="closed_form", direction="SHORT", __cli_raw=True)
            self.assertEqual(mock_cf.call_args.kwargs.get("direction"), "short")

    def test_forecast_barrier_prob_rejects_invalid_direction(self):
        with patch('src.mtdata.forecast.barriers_probabilities.forecast_barrier_hit_probabilities') as mock_mc:
            res = barrier_prob(symbol="EURUSD", method="hmm_mc", direction="SIDEWAYS", __cli_raw=True)
            self.assertIn("error", res)
            self.assertIn("Invalid direction", res["error"])
            mock_mc.assert_not_called()

        with patch('src.mtdata.forecast.barriers_probabilities.forecast_barrier_closed_form') as mock_cf:
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
        mock_pos.price_open = 1.0
        mock_pos.time = 1700000000
        mock_pos.magic = 0
        mock_pos.sl = 0
        mock_pos.tp = 0
        mock_pos.price_current = 1.0
        mock_pos.comment = ""
        mock_pos.swap = 0.0

        now = int(time.time())
        self.mt5.positions_get.return_value = [mock_pos]
        self.mt5.symbol_info_tick.return_value = SimpleNamespace(
            bid=1.0, ask=1.0, time=now, time_msc=now * 1000
        )
        self.mt5.symbol_info.return_value = SimpleNamespace(
            trade_filling_mode=1,
            volume_min=0.01,
            volume_step=0.01,
            digits=5,
            point=0.00001,
        )
        self.mt5.order_send.return_value = MagicMock(
            retcode=self.mt5.TRADE_RETCODE_DONE,
            deal=1,
            order=1,
            volume=1.0,
            price=1.0,
            comment="done",
            profit=10.0,
        )
        self.mt5.retcode_name = lambda code: "DONE"
        self.mt5.last_error.return_value = (0, "")
        
        from src.mtdata.core.trading import trade_close
        from src.mtdata.core.trading.requests import TradeCloseRequest

        # Test close by ticket
        trade_close(request=TradeCloseRequest(ticket=123), __cli_raw=True)
        self.mt5.positions_get.assert_any_call(ticket=123)

        # Live bulk close requires confirm_close_all=true
        self.mt5.positions_get.reset_mock()
        trade_close(
            request=TradeCloseRequest(
                symbol="EURUSD",
                close_all=True,
                confirm_close_all=True,
            ),
            __cli_raw=True,
        )
        self.mt5.positions_get.assert_any_call(symbol="EURUSD")

        # Test close all
        self.mt5.positions_get.reset_mock()
        trade_close(
            request=TradeCloseRequest(close_all=True, confirm_close_all=True),
            __cli_raw=True,
        )
        self.mt5.positions_get.assert_any_call()

    def test_trading_close_pending(self):
        # Mock orders
        mock_order = MagicMock()
        mock_order.ticket = 456
        mock_order.symbol = "EURUSD"
        mock_order.magic = 0

        self.mt5.positions_get.return_value = []
        self.mt5.orders_get.return_value = [mock_order]
        self.mt5.order_send.return_value = MagicMock(
            retcode=self.mt5.TRADE_RETCODE_DONE,
            deal=0,
            order=456,
            volume=0.0,
            price=0.0,
            comment="done",
        )
        self.mt5.retcode_name = lambda code: "DONE"
        self.mt5.last_error.return_value = (0, "")

        from src.mtdata.core.trading import trade_close
        from src.mtdata.core.trading.requests import TradeCloseRequest

        # Test cancel by ticket
        trade_close(request=TradeCloseRequest(ticket=456), __cli_raw=True)
        self.mt5.orders_get.assert_any_call(ticket=456)

        # Live bulk close requires confirm_close_all=true
        self.mt5.orders_get.reset_mock()
        trade_close(
            request=TradeCloseRequest(
                symbol="EURUSD",
                close_all=True,
                confirm_close_all=True,
            ),
            __cli_raw=True,
        )
        self.mt5.orders_get.assert_any_call(symbol="EURUSD")

        # Test cancel all
        self.mt5.orders_get.reset_mock()
        trade_close(
            request=TradeCloseRequest(close_all=True, confirm_close_all=True),
            __cli_raw=True,
        )
        self.mt5.orders_get.assert_any_call()

    def test_trade_get_open_rejects_invalid_symbol(self):
        # Setup mock to simulate symbol_select failure
        # We need to reset all settings before configuring for invalid symbol
        self.mt5.reset_mock()
        self.mt5.symbol_select.return_value = False
        self.mt5.last_error.return_value = "Symbol INVALID not found"
        # Ensure other methods exist but aren't called
        self.mt5.symbol_info.return_value = None
        self.mt5.positions_get.return_value = None

        # Test with invalid symbol
        res = get_open(symbol="INVALID", __cli_raw=True)
        self.assertIsInstance(res, dict)
        self.assertFalse(res.get("success"), f"Expected success=False but got: {res}")
        self.assertIn("error", res, f"Expected error field in: {res}")
        self.assertIn("INVALID", res.get("error", ""))

    def test_trade_get_pending_rejects_invalid_symbol(self):
        # Setup mock to simulate symbol_select failure
        # We need to reset all settings before configuring for invalid symbol
        self.mt5.reset_mock()
        self.mt5.symbol_select.return_value = False
        self.mt5.last_error.return_value = "Symbol INVALID not found"
        # Ensure other methods exist but aren't called
        self.mt5.symbol_info.return_value = None
        self.mt5.orders_get.return_value = None

        # Test with invalid symbol
        res = get_pending(symbol="INVALID", __cli_raw=True)
        self.assertIsInstance(res, dict)
        self.assertFalse(res.get("success"), f"Expected success=False but got: {res}")
        self.assertIn("error", res, f"Expected error field in: {res}")
        self.assertIn("INVALID", res.get("error", ""))

if __name__ == '__main__':
    unittest.main()
