from __future__ import annotations

from collections import namedtuple
from unittest.mock import MagicMock, patch
import sys

sys.modules.setdefault("MetaTrader5", MagicMock())

from mtdata.core.trading import trade_history
from mtdata.utils.mt5 import _mt5_epoch_to_utc
from mtdata.utils.utils import _format_time_minimal


def _install_mock_mt5() -> tuple[MagicMock, object]:
    prev = sys.modules.get("MetaTrader5")
    mt5 = MagicMock()
    sys.modules["MetaTrader5"] = mt5
    return mt5, prev


def test_trade_history_deals_normalizes_time_to_utc_string() -> None:
    mt5, prev = _install_mock_mt5()
    Deal = namedtuple("Deal", ["ticket", "time", "symbol"])
    mt5.history_deals_get.return_value = [Deal(ticket=1, time=1700000000, symbol="EURUSD")]

    with patch("mtdata.core.trading._auto_connect_wrapper", lambda f: f), patch(
        "mtdata.core.trading._use_client_tz", lambda: False
    ):
        out = trade_history(history_kind="deals", __cli_raw=True)
    if prev is not None:
        sys.modules["MetaTrader5"] = prev

    assert isinstance(out, list)
    assert out[0]["time"] == _format_time_minimal(_mt5_epoch_to_utc(1700000000))


def test_trade_history_orders_normalizes_setup_and_done_times() -> None:
    mt5, prev = _install_mock_mt5()
    Order = namedtuple("Order", ["ticket", "time_setup", "time_done", "symbol"])
    mt5.history_orders_get.return_value = [
        Order(ticket=1, time_setup=1700000000, time_done=1700003600, symbol="EURUSD")
    ]

    with patch("mtdata.core.trading._auto_connect_wrapper", lambda f: f), patch(
        "mtdata.core.trading._use_client_tz", lambda: False
    ):
        out = trade_history(history_kind="orders", __cli_raw=True)
    if prev is not None:
        sys.modules["MetaTrader5"] = prev

    assert isinstance(out, list)
    assert out[0]["time_setup"] == _format_time_minimal(_mt5_epoch_to_utc(1700000000))
    assert out[0]["time_done"] == _format_time_minimal(_mt5_epoch_to_utc(1700003600))
