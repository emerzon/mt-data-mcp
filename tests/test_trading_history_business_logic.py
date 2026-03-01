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


def test_trade_history_filters_rows_by_symbol_even_if_mt5_returns_mixed_rows() -> None:
    mt5, prev = _install_mock_mt5()
    Deal = namedtuple("Deal", ["ticket", "time", "symbol"])
    mt5.history_deals_get.return_value = [
        Deal(ticket=1, time=1700000000, symbol="BTCUSD"),
        Deal(ticket=2, time=1700003600, symbol="XAUUSD"),
    ]

    with patch("mtdata.core.trading._auto_connect_wrapper", lambda f: f), patch(
        "mtdata.core.trading._use_client_tz", lambda: False
    ):
        out = trade_history(history_kind="deals", symbol="BTCUSD", __cli_raw=True)
    if prev is not None:
        sys.modules["MetaTrader5"] = prev

    assert isinstance(out, list)
    assert len(out) == 1
    assert out[0]["symbol"] == "BTCUSD"


def test_trade_history_deals_decodes_enum_codes_to_labels() -> None:
    mt5, prev = _install_mock_mt5()
    mt5.DEAL_TYPE_BUY = 0
    mt5.DEAL_ENTRY_IN = 0
    mt5.DEAL_REASON_CLIENT = 0
    Deal = namedtuple("Deal", ["ticket", "time", "symbol", "type", "entry", "reason"])
    mt5.history_deals_get.return_value = [
        Deal(ticket=1, time=1700000000, symbol="EURUSD", type=0, entry=0, reason=0)
    ]

    with patch("mtdata.core.trading._auto_connect_wrapper", lambda f: f), patch(
        "mtdata.core.trading._use_client_tz", lambda: False
    ):
        out = trade_history(history_kind="deals", __cli_raw=True)
    if prev is not None:
        sys.modules["MetaTrader5"] = prev

    assert isinstance(out, list)
    assert out[0]["type"] == "Buy"
    assert out[0]["entry"] == "In"
    assert out[0]["reason"] == "Client"
    assert out[0]["type_code"] == 0
    assert out[0]["entry_code"] == 0
    assert out[0]["reason_code"] == 0


def test_trade_history_deals_extracts_exit_trigger_from_comment() -> None:
    mt5, prev = _install_mock_mt5()
    mt5.DEAL_ENTRY_OUT = 1
    mt5.DEAL_REASON_SL = 4
    Deal = namedtuple("Deal", ["ticket", "time", "symbol", "entry", "reason", "comment"])
    mt5.history_deals_get.return_value = [
        Deal(
            ticket=1,
            time=1700000000,
            symbol="EURUSD",
            entry=1,
            reason=4,
            comment="[sl 64654.92]",
        )
    ]

    with patch("mtdata.core.trading._auto_connect_wrapper", lambda f: f), patch(
        "mtdata.core.trading._use_client_tz", lambda: False
    ):
        out = trade_history(history_kind="deals", __cli_raw=True)
    if prev is not None:
        sys.modules["MetaTrader5"] = prev

    assert isinstance(out, list)
    assert out[0]["exit_trigger"] == "SL"
    assert out[0]["exit_trigger_price"] == 64654.92
    assert out[0]["exit_trigger_source"] == "comment"


def test_trade_history_deals_extracts_exit_trigger_from_reason_when_comment_missing() -> None:
    mt5, prev = _install_mock_mt5()
    mt5.DEAL_ENTRY_OUT = 1
    mt5.DEAL_REASON_TP = 5
    Deal = namedtuple("Deal", ["ticket", "time", "symbol", "entry", "reason", "comment"])
    mt5.history_deals_get.return_value = [
        Deal(
            ticket=1,
            time=1700000000,
            symbol="EURUSD",
            entry=1,
            reason=5,
            comment="manual close",
        )
    ]

    with patch("mtdata.core.trading._auto_connect_wrapper", lambda f: f), patch(
        "mtdata.core.trading._use_client_tz", lambda: False
    ):
        out = trade_history(history_kind="deals", __cli_raw=True)
    if prev is not None:
        sys.modules["MetaTrader5"] = prev

    assert isinstance(out, list)
    assert out[0]["exit_trigger"] == "TP"
    assert out[0]["exit_trigger_price"] is None
    assert out[0]["exit_trigger_source"] == "reason"
