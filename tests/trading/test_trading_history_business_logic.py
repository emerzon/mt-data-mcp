from __future__ import annotations

from collections import namedtuple
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import sys

from mtdata.core.trading import trade_history as _trade_history_tool
from mtdata.core.trading_requests import TradeHistoryRequest
from mtdata.core.trading_use_cases import run_trade_history
from mtdata.utils.mt5 import MT5ConnectionError, _mt5_epoch_to_utc
from mtdata.utils.utils import _format_time_minimal


def _unwrap(fn):
    while hasattr(fn, "__wrapped__"):
        fn = fn.__wrapped__
    return fn


def trade_history(**kwargs):
    raw_output = bool(kwargs.pop("__cli_raw", False))
    request = kwargs.pop("request", None)
    if request is None:
        request = TradeHistoryRequest(**kwargs)
    with patch("mtdata.core.trading_account.ensure_mt5_connection_or_raise", return_value=None):
        if raw_output:
            return _unwrap(_trade_history_tool)(request=request)
        return _trade_history_tool(request=request, __cli_raw=False)


def _install_mock_mt5() -> tuple[MagicMock, object]:
    prev = sys.modules.get("MetaTrader5")
    mt5 = MagicMock()
    sys.modules["MetaTrader5"] = mt5
    return mt5, prev


def test_trade_history_deals_normalizes_time_to_utc_string() -> None:
    mt5, prev = _install_mock_mt5()
    Deal = namedtuple("Deal", ["ticket", "time", "symbol"])
    mt5.history_deals_get.return_value = [Deal(ticket=1, time=1700000000, symbol="EURUSD")]

    with patch("mtdata.core.trading_account._use_client_tz", lambda: False):
        out = trade_history(history_kind="deals", __cli_raw=True)
    if prev is not None:
        sys.modules["MetaTrader5"] = prev

    assert isinstance(out, list)
    assert out[0]["time"] == _format_time_minimal(_mt5_epoch_to_utc(1700000000))
    assert out[0]["timestamp_timezone"] == "UTC"


def test_trade_history_orders_normalizes_setup_and_done_times() -> None:
    mt5, prev = _install_mock_mt5()
    Order = namedtuple("Order", ["ticket", "time_setup", "time_done", "symbol"])
    mt5.history_orders_get.return_value = [
        Order(ticket=1, time_setup=1700000000, time_done=1700003600, symbol="EURUSD")
    ]

    with patch("mtdata.core.trading_account._use_client_tz", lambda: False):
        out = trade_history(history_kind="orders", __cli_raw=True)
    if prev is not None:
        sys.modules["MetaTrader5"] = prev

    assert isinstance(out, list)
    assert out[0]["time_setup"] == _format_time_minimal(_mt5_epoch_to_utc(1700000000))
    assert out[0]["time_done"] == _format_time_minimal(_mt5_epoch_to_utc(1700003600))
    assert out[0]["timestamp_timezone"] == "UTC"


def test_trade_history_filters_rows_by_symbol_even_if_mt5_returns_mixed_rows() -> None:
    mt5, prev = _install_mock_mt5()
    Deal = namedtuple("Deal", ["ticket", "time", "symbol"])
    mt5.history_deals_get.return_value = [
        Deal(ticket=1, time=1700000000, symbol="BTCUSD"),
        Deal(ticket=2, time=1700003600, symbol="XAUUSD"),
    ]

    with patch("mtdata.core.trading_account._use_client_tz", lambda: False):
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

    with patch("mtdata.core.trading_account._use_client_tz", lambda: False):
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

    with patch("mtdata.core.trading_account._use_client_tz", lambda: False):
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

    with patch("mtdata.core.trading_account._use_client_tz", lambda: False):
        out = trade_history(history_kind="deals", __cli_raw=True)
    if prev is not None:
        sys.modules["MetaTrader5"] = prev

    assert isinstance(out, list)
    assert out[0]["exit_trigger"] == "TP"
    assert out[0]["exit_trigger_price"] is None
    assert out[0]["exit_trigger_source"] == "reason"


def test_trade_history_deals_drops_non_informative_noise_columns() -> None:
    mt5, prev = _install_mock_mt5()
    mt5.DEAL_TYPE_BUY = 0
    mt5.DEAL_ENTRY_IN = 0
    mt5.DEAL_REASON_CLIENT = 0
    Deal = namedtuple(
        "Deal",
        [
            "ticket",
            "time",
            "symbol",
            "type",
            "entry",
            "reason",
            "time_msc",
            "external_id",
            "fee",
        ],
    )
    mt5.history_deals_get.return_value = [
        Deal(
            ticket=1,
            time=1700000000,
            symbol="EURUSD",
            type=0,
            entry=0,
            reason=0,
            time_msc=0,
            external_id="",
            fee=0.0,
        )
    ]

    with patch("mtdata.core.trading_account._use_client_tz", lambda: False):
        out = trade_history(history_kind="deals", __cli_raw=True)
    if prev is not None:
        sys.modules["MetaTrader5"] = prev

    assert isinstance(out, list)
    row = out[0]
    assert "time_msc" not in row
    assert "external_id" not in row
    assert "fee" not in row


def test_trade_history_deals_keeps_fee_when_non_zero() -> None:
    mt5, prev = _install_mock_mt5()
    mt5.DEAL_TYPE_BUY = 0
    mt5.DEAL_ENTRY_IN = 0
    mt5.DEAL_REASON_CLIENT = 0
    Deal = namedtuple(
        "Deal",
        [
            "ticket",
            "time",
            "symbol",
            "type",
            "entry",
            "reason",
            "time_msc",
            "external_id",
            "fee",
        ],
    )
    mt5.history_deals_get.return_value = [
        Deal(
            ticket=1,
            time=1700000000,
            symbol="EURUSD",
            type=0,
            entry=0,
            reason=0,
            time_msc=0,
            external_id="",
            fee=1.25,
        )
    ]

    with patch("mtdata.core.trading_account._use_client_tz", lambda: False):
        out = trade_history(history_kind="deals", __cli_raw=True)
    if prev is not None:
        sys.modules["MetaTrader5"] = prev

    assert isinstance(out, list)
    row = out[0]
    assert row["fee"] == 1.25


def test_trade_history_replaces_non_finite_values_with_none() -> None:
    mt5, prev = _install_mock_mt5()
    Deal = namedtuple("Deal", ["ticket", "time", "symbol", "profit"])
    mt5.history_deals_get.return_value = [
        Deal(ticket=1, time=1700000000, symbol="EURUSD", profit=float("nan"))
    ]

    with patch("mtdata.core.trading_account._use_client_tz", lambda: False):
        out = trade_history(history_kind="deals", __cli_raw=True)
    if prev is not None:
        sys.modules["MetaTrader5"] = prev

    assert isinstance(out, list)
    assert out[0]["profit"] is None


def test_run_trade_history_logs_finish_event(caplog) -> None:
    Deal = namedtuple("Deal", ["ticket", "time", "symbol"])
    gateway = SimpleNamespace(
        ensure_connection=lambda: None,
        history_deals_get=lambda from_dt, to_dt, symbol=None: [
            Deal(ticket=1, time=1700000000, symbol="EURUSD")
        ],
    )

    with caplog.at_level("INFO", logger="mtdata.core.trading_use_cases"):
        out = run_trade_history(
            TradeHistoryRequest(history_kind="deals"),
            gateway=gateway,
            use_client_tz=lambda: False,
            format_time_minimal=lambda ts: f"t{int(ts)}",
            format_time_minimal_local=lambda ts: f"lt{int(ts)}",
            mt5_epoch_to_utc=lambda ts: ts,
            parse_start_datetime=lambda value: None,
            normalize_limit=lambda value: value,
            comment_row_metadata=lambda comment: {},
            normalize_ticket_filter=lambda value, name: (None, None),
            normalize_minutes_back=lambda value: (None, None),
            decode_mt5_enum_label=lambda gateway, value, prefix=None: None,
            mt5_config=SimpleNamespace(get_client_tz=lambda: "UTC"),
        )

    assert isinstance(out, list)
    assert any(
        "event=finish operation=trade_history success=True" in record.message
        for record in caplog.records
    )


def test_trade_history_filters_deals_by_position_ticket() -> None:
    mt5, prev = _install_mock_mt5()
    Deal = namedtuple("Deal", ["ticket", "time", "symbol", "position_id"])
    mt5.history_deals_get.return_value = [
        Deal(ticket=1, time=1700000000, symbol="BTCUSD", position_id=111),
        Deal(ticket=2, time=1700003600, symbol="BTCUSD", position_id=222),
    ]

    with patch("mtdata.core.trading_account._use_client_tz", lambda: False):
        out = trade_history(history_kind="deals", symbol="BTCUSD", position_ticket=222, __cli_raw=True)
    if prev is not None:
        sys.modules["MetaTrader5"] = prev

    assert isinstance(out, list)
    assert len(out) == 1
    assert out[0]["ticket"] == 2


def test_trade_history_without_range_uses_full_history_start() -> None:
    mt5, prev = _install_mock_mt5()
    Deal = namedtuple("Deal", ["ticket", "time", "symbol"])
    mt5.history_deals_get.return_value = [
        Deal(ticket=1, time=1700000000, symbol="EURUSD")
    ]

    with patch("mtdata.core.trading_account._use_client_tz", lambda: False):
        out = trade_history(history_kind="deals", __cli_raw=True)
    if prev is not None:
        sys.modules["MetaTrader5"] = prev

    assert isinstance(out, list)
    from_dt, to_dt = mt5.history_deals_get.call_args.args[:2]
    assert abs((to_dt - from_dt).total_seconds() - (7 * 24 * 60 * 60)) < 1.0
    assert to_dt >= from_dt


def test_trade_history_surfaces_mt5_history_exception_with_actionable_hint() -> None:
    mt5, prev = _install_mock_mt5()
    mt5.history_deals_get.side_effect = RuntimeError(
        "<built-in function history_deals_get> returned a result with an exception set"
    )

    with patch("mtdata.core.trading_account._use_client_tz", lambda: False):
        out = trade_history(history_kind="deals", __cli_raw=True)
    if prev is not None:
        sys.modules["MetaTrader5"] = prev

    assert out["error"] == (
        "Failed to fetch deal history from MT5. "
        "Try narrowing the range with --minutes-back, --days, --start, or --end."
    )


def test_trade_history_rejects_start_with_minutes_back() -> None:
    out = trade_history(
        history_kind="deals",
        start="2026-03-01",
        minutes_back=30,
        __cli_raw=True,
    )

    assert out["error"] == "Use either start or minutes_back, not both."


def test_trade_history_surfaces_comment_limit_metadata() -> None:
    mt5, prev = _install_mock_mt5()
    Deal = namedtuple("Deal", ["ticket", "time", "symbol", "comment"])
    mt5.history_deals_get.return_value = [
        Deal(ticket=1, time=1700000000, symbol="BTCUSD", comment="audit short"),
    ]

    with patch("mtdata.core.trading_account._use_client_tz", lambda: False):
        out = trade_history(history_kind="deals", symbol="BTCUSD", __cli_raw=True)
    if prev is not None:
        sys.modules["MetaTrader5"] = prev

    assert isinstance(out, list)
    assert out[0]["comment_max_length"] == 31
    assert out[0]["comment_visible_length"] == len("audit short")
    assert out[0]["comment_may_be_truncated"] is False


def test_trade_history_empty_deals_message_includes_orders_hint() -> None:
    mt5, prev = _install_mock_mt5()
    mt5.history_deals_get.return_value = []

    with patch("mtdata.core.trading_account._use_client_tz", lambda: False):
        out = trade_history(history_kind="deals", __cli_raw=True)
    if prev is not None:
        sys.modules["MetaTrader5"] = prev

    assert out["message"].startswith("No deals found")
    assert "--history-kind orders" in out["message"]


def test_trade_history_small_window_empty_orders_message_includes_propagation_note() -> None:
    mt5, prev = _install_mock_mt5()
    mt5.history_orders_get.return_value = []

    with patch("mtdata.core.trading_account._use_client_tz", lambda: False):
        out = trade_history(history_kind="orders", minutes_back=5, __cli_raw=True)
    if prev is not None:
        sys.modules["MetaTrader5"] = prev

    assert out["message"].startswith("No orders found")
    assert "may take up to a few minutes to reflect very recent events" in out["message"]


def test_trade_history_returns_connection_error_payload() -> None:
    with patch(
        "mtdata.core.trading_account.ensure_mt5_connection_or_raise",
        side_effect=MT5ConnectionError("Failed to connect to MetaTrader5. Ensure MT5 terminal is running."),
    ):
        out = _unwrap(_trade_history_tool)(request=TradeHistoryRequest(history_kind="deals"))

    assert out["error"] == "Failed to connect to MetaTrader5. Ensure MT5 terminal is running."
