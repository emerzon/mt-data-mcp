from __future__ import annotations

import sys
from collections import namedtuple
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from mtdata.core.trading import trade_history as _trade_history_tool
from mtdata.core.trading.account import trade_journal_analyze as _trade_journal_tool
from mtdata.core.trading.positions import normalize_trade_history_output
from mtdata.core.trading.requests import TradeHistoryRequest, TradeJournalAnalyzeRequest
from mtdata.core.trading.use_cases import run_trade_history
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
    with patch(
        "mtdata.core.trading.account.ensure_mt5_connection_or_raise", return_value=None
    ):
        if raw_output:
            return _unwrap(_trade_history_tool)(request=request)
        return _trade_history_tool(request=request, __cli_raw=False)


def trade_journal_analyze(**kwargs):
    raw_output = bool(kwargs.pop("__cli_raw", False))
    request = kwargs.pop("request", None)
    if request is None:
        request = TradeJournalAnalyzeRequest(**kwargs)
    with patch(
        "mtdata.core.trading.account.ensure_mt5_connection_or_raise", return_value=None
    ):
        if raw_output:
            return _unwrap(_trade_journal_tool)(request=request)
        return _trade_journal_tool(request=request, __cli_raw=False)


def _install_mock_mt5() -> tuple[MagicMock, object]:
    prev = sys.modules.get("MetaTrader5")
    mt5 = MagicMock()
    sys.modules["MetaTrader5"] = mt5
    return mt5, prev


def test_trade_history_deals_normalizes_time_to_utc_string() -> None:
    mt5, prev = _install_mock_mt5()
    Deal = namedtuple("Deal", ["ticket", "time", "symbol"])
    mt5.history_deals_get.return_value = [
        Deal(ticket=1, time=1700000000, symbol="EURUSD")
    ]

    with patch("mtdata.core.trading.account._use_client_tz", lambda: False):
        out = trade_history(history_kind="deals", __cli_raw=True)
    if prev is not None:
        sys.modules["MetaTrader5"] = prev

    assert out["success"] is True
    assert out["kind"] == "trade_history"
    assert out["history_kind"] == "deals"
    assert out["count"] == 1
    assert out["items"][0]["time"] == _format_time_minimal(
        _mt5_epoch_to_utc(1700000000)
    )
    assert out["items"][0]["timestamp_timezone"] == "UTC"


def test_trade_history_orders_normalizes_setup_and_done_times() -> None:
    mt5, prev = _install_mock_mt5()
    Order = namedtuple("Order", ["ticket", "time_setup", "time_done", "symbol"])
    mt5.history_orders_get.return_value = [
        Order(ticket=1, time_setup=1700000000, time_done=1700003600, symbol="EURUSD")
    ]

    with patch("mtdata.core.trading.account._use_client_tz", lambda: False):
        out = trade_history(history_kind="orders", __cli_raw=True)
    if prev is not None:
        sys.modules["MetaTrader5"] = prev

    assert out["success"] is True
    assert out["history_kind"] == "orders"
    assert out["count"] == 1
    assert out["items"][0]["time_setup"] == _format_time_minimal(
        _mt5_epoch_to_utc(1700000000)
    )
    assert out["items"][0]["time_done"] == _format_time_minimal(
        _mt5_epoch_to_utc(1700003600)
    )
    assert out["items"][0]["timestamp_timezone"] == "UTC"


def test_trade_history_filters_rows_by_symbol_even_if_mt5_returns_mixed_rows() -> None:
    mt5, prev = _install_mock_mt5()
    Deal = namedtuple("Deal", ["ticket", "time", "symbol"])
    mt5.history_deals_get.return_value = [
        Deal(ticket=1, time=1700000000, symbol="BTCUSD"),
        Deal(ticket=2, time=1700003600, symbol="XAUUSD"),
    ]

    with patch("mtdata.core.trading.account._use_client_tz", lambda: False):
        out = trade_history(history_kind="deals", symbol="BTCUSD", __cli_raw=True)
    if prev is not None:
        sys.modules["MetaTrader5"] = prev

    assert out["success"] is True
    assert out["scope"] == "symbol"
    assert out["count"] == 1
    assert out["items"][0]["symbol"] == "BTCUSD"


def test_trade_history_filters_deals_by_side_alias() -> None:
    mt5, prev = _install_mock_mt5()
    mt5.DEAL_TYPE_BUY = 0
    mt5.DEAL_TYPE_SELL = 1
    Deal = namedtuple("Deal", ["ticket", "time", "symbol", "type"])
    mt5.history_deals_get.return_value = [
        Deal(ticket=1, time=1700000000, symbol="EURUSD", type=0),
        Deal(ticket=2, time=1700003600, symbol="EURUSD", type=1),
    ]

    with patch("mtdata.core.trading.account._use_client_tz", lambda: False):
        out = trade_history(history_kind="deals", side="long", detail="full", __cli_raw=True)
    if prev is not None:
        sys.modules["MetaTrader5"] = prev

    assert out["success"] is True
    assert out["side"] == "BUY"
    assert out["count"] == 1
    assert out["items"][0]["ticket"] == 1
    assert out["items"][0]["type"] == "Buy"


def test_trade_history_request_normalizes_buy_sell_aliases() -> None:
    assert TradeHistoryRequest(side="buy").side == "BUY"
    assert TradeHistoryRequest(side="short").side == "SELL"
    assert TradeHistoryRequest(side="weird").side == "weird"
    assert TradeHistoryRequest().detail == "compact"


def test_trade_history_filters_orders_by_side_prefix() -> None:
    mt5, prev = _install_mock_mt5()
    mt5.ORDER_TYPE_BUY_LIMIT = 2
    mt5.ORDER_TYPE_SELL_STOP = 5
    Order = namedtuple("Order", ["ticket", "time_setup", "symbol", "type"])
    mt5.history_orders_get.return_value = [
        Order(ticket=11, time_setup=1700000000, symbol="EURUSD", type=2),
        Order(ticket=12, time_setup=1700003600, symbol="EURUSD", type=5),
    ]

    with patch("mtdata.core.trading.account._use_client_tz", lambda: False):
        out = trade_history(history_kind="orders", side="sell", detail="full", __cli_raw=True)
    if prev is not None:
        sys.modules["MetaTrader5"] = prev

    assert out["success"] is True
    assert out["side"] == "SELL"
    assert out["count"] == 1
    assert out["items"][0]["ticket"] == 12
    assert out["items"][0]["type"] == "Sell Stop"


def test_trade_history_deals_decodes_enum_codes_to_labels() -> None:
    mt5, prev = _install_mock_mt5()
    mt5.DEAL_TYPE_BUY = 0
    mt5.DEAL_ENTRY_IN = 0
    mt5.DEAL_REASON_CLIENT = 0
    Deal = namedtuple("Deal", ["ticket", "time", "symbol", "type", "entry", "reason"])
    mt5.history_deals_get.return_value = [
        Deal(ticket=1, time=1700000000, symbol="EURUSD", type=0, entry=0, reason=0)
    ]

    with patch("mtdata.core.trading.account._use_client_tz", lambda: False):
        out = trade_history(history_kind="deals", __cli_raw=True)
    if prev is not None:
        sys.modules["MetaTrader5"] = prev

    row = out["items"][0]
    assert row["type"] == "Buy"
    assert row["entry"] == "In"
    assert row["reason"] == "Client"
    assert row["type_code"] == 0
    assert row["entry_code"] == 0
    assert row["reason_code"] == 0


def test_trade_history_deals_extracts_exit_trigger_from_comment() -> None:
    mt5, prev = _install_mock_mt5()
    mt5.DEAL_ENTRY_OUT = 1
    mt5.DEAL_REASON_SL = 4
    Deal = namedtuple(
        "Deal", ["ticket", "time", "symbol", "entry", "reason", "comment"]
    )
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

    with patch("mtdata.core.trading.account._use_client_tz", lambda: False):
        out = trade_history(history_kind="deals", __cli_raw=True)
    if prev is not None:
        sys.modules["MetaTrader5"] = prev

    row = out["items"][0]
    assert row["exit_trigger"] == "SL"
    assert row["exit_trigger_price"] == 64654.92
    assert row["exit_trigger_source"] == "comment"


def test_trade_history_deals_extracts_exit_trigger_from_reason_when_comment_missing() -> (
    None
):
    mt5, prev = _install_mock_mt5()
    mt5.DEAL_ENTRY_OUT = 1
    mt5.DEAL_REASON_TP = 5
    Deal = namedtuple(
        "Deal", ["ticket", "time", "symbol", "entry", "reason", "comment"]
    )
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

    with patch("mtdata.core.trading.account._use_client_tz", lambda: False):
        out = trade_history(history_kind="deals", __cli_raw=True)
    if prev is not None:
        sys.modules["MetaTrader5"] = prev

    row = out["items"][0]
    assert row["exit_trigger"] == "TP"
    assert row["exit_trigger_price"] is None
    assert row["exit_trigger_source"] == "reason"


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

    with patch("mtdata.core.trading.account._use_client_tz", lambda: False):
        out = trade_history(history_kind="deals", __cli_raw=True)
    if prev is not None:
        sys.modules["MetaTrader5"] = prev

    row = out["items"][0]
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

    with patch("mtdata.core.trading.account._use_client_tz", lambda: False):
        out = trade_history(history_kind="deals", __cli_raw=True)
    if prev is not None:
        sys.modules["MetaTrader5"] = prev

    row = out["items"][0]
    assert row["fee"] == 1.25


def test_trade_history_replaces_non_finite_values_with_none() -> None:
    mt5, prev = _install_mock_mt5()
    Deal = namedtuple("Deal", ["ticket", "time", "symbol", "profit"])
    mt5.history_deals_get.return_value = [
        Deal(ticket=1, time=1700000000, symbol="EURUSD", profit=float("nan"))
    ]

    with patch("mtdata.core.trading.account._use_client_tz", lambda: False):
        out = trade_history(history_kind="deals", __cli_raw=True)
    if prev is not None:
        sys.modules["MetaTrader5"] = prev

    assert out["items"][0]["profit"] is None


def test_run_trade_history_logs_finish_event(caplog) -> None:
    Deal = namedtuple("Deal", ["ticket", "time", "symbol"])
    gateway = SimpleNamespace(
        ensure_connection=lambda: None,
        history_deals_get=lambda from_dt, to_dt, symbol=None: [
            Deal(ticket=1, time=1700000000, symbol="EURUSD")
        ],
    )

    with caplog.at_level("INFO", logger="mtdata.core.trading.use_cases"):
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

    with patch("mtdata.core.trading.account._use_client_tz", lambda: False):
        out = trade_history(
            history_kind="deals", symbol="BTCUSD", position_ticket=222, __cli_raw=True
        )
    if prev is not None:
        sys.modules["MetaTrader5"] = prev

    assert out["scope"] == "ticket"
    assert out["count"] == 1
    assert out["items"][0]["ticket"] == 2


def test_trade_history_without_range_uses_full_history_start() -> None:
    mt5, prev = _install_mock_mt5()
    Deal = namedtuple("Deal", ["ticket", "time", "symbol"])
    mt5.history_deals_get.return_value = [
        Deal(ticket=1, time=1700000000, symbol="EURUSD")
    ]

    with patch("mtdata.core.trading.account._use_client_tz", lambda: False):
        out = trade_history(history_kind="deals", __cli_raw=True)
    if prev is not None:
        sys.modules["MetaTrader5"] = prev

    assert out["success"] is True
    from_dt, to_dt = mt5.history_deals_get.call_args.args[:2]
    assert abs((to_dt - from_dt).total_seconds() - (7 * 24 * 60 * 60)) < 1.0
    assert to_dt >= from_dt


def test_trade_history_surfaces_mt5_history_exception_with_actionable_hint() -> None:
    mt5, prev = _install_mock_mt5()
    mt5.history_deals_get.side_effect = RuntimeError(
        "<built-in function history_deals_get> returned a result with an exception set"
    )

    with patch("mtdata.core.trading.account._use_client_tz", lambda: False):
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


def test_trade_history_rejects_invalid_side_filter() -> None:
    out = trade_history(history_kind="deals", side="flat", detail="full", __cli_raw=True)

    assert out["success"] is False
    assert out["error"] == "side must be BUY/SELL or LONG/SHORT."
    assert out["side"] == "flat"


def test_trade_history_surfaces_comment_limit_metadata() -> None:
    mt5, prev = _install_mock_mt5()
    Deal = namedtuple("Deal", ["ticket", "time", "symbol", "comment"])
    mt5.history_deals_get.return_value = [
        Deal(ticket=1, time=1700000000, symbol="BTCUSD", comment="audit short"),
    ]

    with patch("mtdata.core.trading.account._use_client_tz", lambda: False):
        out = trade_history(history_kind="deals", symbol="BTCUSD", __cli_raw=True)
    if prev is not None:
        sys.modules["MetaTrader5"] = prev

    row = out["items"][0]
    assert row["comment_max_length"] == 31
    assert row["comment_visible_length"] == len("audit short")
    assert row["comment_may_be_truncated"] is False


def test_trade_history_empty_deals_message_includes_orders_hint() -> None:
    mt5, prev = _install_mock_mt5()
    mt5.history_deals_get.return_value = []

    with patch("mtdata.core.trading.account._use_client_tz", lambda: False):
        out = trade_history(history_kind="deals", __cli_raw=True)
    if prev is not None:
        sys.modules["MetaTrader5"] = prev

    assert out["message"].startswith("No deals found")
    assert "--history-kind orders" in out["message"]


def test_trade_history_small_window_empty_orders_message_includes_propagation_note() -> (
    None
):
    mt5, prev = _install_mock_mt5()
    mt5.history_orders_get.return_value = []

    with patch("mtdata.core.trading.account._use_client_tz", lambda: False):
        out = trade_history(history_kind="orders", minutes_back=5, __cli_raw=True)
    if prev is not None:
        sys.modules["MetaTrader5"] = prev

    assert out["message"].startswith("No orders found")
    assert (
        "may take up to a few minutes to reflect very recent events" in out["message"]
    )


def test_trade_history_minutes_back_empty_deals_message_mentions_window_not_orders() -> None:
    mt5, prev = _install_mock_mt5()
    mt5.history_deals_get.return_value = []

    with patch("mtdata.core.trading.account._use_client_tz", lambda: False):
        out = trade_history(history_kind="deals", minutes_back=60, __cli_raw=True)
    if prev is not None:
        sys.modules["MetaTrader5"] = prev

    assert "in the last 60 minute(s)" in out["message"]
    assert "--history-kind orders" not in out["message"]


def test_trade_history_converts_minutes_back_window_to_server_time_before_fetch() -> None:
    captured: dict[str, object] = {}
    parsed_end = datetime(2026, 3, 1, 11, 0, 0)

    def history_deals_get(from_dt, to_dt, symbol=None):
        captured["from_dt"] = from_dt
        captured["to_dt"] = to_dt
        captured["symbol"] = symbol
        return []

    gateway = SimpleNamespace(
        ensure_connection=lambda: None,
        history_deals_get=history_deals_get,
    )

    with patch(
        "mtdata.core.trading.use_cases._to_server_naive_dt",
        side_effect=lambda dt: dt + timedelta(hours=2),
    ):
        out = run_trade_history(
            TradeHistoryRequest(
                history_kind="deals",
                symbol="BTCUSD",
                end="2026-03-01 11:00",
                minutes_back=60,
            ),
            gateway=gateway,
            use_client_tz=lambda: False,
            format_time_minimal=lambda ts: f"t{int(ts)}",
            format_time_minimal_local=lambda ts: f"lt{int(ts)}",
            mt5_epoch_to_utc=lambda ts: ts,
            parse_start_datetime=lambda value: parsed_end if value == "2026-03-01 11:00" else None,
            normalize_limit=lambda value: value,
            comment_row_metadata=lambda comment: {},
            normalize_ticket_filter=lambda value, name: (None, None),
            normalize_minutes_back=lambda value: (value, None),
            decode_mt5_enum_label=lambda gateway, value, prefix=None: None,
            mt5_config=SimpleNamespace(get_client_tz=lambda: "UTC"),
        )

    assert captured["from_dt"] == datetime(2026, 3, 1, 12, 0, 0)
    assert captured["to_dt"] == datetime(2026, 3, 1, 13, 0, 0)
    assert captured["symbol"] == "BTCUSD"
    assert out["message"] == "No deals found for BTCUSD in the last 60 minute(s)"


def test_trade_history_returns_connection_error_payload() -> None:
    with patch(
        "mtdata.core.trading.account.ensure_mt5_connection_or_raise",
        side_effect=MT5ConnectionError(
            "Failed to connect to MetaTrader5. Ensure MT5 terminal is running."
        ),
    ):
        out = _unwrap(_trade_history_tool)(
            request=TradeHistoryRequest(history_kind="deals")
        )

    assert (
        out["error"]
        == "Failed to connect to MetaTrader5. Ensure MT5 terminal is running."
    )
    assert out["success"] is False
    assert out["kind"] == "trade_history"
    assert out["count"] == 0
    assert out["items"] == []


def test_normalize_trade_history_output_preserves_upstream_error_metadata() -> None:
    request = TradeHistoryRequest(history_kind="deals", symbol="EURUSD")

    out = normalize_trade_history_output(
        {
            "error": "history lookup failed",
            "error_code": "trade_history_lookup_failed",
            "request_id": "broker-123",
            "details": {"range": "7d"},
            "checked_scopes": ["history"],
        },
        request=request,
    )

    assert out["success"] is False
    assert out["error"] == "history lookup failed"
    assert out["error_code"] == "trade_history_lookup_failed"
    assert out["request_id"] == "broker-123"
    assert out["details"] == {"range": "7d"}
    assert out["checked_scopes"] == ["history"]
    assert out["kind"] == "trade_history"
    assert out["scope"] == "symbol"


def test_trade_history_compact_detail_omits_echoed_filters() -> None:
    out = normalize_trade_history_output(
        [{"ticket": 1, "symbol": "EURUSD"}],
        request=TradeHistoryRequest(
            history_kind="deals",
            detail="compact",
            symbol="EURUSD",
            side="buy",
            limit=5,
        ),
    )

    assert out["history_kind"] == "deals"
    assert out["scope"] == "symbol"
    assert out["count"] == 1
    assert "symbol" not in out
    assert "side" not in out
    assert "limit" not in out


def test_trade_history_empty_message_uses_enveloped_contract() -> None:
    mt5, prev = _install_mock_mt5()
    mt5.history_deals_get.return_value = []

    with patch("mtdata.core.trading.account._use_client_tz", lambda: False):
        out = trade_history(history_kind="deals", __cli_raw=True)
    if prev is not None:
        sys.modules["MetaTrader5"] = prev

    assert out["success"] is True
    assert out["kind"] == "trade_history"
    assert out["history_kind"] == "deals"
    assert out["count"] == 0
    assert out["items"] == []
    assert out["no_action"] is True
    assert out["message"].startswith("No deals found")


def test_trade_journal_analyze_summarizes_realized_exit_deals() -> None:
    history_rows = [
        {
            "ticket": 1,
            "symbol": "EURUSD",
            "entry": "In",
            "type": "Buy",
            "profit": 0.0,
            "commission": -1.0,
            "swap": 0.0,
            "time": "2026-01-01 10:00",
        },
        {
            "ticket": 2,
            "symbol": "EURUSD",
            "entry": "Out",
            "type": "Buy",
            "profit": 25.0,
            "commission": -1.0,
            "swap": -0.5,
            "exit_trigger": "TP",
            "time": "2026-01-01 12:00",
            "volume": 0.1,
        },
        {
            "ticket": 3,
            "symbol": "GBPUSD",
            "entry": "Out",
            "type": "Sell",
            "profit": -10.0,
            "commission": -0.5,
            "swap": 0.0,
            "exit_trigger": "SL",
            "time": "2026-01-02 09:00",
            "volume": 0.2,
        },
    ]

    with patch(
        "mtdata.core.trading.account._run_trade_history_request",
        return_value={
            "success": True,
            "count": len(history_rows),
            "items": history_rows,
        },
    ):
        out = trade_journal_analyze(__cli_raw=True)

    assert out["success"] is True
    assert out["summary"]["closed_deals"] == 2
    assert out["summary"]["wins"] == 1
    assert out["summary"]["losses"] == 1
    assert out["summary"]["net_pnl"] == 13.0
    assert out["summary"]["profit_factor"] == 2.238095238095238
    assert out["breakdowns"]["by_symbol"][0]["symbol"] == "EURUSD"
    assert out["best_trades"][0]["ticket"] == 2
    assert out["worst_trades"][0]["ticket"] == 3


def test_trade_journal_analyze_filters_best_worst_by_pnl_sign() -> None:
    """Verify that best_trades only contains wins and worst_trades only contains losses.
    
    This test validates the fix for the logic error where best_trades and worst_trades
    were mixed together, just sorted differently.
    """
    history_rows = [
        {
            "ticket": 1,
            "symbol": "EURUSD",
            "entry": "Out",
            "type": "Buy",
            "profit": 0.82,
            "commission": 0.0,
            "swap": 0.0,
            "exit_trigger": "TP",
            "time": "2026-01-01 10:00",
        },
        {
            "ticket": 2,
            "symbol": "USDJPY",
            "entry": "Out",
            "type": "Buy",
            "profit": 0.04,
            "commission": 0.0,
            "swap": 0.0,
            "exit_trigger": "TP",
            "time": "2026-01-01 11:00",
        },
        {
            "ticket": 3,
            "symbol": "EURUSD",
            "entry": "Out",
            "type": "Sell",
            "profit": -0.23,
            "commission": 0.0,
            "swap": 0.0,
            "exit_trigger": "SL",
            "time": "2026-01-01 12:00",
        },
    ]

    with patch(
        "mtdata.core.trading.account._run_trade_history_request",
        return_value={
            "success": True,
            "count": len(history_rows),
            "items": history_rows,
        },
    ):
        out = trade_journal_analyze(__cli_raw=True)

    # Verify metrics are correct
    assert out["summary"]["wins"] == 2
    assert out["summary"]["losses"] == 1
    assert out["summary"]["win_rate"] == 2 / 3
    assert out["summary"]["best_trade"] == 0.82
    assert out["summary"]["worst_trade"] == -0.23

    # Verify best_trades only contains winners (positive P&L)
    assert len(out["best_trades"]) == 2
    for trade in out["best_trades"]:
        assert trade["net_pnl"] > 0, f"best_trades should only contain wins, but found ticket {trade['ticket']} with net_pnl {trade['net_pnl']}"

    # Verify worst_trades only contains losers (negative P&L)
    assert len(out["worst_trades"]) == 1
    for trade in out["worst_trades"]:
        assert trade["net_pnl"] < 0, f"worst_trades should only contain losses, but found ticket {trade['ticket']} with net_pnl {trade['net_pnl']}"

    # Verify specific tickets in correct lists
    best_tickets = {trade["ticket"] for trade in out["best_trades"]}
    worst_tickets = {trade["ticket"] for trade in out["worst_trades"]}
    
    assert 1 in best_tickets  # EURUSD +0.82
    assert 2 in best_tickets  # USDJPY +0.04
    assert 3 in worst_tickets  # EURUSD -0.23


def test_trade_journal_analyze_returns_message_when_no_exit_deals_found() -> None:
    history_rows = [
        {
            "ticket": 1,
            "symbol": "EURUSD",
            "entry": "In",
            "type": "Buy",
            "profit": 0.0,
            "commission": -1.0,
            "time": "2026-01-01 10:00",
        },
    ]

    with patch(
        "mtdata.core.trading.account._run_trade_history_request",
        return_value={
            "success": True,
            "count": len(history_rows),
            "items": history_rows,
        },
    ):
        out = trade_journal_analyze(__cli_raw=True)

    assert out["success"] is True
    assert out["summary"]["closed_deals"] == 0
    assert "No realized exit deals found" in out["message"]


def test_trade_journal_analyze_propagates_history_errors() -> None:
    with patch(
        "mtdata.core.trading.account._run_trade_history_request",
        return_value={"error": "boom"},
    ):
        out = trade_journal_analyze(__cli_raw=True)

    assert out == {"error": "boom"}


def test_trade_journal_analyze_forwards_side_filter() -> None:
    captured = {}

    def _fake_history(request):
        captured["request"] = request
        return {"success": True, "count": 0, "items": [], "message": "No deals found"}

    with patch(
        "mtdata.core.trading.account._run_trade_history_request",
        side_effect=_fake_history,
    ):
        out = trade_journal_analyze(side="short", __cli_raw=True)

    assert captured["request"].side == "SELL"
    assert out["success"] is True
    assert out["summary"]["closed_deals"] == 0


def test_trade_journal_request_preserves_invalid_side_for_tool_level_error() -> None:
    assert TradeJournalAnalyzeRequest(side="sideways").side == "sideways"
