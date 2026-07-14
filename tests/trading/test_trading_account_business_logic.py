from __future__ import annotations

import logging
import sys
from collections import namedtuple
from datetime import datetime, timezone
from inspect import signature
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from mtdata.core.trading import account as core_trading_account
from mtdata.core.trading import positions as core_trading_positions
from mtdata.core.trading import trade_account_info
from mtdata.core.trading.requests import TradeGetOpenRequest, TradeGetPendingRequest
from mtdata.core.trading.use_cases import run_trade_get_open, run_trade_get_pending
from mtdata.utils.mt5 import MT5ConnectionError


def _unwrap(fn):
    while hasattr(fn, "__wrapped__"):
        fn = fn.__wrapped__
    return fn


def test_trade_account_info_includes_execution_preflight_fields() -> None:
    mt5 = MagicMock()
    prev = sys.modules.get("MetaTrader5")
    sys.modules["MetaTrader5"] = mt5
    mt5.ACCOUNT_TRADE_MODE_DEMO = 0
    mt5.ACCOUNT_TRADE_MODE_CONTEST = 1
    mt5.ACCOUNT_TRADE_MODE_REAL = 2
    mt5.account_info.return_value = SimpleNamespace(
        balance=10000.0,
        equity=10050.0,
        profit=50.0,
        margin=100.0,
        margin_free=9950.0,
        margin_level=10050.0,
        currency="USD",
        leverage=100,
        trade_allowed=True,
        trade_expert=True,
        server="Demo-Server",
        company="Broker LLC",
        trade_mode=0,
        login=123456,
    )
    mt5.terminal_info.return_value = SimpleNamespace(
        trade_allowed=False,
        tradeapi_disabled=False,
        connected=True,
        community_account=True,
    )

    raw = _unwrap(trade_account_info)
    with patch("mtdata.core.trading.account.ensure_mt5_connection_or_raise", return_value=None):
        out = raw(detail="full")

    if prev is not None:
        sys.modules["MetaTrader5"] = prev

    assert out["success"] is True
    assert out["login"] == 123456
    assert out["profit"] == 50.0
    assert out["floating_pnl"] == 50.0
    assert out["pnl_basis"] == "floating_open_positions"
    assert out["equity_balance_delta"] == 50.0
    assert out["server"] == "Demo-Server"
    assert out["company"] == "Broker LLC"
    assert out["trade_mode"] == "demo"
    assert out["account_type"] == "demo"
    assert out["is_demo"] is True
    assert out["is_live"] is False
    assert out["terminal_trade_allowed"] is False
    assert out["terminal_tradeapi_disabled"] is False
    assert out["terminal_connected"] is True
    assert out["auto_trading_enabled"] is False
    assert out["execution_ready"] is True
    assert out["execution_ready_strict"] is False
    assert out["execution_hard_blockers"] == []
    assert "Terminal AutoTrading is disabled." in out["execution_soft_blockers"]
    assert "Terminal AutoTrading is disabled." in out["execution_blockers"]


def test_trade_account_info_uses_single_detail_output_control() -> None:
    params = signature(_unwrap(trade_account_info)).parameters

    assert list(params) == ["detail"]
    assert params["detail"].default == "compact"
    assert "verbose" not in params


def test_trade_account_info_rounds_margin_level_for_display() -> None:
    gateway = SimpleNamespace(
        ensure_connection=lambda: None,
        account_info=lambda: SimpleNamespace(
            balance=10000.0,
            equity=10050.0,
            profit=50.0,
            margin=100.0,
            margin_free=9950.0,
            margin_level=53231.42857143,
            currency="USD",
            leverage=100,
            trade_allowed=True,
            trade_expert=True,
        ),
        terminal_info=lambda: None,
        build_trade_preflight=lambda account_info=None, terminal_info=None: {},
    )

    raw = _unwrap(trade_account_info)
    with patch.object(core_trading_account, "create_trading_gateway", return_value=gateway):
        out = raw()

    assert out["success"] is True
    assert out["margin_level"] == 53231.43


def test_trade_account_info_compact_detail_includes_account_fields_without_diagnostics() -> None:
    gateway = SimpleNamespace(
        ensure_connection=lambda: None,
        account_info=lambda: SimpleNamespace(
            balance=10000.0,
            equity=10050.0,
            profit=50.0,
            margin=100.0,
            margin_free=9950.0,
            margin_level=1000.0,
            currency="USD",
            leverage=100,
            trade_allowed=True,
            trade_expert=True,
            login=123456,
        ),
        terminal_info=lambda: None,
        build_trade_preflight=lambda account_info=None, terminal_info=None: {
            "login": 123456,
            "server": "Demo-Server",
            "company": "Broker LLC",
            "trade_mode": "demo",
            "account_type": "demo",
            "is_demo": True,
            "is_live": False,
            "execution_ready": True,
            "execution_blockers": [],
        },
    )

    raw = _unwrap(trade_account_info)
    with patch.object(core_trading_account, "create_trading_gateway", return_value=gateway):
        out = raw(detail="compact")

    assert out["balance"] == 10000.0
    assert out["source"] == "mt5_account_snapshot"
    assert out["timezone"] == "UTC"
    assert out["as_of_source"] == "client_utc_clock"
    assert out["as_of"].endswith("Z")
    assert out["retrieved_at"] == out["as_of"]
    assert out["profit"] == 50.0
    assert "floating_pnl" not in out
    assert "pnl_basis" not in out
    assert "equity_balance_delta" not in out
    assert out["margin"] == 100.0
    assert out["margin_free"] == 9950.0
    assert out["leverage"] == 100
    assert out["login"] == 123456
    assert out["server"] == "Demo-Server"
    assert out["company"] == "Broker LLC"
    assert out["trade_mode"] == "demo"
    assert out["account_type"] == "demo"
    assert out["is_demo"] is True
    assert out["is_live"] is False
    assert out["trade_allowed"] is True
    assert out["trade_expert"] is True
    assert "execution_ready" not in out


def test_trade_account_info_includes_terminal_server_clock_when_available() -> None:
    fixed_now = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)

    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed_now if tz is not None else fixed_now.replace(tzinfo=None)

    gateway = SimpleNamespace(
        ensure_connection=lambda: None,
        account_info=lambda: SimpleNamespace(
            balance=10000.0,
            equity=10050.0,
            profit=50.0,
            margin=100.0,
            margin_free=9950.0,
            margin_level=1000.0,
            currency="USD",
            leverage=100,
            trade_allowed=True,
            trade_expert=True,
        ),
        terminal_info=lambda: SimpleNamespace(server_time=fixed_now.timestamp() - 2.0),
        build_trade_preflight=lambda account_info=None, terminal_info=None: {},
    )

    raw = _unwrap(trade_account_info)
    with patch.object(core_trading_account, "create_trading_gateway", return_value=gateway), patch.object(
        core_trading_account,
        "datetime",
        FixedDatetime,
    ):
        out = raw(detail="compact")

    assert out["as_of"] == "2026-01-01T12:00:00Z"
    assert out["as_of_source"] == "client_utc_clock"
    assert out["server_time"] == "2026-01-01T11:59:58Z"
    assert out["server_time_source"] == "mt5_terminal_info.server_time"
    assert out["clock_skew_seconds"] == 2.0


def test_trade_account_info_rejects_unsupported_detail_modes() -> None:
    raw = _unwrap(trade_account_info)

    assert raw(detail="standard") == {
        "error": "Invalid detail level. Use 'compact' or 'full'."
    }
    assert raw(detail="summary") == {
        "error": "Invalid detail level. Use 'compact' or 'full'."
    }


def test_trade_account_info_rejects_unknown_detail() -> None:
    raw = _unwrap(trade_account_info)

    assert raw(detail="basic") == {"error": "Invalid detail level. Use 'compact' or 'full'."}


def test_trade_account_info_returns_connection_error_payload() -> None:
    raw = _unwrap(trade_account_info)

    with patch(
        "mtdata.core.trading.account.ensure_mt5_connection_or_raise",
        side_effect=MT5ConnectionError("Failed to connect to MetaTrader5. Ensure MT5 terminal is running."),
    ):
        out = raw()

    assert out == {"error": "Failed to connect to MetaTrader5. Ensure MT5 terminal is running."}


def test_trade_account_info_logs_finish_event(caplog) -> None:
    gateway = SimpleNamespace(
        ensure_connection=lambda: None,
        account_info=lambda: SimpleNamespace(
            balance=10000.0,
            equity=10050.0,
            profit=50.0,
            margin=100.0,
            margin_free=9950.0,
            margin_level=10050.0,
            currency="USD",
            leverage=100,
            trade_allowed=True,
            trade_expert=True,
        ),
        terminal_info=lambda: None,
        build_trade_preflight=lambda account_info=None, terminal_info=None: {
            "server": "Demo-Server",
            "company": "Broker LLC",
            "trade_mode": "demo",
            "terminal_trade_allowed": True,
            "terminal_tradeapi_disabled": False,
            "terminal_connected": True,
            "auto_trading_enabled": True,
            "execution_ready": True,
            "execution_ready_strict": True,
            "execution_hard_blockers": [],
            "execution_soft_blockers": [],
            "execution_blockers": [],
        },
    )

    raw = _unwrap(trade_account_info)
    with patch.object(core_trading_account, "create_trading_gateway", return_value=gateway), caplog.at_level(logging.DEBUG,
        logger=core_trading_account.logger.name,
    ):
        out = raw()

    assert out["success"] is True
    assert out["balance"] == 10000.0
    assert any(
        "event=finish operation=trade_account_info success=True" in record.message
        for record in caplog.records
    )


def test_run_trade_get_open_logs_finish_event(caplog) -> None:
    Position = namedtuple(
        "Position",
        ["ticket", "symbol", "time_update", "type", "volume", "price_open", "sl", "tp", "price_current", "swap", "profit", "comment", "magic"],
    )
    rows = [
        Position(
            ticket=1,
            symbol="EURUSD",
            time_update=1700000000,
            type=0,
            volume=0.1,
            price_open=1.1,
            sl=1.0,
            tp=1.2,
            price_current=1.15,
            swap=0.0,
            profit=5.0,
            comment="note",
            magic=7,
        )
    ]
    gateway = SimpleNamespace(
        ensure_connection=lambda: None,
        positions_get=lambda ticket=None, symbol=None: rows,
        POSITION_TYPE_BUY=0,
        POSITION_TYPE_SELL=1,
    )

    with caplog.at_level("DEBUG", logger="mtdata.core.trading.use_cases"):
        out = run_trade_get_open(
            TradeGetOpenRequest(),
            gateway=gateway,
            use_client_tz=lambda: False,
            format_time_minimal=lambda ts: f"t{int(ts)}",
            format_time_minimal_local=lambda ts: f"lt{int(ts)}",
            mt5_epoch_to_utc=lambda ts: ts,
            normalize_limit=lambda value: value,
            comment_row_metadata=lambda comment: {
                "comment_visible_length": len(comment or ""),
                "comment_max_length": 31,
                "comment_may_be_truncated": False,
            },
        )

    assert isinstance(out, list)
    assert any(
        "event=finish operation=trade_get_open success=True" in record.message
        for record in caplog.records
    )


def test_run_trade_get_open_accepts_simplenamespace_rows() -> None:
    rows = [
        SimpleNamespace(
            ticket=1,
            symbol="EURUSD",
            time_update=1700000000,
            type=0,
            volume=0.1,
            price_open=1.1,
            sl=1.0,
            tp=1.2,
            price_current=1.15,
            swap=0.0,
            profit=5.0,
            comment="note",
            magic=7,
        )
    ]
    gateway = SimpleNamespace(
        ensure_connection=lambda: None,
        positions_get=lambda ticket=None, symbol=None: rows,
        POSITION_TYPE_BUY=0,
        POSITION_TYPE_SELL=1,
    )

    out = run_trade_get_open(
        TradeGetOpenRequest(),
        gateway=gateway,
        use_client_tz=lambda: False,
        format_time_minimal=lambda ts: f"t{int(ts)}",
        format_time_minimal_local=lambda ts: f"lt{int(ts)}",
        mt5_epoch_to_utc=lambda ts: ts,
        normalize_limit=lambda value: value,
        comment_row_metadata=lambda comment: {
            "comment_visible_length": len(comment or ""),
            "comment_max_length": 31,
            "comment_may_be_truncated": False,
        },
    )

    assert out[0]["ticket"] == 1
    assert out[0]["time"] == "t1700000000"
    assert out[0]["side"] == "BUY"
    assert out[0]["profit"] == 5.0


def test_run_trade_get_open_filters_by_magic() -> None:
    rows = [
        SimpleNamespace(
            ticket=1,
            symbol="EURUSD",
            time_update=1700000000,
            type=0,
            volume=0.1,
            price_open=1.1,
            sl=1.0,
            tp=1.2,
            price_current=1.15,
            swap=0.0,
            profit=5.0,
            comment="a",
            magic=7,
        ),
        SimpleNamespace(
            ticket=2,
            symbol="EURUSD",
            time_update=1700000060,
            type=0,
            volume=0.2,
            price_open=1.2,
            sl=1.1,
            tp=1.3,
            price_current=1.25,
            swap=0.0,
            profit=6.0,
            comment="b",
            magic=9,
        ),
    ]
    gateway = SimpleNamespace(
        ensure_connection=lambda: None,
        positions_get=lambda ticket=None, symbol=None: rows,
        POSITION_TYPE_BUY=0,
        POSITION_TYPE_SELL=1,
    )

    out = run_trade_get_open(
        TradeGetOpenRequest(magic=9),
        gateway=gateway,
        use_client_tz=lambda: False,
        format_time_minimal=lambda ts: f"t{int(ts)}",
        format_time_minimal_local=lambda ts: f"lt{int(ts)}",
        mt5_epoch_to_utc=lambda ts: ts,
        normalize_limit=lambda value: value,
        comment_row_metadata=lambda comment: {},
    )

    assert [row["ticket"] for row in out] == [2]
    assert out[0]["magic"] == 9


def test_run_trade_get_open_filters_losses_and_orders_by_close_priority() -> None:
    rows = [
        SimpleNamespace(
            ticket=1,
            symbol="EURUSD",
            time_update=1700000000,
            type=0,
            volume=0.1,
            price_open=1.1,
            sl=1.0,
            tp=1.2,
            price_current=1.15,
            swap=0.0,
            profit=5.0,
            comment="winner",
            magic=7,
        ),
        SimpleNamespace(
            ticket=2,
            symbol="EURUSD",
            time_update=1700000060,
            type=0,
            volume=0.2,
            price_open=1.2,
            sl=1.1,
            tp=1.3,
            price_current=1.15,
            swap=0.0,
            profit=-7.0,
            comment="larger loss",
            magic=7,
        ),
        SimpleNamespace(
            ticket=3,
            symbol="EURUSD",
            time_update=1700000120,
            type=0,
            volume=0.1,
            price_open=1.3,
            sl=1.2,
            tp=1.4,
            price_current=1.25,
            swap=0.0,
            profit=-1.0,
            comment="smaller loss",
            magic=7,
        ),
    ]
    gateway = SimpleNamespace(
        ensure_connection=lambda: None,
        positions_get=lambda ticket=None, symbol=None: rows,
        POSITION_TYPE_BUY=0,
        POSITION_TYPE_SELL=1,
    )

    out = run_trade_get_open(
        TradeGetOpenRequest(loss_only=True, close_priority="loss_first"),
        gateway=gateway,
        use_client_tz=lambda: False,
        format_time_minimal=lambda ts: f"t{int(ts)}",
        format_time_minimal_local=lambda ts: f"lt{int(ts)}",
        mt5_epoch_to_utc=lambda ts: ts,
        normalize_limit=lambda value: value,
        comment_row_metadata=lambda comment: {},
    )

    assert [row["ticket"] for row in out] == [2, 3]
    assert [row["profit"] for row in out] == [-7.0, -1.0]


def test_run_trade_get_open_rejects_conflicting_profit_filters() -> None:
    gateway = SimpleNamespace(
        ensure_connection=lambda: None,
        positions_get=lambda ticket=None, symbol=None: [
            SimpleNamespace(
                ticket=1,
                symbol="EURUSD",
                time_update=1700000000,
                type=0,
                volume=0.1,
                price_open=1.1,
                sl=1.0,
                tp=1.2,
                price_current=1.15,
                swap=0.0,
                profit=5.0,
                comment="note",
                magic=7,
            )
        ],
        POSITION_TYPE_BUY=0,
        POSITION_TYPE_SELL=1,
    )

    out = run_trade_get_open(
        TradeGetOpenRequest(profit_only=True, loss_only=True),
        gateway=gateway,
        use_client_tz=lambda: False,
        format_time_minimal=lambda ts: f"t{int(ts)}",
        format_time_minimal_local=lambda ts: f"lt{int(ts)}",
        mt5_epoch_to_utc=lambda ts: ts,
        normalize_limit=lambda value: value,
        comment_row_metadata=lambda comment: {},
    )

    assert out[0]["error"] == "profit_only and loss_only cannot both be true."


def test_run_trade_get_pending_logs_finish_event(caplog) -> None:
    Order = namedtuple(
        "Order",
        ["ticket", "symbol", "time_setup", "type", "volume", "price_open", "sl", "tp", "price_current", "comment", "magic"],
    )
    rows = [
        Order(
            ticket=2,
            symbol="EURUSD",
            time_setup=1700000000,
            type=2,
            volume=0.1,
            price_open=1.1,
            sl=1.0,
            tp=1.2,
            price_current=1.15,
            comment="note",
            magic=7,
        )
    ]
    gateway = SimpleNamespace(
        ensure_connection=lambda: None,
        orders_get=lambda ticket=None, symbol=None: rows,
        ORDER_TYPE_BUY=0,
        ORDER_TYPE_SELL=1,
        ORDER_TYPE_BUY_LIMIT=2,
        ORDER_TYPE_SELL_LIMIT=3,
        ORDER_TYPE_BUY_STOP=4,
        ORDER_TYPE_SELL_STOP=5,
        ORDER_TYPE_BUY_STOP_LIMIT=6,
        ORDER_TYPE_SELL_STOP_LIMIT=7,
    )

    with caplog.at_level("DEBUG", logger="mtdata.core.trading.use_cases"):
        out = run_trade_get_pending(
            TradeGetPendingRequest(),
            gateway=gateway,
            use_client_tz=lambda: False,
            format_time_minimal=lambda ts: f"t{int(ts)}",
            format_time_minimal_local=lambda ts: f"lt{int(ts)}",
            mt5_epoch_to_utc=lambda ts: ts,
            normalize_limit=lambda value: value,
            comment_row_metadata=lambda comment: {
                "comment_visible_length": len(comment or ""),
                "comment_max_length": 31,
                "comment_may_be_truncated": False,
            },
        )

    assert isinstance(out, list)
    assert any(
        "event=finish operation=trade_get_pending success=True" in record.message
        for record in caplog.records
    )


def test_run_trade_get_open_limit_prefers_latest_valid_timestamps() -> None:
    Position = namedtuple(
        "Position",
        [
            "ticket",
            "symbol",
            "time_update",
            "type",
            "volume",
            "price_open",
            "sl",
            "tp",
            "price_current",
            "swap",
            "profit",
            "comment",
            "magic",
        ],
    )
    rows = [
        Position(1, "EURUSD", 100, 0, 0.1, 1.1, 1.0, 1.2, 1.15, 0.0, 1.0, "old", 7),
        Position(2, "EURUSD", None, 0, 0.1, 1.1, 1.0, 1.2, 1.15, 0.0, 2.0, "missing", 7),
        Position(3, "EURUSD", 200, 0, 0.1, 1.1, 1.0, 1.2, 1.15, 0.0, 3.0, "mid", 7),
        Position(4, "EURUSD", 300, 0, 0.1, 1.1, 1.0, 1.2, 1.15, 0.0, 4.0, "new", 7),
    ]
    gateway = SimpleNamespace(
        ensure_connection=lambda: None,
        positions_get=lambda ticket=None, symbol=None: rows,
        POSITION_TYPE_BUY=0,
        POSITION_TYPE_SELL=1,
    )

    out = run_trade_get_open(
        TradeGetOpenRequest(limit=2),
        gateway=gateway,
        use_client_tz=lambda: False,
        format_time_minimal=lambda ts: f"t{int(ts)}",
        format_time_minimal_local=lambda ts: f"lt{int(ts)}",
        mt5_epoch_to_utc=lambda ts: ts,
        normalize_limit=lambda value: value,
        comment_row_metadata=lambda comment: {
            "comment_visible_length": len(comment or ""),
            "comment_max_length": 31,
            "comment_may_be_truncated": False,
        },
    )

    rows = out["items"] if isinstance(out, dict) and "items" in out else out
    assert [row["ticket"] for row in rows] == [3, 4]
    assert [row["time"] for row in rows] == ["t200", "t300"]
    if isinstance(out, dict) and "items" in out:
        assert out["has_more"] is True
        assert out["truncated"] is True


def test_run_trade_get_pending_limit_prefers_latest_valid_timestamps() -> None:
    Order = namedtuple(
        "Order",
        [
            "ticket",
            "symbol",
            "time_setup",
            "type",
            "volume",
            "price_open",
            "sl",
            "tp",
            "price_current",
            "comment",
            "magic",
        ],
    )
    rows = [
        Order(11, "EURUSD", 100, 2, 0.1, 1.1, 1.0, 1.2, 1.15, "old", 7),
        Order(12, "EURUSD", None, 2, 0.1, 1.1, 1.0, 1.2, 1.15, "missing", 7),
        Order(13, "EURUSD", 200, 2, 0.1, 1.1, 1.0, 1.2, 1.15, "mid", 7),
        Order(14, "EURUSD", 300, 2, 0.1, 1.1, 1.0, 1.2, 1.15, "new", 7),
    ]
    gateway = SimpleNamespace(
        ensure_connection=lambda: None,
        orders_get=lambda ticket=None, symbol=None: rows,
        ORDER_TYPE_BUY=0,
        ORDER_TYPE_SELL=1,
        ORDER_TYPE_BUY_LIMIT=2,
        ORDER_TYPE_SELL_LIMIT=3,
        ORDER_TYPE_BUY_STOP=4,
        ORDER_TYPE_SELL_STOP=5,
        ORDER_TYPE_BUY_STOP_LIMIT=6,
        ORDER_TYPE_SELL_STOP_LIMIT=7,
    )

    out = run_trade_get_pending(
        TradeGetPendingRequest(limit=2),
        gateway=gateway,
        use_client_tz=lambda: False,
        format_time_minimal=lambda ts: f"t{int(ts)}",
        format_time_minimal_local=lambda ts: f"lt{int(ts)}",
        mt5_epoch_to_utc=lambda ts: ts,
        normalize_limit=lambda value: value,
        comment_row_metadata=lambda comment: {
            "comment_visible_length": len(comment or ""),
            "comment_max_length": 31,
            "comment_may_be_truncated": False,
        },
    )

    rows = out["items"] if isinstance(out, dict) and "items" in out else out
    assert [row["ticket"] for row in rows] == [13, 14]
    assert [row["time"] for row in rows] == ["t200", "t300"]
    if isinstance(out, dict) and "items" in out:
        assert out["has_more"] is True
        assert out["truncated"] is True


def test_run_trade_get_open_uses_snake_case_columns() -> None:
    Position = namedtuple(
        "Position",
        [
            "ticket",
            "symbol",
            "time",
            "type",
            "volume",
            "price_open",
            "sl",
            "tp",
            "price_current",
            "swap",
            "profit",
            "comment",
            "magic",
        ],
    )
    rows = [
        Position(21, "EURUSD", 1700000100, 0, 0.1, 1.1, 1.0, 1.2, 1.15, 0.0, 2.5, "note", 7),
    ]
    gateway = SimpleNamespace(
        ensure_connection=lambda: None,
        positions_get=lambda ticket=None, symbol=None: rows,
        POSITION_TYPE_BUY=0,
        POSITION_TYPE_SELL=1,
    )

    out = run_trade_get_open(
        TradeGetOpenRequest(),
        gateway=gateway,
        use_client_tz=lambda: False,
        format_time_minimal=lambda ts: f"t{int(ts)}",
        format_time_minimal_local=lambda ts: f"lt{int(ts)}",
        mt5_epoch_to_utc=lambda ts: ts,
        normalize_limit=lambda value: value,
        comment_row_metadata=lambda comment: {
            "comment_visible_length": len(comment or ""),
            "comment_max_length": 31,
            "comment_may_be_truncated": False,
        },
    )

    assert out[0]["ticket"] == 21
    assert out[0]["time"] == "t1700000100"
    assert out[0]["side"] == "BUY"
    assert out[0]["entry_price"] == 1.1
    assert out[0]["price_current"] == 1.15
    assert out[0]["comment"] == "note"
    assert out[0]["magic"] == 7
    assert "Ticket" not in out[0]
    assert "type" not in out[0]
    assert "price_open" not in out[0]


def test_run_trade_get_pending_uses_snake_case_columns() -> None:
    Order = namedtuple(
        "Order",
        [
            "ticket",
            "symbol",
            "time_setup",
            "time_expiration",
            "type",
            "volume",
            "price_open",
            "sl",
            "tp",
            "price_current",
            "comment",
            "magic",
        ],
    )
    rows = [
        Order(31, "EURUSD", 1700000200, 0, 2, 0.1, 1.1, 1.0, 1.2, 1.15, "pending", 9),
    ]
    gateway = SimpleNamespace(
        ensure_connection=lambda: None,
        orders_get=lambda ticket=None, symbol=None: rows,
        ORDER_TYPE_BUY=0,
        ORDER_TYPE_SELL=1,
        ORDER_TYPE_BUY_LIMIT=2,
        ORDER_TYPE_SELL_LIMIT=3,
        ORDER_TYPE_BUY_STOP=4,
        ORDER_TYPE_SELL_STOP=5,
        ORDER_TYPE_BUY_STOP_LIMIT=6,
        ORDER_TYPE_SELL_STOP_LIMIT=7,
    )

    out = run_trade_get_pending(
        TradeGetPendingRequest(),
        gateway=gateway,
        use_client_tz=lambda: False,
        format_time_minimal=lambda ts: f"t{int(ts)}",
        format_time_minimal_local=lambda ts: f"lt{int(ts)}",
        mt5_epoch_to_utc=lambda ts: ts,
        normalize_limit=lambda value: value,
        comment_row_metadata=lambda comment: {
            "comment_visible_length": len(comment or ""),
            "comment_max_length": 31,
            "comment_may_be_truncated": False,
        },
    )

    assert out[0]["ticket"] == 31
    assert out[0]["time"] == "t1700000200"
    assert out[0]["expiration"] == "GTC"
    assert out[0]["order_type"] == "BUY_LIMIT"
    assert out[0]["side"] == "BUY"
    assert out[0]["trigger_price"] == 1.1
    assert out[0]["price_current"] == 1.15
    assert out[0]["comment"] == "pending"
    assert out[0]["magic"] == 9
    assert "Ticket" not in out[0]
    assert "type" not in out[0]
    assert "price_open" not in out[0]


def test_run_trade_get_pending_filters_by_magic() -> None:
    rows = [
        SimpleNamespace(
            ticket=31,
            symbol="EURUSD",
            time_setup=1700000200,
            time_expiration=0,
            type=2,
            volume=0.1,
            price_open=1.1,
            sl=1.0,
            tp=1.2,
            price_current=1.15,
            comment="a",
            magic=7,
        ),
        SimpleNamespace(
            ticket=32,
            symbol="EURUSD",
            time_setup=1700000260,
            time_expiration=0,
            type=2,
            volume=0.2,
            price_open=1.2,
            sl=1.1,
            tp=1.3,
            price_current=1.25,
            comment="b",
            magic=9,
        ),
    ]
    gateway = SimpleNamespace(
        ensure_connection=lambda: None,
        orders_get=lambda ticket=None, symbol=None: rows,
        ORDER_TYPE_BUY=0,
        ORDER_TYPE_SELL=1,
        ORDER_TYPE_BUY_LIMIT=2,
        ORDER_TYPE_SELL_LIMIT=3,
        ORDER_TYPE_BUY_STOP=4,
        ORDER_TYPE_SELL_STOP=5,
        ORDER_TYPE_BUY_STOP_LIMIT=6,
        ORDER_TYPE_SELL_STOP_LIMIT=7,
    )

    out = run_trade_get_pending(
        TradeGetPendingRequest(magic=9),
        gateway=gateway,
        use_client_tz=lambda: False,
        format_time_minimal=lambda ts: f"t{int(ts)}",
        format_time_minimal_local=lambda ts: f"lt{int(ts)}",
        mt5_epoch_to_utc=lambda ts: ts,
        normalize_limit=lambda value: value,
        comment_row_metadata=lambda comment: {},
    )

    assert [row["ticket"] for row in out] == [32]
    assert out[0]["magic"] == 9


def test_run_trade_get_open_falls_back_to_time_column() -> None:
    Position = namedtuple(
        "Position",
        [
            "ticket",
            "symbol",
            "time",
            "type",
            "volume",
            "price_open",
            "sl",
            "tp",
            "price_current",
            "swap",
            "profit",
            "comment",
            "magic",
        ],
    )
    rows = [
        Position(21, "EURUSD", 1700000100, 0, 0.1, 1.1, 1.0, 1.2, 1.15, 0.0, 2.5, "note", 7),
    ]
    gateway = SimpleNamespace(
        ensure_connection=lambda: None,
        positions_get=lambda ticket=None, symbol=None: rows,
        POSITION_TYPE_BUY=0,
        POSITION_TYPE_SELL=1,
    )

    out = run_trade_get_open(
        TradeGetOpenRequest(),
        gateway=gateway,
        use_client_tz=lambda: False,
        format_time_minimal=lambda ts: f"t{int(ts)}",
        format_time_minimal_local=lambda ts: f"lt{int(ts)}",
        mt5_epoch_to_utc=lambda ts: ts,
        normalize_limit=lambda value: value,
        comment_row_metadata=lambda comment: {
            "comment_visible_length": len(comment or ""),
            "comment_max_length": 31,
            "comment_may_be_truncated": False,
        },
    )

    assert out[0]["ticket"] == 21
    assert out[0]["time"] == "t1700000100"
    assert out[0]["side"] == "BUY"


def test_run_trade_get_pending_falls_back_to_volume_initial() -> None:
    Order = namedtuple(
        "Order",
        [
            "ticket",
            "symbol",
            "time_setup",
            "type",
            "volume_current",
            "volume_initial",
            "price_open",
            "sl",
            "tp",
            "price_current",
            "comment",
            "magic",
        ],
    )
    rows = [
        Order(22, "EURUSD", 1700000200, 2, None, 0.3, 1.1, 1.0, 1.2, 1.15, "note", 7),
    ]
    gateway = SimpleNamespace(
        ensure_connection=lambda: None,
        orders_get=lambda ticket=None, symbol=None: rows,
        ORDER_TYPE_BUY=0,
        ORDER_TYPE_SELL=1,
        ORDER_TYPE_BUY_LIMIT=2,
        ORDER_TYPE_SELL_LIMIT=3,
        ORDER_TYPE_BUY_STOP=4,
        ORDER_TYPE_SELL_STOP=5,
        ORDER_TYPE_BUY_STOP_LIMIT=6,
        ORDER_TYPE_SELL_STOP_LIMIT=7,
    )

    out = run_trade_get_pending(
        TradeGetPendingRequest(),
        gateway=gateway,
        use_client_tz=lambda: False,
        format_time_minimal=lambda ts: f"t{int(ts)}",
        format_time_minimal_local=lambda ts: f"lt{int(ts)}",
        mt5_epoch_to_utc=lambda ts: ts,
        normalize_limit=lambda value: value,
        comment_row_metadata=lambda comment: {
            "comment_visible_length": len(comment or ""),
            "comment_max_length": 31,
            "comment_may_be_truncated": False,
        },
    )

    assert out[0]["ticket"] == 22
    assert out[0]["volume"] == 0.3
    assert out[0]["order_type"] == "BUY_LIMIT"
    assert out[0]["side"] == "BUY"


def test_trade_get_open_logs_finish_event(caplog) -> None:
    raw = _unwrap(core_trading_positions.trade_get_open)
    gateway = SimpleNamespace(account_info=lambda: SimpleNamespace(currency="USD"))

    def fake_run_trade_get_open(*args, **kwargs):
        assert kwargs["use_client_tz"]() is False
        assert kwargs["format_time_minimal_local"] is kwargs["format_time_minimal"]
        return [{"ticket": 1, "symbol": "EURUSD", "timezone": "UTC"}]

    with patch.object(core_trading_positions, "create_trading_gateway", return_value=gateway), patch.object(
        core_trading_positions,
        "run_trade_get_open",
        side_effect=fake_run_trade_get_open,
    ), caplog.at_level(logging.DEBUG, logger=core_trading_positions.logger.name):
        out = raw(TradeGetOpenRequest(symbol="EURUSD", limit=10))

    assert out["success"] is True
    assert out["count"] == 1
    assert out["items"][0]["ticket"] == 1
    assert out["timezone"] == "UTC"
    assert out["currency"] == "USD"
    assert any(
        "event=finish operation=trade_get_open success=True" in record.message
        for record in caplog.records
    )


def test_trade_get_pending_logs_finish_event(caplog) -> None:
    raw = _unwrap(core_trading_positions.trade_get_pending)

    def fake_run_trade_get_pending(*args, **kwargs):
        assert kwargs["use_client_tz"]() is False
        assert kwargs["format_time_minimal_local"] is kwargs["format_time_minimal"]
        return [{"ticket": 2, "symbol": "EURUSD", "timezone": "UTC"}]

    with patch.object(core_trading_positions, "create_trading_gateway", return_value=object()), patch.object(
        core_trading_positions,
        "run_trade_get_pending",
        side_effect=fake_run_trade_get_pending,
    ), caplog.at_level(logging.DEBUG, logger=core_trading_positions.logger.name):
        out = raw(TradeGetPendingRequest(symbol="EURUSD", limit=10))

    assert out["success"] is True
    assert out["count"] == 1
    assert out["items"][0]["ticket"] == 2
    assert out["timezone"] == "UTC"
    assert any(
        "event=finish operation=trade_get_pending success=True" in record.message
        for record in caplog.records
    )


def test_open_position_quote_context_discloses_basis_and_staleness() -> None:
    now_epoch = datetime(2026, 7, 12, 20, tzinfo=timezone.utc).timestamp()
    payload = {
        "items": [
            {"symbol": "EURUSD", "side": "SELL", "price_current": 1.14155}
        ]
    }
    gateway = SimpleNamespace(
        symbol_info_tick=lambda symbol: SimpleNamespace(
            time=now_epoch - 600,
            time_msc=0,
            bid=1.14139,
            ask=1.14155,
        )
    )

    core_trading_positions._attach_open_position_quote_context(
        payload,
        gateway,
        now_epoch=now_epoch,
    )

    row = payload["items"][0]
    assert row["price_current_basis"] == "ask"
    assert row["quote_time"].endswith("Z")
    assert row["data_age_seconds"] == 600.0
    assert row["data_stale"] is True
    assert row["usable_for_live_trading"] is False
    assert payload["quote_freshness_summary"]["stale_quotes"] == 1


def test_open_position_quote_summary_does_not_count_recent_quote_as_live() -> None:
    now_epoch = datetime(2026, 7, 14, 20, tzinfo=timezone.utc).timestamp()
    payload = {"items": [{"symbol": "EURUSD", "side": "BUY"}]}
    gateway = SimpleNamespace(
        symbol_info_tick=lambda symbol: SimpleNamespace(
            time=now_epoch - 45,
            time_msc=0,
            bid=1.14139,
            ask=1.14155,
        )
    )

    core_trading_positions._attach_open_position_quote_context(
        payload,
        gateway,
        now_epoch=now_epoch,
    )

    assert payload["items"][0]["usable_for_live_trading"] is False
    assert payload["quote_freshness_summary"] == {
        "positions_enriched": 1,
        "stale_quotes": 0,
        "live_usable_quotes": 0,
        "recent_or_delayed_quotes": 1,
    }
