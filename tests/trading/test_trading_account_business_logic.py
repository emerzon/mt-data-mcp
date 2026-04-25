from __future__ import annotations

import logging
import sys
from inspect import signature
from collections import namedtuple
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
        out = raw()

    if prev is not None:
        sys.modules["MetaTrader5"] = prev

    assert out["success"] is True
    assert out["server"] == "Demo-Server"
    assert out["company"] == "Broker LLC"
    assert out["trade_mode"] == "demo"
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
        build_trade_preflight=lambda account_info=None: {},
    )

    raw = _unwrap(trade_account_info)
    with patch.object(core_trading_account, "create_trading_gateway", return_value=gateway):
        out = raw()

    assert out["success"] is True
    assert out["margin_level"] == 53231.43


def test_trade_account_info_summary_detail_is_minimal_snapshot() -> None:
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
        build_trade_preflight=lambda account_info=None: {
            "server": "Demo-Server",
            "company": "Broker LLC",
            "trade_mode": "demo",
            "execution_ready": True,
            "execution_blockers": [],
        },
    )

    raw = _unwrap(trade_account_info)
    with patch.object(core_trading_account, "create_trading_gateway", return_value=gateway):
        out = raw(detail="summary")

    assert set(out) == {
        "success",
        "balance",
        "equity",
        "margin_level",
        "currency",
        "meta",
    }
    assert out["balance"] == 10000.0
    assert out["margin_level"] == 53231.43
    assert out["meta"]["tool"] == "trade_account_info"
    assert "profit" not in out
    assert "margin_free" not in out
    assert "execution_ready" not in out
    assert "server" not in out


def test_trade_account_info_compact_detail_includes_margin_fields_without_identity() -> None:
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
        build_trade_preflight=lambda account_info=None: {
            "server": "Demo-Server",
            "company": "Broker LLC",
            "trade_mode": "demo",
            "execution_ready": True,
            "execution_blockers": [],
        },
    )

    raw = _unwrap(trade_account_info)
    with patch.object(core_trading_account, "create_trading_gateway", return_value=gateway):
        out = raw(detail="compact")

    assert out["balance"] == 10000.0
    assert out["profit"] == 50.0
    assert out["margin"] == 100.0
    assert out["margin_free"] == 9950.0
    assert out["leverage"] == 100
    assert "execution_ready" not in out
    assert "server" not in out


def test_trade_account_info_standard_detail_aliases_compact() -> None:
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
        build_trade_preflight=lambda account_info=None: {
            "server": "Demo-Server",
            "company": "Broker LLC",
            "trade_mode": "demo",
            "execution_ready": True,
            "execution_blockers": [],
        },
    )

    raw = _unwrap(trade_account_info)
    with patch.object(core_trading_account, "create_trading_gateway", return_value=gateway):
        compact = raw(detail="compact")
        standard = raw(detail="standard")

    compact_without_meta = {key: value for key, value in compact.items() if key != "meta"}
    standard_without_meta = {key: value for key, value in standard.items() if key != "meta"}
    assert standard_without_meta == compact_without_meta


def test_trade_account_info_basic_detail_keeps_identity_without_diagnostics() -> None:
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
            trade_expert=False,
        ),
        build_trade_preflight=lambda account_info=None: {
            "server": "Demo-Server",
            "company": "Broker LLC",
            "trade_mode": "demo",
            "execution_ready": False,
            "execution_hard_blockers": ["Expert trading is disabled for the account."],
            "execution_blockers": ["Expert trading is disabled for the account."],
        },
    )

    raw = _unwrap(trade_account_info)
    with patch.object(core_trading_account, "create_trading_gateway", return_value=gateway):
        out = raw(detail="basic")

    assert out["success"] is True
    assert out["server"] == "Demo-Server"
    assert out["company"] == "Broker LLC"
    assert out["trade_mode"] == "demo"
    assert out["trade_expert"] is False
    assert "execution_ready" not in out
    assert "execution_blockers" not in out
    assert out["meta"]["tool"] == "trade_account_info"


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
        build_trade_preflight=lambda account_info=None: {
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
    with patch.object(core_trading_account, "create_trading_gateway", return_value=gateway), caplog.at_level(
        logging.INFO,
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

    with caplog.at_level("INFO", logger="mtdata.core.trading.use_cases"):
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
    assert out[0]["type"] == "BUY"
    assert out[0]["profit"] == 5.0


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

    with caplog.at_level("INFO", logger="mtdata.core.trading.use_cases"):
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

    assert [row["ticket"] for row in out] == [3, 4]
    assert [row["time"] for row in out] == ["t200", "t300"]


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

    assert [row["ticket"] for row in out] == [13, 14]
    assert [row["time"] for row in out] == ["t200", "t300"]


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
    assert out[0]["type"] == "BUY"
    assert out[0]["price_open"] == 1.1
    assert out[0]["price_current"] == 1.15
    assert out[0]["comment"] == "note"
    assert out[0]["magic"] == 7
    assert "Ticket" not in out[0]


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
    assert out[0]["type"] == "BUY_LIMIT"
    assert out[0]["price_open"] == 1.1
    assert out[0]["price_current"] == 1.15
    assert out[0]["comment"] == "pending"
    assert out[0]["magic"] == 9
    assert "Ticket" not in out[0]


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
    assert out[0]["type"] == "BUY"


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
    assert out[0]["type"] == "BUY_LIMIT"


def test_trade_get_open_logs_finish_event(caplog) -> None:
    raw = _unwrap(core_trading_positions.trade_get_open)

    with patch.object(core_trading_positions, "create_trading_gateway", return_value=object()), patch.object(
        core_trading_positions,
        "run_trade_get_open",
        return_value=[{"ticket": 1, "symbol": "EURUSD"}],
    ), caplog.at_level(logging.INFO, logger=core_trading_positions.logger.name):
        out = raw(TradeGetOpenRequest(symbol="EURUSD", limit=10))

    assert out["success"] is True
    assert out["count"] == 1
    assert out["items"][0]["ticket"] == 1
    assert any(
        "event=finish operation=trade_get_open success=True" in record.message
        for record in caplog.records
    )


def test_trade_get_pending_logs_finish_event(caplog) -> None:
    raw = _unwrap(core_trading_positions.trade_get_pending)

    with patch.object(core_trading_positions, "create_trading_gateway", return_value=object()), patch.object(
        core_trading_positions,
        "run_trade_get_pending",
        return_value=[{"ticket": 2, "symbol": "EURUSD"}],
    ), caplog.at_level(logging.INFO, logger=core_trading_positions.logger.name):
        out = raw(TradeGetPendingRequest(symbol="EURUSD", limit=10))

    assert out["success"] is True
    assert out["count"] == 1
    assert out["items"][0]["ticket"] == 2
    assert any(
        "event=finish operation=trade_get_pending success=True" in record.message
        for record in caplog.records
    )
