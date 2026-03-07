from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import sys

from mtdata.core.trading import trade_account_info


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
    with patch("mtdata.core.trading_account._auto_connect_wrapper", lambda f: f):
        out = raw()

    if prev is not None:
        sys.modules["MetaTrader5"] = prev

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
