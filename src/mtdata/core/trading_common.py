"""Shared trading support helpers."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from . import trading_validation


def _trade_mode_text(mt5: Any, account_info: Any) -> Optional[str]:
    trade_mode = getattr(account_info, "trade_mode", None)
    if trade_mode is None:
        return None
    mapping = {
        getattr(mt5, "ACCOUNT_TRADE_MODE_DEMO", 0): "demo",
        getattr(mt5, "ACCOUNT_TRADE_MODE_CONTEST", 1): "contest",
        getattr(mt5, "ACCOUNT_TRADE_MODE_REAL", 2): "real",
    }
    return mapping.get(trade_mode, str(trade_mode))


def _retcode_name(mt5: Any, retcode: Any) -> Optional[str]:
    try:
        ret = int(retcode)
    except Exception:
        return None
    try:
        for attr in dir(mt5):
            if not attr.startswith("TRADE_RETCODE_"):
                continue
            try:
                if int(getattr(mt5, attr)) == ret:
                    return attr
            except Exception:
                continue
    except Exception:
        return None
    return None


def _build_trade_preflight(
    mt5: Any, account_info: Any = None, terminal_info: Any = None
) -> Dict[str, Any]:
    """Summarize account and terminal execution readiness."""
    info = account_info if account_info is not None else None
    term = terminal_info if terminal_info is not None else None
    if info is None:
        try:
            info = mt5.account_info()
        except Exception:
            info = None
    if term is None:
        try:
            term = mt5.terminal_info()
        except Exception:
            term = None

    account_trade_allowed = (
        trading_validation._coerce_optional_bool(getattr(info, "trade_allowed", None))
        if info is not None
        else None
    )
    account_trade_expert = (
        trading_validation._coerce_optional_bool(getattr(info, "trade_expert", None))
        if info is not None
        else None
    )
    terminal_trade_allowed = (
        trading_validation._coerce_optional_bool(getattr(term, "trade_allowed", None))
        if term is not None
        else None
    )
    terminal_tradeapi_disabled = (
        trading_validation._coerce_optional_bool(
            getattr(term, "tradeapi_disabled", None)
        )
        if term is not None
        else None
    )
    terminal_connected = (
        trading_validation._coerce_optional_bool(getattr(term, "connected", None))
        if term is not None
        else None
    )

    hard_blockers: List[str] = []
    soft_blockers: List[str] = []
    if account_trade_allowed is False:
        hard_blockers.append("Account trading is disabled.")
    if account_trade_expert is False:
        hard_blockers.append("Expert trading is disabled for the account.")
    if terminal_trade_allowed is False:
        soft_blockers.append("Terminal AutoTrading is disabled.")
    if terminal_tradeapi_disabled is True:
        hard_blockers.append("Terminal API trading is disabled.")
    if terminal_connected is False:
        hard_blockers.append("Terminal is not connected.")

    auto_trading_enabled = False
    if terminal_trade_allowed is True and terminal_tradeapi_disabled is not True:
        auto_trading_enabled = True

    all_blockers = hard_blockers + soft_blockers

    return {
        "server": getattr(info, "server", None) if info is not None else None,
        "company": getattr(info, "company", None) if info is not None else None,
        "trade_mode": _trade_mode_text(mt5, info) if info is not None else None,
        "trade_mode_raw": getattr(info, "trade_mode", None)
        if info is not None
        else None,
        "login": getattr(info, "login", None) if info is not None else None,
        "account_trade_allowed": account_trade_allowed,
        "account_trade_expert": account_trade_expert,
        "terminal_trade_allowed": terminal_trade_allowed,
        "terminal_tradeapi_disabled": terminal_tradeapi_disabled,
        "terminal_connected": terminal_connected,
        "community_account": trading_validation._coerce_optional_bool(
            getattr(term, "community_account", None)
        )
        if term is not None
        else None,
        "auto_trading_enabled": auto_trading_enabled,
        "execution_ready": len(hard_blockers) == 0,
        "execution_ready_strict": len(all_blockers) == 0,
        "execution_hard_blockers": hard_blockers,
        "execution_soft_blockers": soft_blockers,
        "execution_blockers": all_blockers,
    }
