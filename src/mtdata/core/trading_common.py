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


def _build_trade_preflight(mt5: Any, account_info: Any = None, terminal_info: Any = None) -> Dict[str, Any]:
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

    account_trade_allowed = trading_validation._coerce_optional_bool(getattr(info, "trade_allowed", None)) if info is not None else None
    account_trade_expert = trading_validation._coerce_optional_bool(getattr(info, "trade_expert", None)) if info is not None else None
    terminal_trade_allowed = trading_validation._coerce_optional_bool(getattr(term, "trade_allowed", None)) if term is not None else None
    terminal_tradeapi_disabled = trading_validation._coerce_optional_bool(getattr(term, "tradeapi_disabled", None)) if term is not None else None
    terminal_connected = trading_validation._coerce_optional_bool(getattr(term, "connected", None)) if term is not None else None

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
        "trade_mode_raw": getattr(info, "trade_mode", None) if info is not None else None,
        "login": getattr(info, "login", None) if info is not None else None,
        "account_trade_allowed": account_trade_allowed,
        "account_trade_expert": account_trade_expert,
        "terminal_trade_allowed": terminal_trade_allowed,
        "terminal_tradeapi_disabled": terminal_tradeapi_disabled,
        "terminal_connected": terminal_connected,
        "community_account": trading_validation._coerce_optional_bool(getattr(term, "community_account", None)) if term is not None else None,
        "auto_trading_enabled": auto_trading_enabled,
        "execution_ready": len(hard_blockers) == 0,
        "execution_ready_strict": len(all_blockers) == 0,
        "execution_hard_blockers": hard_blockers,
        "execution_soft_blockers": soft_blockers,
        "execution_blockers": all_blockers,
    }


def _build_trade_preflight_guidance(preflight: Optional[Dict[str, Any]]) -> List[str]:
    """Translate preflight blockers into concise next-step guidance."""
    if not isinstance(preflight, dict):
        return []

    blockers = [str(item).strip() for item in preflight.get("execution_blockers") or [] if str(item).strip()]
    guidance: List[str] = []

    if "Terminal AutoTrading is disabled." in blockers:
        guidance.append("Enable AutoTrading in the MT5 terminal toolbar, then retry the order.")
    if "Terminal API trading is disabled." in blockers:
        guidance.append(
            "In MT5, open Tools > Options > Expert Advisors and enable algorithmic/API trading."
        )
    if "Terminal is not connected." in blockers:
        guidance.append("Reconnect the MT5 terminal to the broker server before placing orders.")
    if "Expert trading is disabled for the account." in blockers:
        guidance.append("Use an account that permits expert trading, or ask the broker to enable it.")
    if "Account trading is disabled." in blockers:
        guidance.append("Check the account permissions/status with the broker before retrying.")

    if guidance:
        return guidance

    if blockers:
        return ["Review the returned preflight blockers and resolve them before retrying the order."]
    return []
