"""Trading account and history views."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from ._mcp_instance import mcp
from . import trading_comments, trading_validation
from .config import mt5_config
from .trading_common import _build_trade_preflight
from .trading_requests import TradeHistoryRequest
from .trading_use_cases import run_trade_history
from ..utils.mt5 import _auto_connect_wrapper, _mt5_epoch_to_utc, mt5_adapter
from ..utils.mt5_enums import decode_mt5_enum_label
from ..utils.utils import (
    _format_time_minimal,
    _format_time_minimal_local,
    _normalize_limit,
    _parse_start_datetime,
    _use_client_tz,
)


@mcp.tool()
def trade_account_info() -> dict:
    """Get account information (balance, equity, profit, margin level, free margin, account type, leverage, currency)."""
    mt5 = mt5_adapter

    @_auto_connect_wrapper
    def _get_account_info():
        info = mt5.account_info()
        if info is None:
            return {"error": "Failed to get account info"}
        preflight = _build_trade_preflight(mt5, account_info=info)
        margin_level: Optional[float] = getattr(info, "margin_level", None)
        margin_level_note: Optional[str] = None
        try:
            margin_val = float(getattr(info, "margin", 0.0) or 0.0)
            ml_val = float(getattr(info, "margin_level", 0.0) or 0.0)
            if margin_val <= 0.0 and ml_val <= 0.0:
                margin_level = None
                margin_level_note = "N/A (no open margin/positions)"
            elif not math.isfinite(ml_val):
                margin_level = None
        except Exception:
            pass

        payload = {
            "balance": info.balance,
            "equity": info.equity,
            "profit": info.profit,
            "margin": info.margin,
            "margin_free": info.margin_free,
            "margin_level": margin_level,
            "currency": info.currency,
            "leverage": info.leverage,
            "trade_allowed": info.trade_allowed,
            "trade_expert": info.trade_expert,
            "server": preflight.get("server"),
            "company": preflight.get("company"),
            "trade_mode": preflight.get("trade_mode"),
            "terminal_trade_allowed": preflight.get("terminal_trade_allowed"),
            "terminal_tradeapi_disabled": preflight.get("terminal_tradeapi_disabled"),
            "terminal_connected": preflight.get("terminal_connected"),
            "auto_trading_enabled": preflight.get("auto_trading_enabled"),
            "execution_ready": preflight.get("execution_ready"),
            "execution_ready_strict": preflight.get("execution_ready_strict"),
            "execution_hard_blockers": preflight.get("execution_hard_blockers"),
            "execution_soft_blockers": preflight.get("execution_soft_blockers"),
            "execution_blockers": preflight.get("execution_blockers"),
        }
        if margin_level_note:
            payload["margin_level_note"] = margin_level_note
        return payload

    return _get_account_info()


@mcp.tool()
def trade_history(request: TradeHistoryRequest) -> List[Dict[str, Any]]:
    """Get deal or order history as tabular data."""
    return run_trade_history(
        request,
        mt5=mt5_adapter,
        auto_connect_wrapper=_auto_connect_wrapper,
        use_client_tz=_use_client_tz,
        format_time_minimal=_format_time_minimal,
        format_time_minimal_local=_format_time_minimal_local,
        mt5_epoch_to_utc=_mt5_epoch_to_utc,
        parse_start_datetime=_parse_start_datetime,
        normalize_limit=_normalize_limit,
        comment_row_metadata=trading_comments._comment_row_metadata,
        normalize_ticket_filter=trading_validation._normalize_ticket_filter,
        normalize_minutes_back=trading_validation._normalize_minutes_back,
        decode_mt5_enum_label=decode_mt5_enum_label,
        mt5_config=mt5_config,
    )
