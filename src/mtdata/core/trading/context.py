"""Trading session context utilities."""

import logging
from typing import Any, Dict

from .._mcp_instance import mcp
from ..execution_logging import run_logged_operation
from ..market_depth import market_ticker
from ..output_contract import ensure_common_meta
from .account import trade_account_info
from .positions import trade_get_open, trade_get_pending
from .requests import TradeGetOpenRequest, TradeGetPendingRequest

logger = logging.getLogger(__name__)


@mcp.tool()
def trade_session_context(symbol: str) -> Dict[str, Any]:
    """Get a consolidated session context including account info, open positions, pending orders, ticker, and computed state for a symbol.

    Parameters: symbol
    """

    def _run() -> Dict[str, Any]:
        # Un-wrap original functions if necessary to bypass double-logging or async mcp wrappers
        acc_func = getattr(trade_account_info, "__wrapped__", trade_account_info)
        ticker_func = getattr(market_ticker, "__wrapped__", market_ticker)
        open_func = getattr(trade_get_open, "__wrapped__", trade_get_open)
        pending_func = getattr(trade_get_pending, "__wrapped__", trade_get_pending)

        account_res = acc_func()
        ticker_res = ticker_func(symbol=symbol)

        open_req = TradeGetOpenRequest(symbol=symbol)
        open_res = open_func(request=open_req)

        pending_req = TradeGetPendingRequest(symbol=symbol)
        pending_res = pending_func(request=pending_req)

        # Determine internal book state
        has_open = bool(open_res.get("success", False) and open_res.get("count", 0) > 0)
        has_pending = bool(pending_res.get("success", False) and pending_res.get("count", 0) > 0)

        if has_open and has_pending:
            state = "mixed"
        elif has_open:
            state = "open_position"
        elif has_pending:
            state = "pending_only"
        else:
            state = "flat"

        return ensure_common_meta(
            {
                "success": True,
                "symbol": symbol,
                "state": state,
                "account": account_res,
                "open_positions": open_res,
                "pending_orders": pending_res,
                "ticker": ticker_res,
            },
            tool_name="trade_session_context",
        )

    return run_logged_operation(
        logger,
        operation="trade_session_context",
        symbol=symbol,
        func=_run,
    )
