"""Trading risk analysis."""

from __future__ import annotations

import logging

from .._mcp_instance import mcp
from ..execution_logging import run_logged_operation
from .gateway import create_trading_gateway
from .requests import TradeRiskAnalyzeRequest
from .use_cases import run_trade_risk_analyze
from ...utils.mt5 import ensure_mt5_connection_or_raise, mt5_adapter

logger = logging.getLogger(__name__)


@mcp.tool()
def trade_risk_analyze(request: TradeRiskAnalyzeRequest) -> dict:
    """Analyze risk exposure for existing positions and calculate position sizing for new trades.

    When sizing a proposed trade, pass direction='long' or direction='short' to
    validate that proposed SL/TP are on the correct side of the entry.
    """
    return run_logged_operation(
        logger,
        operation="trade_risk_analyze",
        symbol=request.symbol,
        func=lambda: run_trade_risk_analyze(
            request,
            gateway=create_trading_gateway(
                adapter=mt5_adapter,
                ensure_connection_impl=ensure_mt5_connection_or_raise,
            ),
        ),
    )
