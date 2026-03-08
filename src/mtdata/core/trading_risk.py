"""Trading risk analysis."""

from __future__ import annotations

import logging
import time

from ._mcp_instance import mcp
from .execution_logging import infer_result_success, log_operation_finish, log_operation_start
from .trading_gateway import MT5TradingGateway
from .trading_requests import TradeRiskAnalyzeRequest
from .trading_use_cases import run_trade_risk_analyze
from ..utils.mt5 import ensure_mt5_connection_or_raise, mt5_adapter

logger = logging.getLogger(__name__)


def _get_trading_gateway() -> MT5TradingGateway:
    return MT5TradingGateway(
        adapter=mt5_adapter,
        ensure_connection_impl=ensure_mt5_connection_or_raise,
    )


@mcp.tool()
def trade_risk_analyze(request: TradeRiskAnalyzeRequest) -> dict:
    """Analyze risk exposure for existing positions and calculate position sizing for new trades."""
    started_at = time.perf_counter()
    log_operation_start(
        logger,
        operation="trade_risk_analyze",
        symbol=request.symbol,
    )
    mt5 = _get_trading_gateway()
    result = run_trade_risk_analyze(
        request,
        gateway=mt5,
    )
    log_operation_finish(
        logger,
        operation="trade_risk_analyze",
        started_at=started_at,
        success=infer_result_success(result),
        symbol=request.symbol,
    )
    return result
