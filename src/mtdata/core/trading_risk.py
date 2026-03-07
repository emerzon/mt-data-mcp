"""Trading risk analysis."""

from __future__ import annotations

from ._mcp_instance import mcp
from .trading_gateway import MT5TradingGateway
from .trading_requests import TradeRiskAnalyzeRequest
from .trading_use_cases import run_trade_risk_analyze
from ..utils.mt5 import ensure_mt5_connection_or_raise, mt5_adapter


def _get_trading_gateway() -> MT5TradingGateway:
    return MT5TradingGateway(
        adapter=mt5_adapter,
        ensure_connection_impl=ensure_mt5_connection_or_raise,
    )


@mcp.tool()
def trade_risk_analyze(request: TradeRiskAnalyzeRequest) -> dict:
    """Analyze risk exposure for existing positions and calculate position sizing for new trades."""
    mt5 = _get_trading_gateway()
    return run_trade_risk_analyze(
        request,
        mt5=mt5,
        ensure_connection=mt5.ensure_connection,
    )
