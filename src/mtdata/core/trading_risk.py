"""Trading risk analysis."""

from __future__ import annotations

from ._mcp_instance import mcp
from .trading_requests import TradeRiskAnalyzeRequest
from .trading_use_cases import run_trade_risk_analyze
from ..utils.mt5 import ensure_mt5_connection_or_raise, mt5_adapter


@mcp.tool()
def trade_risk_analyze(request: TradeRiskAnalyzeRequest) -> dict:
    """Analyze risk exposure for existing positions and calculate position sizing for new trades."""
    return run_trade_risk_analyze(
        request,
        mt5=mt5_adapter,
        ensure_connection=ensure_mt5_connection_or_raise,
    )
