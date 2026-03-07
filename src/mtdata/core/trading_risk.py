"""Trading risk analysis."""

from __future__ import annotations

from ._mcp_instance import mcp
from .trading_requests import TradeRiskAnalyzeRequest
from .trading_use_cases import run_trade_risk_analyze
from ..utils.mt5 import _auto_connect_wrapper, mt5_adapter


@mcp.tool()
def trade_risk_analyze(request: TradeRiskAnalyzeRequest) -> dict:
    """Analyze risk exposure for existing positions and calculate position sizing for new trades."""
    return run_trade_risk_analyze(
        request,
        mt5=mt5_adapter,
        auto_connect_wrapper=_auto_connect_wrapper,
    )
