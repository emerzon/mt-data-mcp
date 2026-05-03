"""Trading risk analysis."""

from __future__ import annotations

import logging

from ...utils.mt5 import ensure_mt5_connection_or_raise, mt5_adapter
from .._mcp_instance import mcp
from ..execution_logging import run_logged_operation
from .gateway import create_trading_gateway
from .requests import TradeRiskAnalyzeRequest, TradeVarCvarRequest
from .use_cases import run_trade_risk_analyze, run_trade_var_cvar_calculate

logger = logging.getLogger(__name__)


@mcp.tool()
def trade_risk_analyze(request: TradeRiskAnalyzeRequest) -> dict:
    """Analyze risk exposure for existing positions and calculate position sizing for new trades.

    Use this for symbol-level trade planning: current exposure, stop-loss risk,
    reward/risk, and optional lot sizing from `desired_risk_pct`, `entry`, and
    `stop_loss`. It is not a portfolio tail-risk model; use
    `trade_var_cvar_calculate` for VaR/CVaR across open positions.
    Compact detail keeps the sizing decision fields; full detail includes
    broker volume rounding diagnostics and incomplete-sizing context.

    When sizing a proposed trade, pass direction='long' or direction='short' to
    validate that proposed SL/TP are on the correct side of the entry. New-trade
    sizing is opt-in: provide desired_risk_pct together with entry and stop_loss
    to calculate lot size.
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


@mcp.tool()
def trade_var_cvar_calculate(request: TradeVarCvarRequest) -> dict:
    """Estimate portfolio VaR/CVaR for current open MT5 positions.

    Use this for account-level tail-risk analysis over the current portfolio.
    For single-symbol stop-loss exposure and new-trade lot sizing, use
    `trade_risk_analyze`. For a lightweight execution snapshot that includes
    account, ticker, open positions, and pending orders, use
    `trade_session_context`.
    """
    return run_logged_operation(
        logger,
        operation="trade_var_cvar_calculate",
        symbol=request.symbol,
        timeframe=request.timeframe,
        func=lambda: run_trade_var_cvar_calculate(
            request,
            gateway=create_trading_gateway(
                adapter=mt5_adapter,
                ensure_connection_impl=ensure_mt5_connection_or_raise,
            ),
        ),
    )
