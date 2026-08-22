"""Trade use-case orchestration package."""

from __future__ import annotations

from mtdata.core.trading.use_cases.close import run_trade_close
from mtdata.core.trading.use_cases.history import run_trade_history
from mtdata.core.trading.use_cases.modify import run_trade_modify
from mtdata.core.trading.use_cases.place import run_trade_place
from mtdata.core.trading.use_cases.query import (
    run_trade_get_open,
    run_trade_get_pending,
)
from mtdata.core.trading.use_cases.risk import (
    run_trade_risk_analyze,
    run_trade_stress_test,
    run_trade_var_cvar_calculate,
)

__all__ = [
    "run_trade_close",
    "run_trade_get_open",
    "run_trade_get_pending",
    "run_trade_history",
    "run_trade_modify",
    "run_trade_place",
    "run_trade_risk_analyze",
    "run_trade_stress_test",
    "run_trade_var_cvar_calculate",
]
