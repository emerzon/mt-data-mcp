"""Explicit tool-module bootstrap for transport adapters."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Final, Iterable, Optional

from ..core._mcp_instance import mcp
from ..core.schema_attach import attach_schemas_to_tools
from ..shared.schema import get_shared_enum_lists

TOOL_MODULE_NAMES: Final[tuple[str, ...]] = (
    "mtdata.core.data",
    "mtdata.core.forecast",
    "mtdata.core.forecast_tasks",
    "mtdata.core.causal",
    "mtdata.core.analytics",
    "mtdata.core.diagnostics",
    "mtdata.core.denoise",
    "mtdata.core.indicators",
    "mtdata.core.market_depth",
    "mtdata.core.market_snapshot",
    "mtdata.core.options",
    "mtdata.core.patterns",
    "mtdata.core.pivot",
    "mtdata.core.volume_profile",
    "mtdata.core.symbols",
    "mtdata.core.regime",
    "mtdata.core.labels",
    "mtdata.core.market_status",
    "mtdata.core.report",
    "mtdata.core.trading",
    "mtdata.core.temporal",
    "mtdata.core.finviz",
    "mtdata.core.news",
    "mtdata.core.tools",
)

_BOOTSTRAPPED_MODULES: dict[str, ModuleType] = {}


def cli_tool_module_names(command: str) -> Optional[tuple[str, ...]]:
    """Return the minimal bootstrap module for a known CLI command family."""
    name = str(command or "").strip().lower().replace("-", "_")
    if not name or name == "tools_list":
        return None
    special = {
        "market_microstructure_analyze": "mtdata.core.analytics",
        "market_relative_strength": "mtdata.core.analytics",
        "portfolio_risk_decompose": "mtdata.core.analytics",
        "strategy_validate": "mtdata.core.analytics",
        "trade_execution_quality": "mtdata.core.analytics",
        "strategy_backtest": "mtdata.core.forecast",
        "volatility_term_structure": "mtdata.core.diagnostics",
        "market_ticker": "mtdata.core.market_depth",
        "market_snapshot": "mtdata.core.market_snapshot",
        "market_status": "mtdata.core.market_status",
        "market_scan": "mtdata.core.symbols",
        "support_resistance_levels": "mtdata.core.pivot",
        "confluence_levels": "mtdata.core.pivot",
        "pivot_compute_points": "mtdata.core.pivot",
        "volume_profile_levels": "mtdata.core.volume_profile",
        "news": "mtdata.core.news",
        "report_generate": "mtdata.core.report",
        "regime_detect": "mtdata.core.regime",
        "patterns_detect": "mtdata.core.patterns",
        "labels_triple_barrier": "mtdata.core.labels",
        "temporal_analyze": "mtdata.core.temporal",
        "wait_event": "mtdata.core.data",
    }
    module_name = special.get(name)
    if module_name is None:
        if name.startswith(("forecast_task_", "forecast_models_")) or name == "forecast_train":
            module_name = "mtdata.core.forecast_tasks"
        elif name.startswith("forecast_"):
            module_name = "mtdata.core.forecast"
        elif name.startswith("trade_"):
            module_name = "mtdata.core.trading"
        elif name.startswith("data_"):
            module_name = "mtdata.core.data"
        elif name.startswith("causal_") or name in {
            "cointegration_test",
            "correlation_matrix",
            "cross_correlation",
        }:
            module_name = "mtdata.core.causal"
        elif name.startswith("denoise_"):
            module_name = "mtdata.core.denoise"
        elif name.startswith("indicators_"):
            module_name = "mtdata.core.indicators"
        elif name.startswith("options_"):
            module_name = "mtdata.core.options"
        elif name.startswith("symbols_"):
            module_name = "mtdata.core.symbols"
        elif name.startswith("finviz_"):
            module_name = "mtdata.core.finviz"
        elif name in {"outliers_detect", "seasonality_detect", "stationarity_test"}:
            module_name = "mtdata.core.diagnostics"
    return (module_name,) if module_name in TOOL_MODULE_NAMES else None


def bootstrap_tools(module_names: Optional[Iterable[str]] = None) -> tuple[ModuleType, ...]:
    """Load selected tool modules, or the complete surface when none are selected."""
    requested = tuple(module_names) if module_names is not None else TOOL_MODULE_NAMES
    unknown = [name for name in requested if name not in TOOL_MODULE_NAMES]
    if unknown:
        raise ValueError(f"Unknown tool bootstrap module(s): {', '.join(unknown)}")
    for name in requested:
        if name not in _BOOTSTRAPPED_MODULES:
            _BOOTSTRAPPED_MODULES[name] = import_module(name)

    # This is intentionally repeatable: warm-shell calls can import another
    # module later, whose newly registered tools still need shared schemas.
    attach_schemas_to_tools(mcp, get_shared_enum_lists())
    return tuple(_BOOTSTRAPPED_MODULES[name] for name in requested)
