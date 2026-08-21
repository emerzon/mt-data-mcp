"""Canonical tool-category identifiers shared by discovery surfaces."""

from __future__ import annotations

TOOL_CATEGORY_IDS = (
    "data",
    "symbols",
    "market",
    "news",
    "forecast",
    "analysis",
    "methods",
    "pattern_regime",
    "research",
    "report",
    "options",
    "trading",
)


def tool_catalog_category(name: str, *, module: str = "") -> str:
    """Return the one public category ID for a tool name."""
    tool_name = str(name or "").strip().lower()
    module_name = str(module or "").strip().lower()
    if tool_name.startswith(("trade_", "portfolio_")) or ".trading" in module_name:
        return "trading"
    if tool_name.startswith(("forecast_", "strategy_")):
        return "forecast"
    if tool_name in {"news", "calendar"} or tool_name.endswith("_news"):
        return "news"
    if tool_name in {"equity_profile", "screener"}:
        return "research"
    if tool_name == "asset_performance":
        return "market"
    if tool_name.startswith("market_"):
        return "market"
    if tool_name.startswith("symbols_"):
        return "symbols"
    if tool_name.startswith("data_") or tool_name == "wait_event":
        return "data"
    if tool_name.startswith(("patterns_", "regime_")):
        return "pattern_regime"
    if tool_name.startswith("options_"):
        return "options"
    if tool_name.startswith("report_"):
        return "report"
    if tool_name.startswith(("denoise_", "indicators_")):
        return "methods"
    if tool_name in {
        "pivot_compute_points",
        "support_resistance_levels",
        "temporal_analyze",
    }:
        return "analysis"
    return "research"


__all__ = ["TOOL_CATEGORY_IDS", "tool_catalog_category"]
