"""Static command catalog used by the lightweight CLI entry point.

Keep this module free of imports from the tool graph.  A parity test guards the
catalog against drift from the dynamically registered MCP tools.
"""

from __future__ import annotations

import os
from collections.abc import Iterable

CLI_COMMAND_NAMES = (
    "causal_discover_signals",
    "cointegration_test",
    "confluence_levels",
    "correlation_matrix",
    "cross_correlation",
    "data_fetch_candles",
    "data_fetch_ticks",
    "denoise_describe",
    "denoise_list_methods",
    "finviz_calendar",
    "finviz_crypto",
    "finviz_description",
    "finviz_earnings",
    "finviz_filters_list",
    "finviz_forex",
    "finviz_fundamentals",
    "finviz_futures",
    "finviz_insider",
    "finviz_insider_activity",
    "finviz_market_news",
    "finviz_news",
    "finviz_peers",
    "finviz_ratings",
    "finviz_screen",
    "forecast_backtest_run",
    "forecast_barrier_optimize",
    "forecast_barrier_prob",
    "forecast_conformal_intervals",
    "forecast_generate",
    "forecast_list_library_models",
    "forecast_list_methods",
    "forecast_models_cleanup",
    "forecast_models_delete",
    "forecast_models_list",
    "forecast_optimize_hints",
    "forecast_task_cancel",
    "forecast_task_cancel_all",
    "forecast_task_list",
    "forecast_task_status",
    "forecast_task_wait",
    "forecast_train",
    "forecast_tune_genetic",
    "forecast_tune_optuna",
    "forecast_volatility_estimate",
    "indicators_describe",
    "indicators_list",
    "labels_triple_barrier",
    "market_microstructure_analyze",
    "market_relative_strength",
    "market_scan",
    "market_snapshot",
    "market_status",
    "market_ticker",
    "news",
    "options_barrier_price",
    "options_chain",
    "options_expirations",
    "options_heston_calibrate",
    "options_provider_status",
    "outliers_detect",
    "patterns_detect",
    "pivot_compute_points",
    "portfolio_risk_decompose",
    "regime_detect",
    "report_generate",
    "seasonality_detect",
    "stationarity_test",
    "strategy_backtest",
    "strategy_validate",
    "support_resistance_levels",
    "symbols_describe",
    "symbols_list",
    "symbols_top_markets",
    "temporal_analyze",
    "tools_list",
    "trade_account_info",
    "trade_close",
    "trade_execution_quality",
    "trade_get_open",
    "trade_get_pending",
    "trade_history",
    "trade_journal_analyze",
    "trade_modify",
    "trade_place",
    "trade_risk_analyze",
    "trade_session_context",
    "trade_stress_test",
    "trade_var_cvar_calculate",
    "volatility_term_structure",
    "volume_profile_levels",
    "wait_event",
)

_OPTIONAL_COMMAND_ENV = {
    "market_depth_fetch": "MTDATA_ENABLE_MARKET_DEPTH_FETCH",
}

_CATEGORY_PREFIXES = (
    ("DATA AND MARKET", ("data_", "market_", "symbols_", "wait_event")),
    ("FORECAST", ("forecast_",)),
    ("TRADING AND RISK", ("trade_", "portfolio_", "strategy_")),
    ("NEWS, OPTIONS, AND FINVIZ", ("news", "options_", "finviz_")),
    (
        "ANALYSIS",
        (
            "causal_",
            "cointegration_",
            "confluence_",
            "correlation_",
            "cross_correlation",
            "denoise_",
            "indicators_",
            "labels_",
            "outliers_",
            "patterns_",
            "pivot_",
            "regime_",
            "seasonality_",
            "stationarity_",
            "support_",
            "temporal_",
            "volatility_",
            "volume_",
        ),
    ),
)


def _env_enabled(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def available_command_names() -> tuple[str, ...]:
    """Return always-on commands plus explicitly enabled gated commands."""
    optional = tuple(
        command
        for command, env_name in _OPTIONAL_COMMAND_ENV.items()
        if _env_enabled(env_name)
    )
    return tuple(sorted((*CLI_COMMAND_NAMES, *optional)))


def _matches_prefix(name: str, prefixes: Iterable[str]) -> bool:
    return any(name == prefix or name.startswith(prefix) for prefix in prefixes)


def format_root_help(program: str) -> str:
    """Render root help without importing any tool implementation modules."""
    names = available_command_names()
    grouped_names: set[str] = set()
    lines = [
        f"usage: {program} [-h] [-V] [--json] [--extras EXTRA[,EXTRA...]]",
        "                  [--fields FIELD[,FIELD...]] [--precision MODE]",
        "                  [--timeframe TIMEFRAME] <command> ...",
        "",
        "MetaTrader 5 research, forecasting, and trading CLI. One-shot commands",
        "load only their tool family; use shell for repeated commands in one warm",
        "process. TOON is the default output format; pass --json for JSON.",
        "",
        "Commands:",
        "  shell  Run interactive commands or a stdin batch in one warm process",
    ]
    for category, prefixes in _CATEGORY_PREFIXES:
        rows = [name for name in names if _matches_prefix(name, prefixes)]
        if not rows:
            continue
        grouped_names.update(rows)
        lines.extend(("", f"  {category}:", f"    {' '.join(rows)}"))
    remaining = [name for name in names if name not in grouped_names]
    if remaining:
        lines.extend(("", "  OTHER:", f"    {' '.join(remaining)}"))
    lines.extend(
        (
            "",
            "Global options:",
            "  -h, --help              Show this help and exit",
            "  -V, --version           Show installed mtdata version and exit",
            "  --json                  Output structured JSON instead of TOON",
            "  --extras EXTRA,...      Include richer output sections",
            "  --fields FIELD,...      Select output fields",
            "  --precision MODE        TOON numeric display precision",
            "  --timeframe TIMEFRAME   Default MT5 timeframe for commands with a",
            "                          timeframe parameter; command-level",
            "                          --timeframe overrides it.",
            "",
            f"Run '{program} <command> --help' for command arguments.",
            f"Run '{program} --help <keyword>' to search detailed command help.",
            "Kebab-case command spellings are also accepted.",
        )
    )
    return "\n".join(lines)


__all__ = ["CLI_COMMAND_NAMES", "available_command_names", "format_root_help"]
