"""Regression test: verify every MCP tool is registered after bootstrap."""

from mtdata.bootstrap.tools import bootstrap_tools, mcp

EXPECTED_TOOL_NAMES = frozenset(
    {
        "causal_discover_signals",
        "cointegration_test",
        "correlation_matrix",
        "data_fetch_candles",
        "data_fetch_ticks",
        "finviz_calendar",
        "finviz_crypto",
        "finviz_description",
        "finviz_earnings",
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
        "forecast_models_delete",
        "forecast_models_list",
        "forecast_optimize_hints",
        "forecast_task_cancel",
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
        "market_status",
        "market_ticker",
        "news",
        "options_barrier_price",
        "options_chain",
        "options_expirations",
        "options_heston_calibrate",
        "patterns_detect",
        "pivot_compute_points",
        "regime_detect",
        "report_generate",
        "strategy_backtest",
        "support_resistance_levels",
        "symbols_describe",
        "symbols_list",
        "symbols_top_markets",
        "market_scan",
        "temporal_analyze",
        "trade_account_info",
        "trade_close",
        "trade_get_open",
        "trade_get_pending",
        "trade_history",
        "trade_journal_analyze",
        "trade_modify",
        "trade_place",
        "trade_risk_analyze",
        "trade_var_cvar_calculate",
        "trade_session_context",
        "wait_event",
    }
)

_MCP_TOP_LEVEL_FORBIDDEN_SCHEMA_KEYS = frozenset(
    {"oneOf", "anyOf", "allOf", "enum", "not"}
)


def test_tool_count_matches_snapshot():
    bootstrap_tools()
    registered = {t.name for t in mcp._tool_manager.list_tools()}
    missing = EXPECTED_TOOL_NAMES - registered
    extra = registered - EXPECTED_TOOL_NAMES
    assert not missing, f"Tools disappeared: {sorted(missing)}"
    assert not extra, f"New tools not in snapshot (update EXPECTED_TOOL_NAMES): {sorted(extra)}"
    assert len(registered) == len(EXPECTED_TOOL_NAMES)


def test_tool_public_schemas_match_mcp_top_level_subset():
    bootstrap_tools()
    tool_map = getattr(getattr(mcp, "_tool_manager", None), "_tools", None)
    assert isinstance(tool_map, dict) and tool_map

    issues: list[str] = []
    for name, tool in sorted(tool_map.items()):
        parameters = getattr(tool, "parameters", None)
        if not isinstance(parameters, dict):
            issues.append(f"{name}: parameters is {type(parameters).__name__}")
            continue

        if parameters.get("type") != "object":
            issues.append(
                f"{name}: top-level type is {parameters.get('type')!r}, expected 'object'"
            )

        forbidden_keys = sorted(
            key for key in _MCP_TOP_LEVEL_FORBIDDEN_SCHEMA_KEYS if key in parameters
        )
        if forbidden_keys:
            issues.append(
                f"{name}: forbidden top-level schema keys present: {forbidden_keys}"
            )

    assert not issues, "Invalid MCP tool schemas:\n" + "\n".join(issues)
