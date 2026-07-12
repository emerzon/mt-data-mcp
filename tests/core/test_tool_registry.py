"""Regression test: verify every MCP tool is registered after bootstrap."""

from mtdata.bootstrap.tools import bootstrap_tools, mcp
from mtdata.core._mcp_tools import registered_tool_catalog
from mtdata.core.tools import tools_list

EXPECTED_TOOL_NAMES = frozenset(
    {
        "causal_discover_signals",
        "cointegration_test",
        "cross_correlation",
        "confluence_levels",
        "correlation_matrix",
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
        "forecast_task_cancel_all",
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
        "outliers_detect",
        "labels_triple_barrier",
        "market_status",
        "market_snapshot",
        "market_ticker",
        "market_microstructure_analyze",
        "market_relative_strength",
        "news",
        "options_barrier_price",
        "options_chain",
        "options_expirations",
        "options_heston_calibrate",
        "options_provider_status",
        "patterns_detect",
        "pivot_compute_points",
        "portfolio_risk_decompose",
        "regime_detect",
        "report_generate",
        "strategy_backtest",
        "strategy_validate",
        "stationarity_test",
        "support_resistance_levels",
        "symbols_describe",
        "symbols_list",
        "symbols_top_markets",
        "market_scan",
        "temporal_analyze",
        "seasonality_detect",
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
        "trade_var_cvar_calculate",
        "trade_session_context",
        "trade_stress_test",
        "tools_list",
        "volume_profile_levels",
        "volatility_term_structure",
        "wait_event",
    }
)

_MCP_TOP_LEVEL_FORBIDDEN_SCHEMA_KEYS = frozenset(
    {"oneOf", "anyOf", "allOf", "enum", "not"}
)


def test_tool_count_matches_snapshot():
    bootstrap_tools()
    registered = {t.name for t in mcp._tool_manager.list_tools()}
    expected = set(EXPECTED_TOOL_NAMES)
    if "market_depth_fetch" in registered:
        expected.add("market_depth_fetch")
    missing = expected - registered
    extra = registered - expected
    assert not missing, f"Tools disappeared: {sorted(missing)}"
    assert not extra, f"New tools not in snapshot (update EXPECTED_TOOL_NAMES): {sorted(extra)}"
    assert len(registered) == len(expected)


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


def test_tools_catalog_standard_detail_includes_parameter_summaries():
    bootstrap_tools()

    compact = registered_tool_catalog(detail="compact")
    standard = registered_tool_catalog(detail="standard")
    full = registered_tool_catalog(detail="full")

    compact_market_scan = next(row for row in compact["tools"] if row["name"] == "market_scan")
    standard_market_scan = next(row for row in standard["tools"] if row["name"] == "market_scan")
    full_market_scan = next(row for row in full["tools"] if row["name"] == "market_scan")

    assert compact["detail"] == "compact"
    assert standard["detail"] == "standard"
    assert full["detail"] == "full"
    assert "parameters" not in compact_market_scan
    assert standard_market_scan["parameters"]["timeframe"] == "optional"
    assert "module" not in standard_market_scan
    assert full_market_scan["parameters"]["timeframe"] == "optional"
    assert full_market_scan["module"] == "mtdata.core.symbols"


def test_tools_list_filters_and_paginates_rows():
    bootstrap_tools()
    raw_tools_list = getattr(tools_list, "__wrapped__", tools_list)

    out = raw_tools_list(category="forecast", limit=3, offset=1)

    assert out["success"] is True
    assert out["filters"] == {"category": "forecast", "search": None}
    assert out["count"] == 3
    assert out["total_count"] > out["count"]
    assert out["offset"] == 1
    assert out["limit"] == 3
    assert out["has_more"] is True
    assert out["pagination"] == {
        "total": out["total_count"],
        "returned": 3,
        "offset": 1,
        "limit": 3,
        "has_more": True,
        "more_available": out["total_count"] - 4,
    }
    assert all(row["category"] == "forecast" for row in out["tools"])


def test_tools_list_compact_keeps_rows_slim_by_default(monkeypatch):
    monkeypatch.delenv("MTDATA_ENABLE_MARKET_DEPTH_FETCH", raising=False)
    bootstrap_tools()
    raw_tools_list = getattr(tools_list, "__wrapped__", tools_list)

    out = raw_tools_list(category="market", search="depth")

    assert out["success"] is True
    row = next(row for row in out["tools"] if row["name"] == "market_depth_fetch")
    assert set(row) == {"name", "category", "description"}
    assert out["gated_tools"] == [
        {
            "enabled": False,
            "enable_env": "MTDATA_ENABLE_MARKET_DEPTH_FETCH",
            "status": "disabled",
            "why_disabled": "Requires broker Level 2/DOM support and is off by default.",
            "recommended_alternative": "market_ticker",
            "name": "market_depth_fetch",
        }
    ]


def test_tools_list_related_tools_are_opt_in():
    bootstrap_tools()
    raw_tools_list = getattr(tools_list, "__wrapped__", tools_list)

    hidden = raw_tools_list(search="data_fetch_candles")
    shown = raw_tools_list(search="data_fetch_candles", include_related=True)

    assert "related_tools" not in hidden["tools"][0]
    assert shown["tools"][0]["related_tools"]
