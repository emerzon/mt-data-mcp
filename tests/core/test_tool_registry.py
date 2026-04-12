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
        "forecast_options_chain",
        "forecast_options_expirations",
        "forecast_quantlib_barrier_price",
        "forecast_quantlib_heston_calibrate",
        "forecast_tune_genetic",
        "forecast_tune_optuna",
        "forecast_volatility_estimate",
        "indicators_describe",
        "indicators_list",
        "labels_triple_barrier",
        "market_status",
        "market_ticker",
        "news",
        "patterns_detect",
        "pivot_compute_points",
        "regime_detect",
        "report_generate",
        "support_resistance_levels",
        "symbols_describe",
        "symbols_list",
        "symbols_top_markets",
        "temporal_analyze",
        "trade_account_info",
        "trade_close",
        "trade_get_open",
        "trade_get_pending",
        "trade_history",
        "trade_modify",
        "trade_place",
        "trade_risk_analyze",
        "trade_session_context",
        "wait_event",
    }
)


def test_tool_count_matches_snapshot():
    bootstrap_tools()
    registered = {t.name for t in mcp._tool_manager.list_tools()}
    missing = EXPECTED_TOOL_NAMES - registered
    extra = registered - EXPECTED_TOOL_NAMES
    assert not missing, f"Tools disappeared: {sorted(missing)}"
    assert not extra, f"New tools not in snapshot (update EXPECTED_TOOL_NAMES): {sorted(extra)}"
    assert len(registered) == len(EXPECTED_TOOL_NAMES)
