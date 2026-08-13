import argparse
import inspect
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

from ....utils.coercion import UNPARSED_BOOL, parse_bool_like

ToolInfo = Dict[str, Any]


_OPTIONAL_POSITIONAL_PARAMS: set[tuple[str, str]] = {
    ("finviz_forex", "symbol"),
    ("finviz_news", "symbol"),
    ("news", "symbol"),
    ("correlation_matrix", "symbols"),
    ("cointegration_test", "symbols"),
    ("market_relative_strength", "symbols"),
    ("market_scan", "symbols"),
    ("causal_discover_signals", "symbols"),
    ("market_status", "symbol"),
    ("trade_close", "symbol"),
    ("trade_execution_quality", "symbol"),
    ("trade_get_open", "symbol"),
    ("trade_get_pending", "symbol"),
    ("trade_place", "symbol"),
    ("trade_risk_analyze", "symbol"),
    ("trade_var_cvar_calculate", "symbol"),
    ("forecast_list_library_models", "library"),
    ("wait_event", "symbol"),
}

_HIDDEN_OPTIONAL_POSITIONAL_FLAGS: set[tuple[str, str]] = {
    ("correlation_matrix", "symbols"),
    ("cointegration_test", "symbols"),
    ("causal_discover_signals", "symbols"),
}

# Choice discovery comes from the same Literal/Pydantic annotations used to
# build public MCP schemas. Keep this map only for exceptional transport-only
# compatibility cases.
_COMMAND_PARAM_CHOICE_OVERRIDES: Dict[tuple[str, str], list[str]] = {
    ("temporal_analyze", "group_by"): [
        "dow",
        "day_of_week",
        "hour",
        "month",
        "session",
        "all",
    ],
}

_POSITIONAL_ONLY_OPTIONAL_PARAMS: set[tuple[str, str]] = set()

_SEARCH_ALIAS_COMMANDS = frozenset(
    {
        "finviz_filters_list",
        "forecast_list_methods",
        "indicators_list",
        "symbols_list",
        "tools_list",
    }
)

_OPTION_ALIAS_DEST_PREFIX = "_cli_option_"

_MULTI_VALUE_SYMBOL_POSITIONAL_COMMANDS = frozenset(
    {
        "causal_discover_signals",
        "correlation_matrix",
        "cointegration_test",
        "cross_correlation",
        "market_relative_strength",
    }
)

_COMMAND_REQUIRED_OPTIONS: set[tuple[str, str]] = {
    ("trade_modify", "ticket"),
    ("trade_stress_test", "shocks"),
}

_NAMED_ONLY_REQUIRED_PARAMS: set[tuple[str, str]] = {
    ("trade_modify", "ticket"),
    ("trade_stress_test", "shocks"),
}

_PRESERVE_OMITTED_DEFAULT_PARAMS: set[tuple[str, str]] = {
    ("data_fetch_candles", "limit"),
}

_COMMAND_PARAM_HELP_OVERRIDES: Dict[tuple[str, str], str] = {
    ("forecast_train", "lookback"): (
        "Maximum historical bars to use for training after applying the requested "
        "time window."
    ),
    ("forecast_train", "as_of"): (
        "Train on closed bars available at this historical reference time. Cannot "
        "be combined with --start/--end."
    ),
    ("forecast_train", "start"): (
        "Optional start of the historical training range. Cannot be combined "
        "with --as-of."
    ),
    ("forecast_train", "end"): (
        "Optional end of the historical training range. Cannot be combined "
        "with --as-of."
    ),
    ("forecast_train", "quantity"): (
        "Train a price-level or return target. Volatility uses "
        "forecast_volatility_estimate and is not separately trainable."
    ),
    ("forecast_train", "wait"): (
        "Wait for training to finish. One-shot CLI and stdin shell batches wait "
        "by default; interactive shell, MCP, and Web API calls submit in the "
        "background unless wait is true."
    ),
    ("data_fetch_candles", "timestamp_format"): (
        "Format each candle's `time` value: iso for the UTC bar-open timestamp "
        "or epoch for UTC epoch seconds."
    ),
    ("data_fetch_ticks", "timestamp_format"): (
        "Format each MT5 tick event's `time` value as an ISO UTC timestamp or "
        "UTC epoch seconds."
    ),
    ("market_ticker", "price_field"): (
        "Omit for the default bid/ask/spread quote snapshot; set bid, ask, mid, "
        "last, or spread for a single-price response."
    ),
    ("patterns_detect", "timeframe"): (
        "Chart timeframe. When omitted, candlestick/classic/harmonic/fractal use "
        "H1, elliott scans H1/H4/D1, and all scans M30/H1/H4/D1/W1."
    ),
    ("symbols_list", "universe"): (
        "Symbol scan universe. When omitted, unfiltered listings use visible "
        "Market Watch symbols while searches use the full broker catalog."
    ),
    ("volume_profile_levels", "source"): (
        "Profile input. auto uses bounded raw ticks when coverage is adequate, "
        "then falls back to the labeled M1-bar approximation for oversized "
        "windows, failed tick fetches, or poor tick-price coverage."
    ),
    ("forecast_list_library_models", "limit"): (
        "Maximum models to return on this page. Omitted compact output uses "
        "20; omitted full output is unbounded."
    ),
    ("correlation_matrix", "method"): "Correlation coefficient: pearson or spearman.",
    ("correlation_matrix", "transform"): (
        "Price transform: log_return, pct, diff, level, or log_level."
    ),
    ("cross_correlation", "method"): "Correlation coefficient: pearson or spearman.",
    ("cross_correlation", "transform"): (
        "Price transform: log_return, pct, diff, level, or log_level."
    ),
    ("stationarity_test", "tests"): (
        "Comma-separated stationarity tests: adf, kpss, pp. "
        "Example: --tests adf,kpss."
    ),
    ("denoise_describe", "method"): (
        "Denoise method to describe. Run denoise_list_methods to list methods "
        "available in this installation."
    ),
    ("trade_var_cvar_calculate", "method"): (
        "Tail-risk method: historical or parametric."
    ),
    ("trade_var_cvar_calculate", "symbol"): (
        "Optional scope: calculate VaR/CVaR for currently open positions in this "
        "symbol. Omit it for the full open portfolio."
    ),
    ("trade_var_cvar_calculate", "transform"): (
        "Return transform: log_return (aliases log_returns/log) or pct "
        "(aliases pct_return/percent/simple_return)."
    ),
    ("data_fetch_candles", "indicators"): "Technical indicators. On PowerShell, quote parenthesized specs such as --indicators \"rsi(14)\", or use shell-safe rsi_14 / sma=20 syntax. JSON arrays like '[{\"name\":\"rsi\",\"params\":[14]}]' and named params like rsi(length=14) also work. Use params syntax, not sma,20.",
    ("data_fetch_candles", "limit"): (
        "Maximum returned bars (default: 20). Queries with --start retain the "
        "earliest matching bars (first-N); otherwise the latest bars are retained. "
        "On an explicit range, omission uses a 100000-bar safety cap. Indicator "
        "warmup bars are fetched in addition to returned rows."
    ),
    ("data_fetch_candles", "start"): (
        "Inclusive range start. Intraday date-only and calendar phrases use UTC. "
        "For D1/W1/MN1 they select broker-session calendar periods and resolve "
        "from broker-local midnight. Adding --limit retains the first N bars."
    ),
    ("data_fetch_candles", "end"): (
        "Inclusive range end. Intraday date-only and calendar phrases end in UTC; "
        "for D1/W1/MN1 they end at the broker-local calendar-period boundary."
    ),
    ("data_fetch_candles", "include_incomplete"): (
        "Include the latest forming candle; defaults to false. Compact responses "
        "expose forming_candle_status=skipped and an inclusion hint when a forming "
        "bar is omitted; full detail also includes counts and booleans."
    ),
    ("data_fetch_ticks", "limit"): (
        "Maximum ticks returned (default 20, maximum 50000). Queries with a "
        "start bound return the earliest matching ticks; otherwise the latest ticks."
    ),
    ("market_status", "symbol"): (
        "Broker symbol for MT5 session/tradability status. If omitted, the "
        "command returns a static major-equity-exchange calendar, not the "
        "connected broker book."
    ),
    ("forecast_task_cancel_all", "status_filter"): (
        "Cancelable task status: all, pending, or running. Defaults to all active tasks."
    ),
    ("indicators_list", "trading_style"): (
        "Filter by broad workflow tags (intraday, swing, or position). Many tags "
        "are category heuristics, not indicator-specific recommendations."
    ),
    ("trade_place", "magic"): "MT5 magic number: integer strategy/order identifier used to group EA or strategy trades. Defaults to configured order_magic when omitted.",
    ("trade_get_open", "magic"): "MT5 magic number filter for positions from one strategy or EA. Omit for all magic numbers.",
    ("trade_get_pending", "magic"): "MT5 magic number filter for pending orders from one strategy or EA. Omit for all magic numbers.",
    ("trade_close", "magic"): "Standalone strategy scope for matching objects in the selected target class. Omit for all magic numbers.",
    ("wait_event", "magic"): "MT5 magic number filter for account events from one strategy or EA. Omit for all magic numbers.",
    ("finviz_screen", "filters"): "Filter key=value pairs, operator aliases like beta_under=1, Finviz shorthand, or JSON object. Examples: 'country=USA,marketcap=mega', 'pe_under=15,beta_under=1', 'cap_largeover,exch_nyse', '{\"Exchange\":\"NASDAQ\",\"Sector\":\"Technology\"}'. Common keys include Exchange, Index, Sector, Industry, Country, Market Cap., P/E, Dividend Yield, RSI (14), Average Volume, and Price.",
    ("finviz_screen", "limit"): "Max screener results to return on this page.",
    ("finviz_screen", "order"): "Finviz sort key. Use --order=-marketcap for descending or --order=price for ascending.",
    ("finviz_news", "limit"): "Max news items to return on this page.",
    ("finviz_insider", "limit"): "Max insider trades to return on this page.",
    ("finviz_insider_activity", "option"): (
        "Insider activity view: latest, latest buys/sales, top week "
        "buys/sales, or top owner trade/buys/sales."
    ),
    ("finviz_calendar", "start"): "Start date (YYYY-MM-DD).",
    ("finviz_calendar", "end"): "End date (YYYY-MM-DD).",
    ("forecast_barrier_optimize", "method"): "Barrier simulation method: mc_gbm, mc_gbm_bb, hmm_mc, garch, bootstrap, heston, jump_diffusion, or auto.",
    ("forecast_barrier_prob", "barrier"): (
        'Barrier object. Use {"kind":"single_price","level":1.1000} for '
        'closed_form, or {"kind":"tp_sl","unit":"pct","take_profit":0.2,'
        '"stop_loss":0.1} for simulation methods. The kind may be omitted '
        "from a complete TP/SL object."
    ),
    ("forecast_barrier_prob", "mu"): (
        "Annual log-return drift override (decimal fraction) on the shared "
        "symbol/timeframe annualization basis."
    ),
    ("forecast_barrier_prob", "sigma"): (
        "Annual return-volatility override (decimal fraction) on the shared "
        "symbol/timeframe annualization basis."
    ),
    ("forecast_volatility_estimate", "method"): (
        "Volatility estimator, such as ewma, rolling_std, har_rv, garch, "
        "arima, theta, or ensemble. Run forecast_list_methods with "
        "--detail standard --search-term NAME to browse the full namespace."
    ),
    ("volatility_term_structure", "horizons"): (
        "Comma-separated realized-volatility horizons in bars, for example 1,5,20."
    ),
    ("market_relative_strength", "horizons"): (
        "Comma-separated ranking horizons in bars; values align one-to-one with --weights."
    ),
    ("market_relative_strength", "weights"): (
        "Comma-separated non-negative ranking weights matching --horizons; normalized "
        "to sum to 1."
    ),
    ("market_relative_strength", "limit"): (
        "Maximum distinct ranked symbols returned across strongest and weakest "
        "tails; the stronger tail receives the extra row for odd limits."
    ),
    ("options_chain", "limit"): "Max option contracts to return.",
    ("options_heston_calibrate", "valuation_date"): (
        "Valuation date in YYYY-MM-DD format; omit for the current UTC date."
    ),
    ("options_barrier_price", "valuation_date"): (
        "Valuation date in YYYY-MM-DD format; omit for the current UTC date."
    ),
    ("volume_profile_levels", "lookback"): (
        "Historical bar count for a timeframe-based profile; requires --timeframe."
    ),
    ("outliers_detect", "limit"): "Max anomalous bars to return.",
    ("temporal_analyze", "limit"): (
        "Max grouped time buckets to return; pagination only, not the analysis window."
    ),
    ("temporal_analyze", "session_calendar"): "Session calendar: auto, fx, or equity.",
    ("seasonality_detect", "max_period"): (
        "Maximum candidate seasonal period in bars; defaults from available samples and "
        "--min-cycles."
    ),
    ("causal_discover_signals", "symbols"): (
        "Comma- or space-separated MT5 symbols (e.g. EURUSD,GBPUSD or "
        "EURUSD GBPUSD); one symbol auto-expands to its MT5 group. Optional "
        "with --group."
    ),
    ("trade_execution_quality", "side"): "Execution fill side filter: buy or sell.",
    ("trade_history", "side"): (
        "For deals, buy/sell filters fill_side and long/short filters "
        "position_side. Order history accepts buy/sell only."
    ),
    ("trade_journal_analyze", "side"): (
        "buy/sell filters exit-fill direction; long/short filters realized "
        "position direction."
    ),
    ("trade_execution_quality", "min_sample"): (
        "Minimum eligible fills required for sufficient execution-quality evidence."
    ),
    ("trade_history", "column_style"): (
        "Trade-history field naming: snake_case or humanized."
    ),
    ("market_microstructure_analyze", "max_ticks"): (
        "Maximum raw ticks retained for microstructure analysis."
    ),
    ("options_barrier_price", "option_type"): "Option side: call or put.",
    ("options_barrier_price", "calendar"): (
        "QuantLib calendar name, such as UnitedStates.NYSE, TARGET, or NullCalendar."
    ),
    ("options_barrier_price", "maturity_basis"): (
        "Interpret maturity_days as calendar_days or business_days in the selected QuantLib calendar."
    ),
    ("options_barrier_price", "barrier"): (
        "Option knock-in/knock-out barrier price level, in the same units as spot "
        "and strike. This is a numeric parametric pricer; it does not fetch a symbol quote."
    ),
    ("strategy_validate", "candidates"): (
        "JSON strategy candidate list. Example: "
        "'[{\"id\":\"cross\",\"type\":\"builtin_strategy\","
        "\"strategy\":\"ema_cross\"}]'. Candidate types are builtin_strategy "
        "and forecast_threshold."
    ),
    ("strategy_validate", "barrier"): (
        "JSON triple-barrier labeling config with horizon, tp_pct, sl_pct, and "
        "same_bar_policy; tp_pct/sl_pct are percentage points (0.5 means 0.5%)."
    ),
    ("options_chain", "symbol"): (
        "Underlying symbol for listed options, e.g. AAPL or SPX."
    ),
    ("options_expirations", "symbol"): (
        "Underlying symbol for listed options, e.g. AAPL or SPX."
    ),
    ("options_heston_calibrate", "symbol"): (
        "Underlying symbol for listed options, e.g. AAPL or SPX."
    ),
    ("options_heston_calibrate", "calendar"): (
        "QuantLib calendar name used by calibration helpers, such as UnitedStates.NYSE or NullCalendar."
    ),
    ("options_heston_calibrate", "maturity_basis"): (
        "Basis for the reported days_to_expiry diagnostic; calibration remains anchored to the contract expiry date."
    ),
    ("options_chain", "expiration"): (
        "Listed option expiration date in YYYY-MM-DD format, e.g. 2026-07-17. "
        "Omit to use the provider's nearest available expiration."
    ),
    ("options_heston_calibrate", "expiration"): (
        "Listed option expiration date in YYYY-MM-DD format, e.g. 2026-07-17. "
        "Omit to use the provider's nearest available expiration."
    ),
    ("forecast_tune_optuna", "search_space"): "Optuna search space (JSON or k=v).",
    ("indicators_list", "detail"): "Output detail: compact table or full rows with aliases and descriptions.",
    ("market_snapshot", "sections"): (
        "Analysis modules to include: quote, status, levels, patterns, regime, "
        "forecast, or all. Defaults to quote,status,levels,patterns."
    ),
    ("market_snapshot", "detail"): (
        "Field verbosity inside selected sections; full does not add sections. "
        "Use --sections all for every snapshot module."
    ),
    ("causal_discover_signals", "limit"): "Max causal link rows to return.",
    ("causal_discover_signals", "window_bars"): (
        "Historical bars per symbol used for causal tests."
    ),
    ("cointegration_test", "symbols"): (
        "Comma- or space-separated MT5 symbols (e.g. EURUSD,GBPUSD or EURUSD GBPUSD); one symbol auto-expands "
        "to its MT5 group. Optional with --group."
    ),
    ("cointegration_test", "limit"): "Max cointegration pair rows to return.",
    ("cointegration_test", "window_bars"): (
        "Historical bars per symbol used for the cointegration test window."
    ),
    ("correlation_matrix", "limit"): "Max correlation pair rows to return.",
    ("correlation_matrix", "window_bars"): (
        "Historical bars per symbol used for the correlation window."
    ),
    ("correlation_matrix", "symbols"): (
        "Comma- or space-separated MT5 symbols (e.g. EURUSD,GBPUSD or EURUSD GBPUSD); one symbol auto-expands "
        "to its MT5 group. Optional with --group."
    ),
    ("cross_correlation", "symbols"): (
        "Comma- or space-separated MT5 symbols (e.g. EURUSD,GBPUSD or EURUSD GBPUSD)."
    ),
    ("market_scan", "symbols"): (
        "Comma-separated MT5 symbols to scan. Optional with --group."
    ),
    ("market_relative_strength", "symbols"): (
        "Comma- or space-separated MT5 symbols to rank (e.g. EURUSD,GBPUSD "
        "or EURUSD GBPUSD). Provide at least two symbols, use --group to rank "
        "an MT5 group, or omit both to rank the visible Market Watch universe. "
        "Use a homogeneous group when comparable peers are required."
    ),
    ("market_scan", "preset"): (
        "Built-in scan preset: oversold, overbought, high-volume, tight-spread, "
        "gap-up, or gap-down. Explicit filter flags override preset defaults."
    ),
    ("market_scan", "rank_order"): (
        "Sort direction for ranked rows: auto, asc/ascending, or desc/descending. "
        "Auto keeps tight spreads and oversold RSI ascending; most other ranks descending."
    ),
    ("market_scan", "quote_usable_only"): (
        "Exclude quotes that are stale, future-dated, locked, inverted, or one-sided. "
        "Defaults to true for spread rankings and the tight-spread preset."
    ),
    ("outliers_detect", "score_fields"): (
        "Comma-separated candle features to score: return, volume, and/or range."
    ),
    ("outliers_detect", "threshold"): (
        "Positive robust-deviation cutoff; 3.5 is a common MAD threshold."
    ),
    ("labels_triple_barrier", "detail"): (
        "Detail level: compact (small outcome sample), standard (recent lookback rows), "
        "summary, or full."
    ),
    ("labels_triple_barrier", "limit"): (
        "Maximum labeled rows for compact/standard output. Compact is capped at "
        "10 and normally shows the recent tail; when that tail is entirely neutral, "
        "it reserves up to two rows for recent resolved TP/SL examples; full returns "
        "the complete labeled series."
    ),
    ("labels_triple_barrier", "lookback"): (
        "Number of labeled entries to calculate; the tool fetches lookback plus "
        "horizon bars."
    ),
    ("labels_triple_barrier", "barriers"): (
        "Required JSON barrier pair. Example: "
        "'{\"kind\":\"tp_sl\",\"unit\":\"pct\",\"take_profit\":0.5,\"stop_loss\":0.5}'. "
        "kind='tp_sl' is optional, so forecast_barrier_prob TP/SL objects can be reused. "
        "pct/ticks are distances from entry; price values are absolute levels."
    ),
    ("labels_triple_barrier", "allow_noncausal_denoise"): (
        "Allow explicitly requested zero-phase denoising. This uses future bars, "
        "sets lookahead_bias=true, and makes labels unsuitable for backtests or training."
    ),
    ("market_scan", "limit"): "Max matching symbols to return.",
    ("news", "limit"): (
        "Global maximum across all news/event buckets. One upcoming event is "
        "reserved when available; use --limit-per-bucket to cap each family separately."
    ),
    ("news", "limit_per_bucket"): (
        "Maximum rows in each news/event family while preserving the separate buckets."
    ),
    ("market_depth_fetch", "require_dom"): "Fail if DOM is unavailable instead of falling back to a quote snapshot.",
    ("patterns_detect", "mode"): "Pattern mode: all, candlestick, classic, harmonic, fractal, or elliott.",
    ("patterns_detect", "engine"): (
        "Classic-mode engine: native or stock_pattern. Omitted classic calls "
        "use native; invalid for other modes."
    ),
    ("report_generate", "template"): (
        "Report template: minimal fast context+forecast (default), basic balanced research, "
        "advanced regimes/HAR/conformal, scalping M5, intraday H1, swing H4/D1, "
        "or position D1/W1. Typical warm runtimes: minimal 3-10s, scalping "
        "15-60s, basic/style templates 30-120s, advanced 60-180s; broker "
        "history and enabled methods can increase them. Use --max-runtime, "
        "--include-sections, or --max-sections to bound work."
    ),
    ("report_generate", "max_runtime"): (
        "Cooperative runtime budget in seconds (1-3600). Sections whose "
        "estimated cost does not fit are omitted, and new sub-tools stop after "
        "the deadline; an active native/MT5 call is allowed to finish safely."
    ),
    ("report_generate", "allow_partial"): (
        "Return success=true when at least one report section is usable while "
        "retaining section_run_status=partial; set false for strict completion."
    ),
    ("report_generate", "progress"): (
        "Write report sub-tool start/finish progress lines to stderr."
    ),
    ("temporal_analyze", "lookback"): (
        "Historical bars used when start/end are omitted. Defaults to a "
        "timeframe-aware seasonal window: 210 days for day-of-week, 60 days "
        "for hour/session, 730 days for month, and 365 days for overall "
        "analysis, bounded to 200-20000 bars (H1 session: 1440 bars)."
    ),
    ("regime_detect", "fetch_limit"): (
        "Historical bars fetched for regime detection. Defaults to the effective "
        "lookback plus warmup bars; use max_regimes for compact output count."
    ),
    ("symbols_list", "limit"): "Max symbols or groups to return.",
    ("symbols_top_markets", "rank_by"): (
        "Leaderboard to compute: abs_price_change_pct (default), all, "
        "spread/spread_pct, tick_volume, price_change/price_change_pct, "
        "or abs_price_change/abs_price_change_pct."
    ),
    ("symbols_top_markets", "limit"): (
        "Max symbols for the selected ranking; per leaderboard when rank_by=all."
    ),
    ("symbols_top_markets", "candidate_limit"): (
        "Candidate partition size (1-250) for a large sorted group/category universe. "
        "Merge each partition's top-N rows to obtain the global leaderboard."
    ),
    ("symbols_top_markets", "candidate_offset"): (
        "Zero-based offset into the deterministic sorted candidate universe. Increment "
        "by candidate_limit until candidate_page.has_more is false."
    ),
    ("trade_close", "close_all"): (
        "Select the whole account when ticket, symbol, and magic are omitted."
    ),
    ("trade_close", "target"): (
        "Object class: positions, pending, or all_exposure. The default positions "
        "target never cancels pending orders."
    ),
    ("trade_close", "confirm_close_all"): (
        "Confirm any live ticketless bulk operation, including symbol and magic scopes."
    ),
    ("trade_close", "dry_run"): (
        "Preview the close request without sending it to the broker."
    ),
    ("trade_close", "pnl_filter"): (
        "Position P&L filter: all, profit, or loss."
    ),
    ("trade_close", "close_priority"): (
        "When multiple positions match, close loss_first, profit_first, or largest_first."
    ),
    ("trade_modify", "dry_run"): (
        "Preview the modification without sending it to the broker."
    ),
    ("trade_modify", "price"): (
        "New pending-order price. Omit when only stop_loss/take_profit change."
    ),
    ("trade_get_pending", "order_type"): (
        "Pending-order filter: BUY_LIMIT, BUY_STOP, BUY_STOP_LIMIT, SELL_LIMIT, "
        "SELL_STOP, or SELL_STOP_LIMIT."
    ),
    ("trade_modify", "idempotency_key"): (
        "Durable dedupe key shared by CLI and server processes. Reusing the same "
        "key and payload within the retention window replays the prior outcome."
    ),
    ("trade_place", "idempotency_key"): (
        "Durable dedupe key shared by CLI and server processes. Reusing the same "
        "key and payload within the retention window replays the prior outcome."
    ),
    ("trade_place", "dry_run"): (
        "Preview the order without sending it to the broker."
    ),
    ("trade_place", "detail"): (
        "Dry-run preview detail: compact for key checks, full for execution diagnostics."
    ),
    ("trade_stress_test", "shocks"): (
        "JSON object mapping symbols to percentage shocks. Examples: "
        "'{\"*\":-2}' or '{\"EURUSD\":-1,\"XAUUSD\":-3}'."
    ),
    ("trade_place", "require_sl_tp"): (
        "Require both stop_loss and take_profit for market orders."
    ),
    ("trade_history", "minutes_back"): (
        "History lookback in minutes. Defaults to 10080 minutes (7 days) when "
        "start/end and minutes_back are omitted."
    ),
    ("trade_journal_analyze", "minutes_back"): (
        "Journal history lookback in minutes. Defaults to 10080 minutes (7 days) "
        "when start/end and minutes_back are omitted."
    ),
    ("trade_journal_analyze", "limit"): (
        "Maximum realized exit deals to analyze. The command pages through raw "
        "history rows as needed (default 50)."
    ),
    ("trade_execution_quality", "minutes_back"): (
        "Execution-history lookback in minutes (default 43200 = 30 days)."
    ),
    ("trade_execution_quality", "limit"): (
        "Maximum eligible fills to analyze (default 200)."
    ),
    ("trade_modify", "expiration"): "Pending order expiration time (dateparser string, UTC epoch seconds, or GTC token).",
    ("trade_place", "expiration"): "Pending order expiration time (dateparser string, UTC epoch seconds, or GTC token).",
    ("wait_event", "symbol"): (
        "Single trading symbol (e.g. EURUSD). Cannot be combined with symbols. "
        "Omit both only for timeframe clock-boundary mode; duration mode "
        "requires symbol or symbols."
    ),
    ("wait_event", "symbols"): (
        "Basket of 1-12 trading symbols. Cannot be combined with symbol; omitted-symbol "
        "watchers apply to every basket member."
    ),
    ("wait_event", "timeframe"): (
        "Candle-boundary wait mode. Cannot be combined with max_wait_seconds. "
        "With inferred watchers, reaching the boundary is a successful completion."
    ),
    ("wait_event", "max_wait_seconds"): (
        "Duration wait mode in seconds. Cannot be combined with timeframe or end_on. "
        "With inferred watchers, elapsed duration is a successful completion."
    ),
    ("wait_event", "poll_interval_seconds"): (
        "Seconds between polls; must be at least 0.1. Omit to use 0.5."
    ),
    ("wait_event", "watch_tick_count_spike"): (
        "Include the inferred tick-count-spike watcher. Ignored with explicit watch_for."
    ),
    ("wait_event", "watch_for"): (
        "Event names or event objects. Examples: order_filled, "
        "'{\"type\":\"order_filled\",\"symbol\":\"EURUSD\"}'. "
        "Put candle_close boundaries in end_on. Omit for the lightweight core "
        "order/position and market-activity watcher set; generated S/R and pivot "
        "zones are not inferred. Explicit watchers make an unmatched timeout or "
        "boundary a failed wait."
    ),
    ("wait_event", "end_on"): (
        "Optional timeframe-mode boundaries. Explicit boundary timeframes must "
        "match the top-level timeframe."
    ),
}

_VOLATILITY_METHOD_LITERAL_MARKERS = {
    "ewma",
    "parkinson",
    "gk",
    "rs",
    "yang_zhang",
    "rolling_std",
    "realized_kernel",
    "har_rv",
    "garch_t",
    "egarch_t",
    "gjr_garch_t",
    "figarch",
}

_FORECAST_METHOD_LITERAL_MARKERS = {
    "theta",
    "naive",
    "arima",
    "chronos2",
    "statsforecast",
}


def _parse_cli_bool_value(value: Any) -> str:
    """Accept the shared bool vocabulary and return argparse's canonical token."""
    parsed = parse_bool_like(value)
    if parsed is UNPARSED_BOOL:
        raise argparse.ArgumentTypeError(
            "expected true/false, 1/0, yes/no, or on/off"
        )
    return "true" if bool(parsed) else "false"


def _case_insensitive_choice_parser(choices: Sequence[str]) -> Callable[[Any], str]:
    canonical = [str(choice) for choice in choices]
    folded: Dict[str, Optional[str]] = {}
    for choice in canonical:
        key = choice.casefold()
        folded[key] = choice if key not in folded else None

    def _parse(value: Any) -> str:
        text = str(value or "").strip()
        if text in canonical:
            return text
        return folded.get(text.casefold()) or text

    return _parse


def _is_forecast_method_literal(
    ptype: Any,
    *,
    is_literal_origin: Callable[[Any], bool],
    get_origin_func: Callable[[Any], Any],
    get_args_func: Callable[[Any], Tuple[Any, ...]],
) -> bool:
    try:
        origin = get_origin_func(ptype)
        if not is_literal_origin(origin):
            return False
        args = {str(v) for v in get_args_func(ptype) if v is not None}
        if args.intersection(_VOLATILITY_METHOD_LITERAL_MARKERS):
            return False
        return bool(args.intersection(_FORECAST_METHOD_LITERAL_MARKERS))
    except Exception:
        return False


def _dedupe_flags(*flags: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(flag for flag in flags if flag))


def _canonicalize_long_option(flag: str) -> str:
    text = str(flag or "").strip()
    if not text.startswith("--"):
        return text
    if "=" in text:
        option, value = text.split("=", 1)
        return f"{option.replace('_', '-')}={value}"
    return text.replace("_", "-")


def _split_visible_and_hidden_flags(*flags: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    visible: list[str] = []
    hidden: list[str] = []
    for flag in _dedupe_flags(*flags):
        canonical = _canonicalize_long_option(flag)
        if canonical and canonical not in visible:
            visible.append(canonical)
        if flag != canonical and flag not in hidden:
            hidden.append(flag)
    return tuple(visible), tuple(hidden)


def should_expose_cli_param(*, cmd_name: Optional[str], param_name: str) -> bool:
    """Return whether a function parameter should surface as a user CLI argument."""
    if str(cmd_name or "") == "finviz_calendar" and str(param_name or "") in {"date_from", "date_to"}:
        return False
    if str(cmd_name or "") == "wait_event" and str(param_name or "") == "instrument":
        return False
    return True


def get_function_info(
    func: Any,
    *,
    schema_get_function_info: Callable[[Any], Dict[str, Any]],
    flatten_request_model_param: Callable[[Dict[str, Any]], Dict[str, Any]],
) -> Dict[str, Any]:
    """Attach the underlying callable to schema introspection data."""
    info = schema_get_function_info(func)
    info["func"] = func
    info = flatten_request_model_param(info)
    if not info.get("doc"):
        info["doc"] = f"Execute {info.get('name') or getattr(func, '__name__', 'function')}"
    for param in info.get("params", []):
        if param.get("type") is None:
            param["type"] = str
        if "required" not in param:
            param["required"] = param.get("default") is None
    return info


def apply_schema_overrides(
    tool: ToolInfo,
    func_info: Dict[str, Any],
    *,
    enrich_schema_with_shared_defs: Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]],
) -> Dict[str, Any]:
    """Apply JSON schema defaults and required flags to CLI parameter metadata."""
    meta = tool.setdefault("meta", {})
    schema = meta.get("schema") or {}
    schema = enrich_schema_with_shared_defs(schema, func_info)
    meta["schema"] = schema
    params_obj = schema.get("parameters") if isinstance(schema.get("parameters"), dict) else schema
    schema_props = params_obj.get("properties") if isinstance(params_obj, dict) else {}
    schema_required = set(params_obj.get("required", [])) if isinstance(params_obj, dict) else set()
    for param in func_info.get("params", []):
        prop = schema_props.get(param["name"]) if isinstance(schema_props, dict) else None
        if isinstance(prop, dict) and "default" in prop and param.get("default") is None:
            param["default"] = prop["default"]
        if param["name"] in schema_required:
            param["required"] = True
    return schema


def extract_function_from_tool_obj(tool_obj: Any) -> Any:
    """Best-effort extraction of the underlying function from an MCP tool object."""
    for attr in ("func", "function", "callable", "handler", "wrapped", "_func"):
        if hasattr(tool_obj, attr) and callable(getattr(tool_obj, attr)):
            return getattr(tool_obj, attr)
    if callable(tool_obj):
        return tool_obj
    return None


def extract_metadata_from_tool_obj(tool_obj: Any) -> Dict[str, Any]:
    """Extract tool descriptions and per-parameter docs from registry objects."""
    meta: Dict[str, Any] = {"description": None, "param_docs": {}, "schema": None}

    for attr in ("description", "doc", "docs"):
        val = getattr(tool_obj, attr, None)
        if isinstance(val, str) and val.strip():
            meta["description"] = val.strip()
            break

    schema = None
    for attr in ("schema", "input_schema", "parameters", "spec"):
        val = getattr(tool_obj, attr, None)
        if isinstance(val, dict) and val:
            schema = val
            break

    if schema:
        meta["schema"] = schema
        if not meta["description"] and isinstance(schema.get("description"), str):
            meta["description"] = schema.get("description")
        params_obj = schema.get("parameters") if isinstance(schema.get("parameters"), dict) else schema
        props = params_obj.get("properties") if isinstance(params_obj, dict) else None
        if isinstance(props, dict):
            for pname, pdef in props.items():
                desc = pdef.get("description") if isinstance(pdef, dict) else None
                if isinstance(desc, str) and desc.strip():
                    meta["param_docs"][pname] = desc.strip()

    return meta


def discover_tools(
    *,
    bootstrap_tools: Callable[[], Tuple[Any, ...]],
    get_registered_tools: Callable[[], Any],
    mcp: Any,
    get_mcp_registry: Callable[[Any], Any],
    debug: Callable[[str], None],
    extract_function_from_tool_obj: Callable[[Any], Any],
    extract_metadata_from_tool_obj: Callable[[Any], Dict[str, Any]],
    errors: Optional[list[str]] = None,
) -> Dict[str, ToolInfo]:
    """Discover CLI-visible tools from the bootstrap and MCP registries."""
    tools: Dict[str, ToolInfo] = {}

    def _module_is_visible(module_name: Any, allowed_modules: set[str], allowed_prefixes: tuple[str, ...]) -> bool:
        if not isinstance(module_name, str):
            return False
        if module_name in allowed_modules:
            return True
        return any(module_name.startswith(prefix) for prefix in allowed_prefixes)

    registry = None
    bootstrapped_modules: Tuple[Any, ...] = ()
    try:
        bootstrapped_modules = tuple(bootstrap_tools())
    except Exception as exc:
        message = f"bootstrap_tools failed: {exc}"
        debug(message)
        if errors is not None:
            errors.append(message)
    try:
        reg = get_registered_tools()
        if reg and hasattr(reg, "items"):
            registry = reg
    except Exception as exc:
        message = f"get_registered_tools failed: {exc}"
        debug(message)
        if errors is not None:
            errors.append(message)
    if mcp is not None:
        try:
            registry = get_mcp_registry(mcp) or registry
        except Exception as exc:
            message = f"get_mcp_registry failed: {exc}"
            debug(message)
            if errors is not None:
                errors.append(message)

    module_names = {
        str(getattr(module, "__name__", "")).strip()
        for module in bootstrapped_modules
        if getattr(module, "__name__", None)
    }
    module_prefixes = tuple(
        f"{module_name.rsplit('.', 1)[0]}."
        for module_name in module_names
        if "." in module_name
    )
    if registry and hasattr(registry, "items"):
        for name, obj in registry.items():
            func = extract_function_from_tool_obj(obj)
            mod = getattr(func, "__module__", None) if func else None
            if func and (not module_names or _module_is_visible(mod, module_names, module_prefixes)):
                meta = extract_metadata_from_tool_obj(obj)
                tools[name] = {"func": func, "meta": meta}

    if tools:
        return tools

    for module in bootstrapped_modules:
        module_name = getattr(module, "__name__", None)
        if not isinstance(module_name, str):
            continue
        for name in dir(module):
            if name.startswith("_"):
                continue
            obj = getattr(module, name)
            if callable(obj) and getattr(obj, "__module__", None) == module_name:
                try:
                    inspect.signature(obj)
                except (TypeError, ValueError):
                    continue
                if isinstance(obj, type):
                    continue
                if name.endswith(("_wrapper",)):
                    continue
                tools[name] = {"func": obj, "meta": {"description": None, "param_docs": {}}}

    return tools


def resolve_param_kwargs(
    param: Dict[str, Any],
    param_docs: Optional[Dict[str, str]],
    *,
    cmd_name: Optional[str],
    param_names: Optional[set],
    param_hints: Dict[str, str],
    debug: Callable[[str], None],
    is_literal_origin: Callable[[Any], bool],
    unwrap_optional_type: Callable[[Any], Tuple[Any, Any]],
    is_typed_dict_type: Callable[[Any], bool],
    get_origin: Callable[[Any], Any],
    get_args: Callable[[Any], Tuple[Any, ...]],
    is_mapping_annotation: Callable[[Any], bool],
) -> Tuple[Dict[str, Any], bool]:
    """Resolve argparse kwargs for a single CLI parameter."""

    def _is_model_type(value: Any) -> bool:
        return isinstance(value, type) and (
            callable(getattr(value, "model_validate", None))
            or callable(getattr(value, "parse_obj", None))
        )

    def _escape_argparse_help(text: Optional[str]) -> Optional[str]:
        return text.replace("%", "%%") if isinstance(text, str) else text

    desc = None
    if param_docs and param["name"] in param_docs:
        desc = param_docs[param["name"]]
    hint = desc or param_hints.get(param["name"])
    override_help = _COMMAND_PARAM_HELP_OVERRIDES.get((str(cmd_name or ""), str(param["name"])))
    if override_help:
        hint = override_help
    fallback_help = f"Value for {str(param['name']).replace('_', ' ')}."
    kwargs = {"help": _escape_argparse_help(hint) or fallback_help, "dest": param["name"]}
    is_mapping_type = False

    if param["name"] == "method" and (
        (cmd_name in {"forecast_generate", "forecast_conformal_intervals", "forecast_tune_genetic", "forecast_tune_optuna"})
        or _is_forecast_method_literal(
            param.get("type"),
            is_literal_origin=is_literal_origin,
            get_origin_func=get_origin,
            get_args_func=get_args,
        )
    ):
        if not (param_names and "library" in param_names):
            help_suffix = " Use forecast_list_methods to browse available methods."
            if "forecast_list_methods" not in kwargs["help"]:
                kwargs["help"] = f"{kwargs['help']}{help_suffix}"
            kwargs["metavar"] = "METHOD"
    else:
        try:
            ptype = param.get("type")
            base_type, origin = unwrap_optional_type(ptype)

            is_mapping_type = is_mapping_annotation(ptype)

            kwargs["type"] = str

            if base_type in (int, float, str):
                kwargs["type"] = base_type
            elif base_type is bool:
                kwargs["type"] = _parse_cli_bool_value
                kwargs["choices"] = ["true", "false"]

            if origin in (list, tuple):
                inner = get_args(ptype)[0] if get_args(ptype) else None
                inner_origin = get_origin(inner)
                if is_literal_origin(inner_origin):
                    choices = [str(v) for v in get_args(inner)]
                    if choices:
                        kwargs["choices"] = choices
                        kwargs["type"] = _case_insensitive_choice_parser(choices)
                    else:
                        kwargs["type"] = str
                    kwargs["nargs"] = "+"
                else:
                    kwargs["type"] = str
                    kwargs["nargs"] = "+"
            elif is_literal_origin(origin):
                choices = [str(v) for v in get_args(base_type)]
                if choices:
                    kwargs["choices"] = choices
                    kwargs["type"] = _case_insensitive_choice_parser(choices)
                else:
                    kwargs["type"] = str
        except Exception as exc:
            debug(f"Type resolution failed for param '{param['name']}': {exc}")
            kwargs["type"] = str

    if not param["required"] and not (param["type"] is bool and param["default"] is None):
        if (str(cmd_name or ""), str(param["name"])) in _PRESERVE_OMITTED_DEFAULT_PARAMS:
            kwargs["default"] = argparse.SUPPRESS
        else:
            kwargs["default"] = param["default"]

    choice_override_key = (str(cmd_name or ""), str(param["name"]))
    choice_override = _COMMAND_PARAM_CHOICE_OVERRIDES.get(choice_override_key)
    if choice_override:
        choices = list(choice_override)
        kwargs["choices"] = choices
        kwargs["type"] = _case_insensitive_choice_parser(choices)

    if choice_override_key == ("temporal_analyze", "group_by"):
        parse_group_by = kwargs["type"]

        def _parse_temporal_group(value: Any) -> str:
            parsed = parse_group_by(value)
            return "dow" if parsed == "day_of_week" else parsed

        kwargs["type"] = _parse_temporal_group

    if choice_override_key == ("trade_place", "order_type") and kwargs.get("choices"):
        parse_choice = _case_insensitive_choice_parser(kwargs["choices"])

        def _parse_order_type(value: Any) -> str:
            normalized = str(value or "").strip().replace("-", "_").replace(" ", "_")
            return parse_choice(normalized)

        kwargs["type"] = _parse_order_type

    if (str(cmd_name or ""), str(param["name"])) == ("indicators_list", "category"):
        kwargs["type"] = lambda value: str(value or "").strip().lower()

    return kwargs, is_mapping_type


def add_dynamic_arguments(  # noqa: C901
    parser: Any,
    param_info: Dict[str, Any],
    *,
    resolve_param_kwargs: Callable[..., Tuple[Dict[str, Any], bool]],
    param_docs: Optional[Dict[str, str]] = None,
    cmd_name: Optional[str] = None,
) -> None:
    """Add CLI arguments for an introspected function schema."""
    has_mapping_param = False

    def _extra_option_flags(param_name: str, cmd_name_value: Optional[str]) -> tuple[str, ...]:
        extras: list[str] = []
        if cmd_name_value == "trade_history" and param_name == "position_ticket":
            extras.append("--ticket")
        if cmd_name_value in {
            "forecast_backtest_run",
            "forecast_tune_genetic",
            "forecast_tune_optuna",
        } and param_name == "methods":
            extras.append("--method")
        if cmd_name_value in _SEARCH_ALIAS_COMMANDS and param_name == "search":
            extras.append("--search-term")
        elif cmd_name_value in _SEARCH_ALIAS_COMMANDS and param_name == "search_term":
            extras.append("--search")
        if cmd_name_value == "temporal_analyze" and param_name == "group_by":
            extras.append("--by")
        if cmd_name_value in {
            "causal_discover_signals",
            "cointegration_test",
            "correlation_matrix",
            "cross_correlation",
        } and param_name == "window_bars":
            extras.append("--lookback")
        return tuple(extras)

    for param in param_info["params"]:
        if not should_expose_cli_param(cmd_name=cmd_name, param_name=str(param.get("name") or "")):
            continue
        hyph = f"--{param['name'].replace('_', '-')}"
        uscr = f"--{param['name']}"
        option_flags, hidden_option_flags = _split_visible_and_hidden_flags(
            hyph,
            uscr,
            *_extra_option_flags(param["name"], cmd_name),
        )

        param_names = {p.get("name") for p in (param_info.get("params") or []) if isinstance(p, dict)}
        kwargs, is_mapping_type = resolve_param_kwargs(
            param,
            param_docs,
            cmd_name=cmd_name,
            param_names=param_names,
        )
        is_required_option = (
            param["required"] and param != param_info["params"][0]
        ) or (str(cmd_name or ""), str(param["name"])) in _COMMAND_REQUIRED_OPTIONS
        if is_required_option:
            kwargs["required"] = True
            kwargs["default"] = argparse.SUPPRESS
            kwargs["help"] = f"{kwargs.get('help') or param['name']} (required)"

        is_optional_bool = param.get("type") is bool and not param.get("required", False)
        allow_optional_positional = (
            str(cmd_name or ""),
            str(param["name"]),
        ) in _OPTIONAL_POSITIONAL_PARAMS

        required_symbol_alias = (
            param["required"]
            and param == param_info["params"][0]
            and str(param["name"]) in {"symbol", "symbols"}
        )
        if required_symbol_alias:
            parser.usage = (
                "%(prog)s (SYMBOL | --symbol SYMBOL) [options]"
                if str(param["name"]) == "symbol"
                else "%(prog)s (SYMBOL [SYMBOL ...] | --symbols SYMBOLS) [options]"
            )
            positional_kwargs = {
                k: v
                for k, v in kwargs.items()
                if k in ("help", "type", "choices", "metavar")
            }
            positional_kwargs["nargs"] = (
                "*"
                if (
                    str(cmd_name or "") in _MULTI_VALUE_SYMBOL_POSITIONAL_COMMANDS
                    and str(param["name"]) == "symbols"
                )
                else "?"
            )
            positional_kwargs["default"] = argparse.SUPPRESS
            positional_kwargs["help"] = (
                f"{positional_kwargs.get('help') or param['name']} (required)"
            )
            parser.add_argument(param["name"], **positional_kwargs)
            option_kwargs = dict(kwargs)
            option_kwargs["dest"] = f"{_OPTION_ALIAS_DEST_PREFIX}{param['name']}"
            option_kwargs.setdefault("metavar", str(param["name"]).upper())
            option_kwargs["default"] = argparse.SUPPRESS
            option_kwargs["required"] = False
            if option_flags:
                parser.add_argument(*option_flags, **option_kwargs)
            if hidden_option_flags:
                hidden_option_kwargs = dict(option_kwargs)
                hidden_option_kwargs["help"] = argparse.SUPPRESS
                parser.add_argument(*hidden_option_flags, **hidden_option_kwargs)
        elif (
            param["required"]
            and param == param_info["params"][0]
            and (str(cmd_name or ""), str(param["name"]))
            not in _NAMED_ONLY_REQUIRED_PARAMS
        ):
            positional_kwargs = {k: v for k, v in kwargs.items() if k in ("help", "type", "choices", "metavar")}
            if (
                str(cmd_name or "") in _MULTI_VALUE_SYMBOL_POSITIONAL_COMMANDS
                and str(param["name"]) == "symbols"
            ):
                positional_kwargs["nargs"] = "+"
            positional_kwargs["help"] = f"{positional_kwargs.get('help') or param['name']} (required)"
            parser.add_argument(param["name"], **positional_kwargs)
        elif allow_optional_positional:
            positional_kwargs = {k: v for k, v in kwargs.items() if k in ("help", "type", "choices", "metavar")}
            positional_kwargs["nargs"] = (
                "*"
                if (
                    str(cmd_name or "") in _MULTI_VALUE_SYMBOL_POSITIONAL_COMMANDS
                    and str(param["name"]) == "symbols"
                )
                else "?"
            )
            positional_kwargs["default"] = argparse.SUPPRESS
            parser.add_argument(param["name"], **positional_kwargs)
            option_kwargs = dict(kwargs)
            option_kwargs["dest"] = f"{_OPTION_ALIAS_DEST_PREFIX}{param['name']}"
            option_kwargs.setdefault("metavar", str(param["name"]).upper())
            option_kwargs["default"] = argparse.SUPPRESS
            if (
                str(cmd_name or "") in _MULTI_VALUE_SYMBOL_POSITIONAL_COMMANDS
                and str(param["name"]) == "symbols"
            ):
                option_kwargs["nargs"] = "+"
            if (
                str(param["name"]) != "symbols"
                or (str(cmd_name or ""), str(param["name"])) in _HIDDEN_OPTIONAL_POSITIONAL_FLAGS
            ):
                option_kwargs["help"] = argparse.SUPPRESS
            positional_key = (str(cmd_name or ""), str(param["name"]))
            if option_flags and positional_key not in _POSITIONAL_ONLY_OPTIONAL_PARAMS:
                parser.add_argument(*option_flags, **option_kwargs)
            if hidden_option_flags and positional_key not in _POSITIONAL_ONLY_OPTIONAL_PARAMS:
                hidden_option_kwargs = dict(option_kwargs)
                hidden_option_kwargs["help"] = argparse.SUPPRESS
                parser.add_argument(*hidden_option_flags, **hidden_option_kwargs)
        else:
            if is_optional_bool:
                local_kwargs = dict(kwargs)
                local_kwargs["nargs"] = "?"
                local_kwargs["const"] = "true"
                if option_flags:
                    parser.add_argument(*option_flags, **local_kwargs)
                if hidden_option_flags:
                    hidden_kwargs = dict(local_kwargs)
                    hidden_kwargs["help"] = argparse.SUPPRESS
                    parser.add_argument(*hidden_option_flags, **hidden_kwargs)
                no_flags, no_hidden_flags = _split_visible_and_hidden_flags(
                    f"--no-{param['name'].replace('_', '-')}",
                    f"--no_{param['name']}",
                )
                if no_flags:
                    parser.add_argument(
                        *no_flags,
                        dest=param["name"],
                        action="store_const",
                        const="false",
                        help=argparse.SUPPRESS,
                    )
                if no_hidden_flags:
                    hidden_no_kwargs = {
                        "dest": param["name"],
                        "action": "store_const",
                        "const": "false",
                        "help": argparse.SUPPRESS,
                    }
                    parser.add_argument(*no_hidden_flags, **hidden_no_kwargs)
            elif is_mapping_type:
                local_kwargs = dict(kwargs)
                local_kwargs["nargs"] = "?"
                local_kwargs["const"] = "__PRESENT__"
                if option_flags:
                    parser.add_argument(*option_flags, **local_kwargs)
                if hidden_option_flags:
                    hidden_kwargs = dict(local_kwargs)
                    hidden_kwargs["help"] = argparse.SUPPRESS
                    parser.add_argument(*hidden_option_flags, **hidden_kwargs)
            else:
                if option_flags:
                    parser.add_argument(*option_flags, **kwargs)
                if hidden_option_flags:
                    hidden_kwargs = dict(kwargs)
                    hidden_kwargs["help"] = argparse.SUPPRESS
                    hidden_kwargs["required"] = False
                    parser.add_argument(*hidden_option_flags, **hidden_kwargs)
        if str(param["name"]) == "minutes_back" and str(cmd_name or "").startswith("trade_"):
            parser.add_argument(
                "--days",
                dest="_trade_days",
                type=float,
                default=argparse.SUPPRESS,
                metavar="DAYS",
                help="Alias for --minutes-back expressed in days.",
        )

        if is_mapping_type:
            has_mapping_param = True
            if param["name"] == "params":
                continue
            params_flags = _dedupe_flags(
                f"--{param['name'].replace('_', '-')}-params",
                f"--{param['name']}_params",
            )
            parser.add_argument(
                *params_flags,
                dest=f"{param['name']}_params",
                type=str,
                default=None,
                help=f"Extra params for {param['name']} (key=value[,key=value])",
            )
    if has_mapping_param:
        parser.add_argument(
            "--set",
            dest="set_overrides",
            action="append",
            default=None,
            metavar="PARAM.KEY=VALUE",
            help="Override nested mapping params, e.g. --set params.window=64.",
        )
