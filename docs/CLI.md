# CLI Guide

The CLI is the one-shot interface for scripts and occasional exploration. Every
tool is one command:

```bash
mtdata-cli <command> [options]
```

Use it for scripts, exploration, and copy-paste workflows. Each one-shot
invocation starts Python and discovers the requested command family. For repeated
local exploration, run `mtdata-cli shell` and enter ordinary command lines
without the `mtdata-cli` prefix; imports remain warm until `exit` or `quit`.
The shell also accepts newline-delimited commands on stdin for non-interactive
batches, ignores blank lines and `#` comments, and exits nonzero if any command
fails. Batch output is NDJSON: each executable input line produces one compact
JSON envelope with `line`, `command`, `success`, and `status`. Parsed child JSON
is nested under `result`; non-JSON output and diagnostics use `output` and
`stderr`.
Repeated agent or application calls should keep a process alive with
`mtdata-stdio`, `mtdata-streamable-http`, or `mtdata-webapi`. The full tool surface is also
available over [MCP](GLOSSARY.md#mcp-model-context-protocol). The Web API
exposes a focused subset with the same canonical payload semantics for
operations shared across interfaces. Default text presentation is
[TOON](GLOSSARY.md#toon); use `--json` for machines.

Background training is rejected by one-shot CLI processes because their workers
would be terminated as soon as the command returned. Run `forecast_train` or
`forecast_generate --async-mode true` inside `mtdata-cli shell`, through MCP, or
through the Web API.

Stuck on an acronym in output (BOCPD, Kelly, CVaR, …)? See the [glossary quick find](GLOSSARY.md#quick-find).

**Related:** [README](../README.md) · [Setup](SETUP.md) · [Glossary](GLOSSARY.md) · [Output contract](OUTPUT.md)

---

## Safety (trading commands)

`trade_*` can place, modify, or close **real** orders on the account logged into MT5.

- Prefer a **demo account** while learning.
- mtdata has no separate paper mode — demo terminal = simulated execution.
- Preview with `--dry-run true` before live sends.

| Safer research | Live execution |
|----------------|----------------|
| `symbols_*`, `market_*`, `data_fetch_*` | `trade_place` |
| `forecast_*`, `regime_detect`, `patterns_detect` | `trade_modify` |
| `report_generate`, `trade_risk_analyze`, `trade_get_*` | `trade_close` |

`trade_place`, `trade_modify`, and `trade_close` default to **preview mode** (`dry_run=true`). Set `--dry-run false` explicitly for a live request. Bulk closes still require `--close-all` and explicit confirmation. Booleans on the CLI are `true` / `false`.

Full runbook: [TRADING_SAFETY.md](TRADING_SAFETY.md).

## Getting help

```bash
# List all commands
mtdata-cli --help

# Search by topic
mtdata-cli --help forecast
mtdata-cli --help barrier
mtdata-cli --help regime

# Help for one command
mtdata-cli forecast_generate --help
mtdata-cli regime_detect --help

# Tool catalog (filter / paginate)
mtdata-cli tools_list --category forecast --json
```

---

## Output contract

Every tool returns the **same canonical payload**; CLI, MCP, and Web API only change presentation. For the full envelope (`success` / error, `detail`, `output_fields`, pagination, error codes), see [OUTPUT.md](OUTPUT.md).

### TOON (Default)
Human-readable compact TOON output:
```bash
mtdata-cli symbols_list --limit 5
```

TOON includes a quick schema hint:
```
data[5]{name,group,description}:
    EURUSD,Forex\Majors,Euro vs US Dollar
    ...
```
- `data[5]` is the number of rows returned
- `{name,group,description}` are the columns/keys in each row

### JSON
Structured output for programmatic use:
```bash
mtdata-cli symbols_list --limit 5 --json
```

For scripts that always require JSON, set `MTDATA_OUTPUT_FORMAT=json` in the
environment or `.env` file. Accepted values are `json` and `toon`; an explicit
`--json` flag always selects JSON.

JSON output always keeps numeric values unminimized. Text output uses
`--precision auto`, which compacts most tools while preserving full precision
for an explicit set of sensitive outputs, including trading, quotes, forecasts,
reports, and price-level tools.

Control display precision explicitly:
```bash
# Preserve full numeric precision in TOON text
mtdata-cli market_ticker EURUSD --precision full

# Compact a large table for token-saving display
mtdata-cli data_fetch_candles EURUSD --limit 200 --precision compact

```

`--precision raw` is accepted as an alias for `full`, and `display` is accepted
as an alias for `compact`. There is no global `--decimals` option; tools with a
domain-specific decimal control document it in their own help. Precision
controls only text presentation; internal tool processing and JSON/raw payloads
keep numeric values.

### Detail and field selection
Compact output is implicit. Use `--detail full` for richer runtime metadata,
diagnostics, request context, and supporting rows:
```bash
mtdata-cli market_status --detail full
```

Use `--output-fields` to project the final response without changing domain
semantics:
```bash
mtdata-cli symbols_describe EURUSD --output-fields symbol,digits,point --json
```

### Exit Codes

| Code | Meaning |
|------|---------|
| `0` | Command completed without a tool error |
| `1` | Tool/provider failure, invalid tool payload, interrupted command, internal CLI error, or no command selected |
| `2` | Argument parsing or command-line usage error reported by `argparse` |

Scripts should parse JSON error fields when they need to distinguish provider,
validation, and internal failures that share exit code `1`.

---

## Common Patterns

### Positional Arguments
Most commands take `symbol` as the first positional argument:
```bash
mtdata-cli forecast_generate EURUSD --horizon 12
mtdata-cli regime_detect EURUSD --method hmm
mtdata-cli data_fetch_candles EURUSD --limit 100
```

### Timeframe
Specify market data granularity with `--timeframe`:
```bash
mtdata-cli data_fetch_candles EURUSD --timeframe M15 --limit 100
mtdata-cli forecast_generate EURUSD --timeframe H4 --horizon 24
```

Available timeframes: `M1`, `M2`, `M3`, `M4`, `M5`, `M6`, `M10`, `M12`, `M15`, `M20`, `M30`, `H1`, `H2`, `H3`, `H4`, `H6`, `H8`, `H12`, `D1`, `W1`, `MN1`. Broker history availability may vary.

### Parameters
Pass method-specific parameters with `--params`:
```bash
mtdata-cli forecast_volatility_estimate EURUSD --method ewma --params "lambda_=0.94"
mtdata-cli regime_detect EURUSD --method hmm --params "n_states=3"
```

Format: `key=value key2=value2` (space-separated), `key=value,key2=value2` (comma-separated), or JSON `{"key": value}` — all three are accepted.

### Reduce Large Outputs (Simplify)
Use `--simplify` to downsample returned rows for charting or large exports.

```bash
# Default simplification (targets ~10% of --limit)
mtdata-cli data_fetch_candles EURUSD --timeframe M1 --limit 5000 --simplify

# Choose an algorithm + target points
mtdata-cli data_fetch_candles EURUSD --timeframe M1 --limit 5000 \
  --simplify lttb --simplify-params "points=500"

# Raw ticks can also be simplified
mtdata-cli data_fetch_ticks EURUSD --limit 20000 \
  --simplify rdp --simplify-params "points=2000"
```

Tick rows preserve MT5 snapshot fields: `bid`, `ask`, `last`, `volume`,
`volume_real`, and `flags`. The volume fields describe the current last trade,
not a row-level tick count; use `flags` to identify trade-change events. Candle
rows continue to use `tick_volume` for the broker's per-bar tick count.

See [SIMPLIFICATION.md](SIMPLIFICATION.md) for algorithms and parameters.

### Method Parameters
For forecast methods, use `--params`:
```bash
mtdata-cli forecast_generate EURUSD --method arima --params "p=2 d=1 q=2"
mtdata-cli forecast_generate EURUSD --method mc_gbm --params "n_sims=2000 seed=42"
```

---

## Date Inputs

Commands accepting `--start` and `--end` parse flexible date strings:
```bash
# Relative dates
mtdata-cli data_fetch_candles EURUSD --start "2 days ago" --end "now"
mtdata-cli data_fetch_candles EURUSD --start "1 week ago"

# Absolute dates
mtdata-cli data_fetch_candles EURUSD --start "2025-12-01" --end "2025-12-31"
```

For candle ranges, bounds are inclusive and resolved in UTC. An ISO date-only
`--start` means 00:00:00 at the beginning of that day, while a date-only
`--end` means 23:59:59.999999 at the end of that day. Include an explicit time
and timezone when automation needs an exact instant, using either an ISO offset
(`2026-08-03T09:30-04:00`) or an IANA name
(`2026-08-03 09:30 America/New_York`). Ambiguous or nonexistent daylight-saving
local times are rejected; use an ISO offset to choose a specific instant.
Candle responses echo the resolved instants and bound modes in `query_applied`.

---

## Command Categories

### Data
| Command | Description |
|---------|-------------|
| `symbols_list` | List available trading symbols |
| `symbols_describe` | Get symbol details (pip size, contract, etc.) |
| `symbols_top_markets` | Rank the top MT5 markets by spread, recent volume, or recent price change |
| `market_scan` | Filter MT5 symbols by spread, price change, volume, RSI, and SMA |
| `data_fetch_candles` | Fetch OHLCV candles with optional indicators |
| `data_fetch_ticks` | Fetch tick data |
| `market_depth_fetch` | Get order book (DOM) — requires `MTDATA_ENABLE_MARKET_DEPTH_FETCH=1` |
| `market_ticker` | Get current bid/ask/spread snapshot |
| `market_snapshot` | Unified pre-trade snapshot (quote, levels, patterns; optional regime/forecast sections) |
| `market_status` | Get market trading hours and session status |
| `wait_event` | **Blocking:** wait for a clock, single-symbol, or basket event using exactly one stopping mode: the next `timeframe` boundary or `max_wait_seconds` duration |

### Forecasting
| Command | Description |
|---------|-------------|
| `forecast_generate` | Generate price forecasts |
| `forecast_list_methods` | List available forecasting methods |
| `forecast_list_library_models` | List models in a specific library |
| `forecast_backtest_run` | Run rolling-origin backtest |
| `strategy_backtest` | Backtest simple indicator-driven trading strategies |
| `forecast_conformal_intervals` | Generate calibrated confidence bands |
| `forecast_volatility_estimate` | Forecast volatility |
| `forecast_tune_genetic` | Optimize model parameters (genetic algorithm) |
| `forecast_tune_optuna` | Optimize model parameters (Bayesian/Optuna) |
| `forecast_optimize_hints` | Genetic search for top forecast configurations across timeframes, methods, and parameters |

### Async Training & Model Store
| Command | Description |
|---------|-------------|
| `forecast_train` | Start a background training job for heavyweight methods (returns a `task_id`) |
| `forecast_task_status` | Poll training progress for a `task_id` |
| `forecast_task_wait` | Wait for a task to finish or until a timeout is reached |
| `forecast_task_cancel` | Cancel a running training task |
| `forecast_task_cancel_all` | Cancel all active training tasks |
| `forecast_task_list` | List active and recent training tasks |
| `forecast_models_list` | List trained models cached on disk |
| `forecast_models_delete` | Delete a stored model by `model_id` |
| `forecast_models_cleanup` | Preview or delete stale/expired stored models |

Trained models are written under `~/.mtdata/models/` by default and reused by
live `forecast_generate` calls with the same method, symbol, timeframe, horizon,
seasonality, preprocessing, and training parameters. Results report model age
in bars, and live reuse is capped at one resolved seasonal cycle. Historical
`as_of` forecasts require an exact training anchor. Task
status is persisted in `~/.mtdata/forecast/jobs.sqlite` by default, so recent
task state can survive process restarts. See
[ENV_VARS.md](ENV_VARS.md#async-training--model-store) for related variables.

### Risk Analysis
| Command | Description |
|---------|-------------|
| `forecast_barrier_prob` | Calculate TP/SL hit probabilities |
| `forecast_barrier_optimize` | Find optimal TP/SL levels |
| `labels_triple_barrier` | Label data with barrier outcomes |
| `regime_detect` | Detect market regimes and change points |

### Time-Series Diagnostics
| Command | Description |
|---------|-------------|
| `stationarity_test` | Run ADF, KPSS, and optional Phillips-Perron tests |
| `seasonality_detect` | Rank dominant periods using autocorrelation and spectral peaks |
| `outliers_detect` | Detect anomalous return, volume, and range bars |
| `volatility_term_structure` | Compare realized volatility across rolling horizons and historical percentiles |

### Indicators & Patterns
| Command | Description |
|---------|-------------|
| `indicators_list` | List available indicators |
| `indicators_describe` | Get indicator details |
| `patterns_detect` | Detect candlestick, chart, harmonic, fractal, and Elliott patterns |
| `volume_profile_levels` | Compute POC, VAH, and VAL from bounded ticks or M1-bar approximation |
| `confluence_levels` | Rank price zones where pivots, support/resistance, Fibonacci, and volume-profile levels cluster |
| `pivot_compute_points` | Calculate pivot levels |
| `support_resistance_levels` | Compute support/resistance levels with Fibonacci swing context |
| `correlation_matrix` | Pairwise correlation matrix between symbols |
| `cross_correlation` | Estimate lead/lag correlation between two symbols |
| `cointegration_test` | Engle-Granger pair tests or Johansen multivariate cointegration |
| `causal_discover_signals` | Granger predictive-link discovery between symbols |

The root `--timeframe` default maps to `--pivot-timeframe` for
`confluence_levels`; an explicit command-level `--pivot-timeframe` takes
precedence. `--sr-timeframe` remains independent and defaults to `auto`.

Volume profile example:

```bash
mtdata-cli volume_profile_levels EURUSD --start "1 day ago" --end "now" \
  --source auto --price-source mid --bucket-points 10 --json
```

You can also derive the window from a lookback:

```bash
mtdata-cli volume_profile_levels EURUSD --timeframe H1 --limit 168 \
  --source auto --bucket-points 10 --json
```

The default tick window is one day and 50,000 ticks. Longer `auto` windows use
the labeled M1-bar approximation unless those caps are raised explicitly.

For fractal + volume-structure confluence, opt in through pattern config:

```bash
mtdata-cli patterns_detect EURUSD --timeframe H1 --mode fractal \
  --config '{"volume_profile":true,"volume_profile_tolerance_points":25}' --json
```

See [LEVELS.md](LEVELS.md) for the full pivots, support/resistance, confluence, and volume-profile reference.

### Denoising
| Command | Description |
|---------|-------------|
| `denoise_list_methods` | List denoise methods with their dependencies, causality support, and auto parameters |
| `denoise_describe` | Describe one denoise method and its supported options and defaults |

Denoising is applied to data via the `--denoise`/`--denoise-params` flags (see [Reduce Large Outputs](#reduce-large-outputs-simplify) and the examples below). Use `denoise_list_methods`/`denoise_describe` to discover method names and parameters first. See [DENOISING.md](DENOISING.md) for the full reference.

### Trading
| Command | Description |
|---------|-------------|
| `trade_account_info` | Get account info |
| `trade_session_context` | Snapshot of broker/session/server-time context for downstream trading prompts |
| `trade_place` | Place orders |
| `trade_close` | Close positions |
| `trade_modify` | Modify orders |
| `trade_get_open` | Get open positions |
| `trade_get_pending` | Get pending orders |
| `trade_history` | Get trading history |
| `trade_journal_analyze` | Summarize realized exit-deal performance |
| `trade_execution_quality` | Analyze slippage, latency, partial fills, fees, and markouts |
| `trade_risk_analyze` | Analyze position risk |
| `trade_var_cvar_calculate` | Estimate portfolio VaR/CVaR from open positions |
| `trade_stress_test` | Apply deterministic percentage shocks to open positions |

See [TRADING_RISK.md](TRADING_RISK.md) for position sizing (fixed-fraction + Kelly), VaR/CVaR, and stress-test parameters and output.
`trade_journal_analyze` reports raw account-currency PnL per exit deal. Those
averages are useful for journal review, but they are not Kelly inputs because
they are not normalized to a consistent stake or unit of risk.

### News
| Command | Description |
|---------|-------------|
| `news` | Unified, ranked news feed (general + symbol-relevant + economic calendar). Pass `--symbol` to focus the feed on an instrument. |

### Advanced MT5-native analytics

| Command | Description |
|---------|-------------|
| `market_microstructure_analyze` | Analyze tick liquidity and feed-appropriate order-flow proxies |
| `strategy_validate` | Run anchored fixed-candidate OOS validation with horizon-safe barrier outcomes and costs |
| `portfolio_risk_decompose` | Decompose filtered-historical VaR/ES and proposed-trade risk |
| `market_relative_strength` | Rank a bounded MT5 universe by robust factor-adjusted momentum and breadth |

See [ADVANCED_ANALYTICS.md](ADVANCED_ANALYTICS.md) for data requirements, examples, and caveats.

### Reports
| Command | Description |
|---------|-------------|
| `report_generate` | Generate consolidated analysis report |

### Temporal Analysis
| Command | Description |
|---------|-------------|
| `temporal_analyze` | Analyze returns, volatility, and volume by time period (day of week, hour, month) |

### Fundamental Data (Finviz)
| Command | Description |
|---------|-------------|
| `finviz_fundamentals` | Get company fundamental metrics (P/E, EPS, market cap, etc.) |
| `finviz_description` | Get company business description |
| `finviz_news` | Get stock-specific or general market news |
| `finviz_market_news` | Get broad market headlines or blog posts |
| `finviz_insider` | Get insider trading activity for a stock |
| `finviz_insider_activity` | Get market-wide insider trading activity |
| `finviz_ratings` | Get analyst ratings history |
| `finviz_peers` | Find peer companies |
| `finviz_screen` | Screen stocks using Finviz filters |
| `finviz_filters_list` | List valid screener filters and accepted values (pair with `finviz_screen`) |
| `finviz_forex` | Get forex pairs performance snapshot |
| `finviz_crypto` | Get cryptocurrency performance snapshot |
| `finviz_futures` | Get futures market performance snapshot |
| `finviz_calendar` | Get economic, earnings, or dividends calendar |
| `finviz_earnings` | Get upcoming earnings announcements |

See [FINVIZ.md](FINVIZ.md) for detailed examples.

### Options & QuantLib
| Command | Description |
|---------|-------------|
| `options_provider_status` | Report configured options-chain provider readiness without querying market data |
| `options_expirations` | List available option expiration dates |
| `options_chain` | Fetch options chain snapshot with filtering |
| `options_barrier_price` | Price a barrier option using QuantLib |
| `options_heston_calibrate` | Calibrate Heston stochastic volatility model |

See [OPTIONS_QUANTLIB.md](OPTIONS_QUANTLIB.md) for detailed examples.

---

## Examples by Task

### Explore Available Symbols
```bash
# List forex pairs
mtdata-cli symbols_list --limit 20

# Get details for a symbol
mtdata-cli symbols_describe EURUSD --json

# Rank the current watchlist by spread, volume, and price change
mtdata-cli symbols_top_markets --rank-by all --limit 5 --timeframe H1 --json

# Include hidden symbols within a bounded comparable category
mtdata-cli symbols_top_markets --rank-by spread --limit 10 --universe all \
  --category forex --json

# Scan visible majors for strong RSI and price above SMA
mtdata-cli market_scan --group "Forex\\Majors" --rsi-above 60 --price-vs-sma above \
  --sma-period 20 --timeframe H1 --lookback 120 --json

# Scan an explicit symbol basket for oversold names with tight spreads
mtdata-cli market_scan EURUSD,GBPUSD,USDJPY --rsi-below 35 --max-spread-pct 0.03 --json

# Multi-symbol selectors use the canonical `symbols` selector.
```

`market_scan` and `symbols_top_markets` keep completed-bar values such as
`close` separate from the current executable quote. When a quote is available,
rows expose `bid`, `ask`, `mid`, and `quote_as_of`; use those fields for a live
mark and the bar fields for ranking and indicator context.

`symbols_list` rejects non-positive limits. `symbols_top_markets` preserves
exact ranking semantics and rejects a filtered candidate universe above 250
symbols before activating hidden quotes; narrow it with `group` or `category`.

### Fetch Market Data
```bash
# Basic candles
mtdata-cli data_fetch_candles EURUSD --timeframe H1 --limit 100

# With indicators
mtdata-cli data_fetch_candles EURUSD --timeframe H1 --limit 100 \
  --indicators "ema(20),rsi(14),macd(12,26,9)"

# With denoising
mtdata-cli data_fetch_candles EURUSD --timeframe H1 --limit 100 \
  --denoise ema --denoise-params "alpha=0.2"
```

### Generate Forecasts
```bash
# Basic forecast
mtdata-cli forecast_generate EURUSD --timeframe H1 --horizon 12 --method theta

# Foundation method
mtdata-cli forecast_generate EURUSD --library pretrained --method chronos2 --horizon 24

# Monte Carlo simulation
mtdata-cli forecast_generate EURUSD --method mc_gbm --params "n_sims=2000"

# Search timeframes + methods + params for the best starting configuration
mtdata-cli forecast_optimize_hints EURUSD --timeframes H1 H4 D1 \
  --methods theta ets --horizon 12 --top-n 5 --json
```

### Backtest Trading Rules
```bash
mtdata-cli strategy_backtest EURUSD --timeframe H1 --strategy sma_cross \
  --fast-period 10 --slow-period 30 --lookback 300 --json

mtdata-cli strategy_backtest EURUSD --timeframe H1 --strategy rsi_reversion \
  --rsi-length 14 --oversold 30 --overbought 70 --position-mode long_only --json

# Historical MT5 bar spread is the default cost model. For a controlled constant:
mtdata-cli strategy_backtest EURUSD --cost-model fixed --spread-bps 1.2 --json
```

Net returns and derived performance metrics are reported only when every trade has
a spread cost. If historical spread data cannot price every trade, the response
keeps gross results, labels the partial result `return_after_known_costs`, reports
the priced-trade coverage, and marks performance metrics unavailable. Use an
explicit fixed spread when a complete, comparable net result is required.

### Analyze Risk
```bash
# Volatility estimate
mtdata-cli forecast_volatility_estimate EURUSD --horizon 12 --method ewma

# Barrier probability
mtdata-cli forecast_barrier_prob EURUSD --horizon 12 \
  --method hmm_mc --barrier '{"kind":"tp_sl","unit":"pct","take_profit":0.5,"stop_loss":0.3}'

# Optimize TP/SL
mtdata-cli forecast_barrier_optimize EURUSD --horizon 12 \
  --grid-style volatility --objective edge
```

### Pre-Trade Snapshot & Session Context
```bash
# One-shot pre-trade snapshot: quote + levels + patterns
mtdata-cli market_snapshot EURUSD --timeframe H1 --json

# Add the optional regime + forecast sections (sections=all)
mtdata-cli market_snapshot EURUSD --timeframe H1 --sections all --horizon 8 --json

# Global exchange status (NYSE, LSE, Tokyo, ...) or one broker symbol's tradability
mtdata-cli market_status --region all --json
mtdata-cli market_status --symbol EURUSD --json

# Consolidated broker/session context (account, open/pending, quote, computed state)
mtdata-cli trade_session_context EURUSD --json
```

In symbol mode, `is_tradable` reflects the broker trade mode (including
close-only symbols), while `can_open_new_positions` additionally requires a
live-ready quote and an active session.

### Place Orders
`trade_place` requires `symbol`, `volume`, and `order_type`.

The examples below intentionally use `--dry-run true`. Remove it, or set `--dry-run false`, only when you are on the intended account and ready to send the order to MT5. See [TRADING_SAFETY.md](TRADING_SAFETY.md) for the dry-run-first workflow, account guardrails, and broker behavior.

Accepted `order_type` values (case-insensitive; `-` or space is normalized to `_`):
`BUY`, `SELL`, `BUY_LIMIT`, `BUY_STOP`, `SELL_LIMIT`, `SELL_STOP`. MT5 numeric constants (`0..5`) and `ORDER_TYPE_*` names are **not** accepted as input — they only appear when *reading* existing orders/positions.

```bash
# Preview a pending order with canonical order_type
mtdata-cli trade_place BTCUSD --volume 0.03 --order-type BUY_LIMIT --price 68750 \
  --stop-loss 67500 --take-profit 72000 --dry-run true

# Case and separators are normalized (buy-stop -> BUY_STOP)
mtdata-cli trade_place BTCUSD --volume 0.03 --order-type buy-stop --price 70200 \
  --stop-loss 69000 --take-profit 73000 --dry-run true

# Preview a market order with explicit protective levels
mtdata-cli trade_place BTCUSD --volume 0.01 --order-type BUY \
  --stop-loss 64500 --take-profit 67200 --dry-run true
```

### Trade Execution Controls

| Flag | Applies To | Description |
|------|------------|-------------|
| `--dry-run` | `trade_place`, `trade_modify`, `trade_close` | Preview the request without sending it to MT5. |
| `--detail` | `trade_place` | Preview detail level; use `full` for execution diagnostics. |
| `--magic` | `trade_place`, `trade_get_open`, `trade_get_pending`, `trade_close` | MT5 magic-number filter or default strategy identifier. |
| `--require-sl-tp` | `trade_place` | Require both stop-loss and take-profit on market orders. |
| `--auto-close-on-sl-tp-fail` | `trade_place` | If SL/TP attachment fails after a market fill, try to close the unprotected position. |
| `--expiration` | `trade_place`, `trade_modify` | Expiration time for pending orders (`dateparser`, UTC epoch seconds, or `GTC`). |
| `--idempotency-key` | `trade_place`, `trade_modify` | Durable dedupe key shared by CLI and server processes within the configured retention window. |
| `--close-all` | `trade_close` | Close all matching positions instead of one ticket. |
| `--pnl-filter` | `trade_close`, `trade_get_open` | Filter positions by `all`, `profit`, or `loss`. |
| `--close-priority` | `trade_close` | When multiple positions match, close `loss_first`, `profit_first`, or `largest_first`. |

Every `trade_place`, `trade_modify`, and `trade_close` response includes a
`correlation_id`. The same value appears in execution logs; live order results
also log MT5's numeric `request_id` as `mt5_request_id`. Idempotent replays expose
both the current `correlation_id` and the original invocation's
`original_correlation_id`.

For account-level safety, configure trade guardrails in [ENV_VARS.md](ENV_VARS.md#trade-guardrails) before moving from preview to live execution.

### Close or Modify Positions
Use exact tickets whenever possible. `trade_close` defaults to preview mode;
set `--dry-run false` explicitly only when you intend a live close:

```bash
mtdata-cli trade_get_open --json
mtdata-cli trade_modify 123456789 --stop-loss 60500 --take-profit 62500
mtdata-cli trade_close --ticket 123456789 --volume 0.05 --dry-run true
```

Be especially careful with `trade_close --close-all`; it targets every matching open position.

### Review Trade Journal
```bash
mtdata-cli trade_journal_analyze --minutes-back 10080 --json
mtdata-cli trade_journal_analyze --symbol EURUSD --minutes-back 43200 --breakdown-limit 5 --json
```

`trade_history` and `trade_journal_analyze` default to a 7-day lookback (`--minutes-back 10080`) when you do not pass a time window explicitly.
For Kelly sizing in `trade_risk_analyze`, provide a `--sizing` JSON object with
`win_rate`, `avg_win`, and `avg_loss` derived from complete trade lifecycles whose
returns are normalized consistently (for example, R-multiples). Do not map the
raw `trade_journal_analyze` PnL averages into those flags.

### Estimate Portfolio Tail Risk
```bash
mtdata-cli trade_var_cvar_calculate --timeframe H1 --lookback 500 --confidence 95 --json
mtdata-cli trade_var_cvar_calculate --symbol EURUSD --method gaussian --transform pct --lookback 300 --json
```

### Stress Open Positions
```bash
mtdata-cli trade_stress_test '{"EURUSD":-2.0,"GBPUSD":-1.5}' --json
mtdata-cli trade_stress_test '{"*":-3.0}' --detail full --json
```

Shock values are percentage price moves. `*` is a fallback for any open-position symbol without an explicit shock. The tool is read-only and reports estimated P&L and equity impact.

### Detect Patterns and Regimes
```bash
# Candlestick patterns
mtdata-cli patterns_detect EURUSD --mode candlestick --robust-only true

# Harmonic Fibonacci-ratio patterns
mtdata-cli patterns_detect EURUSD --mode harmonic --limit 800

# Regime detection
mtdata-cli regime_detect EURUSD --method hmm --params "n_states=2"

# Change-point detection
mtdata-cli regime_detect EURUSD --method bocpd --threshold 0.5
```

### Compare Cross-Symbol Relationships (Exploratory)
```bash
# Rank co-moving symbols with transformed-return correlations
mtdata-cli correlation_matrix "EURUSD,GBPUSD,USDJPY" --timeframe H1 \
  --window-bars 500 --method pearson --transform log_return --json

# Use an explicit MT5 group path instead of naming symbols one-by-one
mtdata-cli correlation_matrix --group "Forex\\Majors" --timeframe H1 \
  --window-bars 500 --limit 120 --method pearson --transform log_return --detail full --json

# Find candidate mean-reverting pairs inside an MT5 group
mtdata-cli cointegration_test --group "Forex\\Majors" --timeframe H1 \
  --window-bars 400 --transform log_level --significance 0.05 --json

# Compare a few symbols directly
mtdata-cli causal_discover_signals "EURUSD,GBPUSD,USDJPY" --timeframe H1 \
  --window-bars 800 --max-lag 5 --transform log_return --significance 0.05

# Pass a single symbol to auto-expand its visible MT5 group (e.g., Forex\\Majors)
mtdata-cli causal_discover_signals EURUSD --timeframe H1 --window-bars 800
```

For `market_scan`, `correlation_matrix`, `cointegration_test`, and
`causal_discover_signals`, use the canonical `symbols` selector for
multi-symbol integrations. `group` remains mutually exclusive with explicit
symbol selectors.

See [CAUSAL_DISCOVERY.md](CAUSAL_DISCOVERY.md) for interpretation and caveats.

---

## Tips

### Pipe Output to jq for JSON Processing
```bash
mtdata-cli forecast_generate EURUSD --json | jq '.forecast'
```

### Save Output to File
```bash
mtdata-cli data_fetch_candles EURUSD --limit 1000 --json > eurusd_data.json
```

### Debug Mode
Set the debug environment variable for verbose CLI logging:

PowerShell:
```powershell
$env:MTDATA_CLI_DEBUG = "1"
mtdata-cli forecast_generate EURUSD
$env:MTDATA_CLI_DEBUG = $null
```

Bash:
```bash
MTDATA_CLI_DEBUG=1 mtdata-cli forecast_generate EURUSD
```

---

## See Also

- [SETUP.md](SETUP.md) — Installation guide
- [OUTPUT.md](OUTPUT.md) — Response envelope, `detail`/`output_fields`, and error codes
- [TIMESTAMPS.md](TIMESTAMPS.md) — Timezone policy for inputs and output
- [TRADING_SAFETY.md](TRADING_SAFETY.md) — Dry-run-first trading runbook and guardrails
- [EXAMPLE.md](EXAMPLE.md) — Complete workflow example
- [FINVIZ.md](FINVIZ.md) — Fundamental data commands
- [OPTIONS_QUANTLIB.md](OPTIONS_QUANTLIB.md) — Options and QuantLib commands
- [TEMPORAL.md](TEMPORAL.md) — Temporal analysis
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) — Common issues
