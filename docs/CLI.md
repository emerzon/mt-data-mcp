# CLI Guide

The CLI is the quickest way to explore mtdata capabilities. All tools are accessible via `mtdata-cli <command>`.

**Related:**
- [README.md](../README.md) — Project overview
- [SETUP.md](SETUP.md) — Installation and configuration
- [GLOSSARY.md](GLOSSARY.md) — Term definitions

---

## Safety (Trading Commands)

`trade_*` commands can place/modify/close real orders on the account currently logged into MT5 (demo or live). Use a demo account until you're confident in your setup.

There is no built-in “paper trading” mode in mtdata; for simulated execution use an MT5 demo account and double-check which account is logged in before running any `trade_*` commands.

Use read-only commands for research and reserve execution commands for intentional account operations:

| Safer Research Commands | Live Execution Commands |
|-------------------------|-------------------------|
| `symbols_*`, `market_*`, `data_fetch_*` | `trade_place` |
| `forecast_*`, `regime_detect`, `patterns_detect` | `trade_modify` |
| `report_generate`, `trade_risk_analyze`, `trade_get_*` | `trade_close` |

`trade_place`, `trade_modify`, and `trade_close` all default to `dry_run=false`,
so they execute live unless you pass `--dry-run true` to preview. Bulk closes
still require `--close-all` and explicit confirmation.
The CLI expects boolean values as `true` or `false`.

## Getting Help

**List all commands:**
```bash
mtdata-cli --help
```

**Search for commands by topic:**
```bash
mtdata-cli --help forecast
mtdata-cli --help barrier
mtdata-cli --help regime
```

**Get help for a specific command:**
```bash
mtdata-cli forecast_generate --help
mtdata-cli regime_detect --help
```

**Discover tools programmatically:**
```bash
mtdata-cli tools_list --category forecast --json   # filter/paginate the tool catalog
```

---

## Output Contract

Every tool returns the same canonical payload; the transport only adapts presentation. For the full response envelope — the `success`/error structure, `detail` levels, `extras` sections, pagination, and error codes — see [OUTPUT.md](OUTPUT.md).

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

JSON output keeps numeric values unminimized by default. Text output uses
`--precision auto`, which preserves full precision for trading and price-level
tools while compacting known large tables such as candles and scans.

Control display precision explicitly:
```bash
# Preserve full numeric precision in TOON text
mtdata-cli market_ticker EURUSD --precision full

# Compact a large table for token-saving display
mtdata-cli data_fetch_candles EURUSD --limit 200 --precision compact

# Use a deterministic display decimal count
mtdata-cli data_fetch_candles EURUSD --limit 200 --precision compact --decimals 5
```

`--precision raw` is accepted as an alias for `full`, and `display` is accepted
as an alias for `compact`. Precision controls only presentation; internal tool
processing and JSON/raw payloads keep numeric values.

### Extras
Compact output is implicit. For richer sections such as runtime metadata,
diagnostics, echoed request context, raw rows, or method documentation, use
`--extras`:
```bash
mtdata-cli market_status --extras metadata,diagnostics
```

Use `--extras all` when you need every supported richer section.

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
| `wait_event` | Stream real-time market events |

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

Trained models are written under `~/.mtdata/models/` by default and re-used automatically by subsequent `forecast_generate` calls with the same method/symbol/timeframe/params. Task status is persisted in `~/.mtdata/forecast/jobs.sqlite` by default, so recent task state can survive process restarts. See [ENV_VARS.md](ENV_VARS.md#async-training--model-store) for the related environment variables.

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
| `causal_discover_signals` | Granger-style causal discovery between symbols |

Volume profile example:

```bash
mtdata-cli volume_profile_levels EURUSD --start "1 week ago" --end "now" \
  --source auto --price-source mid --bucket-points 10 --json
```

You can also derive the window from a lookback:

```bash
mtdata-cli volume_profile_levels EURUSD --timeframe H1 --limit 168 \
  --source auto --bucket-points 10 --json
```

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
For Kelly sizing, `trade_journal_analyze` is the quickest way to derive
`win_rate`, `avg_win`, and `avg_loss` inputs from realized trade history.

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

# Opt into a slower full-universe scan when you need hidden tradable symbols too
mtdata-cli symbols_top_markets --rank-by spread --limit 10 --universe all --json

# Scan visible majors for strong RSI and price above SMA
mtdata-cli market_scan --group "Forex\\Majors" --rsi-above 60 --price-vs-sma above \
  --sma-period 20 --timeframe H1 --lookback 120 --json

# Scan an explicit symbol basket for oversold names with tight spreads
mtdata-cli market_scan EURUSD,GBPUSD,USDJPY --rsi-below 35 --max-spread-pct 0.03 --json

# Multi-symbol selectors use the canonical `symbols` selector.
```

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
```

### Analyze Risk
```bash
# Volatility estimate
mtdata-cli forecast_volatility_estimate EURUSD --horizon 12 --method ewma

# Barrier probability
mtdata-cli forecast_barrier_prob EURUSD --horizon 12 \
  --method hmm_mc --tp-pct 0.5 --sl-pct 0.3

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
| `--idempotency-key` | `trade_place`, `trade_modify` | Dedupe repeated requests within the current process only (in-memory ~5-minute TTL; not persisted across restarts or shared across workers). |
| `--close-all` | `trade_close` | Close all matching positions instead of one ticket. |
| `--profit-only` / `--loss-only` | `trade_close` | Restrict closes to positions currently in profit or loss. |
| `--close-priority` | `trade_close` | When multiple positions match, close `loss_first`, `profit_first`, or `largest_first`. |

For account-level safety, configure trade guardrails in [ENV_VARS.md](ENV_VARS.md#trade-guardrails) before moving from preview to live execution.

### Close or Modify Positions
Use exact tickets whenever possible. Because `trade_close` defaults to live
execution, pass `--dry-run true` explicitly when previewing a close:

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
For Kelly sizing in `trade_risk_analyze`, map `summary.win_rate`,
`summary.avg_win`, and `summary.avg_loss` from `trade_journal_analyze` to
`--kelly-win-rate`, `--kelly-avg-win`, and `--kelly-avg-loss`.

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
  --window-bars 500 --limit 120 --method pearson --transform log_return --extras metadata --json

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
- [OUTPUT.md](OUTPUT.md) — Response envelope, `detail`/`extras`, and error codes
- [TIMESTAMPS.md](TIMESTAMPS.md) — Timezone policy for inputs and output
- [TRADING_SAFETY.md](TRADING_SAFETY.md) — Dry-run-first trading runbook and guardrails
- [EXAMPLE.md](EXAMPLE.md) — Complete workflow example
- [FINVIZ.md](FINVIZ.md) — Fundamental data commands
- [OPTIONS_QUANTLIB.md](OPTIONS_QUANTLIB.md) — Options and QuantLib commands
- [TEMPORAL.md](TEMPORAL.md) — Temporal analysis
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) — Common issues
