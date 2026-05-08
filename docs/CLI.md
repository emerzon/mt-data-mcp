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

When available, add `--dry-run true` first. The CLI expects boolean values as `true` or `false`, for example `--dry-run true`.

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

---

## Output Contract

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

Available timeframes: `M1`, `M5`, `M15`, `M30`, `H1`, `H4`, `D1`, `W1`, `MN1`

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

### Async Training & Model Store
| Command | Description |
|---------|-------------|
| `forecast_train` | Start a background training job for heavyweight methods (returns a `task_id`) |
| `forecast_task_status` | Poll training progress for a `task_id` |
| `forecast_task_wait` | Wait for a task to finish or until a timeout is reached |
| `forecast_task_cancel` | Cancel a running training task |
| `forecast_task_list` | List active and recent training tasks |
| `forecast_models_list` | List trained models cached on disk |
| `forecast_models_delete` | Delete a stored model by `model_id` |

Trained models are written under `~/.mtdata/models/` by default and re-used automatically by subsequent `forecast_generate` calls with the same method/symbol/timeframe/params. Task status is persisted in `~/.mtdata/forecast/jobs.sqlite` by default, so recent task state can survive process restarts. See [ENV_VARS.md](ENV_VARS.md#async-training--model-store) for the related environment variables.

### Risk Analysis
| Command | Description |
|---------|-------------|
| `forecast_barrier_prob` | Calculate TP/SL hit probabilities |
| `forecast_barrier_optimize` | Find optimal TP/SL levels |
| `labels_triple_barrier` | Label data with barrier outcomes |
| `regime_detect` | Detect market regimes and change points |

### Indicators & Patterns
| Command | Description |
|---------|-------------|
| `indicators_list` | List available indicators |
| `indicators_describe` | Get indicator details |
| `patterns_detect` | Detect candlestick/chart patterns |
| `pivot_compute_points` | Calculate pivot levels |
| `support_resistance_levels` | Compute support/resistance levels with Fibonacci swing context |
| `correlation_matrix` | Pairwise correlation matrix between symbols |
| `cointegration_test` | Pairwise cointegration test between symbols |
| `causal_discover_signals` | Granger-style causal discovery between symbols |

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
| `trade_risk_analyze` | Analyze position risk |
| `trade_var_cvar_calculate` | Estimate portfolio VaR/CVaR from open positions |

### News
| Command | Description |
|---------|-------------|
| `news` | Unified, ranked news feed (general + symbol-relevant + economic calendar). Pass `--symbol` to focus the feed on an instrument. |

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
| `finviz_forex` | Get forex pairs performance snapshot |
| `finviz_crypto` | Get cryptocurrency performance snapshot |
| `finviz_futures` | Get futures market performance snapshot |
| `finviz_calendar` | Get economic, earnings, or dividends calendar |
| `finviz_earnings` | Get upcoming earnings announcements |

See [FINVIZ.md](FINVIZ.md) for detailed examples.

### Options & QuantLib
| Command | Description |
|---------|-------------|
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

# Multi-symbol selectors prefer `symbols`, but MCP/Web API calls may also send
# the compatibility alias `symbol` when only one selector string is available.
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

### Place Orders
`trade_place` requires `symbol`, `volume`, and `order_type`.

The examples below intentionally use `--dry-run true`. Remove it, or set `--dry-run false`, only when you are on the intended account and ready to send the order to MT5.

Accepted `order_type` forms:
- Canonical: `BUY`, `SELL`, `BUY_LIMIT`, `BUY_STOP`, `SELL_LIMIT`, `SELL_STOP`
- MT5 aliases: `ORDER_TYPE_BUY`, `ORDER_TYPE_BUY_LIMIT`, etc.
- MT5 numeric constants: `0..5`

```bash
# Preview a pending order with canonical order_type
mtdata-cli trade_place BTCUSD --volume 0.03 --order-type BUY_LIMIT --price 68750 \
  --stop-loss 67500 --take-profit 72000 --dry-run true

# Same order_type using MT5 alias, still as a dry run
mtdata-cli trade_place BTCUSD --volume 0.03 --order-type ORDER_TYPE_BUY_LIMIT --price 68750 \
  --stop-loss 67500 --take-profit 72000 --dry-run true

# Same order_type using MT5 numeric constant
mtdata-cli trade_place BTCUSD --volume 0.03 --order-type 2 --price 68750 \
  --stop-loss 67500 --take-profit 72000 --dry-run true

# Preview a market order with explicit protective levels
mtdata-cli trade_place BTCUSD --volume 0.01 --order-type BUY \
  --stop-loss 64500 --take-profit 67200 --dry-run true
```

For account-level safety, configure trade guardrails in [ENV_VARS.md](ENV_VARS.md#trade-guardrails) before moving from preview to live execution.

### Close or Modify Positions
Use exact tickets whenever possible. Preview closes before execution when `--dry-run true` is supported:

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

### Estimate Portfolio Tail Risk
```bash
mtdata-cli trade_var_cvar_calculate --timeframe H1 --lookback 500 --confidence 95 --json
mtdata-cli trade_var_cvar_calculate --symbol EURUSD --method gaussian --transform pct --lookback 300 --json
```

### Detect Patterns and Regimes
```bash
# Candlestick patterns
mtdata-cli patterns_detect EURUSD --mode candlestick --robust-only true

# Regime detection
mtdata-cli regime_detect EURUSD --method hmm --params "n_states=2"

# Change-point detection
mtdata-cli regime_detect EURUSD --method bocpd --threshold 0.5
```

### Compare Cross-Symbol Relationships (Exploratory)
```bash
# Rank co-moving symbols with transformed-return correlations
mtdata-cli correlation_matrix "EURUSD,GBPUSD,USDJPY" --timeframe H1 \
  --limit 500 --method pearson --transform log_return --json

# Use an explicit MT5 group path instead of naming symbols one-by-one
mtdata-cli correlation_matrix --group "Forex\\Majors" --timeframe H1 \
  --limit 120 --method pearson --transform log_return --extras metadata --json

# Find candidate mean-reverting pairs inside an MT5 group
mtdata-cli cointegration_test --group "Forex\\Majors" --timeframe H1 \
  --limit 400 --transform log_level --significance 0.05 --json

# Compare a few symbols directly
mtdata-cli causal_discover_signals "EURUSD,GBPUSD,USDJPY" --timeframe H1 \
  --limit 800 --max-lag 5 --transform log_return --significance 0.05

# Pass a single symbol to auto-expand its visible MT5 group (e.g., Forex\\Majors)
mtdata-cli causal_discover_signals EURUSD --timeframe H1 --limit 800
```

For `market_scan`, `correlation_matrix`, `cointegration_test`, and
`causal_discover_signals`, prefer the canonical `symbols` selector for new
multi-symbol integrations. The compatibility alias `symbol` is also accepted,
but `symbol`/`symbols` must agree when both are supplied, and `group` remains
mutually exclusive with explicit symbol selectors.

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
- [EXAMPLE.md](EXAMPLE.md) — Complete workflow example
- [FINVIZ.md](FINVIZ.md) — Fundamental data commands
- [OPTIONS_QUANTLIB.md](OPTIONS_QUANTLIB.md) — Options and QuantLib commands
- [TEMPORAL.md](TEMPORAL.md) — Temporal analysis
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) — Common issues

