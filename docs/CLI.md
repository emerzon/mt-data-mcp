# CLI Guide

The CLI is the quickest way to explore mtdata capabilities. All tools are accessible via `python cli.py <command>`.

**Related:**
- [README.md](../README.md) — Project overview
- [SETUP.md](SETUP.md) — Installation and configuration
- [GLOSSARY.md](GLOSSARY.md) — Term definitions

---

## Safety (Trading Commands)

`trading_*` commands can place/modify/close real orders on the account currently logged into MT5 (demo or live). Use a demo account until you're confident in your setup.

There is no built-in “paper trading” mode in mtdata; for simulated execution use an MT5 demo account and double-check which account is logged in before running any `trading_*` commands.

## Getting Help

**List all commands:**
```bash
python cli.py --help
```

**Search for commands by topic:**
```bash
python cli.py --help forecast
python cli.py --help barrier
python cli.py --help regime
```

**Get help for a specific command:**
```bash
python cli.py forecast_generate --help
python cli.py regime_detect --help
```

---

## Output Formats

### Text (Default)
Human-readable compact output:
```bash
python cli.py symbols_list --limit 5
```

The text format includes a quick schema hint:
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
python cli.py symbols_list --limit 5 --format json
```

### Verbose
Include additional metadata:
```bash
python cli.py forecast_generate EURUSD --horizon 12 --verbose
```

Tip: with `--format json`, `--verbose` also adds `cli_meta` (including timezone hints) to many tool results.

---

## Common Patterns

### Positional Arguments
Most commands take `symbol` as the first positional argument:
```bash
python cli.py forecast_generate EURUSD --horizon 12
python cli.py regime_detect EURUSD --method hmm
python cli.py data_fetch_candles EURUSD --limit 100
```

### Timeframe
Specify market data granularity with `--timeframe`:
```bash
python cli.py data_fetch_candles EURUSD --timeframe M15 --limit 100
python cli.py forecast_generate EURUSD --timeframe H4 --horizon 24
```

Available timeframes: `M1`, `M5`, `M15`, `M30`, `H1`, `H4`, `D1`, `W1`, `MN1`

### Parameters
Pass method-specific parameters with `--params`:
```bash
python cli.py forecast_volatility_estimate EURUSD --method ewma --params "lambda=0.94"
python cli.py regime_detect EURUSD --method hmm --params "n_states=3"
```

Format: `key=value,key2=value2` or JSON `{"key": value}`

### Reduce Large Outputs (Simplify)
Use `--simplify` to downsample returned rows for charting or large exports.

```bash
# Default simplification (targets ~10% of --limit)
python cli.py data_fetch_candles EURUSD --timeframe M1 --limit 5000 --simplify

# Choose an algorithm + target points
python cli.py data_fetch_candles EURUSD --timeframe M1 --limit 5000 \
  --simplify lttb --simplify-params "points=500"

# Raw ticks (rows output) can also be simplified
python cli.py data_fetch_ticks EURUSD --output rows --limit 20000 \
  --simplify rdp --simplify-params "points=2000"
```

See [SIMPLIFICATION.md](SIMPLIFICATION.md) for algorithms and parameters.

### Model Parameters
For forecast models, use `--model-params`:
```bash
python cli.py forecast_generate EURUSD --model arima --model-params "p=2 d=1 q=2"
python cli.py forecast_generate EURUSD --model mc_gbm --model-params "n_sims=2000 seed=42"
```

---

## Date Inputs

Commands accepting `--start` and `--end` parse flexible date strings:
```bash
# Relative dates
python cli.py data_fetch_candles EURUSD --start "2 days ago" --end "now"
python cli.py data_fetch_candles EURUSD --start "1 week ago"

# Absolute dates
python cli.py data_fetch_candles EURUSD --start "2025-12-01" --end "2025-12-31"
```

---

## Command Categories

### Data
| Command | Description |
|---------|-------------|
| `symbols_list` | List available trading symbols |
| `symbols_describe` | Get symbol details (pip size, contract, etc.) |
| `data_fetch_candles` | Fetch OHLCV candles with optional indicators |
| `data_fetch_ticks` | Fetch tick data |
| `market_depth_fetch` | Get order book (DOM) |

### Forecasting
| Command | Description |
|---------|-------------|
| `forecast_generate` | Generate price forecasts |
| `forecast_list_methods` | List available forecasting methods |
| `forecast_list_library_models` | List models in a specific library |
| `forecast_backtest_run` | Run rolling-origin backtest |
| `forecast_conformal_intervals` | Generate calibrated confidence bands |
| `forecast_volatility_estimate` | Forecast volatility |
| `forecast_tune_genetic` | Optimize model parameters |

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
| `causal_discover_signals` | Granger-style causal discovery between symbols |

### Trading
| Command | Description |
|---------|-------------|
| `trading_account_info` | Get account info |
| `trading_place` | Place orders |
| `trading_close` | Close positions |
| `trading_modify` | Modify orders |
| `trading_open_get` | Get open positions |
| `trading_history` | Get trading history |
| `trading_risk_analyze` | Analyze position risk |

### Reports
| Command | Description |
|---------|-------------|
| `report_generate` | Generate consolidated analysis report |

---

## Examples by Task

### Explore Available Symbols
```bash
# List forex pairs
python cli.py symbols_list --limit 20

# Get details for a symbol
python cli.py symbols_describe EURUSD --format json
```

### Fetch Market Data
```bash
# Basic candles
python cli.py data_fetch_candles EURUSD --timeframe H1 --limit 100

# With indicators
python cli.py data_fetch_candles EURUSD --timeframe H1 --limit 100 \
  --indicators "ema(20),rsi(14),macd(12,26,9)"

# With denoising
python cli.py data_fetch_candles EURUSD --timeframe H1 --limit 100 \
  --denoise ema --denoise-params "alpha=0.2"
```

### Generate Forecasts
```bash
# Basic forecast
python cli.py forecast_generate EURUSD --timeframe H1 --horizon 12 --model theta

# Foundation model
python cli.py forecast_generate EURUSD --library pretrained --model chronos2 --horizon 24

# Monte Carlo simulation
python cli.py forecast_generate EURUSD --model mc_gbm --model-params "n_sims=2000"
```

### Analyze Risk
```bash
# Volatility estimate
python cli.py forecast_volatility_estimate EURUSD --horizon 12 --method ewma

# Barrier probability
python cli.py forecast_barrier_prob EURUSD --horizon 12 \
  --method mc --mc-method hmm_mc --tp-pct 0.5 --sl-pct 0.3

# Optimize TP/SL
python cli.py forecast_barrier_optimize EURUSD --horizon 12 \
  --grid-style volatility --objective edge
```

### Detect Patterns and Regimes
```bash
# Candlestick patterns
python cli.py patterns_detect EURUSD --mode candlestick --robust-only true

# Regime detection
python cli.py regime_detect EURUSD --method hmm --params "n_states=2"

# Change-point detection
python cli.py regime_detect EURUSD --method bocpd --threshold 0.5
```

### Discover Causal Links (Exploratory)
```bash
# Compare a few symbols directly
python cli.py causal_discover_signals "EURUSD,GBPUSD,USDJPY" --timeframe H1 \
  --limit 800 --max-lag 5 --transform log_return --significance 0.05

# Pass a single symbol to auto-expand its visible MT5 group (e.g., Forex\\Majors)
python cli.py causal_discover_signals EURUSD --timeframe H1 --limit 800
```

See [CAUSAL_DISCOVERY.md](CAUSAL_DISCOVERY.md) for interpretation and caveats.

---

## Tips

### Pipe Output to jq for JSON Processing
```bash
python cli.py forecast_generate EURUSD --format json | jq '.forecast'
```

### Save Output to File
```bash
python cli.py data_fetch_candles EURUSD --limit 1000 --format json > eurusd_data.json
```

### Debug Mode
Set environment variable for verbose debugging:
```bash
MTDATA_CLI_DEBUG=1 python cli.py forecast_generate EURUSD
```

---

## See Also

- [SETUP.md](SETUP.md) — Installation guide
- [EXAMPLE.md](EXAMPLE.md) — Complete workflow example
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) — Common issues
