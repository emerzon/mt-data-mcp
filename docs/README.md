# Documentation Index

This folder contains user-facing documentation for mtdata. Start with setup, then follow a guided workflow before exploring individual topics.

Safety note: `trade_*` commands can place/modify/close real orders on the account currently logged into MT5 (demo or live). Use a demo account until you're confident in your setup.

## Learning Path

1. [SETUP.md](SETUP.md) — Install, connect MT5, configure timezone
2. [GLOSSARY.md](GLOSSARY.md) — Trading + forecasting terms (start here if new)
3. [CLI.md](CLI.md) — How to run commands and read output
4. [SAMPLE-TRADE.md](SAMPLE-TRADE.md) — A beginner-friendly workflow with explanations
5. [SAMPLE-TRADE-ADVANCED.md](SAMPLE-TRADE-ADVANCED.md) — More methods + stricter risk/execution gates
6. Deep dives: [FORECAST.md](FORECAST.md), [BARRIER_FUNCTIONS.md](BARRIER_FUNCTIONS.md), [TECHNICAL_INDICATORS.md](TECHNICAL_INDICATORS.md)
7. Specialized: [FINVIZ.md](FINVIZ.md), [TEMPORAL.md](TEMPORAL.md), [OPTIONS_QUANTLIB.md](OPTIONS_QUANTLIB.md)

## Getting Started

| Document | Description |
|----------|-------------|
| [SETUP.md](SETUP.md) | Installation, MT5 connection, environment variables |
| [CLI.md](CLI.md) | Command conventions, output formats, help system |
| [GLOSSARY.md](GLOSSARY.md) | **Start here** — Explanations of all technical terms |

## Core Topics

| Document | Description |
|----------|-------------|
| [FORECAST.md](FORECAST.md) | Price forecasting methods and usage |
| [forecast/BACKTESTING.md](forecast/BACKTESTING.md) | Rolling backtests and parameter optimization |
| [forecast/VOLATILITY.md](forecast/VOLATILITY.md) | Volatility estimation and forecasting |
| [forecast/REGIMES.md](forecast/REGIMES.md) | Market regime and change-point detection |
| [forecast/UNCERTAINTY.md](forecast/UNCERTAINTY.md) | Confidence and conformal intervals |
| [BARRIER_FUNCTIONS.md](BARRIER_FUNCTIONS.md) | TP/SL hit probability and optimization |
| [TECHNICAL_INDICATORS.md](TECHNICAL_INDICATORS.md) | Available indicators and how to use them |
| [DENOISING.md](DENOISING.md) | Smoothing filters and noise reduction |
| [SIMPLIFICATION.md](SIMPLIFICATION.md) | Downsampling and output reduction (`--simplify`) |
| [CAUSAL_DISCOVERY.md](CAUSAL_DISCOVERY.md) | Granger-style causal signal discovery |
| [TEMPORAL.md](TEMPORAL.md) | Session effects, day-of-week, hourly and monthly patterns |
| [forecast/PATTERN_SEARCH.md](forecast/PATTERN_SEARCH.md) | Pattern detection and similarity search |

## External Data & Options

| Document | Description |
|----------|-------------|
| [FINVIZ.md](FINVIZ.md) | US equity fundamentals, screening, news, insider activity, macro snapshots |
| [OPTIONS_QUANTLIB.md](OPTIONS_QUANTLIB.md) | Options chains, QuantLib barrier pricing, Heston calibration |
| [WEB_API.md](WEB_API.md) | FastAPI endpoints powering the Web UI |

## Tutorials

| Document | Description |
|----------|-------------|
| [SAMPLE-TRADE.md](SAMPLE-TRADE.md) | Beginner step-by-step trade analysis guide |
| [SAMPLE-TRADE-ADVANCED.md](SAMPLE-TRADE-ADVANCED.md) | Advanced workflow with regimes, HAR-RV, barriers |
| [EXAMPLE.md](EXAMPLE.md) | Complete end-to-end command-driven research loop |

## Common Workflows (Recipes)

These are research workflows (not financial advice). For a guided narrative, start with [SAMPLE-TRADE.md](SAMPLE-TRADE.md).

### 1) Quick market snapshot (no trading)
```bash
mtdata-cli symbols_describe EURUSD --json
mtdata-cli data_fetch_candles EURUSD --timeframe H1 --limit 200 --json
mtdata-cli forecast_generate EURUSD --timeframe H1 --horizon 12 --model theta --json
```

### 2) TP/SL odds for a trade idea
```bash
mtdata-cli forecast_barrier_prob EURUSD --timeframe H1 --horizon 12 \
  --method mc --mc-method mc_gbm --direction long --tp-pct 0.4 --sl-pct 0.6 --json
```

### 3) Scan a small watchlist (PowerShell)
```powershell
$symbols = "EURUSD","GBPUSD","USDJPY"
$symbols | % { mtdata-cli forecast_volatility_estimate $_ --timeframe H1 --horizon 12 --method ewma --json }
```

## Troubleshooting

| Document | Description |
|----------|-------------|
| [TROUBLESHOOTING.md](TROUBLESHOOTING.md) | Common issues and solutions |

## Quick Reference

**List available commands:**
```bash
mtdata-cli --help
```

**Search for a command:**
```bash
mtdata-cli --help forecast
mtdata-cli --help barrier
```

**Get help for a specific command:**
```bash
mtdata-cli forecast_generate --help
mtdata-cli regime_detect --help
```

