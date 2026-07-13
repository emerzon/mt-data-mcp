# Documentation Index

Friendly guides and references for **mtdata** — the MT5 research and automation toolkit (CLI, MCP, Web API). For a high-level introduction, start at the [project README](../README.md).

**How to use this folder:** set up once, practice a **read-only** workflow, then open deeper pages only when you need them. You do not need to read everything end-to-end.

> **Safety:** `trade_*` commands can place, modify, or close **real** orders on the MT5 account that is currently logged in (demo or live). Prefer a demo account until you are confident in your setup. See [TRADING_SAFETY.md](TRADING_SAFETY.md).

## Choose your path

| Goal | Start here | Then read |
|------|------------|-----------|
| Install and confirm MT5 connectivity | [SETUP.md](SETUP.md) | [TROUBLESHOOTING.md](TROUBLESHOOTING.md) if anything fails |
| Learn the command line safely | [CLI.md](CLI.md) | [GLOSSARY.md](GLOSSARY.md), [SAMPLE-TRADE.md](SAMPLE-TRADE.md) |
| Build a research workflow | [EXAMPLE.md](EXAMPLE.md) | [REPORTS.md](REPORTS.md), [FORECAST.md](FORECAST.md), [BARRIER_FUNCTIONS.md](BARRIER_FUNCTIONS.md) |
| Integrate with an app or agent | [WEB_API.md](WEB_API.md) | [DEPLOYMENT.md](DEPLOYMENT.md), [ENV_VARS.md](ENV_VARS.md), [CLI.md](CLI.md) |
| Prepare for trade execution | [SAMPLE-TRADE-ADVANCED.md](SAMPLE-TRADE-ADVANCED.md) | [ENV_VARS.md](ENV_VARS.md#trade-guardrails), [TRADING_SAFETY.md](TRADING_SAFETY.md) |

## Learning path

1. [SETUP.md](SETUP.md) — Install, connect MT5, configure timezone
2. [GLOSSARY.md](GLOSSARY.md) — Trading and forecasting terms (start here if you are new)
3. [CLI.md](CLI.md) — How to run commands and read output
4. [SAMPLE-TRADE.md](SAMPLE-TRADE.md) — Beginner-friendly workflow with explanations
5. [SAMPLE-TRADE-ADVANCED.md](SAMPLE-TRADE-ADVANCED.md) — More methods and stricter risk/execution gates
6. Deep dives: [FORECAST.md](FORECAST.md), [TIME_SERIES_DIAGNOSTICS.md](TIME_SERIES_DIAGNOSTICS.md), [BARRIER_FUNCTIONS.md](BARRIER_FUNCTIONS.md), [TECHNICAL_INDICATORS.md](TECHNICAL_INDICATORS.md)
7. Specialized: [FINVIZ.md](FINVIZ.md), [TEMPORAL.md](TEMPORAL.md), [OPTIONS_QUANTLIB.md](OPTIONS_QUANTLIB.md)

## Getting Started

| Document | Description |
|----------|-------------|
| [SETUP.md](SETUP.md) | Installation, MT5 connection, environment variables |
| [ENV_VARS.md](ENV_VARS.md) | Complete `.env` reference (MT5, MCP, Web API, GPU, etc.) |
| [CLI.md](CLI.md) | Command conventions, output formats, help system |
| [OUTPUT.md](OUTPUT.md) | Response envelope, `detail`/`extras`, pagination, and error codes |
| [TIMESTAMPS.md](TIMESTAMPS.md) | Timezone policy: broker time, UTC, client-local, and provider time |
| [GLOSSARY.md](GLOSSARY.md) | **Dense terms** — BOCPD, Kelly, VaR, Granger, etc. ([quick find](GLOSSARY.md#quick-find)) |
| [LIMITATIONS.md](LIMITATIONS.md) | Practical caveats, provider limits, and documentation gaps |
| [DEPLOYMENT.md](DEPLOYMENT.md) | Run the MCP server or Web API as a persistent local service |

## Core Topics

| Document | Description |
|----------|-------------|
| [FORECAST.md](FORECAST.md) | Price forecasting methods, async training, and model store |
| [REPORTS.md](REPORTS.md) | Generate packaged market summaries, choose a template, and filter output sections |
| [forecast/METHODS.md](forecast/METHODS.md) | Per-method reference: categories, libraries, default parameters, dependencies |
| [forecast/FORECAST_GENERATE.md](forecast/FORECAST_GENERATE.md) | Detailed `forecast_generate` reference (parameters, quantity modes, pipeline) |
| [forecast/BACKTESTING.md](forecast/BACKTESTING.md) | Rolling backtests and parameter optimization |
| [forecast/VOLATILITY.md](forecast/VOLATILITY.md) | Volatility estimation and forecasting |
| [forecast/REGIMES.md](forecast/REGIMES.md) | Market regime and change-point detection |
| [TIME_SERIES_DIAGNOSTICS.md](TIME_SERIES_DIAGNOSTICS.md) | Stationarity, automatic seasonality, outlier, and volatility-cone analysis |
| [forecast/UNCERTAINTY.md](forecast/UNCERTAINTY.md) | Confidence and conformal intervals |
| [BARRIER_FUNCTIONS.md](BARRIER_FUNCTIONS.md) | TP/SL hit probability, optimization, and statistical robustness checks |
| [TECHNICAL_INDICATORS.md](TECHNICAL_INDICATORS.md) | Available indicators and how to use them |
| [LEVELS.md](LEVELS.md) | Pivots, support/resistance, confluence zones, and volume profile |
| [DENOISING.md](DENOISING.md) | Smoothing filters and noise reduction |
| [SIMPLIFICATION.md](SIMPLIFICATION.md) | Downsampling and output reduction (`--simplify`) |
| [CAUSAL_DISCOVERY.md](CAUSAL_DISCOVERY.md) | Granger-style causal signal discovery |
| [ADVANCED_ANALYTICS.md](ADVANCED_ANALYTICS.md) | Tick microstructure, execution quality, robust strategy validation, portfolio risk decomposition, and relative strength |
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

## Trading

| Document | Description |
|----------|-------------|
| [TRADING_RISK.md](TRADING_RISK.md) | Read-only risk analytics: position sizing (fixed-fraction + Kelly), VaR/CVaR, and scenario stress tests |
| [TRADING_SAFETY.md](TRADING_SAFETY.md) | Safety runbook: dry-run previews, guardrails, validation, and broker behavior for `trade_*` |
| [BARRIER_FUNCTIONS.md](BARRIER_FUNCTIONS.md) | TP/SL hit probability, optimization, and statistical robustness checks |

## Common workflows (recipes)

Research-only snippets (not financial advice). For a guided narrative, use [SAMPLE-TRADE.md](SAMPLE-TRADE.md).

### 1) Quick market snapshot (no trading)

```bash
mtdata-cli symbols_describe EURUSD --json
mtdata-cli data_fetch_candles EURUSD --timeframe H1 --limit 200 --json
mtdata-cli forecast_generate EURUSD --timeframe H1 --horizon 12 --method theta --json
```

### 2) TP/SL odds for a trade idea

```bash
mtdata-cli forecast_barrier_prob EURUSD --timeframe H1 --horizon 12 \
  --method mc_gbm --direction long --tp-pct 0.4 --sl-pct 0.6 --json
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

## Where details live

Practical caveats: [LIMITATIONS.md](LIMITATIONS.md). Dedicated references for common “how does this work?” questions:

- Forecast methods, defaults, and dependencies → [forecast/METHODS.md](forecast/METHODS.md) (reproducibility notes in [FORECAST.md](FORECAST.md#reproducibility-notes))
- Response envelope, `detail` / `extras`, pagination, errors → [OUTPUT.md](OUTPUT.md)
- Timezones (broker, UTC, client-local, provider) → [TIMESTAMPS.md](TIMESTAMPS.md)
- Trading dry-run, guardrails, broker behavior → [TRADING_SAFETY.md](TRADING_SAFETY.md)
- Long-running MCP / Web API service setup → [DEPLOYMENT.md](DEPLOYMENT.md)

## Quick reference

```bash
# List commands
mtdata-cli --help

# Search by keyword
mtdata-cli --help forecast
mtdata-cli --help barrier

# Help for one command
mtdata-cli forecast_generate --help
mtdata-cli regime_detect --help
```
