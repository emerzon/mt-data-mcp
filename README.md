# mtdata

**mtdata** is a research and automation toolkit for MetaTrader 5 (MT5). It provides tools for forecasting, regime detection, signal processing, risk analysis, and reporting—bridging raw market data with data science.

It runs as a **Model Context Protocol (MCP)** server or a standalone **CLI**.

## Who Is This For?

- **Newer traders / learners:** Follow guided workflows (no quant background required).
- **Systematic traders:** Prototype ideas, backtest quickly, and automate via CLI/MCP.
- **Data folks:** Pull MT5 market data into repeatable analysis pipelines.

## Platform Support (Important)

- **Windows is required** to run MetaTrader 5 (and therefore to run `mtdata` against MT5).
- If you're on macOS/Linux, run `mtdata` on a **Windows VM or Windows machine** and connect remotely (MCP/Web API).
- **Python 3.14 is the supported runtime** for the packaged dependency set in this repo.

## Safety First

- `mtdata` includes `trade_*` commands that can **place/modify/close real orders** on the account currently logged into MT5.
- Use a **demo account** until you understand the tools and your broker setup.
- There is no built-in “paper trading” mode in mtdata; use an MT5 demo account for simulated execution.
- If you only want research, stick to `data_*`, `forecast_*`, `regime_*`, `patterns_*`, and `report_*` commands.

## Capabilities

| Category | What It Does | Key Tools |
|----------|--------------|-----------|
| **Data** | Fetch candles, ticks, market depth from MT5 | `data_fetch_candles`, `data_fetch_ticks`, `market_depth_fetch` |
| **Forecasting** | Predict price paths with classical or ML models | `forecast_generate`, `forecast_backtest_run` |
| **Volatility** | Estimate future price movement magnitude | `forecast_volatility_estimate` |
| **Regimes** | Detect trending, ranging, or crisis market states | `regime_detect` |
| **Barriers** | Calculate TP/SL hit probabilities via simulation | `forecast_barrier_prob`, `forecast_barrier_optimize` |
| **Patterns** | Identify candlestick and chart patterns | `patterns_detect` |
| **Indicators** | Compute 100+ technical indicators | `data_fetch_candles --indicators` |
| **Denoising** | Smooth price data to reveal trends | `--denoise` option |
| **Trading** | Place orders, manage positions | `trade_place`, `trade_close` |

## Quick Start

**Prerequisites:** Windows + Python 3.14 + MetaTrader 5 installed and running (demo account recommended).

```bash
# Install dependencies
pip install -r requirements.txt

# Verify MT5 connection (lists symbols from the running terminal)
python cli.py symbols_list --limit 5

# Fetch recent candles
python cli.py data_fetch_candles EURUSD --timeframe H1 --limit 50

# Generate a baseline price forecast
python cli.py forecast_generate EURUSD --timeframe H1 --horizon 12 --model theta
```

Notes:
- The default Python 3.14 install includes the validated Chronos/TimesFM, StatsForecast, sktime, and mlforecast paths.
- `gluonts`/Lag-Llama, `hnswlib`, and `tsdownsample` are intentionally excluded from the default 3.14 environment because current wheel support is incomplete or incompatible.

## Documentation

New here? Follow this learning path:
`docs/SETUP.md` → `docs/GLOSSARY.md` → `docs/CLI.md` → `docs/SAMPLE-TRADE.md` (then `docs/SAMPLE-TRADE-ADVANCED.md` and deep dives).

### Getting Started
- **[Setup & Configuration](docs/SETUP.md)** — Installation, MT5 connection, environment variables
- **[CLI Guide](docs/CLI.md)** — Command conventions, output formats, help system
- **[Glossary](docs/GLOSSARY.md)** — Explanations of all technical terms with real-world examples
- **[Docs Index](docs/README.md)** — One-page map of all docs

### Core Topics
- **[Forecasting](docs/FORECAST.md)** — Price prediction methods (Theta, ARIMA, Chronos, etc.)
- **[Volatility](docs/forecast/VOLATILITY.md)** — Estimating price movement magnitude
- **[Regime Detection](docs/forecast/REGIMES.md)** — Identifying market states (trending vs. ranging)
- **[Barrier Analysis](docs/BARRIER_FUNCTIONS.md)** — TP/SL hit probability calculation
- **[Technical Indicators](docs/TECHNICAL_INDICATORS.md)** — Available indicators and usage
- **[Denoising](docs/DENOISING.md)** — Smoothing filters to reveal trends
- **[Pattern Detection](docs/forecast/PATTERN_SEARCH.md)** — Candlestick and chart patterns

### Tutorials
- **[Sample Trade Workflow](docs/SAMPLE-TRADE.md)** — Step-by-step analysis for a trade decision
- **[Advanced Playbook](docs/SAMPLE-TRADE-ADVANCED.md)** — Regime filters, conformal intervals, barrier optimization
- **[End-to-End Example](docs/EXAMPLE.md)** — Complete research loop with all tools

### Reference
- **[Troubleshooting](docs/TROUBLESHOOTING.md)** — Common issues and fixes

## Configuration

Create a `.env` file in the project root:

```ini
MT5_LOGIN=12345678
MT5_PASSWORD=your_password
MT5_SERVER=your_broker_server
MT5_TIME_OFFSET_MINUTES=0  # Adjust if broker server time differs from UTC
```

## Architecture

```
mtdata/
├── cli.py              # CLI entry point
├── server.py           # MCP server entry point
├── webui.py            # Web API (FastAPI + React UI)
├── src/mtdata/
│   ├── core/           # Tool registry, schemas, server logic
│   ├── forecast/       # Forecasting methods and engines
│   ├── patterns/       # Pattern detection algorithms
│   ├── services/       # MT5 data access layer
│   └── utils/          # Shared utilities
├── webui/              # React + Vite frontend
├── docs/               # Documentation
└── tests/              # Test suite
```

## License

MIT
