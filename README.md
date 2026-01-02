# mtdata

**mtdata** is a research and automation toolkit for MetaTrader 5 (MT5). It provides tools for forecasting, regime detection, signal processing, risk analysis, and reporting—bridging raw market data with data science.

It runs as a **Model Context Protocol (MCP)** server or a standalone **CLI**.

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
| **Trading** | Place orders, manage positions | `trading_place`, `trading_close` |

## Quick Start

**Prerequisites:** MetaTrader 5 installed and running (demo account works).

```bash
# Install dependencies
pip install -r requirements.txt

# List available symbols
python cli.py symbols_list --limit 10

# Generate a price forecast
python cli.py forecast_generate EURUSD --timeframe H1 --horizon 12 --model theta

# Detect market regime
python cli.py regime_detect EURUSD --timeframe H1 --method hmm --params "n_states=2"

# Estimate volatility
python cli.py forecast_volatility_estimate EURUSD --timeframe H1 --horizon 12 --method ewma
```

## Documentation

### Getting Started
- **[Setup & Configuration](docs/SETUP.md)** — Installation, MT5 connection, environment variables
- **[CLI Guide](docs/CLI.md)** — Command conventions, output formats, help system
- **[Glossary](docs/GLOSSARY.md)** — Explanations of all technical terms with real-world examples

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
