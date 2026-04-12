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
| **Data** | Fetch candles, ticks, market depth, and ranked market scans from MT5 | `data_fetch_candles`, `data_fetch_ticks`, `market_depth_fetch`, `market_ticker`, `symbols_top_markets` |
| **Forecasting** | Predict price paths with classical, ML, or foundation models | `forecast_generate`, `forecast_backtest_run` |
| **Volatility** | Estimate future price movement magnitude | `forecast_volatility_estimate` |
| **Regimes** | Detect trending, ranging, or crisis market states | `regime_detect` |
| **Barriers** | Calculate TP/SL hit probabilities via simulation | `forecast_barrier_prob`, `forecast_barrier_optimize` |
| **Patterns** | Identify candlestick, chart, Elliott, and fractal patterns | `patterns_detect` |
| **Indicators** | Compute 100+ technical indicators | `data_fetch_candles --indicators` |
| **Denoising** | Smooth price data to reveal trends | `--denoise` option |
| **Temporal** | Discover session effects and seasonal patterns | `temporal_analyze` |
| **Trading** | Place orders, manage positions | `trade_place`, `trade_close` |
| **Fundamentals** | US equity data, screening, news, calendars | `finviz_fundamentals`, `finviz_screen`, `finviz_calendar` |
| **Options** | Options chains and QuantLib barrier pricing | `forecast_options_chain`, `forecast_quantlib_barrier_price` |

## Quick Start

**Prerequisites:** Windows + Python 3.14 + MetaTrader 5 installed and running (demo account recommended). For the full install on Windows, also install Visual Studio Build Tools 2022 with the **Desktop development with C++** workload.

```bash
# Optional but recommended: create an isolated conda environment first
conda create -n mtdata python=3.14 -y
conda activate mtdata

# Lean core install
pip install -e .

# Full stable research/web install
pip install -r requirements.txt

# Verify MT5 connection (lists symbols from the running terminal)
mtdata-cli symbols_list --limit 5

# Scan the current MT5 watchlist for top markets
mtdata-cli symbols_top_markets --rank-by all --limit 5 --timeframe H1

# Fetch recent candles
mtdata-cli data_fetch_candles EURUSD --timeframe H1 --limit 50

# Generate a baseline price forecast
mtdata-cli forecast_generate EURUSD --timeframe H1 --horizon 12 --method theta
```

Notes:
- `pip install -e .` now installs the lean core package only.
- `pip install -r requirements.txt` installs the validated Python 3.14 stack from package-index releases, including Chronos, StatsForecast, sktime, mlforecast, news embeddings, and the Web API.
- NeuralForecast-based models (`nhits`, `tft`, `patchtst`, `nbeatsx`) are not installed by `requirements.txt` or any package extra today; install them manually with `pip install neuralforecast torch` if you want to experiment with them.
- Conda is a supported way to isolate the install before running the pip commands above.
- Git-backed add-ons stay explicit: `pip install -e .[forecast-timesfm]` for TimesFM, `pip install -e .[patterns-ext]` for `stock-pattern`, `pip install -e .[news-ycnbc]` for the CNBC adapter, or `pip install -e .[all-git]` for everything in one go.
- GluonTS/Lag-Llama and GluonTS `gt_*` methods remain excluded from the supported Python 3.14 environment because upstream runtime constraints are still incompatible.
- Optional accelerators `hnswlib` and `tsdownsample` remain excluded from the supported default install, but an opt-in native/source-build path is documented via `requirements-optional-src.txt` and [docs/SETUP.md](docs/SETUP.md).

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
- **[Temporal Analysis](docs/TEMPORAL.md)** — Session effects, day-of-week, and seasonal patterns

### External Data & Options
- **[Finviz Fundamentals](docs/FINVIZ.md)** — US equity data, screening, news, calendars
- **[Options & QuantLib](docs/OPTIONS_QUANTLIB.md)** — Options chains, barrier pricing, Heston calibration

### Tutorials
- **[Sample Trade Workflow](docs/SAMPLE-TRADE.md)** — Step-by-step analysis for a trade decision
- **[Advanced Playbook](docs/SAMPLE-TRADE-ADVANCED.md)** — Regime filters, conformal intervals, barrier optimization
- **[End-to-End Example](docs/EXAMPLE.md)** — Complete research loop with all tools

### Reference
- **[Environment Variables](docs/ENV_VARS.md)** — Complete `.env` reference (MT5, MCP, Web API, GPU, etc.)
- **[Web API](docs/WEB_API.md)** — REST endpoints for the Web UI and integrations
- **[Troubleshooting](docs/TROUBLESHOOTING.md)** — Common issues and fixes

## Configuration

Create a `.env` file in the project root:

```ini
MT5_LOGIN=12345678
MT5_PASSWORD=your_password
MT5_SERVER=your_broker_server
MT5_SERVER_TZ=Europe/Athens   # Or use MT5_TIME_OFFSET_MINUTES=120
```

mtdata reads dozens of environment variables covering MT5 connection, timezone, MCP/Web API server settings, news embeddings, Finviz tuning, GPU acceleration, and more. See **[Environment Variables Reference](docs/ENV_VARS.md)** for the full list and a starter `.env` template.

## Architecture

```
mtdata/
├── requirements-optional-src.txt  # Opt-in native/source-built accelerators
├── src/mtdata/
│   ├── bootstrap/      # Runtime startup, settings, tool loading
│   ├── core/           # Tool registry, schemas, server logic, all 57 MCP tools
│   │   ├── cli/        # Dynamic CLI (argparse, formatting, parsing, runtime)
│   │   ├── regime/     # Regime detection (HMM, BOCPD, MS-AR)
│   │   └── report_templates/  # Report generation templates
│   ├── forecast/       # Forecasting methods, engines, and method registry
│   ├── patterns/       # Pattern detection algorithms
│   ├── services/       # MT5 data access, Finviz, options data
│   ├── shared/         # Shared constants, schemas, validators
│   └── utils/          # Shared utilities (indicators, denoising, etc.)
├── webui/              # React + Vite frontend
├── docs/               # Documentation
└── tests/              # Test suite
```

## License

MIT
