# Setup & Configuration

Get mtdata installed, talking to MetaTrader 5, and through a **safe first workflow** — all before you enable any trading.

You do not need every optional dependency on day one. Start lean, confirm candles work, then add forecasting or web extras as you need them.

**Related:** [README](../README.md) · [Env vars](ENV_VARS.md) · [CLI](CLI.md) · [Troubleshooting](TROUBLESHOOTING.md)

---

> **Platform:** The MetaTrader 5 Python integration is **Windows-only**. On macOS/Linux, run mtdata on a Windows machine or VM and connect remotely (MCP or Web API).

## What you need

| Requirement | Detail |
|-------------|--------|
| **OS** | Windows (required for MT5) |
| **Python** | **3.14** |
| **MetaTrader 5** | Installed, running, and logged in (demo recommended) |
| **Build tools** | Visual Studio Build Tools 2022 with **Desktop development with C++** for the full install, Git-backed extras, and optional native builds |

## Recommended first run

1. Install the lean package: `pip install -e .`
2. Confirm MT5 connectivity: `mtdata-cli symbols_list --limit 10`
3. Set `CLIENT_TZ=UTC` for deterministic timestamp presentation
4. Stay **read-only** at first: symbols, candles, forecast, report
5. Only then try `trade_*` — on a **demo** account, with `--dry-run true` when the command supports it

### Install path cheatsheet

| Need | Install |
|------|---------|
| CLI, MT5 data, indicators, core analysis | `pip install -e .` |
| Full research stack used by most docs | `pip install -r requirements.txt` |
| Web API / UI backend only | `pip install -e .[web]` |
| Heavy forecast extras | `pip install -e .[forecast-classical]` and/or `pip install -e .[forecast-foundation]` |
| Git-backed experiments (TimesFM, etc.) | Install that extra only when you need it |

---

## Installation

### 1. Create an isolated conda environment (optional but recommended)

If you use Conda/Miniconda, start with a clean Python 3.14 environment:

```bash
conda create -n mtdata python=3.14 -y
conda activate mtdata
python --version
```

If you prefer not to use conda, make sure the active interpreter is Python 3.14 before continuing.

### 2. Install the Lean Core Package

From the repository root:

```bash
pip install -e .
```

### 3. Install the Full Stable Feature Set (Optional)

For the validated research/web environment used in local development and most docs/examples:

```bash
pip install -r requirements.txt
```

This path intentionally stays on package-index releases. Git-backed add-ons such as TimesFM, `stock-pattern`, and `ycnbc` stay opt-in so the default install does not depend on Git checkouts.
NeuralForecast-based models are also kept out of this default path; current Windows Python 3.14 resolution can fail on transitive dependencies, so treat `nhits`, `tft`, `patchtst`, and `nbeatsx` as manual/nonstandard setup.

### 4. Optional Dependencies

The base package is intentionally lean. Install extras as needed:

> Windows note: install Visual Studio Build Tools 2022 with the **Desktop development with C++** workload before the full install or any Git-backed extra. Several optional dependencies include native extensions, and pip may need MSVC when a compatible wheel is unavailable. Git-backed extras also require Git on `PATH`.

- Classical forecasting / optimization:
  `pip install -e .[forecast-classical]`
- Foundation-model forecasting (Chronos / Chronos-Bolt):
  `pip install -e .[forecast-foundation]`
- TimesFM (Git-backed):
  `pip install -e .[forecast-timesfm]`
- Web API:
  `pip install -e .[web]`
- Experimental pattern engines (Git-backed, requires manual install):
  `pip install -e .[patterns-ext]` (Note: stock-pattern requires manual copy to site-packages; see below)
- News embeddings (semantic reranking):
  `pip install -e .[news-embeddings]`
- CNBC news source (Git-backed):
  `pip install -e .[news-ycnbc]`
- Everything from package indexes:
  `pip install -e .[all]`
- Everything including Git-backed extras:
  `pip install -e .[all-git]`

Feature notes:

- Causal discovery (`causal_discover_signals`) and classical ARIMA/ETS: `statsmodels`
- Wavelet denoising: `PyWavelets`
- Dimred UMAP (Web UI / analysis): `umap-learn`
- Foundation models:
  - Chronos (`chronos2`, `chronos_bolt`): `chronos-forecasting`, `torch`
  - TimesFM (`timesfm`): `timesfm`, `torch` (install with `pip install -e .[forecast-timesfm]`; Git-backed extra)
  - GluonTS / Lag-Llama are not shipped in mtdata because they are unsupported on the project's Python 3.14 runtime
- Forecasting libraries: `statsforecast`, `sktime`, `mlforecast` (plus `lightgbm` for GBMs)
- Volatility (GARCH/ARCH): `arch`
- Simplification accelerator: `tsdownsample` is included in the full package-index install path (`requirements.txt` / `[all]`) and remains optional for lean installs
- Optional pattern-search accelerator omitted from the default Python 3.14 install: `hnswlib` (see the opt-in helper file `requirements-optional-src.txt` below)
- Barrier option pricing & Heston calibration: `QuantLib`
- Bayesian hyperparameter optimization: `optuna`
- Neural network forecasters (`nhits`, `tft`, `patchtst`, `nbeatsx`): manual/nonstandard setup only; not included in `requirements.txt` or a package extra

### 5. Optional Native Accelerator Source-Build Path

Use this only if you explicitly want the extra accelerator that is omitted from the validated default Python 3.14 stack:

- `hnswlib` for the `hnsw` analog-search engine

Helper file:

```bash
pip install -r requirements-optional-src.txt
```

Notes:

- This path is opt-in and not part of the project's supported default Python 3.14 environment.
- On Python 3.14, pip builds `hnswlib` from source because a compatible Windows wheel is unavailable.
- `hnswlib` is a C++ extension build. On Windows, install Visual Studio Build Tools 2022 with the **Desktop development with C++** workload first.
- If you only want this accelerator, install it directly instead of the helper file:
  - `pip install hnswlib==0.8.0`
- If you want the LTTB simplification accelerator without the full `[all]` extra, install it directly with `pip install "tsdownsample>=0.1.5"`.

Tip: `mtdata-cli forecast_list_methods --json` shows `available` and `requires` per method.

### 6. Manual stock-pattern Installation

The `stock-pattern` library (used by `patterns-ext`) does not have a `setup.py` or `pyproject.toml`, so it cannot be installed via pip. Install manually:

**Windows:**
```powershell
# Clone and copy to site-packages
git clone --depth 1 https://github.com/BennyThadikaran/stock-pattern.git $env:TEMP\stock-pattern-src
robocopy $env:TEMP\stock-pattern-src\src $env:CONDA_PREFIX\Lib\site-packages\stock_pattern /E
```

**Linux/macOS:**
```bash
# Clone and copy to site-packages
git clone --depth 1 https://github.com/BennyThadikaran/stock-pattern.git /tmp/stock-pattern-src
cp -r /tmp/stock-pattern-src/src/* $(python -c "import site; print(site.getsitepackages()[0])")/stock_pattern/
```

Verify installation:
```python
python -c "import stock_pattern.utils; print('stock_pattern installed')"
```

---

## MetaTrader 5 Setup

### 1. Install MetaTrader 5

Download from your broker or [MetaQuotes](https://www.metatrader5.com/en/download).

### 2. Launch and Login

1. Start the MetaTrader 5 terminal
2. Log in to your broker account (demo account works and is recommended for first use)
3. Keep the terminal running while using mtdata

If you don't have a broker account yet, you can still get started:
- In MT5: **File → Open an Account** → choose a demo provider (often **MetaQuotes-Demo**) → create a demo account.
- Confirm prices are updating in **Market Watch** (this avoids “stale”/empty data).

### 3. Verify Connection

```bash
mtdata-cli symbols_list --limit 10
```

Optional deeper check:
```bash
mtdata-cli trade_account_info --json
```

Expected output:
```
data[10]{name,group,description}:
    EURUSD,Forex\Majors,Euro vs US Dollar
    GBPUSD,Forex\Majors,Great Britain Pound vs US Dollar
    ...
```

If you don't see symbols (or you get a connection error):
- Make sure MT5 is **running** and **logged in**
- Make sure the symbol is visible in **Market Watch** (right-click → “Show All”)
- If you have multiple MT5 terminals installed, close extras and retry
- See [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

---

## Environment Variables

> **Full reference:** [ENV_VARS.md](ENV_VARS.md) documents the full environment-variable surface (MCP server, Web API, news embeddings, Finviz, GPU, market depth, CLI debug, and more) with a starter `.env` template.

Create a `.env` file in the project root for configuration:

```ini
# MT5 Credentials (optional - for unattended login)
MT5_LOGIN=12345678
MT5_PASSWORD=your_password
MT5_SERVER=your_broker_server

# Deterministic timestamp presentation
CLIENT_TZ=UTC

# Optional broker session/calendar timezone
MT5_SERVER_TZ=Europe/Athens

# Optional trade guardrails
MTDATA_TRADE_GUARDRAILS_ENABLED=1
MTDATA_TRADE_ALLOWED_SYMBOLS=EURUSD,BTCUSD,XAUUSD
MTDATA_TRADE_MAX_VOLUME_BY_SYMBOL=EURUSD:0.50,BTCUSD:0.03
MTDATA_TRADE_MAX_RISK_PCT_OF_EQUITY=1.5
```

For the full guardrail surface, including blocklists, wallet-risk limits, and pending-order modification behavior, see [ENV_VARS.md](ENV_VARS.md#trade-guardrails).

### Timezone Configuration

MT5 Python request datetimes and returned epochs are UTC. Broker timezone
settings do not normalize or shift those absolute timestamps. Set `CLIENT_TZ`
to control presentation, and retain the payload `timezone` field with saved data.

`MT5_SERVER_TZ` and `MT5_TIME_OFFSET_MINUTES` are optional broker-session
settings. Use one method when broker-local calendar boundaries matter.
`MT5_SERVER_TZ` is preferred because it handles DST. If both are set and
`MT5_TIME_OFFSET_MINUTES` is non-zero, the fixed session offset wins.

**Option 1: Offset in minutes**
```ini
MT5_TIME_OFFSET_MINUTES=120  # Server is UTC+2 (e.g., Eastern European)
MT5_TIME_OFFSET_MINUTES=-240 # Server is UTC-4 (e.g., Eastern US)
```

**Option 2: Timezone name**
```ini
MT5_SERVER_TZ=Europe/Athens
MT5_SERVER_TZ=America/New_York
```

An incorrect broker-session setting can misclassify broker-local daily pivots or
session filters, but it does not change candle, tick, order, or deal epochs.

---

## Running mtdata

Same toolkit, three surfaces — pick what fits the job.

### CLI

```bash
# After `pip install -e .`
mtdata-cli <command> [options]
```

### MCP server

Three transports for AI clients and custom hosts:

```bash
# SSE transport (default) — for browser/HTTP-based MCP clients
mtdata-sse

# stdio transport — for IDE integrations (Claude Desktop, VS Code, etc.)
mtdata-stdio

# Streamable HTTP transport
mtdata-streamable-http
```

After editable install, these entry points are available:

| Entry Point | Transport | Usage |
|-------------|-----------|-------|
| `mtdata-sse` | SSE (default) | `mtdata-sse` |
| `mtdata-stdio` | stdio | `mtdata-stdio` |
| `mtdata-streamable-http` | Streamable HTTP | `mtdata-streamable-http` |
| `mtdata-webapi` | Web API (FastAPI) | `mtdata-webapi` |
| `mtdata-cli` | CLI | `mtdata-cli <command>` |

**MCP server environment variables:**

| Variable | Default | Description |
|----------|---------|-------------|
| `MCP_TRANSPORT` | `sse` | Transport mode: `sse`, `stdio`, `streamable-http` |
| `FASTMCP_HOST` | `127.0.0.1` | Bind address |
| `FASTMCP_ALLOW_REMOTE` | `0` | Set to `1` to allow non-loopback binds such as `0.0.0.0` |
| `FASTMCP_PORT` | `8000` | Listen port |
| `FASTMCP_MOUNT_PATH` | `/` | Mount path |
| `FASTMCP_SSE_PATH` | `/sse` | SSE event stream path |
| `FASTMCP_MESSAGE_PATH` | `/message` | Message endpoint path |
| `FASTMCP_LOG_LEVEL` | (default) | Logging level |

**Claude Desktop configuration example** (`claude_desktop_config.json`):
```json
{
  "mcpServers": {
    "mtdata": {
      "command": "mtdata-stdio"
    }
  }
}
```

### Web API

```bash
mtdata-webapi
```

Starts a FastAPI server.

- Health check: `http://localhost:8000/health`
- API base paths: `http://localhost:8000/api` and `http://localhost:8000/api/v1`
- React UI after `cd webui && npm install && npm run build`:
  `http://localhost:8000/app`

Web UI / API configuration:
- `WEBAPI_HOST` (default `127.0.0.1`)
- `WEBAPI_PORT` (default `8000`)
- `WEBAPI_ALLOW_REMOTE=1` to permit a non-loopback bind
- `WEBAPI_AUTH_TOKEN` to require `Authorization: Bearer <token>` or `X-API-Key: <token>` on API requests
- `CORS_ORIGINS` with explicit origins only (wildcard `*` is rejected when credentials are enabled)
- `WEBUI_DIST_DIR` to override the built UI directory

The Python package does not ship generated `webui/dist/` assets. Without a
production build, the REST API remains available and `/app` is not mounted; use
`npm run dev` for frontend development or `npm run build` before deployment.

See [WEB_API.md](WEB_API.md) for endpoint details.

---

## Verifying Installation

Run these commands to verify everything works:

```bash
# List symbols
mtdata-cli symbols_list --limit 5

# Get symbol details
mtdata-cli symbols_describe EURUSD --json

# Fetch candles
mtdata-cli data_fetch_candles EURUSD --timeframe H1 --limit 100

# List forecast methods
mtdata-cli forecast_list_methods

# Generate a forecast
mtdata-cli forecast_generate EURUSD --timeframe H1 --horizon 12 --method theta
```

---

## Project Structure

```
mtdata/
├── requirements.txt    # Python dependencies
├── requirements-optional-src.txt  # Opt-in native/source-built accelerators
├── pyproject.toml      # Package configuration
├── .env                # Local configuration (create this)
├── src/mtdata/
│   ├── bootstrap/      # Runtime startup, settings, tool loading
│   ├── core/           # Tool registry, server, CLI logic, all MCP tool modules
│   │   ├── cli/        # Dynamic CLI (argparse) + parsing/, runtime/ subpackages
│   │   ├── data/       # `data_fetch_*` and `wait_event` tools
│   │   ├── regime/     # Regime detection (HMM, BOCPD, MS-AR)
│   │   ├── report/     # `report_generate` runtime
│   │   ├── report_templates/  # Per-style report templates
│   │   └── trading/    # `trade_*` tools and account / risk modules
│   ├── forecast/       # Forecasting methods
│   ├── patterns/       # Pattern detection
│   ├── services/       # MT5 data access, Finviz, options data
│   ├── shared/         # Shared constants, schemas, validators
│   └── utils/          # Shared utilities (indicators, denoising, etc.)
├── webui/              # React frontend
├── docs/               # Documentation
└── tests/              # Test suite
```

---

## Development Setup

### Running Tests

```bash
# Install test dependencies
pip install pytest

# Run tests
python -m pytest tests/
```

### Web UI Development

```bash
cd webui
npm install
npm run dev     # Development server
npm run build   # Production build
```

---

## Troubleshooting Setup

### "Could not connect to MT5"

1. Ensure MT5 terminal is running
2. Ensure you're logged in
3. Check if MT5 is in portable mode (may need standard installation)

### Import Errors

1. Verify Python version: `python --version` (need 3.14)
2. Reinstall dependencies: `pip install -r requirements.txt`
3. Try editable install: `pip install -e .`

### Timezone Issues

1. Check the payload's `timezone` field
2. Set `CLIENT_TZ=UTC` for deterministic presentation
3. Verify with: `mtdata-cli data_fetch_candles EURUSD --limit 1 --json`

Configure `MT5_SERVER_TZ` only for broker-local session/calendar calculations.

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for more issues.

---

## Next steps

Once candles list and a simple forecast run:

1. [CLI.md](CLI.md) — Commands, help, and output formats
2. [GLOSSARY.md](GLOSSARY.md) — Terms used across the docs
3. [SAMPLE-TRADE.md](SAMPLE-TRADE.md) — Guided research workflow
4. [EXAMPLE.md](EXAMPLE.md) — Compact end-to-end research loop
5. [LIMITATIONS.md](LIMITATIONS.md) — Caveats before deep integrations

Also useful later: [FINVIZ.md](FINVIZ.md) · [TEMPORAL.md](TEMPORAL.md) · [FORECAST.md](FORECAST.md)
