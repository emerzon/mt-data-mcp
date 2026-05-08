# Setup & Configuration

Use this guide to install mtdata, connect it to MetaTrader 5, and verify a safe read-only workflow before enabling any trading actions.

**Related:**
- [README.md](../README.md) — Project overview
- [ENV_VARS.md](ENV_VARS.md) — Complete environment variables reference
- [CLI.md](CLI.md) — Command usage
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) — Common issues

---

> Important: The MetaTrader 5 Python integration is **Windows-only**. If you're on macOS/Linux, run `mtdata` on a Windows VM or Windows machine and connect remotely (MCP/Web API).

## Requirements

- **Operating System:** Windows (required for MetaTrader 5)
- **Python:** 3.14
- **MetaTrader 5:** Installed and running
- **Windows Build Tools:** Visual Studio Build Tools 2022 with the **Desktop development with C++** workload for `pip install -r requirements.txt`, Git-backed extras, and optional native accelerators
- **Rust toolchain (optional):** Required only if you opt into the `tsdownsample` source-build path described below

---

## Recommended First-Run Path

1. Install the lean package with `pip install -e .`.
2. Confirm MT5 connectivity with `mtdata-cli symbols_list --limit 10`.
3. Set broker timezone configuration before relying on timestamps or backtests.
4. Run read-only examples first: symbols, candles, forecast, and report commands.
5. Use a demo account and `--dry-run true` before testing any `trade_*` command that supports it.

Choose the smallest install that fits your current task:

| Need | Install Path |
|------|--------------|
| CLI, MT5 data, indicators, core analysis | `pip install -e .` |
| Validated local research stack used by most docs | `pip install -r requirements.txt` |
| Web API / Web UI backend only | `pip install -e .[web]` |
| Heavy forecasting and optimization extras | `pip install -e .[forecast-classical]` and/or `pip install -e .[forecast-foundation]` |
| Git-backed experimental integrations | Install the specific Git-backed extra only when needed |

---

## Installation

### 1. Create an Isolated Conda Environment (Optional but Recommended)

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
NeuralForecast-based models are also kept out of this default path; if you want `nhits`, `tft`, `patchtst`, or `nbeatsx`, install them manually with `pip install neuralforecast torch`.

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
  - Lag-Llama (`lag_llama`): documented for completeness, but not part of the supported Python 3.14 environment because `gluonts`/Lag-Llama are still constrained by upstream compatibility
- Forecasting libraries: `statsforecast`, `sktime`, `mlforecast` (plus `lightgbm` for GBMs)
- Volatility (GARCH/ARCH): `arch`
- Optional pattern/simplification accelerators omitted from the default Python 3.14 install: `hnswlib`, `tsdownsample` (see the opt-in helper file `requirements-optional-src.txt` below)
- Barrier option pricing & Heston calibration: `QuantLib`
- Bayesian hyperparameter optimization: `optuna`
- Neural network forecasters (`nhits`, `tft`, `patchtst`, `nbeatsx`): manual install only via `pip install neuralforecast torch`; not included in `requirements.txt` or a package extra

### 5. Optional Native Accelerator Source-Build Path

Use this only if you explicitly want the extra accelerators that are omitted from the validated default Python 3.14 stack:

- `hnswlib` for the `hnsw` analog-search engine
- `tsdownsample` for faster LTTB simplification

Helper file:

```bash
pip install -r requirements-optional-src.txt
```

Notes:

- This path is opt-in and not part of the project's supported default Python 3.14 environment.
- On Python 3.14, pip may build these packages from source when a compatible wheel is unavailable.
- `hnswlib` is a C++ extension build. On Windows, install Visual Studio Build Tools 2022 with the **Desktop development with C++** workload first.
- `tsdownsample` is a Rust/maturin build. On Windows, install both the Visual Studio C/C++ build tools and a Rust toolchain via `rustup`.
- If you only want one accelerator, install it directly instead of the helper file:
  - `pip install hnswlib==0.8.0`
  - `pip install tsdownsample==0.1.4.1`

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

# Timezone Configuration (choose one server-time method)
MT5_SERVER_TZ=Europe/Athens  # Timezone name
# OR
MT5_TIME_OFFSET_MINUTES=120  # If server is UTC+2

# Optional trade guardrails
MTDATA_TRADE_GUARDRAILS_ENABLED=1
MTDATA_TRADE_ALLOWED_SYMBOLS=EURUSD,BTCUSD,XAUUSD
MTDATA_TRADE_MAX_VOLUME_BY_SYMBOL=EURUSD:0.50,BTCUSD:0.03
MTDATA_TRADE_MAX_RISK_PCT_OF_EQUITY=1.5
```

For the full guardrail surface, including blocklists, wallet-risk limits, and pending-order modification behavior, see [ENV_VARS.md](ENV_VARS.md#trade-guardrails).

### Timezone Configuration

MT5 server times vary by broker. Configure timezone for correct timestamp normalization.

Use one method at a time. `MT5_SERVER_TZ` is preferred because it handles DST; `MT5_TIME_OFFSET_MINUTES` is useful when you only know a fixed broker offset. If both are set and `MT5_TIME_OFFSET_MINUTES` is non-zero, the fixed offset wins and mtdata logs a warning because static offsets do not adjust for DST.

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

**How to determine your broker's timezone/offset (practical):**
1. Prefer `MT5_SERVER_TZ` if you know the broker's IANA timezone name (handles DST automatically).
2. If you don't know it, estimate an offset during active market hours (so ticks are current):
   ```bash
   python scripts/detect_mt5_time_offset.py --symbol EURUSD
   ```
   Then set `MT5_TIME_OFFSET_MINUTES` to the recommended value.

Set `MT5_SERVER_TZ` or `MT5_TIME_OFFSET_MINUTES` explicitly before starting `mtdata-webapi`; the Web API no longer mutates process environment state at startup.

What happens if it's wrong?
- Candle timestamps may be shifted, which can affect **daily pivots**, **session filters**, and **backtests**.

---

## Running mtdata

### CLI

```bash
# After `pip install -e .`
mtdata-cli <command> [options]
```

### MCP Server

mtdata supports three MCP transport modes:

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
- React UI: `http://localhost:8000/app`

Web UI / API configuration:
- `WEBAPI_HOST` (default `127.0.0.1`)
- `WEBAPI_PORT` (default `8000`)
- `WEBAPI_ALLOW_REMOTE=1` to permit a non-loopback bind
- `WEBAPI_AUTH_TOKEN` to require `Authorization: Bearer <token>` or `X-API-Key: <token>` on API requests
- `CORS_ORIGINS` with explicit origins only (wildcard `*` is rejected when credentials are enabled)
- `WEBUI_DIST_DIR` to override the built UI directory

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
│   │   ├── reports/    # Shared report helpers
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

1. Check server time in MT5: Tools → Options → Server
2. Set `MT5_SERVER_TZ` in `.env` when you know the broker's IANA timezone, or `MT5_TIME_OFFSET_MINUTES` when you only know a fixed offset
3. Verify with: `mtdata-cli data_fetch_candles EURUSD --limit 1 --json`

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for more issues.

---

## Next Steps

- [CLI.md](CLI.md) — Learn command usage
- [EXAMPLE.md](EXAMPLE.md) — Follow an end-to-end workflow
- [GLOSSARY.md](GLOSSARY.md) — Understand terminology
- [LIMITATIONS.md](LIMITATIONS.md) — Review practical caveats before deeper integrations
- [FINVIZ.md](FINVIZ.md) — Fundamental data and screening
- [TEMPORAL.md](TEMPORAL.md) — Session and seasonal analysis

