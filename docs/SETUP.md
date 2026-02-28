# Setup & Configuration

Installation and configuration guide for mtdata.

**Related:**
- [README.md](../README.md) — Project overview
- [CLI.md](CLI.md) — Command usage
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) — Common issues

---

> Important: The MetaTrader 5 Python integration is **Windows-only**. If you're on macOS/Linux, run `mtdata` on a Windows VM or Windows machine and connect remotely (MCP/Web API).

## Requirements

- **Operating System:** Windows (required for MetaTrader 5)
- **Python:** 3.14
- **MetaTrader 5:** Installed and running

---

## Installation

### 1. Install Python Dependencies

From the repository root:

```bash
pip install -r requirements.txt
```

### 2. Editable Install (Optional)

For development or to use entry points (`mtdata-server`, `mtdata-cli`):

```bash
pip install -e .
```

### 3. Optional Dependencies

`requirements.txt` is the supported Python 3.14 dependency set. If you are installing selectively, these features require extra packages:

- Causal discovery (`causal_discover_signals`) and classical ARIMA/ETS: `statsmodels`
- Wavelet denoising: `PyWavelets`
- Dimred UMAP (Web UI / analysis): `umap-learn`
- Foundation models:
  - Chronos (`chronos2`, `chronos_bolt`): `chronos-forecasting`, `torch`
  - TimesFM (`timesfm`): `timesfm`, `torch` (installed from Git in `requirements.txt`)
  - Lag-Llama (`lag_llama`): not included in the default Python 3.14 environment because `gluonts`/Lag-Llama are still constrained by upstream compatibility
- Forecasting libraries: `statsforecast`, `sktime`, `mlforecast` (plus `lightgbm` for GBMs)
- Volatility (GARCH/ARCH): `arch`
- Optional pattern/simplification accelerators omitted from the default Python 3.14 install: `hnswlib`, `tsdownsample`

Tip: `python cli.py forecast_list_methods --format json` shows `available` and `requires` per method.

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
python cli.py symbols_list --limit 10
```

Optional deeper check:
```bash
python cli.py trade_account_info --format json
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
- See [TROUBLESHOOTING.md](TROUBLESHOOTING.md#connection-issues)

---

## Environment Variables

Create a `.env` file in the project root for configuration:

```ini
# MT5 Credentials (optional - for unattended login)
MT5_LOGIN=12345678
MT5_PASSWORD=your_password
MT5_SERVER=your_broker_server

# Timezone Configuration (choose one method)
MT5_TIME_OFFSET_MINUTES=120  # If server is UTC+2
# OR
MT5_SERVER_TZ=Europe/Athens  # Timezone name
```

### Timezone Configuration

MT5 server times vary by broker. Configure timezone for correct timestamp normalization.

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

Tip: `python webui.py` will attempt to auto-detect and apply `MT5_TIME_OFFSET_MINUTES` at startup if neither `MT5_SERVER_TZ` nor `MT5_TIME_OFFSET_MINUTES` is set. This is best-effort and may return 0 when the market is closed.

What happens if it's wrong?
- Candle timestamps may be shifted, which can affect **daily pivots**, **session filters**, and **backtests**.

---

## Running mtdata

### CLI

```bash
# Direct execution
python cli.py <command> [options]

# After editable install
mtdata-cli <command> [options]
```

### MCP Server

```bash
# Direct execution
python server.py

# After editable install
mtdata-server
```

### Web API

```bash
python webui.py
```

Starts a FastAPI server with a React UI at `http://localhost:8000`.

Web UI / API configuration:
- `MTDATA_WEBUI_HOST` (default `127.0.0.1`)
- `MTDATA_WEBUI_PORT` (default `8000`)
- `MTDATA_WEBUI_RELOAD=1` for auto-reload (dev only)

See [WEB_API.md](WEB_API.md) for endpoint details.

---

## Verifying Installation

Run these commands to verify everything works:

```bash
# List symbols
python cli.py symbols_list --limit 5

# Get symbol details
python cli.py symbols_describe EURUSD --format json

# Fetch candles
python cli.py data_fetch_candles EURUSD --timeframe H1 --limit 100

# List forecast methods
python cli.py forecast_list_methods

# Generate a forecast
python cli.py forecast_generate EURUSD --timeframe H1 --horizon 12 --model theta
```

---

## Project Structure

```
mtdata/
├── cli.py              # CLI entry point
├── server.py           # MCP server entry point
├── webui.py            # Web API entry point
├── requirements.txt    # Python dependencies
├── pyproject.toml      # Package configuration
├── .env                # Local configuration (create this)
├── src/mtdata/
│   ├── core/           # Tool registry, server, CLI logic
│   ├── forecast/       # Forecasting methods
│   ├── patterns/       # Pattern detection
│   ├── services/       # MT5 data access
│   └── utils/          # Shared utilities
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
2. Set `MT5_TIME_OFFSET_MINUTES` in `.env`
3. Verify with: `python cli.py data_fetch_candles EURUSD --limit 1 --format json`

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for more issues.

---

## Next Steps

- [CLI.md](CLI.md) — Learn command usage
- [EXAMPLE.md](EXAMPLE.md) — Follow an end-to-end workflow
- [GLOSSARY.md](GLOSSARY.md) — Understand terminology
