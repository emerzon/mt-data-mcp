# Setup & Configuration

**Related documentation:**
- [README.md](../README.md) - Project overview
- [CLI.md](CLI.md) - CLI usage patterns
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Common issues

This project runs on top of a local MetaTrader 5 terminal. The server attaches to the terminal and exposes tools over MCP; `cli.py` is a convenient local way to call those tools.

## Requirements

- Windows (MetaTrader 5 requirement)
- MetaTrader 5 terminal installed
- Python 3.10+

## Install

From the repository root:

```bash
pip install -r requirements.txt
```

Optional (recommended for development): install editable so the `mtdata-server` / `mtdata-cli` entrypoints are available.

```bash
pip install -e .
```

## MetaTrader 5 terminal

1. Launch the MetaTrader 5 terminal.
2. Log in to your broker account (optional if you already stay logged in).
3. Keep the terminal running while you use the server/CLI.

## Environment variables (`.env`)

Create `.env` in the repo root (optional). This is mainly for unattended runs.

```ini
MT5_LOGIN=12345678
MT5_PASSWORD=...
MT5_SERVER=...
```

### Timezone normalization

If your broker server time is not UTC, set:

```ini
MT5_TIME_OFFSET_MINUTES=120
```

- Positive values mean “server time is ahead of UTC” (UTC+2 = `120`).
- Negative values mean “server time is behind UTC” (UTC-4 = `-240`).
- The server normalizes timestamps to UTC using this offset.

## Run

Start the server:

```bash
python server.py
# or: mtdata-server
```

Run a couple sanity checks:

```bash
python cli.py symbols_list --limit 20
python cli.py symbols_describe EURUSD --format json
python cli.py data_fetch_candles EURUSD --timeframe H1 --limit 300 --format json
```

