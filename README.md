# mtdata-mcp-server

`mtdata-mcp-server` is a Model Context Protocol (MCP) server + CLI that connects to a running MetaTrader 5 terminal and exposes market data, analytics, and (optionally) trading operations.

## What it does

- Market data: symbols, OHLCV candles, ticks, and DOM/market depth.
- Analytics: technical indicators, denoising/smoothing, forecasting, volatility estimation, and risk/barrier analytics.
- Workflows: a one-shot `report_generate` tool and step-by-step trading guides in `docs/`.

## Requirements

- Windows (MetaTrader 5 requirement)
- MetaTrader 5 terminal installed and running
- Python 3.10+

## Install

```bash
pip install -r requirements.txt
```

For editable installs (enables `mtdata-server` / `mtdata-cli` entrypoints):

```bash
pip install -e .
```

## Quick start

Start the server:

```bash
python server.py
# or: mtdata-server
```

Try a couple CLI calls:

```bash
python cli.py symbols_list --limit 20
python cli.py data_fetch_candles EURUSD --timeframe H1 --limit 300 --format json
python cli.py report_generate EURUSD --template basic
```

## Configuration

Create a `.env` file (optional) to preconfigure MT5 credentials:

```ini
MT5_LOGIN=12345678
MT5_PASSWORD=...
MT5_SERVER=...
```

If your MT5 server time is not UTC, set `MT5_TIME_OFFSET_MINUTES` (can be negative). All timestamps are normalized to UTC.

See `docs/SETUP.md` for setup and troubleshooting.

## Documentation

Start here: `docs/README.md`

- Trading workflows: `docs/SAMPLE-TRADE.md`, `docs/SAMPLE-TRADE-ADVANCED.md`, `docs/EXAMPLE.md`
- Forecasting: `docs/FORECAST.md`
- Barrier analytics: `docs/BARRIER_FUNCTIONS.md`
- Denoising: `docs/DENOISING.md`
- Technical indicators: `docs/TECHNICAL_INDICATORS.md`
- Troubleshooting: `docs/TROUBLESHOOTING.md`

## Notes

- This project is for research/automation tooling; it is not financial advice.
- Keep credentials out of git: copy `.env.example` to `.env` and do not commit it.

## License

MIT
