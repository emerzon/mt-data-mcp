# CLI Guide

**Related documentation:**
- [README.md](../README.md) - Project overview
- [SETUP.md](SETUP.md) - Installation and MT5 setup
- [EXAMPLE.md](EXAMPLE.md) - End-to-end workflow

The CLI is a local convenience wrapper around the server tools. It’s the quickest way to explore capabilities and copy-paste runnable examples.

## Basics

Show all commands:

```bash
python cli.py --help
```

Show command-specific help:

```bash
python cli.py data_fetch_candles --help
python cli.py forecast_generate --help
```

Search help by keyword:

```bash
python cli.py --help forecast
python cli.py --help denoise
```

## Output formats

- Default output is compact “text” meant for humans.
- For structured output, add `--format json`.

Example:

```bash
python cli.py symbols_describe EURUSD --format json
```

## Common conventions

- `symbol` is usually a positional argument (e.g., `EURUSD`).
- `timeframe` is an option on most tools (e.g., `--timeframe H1`).
- Many tools accept `--params` as “JSON or `k=v`” for advanced configuration.

## Useful discovery commands

Forecast methods (what’s available on your machine):

```bash
python cli.py forecast_list_methods --format json
python cli.py forecast_list_library_models native --format json
python cli.py forecast_list_library_models statsforecast --format json
python cli.py forecast_list_library_models sktime --format json
python cli.py forecast_list_library_models pretrained --format json
```

Technical indicators:

```bash
python cli.py indicators_list --limit 50
python cli.py indicators_list --category momentum --limit 50
python cli.py indicators_describe rsi --format json
```

## Date inputs

Some tools accept `--start` / `--end` and parse flexible date strings via `dateparser`, for example:

```bash
python cli.py data_fetch_candles EURUSD --timeframe H1 --start "2 days ago" --end "now" --format json
```

