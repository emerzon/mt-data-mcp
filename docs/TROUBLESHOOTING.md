# Troubleshooting

**Related documentation:**
- [SETUP.md](SETUP.md) - Installation and MT5 setup
- [CLI.md](CLI.md) - CLI conventions and help
- [EXAMPLE.md](EXAMPLE.md) - Working command examples

This page collects the most common issues when running the server/CLI and how to resolve them.

## “Could not connect to MT5” / empty data

Checklist:

1. Ensure the MetaTrader 5 terminal is installed and running.
2. Make sure the terminal is logged in (or provide credentials in `.env`).
3. Try a minimal command:

```bash
python cli.py symbols_list --limit 10
```

If symbol listing works but candles fail, confirm the symbol exists and is visible in Market Watch.

## Validation errors (missing required fields)

Example error:

```
1 validation error for InputSchema
symbol
  Field required [type=missing, input_value={}, input_type=dict]
```

What it means: a tool needs a required parameter and you didn’t pass it.

Fix:

- Use command help to see required arguments:

```bash
python cli.py data_fetch_candles --help
```

- Many tools require `symbol` as a positional argument, for example:

```bash
python cli.py data_fetch_candles EURUSD --timeframe H1 --limit 200
```

## “Invalid choice” for timeframe/mode/method

Many tools restrict values (timeframes, modes, methods). Use `--help` to see allowed values:

```bash
python cli.py forecast_volatility_estimate --help
python cli.py patterns_detect --help
```

## Output is hard to parse

Use JSON output:

```bash
python cli.py symbols_describe EURUSD --format json
python cli.py report_generate EURUSD --template basic --format json
```

## Getting unstuck quickly

- Run `python cli.py --help <keyword>` to search for a command by topic:

```bash
python cli.py --help forecast
python cli.py --help barrier
python cli.py --help indicators
```

