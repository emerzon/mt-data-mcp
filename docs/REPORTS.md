# Report generation

`report_generate` packages several read-only analysis steps into one structured
market summary. Use it when you want a repeatable overview rather than calling
context, forecast, level, pattern, barrier, and regime tools separately.

Reports are research output, not trade instructions. A report can also be
partial when a provider, optional dependency, or sub-analysis is unavailable;
inspect its section statuses and diagnostics before relying on it.

**Related:** [CLI](CLI.md) · [Output contract](OUTPUT.md) · [Forecasting](FORECAST.md) · [Regimes](forecast/REGIMES.md) · [Barriers](BARRIER_FUNCTIONS.md)

---

## Quick start

```bash
mtdata-cli report_generate EURUSD --timeframe H1 --template basic
```

The CLI defaults to compact TOON text. Use `--json` for a machine-readable
payload or `--detail full` when you need the complete supported report content.
CLI and MCP preserve the same canonical report payload; output format only
changes its final presentation.

## Choose a template

| Template | Intended use |
|----------|--------------|
| `minimal` | Fast context and direct forecast path |
| `basic` | Balanced general-purpose report (default) |
| `advanced` | Adds regime, HAR-RV, and conformal analysis |
| `scalping` | Short-horizon M5 setup |
| `intraday` | H1 setup |
| `swing` | H4/D1 setup |
| `position` | D1/W1 setup |

`minimal` is the quick path. The other templates may perform several MT5
fetches and invoke pivots, patterns, backtests, barriers, or regime checks.
Runtime and dependency requirements therefore vary by template. Section
filters are applied to the completed report, so they reduce output size rather
than computation.

## Control template, scope, and output

```bash
# Fast overview
mtdata-cli report_generate EURUSD --template minimal --timeframe H1

# Style-specific report with an explicit forecast horizon
mtdata-cli report_generate EURUSD --template swing --timeframe H4 --horizon 12

# Keep only selected computed sections
mtdata-cli report_generate EURUSD --template basic \
  --include-sections context,forecast,barriers --max-sections 3 --json

# Restrict candidate forecast methods and apply denoising
mtdata-cli report_generate EURUSD --template basic \
  --methods theta,arima --denoise kalman --json
```

Useful controls:

- `--horizon` sets the forecast horizon in bars.
- `--timeframe`, `--start`, and `--end` constrain the requested market window.
- `--methods` supplies comma- or space-separated forecast methods.
- `--include-sections` filters the sections returned after computation;
  `--max-sections` caps their count.
- `--denoise` and `--denoise-params` configure optional input smoothing.
- `--params` supplies template and sub-tool overrides such as context limits,
  backtest settings, barrier grids, or additional timeframes.
- `--detail` controls canonical response detail; `--extras` requests supported
  richer envelope sections in TOON output.

Run `mtdata-cli report_generate --help` for the current parameter list and
template descriptions.

## Reading the result

Full reports contain a `sections` mapping plus summary and status information.
Section names depend on the template and may include context, forecast,
backtest, volatility, pivots, patterns, barriers, regimes, or multi-timeframe
variants. Check the report-level and per-section status before consuming a
value: a successful report envelope can still describe omitted or partial
sections.

For automation, prefer `--json` and follow the stable envelope rules in
[OUTPUT.md](OUTPUT.md). Do not parse the human-oriented TOON rendering.
