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
mtdata-cli report_generate EURUSD --timeframe H1
```

The command defaults to the fast `minimal` template (context and forecast only)
and compact TOON text. Use `--template basic` for levels, patterns, barriers,
and broader risk context; use `--json` for a machine-readable payload or
`--detail full` for all content supported by the selected template. CLI and MCP
preserve the same canonical report payload; output format only changes its final
presentation.

## Choose a template

| Template | Typical warm runtime | Design and intended use |
|----------|----------------------|-------------------------|
| `minimal` | 3-10 seconds | Default fast path: context and direct forecast only |
| `basic` | 30-120 seconds | Shared general-purpose research pipeline; opt in explicitly |
| `advanced` | 60-180 seconds | Extends `basic` with regime, HAR-RV, and conformal sections |
| `scalping` | 15-60 seconds | Specialized short-horizon M5 path with tick-aware barrier logic |
| `intraday` | 30-120 seconds | `basic` preset with H1-oriented defaults |
| `swing` | 30-120 seconds | `basic` preset with H4/D1-oriented defaults |
| `position` | 30-120 seconds | `basic` preset with D1/W1-oriented defaults |

`intraday`, `swing`, and `position` select different default timeframes,
lookbacks, backtest sampling, barrier ranges, and multi-timeframe inputs. They
do not define different analytics or section schemas: each runs the `basic`
pipeline with its preset parameters.

`minimal` is the bounded interactive default. The other templates may perform several MT5
fetches and invoke pivots, patterns, backtests, barriers, or regime checks.
Runtime and dependency requirements therefore vary by template. Section
controls select the sections to execute and return, while internal
dependencies may still run when a requested section requires them.
The ranges above are guidance, not deadlines: broker history synchronization,
explicit model choices, and cold model initialization can take longer.

## Control template, scope, and output

```bash
# Fast overview
mtdata-cli report_generate EURUSD --template minimal --timeframe H1

# Basic-pipeline style preset with an explicit forecast horizon
mtdata-cli report_generate EURUSD --template swing --timeframe H4 --horizon 12

# Keep only selected computed sections
mtdata-cli report_generate EURUSD --template basic \
  --include-sections context,forecast,barriers --max-sections 3 --json

# Return the useful subset planned for a 10-second budget and show progress
mtdata-cli report_generate EURUSD --template basic \
  --max-runtime 10 --progress true --json

# Restrict candidate forecast methods and apply denoising
mtdata-cli report_generate EURUSD --template basic \
  --methods theta,arima --denoise kalman --json
```

Useful controls:

- `--horizon` sets the forecast horizon in bars.
- `--timeframe`, `--start`, and `--end` constrain the requested market window.
- When `--start` or `--end` bounds a report, sections that only support current-market
  analysis are not run. Their section payloads use `status: omitted` with reason
  `current_only_section_omitted`, and the report is marked partial rather than mixing
  current data into the bounded analysis.
- `--methods` supplies comma- or space-separated forecast methods.
- `--include-sections` selects the sections to execute and return; required
  internal dependencies may run but cannot independently make the request
  successful. `--max-sections` caps the selected count.
- `--max-runtime` supplies a cooperative wall-clock budget. The runner first
  schedules a section subset whose estimated cost fits, then stops starting
  report sub-tools once the deadline passes. An already-running native or MT5
  call cannot be safely preempted, so a single call can finish just beyond the
  requested budget. `runtime_plan` records estimates, omissions, elapsed time,
  and whether the budget was exhausted.
- `--progress true` writes sub-tool start/finish events to stderr while stdout
  remains the final structured report.
- `--allow-partial` defaults to `true`: a report with at least one usable
  section returns `success:true` and `section_run_status:partial`. Set it to
  `false` when a caller requires every selected section to complete cleanly.
- `--denoise` and `--denoise-params` configure optional input smoothing.
- `--params` supplies template and sub-tool overrides such as context limits,
  backtest settings, barrier grids, or additional timeframes.
- Scalping and intraday `market` sections always obtain Level 1 bid/ask/spread
  from `market_ticker`; broker DOM is optional and reports `depth_status` as
  `available`, `quote_only`, `disabled`, or `unavailable`. The
  `execution_gates` section always returns a gate decision. Configure an
  additional spread cap with `params.spread_max_ticks` or
  `params.spread_max_pips`; without one, the gate checks quote readiness and a
  valid positive spread.
- `--detail` controls canonical response detail; use `--detail full` for richer
  metadata and diagnostics.

Run `mtdata-cli report_generate --help` for the current parameter list and
template descriptions.

## Reading the result

Full reports contain a `sections` mapping plus summary and status information.
Section names depend on the template and may include context, forecast,
backtest, volatility, pivot, patterns, barriers, regime, or multi-timeframe
variants. Check the report-level and per-section status before consuming a
value: a successful report envelope can still describe omitted or partial
sections.

`section_run_status` reports whether scheduled sections completed (`complete`,
`partial`, or `failed`). `content_detail` separately reports how much content
was returned (`summary_only`, `selected_sections`, or `full_sections`). Compact
responses are therefore explicitly `content_detail: summary_only` even when all
scheduled sections ran successfully. Context trend windows are calculated from
consecutive source-timeframe candles; unavailable long windows are `null`
rather than silently shortened.

For automation, prefer `--json` and follow the stable envelope rules in
[OUTPUT.md](OUTPUT.md). Do not parse the human-oriented TOON rendering.
