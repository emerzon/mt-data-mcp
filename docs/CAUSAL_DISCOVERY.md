# Causal Signal Discovery

The `causal_discover_signals` tool performs **pairwise Granger-style causal discovery** between symbols using recent MT5 close prices. It is intended for exploratory analysis (feature discovery / watchlist relationships), not “true causality”.

**Related:**
- [CLI.md](CLI.md) — Command usage and output formats
- [SETUP.md](SETUP.md) — MT5 connection and dependencies
- [GLOSSARY.md](GLOSSARY.md) — Time series terms

---

## Quick Start

```bash
# Provide an explicit list of symbols
python cli.py causal_discover_signals "EURUSD,GBPUSD,USDJPY" --timeframe H1 \
  --limit 800 --max-lag 5 --transform log_return --significance 0.05

# Provide a single symbol to auto-expand its visible MT5 group (e.g., Forex\Majors)
python cli.py causal_discover_signals EURUSD --timeframe H1 --limit 800
```

---

## What It Does

For each ordered pair of symbols `(cause → effect)`, the tool:
1. Fetches and aligns overlapping close-price histories
2. Applies a transform (by default `log_return`) to improve stationarity
3. Optionally z-scores the series (`normalize=true`)
4. Runs Granger causality tests for lags `1..max_lag`
5. Reports the **best (lowest p-value) lag** per pair using `ssr_ftest`

---

## Parameters

| Parameter | Default | Description |
|----------|---------|-------------|
| `symbols` | (required) | Comma-separated MT5 symbols. If you pass **one** symbol, mtdata expands to other visible symbols in the same MT5 group. |
| `timeframe` | `H1` | Bar timeframe (`M15`, `H1`, etc.). |
| `limit` | `500` | Bars per symbol to analyze (after alignment). |
| `max_lag` | `5` | Maximum lag to test (≥ 1). |
| `significance` | `0.05` | Alpha threshold for reporting “causal” links. |
| `transform` | `log_return` | One of: `log_return`, `pct`, `diff`, `level`. |
| `normalize` | `true` | Z-score each series before testing. |

---

## Output

The tool returns a plain-text table:

```
Effect <- Cause | Lag | p-value | Samples | Conclusion
```

- **Conclusion** is `causal` when `p-value < significance`, otherwise `no-link`.
- **Lag** is the best-performing lag for the pair under the selected test statistic.

Tip: `--format json` wraps the text as `{"text": "..."}`.

---

## Interpretation and Caveats

- Granger causality is a **predictive** notion: “past values of A help predict B” under the model assumptions.
- Results are **pairwise** (not a full causal graph) and can be confounded by common drivers (USD strength, risk-on/off, sessions).
- Use transforms (returns/diffs) and sufficient history; non-stationary levels can produce misleading links.
- Validate any signal with out-of-sample testing before using it in a strategy.

---

## Dependencies

`causal_discover_signals` requires `statsmodels`. If it is not installed, the tool returns a readable error message.

