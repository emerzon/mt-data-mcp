# Causal Signal Discovery

The `causal_discover_signals` tool performs **pairwise Granger-style causal discovery** between symbols using recent MT5 close prices. It is intended for exploratory analysis (feature discovery / watchlist relationships), not “true causality”.

If you want **co-movement** rather than lead/lag structure, use `correlation_matrix` to calculate pairwise Pearson or Spearman correlations on transformed MT5 price series. If you want candidate **mean-reverting / spread** relationships, use `cointegration_test`.

**Related:**
- [CLI.md](CLI.md) — Command usage and output formats
- [SETUP.md](SETUP.md) — MT5 connection and dependencies
- [GLOSSARY.md](GLOSSARY.md) — Time series terms

---

## Quick Start

```bash
# Compare symbols by correlation strength
mtdata-cli correlation_matrix "EURUSD,GBPUSD,USDJPY" --timeframe H1 \
  --limit 500 --method pearson --transform log_return --json

# Use an explicit MT5 group path for easier basket selection
mtdata-cli correlation_matrix --group "Forex\\Majors" --timeframe H1 \
  --limit 120 --method pearson --transform log_return --json

# Test an MT5 group for candidate cointegrated pairs
mtdata-cli cointegration_test --group "Forex\\Majors" --timeframe H1 \
  --limit 400 --transform log_level --significance 0.05 --json

# Provide an explicit list of symbols
mtdata-cli causal_discover_signals "EURUSD,GBPUSD,USDJPY" --timeframe H1 \
  --limit 800 --max-lag 5 --transform log_return --significance 0.05

# Provide a single symbol to auto-expand its visible MT5 group (e.g., Forex\Majors)
mtdata-cli causal_discover_signals EURUSD --timeframe H1 --limit 800
```

---

## What It Does

### `correlation_matrix`

For each unordered pair of symbols `(A, B)`, the tool:
1. Fetches recent close-price histories
2. Applies a transform (by default `log_return`)
3. Computes pairwise correlations on overlapping transformed samples
4. Returns canonical ranked pair rows plus optional matrix/highlight views

It accepts either:

- an explicit `symbols` list (or compatibility alias `symbol`), or
- a `group` path that matches the MT5 symbol groups exposed by `symbols_list --list-mode groups`

`items` is the canonical compact payload. Each row includes the correlation,
sample count, and pairwise period window; `context` records the timeframe,
limit, transform, and `min_overlap` used. Use `extras=metadata` when you also
need derived convenience views such as `matrix`; omit `extras` to keep only the
ranked pair rows plus summary highlights.

### `cointegration_test`

For each unordered pair of symbols `(A, B)`, the tool:
1. Fetches recent price histories
2. Applies a level-style transform (`log_level` by default)
3. Runs Engle-Granger cointegration tests in both orientations
4. Keeps the orientation with the lower p-value and reports spread diagnostics

It accepts either:

- an explicit `symbols` list (or compatibility alias `symbol`), or
- a `group` path that matches the MT5 symbol groups exposed by `symbols_list --list-mode groups`

### `causal_discover_signals`

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

Tip: `--json` returns the structured payload instead of default TOON text.

---

## Interpretation and Caveats

- `correlation_matrix`, `cointegration_test`, `causal_discover_signals`, and `market_scan` all normalize `symbol` vs `symbols` the same way. Prefer `symbols` for new multi-symbol calls.
- `group` remains mutually exclusive with explicit symbol selectors.

- Use **`correlation_matrix`** when you want a fast view of which symbols move together or in opposite directions.
- Use **`cointegration_test`** when you want candidate pairs or baskets whose price levels may share a stable long-run relationship.
- Use **`causal_discover_signals`** when you specifically want to test whether lagged values of one symbol add predictive information for another.
- Granger causality is a **predictive** notion: “past values of A help predict B” under the model assumptions.
- Results are **pairwise** (not a full causal graph) and can be confounded by common drivers (USD strength, risk-on/off, sessions).
- Use transforms (returns/diffs) and sufficient history; non-stationary levels can produce misleading links.
- Validate any signal with out-of-sample testing before using it in a strategy.

---

## Dependencies

`causal_discover_signals` and `cointegration_test` require `statsmodels`. If it is not installed, the tools return a readable error message.

---

## See Also

- [FORECAST.md](FORECAST.md) — Forecasting methods overview
- [TECHNICAL_INDICATORS.md](TECHNICAL_INDICATORS.md) — Indicator reference
- [GLOSSARY.md](GLOSSARY.md) — Term definitions


