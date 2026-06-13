# Trading Risk Analytics

These are **read-only** risk tools. They inspect the account and open positions and
return analytics — they never place, modify, or close orders. Use them to size a new
trade, estimate tail risk, and stress-test the portfolio against scenarios.

| Tool | Answers |
|------|---------|
| `trade_risk_analyze` | "How big can this trade be?" and "How much am I risking right now?" |
| `trade_var_cvar_calculate` | "What is my statistical worst-case loss over one bar?" |
| `trade_stress_test` | "What happens to my open positions under these price shocks?" |

See [SAMPLE-TRADE.md](SAMPLE-TRADE.md) and [SAMPLE-TRADE-ADVANCED.md](SAMPLE-TRADE-ADVANCED.md)
for end-to-end workflows, and [ENV_VARS.md](ENV_VARS.md#trade-guardrails) for guardrails.

---

## `trade_risk_analyze`

Analyzes current open/pending risk and, when given a proposed entry and stop, sizes a
new trade. Supports two sizing methods: fixed-fraction and Kelly.

```bash
# Current portfolio risk only
mtdata-cli trade_risk_analyze --json

# Size a new long: risk-based volume from entry + stop
mtdata-cli trade_risk_analyze --symbol EURUSD --direction long \
  --entry 1.0850 --stop-loss 1.0800 --desired-risk-pct 1.0 --json
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `symbol` | — | Symbol for new-trade sizing; omit for portfolio-only risk. |
| `desired_risk_pct` | — | Target account risk (%) for `fixed_fraction`, and a cap for `kelly`. |
| `sizing_method` | `fixed_fraction` | `fixed_fraction` uses `desired_risk_pct`; `kelly` uses win-rate and average win/loss. |
| `strict_risk` | `true` | Return `suggested_volume=0.0` if the broker minimum lot would exceed `desired_risk_pct`. |
| `include_pending` | `true` | Include contingent stop-loss risk from pending orders in portfolio totals. |
| `direction` | — | `long`/`short` (aliases accepted) for the proposed trade. |
| `entry` | — | Proposed entry price. With `symbol`+`stop_loss` but no entry, it is resolved from the live tick (ask for long, bid for short, mid otherwise). |
| `stop_loss` | — | Proposed stop (alias `sl`). Required to compute risk-based volume. |
| `take_profit` | — | Optional target (alias `tp`) for reward/risk context. |

### Kelly sizing

Set `sizing_method=kelly` and supply edge statistics:

```bash
mtdata-cli trade_risk_analyze --symbol EURUSD --direction long \
  --entry 1.0850 --stop-loss 1.0800 --sizing-method kelly \
  --kelly-win-rate 0.55 --kelly-avg-win 0.012 --kelly-avg-loss 0.010 \
  --kelly-fraction-multiplier 0.5 --kelly-max-risk-pct 2.0 --json
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `kelly_win_rate` | — | Win probability as a fraction in `[0, 1]`. |
| `kelly_avg_win` | — | Average winning return. |
| `kelly_avg_loss` | — | Average losing return magnitude. |
| `kelly_fraction_multiplier` | `0.5` | Multiplier on the raw Kelly fraction (half-Kelly = `0.5`). |
| `kelly_max_risk_pct` | `2.0` | Hard cap on account risk (%) for Kelly sizing. |
| `kelly_metrics` | — | Dict alternative carrying `win_rate`, `avg_win_return`, `avg_loss_return`; explicit `kelly_*` fields override it. |

The Kelly fraction is `win_rate − (1 − win_rate) / (avg_win / |avg_loss|)`, then scaled
by `kelly_fraction_multiplier` and capped by `kelly_max_risk_pct` (and
`desired_risk_pct` when set). On a non-positive edge the tool reports
`status="kelly_no_edge"` and a suggested volume of `0.0`.

---

## `trade_var_cvar_calculate`

Estimates one-bar Value at Risk (VaR) and Conditional VaR (CVaR, a.k.a. Expected
Shortfall) for the current open positions — either the whole portfolio or a single
symbol.

```bash
# Portfolio VaR/CVaR at 95% over one H1 bar
mtdata-cli trade_var_cvar_calculate --timeframe H1 --lookback 500 --confidence 95 --json

# Symbol-scoped, parametric/Gaussian, percentage returns
mtdata-cli trade_var_cvar_calculate --symbol EURUSD --method gaussian \
  --transform pct --lookback 300 --json
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `symbol` | — | Restrict to one symbol's exposure; omit for the full portfolio. |
| `timeframe` | `H1` | Return interval and the one-bar holding period. |
| `lookback` | `500` | Historical bars used to build the return distribution. |
| `confidence` | `0.95` | Confidence level. Accepts a fraction (`0.95`, `0.99`) or a percentage (`95`); must resolve to `0 < confidence < 1`. |
| `method` | `historical` | `historical` (empirical tail) or `gaussian` (aliases `normal`/`parametric`). |
| `transform` | `log_return` | Return transform: `log_return` or `pct`. |
| `min_observations` | `50` | Minimum aligned observations before estimating risk. |

**Output** includes `var` and `cvar`, position/exposure counts, and — by detail level —
per-position and per-symbol exposure breakdowns. With no open positions, `--detail full`
returns the legacy zero-filled arrays.

---

## `trade_stress_test`

Applies deterministic percentage price shocks to open positions and reports the P&L
impact. Useful for "what if EURUSD drops 2% and everything else 3%?" scenarios.

```bash
# Per-symbol shocks
mtdata-cli trade_stress_test '{"EURUSD":-2.0,"GBPUSD":-1.5}' --json

# Wildcard shock applied to every position without an explicit entry
mtdata-cli trade_stress_test '{"*":-3.0}' --detail full --json
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `shocks` | (required) | Per-symbol percentage shocks, e.g. `{"EURUSD": -2.0}`. Use `"*"` as a fallback for symbols without an explicit entry. |
| `include_unshocked` | `false` | Include positions that received no shock (no exact match and no `"*"` fallback). |
| `detail` | `compact` | `full` adds per-position diagnostics. |

**Output** is a list of `items` (one per evaluated position) with `ticket`, `symbol`,
`side`, `volume`, `shock_pct`, `current_price`, `shocked_price`, and `pnl_impact`, plus
totals: `total_pnl_impact`, `positions_total`/`evaluated`/`shocked`, and — when account
metadata is available — `equity_before`/`equity_after`/`impact_pct`.

---

## Caveats

- All three tools read live MT5 state; results change as positions and quotes move.
- VaR/CVaR assume the recent return distribution persists and use a single-bar holding
  period; they are not a guarantee of maximum loss.
- Stress shocks are deterministic and linear in price; they do not model spread
  widening, gaps, swaps, or correlation breaks.
- Kelly sizing is only as good as its inputs — estimate `win_rate` and average win/loss
  from a sufficient out-of-sample track record (see `trade_journal_analyze`).

## See Also

- [BARRIER_FUNCTIONS.md](BARRIER_FUNCTIONS.md) — TP/SL hit probabilities
- [GLOSSARY.md](GLOSSARY.md) — VaR, CVaR, Kelly, and risk term definitions
- [SAMPLE-TRADE-ADVANCED.md](SAMPLE-TRADE-ADVANCED.md) — Risk gating in a full workflow
