# Trade idea composer

**Audience:** User

Turn one symbol, timeframe, and horizon into a **preview-only research idea**: a short narrative, exits, size, gates, and a dry-run order check. This is the sample-trade loop as a single command.

It is **not** a buy or sell instruction, and it **cannot** send a live order.

**Dense terms:** [Barrier](GLOSSARY.md#barrier) · [Dry-run](GLOSSARY.md#dry-run) · [Confluence](GLOSSARY.md#confluence) · [Fixed-fraction sizing](GLOSSARY.md#fixed-fraction-sizing)

**Related:** [Sample trade](SAMPLE-TRADE.md) · [Barriers](BARRIER_FUNCTIONS.md) · [Trading safety](TRADING_SAFETY.md) · [Reports](REPORTS.md) · [Web UI](WEBUI.md)

---

## Quick start

```bash
mtdata-cli trade_idea_compose EURUSD --timeframe H1 --horizon 12 --template quick
```

Read `direction`, `narrative`, `geometry`, `sizing.suggested_volume`, and `preview.preview_ok`. If `direction` is `stand_down`, the composer is telling you the idea did not clear its gates — not that you should fade it.

The same payload is available from MCP as `trade_idea_compose` and from HTTP as `POST /api/v1/trade-ideas`. In the Web UI, use the **Idea** button.

---

## What it does

The composer **reuses existing tools**. It does not invent new forecast or barrier math.

| Step | Tool | Quick | Standard |
|------|------|-------|----------|
| Session + quote | `trade_session_context` | live only | live only |
| Structure | `confluence_levels` | no | yes (and may snap TP/SL toward nearby zones) |
| Price path | `forecast_conformal_intervals` (Theta) for auto; `forecast_generate` for an explicit side | yes | yes |
| Typical movement | `forecast_volatility_estimate` (EWMA) | yes | yes |
| One TP/SL pair | `forecast_barrier_prob` (0.40% / 0.60%) | yes | yes |
| Size | `trade_risk_analyze` (fixed-fraction) | live only | live only |
| Preview | `trade_place` with `dry_run=true` | live only | live only |

`--direction auto` (default) calibrates Theta residual-quantile bands over 50
rolling historical anchors, spaced by at least the requested horizon. It selects
a side only when the calibrated horizon band excludes the last-price anchor. The
result's `forecast` section identifies the method, interval method, alpha,
calibration sample, and exact interval gate basis. A neutral direction,
insufficient calibration, an interval containing the anchor, or unavailable
uncertainty stands down; the composer does not infer a side from the slope
between forecast steps. It also stands down when the barrier sketch says the
stop is more likely to hit first.

Auto mode is therefore materially slower than `--direction long` or
`--direction short`: it fits 50 rolling backtest forecasts before the current
forecast. Explicit directions use the point forecast only, while all other
quote, barrier, sizing, and preview safety gates remain in force.

`--as-of` makes the idea historical and **research-only**: no live session or
quote, no live sizing, and no dry-run preview. Historical geometry uses the
barrier analysis's cutoff-bound reference price.

---

## How to read the result

| Field | Meaning |
|-------|---------|
| `direction` | `long`, `short`, or `stand_down` |
| `direction_basis` | `forecast_vs_last_price` for auto direction, or `requested` for an explicit side |
| `suggested_direction` | Forecast-based hint; may differ from `direction` |
| `forecast.calibration` | Auto mode's requested anchors, minimum usable residual sample, empirical coverage, and sufficiency status |
| `forecast.forecast_vs_last_price.direction_interval_basis` | Exact comparison used by the auto direction gate |
| `actionability` | Always `preview_only` or `research`. Never live. |
| `gates` | `pass` / `fail` / `skip` for quote, session, forecast, barriers, SL/TP, sizing, preview |
| `preview.preview_ok` | Local dry-run eligibility. Still not a broker fill. |
| `partial_failure` | Some sections failed; do not infer the missing ones |

Reports (`report_generate`) remain research packages. This command is the **decision artifact** that adds size, gates, and a dry-run preview.

---

## Safety

- The composer **rejects** any live send. There is no `dry_run=false` flag.
- Stale, locked, or non-tradable quotes stand down and keep `suggested_volume` at `0`.
- Prefer a demo account even for previews that you later copy into `trade_place`.
- See [TRADING_SAFETY.md](TRADING_SAFETY.md) before you ever set `--dry-run false` on `trade_place` itself.

---

## See also

- [SAMPLE-TRADE.md](SAMPLE-TRADE.md) — the same questions as separate commands
- [SAMPLE-TRADE-WEBUI.md](SAMPLE-TRADE-WEBUI.md) — run `trade_idea_compose` from **Tools** until the Idea panel ships
- [REPORTS.md](REPORTS.md) — packaged research without sizing or dry-run
