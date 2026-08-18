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

The same payload is available from MCP as `trade_idea_compose` and from HTTP as `POST /api/v1/trade-ideas`.

---

## What it does

The composer **reuses existing tools**. It does not invent new forecast or barrier math.

| Step | Tool | Quick | Standard |
|------|------|-------|----------|
| Session + quote | `trade_session_context` | yes | yes |
| Structure | `confluence_levels` | no | yes (and may snap TP/SL toward nearby zones) |
| Price path | `forecast_generate` (Theta) | yes | yes |
| Typical movement | `forecast_volatility_estimate` (EWMA) | yes | yes |
| One TP/SL pair | `forecast_barrier_prob` (0.40% / 0.60%) | yes | yes |
| Size | `trade_risk_analyze` (fixed-fraction) | live only | live only |
| Preview | `trade_place` with `dry_run=true` | live only | live only |

`--direction auto` (default) may *suggest* long or short from the forecast path. If the forecast is flat, or the barrier sketch says the stop is more likely to hit first, the idea stands down.

`--as-of` makes the idea historical and **research-only**: no live sizing and no dry-run preview.

---

## How to read the result

| Field | Meaning |
|-------|---------|
| `direction` | `long`, `short`, or `stand_down` |
| `suggested_direction` | Forecast-based hint; may differ from `direction` |
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
