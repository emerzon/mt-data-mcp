# Trading Safety Runbook

The `trade_*` tools send **real requests** to the MT5 account currently logged into the terminal. This runbook is the safety-first workflow for `trade_place`, `trade_modify`, and `trade_close`: how to preview, what gets validated, how account guardrails work, and broker-specific behavior to expect.

> ⚠️ **These tools default to LIVE execution.** `dry_run` defaults to **`false`** on `trade_place`, `trade_modify`, and `trade_close`. If you do not pass `--dry-run true`, the request is sent to MT5. Use a **demo account** until you fully trust your setup — mtdata has no separate paper-trading mode.

---

## Golden rules

1. **Preview first.** Add `--dry-run true` to every order until you have verified the request.
2. **Trade a demo account** while learning. There is no simulated mode besides an MT5 demo.
3. **Enable guardrails** on any account that can place orders (see [Account guardrails](#account-guardrails)).
4. **Use exact tickets** for `trade_modify` / `trade_close`; be extremely careful with `--close-all`.
5. **Attach protective levels.** `trade_place` requires SL **and** TP on market orders by default (`--require-sl-tp`).

---

## Preview with `--dry-run`

A dry run routes and validates the request **without sending it to MT5**. The `trade_place` preview returns markers you can assert on:

```jsonc
{
  "dry_run": true,
  "no_action": true,
  "would_send_order": false,
  "dry_run_simulated": true,
  "validation_scope": "request_routing_only",
  "validation_not_performed": [ /* broker acceptance, live price-distance rules, margin/funds, fillability, SL/TP attachment */ ],
  "guardrails_preview": { /* which guardrails would apply */ }
}
```

**What a dry run *does* check:** required fields, order-type validity, market-vs-pending routing, and a guardrails preview.

**What a dry run *cannot* check** (only a live send confirms these): final broker acceptance, live price-distance/stops rules, margin and funds, fillability, and SL/TP attachment after a market fill. Treat a clean preview as necessary, not sufficient.

---

## `trade_place`

Requires `symbol`, `volume`, and `order_type`.

| Flag | Default | Notes |
|------|---------|-------|
| `--symbol` | — | Broker symbol |
| `--volume` | — | Lots (validated against broker min/max/step) |
| `--order-type` | — | See [Order types](#order-types) |
| `--price` | — | Entry for pending orders; **omit for market orders** |
| `--stop-loss` / `--sl` | — | Stop-loss price |
| `--take-profit` / `--tp` | — | Take-profit price |
| `--deviation` | `20` | Max slippage in points (market orders) |
| `--require-sl-tp` | `true` | Require both SL and TP on market orders |
| `--auto-close-on-sl-tp-fail` | `true` | Close the position if SL/TP attachment fails after a fill |
| `--expiration` | — | Pending-order expiry (`dateparser`, UTC epoch seconds, or `GTC`) |
| `--magic` | `MTDATA_ORDER_MAGIC` | Strategy identifier stamped on the order |
| `--comment` | — | Free-text order comment |
| `--idempotency-key` | — | In-process dedupe (~5-min TTL; not persisted/shared) |
| `--dry-run` | `false` | **Preview only when `true`** |
| `--detail` | `compact` | Use `full` for execution diagnostics |

```bash
# Preview a market buy with protective levels
mtdata-cli trade_place EURUSD --volume 0.10 --order-type BUY \
  --stop-loss 1.0850 --take-profit 1.0950 --dry-run true

# Go live (only on the intended account)
mtdata-cli trade_place EURUSD --volume 0.10 --order-type BUY \
  --stop-loss 1.0850 --take-profit 1.0950 --dry-run false
```

### Order types

`order_type` accepts these **canonical strings** (case-insensitive; `-` or space becomes `_`):

`BUY`, `SELL`, `BUY_LIMIT`, `BUY_STOP`, `SELL_LIMIT`, `SELL_STOP`

MT5 numeric constants (`0..5`) and `ORDER_TYPE_*` names are **rejected** as input — they only appear when *reading* existing orders/positions. Market orders use `BUY`/`SELL` (no `--price`); the four `*_LIMIT`/`*_STOP` types are pending orders and require `--price`.

---

## `trade_modify`

Modifies an existing order/position by ticket.

| Flag | Default | Notes |
|------|---------|-------|
| `--ticket` | — | **Required** |
| `--price` | — | New pending-order price |
| `--stop-loss` / `--sl` | — | New stop-loss |
| `--take-profit` / `--tp` | — | New take-profit |
| `--expiration` | — | New pending-order expiry |
| `--comment` | — | Updated comment |
| `--idempotency-key` | — | In-process dedupe |
| `--dry-run` | `false` | **Preview only when `true`** |

```bash
mtdata-cli trade_get_open --json
mtdata-cli trade_modify 123456789 --stop-loss 1.0860 --take-profit 1.0980 --dry-run true
```

Guardrails apply to `trade_modify` only for pending-order changes and SL changes that **increase** risk; risk-reducing changes stay allowed.

---

## `trade_close`

Closes one position, a filtered set, or all — with an extra confirmation for bulk closes.

| Flag | Default | Notes |
|------|---------|-------|
| `--ticket` | — | Close a single position |
| `--volume` | — | Partial-close size (validated against broker step) |
| `--symbol` | — | Restrict closes to a symbol |
| `--magic` | — | Restrict closes to a magic number |
| `--close-all` | `false` | Close every matching position |
| `--confirm-close-all` | `false` | **Required** when `--close-all` is set and not a dry run |
| `--profit-only` / `--loss-only` | `false` | Only close winners / losers |
| `--close-priority` | — | `loss_first`, `profit_first`, or `largest_first` |
| `--deviation` | `20` | Max slippage in points |
| `--dry-run` | `false` | **Preview only when `true`** |

```bash
# Preview a partial close of one ticket
mtdata-cli trade_close --ticket 123456789 --volume 0.05 --dry-run true

# Close ALL positions — requires the explicit confirmation flag when live
mtdata-cli trade_close --close-all --confirm-close-all true --dry-run false
```

There is no separate "confirm" token for `trade_place`/`trade_modify`; the only extra safety gate is `--confirm-close-all` for bulk closes.

---

## Account guardrails

Guardrails are optional pre-trade controls that reject risky orders **before** they reach MT5. They are evaluated when `MTDATA_TRADE_GUARDRAILS_ENABLED=1` **or** whenever any individual guardrail variable is set (demo accounts are skipped when `MTDATA_TRADE_GUARDRAILS_IGNORE_ON_DEMO=true`).

Guardrails span several layers:

| Layer | Rejects when… | Key variables |
|-------|---------------|---------------|
| Kill switch | Trading is disabled | `MTDATA_TRADING_ENABLED=0` |
| Symbol rules | Symbol is blocked or not allowlisted | `MTDATA_TRADE_ALLOWED_SYMBOLS`, `MTDATA_TRADE_BLOCKED_SYMBOLS` |
| Volume caps | Order volume exceeds a global or per-symbol cap | `MTDATA_TRADE_MAX_VOLUME`, `MTDATA_TRADE_MAX_VOLUME_BY_SYMBOL` |
| Safety policy | Missing SL, excessive deviation, or non-reducing order | `MTDATA_TRADE_SAFETY_REQUIRE_STOP_LOSS`, `MTDATA_TRADE_SAFETY_MAX_DEVIATION`, `MTDATA_TRADE_SAFETY_REDUCE_ONLY` |
| Account risk | Margin too low, floating loss or exposure too high | `MTDATA_TRADE_MIN_MARGIN_LEVEL_PCT`, `MTDATA_TRADE_MAX_FLOATING_LOSS`, `MTDATA_TRADE_MAX_TOTAL_EXPOSURE_LOTS` |
| Wallet risk | Post-trade risk exceeds a % of equity/balance/free margin | `MTDATA_TRADE_MAX_RISK_PCT_OF_EQUITY`, `MTDATA_TRADE_MAX_RISK_PCT_OF_BALANCE`, `MTDATA_TRADE_MAX_RISK_PCT_OF_FREE_MARGIN` |

> **Note:** A per-symbol volume map (`MTDATA_TRADE_MAX_VOLUME_BY_SYMBOL`) also acts as an allowlist — a symbol missing from the map is rejected. Wallet-risk caps require a quantifiable stop-loss and valid broker tick metadata.

Reduce-only checks the current open positions before allowing an opposite-side
order no larger than the net position. On hedging accounts, `trade_place` cannot
guarantee a reduction, so use `trade_close` with a position ticket instead.

See [ENV_VARS.md § Trade Guardrails](ENV_VARS.md#trade-guardrails) for every variable, defaults, formats, and a ready-to-copy `.env` block. A dry run returns a `guardrails_preview` so you can confirm which rules would fire before going live.

---

## Pre-trade validation & broker behavior

Even with guardrails off, mtdata validates each order against broker constraints before submission:

- **Volume** — must be numeric, positive, finite, within the symbol's `volume_min`/`volume_max`, and aligned to `volume_step` (misaligned sizes are rejected with an aligned suggestion).
- **Pending price side** — `buy_limit` must sit below ask, `buy_stop` above ask, `sell_limit` above bid, `sell_stop` below bid.
- **Stops distance** — SL/TP and pending prices must respect the broker's minimum stops/freeze level.
- **Symbol readiness** — the symbol must be selectable and have live bid/ask.
- **Filling mode** — mtdata resolves a broker-compatible filling mode for market fills and closes.
- **Margin** — a market-order preview estimates required margin.

Because these depend on **live** broker state, they are only fully enforced on a real send — another reason to keep position sizes small when first going live.

---

## Live-trade checklist

1. Confirm the account: `mtdata-cli trade_account_info --json` (verify it's the intended demo/live account).
2. Snapshot context: `mtdata-cli trade_session_context EURUSD --json`.
3. Configure guardrails in `.env` (allowlist, volume caps, risk %). Restart mtdata.
4. Preview: run the order with `--dry-run true`; inspect `guardrails_preview` and `validation_not_performed`.
5. Go live with a **small** size and `--dry-run false`.
6. Verify: `mtdata-cli trade_get_open --json`, then manage with `trade_modify` / `trade_close`.

---

## See Also

- [CLI.md § Trading](CLI.md#trading) — Command list and execution controls
- [ENV_VARS.md § Trade Guardrails](ENV_VARS.md#trade-guardrails) — Full guardrail variable reference
- [TRADING_RISK.md](TRADING_RISK.md) — Position sizing, VaR/CVaR, and stress tests
- [SAMPLE-TRADE-ADVANCED.md](SAMPLE-TRADE-ADVANCED.md) — An end-to-end analysis-to-execution workflow
- [OUTPUT.md](OUTPUT.md) — Response envelope and error codes
