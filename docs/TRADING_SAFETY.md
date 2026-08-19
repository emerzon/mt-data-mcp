# Trading safety runbook

**Audience:** User

If you only skim one trading doc, make it this one. The `trade_*` tools send **real requests** to the MT5 account currently logged into the terminal. This runbook covers previewing orders, validation, account guardrails, and broker quirks for `trade_place`, `trade_modify`, and `trade_close`.

> **These tools default to preview mode.** `dry_run` defaults to **`true`**. A request reaches MT5 only when you explicitly set `--dry-run false`. Use a **demo account** until you trust your setup — mtdata has no separate paper-trading mode. `trade_idea_compose` is stricter: it is preview-only and has no live-send flag.

**Dense terms:** [Dry-run](GLOSSARY.md#dry-run) · [Trade guardrails](GLOSSARY.md#trade-guardrails) · [Slippage](GLOSSARY.md#slippage) · [Lot size](GLOSSARY.md#lot-size) · [TP/SL](GLOSSARY.md#tpsl-take-profit--stop-loss)

**Related:** [Trade ideas](TRADE_IDEAS.md) · [Risk analytics](TRADING_RISK.md) · [Env vars (guardrails)](ENV_VARS.md#trade-guardrails) · [CLI](CLI.md) · [Sample trade](SAMPLE-TRADE.md) · [Glossary](GLOSSARY.md)

---

## Golden rules

1. **Preview first** — `--dry-run true` until the request looks right.
2. **Demo while learning** — no simulated mode except an MT5 demo.
3. **Enable guardrails** on any account that can place orders ([Account guardrails](#account-guardrails)).
4. **Exact tickets** for modify/close; treat `--close-all` as nuclear.
5. **Protective levels** — market orders require SL **and** TP by default (`--require-sl-tp`).

MT5 tickets and magic numbers are unsigned 64-bit identifiers. Ticket inputs
accept `1..18446744073709551615`; magic accepts
`0..18446744073709551615`, where zero is a real manual/untagged strategy scope,
not an omitted filter. Decimal input is parsed without floating-point
conversion. When an identifier exceeds JavaScript's exact integer range, JSON
responses also include a sibling such as `ticket_exact` or `magic_exact` as a
canonical decimal string and set
`identifier_encoding=decimal_string_in_exact_fields`.

---

## Preview with `--dry-run`

A dry run routes and validates the request **without sending it to MT5**. The `trade_place` preview returns markers you can assert on:

```jsonc
{
  "dry_run": true,
  "no_action": true,
  "would_send_order": false,
  "dry_run_simulated": true,
  "preview_ok": true,
  "validation_passed": true,
  "validation_scope": "local_preview_plus_estimates",
  "preview_checks_performed": [ /* checks actually completed */ ],
  "checks_not_performed": [ /* for example, margin_estimate when unavailable */ ],
  "broker_validation_not_performed": [ /* broker acceptance/enforcement, margin reservation, fillability, SL/TP attachment */ ],
  "guardrails_preview": { /* which guardrails would apply */ }
}
```

Eligible previews return `success=true` and `preview_ok=true`. A preview that is
not eligible for live submission returns `success=false`,
`error_code=preview_blocked`, and `preview_ok=false` while retaining the preview
body and its actionable `blockers`. This includes missing required SL/TP,
closed-market or stale-quote checks, and other local safety failures.
Ticketless bulk `trade_close` previews also remain blocked until
`--confirm-close-all true` is present; `required_confirmation` and
`validation.live_submission_eligible` make that remaining live gate explicit.
The CLI prints those blocked previews and exits `1`; an eligible preview exits `0`.
Compact output always retains these gate fields and the broker-validation
limitations. Its `guardrails_preview` summary retains `enabled`, `blocked`,
`ignored_for_demo`, and `checks_not_performed`; standard/full detail includes
the complete guardrail diagnostics.

**What a dry run *does* check:** required fields, order-type validity,
market-vs-pending routing, an indicative margin estimate when MT5 exposes one,
and a guardrails preview. For pending orders, `margin_required_when_filled` is
calculated with the corresponding active BUY/SELL action at the requested entry
price; `margin_estimate_basis` records that assumption.

**What a dry run *cannot* check** (only a live send confirms these): final broker
acceptance, live price-distance/stops rules, the final margin reservation and
funds decision, fillability, and SL/TP attachment after a market fill. Treat a
clean preview as necessary, not sufficient.

---

## `trade_place`

Requires `symbol`, `volume`, and `order_type`.

| Flag | Default | Notes |
|------|---------|-------|
| `symbol` | — | Broker symbol |
| `--volume` | — | Lots (validated against broker min/max/step) |
| `--order-type` | — | See [Order types](#order-types) |
| `--price` | — | Entry for pending orders; **omit for market orders** |
| `--stop-loss` | — | Stop-loss price |
| `--take-profit` | — | Take-profit price |
| `--deviation` | `20` | Max slippage in points (market orders) |
| `--require-sl-tp` | `true` | Require both SL and TP on market orders |
| `--expiration` | — | Future pending-order expiry (`dateparser` or positive UTC epoch seconds); literal `GTC` means no expiry |
| `--magic` | `MTDATA_ORDER_MAGIC` | Strategy identifier stamped on the order |
| `--comment` | — | Free-text order comment |
| `--idempotency-key` | — | Durable dedupe shared across processes/restarts (24-hour default retention) |
| `--dry-run` | `true` | Set `false` explicitly for live execution |
| `--detail` | `compact` | Use `full` for execution diagnostics |

If required SL/TP protection cannot be attached after a market fill, mtdata
always attempts to close the unprotected position. This fail-safe is not optional.

Idempotency outcomes are stored in `MTDATA_TRADE_IDEMPOTENCY_DB`. If a process
stops after reserving a key but before recording the broker outcome, retries for
that key fail closed. Reconcile the order or modification in MT5 before removing
the unresolved database row; never clear it merely to make a retry proceed.

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

`BUY`, `SELL`, `BUY_LIMIT`, `BUY_STOP`, `BUY_STOP_LIMIT`, `SELL_LIMIT`,
`SELL_STOP`, `SELL_STOP_LIMIT`

MT5 numeric constants and `ORDER_TYPE_*` names are **rejected** as input — they
only appear when *reading* existing orders/positions. Market orders use
`BUY`/`SELL` (no `--price`). Every pending order requires `--price`. For a
stop-limit order, `--price` is the stop trigger and `--stop-limit-price` is the
limit leg activated after the trigger. A buy stop-limit's limit price must be at
or below its trigger; a sell stop-limit's limit price must be at or above it.

```bash
# Trigger above the ask, then activate a buy limit at or below that trigger
mtdata-cli trade_place EURUSD --volume 0.10 --order-type BUY_STOP_LIMIT \
  --price 1.1050 --stop-limit-price 1.1045 --dry-run true
```

---

## `trade_modify`

Modifies an existing order/position by ticket.

At least one of `price`, `stop_limit_price`, `stop_loss`, `take_profit`,
`expiration`, or `comment` must be supplied. An explicit value that already
matches the live object is a successful idempotent no-change request; omitting
every modification field is an error.

| Flag | Default | Notes |
|------|---------|-------|
| `ticket` | — | **Required** |
| `--price` | — | New pending-order price |
| `--stop-limit-price` | — | New limit leg for an existing stop-limit order |
| `--stop-loss` | — | New stop-loss |
| `--take-profit` | — | New take-profit |
| `--expiration` | — | New future pending-order expiry, or literal `GTC` |
| `--comment` | — | Updated comment |
| `--idempotency-key` | — | Durable dedupe shared across processes/restarts |
| `--dry-run` | `true` | Preview by default; set `false` explicitly for a live modification |

```bash
mtdata-cli trade_get_open --json
mtdata-cli trade_modify --ticket 123456789 --stop-loss 1.0860 --take-profit 1.0980 --dry-run true
```

Guardrails apply to `trade_modify` only for pending-order changes and SL changes that **increase** risk; risk-reducing changes stay allowed.

---

## `trade_close`

`trade_close` acts on one explicit object class. It closes positions by default,
cancels pending orders only with `--target pending`, and flattens both classes
with `--target all_exposure`. There is no automatic position-to-order fallback.

| Flag | Default | Notes |
|------|---------|-------|
| `--ticket` | — | Act on one ticket in the selected target class |
| `--target` | `positions` | `positions`, `pending`, or `all_exposure` (bulk scopes only) |
| `--volume` | — | Partial-close size (validated against broker step) |
| `--symbol` | — | Restrict closes to a symbol |
| `--magic` | — | Restrict closes to a magic number |
| `--close-all` | `false` | Select the whole account when ticket, symbol, and magic are omitted |
| `--confirm-close-all` | `false` | **Required** for any ticketless live bulk operation |
| `--pnl-filter` | `all` | Close all matches, only winners (`profit`), or only losers (`loss`) |
| `--close-priority` | — | `loss_first`, `profit_first`, or `largest_first` |
| `--deviation` | `20` | Max slippage in points |
| `--dry-run` | `true` | Preview by default; set `false` explicitly for a live close |

```bash
# Preview a partial close of one ticket
mtdata-cli trade_close --ticket 123456789 --volume 0.05 --dry-run true

# Cancel one pending order; default target=positions would not cancel it
mtdata-cli trade_close --ticket 987654321 --target pending --dry-run false

# Close all positions account-wide
mtdata-cli trade_close --close-all --confirm-close-all true --dry-run false

# Close positions and cancel pending orders for one strategy
mtdata-cli trade_close --magic 3001 --target all_exposure \
  --confirm-close-all true --dry-run false
```

For `all_exposure`, the response keeps `closed_positions` and
`cancelled_pending_orders` as separate result legs and reports partial failures;
one failed leg does not prevent the other from being attempted. There is no
separate "confirm" token for `trade_place`/`trade_modify`; the extra
`--confirm-close-all` gate applies to every ticketless live bulk close.
Dry-run bulk previews can still enumerate the matching exposure without the
flag, but they report `preview_ok=false` and
`required_confirmation="--confirm-close-all true"`. Add the confirmation to
the preview as well when you want to verify that the same request is locally
eligible to switch to `--dry-run false`.

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

> **Note:** A per-symbol volume map (`MTDATA_TRADE_MAX_VOLUME_BY_SYMBOL`) also acts as an allowlist — a symbol missing from the map is rejected. Exposure and wallet-risk caps include both open positions and pending orders. Wallet-risk caps fail closed when any position or pending order lacks a quantifiable stop-loss or valid broker tick metadata.

Reduce-only checks the current open positions before allowing an opposite-side
order no larger than the net position. On hedging accounts, `trade_place` cannot
guarantee a reduction, so use `trade_close` with a position ticket instead.

See [ENV_VARS.md § Trade Guardrails](ENV_VARS.md#trade-guardrails) for every variable, defaults, formats, and a ready-to-copy `.env` block. A dry run returns a `guardrails_preview` so you can confirm which rules would fire before going live.

Live market and pending placements are serialized within one mtdata process so
the portfolio snapshot, guardrail decision, and broker submission are atomic
against concurrent tool calls. Separate mtdata processes connected to the same
account do not share that lock; use a single live-trade executor per MT5 account
when exposure or wallet-risk caps must be enforced across clients.

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
4. Preview: run the order with `--dry-run true`; inspect `guardrails_preview`, `preview_checks_performed`, `checks_not_performed`, and `broker_validation_not_performed`. A margin estimate appears under performed checks only when MT5 returned a finite estimate.
5. Go live with a **small** size and `--dry-run false`.
6. Verify: `mtdata-cli trade_get_open --json`, then manage with `trade_modify` / `trade_close`.

---

## Account and journal (read-only)

Look at the account **without** placing an order. None of these send
`trade_place` / `trade_modify` / `trade_close`.

| Question | Tool |
|----------|------|
| Which account is logged in? Balance, equity, margin? | `trade_account_info` |
| What is open right now? | `trade_get_open` |
| What is waiting as a pending order? | `trade_get_pending` |
| What filled recently? | `trade_history` |
| How did closed trades perform? | `trade_journal_analyze` |
| Session + quote + exposure in one bundle | `trade_session_context` |

```bash
mtdata-cli trade_account_info --json
mtdata-cli trade_get_open --json
mtdata-cli trade_get_pending --json
mtdata-cli trade_history --history-kind deals --minutes-back 10080 --json
mtdata-cli trade_journal_analyze --minutes-back 10080 --json
mtdata-cli trade_journal_analyze --magic 3001 --minutes-back 10080 --json
mtdata-cli trade_session_context EURUSD --json
```

`trade_history` and `trade_journal_analyze` default to the last 7 days when you
omit a window.
On accounts shared by multiple strategies, pass `--magic` to either command so
history pagination and journal metrics are scoped to one MT5 strategy identifier.

**History vs journal:** history is the raw deal/order tape. The journal
summarizes *exit* deals (wins, losses, averages) for review. It matches entry
fills by position ticket and allocates their commission and fees by closed
volume. Check `entry_cost_coverage`: an entry outside the requested history
window leaves that exit on the explicitly reported exit-deal-only PnL basis.

**Do not paste journal averages into Kelly sizing.**
`trade_journal_analyze` reports profit and loss in account currency per exit,
including matched entry costs where `entry_cost_coverage` permits.
[Kelly](GLOSSARY.md#kelly-criterion) needs a win rate and average win/loss that
are normalized to a consistent stake (for example R-multiples). Build those
inputs on purpose; see [TRADING_RISK.md](TRADING_RISK.md).

**Dense terms:** [Balance / equity / free margin](GLOSSARY.md#balance-equity-and-free-margin) · [Margin](GLOSSARY.md#margin-and-leverage) · [Magic number](GLOSSARY.md#magic-number)

---

## See Also

- [CLI.md § Trading](CLI.md#trading) — Command list and execution controls
- [ENV_VARS.md § Trade Guardrails](ENV_VARS.md#trade-guardrails) — Full guardrail variable reference
- [TRADING_RISK.md](TRADING_RISK.md) — Position sizing, VaR/CVaR, and stress tests
- [SAMPLE-TRADE-ADVANCED.md](SAMPLE-TRADE-ADVANCED.md) — An end-to-end analysis-to-execution workflow
- [Account terms](GLOSSARY.md#balance-equity-and-free-margin)
- [OUTPUT.md](OUTPUT.md) — Response envelope and error codes
