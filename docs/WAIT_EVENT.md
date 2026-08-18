# Wait for a candle close or a fill

**Audience:** User

Sometimes a script or assistant should **pause** until something happens —
the current hour closes, or an order fills — instead of polling forever by
hand. `wait_event` does that pause for you.

It **blocks** until it finishes or times out. That is useful in the CLI or an
MCP session. It is a poor fit for the Web UI Tools runner, which has no
progress bar for a long wait.

**Dense terms:** [Timeframe](GLOSSARY.md#timeframe-and-candles-bars) · [Pending vs market order](GLOSSARY.md#market-vs-pending-orders)

**Related:** [CLI](CLI.md) · [Market discovery](MARKET.md) · [Trading safety](TRADING_SAFETY.md) · [Web UI](WEBUI.md)

---

## Pick a wait target

Choose at least one:

- a **timeframe** — stop at the next candle boundary (for example the next H1 close),
- **`max_wait_seconds`** — stop after a fixed number of seconds, or bound a timeframe wait.

Do not omit both. Setting both creates a bounded boundary wait: the command
returns at the candle boundary when it fits within the budget, or reports
`wait_budget_exceeded` without starting an over-budget sleep.

---

## Example 1 — wait for the next H1 close

```bash
mtdata-cli wait_event EURUSD --timeframe H1 --watch-for '[]' \
  --max-wait-seconds 3700 --json
```

The explicit empty watch list makes this a candle-boundary-only wait, so the
next research step runs on a *completed* hour rather than returning early for
an inferred market or account event. The 3,700-second budget covers at most one
H1 boundary plus the default close buffer, so an unattended call is bounded.

Use a shorter budget when the script should start only near the boundary:

```bash
mtdata-cli wait_event EURUSD --timeframe H1 --watch-for '[]' \
  --max-wait-seconds 300 --json
```

That five-minute budget returns `wait_budget_exceeded` without sleeping unless
the next buffered H1 close already fits inside it.

A timeframe wait with no symbol and no extra watch list is a pure clock wait
(no candle payload). Passing the symbol includes a best-effort closed-candle
snapshot when the boundary hits. If a symbol is supplied and `--watch-for` is
omitted, the command installs its inferred watcher set and may return mid-bar.

---

## Example 2 — wait up to 30 seconds for a fill

```bash
mtdata-cli wait_event EURUSD --max-wait-seconds 30 \
  --watch-for '[{"type":"order_filled","symbol":"EURUSD"}]' --json
```

If nothing fills in time, the command fails (`success=false`,
`error_code=wait_event_timeout`) and the CLI exits nonzero. That is
intentional: a script can decide whether to retry.

Omitting `--watch-for` in duration mode makes this a pure timer. It does not
connect to MT5 or poll order, position, or market state. Completion reports
`success=true`, `matched=false`, `timed_out=false`, `timer_only=true`, and
`completion_reason=duration_elapsed`. `timed_out` is reserved for an explicit
event wait whose deadline expires.

Pass an explicit `--watch-for` when the wait should return early for an event,
as in the fill example above. In timeframe mode, omitted watchers still include
the lightweight order/position lifecycle, pending-stop proximity, activity,
and tick-count set. They do **not** fetch support/resistance or pivot zones.

---

## Web UI warning

Do **not** run `wait_event` from **Tools** in the chart workspace for a long
block. Use `mtdata-cli` or an MCP assistant instead. See [WEBUI.md](WEBUI.md#do-not-run-these-from-the-browser).

---

## Deeper detail

- Basket: pass `--symbols` (up to 12) *or* one `--symbol`, not both. The wait
  returns on the first match.
- `accept_preexisting=true` returns immediately if a state-style watcher is
  already true at startup. The default waits for a *new* transition.
- Put candle-close boundaries in `end_on`, not in `watch_for`.
- `--detail full` adds polling and timing diagnostics.
- Tool contract and edge cases live in the `wait_event --help` text.
