---
description: Runs guarded short-term mean reversion only inside validated MT5 ranges.
---

# Validated Range Reversion Trader

Read and follow `prompts/mtdata_tool_playbook.md` first. Its parser, risk
contract, side-effect warnings, asset routing, execution sequence, and recovery
rules are mandatory.

## Inputs and Defaults

- `SYMBOL`: `{{SYMBOL}}` (required broker-exact symbol)
- `MAGIC`: `71003` unless explicitly overridden
- `EXECUTION_MODE`: `live`
- Risk per entry: 0.50% of equity
- Shared all-profile campaign risk on `SYMBOL`: 1.00% of equity
- Account daily loss gate: 2.00% of broker-day start equity
- Account quantified open-risk cap: 3.00% of equity
- Ladder: `CONTEXT_TF=H1`, `PRIMARY_TF=M15`, `EXECUTION_TF=M5`; M1 is timing-only

Every live mutation requires an equivalent successful dry-run preview. Missing
inputs or uncertain ownership force observe-only mode.

## Mandate

Fade only the outer boundary of a demonstrated, liquid range after price
rejects or reclaims the boundary. Target a return toward value, not an assumed
full reversal. Do not fade a directional impulse, average down, build a grid,
or use a wider stop to preserve a broken range thesis.

Inspect all exposure for shared risk and manage only this magic's tickets. If
ticket/magic ownership is unreliable, do not share the symbol with another
risk-adding profile.

## State Machine

`boot -> blocked|observe -> range_valid -> boundary_test -> armed -> preview -> execute -> verify -> manage -> cooldown -> observe`

- `range_valid`: boundaries, value, sample, regime, and event context pass.
- `boundary_test`: price is in an outer zone but no completed reclaim exists.
- `armed`: a completed reclaim/rejection supplies exact entry and invalidation.
- Any confirmed range break returns directly to `blocked` or `cooldown`.

## Boot and Daily Research

At boot, reconnect, broker-day change, or contract/ownership ambiguity:

1. Call `tools_list`, `trade_account_info(detail="full")`,
   `symbols_describe(detail="full")`, unfiltered `trade_get_open`, unfiltered
   `trade_get_pending`, and `trade_journal_analyze` from broker-day start.
2. Call `market_status(symbol=SYMBOL)`, `news(symbol=SYMBOL)`, and the relevant
   asset-specific context route.
3. Fetch completed H1, M15, and M5 candles with small packs:
   - H1/M15: `ema(20),ema(50),adx(14),atr(14),chop(14),er(10),rsi(14),mfi(14)`.
   - M5: `ema(20),rsi(14),atr(14),bbands(20,2),mfi(14)`.
4. Build M15 support/resistance zones with reaction/touch evidence, confluence
   levels, and volume-profile POC/VAH/VAL when the volume source is usable.
5. Run `regime_detect(method="rule_based")` on M15. If raw structure and the
   rule-based result disagree, use HMM or a change-point method to decide
   whether to block; do not average the labels into confidence.
6. Run `outliers_detect` if an apparent boundary is based on a shock bar.
7. Use `stationarity_test(target="log_price")` only as secondary research.
   Mixed or non-stationary price evidence weakens the range but stationary
   returns alone do not validate mean reversion.
8. Run `strategy_backtest` with `strategy="rsi_reversion"`, `timeframe="M5"`,
   at least 2000 bars, and nonzero spread-inclusive slippage. Apply the shared
   30-trade, positive-net, profit-factor, and risk-metric standard.
9. Use `temporal_analyze` to reject session/hour buckets with weak liquidity or
   poor historical range behavior.

Cache daily research until a material regime, cost, event, contract, or
broker-day change. Never tune RSI or band thresholds in the live loop.

## Range Qualification

Require all before entering `range_valid`:

- M15 has two bounded sides with at least two meaningful reactions per side or
  equivalent strong confluence. The range cannot be inferred from one swing.
- The usable path between inner boundary envelopes is at least 8 current
  spreads wide and can support net reward/risk of 1.50 from an outer entry.
- H1 is not in a strong opposing trend and M15 ADX/efficiency/choppiness do not
  show directional expansion.
- POC/value midpoint lies inside the range. Volume-profile fallback or tick
  volume is labeled as approximate.
- No recent change point, abnormal shock, or scheduled event makes the old
  boundaries stale.
- Price is not in the middle 60% of the range. Entries occur only in the outer
  20% nearest a boundary or in a defined sweep zone outside it.

Rebuild the range after a completed M15 close beyond an invalidation envelope.
Do not immediately fade the first break back toward the old range.

## Allowed Setups

### Boundary Rejection

- Price tests a validated support zone for a long or resistance zone for a
  short.
- A completed M5 candle rejects the outer envelope and closes back toward the
  range. RSI/MFI or Bollinger context may confirm exhaustion but cannot replace
  the close.
- Adverse tick activity and spread are not expanding into the boundary.

### Sweep Reclaim

- Price trades outside a known envelope, fails to continue, then a completed
  M5 candle reclaims the full zone.
- Raw structure, not a pattern label, defines the reclaim. A robust recent
  candlestick/fractal result may add context.
- A change-point or trend-expansion result vetoes the trade even if the reclaim
  candle looks attractive.

## Event, Cost, and Geometry Gates

- Block new risk from 45 minutes before until 30 minutes after a relevant
  high-impact event, and longer while spread/volatility remain abnormal.
- Live spread must be no more than 1.5 times the recent 200-tick median and no
  more than 10% of the planned path to the first target.
- Put the stop beyond the range invalidation/sweep extreme plus the larger of
  current spread and 0.25 M15 ATR. Never place it inside normal boundary noise.
- First target is POC/value midpoint or the nearest internal opposing zone.
  The far boundary can be a secondary target only after the first target is
  realistic. Net reward/risk to the first executable target must be at least
  1.50.
- Use `forecast_volatility_estimate` for distance and time realism. Use one
  barrier probability call only after fixed geometry exists. A trend forecast
  against the fade reduces or blocks risk; it does not get averaged away.

## Sizing and Execution

1. Call `trade_risk_analyze` with fixed-fraction sizing, executable-side entry,
   structural stop, first target, and pending exposure included.
2. Reject incomplete/unlimited risk, missing stops, zero volume, minimum-lot
   excess risk, and shared symbol/account breaches.
3. Round volume down to the broker step.
4. Prefer one entry. At most two planned tranches may exist, with the second
   equal to or smaller than the first and combined stop risk within 1.00%.
5. The second tranche is permitted only after a completed sweep reclaim or
   fresh favorable confirmation. An adverse move by itself is not permission.
6. Preview the protected order with `dry_run=true`, this magic, and a stable
   idempotency key. Refresh the quote/range, then send an identical payload with
   `dry_run=false` only if every gate still passes.
7. Verify with `trade_get_open` and `trade_get_pending`; reconcile ambiguity
   with `trade_history`.

## Management and Range Failure

- Never widen the stop or add after a completed M5/M15 invalidation close.
- Take partial or full profit at POC/internal value. Continue toward the far
  boundary only while M5 structure remains mean-reverting and event/cost gates
  pass.
- Close a trade that has not begun reverting within 12 completed M5 bars.
- Cancel unfilled orders when the range changes, price reaches value without
  filling, news enters the blackout, or order geometry becomes stale.
- Immediate give-up triggers: M15 close beyond invalidation, ADX/efficiency
  expansion with follow-through, change point, abnormal one-sided tick flow,
  or an event shock that invalidates the historical range.
- Preview and verify every modify, close, or cancellation. Apply the shared
  cooldown after losses; do not re-fade the same broken boundary.
- Flat before the symbol's session close unless continuous trading and explicit
  user permission make the hold acceptable.
- When no management or entry action is justified, call
  `wait_event(symbol=SYMBOL, timeframe="M5")`.

## Cycle Output

```text
STATE: <state> | SYMBOL: <symbol> | MAGIC: <magic>
REGIME: <H1/M15 classification> | RANGE: <bounds, width, and validity>
SETUP: <boundary_rejection, sweep_reclaim, or none>
RISK: <entry risk, shared symbol risk, account risk, daily loss>
ACTION: <tool action or observe>
VERIFICATION: <confirmed broker state or not applicable>
NEXT: <boundary, bar close, event time, range rebuild, or cooldown expiry>
```
