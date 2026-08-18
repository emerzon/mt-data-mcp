# Sample trade in the Web UI

**Audience:** User

Same research questions as [SAMPLE-TRADE.md](SAMPLE-TRADE.md), answered with
the chart workspace instead of the command line. Still a **research example**,
not financial advice. Use a demo account.

**Dense terms:** [EMA / RSI / MACD](GLOSSARY.md#moving-average) · [Pivot points](GLOSSARY.md#pivot-points) · [EWMA vol](GLOSSARY.md#ewma-exponentially-weighted-moving-average) · [Theta](GLOSSARY.md#theta-method) · [Barrier](GLOSSARY.md#barrier)

**Related:** [Web UI tour](WEBUI.md) · [CLI walkthrough](SAMPLE-TRADE.md) · [Advanced playbook](SAMPLE-TRADE-ADVANCED.md) · [Glossary](GLOSSARY.md)

You will need the UI running: [WEBUI.md](WEBUI.md#you-will-need).

### Fast path

Click **Idea**, leave the defaults, and compose. Read `direction`, `narrative`, geometry, and whether the dry-run preview is ok. Entry, take-profit, and stop-loss lines appear on the chart. That is the whole beginner path as one preview-only idea. It cannot place an order. Details: [TRADE_IDEAS.md](TRADE_IDEAS.md).

The numbered steps below still show the same questions on the chart and in dedicated panels.

---

## 1. Recent prices and a few indicators

1. Set symbol `EURUSD` and timeframe `H1`.
2. Confirm candles appear (about 200 bars is plenty — scroll left if you want more history).
3. Open **Indicators** and click **Sample trade**. That draws EMA 20 and EMA 50
   on the candles, plus RSI 14 and MACD in their own panes — the same set as
   the CLI guide. You can also tick them one by one. The choice is remembered
   for this symbol and timeframe.

**How to read it:** price above both moving averages often means short-term
strength. RSI near 70 is “stretched up,” near 30 is “stretched down.” None of
that is an order.

---

## 2–3. Daily range and pivot levels

1. Switch timeframe to `D1` for a moment if you want to *see* the daily
   candles, then switch back to `H1` for the working chart.
2. Turn on the **pivot** overlay (classic is the default).
3. Optionally turn on **support/resistance**.

Pivots are formula levels from the last completed bar (traditionally
yesterday’s high, low, and close). Price sitting just under the first
resistance and above the pivot is a common “test this level” picture — not a
signal by itself.

---

## 4. How far might price travel?

Open **Forecast** → **Volatility**. Keep method `ewma` and horizon `12` on H1
(about half a day). Run it.

The number is a typical swing size, not a promise. Use it so targets and stops
are not wildly smaller or larger than normal movement.

---

## 5. A simple price forecast

**Forecast** → **Price**. Method `theta`, horizon `12`, quantity `price`. Run it.
A forecast line overlays the chart.

Theta is a fast baseline. Treat the path as “one plausible sketch.” If you
need a band around it, run `forecast_conformal_intervals` from **Tools** (see
the [advanced playbook](SAMPLE-TRADE-ADVANCED.md)).

---

## 6. Odds for one take-profit / stop-loss pair

In **Tools**, run `forecast_barrier_prob` with a modest pair, for example
take-profit `0.40` and stop-loss `0.60` percent, direction `long`, horizon `12`.

Read three probabilities:

- hit the profit target first
- hit the stop first
- hit neither before time runs out

That is enough for the beginner path. Searching a full grid with
`forecast_barrier_optimize` (HMM paths, refine, Kelly) belongs in
[SAMPLE-TRADE-ADVANCED.md](SAMPLE-TRADE-ADVANCED.md).

---

## Putting it together

Combine structure (pivots), typical movement (volatility), a baseline forecast,
and one barrier pair into a *hypothesis*. Preview any order with dry-run on a
demo account. Do not run `wait_event` or live `trade_place` from this page’s
happy path.

Next: [SAMPLE-TRADE.md](SAMPLE-TRADE.md) for the same steps as copy-paste CLI,
or [TRADING_SAFETY.md](TRADING_SAFETY.md) if you move toward execution.
