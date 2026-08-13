# Find a market and read a quote

**Audience:** User

Before you forecast or think about a trade, answer four plain questions:

1. Does my broker list this symbol?
2. Is the market open (or is this quote stale)?
3. What is the current buy/sell price?
4. Which names on my list look active or cheap to trade right now?

**Dense terms:** [Bid / ask / spread](GLOSSARY.md#bidask-and-spread) · [Pip](GLOSSARY.md#pip) · [OHLCV](GLOSSARY.md#ohlcv) · [Support and resistance](GLOSSARY.md#support-and-resistance)

**Related:** [CLI](CLI.md) · [Web UI](WEBUI.md) · [Levels](LEVELS.md) · [Temporal sessions](TEMPORAL.md) · [Trading safety](TRADING_SAFETY.md)

---

## Quick start (read-only)

```bash
# What can MetaTrader 5 see?
mtdata-cli symbols_list --limit 10

# Details for one name (pip size, contract, live quote fields)
mtdata-cli symbols_describe EURUSD --json

# Latest bid / ask / spread
mtdata-cli market_ticker EURUSD --json

# Is a major stock exchange open, or can this FX symbol take new orders?
mtdata-cli market_status --region all --json
mtdata-cli market_status --symbol EURUSD --json
```

In the Web UI, picking a symbol and turning **Live** on is the ticker. **Tools**
can run any command in this page.

---

## Completed bars vs the live quote

A **completed bar** is a finished candle (yesterday’s close, last hour’s close).
A **live quote** is the current bid and ask.

Ranking tools keep those apart on purpose:

- Use bar fields (`close`, RSI, SMA) for “how did this market behave?”
- Use `bid`, `ask`, `mid`, and `quote_as_of` for “what would I pay *now*?”

If a live quote is locked, one-sided, or otherwise unsafe, spread-ranked scans
drop it unless you pass `--quote-usable-only false` to inspect it on purpose.

---

## Rank and scan a list

```bash
# Rank the current watchlist by spread, volume, and recent change
mtdata-cli symbols_top_markets --rank-by all --limit 5 --timeframe H1 --json

# Scan visible majors: strong RSI and price above its average
mtdata-cli market_scan --group "Forex\\Majors" --rsi-above 60 --price-vs-sma above \
  --sma-period 20 --timeframe H1 --lookback 120 --json
```

`symbols_top_markets` ranks. `market_scan` **filters** (spread caps, RSI
bands, and similar). Price-change ranks compare the previous completed close
with the latest completed close over exactly one requested timeframe bar.

---

## One-shot pre-trade snapshot

`market_snapshot` packs a quote, nearby levels, and patterns so you do not
have to call five tools:

```bash
mtdata-cli market_snapshot EURUSD --timeframe H1 --json

# Also attach optional regime + forecast sections
mtdata-cli market_snapshot EURUSD --timeframe H1 --sections all --horizon 8 --json
```

Still read-only. It does not place an order.

---

## Session context (when you are closer to trading)

```bash
mtdata-cli trade_session_context EURUSD --json
```

This is the “what is my account and this symbol doing right now?” bundle:
session, quote, and open or pending exposure. It does not send orders.

For “is this equity venue open?” vs “can I open a *new* position on this
broker symbol?”, `market_status` reports both. `can_open_new_positions` needs
a live-ready quote and an active session, not only a tradable symbol mode.

---

## Deeper detail

- Order book (depth / DOM) is off unless you set
  `MTDATA_ENABLE_MARKET_DEPTH_FETCH=1` *and* your broker supplies it.
- `wait_event` can pause until the next candle close or a fill — see
  [WAIT_EVENT.md](WAIT_EVENT.md). Do not run long waits from the Web UI.
- Quote quality and scan limits: [CLI.md](CLI.md#explore-available-symbols).
