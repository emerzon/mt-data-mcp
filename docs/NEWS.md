# News and calendar

**Audience:** User

See what is happening around a market — headlines and scheduled economic
releases — without opening a separate news terminal.

The everyday command is `news`. It merges Finviz, MetaTrader 5 calendar items,
and CNBC when that extra is installed, then ranks the mix. Use the
`finviz_news` family only when you need raw Finviz pages, blogs, or provider
pagination.

**Dense terms:** [Finviz](GLOSSARY.md#finviz)

**Related:** [Finviz reference](FINVIZ.md) · [CLI](CLI.md) · [Sample trade](SAMPLE-TRADE.md) · [Env vars (optional embeddings)](ENV_VARS.md#news-embeddings) (Operator)

---

## Quick start (read-only)

```bash
# Broad recent headlines
mtdata-cli news

# Focus on one instrument (FX, stocks, or crypto names your broker lists)
mtdata-cli news EURUSD --json
mtdata-cli news AAPL --json
```

In the Web UI, open **Tools**, search `news`, and run it with or without a
symbol. Same idea from an assistant: “Show me `news` for EURUSD. Do not trade.”

---

## What the buckets mean

With no symbol, you get the most important recent **general** items plus the
market-wide upcoming and recent economic calendar. Symbol requests narrow
calendar events to the instrument's currencies or macro exposure.

With a symbol, the result is split so you can scan quickly:

| Bucket | Plain idea |
|--------|------------|
| `general_news` | Market-wide headlines that are important even if they do not name your symbol. |
| `related_news` | Items that look relevant to this instrument (name, aliases, or theme). |
| `impact_news` | High-importance shocks (for example energy or geopolitical) that can move many markets. |
| `upcoming_events` | **Future** calendar releases tied to this instrument — the “what is still ahead” list. |
| `recent_events` | **Already published** calendar prints — useful for “what just came out.” |

Headline rows with a provider-observed instant use `published_at`. Some Finviz
market headlines expose only a date; those rows use `publication_date`,
`timestamp_precision=date`, and `source_timezone=America/New_York` instead of
an invented midnight timestamp. Provider order breaks ties among these
date-only rows. Economic-calendar rows use `scheduled_at`, including rows in
both event buckets, so a future release time cannot be mistaken for an article
publication time.

Broad compact output returns a global page of at most 10 rows, reserves the
next upcoming event (or a recent release when no future event remains), and
includes `pagination` plus `bucket_truncation` metadata. Use `--limit` for a
different global page size, `--limit-per-bucket` for independent bucket caps,
or `--detail full` for the uncapped selected buckets and richer matching
diagnostics. Compact symbol news keeps up to five rows per bucket by default.
The related-news selector reserves up to five of the newest direct-symbol
headlines before filling the remaining internal selection by relevance. Full
detail exposes `related_selection`, including whether that selection was
truncated. For the complete provider-ordered US-equity page, continue with
`finviz_news SYMBOL`; public `news` limits paginate the selected multi-source
feed rather than the raw provider candidate pool.
Calendar rows show both the absolute UTC `scheduled_at` timestamp and the
convenience `relative_time` label in the default TOON view.

Full detail also adds, when available, a `market_context` quote snapshot.
Finviz snapshot performance is expressed in canonical `*_pct` metadata fields
using percentage points (`1.0 = 1%`), and summaries render a `%` sign. The
high-frequency provider fractions are never exposed as unqualified decimals.

A small `--limit` still tries to keep at least one upcoming event visible so a
tight cap does not hide the next scheduled release.

---

## When to use Finviz tools instead

| Need | Tool |
|------|------|
| One stock’s Finviz news page | `finviz_news NVDA` |
| Broad Finviz headlines or blogs | `finviz_market_news` |
| Economic or earnings calendar only | `finviz_calendar` / `finviz_earnings` |
| Fundamentals, screeners, insiders | See [FINVIZ.md](FINVIZ.md) |

Finviz US-equity data is delayed about 15–20 minutes. Treat it as research
context, not a live tape.

---

## Deeper detail

- Matching uses symbol aliases, asset-class words, MetaTrader 5 metadata, and a
  lightweight text-similarity score. It is a helper, not a guarantee that every
  headline will move the price.
- Optional embedding rerank (downloads a model on first use) is **off** by
  default. See [ENV_VARS.md](ENV_VARS.md#news-embeddings).
- CNBC via `ycnbc` is an opt-in extra (`pip install -e ".[news-ycnbc]"`).
