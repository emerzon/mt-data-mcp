# Finviz fundamentals

Pull **US equity fundamentals**, screens, news, insider activity, and macro snapshots (forex, crypto, futures, economic calendars) via [Finviz](https://finviz.com) — useful when your MT5 workflow also needs equity or calendar context.

These tools complement MT5; they do not replace the terminal for live FX/CFD quotes.

Finviz `symbol` values are exchange tickers such as `AAPL`, not necessarily the
connected broker's MT5 identifiers. A broker may expose several contracts for
one ticker (for example a cash-session and a 24-hour CFD), so mtdata does not
guess between them. If an MT5 tool rejects the bare ticker, inspect its
structured `details.did_you_mean` candidates and choose the intended contract;
use `symbols_list --search AAPL` when a wider catalog view is needed.

**Related:** [CLI](CLI.md) · [Glossary](GLOSSARY.md) · [Setup](SETUP.md)

---

> **Note:** Coverage is mainly **US-listed equities** plus global macro snapshots. Data may be delayed 15–20 minutes depending on the source.

---

## Quick Start

```bash
# Company fundamentals (P/E, EPS, market cap, etc.)
mtdata-cli finviz_fundamentals AAPL --json

# Latest news for a stock
mtdata-cli finviz_news NVDA --json

# Screen for undervalued tech stocks
mtdata-cli finviz_screen --filters '{"Sector": "Technology", "P/E": "Under 15"}' --json

# This week's economic calendar
mtdata-cli finviz_calendar --json

# Forex performance snapshot
mtdata-cli finviz_forex --json
```

---

## Company Research

### `finviz_fundamentals`

Get fundamental metrics for a US stock.

```bash
mtdata-cli finviz_fundamentals AAPL --json
```

**Returns:** P/E, Forward P/E, EPS, market cap, sector, industry, dividend yield, 52-week range, analyst recommendations, and 60+ other metrics.

### `finviz_description`

Get a company's business description.

```bash
mtdata-cli finviz_description TSLA --json
```

### `finviz_peers`

Find peer companies in the same sector/industry.

```bash
mtdata-cli finviz_peers MSFT --json
```

**Returns:** List of ticker symbols for comparable companies.

### `finviz_ratings`

Get analyst ratings history.

```bash
mtdata-cli finviz_ratings GOOGL --json
```

**Returns:** Date, analyst firm, rating action (upgrade/downgrade/initiate), rating, and price target.

---

## News

### `finviz_news`

Get stock-specific market news. Use `finviz_market_news` for general market
headlines/blogs, or the general `news` tool for the unified news workflow.

```bash
# Stock-specific news
mtdata-cli finviz_news NVDA --limit 10 --json
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `symbol` | (required) | Stock ticker. |
| `--limit` | 20 | Max news items |
| `--page` | 1 | Pagination page |

Stock-specific responses use a normalized `items` list with `title`, `source`,
`published_at`, and `url`. Ticker-page membership is reported as
`provider_associated`; only an explicit ticker token in the headline is labeled
`direct_symbol`. The response includes the provider context and relevance basis
because ticker pages can also contain peer, industry, and macro stories.
Finviz tools may accept one-based `--page` inputs,
but responses use the shared offset-based `pagination` object documented in
[OUTPUT.md](OUTPUT.md#pagination); flat provider `page`/`pages` fields are not
emitted.

### `finviz_market_news`

Get broad financial market headlines or blog posts.

```bash
mtdata-cli finviz_market_news --news-type news --limit 20 --json
mtdata-cli finviz_market_news --news-type blogs --json
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--news-type` | `news` | `news` for headlines, `blogs` for blog posts |
| `--limit` | 20 | Max items |
| `--page` | 1 | Pagination page |

---

## Insider Trading

### `finviz_insider`

Get insider trading activity for a specific stock.

```bash
mtdata-cli finviz_insider AAPL --limit 10 --json
```

**Returns:** Owner name, relationship (CEO, CFO, Director, etc.), transaction type (buy/sell), shares, value, and date.

### `finviz_insider_activity`

Get market-wide insider trading activity.

```bash
# Latest insider trades across the market
mtdata-cli finviz_insider_activity --option latest --json

# Top insider buys this week
mtdata-cli finviz_insider_activity --option "top week" --json

# Only insider buys
mtdata-cli finviz_insider_activity --option "insider buy" --json
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--option` | `latest` | `latest`, `top week`, `top owner trade`, `insider buy`, `insider sale` |
| `--limit` | 50 | Max items |
| `--page` | 1 | Pagination page |

---

## Stock Screening

### `finviz_screen`

Screen stocks using Finviz's powerful filter engine.

```bash
# Tech stocks on NASDAQ
mtdata-cli finviz_screen --filters '{"Exchange": "NASDAQ", "Sector": "Technology"}' --json

# Same screen using compact key=value syntax
mtdata-cli finviz_screen --filters "exchange=NASDAQ,sector=Technology" --json

# Discrete comparison aliases for Finviz filters
mtdata-cli finviz_screen --filters "pe_under=15,beta_under=1" --json

# Same screen using native Finviz shorthand tokens
mtdata-cli finviz_screen --filters "exch_nasd,sec_technology" --json

# Large-cap value stocks
mtdata-cli finviz_screen --filters '{"Market Cap.": "Large ($10bln to $200bln)", "P/E": "Under 15"}' --json

# High-dividend stocks with valuation view
mtdata-cli finviz_screen --filters '{"Dividend Yield": "Over 5%"}' --view valuation --json

# Sort by market cap descending
mtdata-cli finviz_screen --filters '{"Sector": "Healthcare"}' --order=-marketcap --json
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--filters` | (optional) | JSON object, `key=value` pairs, or Finviz shorthand tokens |
| `--order` | (optional) | Sort: e.g., `--order=-marketcap` (desc), `--order=price` (asc) |
| `--limit` | 20 | Max results per page |
| `--page` | 1 | Pagination page |
| `--view` | `overview` | `overview`, `valuation`, `financial`, `ownership`, `performance`, `technical` |

**Common filter keys:** `Exchange`, `Index`, `Sector`, `Industry`, `Country`, `Market Cap.`, `P/E`, `Forward P/E`, `PEG`, `P/S`, `P/B`, `Dividend Yield`, `EPS growth this year`, `Return on Equity`, `Current Ratio`, `Analyst Recom.`, `RSI (14)`, `50-Day Simple Moving Average`, `Average Volume`, `Price`, `Beta`.

**Filter formats:** JSON uses exact Finviz names, for example `{"Exchange":"NASDAQ"}`. Key-value pairs use compact keys and values such as `country=USA,marketcap=mega`; discrete comparison aliases such as `pe_under=15` and `beta_under=1` map to Finviz's available "Under/Over" filter options. Native shorthand uses Finviz URL tokens such as `cap_largeover,exch_nyse`; invalid tokens are reported in the error details.

### `finviz_filters_list`

Discover valid screener filters and their accepted values/tokens before building a `finviz_screen` query.

```bash
# List available filters
mtdata-cli finviz_filters_list --json

# Search filters by name
mtdata-cli finviz_filters_list --search dividend --json

# Show accepted values for one filter
mtdata-cli finviz_filters_list --filter-name "Market Cap." --json
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--search` | (optional) | Case-insensitive substring match; matched rows include accepted values/tokens |
| `--filter-name` | (optional) | Show accepted values and tokens for a single filter |
| `--limit` | 20 | Max filters per page |
| `--offset` | 0 | Pagination offset |

---

## Macro Market Snapshots

Forex, crypto, and futures snapshots use the same row schema at every detail
level: `price` is a delayed reference price, and `perf_*_pct` values are
percent (`1.0 = 1%`). Full detail may add source fields, but it does
not rename or re-unit those canonical values. Check the response `units` and
`performance_format` fields when consuming rows programmatically.

### `finviz_forex`

Get forex currency pairs performance.

```bash
mtdata-cli finviz_forex --json
```

**Returns:** Performance data for major currency pairs (daily change, weekly change, etc.).

### `finviz_crypto`

Get cryptocurrency performance.

```bash
mtdata-cli finviz_crypto --json
```

Finviz may round prices for very low-priced tokens to zero. In that case the
tool returns `price_status: unavailable_provider_rounded_zero`, omits `price`,
and includes a warning instead of presenting the rounded value as a tradable
zero price.

**Returns:** Price, daily change, volume, and market cap for major cryptocurrencies.

### `finviz_futures`

Get futures market performance.

```bash
mtdata-cli finviz_futures --json
```

**Returns:** Performance data for major futures contracts (commodities, indices, bonds, currencies).

---

## Calendars

### `finviz_calendar`

Get economic, earnings, or dividends calendar.

```bash
# Economic calendar (default)
mtdata-cli finviz_calendar --json

# Earnings calendar
mtdata-cli finviz_calendar --calendar earnings --json

# High-impact economic events only
mtdata-cli finviz_calendar --calendar economic --impact high --json

# Date range filter
mtdata-cli finviz_calendar --start 2026-03-01 --end 2026-03-15 --json
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--calendar` | `economic` | `economic`, `earnings`, or `dividends` |
| `--impact` | (all) | Economic only: `low`, `medium`, `high` |
| `--start` | current New York date | Start date `YYYY-MM-DD`; omitted ranges anchor to `America/New_York`, independent of host timezone. |
| `--end` | (optional) | End date `YYYY-MM-DD` |
| `--upcoming` | live default | Economic only: include unreleased future events; defaults on when no explicit date range is supplied and off for explicit ranges. |
| `--limit` | 20 | Max events after country and upcoming filters, ordered by scheduled time |
| `--page` | 1 | Pagination page |

Economic calendar data is based on Finviz JSON API fields: `date`, `event`,
`ticker`, `importance` (`1` low, `2` medium, `3` high), `actual`, `forecast`,
`previous`, `category`, `reference`, and `referenceDate` when present. The
`finviz_calendar` tool presents these as normalized keys, including `symbol`
for Finviz `ticker` and `reference_date` for `referenceDate`.
Root output includes `date_from`, `date_to`, and `calendar_timezone` so defaulted
calendar ranges remain explicit. Event timestamps use the separate root
`timezone` field. Economic rows also include `country_attribution` (`provider`,
`inferred`, or `unknown`); country-filtered responses warn when unknown rows
were excluded and paginate the filtered collection.

### `finviz_earnings`

Get a period-based earnings calendar. Current-week and current-month results
omit dates before the current New York calendar date before pagination, so the
default first page starts with upcoming reports. Pass `--include-elapsed true`
for the complete period-start archive. `previous-week` is always an archive.

```bash
mtdata-cli finviz_earnings --period this-week --json
mtdata-cli finviz_earnings --period this-week --include-elapsed true --json
mtdata-cli finviz_earnings --period next-week --json
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--period` | `this-week` | `this-week`, `next-week`, `previous-week`, `this-month` |
| `--limit` | 10 | Max items |
| `--page` | 1 | Pagination page |
| `--include-elapsed` | false | Include earlier dates for current periods |

---

## Quick Reference

| Task | Command |
|------|---------|
| Company fundamentals | `mtdata-cli finviz_fundamentals AAPL` |
| Company description | `mtdata-cli finviz_description TSLA` |
| Peer companies | `mtdata-cli finviz_peers MSFT` |
| Analyst ratings | `mtdata-cli finviz_ratings GOOGL` |
| Stock news | `mtdata-cli finviz_news NVDA` |
| Market news | `mtdata-cli finviz_market_news` |
| Insider trades (stock) | `mtdata-cli finviz_insider AAPL` |
| Insider trades (market) | `mtdata-cli finviz_insider_activity` |
| Stock screener | `mtdata-cli finviz_screen --filters '{"Sector":"Technology"}'` |
| List screener filters | `mtdata-cli finviz_filters_list` |
| Forex snapshot | `mtdata-cli finviz_forex` |
| Crypto snapshot | `mtdata-cli finviz_crypto` |
| Futures snapshot | `mtdata-cli finviz_futures` |
| Economic calendar | `mtdata-cli finviz_calendar` |
| Earnings calendar | `mtdata-cli finviz_earnings` |

---

## See Also

- [CLI.md](CLI.md) — Command usage
- [GLOSSARY.md](GLOSSARY.md) — Term definitions
- [SAMPLE-TRADE.md](SAMPLE-TRADE.md) — Trade analysis workflow
