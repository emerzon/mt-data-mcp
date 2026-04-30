# Finviz Fundamental Data

The `finviz_*` tools provide access to US equity fundamental data, screening, news, insider activity, and macro market snapshots (forex, crypto, futures, economic calendars) via [Finviz](https://finviz.com).

**Related:**
- [CLI.md](CLI.md) — Command usage and output formats
- [GLOSSARY.md](GLOSSARY.md) — Term definitions
- [SETUP.md](SETUP.md) — Installation and configuration

---

> **Note:** Finviz data covers **US-listed equities** and global macro snapshots. It is separate from MT5 market data. Data may be delayed 15–20 minutes depending on the source.

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

Get stock-specific or general market news.

```bash
# Stock-specific news
mtdata-cli finviz_news NVDA --limit 10 --json

# General market news (no symbol)
mtdata-cli finviz_news --limit 20 --json
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `symbol` | (optional) | Stock ticker. Omit for general news. |
| `--limit` | 20 | Max news items |
| `--page` | 1 | Pagination page |

Stock-specific responses keep the legacy `news` rows and also include a
normalized `items` list with `title`, `source`, `published_at`, and `url`.

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

# Large-cap value stocks
mtdata-cli finviz_screen --filters '{"Market Cap.": "Large ($10bln to $200bln)", "P/E": "Under 15"}' --json

# High-dividend stocks with valuation view
mtdata-cli finviz_screen --filters '{"Dividend Yield": "Over 5%"}' --view valuation --json

# Sort by market cap descending
mtdata-cli finviz_screen --filters '{"Sector": "Healthcare"}' --order "-marketcap" --json
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--filters` | (optional) | JSON string of filter criteria |
| `--order` | (optional) | Sort: e.g., `-marketcap` (desc), `price` (asc) |
| `--limit` | 50 | Max results |
| `--page` | 1 | Pagination page |
| `--view` | `overview` | `overview`, `valuation`, `financial`, `ownership`, `performance`, `technical` |

**Common filter keys:** `Exchange`, `Index`, `Sector`, `Industry`, `Country`, `Market Cap.`, `P/E`, `Forward P/E`, `PEG`, `P/S`, `P/B`, `Dividend Yield`, `EPS growth this year`, `Return on Equity`, `Current Ratio`, `Analyst Recom.`, `RSI (14)`, `50-Day Simple Moving Average`, `Average Volume`, `Price`, `Beta`.

---

## Macro Market Snapshots

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
mtdata-cli finviz_calendar --date-from 2026-03-01 --date-to 2026-03-15 --json
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--calendar` | `economic` | `economic`, `earnings`, or `dividends` |
| `--impact` | (all) | Economic only: `low`, `medium`, `high` |
| `--date-from` | (optional) | Start date `YYYY-MM-DD` |
| `--date-to` | (optional) | End date `YYYY-MM-DD` |
| `--limit` | 100 | Max events |
| `--page` | 1 | Pagination page |

Economic calendar data is based on Finviz JSON API fields: `date`, `event`,
`ticker`, `importance` (`1` low, `2` medium, `3` high), `actual`, `forecast`,
`previous`, `category`, `reference`, and `referenceDate` when present. The
`finviz_calendar` tool presents these as normalized keys, including `symbol`
for Finviz `ticker` and `reference_date` for `referenceDate`.

### `finviz_earnings`

Get upcoming earnings announcements.

```bash
mtdata-cli finviz_earnings --period "This Week" --json
mtdata-cli finviz_earnings --period "Next Week" --json
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--period` | `This Week` | `This Week`, `Next Week`, `Previous Week`, `This Month` |
| `--limit` | 50 | Max items |
| `--page` | 1 | Pagination page |

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
