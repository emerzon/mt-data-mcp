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
python cli.py finviz_fundamentals AAPL --json

# Latest news for a stock
python cli.py finviz_news NVDA --json

# Screen for undervalued tech stocks
python cli.py finviz_screen --filters '{"Sector": "Technology", "P/E": "Under 15"}' --json

# This week's economic calendar
python cli.py finviz_calendar --json

# Forex performance snapshot
python cli.py finviz_forex --json
```

---

## Company Research

### `finviz_fundamentals`

Get fundamental metrics for a US stock.

```bash
python cli.py finviz_fundamentals AAPL --json
```

**Returns:** P/E, Forward P/E, EPS, market cap, sector, industry, dividend yield, 52-week range, analyst recommendations, and 60+ other metrics.

### `finviz_description`

Get a company's business description.

```bash
python cli.py finviz_description TSLA --json
```

### `finviz_peers`

Find peer companies in the same sector/industry.

```bash
python cli.py finviz_peers MSFT --json
```

**Returns:** List of ticker symbols for comparable companies.

### `finviz_ratings`

Get analyst ratings history.

```bash
python cli.py finviz_ratings GOOGL --json
```

**Returns:** Date, analyst firm, rating action (upgrade/downgrade/initiate), rating, and price target.

---

## News

### `finviz_news`

Get stock-specific or general market news.

```bash
# Stock-specific news
python cli.py finviz_news NVDA --limit 10 --json

# General market news (no symbol)
python cli.py finviz_news --limit 20 --json
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `symbol` | (optional) | Stock ticker. Omit for general news. |
| `--limit` | 20 | Max news items |
| `--page` | 1 | Pagination page |

### `finviz_market_news`

Get broad financial market headlines or blog posts.

```bash
python cli.py finviz_market_news --news-type news --limit 20 --json
python cli.py finviz_market_news --news-type blogs --json
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
python cli.py finviz_insider AAPL --limit 10 --json
```

**Returns:** Owner name, relationship (CEO, CFO, Director, etc.), transaction type (buy/sell), shares, value, and date.

### `finviz_insider_activity`

Get market-wide insider trading activity.

```bash
# Latest insider trades across the market
python cli.py finviz_insider_activity --option latest --json

# Top insider buys this week
python cli.py finviz_insider_activity --option "top week" --json

# Only insider buys
python cli.py finviz_insider_activity --option "insider buy" --json
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
python cli.py finviz_screen --filters '{"Exchange": "NASDAQ", "Sector": "Technology"}' --json

# Large-cap value stocks
python cli.py finviz_screen --filters '{"Market Cap.": "Large ($10bln to $200bln)", "P/E": "Under 15"}' --json

# High-dividend stocks with valuation view
python cli.py finviz_screen --filters '{"Dividend Yield": "Over 5%"}' --view valuation --json

# Sort by market cap descending
python cli.py finviz_screen --filters '{"Sector": "Healthcare"}' --order "-marketcap" --json
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
python cli.py finviz_forex --json
```

**Returns:** Performance data for major currency pairs (daily change, weekly change, etc.).

### `finviz_crypto`

Get cryptocurrency performance.

```bash
python cli.py finviz_crypto --json
```

**Returns:** Price, daily change, volume, and market cap for major cryptocurrencies.

### `finviz_futures`

Get futures market performance.

```bash
python cli.py finviz_futures --json
```

**Returns:** Performance data for major futures contracts (commodities, indices, bonds, currencies).

---

## Calendars

### `finviz_calendar`

Get economic, earnings, or dividends calendar.

```bash
# Economic calendar (default)
python cli.py finviz_calendar --json

# Earnings calendar
python cli.py finviz_calendar --calendar earnings --json

# High-impact economic events only
python cli.py finviz_calendar --calendar economic --impact high --json

# Date range filter
python cli.py finviz_calendar --date-from 2026-03-01 --date-to 2026-03-15 --json
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--calendar` | `economic` | `economic`, `earnings`, or `dividends` |
| `--impact` | (all) | Economic only: `low`, `medium`, `high` |
| `--date-from` | (optional) | Start date `YYYY-MM-DD` |
| `--date-to` | (optional) | End date `YYYY-MM-DD` |
| `--limit` | 100 | Max events |
| `--page` | 1 | Pagination page |

### `finviz_earnings`

Get upcoming earnings announcements.

```bash
python cli.py finviz_earnings --period "This Week" --json
python cli.py finviz_earnings --period "Next Week" --json
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
| Company fundamentals | `python cli.py finviz_fundamentals AAPL` |
| Company description | `python cli.py finviz_description TSLA` |
| Peer companies | `python cli.py finviz_peers MSFT` |
| Analyst ratings | `python cli.py finviz_ratings GOOGL` |
| Stock news | `python cli.py finviz_news NVDA` |
| Market news | `python cli.py finviz_market_news` |
| Insider trades (stock) | `python cli.py finviz_insider AAPL` |
| Insider trades (market) | `python cli.py finviz_insider_activity` |
| Stock screener | `python cli.py finviz_screen --filters '{"Sector":"Technology"}'` |
| Forex snapshot | `python cli.py finviz_forex` |
| Crypto snapshot | `python cli.py finviz_crypto` |
| Futures snapshot | `python cli.py finviz_futures` |
| Economic calendar | `python cli.py finviz_calendar` |
| Earnings calendar | `python cli.py finviz_earnings` |

---

## See Also

- [CLI.md](CLI.md) — Command usage
- [GLOSSARY.md](GLOSSARY.md) — Term definitions
- [SAMPLE-TRADE.md](SAMPLE-TRADE.md) — Trade analysis workflow
