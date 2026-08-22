# services/ — Data Access Layer

Thin service layer for external data sources. No business logic — data retrieval only. Python modules including the `data_service/`, `finviz/`, and `research/` packages.

## FILE MAP

| File | Lines | Purpose |
|------|-------|---------|
| `data_service/` | package | **MT5 history gateway**: candles, ticks. `__init__.py` re-exports the public API; `query.py` (datetime/calendar bounds, `query_applied`), `errors.py` (no-data/weekend/last-error), `candles.py` (warmup, denoise/indicators, `fetch_candles`), `ticks.py` (flag decode, backward/forward fetch, `fetch_ticks`). `core/data/` stays the MCP/wait-event layer. |
| `finviz/` | package | Finviz web scraping: fundamentals, screening, news, economic calendar |
| `research/` | package | Capability registry for news/calendar/profile/screener adapters |
| `options_service.py` | — | Options chain data retrieval |
| `news_service.py` | — | MT5/news data retrieval |
| `unified_news.py` | — | Unified news provider orchestration |
| `news_embeddings.py` | — | Optional news embedding/reranking backend |
| `news_text.py` | — | News text helpers |
| `__init__.py` | — | Empty |

## CONVENTIONS

- Services are consumed by `core/` tool modules (`core/data/`, `core/finviz/`, etc.) — never called directly by end users.
- `data_service/` handles MT5 connection init, credential loading from `.env`, and all MetaTrader5 history API calls.
- `finviz/` uses the `finvizfinance` library for web scraping — no API key required.

## ANTI-PATTERNS

- **Never** add business logic (forecasting, pattern detection, etc.) to service files — they are pure data access.
- **Never** call MT5 API functions outside `data_service/` or `utils/mt5.py` — centralize connection management.
- `data_service/` is the MT5 history gateway — when modifying, ensure MT5 connection guards are preserved.
