# services/ — Data Access Layer

Thin service layer for external data sources. No business logic — data retrieval only. 4 files.

## FILE MAP

| File | Lines | Purpose |
|------|-------|---------|
| `data_service.py` | 1209 | **Single gateway** for all MT5 data: candles, ticks, market depth, symbols, account info |
| `finviz/` | package | Finviz web scraping: fundamentals, screening, news, economic calendar |
| `options_service.py` | — | Options chain data retrieval |
| `__init__.py` | — | Empty |

## CONVENTIONS

- Services are consumed by `core/` tool modules (`core/data.py`, `core/finviz.py`, etc.) — never called directly by end users.
- `data_service.py` handles MT5 connection init, credential loading from `.env`, and all MetaTrader5 API calls.
- `finviz/` uses the `finvizfinance` library for web scraping — no API key required.

## ANTI-PATTERNS

- **Never** add business logic (forecasting, pattern detection, etc.) to service files — they are pure data access.
- **Never** call MT5 API functions outside `data_service.py` or `utils/mt5.py` — centralize connection management.
- `data_service.py` is the largest file here (1209 lines) — when modifying, ensure MT5 connection guards are preserved.
