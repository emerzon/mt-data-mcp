# core/ — API-Facing Layer

All 57 MCP tools (+1 conditional), CLI commands, web API endpoints, and server transport logic live here. This is the largest module (56 files, 8 files >500 lines).

## FILE MAP

### MCP Tools (one file per domain)

| File | Lines | Domain |
|------|-------|--------|
| `data.py` | — | Data fetch tools (candles, ticks, market depth) |
| `forecast.py` | 792 | Forecast generation/backtest tools |
| `trading.py` | — | Trading tool entry points |
| `patterns.py` | 706 | Pattern detection tools |
| `regime.py` | 1264 | Regime detection (HMM + rule-based) |
| `temporal.py` | 555 | Temporal/seasonal analysis tools |
| `causal.py` | 534 | Causal analysis tools |
| `finviz.py` | 518 | Finviz fundamentals/screening tools |
| `indicators.py` | — | Indicator computation tools |
| `market_depth.py` | — | Market depth tools |
| `symbols.py` | — | Symbol listing/search tools |
| `labels.py` | — | Labeling tools |
| `features.py` | — | Feature extraction tools |
| `pivot.py` | — | Pivot point tools |

### Trading (split by concern)

| File | Lines | Purpose |
|------|-------|---------|
| `trading_use_cases.py` | 1414 | Orchestration layer |
| `trading_orders.py` | 636 | Order placement/modification |
| `trading_execution.py` | 620 | Execution logic |
| `trading_positions.py` | — | Position queries |
| `trading_risk.py` | — | Risk calculations |
| `trading_validation.py` | — | Input validation |
| `trading_gateway.py` | — | MT5 trade API wrapper |
| `trading_account.py` | — | Account info queries |
| `trading_common.py` | — | Shared trading types |
| `trading_comments.py` | — | Order comment encoding |
| `trading_time.py` | — | Trading time helpers |
| `trading_requests.py` | — | Request models |

### Reports

| File | Lines | Purpose |
|------|-------|---------|
| `report.py` | — | Report tool entry points |
| `report_use_cases.py` | — | Report orchestration |
| `report_rendering.py` | 768 | Markdown/HTML rendering |
| `report_utils.py` | 607 | Report helper functions |
| `report_shared.py` | — | Shared report types |
| `report_requests.py` | — | Report request models |
| `report_templates/` | 8 files | Template implementations (basic.py = 1032 lines) |
| `reports/` | — | Generated report output |

### Server, CLI, Web API

| File | Purpose |
|------|---------|
| `server.py` | MCP server entry (SSE, stdio, streamable-HTTP) |
| `cli.py` (1514) | Dynamic CLI — discovers tools, builds argparse subparsers |
| `web_api.py` | Web API entry point |
| `web_api_runtime.py` | FastAPI app creation, CORS, uvicorn |
| `web_api_handlers.py` (697) | REST route handlers |
| `web_api_models.py` | Request/response Pydantic models |
| `server_utils.py` | Server helper functions |

### Infrastructure

| File | Purpose |
|------|---------|
| `_mcp_instance.py` | Singleton MCP server instance |
| `_mcp_tools.py` | Tool registration decorator, type coercion |
| `config.py` | Environment/MT5 config loading |
| `constants.py` | SERVICE_NAME, TIMEFRAME_MAP, TIMEFRAME_SECONDS |
| `schema.py` | Core-level Pydantic schemas |
| `schema_attach.py` | Schema attachment helpers |
| `error_envelope.py` | Standardized error response format |
| `execution_logging.py` | Execution/query logging |
| `unified_params.py` | Shared parameter definitions |
| `patterns_support.py` (896) | Complex pattern detection helpers |
| `patterns_requests.py` | Pattern request models |
| `patterns_use_cases.py` | Pattern detection orchestration |
| `data_requests.py` | Data request models |
| `data_use_cases.py` | Data fetch orchestration |
| `indicators_docs.py` | Indicator documentation strings |
| `mt5_gateway.py` | MT5 connection management |

## CONVENTIONS

- **Domain pattern**: `{domain}.py` (tools) → `{domain}_requests.py` (input models) → `{domain}_use_cases.py` (orchestration).
- **`__init__.py` is intentionally empty** — import modules directly to avoid circular deps.
- **`_` prefix** on `_mcp_instance.py` and `_mcp_tools.py` signals internal infrastructure.

## ANTI-PATTERNS

- **Never** import from `core/__init__.py`. Always `from mtdata.core.data import ...`.
- **Never** add business logic here — delegate to `forecast/`, `services/`, `utils/`, `patterns/`.
- **Never** bypass `error_envelope.py` for error responses in tool functions.
