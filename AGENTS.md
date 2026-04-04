# PROJECT KNOWLEDGE BASE

**Generated:** 2026-03-07
**Commit:** ca6b231
**Branch:** main

## OVERVIEW

MetaTrader 5 research/automation toolkit exposing 52+ MCP tools for forecasting, regime detection, pattern recognition, signal processing, and trading. Python 3.14 backend (MCP server + CLI + FastAPI web API) with React/Vite frontend. ~88k Python LOC, ~3.4k TypeScript LOC.

## STRUCTURE

```
mtdata/
├── src/mtdata/
│   ├── bootstrap/    # Runtime init, settings, tool registration (4 files)
│   ├── core/         # 52 MCP tools, CLI, server, web API, all API-facing logic (56 files)
│   ├── forecast/     # Forecasting engines, backtests, methods registry (34 files)
│   │   └── methods/  # Individual model implementations (12 files)
│   ├── patterns/     # Chart/candlestick/Elliott wave detection (12 files)
│   │   └── classic_impl/  # Classic pattern algorithm implementations
│   ├── services/     # MT5 gateway, Finviz, options data access (4 files)
│   ├── shared/       # Cross-module schemas and constants (3 files)
│   └── utils/        # Indicators, denoising, dimension reduction, formatting (15 files)
├── tests/            # 122 test files, hybrid pytest/unittest.TestCase
├── webui/            # React + Vite + Tailwind frontend
│   └── src/          # App.tsx, 4 components, hooks, API client, chart lib
├── docs/             # User-facing documentation (18 files)
├── scripts/          # MT5 time offset detection, backtest plotting
└── prompts/          # Prompt templates
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Add/modify MCP tool | `src/mtdata/core/` | Each domain has its own file (data.py, forecast.py, trading.py, etc.) |
| Add forecast method | `src/mtdata/forecast/methods/` + `forecast_registry.py` | Register in registry, implement interface |
| Fix MT5 data access | `src/mtdata/services/data_service.py` | 1209 lines, main data gateway |
| Fix Finviz integration | `src/mtdata/services/finviz/` | Package implementation for web scraping |
| Modify pattern detection | `src/mtdata/patterns/` | `classic.py` delegates to `classic_impl/` |
| Change indicators | `src/mtdata/utils/indicators.py` | 100+ technical indicators |
| Edit denoising filters | `src/mtdata/utils/denoise/` | Package implementation for denoising filters |
| Modify web UI | `webui/src/` | App.tsx (547 lines) is main, 4 components |
| Server/transport config | `src/mtdata/core/server.py` | SSE, stdio, streamable-HTTP modes |
| CLI changes | `src/mtdata/core/cli/` | Package surface for dynamic tool discovery |
| Trading logic | `src/mtdata/core/trading_*.py` | Split across ~12 files by concern |
| Report generation | `src/mtdata/core/report_*.py` + `report_templates/` | Templates in subdirectory |
| Regime detection | `src/mtdata/core/regime/` | Package implementation for HMM + rule-based detection |
| Shared schemas | `src/mtdata/shared/schema.py` | 791 lines, Pydantic models |
| Runtime/env setup | `src/mtdata/bootstrap/` | Settings, tool bootstrap, runtime init |

## ENTRY POINTS

No root wrapper scripts. All entry points are console scripts in `pyproject.toml`:
- `mtdata-cli` → `mtdata.core.cli:main` — Dynamic CLI with argparse subparsers per tool
- `mtdata-sse` → `mtdata.core.server:main_sse` — MCP server (SSE transport)
- `mtdata-stdio` → `mtdata.core.server:main_stdio` — MCP server (stdio transport)
- `mtdata-streamable-http` → `mtdata.core.server:main_streamable_http` — MCP server (HTTP transport)
- `mtdata-webapi` → `mtdata.core.web_api:main_webapi` — FastAPI + bundled UI on :8000

Request flow: `entry point → load_environment() → bootstrap_tools() → mcp.run() / uvicorn.run()`

## CONVENTIONS

- **Python**: PEP 8, 4-space indent, type hints on public functions. `snake_case` functions/modules, `PascalCase` classes, `UPPER_SNAKE_CASE` constants.
- **Imports**: Import individual modules from `core/` directly — `__init__.py` is intentionally empty to prevent circular deps.
- **Domain logic**: Lives in `src/mtdata/*` modules, never in entry scripts.
- **TypeScript**: Strict mode. Components in `webui/src/components/` with `PascalCase` filenames. Hooks prefixed `use`.
- **No linter/formatter configs**: No ESLint, Prettier, Ruff, or mypy config files. Conventions are implicit.
- **Commit style**: `<area>: <imperative summary>` (e.g., `core: dedupe trading query logging`).

## ANTI-PATTERNS (THIS PROJECT)

- **Never** import everything from `core/__init__.py` — causes circular dependencies.
- **Never** commit `.env` or credentials. MT5 credentials go in `.env` only.
- **Never** place real orders without validating account context — `trade_*` tools execute real trades.
- **Never** re-normalize already-normalized time data (see `forecast/common.py`).
- **Do not** add domain logic to entry scripts or bootstrap modules — they are thin wrappers.
- **Do not** use `pkg_resources` — deprecated, suppress warnings if unavoidable in dependencies.

## DEPENDENCY GROUPS

| Group | Purpose | Key Packages |
|-------|---------|-------------|
| Core | Always installed | MetaTrader5, fastmcp, pandas, numpy, scipy, scikit-learn, matplotlib, finvizfinance |
| forecast-classical | Classical/ML models | arch, statsforecast, sktime, mlforecast, optuna, QuantLib, lightgbm |
| forecast-foundation | Deep learning models | torch, chronos-forecasting, transformers, timesfm |
| patterns-ext | External pattern lib | precise-patterns (git dep) |
| web | Web UI backend | fastapi, uvicorn |
| all | Everything | Union of all above |

**Excluded from 3.14**: gluonts/Lag-Llama, hnswlib, tsdownsample (wheel incompatibility).

## COMMANDS

```bash
pip install -e .                          # Lean core install
pip install -r requirements.txt           # Full install (all extras)
mtdata-cli --help                         # CLI commands
python -m pytest tests/                   # Full test suite
python -m pytest tests/test_data_service.py  # Single test file
cd webui && npm install && npm run dev    # Frontend dev server (:5173 → proxy :8000)
cd webui && npm run build                 # Production frontend bundle
```

## TESTING

- Runner: pytest. Both `pytest`-style and `unittest.TestCase` present.
- Naming: `test_*.py` files, `test_*` functions.
- **Mock MT5**: Always mock/stub MT5 interactions unless explicitly testing integration. `conftest.py` handles `sys.modules` stubbing for MT5 and torch.
- File suffixes signal scope: `_pure` (unit), `_business_logic` (logic), `_coverage` (comprehensive), `_extended` (integration).
- No CI pipeline. No coverage gate. Manual test runs only.

## NOTES

- **Windows required**: MT5 only runs on Windows. macOS/Linux users connect remotely via MCP/Web API.
- **Python 3.14 only**: Pinned in `pyproject.toml`. Some deps have version ceilings (numpy <2.4, pandas <3, scikit-learn <1.8).
- **Large files**: 67 files exceed 500 lines. Core has the most complexity (8 files >500 lines). Forecast methods and utils also heavy.
- **No CI/CD**: No GitHub Actions, Makefile, Docker, or pre-commit hooks. All builds/tests are manual.
- **CORS**: Web API has `allow_credentials=True` with permissive CORS in dev (see `web_api_runtime.py`).
