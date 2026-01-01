# Repository Guidelines

## Project Structure & Module Organization
- `src/mtdata/`: Python package (MCP server, CLI, forecasting, reporting).
  - `core/`: tool registry + schemas, `server.py`, `cli.py`, FastAPI app (`web_api.py`).
  - `services/`: MetaTrader5 data access and service wrappers.
  - `utils/`, `forecast/`, `patterns/`: shared algorithms and helpers.
- `tests/`: unit/smoke tests (`test_*.py`).
- `docs/`: user guides and troubleshooting.
- `webui/`: Vite + React + Tailwind UI (build output: `webui/dist/`).
- `scripts/`: one-off utilities (e.g., plotting/backtest helpers).

## Build, Test, and Development Commands
- Install Python deps: `python -m pip install -r requirements.txt`
- Editable install (enables entry points): `python -m pip install -e .`
- Run MCP server: `python server.py` (or `mtdata-server` after install)
- Use CLI: `python cli.py symbols_list` (or `mtdata-cli symbols_list`)
- Run Web API: `python webui.py` (FastAPI via Uvicorn)
- WebUI dev: `cd webui && npm install && npm run dev`
- WebUI build/preview: `cd webui && npm run build && npm run preview`

## Coding Style & Naming Conventions
- Python: 4-space indentation, type hints for new/changed code, keep imports package-relative inside `src/mtdata/`.
- Naming: `snake_case` modules/functions, `PascalCase` classes, `UPPER_SNAKE_CASE` constants.
- Keep changes focused; update `docs/` when behavior or CLI output changes.

## Testing Guidelines
- Primary runner: `python -m pytest` (install if needed: `python -m pip install pytest`).
- Prefer deterministic unit tests; mock MetaTrader5/network access unless explicitly writing an integration test.

## Commit & Pull Request Guidelines
- Commit history favors short, scoped messages (e.g., `core: ...`, `forecast: ...`, `utils: ...`); use `scope: imperative summary` when possible.
- PRs: include what/why, how to test (commands + expected result), and screenshots for `webui/` changes.

## Security & Configuration Tips
- Do not commit secrets. Copy `.env.example` → `.env` for `MT5_LOGIN`, `MT5_PASSWORD`, `MT5_SERVER`, and timezone settings (`MT5_SERVER_TZ` or `MT5_TIME_OFFSET_MINUTES`).
- Use a demo account for any trading-related development.

