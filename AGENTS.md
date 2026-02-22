# Repository Guidelines

## Project Structure & Module Organization
Primary Python code lives in `src/mtdata/`:
- `core/`: CLI/MCP server wiring, schemas, API-facing tools.
- `forecast/`: forecasting engines, backtests, and method registries.
- `services/`: MT5/Finviz data access.
- `patterns/` and `utils/`: detection logic and shared helpers.

Entry scripts at repo root (`cli.py`, `server.py`, `webui.py`) are thin wrappers. Tests are under `tests/` (mostly `test_*.py`). Frontend assets live in `webui/` (`src/`, `index.html`, Vite config). Documentation is in `docs/`, and utility scripts are in `scripts/`.

## Build, Test, and Development Commands
- `pip install -r requirements.txt`: install Python dependencies.
- `pip install -e .`: editable install with console entry points (`mtdata-cli`, `mtdata-sse`, `mtdata-stdio`, `mtdata-webapi`).
- `python cli.py --help`: inspect available CLI commands.
- `python server.py`: start the MCP server.
- `python webui.py`: start FastAPI + bundled UI backend on `127.0.0.1:8000`.
- `python -m pytest tests/`: run full test suite.
- `python -m pytest tests/test_data_service.py`: run a focused test file.
- `cd webui && npm install && npm run dev`: run frontend dev server.
- `cd webui && npm run build`: produce production frontend bundle.

## Coding Style & Naming Conventions
Use 4-space indentation in Python and follow PEP 8 with type hints on new/changed public functions. Use `snake_case` for modules/functions, `PascalCase` for classes, and `UPPER_SNAKE_CASE` for constants. Keep domain logic in `src/mtdata/*` modules, not wrapper scripts. In React/TypeScript, keep components in `webui/src/components` with `PascalCase` filenames and hooks prefixed with `use`.

## Testing Guidelines
Pytest is the runner; both `pytest`-style and `unittest.TestCase` tests are present. Name new tests `test_*.py` and test functions `test_*`. Prefer deterministic tests with MT5 interactions mocked/stubbed unless explicitly validating integration behavior. No strict coverage gate is defined; maintain or improve coverage in touched areas.

## Commit & Pull Request Guidelines
Recent history favors short, imperative subjects, often scoped (for example, `core: ...`, `forecast: ...`, `docs: ...`). Prefer: `<area>: <imperative summary>`. Keep commits focused and avoid placeholder messages (for example, `ckpt`).

PRs should include:
- concise problem/solution summary,
- linked issue (if applicable),
- exact test commands run and results,
- screenshots or recordings for `webui/` changes,
- environment notes for MT5-dependent behavior.

## Security & Configuration Tips
Keep credentials in `.env` and never commit secrets. MT5 actions can place real orders; validate account context and use a demo account for development and testing.
