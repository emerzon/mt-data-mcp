# Repository Guidelines

## Project Structure & Module Organization
- Source code lives under `src/mtdata/`:
  - `core/` — server and CLI entrypoints, config, constants.
  - `utils/` — MT5 adapters, indicators, denoise, helpers.
- Top-level wrappers: `server.py`, `cli.py` (delegate to `src/mtdata/core`).
- Tests and utilities: `tests/` (see `tests/test_forecast_methods.py`).
- Packaging/config: `pyproject.toml`, `requirements.txt`, `.env(.example)`.

## Build, Test, and Development Commands
- Create venv: `python -m venv .venv && source .venv/bin/activate` (Windows: `\\.venv\\Scripts\\activate`).
- Install deps: `pip install -r requirements.txt` (dev install: `pip install -e .`).
- Run server: `python server.py` (or, after editable install, `mtdata-server`).
- Run CLI: `python cli.py --help` (or `mtdata-cli --help`).
- Run forecasting test tool: `python tests/test_forecast_methods.py EURUSD H1 12` (writes JSON to `tests/test_results/`).

## Coding Style & Naming Conventions
- Python 3.8+; follow PEP 8 with 4‑space indentation.
- Use type hints and docstrings for public functions.
- Names: modules/functions `snake_case`, classes `CamelCase`, constants `UPPER_SNAKE`.
- Place new server tools in `src/mtdata/core/server.py`; CLI args/wiring in `src/mtdata/core/cli.py`; shared helpers in `src/mtdata/utils/`.

## Testing Guidelines
- Preferred location: `tests/` with files named `test_*.py`.
- Existing suite: `tests/test_forecast_methods.py` (invoked as a script; no pytest required).
- Ensure new features have a runnable example or scriptable test; include sample CLI commands in PR description.

## Commit & Pull Request Guidelines
- Commits: concise, imperative subject; scope prefix when helpful (e.g., `core: add LTTB resample mode`).
- Reference issues with `#123` and explain user impact in the body.
- PRs must include: clear description, linked issues, how to run/verify (commands and expected outputs), and docs updates (`README.md`, `.env.example`) when config or behaviors change.

## Security & Configuration Tips
- Do not commit secrets. Use `.env` (copy from `.env.example`).
- MT5 credentials are optional; the server can attach to an already logged‑in terminal.
- If adding new env vars, document them and provide safe defaults.
