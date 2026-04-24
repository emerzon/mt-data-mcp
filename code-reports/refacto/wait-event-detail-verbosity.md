# wait_event detail verbosity

## Source report

- Original report: `code-reports/wait_event_verbose_not_detail.md`

## Status

Deferred for larger refactor.

## Validated issue

`wait_event` uses `verbose: bool = False` to control whether polling, timing, criteria, and event details are included, while most other user-facing tools use a string `detail` parameter for compact/full output control.

## Evidence

- `src/mtdata/core/data/__init__.py::wait_event` defines `verbose: bool = False`.
- The docstring tells callers to set `verbose=true` for polling/timing details and the full criteria echo.
- `src/mtdata/core/data/__init__.py::_compact_wait_event_public_result` keeps all fields when `verbose` is truthy and strips timing/polling/criteria fields otherwise.
- `tests/core/test_wait_event.py` exercises the public `wait_event` wrapper and mocks `_compact_wait_event_public_result`, indicating the wrapper behavior is part of the tested surface.

## Why this should not be fixed inline

Replacing `verbose` with `detail` changes the public MCP schema and generated CLI arguments. Keeping `verbose` while adding `detail` would increase the API surface and duplicate output controls, which is the same anti-pattern this report identifies.

## Recommended approach

Handle `wait_event` in the same output-control migration as other verbosity inconsistencies. Decide whether deprecated compatibility is required. If not, replace `verbose` with `detail: CompactFullDetailLiteral = "compact"` and map `detail="full"` to the current verbose behavior. If compatibility is required, define a short-lived alias and tests for deprecation messaging.

## Scope

- Files:
  - `src/mtdata/core/data/__init__.py`
  - `tests/core/test_wait_event.py`
  - CLI/schema tests covering `wait_event`
- Symbols:
  - `wait_event`
  - `_compact_wait_event_public_result`
- Tests:
  - Public wait-event wrapper tests
  - Wait-event compact/full payload tests
  - CLI/schema argument generation tests
- Docs/config/CLI/API affected:
  - MCP schema for `wait_event`
  - generated CLI help for `wait_event`

## Risks

- Existing clients may still pass `verbose=true`.
- Adding both controls can create precedence ambiguity.
- Removing `verbose` without a migration can break scripts that rely on full wait-event diagnostics.

## Verification plan

- Add tests proving `detail="compact"` and `detail="full"` map to current compact/verbose behavior.
- Run `python -m pytest tests/core/test_wait_event.py tests/services/test_wait_event_use_cases.py`.
- Run relevant CLI/schema tests for generated `wait_event` parameters.
- Inspect generated tool schema to confirm only the intended output-control parameter is exposed.

## Expected impact

- Estimated LOC reduction: small after deprecated compatibility is removed.
- Complexity reduction: low to medium by standardizing output-control semantics.
- Maintenance benefit: fewer special-case verbosity parameters across tools.
