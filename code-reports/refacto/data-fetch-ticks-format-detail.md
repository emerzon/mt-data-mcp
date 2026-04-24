# data_fetch_ticks format/detail split

## Source report

- Original report: `code-reports/data_fetch_ticks_format_vs_detail.md`
- Original report: `code-reports/data-fetch-ticks-format-vs-detail.md`
- Original report: `code-reports/02-data_fetch_ticks-format-vs-detail.md`

## Status

Deferred for larger refactor.

## Validated issue

`data_fetch_ticks` uses a `format` request field to choose between summary, stats, and raw row output, while `data_fetch_candles` and many other tools use `detail` for verbosity. The tick tool also accepts `compact` and `full` as aliases for `summary` and `stats`, which makes the shared detail vocabulary mean a tick-output shape rather than normal metadata/detail verbosity.

## Evidence

- `src/mtdata/core/data/requests.py::DataFetchCandlesRequest` defines `detail: CompactFullDetailLiteral = "compact"`.
- `src/mtdata/core/data/requests.py::DataFetchTicksRequest` defines `format: Literal["summary", "stats", "rows", "compact", "full"] = "summary"`.
- `DataFetchTicksRequest` rejects the removed `output` field with a message that directs callers to `format`.
- `DataFetchTicksRequest._normalize_format` normalizes shared aliases such as `compact` and `full` through `normalize_output_detail`.
- `src/mtdata/core/data/__init__.py::data_fetch_ticks` documents and logs `format=request.format`.
- `tests/services/test_data_use_cases.py` asserts the old `output` field is rejected and that `compact`/`full` normalize to `summary`/`stats`.
- `docs/CLI.md` and `docs/SIMPLIFICATION.md` show `data_fetch_ticks --format rows`.

## Why this should not be fixed inline

The parameter has already been renamed from `output` to `format`, and the current request model, docs, CLI examples, and tests explicitly support `format`. Replacing it with `detail` or splitting it into `detail` plus an output-shape parameter changes a public data-fetching tool contract and needs a planned migration.

## Recommended approach

Separate verbosity from output shape. Keep or introduce a dedicated shape parameter such as `mode` or `output_mode` for `summary`, `stats`, and `rows`, and use `detail` only for compact/full verbosity semantics. Decide whether `format` remains a temporary alias and define deprecation warnings/tests before removing it from the public schema.

## Scope

- Files:
  - `src/mtdata/core/data/requests.py`
  - `src/mtdata/core/data/__init__.py`
  - `src/mtdata/core/data/use_cases.py`
  - `tests/services/test_data_use_cases.py`
  - CLI/schema tests for `data_fetch_ticks`
  - `docs/CLI.md`
  - `docs/SIMPLIFICATION.md`
- Symbols:
  - `DataFetchTicksRequest`
  - `data_fetch_ticks`
  - `run_data_fetch_ticks`
  - `_run_data_fetch_ticks_impl`
- Tests:
  - Data fetch request validation tests
  - Tick summary/stats/rows behavior tests
  - CLI help and argument parsing tests
- Docs/config/CLI/API affected:
  - MCP schema for `data_fetch_ticks`
  - generated CLI flag names
  - documentation examples using `--format`

## Risks

- Existing clients may rely on `format="rows"` for raw ticks.
- Existing clients may rely on `format="compact"` and `format="full"` aliases.
- Renaming a parameter after the previous `output -> format` migration could create another compatibility churn cycle.
- Splitting detail and output shape may require non-trivial response-contract tests.

## Verification plan

- Add request-model tests for the selected canonical parameter names and any supported aliases.
- Run `python -m pytest tests/services/test_data_use_cases.py`.
- Run relevant CLI coverage tests for `data_fetch_ticks`.
- Validate docs examples and generated help/schema output.

## Expected impact

- Estimated LOC reduction: small after legacy aliases are removed.
- Complexity reduction: medium by separating output shape from verbosity.
- Maintenance benefit: more predictable data-fetching tool parameters and fewer overloaded `format` meanings.
