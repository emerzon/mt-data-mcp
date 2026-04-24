# labels_triple_barrier summary_only compatibility

## Source report

- Original report: `code-reports/labels-summary-only-deprecated-duplicate.md`

## Status

Deferred for larger refactor.

## Validated issue

`labels_triple_barrier` exposes two ways to request summary-style output: `detail="summary"` or legacy `detail="summary_only"`, plus a deprecated `summary_only` boolean flag that forces summary mode. The CLI layer already hides the duplicate `summary_only` flag, but the MCP function signature still includes it.

## Evidence

- `src/mtdata/core/labels.py` defines `detail: Literal["full", "summary", "compact", "summary_only"]` and `summary_only: bool = False`.
- The implementation normalizes `detail="summary_only"` to `summary` and appends a deprecation warning.
- The implementation treats truthy `summary_only` as summary mode and appends a deprecation warning.
- `src/mtdata/core/cli_discovery.py` contains explicit handling to hide the `labels_triple_barrier` `summary_only` shadow flag from CLI output.
- `tests/core/test_labels_coverage.py` verifies both legacy `summary_only=True` and `detail="summary_only"` still work with warnings.
- `tests/core/test_cli_coverage.py` verifies the CLI hides the `summary_only` flag while still accepting the legacy detail alias.

## Why this should not be fixed inline

Removing either legacy path changes supported behavior covered by tests and visible through the MCP tool signature. The compatibility policy for deprecated detail aliases and shadow flags needs to be decided across tools before deleting the labels-specific behavior.

## Recommended approach

Audit all `summary_only` aliases and deprecated verbosity paths across the project, then choose a single migration policy. If removal is approved, delete the `summary_only` flag from `labels_triple_barrier`, remove `summary_only` from the labels detail literal, update CLI hiding logic, and update tests to assert the new canonical-only behavior.

## Scope

- Files:
  - `src/mtdata/core/labels.py`
  - `src/mtdata/core/cli_discovery.py`
  - `src/mtdata/shared/schema.py`
  - `tests/core/test_labels_coverage.py`
  - `tests/core/test_cli_coverage.py`
- Symbols:
  - `labels_triple_barrier`
  - `normalize_cli_literal_choices`
  - CLI hidden-parameter handling for `summary_only`
  - `CANONICAL_OUTPUT_DETAIL_ALIASES`
- Tests:
  - Labels output detail tests
  - CLI argument generation tests
  - Shared output detail alias tests
- Docs/config/CLI/API affected:
  - MCP schema for `labels_triple_barrier`
  - generated CLI help for labels

## Risks

- Existing MCP clients may still pass `summary_only=True`.
- Existing users may still pass `detail="summary_only"`.
- Removing only the labels-specific flag while leaving shared aliases elsewhere could make detail semantics less consistent.

## Verification plan

- Update tests to reflect the selected deprecation policy.
- Run `python -m pytest tests/core/test_labels_coverage.py tests/core/test_cli_coverage.py tests/core/test_output_contract.py`.
- Inspect generated CLI help for `labels_triple_barrier` and MCP schema parameters.

## Expected impact

- Estimated LOC reduction: small.
- Complexity reduction: low to medium by removing one duplicate public control path.
- Maintenance benefit: clearer verbosity contract for label generation and fewer deprecated choices in schema output.
