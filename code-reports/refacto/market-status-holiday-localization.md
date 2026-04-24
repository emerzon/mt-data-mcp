# market_status holiday localization

## Source report

- Original report: `code-reports/11-market-status-holiday-language.md`

## Status

Deferred for larger refactor.

## Validated issue

`market_status` returns holiday names from the upstream holiday calendars as-is. For non-English markets, those names may be localized or in non-Latin scripts, which makes compact holiday summaries hard for English-speaking users to scan.

## Evidence

- `src/mtdata/core/market_status.py::_get_holidays` calls `holidays.country_holidays(country, years=[...])`.
- `src/mtdata/core/market_status.py::_is_holiday` returns `str(h[date_key])` directly.
- `_check_market_status` includes the returned holiday string as `holiday` and in the human-readable message.
- `_get_upcoming_holidays` includes the returned holiday string in `upcoming_holidays`.
- `_summarize_upcoming_holiday` preserves the `holiday` field in compact summaries without translation.
- Tests currently assert passthrough English fake holiday names, not locale normalization.

## Why this should not be fixed inline

A reliable fix requires a localization policy, not a one-off mapping. Holiday names vary by country, year, substitute-day rules, and upstream package behavior. A partial hand-maintained translation table would be incomplete and could silently mislabel holidays.

## Recommended approach

Decide whether `market_status` should expose native names, English names, or both. If both, add a `holiday_en` or `holiday_display` field and a tested mapping/provider layer. Prefer using upstream locale support if available; otherwise maintain a small explicit mapping only for supported exchanges and document fallback behavior.

## Scope

- Files:
  - `src/mtdata/core/market_status.py`
  - market-status tests
  - docs/CLI examples for `market_status`
- Symbols:
  - `_get_holidays`
  - `_is_holiday`
  - `_get_upcoming_holidays`
  - `_summarize_upcoming_holiday`
  - `market_status`
- Tests:
  - Native-name passthrough or translation fallback tests
  - Compact/full holiday payload tests
  - Locale/display parameter tests if a parameter is introduced
- Docs/config/CLI/API affected:
  - `market_status` response schema
  - generated CLI output
  - holiday field documentation

## Risks

- Incorrect translations could mislead users about market closures.
- Upstream holiday names may change across package versions.
- Adding locale parameters changes the public tool schema.
- Replacing native names could remove useful local context for non-English users.

## Verification plan

- Add fixtures for representative JP, CN, HK, DE, FR, UK, US, and AU holidays.
- Verify compact and full holiday outputs include the chosen display fields.
- Run all market-status tests.
- Manually inspect representative outputs around major international holidays.

## Expected impact

- Estimated LOC reduction: none; this is a UX consistency refactor.
- Complexity reduction: medium for users scanning holiday output.
- Maintenance benefit: clearer, documented holiday-name semantics.
