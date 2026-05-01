# tests/ — Test Suite

235 test files, hybrid pytest/unittest.TestCase. No CI — manual runs only. No pytest.ini — all config in `conftest.py`.

## NAMING CONVENTIONS

Files mirror `src/mtdata/` module structure with suffix indicating scope:

| Suffix | Scope | Max Lines | Example |
|--------|-------|-----------|---------|
| `_pure` | Unit tests for pure functions, no mocks | 500 | `test_indicators_pure.py` |
| `_business_logic` | Logic tests, may mock external deps | 800 | `test_core_forecast_business_logic.py` |
| `_coverage` | Comprehensive coverage of a module | 1,000 | `test_cli_coverage.py` |
| `_extended` | Integration/extended scenarios | 600 | `test_elliott_extended.py` |
| _(no suffix)_ | General tests | 500 | `test_data_service.py` |

### Size Guidelines

- Files larger than their suffix limit should be split into subdirectories:
  - `tests/core/cli/` - CLI tests (formerly `test_cli_coverage.py`)
  - `tests/patterns/coverage/` - Pattern coverage tests (formerly `test_patterns_core_coverage.py`)
  - `tests/trading/coverage/` - Trading coverage tests (formerly `test_trading_coverage.py`)
  - `tests/services/wait_event/` - Wait event tests (formerly `test_wait_event_use_cases.py`)

### Test Domains

| Prefix | Covers | File Count |
|--------|--------|------------|
| `test_trading_*` | Trading logic | 20+ |
| `test_forecast_*` | Forecast engine | 60+ |
| `test_report_*` | Report generation | 10+ |
| `test_data_*` | Data service | 15+ |
| `test_patterns_*` | Pattern detection | 20+ |
| `test_regime_*` | Regime detection | 15+ |
| `test_volatility_*` | Volatility | 10+ |
| `test_indicators_*` | Indicators | 15+ |
| `test_denoise_*` | Denoising | 10+ |
| `test_web_api_*` | Web API | 10+ |
| `test_cli_*` | CLI | 5+ |
| `test_server_*` | MCP server | 5+ |

## SUBDIRECTORIES

Large modules use subdirectories for organization:

| Directory | Contains | Original File |
|-----------|----------|---------------|
| `tests/core/cli/` | 5 focused CLI test files | `test_cli_coverage.py` (6,006 lines) |
| `tests/patterns/coverage/` | 7 focused pattern test files | `test_patterns_core_coverage.py` (2,461 lines) |
| `tests/patterns/` | 3 split pattern business logic files | `test_patterns_business_logic.py` (3,736 lines) |
| `tests/trading/coverage/` | 7 focused trading test files | `test_trading_coverage.py` (2,837 lines) |
| `tests/services/wait_event/` | 4 focused wait event files | `test_wait_event_use_cases.py` (2,744 lines) |

## CONFTEST FIXTURES

`conftest.py` (70 lines) handles critical test infrastructure:

- **`sys.modules` stubbing**: Injects `MagicMock` for `MetaTrader5` into `sys.modules` at import time so tests run without a real MT5 terminal.
- **`mt5_module` fixture**: Per-test MT5 stub with automatic cleanup.
- **`pytest_runtest_setup/teardown` hooks**: Clear scipy's `_issubclass_fast` LRU cache between tests to prevent pollution from fake `torch` module injection.
- **Path setup**: Adds `src/` and project root to `sys.path`.

## MOCKING PATTERNS

- **MT5 interactions**: Always use `unittest.mock.patch` or the `mt5_module` fixture. Never call real MT5 API in tests.
- **torch/GPU**: Some `_coverage` tests inject a fake `torch` module — conftest clears scipy cache to prevent cross-test leaks.
- **Style**: Both `def test_*()` (pytest-style) and `class TestX(unittest.TestCase)` coexist. New tests can use either.

## ANTI-PATTERNS

- **Never** run tests against a live MT5 account — always mock.
- **Never** delete failing tests to "pass" — fix the underlying issue.
- **Never** import `conftest.py` directly — pytest discovers it automatically.
- **Never** let coverage files grow beyond 1,000 lines — split into subdirectories when they approach the limit.

## CLEANUP HISTORY

2026-05-01: Phase 1-3 test reorganization
- Renamed 4 files to match `_business_logic` convention
- Deleted 3 duplicate files (cli, data_service, trading business_logic)
- Split 4 large files into 19 focused subfiles in 5 subdirectories
- Total: 235 test files (was 213), +22 files but much better organization
