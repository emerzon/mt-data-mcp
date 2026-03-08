# tests/ — Test Suite

122 test files, hybrid pytest/unittest.TestCase. No CI — manual runs only. No pytest.ini — all config in `conftest.py`.

## NAMING CONVENTIONS

Files mirror `src/mtdata/` module structure with suffix indicating scope:

| Suffix | Scope | Example |
|--------|-------|---------|
| `_pure` | Unit tests for pure functions, no mocks | `test_indicators_pure.py` |
| `_business_logic` | Logic tests, may mock external deps | `test_core_forecast_business_logic.py` |
| `_coverage` | Comprehensive coverage of a module | `test_cli_coverage.py` (2050 lines) |
| `_extended` | Integration/extended scenarios | `test_elliott_extended.py` |
| _(no suffix)_ | General tests | `test_data_service.py` |

### Test Domains

| Prefix | Covers | Largest File |
|--------|--------|-------------|
| `test_trading_*` | Trading logic | `test_trading_coverage.py` (1844) |
| `test_forecast_*` | Forecast engine | `test_forecast_barriers.py` (1107) |
| `test_report_*` | Report generation | `test_report_utils_pure.py` (1705) |
| `test_data_*` | Data service | `test_data_service_coverage.py` (1393) |
| `test_patterns_*` | Pattern detection | `test_patterns_core_coverage.py` (1005) |
| `test_regime_*` | Regime detection | `test_regime_core_coverage.py` (837) |
| `test_volatility_*` | Volatility | `test_volatility_coverage.py` (1035) |
| `test_indicators_*` | Indicators | `test_indicators_pure.py` (931) |
| `test_denoise_*` | Denoising | `test_denoise_pure.py` (888) |
| `test_web_api_*` | Web API | `test_web_api_coverage.py` (1097) |
| `test_cli_*` | CLI | `test_cli_coverage.py` (2050) |
| `test_server_*` | MCP server | `test_server_coverage.py` (665) |

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
