# MetaTrader5 Forecast Methods Test Suite

This directory contains comprehensive test tools for the MetaTrader5 forecasting functionality.

## Files

### `test_forecast_methods.py`
The main comprehensive test suite for all forecasting methods.

**Features:**
- Tests all 11 available forecast methods
- Clean output with no statsmodels warnings
- Comprehensive performance analysis  
- Detailed JSON reports with timestamps
- Category-based method grouping
- Statistical analysis and trend detection

**Usage:**
```bash
# Basic usage (defaults: EURUSD, H1, 12 periods)
python tests/test_forecast_methods.py

# With specific parameters  
python tests/test_forecast_methods.py GBPUSD M30 8
python tests/test_forecast_methods.py USDJPY D1 5

# Arguments:
# 1. symbol     - Trading symbol (default: EURUSD)
# 2. timeframe  - MT5 timeframe (default: H1) 
# 3. horizon    - Forecast periods (default: 12)
```

**Output:**
- Console display of all test results
- JSON file saved to `test_results/` directory
- Performance categorization and analysis
- Success rate assessment

## Test Results Directory

Test results are automatically saved to the `test_results/` directory with timestamped filenames:
```
test_results/forecast_test_EURUSD_H1_20250901_130558.json
```

## Method Categories

The test suite organizes methods into four categories:

### Simple Baselines
- `naive` - Repeat last observed value
- `drift` - Linear drift extrapolation  
- `seasonal_naive` - Repeat seasonal patterns

### Exponential Smoothing
- `ses` - Simple Exponential Smoothing
- `holt` - Linear trend method
- `holt_winters_add` - Additive seasonality
- `holt_winters_mul` - Multiplicative seasonality

### Advanced Methods  
- `theta` - Hybrid trend/level method
- `fourier_ols` - Fourier series regression

### ARIMA Family
- `arima` - Non-seasonal ARIMA
- `sarima` - Seasonal ARIMA

## Example Output

```
MetaTrader5 Forecast Methods Test Suite
Symbol: EURUSD | Timeframe: H1 | Horizon: 6 periods
================================================================================

Testing 11 forecast methods...
================================================================================
[ 1/11] Testing naive                [SUCCESS] Trend: FLAT, Mean: 1.170440
[ 2/11] Testing drift                [SUCCESS] Trend: UP, Mean: 1.170615
[ 3/11] Testing seasonal_naive       [SUCCESS] Trend: DOWN, Mean: 1.169165
...

====================================================================================================
DETAILED FORECAST METHOD ANALYSIS
====================================================================================================
Total Successful Methods: 10
Total Forecast Points: 60

DATA PERIOD ANALYSIS:
Average Lookback Period: 136 bars (training data)
Forecast Period: 2025-09-01T21 to 2025-09-02T02

FORECAST VALUE ANALYSIS:
Overall Price Range: 1.165970 - 1.170760
Overall Mean Price: 1.169494
Overall Price Std Dev: 0.001366
...

======================================================================
TEST SUITE SUMMARY
======================================================================
Symbol: EURUSD
Timeframe: H1
Training Data: ~136 bars (historical data used)
Forecast Horizon: 6 periods  
Forecast Period: 2025-09-01T21 to 2025-09-02T02

Methods Tested: 11
Successful: 10
Failed: 1
Success Rate: 90.9%

[EXCELLENT] Forecast system is working exceptionally well!
```

## Performance Thresholds

- **90%+**: Excellent - System working exceptionally well
- **75-89%**: Very Good - Most methods working correctly  
- **50-74%**: Good - Majority of methods functional
- **25-49%**: Moderate - Some methods need attention
- **<25%**: Poor - Significant issues detected

## Notes

- All statsmodels warnings are properly suppressed for clean output
- Tests work with any valid MT5 symbol and timeframe
- Confidence intervals are tested and reported
- Results include trend analysis and volatility measures
- JSON output preserves all test data for further analysis