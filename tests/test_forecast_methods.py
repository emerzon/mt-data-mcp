#!/usr/bin/env python3
"""
Comprehensive test suite for MetaTrader5 forecasting functionality.

This test suite:
1. Tests the get_forecast_methods function
2. Tests each available forecast method with current data
3. Handles statsmodels warnings appropriately  
4. Provides detailed diagnostics and performance analysis
5. Prints concise, readable summaries to stdout

Usage:
    python tests/test_forecast_methods.py [symbol] [timeframe] [horizon]
    
Examples:
    python tests/test_forecast_methods.py EURUSD H1 12
    python tests/test_forecast_methods.py GBPUSD M30 6
    python tests/test_forecast_methods.py USDJPY D1 5

Arguments:
    symbol     - Trading symbol (default: EURUSD)
    timeframe  - MT5 timeframe (default: H1) 
    horizon    - Forecast periods (default: 12)
"""

import sys
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import server functions
from server import get_forecast_methods, get_forecast, get_rates, TIMEFRAME_MAP

# Setup logging with warning suppression for statsmodels
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress common statsmodels warnings that are informational rather than errors
warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")
warnings.filterwarnings("ignore", message=".*optimization failed to converge.*")
warnings.filterwarnings("ignore", message=".*Non-stationary starting.*")
warnings.filterwarnings("ignore", message=".*Non-invertible starting.*")

class ForecastTestSuite:
    """Comprehensive forecast testing with clean output and detailed analysis."""
    
    def __init__(self, symbol: str = "EURUSD", timeframe: str = "H1"):
        self.symbol = symbol
        self.timeframe = timeframe
        self.test_results = {}
        
    def test_forecast_methods_api(self) -> Dict[str, Any]:
        """Test the get_forecast_methods function with comprehensive output."""
        logger.info("Testing get_forecast_methods API...")
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                methods_data = get_forecast_methods()
            
            if "error" in methods_data:
                logger.error(f"Error getting forecast methods: {methods_data['error']}")
                return methods_data
                
            logger.info(f"Successfully retrieved {len(methods_data['methods'])} forecast methods")
            
            # Build available methods list without verbose stdout listing
            categories = {
                "Simple Baselines": {
                    "methods": ["naive", "drift", "seasonal_naive"],
                    "description": "Basic forecasting methods for benchmarking"
                },
                "Exponential Smoothing": {
                    "methods": ["ses", "holt", "holt_winters_add", "holt_winters_mul"],
                    "description": "Exponential smoothing family methods"
                },
                "Advanced Methods": {
                    "methods": ["theta", "fourier_ols"],
                    "description": "Hybrid and frequency-domain methods"
                },
                "ARIMA Family": {
                    "methods": ["arima", "sarima"],
                    "description": "Autoregressive integrated moving average models"
                }
            }
            available_methods = [m['method'] for m in methods_data['methods'] if m.get('available')]
            
            return {
                "success": True, 
                "methods": methods_data['methods'], 
                "available_methods": available_methods,
                "categories": categories
            }
            
        except Exception as e:
            error_msg = f"Exception testing forecast methods: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg}

    def get_training_period_info(self, lookback_bars: int) -> Dict[str, Any]:
        """Get information about the training data period."""
        try:
            # Get historical data to determine training period
            rates_result = get_rates(
                symbol=self.symbol,
                timeframe=self.timeframe,
                candles=max(lookback_bars, 50),  # Get at least the lookback amount
                ohlcv=['C']
            )
            
            if "error" in rates_result or not rates_result.get('csv_data'):
                return {"error": "Could not retrieve training period data"}
            
            # Parse the CSV data to get timestamps (rows are oldest -> newest)
            csv_lines = [ln for ln in rates_result['csv_data'].strip().split('\n') if ln.strip()]
            if len(csv_lines) >= 2:
                oldest_line = csv_lines[0]
                newest_line = csv_lines[-1]
                
                # Extract timestamps (first column before comma)
                training_start = oldest_line.split(',')[0] if ',' in oldest_line else oldest_line
                training_end = newest_line.split(',')[0] if ',' in newest_line else newest_line
                
                return {
                    "training_start": training_start,
                    "training_end": training_end,
                    "training_bars": len(csv_lines)
                }
            
            return {"error": "Insufficient training data"}
            
        except Exception as e:
            return {"error": f"Error getting training period: {str(e)}"}

    def get_actual_validation_values(self, horizon: int) -> Dict[str, Any]:
        """Deprecated in favor of backtest anchor selection."""
        try:
            return {"error": "Use _pick_backtest_anchor_and_validation instead"}
        
        except Exception as e:
            return {"error": f"Error getting validation values: {str(e)}"}

    def _pick_backtest_anchor_and_validation(self, horizon: int) -> Dict[str, Any]:
        """Select a backtest anchor (as_of) and return the next horizon actual values.

        Approach: fetch enough recent bars, choose anchor at (len - horizon - 1), and
        take the following `horizon` bars as actuals. Returns timestamps as strings.
        """
        try:
            bars_needed = max(horizon + 60, 100)
            rates_result = get_rates(
                symbol=self.symbol,
                timeframe=self.timeframe,
                candles=bars_needed,
                ohlcv=['C']
            )
            if "error" in rates_result or not rates_result.get('csv_data'):
                return {"error": "Could not retrieve data for anchor selection"}
            lines = [ln for ln in rates_result['csv_data'].strip().split('\n') if ln.strip()]
            if len(lines) < horizon + 2:
                return {"error": "Insufficient data to select backtest anchor"}
            anchor_idx = len(lines) - horizon - 1
            if anchor_idx < 0:
                anchor_idx = 0
            anchor_time = lines[anchor_idx].split(',')[0]
            validation = []
            timestamps = []
            for line in lines[anchor_idx+1:anchor_idx+1+horizon]:
                parts = line.split(',')
                if len(parts) >= 2:
                    timestamps.append(parts[0])
                    try:
                        validation.append(float(parts[1]))
                    except Exception:
                        validation.append(np.nan)
            if not validation or len(validation) < horizon:
                return {"error": "Failed to build full validation window"}
            return {
                "anchor_time": anchor_time,
                "actual_values": validation,
                "timestamps": timestamps,
            }
        except Exception as e:
            return {"error": f"Error picking backtest anchor: {e}"}

    def test_single_method(self, method: str, horizon: int = 12, **kwargs) -> Dict[str, Any]:
        """Test a single forecast method with comprehensive error handling."""
        logger.info(f"Testing {method} method...")
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                # Choose a backtest anchor and validation window in history
                vb = self._pick_backtest_anchor_and_validation(horizon)
                if 'error' in vb:
                    return {"method": method, "success": False, "error": vb['error']}

                result = get_forecast(
                    symbol=self.symbol,
                    timeframe=self.timeframe,
                    horizon=horizon,
                    method=method,
                    as_of=vb['anchor_time'],
                    **kwargs
                )
            
            if "error" in result:
                return {"method": method, "success": False, "error": result["error"]}
            
            # Extract and validate forecast data
            forecast_values = result.get("forecast_price", [])
            lower_values = result.get("lower_price", [])
            upper_values = result.get("upper_price", [])
            times = result.get("times", [])
            params_used = result.get("params_used", {})
            lookback_used = result.get("lookback_used", 0)
            seasonality_period = result.get("seasonality_period", None)
            
            # Calculate comprehensive forecast statistics
            if forecast_values:
                forecast_stats = {
                    "count": len(forecast_values),
                    "mean": np.mean(forecast_values),
                    "std": np.std(forecast_values),
                    "min": np.min(forecast_values),
                    "max": np.max(forecast_values),
                    "range": np.max(forecast_values) - np.min(forecast_values),
                    "trend": self._calculate_trend(forecast_values),
                    "volatility": np.std(np.diff(forecast_values)) if len(forecast_values) > 1 else 0,
                    "first_value": forecast_values[0] if forecast_values else None,
                    "last_value": forecast_values[-1] if forecast_values else None
                }
            else:
                forecast_stats = {}
            
            # Attach validation info to result for downstream analysis
            validation_values = vb.get("actual_values", [])
            validation_timestamps = vb.get("timestamps", [])

            # Check for confidence intervals
            has_intervals = bool(lower_values and upper_values)
            interval_width = np.mean(np.array(upper_values) - np.array(lower_values)) if has_intervals else 0
            
            return {
                "method": method,
                "success": True,
                "forecast_values": forecast_values,
                "lower_values": lower_values,
                "upper_values": upper_values,
                "times": times,
                "params_used": params_used,
                "stats": forecast_stats,
                "has_intervals": has_intervals,
                "avg_interval_width": interval_width,
                "lookback_used": lookback_used,
                "seasonality_period": seasonality_period,
                "anchor_time": vb.get("anchor_time"),
                "validation_values": validation_values,
                "validation_timestamps": validation_timestamps,
                "forecast_start": times[0] if times else None,
                "forecast_end": times[-1] if times else None,
                "full_result": result
            }
            
        except Exception as e:
            return {"method": method, "success": False, "error": f"Exception: {str(e)}"}

    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate the overall trend direction of forecast values."""
        if len(values) < 2:
            return "FLAT"
        
        # Simple linear regression slope
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        # Use relative threshold based on value magnitude
        threshold = abs(np.mean(values)) * 1e-6 if np.mean(values) != 0 else 1e-6
        
        if slope > threshold:
            return "UP"
        elif slope < -threshold:
            return "DOWN"
        else:
            return "FLAT"

    def run_comprehensive_test(self, horizon: int = 12) -> Dict[str, Any]:
        """Run comprehensive testing of all available forecast methods."""
        logger.info(f"Running comprehensive forecast test suite for {self.symbol} ({self.timeframe})...")
        
        # Test API first
        methods_result = self.test_forecast_methods_api()
        if "error" in methods_result:
            return methods_result
        
        available_methods = methods_result.get("available_methods", [])
        if not available_methods:
            return {"error": "No available forecast methods found"}
        
        # Test each method
        test_results = {}
        successful_tests = 0
        training_period_info = None
        validation_values_info = None
        
        print(f"\nTesting {len(available_methods)} forecast methods...")
        print("=" * 80)
        
        first_success_validation = None
        for i, method in enumerate(available_methods, 1):
            print(f"[{i:2d}/{len(available_methods)}] Testing {method:<20}", end=" ")
            
            result = self.test_single_method(method, horizon)
            test_results[method] = result
            
            # Get training period info from the first successful result
            if result.get("success") and training_period_info is None:
                lookback = result.get("lookback_used", 0)
                if lookback > 0:
                    training_period_info = self.get_training_period_info(lookback)
            
            # Capture validation values from the first successful result
            if result.get("success") and first_success_validation is None:
                first_success_validation = {
                    "actual_values": result.get("validation_values", []),
                    "timestamps": result.get("validation_timestamps", []),
                }
            
            if result.get("success"):
                successful_tests += 1
                stats = result.get("stats", {})
                trend = stats.get("trend", "N/A")
                mean_val = stats.get("mean", 0)
                ci = result.get("avg_interval_width", 0)
                has_ci = "Y" if result.get("has_intervals") else "N"
                print(f"[SUCCESS] Trend: {trend:<4}  Mean: {mean_val:>12.6f}  CI?: {has_ci}  Avg CI Width: {ci:>10.6f}")
            else:
                error = result.get("error", "Unknown error")[:40]
                print(f"[FAILED]  {error}")

        print("=" * 80)

        # Compact summary table
        print("\nSUMMARY")
        print("-" * 80)
        header = f"{'Method':<20} {'Status':<8} {'Mean':>12} {'Trend':<6} {'CI?':<4} {'Avg CI Width':>13}"
        print(header)
        print("-" * 80)
        for method in available_methods:
            r = test_results.get(method, {})
            status = "OK" if r.get("success") else "FAIL"
            stats = r.get("stats", {})
            mean_val = stats.get("mean", float('nan'))
            trend = stats.get("trend", "-")
            has_ci = "Y" if r.get("has_intervals") else "N"
            ciw = r.get("avg_interval_width", float('nan'))
            print(f"{method:<20} {status:<8} {mean_val:>12.6f} {trend:<6} {has_ci:<4} {ciw:>13.6f}")
        print("-" * 80)

        # Build validation info block
        if first_success_validation and first_success_validation.get("actual_values"):
            av = np.array(first_success_validation["actual_values"], dtype=float)
            validation_values_info = {
                "actual_values": first_success_validation["actual_values"],
                "timestamps": first_success_validation["timestamps"],
                "validation_stats": {
                    "mean": float(np.nanmean(av)),
                    "std": float(np.nanstd(av)),
                    "min": float(np.nanmin(av)),
                    "max": float(np.nanmax(av)),
                    "range": float(np.nanmax(av) - np.nanmin(av)),
                    "trend": self._calculate_trend(av.tolist())
                }
            }

        return {
            "success": True,
            "methods_info": methods_result,
            "test_results": test_results,
            "training_period_info": training_period_info,
            "validation_values_info": validation_values_info,
            "total_methods": len(available_methods),
            "successful_methods": successful_tests,
            "success_rate": successful_tests / len(available_methods) * 100 if available_methods else 0
        }

    def create_detailed_analysis(self, comprehensive_results: Dict[str, Any]) -> str:
        """Create detailed analysis of test results."""
        test_results = comprehensive_results.get("test_results", {})
        successful_results = {k: v for k, v in test_results.items() if v.get("success")}
        # Build validation info from first successful result
        validation_values_info = {}
        
        analysis = [
            "\n" + "="*80,
            "DETAILED FORECAST METHOD ANALYSIS",
            "="*80
        ]
        
        if not successful_results:
            analysis.append("No successful forecasts to analyze.")
            return "\n".join(analysis)
        
        # Overall statistics and data period information
        total_forecasts = sum(len(r.get("forecast_values", [])) for r in successful_results.values())
        all_values = []
        lookbacks = []
        forecast_periods = []
        
        for result in successful_results.values():
            all_values.extend(result.get("forecast_values", []))
            lookbacks.append(result.get("lookback_used", 0))
            if result.get("forecast_start") and result.get("forecast_end"):
                forecast_periods.append(f"{result.get('forecast_start')} to {result.get('forecast_end')}")
        
        if all_values:
            # Get representative forecast period (they should all be the same)
            forecast_period = forecast_periods[0] if forecast_periods else "N/A"
            avg_lookback = np.mean(lookbacks) if lookbacks else 0
            
            # Get training period information
            training_info = comprehensive_results.get("training_period_info", {})
            training_start = training_info.get("training_start", "N/A")
            training_end = training_info.get("training_end", "N/A")
            actual_training_bars = training_info.get("training_bars", 0)
            
            analysis.extend([
                f"Total Successful Methods: {len(successful_results)}",
                f"Total Forecast Points: {total_forecasts}",
                f"",
                f"DATA PERIOD ANALYSIS:",
                f"Training Data Period: {training_start} to {training_end}",
                f"Training Data Bars: {actual_training_bars} bars (actual historical data)",
                f"Average Lookback Used: {avg_lookback:.0f} bars (used by forecast methods)",
                f"Forecast Period: {forecast_period}",
                f"",
                f"FORECAST VALUE ANALYSIS:",
                f"Overall Price Range: {min(all_values):.6f} - {max(all_values):.6f}",
                f"Overall Mean Price: {np.mean(all_values):.6f}",
                f"Overall Price Std Dev: {np.std(all_values):.6f}",
                ""
            ])
        
        # Method comparison table with actual values
        actual_values = validation_values_info.get("actual_values", [])
        actual_stats = validation_values_info.get("validation_stats", {})
        
        if actual_values:
            actual_mean = actual_stats.get("mean", 0)
            actual_trend = actual_stats.get("trend", "N/A")
            analysis.extend([
                f"{'Method':<18} {'Pred Mean':<10} {'Act Mean':<10} {'MAE':<8} {'Std Dev':<10} {'Range':<10} {'Trend':<6} {'Act Trend':<9} {'Confidence':<10}",
                "-" * 120
            ])
        else:
            analysis.extend([
                f"{'Method':<18} {'Mean':<10} {'Std Dev':<10} {'Range':<10} {'Trend':<6} {'Volatility':<12} {'Confidence':<10}",
                "-" * 100
            ])
        
        # Sort by mean forecast value
        sorted_methods = sorted(successful_results.items(), 
                               key=lambda x: x[1].get("stats", {}).get("mean", 0))
        
        for method, result in sorted_methods:
            stats = result.get("stats", {})
            mean_val = stats.get("mean", 0)
            std_val = stats.get("std", 0)
            range_val = stats.get("range", 0)
            trend = stats.get("trend", "N/A")
            volatility = stats.get("volatility", 0)
            has_intervals = "YES" if result.get("has_intervals") else "NO"
            
            if actual_values:
                # Calculate MAE between forecast and actual values
                forecast_values = result.get("forecast_values", [])
                mae = 0
                if len(forecast_values) == len(actual_values):
                    mae = np.mean(np.abs(np.array(forecast_values) - np.array(actual_values)))
                elif len(forecast_values) > 0 and len(actual_values) > 0:
                    # Handle different lengths by using the minimum length
                    min_len = min(len(forecast_values), len(actual_values))
                    mae = np.mean(np.abs(np.array(forecast_values[:min_len]) - np.array(actual_values[:min_len])))
                
                analysis.append(
                    f"{method:<18} {mean_val:<10.6f} {actual_mean:<10.6f} {mae:<8.6f} {std_val:<10.6f} {range_val:<10.6f} {trend:<6} {actual_trend:<9} {has_intervals:<10}"
                )
            else:
                analysis.append(
                    f"{method:<18} {mean_val:<10.6f} {std_val:<10.6f} {range_val:<10.6f} {trend:<6} {volatility:<12.6f} {has_intervals:<10}"
                )
        
        # Category analysis
        categories = {
            "Simple Baselines": ["naive", "drift", "seasonal_naive"],
            "Exponential Smoothing": ["ses", "holt", "holt_winters_add", "holt_winters_mul"],
            "Advanced Methods": ["theta", "fourier_ols"],
            "ARIMA Family": ["arima", "sarima"]
        }
        
        analysis.extend([
            "",
            "PERFORMANCE BY METHOD CATEGORY:",
            "-" * 60
        ])
        
        for category, methods in categories.items():
            category_results = {k: v for k, v in successful_results.items() if k in methods}
            if category_results:
                means = [r["stats"]["mean"] for r in category_results.values()]
                volatilities = [r["stats"]["volatility"] for r in category_results.values()]
                success_count = len(category_results)
                total_in_category = len(methods)
                success_rate = success_count / total_in_category * 100
                
                analysis.append(
                    f"{category:<22}: {success_count}/{total_in_category} methods ({success_rate:.1f}%), "
                    f"Avg Mean: {np.mean(means):.6f}, Avg Volatility: {np.mean(volatilities):.6f}"
                )
        
        analysis.append("="*80)
        return "\n".join(analysis)

    # Removed file saving; we print readable summaries to stdout only

def main():
    """Main function to run the comprehensive forecast test suite."""
    # Parse command line arguments
    symbol = sys.argv[1] if len(sys.argv) > 1 else "EURUSD"
    timeframe = sys.argv[2] if len(sys.argv) > 2 else "H1"
    horizon = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    
    # Validate timeframe
    if timeframe not in TIMEFRAME_MAP:
        logger.error(f"Invalid timeframe: {timeframe}. Valid options: {list(TIMEFRAME_MAP.keys())}")
        sys.exit(1)
    
    print(f"\nMetaTrader5 Forecast Methods Test Suite")
    print(f"Symbol: {symbol} | Timeframe: {timeframe} | Horizon: {horizon} periods")
    print("="*80)
    
    # Initialize test suite
    test_suite = ForecastTestSuite(symbol=symbol, timeframe=timeframe)
    
    # Run comprehensive test
    comprehensive_results = test_suite.run_comprehensive_test(horizon=horizon)
    
    if "error" in comprehensive_results:
        logger.error(f"Test suite failed: {comprehensive_results['error']}")
        sys.exit(1)
    
    # Display detailed analysis
    detailed_analysis = test_suite.create_detailed_analysis(comprehensive_results)
    print(detailed_analysis)
    
    # Final summary with enhanced period information
    total = comprehensive_results.get("total_methods", 0)
    successful = comprehensive_results.get("successful_methods", 0)
    success_rate = comprehensive_results.get("success_rate", 0)
    
    # Get period information from successful results
    test_results = comprehensive_results.get("test_results", {})
    successful_results = {k: v for k, v in test_results.items() if v.get("success")}
    
    avg_lookback = 0
    forecast_period = "N/A"
    training_period = "N/A"
    if successful_results:
        lookbacks = [r.get("lookback_used", 0) for r in successful_results.values()]
        avg_lookback = np.mean(lookbacks) if lookbacks else 0
        
        # Get forecast period from any successful result
        first_result = next(iter(successful_results.values()))
        if first_result.get("forecast_start") and first_result.get("forecast_end"):
            forecast_period = f"{first_result.get('forecast_start')} to {first_result.get('forecast_end')}"
    
    # Get training period information
    training_info = comprehensive_results.get("training_period_info", {})
    if training_info and not training_info.get("error"):
        training_start = training_info.get("training_start", "N/A")
        training_end = training_info.get("training_end", "N/A")
        actual_training_bars = training_info.get("training_bars", 0)
        training_period = f"{training_start} to {training_end} ({actual_training_bars} bars)"
    
    print(f"\n" + "="*70)
    print("TEST SUITE SUMMARY")
    print("="*70)
    print(f"Symbol: {symbol}")
    print(f"Timeframe: {timeframe}")
    print(f"")
    print(f"Training Data Period: {training_period}")
    print(f"Average Lookback Used: {avg_lookback:.0f} bars (by forecast methods)")
    print(f"Forecast Horizon: {horizon} periods")
    print(f"Forecast Period: {forecast_period}")
    print(f"")
    print(f"Methods Tested: {total}")
    print(f"Successful: {successful}")
    print(f"Failed: {total - successful}")
    print(f"Success Rate: {success_rate:.1f}%")
    
    # Performance assessment
    if success_rate >= 90:
        print(f"\n[EXCELLENT] Forecast system is working exceptionally well!")
    elif success_rate >= 75:
        print(f"\n[VERY GOOD] Most forecast methods are working correctly.")
    elif success_rate >= 50:
        print(f"\n[GOOD] Majority of forecast methods are functional.")
    elif success_rate >= 25:
        print(f"\n[MODERATE] Some forecast methods need attention.")
    else:
        print(f"\n[POOR] Significant issues detected with forecast methods.")
    
    print("="*70)

if __name__ == "__main__":
    main()
