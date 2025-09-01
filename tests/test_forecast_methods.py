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
    python tests/test_forecast_methods.py [symbol] [timeframe] [horizon] [OPTIONS]
    
Examples:
    # Standard forecast method comparison
    python tests/test_forecast_methods.py EURUSD H1 12
    python tests/test_forecast_methods.py GBPUSD M30 6 --backtest
    
    # Single denoise method testing
    python tests/test_forecast_methods.py EURUSD H1 12 --denoise ema --denoise-span 5
    python tests/test_forecast_methods.py GBPUSD M30 6 --backtest --denoise sma --denoise-span 20
    
    # Compare ALL available denoise methods (NEW!)
    python tests/test_forecast_methods.py EURUSD H1 12 --denoise-compare
    python tests/test_forecast_methods.py GBPUSD M30 6 --denoise-compare --backtest
    python tests/test_forecast_methods.py USDJPY D1 5 --denoise-compare --denoise-method arima

Arguments:
    symbol         - Trading symbol (default: EURUSD)
    timeframe      - MT5 timeframe (default: H1) 
    horizon        - Forecast periods (default: 12)
    
Options:
    --backtest       - Enable multiple backtest mode for robust testing
    --backtests N    - Number of backtest points to test (default: 3, requires --backtest)
    --denoise METHOD - Apply specific denoising filter (e.g., ema, sma, wavelet)
    --denoise-span N - Denoising filter span/window size (default: 10)
    --denoise-compare - Test ALL available denoise methods and rank them (NEW!)
    --denoise-method - Target forecast method for denoise comparison (default: theta)
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
from server import get_forecast_methods, get_forecast, get_rates, get_denoise_methods, TIMEFRAME_MAP

# Setup logging with warning suppression for statsmodels
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# Set server module to WARNING to reduce noise
logging.getLogger('server').setLevel(logging.WARNING)

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
        
    def get_available_denoise_methods(self) -> List[str]:
        """Get list of available denoise methods."""
        try:
            denoise_methods = get_denoise_methods()
            if "error" in denoise_methods:
                return ["none"]  # Fallback to no denoising
            
            available = []
            for method in denoise_methods.get("methods", []):
                if method.get("available", False):
                    method_name = method.get("method", "")
                    if method_name != "none":  # Skip 'none' as it means no denoising
                        available.append(method_name)
            
            # Always include no denoising as baseline
            return ["none"] + available
        except Exception:
            return ["none"]  # Fallback to no denoising
        
    def test_forecast_methods_api(self) -> Dict[str, Any]:
        """Test the get_forecast_methods function with comprehensive output."""
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                methods_data = get_forecast_methods()
            
            if "error" in methods_data:
                logger.error(f"Error getting forecast methods: {methods_data['error']}")
                return methods_data
            
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

    def _pick_backtest_anchor_and_validation(self, horizon: int, min_lookback_periods: int = 200) -> Dict[str, Any]:
        """Select a backtest anchor (as_of) and return the next horizon actual values.

        Approach: fetch sufficient historical data, choose anchor far enough back to ensure
        proper historical backtesting, and take the following `horizon` bars as actuals.
        
        Args:
            horizon: Number of periods to forecast
            min_lookback_periods: Minimum periods to go back from present for anchor selection
        """
        try:
            # Fetch much more historical data for proper backtesting
            bars_needed = max(horizon + min_lookback_periods + 100, 500)
            rates_result = get_rates(
                symbol=self.symbol,
                timeframe=self.timeframe,
                candles=bars_needed,
                ohlcv=['C']
            )
            if "error" in rates_result or not rates_result.get('csv_data'):
                return {"error": "Could not retrieve data for anchor selection"}
            
            lines = [ln for ln in rates_result['csv_data'].strip().split('\n') if ln.strip()]
            if len(lines) < horizon + min_lookback_periods + 50:
                return {"error": f"Insufficient data for proper backtesting. Need at least {horizon + min_lookback_periods + 50} bars, got {len(lines)}"}
            
            # Select anchor point well back in history for true backtesting
            anchor_idx = len(lines) - horizon - min_lookback_periods
            if anchor_idx < 0:
                anchor_idx = max(0, len(lines) - horizon - 50)  # Fallback with warning
                
            anchor_time = lines[anchor_idx].split(',')[0]
            validation = []
            timestamps = []
            
            # Extract validation data from the period immediately following the anchor
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
                
            # Extract training period info (from start to anchor)
            training_start = lines[0].split(',')[0] if lines else "N/A"
            training_end = anchor_time
            training_bars = anchor_idx + 1
            
            # Extract validation period info
            validation_start = timestamps[0] if timestamps else "N/A"  
            validation_end = timestamps[-1] if timestamps else "N/A"
            
            return {
                "anchor_time": anchor_time,
                "actual_values": validation,
                "timestamps": timestamps,
                "anchor_idx": anchor_idx,
                "total_bars": len(lines),
                "lookback_periods": len(lines) - anchor_idx - horizon,
                "training_start": training_start,
                "training_end": training_end,
                "training_bars": training_bars,
                "validation_start": validation_start,
                "validation_end": validation_end,
                "validation_bars": len(validation)
            }
        except Exception as e:
            return {"error": f"Error picking backtest anchor: {e}"}

    def test_multiple_backtests(self, method: str, horizon: int = 12, num_tests: int = 3, denoise_spec: Optional[Dict] = None, **kwargs) -> Dict[str, Any]:
        """Test a single method at multiple historical points for robust backtesting."""
        
        all_results = []
        successful_tests = 0
        
        # Space out the backtest points across available history
        min_lookbacks = [200, 300, 400][:num_tests]  # Different historical depths
        
        for i, min_lookback in enumerate(min_lookbacks):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    
                    # Choose a backtest anchor at different historical points
                    vb = self._pick_backtest_anchor_and_validation(horizon, min_lookback)
                    if 'error' in vb:
                        all_results.append({"test": i+1, "success": False, "error": vb['error']})
                        continue

                    forecast_kwargs = kwargs.copy()
                    if denoise_spec:
                        forecast_kwargs['denoise'] = denoise_spec
                        
                    result = get_forecast(
                        symbol=self.symbol,
                        timeframe=self.timeframe,
                        horizon=horizon,
                        method=method,
                        as_of=vb['anchor_time'],
                        **forecast_kwargs
                    )
                    
                    if "error" in result:
                        all_results.append({"test": i+1, "success": False, "error": result["error"]})
                        continue
                    
                    # Calculate backtest metrics
                    forecast_values = result.get("forecast_price", [])
                    actual_values = vb.get("actual_values", [])
                    
                    if forecast_values and actual_values and len(forecast_values) == len(actual_values):
                        mae = np.mean(np.abs(np.array(forecast_values) - np.array(actual_values)))
                        mse = np.mean((np.array(forecast_values) - np.array(actual_values))**2)
                        rmse = np.sqrt(mse)
                        
                        # Calculate directional accuracy
                        if len(actual_values) > 1:
                            actual_direction = np.sign(np.diff(actual_values))
                            forecast_direction = np.sign(np.diff(forecast_values))
                            directional_accuracy = np.mean(actual_direction == forecast_direction)
                        else:
                            directional_accuracy = 0.5
                    else:
                        mae = mse = rmse = directional_accuracy = np.nan
                    
                    all_results.append({
                        "test": i+1,
                        "success": True,
                        "anchor_time": vb.get('anchor_time'),
                        "lookback_periods": vb.get('lookback_periods', 0),
                        "mae": mae,
                        "mse": mse,
                        "rmse": rmse,
                        "directional_accuracy": directional_accuracy,
                        "forecast_values": forecast_values,
                        "actual_values": actual_values,
                        "forecast_mean": np.mean(forecast_values) if forecast_values else np.nan,
                        "actual_mean": np.mean(actual_values) if actual_values else np.nan,
                        "training_start": vb.get("training_start"),
                        "training_end": vb.get("training_end"), 
                        "training_bars": vb.get("training_bars"),
                        "validation_start": vb.get("validation_start"),
                        "validation_end": vb.get("validation_end"),
                        "validation_bars": vb.get("validation_bars"),
                        "denoise_used": denoise_spec
                    })
                    successful_tests += 1
                    
            except Exception as e:
                all_results.append({"test": i+1, "success": False, "error": f"Exception: {str(e)}"})
        
        # Aggregate results
        if successful_tests > 0:
            successful_results = [r for r in all_results if r.get("success")]
            avg_mae = np.mean([r["mae"] for r in successful_results if not np.isnan(r["mae"])])
            avg_rmse = np.mean([r["rmse"] for r in successful_results if not np.isnan(r["rmse"])])
            avg_directional_accuracy = np.mean([r["directional_accuracy"] for r in successful_results if not np.isnan(r["directional_accuracy"])])
            
            return {
                "method": method,
                "success": True,
                "num_tests": num_tests,
                "successful_tests": successful_tests,
                "all_results": all_results,
                "avg_mae": avg_mae,
                "avg_rmse": avg_rmse,
                "avg_directional_accuracy": avg_directional_accuracy,
                "consistency": successful_tests / num_tests
            }
        else:
            return {
                "method": method,
                "success": False,
                "num_tests": num_tests,
                "successful_tests": 0,
                "all_results": all_results,
                "error": "No successful backtests"
            }

    def test_single_method(self, method: str, horizon: int = 12, min_lookback_periods: int = 200, denoise_spec: Optional[Dict] = None, **kwargs) -> Dict[str, Any]:
        """Test a single forecast method with comprehensive error handling."""
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                # Choose a backtest anchor and validation window in history
                vb = self._pick_backtest_anchor_and_validation(horizon, min_lookback_periods)
                if 'error' in vb:
                    return {"method": method, "success": False, "error": vb['error']}

                forecast_kwargs = kwargs.copy()
                if denoise_spec:
                    forecast_kwargs['denoise'] = denoise_spec
                    
                result = get_forecast(
                    symbol=self.symbol,
                    timeframe=self.timeframe,
                    horizon=horizon,
                    method=method,
                    as_of=vb['anchor_time'],
                    **forecast_kwargs
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
                "training_start": vb.get("training_start"),
                "training_end": vb.get("training_end"),
                "training_bars": vb.get("training_bars"),
                "validation_start": vb.get("validation_start"),
                "validation_end": vb.get("validation_end"),
                "validation_bars": vb.get("validation_bars"),
                "denoise_used": denoise_spec,
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

    def run_comprehensive_test(self, horizon: int = 12, use_multiple_backtests: bool = False, num_backtests: int = 3, denoise_spec: Optional[Dict] = None) -> Dict[str, Any]:
        """Run comprehensive testing of all available forecast methods.
        
        Args:
            horizon: Number of periods to forecast
            use_multiple_backtests: If True, run multiple backtests per method for robustness
            num_backtests: Number of backtest points to test per method
            denoise_spec: Optional denoise specification to apply to training data
        """
        test_type = f"multiple backtest ({num_backtests} points)" if use_multiple_backtests else "single backtest"
        
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
            
            if use_multiple_backtests:
                result = self.test_multiple_backtests(method, horizon, num_backtests, denoise_spec)
            else:
                result = self.test_single_method(method, horizon, denoise_spec=denoise_spec)
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
                if use_multiple_backtests:
                    # Display multiple backtest results with period info
                    mae = result.get("avg_mae", 0)
                    rmse = result.get("avg_rmse", 0)
                    dir_acc = result.get("avg_directional_accuracy", 0.5)
                    consistency = result.get("consistency", 0)
                    successful_bt = result.get("successful_tests", 0)
                    total_bt = result.get("num_tests", 1)
                    print(f"[SUCCESS] Backtests: {successful_bt}/{total_bt}  MAE: {mae:>8.6f}  RMSE: {rmse:>8.6f}  Dir.Acc: {dir_acc:>5.1%}  Consistency: {consistency:>5.1%}")
                    
                    # Show details for each backtest point
                    all_results = result.get("all_results", [])
                    for bt_result in all_results:
                        if bt_result.get("success"):
                            test_num = bt_result.get("test", "?")
                            tr_start = bt_result.get("training_start", "N/A")
                            tr_end = bt_result.get("training_end", "N/A")
                            tr_bars = bt_result.get("training_bars", 0)
                            val_start = bt_result.get("validation_start", "N/A")
                            val_end = bt_result.get("validation_end", "N/A")
                            val_bars = bt_result.get("validation_bars", 0)
                            mae_single = bt_result.get("mae", 0)
                            print(f"          Test {test_num}: Train[{tr_start} to {tr_end}, {tr_bars}bars] -> Val[{val_start} to {val_end}, {val_bars}bars] MAE:{mae_single:.6f}")
                else:
                    # Display single backtest results with period info
                    stats = result.get("stats", {})
                    trend = stats.get("trend", "N/A")
                    mean_val = stats.get("mean", 0)
                    ci = result.get("avg_interval_width", 0)
                    has_ci = "Y" if result.get("has_intervals") else "N"
                    tr_start = result.get("training_start", "N/A")
                    tr_end = result.get("training_end", "N/A")
                    tr_bars = result.get("training_bars", 0)
                    val_start = result.get("validation_start", "N/A")
                    val_end = result.get("validation_end", "N/A")
                    val_bars = result.get("validation_bars", 0)
                    print(f"[SUCCESS] Trend: {trend:<4}  Mean: {mean_val:>12.6f}  CI?: {has_ci}  Avg CI Width: {ci:>10.6f}")
                    print(f"          Train[{tr_start} to {tr_end}, {tr_bars}bars] -> Val[{val_start} to {val_end}, {val_bars}bars]")
            else:
                error = result.get("error", "Unknown error")[:40]
                print(f"[FAILED]  {error}")

        print("=" * 80)

        # Compact summary table
        print("\nSUMMARY")
        print("-" * 80)
        if use_multiple_backtests:
            header = f"{'Method':<20} {'Status':<8} {'Tests':<7} {'Avg MAE':>10} {'Avg RMSE':>10} {'Dir.Acc':>8} {'Consistency':>11}"
            print(header)
            print("-" * 80)
            for method in available_methods:
                r = test_results.get(method, {})
                status = "OK" if r.get("success") else "FAIL"
                tests = f"{r.get('successful_tests', 0)}/{r.get('num_tests', 1)}" if r.get("success") else "0/0"
                mae = r.get("avg_mae", float('nan'))
                rmse = r.get("avg_rmse", float('nan'))
                dir_acc = r.get("avg_directional_accuracy", float('nan'))
                consistency = r.get("consistency", float('nan'))
                print(f"{method:<20} {status:<8} {tests:<7} {mae:>10.6f} {rmse:>10.6f} {dir_acc:>8.1%} {consistency:>11.1%}")
        else:
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

    def run_denoise_comparison_test(self, horizon: int = 12, use_multiple_backtests: bool = False, num_backtests: int = 3, target_method: str = "theta") -> Dict[str, Any]:
        """Run comprehensive denoising comparison for a specific forecast method.
        
        Args:
            horizon: Number of periods to forecast
            use_multiple_backtests: If True, run multiple backtests per denoise method
            num_backtests: Number of backtest points to test per denoise method
            target_method: Which forecast method to use for denoise comparison
        """
        # Get available denoise methods
        denoise_methods = self.get_available_denoise_methods()
        
        print(f"\nTesting {len(denoise_methods)} denoise methods with {target_method} forecasting...")
        print("=" * 80)
        
        denoise_results = {}
        successful_tests = 0
        
        for i, denoise_method in enumerate(denoise_methods, 1):
            print(f"[{i:2d}/{len(denoise_methods)}] Testing {denoise_method:<15}", end=" ")
            
            # Create denoise spec (none means no denoising)
            if denoise_method == "none":
                denoise_spec = None
            else:
                denoise_spec = {
                    "method": denoise_method,
                    "when": "pre_ti",
                    "columns": ["close"],
                    "keep_original": False,
                    "params": {"span": 10}  # Default span for all methods
                }
                
                # Some methods need different parameters
                if denoise_method in ["wavelet"]:
                    denoise_spec["params"] = {"wavelet": "db4", "mode": "soft"}
                elif denoise_method in ["emd", "eemd", "ceemdan"]:
                    denoise_spec["params"] = {"n_imfs": 3}
                elif denoise_method in ["lowpass_fft"]:
                    denoise_spec["params"] = {"cutoff": 0.1}
            
            # Run the test with this denoise method
            if use_multiple_backtests:
                result = self.test_multiple_backtests(target_method, horizon, num_backtests, denoise_spec)
            else:
                result = self.test_single_method(target_method, horizon, denoise_spec=denoise_spec)
            
            denoise_results[denoise_method] = result
            
            if result.get("success"):
                successful_tests += 1
                if use_multiple_backtests:
                    mae = result.get("avg_mae", 0)
                    rmse = result.get("avg_rmse", 0)
                    dir_acc = result.get("avg_directional_accuracy", 0.5)
                    consistency = result.get("consistency", 0)
                    print(f"[SUCCESS] MAE: {mae:>8.6f}  RMSE: {rmse:>8.6f}  Dir.Acc: {dir_acc:>5.1%}  Consistency: {consistency:>5.1%}")
                else:
                    stats = result.get("stats", {})
                    mean_val = stats.get("mean", 0)
                    trend = stats.get("trend", "N/A")
                    ci = result.get("avg_interval_width", 0)
                    print(f"[SUCCESS] Mean: {mean_val:>12.6f}  Trend: {trend:<4}  Avg CI Width: {ci:>10.6f}")
            else:
                error = result.get("error", "Unknown error")[:40]
                print(f"[FAILED]  {error}")
        
        print("=" * 80)
        
        return {
            "success": True,
            "target_method": target_method,
            "denoise_results": denoise_results,
            "total_denoise_methods": len(denoise_methods),
            "successful_denoise_methods": successful_tests,
            "denoise_success_rate": successful_tests / len(denoise_methods) * 100 if denoise_methods else 0
        }

    def create_detailed_analysis(self, comprehensive_results: Dict[str, Any]) -> str:
        """Create detailed analysis of test results."""
        test_results = comprehensive_results.get("test_results", {})
        successful_results = {k: v for k, v in test_results.items() if v.get("success")}
        # Determine if we're in single or multiple backtest mode
        is_multiple_backtest = any("avg_mae" in r for r in successful_results.values())
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
        
        # Adjust headers based on mode
        if is_multiple_backtest:
            if actual_values:
                actual_mean = actual_stats.get("mean", 0)
                actual_trend = actual_stats.get("trend", "N/A")
                analysis.extend([
                    f"{'Method':<18} {'Avg MAE':<10} {'Act Mean':<10} {'Comp MAE':<8} {'Avg RMSE':<10} {'Dir.Acc':<10} {'Consist':<8} {'Act Trend':<9} {'Tests':<10}",
                    "-" * 120
                ])
            else:
                analysis.extend([
                    f"{'Method':<18} {'Avg MAE':<10} {'Avg RMSE':<10} {'Dir.Acc':<10} {'Consist':<8} {'Success':<12} {'Tests':<10}",
                    "-" * 100
                ])
        else:
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
        
        if is_multiple_backtest:
            # For multiple backtest mode, sort by average MAE (lower is better)
            sorted_methods = sorted(successful_results.items(), 
                                   key=lambda x: x[1].get("avg_mae", float('inf')))
        else:
            # For single backtest mode, sort by mean forecast value
            sorted_methods = sorted(successful_results.items(), 
                                   key=lambda x: x[1].get("stats", {}).get("mean", 0))
        
        for method, result in sorted_methods:
            if is_multiple_backtest:
                # Multiple backtest mode - use different metrics
                mean_val = result.get("avg_mae", 0)  # Show MAE instead of mean
                std_val = result.get("avg_rmse", 0)  # Show RMSE instead of std
                range_val = result.get("avg_directional_accuracy", 0)  # Show directional accuracy
                trend = f"{result.get('consistency', 0):.1%}"  # Show consistency as trend
                volatility = result.get("successful_tests", 0) / result.get("num_tests", 1)  # Success rate
                has_intervals = f"{result.get('num_tests', 0)}BT"  # Number of backtests
            else:
                # Single backtest mode - use original stats
                stats = result.get("stats", {})
                mean_val = stats.get("mean", 0)
                std_val = stats.get("std", 0)
                range_val = stats.get("range", 0)
                trend = stats.get("trend", "N/A")
                volatility = stats.get("volatility", 0)
                has_intervals = "YES" if result.get("has_intervals") else "NO"
            
            if is_multiple_backtest:
                if actual_values:
                    # For multiple backtest mode with actual values
                    analysis.append(
                        f"{method:<18} {mean_val:<10.6f} {actual_mean:<10.6f} {mean_val:<8.6f} {std_val:<10.6f} {range_val:<10.1%} {trend:<8} {actual_trend:<9} {has_intervals:<10}"
                    )
                else:
                    # For multiple backtest mode without actual values
                    analysis.append(
                        f"{method:<18} {mean_val:<10.6f} {std_val:<10.6f} {range_val:<10.1%} {trend:<8} {volatility:<12.1%} {has_intervals:<10}"
                    )
            else:
                if actual_values:
                    # Calculate MAE between forecast and actual values for single backtest
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
                    # Single backtest mode without actual values
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
                # Handle both single and multiple backtest results
                if any("stats" in r for r in category_results.values()):
                    # Single backtest mode
                    means = [r["stats"]["mean"] for r in category_results.values() if "stats" in r]
                    volatilities = [r["stats"]["volatility"] for r in category_results.values() if "stats" in r]
                    success_count = len(category_results)
                    total_in_category = len(methods)
                    success_rate = success_count / total_in_category * 100
                    
                    analysis.append(
                        f"{category:<22}: {success_count}/{total_in_category} methods ({success_rate:.1f}%), "
                        f"Avg Mean: {np.mean(means):.6f}, Avg Volatility: {np.mean(volatilities):.6f}"
                    )
                else:
                    # Multiple backtest mode
                    maes = [r["avg_mae"] for r in category_results.values() if "avg_mae" in r and not np.isnan(r["avg_mae"])]
                    rmses = [r["avg_rmse"] for r in category_results.values() if "avg_rmse" in r and not np.isnan(r["avg_rmse"])]
                    dir_accs = [r["avg_directional_accuracy"] for r in category_results.values() if "avg_directional_accuracy" in r and not np.isnan(r["avg_directional_accuracy"])]
                    success_count = len(category_results)
                    total_in_category = len(methods)
                    success_rate = success_count / total_in_category * 100
                    
                    analysis.append(
                        f"{category:<22}: {success_count}/{total_in_category} methods ({success_rate:.1f}%), "
                        f"Avg MAE: {np.mean(maes):.6f}, Avg Dir.Acc: {np.mean(dir_accs):.1%}"
                    )
        
        analysis.append("="*80)
        return "\n".join(analysis)

    def create_denoise_ranking_analysis(self, denoise_comparison_results: Dict[str, Any]) -> str:
        """Create detailed ranking analysis of denoise method comparison."""
        denoise_results = denoise_comparison_results.get("denoise_results", {})
        target_method = denoise_comparison_results.get("target_method", "unknown")
        successful_results = {k: v for k, v in denoise_results.items() if v.get("success")}
        
        analysis = [
            "\n" + "="*80,
            f"DENOISE METHOD RANKING ANALYSIS FOR {target_method.upper()}",
            "="*80
        ]
        
        if not successful_results:
            analysis.append("No successful denoise tests to analyze.")
            return "\n".join(analysis)
        
        # Determine if we're in single or multiple backtest mode
        is_multiple_backtest = any("avg_mae" in r for r in successful_results.values())
        
        # Create ranking based on performance metric
        if is_multiple_backtest:
            # Sort by MAE (lower is better)
            sorted_methods = sorted(successful_results.items(), 
                                   key=lambda x: x[1].get("avg_mae", float('inf')))
            
            analysis.extend([
                f"RANKING BY AVERAGE MAE (Lower = Better)",
                f"{'Rank':<6} {'Denoise Method':<15} {'Avg MAE':<10} {'Avg RMSE':<10} {'Dir.Acc':<8} {'Consistency':<11} {'Tests':<7}",
                "-" * 80
            ])
            
            best_mae = sorted_methods[0][1].get("avg_mae", float('inf'))
            
            for rank, (method, result) in enumerate(sorted_methods, 1):
                mae = result.get("avg_mae", 0)
                rmse = result.get("avg_rmse", 0)
                dir_acc = result.get("avg_directional_accuracy", 0.5)
                consistency = result.get("consistency", 0)
                successful_bt = result.get("successful_tests", 0)
                total_bt = result.get("num_tests", 1)
                
                # Calculate improvement vs best
                if rank == 1:
                    improvement = "BEST"
                else:
                    pct_worse = ((mae - best_mae) / best_mae) * 100
                    improvement = f"+{pct_worse:.1f}%"
                
                analysis.append(
                    f"#{rank:<5} {method:<15} {mae:<10.6f} {rmse:<10.6f} {dir_acc:<8.1%} {consistency:<11.1%} {successful_bt}/{total_bt:<4} [{improvement}]"
                )
                
        else:
            # Single backtest mode - could rank by various metrics, let's use mean forecast value consistency
            # For now, just show them in order with stats
            analysis.extend([
                f"DENOISE METHOD COMPARISON (Single Backtest Mode)",
                f"{'Method':<15} {'Mean':<12} {'Trend':<6} {'CI Width':<10} {'Status':<8}",
                "-" * 60
            ])
            
            for method, result in successful_results.items():
                stats = result.get("stats", {})
                mean_val = stats.get("mean", 0)
                trend = stats.get("trend", "N/A")
                ci_width = result.get("avg_interval_width", 0)
                
                analysis.append(
                    f"{method:<15} {mean_val:<12.6f} {trend:<6} {ci_width:<10.6f} SUCCESS"
                )
        
        # Add performance insights
        if is_multiple_backtest and len(successful_results) > 1:
            analysis.extend([
                "",
                "PERFORMANCE INSIGHTS:",
                "-" * 40
            ])
            
            best_method, best_result = sorted_methods[0]
            worst_method, worst_result = sorted_methods[-1]
            
            best_mae = best_result.get("avg_mae", 0)
            worst_mae = worst_result.get("avg_mae", 0)
            improvement_pct = ((worst_mae - best_mae) / worst_mae) * 100
            
            analysis.extend([
                f"Best Denoise Method: {best_method} (MAE: {best_mae:.6f})",
                f"Worst Denoise Method: {worst_method} (MAE: {worst_mae:.6f})",
                f"Performance Range: {improvement_pct:.1f}% improvement from best to worst",
                ""
            ])
            
            # Find methods that beat no denoising
            if "none" in successful_results:
                none_mae = successful_results["none"].get("avg_mae", 0)
                better_than_none = [
                    (method, result) for method, result in successful_results.items()
                    if method != "none" and result.get("avg_mae", float('inf')) < none_mae
                ]
                
                if better_than_none:
                    analysis.append(f"Methods Better Than No Denoising ({len(better_than_none)}/{len(successful_results)-1}):")
                    for method, result in better_than_none:
                        mae = result.get("avg_mae", 0)
                        improvement = ((none_mae - mae) / none_mae) * 100
                        analysis.append(f"  • {method}: {mae:.6f} MAE ({improvement:.1f}% better)")
                else:
                    analysis.append("No denoising methods improved upon the baseline (no denoising)")
        
        analysis.append("="*80)
        return "\n".join(analysis)

    # Removed file saving; we print readable summaries to stdout only

def main():
    """Main function to run the comprehensive forecast test suite."""
    # Parse command line arguments
    symbol = sys.argv[1] if len(sys.argv) > 1 else "EURUSD"
    timeframe = sys.argv[2] if len(sys.argv) > 2 else "H1"
    horizon = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    
    # Check for mode flags
    use_multiple_backtests = "--backtest" in sys.argv or "-b" in sys.argv
    denoise_comparison_mode = "--denoise-compare" in sys.argv
    num_backtests = 3  # Default number of backtest points
    
    # Parse number of backtests if specified
    if "--backtests" in sys.argv:
        try:
            idx = sys.argv.index("--backtests")
            if idx + 1 < len(sys.argv):
                num_backtests = int(sys.argv[idx + 1])
        except (ValueError, IndexError):
            num_backtests = 3
    
    # Check for denoise option
    denoise_spec = None
    if "--denoise" in sys.argv:
        try:
            idx = sys.argv.index("--denoise")
            if idx + 1 < len(sys.argv):
                denoise_method = sys.argv[idx + 1]
                # Default denoise spec - simple EMA denoising
                denoise_spec = {
                    "method": denoise_method,
                    "when": "pre_ti",  # Apply before forecasting
                    "columns": ["close"],
                    "keep_original": False,
                    "params": {"span": 10}  # Default EMA span
                }
                # Check for denoise span parameter
                if "--denoise-span" in sys.argv:
                    span_idx = sys.argv.index("--denoise-span")
                    if span_idx + 1 < len(sys.argv):
                        try:
                            denoise_spec["params"]["span"] = int(sys.argv[span_idx + 1])
                        except ValueError:
                            pass
        except (ValueError, IndexError):
            # Default to EMA with span 10
            denoise_spec = {
                "method": "ema",
                "when": "pre_ti", 
                "columns": ["close"],
                "keep_original": False,
                "params": {"span": 10}
            }
    
    # Check for denoise comparison target method
    denoise_target_method = "theta"  # Default method for denoise comparison
    if "--denoise-method" in sys.argv:
        try:
            idx = sys.argv.index("--denoise-method")
            if idx + 1 < len(sys.argv):
                denoise_target_method = sys.argv[idx + 1]
        except (ValueError, IndexError):
            pass
    
    # Validate timeframe
    if timeframe not in TIMEFRAME_MAP:
        logger.error(f"Invalid timeframe: {timeframe}. Valid options: {list(TIMEFRAME_MAP.keys())}")
        sys.exit(1)
    
    # Update title based on mode
    if denoise_comparison_mode:
        mode_info = f" (Denoise Comparison: {denoise_target_method})"
        if use_multiple_backtests:
            mode_info += f" + Multiple Backtests ({num_backtests} points)"
    else:
        mode_info = f" (Multiple Backtests: {num_backtests} points)" if use_multiple_backtests else " (Single Backtest)"
        if denoise_spec:
            mode_info += f" + Denoise[{denoise_spec['method']} span={denoise_spec['params']['span']}]"
    
    print(f"\nMetaTrader5 Forecast Methods Test Suite{mode_info}")
    print(f"Symbol: {symbol} | Timeframe: {timeframe} | Horizon: {horizon} periods")
    
    if denoise_comparison_mode:
        print(f"Denoise Comparison Mode: Testing all available denoise methods with {denoise_target_method} forecasting")
        if use_multiple_backtests:
            print(f"Each denoise method tested at {num_backtests} different historical points")
        else:
            print(f"Each denoise method tested at one historical point (~200 periods back)")
    else:
        if use_multiple_backtests:
            print(f"Backtest Mode: Testing each method at {num_backtests} different historical points")
        else:
            print(f"Single Test Mode: Testing each method at one historical point (~200 periods back)")
        if denoise_spec:
            print(f"Denoising: Applying {denoise_spec['method']} filter (span={denoise_spec['params']['span']}) to training data before forecasting")
    
    print("="*80)
    
    # Initialize test suite
    test_suite = ForecastTestSuite(symbol=symbol, timeframe=timeframe)
    
    # Run appropriate test based on mode
    if denoise_comparison_mode:
        # Run denoise comparison test
        comparison_results = test_suite.run_denoise_comparison_test(
            horizon=horizon,
            use_multiple_backtests=use_multiple_backtests, 
            num_backtests=num_backtests,
            target_method=denoise_target_method
        )
        
        if "error" in comparison_results:
            logger.error(f"Denoise comparison failed: {comparison_results['error']}")
            sys.exit(1)
            
        # Display denoise ranking analysis
        denoise_analysis = test_suite.create_denoise_ranking_analysis(comparison_results)
        print(denoise_analysis)
        
        # Summary for denoise comparison
        total_denoise = comparison_results.get("total_denoise_methods", 0)
        successful_denoise = comparison_results.get("successful_denoise_methods", 0)
        denoise_success_rate = comparison_results.get("denoise_success_rate", 0)
        
        print(f"\n" + "="*70)
        print("DENOISE COMPARISON SUMMARY")
        print("="*70)
        print(f"Target Method: {denoise_target_method}")
        print(f"Symbol: {symbol}")
        print(f"Timeframe: {timeframe}")
        print(f"Forecast Horizon: {horizon} periods")
        print(f"")
        print(f"Denoise Methods Tested: {total_denoise}")
        print(f"Successful: {successful_denoise}")
        print(f"Failed: {total_denoise - successful_denoise}")
        print(f"Success Rate: {denoise_success_rate:.1f}%")
        
        # Performance assessment
        if denoise_success_rate >= 90:
            print(f"\n[EXCELLENT] Denoise comparison completed successfully!")
        elif denoise_success_rate >= 75:
            print(f"\n[VERY GOOD] Most denoise methods tested successfully.")
        elif denoise_success_rate >= 50:
            print(f"\n[GOOD] Majority of denoise methods worked.")
        elif denoise_success_rate >= 25:
            print(f"\n[MODERATE] Some denoise methods need attention.")
        else:
            print(f"\n[POOR] Significant issues with denoise methods.")
        
        print("="*70)
        
    else:
        # Run standard comprehensive test
        comprehensive_results = test_suite.run_comprehensive_test(
            horizon=horizon, 
            use_multiple_backtests=use_multiple_backtests,
            num_backtests=num_backtests,
            denoise_spec=denoise_spec
        )
        
        if "error" in comprehensive_results:
            logger.error(f"Test suite failed: {comprehensive_results['error']}")
            sys.exit(1)
        
        # Display detailed analysis
        detailed_analysis = test_suite.create_detailed_analysis(comprehensive_results)
        print(detailed_analysis)
    
    # Final summary with enhanced period information (only for standard mode)
    if not denoise_comparison_mode:
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
