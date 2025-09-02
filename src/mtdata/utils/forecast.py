from typing import Any, Dict, List


def get_forecast_methods_data(_SM_ETS_AVAILABLE: bool, _SM_SARIMAX_AVAILABLE: bool) -> Dict[str, Any]:
    methods: List[Dict[str, Any]] = []

    def add(method: str, available: bool, description: str, params: List[Dict[str, Any]], defaults: Dict[str, Any]) -> None:
        methods.append({
            "method": method,
            "available": bool(available),
            "description": description,
            "params": params,
            "defaults": defaults,
        })

    common_defaults = {
        "timeframe": "H1",
        "horizon": 12,
        "lookback": None,
        "as_of": None,
        "ci_alpha": 0.05,  # null to disable intervals
        "target": "price",
    }

    add("naive", True, "Repeat last observed value (random walk baseline).", [], common_defaults)
    add("drift", True, "Linear drift from first to last observation; strong simple baseline.", [], common_defaults)
    add("seasonal_naive", True, "Repeat last season's values; requires seasonality period m.", [
        {"name": "seasonality", "type": "int", "default": None, "description": "Season period m; auto from timeframe if omitted."},
    ], common_defaults)
    add("theta", True, "Fast Theta-style: average of linear trend extrapolation and SES level.", [
        {"name": "alpha", "type": "float", "default": 0.2, "description": "SES smoothing for level component."},
    ], common_defaults)
    add("fourier_ols", True, "Fourier regression with K harmonics and optional linear trend.", [
        {"name": "seasonality", "type": "int", "default": None, "description": "Season period m; auto by timeframe if omitted."},
        {"name": "K", "type": "int", "default": "min(3, m/2)", "description": "Number of harmonics."},
        {"name": "trend", "type": "bool", "default": True, "description": "Include linear trend term."},
    ], common_defaults)
    add("ses", _SM_ETS_AVAILABLE, "Simple Exponential Smoothing (statsmodels).", [
        {"name": "alpha", "type": "float", "default": None, "description": "Smoothing level; optimized if None."},
    ], common_defaults)
    add("holt", _SM_ETS_AVAILABLE, "Holt's linear trend with optional damping (statsmodels).", [
        {"name": "damped", "type": "bool", "default": True, "description": "Use damped trend."},
    ], common_defaults)
    add("holt_winters_add", _SM_ETS_AVAILABLE, "Additive Holt-Winters with additive seasonality (statsmodels).", [
        {"name": "seasonality", "type": "int", "default": None, "description": "Season period m; required."},
    ], common_defaults)
    add("holt_winters_mul", _SM_ETS_AVAILABLE, "Additive trend with multiplicative seasonality (statsmodels).", [
        {"name": "seasonality", "type": "int", "default": None, "description": "Season period m; required."},
    ], common_defaults)
    add("arima", _SM_SARIMAX_AVAILABLE, "Non-seasonal ARIMA via SARIMAX (statsmodels).", [
        {"name": "p", "type": "int", "default": 1, "description": "AR order."},
        {"name": "d", "type": "int", "default": "0 (return) or 1 (price)", "description": "Differencing order."},
        {"name": "q", "type": "int", "default": 1, "description": "MA order."},
        {"name": "trend", "type": "str", "default": "c", "description": "Trend: 'c' constant, 'n' none."},
    ], common_defaults)
    add("sarima", _SM_SARIMAX_AVAILABLE, "Seasonal ARIMA via SARIMAX (statsmodels).", [
        {"name": "p", "type": "int", "default": 1, "description": "AR order."},
        {"name": "d", "type": "int", "default": "0 (return) or 1 (price)", "description": "Differencing order."},
        {"name": "q", "type": "int", "default": 1, "description": "MA order."},
        {"name": "P", "type": "int", "default": 0, "description": "Seasonal AR order."},
        {"name": "D", "type": "int", "default": "0 (return) or 1 (price)", "description": "Seasonal differencing order."},
        {"name": "Q", "type": "int", "default": 0, "description": "Seasonal MA order."},
        {"name": "seasonality", "type": "int", "default": None, "description": "Season period m; auto by timeframe if omitted."},
        {"name": "trend", "type": "str", "default": "c", "description": "Trend: 'c' constant, 'n' none."},
    ], common_defaults)

    return {"success": True, "schema_version": 1, "methods": methods}

