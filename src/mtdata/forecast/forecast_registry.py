"""
Forecast method registry and metadata management.

Centralizes method definitions, requirements, and availability checking.
"""

from typing import Any, Dict, List

# Import availability checkers
try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing as _ETS  # type: ignore
    _SM_ETS_AVAILABLE = True
except Exception:
    _SM_ETS_AVAILABLE = False

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX as _SARIMAX  # type: ignore
    _SM_SARIMAX_AVAILABLE = True
except Exception:
    _SM_SARIMAX_AVAILABLE = False

try:
    import importlib.util as _importlib_util
    _NF_AVAILABLE = _importlib_util.find_spec("neuralforecast") is not None
    _MLF_AVAILABLE = _importlib_util.find_spec("mlforecast") is not None
    _SF_AVAILABLE = _importlib_util.find_spec("statsforecast") is not None
    _LGB_AVAILABLE = _importlib_util.find_spec("lightgbm") is not None
    _CHRONOS_AVAILABLE = _importlib_util.find_spec("chronos") is not None or _importlib_util.find_spec("transformers") is not None
    _TIMESFM_AVAILABLE = _importlib_util.find_spec("timesfm") is not None or _importlib_util.find_spec("transformers") is not None
    _LAG_LLAMA_AVAILABLE = _importlib_util.find_spec("lag_llama") is not None
except Exception:
    _NF_AVAILABLE = False
    _MLF_AVAILABLE = False
    _SF_AVAILABLE = False
    _LGB_AVAILABLE = False
    _CHRONOS_AVAILABLE = False
    _TIMESFM_AVAILABLE = False
    _LAG_LLAMA_AVAILABLE = False


def get_forecast_methods_data() -> Dict[str, Any]:
    """Return metadata about available forecast methods and their requirements."""
    methods: List[Dict[str, Any]] = []

    def add(method: str, description: str, params: List[Dict[str, Any]], requires: List[str], supports: Dict[str, bool]) -> None:
        available = True
        reqs = list(requires)
        
        # Check availability based on method type
        if method in ("ses", "holt", "holt_winters_add", "holt_winters_mul") and not _SM_ETS_AVAILABLE:
            available = False; reqs.append("statsmodels")
        if method in ("arima", "sarima") and not _SM_SARIMAX_AVAILABLE:
            available = False; reqs.append("statsmodels")
        if method in ("nhits", "nbeatsx", "tft", "patchtst") and not _NF_AVAILABLE:
            available = False; reqs.append("neuralforecast[torch]")
        if method.startswith("sf_") and not _SF_AVAILABLE:
            available = False; reqs.append("statsforecast")
        if method == "mlf_rf" and not _MLF_AVAILABLE:
            available = False; reqs.append("mlforecast, scikit-learn")
        if method == "mlf_lightgbm" and (not _MLF_AVAILABLE or not _LGB_AVAILABLE):
            available = False; reqs.append("mlforecast, lightgbm")
        if method == "chronos_bolt" and not _CHRONOS_AVAILABLE:
            available = False; reqs.append("chronos or transformers")
        if method == "timesfm" and not _TIMESFM_AVAILABLE:
            available = False; reqs.append("timesfm or transformers")
        if method == "lag_llama" and not _LAG_LLAMA_AVAILABLE:
            available = False; reqs.append("lag_llama or transformers")
        
        methods.append({
            "method": method,
            "available": bool(available),
            "requires": sorted(set(reqs)),
            "description": description,
            "params": params,
            "supports": supports,
        })

    # Define all methods
    _register_classical_methods(add)
    _register_ets_methods(add)
    _register_monte_carlo_methods(add)
    _register_neural_methods(add)
    _register_statistical_methods(add)
    _register_ml_methods(add)
    _register_pretrained_methods(add)

    return {
        "methods": methods,
        "total": len(methods),
        "categories": {
            "classical": ["naive", "drift", "seasonal_naive", "theta", "fourier_ols"],
            "ets": ["ses", "holt", "holt_winters_add", "holt_winters_mul"],
            "arima": ["arima", "sarima"],
            "monte_carlo": ["mc_gbm", "hmm_mc"],
            "neural": ["nhits", "nbeatsx", "tft", "patchtst"],
            "statsforecast": ["sf_autoarima", "sf_theta", "sf_autoets", "sf_seasonalnaive"],
            "machine_learning": ["mlf_rf", "mlf_lightgbm"],
            "pretrained": ["chronos_bolt", "timesfm", "lag_llama"],
            "ensemble": ["ensemble"]
        }
    }


def _register_classical_methods(add_func):
    """Register classical forecasting methods."""
    add_func("naive", "Repeat last value forward.", [], [], {"price": True, "return": True, "ci": True})
    add_func("seasonal_naive", "Repeat last seasonal value (period m).", [
        {"name": "seasonality", "type": "int", "default": None, "description": "Seasonal period. Auto by timeframe if omitted."},
    ], [], {"price": True, "return": True, "ci": True})
    add_func("drift", "Line from first to last with constant slope.", [], [], {"price": True, "return": True, "ci": True})
    add_func("theta", "Theta method (SES + trend).", [
        {"name": "alpha", "type": "float", "default": 0.2, "description": "SES smoothing factor for theta blend."},
    ], [], {"price": True, "return": True, "ci": True})
    add_func("fourier_ols", "Fourier series regression with optional trend.", [
        {"name": "seasonality", "type": "int", "default": None, "description": "Seasonal period m."},
        {"name": "K", "type": "int", "default": 3, "description": "Number of Fourier harmonics."},
        {"name": "trend", "type": "bool", "default": True, "description": "Include linear trend."},
    ], [], {"price": True, "return": True, "ci": True})


def _register_ets_methods(add_func):
    """Register ETS/Holt-Winters methods."""
    add_func("ses", "Simple Exponential Smoothing (statsmodels).", [
        {"name": "alpha", "type": "float|null", "default": None, "description": "If None, estimated by MLE."},
    ], [], {"price": True, "return": True, "ci": True})
    add_func("holt", "Holt's linear trend (statsmodels).", [
        {"name": "damped", "type": "bool", "default": True, "description": "Use damped trend variant."},
    ], [], {"price": True, "return": True, "ci": True})
    add_func("holt_winters_add", "Additive Holt-Winters (statsmodels).", [
        {"name": "seasonality", "type": "int", "default": None, "description": "Seasonal period m."},
    ], [], {"price": True, "return": True, "ci": True})
    add_func("holt_winters_mul", "Multiplicative Holt-Winters (statsmodels).", [
        {"name": "seasonality", "type": "int", "default": None, "description": "Seasonal period m."},
    ], [], {"price": True, "return": True, "ci": True})
    add_func("arima", "Non-seasonal ARIMA via SARIMAX.", [
        {"name": "p", "type": "int", "default": 1}, {"name": "d", "type": "int", "default": 0}, {"name": "q", "type": "int", "default": 1},
        {"name": "trend", "type": "str", "default": "c"},
    ], [], {"price": True, "return": True, "ci": True})
    add_func("sarima", "Seasonal ARIMA via SARIMAX.", [
        {"name": "p", "type": "int", "default": 1}, {"name": "d", "type": "int", "default": 0}, {"name": "q", "type": "int", "default": 1},
        {"name": "P", "type": "int", "default": 0}, {"name": "D", "type": "int", "default": 1}, {"name": "Q", "type": "int", "default": 0},
        {"name": "seasonality", "type": "int", "default": None},
        {"name": "trend", "type": "str", "default": "c"},
    ], [], {"price": True, "return": True, "ci": True})


def _register_monte_carlo_methods(add_func):
    """Register Monte Carlo methods."""
    add_func("mc_gbm", "Monte Carlo with GBM calibrated from log-returns.", [
        {"name": "n_sims", "type": "int", "default": 500},
        {"name": "seed", "type": "int", "default": 42},
    ], [], {"price": True, "return": True, "ci": True})
    add_func("hmm_mc", "Monte Carlo with Gaussian HMM regimes over returns.", [
        {"name": "n_states", "type": "int", "default": 2},
        {"name": "n_sims", "type": "int", "default": 500},
        {"name": "seed", "type": "int", "default": 42},
    ], [], {"price": True, "return": True, "ci": True})


def _register_neural_methods(add_func):
    """Register neural network methods."""
    add_func("nhits", "NeuralForecast NHITS (PyTorch).", [], ["neuralforecast[torch]"], {"price": True, "return": True, "ci": False})
    add_func("nbeatsx", "NeuralForecast NBEATSx (PyTorch).", [], ["neuralforecast[torch]"], {"price": True, "return": True, "ci": False})
    add_func("tft", "NeuralForecast TFT (PyTorch).", [], ["neuralforecast[torch]"], {"price": True, "return": True, "ci": False})
    add_func("patchtst", "NeuralForecast PatchTST (PyTorch).", [], ["neuralforecast[torch]"], {"price": True, "return": True, "ci": False})


def _register_statistical_methods(add_func):
    """Register StatsForecast methods."""
    add_func("sf_autoarima", "StatsForecast AutoARIMA.", [], ["statsforecast"], {"price": True, "return": True, "ci": False})
    add_func("sf_theta", "StatsForecast Theta.", [], ["statsforecast"], {"price": True, "return": True, "ci": False})
    add_func("sf_autoets", "StatsForecast AutoETS.", [], ["statsforecast"], {"price": True, "return": True, "ci": False})
    add_func("sf_seasonalnaive", "StatsForecast SeasonalNaive.", [], ["statsforecast"], {"price": True, "return": True, "ci": False})


def _register_ml_methods(add_func):
    """Register machine learning methods."""
    add_func("mlf_rf", "MLForecast Random Forest", [
        {"name": "n_estimators", "type": "int", "description": "Number of trees (default: 100)"},
        {"name": "max_depth", "type": "int", "description": "Maximum depth (default: None)"},
        {"name": "lags", "type": "list", "description": "Lag features to use (auto if omitted)"}
    ], ["mlforecast", "scikit-learn"], {"price": True, "return": True, "volatility": True, "ci": True})
    
    add_func("mlf_lightgbm", "MLForecast LightGBM", [
        {"name": "num_leaves", "type": "int", "description": "Number of leaves (default: 31)"},
        {"name": "learning_rate", "type": "float", "description": "Learning rate (default: 0.1)"},
        {"name": "lags", "type": "list", "description": "Lag features to use (auto if omitted)"}
    ], ["mlforecast", "lightgbm"], {"price": True, "return": True, "volatility": True, "ci": True})


def _register_pretrained_methods(add_func):
    """Register pre-trained model methods."""
    add_func("chronos_bolt", "Amazon Chronos-BOLT pre-trained time series model", [
        {"name": "device", "type": "str", "description": "Compute device (cpu/cuda, default: auto)"},
        {"name": "limit_prediction_length", "type": "bool", "description": "Limit to horizon (default: True)"}
    ], ["torch", "transformers"], {"price": True, "return": True, "volatility": True, "ci": True})
    
    add_func("timesfm", "Google TimesFM pre-trained time series foundation model", [
        {"name": "device", "type": "str", "description": "Compute device (cpu/cuda, default: auto)"},
        {"name": "batch_size", "type": "int", "description": "Batch size (default: 16)"}
    ], ["torch", "transformers"], {"price": True, "return": True, "volatility": True, "ci": True})
    
    add_func("lag_llama", "Lag-Llama pre-trained time series model", [
        {"name": "device", "type": "str", "description": "Compute device (cpu/cuda, default: auto)"},
        {"name": "batch_size", "type": "int", "description": "Batch size (default: 1)"}
    ], ["torch", "transformers"], {"price": True, "return": True, "volatility": True, "ci": True})


# Availability flags that can be imported by other modules
__all__ = [
    'get_forecast_methods_data',
    '_SM_ETS_AVAILABLE',
    '_SM_SARIMAX_AVAILABLE', 
    '_NF_AVAILABLE',
    '_MLF_AVAILABLE',
    '_SF_AVAILABLE',
    '_LGB_AVAILABLE',
    '_CHRONOS_AVAILABLE',
    '_TIMESFM_AVAILABLE',
    '_LAG_LLAMA_AVAILABLE'
]