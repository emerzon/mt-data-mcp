"""Forecast methods registry and metadata."""
from typing import Any, Dict, List
import importlib.util as _importlib_util

# Availability checks
try:
    from statsmodels.tsa.holtwinters import SimpleExpSmoothing as _SES, ExponentialSmoothing as _ETS  # type: ignore
    _SM_ETS_AVAILABLE = True
except Exception:
    _SM_ETS_AVAILABLE = False

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX as _SARIMAX  # type: ignore
    _SM_SARIMAX_AVAILABLE = True
except Exception:
    _SM_SARIMAX_AVAILABLE = False

_NF_AVAILABLE = _importlib_util.find_spec("neuralforecast") is not None
_SF_AVAILABLE = _importlib_util.find_spec("statsforecast") is not None
_MLF_AVAILABLE = _importlib_util.find_spec("mlforecast") is not None
_LGB_AVAILABLE = _importlib_util.find_spec("lightgbm") is not None
_CHRONOS_AVAILABLE = _importlib_util.find_spec("chronos") is not None
_TIMESFM_AVAILABLE = _importlib_util.find_spec("timesfm") is not None
_LAG_LLAMA_AVAILABLE = (_importlib_util.find_spec("lag_llama") is not None)
_MOIRAI_AVAILABLE = (_importlib_util.find_spec("uni2ts") is not None)


def get_forecast_methods_data() -> Dict[str, Any]:
    """Return metadata about available forecast methods and their requirements."""
    methods: List[Dict[str, Any]] = []

    def add(method: str, description: str, params: List[Dict[str, Any]], 
            requires: List[str], supports: Dict[str, bool]) -> None:
        available = True
        reqs = list(requires)
        
        if method in ("ses", "holt", "holt_winters_add", "holt_winters_mul") and not _SM_ETS_AVAILABLE:
            available = False
            reqs.append("statsmodels")
        if method in ("arima", "sarima") and not _SM_SARIMAX_AVAILABLE:
            available = False
            reqs.append("statsmodels")
        if method in ("nhits", "nbeatsx", "tft", "patchtst") and not _NF_AVAILABLE:
            available = False
            reqs.append("neuralforecast[torch]")
        if method.startswith("sf_") and not _SF_AVAILABLE:
            available = False
            reqs.append("statsforecast")
        if method == "mlf_rf" and not _MLF_AVAILABLE:
            available = False
            reqs.append("mlforecast, scikit-learn")
        if method == "mlf_lightgbm" and (not _MLF_AVAILABLE or not _LGB_AVAILABLE):
            available = False
            reqs.append("mlforecast, lightgbm")
        if method in ("chronos_bolt", "chronos2") and not _CHRONOS_AVAILABLE:
            available = False
            reqs.append("chronos-forecasting")
        if method == "timesfm" and not _TIMESFM_AVAILABLE:
            available = False
            reqs.append("timesfm")
        if method == "lag_llama" and not _LAG_LLAMA_AVAILABLE:
            available = False
            reqs.append("lag-llama, gluonts, torch")
        if method == "moirai" and not _MOIRAI_AVAILABLE:
            available = False
            reqs.append("uni2ts[moirai], torch")
        if method == "ensemble":
            available = False
            reqs.append("not implemented")
            
        methods.append({
            "method": method,
            "available": bool(available),
            "requires": sorted(set(reqs)),
            "description": description,
            "params": params,
            "supports": supports,
        })

    # Baselines
    add("naive", "Repeat last value forward.", [], [], 
        {"price": True, "return": True, "ci": True})
    add("seasonal_naive", "Repeat last seasonal value (period m).", [
        {"name": "seasonality", "type": "int", "default": None, 
         "description": "Seasonal period. Auto by timeframe if omitted."},
    ], [], {"price": True, "return": True, "ci": True})
    add("drift", "Line from first to last with constant slope.", [], [], 
        {"price": True, "return": True, "ci": True})
    
    # Classical
    add("theta", "Theta method (SES + trend).", [
        {"name": "seasonality", "type": "int", "default": None, 
         "description": "Seasonal period for deseasonalization."},
    ], [], {"price": True, "return": True, "ci": True})
    add("fourier_ols", "Fourier series regression.", [
        {"name": "seasonality", "type": "int", "default": None},
        {"name": "fourier_order", "type": "int", "default": 5},
    ], [], {"price": True, "return": True, "ci": True})
    
    # ETS
    add("ses", "Simple Exponential Smoothing.", [
        {"name": "alpha", "type": "float", "default": None},
    ], ["statsmodels"], {"price": True, "return": True, "ci": True})
    add("holt", "Holt linear trend.", [
        {"name": "alpha", "type": "float", "default": None},
        {"name": "beta", "type": "float", "default": None},
    ], ["statsmodels"], {"price": True, "return": True, "ci": True})
    add("holt_winters_add", "Holt-Winters additive.", [
        {"name": "seasonality", "type": "int", "default": None},
    ], ["statsmodels"], {"price": True, "return": True, "ci": True})
    add("holt_winters_mul", "Holt-Winters multiplicative.", [
        {"name": "seasonality", "type": "int", "default": None},
    ], ["statsmodels"], {"price": True, "return": True, "ci": True})
    
    # ARIMA
    add("arima", "Non-seasonal ARIMA.", [
        {"name": "p", "type": "int", "default": 1},
        {"name": "d", "type": "int", "default": 1},
        {"name": "q", "type": "int", "default": 1},
    ], ["statsmodels"], {"price": True, "return": True, "ci": True})
    add("sarima", "Seasonal ARIMA.", [
        {"name": "p", "type": "int", "default": 1},
        {"name": "d", "type": "int", "default": 1},
        {"name": "q", "type": "int", "default": 1},
        {"name": "P", "type": "int", "default": 1},
        {"name": "D", "type": "int", "default": 1},
        {"name": "Q", "type": "int", "default": 1},
        {"name": "seasonality", "type": "int", "default": None},
    ], ["statsmodels"], {"price": True, "return": True, "ci": True})
    
    # Monte Carlo
    add("mc_gbm", "Geometric Brownian Motion Monte Carlo.", [
        {"name": "n_sims", "type": "int", "default": 1000},
        {"name": "seed", "type": "int", "default": None},
    ], [], {"price": True, "return": False, "ci": True, "distribution": True})
    add("hmm_mc", "HMM-based regime Monte Carlo.", [
        {"name": "n_states", "type": "int", "default": 2},
        {"name": "n_sims", "type": "int", "default": 1000},
        {"name": "seed", "type": "int", "default": None},
    ], [], {"price": True, "return": False, "ci": True, "distribution": True})
    
    # StatsForecast
    add("sf_autoarima", "StatsForecast AutoARIMA.", [
        {"name": "seasonality", "type": "int", "default": None},
        {"name": "stepwise", "type": "bool", "default": True},
    ], ["statsforecast"], {"price": True, "return": True, "ci": True})
    add("sf_theta", "StatsForecast Theta.", [
        {"name": "seasonality", "type": "int", "default": None},
    ], ["statsforecast"], {"price": True, "return": True, "ci": True})
    add("sf_autoets", "StatsForecast AutoETS.", [
        {"name": "seasonality", "type": "int", "default": None},
    ], ["statsforecast"], {"price": True, "return": True, "ci": True})
    add("sf_seasonalnaive", "StatsForecast SeasonalNaive.", [
        {"name": "seasonality", "type": "int", "default": None},
    ], ["statsforecast"], {"price": True, "return": True, "ci": True})
    
    # MLForecast
    add("mlf_rf", "MLForecast Random Forest.", [
        {"name": "lags", "type": "list", "default": [1, 2, 3]},
        {"name": "n_estimators", "type": "int", "default": 100},
    ], ["mlforecast", "scikit-learn"], {"price": True, "return": True, "ci": False})
    add("mlf_lightgbm", "MLForecast LightGBM.", [
        {"name": "lags", "type": "list", "default": [1, 2, 3]},
        {"name": "n_estimators", "type": "int", "default": 100},
    ], ["mlforecast", "lightgbm"], {"price": True, "return": True, "ci": False})
    
    # NeuralForecast
    add("nhits", "NeuralForecast NHITS.", [
        {"name": "max_epochs", "type": "int", "default": 50},
        {"name": "input_size", "type": "int", "default": 128},
    ], ["neuralforecast[torch]"], {"price": True, "return": True, "ci": True})
    add("nbeatsx", "NeuralForecast NBEATS-X.", [
        {"name": "max_epochs", "type": "int", "default": 50},
        {"name": "input_size", "type": "int", "default": 128},
    ], ["neuralforecast[torch]"], {"price": True, "return": True, "ci": True})
    add("tft", "NeuralForecast TFT.", [
        {"name": "max_epochs", "type": "int", "default": 50},
        {"name": "input_size", "type": "int", "default": 128},
    ], ["neuralforecast[torch]"], {"price": True, "return": True, "ci": True})
    add("patchtst", "NeuralForecast PatchTST.", [
        {"name": "max_epochs", "type": "int", "default": 50},
        {"name": "input_size", "type": "int", "default": 128},
    ], ["neuralforecast[torch]"], {"price": True, "return": True, "ci": True})
    
    # Foundation models
    add("chronos_bolt", "Chronos-2 foundation model (Chronos-Bolt successor; upstream model supports cross-learning, multivariate targets, and covariates—current adapter is univariate).", [
        {"name": "model_name", "type": "str", "default": "amazon/chronos-2"},
        {"name": "context_length", "type": "int", "default": 512},
        {"name": "quantiles", "type": "list", "default": [0.5]},
        {"name": "device_map", "type": "str", "default": "auto"},
    ], ["chronos-forecasting>=2.0.0"], {"price": True, "return": True, "ci": True})
    add("chronos2", "Chronos-2 foundation model (preferred name; same as chronos_bolt). Upstream supports cross-learning, multivariate targets, and covariates—current adapter is univariate.", [
        {"name": "model_name", "type": "str", "default": "amazon/chronos-2"},
        {"name": "context_length", "type": "int", "default": 512},
        {"name": "quantiles", "type": "list", "default": [0.5]},
        {"name": "device_map", "type": "str", "default": "auto"},
    ], ["chronos-forecasting>=2.0.0"], {"price": True, "return": True, "ci": True})
    add("timesfm", "Google TimesFM foundation model.", [
        {"name": "context_length", "type": "int", "default": 512},
    ], ["timesfm"], {"price": True, "return": True, "ci": True})
    add("lag_llama", "Lag-Llama foundation model.", [
        {"name": "context_length", "type": "int", "default": 512},
    ], ["lag-llama", "gluonts", "torch"], {"price": True, "return": True, "ci": True})
    add("moirai", "Salesforce Moirai (uni2ts) foundation model.", [
        {"name": "context_length", "type": "int", "default": 512},
        {"name": "variant", "type": "str", "default": "1.0-R-small"}
    ], ["uni2ts"], {"price": True, "return": True, "ci": True})
    
    return {"methods": methods}
