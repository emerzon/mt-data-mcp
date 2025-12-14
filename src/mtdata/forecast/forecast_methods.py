"""
Forecast method definitions and metadata.
"""

from typing import Any, Dict, List, Literal
import importlib.util as _importlib_util

# Local fallbacks for typing aliases (avoid import cycle)
try:
    from mtdata.core.server import ForecastMethodLiteral, TimeframeLiteral  # type: ignore
except Exception:  # runtime fallback
    ForecastMethodLiteral = str
    TimeframeLiteral = str

# Supported forecast methods
FORECAST_METHODS = (
    "naive",
    "seasonal_naive",
    "drift",
    "theta",
    "fourier_ols",
    "ses",
    "holt",
    "holt_winters_add",
    "holt_winters_mul",
    "arima",
    "sarima",
    "mc_gbm",
    "hmm_mc",
    "nhits",
    "nbeatsx",
    "tft",
    "patchtst",
    "sf_autoarima",
    "sf_theta",
    "sf_autoets",
    "sf_seasonalnaive",
    "mlf_rf",
    "mlf_lightgbm",
    "chronos_bolt",
    "chronos2",
    "timesfm",
    "lag_llama",
    "gt_deepar",
    "gt_sfeedforward",
    "gt_prophet",
    "gt_tft",
    "gt_wavenet",
    "gt_deepnpts",
    "gt_mqf2",
    "gt_npts",
    "ensemble",
    "analog",
)


def get_forecast_methods_data() -> Dict[str, Any]:
    """Get comprehensive data about available forecast methods."""
    methods = []

    module_name_overrides = {
        # pip name -> importable module name
        "scikit-learn": "sklearn",
        "lag-llama": "lag_llama",
        "chronos-forecasting>=2.0.0": "chronos",
        "pandas-ta-classic": "pandas_ta_classic",
        "pandas-ta": "pandas_ta",
        "python-dotenv": "dotenv",
    }

    def _is_available(requirement: str) -> bool:
        name = str(requirement).strip()
        if not name:
            return True
        # Strip simple version constraints like "pkg>=1.2"
        for sep in (">=", "==", "<=", "~=", ">", "<"):
            if sep in name:
                name = name.split(sep, 1)[0].strip()
                break
        name = module_name_overrides.get(name, name)
        try:
            return _importlib_util.find_spec(name) is not None
        except Exception:
            return False

    def add(method: str, description: str, params: List[Dict[str, Any]], requires: List[str], supports: Dict[str, bool]) -> None:
        missing = [pkg for pkg in (requires or []) if not _is_available(pkg)]
        methods.append({
            "method": method,
            "description": description,
            "params": params,
            "requires": requires,
            "supports": supports,
            "available": len(missing) == 0,
            "missing": missing,
        })

    # Classical methods
    add("naive", "Last value carried forward",
        [], [],
        {"price": True, "return": True, "volatility": True, "ci": False})

    add("drift", "Linear trend extension from last observation",
        [], ["statsmodels"],
        {"price": True, "return": True, "volatility": True, "ci": False})

    add("seasonal_naive", "Last seasonal period value carried forward",
        [{"name": "seasonality", "type": "int", "description": "Seasonal period (auto if omitted)"}],
        ["pandas"],
        {"price": True, "return": True, "volatility": True, "ci": False})

    add("theta", "Theta decomposition with SES and linear drift components",
        [{"name": "seasonality", "type": "int", "description": "Seasonal period (auto if omitted)"},
         {"name": "alpha", "type": "float", "description": "Smoothing parameter (0.1-1.0, auto if omitted)"},
         {"name": "drift", "type": "bool", "description": "Include drift component (default: True)"}],
        ["statsmodels"],
        {"price": True, "return": True, "volatility": True, "ci": True})

    add("fourier_ols", "Fourier series regression with ordinary least squares",
        [{"name": "seasonality", "type": "int", "description": "Base seasonal period (auto if omitted)"},
         {"name": "terms", "type": "int", "description": "Number of Fourier terms (default: 3)"},
         {"name": "trend", "type": "bool", "description": "Include linear trend (default: True)"}],
        ["statsmodels", "pandas"],
        {"price": True, "return": True, "volatility": True, "ci": True})

    # ETS family
    add("ses", "Simple exponential smoothing",
        [{"name": "alpha", "type": "float", "description": "Smoothing parameter (0.1-1.0, auto if omitted)"},
         {"name": "damped", "type": "bool", "description": "Apply damping (default: False)"}],
        ["statsmodels"],
        {"price": True, "return": True, "volatility": True, "ci": True})

    add("holt", "Holt's linear trend method",
        [{"name": "alpha", "type": "float", "description": "Level smoothing (0.1-1.0, auto if omitted)"},
         {"name": "beta", "type": "float", "description": "Trend smoothing (0.1-1.0, auto if omitted)"},
         {"name": "damped", "type": "bool", "description": "Apply damping (default: False)"}],
        ["statsmodels"],
        {"price": True, "return": True, "volatility": True, "ci": True})

    add("holt_winters_add", "Holt-Winters additive seasonality",
        [{"name": "seasonality", "type": "int", "description": "Seasonal period (auto if omitted)"},
         {"name": "alpha", "type": "float", "description": "Level smoothing (0.1-1.0, auto if omitted)"},
         {"name": "beta", "type": "float", "description": "Trend smoothing (0.1-1.0, auto if omitted)"},
         {"name": "gamma", "type": "float", "description": "Seasonal smoothing (0.1-1.0, auto if omitted)"},
         {"name": "damped", "type": "bool", "description": "Apply damping (default: False)"}],
        ["statsmodels"],
        {"price": True, "return": True, "volatility": True, "ci": True})

    add("holt_winters_mul", "Holt-Winters multiplicative seasonality",
        [{"name": "seasonality", "type": "int", "description": "Seasonal period (auto if omitted)"},
         {"name": "alpha", "type": "float", "description": "Level smoothing (0.1-1.0, auto if omitted)"},
         {"name": "beta", "type": "float", "description": "Trend smoothing (0.1-1.0, auto if omitted)"},
         {"name": "gamma", "type": "float", "description": "Seasonal smoothing (0.1-1.0, auto if omitted)"},
         {"name": "damped", "type": "bool", "description": "Apply damping (default: False)"}],
        ["statsmodels"],
        {"price": True, "return": False, "volatility": False, "ci": True})

    # ARIMA family
    add("arima", "Autoregressive Integrated Moving Average",
        [{"name": "order", "type": "tuple", "description": "(p,d,q) order (auto if omitted)"},
         {"name": "alpha", "type": "float", "description": "Significance level for CI (default: 0.05)"}],
        ["statsmodels", "pmdarima"],
        {"price": True, "return": True, "volatility": True, "ci": True})

    add("sarima", "Seasonal ARIMA",
        [{"name": "order", "type": "tuple", "description": "(p,d,q) order (auto if omitted)"},
         {"name": "seasonal_order", "type": "tuple", "description": "(P,D,Q,s) order (auto if omitted)"},
         {"name": "seasonality", "type": "int", "description": "Seasonal period (auto if omitted)"},
         {"name": "alpha", "type": "float", "description": "Significance level for CI (default: 0.05)"}],
        ["statsmodels", "pmdarima"],
        {"price": True, "return": True, "volatility": True, "ci": True})

    # Monte Carlo methods
    add("mc_gbm", "Geometric Brownian Motion Monte Carlo simulation",
        [{"name": "n_sims", "type": "int", "description": "Number of simulations (default: 1000)"},
         {"name": "seed", "type": "int", "description": "Random seed (default: 42)"},
         {"name": "mu", "type": "float", "description": "Drift rate (auto-calibrated if omitted)"},
         {"name": "sigma", "type": "float", "description": "Volatility (auto-calibrated if omitted)"}],
        ["numpy", "scipy"],
        {"price": True, "return": True, "volatility": False, "ci": True})

    add("hmm_mc", "Hidden Markov Model Monte Carlo simulation",
        [{"name": "n_states", "type": "int", "description": "Number of regime states (default: 2)"},
         {"name": "n_sims", "type": "int", "description": "Number of simulations (default: 1000)"},
         {"name": "seed", "type": "int", "description": "Random seed (default: 42)"}],
        ["hmmlearn", "numpy"],
        {"price": True, "return": True, "volatility": False, "ci": True})

    # Neural networks (require external dependencies)
    add("nhits", "N-HiTS neural network",
        [{"name": "input_size", "type": "int", "description": "Input context length (auto if omitted)"},
         {"name": "epochs", "type": "int", "description": "Training epochs (default: 100)"},
         {"name": "learning_rate", "type": "float", "description": "Learning rate (default: 0.001)"}],
        ["torch", "pytorch_forecasting"],
        {"price": True, "return": True, "volatility": True, "ci": True})

    add("nbeatsx", "N-BEATS-X neural network with exogenous features",
        [{"name": "input_size", "type": "int", "description": "Input context length (auto if omitted)"},
         {"name": "epochs", "type": "int", "description": "Training epochs (default: 100)"},
         {"name": "learning_rate", "type": "float", "description": "Learning rate (default: 0.001)"}],
        ["torch", "pytorch_forecasting"],
        {"price": True, "return": True, "volatility": True, "ci": True})

    add("tft", "Temporal Fusion Transformer",
        [{"name": "input_size", "type": "int", "description": "Input context length (auto if omitted)"},
         {"name": "epochs", "type": "int", "description": "Training epochs (default: 50)"},
         {"name": "learning_rate", "type": "float", "description": "Learning rate (default: 0.001)"}],
        ["torch", "pytorch_forecasting"],
        {"price": True, "return": True, "volatility": True, "ci": True})

    add("patchtst", "PatchTST transformer",
        [{"name": "input_size", "type": "int", "description": "Input context length (auto if omitted)"},
         {"name": "epochs", "type": "int", "description": "Training epochs (default: 100)"},
         {"name": "learning_rate", "type": "float", "description": "Learning rate (default: 0.001)"}],
        ["torch", "patchtst"],
        {"price": True, "return": True, "volatility": True, "ci": True})

    # StatsForecast methods
    add("sf_autoarima", "StatsForecast AutoARIMA",
        [{"name": "seasonality", "type": "int", "description": "Seasonal period (auto if omitted)"},
         {"name": "alpha", "type": "float", "description": "Significance level for CI (default: 0.05)"}],
        ["statsforecast"],
        {"price": True, "return": True, "volatility": True, "ci": True})

    add("sf_theta", "StatsForecast Theta method",
        [{"name": "seasonality", "type": "int", "description": "Seasonal period (auto if omitted)"},
         {"name": "decomposition_type", "type": "str", "description": "Multiplicative or additive (auto if omitted)"}],
        ["statsforecast"],
        {"price": True, "return": True, "volatility": True, "ci": True})

    add("sf_autoets", "StatsForecast AutoETS",
        [{"name": "seasonality", "type": "int", "description": "Seasonal period (auto if omitted)"},
         {"name": "alpha", "type": "float", "description": "Significance level for CI (default: 0.05)"}],
        ["statsforecast"],
        {"price": True, "return": True, "volatility": True, "ci": True})

    add("sf_seasonalnaive", "StatsForecast Seasonal Naive",
        [{"name": "seasonality", "type": "int", "description": "Seasonal period (auto if omitted)"}],
        ["statsforecast"],
        {"price": True, "return": True, "volatility": True, "ci": True})

    # Machine Learning methods
    add("mlf_rf", "MLForecast Random Forest",
        [{"name": "n_estimators", "type": "int", "description": "Number of trees (default: 100)"},
         {"name": "max_depth", "type": "int", "description": "Maximum depth (default: None)"},
         {"name": "lags", "type": "list", "description": "Lag features to use (auto if omitted)"}],
        ["mlforecast", "scikit-learn"],
        {"price": True, "return": True, "volatility": True, "ci": True})

    add("mlf_lightgbm", "MLForecast LightGBM",
        [{"name": "num_leaves", "type": "int", "description": "Number of leaves (default: 31)"},
         {"name": "learning_rate", "type": "float", "description": "Learning rate (default: 0.1)"},
         {"name": "lags", "type": "list", "description": "Lag features to use (auto if omitted)"}],
        ["mlforecast", "lightgbm"],
        {"price": True, "return": True, "volatility": True, "ci": True})

    # Pre-trained models
    add("chronos_bolt", "Chronos-2 foundation model (alias: chronos2). Successor to Chronos-Bolt; upstream supports cross-learning, multivariate, and covariates—current adapter uses univariate target only.",
        [{"name": "model_name", "type": "str", "description": "Hugging Face model id (default: amazon/chronos-2)"},
         {"name": "context_length", "type": "int", "description": "Context window length (auto if omitted)"},
         {"name": "quantiles", "type": "list", "description": "Quantile levels to return (default: [0.5])"},
         {"name": "device_map", "type": "str", "description": "Device map for loading (default: auto)"}],
        ["chronos-forecasting>=2.0.0", "torch"],
        {"price": True, "return": True, "volatility": True, "ci": True})

    add("chronos2", "Chronos-2 foundation model (preferred name; same implementation as chronos_bolt). Upstream supports cross-learning, multivariate, and covariates—current adapter uses univariate target only.",
        [{"name": "model_name", "type": "str", "description": "Hugging Face model id (default: amazon/chronos-2)"},
         {"name": "context_length", "type": "int", "description": "Context window length (auto if omitted)"},
         {"name": "quantiles", "type": "list", "description": "Quantile levels to return (default: [0.5])"},
         {"name": "device_map", "type": "str", "description": "Device map for loading (default: auto)"}],
        ["chronos-forecasting>=2.0.0", "torch"],
        {"price": True, "return": True, "volatility": True, "ci": True})

    add("timesfm", "Google TimesFM pre-trained time series foundation model",
        [{"name": "device", "type": "str", "description": "Compute device (cpu/cuda, default: auto)"},
         {"name": "batch_size", "type": "int", "description": "Batch size (default: 16)"}],
        ["timesfm", "torch"],
        {"price": True, "return": True, "volatility": True, "ci": True})

    add("lag_llama", "Lag-Llama pre-trained time series model",
        [{"name": "device", "type": "str", "description": "Compute device (cpu/cuda, default: auto)"},
         {"name": "batch_size", "type": "int", "description": "Batch size (default: 1)"}],
        ["lag-llama", "gluonts", "torch"],
        {"price": True, "return": True, "volatility": True, "ci": True})

    # GluonTS Torch quick-train models
    add("gt_deepar", "GluonTS DeepAR (quick train on series)",
        [{"name": "context_length", "type": "int", "description": "Input window length (default: min(64,n))"},
         {"name": "train_epochs", "type": "int", "description": "Training epochs (default: 5)"},
         {"name": "batch_size", "type": "int", "description": "Batch size (default: 32)"},
         {"name": "learning_rate", "type": "float", "description": "Learning rate (default: 1e-3)"},
         {"name": "freq", "type": "str", "description": "Pandas frequency string (auto from timeframe)"}],
        ["gluonts", "torch"],
        {"price": True, "return": True, "volatility": True, "ci": True})

    add("gt_sfeedforward", "GluonTS SimpleFeedForward (quick train)",
        [{"name": "context_length", "type": "int", "description": "Input window length (default: min(64,n))"},
         {"name": "train_epochs", "type": "int", "description": "Training epochs (default: 5)"},
         {"name": "batch_size", "type": "int", "description": "Batch size (default: 32)"},
         {"name": "learning_rate", "type": "float", "description": "Learning rate (default: 1e-3)"},
         {"name": "freq", "type": "str", "description": "Pandas frequency string (auto from timeframe)"}],
        ["gluonts", "torch"],
        {"price": True, "return": True, "volatility": True, "ci": True})

    add("gt_prophet", "GluonTS Prophet wrapper",
        [{"name": "freq", "type": "str", "description": "Pandas frequency string (auto from timeframe)"},
         {"name": "prophet_params", "type": "dict", "description": "Passed to ProphetPredictor (growth, seasonality_mode, ... )"}],
        ["gluonts", "prophet"],
        {"price": True, "return": True, "volatility": True, "ci": True})

    add("gt_tft", "GluonTS Temporal Fusion Transformer (quick train)",
        [{"name": "context_length", "type": "int", "description": "Input window length (default: min(128,n))"},
         {"name": "train_epochs", "type": "int", "description": "Training epochs (default: 5)"},
         {"name": "batch_size", "type": "int", "description": "Batch size (default: 32)"},
         {"name": "learning_rate", "type": "float", "description": "Learning rate (default: 1e-3)"},
         {"name": "hidden_size", "type": "int", "description": "Model width (default: 64)"},
         {"name": "dropout", "type": "float", "description": "Dropout (default: 0.1)"},
         {"name": "freq", "type": "str", "description": "Pandas frequency (auto from timeframe)"}],
        ["gluonts", "torch"],
        {"price": True, "return": True, "volatility": True, "ci": True})

    add("gt_wavenet", "GluonTS WaveNet (quick train)",
        [{"name": "context_length", "type": "int", "description": "Input window length (default: min(128,n))"},
         {"name": "train_epochs", "type": "int", "description": "Training epochs (default: 5)"},
         {"name": "batch_size", "type": "int", "description": "Batch size (default: 32)"},
         {"name": "learning_rate", "type": "float", "description": "Learning rate (default: 1e-3)"},
         {"name": "dilation_depth", "type": "int", "description": "Dilation depth (default: 5)"},
         {"name": "num_blocks", "type": "int", "description": "WaveNet blocks (default: 1)"},
         {"name": "freq", "type": "str", "description": "Pandas frequency (auto from timeframe)"}],
        ["gluonts", "torch"],
        {"price": True, "return": True, "volatility": True, "ci": True})

    add("gt_deepnpts", "GluonTS DeepNPTS (quick train)",
        [{"name": "context_length", "type": "int", "description": "Input window length (default: min(128,n))"},
         {"name": "train_epochs", "type": "int", "description": "Training epochs (default: 5)"},
         {"name": "batch_size", "type": "int", "description": "Batch size (default: 32)"},
         {"name": "learning_rate", "type": "float", "description": "Learning rate (default: 1e-3)"},
         {"name": "freq", "type": "str", "description": "Pandas frequency (auto from timeframe)"}],
        ["gluonts", "torch"],
        {"price": True, "return": True, "volatility": True, "ci": True})

    add("gt_mqf2", "GluonTS MQF2 (quick train, quantile-focused)",
        [{"name": "context_length", "type": "int", "description": "Input window length (default: min(128,n))"},
         {"name": "train_epochs", "type": "int", "description": "Training epochs (default: 5)"},
         {"name": "batch_size", "type": "int", "description": "Batch size (default: 32)"},
         {"name": "learning_rate", "type": "float", "description": "Learning rate (default: 1e-3)"},
         {"name": "freq", "type": "str", "description": "Pandas frequency (auto from timeframe)"},
         {"name": "quantiles", "type": "list", "description": "Quantiles to return (e.g., [0.05,0.5,0.95])"}],
        ["gluonts", "torch"],
        {"price": True, "return": True, "volatility": True, "ci": True})

    add("gt_npts", "GluonTS NPTS (non-parametric, fast)",
        [{"name": "freq", "type": "str", "description": "Pandas frequency string (auto from timeframe)"},
         {"name": "season_length", "type": "int", "description": "Season length (default: 1)"},
         {"name": "kernel", "type": "str", "description": "Kernel: parzen|mean|median (default: parzen)"},
         {"name": "window_size", "type": "int", "description": "Window size (default: min(256,n))"}],
        ["gluonts"],
        {"price": True, "return": True, "volatility": True, "ci": True})

    # Ensemble methods
    add("ensemble", "Adaptive ensemble with averaging, Bayesian model averaging, or stacking",
        [{"name": "methods", "type": "list", "description": "Methods to ensemble (default: naive,theta,fourier_ols)"},
         {"name": "mode", "type": "str", "description": "average|bma|stacking (default: average)"},
         {"name": "weights", "type": "list", "description": "Manual weights when mode=average"},
         {"name": "cv_points", "type": "int", "description": "Walk-forward anchors for weighting (default: 2*len(methods))"},
         {"name": "min_train_size", "type": "int", "description": "Minimum history per CV anchor (default: max(30, horizon*3))"},
         {"name": "method_params", "type": "dict", "description": "Per-method parameter overrides"},
         {"name": "expose_components", "type": "bool", "description": "Include component forecasts in response (default: True)"}],
        [],
        {"price": True, "return": True, "volatility": True, "ci": False})

    # Analog / Nearest Neighbor
    add("analog", "Nearest-neighbor search based on historical patterns",
        [{"name": "window_size", "type": "int", "description": "Length of pattern to match (default: 64)"},
         {"name": "search_depth", "type": "int", "description": "Bars back to search (default: 5000)"},
         {"name": "top_k", "type": "int", "description": "Number of analogs (default: 20)"},
         {"name": "metric", "type": "str", "description": "Similarity metric: euclidean|cosine|correlation (default: euclidean)"},
         {"name": "scale", "type": "str", "description": "zscore|minmax|none (default: zscore)"},
         {"name": "refine_metric", "type": "str", "description": "dtw|softdtw|affine|ncc|none (default: dtw)"},
         {"name": "search_engine", "type": "str", "description": "ckdtree|hnsw|matrix_profile|mass (default: ckdtree)"},
         {"name": "secondary_timeframes", "type": "str|list", "description": "List of timeframes to ensemble (e.g. 'D1,H4')"}],
        ["scipy", "numpy"],
        {"price": True, "return": False, "volatility": False, "ci": True})

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
            "pretrained": ["chronos_bolt", "chronos2", "timesfm", "lag_llama"],
            "ensemble": ["ensemble"]
        }
    }


def get_method_category(method: str) -> str:
    """Get the category of a forecast method."""
    categories = {
        "naive": "classical", "drift": "classical", "seasonal_naive": "classical",
        "theta": "classical", "fourier_ols": "classical",
        "ses": "ets", "holt": "ets", "holt_winters_add": "ets", "holt_winters_mul": "ets",
        "arima": "arima", "sarima": "arima",
        "mc_gbm": "monte_carlo", "hmm_mc": "monte_carlo",
        "nhits": "neural", "nbeatsx": "neural", "tft": "neural", "patchtst": "neural",
        "sf_autoarima": "statsforecast", "sf_theta": "statsforecast",
        "sf_autoets": "statsforecast", "sf_seasonalnaive": "statsforecast",
        "mlf_rf": "machine_learning", "mlf_lightgbm": "machine_learning",
        "chronos_bolt": "pretrained", "chronos2": "pretrained", "timesfm": "pretrained", "lag_llama": "pretrained",
        "ensemble": "ensemble",
        "analog": "analog"
    }
    return categories.get(method, "unknown")


def get_method_requirements(method: str) -> List[str]:
    """Get the list of required packages for a method."""
    method_data = get_forecast_methods_data()
    for m in method_data["methods"]:
        if m["method"] == method:
            return m["requires"]
    return []


def get_method_supports(method: str) -> Dict[str, bool]:
    """Get the supported data types and features for a method."""
    method_data = get_forecast_methods_data()
    for m in method_data["methods"]:
        if m["method"] == method:
            return m["supports"]
    return {"price": False, "return": False, "volatility": False, "ci": False}


def validate_method_params(method: str, params: Dict[str, Any]) -> List[str]:
    """Validate method parameters and return list of errors."""
    errors = []

    # Get method definition
    method_data = get_forecast_methods_data()
    method_def = None
    for m in method_data["methods"]:
        if m["method"] == method:
            method_def = m
            break

    if not method_def:
        errors.append(f"Unknown method: {method}")
        return errors

    # Check parameter types
    for param_def in method_def["params"]:
        param_name = param_def["name"]
        param_type = param_def["type"]

        if param_name in params:
            param_value = params[param_name]

            # Type validation
            if param_type == "int":
                try:
                    int(param_value)
                except (ValueError, TypeError):
                    errors.append(f"Parameter '{param_name}' should be an integer")
            elif param_type == "float":
                try:
                    float(param_value)
                except (ValueError, TypeError):
                    errors.append(f"Parameter '{param_name}' should be a float")
            elif param_type == "bool":
                if not isinstance(param_value, bool):
                    errors.append(f"Parameter '{param_name}' should be a boolean")
            elif param_type == "tuple":
                if not isinstance(param_value, (list, tuple)):
                    errors.append(f"Parameter '{param_name}' should be a tuple or list")
                elif len(param_value) != 3:  # Assuming (p,d,q) order
                    errors.append(f"Parameter '{param_name}' should have 3 elements")

    return errors
