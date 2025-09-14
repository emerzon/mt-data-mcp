from typing import Any, Dict, List


def get_forecast_methods_data(
    _SM_ETS_AVAILABLE: bool,
    _SM_SARIMAX_AVAILABLE: bool,
    _NF_AVAILABLE: bool = False,
    _SF_AVAILABLE: bool = False,
    _MLF_AVAILABLE: bool = False,
    _CHRONOS_AVAILABLE: bool = False,
    _TIMESFM_AVAILABLE: bool = False,
    _LAG_LLAMA_AVAILABLE: bool = False,
    _ARCH_AVAILABLE: bool = False,
) -> Dict[str, Any]:
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

    add("naive", True, "Repeat last observed value (random walk baseline).", [], {**common_defaults, "quantity": "price"})
    add("drift", True, "Linear drift from first to last observation; strong simple baseline.", [], {**common_defaults, "quantity": "price"})
    add("seasonal_naive", True, "Repeat last season's values; requires seasonality period m.", [
        {"name": "seasonality", "type": "int", "default": None, "description": "Season period m; auto from timeframe if omitted."},
    ], {**common_defaults, "quantity": "price"})
    add("theta", True, "Fast Theta-style: average of linear trend extrapolation and SES level.", [
        {"name": "alpha", "type": "float", "default": 0.2, "description": "SES smoothing for level component."},
    ], {**common_defaults, "quantity": "price"})
    add("fourier_ols", True, "Fourier regression with K harmonics and optional linear trend.", [
        {"name": "seasonality", "type": "int", "default": None, "description": "Season period m; auto by timeframe if omitted."},
        {"name": "K", "type": "int", "default": "min(3, m/2)", "description": "Number of harmonics."},
        {"name": "trend", "type": "bool", "default": True, "description": "Include linear trend term."},
    ], {**common_defaults, "quantity": "price"})
    add("ses", _SM_ETS_AVAILABLE, "Simple Exponential Smoothing (statsmodels).", [
        {"name": "alpha", "type": "float", "default": None, "description": "Smoothing level; optimized if None."},
    ], {**common_defaults, "quantity": "price"})
    add("holt", _SM_ETS_AVAILABLE, "Holt's linear trend with optional damping (statsmodels).", [
        {"name": "damped", "type": "bool", "default": True, "description": "Use damped trend."},
    ], {**common_defaults, "quantity": "price"})
    add("holt_winters_add", _SM_ETS_AVAILABLE, "Additive Holt-Winters with additive seasonality (statsmodels).", [
        {"name": "seasonality", "type": "int", "default": None, "description": "Season period m; required."},
    ], {**common_defaults, "quantity": "price"})
    add("holt_winters_mul", _SM_ETS_AVAILABLE, "Additive trend with multiplicative seasonality (statsmodels).", [
        {"name": "seasonality", "type": "int", "default": None, "description": "Season period m; required."},
    ], {**common_defaults, "quantity": "price"})
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

    # NeuralForecast (optional, heavy)
    nf_common_params = [
        {"name": "max_epochs", "type": "int", "default": 50, "description": "Training epochs (keep small for responsiveness)."},
        {"name": "input_size", "type": "int", "default": None, "description": "Insample window length; auto from data if omitted."},
        {"name": "batch_size", "type": "int", "default": 32, "description": "Mini-batch size."},
        {"name": "learning_rate", "type": "float", "default": None, "description": "Optimizer LR; framework default if None."},
        {"name": "model_params", "type": "dict", "default": {}, "description": "Advanced model constructor kwargs (passed through)."},
    ]
    add("nhits", _NF_AVAILABLE, "Deep learning NHITS model (NeuralForecast).", nf_common_params, {**common_defaults, "quantity": "price"})
    add("nbeatsx", _NF_AVAILABLE, "NBEATSx: Explanatory N-BEATS variant (NeuralForecast).", nf_common_params, {**common_defaults, "quantity": "price"})
    add("tft", _NF_AVAILABLE, "Temporal Fusion Transformer (NeuralForecast).", nf_common_params, {**common_defaults, "quantity": "price"})
    add("patchtst", _NF_AVAILABLE, "PatchTST transformer for long horizons (NeuralForecast).", nf_common_params, {**common_defaults, "quantity": "price"})

    # One-shot foundation models (optional)
    add("chronos_bolt", _CHRONOS_AVAILABLE, "One-shot pretrained foundation model (Amazon Chronos-Bolt).", [
        {"name": "model_name", "type": "str", "default": "amazon/chronos-bolt-base", "description": "HF model repo id."},
        {"name": "context_length", "type": "int", "default": None, "description": "Number of tail points to feed (auto if None)."},
        {"name": "device", "type": "str", "default": None, "description": "Device index or name (e.g., 'cpu', 'cuda:0'). Overrides device_map if set."},
        {"name": "device_map", "type": "str", "default": "auto", "description": "Accelerate device map (e.g., 'auto')."},
        {"name": "quantization", "type": "str", "default": None, "description": "Quantization preset if supported: 'int8'|'int4'."},
        {"name": "quantiles", "type": "list[float]", "default": None, "description": "Optional quantiles to predict (e.g., [0.05,0.5,0.95])."},
        {"name": "revision", "type": "str", "default": None, "description": "HF repo revision (branch/tag/commit)."},
        {"name": "trust_remote_code", "type": "bool", "default": False, "description": "Allow custom code from repo (HF trust_remote_code)."},
    ], {**common_defaults, "quantity": "price"})
    add("timesfm", _TIMESFM_AVAILABLE, "One-shot pretrained foundation model (Google TimesFM).", [
        {"name": "model_name", "type": "str", "default": "google/timesfm-1.0-200m", "description": "HF model repo id."},
        {"name": "context_length", "type": "int", "default": None, "description": "Number of tail points to feed (auto if None)."},
        {"name": "device", "type": "str", "default": None, "description": "Device index or name (e.g., 'cpu', 'cuda:0'). Overrides device_map if set."},
        {"name": "device_map", "type": "str", "default": "auto", "description": "Accelerate device map (e.g., 'auto')."},
        {"name": "quantization", "type": "str", "default": None, "description": "Quantization preset if supported: 'int8'|'int4'."},
        {"name": "quantiles", "type": "list[float]", "default": None, "description": "Optional quantiles to predict (e.g., [0.05,0.5,0.95])."},
        {"name": "revision", "type": "str", "default": None, "description": "HF repo revision (branch/tag/commit)."},
        {"name": "trust_remote_code", "type": "bool", "default": False, "description": "Allow custom code from repo (HF trust_remote_code)."},
    ], {**common_defaults, "quantity": "price"})
    add("lag_llama", _LAG_LLAMA_AVAILABLE, "One-shot pretrained foundation model (Lag-Llama).", [
        {"name": "model_name", "type": "str", "default": None, "description": "HF model repo id (required)."},
        {"name": "context_length", "type": "int", "default": None, "description": "Number of tail points to feed (auto if None)."},
        {"name": "device", "type": "str", "default": None, "description": "Device index or name (e.g., 'cpu', 'cuda:0'). Overrides device_map if set."},
        {"name": "device_map", "type": "str", "default": "auto", "description": "Accelerate device map (e.g., 'auto')."},
        {"name": "quantization", "type": "str", "default": None, "description": "Quantization preset if supported: 'int8'|'int4'."},
        {"name": "quantiles", "type": "list[float]", "default": None, "description": "Optional quantiles to predict (e.g., [0.05,0.5,0.95])."},
        {"name": "revision", "type": "str", "default": None, "description": "HF repo revision (branch/tag/commit)."},
        {"name": "trust_remote_code", "type": "bool", "default": False, "description": "Allow custom code from repo (HF trust_remote_code)."},
    ], common_defaults)

    # StatsForecast (optional classical, fast)
    add("sf_autoarima", _SF_AVAILABLE, "AutoARIMA via StatsForecast (fast, numba).", [
        {"name": "seasonality", "type": "int", "default": None, "description": "Season period m; auto by timeframe if omitted."},
        {"name": "stepwise", "type": "bool", "default": True, "description": "Use stepwise search (faster)."},
        {"name": "d", "type": "int", "default": None, "description": "Non-seasonal differencing order (auto if None)."},
        {"name": "D", "type": "int", "default": None, "description": "Seasonal differencing order (auto if None)."},
    ], common_defaults)
    add("sf_theta", _SF_AVAILABLE, "Theta model via StatsForecast.", [
        {"name": "seasonality", "type": "int", "default": None, "description": "Season period m; auto by timeframe if omitted."},
    ], common_defaults)
    add("sf_autoets", _SF_AVAILABLE, "AutoETS (exponential smoothing) via StatsForecast.", [
        {"name": "seasonality", "type": "int", "default": None, "description": "Season period m; auto by timeframe if omitted."},
    ], common_defaults)
    add("sf_seasonalnaive", _SF_AVAILABLE, "Seasonal Naive via StatsForecast.", [
        {"name": "seasonality", "type": "int", "default": None, "description": "Season period m; auto by timeframe if omitted."},
    ], common_defaults)

    # MLForecast (optional gradient-boosted / tree models over lags)
    add("mlf_rf", _MLF_AVAILABLE, "MLForecast with sklearn RandomForestRegressor over lag features.", [
        {"name": "lags", "type": "list[int]", "default": "auto", "description": "Lag indices; auto based on timeframe if omitted."},
        {"name": "rolling_agg", "type": "str", "default": "mean", "description": "Rolling stat for lag windows: 'mean'|'min'|'max'|'std'|None."},
        {"name": "n_estimators", "type": "int", "default": 200, "description": "RF trees."},
        {"name": "max_depth", "type": "int", "default": None, "description": "RF max depth."},
    ], {**common_defaults, "quantity": "price"})
    add("mlf_lightgbm", _MLF_AVAILABLE, "MLForecast with LightGBM regressor over lag features.", [
        {"name": "lags", "type": "list[int]", "default": "auto", "description": "Lag indices; auto based on timeframe if omitted."},
        {"name": "rolling_agg", "type": "str", "default": "mean", "description": "Rolling stat for lag windows: 'mean'|'min'|'max'|'std'|None."},
        {"name": "n_estimators", "type": "int", "default": 200, "description": "Boosting rounds."},
        {"name": "learning_rate", "type": "float", "default": 0.05, "description": "LightGBM learning rate."},
        {"name": "num_leaves", "type": "int", "default": 31, "description": "LightGBM num_leaves."},
        {"name": "max_depth", "type": "int", "default": -1, "description": "LightGBM max_depth (-1 for no limit)."},
    ], {**common_defaults, "quantity": "price"})

    # Volatility methods are listed via forecast_volatility (see docs). Price/return methods are listed here.

    # Ensemble meta-method
    add("ensemble", True, "Aggregate multiple base forecasts (mean/median/weighted).", [
        {"name": "methods", "type": "list[str]", "default": "['theta','fourier_ols','holt'] (if available)", "description": "Base methods to combine; 'ensemble' is not allowed."},
        {"name": "aggregator", "type": "str", "default": "mean", "description": "Aggregation: 'mean', 'median', or 'weighted'."},
        {"name": "weights", "type": "list[float]", "default": None, "description": "Optional weights aligned with methods; auto-normalized."},
        {"name": "expose_components", "type": "bool", "default": True, "description": "Include per-method forecasts and weights in output."},
    ], common_defaults)

    return {"success": True, "schema_version": 1, "methods": methods}
