"""
Forecast method registry and metadata management.

Centralizes method definitions, requirements, and availability checking.
"""

import importlib as _importlib
import importlib.util as _importlib_util
import sys
from functools import lru_cache
from typing import Any, Dict, List, Tuple, Type

from .interface import ForecastMethod


class ForecastRegistry:
    """Registry for forecasting methods."""

    _methods: Dict[str, Type[ForecastMethod]] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register a forecast method class."""
        def decorator(method_cls: Type[ForecastMethod]):
            cls._methods[name] = method_cls
            return method_cls
        return decorator

    @classmethod
    def get(cls, name: str) -> ForecastMethod:
        """Get an instance of a registered forecast method."""
        if name not in cls._methods:
            raise ValueError(f"Unknown method: {name}")
        return cls._methods[name]()

    @classmethod
    def list_available(cls) -> List[str]:
        """List names of all registered methods."""
        return list(cls._methods)

    @classmethod
    def get_class(cls, name: str) -> Type[ForecastMethod]:
        """Get the class of a registered forecast method."""
        if name not in cls._methods:
            raise ValueError(f"Unknown method: {name}")
        return cls._methods[name]

    @classmethod
    def get_all_method_names(cls) -> List[str]:
        """Get all available forecast method names from the registered classes."""
        return sorted(cls._methods.keys())

    @classmethod
    def get_method_info(cls, name: str) -> Dict[str, Any]:
        """Return capability metadata for a single registered method."""
        inst = cls.get(name)
        return {
            "name": name,
            "category": inst.category,
            "supports_training": inst.supports_training,
            "train_supports_cancel": inst.train_supports_cancel,
            "train_supports_progress": inst.train_supports_progress,
            "training_category": inst.training_category,
        }

    @classmethod
    def list_trainable(cls) -> List[str]:
        """Return names of methods that support the train/predict lifecycle."""
        return [
            name for name in cls._methods
            if cls._methods[name]().supports_training
        ]

DEFAULT_METHOD_SUPPORTS: Dict[str, bool] = {
    "price": True,
    "return": True,
    "volatility": False,
    "ci": False,
}
METHOD_DESCRIPTIONS: Dict[str, str] = {
    "arima": "ARIMA statistical model for autoregressive, differenced, moving-average forecasts.",
    "sarima": "Seasonal ARIMA model for trend, autocorrelation, and repeating seasonal structure.",
    "drift": "Linear drift baseline that extrapolates the historical start-to-end slope.",
    "fourier_ols": "Fourier regression with optional trend for smooth seasonal cycles.",
    "naive": "Naive baseline that repeats the latest observed value.",
    "seasonal_naive": "Seasonal naive baseline that repeats the last observed seasonal cycle.",
    "theta": "Classical Theta method combining linear trend and exponential smoothing.",
    "holt": "Holt exponential smoothing with level and trend components.",
    "holt_winters_add": "Holt-Winters exponential smoothing with additive seasonality.",
    "holt_winters_mul": "Holt-Winters exponential smoothing with multiplicative seasonality.",
    "ses": "Simple exponential smoothing for level-only series without explicit trend.",
    "mlf_lightgbm": "MLForecast adapter using LightGBM regressors with lag features.",
    "mlf_rf": "MLForecast adapter using random forests with lag features.",
    "hmm_mc": "Monte Carlo simulation with regime transitions estimated by an HMM.",
    "mc_gbm": "Geometric Brownian motion Monte Carlo simulation for price paths.",
    "nbeatsx": "NeuralForecast NBEATSx deep model for exogenous-aware time-series forecasts.",
    "nhits": "NeuralForecast NHITS deep model for hierarchical interpolation forecasts.",
    "patchtst": "NeuralForecast PatchTST transformer model for patched time-series windows.",
    "tft": "NeuralForecast Temporal Fusion Transformer for multivariate sequence forecasts.",
    "chronos2": "Chronos-2 pretrained foundation model for probabilistic time-series forecasts.",
    "chronos_bolt": "Chronos Bolt pretrained foundation model for fast time-series forecasts.",
    "timesfm": "TimesFM pretrained foundation model for long-context time-series forecasts.",
    "lag_llama": "Lag-Llama pretrained probabilistic model for univariate time-series forecasts.",
    "gt_deepar": "GluonTS DeepAR recurrent probabilistic forecasting model.",
    "gt_deepnpts": "GluonTS DeepNPTS nonparametric probabilistic forecasting model.",
    "gt_mqf2": "GluonTS MQF2 multivariate quantile forecasting model.",
    "gt_npts": "GluonTS NPTS nonparametric probabilistic forecasting model.",
    "gt_prophet": "GluonTS Prophet adapter for trend and seasonality forecasting.",
    "gt_sfeedforward": "GluonTS simple feed-forward neural forecasting model.",
    "gt_tft": "GluonTS Temporal Fusion Transformer forecasting model.",
    "gt_wavenet": "GluonTS WaveNet sequence model for probabilistic forecasting.",
}

_FORECAST_METHOD_MODULES = (
    "classical",
    "ets_arima",
    "statsforecast",
    "mlforecast",
    "pretrained",
    "neural",
    "sktime",
    "gluonts_extra",
    "analog",
    "ensemble",
    "monte_carlo",
)
_PYTHON_314_PLUS = sys.version_info >= (3, 14)
_GLUONTS_EXTRA_METHODS = {
    "gt_deepar",
    "gt_sfeedforward",
    "gt_prophet",
    "gt_tft",
    "gt_wavenet",
    "gt_deepnpts",
    "gt_mqf2",
    "gt_npts",
}
_GLUONTS_PYTHON_RUNTIME_REQUIREMENT = "Python < 3.14 (GluonTS methods are unsupported in this project)"
_OPTIONAL_FORECAST_METHOD_MODULES = frozenset(
    {
        "statsforecast",
        "mlforecast",
        "pretrained",
        "neural",
        "sktime",
        "gluonts_extra",
    }
)
_LOADED_FORECAST_METHOD_MODULES: set[str] = set()
_FAILED_OPTIONAL_FORECAST_MODULES: Dict[str, str] = {}

# Import availability checkers
try:
    _SM_ETS_AVAILABLE = _importlib_util.find_spec("statsmodels.tsa.holtwinters") is not None
except Exception:
    _SM_ETS_AVAILABLE = False

try:
    _SM_SARIMAX_AVAILABLE = _importlib_util.find_spec("statsmodels.tsa.statespace.sarimax") is not None
except Exception:
    _SM_SARIMAX_AVAILABLE = False

try:
    _NF_AVAILABLE = _importlib_util.find_spec("neuralforecast") is not None
    _MLF_AVAILABLE = _importlib_util.find_spec("mlforecast") is not None
    _SF_AVAILABLE = _importlib_util.find_spec("statsforecast") is not None
    _LGB_AVAILABLE = _importlib_util.find_spec("lightgbm") is not None
    _CHRONOS_AVAILABLE = _importlib_util.find_spec("chronos") is not None
    _TIMESFM_AVAILABLE = _importlib_util.find_spec("timesfm") is not None
    _LAG_LLAMA_AVAILABLE = _importlib_util.find_spec("lag_llama") is not None
    _SKTIME_AVAILABLE = _importlib_util.find_spec("sktime") is not None
except Exception:
    _NF_AVAILABLE = False
    _MLF_AVAILABLE = False
    _SF_AVAILABLE = False
    _LGB_AVAILABLE = False
    _CHRONOS_AVAILABLE = False
    _TIMESFM_AVAILABLE = False
    _LAG_LLAMA_AVAILABLE = False
    _SKTIME_AVAILABLE = False


def _find_method_definition(
    method: str,
    method_data: Dict[str, Any] | None = None,
) -> Dict[str, Any] | None:
    data = method_data if isinstance(method_data, dict) else get_forecast_methods_data()
    methods = data.get("methods") if isinstance(data, dict) else None
    if not isinstance(methods, list):
        return None
    for method_def in methods:
        if isinstance(method_def, dict) and method_def.get("method") == method:
            return method_def
    return None


def _build_forecast_methods_snapshot() -> Tuple[List[Dict[str, Any]], Dict[str, List[str]]]:
    methods: List[Dict[str, Any]] = []
    categories: Dict[str, List[str]] = {}

    for method in ForecastRegistry.get_all_method_names():
        try:
            cls = ForecastRegistry.get_class(method)
            inst = cls()
        except Exception:
            if method == "ensemble":
                entry = _ensemble_metadata()
                methods.append(entry)
                categories.setdefault("ensemble", []).append(method)
            continue

        supports = inst.supports_features or dict(DEFAULT_METHOD_SUPPORTS)
        requires = list(getattr(inst, "required_packages", []) or [])
        params = getattr(inst, "PARAMS", None)
        if not isinstance(params, list):
            params = []
        desc = _extract_description(cls, method)
        available, reqs = _check_requirements(method, requires)
        cat = str(getattr(inst, "category", "unknown") or "unknown").lower()

        entry = {
            "method": method,
            "category": cat,
            "available": bool(available),
            "requires": sorted(set(reqs)),
            "description": desc,
            "params": params,
            "supports": supports,
        }
        methods.append(entry)
        categories.setdefault(cat, []).append(method)

    return methods, categories


def get_forecast_methods_data() -> Dict[str, Any]:
    """Return metadata about available forecast methods and their requirements.

    This is derived from ForecastRegistry to avoid drift.
    """
    _ensure_registry_loaded()
    methods, categories = _build_forecast_methods_snapshot()

    return {
        "methods": methods,
        "total": len(methods),
        "categories": categories,
    }


def get_forecast_method_availability_snapshot() -> Dict[str, bool]:
    """Return method availability derived from the registry-backed method snapshot."""
    _ensure_registry_loaded()
    methods, _ = _build_forecast_methods_snapshot()
    return {
        str(method_def.get("method")): bool(method_def.get("available"))
        for method_def in methods
        if isinstance(method_def, dict) and method_def.get("method")
    }




def _extract_description(cls: Any, fallback: str) -> str:
    doc = getattr(cls, "__doc__", None)
    if isinstance(doc, str):
        line = doc.strip().splitlines()
        if line and line[0].strip():
            return line[0].strip()
    fallback_text = str(fallback)
    return METHOD_DESCRIPTIONS.get(fallback_text, fallback_text)


@lru_cache(maxsize=1)
def _check_chronos_runtime_support() -> Tuple[bool, List[str]]:
    """Verify Chronos exposes the public pipeline APIs required by our adapters."""
    reqs: List[str] = []
    try:
        chronos_mod = _importlib.import_module("chronos")
    except Exception:
        return False, ["chronos-forecasting"]

    top_level_pipelines = ("Chronos2Pipeline", "ChronosBoltPipeline", "ChronosPipeline")
    if any(hasattr(chronos_mod, attr) for attr in top_level_pipelines):
        return True, reqs

    # Newer Chronos-2 builds may expose the pipeline only from chronos.chronos2.
    try:
        chronos2_mod = _importlib.import_module("chronos.chronos2")
    except Exception:
        reqs.append("chronos pipeline API")
        return False, reqs

    if not hasattr(chronos2_mod, "Chronos2Pipeline"):
        reqs.append("chronos pipeline API")

    return len(reqs) == 0, reqs


def _check_requirements(method: str, requires: List[str]) -> Tuple[bool, List[str]]:
    available = True
    reqs = list(requires or [])

    # Check availability based on method type and runtime flags.
    if method in ("ses", "holt", "holt_winters_add", "holt_winters_mul", "ets") and not _SM_ETS_AVAILABLE:
        available = False; reqs.append("statsmodels")
    if method in ("arima", "sarima") and not _SM_SARIMAX_AVAILABLE:
        available = False; reqs.append("statsmodels")
    if method == "statsforecast" and not _SF_AVAILABLE:
        available = False; reqs.append("statsforecast")
    if method == "mlforecast" and not _MLF_AVAILABLE:
        available = False; reqs.append("mlforecast")
    if method == "mlf_rf" and not _MLF_AVAILABLE:
        available = False; reqs.append("mlforecast, scikit-learn")
    if method == "mlf_lightgbm" and (not _MLF_AVAILABLE or not _LGB_AVAILABLE):
        available = False; reqs.append("mlforecast, lightgbm")
    if method in ("chronos_bolt", "chronos2"):
        if not _CHRONOS_AVAILABLE:
            available = False; reqs.append("chronos-forecasting")
        else:
            chronos_ok, chronos_reqs = _check_chronos_runtime_support()
            if not chronos_ok:
                available = False
                reqs.extend(chronos_reqs)
    if method == "timesfm" and not _TIMESFM_AVAILABLE:
        available = False; reqs.append("timesfm")
    if method == "lag_llama" and not _LAG_LLAMA_AVAILABLE:
        available = False; reqs.append("lag-llama, gluonts, torch")
    if method == "sktime" and not _SKTIME_AVAILABLE:
        available = False; reqs.append("sktime")
    if method in _GLUONTS_EXTRA_METHODS and _PYTHON_314_PLUS:
        available = False
        reqs.append(_GLUONTS_PYTHON_RUNTIME_REQUIREMENT)

    module_name_overrides = {
        "scikit-learn": "sklearn",
        "lag-llama": "lag_llama",
        "chronos-forecasting": "chronos",
        "chronos-forecasting>=2.0.0": "chronos",
        "python-dotenv": "dotenv",
    }
    for req in list(reqs):
        name = str(req).strip()
        if not name:
            continue
        if name.lower().startswith("python "):
            continue
        for sep in (">=", "==", "<=", "~=", ">", "<"):
            if sep in name:
                name = name.split(sep, 1)[0].strip()
                break
        name = module_name_overrides.get(name, name)
        try:
            if _importlib_util.find_spec(name) is None:
                available = False
        except Exception:
            available = False

    return available, reqs


def _ensure_registry_loaded() -> None:
    """Ensure ForecastRegistry is populated by importing method modules."""
    base_package = f"{__package__}.methods"
    for module_name in _FORECAST_METHOD_MODULES:
        if module_name in _LOADED_FORECAST_METHOD_MODULES:
            continue
        if module_name in _FAILED_OPTIONAL_FORECAST_MODULES:
            continue
        try:
            _importlib.import_module(f"{base_package}.{module_name}")
        except ModuleNotFoundError as exc:
            if module_name not in _OPTIONAL_FORECAST_METHOD_MODULES:
                raise
            _FAILED_OPTIONAL_FORECAST_MODULES[module_name] = str(exc)
            continue
        except ImportError as exc:
            if module_name not in _OPTIONAL_FORECAST_METHOD_MODULES:
                raise
            _FAILED_OPTIONAL_FORECAST_MODULES[module_name] = str(exc)
            continue
        _LOADED_FORECAST_METHOD_MODULES.add(module_name)


def _ensemble_metadata() -> Dict[str, Any]:
    return {
        "method": "ensemble",
        "category": "ensemble",
        "available": True,
        "requires": [],
        "description": "Adaptive ensemble with averaging, Bayesian model averaging, or stacking.",
        "params": [
            {"name": "methods", "type": "list", "description": "Methods to ensemble (default: naive,theta,fourier_ols)"},
            {"name": "mode", "type": "str", "description": "average|bma|stacking (default: average)"},
            {"name": "weights", "type": "list", "description": "Manual weights when mode=average"},
            {"name": "cv_points", "type": "int", "description": "Walk-forward anchors for weighting (default: 2*len(methods))"},
            {"name": "min_train_size", "type": "int", "description": "Minimum history per CV anchor (default: max(30, horizon*3))"},
            {"name": "method_params", "type": "dict", "description": "Per-method parameter overrides"},
            {"name": "expose_components", "type": "bool", "description": "Include component forecasts in response (default: True)"},
        ],
        "supports": {"price": True, "return": True, "volatility": True, "ci": False},
    }


# Availability flags that can be imported by other modules
__all__ = [
    'ForecastRegistry',
    'get_forecast_methods_data',
    'get_forecast_method_availability_snapshot',
    '_SM_ETS_AVAILABLE',
    '_SM_SARIMAX_AVAILABLE',
    '_NF_AVAILABLE',
    '_MLF_AVAILABLE',
    '_SF_AVAILABLE',
    '_LGB_AVAILABLE',
    '_CHRONOS_AVAILABLE',
    '_TIMESFM_AVAILABLE',
    '_LAG_LLAMA_AVAILABLE',
]
