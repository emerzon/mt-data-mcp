"""
Forecast method registry and metadata management.

Centralizes method definitions, requirements, and availability checking.
"""

from typing import Any, Dict, List, Tuple
from functools import lru_cache
import importlib as _importlib
import importlib.util as _importlib_util

from .registry import ForecastRegistry

DEFAULT_METHOD_SUPPORTS: Dict[str, bool] = {
    "price": True,
    "return": True,
    "volatility": False,
    "ci": False,
}

# Import availability checkers
try:
    _SM_ETS_AVAILABLE = (
        _importlib_util.find_spec("statsmodels.tsa.holtwinters") is not None
    )
except Exception:
    _SM_ETS_AVAILABLE = False

try:
    _SM_SARIMAX_AVAILABLE = (
        _importlib_util.find_spec("statsmodels.tsa.statespace.sarimax") is not None
    )
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


def get_forecast_methods_data() -> Dict[str, Any]:
    """Return metadata about available forecast methods and their requirements.

    This is derived from ForecastRegistry to avoid drift.
    """
    _ensure_registry_loaded()
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

    return {
        "methods": methods,
        "total": len(methods),
        "categories": categories,
    }


def _extract_description(cls: Any, fallback: str) -> str:
    doc = getattr(cls, "__doc__", None)
    if isinstance(doc, str):
        line = doc.strip().splitlines()
        if line and line[0].strip():
            return line[0].strip()
    return str(fallback)


@lru_cache(maxsize=1)
def _check_chronos_runtime_support() -> Tuple[bool, List[str]]:
    """Verify Chronos exposes APIs required by our adapters."""
    reqs: List[str] = []
    try:
        chronos_mod = _importlib.import_module("chronos")
    except Exception:
        return False, ["chronos-forecasting"]

    if not any(
        hasattr(chronos_mod, attr)
        for attr in ("Chronos2Pipeline", "ChronosBoltPipeline", "ChronosPipeline")
    ):
        reqs.append("chronos pipeline API")

    # Some incompatible chronos versions import but fail at runtime due missing internals.
    try:
        chronos2_mod = _importlib.import_module("chronos.chronos2")
        if not hasattr(chronos2_mod, "ChronosBoltModelForForecasting"):
            reqs.append("chronos-forecasting>=2.0.0")
    except Exception:
        reqs.append("chronos-forecasting>=2.0.0")

    return len(reqs) == 0, reqs


def _check_requirements(method: str, requires: List[str]) -> Tuple[bool, List[str]]:
    available = True
    reqs = list(requires or [])

    # Check availability based on method type and runtime flags.
    if (
        method in ("ses", "holt", "holt_winters_add", "holt_winters_mul", "ets")
        and not _SM_ETS_AVAILABLE
    ):
        available = False
        reqs.append("statsmodels")
    if method in ("arima", "sarima") and not _SM_SARIMAX_AVAILABLE:
        available = False
        reqs.append("statsmodels")
    if method == "statsforecast" and not _SF_AVAILABLE:
        available = False
        reqs.append("statsforecast")
    if method == "mlforecast" and not _MLF_AVAILABLE:
        available = False
        reqs.append("mlforecast")
    if method == "mlf_rf" and not _MLF_AVAILABLE:
        available = False
        reqs.append("mlforecast, scikit-learn")
    if method == "mlf_lightgbm" and (not _MLF_AVAILABLE or not _LGB_AVAILABLE):
        available = False
        reqs.append("mlforecast, lightgbm")
    if method in ("chronos_bolt", "chronos2"):
        if not _CHRONOS_AVAILABLE:
            available = False
            reqs.append("chronos-forecasting")
        else:
            chronos_ok, chronos_reqs = _check_chronos_runtime_support()
            if not chronos_ok:
                available = False
                reqs.extend(chronos_reqs)
    if method == "timesfm" and not _TIMESFM_AVAILABLE:
        available = False
        reqs.append("timesfm")
    if method == "lag_llama" and not _LAG_LLAMA_AVAILABLE:
        available = False
        reqs.append("lag-llama, gluonts, torch")
    if method == "sktime" and not _SKTIME_AVAILABLE:
        available = False
        reqs.append("sktime")

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
    try:
        from .methods import classical  # noqa: F401
        from .methods import ets_arima  # noqa: F401
        from .methods import statsforecast  # noqa: F401
        from .methods import mlforecast  # noqa: F401
        from .methods import pretrained  # noqa: F401
        from .methods import neural  # noqa: F401
        from .methods import sktime  # noqa: F401
        from .methods import gluonts_extra  # noqa: F401
        from .methods import analog  # noqa: F401
        from .methods import ensemble  # noqa: F401
        from .methods import monte_carlo  # noqa: F401
    except Exception:
        pass


def _ensemble_metadata() -> Dict[str, Any]:
    return {
        "method": "ensemble",
        "category": "ensemble",
        "available": True,
        "requires": [],
        "description": "Adaptive ensemble with averaging, Bayesian model averaging, or stacking.",
        "params": [
            {
                "name": "methods",
                "type": "list",
                "description": "Methods to ensemble (default: naive,theta,fourier_ols)",
            },
            {
                "name": "mode",
                "type": "str",
                "description": "average|bma|stacking (default: average)",
            },
            {
                "name": "weights",
                "type": "list",
                "description": "Manual weights when mode=average",
            },
            {
                "name": "cv_points",
                "type": "int",
                "description": "Walk-forward anchors for weighting (default: 2*len(methods))",
            },
            {
                "name": "min_train_size",
                "type": "int",
                "description": "Minimum history per CV anchor (default: max(30, horizon*3))",
            },
            {
                "name": "method_params",
                "type": "dict",
                "description": "Per-method parameter overrides",
            },
            {
                "name": "expose_components",
                "type": "bool",
                "description": "Include component forecasts in response (default: True)",
            },
        ],
        "supports": {"price": True, "return": True, "volatility": True, "ci": False},
    }


# Availability flags that can be imported by other modules
__all__ = [
    "get_forecast_methods_data",
    "_SM_ETS_AVAILABLE",
    "_SM_SARIMAX_AVAILABLE",
    "_NF_AVAILABLE",
    "_MLF_AVAILABLE",
    "_SF_AVAILABLE",
    "_LGB_AVAILABLE",
    "_CHRONOS_AVAILABLE",
    "_TIMESFM_AVAILABLE",
    "_LAG_LLAMA_AVAILABLE",
]
