from __future__ import annotations

import difflib
import importlib
import pkgutil
from functools import lru_cache
from typing import Any, Dict, Optional, Tuple

from .backtest import forecast_backtest as _forecast_backtest_impl
from .exceptions import ForecastError
from .forecast import forecast as _forecast_impl
from .requests import ForecastBacktestRequest, ForecastGenerateRequest


@lru_cache(maxsize=1)
def _discover_sktime_forecasters() -> Dict[str, Tuple[str, str]]:
    """Return mapping of forecaster class name (lower) -> (class_name, dotted path)."""
    try:
        import sktime.forecasting as _sf  # type: ignore
        from sktime.forecasting.base import BaseForecaster  # type: ignore
    except Exception:
        return {}

    mapping: Dict[str, Tuple[str, str]] = {}

    def _skip_module(mod_name: str) -> bool:
        parts = mod_name.split(".")
        if "tests" in parts:
            return True
        if any(part.startswith("test") for part in parts):
            return True
        return False

    for mod in pkgutil.walk_packages(getattr(_sf, "__path__", []), _sf.__name__ + "."):
        mod_name = getattr(mod, "name", None)
        if not isinstance(mod_name, str) or _skip_module(mod_name):
            continue
        try:
            module = importlib.import_module(mod_name)
        except Exception:
            continue
        for _, obj in vars(module).items():
            if not isinstance(obj, type):
                continue
            if obj is BaseForecaster:
                continue
            name = getattr(obj, "__name__", None)
            if not isinstance(name, str) or not name or name.startswith("_"):
                continue
            try:
                if not issubclass(obj, BaseForecaster):
                    continue
            except Exception:
                continue
            key = name.lower()
            if key not in mapping:
                mapping[key] = (name, f"{obj.__module__}.{name}")
    return mapping


def _normalize_forecaster_name(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def _resolve_sktime_forecaster(model: str) -> Optional[Tuple[str, str]]:
    """Resolve a user-provided model name to (class_name, dotted_path)."""
    model_s = str(model or "").strip()
    if not model_s:
        return None

    mapping = _discover_sktime_forecasters()
    if not mapping:
        return None

    exact = mapping.get(model_s.lower())
    if exact:
        return exact

    norm_map: Dict[str, Tuple[str, str]] = {}
    for _, (cls_name, dotted) in mapping.items():
        norm_map.setdefault(_normalize_forecaster_name(cls_name), (cls_name, dotted))

    query_norm = _normalize_forecaster_name(model_s)
    if query_norm in norm_map:
        return norm_map[query_norm]

    starts = [value for key, value in norm_map.items() if key.startswith(query_norm)]
    if starts:
        return sorted(starts, key=lambda item: len(item[0]))[0]

    contains = [value for key, value in norm_map.items() if query_norm and query_norm in key]
    if contains:
        return sorted(contains, key=lambda item: len(item[0]))[0]

    candidates = difflib.get_close_matches(query_norm, list(norm_map.keys()), n=1, cutoff=0.6)
    if candidates:
        return norm_map[candidates[0]]
    return None


def run_forecast_generate(
    request: ForecastGenerateRequest,
    *,
    forecast_impl: Any = _forecast_impl,
    resolve_sktime_forecaster: Any = _resolve_sktime_forecaster,
) -> Dict[str, Any]:
    lib = str(request.library or "native").strip().lower()
    model = str(request.model or "").strip()
    params = dict(request.model_params or {})
    legacy_method = str(request.method or "").strip()

    if legacy_method:
        resolved_method = legacy_method
    elif lib in ("", "native"):
        resolved_method = model or "theta"
    elif lib == "statsforecast":
        if not model:
            raise ForecastError("model is required for library=statsforecast")
        resolved_method = "statsforecast"
        params.setdefault("model_name", model)
    elif lib == "sktime":
        query = model.strip() if model else "ThetaForecaster"
        if "." in query:
            resolved_method = "sktime"
            params.setdefault("estimator", query)
        else:
            found = resolve_sktime_forecaster(query)
            if not found:
                raise ForecastError(f"Unknown sktime forecaster '{query}'")
            _, dotted = found
            resolved_method = "sktime"
            params.setdefault("estimator", dotted)
    elif lib == "pretrained":
        resolved_method = model or "chronos2"
    elif lib == "mlforecast":
        if not model:
            raise ForecastError("model is required for library=mlforecast")
        resolved_method = "mlforecast"
        params.setdefault("model", model)
    else:
        raise ForecastError(f"Unsupported library: {request.library}")

    out = forecast_impl(
        symbol=request.symbol,
        timeframe=request.timeframe,
        method=str(resolved_method),
        horizon=request.horizon,
        lookback=request.lookback,
        as_of=request.as_of,
        params=params,
        ci_alpha=request.ci_alpha,
        quantity=request.quantity,
        target="price",
        denoise=request.denoise,
        features=request.features or {},
        dimred_method=request.dimred_method,
        dimred_params=request.dimred_params,
        target_spec=request.target_spec,
    )

    if (
        isinstance(out, dict)
        and not legacy_method
        and lib in ("", "native")
        and str(resolved_method).strip().lower() == "theta"
    ):
        warning = (
            "Using native theta. StatsForecast theta is available via "
            f"`mtdata-cli forecast_generate {request.symbol} --timeframe {request.timeframe} "
            f"--library statsforecast --model Theta --horizon {request.horizon}` "
            "and may produce different forecasts/interval behavior."
        )
        warnings_out = out.get("warnings")
        if not isinstance(warnings_out, list):
            warnings_out = []
        if warning not in warnings_out:
            warnings_out.append(warning)
        out["warnings"] = warnings_out
    return out


def run_forecast_backtest(
    request: ForecastBacktestRequest,
    *,
    backtest_impl: Any = _forecast_backtest_impl,
) -> Dict[str, Any]:
    return backtest_impl(
        symbol=request.symbol,
        timeframe=request.timeframe,
        horizon=request.horizon,
        steps=request.steps,
        spacing=request.spacing,
        methods=request.methods,
        params_per_method=request.params_per_method,
        quantity=request.quantity,
        target=request.target,
        denoise=request.denoise,
        params=request.params,
        features=request.features,
        dimred_method=request.dimred_method,
        dimred_params=request.dimred_params,
        slippage_bps=request.slippage_bps,
        trade_threshold=request.trade_threshold,
        detail=request.detail,
    )
