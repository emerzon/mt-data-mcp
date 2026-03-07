from __future__ import annotations

import difflib
import importlib
import pkgutil
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from .backtest import forecast_backtest as _forecast_backtest_impl
from .exceptions import ForecastError
from .forecast import forecast as _forecast_impl
from .requests import (
    ForecastBacktestRequest,
    ForecastConformalIntervalsRequest,
    ForecastGenerateRequest,
    ForecastTuneGeneticRequest,
    ForecastTuneOptunaRequest,
)


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


def run_forecast_conformal_intervals(
    request: ForecastConformalIntervalsRequest,
    *,
    backtest_impl: Any = _forecast_backtest_impl,
    forecast_impl: Any = _forecast_impl,
) -> Dict[str, Any]:
    # 1) Rolling backtest to collect residuals.
    bt = backtest_impl(
        symbol=request.symbol,
        timeframe=request.timeframe,
        horizon=int(request.horizon),
        steps=int(request.steps),
        spacing=int(request.spacing),
        methods=[str(request.method)],
        denoise=request.denoise,
        params={str(request.method): dict(request.params or {})},
        detail="full",
    )
    if "error" in bt:
        return bt
    res = bt.get("results", {}).get(str(request.method))
    if not res or not res.get("details"):
        return {"error": "Conformal calibration failed: no backtest details"}

    # Build per-step residuals |y_hat_i - y_i|.
    fh = int(request.horizon)
    errs: List[List[float]] = [[] for _ in range(fh)]
    for detail in res["details"]:
        fc = detail.get("forecast")
        act = detail.get("actual")
        if not fc or not act:
            continue
        width = min(len(fc), len(act), fh)
        for i in range(width):
            try:
                errs[i].append(abs(float(fc[i]) - float(act[i])))
            except Exception:
                continue

    import numpy as _np

    q = 1.0 - float(request.alpha)
    qerrs = [
        float(_np.quantile(_np.array(err, dtype=float), q)) if err else float("nan")
        for err in errs
    ]

    # 2) Forecast now (latest).
    out = forecast_impl(
        symbol=request.symbol,
        timeframe=request.timeframe,
        method=request.method,
        horizon=int(request.horizon),
        params=request.params,
        denoise=request.denoise,
    )
    if "error" in out:
        return out
    yhat = out.get("forecast_price") or []
    if not yhat:
        return {"error": "Empty point forecast for conformal intervals"}
    yhat_arr = _np.array(yhat, dtype=float)
    fh_eff = min(fh, yhat_arr.size)
    lo = _np.empty(fh_eff, dtype=float)
    hi = _np.empty(fh_eff, dtype=float)
    for i in range(fh_eff):
        err = qerrs[i] if i < len(qerrs) and _np.isfinite(qerrs[i]) else 0.0
        lo[i] = yhat_arr[i] - err
        hi[i] = yhat_arr[i] + err

    result = dict(out)
    result["conformal"] = {
        "alpha": float(request.alpha),
        "calibration_steps": int(request.steps),
        "calibration_spacing": int(request.spacing),
        "per_step_q": [float(v) for v in qerrs],
    }
    result["lower_price"] = [float(v) for v in lo.tolist()]
    result["upper_price"] = [float(v) for v in hi.tolist()]
    result["ci_alpha"] = float(request.alpha)
    return result


def _resolve_tuning_search_space(
    request: ForecastTuneGeneticRequest | ForecastTuneOptunaRequest,
) -> tuple[Optional[str], Dict[str, Any]]:
    method_for_search: Optional[str] = request.method
    from ..forecast.tune import default_search_space as _default_search_space

    search_space = dict(request.search_space or {})
    if not search_space:
        if isinstance(request.methods, (list, tuple)) and len(request.methods) > 0:
            return None, _default_search_space(method=None, methods=request.methods)
        return method_for_search, _default_search_space(method=method_for_search, methods=None)
    if isinstance(request.methods, (list, tuple)) and len(request.methods) > 0:
        method_for_search = None
    return method_for_search, search_space


def run_forecast_tune_genetic(
    request: ForecastTuneGeneticRequest,
    *,
    genetic_search_impl: Any,
) -> Dict[str, Any]:
    method_for_search, search_space = _resolve_tuning_search_space(request)
    return genetic_search_impl(
        symbol=request.symbol,
        timeframe=request.timeframe,
        method=str(method_for_search) if method_for_search is not None else None,
        methods=request.methods,
        horizon=int(request.horizon),
        steps=int(request.steps),
        spacing=int(request.spacing),
        search_space=search_space,
        metric=str(request.metric),
        mode=str(request.mode),
        population=int(request.population),
        generations=int(request.generations),
        crossover_rate=float(request.crossover_rate),
        mutation_rate=float(request.mutation_rate),
        seed=int(request.seed),
        trade_threshold=float(request.trade_threshold),
        denoise=request.denoise,
        features=request.features,
        dimred_method=request.dimred_method,
        dimred_params=request.dimred_params,
    )


def run_forecast_tune_optuna(
    request: ForecastTuneOptunaRequest,
    *,
    optuna_search_impl: Any,
) -> Dict[str, Any]:
    method_for_search, search_space = _resolve_tuning_search_space(request)
    return optuna_search_impl(
        symbol=request.symbol,
        timeframe=request.timeframe,
        method=str(method_for_search) if method_for_search is not None else None,
        methods=request.methods,
        horizon=int(request.horizon),
        steps=int(request.steps),
        spacing=int(request.spacing),
        search_space=search_space,
        metric=str(request.metric),
        mode=str(request.mode),
        n_trials=int(request.n_trials),
        timeout=float(request.timeout) if request.timeout is not None else None,
        n_jobs=int(request.n_jobs),
        sampler=str(request.sampler),
        pruner=str(request.pruner),
        study_name=str(request.study_name) if request.study_name is not None else None,
        storage=str(request.storage) if request.storage is not None else None,
        seed=int(request.seed),
        trade_threshold=float(request.trade_threshold),
        denoise=request.denoise,
        features=request.features,
        dimred_method=request.dimred_method,
        dimred_params=request.dimred_params,
    )
