from __future__ import annotations

import difflib
import importlib
import logging
import os
import pkgutil
import time
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from .backtest import execute_forecast_backtest as _forecast_backtest_impl
from .capabilities import resolve_capability_request
from .exceptions import ForecastError, raise_if_error_result
from .forecast import execute_forecast as _forecast_impl
from ..core.execution_logging import (
    infer_result_success,
    log_operation_exception,
    log_operation_finish,
    log_operation_start,
)
from .requests import (
    ForecastBacktestRequest,
    ForecastBarrierOptimizeRequest,
    ForecastBarrierProbRequest,
    ForecastConformalIntervalsRequest,
    ForecastGenerateRequest,
    ForecastTuneGeneticRequest,
    ForecastTuneOptunaRequest,
    ForecastVolatilityEstimateRequest,
)

logger = logging.getLogger(__name__)


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


def _resolve_sktime_forecaster(method: str) -> Optional[Tuple[str, str]]:
    """Resolve a user-provided method name to (class_name, dotted_path)."""
    method_s = str(method or "").strip()
    if not method_s:
        return None

    mapping = _discover_sktime_forecasters()
    if not mapping:
        return None

    exact = mapping.get(method_s.lower())
    if exact:
        return exact

    norm_map: Dict[str, Tuple[str, str]] = {}
    for _, (cls_name, dotted) in mapping.items():
        norm_map.setdefault(_normalize_forecaster_name(cls_name), (cls_name, dotted))

    query_norm = _normalize_forecaster_name(method_s)
    if query_norm in norm_map:
        return norm_map[query_norm]

    starts = [value for key, value in norm_map.items() if key.startswith(query_norm)]
    if starts:
        return sorted(starts, key=lambda item: len(item[0]))[0]

    contains = [value for key, value in norm_map.items() if query_norm and query_norm in key]
    if contains:
        return sorted(contains, key=lambda item: len(item[0]))[0]

    candidates = difflib.get_close_matches(query_norm, list(norm_map), n=1, cutoff=0.6)
    if candidates:
        return norm_map[candidates[0]]
    return None


def run_forecast_generate(
    request: ForecastGenerateRequest,
    *,
    forecast_impl: Any = _forecast_impl,
    resolve_sktime_forecaster: Any = _resolve_sktime_forecaster,
    log_events: bool = True,
) -> Dict[str, Any]:
    started_at = time.perf_counter()
    lib = str(request.library or "native").strip().lower()
    method = str(request.method or "").strip()
    params = dict(request.params or {})
    if log_events:
        log_operation_start(
            logger,
            operation="forecast_generate",
            symbol=request.symbol,
            timeframe=request.timeframe,
            library=lib or "native",
            method=method or None,
        )

    def _finish(result: Dict[str, Any], *, resolved_method: Optional[str] = None) -> Dict[str, Any]:
        if log_events:
            log_operation_finish(
                logger,
                operation="forecast_generate",
                started_at=started_at,
                success=infer_result_success(result),
                symbol=request.symbol,
                timeframe=request.timeframe,
                library=lib or "native",
                method=method or None,
                resolved_method=resolved_method,
            )
        return result

    try:
        capability_requested = ":" in method
        lib, method, params = resolve_capability_request(
            library=lib,
            method=method,
            params=params,
            discover_sktime_forecasters=_discover_sktime_forecasters,
        ) if capability_requested else (lib, method, params)
        if capability_requested:
            if lib in ("", "native"):
                resolved_method = method or "theta"
            elif lib == "statsforecast":
                resolved_method = "statsforecast"
            elif lib == "sktime":
                resolved_method = "sktime"
            elif lib == "pretrained":
                resolved_method = method or "chronos2"
            elif lib == "mlforecast":
                resolved_method = "mlforecast"
            else:
                raise ForecastError(f"Unsupported library: {lib}")
        elif lib in ("", "native"):
            resolved_method = method or "theta"
        elif lib == "statsforecast":
            if not method:
                raise ForecastError("method is required for library=statsforecast")
            resolved_method = "statsforecast"
            params.setdefault("model_name", method)
        elif lib == "sktime":
            query = method.strip() if method else "ThetaForecaster"
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
            resolved_method = method or "chronos2"
        elif lib == "mlforecast":
            if not method:
                raise ForecastError("method is required for library=mlforecast")
            resolved_method = "mlforecast"
            params.setdefault("model", method)
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
            denoise=request.denoise,
            features=request.features or {},
            dimred_method=request.dimred_method,
            dimred_params=request.dimred_params,
            target_spec=request.target_spec,
        )
        if isinstance(out, dict) and "success" not in out and infer_result_success(out):
            out["success"] = True

        if (
            isinstance(out, dict)
            and lib in ("", "native")
            and str(resolved_method).strip().lower() == "theta"
        ):
            warning = (
                "Using native theta. StatsForecast theta is available via "
                f"`mtdata-cli forecast_generate {request.symbol} --timeframe {request.timeframe} "
                f"--library statsforecast --method Theta --horizon {request.horizon}` "
                "and may produce different forecasts/interval behavior."
            )
            warnings_out = out.get("warnings")
            if not isinstance(warnings_out, list):
                warnings_out = []
            has_interval_warning = any(
                "forecast_conformal_intervals" in str(item)
                or "confidence intervals are unavailable" in str(item)
                for item in warnings_out
            )
            if warning not in warnings_out and not has_interval_warning:
                warnings_out.append(warning)
            out["warnings"] = warnings_out
        return _finish(out, resolved_method=str(resolved_method))
    except Exception as exc:
        if log_events:
            log_operation_exception(
                logger,
                operation="forecast_generate",
                started_at=started_at,
                exc=exc,
                symbol=request.symbol,
                timeframe=request.timeframe,
                library=lib or "native",
                method=method or None,
            )
        raise


def run_forecast_backtest(
    request: ForecastBacktestRequest,
    *,
    backtest_impl: Any = _forecast_backtest_impl,
) -> Dict[str, Any]:
    started_at = time.perf_counter()
    log_operation_start(
        logger,
        operation="forecast_backtest",
        symbol=request.symbol,
        timeframe=request.timeframe,
        horizon=request.horizon,
        methods=len(request.methods or []),
    )
    try:
        result = backtest_impl(
            symbol=request.symbol,
            timeframe=request.timeframe,
            horizon=request.horizon,
            steps=request.steps,
            spacing=request.spacing,
            methods=request.methods,
            params_per_method=request.params_per_method,
            quantity=request.quantity,
            denoise=request.denoise,
            params=request.params,
            features=request.features,
            dimred_method=request.dimred_method,
            dimred_params=request.dimred_params,
            slippage_bps=request.slippage_bps,
            trade_threshold=request.trade_threshold,
            detail=request.detail,
        )
    except Exception as exc:
        log_operation_exception(
            logger,
            operation="forecast_backtest",
            started_at=started_at,
            exc=exc,
            symbol=request.symbol,
            timeframe=request.timeframe,
            horizon=request.horizon,
        )
        raise
    log_operation_finish(
        logger,
        operation="forecast_backtest",
        started_at=started_at,
        success=infer_result_success(result),
        symbol=request.symbol,
        timeframe=request.timeframe,
        horizon=request.horizon,
        methods=len(request.methods or []),
    )
    return result


def run_forecast_conformal_intervals(
    request: ForecastConformalIntervalsRequest,
    *,
    backtest_impl: Any = _forecast_backtest_impl,
    forecast_impl: Any = _forecast_impl,
) -> Dict[str, Any]:
    started_at = time.perf_counter()
    log_operation_start(
        logger,
        operation="forecast_conformal_intervals",
        symbol=request.symbol,
        timeframe=request.timeframe,
        method=request.method,
        horizon=request.horizon,
    )
    try:
        # 1) Rolling backtest to collect residuals.
        bt = raise_if_error_result(backtest_impl(
            symbol=request.symbol,
            timeframe=request.timeframe,
            horizon=int(request.horizon),
            steps=int(request.steps),
            spacing=int(request.spacing),
            methods=[str(request.method)],
            denoise=request.denoise,
            params_per_method={str(request.method): dict(request.params or {})},
            detail="full",
        ))
        res = bt.get("results", {}).get(str(request.method))
        if not res or not res.get("details"):
            raise ForecastError("Conformal calibration failed: no backtest details")

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

        q = 1.0 - float(request.ci_alpha)
        qerrs = [
            float(_np.quantile(_np.array(err, dtype=float), q)) if err else float("nan")
            for err in errs
        ]

        # 2) Forecast now (latest).
        out = raise_if_error_result(forecast_impl(
            symbol=request.symbol,
            timeframe=request.timeframe,
            method=request.method,
            horizon=int(request.horizon),
            params=request.params,
            denoise=request.denoise,
        ))
        yhat = out.get("forecast_price") or []
        if not yhat:
            raise ForecastError("Empty point forecast for conformal intervals")
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
            "ci_alpha": float(request.ci_alpha),
            "calibration_steps": int(request.steps),
            "calibration_spacing": int(request.spacing),
            "per_step_q": [float(v) for v in qerrs],
        }
        result["lower_price"] = [float(v) for v in lo.tolist()]
        result["upper_price"] = [float(v) for v in hi.tolist()]
        result["ci_alpha"] = float(request.ci_alpha)
    except Exception as exc:
        log_operation_exception(
            logger,
            operation="forecast_conformal_intervals",
            started_at=started_at,
            exc=exc,
            symbol=request.symbol,
            timeframe=request.timeframe,
            method=request.method,
            horizon=request.horizon,
        )
        raise
    log_operation_finish(
        logger,
        operation="forecast_conformal_intervals",
        started_at=started_at,
        success=infer_result_success(result),
        symbol=request.symbol,
        timeframe=request.timeframe,
        method=request.method,
        horizon=request.horizon,
    )
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
    started_at = time.perf_counter()
    log_operation_start(
        logger,
        operation="forecast_tune_genetic",
        symbol=request.symbol,
        timeframe=request.timeframe,
        method=request.method,
        methods=len(request.methods or []),
    )
    method_for_search, search_space = _resolve_tuning_search_space(request)
    try:
        result = genetic_search_impl(
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
    except Exception as exc:
        log_operation_exception(
            logger,
            operation="forecast_tune_genetic",
            started_at=started_at,
            exc=exc,
            symbol=request.symbol,
            timeframe=request.timeframe,
            method=request.method,
        )
        raise
    log_operation_finish(
        logger,
        operation="forecast_tune_genetic",
        started_at=started_at,
        success=infer_result_success(result),
        symbol=request.symbol,
        timeframe=request.timeframe,
        method=request.method,
        methods=len(request.methods or []),
    )
    return result


def run_forecast_tune_optuna(
    request: ForecastTuneOptunaRequest,
    *,
    optuna_search_impl: Any,
) -> Dict[str, Any]:
    started_at = time.perf_counter()
    log_operation_start(
        logger,
        operation="forecast_tune_optuna",
        symbol=request.symbol,
        timeframe=request.timeframe,
        method=request.method,
        methods=len(request.methods or []),
    )
    method_for_search, search_space = _resolve_tuning_search_space(request)
    try:
        result = optuna_search_impl(
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
    except Exception as exc:
        log_operation_exception(
            logger,
            operation="forecast_tune_optuna",
            started_at=started_at,
            exc=exc,
            symbol=request.symbol,
            timeframe=request.timeframe,
            method=request.method,
        )
        raise
    log_operation_finish(
        logger,
        operation="forecast_tune_optuna",
        started_at=started_at,
        success=infer_result_success(result),
        symbol=request.symbol,
        timeframe=request.timeframe,
        method=request.method,
        methods=len(request.methods or []),
    )
    return result


def run_forecast_barrier_prob(
    request: ForecastBarrierProbRequest,
    *,
    build_barrier_kwargs: Any,
    normalize_trade_direction: Any,
    barrier_hit_probabilities_impl: Any,
    barrier_closed_form_impl: Any,
) -> Dict[str, Any]:
    started_at = time.perf_counter()
    method_raw = str(request.method or "hmm_mc").lower().strip()
    method_aliases = {"mc": "hmm_mc"}
    method_val = method_aliases.get(method_raw, method_raw)
    mc_methods = {
        "auto",
        "bootstrap",
        "garch",
        "heston",
        "hmm_mc",
        "jump_diffusion",
        "mc_gbm",
        "mc_gbm_bb",
    }
    log_operation_start(
        logger,
        operation="forecast_barrier_prob",
        symbol=request.symbol,
        timeframe=request.timeframe,
        method=method_val,
        direction=request.direction,
    )

    direction, direction_error = normalize_trade_direction(request.direction)
    if direction_error:
        result = {"error": direction_error}
        log_operation_finish(
            logger,
            operation="forecast_barrier_prob",
            started_at=started_at,
            success=False,
            symbol=request.symbol,
            timeframe=request.timeframe,
            method=method_val,
            direction=request.direction,
        )
        return result

    try:
        if method_val in mc_methods:
            barrier_kwargs = build_barrier_kwargs(request.model_dump())
            result = barrier_hit_probabilities_impl(
                symbol=request.symbol,
                timeframe=request.timeframe,
                horizon=request.horizon,
                method=method_val,
                direction=direction,
                **barrier_kwargs,
                params=request.params,
                denoise=request.denoise,
            )
            log_operation_finish(
                logger,
                operation="forecast_barrier_prob",
                started_at=started_at,
                success=infer_result_success(result),
                symbol=request.symbol,
                timeframe=request.timeframe,
                method=method_val,
                direction=direction,
            )
            return result

        if method_val == "closed_form":
            result = barrier_closed_form_impl(
                symbol=request.symbol,
                timeframe=request.timeframe,
                horizon=request.horizon,
                direction=direction,
                barrier=request.barrier,
                mu=request.mu,
                sigma=request.sigma,
                denoise=request.denoise,
            )
            log_operation_finish(
                logger,
                operation="forecast_barrier_prob",
                started_at=started_at,
                success=infer_result_success(result),
                symbol=request.symbol,
                timeframe=request.timeframe,
                method=method_val,
                direction=direction,
            )
            return result
    except Exception as exc:
        log_operation_exception(
            logger,
            operation="forecast_barrier_prob",
            started_at=started_at,
            exc=exc,
            symbol=request.symbol,
            timeframe=request.timeframe,
            method=method_val,
            direction=direction,
        )
        raise

    result = {"error": f"Unknown method: {request.method}"}
    log_operation_finish(
        logger,
        operation="forecast_barrier_prob",
        started_at=started_at,
        success=False,
        symbol=request.symbol,
        timeframe=request.timeframe,
        method=method_val,
        direction=direction,
    )
    return result


def run_forecast_barrier_optimize(
    request: ForecastBarrierOptimizeRequest,
    *,
    parse_kv_or_json: Any,
    barrier_optimize_impl: Any,
    cpu_count: Any = os.cpu_count,
) -> Dict[str, Any]:
    started_at = time.perf_counter()
    log_operation_start(
        logger,
        operation="forecast_barrier_optimize",
        symbol=request.symbol,
        timeframe=request.timeframe,
        method=request.method,
        direction=request.direction,
    )
    params_norm = parse_kv_or_json(request.params)
    if not isinstance(params_norm, dict):
        params_norm = {}
    defaults = {
        "optimizer": "optuna",
        "sampler": "tpe",
        "pruner": "median",
        "n_jobs": int((cpu_count() or 1)),
        "seed": 42,
    }
    for key, value in defaults.items():
        if key not in params_norm:
            params_norm[key] = value

    try:
        result = barrier_optimize_impl(
            symbol=request.symbol,
            timeframe=request.timeframe,
            horizon=request.horizon,
            method=request.method,
            direction=request.direction,
            mode=request.mode,
            tp_min=request.tp_min,
            tp_max=request.tp_max,
            tp_steps=request.tp_steps,
            sl_min=request.sl_min,
            sl_max=request.sl_max,
            sl_steps=request.sl_steps,
            params=params_norm,
            denoise=request.denoise,
            objective=request.objective,
            return_grid=request.return_grid,
            top_k=request.top_k,
            output=request.output,
            viable_only=request.viable_only,
            concise=request.concise,
            grid_style=request.grid_style,
            preset=request.preset,
            vol_window=request.vol_window,
            vol_min_mult=request.vol_min_mult,
            vol_max_mult=request.vol_max_mult,
            vol_steps=request.vol_steps,
            vol_sl_extra=request.vol_sl_extra,
            vol_floor_pct=request.vol_floor_pct,
            vol_floor_pips=request.vol_floor_pips,
            ratio_min=request.ratio_min,
            ratio_max=request.ratio_max,
            ratio_steps=request.ratio_steps,
            refine=request.refine,
            refine_radius=request.refine_radius,
            refine_steps=request.refine_steps,
            min_prob_win=request.min_prob_win,
            max_prob_no_hit=request.max_prob_no_hit,
            max_median_time=request.max_median_time,
            fast_defaults=request.fast_defaults,
            search_profile=request.search_profile,
        )
    except Exception as exc:
        log_operation_exception(
            logger,
            operation="forecast_barrier_optimize",
            started_at=started_at,
            exc=exc,
            symbol=request.symbol,
            timeframe=request.timeframe,
            method=request.method,
            direction=request.direction,
        )
        raise
    log_operation_finish(
        logger,
        operation="forecast_barrier_optimize",
        started_at=started_at,
        success=infer_result_success(result),
        symbol=request.symbol,
        timeframe=request.timeframe,
        method=request.method,
        direction=request.direction,
    )
    return result


def run_forecast_volatility_estimate(
    request: ForecastVolatilityEstimateRequest,
    *,
    forecast_volatility_impl: Any,
) -> Dict[str, Any]:
    started_at = time.perf_counter()
    log_operation_start(
        logger,
        operation="forecast_volatility_estimate",
        symbol=request.symbol,
        timeframe=request.timeframe,
        method=request.method,
        horizon=request.horizon,
    )
    try:
        result = forecast_volatility_impl(
            symbol=request.symbol,
            timeframe=request.timeframe,
            horizon=request.horizon,
            method=request.method,
            proxy=request.proxy,
            params=request.params,
            as_of=request.as_of,
            denoise=request.denoise,
        )
    except Exception as exc:
        log_operation_exception(
            logger,
            operation="forecast_volatility_estimate",
            started_at=started_at,
            exc=exc,
            symbol=request.symbol,
            timeframe=request.timeframe,
            method=request.method,
            horizon=request.horizon,
        )
        raise
    log_operation_finish(
        logger,
        operation="forecast_volatility_estimate",
        started_at=started_at,
        success=infer_result_success(result),
        symbol=request.symbol,
        timeframe=request.timeframe,
        method=request.method,
        horizon=request.horizon,
    )
    return result
