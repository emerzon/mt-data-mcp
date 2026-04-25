from __future__ import annotations

import difflib
import importlib
import logging
import math
import os
import pkgutil
import time
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from ..core.execution_logging import (
    infer_result_success,
    log_operation_exception,
    log_operation_finish,
    log_operation_start,
)
from ..core.output_contract import attach_collection_contract
from .backtest import execute_forecast_backtest as _forecast_backtest_impl
from .backtest import _compact_metrics_payload
from .capabilities import resolve_capability_request
from .barriers_shared import barrier_method_error, normalize_barrier_method
from .exceptions import ForecastError, raise_if_error_result
from .forecast import execute_forecast as _forecast_impl
from .requests import (
    ForecastBacktestRequest,
    ForecastBarrierOptimizeRequest,
    ForecastBarrierProbRequest,
    ForecastConformalIntervalsRequest,
    ForecastGenerateRequest,
    ForecastOptimizeHintsRequest,
    ForecastTuneGeneticRequest,
    ForecastTuneOptunaRequest,
    ForecastVolatilityEstimateRequest,
    StrategyBacktestRequest,
)

logger = logging.getLogger(__name__)


def _normalize_trader_detail(value: Any, *, default: str = "compact") -> str:
    normalized = str(default if value is None else value).strip().lower()
    if normalized in {"summary", "summary_only"}:
        return "compact"
    if normalized == "full":
        return "full"
    if normalized == "standard":
        return "standard"
    return "compact"


def _forecast_interval_summary(payload: Dict[str, Any]) -> Optional[Dict[str, float]]:
    lower_key = next(
        (
            key
            for key in ("lower_price", "lower_return", "lower")
            if isinstance(payload.get(key), list)
        ),
        None,
    )
    if lower_key is None:
        return None
    upper_key = lower_key.replace("lower", "upper", 1)
    lower_vals = payload.get(lower_key)
    upper_vals = payload.get(upper_key)
    if not isinstance(lower_vals, list) or not isinstance(upper_vals, list) or not lower_vals or not upper_vals:
        return None
    try:
        widths = [
            float(upper) - float(lower)
            for lower, upper in zip(lower_vals, upper_vals)
        ]
        if not widths:
            return None
        widths_sorted = sorted(widths)
        return {
            "first_low": float(lower_vals[0]),
            "first_high": float(upper_vals[0]),
            "last_low": float(lower_vals[-1]),
            "last_high": float(upper_vals[-1]),
            "median_width": float(widths_sorted[len(widths_sorted) // 2]),
        }
    except Exception:
        return None


def _apply_forecast_generate_detail(
    payload: Dict[str, Any],
    request: ForecastGenerateRequest,
) -> Dict[str, Any]:
    if not isinstance(payload, dict) or payload.get("error"):
        return payload

    detail_value = _normalize_trader_detail(getattr(request, "detail", "compact"))
    if detail_value in {"standard", "full"}:
        out = dict(payload)
        out.setdefault("symbol", request.symbol)
        out.setdefault("timeframe", request.timeframe)
        out["detail"] = detail_value
        return attach_collection_contract(
            out,
            collection_kind="time_series",
            series=_forecast_generate_series_rows(out),
            include_contract_meta=detail_value == "full",
        )

    compact: Dict[str, Any] = {
        "success": bool(payload.get("success", True)),
        "symbol": request.symbol,
        "timeframe": request.timeframe,
        "method": payload.get("method"),
        "horizon": payload.get("horizon"),
        "quantity": payload.get("quantity"),
        "detail": "compact",
    }
    ci_unavailable = str(payload.get("ci_status") or "").strip().lower() == "unavailable"
    for key in (
        "last_observation_time",
        "forecast_start_time",
        "forecast_time",
        "forecast_price",
        "forecast_return",
        "ci_status",
        "ci_available",
        "ci_alpha",
        "warnings",
    ):
        if ci_unavailable and key.startswith("ci_"):
            continue
        value = payload.get(key)
        if value not in (None, "", [], {}):
            compact[key] = value
    interval_summary = _forecast_interval_summary(payload)
    if interval_summary:
        compact["interval_summary"] = interval_summary
    for key, value in payload.items():
        if key in compact:
            continue
        if key in {
            "base_col",
            "last_observation_epoch",
            "forecast_start_epoch",
            "forecast_anchor",
            "forecast_step_seconds",
            "forecast_epoch",
            "last_price",
            "last_price_close",
            "last_price_source",
            "lower_price",
            "upper_price",
            "lower_return",
            "upper_return",
            "lower",
            "upper",
            "ci",
        }:
            continue
        if ci_unavailable and str(key).startswith("ci_"):
            continue
        compact[key] = value
    return attach_collection_contract(
        compact,
        collection_kind="time_series",
        series=_forecast_generate_series_rows(compact),
        include_contract_meta=False,
    )


def _forecast_generate_series_rows(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    times = payload.get("forecast_time")
    prices = payload.get("forecast_price")
    if not isinstance(times, list) or not isinstance(prices, list):
        return []

    optional_series = {
        "forecast_return": payload.get("forecast_return"),
        "lower_price": payload.get("lower_price"),
        "upper_price": payload.get("upper_price"),
    }
    rows: List[Dict[str, Any]] = []
    for idx, time_value in enumerate(times):
        row: Dict[str, Any] = {
            "time": time_value,
            "forecast_price": prices[idx] if idx < len(prices) else None,
        }
        for key, values in optional_series.items():
            if isinstance(values, list) and idx < len(values):
                row[key] = values[idx]
        rows.append(row)
    return rows


def _specific_forecast_method_name(
    *,
    requested_method: str,
    resolved_method: str,
    resolved_library: str,
    params: Dict[str, Any],
) -> str:
    requested = str(requested_method or "").strip()
    if ":" in requested:
        requested = requested.split(":", 1)[1].strip()
    if requested and requested.lower() != str(resolved_method or "").strip().lower():
        return requested

    selector_key_by_library = {
        "statsforecast": "model_name",
        "sktime": "estimator",
        "mlforecast": "model",
    }
    selector_key = selector_key_by_library.get(resolved_library)
    if selector_key:
        selector_value = params.get(selector_key)
        if selector_value not in (None, "", [], {}):
            return str(selector_value)
    return str(resolved_method or requested or "").strip()


def _annotate_forecast_generate_method(
    payload: Dict[str, Any],
    *,
    requested_method: str,
    resolved_method: str,
    resolved_library: str,
    params: Dict[str, Any],
) -> None:
    if not isinstance(payload, dict) or payload.get("error"):
        return
    library_name = str(resolved_library or "native").strip().lower() or "native"
    if library_name in {"", "native"}:
        return

    payload["library"] = library_name
    adapter_method = str(resolved_method or "").strip().lower()
    output_method = str(payload.get("method") or "").strip().lower()
    if output_method in {"", adapter_method}:
        payload["method"] = _specific_forecast_method_name(
            requested_method=requested_method,
            resolved_method=resolved_method,
            resolved_library=library_name,
            params=params,
        )


def _apply_barrier_prob_detail(
    payload: Dict[str, Any],
    request: ForecastBarrierProbRequest,
) -> Dict[str, Any]:
    if not isinstance(payload, dict) or payload.get("error"):
        return payload

    def _set_if_present(target: Dict[str, Any], key: str, value: Any) -> None:
        if value not in (None, "", [], {}):
            target[key] = value

    detail_value = _normalize_trader_detail(getattr(request, "detail", "compact"))
    if detail_value == "full":
        out = dict(payload)
        out["detail"] = "full"
        return out

    if "prob_hit" in payload:
        closed_form: Dict[str, Any] = {
            "success": bool(payload.get("success", True)),
            "detail": detail_value,
        }
        for key in (
            "symbol",
            "timeframe",
            "direction",
            "horizon",
            "barrier",
            "last_price",
            "prob_hit",
        ):
            _set_if_present(closed_form, key, payload.get(key))
        if detail_value == "standard":
            for key in ("already_hit", "mu_annual", "log_drift_annual", "sigma_annual"):
                value = payload.get(key)
                if value not in (None, "", [], {}):
                    closed_form[key] = value
        if set(closed_form) == {"success", "detail"}:
            return dict(payload)
        return closed_form

    if detail_value == "standard":
        out = dict(payload)
        out.pop("tp_hit_prob_by_t", None)
        out.pop("sl_hit_prob_by_t", None)
        out.pop("sim_meta", None)
        out.pop("model_summary", None)
        out["detail"] = "standard"
        return out

    compact: Dict[str, Any] = {
        "success": bool(payload.get("success", True)),
        "detail": "compact",
    }
    for key in (
        "symbol",
        "timeframe",
        "method",
        "direction",
        "horizon",
        "last_price",
        "tp_price",
        "sl_price",
        "prob_tp_first",
        "prob_sl_first",
        "prob_no_hit",
        "edge",
    ):
        _set_if_present(compact, key, payload.get(key))
    confidence: Dict[str, Any] = {}
    for key in ("prob_tp_first_ci95", "prob_sl_first_ci95", "prob_no_hit_ci95"):
        value = payload.get(key)
        if value not in (None, "", [], {}):
            confidence[key] = value
    if confidence:
        compact["confidence"] = confidence
    timing: Dict[str, Any] = {}
    for source_key, target_key in (
        ("time_to_tp_bars", "tp"),
        ("time_to_sl_bars", "sl"),
    ):
        value = payload.get(source_key)
        if isinstance(value, dict) and any(val not in (None, "") for val in value.values()):
            timing[target_key] = {
                key: value.get(key)
                for key in ("mean", "median")
                if value.get(key) not in (None, "")
            }
    if timing:
        compact["timing_bars"] = timing
    if payload.get("warnings") not in (None, "", [], {}):
        compact["warnings"] = payload.get("warnings")
    for key, value in payload.items():
        if key in compact:
            continue
        if key in {
            "prob_tp_first_ci95",
            "prob_tp_first_se",
            "prob_sl_first_ci95",
            "prob_sl_first_se",
            "prob_no_hit_ci95",
            "prob_tie",
            "prob_tie_se",
            "prob_no_hit_se",
            "prob_tie_ci95",
            "tp_hit_prob_by_t",
            "sl_hit_prob_by_t",
            "time_to_tp_bars",
            "time_to_sl_bars",
            "sim_meta",
            "model_summary",
        }:
            continue
        compact[key] = value
    if set(compact) == {"success", "detail"}:
        return dict(payload)
    return compact


def _closed_form_barrier_input_error(request: ForecastBarrierProbRequest) -> Optional[str]:
    try:
        barrier_value = float(request.barrier)
    except (TypeError, ValueError):
        barrier_value = 0.0
    if barrier_value > 0.0:
        return None
    for field_name in (
        "tp_abs",
        "sl_abs",
        "tp_pct",
        "sl_pct",
        "tp_ticks",
        "sl_ticks",
        "tp_pips",
        "sl_pips",
    ):
        if getattr(request, field_name, None) is not None:
            return (
                "The closed_form method uses the absolute barrier parameter and "
                "does not consume TP/SL inputs such as tp_pct/sl_pct, tp_abs/sl_abs, "
                "or tick-based barriers. Provide barrier as a positive price, or use "
                "a Monte Carlo method such as mc_gbm for TP/SL barrier inputs."
            )
    return None


def _is_interval_unavailable_warning(value: Any) -> bool:
    text = str(value)
    return (
        "forecast_conformal_intervals" in text
        or "confidence intervals are unavailable" in text
    )


def _compact_backtest_result(result: Dict[str, Any]) -> Dict[str, Any]:
    raw_results = result.get("results")
    if not isinstance(raw_results, dict):
        return result

    def _sort_metric(value: Any) -> Optional[float]:
        try:
            value_f = float(value)
        except (TypeError, ValueError):
            return None
        return value_f if math.isfinite(value_f) else None

    ranked_methods: list[Dict[str, Any]] = []
    for method_name, method_payload in raw_results.items():
        if not isinstance(method_payload, dict):
            ranked_methods.append({"method": method_name, "result": method_payload})
            continue
        details = method_payload.get("details")
        details_count = len(details) if isinstance(details, list) else None
        metrics = (
            method_payload.get("metrics")
            if isinstance(method_payload.get("metrics"), dict)
            else {}
        )
        method_out: Dict[str, Any] = {"method": method_name}
        for key in (
            "success",
            "avg_rmse",
            "avg_mae",
            "avg_directional_accuracy",
            "successful_tests",
            "num_tests",
            "metrics_available",
            "metrics_reason",
        ):
            if key in method_payload:
                method_out[key] = method_payload[key]
        for key in (
            "win_rate",
            "win_rate_display",
            "max_drawdown",
            "avg_return",
            "avg_return_per_trade",
            "trades_observed",
        ):
            if key in metrics:
                method_out[key] = metrics[key]
        if isinstance(details, list):
            method_out["details_count"] = len(details)
        ranked_row = dict(method_out)
        ranked_row["_sort_metric"] = _sort_metric(
            method_payload.get("avg_rmse", method_payload.get("avg_mae"))
        )
        ranked_methods.append(ranked_row)

    compact_out = dict(result)
    compact_out.pop("results", None)
    ranked_methods.sort(
        key=lambda row: (
            row.get("_sort_metric") is None,
            row.get("_sort_metric") if row.get("_sort_metric") is not None else 0.0,
            str(row.get("method") or ""),
        )
    )
    compact_out["ranked_methods"] = [
        {key: value for key, value in row.items() if key != "_sort_metric"}
        for row in ranked_methods
    ]
    return compact_out


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


def _finite_sample_conformal_quantile(values: List[float], alpha: float) -> float:
    if not values:
        return float("nan")

    import numpy as _np

    arr = _np.asarray(values, dtype=float)
    if _np.isnan(arr).any():
        return float("nan")

    n = int(arr.size)
    rank = max(1, min(n, math.ceil((n + 1) * (1.0 - float(alpha)))))
    return float(_np.partition(arr, rank - 1)[rank - 1])


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
        requested_method = method
        original_resolution = (lib, method, dict(params))
        lib, method, params = resolve_capability_request(
            library=lib,
            method=method,
            params=params,
            discover_sktime_forecasters=_discover_sktime_forecasters,
        )
        capability_requested = capability_requested or (lib, method, params) != original_resolution
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
            async_mode=getattr(request, 'async_mode', False),
            model_id=getattr(request, 'model_id', None),
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
                _is_interval_unavailable_warning(item) for item in warnings_out
            )
            if warning not in warnings_out and not has_interval_warning:
                warnings_out.append(warning)
            out["warnings"] = warnings_out
        if isinstance(out, dict):
            _annotate_forecast_generate_method(
                out,
                requested_method=requested_method,
                resolved_method=str(resolved_method),
                resolved_library=lib,
                params=params,
            )
        out = _apply_forecast_generate_detail(out, request)
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
    if str(request.detail or "compact").strip().lower() == "compact":
        return _compact_backtest_result(result)
    return result


def run_strategy_backtest(
    request: StrategyBacktestRequest,
    *,
    strategy_backtest_impl: Any,
) -> Dict[str, Any]:
    started_at = time.perf_counter()
    log_operation_start(
        logger,
        operation="strategy_backtest",
        symbol=request.symbol,
        timeframe=request.timeframe,
        strategy=request.strategy,
        lookback=request.lookback,
    )
    try:
        result = strategy_backtest_impl(
            symbol=request.symbol,
            timeframe=request.timeframe,
            strategy=request.strategy,
            lookback=request.lookback,
            detail=request.detail,
            position_mode=request.position_mode,
            fast_period=request.fast_period,
            slow_period=request.slow_period,
            rsi_length=request.rsi_length,
            oversold=request.oversold,
            overbought=request.overbought,
            max_hold_bars=request.max_hold_bars,
            slippage_bps=request.slippage_bps,
        )
    except Exception as exc:
        log_operation_exception(
            logger,
            operation="strategy_backtest",
            started_at=started_at,
            exc=exc,
            symbol=request.symbol,
            timeframe=request.timeframe,
            strategy=request.strategy,
        )
        raise
    log_operation_finish(
        logger,
        operation="strategy_backtest",
        started_at=started_at,
        success=infer_result_success(result),
        symbol=request.symbol,
        timeframe=request.timeframe,
        strategy=request.strategy,
        lookback=request.lookback,
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

        qerrs = [
            _finite_sample_conformal_quantile(err, float(request.ci_alpha))
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
        result["ci_status"] = "available"
        result["ci_available"] = True
        warnings_out = result.get("warnings")
        if isinstance(warnings_out, list):
            filtered_warnings = [
                item for item in warnings_out if not _is_interval_unavailable_warning(item)
            ]
            if filtered_warnings:
                result["warnings"] = filtered_warnings
            else:
                result.pop("warnings", None)
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
    method_val = normalize_barrier_method(
        request.method or "hmm_mc",
        allow_closed_form=True,
    )
    if method_val is None:
        method_val = str(request.method or "hmm_mc").lower().strip()
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
            result = _apply_barrier_prob_detail(result, request)
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
            input_error = _closed_form_barrier_input_error(request)
            if input_error is not None:
                result = {"error": input_error, "error_code": "invalid_input"}
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
            result = _apply_barrier_prob_detail(result, request)
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

    result = {
        "error": barrier_method_error(request.method, allow_closed_form=True),
        "error_code": "unsupported_method",
    }
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
    method_val = normalize_barrier_method(request.method or "auto", allow_ensemble=True)
    method_supported = method_val is not None
    if method_val is None:
        method_val = str(request.method or "auto").lower().strip()
    log_operation_start(
        logger,
        operation="forecast_barrier_optimize",
        symbol=request.symbol,
        timeframe=request.timeframe,
        method=method_val,
        direction=request.direction,
    )
    if not method_supported:
        result = {
            "error": barrier_method_error(request.method, allow_ensemble=True),
            "error_code": "unsupported_method",
        }
        log_operation_finish(
            logger,
            operation="forecast_barrier_optimize",
            started_at=started_at,
            success=False,
            symbol=request.symbol,
            timeframe=request.timeframe,
            method=method_val,
            direction=request.direction,
        )
        return result
    params_norm = parse_kv_or_json(request.params)
    if not isinstance(params_norm, dict):
        params_norm = {}
    if str(params_norm.get("optimizer", "")).strip().lower() == "optuna":
        optuna_defaults = {
            "sampler": "tpe",
            "pruner": "median",
            "n_jobs": int((cpu_count() or 1)),
        }
        for key, value in optuna_defaults.items():
            if key not in params_norm:
                params_norm[key] = value

    detail_value = _normalize_trader_detail(getattr(request, "detail", "compact"))
    format_value = request.output_mode
    concise_value = request.concise
    return_grid_value = request.return_grid
    field_set = getattr(request, "model_fields_set", set())
    if "detail" in field_set or (
        "output_mode" not in field_set
        and "concise" not in field_set
        and "return_grid" not in field_set
    ):
        if detail_value == "full":
            format_value = "full"
            concise_value = False
        elif detail_value == "standard":
            format_value = "summary"
            concise_value = False
        else:
            format_value = "summary"
            concise_value = True
            return_grid_value = False

    try:
        result = barrier_optimize_impl(
            symbol=request.symbol,
            timeframe=request.timeframe,
            horizon=request.horizon,
            method=method_val,
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
            return_grid=return_grid_value,
            top_k=request.top_k,
            output_mode=format_value,
            viable_only=request.viable_only,
            concise=concise_value,
            grid_style=request.grid_style,
            preset=request.preset,
            vol_window=request.vol_window,
            vol_min_mult=request.vol_min_mult,
            vol_max_mult=request.vol_max_mult,
            vol_steps=request.vol_steps,
            vol_sl_multiplier=request.vol_sl_multiplier,
            vol_floor_pct=request.vol_floor_pct,
            vol_floor_ticks=request.vol_floor_ticks,
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
            statistical_robustness=request.statistical_robustness,
            target_ci_width=request.target_ci_width,
            n_seeds_stability=request.n_seeds_stability,
            enable_bootstrap=request.enable_bootstrap,
            n_bootstrap=request.n_bootstrap,
            enable_convergence_check=request.enable_convergence_check,
            convergence_window=request.convergence_window,
            convergence_threshold=request.convergence_threshold,
            enable_power_analysis=request.enable_power_analysis,
            power_effect_size=request.power_effect_size,
            enable_sensitivity_analysis=request.enable_sensitivity_analysis,
            sensitivity_params=request.sensitivity_params,
        )
        if isinstance(result, dict) and not result.get("error"):
            result = dict(result)
            result["detail"] = detail_value
    except Exception as exc:
        log_operation_exception(
            logger,
            operation="forecast_barrier_optimize",
            started_at=started_at,
            exc=exc,
            symbol=request.symbol,
            timeframe=request.timeframe,
            method=method_val,
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
        method=method_val,
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
            detail=request.detail,
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


def run_forecast_optimize_hints(
    request: ForecastOptimizeHintsRequest,
    *,
    optimize_hints_impl: Any,
) -> Dict[str, Any]:
    """Run genetic search for optimal forecast settings across multiple dimensions.

    Searches across timeframes, methods, parameters, and optionally feature indicators
    to find top-N configurations ranked by composite fitness score.
    """
    started_at = time.perf_counter()
    log_operation_start(
        logger,
        operation="forecast_optimize_hints",
        symbol=request.symbol,
        timeframe=request.timeframe,
        methods=len(request.methods or []),
    )

    # Resolve timeframes to search
    timeframes_to_search = request.timeframes
    if not timeframes_to_search and request.timeframe:
        timeframes_to_search = [request.timeframe]
    if not timeframes_to_search:
        timeframes_to_search = ['H1', 'H4', 'D1']

    try:
        result = optimize_hints_impl(
            symbol=request.symbol,
            timeframes=timeframes_to_search,
            methods=request.methods,
            horizon=int(request.horizon),
            steps=int(request.steps),
            spacing=int(request.spacing),
            fitness_metric=str(request.fitness_metric or 'composite'),
            fitness_weights=request.fitness_weights,
            population=int(request.population),
            generations=int(request.generations),
            crossover_rate=float(request.crossover_rate),
            mutation_rate=float(request.mutation_rate),
            seed=int(request.seed),
            max_search_time_seconds=float(request.max_search_time_seconds)
            if request.max_search_time_seconds is not None
            else None,
            denoise=request.denoise,
            features=request.features,
            dimred_method=request.dimred_method,
            dimred_params=request.dimred_params,
            top_n=int(request.top_n),
            include_feature_genes=bool(request.include_feature_genes),
        )
    except Exception as exc:
        log_operation_exception(
            logger,
            operation="forecast_optimize_hints",
            started_at=started_at,
            exc=exc,
            symbol=request.symbol,
            timeframe=request.timeframe,
        )
        raise
    log_operation_finish(
        logger,
        operation="forecast_optimize_hints",
        started_at=started_at,
        success=infer_result_success(result),
        symbol=request.symbol,
        timeframe=request.timeframe,
        methods=len(request.methods or []),
    )
    return result
