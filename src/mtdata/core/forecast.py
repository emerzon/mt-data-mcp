from typing import Any, Dict, Optional, List, Literal, Tuple
import logging
import time

from .mt5_gateway import create_mt5_gateway
from .schema import TimeframeLiteral, DenoiseSpec, ForecastMethodLiteral
from ._mcp_instance import mcp
from .execution_logging import infer_result_success, log_operation_finish, log_operation_start
from ..forecast.forecast import forecast as _forecast_impl
from ..forecast.exceptions import ForecastError
from ..forecast.backtest import forecast_backtest as _forecast_backtest_impl
from ..forecast.requests import (
    ForecastBacktestRequest,
    ForecastBarrierOptimizeRequest,
    ForecastBarrierProbRequest,
    ForecastConformalIntervalsRequest,
    ForecastGenerateRequest,
    ForecastTuneGeneticRequest,
    ForecastTuneOptunaRequest,
    ForecastVolatilityEstimateRequest,
)
from ..forecast.use_cases import (
    _discover_sktime_forecasters,
    _resolve_sktime_forecaster,
    run_forecast_backtest,
    run_forecast_barrier_optimize,
    run_forecast_barrier_prob,
    run_forecast_conformal_intervals,
    run_forecast_generate,
    run_forecast_tune_genetic,
    run_forecast_tune_optuna,
    run_forecast_volatility_estimate,
)
from ..forecast.volatility import forecast_volatility as _forecast_volatility_impl
from ..forecast.forecast import get_forecast_methods_data as _get_forecast_methods_data
from ..forecast.tune import genetic_search_forecast_params as _genetic_search_impl
from ..forecast.tune import optuna_search_forecast_params as _optuna_search_impl
from ..utils.mt5 import MT5ConnectionError, ensure_mt5_connection_or_raise
from ..utils.utils import parse_kv_or_json as _parse_kv_or_json
from ..utils.barriers import (
    build_barrier_kwargs_from as _build_barrier_kwargs_from,
    normalize_trade_direction as _normalize_trade_direction,
)

logger = logging.getLogger(__name__)


def _get_mt5_gateway():
    return create_mt5_gateway(ensure_connection_impl=ensure_mt5_connection_or_raise)


def _forecast_connection_error() -> Optional[Dict[str, Any]]:
    mt5 = _get_mt5_gateway()
    try:
        mt5.ensure_connection()
    except MT5ConnectionError as exc:
        return {"error": str(exc)}
    return None


def _log_forecast_start(operation: str, **fields: Any) -> float:
    started_at = time.perf_counter()
    log_operation_start(logger, operation=operation, **fields)
    return started_at


def _log_forecast_finish(operation: str, started_at: float, result: Dict[str, Any], **fields: Any) -> Dict[str, Any]:
    log_operation_finish(
        logger,
        operation=operation,
        started_at=started_at,
        success=infer_result_success(result),
        **fields,
    )
    return result

@mcp.tool()
def forecast_generate(request: ForecastGenerateRequest) -> Dict[str, Any]:
    """Generate forecasts for the next `horizon` bars using a selected model.

    Supports native or library-backed models with optional preprocessing.
    Delegates to `mtdata.forecast.forecast`.
    """
    started_at = _log_forecast_start(
        "forecast_generate",
        symbol=request.symbol,
        timeframe=request.timeframe,
        library=request.library,
        model=request.model,
    )
    connection_error = _forecast_connection_error()
    if connection_error is not None:
        return _log_forecast_finish(
            "forecast_generate",
            started_at,
            connection_error,
            symbol=request.symbol,
            timeframe=request.timeframe,
            library=request.library,
            model=request.model,
        )
    try:
        result = run_forecast_generate(
            request,
            forecast_impl=_forecast_impl,
            resolve_sktime_forecaster=_resolve_sktime_forecaster,
        )
        return _log_forecast_finish(
            "forecast_generate",
            started_at,
            result,
            symbol=request.symbol,
            timeframe=request.timeframe,
            library=request.library,
            model=request.model,
        )
    except ForecastError as exc:
        return _log_forecast_finish(
            "forecast_generate",
            started_at,
            {"error": str(exc)},
            symbol=request.symbol,
            timeframe=request.timeframe,
            library=request.library,
            model=request.model,
        )


@mcp.tool()
def forecast_list_library_models(
    library: Literal["native", "statsforecast", "sktime", "pretrained", "mlforecast"],
) -> Dict[str, Any]:
    """List available model names within a forecast library.

    - statsforecast: lists `statsforecast.models.*` class names.
    - sktime: lists supported aliases plus notes for using dotted estimator paths.
    """
    lib = str(library).strip().lower()
    if lib == "native":
        # "Native" methods are mtdata-provided top-level algorithms (as opposed to
        # external-library model spaces like sktime/statsforecast).
        try:
            from mtdata.forecast.forecast_methods import FORECAST_METHODS as _METHODS
        except Exception:
            _METHODS = ()

        excluded = {"statsforecast", "sktime", "mlforecast", "chronos2", "chronos_bolt", "timesfm", "lag_llama"}
        models = [m for m in _METHODS if m not in excluded]
        return {
            "library": lib,
            "models": sorted(models),
            "usage": [
                "mtdata-cli forecast_generate SYMBOL --library native --model analog",
                "mtdata-cli forecast_generate SYMBOL --library native --model theta",
            ],
        }

    if lib == "statsforecast":
        try:
            from statsforecast import models as _models  # type: ignore
        except Exception as ex:
            return {"library": lib, "error": f"statsforecast import failed: {ex}"}
        names: List[str] = []
        for attr in dir(_models):
            if attr.startswith("_"):
                continue
            obj = getattr(_models, attr, None)
            if not isinstance(obj, type):
                continue
            if getattr(obj, "__module__", None) != getattr(_models, "__name__", None):
                continue
            if not any(callable(getattr(obj, a, None)) for a in ("fit", "forecast", "predict")):
                continue
            names.append(attr)
        names = sorted(set(names))
        return {
            "library": lib,
            "models": names,
            "usage": "mtdata-cli forecast_generate SYMBOL --library statsforecast --model AutoARIMA",
        }

    if lib == "sktime":
        mapping = _discover_sktime_forecasters()
        forecasters = sorted({v[0] for v in mapping.values()})
        return {
            "library": lib,
            "models": forecasters,
            "usage": [
                "mtdata-cli forecast_generate SYMBOL --library sktime --model theta",
                "mtdata-cli forecast_generate SYMBOL --library sktime --model ThetaForecaster",
                "mtdata-cli forecast_generate SYMBOL --library sktime --model sktime.forecasting.theta.ThetaForecaster --model-params \"sp=24\"",
            ],
            "note": "The --model value is matched to the closest available forecaster name; you can also pass a dotted class path. Constructor kwargs go in --model-params (or use --set model.<k>=<v>).",
        }

    if lib == "pretrained":
        # These are the pretrained adapters shipped with mtdata.
        pretrained = [
            {
                "model": "chronos2",
                "requires": ["chronos-forecasting>=2.0.0", "torch"],
                "notes": "Hugging Face model id via params.model_name (default: amazon/chronos-bolt-base for compatibility).",
            },
            {
                "model": "chronos_bolt",
                "requires": ["chronos-forecasting>=2.0.0", "torch"],
                "notes": "Same adapter as chronos2; different default naming.",
            },
            {
                "model": "timesfm",
                "requires": ["timesfm", "torch"],
                "notes": "Uses timesfm 2.x (GitHub) API; runs without downloading external weights.",
            },
            {
                "model": "lag_llama",
                "requires": ["lag-llama", "gluonts", "torch"],
                "notes": "May not be installable on Python 3.13 due to upstream pins; included for completeness.",
            },
        ]
        return {
            "library": lib,
            "models": pretrained,
            "usage": [
                "mtdata-cli forecast_generate SYMBOL --library pretrained --model chronos2",
                "mtdata-cli forecast_generate SYMBOL --library pretrained --model timesfm",
            ],
        }

    if lib == "mlforecast":
        return {
            "library": lib,
            "note": "Use `--model <dotted sklearn/lightgbm regressor class>` plus optional constructor kwargs in --model-params (or use --set model.<k>=<v>).",
            "usage": [
                "mtdata-cli forecast_generate SYMBOL --library mlforecast --model sklearn.ensemble.RandomForestRegressor --model-params \"n_estimators=200\"",
                "mtdata-cli forecast_generate SYMBOL --library native --model mlf_rf",
            ],
        }

    return {"library": lib, "error": "Unsupported library (supported: native, statsforecast, sktime, pretrained, mlforecast)"}


@mcp.tool()
def forecast_backtest_run(request: ForecastBacktestRequest) -> Dict[str, Any]:
    """Rolling-origin backtest over historical anchors using the forecast tool."""
    started_at = _log_forecast_start(
        "forecast_backtest_run",
        symbol=request.symbol,
        timeframe=request.timeframe,
        horizon=request.horizon,
        detail=request.detail,
    )
    connection_error = _forecast_connection_error()
    if connection_error is not None:
        return _log_forecast_finish(
            "forecast_backtest_run",
            started_at,
            connection_error,
            symbol=request.symbol,
            timeframe=request.timeframe,
            horizon=request.horizon,
            detail=request.detail,
        )
    result = run_forecast_backtest(request, backtest_impl=_forecast_backtest_impl)
    return _log_forecast_finish(
        "forecast_backtest_run",
        started_at,
        result,
        symbol=request.symbol,
        timeframe=request.timeframe,
        horizon=request.horizon,
        detail=request.detail,
    )


@mcp.tool()
def forecast_volatility_estimate(
    request: ForecastVolatilityEstimateRequest,
) -> Dict[str, Any]:
    """Forecast volatility over `horizon` bars using direct estimators or proxies."""
    started_at = _log_forecast_start(
        "forecast_volatility_estimate",
        symbol=request.symbol,
        timeframe=request.timeframe,
        horizon=request.horizon,
        method=request.method,
    )
    connection_error = _forecast_connection_error()
    if connection_error is not None:
        return _log_forecast_finish(
            "forecast_volatility_estimate",
            started_at,
            connection_error,
            symbol=request.symbol,
            timeframe=request.timeframe,
            horizon=request.horizon,
            method=request.method,
        )
    result = run_forecast_volatility_estimate(
        request,
        forecast_volatility_impl=_forecast_volatility_impl,
    )
    return _log_forecast_finish(
        "forecast_volatility_estimate",
        started_at,
        result,
        symbol=request.symbol,
        timeframe=request.timeframe,
        horizon=request.horizon,
        method=request.method,
    )


@mcp.tool()
def forecast_list_methods(
    detail: Literal["compact", "full"] = "compact",  # type: ignore
    limit: Optional[int] = None,
    search: Optional[str] = None,
) -> Dict[str, Any]:
    """List forecast methods and availability.

    - detail='compact' (default): concise list suitable for terminal usage.
    - detail='full': include full parameter docs and supports metadata.
    """
    try:
        data = _get_forecast_methods_data()
        detail_value = str(detail or "compact").strip().lower()
        search_value = str(search or "").strip().lower()
        limit_value: Optional[int] = None
        if limit is not None:
            try:
                limit_value = int(limit)
            except Exception:
                return {"error": f"Invalid limit: {limit}. Must be a positive integer."}
            if limit_value <= 0:
                return {"error": f"Invalid limit: {limit_value}. Must be >= 1."}

        categories_raw = data.get("categories") if isinstance(data.get("categories"), dict) else {}
        method_to_category: Dict[str, str] = {}
        if isinstance(categories_raw, dict):
            for cat_name, names in categories_raw.items():
                if not isinstance(names, list):
                    continue
                for name in names:
                    if name is None:
                        continue
                    method_to_category[str(name)] = str(cat_name)

        def _method_matches(item: Dict[str, Any]) -> bool:
            if not search_value:
                return True
            method_name = str(item.get("method") or "")
            desc = str(item.get("description") or "")
            cat = method_to_category.get(method_name, "")
            haystack = " ".join((method_name, desc, cat)).lower()
            return search_value in haystack

        def _namespace_info(method_name: str, category: str) -> Tuple[str, str, str]:
            method_norm = str(method_name or "").strip()
            cat_norm = str(category or "").strip().lower()
            if method_norm.startswith("sf_") or cat_norm == "statsforecast":
                concept = method_norm[3:] if method_norm.startswith("sf_") else method_norm
                return "statsforecast", concept, f"statsforecast:{concept}"
            if method_norm.startswith("skt_") or cat_norm == "sktime":
                concept = method_norm[4:] if method_norm.startswith("skt_") else method_norm
                return "sktime", concept, f"sktime:{concept}"
            pretrained_names = {"chronos2", "chronos_bolt", "timesfm", "lag_llama"}
            if method_norm in pretrained_names or cat_norm == "pretrained":
                return "pretrained", method_norm, f"pretrained:{method_norm}"
            if method_norm.startswith("mlf_") or cat_norm in {"mlforecast", "ml"}:
                return "mlforecast", method_norm, f"mlforecast:{method_norm}"
            return "native", method_norm, f"native:{method_norm}"

        if detail_value == "full":
            methods_full = data.get("methods")
            if not isinstance(methods_full, list):
                return data
            enriched_full: List[Dict[str, Any]] = []
            for row in methods_full:
                if not isinstance(row, dict):
                    continue
                method_name = str(row.get("method") or "")
                category = method_to_category.get(method_name, "other")
                namespace, concept, method_id = _namespace_info(method_name, category)
                row_out = dict(row)
                row_out["category"] = category
                row_out["namespace"] = namespace
                row_out["concept"] = concept
                row_out["method_id"] = method_id
                enriched_full.append(row_out)
            filtered_full = [row for row in enriched_full if _method_matches(row)]
            if limit_value is not None:
                filtered_full = filtered_full[:limit_value]
            out_full = dict(data)
            out_full["methods"] = filtered_full
            out_full["total"] = len(filtered_full)
            out_full["filters"] = {
                "search": search_value or None,
                "limit": limit_value,
            }
            out_full["note"] = (
                "Methods include namespace/concept/method_id fields to disambiguate similarly named implementations "
                "(for example native:theta vs statsforecast:theta)."
            )
            return out_full
        methods = data.get("methods")
        if not isinstance(methods, list):
            return data

        if any(not isinstance(item, dict) for item in methods):
            return data

        compact_methods: List[Dict[str, Any]] = []
        available_count = 0
        unavailable_count = 0
        by_category: Dict[str, List[Dict[str, Any]]] = {}
        for item in methods:
            if not _method_matches(item):
                continue
            name = item.get("method")
            if name in (None, ""):
                continue
            method_name = str(name)
            available = bool(item.get("available"))
            if available:
                available_count += 1
            else:
                unavailable_count += 1

            row: Dict[str, Any] = {
                "method": method_name,
                "available": available,
            }
            desc = str(item.get("description") or "").strip()
            if desc:
                row["description"] = desc.splitlines()[0].strip()
            cat = method_to_category.get(method_name)
            row["category"] = cat or "other"
            namespace, concept, method_id = _namespace_info(method_name, row["category"])
            row["namespace"] = namespace
            row["concept"] = concept
            row["method_id"] = method_id
            params = item.get("params")
            if isinstance(params, list):
                row["params_count"] = len(params)
            compact_methods.append(row)
            by_category.setdefault(str(row["category"]), []).append(row)

        category_summary: List[Dict[str, Any]] = []
        selected_methods: List[Dict[str, Any]] = []
        for category in sorted(by_category.keys()):
            rows = list(by_category.get(category, []))
            rows.sort(key=lambda row: (not bool(row.get("available")), str(row.get("method"))))
            n_total = len(rows)
            n_available = int(sum(1 for row in rows if bool(row.get("available"))))
            per_category_cap = 3 if category == "statsforecast" else (8 if category == "other" else 2)
            picks = rows[:per_category_cap]
            selected_methods.extend(picks)
            category_summary.append(
                {
                    "category": category,
                    "total": n_total,
                    "available": n_available,
                    "unavailable": int(n_total - n_available),
                    "examples": [str(row.get("method")) for row in picks],
                    "hidden": int(max(0, n_total - len(picks))),
                }
            )

        selected_methods.sort(
            key=lambda row: (
                str(row.get("category")) == "other",
                str(row.get("category")),
                not bool(row.get("available")),
                str(row.get("method")),
            )
        )
        if limit_value is not None:
            selected_methods = selected_methods[:limit_value]
        return {
            "detail": "compact",
            "total": int(data.get("total") or len(compact_methods)),
            "total_filtered": int(len(compact_methods)),
            "available": available_count,
            "unavailable": unavailable_count,
            "categories": categories_raw,
            "category_summary": category_summary,
            "methods": selected_methods,
            "methods_shown": int(len(selected_methods)),
            "methods_hidden": int(max(0, len(compact_methods) - len(selected_methods))),
            "note": "Compact view groups methods by category and shows a small representative subset. Pass detail='full' for exhaustive docs.",
            "filters": {
                "search": search_value or None,
                "limit": limit_value,
            },
        }
    except Exception as e:
        return {"error": f"Error listing forecast methods: {e}"}


@mcp.tool()
def forecast_conformal_intervals(request: ForecastConformalIntervalsRequest) -> Dict[str, Any]:
    """Conformalized forecast intervals via rolling-origin calibration.

    - Calibrates per-step absolute residual quantiles using `steps` historical anchors (spaced by `spacing`).
    - Returns point forecast (from `method`) and conformal bands per step.
    """
    started_at = _log_forecast_start(
        "forecast_conformal_intervals",
        symbol=request.symbol,
        timeframe=request.timeframe,
        method=request.method,
        horizon=request.horizon,
    )
    connection_error = _forecast_connection_error()
    if connection_error is not None:
        return _log_forecast_finish(
            "forecast_conformal_intervals",
            started_at,
            connection_error,
            symbol=request.symbol,
            timeframe=request.timeframe,
            method=request.method,
            horizon=request.horizon,
        )
    try:
        result = run_forecast_conformal_intervals(
            request,
            backtest_impl=_forecast_backtest_impl,
            forecast_impl=_forecast_impl,
        )
        return _log_forecast_finish(
            "forecast_conformal_intervals",
            started_at,
            result,
            symbol=request.symbol,
            timeframe=request.timeframe,
            method=request.method,
            horizon=request.horizon,
        )
    except ForecastError as exc:
        return _log_forecast_finish(
            "forecast_conformal_intervals",
            started_at,
            {"error": str(exc)},
            symbol=request.symbol,
            timeframe=request.timeframe,
            method=request.method,
            horizon=request.horizon,
        )
    except Exception as e:
        return _log_forecast_finish(
            "forecast_conformal_intervals",
            started_at,
            {"error": f"Error computing conformal forecast: {str(e)}"},
            symbol=request.symbol,
            timeframe=request.timeframe,
            method=request.method,
            horizon=request.horizon,
        )


@mcp.tool()
def forecast_tune_genetic(request: ForecastTuneGeneticRequest) -> Dict[str, Any]:
    """Genetic search over method params to optimize a backtest metric.

    - search_space: dict or JSON like {param: {type, min, max, choices?, log?}}
    - metric: e.g., 'avg_rmse', 'avg_mae', 'avg_directional_accuracy'
    - mode: 'min' or 'max'
    """
    started_at = _log_forecast_start(
        "forecast_tune_genetic",
        symbol=request.symbol,
        timeframe=request.timeframe,
        method=request.method,
        metric=request.metric,
    )
    connection_error = _forecast_connection_error()
    if connection_error is not None:
        return _log_forecast_finish(
            "forecast_tune_genetic",
            started_at,
            connection_error,
            symbol=request.symbol,
            timeframe=request.timeframe,
            method=request.method,
            metric=request.metric,
        )
    try:
        result = run_forecast_tune_genetic(
            request,
            genetic_search_impl=_genetic_search_impl,
        )
        return _log_forecast_finish(
            "forecast_tune_genetic",
            started_at,
            result,
            symbol=request.symbol,
            timeframe=request.timeframe,
            method=request.method,
            metric=request.metric,
        )
    except Exception as e:
        return _log_forecast_finish(
            "forecast_tune_genetic",
            started_at,
            {"error": f"Error in genetic tuning: {e}"},
            symbol=request.symbol,
            timeframe=request.timeframe,
            method=request.method,
            metric=request.metric,
        )


@mcp.tool()
def forecast_tune_optuna(request: ForecastTuneOptunaRequest) -> Dict[str, Any]:
    """Optuna search over method params to optimize a backtest metric."""
    started_at = _log_forecast_start(
        "forecast_tune_optuna",
        symbol=request.symbol,
        timeframe=request.timeframe,
        method=request.method,
        metric=request.metric,
    )
    connection_error = _forecast_connection_error()
    if connection_error is not None:
        return _log_forecast_finish(
            "forecast_tune_optuna",
            started_at,
            connection_error,
            symbol=request.symbol,
            timeframe=request.timeframe,
            method=request.method,
            metric=request.metric,
        )
    try:
        result = run_forecast_tune_optuna(
            request,
            optuna_search_impl=_optuna_search_impl,
        )
        return _log_forecast_finish(
            "forecast_tune_optuna",
            started_at,
            result,
            symbol=request.symbol,
            timeframe=request.timeframe,
            method=request.method,
            metric=request.metric,
        )
    except Exception as e:
        return _log_forecast_finish(
            "forecast_tune_optuna",
            started_at,
            {"error": f"Error in optuna tuning: {e}"},
            symbol=request.symbol,
            timeframe=request.timeframe,
            method=request.method,
            metric=request.metric,
        )


@mcp.tool()
def forecast_options_expirations(
    symbol: str,
) -> Dict[str, Any]:
    """Fetch available option expirations for a symbol."""
    from ..services.options_service import get_options_expirations as _impl
    return _impl(symbol=symbol)


@mcp.tool()
def forecast_options_chain(
    symbol: str,
    expiration: Optional[str] = None,
    option_type: Literal["call", "put", "both"] = "both",  # type: ignore
    min_open_interest: int = 0,
    min_volume: int = 0,
    limit: int = 200,
) -> Dict[str, Any]:
    """Fetch options chain snapshots for a symbol."""
    from ..services.options_service import get_options_chain as _impl
    return _impl(
        symbol=symbol,
        expiration=expiration,
        option_type=option_type,
        min_open_interest=int(min_open_interest),
        min_volume=int(min_volume),
        limit=int(limit),
    )


@mcp.tool()
def forecast_quantlib_barrier_price(
    spot: float,
    strike: float,
    barrier: float,
    maturity_days: int,
    option_type: Literal["call", "put"] = "call",  # type: ignore
    barrier_type: Literal["up_in", "up_out", "down_in", "down_out"] = "up_out",  # type: ignore
    risk_free_rate: float = 0.02,
    dividend_yield: float = 0.0,
    volatility: float = 0.2,
    rebate: float = 0.0,
) -> Dict[str, Any]:
    """Price a barrier option using QuantLib."""
    from ..forecast.quantlib_tools import price_barrier_option_quantlib as _impl
    return _impl(
        spot=float(spot),
        strike=float(strike),
        barrier=float(barrier),
        maturity_days=int(maturity_days),
        option_type=option_type,
        barrier_type=barrier_type,
        risk_free_rate=float(risk_free_rate),
        dividend_yield=float(dividend_yield),
        volatility=float(volatility),
        rebate=float(rebate),
    )


@mcp.tool()
def forecast_quantlib_heston_calibrate(
    symbol: str,
    expiration: Optional[str] = None,
    option_type: Literal["call", "put", "both"] = "call",  # type: ignore
    risk_free_rate: float = 0.02,
    dividend_yield: float = 0.0,
    min_open_interest: int = 0,
    min_volume: int = 0,
    max_contracts: int = 25,
) -> Dict[str, Any]:
    """Calibrate Heston parameters from an option chain using QuantLib."""
    from ..forecast.quantlib_tools import calibrate_heston_quantlib_from_options as _impl
    return _impl(
        symbol=symbol,
        expiration=expiration,
        option_type=option_type,
        risk_free_rate=float(risk_free_rate),
        dividend_yield=float(dividend_yield),
        min_open_interest=int(min_open_interest),
        min_volume=int(min_volume),
        max_contracts=int(max_contracts),
    )


@mcp.tool()
def forecast_barrier_prob(
    request: ForecastBarrierProbRequest,
) -> Dict[str, Any]:
    """Calculate probability of price hitting TP/SL barriers using Monte Carlo or Closed Form methods.
    
    **REQUIRED**: symbol parameter must be provided
    
    Use Cases:
    ----------
    - Validate TP/SL levels before entering a trade
    - Assess probability of hitting profit target vs stop loss
    - Optimize barrier levels based on probability analysis
    
    Parameters:
    -----------
    symbol : str (REQUIRED)
        Trading symbol to analyze (e.g., "EURUSD", "BTCUSD")
    
    timeframe : str, optional (default="H1")
        Analysis timeframe: "M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1", "MN1"
    
    horizon : int, optional (default=12)
        Number of bars to forecast ahead
    
    method : str, optional (default="mc")
        Calculation method:
        - "mc": Monte Carlo simulation (more flexible, handles complex scenarios)
        - "closed_form": Analytical solution (faster, simpler assumptions)
        - "auto": Alias for Monte Carlo with mc_method="auto"
    
    Monte Carlo Parameters (method="mc"):
    -------------------------------------
    mc_method : str, optional (default="hmm_mc")
        - "hmm_mc": Hidden Markov Model-based MC
        - "mc_gbm": Geometric Brownian Motion MC
        - "mc_gbm_bb": GBM MC with Brownian-bridge barrier correction
        - "heston": Heston stochastic volatility MC
        - "jump_diffusion": Merton jump-diffusion MC
        - "garch": GARCH(1,1) volatility model (requires 'arch' package)
        - "bootstrap": Circular block bootstrap (historical simulation)
        - "auto": auto-select based on symbol/timeframe + recent returns

    direction : str, optional (default="long")
        Trade direction: "long" / "short"
    
    tp_abs : float, optional
        Absolute take profit price level
    
    sl_abs : float, optional
        Absolute stop loss price level
    
    tp_pct : float, optional
        Take profit as percentage (e.g., 2.0 for 2%)
    
    sl_pct : float, optional
        Stop loss as percentage
    
    tp_pips : float, optional
        Take profit in ticks (trade_tick_size)
    
    sl_pips : float, optional
        Stop loss in ticks (trade_tick_size)
    
    Closed Form Parameters (method="closed_form"):
    ----------------------------------------------
    barrier : float, optional (default=0.0)
        Target barrier level
    
    mu : float, optional
        Drift parameter (calculated if not provided)
    
    sigma : float, optional
        Volatility parameter (calculated if not provided)
    
    Returns:
    --------
    dict
        Probability analysis including:
        - success: bool
        - symbol: str
        - probabilities: dict with TP/SL hit probabilities
        - method_used: str
    
    Examples:
    ---------
    # Check probability of hitting TP vs SL (Monte Carlo)
    forecast_barrier_prob(
        symbol="EURUSD",
        method="mc",
        direction="long",
        tp_abs=1.1100,
        sl_abs=1.0950
    )
    
    # Use percentage-based barriers
    forecast_barrier_prob(
        symbol="EURUSD",
        direction="long",
        tp_pct=2.0,
        sl_pct=1.0
    )
    
    # Closed form calculation (faster)
    forecast_barrier_prob(
        symbol="GBPUSD",
        method="closed_form",
        direction="long",
        barrier=1.2700
    )
    """
    started_at = _log_forecast_start(
        "forecast_barrier_prob",
        symbol=request.symbol,
        timeframe=request.timeframe,
        method=request.method,
        direction=request.direction,
    )
    connection_error = _forecast_connection_error()
    if connection_error is not None:
        return _log_forecast_finish(
            "forecast_barrier_prob",
            started_at,
            connection_error,
            symbol=request.symbol,
            timeframe=request.timeframe,
            method=request.method,
            direction=request.direction,
        )
    from ..forecast.barriers import (
        forecast_barrier_closed_form as _barrier_closed_form_impl,
        forecast_barrier_hit_probabilities as _barrier_hit_probabilities_impl,
    )

    result = run_forecast_barrier_prob(
        request,
        build_barrier_kwargs=_build_barrier_kwargs_from,
        normalize_trade_direction=_normalize_trade_direction,
        barrier_hit_probabilities_impl=_barrier_hit_probabilities_impl,
        barrier_closed_form_impl=_barrier_closed_form_impl,
    )
    return _log_forecast_finish(
        "forecast_barrier_prob",
        started_at,
        result,
        symbol=request.symbol,
        timeframe=request.timeframe,
        method=request.method,
        direction=request.direction,
    )


@mcp.tool()
def forecast_barrier_optimize(
    request: ForecastBarrierOptimizeRequest,
) -> Dict[str, Any]:
    """Optimize TP/SL barriers with support for presets, volatility scaling, ratios, and two-stage refinement."""
    started_at = _log_forecast_start(
        "forecast_barrier_optimize",
        symbol=request.symbol,
        timeframe=request.timeframe,
        method=request.method,
        direction=request.direction,
    )
    connection_error = _forecast_connection_error()
    if connection_error is not None:
        return _log_forecast_finish(
            "forecast_barrier_optimize",
            started_at,
            connection_error,
            symbol=request.symbol,
            timeframe=request.timeframe,
            method=request.method,
            direction=request.direction,
        )
    from ..forecast.barriers import forecast_barrier_optimize as _barrier_optimize_impl

    result = run_forecast_barrier_optimize(
        request,
        parse_kv_or_json=_parse_kv_or_json,
        barrier_optimize_impl=_barrier_optimize_impl,
    )
    return _log_forecast_finish(
        "forecast_barrier_optimize",
        started_at,
        result,
        symbol=request.symbol,
        timeframe=request.timeframe,
        method=request.method,
        direction=request.direction,
    )
