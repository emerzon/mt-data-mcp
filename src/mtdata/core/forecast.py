import logging
from functools import lru_cache
from importlib import import_module
from typing import Any, Dict, List, Literal, Optional

from ..forecast.exceptions import ForecastError
from ..forecast.requests import (
    ForecastBacktestRequest,
    ForecastBarrierOptimizeRequest,
    ForecastBarrierProbRequest,
    ForecastConformalIntervalsRequest,
    ForecastGenerateRequest,
    ForecastTuneGeneticRequest,
    ForecastTuneOptunaRequest,
    ForecastVolatilityEstimateRequest,
    StrategyBacktestRequest,
)
from ..utils.barriers import (
    build_barrier_kwargs_from as _build_barrier_kwargs_from,
)
from ..utils.barriers import (
    normalize_trade_direction,
)
from ..utils.mt5 import ensure_mt5_connection_or_raise
from ..utils.utils import parse_kv_or_json as _parse_kv_or_json
from ._mcp_instance import mcp
from .execution_logging import run_logged_operation
from .mt5_gateway import get_mt5_gateway, mt5_connection_error

logger = logging.getLogger(__name__)


def _forecast_module():
    return import_module("mtdata.forecast.forecast")


def _forecast_backtest_module():
    return import_module("mtdata.forecast.backtest")


def _forecast_use_cases_module():
    return import_module("mtdata.forecast.use_cases")


def _forecast_volatility_module():
    return import_module("mtdata.forecast.volatility")


def _forecast_tune_module():
    return import_module("mtdata.forecast.tune")


def _forecast_capabilities_module():
    return import_module("mtdata.forecast.capabilities")


def _forecast_impl(**kwargs):
    module = _forecast_module()
    func = getattr(module, "execute_forecast", module.forecast)
    return func(**kwargs)


def _forecast_backtest_impl(**kwargs):
    module = _forecast_backtest_module()
    func = getattr(module, "execute_forecast_backtest", module.forecast_backtest)
    return func(**kwargs)


def _strategy_backtest_impl(**kwargs):
    module = _forecast_backtest_module()
    func = getattr(module, "execute_strategy_backtest", module.strategy_backtest)
    return func(**kwargs)


def _forecast_volatility_impl(**kwargs):
    return _forecast_volatility_module().forecast_volatility(**kwargs)


def _get_forecast_methods_data():
    return _forecast_module().get_forecast_methods_data()


@lru_cache(maxsize=1)
def _get_registered_forecast_capabilities():
    return _forecast_capabilities_module().get_registered_capabilities()


def _get_library_forecast_capabilities(*args, **kwargs):
    return _forecast_capabilities_module().get_library_capabilities(*args, **kwargs)


def _genetic_search_impl(**kwargs):
    return _forecast_tune_module().genetic_search_forecast_params(**kwargs)


def _optuna_search_impl(**kwargs):
    return _forecast_tune_module().optuna_search_forecast_params(**kwargs)


def _discover_sktime_forecasters():
    return _forecast_use_cases_module()._discover_sktime_forecasters()


def _clear_discover_sktime_forecasters_cache() -> None:
    func = getattr(_forecast_use_cases_module(), "_discover_sktime_forecasters", None)
    cache_clear = getattr(func, "cache_clear", None)
    if callable(cache_clear):
        cache_clear()


_discover_sktime_forecasters.cache_clear = _clear_discover_sktime_forecasters_cache


def _resolve_sktime_forecaster(*args, **kwargs):
    return _forecast_use_cases_module()._resolve_sktime_forecaster(*args, **kwargs)


def run_forecast_generate(*args, **kwargs):
    return _forecast_use_cases_module().run_forecast_generate(*args, **kwargs)


def run_forecast_backtest(*args, **kwargs):
    return _forecast_use_cases_module().run_forecast_backtest(*args, **kwargs)


def run_strategy_backtest(*args, **kwargs):
    return _forecast_use_cases_module().run_strategy_backtest(*args, **kwargs)


def run_forecast_conformal_intervals(*args, **kwargs):
    return _forecast_use_cases_module().run_forecast_conformal_intervals(*args, **kwargs)


def run_forecast_tune_genetic(*args, **kwargs):
    return _forecast_use_cases_module().run_forecast_tune_genetic(*args, **kwargs)


def run_forecast_tune_optuna(*args, **kwargs):
    return _forecast_use_cases_module().run_forecast_tune_optuna(*args, **kwargs)


def run_forecast_barrier_prob(*args, **kwargs):
    return _forecast_use_cases_module().run_forecast_barrier_prob(*args, **kwargs)


def run_forecast_barrier_optimize(*args, **kwargs):
    return _forecast_use_cases_module().run_forecast_barrier_optimize(*args, **kwargs)


def run_forecast_volatility_estimate(*args, **kwargs):
    return _forecast_use_cases_module().run_forecast_volatility_estimate(*args, **kwargs)


def _forecast_connection_error() -> Optional[Dict[str, Any]]:
    return mt5_connection_error(
        get_mt5_gateway(ensure_connection_impl=ensure_mt5_connection_or_raise),
    )


def _run_forecast_operation(
    operation: str,
    *,
    func,
    require_connection: bool = False,
    generic_error_prefix: Optional[str] = None,
    catch_forecast_error: bool = False,
    **fields: Any,
) -> Dict[str, Any]:
    def _wrapped() -> Dict[str, Any]:
        if require_connection:
            connection_error = _forecast_connection_error()
            if connection_error is not None:
                return connection_error
        try:
            return func()
        except ForecastError as exc:
            if catch_forecast_error:
                return {"error": str(exc)}
            raise
        except Exception as exc:
            if generic_error_prefix is not None:
                return {"error": f"{generic_error_prefix}{exc}"}
            raise

    return run_logged_operation(
        logger,
        operation=operation,
        func=_wrapped,
        **fields,
    )

@mcp.tool()
def forecast_generate(request: ForecastGenerateRequest) -> Dict[str, Any]:
    """Generate forecasts for the next `horizon` bars using a selected method.

    Supports native or library-backed methods with optional preprocessing.
    Delegates to `mtdata.forecast.forecast`.
    """
    def _execute() -> Dict[str, Any]:
        return run_forecast_generate(
            request,
            forecast_impl=_forecast_impl,
            resolve_sktime_forecaster=_resolve_sktime_forecaster,
            log_events=False,
        )

    return _run_forecast_operation(
        "forecast_generate",
        symbol=request.symbol,
        timeframe=request.timeframe,
        library=request.library,
        method=request.method,
        require_connection=True,
        catch_forecast_error=True,
        func=_execute,
    )


@mcp.tool()
def forecast_list_library_models(
    library: Literal["native", "statsforecast", "sktime", "pretrained", "mlforecast"],
) -> Dict[str, Any]:
    """List available model names within a forecast library.

    - statsforecast: lists `statsforecast.models.*` class names.
    - sktime: lists supported aliases plus notes for using dotted estimator paths.
    """
    return _run_forecast_operation(
        "forecast_list_library_models",
        library=library,
        func=lambda: _forecast_list_library_models_impl(library),
    )


@mcp.tool()
def forecast_backtest_run(request: ForecastBacktestRequest) -> Dict[str, Any]:
    """Rolling-origin backtest over historical anchors using the forecast tool."""
    def _execute() -> Dict[str, Any]:
        return run_forecast_backtest(
            request,
            backtest_impl=_forecast_backtest_impl,
        )

    return _run_forecast_operation(
        "forecast_backtest_run",
        symbol=request.symbol,
        timeframe=request.timeframe,
        horizon=request.horizon,
        detail=request.detail,
        require_connection=True,
        func=_execute,
    )


@mcp.tool()
def strategy_backtest(request: StrategyBacktestRequest) -> Dict[str, Any]:
    """Backtest simple indicator-driven trading strategies on MT5 candle history."""
    def _execute() -> Dict[str, Any]:
        return run_strategy_backtest(
            request,
            strategy_backtest_impl=_strategy_backtest_impl,
        )

    return _run_forecast_operation(
        "strategy_backtest",
        symbol=request.symbol,
        timeframe=request.timeframe,
        strategy=request.strategy,
        require_connection=True,
        func=_execute,
    )


@mcp.tool()
def forecast_volatility_estimate(
    request: ForecastVolatilityEstimateRequest,
) -> Dict[str, Any]:
    """Forecast volatility over `horizon` bars using direct estimators or proxies."""
    def _execute() -> Dict[str, Any]:
        return run_forecast_volatility_estimate(
            request,
            forecast_volatility_impl=_forecast_volatility_impl,
        )

    return _run_forecast_operation(
        "forecast_volatility_estimate",
        symbol=request.symbol,
        timeframe=request.timeframe,
        horizon=request.horizon,
        method=request.method,
        require_connection=True,
        func=_execute,
    )


@mcp.tool()
def forecast_list_methods(
    detail: Literal["compact", "full"] = "compact",  # type: ignore
    limit: Optional[int] = None,
    search: Optional[str] = None,
) -> Dict[str, Any]:
    """List forecast methods and availability.

    - detail='compact' (default): concise list with availability and `supports_ci` guidance.
    - detail='full': include full parameter docs and supports metadata.
    """
    return _run_forecast_operation(
        "forecast_list_methods",
        detail=detail,
        limit=limit,
        search=search,
        func=lambda: _forecast_list_methods_impl(detail=detail, limit=limit, search=search),
    )


@mcp.tool()
def forecast_conformal_intervals(request: ForecastConformalIntervalsRequest) -> Dict[str, Any]:
    """Conformalized forecast intervals via rolling-origin calibration.

    - Calibrates per-step absolute residual quantiles using `steps` historical anchors (spaced by `spacing`).
    - Returns point forecast (from `method`) and conformal bands per step.
    """
    def _execute() -> Dict[str, Any]:
        return run_forecast_conformal_intervals(
            request,
            backtest_impl=_forecast_backtest_impl,
            forecast_impl=_forecast_impl,
        )

    return _run_forecast_operation(
        "forecast_conformal_intervals",
        symbol=request.symbol,
        timeframe=request.timeframe,
        method=request.method,
        horizon=request.horizon,
        require_connection=True,
        catch_forecast_error=True,
        generic_error_prefix="Error computing conformal forecast: ",
        func=_execute,
    )


@mcp.tool()
def forecast_tune_genetic(request: ForecastTuneGeneticRequest) -> Dict[str, Any]:
    """Genetic search over method params to optimize a backtest metric.

    - search_space: dict or JSON like {param: {type, min, max, choices?, log?}}
    - metric: e.g., 'avg_rmse', 'avg_mae', 'avg_directional_accuracy'
    - mode: 'min' or 'max'
    """
    def _execute() -> Dict[str, Any]:
        return run_forecast_tune_genetic(
            request,
            genetic_search_impl=_genetic_search_impl,
        )

    return _run_forecast_operation(
        "forecast_tune_genetic",
        symbol=request.symbol,
        timeframe=request.timeframe,
        method=request.method,
        metric=request.metric,
        require_connection=True,
        generic_error_prefix="Error in genetic tuning: ",
        func=_execute,
    )


@mcp.tool()
def forecast_tune_optuna(request: ForecastTuneOptunaRequest) -> Dict[str, Any]:
    """Optuna search over method params to optimize a backtest metric."""
    def _execute() -> Dict[str, Any]:
        return run_forecast_tune_optuna(
            request,
            optuna_search_impl=_optuna_search_impl,
        )

    return _run_forecast_operation(
        "forecast_tune_optuna",
        symbol=request.symbol,
        timeframe=request.timeframe,
        method=request.method,
        metric=request.metric,
        require_connection=True,
        generic_error_prefix="Error in optuna tuning: ",
        func=_execute,
    )


@mcp.tool()
def forecast_options_expirations(
    symbol: str,
) -> Dict[str, Any]:
    """Fetch option expirations via Yahoo Finance; provider availability/auth can change."""
    from ..services.options_service import get_options_expirations as _impl
    return _run_forecast_operation(
        "forecast_options_expirations",
        symbol=symbol,
        func=lambda: _impl(symbol=symbol),
    )


@mcp.tool()
def forecast_options_chain(
    symbol: str,
    expiration: Optional[str] = None,
    option_type: Literal["call", "put", "both"] = "both",  # type: ignore
    min_open_interest: int = 0,
    min_volume: int = 0,
    limit: int = 200,
) -> Dict[str, Any]:
    """Fetch option-chain snapshots via Yahoo Finance; provider availability/auth can change."""
    from ..services.options_service import get_options_chain as _impl
    return _run_forecast_operation(
        "forecast_options_chain",
        symbol=symbol,
        expiration=expiration,
        option_type=option_type,
        limit=limit,
        func=lambda: _impl(
            symbol=symbol,
            expiration=expiration,
            option_type=option_type,
            min_open_interest=int(min_open_interest),
            min_volume=int(min_volume),
            limit=int(limit),
        ),
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
    return _run_forecast_operation(
        "forecast_quantlib_barrier_price",
        option_type=option_type,
        barrier_type=barrier_type,
        maturity_days=maturity_days,
        func=lambda: _impl(
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
        ),
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
    from ..forecast.quantlib_tools import (
        calibrate_heston_quantlib_from_options as _impl,
    )
    return _run_forecast_operation(
        "forecast_quantlib_heston_calibrate",
        symbol=symbol,
        expiration=expiration,
        option_type=option_type,
        max_contracts=max_contracts,
        func=lambda: _impl(
            symbol=symbol,
            expiration=expiration,
            option_type=option_type,
            risk_free_rate=float(risk_free_rate),
            dividend_yield=float(dividend_yield),
            min_open_interest=int(min_open_interest),
            min_volume=int(min_volume),
            max_contracts=int(max_contracts),
        ),
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
    
    method : str, optional (default="hmm_mc")
        Calculation method:
        - "hmm_mc": Hidden Markov Model Monte Carlo
        - "mc_gbm": Geometric Brownian Motion Monte Carlo
        - "mc_gbm_bb": GBM Monte Carlo with Brownian-bridge correction
        - "garch": GARCH(1,1) Monte Carlo
        - "bootstrap": Historical bootstrap Monte Carlo
        - "heston": Heston stochastic-volatility Monte Carlo
        - "jump_diffusion": Merton jump-diffusion Monte Carlo
        - "closed_form": Analytical solution (faster, simpler assumptions)
        - "auto": Auto-select a Monte Carlo engine

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
        method="hmm_mc",
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
    def _execute() -> Dict[str, Any]:
        from ..forecast.barriers import (
            forecast_barrier_closed_form as _barrier_closed_form_impl,
        )
        from ..forecast.barriers import (
            forecast_barrier_hit_probabilities as _barrier_hit_probabilities_impl,
        )

        return run_forecast_barrier_prob(
            request,
            build_barrier_kwargs=_build_barrier_kwargs_from,
            normalize_trade_direction=normalize_trade_direction,
            barrier_hit_probabilities_impl=_barrier_hit_probabilities_impl,
            barrier_closed_form_impl=_barrier_closed_form_impl,
        )

    return _run_forecast_operation(
        "forecast_barrier_prob",
        symbol=request.symbol,
        timeframe=request.timeframe,
        method=request.method,
        direction=request.direction,
        require_connection=True,
        func=_execute,
    )


@mcp.tool()
def forecast_barrier_optimize(
    request: ForecastBarrierOptimizeRequest,
) -> Dict[str, Any]:
    """Optimize TP/SL barriers with support for presets, volatility scaling, ratios, and two-stage refinement."""
    def _execute() -> Dict[str, Any]:
        from ..forecast.barriers import (
            forecast_barrier_optimize as _barrier_optimize_impl,
        )

        return run_forecast_barrier_optimize(
            request,
            parse_kv_or_json=_parse_kv_or_json,
            barrier_optimize_impl=_barrier_optimize_impl,
        )

    return _run_forecast_operation(
        "forecast_barrier_optimize",
        symbol=request.symbol,
        timeframe=request.timeframe,
        method=request.method,
        direction=request.direction,
        require_connection=True,
        func=_execute,
    )


def _forecast_list_library_models_impl(
    library: Literal["native", "statsforecast", "sktime", "pretrained", "mlforecast"],
) -> Dict[str, Any]:
    lib = str(library).strip().lower()
    capabilities = _get_library_forecast_capabilities(
        lib,
        discover_sktime_forecasters=_discover_sktime_forecasters,
    )
    if lib == "native":
        return {
            "library": lib,
            "models": [str(row.get("method")) for row in capabilities],
            "capabilities": capabilities,
            "usage": [
                "mtdata-cli forecast_generate SYMBOL --library native --method analog",
                "mtdata-cli forecast_generate SYMBOL --library native --method theta",
            ],
        }

    if lib == "statsforecast":
        try:
            pass  # type: ignore
        except Exception as exc:
            return {"library": lib, "error": f"statsforecast import failed: {exc}"}

        return {
            "library": lib,
            "models": [str(row.get("display_name")) for row in capabilities],
            "capabilities": capabilities,
            "usage": "mtdata-cli forecast_generate SYMBOL --library statsforecast --method AutoARIMA",
        }

    if lib == "sktime":
        return {
            "library": lib,
            "models": [str(row.get("display_name")) for row in capabilities],
            "capabilities": capabilities,
            "usage": [
                "mtdata-cli forecast_generate SYMBOL --library sktime --method theta",
                "mtdata-cli forecast_generate SYMBOL --library sktime --method ThetaForecaster",
                "mtdata-cli forecast_generate SYMBOL --library sktime --method sktime.forecasting.theta.ThetaForecaster --params \"sp=24\"",
            ],
            "note": "The --method value is matched to the closest available forecaster name; you can also pass a dotted class path. Constructor kwargs go in --params (or use --set method.<k>=<v>).",
        }

    if lib == "pretrained":
        models = []
        for row in capabilities:
            model_row = {
                "method": str(row.get("method")),
                "requires": list(row.get("requires") or []),
                "params": [
                    dict(param)
                    for param in (row.get("params") or [])
                    if isinstance(param, dict)
                ],
            }
            notes = row.get("notes")
            if notes:
                model_row["notes"] = str(notes)
            models.append(model_row)
        return {
            "library": lib,
            "models": models,
            "capabilities": capabilities,
            "usage": [
                "mtdata-cli forecast_generate SYMBOL --library pretrained --method chronos2",
                "mtdata-cli forecast_generate SYMBOL --library pretrained --method timesfm",
            ],
        }

    if lib == "mlforecast":
        return {
            "library": lib,
            "capabilities": capabilities,
            "note": "Use `--method <dotted sklearn/lightgbm regressor class>` plus optional constructor kwargs in --params (or use --set method.<k>=<v>).",
            "usage": [
                "mtdata-cli forecast_generate SYMBOL --library mlforecast --method sklearn.ensemble.RandomForestRegressor --params \"n_estimators=200\"",
                "mtdata-cli forecast_generate SYMBOL --library native --method mlf_rf",
            ],
        }

    return {"library": lib, "error": "Unsupported library (supported: native, statsforecast, sktime, pretrained, mlforecast)"}


def _forecast_list_methods_impl(  # noqa: C901
    *,
    detail: Literal["compact", "full"] = "compact",
    limit: Optional[int] = None,
    search: Optional[str] = None,
) -> Dict[str, Any]:
    try:
        data = _get_forecast_methods_data()
        capabilities = _get_registered_forecast_capabilities()
        capability_by_method = {
            str(row.get("method")): row
            for row in capabilities
            if isinstance(row, dict) and row.get("method")
        }
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
                row_out = dict(row)
                row_out["category"] = category
                capability = capability_by_method.get(method_name, {})
                row_out["namespace"] = capability.get("namespace", "native")
                row_out["concept"] = capability.get("concept", method_name)
                row_out["method_id"] = capability.get("capability_id", f"native:{method_name}")
                row_out["capability_id"] = capability.get("capability_id", row_out["method_id"])
                row_out["adapter_method"] = capability.get("adapter_method", method_name)
                row_out["selector"] = capability.get("selector", {"mode": "method"})
                row_out["execution"] = capability.get(
                    "execution",
                    {"library": row_out["namespace"], "method": row_out["adapter_method"]},
                )
                supports = capability.get("supports", row_out.get("supports"))
                if isinstance(supports, dict) and isinstance(supports.get("ci"), bool):
                    row_out["supports_ci"] = bool(supports.get("ci"))
                row_out["display_name"] = capability.get("display_name", method_name)
                row_out["aliases"] = capability.get("aliases", [])
                row_out["source"] = capability.get("source", "registry")
                enriched_full.append(row_out)
            filtered_full = [row for row in enriched_full if _method_matches(row)]
            total_filtered = len(filtered_full)
            if limit_value is not None:
                filtered_full = filtered_full[:limit_value]
            available_count = int(sum(1 for row in filtered_full if bool(row.get("available"))))
            out_full = dict(data)
            out_full["detail"] = "full"
            out_full["methods"] = filtered_full
            out_full["total"] = int(data.get("total") or len(enriched_full))
            out_full["total_filtered"] = int(total_filtered)
            out_full["available"] = available_count
            out_full["unavailable"] = int(len(filtered_full) - available_count)
            out_full["methods_shown"] = int(len(filtered_full))
            out_full["methods_hidden"] = int(max(0, total_filtered - len(filtered_full)))
            out_full["filters"] = {
                "search": search_value or None,
                "limit": limit_value,
            }
            out_full["note"] = (
                "Methods include namespace/concept/method_id fields to disambiguate similarly named implementations "
                "(for example native:theta vs statsforecast:theta). "
                "`supports_ci` indicates whether the method reports built-in interval support."
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
            capability = capability_by_method.get(method_name, {})
            if row["category"] in {"statsforecast", "sktime", "mlforecast", "pretrained"}:
                namespace = capability.get("namespace")
                if isinstance(namespace, str) and namespace.strip():
                    row["namespace"] = namespace
            supports = capability.get("supports", item.get("supports"))
            if isinstance(supports, dict) and isinstance(supports.get("ci"), bool):
                row["supports_ci"] = bool(supports.get("ci"))
            elif isinstance(item.get("supports_ci"), bool):
                row["supports_ci"] = bool(item.get("supports_ci"))
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
            "note": "Compact view groups methods by category, includes `supports_ci` guidance, and shows a small representative subset. Use --detail full to see all methods.",
            "filters": {
                "search": search_value or None,
                "limit": limit_value,
            },
        }
    except Exception as exc:
        return {"error": f"Error listing forecast methods: {exc}"}
