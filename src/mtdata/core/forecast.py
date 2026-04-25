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
    ForecastOptimizeHintsRequest,
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
from .cli_formatting import _sanitize_json_compat
from .error_envelope import build_error_payload
from .execution_logging import run_logged_operation
from .mt5_gateway import get_mt5_gateway, mt5_connection_error
from .schema import CompactFullDetailLiteral

logger = logging.getLogger(__name__)


def _forecast_module():
    return import_module("mtdata.forecast.forecast")


def _forecast_backtest_module():
    return import_module("mtdata.forecast.backtest")


def _forecast_use_cases_module():
    return import_module("mtdata.forecast.use_cases")


def _forecast_methods_module():
    return import_module("mtdata.forecast.forecast_methods")


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


def _get_forecast_methods_snapshot():
    return _forecast_methods_module().get_forecast_methods_snapshot(
        method_data=_get_forecast_methods_data(),
        capabilities=_get_registered_forecast_capabilities(),
    )


@lru_cache(maxsize=1)
def _get_registered_forecast_capabilities():
    return _forecast_capabilities_module().get_registered_capabilities()


def _get_library_forecast_capabilities(*args, **kwargs):
    return _forecast_capabilities_module().get_library_capabilities(*args, **kwargs)


def _genetic_search_impl(**kwargs):
    return _forecast_tune_module().genetic_search_forecast_params(**kwargs)


def _optuna_search_impl(**kwargs):
    return _forecast_tune_module().optuna_search_forecast_params(**kwargs)


def _optimize_hints_impl(**kwargs):
    return _forecast_tune_module().genetic_search_optimize_hints(**kwargs)


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


def run_forecast_optimize_hints(*args, **kwargs):
    return _forecast_use_cases_module().run_forecast_optimize_hints(*args, **kwargs)


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


def _forecast_error_payload(message: Any, *, operation: str) -> Dict[str, Any]:
    return build_error_payload(
        message,
        code=f"{operation}_error",
        operation=operation,
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
                return _forecast_error_payload(exc, operation=operation)
            raise
        except Exception as exc:
            if generic_error_prefix is not None:
                return _forecast_error_payload(
                    f"{generic_error_prefix}{exc}",
                    operation=operation,
                )
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
    show_unavailable: bool = False,
) -> Dict[str, Any]:
    """List available model names within a forecast library.

    - statsforecast: lists `statsforecast.models.*` class names.
    - sktime: lists supported aliases plus notes for using dotted estimator paths.
    """
    return _run_forecast_operation(
        "forecast_list_library_models",
        library=library,
        show_unavailable=show_unavailable,
        func=lambda: _forecast_list_library_models_impl(
            library,
            show_unavailable=show_unavailable,
        ),
    )


@mcp.tool()
def forecast_backtest_run(request: ForecastBacktestRequest) -> Dict[str, Any]:
    """Rolling-origin backtest over historical anchors using the forecast tool."""
    def _execute() -> Dict[str, Any]:
        return run_forecast_backtest(
            request,
            backtest_impl=_forecast_backtest_impl,
        )

    result = _run_forecast_operation(
        "forecast_backtest_run",
        symbol=request.symbol,
        timeframe=request.timeframe,
        horizon=request.horizon,
        detail=request.detail,
        require_connection=True,
        func=_execute,
    )
    return _sanitize_json_compat(result)


@mcp.tool()
def strategy_backtest(request: StrategyBacktestRequest) -> Dict[str, Any]:
    """Backtest simple indicator-driven trading strategies on MT5 candle history."""
    def _execute() -> Dict[str, Any]:
        return run_strategy_backtest(
            request,
            strategy_backtest_impl=_strategy_backtest_impl,
        )

    result = _run_forecast_operation(
        "strategy_backtest",
        symbol=request.symbol,
        timeframe=request.timeframe,
        strategy=request.strategy,
        require_connection=True,
        func=_execute,
    )
    return _sanitize_json_compat(result)


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
    detail: CompactFullDetailLiteral = "compact",  # type: ignore
    limit: Optional[int] = None,
    search_term: Optional[str] = None,
    library: Optional[
        Literal["native", "statsforecast", "sktime", "pretrained", "mlforecast"]
    ] = None,
    show_unavailable: bool = False,
) -> Dict[str, Any]:
    """List forecast methods and availability.

    - detail='compact' (default): concise list with availability and `supports_ci` guidance.
    - detail='full': include full parameter docs and supports metadata.
    """
    search_term_value = str(search_term or "").strip() or None
    return _run_forecast_operation(
        "forecast_list_methods",
        detail=detail,
        limit=limit,
        search_term=search_term_value,
        library=library,
        show_unavailable=show_unavailable,
        func=lambda: _forecast_list_methods_impl(
            detail=detail,
            limit=limit,
            search=search_term_value,
            library=library,
            show_unavailable=show_unavailable,
        ),
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
def forecast_optimize_hints(request: ForecastOptimizeHintsRequest) -> Dict[str, Any]:
    """Genetic search for optimal forecast settings across timeframes, methods, and parameters.

    Searches over timeframes, algorithms, and algorithm-specific parameters to find
    top-N configurations ranked by composite fitness score (Sharpe ratio, win rate,
    inverse drawdown, and average return).

    Parameters:
    -----------
    symbol : str
        Trading symbol (e.g., 'EURUSD')
    
    timeframes : list[str], optional
        Timeframes to search (e.g., ['H1', 'H4', 'D1']).
        If not provided, defaults to ['H1', 'H4', 'D1', 'W1'].
    
    methods : list[str], optional
        Forecast methods to search over.
        If not provided, defaults to fast + pretrained methods
        (theta, ARIMA, chronos, timesfm, etc.).
    
    horizon : int, optional (default=12)
        Forecast horizon in bars.
    
    steps : int, optional (default=5)
        Number of backtest anchors.
    
    spacing : int, optional (default=20)
        Spacing between anchors in bars.
    
    population : int, optional (default=20)
        Genetic algorithm population size (1-100).
    
    generations : int, optional (default=15)
        Number of generations to evolve (1-100).
    
    crossover_rate : float, optional (default=0.6)
        Crossover probability (0-1).
    
    mutation_rate : float, optional (default=0.3)
        Mutation probability (0-1).
    
    fitness_metric : str, optional (default='composite')
        Fitness metric: 'composite' (default) or specific metric
        ('avg_rmse', 'sharpe_ratio', 'win_rate', 'calmar_ratio', etc.).
    
    fitness_weights : dict, optional
        Custom weights for composite fitness (metric -> weight).
        Default: {'sharpe_ratio': 0.4, 'win_rate': 0.3, 'inverse_max_drawdown': 0.2, 'avg_return': 0.1}
    
    seed : int, optional (default=42)
        Random seed for reproducibility.
    
    max_search_time_seconds : float, optional
        Maximum time allowed for search (seconds). No limit if None.
    
    top_n : int, optional (default=5)
        Number of top configurations to return (1-20).
    
    include_feature_genes : bool, optional (default=False)
        If True, include technical indicator parameters (RSI, MACD, etc.) in search.
    
    denoise : DenoiseSpec, optional
        Denoising configuration for preprocessing.
    
    features : dict, optional
        Additional feature configurations.
    
    dimred_method : str, optional
        Dimensionality reduction method for multivariate forecasts.
    
    dimred_params : dict, optional
        Parameters for dimensionality reduction.

    Returns:
    --------
    dict
        Success response with:
        - hints: list of top-N configurations ranked by fitness
          Each hint contains: rank, timeframe, method, method_params, fitness_score, backtest_metrics
        - search_summary: summary of search parameters and results
        - history_tail: last 10 generations' statistics
    """
    from ..forecast.requests import ForecastOptimizeHintsRequest as ReqModel

    # Validate request type if needed
    if not isinstance(request, ReqModel):
        request = ReqModel(**dict(request))

    def _execute() -> Dict[str, Any]:
        return run_forecast_optimize_hints(
            request,
            optimize_hints_impl=_optimize_hints_impl,
        )

    return _run_forecast_operation(
        "forecast_optimize_hints",
        symbol=request.symbol,
        timeframe=request.timeframe,
        require_connection=True,
        generic_error_prefix="Error in optimize hints search: ",
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
    *,
    show_unavailable: bool = False,
) -> Dict[str, Any]:
    lib = str(library).strip().lower()
    capabilities_raw = _get_library_forecast_capabilities(
        lib,
        discover_sktime_forecasters=_discover_sktime_forecasters,
    )
    capabilities_all = capabilities_raw if isinstance(capabilities_raw, list) else []
    capabilities = [
        row
        for row in capabilities_all
        if show_unavailable or row.get("available") is not False
    ]
    method_rows = _forecast_library_method_rows(capabilities)
    available_selected = int(
        sum(1 for row in capabilities if row.get("available") is not False)
    )
    unavailable_selected = int(len(capabilities) - available_selected)
    unavailable_total = int(
        sum(1 for row in capabilities_all if row.get("available") is False)
    )
    availability_meta = {
        "total": len(capabilities_all),
        "total_filtered": len(capabilities),
        "available": available_selected,
        "unavailable": unavailable_selected,
        "unavailable_hidden": 0 if show_unavailable else unavailable_total,
        "filters": {"show_unavailable": bool(show_unavailable)},
    }
    if lib == "native":
        return {
            "library": lib,
            "models": [str(row.get("method")) for row in capabilities],
            "methods": method_rows,
            "capabilities": capabilities,
            **availability_meta,
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
            "methods": method_rows,
            "capabilities": capabilities,
            **availability_meta,
            "usage": "mtdata-cli forecast_generate SYMBOL --library statsforecast --method AutoARIMA",
        }

    if lib == "sktime":
        return {
            "library": lib,
            "models": [str(row.get("display_name")) for row in capabilities],
            "methods": method_rows,
            "capabilities": capabilities,
            **availability_meta,
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
            "methods": method_rows,
            "capabilities": capabilities,
            **availability_meta,
            "usage": [
                "mtdata-cli forecast_generate SYMBOL --library pretrained --method chronos2",
                "mtdata-cli forecast_generate SYMBOL --library pretrained --method timesfm",
            ],
        }

    if lib == "mlforecast":
        return {
            "library": lib,
            "methods": method_rows,
            "capabilities": capabilities,
            **availability_meta,
            "note": "Use `--method <dotted sklearn/lightgbm regressor class>` plus optional constructor kwargs in --params (or use --set method.<k>=<v>).",
            "usage": [
                "mtdata-cli forecast_generate SYMBOL --library mlforecast --method sklearn.ensemble.RandomForestRegressor --params \"n_estimators=200\"",
                "mtdata-cli forecast_generate SYMBOL --library native --method mlf_rf",
            ],
        }

    return {"library": lib, "error": "Unsupported library (supported: native, statsforecast, sktime, pretrained, mlforecast)"}


def _forecast_library_method_rows(capabilities: Any) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not isinstance(capabilities, list):
        return rows
    for capability in capabilities:
        if not isinstance(capability, dict):
            continue
        method = str(capability.get("method") or "").strip()
        if not method:
            continue
        row: Dict[str, Any] = {"method": method}
        row["available"] = capability.get("available") is not False
        display_name = str(capability.get("display_name") or "").strip()
        if display_name:
            row["model"] = display_name
        selector = capability.get("selector")
        if isinstance(selector, dict):
            selector_value = selector.get("value")
            if selector_value is not None:
                row["selector_value"] = selector_value
            selector_key = selector.get("key")
            if selector_key is not None:
                row["selector_key"] = selector_key
        execution = capability.get("execution")
        if isinstance(execution, dict):
            library = execution.get("library")
            if library is not None:
                row["library"] = library
        rows.append(row)
    return rows


def _forecast_list_methods_impl(  # noqa: C901
    *,
    detail: CompactFullDetailLiteral = "compact",
    limit: Optional[int] = None,
    search: Optional[str] = None,
    library: Optional[str] = None,
    show_unavailable: bool = True,
) -> Dict[str, Any]:
    try:
        snapshot = _get_forecast_methods_snapshot()
        data = snapshot.get("data") if isinstance(snapshot, dict) else {}
        if not isinstance(data, dict):
            data = {}
        detail_value = str(detail or "compact").strip().lower()
        search_value = str(search or "").strip().lower()
        library_value = str(library or "").strip().lower()
        supported_libraries = {
            "native",
            "statsforecast",
            "sktime",
            "pretrained",
            "mlforecast",
        }
        if library_value and library_value not in supported_libraries:
            return {
                "error": (
                    "Invalid library filter. Use native, statsforecast, sktime, "
                    "pretrained, or mlforecast."
                )
            }
        limit_value: Optional[int] = None
        if limit is not None:
            try:
                limit_value = int(limit)
            except Exception:
                return {"error": f"Invalid limit: {limit}. Must be a positive integer."}
            if limit_value <= 0:
                return {"error": f"Invalid limit: {limit_value}. Must be >= 1."}

        categories_raw = data.get("categories") if isinstance(data.get("categories"), dict) else {}
        method_to_category = snapshot.get("method_to_category") if isinstance(snapshot, dict) else {}
        if not isinstance(method_to_category, dict):
            method_to_category = {}

        def _item_library(item: Dict[str, Any]) -> str:
            method_name = str(item.get("method") or "")
            for key in ("namespace", "library"):
                value = str(item.get(key) or "").strip().lower()
                if value in supported_libraries:
                    return value
            execution = item.get("execution")
            if isinstance(execution, dict):
                value = str(execution.get("library") or "").strip().lower()
                if value in supported_libraries:
                    return value
            category = str(
                item.get("category") or method_to_category.get(method_name) or ""
            ).strip().lower()
            if category in supported_libraries:
                return category
            return "native"

        def _method_matches(item: Dict[str, Any]) -> bool:
            if library_value and _item_library(item) != library_value:
                return False
            if not show_unavailable and not bool(item.get("available")):
                return False
            if not search_value:
                return True
            method_name = str(item.get("method") or "")
            desc = str(item.get("description") or "")
            cat = method_to_category.get(method_name, "")
            haystack = " ".join((method_name, desc, cat)).lower()
            return search_value in haystack

        if detail_value == "full":
            if not bool(snapshot.get("methods_valid")):
                return data
            methods_full = snapshot.get("methods")
            if not isinstance(methods_full, list):
                return data
            filtered_full = [
                dict(row)
                for row in methods_full
                if isinstance(row, dict) and _method_matches(row)
            ]
            total_filtered = len(filtered_full)
            if limit_value is not None:
                filtered_full = filtered_full[:limit_value]
            available_count = int(sum(1 for row in filtered_full if bool(row.get("available"))))
            out_full = dict(data)
            out_full["detail"] = "full"
            out_full["methods"] = filtered_full
            out_full["total"] = int(data.get("total") or len(methods_full))
            out_full["total_filtered"] = int(total_filtered)
            out_full["available"] = available_count
            out_full["unavailable"] = int(len(filtered_full) - available_count)
            out_full["methods_shown"] = int(len(filtered_full))
            out_full["methods_hidden"] = int(max(0, total_filtered - len(filtered_full)))
            out_full["filters"] = {
                "search": search_value or None,
                "limit": limit_value,
                "library": library_value or None,
                "show_unavailable": bool(show_unavailable),
            }
            out_full["note"] = (
                "Methods include namespace/concept/method_id fields to disambiguate similarly named implementations "
                "(for example native:theta vs statsforecast:theta). "
                "`supports_ci` indicates whether the method reports built-in interval support."
            )
            return out_full

        if not bool(snapshot.get("methods_valid")):
            return data
        methods = snapshot.get("methods")
        if not isinstance(methods, list):
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
            cat = item.get("category") or method_to_category.get(method_name)
            row["category"] = str(cat or "other")
            if row["category"] in {"statsforecast", "sktime", "mlforecast", "pretrained"}:
                namespace = item.get("namespace")
                if isinstance(namespace, str) and namespace.strip():
                    row["namespace"] = namespace
            supports = item.get("supports")
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
            selected_methods.extend(rows)
            category_summary.append(
                {
                    "category": category,
                    "total": n_total,
                    "available": n_available,
                    "unavailable": int(n_total - n_available),
                    "examples": [str(row.get("method")) for row in rows[:3]],
                    "hidden": 0,
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
            "note": "Compact view includes all filtered methods with compact columns; set limit to cap rows or detail='full' for complete metadata.",
            "filters": {
                "search": search_value or None,
                "limit": limit_value,
                "library": library_value or None,
                "show_unavailable": bool(show_unavailable),
            },
        }
    except Exception as exc:
        return {"error": f"Error listing forecast methods: {exc}"}
