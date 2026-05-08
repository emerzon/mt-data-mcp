import logging
import multiprocessing as mp
import os
import time
import traceback
from functools import lru_cache
from importlib import import_module
from typing import Any, Dict, List, Literal, Optional

from ..forecast.barriers_shared import (
    BARRIER_METHOD_ALIASES,
    BARRIER_MONTE_CARLO_METHODS,
)
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
from ..shared.schema import CompactFullDetailLiteral
from ..utils.barriers import (
    build_barrier_kwargs_from as _build_barrier_kwargs_from,
)
from ..utils.barriers import (
    normalize_trade_direction,
)
from ..utils.mt5 import ensure_mt5_connection_or_raise
from ..utils.utils import parse_kv_or_json as _parse_kv_or_json
from ._mcp_instance import mcp
from .error_envelope import build_error_payload
from .execution_logging import run_logged_operation
from .mt5_gateway import create_mt5_gateway, mt5_connection_error

logger = logging.getLogger(__name__)
_FORECAST_LIST_METHODS_DEFAULT_COMPACT_LIMIT = 20

_FORECAST_PROCESS_ISOLATION_ENV = "MTDATA_FORECAST_PROCESS_ISOLATION"
_FORECAST_PROCESS_TIMEOUT_ENV = "MTDATA_FORECAST_PROCESS_TIMEOUT_SECONDS"
_FORECAST_PROCESS_CHILD_ENV = "MTDATA_FORECAST_PROCESS_CHILD"
_FORECAST_PROCESS_ISOLATION_DEFAULT = "gpu"
_FORECAST_ISOLATABLE_OPERATIONS = frozenset(
    {
        "forecast_generate",
        "forecast_list_library_models",
        "forecast_backtest_run",
        "strategy_backtest",
        "forecast_volatility_estimate",
        "forecast_list_methods",
        "forecast_conformal_intervals",
        "forecast_tune_genetic",
        "forecast_tune_optuna",
        "forecast_optimize_hints",
        "forecast_barrier_prob",
        "forecast_barrier_optimize",
    }
)
_FORECAST_CONNECTION_REQUIRED_OPERATIONS = frozenset(
    {
        "forecast_generate",
        "forecast_backtest_run",
        "strategy_backtest",
        "forecast_volatility_estimate",
        "forecast_conformal_intervals",
        "forecast_tune_genetic",
        "forecast_tune_optuna",
        "forecast_optimize_hints",
        "forecast_barrier_prob",
        "forecast_barrier_optimize",
    }
)


_FORECAST_TIMESTAMP_OPERATIONS = frozenset(
    {
        "forecast_generate",
        "forecast_backtest_run",
        "forecast_conformal_intervals",
        "forecast_volatility_estimate",
    }
)


def _attach_timestamp_timezone(result: Dict[str, Any], *, operation: str) -> Dict[str, Any]:
    if (
        operation in _FORECAST_TIMESTAMP_OPERATIONS
        and isinstance(result, dict)
        and "error" not in result
    ):
        result.setdefault("timezone", "UTC")
    return result


def _forecast_process_isolation_mode() -> str:
    raw = os.environ.get(
        _FORECAST_PROCESS_ISOLATION_ENV,
        _FORECAST_PROCESS_ISOLATION_DEFAULT,
    )
    mode = str(raw or "").strip().lower()
    if mode in {"", "auto"}:
        return _FORECAST_PROCESS_ISOLATION_DEFAULT
    if mode in {"0", "false", "no", "off", "none", "disabled"}:
        return "off"
    if mode in {"1", "true", "yes", "on", "all", "always"}:
        return "all"
    if mode == "gpu":
        return "gpu"
    logger.warning(
        "Invalid %s=%r; using %s",
        _FORECAST_PROCESS_ISOLATION_ENV,
        raw,
        _FORECAST_PROCESS_ISOLATION_DEFAULT,
    )
    return _FORECAST_PROCESS_ISOLATION_DEFAULT


def _forecast_process_timeout_seconds() -> Optional[float]:
    raw = os.environ.get(_FORECAST_PROCESS_TIMEOUT_ENV)
    if raw is None or str(raw).strip() == "":
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        logger.warning("Invalid %s=%r; forecast child processes will not be time-limited", _FORECAST_PROCESS_TIMEOUT_ENV, raw)
        return None
    if value <= 0:
        return None
    return value


def _in_forecast_process_child() -> bool:
    return os.environ.get(_FORECAST_PROCESS_CHILD_ENV) == "1"


def _model_payload(request: Any) -> Dict[str, Any]:
    dump = getattr(request, "model_dump", None)
    if callable(dump):
        return dict(dump(mode="json"))
    return dict(request or {})


def _is_pretrained_gpu_request(payload: Dict[str, Any]) -> bool:
    library = str(payload.get("library") or "native").strip().lower()
    method = str(payload.get("method") or "").strip().lower()
    if library == "pretrained":
        return True
    if method.startswith("pretrained:"):
        return True
    return False


def _forecast_payload_may_use_gpu(operation: str, payload: Dict[str, Any]) -> bool:
    try:
        from ..forecast.gpu_runtime import (
            forecast_method_may_use_gpu,
            forecast_methods_may_use_gpu,
        )

        if operation == "forecast_generate":
            method = payload.get("method")
            if _is_pretrained_gpu_request(payload):
                method = method or "chronos2"
            return forecast_method_may_use_gpu(method, payload.get("params"))

        if operation == "forecast_backtest_run":
            return forecast_methods_may_use_gpu(
                payload.get("methods"),
                params_per_method=payload.get("params_per_method"),
                params=payload.get("params"),
            )

        if operation == "forecast_conformal_intervals":
            return forecast_method_may_use_gpu(payload.get("method"), payload.get("params"))

        if operation in {"forecast_tune_genetic", "forecast_tune_optuna"}:
            methods = payload.get("methods") or payload.get("method")
            return forecast_methods_may_use_gpu(methods, params=payload.get("params"))

        if operation == "forecast_optimize_hints":
            methods = payload.get("methods")
            if not methods:
                return True
            return forecast_methods_may_use_gpu(methods)
    except Exception as exc:
        logger.debug("Forecast process-isolation GPU detection failed for %s: %s", operation, exc)
    return False


def _should_isolate_forecast_operation(operation: str, payload: Optional[Dict[str, Any]]) -> bool:
    if _in_forecast_process_child():
        return False
    if operation not in _FORECAST_ISOLATABLE_OPERATIONS:
        return False
    mode = _forecast_process_isolation_mode()
    if mode == "off":
        return False
    if mode == "all":
        return True
    return _forecast_payload_may_use_gpu(operation, dict(payload or {}))


def _send_forecast_process_message(channel: Any, message: Dict[str, Any]) -> None:
    send = getattr(channel, "send", None)
    if callable(send):
        send(message)
        return
    put = getattr(channel, "put", None)
    if callable(put):
        put(message)
        return
    raise RuntimeError("Invalid forecast child result channel")


def _forecast_process_entry(operation: str, payload: Dict[str, Any], result_channel: Any) -> None:
    os.environ[_FORECAST_PROCESS_CHILD_ENV] = "1"
    try:
        result = _run_forecast_payload_direct(operation, payload)
    except ForecastError as exc:
        _send_forecast_process_message(
            result_channel,
            {
                "status": "forecast_error",
                "message": str(exc),
            },
        )
    except BaseException as exc:
        _send_forecast_process_message(
            result_channel,
            {
                "status": "exception",
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
    else:
        _send_forecast_process_message(result_channel, {"status": "ok", "result": result})
    finally:
        try:
            from ..forecast.gpu_runtime import cleanup_forecast_gpu_runtime

            cleanup_forecast_gpu_runtime(clear_model_cache=True)
        except Exception:
            pass
        try:
            close = getattr(result_channel, "close", None)
            if callable(close):
                close()
        except Exception:
            pass


def _run_forecast_payload_direct(operation: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    if operation in _FORECAST_CONNECTION_REQUIRED_OPERATIONS:
        ensure_mt5_connection_or_raise()

    if operation == "forecast_generate":
        request = ForecastGenerateRequest(**payload)
        return run_forecast_generate(
            request,
            forecast_impl=_forecast_impl,
            resolve_sktime_forecaster=_resolve_sktime_forecaster,
            log_events=False,
        )

    if operation == "forecast_list_library_models":
        return _forecast_list_library_models_impl(
            payload["library"],
            show_unavailable=bool(payload.get("show_unavailable", False)),
        )

    if operation == "forecast_backtest_run":
        request = ForecastBacktestRequest(**payload)
        return run_forecast_backtest(
            request,
            backtest_impl=_forecast_backtest_impl,
        )

    if operation == "strategy_backtest":
        request = StrategyBacktestRequest(**payload)
        return run_strategy_backtest(
            request,
            strategy_backtest_impl=_strategy_backtest_impl,
        )

    if operation == "forecast_volatility_estimate":
        request = ForecastVolatilityEstimateRequest(**payload)
        return run_forecast_volatility_estimate(
            request,
            forecast_volatility_impl=_forecast_volatility_impl,
        )

    if operation == "forecast_list_methods":
        return _forecast_list_methods_impl(
            detail=payload.get("detail", "compact"),
            limit=payload.get("limit"),
            search=payload.get("search_term"),
            category=payload.get("category"),
            library=payload.get("library"),
            supports_ci=payload.get("supports_ci"),
            show_unavailable=bool(payload.get("show_unavailable", False)),
        )

    if operation == "forecast_conformal_intervals":
        request = ForecastConformalIntervalsRequest(**payload)
        return run_forecast_conformal_intervals(
            request,
            backtest_impl=_forecast_backtest_impl,
            forecast_impl=_forecast_impl,
        )

    if operation == "forecast_tune_genetic":
        request = ForecastTuneGeneticRequest(**payload)
        return run_forecast_tune_genetic(
            request,
            genetic_search_impl=_genetic_search_impl,
        )

    if operation == "forecast_tune_optuna":
        request = ForecastTuneOptunaRequest(**payload)
        return run_forecast_tune_optuna(
            request,
            optuna_search_impl=_optuna_search_impl,
        )

    if operation == "forecast_optimize_hints":
        request = ForecastOptimizeHintsRequest(**payload)
        return run_forecast_optimize_hints(
            request,
            optimize_hints_impl=_optimize_hints_impl,
        )

    if operation == "forecast_barrier_prob":
        from ..forecast.barriers_probabilities import (
            forecast_barrier_closed_form as _barrier_closed_form_impl,
        )
        from ..forecast.barriers_probabilities import (
            forecast_barrier_hit_probabilities as _barrier_hit_probabilities_impl,
        )

        request = ForecastBarrierProbRequest(**payload)
        return run_forecast_barrier_prob(
            request,
            build_barrier_kwargs=_build_barrier_kwargs_from,
            normalize_trade_direction=normalize_trade_direction,
            barrier_hit_probabilities_impl=_barrier_hit_probabilities_impl,
            barrier_closed_form_impl=_barrier_closed_form_impl,
        )

    if operation == "forecast_barrier_optimize":
        from ..forecast.barriers_optimization import (
            forecast_barrier_optimize as _barrier_optimize_impl,
        )

        request = ForecastBarrierOptimizeRequest(**payload)
        return run_forecast_barrier_optimize(
            request,
            parse_kv_or_json=_parse_kv_or_json,
            barrier_optimize_impl=_barrier_optimize_impl,
        )

    raise ValueError(f"Unsupported isolated forecast operation: {operation}")


def _run_forecast_payload_in_process(operation: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    process = ctx.Process(
        target=_forecast_process_entry,
        args=(operation, dict(payload), child_conn),
        daemon=False,
    )
    process.start()
    try:
        child_conn.close()
    except Exception:
        pass
    timeout_seconds = _forecast_process_timeout_seconds()
    deadline = time.monotonic() + timeout_seconds if timeout_seconds is not None else None
    message: Optional[Dict[str, Any]] = None
    try:
        while message is None:
            if parent_conn.poll(0.1):
                raw_message = parent_conn.recv()
                if isinstance(raw_message, dict):
                    message = raw_message
                else:
                    message = {"status": "ok", "result": raw_message}
                break
            if not process.is_alive():
                process.join(timeout=0.1)
                if parent_conn.poll(0.1):
                    raw_message = parent_conn.recv()
                    if isinstance(raw_message, dict):
                        message = raw_message
                    else:
                        message = {"status": "ok", "result": raw_message}
                    break
                exit_code = process.exitcode
                raise RuntimeError(
                    f"{operation} child process exited without returning a result"
                    + (f" (exitcode={exit_code})" if exit_code is not None else "")
                )
            if deadline is not None and time.monotonic() >= deadline:
                process.terminate()
                process.join(timeout=5.0)
                raise TimeoutError(
                    f"{operation} child process timed out after {timeout_seconds} seconds"
                )
    finally:
        try:
            parent_conn.close()
        except Exception:
            pass
        if process.is_alive():
            if message is not None:
                process.join(timeout=5.0)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5.0)
            else:
                process.join(timeout=0.1)

    status = str(message.get("status") or "")
    if status == "ok":
        result = message.get("result")
        if isinstance(result, dict):
            return result
        return {"success": True, "result": result}
    if status == "forecast_error":
        raise ForecastError(str(message.get("message") or "Forecast child process failed"))
    if status == "exception":
        tb = str(message.get("traceback") or "")
        if tb:
            logger.debug("Forecast child traceback for %s:\n%s", operation, tb)
        exc_type = str(message.get("type") or "RuntimeError")
        exc_message = str(message.get("message") or "Forecast child process failed")
        raise RuntimeError(f"{operation} child process failed with {exc_type}: {exc_message}")
    raise RuntimeError(f"{operation} child process returned an invalid status: {status or '<missing>'}")


def _lazy_module(module_name: str):
    return lambda: import_module(module_name)


_forecast_module = _lazy_module("mtdata.forecast.forecast")
_forecast_backtest_module = _lazy_module("mtdata.forecast.backtest")
_forecast_use_cases_module = _lazy_module("mtdata.forecast.use_cases")
_forecast_methods_module = _lazy_module("mtdata.forecast.forecast_methods")
_forecast_volatility_module = _lazy_module("mtdata.forecast.volatility")
_forecast_tune_module = _lazy_module("mtdata.forecast.tune")
_forecast_capabilities_module = _lazy_module("mtdata.forecast.capabilities")


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
        create_mt5_gateway(ensure_connection_impl=ensure_mt5_connection_or_raise),
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
    process_payload: Optional[Dict[str, Any]] = None,
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
            if _should_isolate_forecast_operation(operation, process_payload):
                result = _run_forecast_payload_in_process(operation, dict(process_payload or {}))
            else:
                result = func()
            return _attach_timestamp_timezone(result, operation=operation)
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
        process_payload=_model_payload(request),
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
        process_payload={
            "library": library,
            "show_unavailable": bool(show_unavailable),
        },
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

    return _run_forecast_operation(
        "forecast_backtest_run",
        symbol=request.symbol,
        timeframe=request.timeframe,
        horizon=request.horizon,
        detail=request.detail,
        require_connection=True,
        process_payload=_model_payload(request),
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
        process_payload=_model_payload(request),
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
        process_payload=_model_payload(request),
        func=_execute,
    )


@mcp.tool()
def forecast_list_methods(
    detail: CompactFullDetailLiteral = "compact",  # type: ignore
    limit: Optional[int] = None,
    search_term: Optional[str] = None,
    category: Optional[str] = None,
    library: Optional[
        Literal["native", "statsforecast", "sktime", "pretrained", "mlforecast"]
    ] = None,
    supports_ci: Optional[bool] = None,
    show_unavailable: bool = False,
) -> Dict[str, Any]:
    """List forecast methods and availability.

    Compact output is the default. Use extras='metadata' to include full
    parameter docs and supports metadata.
    """
    search_term_value = str(search_term or "").strip() or None
    return _run_forecast_operation(
        "forecast_list_methods",
        detail=detail,
        limit=limit,
        search_term=search_term_value,
        category=category,
        library=library,
        supports_ci=supports_ci,
        show_unavailable=show_unavailable,
        process_payload={
            "detail": detail,
            "limit": limit,
            "search_term": search_term_value,
            "category": category,
            "library": library,
            "supports_ci": supports_ci,
            "show_unavailable": bool(show_unavailable),
        },
        func=lambda: _forecast_list_methods_impl(
            detail=detail,
            limit=limit,
            search=search_term_value,
            category=category,
            library=library,
            supports_ci=supports_ci,
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
        process_payload=_model_payload(request),
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
        process_payload=_model_payload(request),
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
        process_payload=_model_payload(request),
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
        process_payload=_model_payload(request),
        func=_execute,
    )


@mcp.tool()
def options_expirations(
    symbol: str,
) -> Dict[str, Any]:
    """Fetch option expirations via Yahoo Finance; provider availability/auth can change."""
    from ..services.options_service import get_options_expirations as _impl
    return _run_forecast_operation(
        "options_expirations",
        symbol=symbol,
        func=lambda: _impl(symbol=symbol),
    )


@mcp.tool()
def options_chain(
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
        "options_chain",
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
def options_barrier_price(
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
        "options_barrier_price",
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
def options_heston_calibrate(
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
        "options_heston_calibrate",
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
    
    tp_ticks : float, optional
        Take profit in ticks (trade_tick_size)
    
    sl_ticks : float, optional
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
        from ..forecast.barriers_probabilities import (
            forecast_barrier_closed_form as _barrier_closed_form_impl,
        )
        from ..forecast.barriers_probabilities import (
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
        process_payload=_model_payload(request),
        func=_execute,
    )


@mcp.tool()
def forecast_barrier_optimize(
    request: ForecastBarrierOptimizeRequest,
) -> Dict[str, Any]:
    """Optimize TP/SL barriers with support for presets, volatility scaling, ratios, and two-stage refinement."""
    def _execute() -> Dict[str, Any]:
        from ..forecast.barriers_optimization import (
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
        process_payload=_model_payload(request),
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


def _forecast_method_params(params: Any) -> List[Dict[str, Any]]:
    if not isinstance(params, list):
        return []
    rows: List[Dict[str, Any]] = []
    for item in params:
        if not isinstance(item, dict):
            continue
        row = {
            key: item.get(key)
            for key in ("name", "type", "default", "description")
            if item.get(key) not in (None, "", [], {})
        }
        if row:
            rows.append(row)
    return rows


def _forecast_list_full_row(
    item: Dict[str, Any],
    *,
    method_to_category: Dict[str, Any],
) -> Dict[str, Any]:
    method_name = str(item.get("method") or "")
    category = item.get("category") or method_to_category.get(method_name) or "other"
    params = _forecast_method_params(item.get("params"))
    row: Dict[str, Any] = {
        "method": method_name,
        "category": str(category),
        "available": bool(item.get("available")),
        "params_count": len(params),
    }
    desc = str(item.get("description") or "").strip()
    if desc and desc.lower() != method_name.lower():
        row["description"] = desc.splitlines()[0].strip()
    supports = item.get("supports")
    if isinstance(supports, dict) and isinstance(supports.get("ci"), bool):
        row["supports_ci"] = bool(supports.get("ci"))
    elif isinstance(item.get("supports_ci"), bool):
        row["supports_ci"] = bool(item.get("supports_ci"))
    if params:
        row["params"] = params
    requires = item.get("requires")
    if isinstance(requires, list) and requires:
        row["requires"] = [str(req) for req in requires if str(req).strip()]
    aliases = item.get("aliases")
    if isinstance(aliases, list) and aliases:
        row["aliases"] = [str(alias) for alias in aliases if str(alias).strip()]
    metadata_namespace = str(item.get("namespace") or item.get("library") or "").strip().lower()
    if metadata_namespace and metadata_namespace != "native":
        for key in ("method_id", "capability_id", "adapter_method", "selector", "execution"):
            value = item.get(key)
            if value not in (None, "", [], {}):
                row[key] = value
    execution = item.get("execution")
    if isinstance(execution, dict) and execution.get("library") not in (None, ""):
        row["library"] = execution.get("library")
    elif item.get("namespace") not in (None, ""):
        row["library"] = item.get("namespace")
    return row


def _forecast_list_methods_impl(  # noqa: C901
    *,
    detail: CompactFullDetailLiteral = "compact",
    limit: Optional[int] = None,
    search: Optional[str] = None,
    category: Optional[str] = None,
    library: Optional[str] = None,
    supports_ci: Optional[bool] = None,
    show_unavailable: bool = False,
) -> Dict[str, Any]:
    try:
        snapshot = _get_forecast_methods_snapshot()
        data = snapshot.get("data") if isinstance(snapshot, dict) else {}
        if not isinstance(data, dict):
            data = {}
        detail_value = str(detail or "compact").strip().lower()
        search_value = str(search or "").strip().lower()
        category_filter_value = str(category or "").strip().lower()
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
        compact_default_limit_applies = (
            detail_value != "full"
            and limit_value is None
            and not search_value
            and not category_filter_value
            and not library_value
            and supports_ci is None
            and not bool(show_unavailable)
        )
        effective_limit_value = (
            _FORECAST_LIST_METHODS_DEFAULT_COMPACT_LIMIT
            if compact_default_limit_applies
            else limit_value
        )

        categories_raw = data.get("categories") if isinstance(data.get("categories"), dict) else {}
        method_to_category = snapshot.get("method_to_category") if isinstance(snapshot, dict) else {}
        if not isinstance(method_to_category, dict):
            method_to_category = {}

        def _item_category(item: Dict[str, Any]) -> str:
            method_name = str(item.get("method") or "")
            return str(
                item.get("category") or method_to_category.get(method_name) or "other"
            ).strip().lower()

        def _item_supports_ci(item: Dict[str, Any]) -> Optional[bool]:
            supports = item.get("supports")
            if isinstance(supports, dict) and isinstance(supports.get("ci"), bool):
                return bool(supports.get("ci"))
            if isinstance(item.get("supports_ci"), bool):
                return bool(item.get("supports_ci"))
            return None

        def _item_library(item: Dict[str, Any]) -> str:
            for key in ("namespace", "library"):
                value = str(item.get(key) or "").strip().lower()
                if value in supported_libraries:
                    return value
            execution = item.get("execution")
            if isinstance(execution, dict):
                value = str(execution.get("library") or "").strip().lower()
                if value in supported_libraries:
                    return value
            category = _item_category(item)
            if category in supported_libraries:
                return category
            return "native"

        def _method_matches(item: Dict[str, Any]) -> bool:
            if library_value and _item_library(item) != library_value:
                return False
            if category_filter_value and _item_category(item) != category_filter_value:
                return False
            if supports_ci is not None and _item_supports_ci(item) is not bool(supports_ci):
                return False
            if not show_unavailable and not bool(item.get("available")):
                return False
            if not search_value:
                return True
            method_name = str(item.get("method") or "")
            desc = str(item.get("description") or "")
            cat = _item_category(item)
            namespace = str(item.get("namespace") or "")
            haystack = " ".join((method_name, desc, cat, namespace)).lower()
            return search_value in haystack

        barrier_methods = {
            "methods": list(BARRIER_MONTE_CARLO_METHODS),
            "aliases": {
                key: value
                for key, value in BARRIER_METHOD_ALIASES.items()
                if key.startswith("monte_carlo")
            },
            "probability_only_methods": ["closed_form"],
            "optimizer_only_methods": ["ensemble"],
            "note": "Barrier methods are for forecast_barrier_prob and forecast_barrier_optimize; forecast_generate uses the main forecast method registry.",
        }

        if detail_value == "full":
            if not bool(snapshot.get("methods_valid")):
                return data
            methods_full = snapshot.get("methods")
            if not isinstance(methods_full, list):
                return data
            filtered_full = [
                _forecast_list_full_row(row, method_to_category=method_to_category)
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
                "category": category_filter_value or None,
                "limit": limit_value,
                "library": library_value or None,
                "supports_ci": supports_ci,
                "show_unavailable": bool(show_unavailable),
            }
            out_full["note"] = (
                "Full view includes trader-facing method metadata and structured params; "
                "use search_term, library, or limit to narrow large catalogs."
            )
            out_full["barrier_methods"] = barrier_methods
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
            if desc and desc.lower() != method_name.lower():
                row["description"] = desc.splitlines()[0].strip()
            row["category"] = _item_category(item)
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
            requires = item.get("requires")
            if not available and isinstance(requires, list) and requires:
                req_text = ", ".join(str(req) for req in requires if str(req).strip())
                if req_text:
                    row["unavailable_reason"] = "Requires: " + req_text
            if not available and "unavailable_reason" not in row:
                row["unavailable_reason"] = "Unavailable in the current environment."
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
        if effective_limit_value is not None:
            selected_methods = selected_methods[:effective_limit_value]
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
            "barrier_methods": barrier_methods,
            "note": "Compact view caps unfiltered method rows by default; use category, library, search_term, or limit to narrow or expand rows.",
            "filters": {
                "search": search_value or None,
                "category": category_filter_value or None,
                "limit": effective_limit_value,
                "library": library_value or None,
                "supports_ci": supports_ci,
                "show_unavailable": bool(show_unavailable),
            },
        }
    except Exception as exc:
        return {"error": f"Error listing forecast methods: {exc}"}
