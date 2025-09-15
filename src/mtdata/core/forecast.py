
from typing import Any, Dict, Optional, List, Literal

from .schema import TimeframeLiteral, DenoiseSpec, ForecastMethodLiteral
from .server import mcp, _auto_connect_wrapper
from ..forecast.forecast import forecast as _forecast_impl
from ..forecast.backtest import forecast_backtest as _forecast_backtest_impl
from ..forecast.volatility import forecast_volatility as _forecast_volatility_impl

@mcp.tool()
@_auto_connect_wrapper
def forecast(
    symbol: str,
    timeframe: TimeframeLiteral = "H1",
    method: ForecastMethodLiteral = "theta",
    horizon: int = 12,
    lookback: Optional[int] = None,
    as_of: Optional[str] = None,
    params: Optional[Dict[str, Any]] = None,
    ci_alpha: Optional[float] = 0.05,
    quantity: Literal['price','return','volatility'] = 'price',  # type: ignore
    target: Literal['price','return'] = 'price',  # type: ignore
    denoise: Optional[DenoiseSpec] = None,
    features: Optional[Dict[str, Any]] = None,
    dimred_method: Optional[str] = None,
    dimred_params: Optional[Dict[str, Any]] = None,
    target_spec: Optional[Dict[str, Any]] = None,
    timezone: str = "auto",
) -> Dict[str, Any]:
    """Fast forecasts for the next `horizon` bars using lightweight methods.

    Delegates to the implementation under `mtdata.forecast.forecast`.
    """
    return _forecast_impl(
        symbol=symbol,
        timeframe=timeframe,  # type: ignore[arg-type]
        method=method,        # type: ignore[arg-type]
        horizon=horizon,
        lookback=lookback,
        as_of=as_of,
        params=params,
        ci_alpha=ci_alpha,
        quantity=quantity,    # type: ignore[arg-type]
        target=target,        # type: ignore[arg-type]
        denoise=denoise,
        features=features,
        dimred_method=dimred_method,
        dimred_params=dimred_params,
        target_spec=target_spec,
        timezone=timezone,
    )


@mcp.tool()
@_auto_connect_wrapper
def forecast_backtest(
    symbol: str,
    timeframe: TimeframeLiteral = "H1",
    horizon: int = 12,
    steps: int = 5,
    spacing: int = 20,
    methods: Optional[List[str]] = None,
    params_per_method: Optional[Dict[str, Any]] = None,
    quantity: Literal['price','return','volatility'] = 'price',  # type: ignore
    target: Literal['price','return'] = 'price',  # type: ignore
    denoise: Optional[DenoiseSpec] = None,
    params: Optional[Dict[str, Any]] = None,
    features: Optional[Dict[str, Any]] = None,
    dimred_method: Optional[str] = None,
    dimred_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Rolling-origin backtest over historical anchors using the forecast tool."""
    return _forecast_backtest_impl(
        symbol=symbol,
        timeframe=timeframe,  # type: ignore[arg-type]
        horizon=horizon,
        steps=steps,
        spacing=spacing,
        methods=methods,
        params_per_method=params_per_method,
        quantity=quantity,    # type: ignore[arg-type]
        target=target,        # type: ignore[arg-type]
        denoise=denoise,
        params=params,
        features=features,
        dimred_method=dimred_method,
        dimred_params=dimred_params,
    )


@mcp.tool()
@_auto_connect_wrapper
def forecast_volatility(
    symbol: str,
    timeframe: TimeframeLiteral = "H1",
    horizon: int = 1,
    method: Literal['ewma','parkinson','gk','rs','yang_zhang','rolling_std','garch','egarch','gjr_garch','arima','sarima','ets','theta'] = 'ewma',  # type: ignore
    proxy: Optional[Literal['squared_return','abs_return','log_r2']] = None,  # type: ignore
    params: Optional[Dict[str, Any]] = None,
    as_of: Optional[str] = None,
    denoise: Optional[DenoiseSpec] = None,
) -> Dict[str, Any]:
    """Forecast volatility over `horizon` bars using direct estimators or proxies."""
    return _forecast_volatility_impl(
        symbol=symbol,
        timeframe=timeframe,  # type: ignore[arg-type]
        horizon=horizon,
        method=method,        # type: ignore[arg-type]
        proxy=proxy,          # type: ignore[arg-type]
        params=params,
        as_of=as_of,
        denoise=denoise,
    )
