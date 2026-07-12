"""FastAPI app exposing WebUI-ready endpoints that wrap existing mtdata tools."""

from __future__ import annotations

import hmac
import logging
from functools import lru_cache
from importlib.util import find_spec as _find_spec
from typing import Any, Dict, Literal, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from ..bootstrap.runtime import is_loopback_host, load_web_api_runtime_settings
from ..bootstrap.settings import load_environment, mt5_config
from ..forecast.common import fetch_history as _fetch_history_impl
from ..forecast.forecast import get_forecast_methods_data as _get_methods_impl
from ..forecast.volatility import (
    get_volatility_methods_data as _get_vol_methods,
)
from ..services.data_service import fetch_candles as _fetch_candles_impl
from ..shared.constants import TIMEFRAME_MAP
from ..utils.denoise import get_denoise_methods_data as _get_denoise_methods
from ..utils.denoise import normalize_denoise_spec as _norm_dn
from ..utils.dimred import list_dimred_methods as _list_dimred_methods
from ..utils.mt5 import (
    _ensure_symbol_ready,
    ensure_mt5_connection_or_raise,
    mt5,
    mt5_connection,
)
from ..utils.symbol import _extract_group_path as _extract_group_path_util
from .error_envelope import build_error_payload
from .forecast import (
    forecast_backtest_run as _forecast_backtest_tool,
)
from .forecast import (
    forecast_generate as _forecast_generate_tool,
)
from .forecast import (
    forecast_volatility_estimate as _forecast_volatility_tool,
)
from .forecast_tasks import forecast_models_list as _forecast_models_list_tool
from .mt5_gateway import create_mt5_gateway, mt5_connection_error
from .pivot import pivot_compute_points
from .tool_calling import call_tool_sync_structured, unwrap_tool_callable
from .web_api_handlers import (
    get_denoise_methods_response as _get_denoise_methods_response,
)
from .web_api_handlers import (
    get_dimred_methods_response as _get_dimred_methods_response,
)
from .web_api_handlers import (
    get_history_response as _get_history_response,
)
from .web_api_handlers import (
    get_instruments_response as _get_instruments_response,
)
from .web_api_handlers import (
    get_methods_response as _get_methods_response,
)
from .web_api_handlers import (
    get_models_response as _get_models_response,
)
from .web_api_handlers import (
    get_pivots_response as _get_pivots_response,
)
from .web_api_handlers import (
    get_support_resistance_response as _get_support_resistance_response,
)
from .web_api_handlers import (
    get_tick_response as _get_tick_response,
)
from .web_api_handlers import (
    get_vol_methods_response as _get_vol_methods_response,
)
from .web_api_handlers import (
    get_wavelets_response as _get_wavelets_response,
)
from .web_api_handlers import (
    post_backtest_response as _post_backtest_response,
)
from .web_api_handlers import (
    post_forecast_price_response as _post_forecast_price_response,
)
from .web_api_handlers import (
    post_forecast_volatility_response as _post_forecast_volatility_response,
)
from .web_api_models import BacktestBody, ForecastPriceBody, ForecastVolBody
from .web_api_runtime import (
    SafeJSONResponse,
    create_web_api_app,
    mount_webui,
    run_webapi,
)

API_PREFIXES = ("/api", "/api/v1")

logger = logging.getLogger(__name__)
_bearer_auth = HTTPBearer(auto_error=False)


def _raise_auth_error(status_code: int, message: str, *, code: str, headers: Optional[Dict[str, str]] = None) -> None:
    payload = build_error_payload(message, code=code, operation="web_api_auth")
    logger.warning(
        "transport=web_api operation=%s request_id=%s status=%s error=%s",
        "web_api_auth",
        payload["request_id"],
        status_code,
        payload["error"],
    )
    raise HTTPException(status_code=status_code, detail=payload, headers=headers)


def _is_local_api_client(request: Request) -> bool:
    headers = getattr(request, "headers", None)
    forwarded = None
    if headers is not None:
        try:
            forwarded = (
                headers.get("x-forwarded-for")
                or headers.get("forwarded")
                or headers.get("x-real-ip")
            )
        except Exception:
            forwarded = None
    if isinstance(forwarded, str) and forwarded.strip():
        return False
    client_host = getattr(getattr(request, "client", None), "host", None)
    client_text = str(client_host or "").strip().lower()
    return client_text == "testclient" or is_loopback_host(client_text)


@lru_cache(maxsize=1)
def _get_api_access_runtime_settings():
    return load_web_api_runtime_settings()


def _clear_api_access_runtime_settings_cache() -> None:
    _get_api_access_runtime_settings.cache_clear()


def _require_api_access(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_auth),
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> None:
    runtime = _get_api_access_runtime_settings()
    configured_token = str(runtime.auth_token or "").strip()
    supplied_token = None
    if isinstance(credentials, HTTPAuthorizationCredentials):
        scheme = str(credentials.scheme or "").strip().lower()
        token = str(credentials.credentials or "").strip()
        if scheme == "bearer" and token:
            supplied_token = token
    if not supplied_token and isinstance(x_api_key, str) and x_api_key.strip():
        supplied_token = x_api_key.strip()

    if configured_token:
        if supplied_token and hmac.compare_digest(supplied_token, configured_token):
            return
        _raise_auth_error(
            401,
            "Missing or invalid API token.",
            code="web_api_auth_required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if _is_local_api_client(request):
        return

    _raise_auth_error(
        403,
        "Remote API access requires WEBAPI_AUTH_TOKEN.",
        code="web_api_remote_forbidden",
    )

load_environment()
app = create_web_api_app()
api_router = APIRouter(dependencies=[Depends(_require_api_access)])


def _list_sktime_forecasters() -> Dict[str, Any]:
    if _find_spec("sktime") is None:
        return {"available": False, "error": "sktime not installed", "estimators": []}
    try:
        from sktime.registry import all_estimators  # type: ignore

        estimators = all_estimators(estimator_types="forecaster", as_dataframe=True)
        items = []
        for row in estimators.to_dict("records"):
            cls = row.get("object") or row.get("class")
            name = row.get("name") or getattr(cls, "__name__", None)
            module = row.get("module") or getattr(cls, "__module__", None)
            if not cls or not name or not module:
                continue
            items.append({"name": str(name), "class_path": f"{module}.{name}"})
        items.sort(key=lambda item: item["name"].lower())
        return {"available": True, "estimators": items}
    except Exception as exc:
        return {"available": False, "error": str(exc), "estimators": []}


def _call_tool_raw(func: Any) -> Any:
    return unwrap_tool_callable(func)


def _get_models_impl(*, method: Optional[str] = None, detail: str = "compact") -> Any:
    return _call_tool_raw(_forecast_models_list_tool)(method=method, detail=detail)


def _run_forecast_generate_impl(request: Any) -> Dict[str, Any]:
    return call_tool_sync_structured(_forecast_generate_tool, request=request)


def _run_forecast_backtest_impl(request: Any) -> Dict[str, Any]:
    return call_tool_sync_structured(_forecast_backtest_tool, request=request)


def _forecast_vol_impl(request: Any) -> Dict[str, Any]:
    return call_tool_sync_structured(_forecast_volatility_tool, request=request)


def _web_api_gateway():
    return create_mt5_gateway(
        adapter=mt5,
        ensure_connection_impl=mt5_connection._ensure_connection,
    )


def _readiness_payload() -> tuple[Dict[str, Any], int]:
    connection_error = mt5_connection_error(
        create_mt5_gateway(
            adapter=mt5,
            ensure_connection_impl=ensure_mt5_connection_or_raise,
        )
    )
    if connection_error is None:
        return (
            {
                "service": "mtdata-webui",
                "status": "ok",
                "ready": True,
                "components": {
                    "mt5_connection": {
                        "status": "ok",
                    }
                },
            },
            200,
        )
    component = {
        "status": "error",
        "error": connection_error.get("error"),
        "error_code": connection_error.get("error_code"),
    }
    if connection_error.get("request_id"):
        component["request_id"] = connection_error["request_id"]
    if connection_error.get("remediation"):
        component["remediation"] = connection_error["remediation"]
    return (
        {
            "service": "mtdata-webui",
            "status": "degraded",
            "ready": False,
            "components": {
                "mt5_connection": component,
            },
        },
        503,
    )


@api_router.get("/timeframes")
def get_timeframes() -> Dict[str, Any]:
    return {"timeframes": list(TIMEFRAME_MAP)}


@api_router.get("/instruments")
def get_instruments(search: Optional[str] = Query(None), limit: Optional[int] = Query(None, ge=1)) -> Dict[str, Any]:
    return _get_instruments_response(
        search=search,
        limit=limit,
        mt5=_web_api_gateway(),
        extract_group_path=_extract_group_path_util,
    )


@api_router.get("/methods")
def get_methods(
    extras: Optional[str] = None,
) -> Dict[str, Any]:
    return _get_methods_response(get_methods_impl=_get_methods_impl, extras=extras)


@api_router.get("/models")
def get_models(
    method: Optional[str] = Query(None),
    extras: Optional[str] = None,
) -> Dict[str, Any]:
    return _get_models_response(
        get_models_impl=_get_models_impl,
        method=method,
        extras=extras,
    )


@api_router.get("/volatility/methods")
def get_vol_methods() -> Dict[str, Any]:
    return _get_vol_methods_response(get_vol_methods=_get_vol_methods)


@api_router.get("/sktime/estimators")
def get_sktime_estimators() -> Dict[str, Any]:
    return _list_sktime_forecasters()


@api_router.get("/denoise/methods")
def get_denoise_methods() -> Dict[str, Any]:
    return _get_denoise_methods_response(get_denoise_methods=_get_denoise_methods)


@api_router.get("/dimred/methods")
def get_dimred_methods() -> Dict[str, Any]:
    return _get_dimred_methods_response(list_dimred_methods=_list_dimred_methods)


@api_router.get("/denoise/wavelets")
def get_wavelets() -> Dict[str, Any]:
    return _get_wavelets_response()


@api_router.get("/history")
def get_history(
    symbol: str = Query(...),
    timeframe: str = Query("H1"),
    limit: int = Query(20, ge=1, le=20000),
    start: Optional[str] = Query(None),
    end: Optional[str] = Query(None),
    ohlcv: Optional[str] = Query("ohlc"),
    include_spread: bool = Query(
        False,
        description="Append historical candle spread to each row.",
    ),
    include_incomplete: bool = Query(False, description="Include the latest forming candle."),
    allow_stale: bool = Query(False, description="Return data even when freshness checks fail."),
    indicators: Optional[str] = Query(
        None,
        description="Indicator specification forwarded to data_fetch_candles.",
    ),
    timestamp_format: Literal["epoch", "iso"] = Query(
        "epoch",
        description="Timestamp encoding for returned candle rows.",
    ),
    denoise_method: Optional[str] = Query(None, description="Denoise method name; if set, returns extra *_dn columns."),
    denoise_params: Optional[str] = Query(None, description="JSON or k=v list of denoise params."),
) -> Dict[str, Any]:
    return _get_history_response(
        symbol=symbol,
        timeframe=timeframe,
        limit=limit,
        start=start,
        end=end,
        ohlcv=ohlcv,
        include_spread=include_spread,
        include_incomplete=include_incomplete,
        allow_stale=allow_stale,
        indicators=indicators,
        timestamp_format=timestamp_format,
        denoise_method=denoise_method,
        denoise_params=denoise_params,
        fetch_candles_impl=_fetch_candles_impl,
        get_denoise_methods=_get_denoise_methods,
        normalize_denoise_spec=_norm_dn,
        mt5_config=mt5_config,
    )


@api_router.get("/pivots")
def get_pivots(
    symbol: str = Query(...),
    timeframe: str = Query("H1"),
    method: str = Query("classic"),
) -> Dict[str, Any]:
    return _get_pivots_response(
        symbol=symbol,
        timeframe=timeframe,
        method=method,
        pivot_tool=pivot_compute_points,
        call_tool_raw=_call_tool_raw,
    )


@api_router.get("/support-resistance")
def get_support_resistance(
    symbol: str = Query(...),
    timeframe: str = Query("H1"),
    lookback: Optional[int] = Query(None, ge=100, le=20000),
    tolerance_pct: float = Query(0.0015, ge=0.0, le=0.05),
    min_touches: int = Query(2, ge=1),
    max_levels: int = Query(4, ge=1, le=20),
    max_distance_pct: Optional[float] = Query(5.0, ge=0.0, le=100.0),
    volume_weighting: Literal["off", "auto"] = Query("off"),
    reaction_bars: int = Query(6, ge=1),
    adx_period: int = Query(14, ge=1),
    decay_half_life_bars: Optional[int] = Query(None, ge=1),
    extras: Optional[str] = None,
) -> Dict[str, Any]:
    return _get_support_resistance_response(
        symbol=symbol,
        timeframe=timeframe,
        lookback=lookback,
        tolerance_pct=tolerance_pct,
        min_touches=min_touches,
        max_levels=max_levels,
        max_distance_pct=max_distance_pct,
        volume_weighting=volume_weighting,
        reaction_bars=reaction_bars,
        adx_period=adx_period,
        decay_half_life_bars=decay_half_life_bars,
        extras=extras,
        fetch_history_impl=_fetch_history_impl,
    )


@api_router.get("/tick")
def get_tick(symbol: str = Query(...)) -> Dict[str, Any]:
    return _get_tick_response(
        symbol=symbol,
        mt5=_web_api_gateway(),
        ensure_symbol_ready=_ensure_symbol_ready,
    )


@api_router.post("/forecast/price")
def post_forecast_price(body: ForecastPriceBody) -> Dict[str, Any]:
    return _post_forecast_price_response(body=body, forecast_generate_use_case=_run_forecast_generate_impl)


@api_router.post("/forecast/volatility")
def post_forecast_volatility(body: ForecastVolBody) -> Dict[str, Any]:
    return _post_forecast_volatility_response(body=body, forecast_vol_impl=_forecast_vol_impl)


@api_router.post("/backtest")
def post_backtest(body: BacktestBody) -> Dict[str, Any]:
    return _post_backtest_response(body=body, backtest_use_case=_run_forecast_backtest_impl)


@api_router.get("/health")
def health() -> Dict[str, Any]:
    return {"service": "mtdata-webui", "status": "ok"}


@api_router.get("/ready")
def ready() -> SafeJSONResponse:
    payload, status_code = _readiness_payload()
    return SafeJSONResponse(status_code=status_code, content=payload)


@app.get("/health")
def health_root() -> Dict[str, Any]:
    return health()


@app.get("/ready")
def ready_root() -> SafeJSONResponse:
    return ready()


@app.get("/")
def root() -> Dict[str, Any]:
    return health()


for _prefix in API_PREFIXES:
    app.include_router(api_router, prefix=_prefix)


mount_webui(app)


def main_webapi() -> None:
    """Entry point to run the FastAPI web server."""
    load_environment()
    _clear_api_access_runtime_settings_cache()
    run_webapi(app)
