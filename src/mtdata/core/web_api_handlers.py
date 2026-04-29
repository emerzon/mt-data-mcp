"""Endpoint orchestration helpers for the Web API transport."""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, List, NoReturn, Optional

from fastapi import HTTPException

from ..forecast.exceptions import ForecastError
from ..forecast.forecast_methods import get_forecast_methods_payload
from ..utils.mt5 import MT5ConnectionError
from ..utils.support_resistance import compact_support_resistance_payload
from ..utils.utils import _UNPARSED_BOOL, _parse_bool_like
from .cli_formatting import _sanitize_json_compat
from .error_envelope import build_http_error_detail
from .mt5_gateway import get_default_mt5_gateway
from .output_contract import ensure_common_meta, output_extras_shape_detail
from .pivot import compute_support_resistance_payload
from .runtime_metadata import build_runtime_timezone_meta
from .tool_calling import resolve_sync_tool_result
from .web_api_models import BacktestBody, ForecastPriceBody, ForecastVolBody

logger = logging.getLogger(__name__)


def _shape_detail_from_extras(extras: Any) -> str:
    try:
        return output_extras_shape_detail(extras)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _http_error(
    status_code: int,
    message: Any,
    *,
    code: str,
    operation: str,
    details: Optional[Dict[str, Any]] = None,
) -> HTTPException:
    payload = build_http_error_detail(
        message,
        code=code,
        operation=operation,
        details=details,
    )
    log_fn = logger.error if status_code >= 500 else logger.warning
    log_fn(
        "transport=web_api operation=%s request_id=%s status=%s error=%s",
        operation,
        payload["request_id"],
        status_code,
        payload["error"],
    )
    return HTTPException(status_code=status_code, detail=payload)


def _raise_history_fetch_error(exc: Exception) -> NoReturn:
    if isinstance(exc, MT5ConnectionError):
        raise _http_error(
            503,
            str(exc),
            code="history_mt5_unavailable",
            operation="get_history",
        )
    if isinstance(exc, (ValueError, TypeError, KeyError)):
        raise _http_error(
            400,
            f"history fetch failed: {exc}",
            code="history_fetch_failed",
            operation="get_history",
        )
    logger.exception("transport=web_api operation=get_history unhandled_exception")
    raise _http_error(
        500,
        "History fetch failed.",
        code="history_fetch_internal_error",
        operation="get_history",
    )


def _raise_internal_handler_error(*, operation: str, code: str, message: str) -> NoReturn:
    logger.exception("transport=web_api operation=%s unhandled_exception", operation)
    raise _http_error(500, message, code=code, operation=operation)


def _require_mt5_connection() -> None:
    mt5 = get_default_mt5_gateway()
    try:
        mt5.ensure_connection()
    except MT5ConnectionError as exc:
        raise _http_error(
            503,
            str(exc),
            code="mt5_connection_error",
            operation="require_mt5_connection",
        )


def _history_denoise_bool(value: Any, *, field_name: str) -> bool:
    parsed = _parse_bool_like(value)
    if parsed is _UNPARSED_BOOL:
        raise _http_error(
            400,
            f"denoise_params.{field_name} must be a boolean value.",
            code="denoise_params_invalid",
            operation="get_history",
        )
    return bool(parsed)


def _history_denoise_choice(value: Any, *, field_name: str, allowed: set[str]) -> str:
    if not isinstance(value, str):
        raise _http_error(
            400,
            f"denoise_params.{field_name} must be a string.",
            code="denoise_params_invalid",
            operation="get_history",
        )
    normalized = value.strip().lower()
    if normalized not in allowed:
        allowed_text = ", ".join(sorted(allowed))
        raise _http_error(
            400,
            f"denoise_params.{field_name} must be one of: {allowed_text}.",
            code="denoise_params_invalid",
            operation="get_history",
        )
    return normalized


def _history_denoise_params_dict(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise _http_error(
            400,
            "denoise_params.params must be a JSON object.",
            code="denoise_params_invalid",
            operation="get_history",
        )
    return dict(value)


def _history_denoise_columns(value: Any) -> List[str]:
    if isinstance(value, str):
        columns = [col.strip() for col in value.split(",") if col.strip()]
    elif isinstance(value, list):
        columns = []
        for index, item in enumerate(value):
            if not isinstance(item, str):
                raise _http_error(
                    400,
                    f"denoise_params.columns[{index}] must be a string.",
                    code="denoise_params_invalid",
                    operation="get_history",
                )
            name = item.strip()
            if name:
                columns.append(name)
    else:
        raise _http_error(
            400,
            "denoise_params.columns must be a string or list of strings.",
            code="denoise_params_invalid",
            operation="get_history",
        )
    if not columns:
        raise _http_error(
            400,
            "denoise_params.columns must contain at least one column name.",
            code="denoise_params_invalid",
            operation="get_history",
        )
    return columns


def get_instruments_response(
    *,
    search: Optional[str],
    limit: Optional[int],
    mt5: Any,
    extract_group_path: Callable[[Any], str],
) -> Dict[str, Any]:
    _require_mt5_connection()
    symbols = mt5.symbols_get()
    if symbols is None:
        raise _http_error(
            500,
            f"symbols_get failed: {mt5.last_error()}",
            code="symbols_get_failed",
            operation="get_instruments",
        )
    items: List[Dict[str, Any]] = []
    query = (search or "").strip().lower()
    only_visible = False if query else True
    for symbol in symbols:
        try:
            if only_visible and not getattr(symbol, "visible", False):
                continue
            name = getattr(symbol, "name", "") or ""
            desc = getattr(symbol, "description", "") or ""
            group = extract_group_path(symbol)
            if query:
                haystack = " ".join([name, desc, group]).lower()
                if query not in haystack:
                    continue
            items.append({"name": name, "group": group, "description": desc})
        except Exception:
            continue
    if limit and limit > 0:
        items = items[: int(limit)]
    return {"items": items}


def _compact_forecast_method_definition(method_def: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key in ("method", "available", "requires", "category", "description", "supports_ci"):
        value = method_def.get(key)
        if value is not None:
            out[key] = value
    return out


def get_methods_response(
    *,
    get_methods_impl: Callable[[], Any],
    extras: Any = None,
) -> Dict[str, Any]:
    data = get_methods_impl()
    if not isinstance(data, dict) or data.get("methods") is None:
        return {"methods": []}
    methods = data.get("methods")
    if not isinstance(methods, list):
        return {"methods": []}
    try:
        payload = get_forecast_methods_payload(method_data=data)
    except Exception:
        return data
    detail = _shape_detail_from_extras(extras)
    if detail != "compact":
        return payload
    methods_payload = payload.get("methods")
    if not isinstance(methods_payload, list):
        return {"methods": []}
    out = dict(payload)
    out["methods"] = [
        _compact_forecast_method_definition(method_def)
        for method_def in methods_payload
        if isinstance(method_def, dict)
    ]
    out["detail"] = "compact"
    return out


def get_models_response(
    *,
    get_models_impl: Callable[..., Any],
    method: Optional[str],
    extras: Any = None,
) -> Dict[str, Any]:
    detail = _shape_detail_from_extras(extras)
    data = get_models_impl(method=method, detail=detail)
    if not isinstance(data, dict):
        return {"success": True, "detail": detail, "count": 0, "models": []}
    models = data.get("models")
    if not isinstance(models, list):
        return {"success": True, "detail": detail, "count": 0, "models": []}
    return data


def get_vol_methods_response(*, get_vol_methods: Callable[[], Any]) -> Dict[str, Any]:
    data = get_vol_methods()
    if not isinstance(data, dict):
        return {"methods": []}
    return data


def get_denoise_methods_response(*, get_denoise_methods: Callable[[], Any]) -> Dict[str, Any]:
    data = get_denoise_methods()
    if isinstance(data, dict) and data.get("methods") is not None:
        return data
    return {"methods": []}


def get_dimred_methods_response(*, list_dimred_methods: Callable[[], Dict[str, Any]]) -> Dict[str, Any]:
    base = list_dimred_methods()
    param_suggestions: Dict[str, Any] = {
        "pca": [
            {"name": "n_components", "type": "int", "default": 5, "description": "Target components (1..features)."},
        ],
        "svd": [
            {"name": "n_components", "type": "int", "default": 5, "description": "Target components for TruncatedSVD."},
        ],
        "spca": [{"name": "n_components", "type": "int", "default": 5}],
        "kpca": [
            {"name": "n_components", "type": "int", "default": 5},
            {"name": "kernel", "type": "str", "default": "rbf"},
            {"name": "gamma", "type": "float|null", "default": None},
        ],
        "isomap": [
            {"name": "n_components", "type": "int", "default": 2},
            {"name": "n_neighbors", "type": "int", "default": 10},
        ],
        "laplacian": [
            {"name": "n_components", "type": "int", "default": 2},
            {"name": "n_neighbors", "type": "int", "default": 10},
        ],
        "umap": [
            {"name": "n_components", "type": "int", "default": 2},
            {"name": "n_neighbors", "type": "int", "default": 15},
            {"name": "min_dist", "type": "float", "default": 0.1},
        ],
        "diffusion": [
            {"name": "n_components", "type": "int", "default": 2},
            {"name": "alpha", "type": "float", "default": 0.5},
            {"name": "epsilon", "type": "float|null", "default": None},
            {"name": "k", "type": "int|null", "default": None},
        ],
        "tsne": [
            {"name": "n_components", "type": "int", "default": 2},
            {"name": "perplexity", "type": "float", "default": 30.0},
            {"name": "learning_rate", "type": "float", "default": 200.0},
            {"name": "n_iter", "type": "int", "default": 1000},
        ],
        "dreams_cne": [
            {"name": "n_components", "type": "int", "default": 2},
            {"name": "k", "type": "int", "default": 15},
            {"name": "negative_samples", "type": "int", "default": 500},
            {"name": "n_epochs", "type": "int", "default": 250},
            {"name": "batch_size", "type": "int", "default": 4096},
            {"name": "learning_rate", "type": "float", "default": 0.001},
            {"name": "parametric", "type": "bool", "default": True},
            {"name": "device", "type": "str", "default": "auto"},
            {"name": "regularizer", "type": "bool", "default": True},
            {"name": "reg_lambda", "type": "float", "default": 0.0005},
            {"name": "reg_scaling", "type": "str", "default": "norm"},
        ],
    }
    items = []
    for name, info in base.items():
        items.append(
            {
                "method": name,
                "available": bool(info.get("available")),
                "description": info.get("description"),
                "params": param_suggestions.get(name, []),
            }
        )
    return {"methods": items}


def get_wavelets_response() -> Dict[str, Any]:
    try:
        import pywt  # type: ignore
    except Exception:
        return {"available": False, "families": [], "wavelets": [], "by_family": {}}
    try:
        families = list(pywt.families())  # type: ignore[attr-defined]
    except Exception:
        families = []
    by_family: Dict[str, List[str]] = {}
    flat: List[str] = []
    if families:
        for family in families:
            names: List[str] = []
            try:
                names = list(pywt.wavelist(family))  # type: ignore[attr-defined]
            except Exception:
                try:
                    names = list(pywt.wavelist(family, kind="discrete"))  # type: ignore[attr-defined]
                except Exception:
                    names = []
            by_family[family] = names
            for wavelet in names:
                if wavelet not in flat:
                    flat.append(wavelet)
    else:
        try:
            flat = list(pywt.wavelist(kind="discrete"))  # type: ignore[attr-defined]
        except Exception:
            try:
                flat = list(pywt.wavelist())  # type: ignore[attr-defined]
            except Exception:
                flat = []
    return {"available": True, "families": families, "wavelets": flat, "by_family": by_family}


def get_history_response(  # noqa: C901
    *,
    symbol: str,
    timeframe: str,
    limit: int,
    start: Optional[str],
    end: Optional[str],
    ohlcv: Optional[str],
    include_spread: bool,
    include_incomplete: bool,
    denoise_method: Optional[str],
    denoise_params: Optional[str],
    fetch_candles_impl: Callable[..., Any],
    get_denoise_methods: Callable[[], Any],
    normalize_denoise_spec: Callable[..., Any],
    mt5_config: Any,
    include_used_timezone: bool,
) -> Dict[str, Any]:
    _require_mt5_connection()
    denoise_method_val = denoise_method.strip() if isinstance(denoise_method, str) else None
    denoise_params_val = denoise_params if isinstance(denoise_params, str) else None

    denoise_spec: Optional[Dict[str, Any]] = None
    if denoise_method_val:
        try:
            meta = get_denoise_methods()
            if isinstance(meta, dict):
                methods = {method.get("method"): method for method in (meta.get("methods") or [])}
                method_meta = methods.get(denoise_method_val)
                if not method_meta or not bool(method_meta.get("available", True)):
                    req = method_meta.get("requires") if method_meta else ""
                    suffix = f"Requires {req}" if req else ""
                    raise _http_error(
                        400,
                        f"Denoise method '{denoise_method_val}' is not available. {suffix}".strip(),
                        code="denoise_method_unavailable",
                        operation="get_history",
                    )
        except HTTPException:
            raise
        except Exception as exc:
            logger.warning(
                "transport=web_api operation=get_history denoise_validation_failed error=%s",
                exc,
            )
            raise _http_error(
                500,
                "Denoise method validation failed.",
                code="denoise_validation_failed",
                operation="get_history",
            )

        spec_input: Dict[str, Any] = {
            "method": denoise_method_val,
            "when": "post_ti",
            "columns": ["close"],
            "keep_original": True,
            "suffix": "_dn",
            "params": {},
        }
        if denoise_params_val:
            try:
                payload = json.loads(denoise_params_val)
                if isinstance(payload, dict):
                    if "params" in payload:
                        spec_input["params"] = _history_denoise_params_dict(payload.pop("params"))
                    else:
                        reserved = {"columns", "when", "causality", "keep_original"}
                        extra_params = {key: value for key, value in payload.items() if key not in reserved}
                        if extra_params:
                            spec_input["params"] = extra_params
                    if "columns" in payload:
                        spec_input["columns"] = _history_denoise_columns(payload["columns"])
                    if "when" in payload:
                        spec_input["when"] = _history_denoise_choice(
                            payload["when"],
                            field_name="when",
                            allowed={"post_ti", "pre_ti"},
                        )
                    if "causality" in payload:
                        spec_input["causality"] = _history_denoise_choice(
                            payload["causality"],
                            field_name="causality",
                            allowed={"causal", "zero_phase"},
                        )
                    if "keep_original" in payload:
                        spec_input["keep_original"] = _history_denoise_bool(
                            payload["keep_original"],
                            field_name="keep_original",
                        )
                else:
                    raise ValueError("payload not dict")
            except HTTPException:
                raise
            except Exception:
                params_dict: Dict[str, Any] = {}
                for part in denoise_params_val.split(","):
                    if "=" in part:
                        key, value = part.split("=", 1)
                        key = key.strip()
                        value = value.strip()
                        if key in params_dict:
                            raise _http_error(
                                400,
                                f"denoise_params contains duplicate key '{key}'",
                                code="denoise_params_invalid",
                                operation="get_history",
                            )
                        try:
                            params_dict[key] = float(value) if value.replace(".", "", 1).lstrip("-").isdigit() else value
                        except Exception:
                            params_dict[key] = value
                spec_input["params"] = params_dict
        denoise_spec = normalize_denoise_spec(spec_input, default_when="post_ti")

    try:
        result = fetch_candles_impl(
            symbol=symbol,
            timeframe=timeframe,  # type: ignore[arg-type]
            limit=int(limit),
            start=start,
            end=end,
            ohlcv=ohlcv,
            include_spread=include_spread,
            indicators=None,
            denoise=denoise_spec,
            simplify=None,
            include_incomplete=include_incomplete,
            time_as_epoch=True,
        )
    except Exception as exc:
        _raise_history_fetch_error(exc)

    if not isinstance(result, dict):
        raise _http_error(500, "Unexpected history payload", code="history_payload_invalid", operation="get_history")
    if result.get("error"):
        raise _http_error(400, str(result["error"]), code="history_tool_error", operation="get_history")

    rows_raw = result.get("data")
    rows: List[Dict[str, Any]] = rows_raw if isinstance(rows_raw, list) else []

    if not include_incomplete and bool(result.get("last_candle_open")) and rows:
        rows = rows[:-1]

    result_out = dict(result)
    result_out["data"] = rows
    result_out["candles"] = len(rows)
    requested_value = result.get("candles_requested")
    try:
        candles_requested = int(requested_value)
    except Exception:
        candles_requested = int(len(rows))
    result_out["candles_requested"] = candles_requested
    excluded_value = result.get("candles_excluded")
    try:
        candles_excluded = int(excluded_value)
    except Exception:
        candles_excluded = max(0, candles_requested - int(len(rows)))
    result_out["candles_excluded"] = max(0, candles_excluded)
    response = ensure_common_meta(
        result_out,
        tool_name="data_fetch_candles",
        mt5_config=mt5_config,
    )
    meta_in = response.get("meta")
    if not isinstance(meta_in, dict):
        return response
    runtime_in = meta_in.get("runtime")
    if not isinstance(runtime_in, dict):
        return response
    timezone_in = runtime_in.get("timezone")
    if include_used_timezone:
        if isinstance(timezone_in, dict) and "used" in timezone_in:
            return response

        timezone_out = build_runtime_timezone_meta(
            response,
            mt5_config=mt5_config,
            include_local=True,
            include_now=False,
        )
        runtime_out = dict(runtime_in)
        runtime_out["timezone"] = timezone_out
        meta_out = dict(meta_in)
        meta_out["runtime"] = runtime_out

        response_out = dict(response)
        response_out["meta"] = meta_out
        return response_out

    if not isinstance(timezone_in, dict):
        return response

    runtime_out = dict(runtime_in)
    runtime_out.pop("timezone", None)
    meta_out = dict(meta_in)
    if runtime_out:
        meta_out["runtime"] = runtime_out
    else:
        meta_out.pop("runtime", None)

    response_out = dict(response)
    if meta_out:
        response_out["meta"] = meta_out
    else:
        response_out.pop("meta", None)
    return response_out


def get_pivots_response(
    *,
    symbol: str,
    timeframe: str,
    method: str,
    pivot_tool: Any,
    call_tool_raw: Callable[[Any], Any],
) -> Dict[str, Any]:
    tool = call_tool_raw(pivot_tool)
    try:
        result = resolve_sync_tool_result(tool(symbol=symbol, timeframe=timeframe))
    except TypeError:
        result = resolve_sync_tool_result(pivot_tool(symbol=symbol, timeframe=timeframe))
    except Exception as exc:
        raise _http_error(
            500,
            f"pivot compute failed: {exc}",
            code="pivot_compute_failed",
            operation="get_pivots",
        )

    if isinstance(result, str):
        try:
            result = json.loads(result)
        except Exception:
            raise _http_error(
                500,
                "Unexpected pivot output format",
                code="pivot_output_invalid",
                operation="get_pivots",
            )

    if isinstance(result, dict) and result.get("error"):
        raise _http_error(
            400,
            str(result["error"]),
            code="pivot_tool_error",
            operation="get_pivots",
        )
    if not isinstance(result, dict):
        raise _http_error(
            500,
            "Pivot tool returned non-JSON payload",
            code="pivot_payload_invalid",
            operation="get_pivots",
        )

    levels = []
    method_key = str(method).lower().strip()
    for row in result.get("levels", []) or []:
        level_name = row.get("level") or row.get("Level")
        value = row.get(method_key)
        if level_name is None or value is None:
            continue
        try:
            levels.append({"level": str(level_name), "value": float(value)})
        except Exception:
            continue
    if not levels:
        raise _http_error(
            404,
            f"No pivot levels for method {method}",
            code="pivot_levels_missing",
            operation="get_pivots",
        )
    return {
        "levels": levels,
        "period": result.get("period"),
        "symbol": result.get("symbol", symbol),
        "timeframe": result.get("timeframe", timeframe),
        "method": method_key,
    }


def get_support_resistance_response(
    *,
    symbol: str,
    timeframe: str,
    limit: int,
    tolerance_pct: float,
    min_touches: int,
    max_levels: int,
    extras: Any,
    fetch_history_impl: Callable[..., Any],
) -> Dict[str, Any]:
    try:
        result = compute_support_resistance_payload(
            fetch_history_impl=fetch_history_impl,
            symbol=symbol,
            timeframe=timeframe,
            limit=int(limit),
            tolerance_pct=float(tolerance_pct),
            min_touches=int(min_touches),
            max_levels=int(max_levels),
            max_distance_pct=None,
            volume_weighting="off",
            reaction_bars=6,
            adx_period=14,
            decay_half_life_bars=None,
        )
    except Exception as exc:
        message = str(exc)
        status_code = 404 if "No history available" in message else 400
        detail = message if status_code == 404 else f"history fetch failed: {message}"
        raise _http_error(
            status_code,
            detail,
            code="support_resistance_history_failed" if status_code != 404 else "support_resistance_history_missing",
            operation="get_support_resistance",
        )

    if not isinstance(result, dict) or not result.get("levels"):
        raise _http_error(
            404,
            "No support/resistance levels detected",
            code="support_resistance_levels_missing",
            operation="get_support_resistance",
        )
    if _shape_detail_from_extras(extras) == "compact":
        return compact_support_resistance_payload(result)
    return result


def get_tick_response(
    *,
    symbol: str,
    mt5: Any,
    ensure_symbol_ready: Callable[[str], Any],
) -> Dict[str, Any]:
    _require_mt5_connection()
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        err = ensure_symbol_ready(symbol)
        if err:
            info = mt5.symbol_info(symbol)
            if info is None:
                raise _http_error(404, f"Unknown symbol {symbol}", code="unknown_symbol", operation="get_tick")
            raise _http_error(500, str(err), code="tick_symbol_ready_failed", operation="get_tick")
        tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        raise _http_error(404, f"No tick data for {symbol}", code="tick_data_missing", operation="get_tick")
    return {
        "symbol": symbol,
        "time": float(tick.time),
        "bid": float(tick.bid),
        "ask": float(tick.ask),
        "last": float(tick.last),
        "volume": float(tick.volume),
    }


def post_forecast_price_response(*, body: ForecastPriceBody, forecast_generate_use_case: Callable[..., Any]) -> Dict[str, Any]:
    try:
        result = forecast_generate_use_case(body.to_domain_request())
    except ForecastError as exc:
        raise _http_error(400, str(exc), code="forecast_error", operation="post_forecast_price")
    if isinstance(result, dict) and result.get("error"):
        raise _http_error(400, str(result["error"]), code="forecast_tool_error", operation="post_forecast_price")
    # Sanitize NaN/inf values to null for JSON compatibility
    return _sanitize_json_compat(result)


def post_forecast_volatility_response(*, body: ForecastVolBody, forecast_vol_impl: Callable[..., Any]) -> Dict[str, Any]:
    try:
        result = forecast_vol_impl(
            symbol=body.symbol,
            timeframe=body.timeframe,  # type: ignore[arg-type]
            horizon=body.horizon,
            method=body.method,  # type: ignore[arg-type]
            proxy=body.proxy,  # type: ignore[arg-type]
            params=body.params,
            as_of=body.as_of,
            denoise=body.denoise,
        )
    except HTTPException:
        raise
    except ForecastError as exc:
        raise _http_error(400, str(exc), code="forecast_volatility_error", operation="post_forecast_volatility")
    except MT5ConnectionError as exc:
        raise _http_error(
            503,
            str(exc),
            code="forecast_volatility_mt5_unavailable",
            operation="post_forecast_volatility",
        )
    except Exception:
        _raise_internal_handler_error(
            operation="post_forecast_volatility",
            code="forecast_volatility_internal_error",
            message="Forecast volatility computation failed.",
        )
    if isinstance(result, dict) and result.get("error"):
        raise _http_error(400, str(result["error"]), code="forecast_volatility_error", operation="post_forecast_volatility")
    # Sanitize NaN/inf values to null for JSON compatibility
    return _sanitize_json_compat(result)


def post_backtest_response(*, body: BacktestBody, backtest_use_case: Callable[..., Any]) -> Dict[str, Any]:
    try:
        result = backtest_use_case(body.to_domain_request())
    except HTTPException:
        raise
    except ForecastError as exc:
        raise _http_error(400, str(exc), code="backtest_error", operation="post_backtest")
    except MT5ConnectionError as exc:
        raise _http_error(503, str(exc), code="backtest_mt5_unavailable", operation="post_backtest")
    except Exception:
        _raise_internal_handler_error(
            operation="post_backtest",
            code="backtest_internal_error",
            message="Backtest computation failed.",
        )
    if isinstance(result, dict) and result.get("error"):
        raise _http_error(400, str(result["error"]), code="backtest_error", operation="post_backtest")
    # Sanitize NaN/inf values to null for JSON compatibility
    return _sanitize_json_compat(result)
