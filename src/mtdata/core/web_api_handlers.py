"""Endpoint orchestration helpers for the Web API transport."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from fastapi import HTTPException

from ..forecast.exceptions import ForecastError
from ..utils.mt5 import MT5ConnectionError
from .runtime_metadata import build_runtime_timezone_meta
from .error_envelope import build_http_error_detail
from .mt5_gateway import get_default_mt5_gateway
from .web_api_models import BacktestBody, ForecastPriceBody, ForecastVolBody

logger = logging.getLogger(__name__)


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


def _require_mt5_connection() -> None:
    mt5 = get_default_mt5_gateway()
    try:
        mt5.ensure_connection()
    except MT5ConnectionError as exc:
        raise _http_error(
            500,
            str(exc),
            code="mt5_connection_error",
            operation="require_mt5_connection",
        )


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


def get_methods_response(*, get_methods_impl: Callable[[], Any]) -> Dict[str, Any]:
    data = get_methods_impl()
    if not isinstance(data, dict) or data.get("methods") is None:
        return {"methods": []}
    try:
        import importlib.util as _importlib_util

        def _has(module_name: str) -> bool:
            try:
                return _importlib_util.find_spec(module_name) is not None
            except Exception:
                return False

        for method in data["methods"]:
            name = method.get("method")
            if name == "timesfm":
                ok = _has("timesfm")
                if ok:
                    ok = _has("timesfm.timesfm_2p5_torch") or _has("timesfm.timesfm_2p5") or ok
                method["available"] = bool(ok)
                if ok:
                    method["requires"] = []
            elif name in ("chronos_bolt", "chronos2"):
                ok = _has("chronos")
                method["available"] = bool(ok)
                if ok:
                    method["requires"] = []
            elif name == "lag_llama":
                ok = _has("lag_llama")
                method["available"] = bool(ok)
                if ok:
                    method["requires"] = []
    except Exception:
        pass
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


def get_history_response(
    *,
    symbol: str,
    timeframe: str,
    limit: int,
    start: Optional[str],
    end: Optional[str],
    ohlcv: Optional[str],
    include_incomplete: bool,
    denoise_method: Optional[str],
    denoise_params: Optional[str],
    fetch_candles_impl: Callable[..., Any],
    get_denoise_methods: Callable[[], Any],
    normalize_denoise_spec: Callable[..., Any],
    mt5_config: Any,
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
        except Exception:
            pass

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
                        spec_input["params"] = payload.pop("params") or {}
                    else:
                        reserved = {"columns", "when", "causality", "keep_original"}
                        extra_params = {key: value for key, value in payload.items() if key not in reserved}
                        if extra_params:
                            spec_input["params"] = extra_params
                    if "columns" in payload:
                        cols = payload["columns"]
                        if isinstance(cols, str):
                            cols = [col.strip() for col in cols.split(",") if col.strip()]
                        elif isinstance(cols, list):
                            cols = [str(col).strip() for col in cols if str(col).strip()]
                        if cols:
                            spec_input["columns"] = cols
                    if "when" in payload:
                        spec_input["when"] = payload["when"]
                    if "causality" in payload:
                        spec_input["causality"] = payload["causality"]
                    if "keep_original" in payload:
                        spec_input["keep_original"] = bool(payload["keep_original"])
                else:
                    raise ValueError("payload not dict")
            except Exception:
                params_dict: Dict[str, Any] = {}
                for part in denoise_params_val.split(","):
                    if "=" in part:
                        key, value = part.split("=", 1)
                        key = key.strip()
                        value = value.strip()
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
            indicators=None,
            denoise=denoise_spec,
            simplify=None,
            time_as_epoch=True,
        )
    except Exception as exc:
        raise _http_error(400, f"history fetch failed: {exc}", code="history_fetch_failed", operation="get_history")

    if not isinstance(result, dict):
        raise _http_error(500, "Unexpected history payload", code="history_payload_invalid", operation="get_history")
    if result.get("error"):
        raise _http_error(400, str(result["error"]), code="history_tool_error", operation="get_history")

    rows_raw = result.get("data")
    rows: List[Dict[str, Any]] = rows_raw if isinstance(rows_raw, list) else []

    if not include_incomplete and bool(result.get("last_candle_open")) and rows:
        rows = rows[:-1]

    return {
        "bars": rows,
        "meta": {
            "runtime": {
                "timezone": build_runtime_timezone_meta(
                    result,
                    mt5_config=mt5_config,
                    include_local=False,
                    include_now=False,
                ),
            },
        },
    }


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
        result = tool(symbol=symbol, timeframe=timeframe)
    except TypeError:
        result = pivot_tool(symbol=symbol, timeframe=timeframe)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"pivot compute failed: {exc}")

    if isinstance(result, str):
        try:
            result = json.loads(result)
        except Exception:
            raise HTTPException(status_code=500, detail="Unexpected pivot output format")

    if isinstance(result, dict) and result.get("error"):
        raise HTTPException(status_code=400, detail=str(result["error"]))
    if not isinstance(result, dict):
        raise HTTPException(status_code=500, detail="Pivot tool returned non-JSON payload")

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
        raise HTTPException(status_code=404, detail=f"No pivot levels for method {method}")
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
    fetch_history_impl: Callable[..., Any],
) -> Dict[str, Any]:
    try:
        need = int(limit)
        frame = fetch_history_impl(symbol=symbol, timeframe=timeframe, need=need)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"history fetch failed: {exc}")
    if frame is None or frame.empty:
        raise HTTPException(status_code=404, detail="No history available")
    required_cols = ("high", "low", "close")
    missing = [col for col in required_cols if col not in frame.columns]
    if missing:
        missing_cols = ", ".join(missing)
        raise HTTPException(status_code=400, detail=f"Missing columns: {missing_cols}")
    if len(frame) < 3:
        raise HTTPException(status_code=400, detail="Need at least 3 bars to compute support/resistance levels")

    times = frame["time"].tolist() if "time" in frame.columns else []

    def _coerce_series(series: List[Any]) -> List[float]:
        out: List[float] = []
        for value in series:
            try:
                out.append(float(value))
            except Exception:
                out.append(float("nan"))
        return out

    highs = _coerce_series(frame["high"].tolist())
    lows = _coerce_series(frame["low"].tolist())

    def _to_epoch(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            if isinstance(value, (int, float)):
                return float(value)
            if hasattr(value, "timestamp"):
                return float(value.timestamp())
        except Exception:
            return None
        return None

    epochs = [_to_epoch(value) for value in times]

    def _format_time(timestamp: Optional[float]) -> Optional[str]:
        if timestamp is None:
            return None
        try:
            return datetime.fromtimestamp(float(timestamp), tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
        except Exception:
            return None

    def _find_extrema(values: List[float], comparator: Callable[[float, float, float], bool]) -> List[int]:
        indices: List[int] = []
        for index in range(1, len(values) - 1):
            try:
                center = float(values[index])
                previous = float(values[index - 1])
                nxt = float(values[index + 1])
            except Exception:
                continue
            if comparator(center, previous, nxt):
                indices.append(index)
        return indices

    total_bars = len(highs)

    def _cluster(indices: List[int], values: List[float], level_type: str, limit_per_type: int) -> List[Dict[str, Any]]:
        clusters: List[Dict[str, Any]] = []
        for index in sorted(indices, key=lambda item: values[item], reverse=(level_type == "resistance")):
            try:
                value = float(values[index])
            except Exception:
                continue
            assigned = None
            for cluster in clusters:
                ref = cluster["value"]
                threshold = max(abs(ref), abs(value)) * tolerance_pct
                if threshold <= 0:
                    threshold = tolerance_pct
                if abs(ref - value) <= threshold:
                    cluster["value"] = (cluster["value"] * cluster["touches"] + value) / (cluster["touches"] + 1)
                    cluster["touches"] += 1
                    cluster["indices"].append(index)
                    timestamp = epochs[index] if index < len(epochs) else None
                    if timestamp is not None:
                        if cluster["last_time"] is None or timestamp > cluster["last_time"]:
                            cluster["last_time"] = timestamp
                        if cluster["first_time"] is None or timestamp < cluster["first_time"]:
                            cluster["first_time"] = timestamp
                    assigned = cluster
                    break
            if assigned is None:
                timestamp = epochs[index] if index < len(epochs) else None
                clusters.append(
                    {
                        "type": level_type,
                        "value": value,
                        "touches": 1,
                        "indices": [index],
                        "first_time": timestamp,
                        "last_time": timestamp,
                    }
                )
        usable = [cluster for cluster in clusters if cluster["touches"] >= min_touches]
        if not usable and clusters:
            usable = clusters[:1]

        def _sort_key(cluster: Dict[str, Any]) -> tuple[int, int, float]:
            last_index = max(cluster["indices"])
            value_key = -float(cluster["value"]) if level_type == "support" else float(cluster["value"])
            return (cluster["touches"], last_index, value_key)

        usable.sort(key=_sort_key, reverse=True)
        out: List[Dict[str, Any]] = []
        for cluster in usable[:limit_per_type]:
            last_index = max(cluster["indices"])
            recency = 0.0
            if total_bars > 1:
                recency = max(0.0, 1.0 - (total_bars - 1 - last_index) / float(total_bars))
            out.append(
                {
                    "type": level_type,
                    "value": float(round(cluster["value"], 6)),
                    "touches": int(cluster["touches"]),
                    "score": float(round(cluster["touches"] + recency, 4)),
                    "first_touch": _format_time(cluster["first_time"]),
                    "last_touch": _format_time(cluster["last_time"]),
                }
            )
        return out

    resistance_levels = _cluster(_find_extrema(highs, lambda c, p, n: c >= p and c >= n), highs, "resistance", max_levels)
    support_levels = _cluster(_find_extrema(lows, lambda c, p, n: c <= p and c <= n), lows, "support", max_levels)

    def _first_valid(values: List[Optional[float]]) -> Optional[float]:
        for item in values:
            if item is not None:
                return item
        return None

    def _last_valid(values: List[Optional[float]]) -> Optional[float]:
        for item in reversed(values):
            if item is not None:
                return item
        return None

    window: Dict[str, Optional[str]] = {}
    start = _first_valid(epochs)
    end = _last_valid(epochs)
    if start is not None or end is not None:
        window = {"start": _format_time(start), "end": _format_time(end)}

    levels = resistance_levels + support_levels
    if not levels:
        raise HTTPException(status_code=404, detail="No support/resistance levels detected")

    response: Dict[str, Any] = {
        "symbol": symbol,
        "timeframe": timeframe,
        "limit": int(limit),
        "method": "swing",
        "tolerance_pct": float(tolerance_pct),
        "min_touches": int(min_touches),
        "levels": levels,
    }
    if window:
        response["window"] = window
    return response


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
    return result


def post_forecast_volatility_response(*, body: ForecastVolBody, forecast_vol_impl: Callable[..., Any]) -> Dict[str, Any]:
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
    if isinstance(result, dict) and result.get("error"):
        raise _http_error(400, str(result["error"]), code="forecast_volatility_error", operation="post_forecast_volatility")
    return result


def post_backtest_response(*, body: BacktestBody, backtest_use_case: Callable[..., Any]) -> Dict[str, Any]:
    result = backtest_use_case(body.to_domain_request())
    if isinstance(result, dict) and result.get("error"):
        raise _http_error(400, str(result["error"]), code="backtest_error", operation="post_backtest")
    return result
