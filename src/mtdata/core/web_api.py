"""
FastAPI app exposing WebUI-ready endpoints that wrap existing mtdata tools.

Initial scope:
- GET /api/timeframes
- GET /api/instruments
- GET /api/history
- GET /api/methods
- POST /api/forecast/price
- POST /api/forecast/volatility
- POST /api/backtest

This module reuses existing functions in src.mtdata.core and src.mtdata.forecast.
It performs light payload normalization for tabular endpoints and keeps parameter
surfaces close to the underlying tools. Advanced params are accepted as dicts.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.staticfiles import StaticFiles

from .constants import TIMEFRAME_MAP
from ..forecast.forecast import (
    forecast as _forecast_impl,
    get_forecast_methods_data as _get_methods_impl,
)
from ..forecast.volatility import (
    forecast_volatility as _forecast_vol_impl,
    get_volatility_methods_data as _get_vol_methods,
)
from ..forecast.backtest import forecast_backtest as _backtest_impl
from ..forecast.common import fetch_history as _fetch_history_impl
from .data import data_fetch_candles as _data_fetch_candles
from .pivot import pivot_compute_points
from importlib.util import find_spec as _find_spec

def _list_sktime_forecasters() -> Dict[str, Any]:
    if _find_spec('sktime') is None:
        return {"available": False, "error": "sktime not installed", "estimators": []}
    try:
        from sktime.registry import all_estimators  # type: ignore
        ests = all_estimators(estimator_types="forecaster", as_dataframe=True)
        items = []
        for _, row in ests.iterrows():
            cls = row.get('object') or row.get('class')
            name = row.get('name') or getattr(cls, '__name__', None)
            module = row.get('module') or getattr(cls, '__module__', None)
            if not cls or not name or not module:
                continue
            class_path = f"{module}.{name}"
            items.append({
                "name": str(name),
                "class_path": class_path,
            })
        items.sort(key=lambda x: x['name'].lower())
        return {"available": True, "estimators": items}
    except Exception as e:
        return {"available": False, "error": str(e), "estimators": []}

def _call_tool_raw(func):
    raw = getattr(func, '__wrapped__', None)
    return raw if callable(raw) else func

from ..utils.mt5 import mt5_connection, _ensure_symbol_ready
from ..utils.symbol import _extract_group_path as _extract_group_path_util
from ..utils.denoise import get_denoise_methods_data as _get_denoise_methods
from ..utils.denoise import _apply_denoise as _apply_dn, normalize_denoise_spec as _norm_dn
from ..utils.dimred import list_dimred_methods as _list_dimred_methods
import MetaTrader5 as mt5


class ForecastPriceBody(BaseModel):
    symbol: str
    timeframe: str = Field("H1")
    method: str = Field("theta")
    horizon: int = Field(12, ge=1)
    lookback: Optional[int] = Field(None, ge=1)
    as_of: Optional[str] = None
    params: Optional[Dict[str, Any]] = None
    ci_alpha: Optional[float] = Field(0.05, ge=0.0, le=0.5)
    quantity: str = Field("price")  # 'price' | 'return' | 'volatility'
    target: str = Field("price")     # 'price' | 'return'
    denoise: Optional[Dict[str, Any]] = None
    features: Optional[Dict[str, Any]] = None
    dimred_method: Optional[str] = None
    dimred_params: Optional[Dict[str, Any]] = None
    target_spec: Optional[Dict[str, Any]] = None


class ForecastVolBody(BaseModel):
    symbol: str
    timeframe: str = Field("H1")
    horizon: int = Field(1, ge=1)
    method: str = Field("ewma")
    proxy: Optional[str] = None
    params: Optional[Dict[str, Any]] = None
    as_of: Optional[str] = None
    denoise: Optional[Dict[str, Any]] = None


class BacktestBody(BaseModel):
    symbol: str
    timeframe: str = Field("H1")
    horizon: int = Field(12, ge=1)
    steps: int = Field(5, ge=1)
    spacing: int = Field(20, ge=1)
    methods: Optional[List[str]] = None
    params_per_method: Optional[Dict[str, Any]] = None
    quantity: str = Field("price")
    target: str = Field("price")
    denoise: Optional[Dict[str, Any]] = None
    params: Optional[Dict[str, Any]] = None
    features: Optional[Dict[str, Any]] = None
    dimred_method: Optional[str] = None
    dimred_params: Optional[Dict[str, Any]] = None
    slippage_bps: float = 0.0
    trade_threshold: float = 0.0


app = FastAPI(title="mtdata-webui", version="0.1.0")

# Permissive CORS by default (customize via reverse proxy in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/timeframes")
def get_timeframes() -> Dict[str, Any]:
    return {"timeframes": list(TIMEFRAME_MAP.keys())}


@app.get("/api/instruments")
def get_instruments(search: Optional[str] = Query(None), limit: Optional[int] = Query(None, ge=1)) -> Dict[str, Any]:
    if not mt5_connection._ensure_connection():
        raise HTTPException(status_code=500, detail="Failed to connect to MetaTrader5.")
    symbols = mt5.symbols_get()
    if symbols is None:
        raise HTTPException(status_code=500, detail=f"symbols_get failed: {mt5.last_error()}")
    items: List[Dict[str, Any]] = []
    q = (search or "").strip().lower()
    only_visible = False if q else True
    for s in symbols:
        try:
            if only_visible and not getattr(s, 'visible', False):
                continue
            name = getattr(s, 'name', '') or ''
            desc = getattr(s, 'description', '') or ''
            group = _extract_group_path_util(s)
            if q:
                hay = " ".join([name, desc, group]).lower()
                if q not in hay:
                    continue
            items.append({"name": name, "group": group, "description": desc})
        except Exception:
            continue
    if limit and limit > 0:
        items = items[: int(limit)]
    return {"items": items}


@app.get("/api/methods")
def get_methods() -> Dict[str, Any]:
    data = _get_methods_impl()
    if not isinstance(data, dict) or data.get("methods") is None:
        return {"methods": []}
    # Re-check dynamic availability for certain packages in the live process
    try:
        import importlib.util as _il
        def _has(mod: str) -> bool:
            try:
                return _il.find_spec(mod) is not None
            except Exception:
                return False
        for m in data["methods"]:
            name = m.get("method")
            if name == "timesfm":
                ok = _has("timesfm")
                if ok:
                    # Check common submodules used by our adapter
                    ok = _has("timesfm.timesfm_2p5_torch") or _has("timesfm.timesfm_2p5") or ok
                m["available"] = bool(ok)
                if ok:
                    m["requires"] = []
            elif name in ("chronos_bolt", "chronos2"):
                ok = _has("chronos")
                m["available"] = bool(ok)
                if ok:
                    m["requires"] = []
            elif name == "lag_llama":
                ok = _has("lag_llama")
                m["available"] = bool(ok)
                if ok:
                    m["requires"] = []
    except Exception:
        pass
    return data

@app.get("/api/volatility/methods")
def get_vol_methods() -> Dict[str, Any]:
    data = _get_vol_methods()
    if not isinstance(data, dict):
        return {"methods": []}
    return data


@app.get("/api/sktime/estimators")
def get_sktime_estimators() -> Dict[str, Any]:
    return _list_sktime_forecasters()


@app.get("/api/denoise/methods")
def get_denoise_methods() -> Dict[str, Any]:
    data = _get_denoise_methods()
    if isinstance(data, dict) and data.get("methods") is not None:
        return data
    return {"methods": []}


@app.get("/api/dimred/methods")
def get_dimred_methods() -> Dict[str, Any]:
    base = _list_dimred_methods()
    # Suggest parameter schemas for common methods
    param_suggestions: Dict[str, Any] = {
        "pca": [
            {"name": "n_components", "type": "int", "default": 5, "description": "Target components (1..features)."},
        ],
        "svd": [
            {"name": "n_components", "type": "int", "default": 5, "description": "Target components for TruncatedSVD."},
        ],
        "spca": [
            {"name": "n_components", "type": "int", "default": 5},
        ],
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
    for k, v in base.items():
        items.append({
            "method": k,
            "available": bool(v.get("available")),
            "description": v.get("description"),
            "params": param_suggestions.get(k, []),
        })
    return {"methods": items}


@app.get("/api/denoise/wavelets")
def get_wavelets() -> Dict[str, Any]:
    """List available discrete wavelet names grouped by family (if PyWavelets installed)."""
    try:
        import pywt  # type: ignore
    except Exception:
        return {"available": False, "families": [], "wavelets": [], "by_family": {}}
    try:
        # Most PyWavelets versions expose families() without arguments
        fams = list(pywt.families())  # type: ignore[attr-defined]
    except Exception:
        fams = []
    by_family: Dict[str, list] = {}
    flat: list = []
    if fams:
        for f in fams:
            names: list = []
            try:
                names = list(pywt.wavelist(f))  # type: ignore[attr-defined]
            except Exception:
                try:
                    names = list(pywt.wavelist(f, kind='discrete'))  # older/newer API variants
                except Exception:
                    names = []
            by_family[f] = names
            for w in names:
                if w not in flat:
                    flat.append(w)
    else:
        # Fallback: just list all discrete wavelets if grouping is unavailable
        try:
            flat = list(pywt.wavelist(kind='discrete'))  # type: ignore[attr-defined]
        except Exception:
            try:
                flat = list(pywt.wavelist())  # type: ignore[attr-defined]
            except Exception:
                flat = []
    return {"available": True, "families": fams, "wavelets": flat, "by_family": by_family}


@app.get("/api/history")
def get_history(
    symbol: str = Query(...),
    timeframe: str = Query("H1"),
    limit: int = Query(500, ge=1, le=20000),
    start: Optional[str] = Query(None),
    end: Optional[str] = Query(None),
    ohlcv: Optional[str] = Query("ohlc"),
    include_incomplete: bool = Query(False, description="Include the latest forming candle."),
    denoise_method: Optional[str] = Query(None, description="Denoise method name; if set, returns extra *_dn columns."),
    denoise_params: Optional[str] = Query(None, description="JSON or k=v list of denoise params."),
) -> Dict[str, Any]:
    if not mt5_connection._ensure_connection():
        raise HTTPException(status_code=500, detail="Failed to connect to MetaTrader5.")
    # If denoise requested, use data_fetch_candles to include *_dn columns; else use fast fetch
    if denoise_method:
        # Validate availability first
        try:
            dn_meta = _get_denoise_methods()
            if isinstance(dn_meta, dict):
                methods = {m.get('method'): m for m in (dn_meta.get('methods') or [])}
                m = methods.get(denoise_method)
                if not m or not bool(m.get('available', True)):
                    req = m.get('requires') if m else ''
                    raise HTTPException(status_code=400, detail=f"Denoise method '{denoise_method}' is not available. {('Requires ' + str(req)) if req else ''}")
        except HTTPException:
            raise
        except Exception:
            pass
        # Fetch bars first, then apply denoise locally to avoid text encoding/decoding
        try:
            need = int(limit)
            df = _fetch_history_impl(symbol=symbol, timeframe=timeframe, need=need, as_of=end, drop_last_live=not include_incomplete)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"history fetch failed: {e}")
        # Build denoise spec and apply

        spec_input: Dict[str, Any] = {
            "method": denoise_method,
            "when": "post_ti",
            "columns": ["close"],
            "keep_original": True,
            "suffix": "_dn",
            "params": {},
        }
        if denoise_params:
            try:
                import json as _json
                payload = _json.loads(denoise_params)
                if isinstance(payload, dict):
                    if 'params' in payload:
                        spec_input['params'] = payload.pop('params') or {}
                    else:
                        # treat remaining numeric pairs as params unless reserved keys used
                        reserved = {'columns', 'when', 'causality', 'keep_original'}
                        extra_params = {k: v for k, v in payload.items() if k not in reserved}
                        if extra_params:
                            spec_input['params'] = extra_params
                    if 'columns' in payload:
                        cols = payload['columns']
                        if isinstance(cols, str):
                            cols = [c.strip() for c in cols.split(',') if c.strip()]
                        elif isinstance(cols, list):
                            cols = [str(c).strip() for c in cols if str(c).strip()]
                        if cols:
                            spec_input['columns'] = cols
                    if 'when' in payload:
                        spec_input['when'] = payload['when']
                    if 'causality' in payload:
                        spec_input['causality'] = payload['causality']
                    if 'keep_original' in payload:
                        spec_input['keep_original'] = bool(payload['keep_original'])
                else:
                    raise ValueError('payload not dict')
            except Exception:
                params_dict: Dict[str, Any] = {}
                for part in str(denoise_params).split(','):
                    if '=' in part:
                        k, v = part.split('=', 1)
                        k = k.strip(); v = v.strip()
                        try:
                            params_dict[k] = float(v) if v.replace('.', '', 1).lstrip('-').isdigit() else v
                        except Exception:
                            params_dict[k] = v
                spec_input['params'] = params_dict
        spec = _norm_dn(spec_input, default_when='post_ti')
        try:
            _apply_dn(df, spec, default_when='post_ti')
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"denoise failed: {e}")
        cols_base = ['time', 'open', 'high', 'low', 'close', 'tick_volume']
        cols_extra = []
        if 'close_dn' in df.columns:
            cols_extra.append('close_dn')
        rows: List[Dict[str, Any]] = []
        for _, r in df.iterrows():
            rec: Dict[str, Any] = {}
            for k in cols_base + cols_extra:
                if k in df.columns:
                    rec[k] = float(r[k])
            rows.append(rec)
        return {"bars": rows}
    # Fast path without denoise
    try:
        need = int(limit)
        df = _fetch_history_impl(symbol=symbol, timeframe=timeframe, need=need, as_of=end, drop_last_live=not include_incomplete)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"history fetch failed: {e}")

    cols = ['time', 'open', 'high', 'low', 'close', 'tick_volume']
    rows: List[Dict[str, Any]] = []
    try:
        for _, r in df.iterrows():
            rows.append({k: float(r[k]) if k != 'time' else float(r[k]) for k in cols if k in df.columns})
    except Exception:
        pass
    return {"bars": rows}




@app.get("/api/pivots")
def get_pivots(
    symbol: str = Query(...),
    timeframe: str = Query("D1"),
    method: str = Query("classic"),
) -> Dict[str, Any]:
    tool = _call_tool_raw(pivot_compute_points)
    try:
        res = tool(symbol=symbol, timeframe=timeframe)
    except TypeError:
        res = pivot_compute_points(symbol=symbol, timeframe=timeframe)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"pivot compute failed: {exc}")

    if isinstance(res, str):
        try:
            res = json.loads(res)
        except Exception:
            raise HTTPException(status_code=500, detail="Unexpected pivot output format")

    if isinstance(res, dict) and res.get("error"):
        raise HTTPException(status_code=400, detail=str(res["error"]))
    if not isinstance(res, dict):
        raise HTTPException(status_code=500, detail="Pivot tool returned non-JSON payload")

    levels = []
    method_key = str(method).lower().strip()
    for row in res.get("levels", []) or []:
        lvl_name = row.get("level") or row.get("Level")
        val = row.get(method_key)
        if lvl_name is None or val is None:
            continue
        try:
            levels.append({"level": str(lvl_name), "value": float(val)})
        except Exception:
            continue
    if not levels:
        raise HTTPException(status_code=404, detail=f"No pivot levels for method {method}")
    return {
        "levels": levels,
        "period": res.get("period"),
        "symbol": res.get("symbol", symbol),
        "timeframe": res.get("timeframe", timeframe),
        "method": method_key,
    }


@app.get("/api/support-resistance")
def get_support_resistance(
    symbol: str = Query(...),
    timeframe: str = Query("H1"),
    limit: int = Query(800, ge=100, le=20000),
    tolerance_pct: float = Query(0.0015, ge=0.0, le=0.05),
    min_touches: int = Query(2, ge=1),
    max_levels: int = Query(4, ge=1, le=20),
) -> Dict[str, Any]:
    try:
        need = int(limit)
        df = _fetch_history_impl(symbol=symbol, timeframe=timeframe, need=need)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"history fetch failed: {e}")
    if df is None or df.empty:
        raise HTTPException(status_code=404, detail="No history available")
    required_cols = ("high", "low", "close")
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        missing_cols = ", ".join(missing)
        raise HTTPException(status_code=400, detail=f"Missing columns: {missing_cols}")
    if len(df) < 3:
        raise HTTPException(status_code=400, detail="Need at least 3 bars to compute support/resistance levels")

    times = df["time"].tolist() if "time" in df.columns else []

    def _coerce_series(series: List[Any]) -> List[float]:
        out: List[float] = []
        for val in series:
            try:
                out.append(float(val))
            except Exception:
                out.append(float("nan"))
        return out

    highs = _coerce_series(df["high"].tolist())
    lows = _coerce_series(df["low"].tolist())

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

    epochs = [_to_epoch(v) for v in times]

    def _format_time(ts: Optional[float]) -> Optional[str]:
        if ts is None:
            return None
        try:
            return datetime.utcfromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M")
        except Exception:
            return None

    def _find_extrema(values: List[float], comparator) -> List[int]:
        idxs: List[int] = []
        for i in range(1, len(values) - 1):
            try:
                center = float(values[i])
                prev = float(values[i - 1])
                nxt = float(values[i + 1])
            except Exception:
                continue
            if comparator(center, prev, nxt):
                idxs.append(i)
        return idxs

    total_bars = len(highs)

    def _cluster(indices: List[int], values: List[float], level_type: str, limit_per_type: int) -> List[Dict[str, Any]]:
        clusters: List[Dict[str, Any]] = []
        for idx in sorted(indices, key=lambda j: values[j], reverse=(level_type == "resistance")):
            try:
                val = float(values[idx])
            except Exception:
                continue
            assigned = None
            for cluster in clusters:
                ref = cluster["value"]
                threshold = max(abs(ref), abs(val)) * tolerance_pct
                if threshold <= 0:
                    threshold = tolerance_pct
                if abs(ref - val) <= threshold:
                    cluster["value"] = (cluster["value"] * cluster["touches"] + val) / (cluster["touches"] + 1)
                    cluster["touches"] += 1
                    cluster["indices"].append(idx)
                    ts = epochs[idx] if idx < len(epochs) else None
                    if ts is not None:
                        if cluster["last_time"] is None or ts > cluster["last_time"]:
                            cluster["last_time"] = ts
                        if cluster["first_time"] is None or ts < cluster["first_time"]:
                            cluster["first_time"] = ts
                    assigned = cluster
                    break
            if assigned is None:
                ts = epochs[idx] if idx < len(epochs) else None
                clusters.append({
                    "type": level_type,
                    "value": val,
                    "touches": 1,
                    "indices": [idx],
                    "first_time": ts,
                    "last_time": ts,
                })
        usable = [c for c in clusters if c["touches"] >= min_touches]
        if not usable and clusters:
            usable = clusters[:1]

        def sort_key(cluster: Dict[str, Any]):
            last_idx = max(cluster["indices"])
            value_key = -float(cluster["value"]) if level_type == "support" else float(cluster["value"])
            return (cluster["touches"], last_idx, value_key)

        usable.sort(key=sort_key, reverse=True)
        out: List[Dict[str, Any]] = []
        for cluster in usable[:limit_per_type]:
            last_idx = max(cluster["indices"])
            recency = 0.0
            if total_bars > 1:
                recency = max(0.0, 1.0 - (total_bars - 1 - last_idx) / float(total_bars))
            out.append({
                "type": level_type,
                "value": float(round(cluster["value"], 6)),
                "touches": int(cluster["touches"]),
                "score": float(round(cluster["touches"] + recency, 4)),
                "first_touch": _format_time(cluster["first_time"]),
                "last_touch": _format_time(cluster["last_time"]),
            })
        return out

    resistance_levels = _cluster(_find_extrema(highs, lambda c, p, n: c >= p and c >= n), highs, "resistance", max_levels)
    support_levels = _cluster(_find_extrema(lows, lambda c, p, n: c <= p and c <= n), lows, "support", max_levels)

    def _first_valid(seq: List[Optional[float]]) -> Optional[float]:
        for item in seq:
            if item is not None:
                return item
        return None

    def _last_valid(seq: List[Optional[float]]) -> Optional[float]:
        for item in reversed(seq):
            if item is not None:
                return item
        return None

    window = {}
    start = _first_valid(epochs)
    end = _last_valid(epochs)
    if start is not None or end is not None:
        window = {
            "start": _format_time(start),
            "end": _format_time(end),
        }

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


@app.get("/api/tick")
def get_tick(symbol: str = Query(...)) -> Dict[str, Any]:
    if not mt5_connection._ensure_connection():
        raise HTTPException(status_code=500, detail="Failed to connect to MetaTrader5.")
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        err = _ensure_symbol_ready(symbol)
        if err:
            info = mt5.symbol_info(symbol)
            if info is None:
                raise HTTPException(status_code=404, detail=f"Unknown symbol {symbol}")
            raise HTTPException(status_code=500, detail=str(err))
        tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        raise HTTPException(status_code=404, detail=f"No tick data for {symbol}")
    return {
        "symbol": symbol,
        "time": float(tick.time),
        "bid": float(tick.bid),
        "ask": float(tick.ask),
        "last": float(tick.last),
        "volume": float(tick.volume),
    }

@app.post("/api/forecast/price")
def post_forecast_price(body: ForecastPriceBody) -> Dict[str, Any]:
    res = _forecast_impl(
        symbol=body.symbol,
        timeframe=body.timeframe,  # type: ignore[arg-type]
        method=body.method,        # type: ignore[arg-type]
        horizon=body.horizon,
        lookback=body.lookback,
        as_of=body.as_of,
        params=body.params,
        ci_alpha=body.ci_alpha,
        quantity=body.quantity,    # type: ignore[arg-type]
        target=body.target,        # type: ignore[arg-type]
        denoise=body.denoise,
        features=body.features,
        dimred_method=body.dimred_method,
        dimred_params=body.dimred_params,
        target_spec=body.target_spec,
    )
    if isinstance(res, dict) and res.get("error"):
        raise HTTPException(status_code=400, detail=str(res["error"]))
    return res  # already JSON-like with forecast_price/forecast_time and optional intervals


@app.post("/api/forecast/volatility")
def post_forecast_volatility(body: ForecastVolBody) -> Dict[str, Any]:
    res = _forecast_vol_impl(
        symbol=body.symbol,
        timeframe=body.timeframe,  # type: ignore[arg-type]
        horizon=body.horizon,
        method=body.method,        # type: ignore[arg-type]
        proxy=body.proxy,          # type: ignore[arg-type]
        params=body.params,
        as_of=body.as_of,
        denoise=body.denoise,
    )
    if isinstance(res, dict) and res.get("error"):
        raise HTTPException(status_code=400, detail=str(res["error"]))
    return res


@app.post("/api/backtest")
def post_backtest(body: BacktestBody) -> Dict[str, Any]:
    res = _backtest_impl(
        symbol=body.symbol,
        timeframe=body.timeframe,  # type: ignore[arg-type]
        horizon=body.horizon,
        steps=body.steps,
        spacing=body.spacing,
        methods=body.methods,
        params_per_method=body.params_per_method,
        quantity=body.quantity,    # type: ignore[arg-type]
        target=body.target,        # type: ignore[arg-type]
        denoise=body.denoise,
        params=body.params,
        features=body.features,
        dimred_method=body.dimred_method,
        dimred_params=body.dimred_params,
        slippage_bps=body.slippage_bps,
        trade_threshold=body.trade_threshold,
    )
    if isinstance(res, dict) and res.get("error"):
        raise HTTPException(status_code=400, detail=str(res["error"]))
    return res


@app.get("/")
def root() -> Dict[str, Any]:
    return {"service": "mtdata-webui", "status": "ok"}

# Optionally serve the built SPA if available at webui/dist
try:
    app.mount("/app", StaticFiles(directory="webui/dist", html=True), name="webui")
except Exception:
    pass
