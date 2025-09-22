from typing import Any, Dict, List, Optional, Tuple, Literal
import numpy as np
import pandas as pd
from datetime import datetime as _dt
import json
import math
import MetaTrader5 as mt5

from ..core.constants import TIMEFRAME_MAP, TIMEFRAME_SECONDS
from ..core.schema import TimeframeLiteral, DenoiseSpec
from ..utils.mt5 import _mt5_epoch_to_utc, _mt5_copy_rates_from, _ensure_symbol_ready
from ..utils.utils import _format_time_minimal as _format_time_minimal_util
from .volatility import forecast_volatility
from .forecast import forecast
from ..utils.denoise import normalize_denoise_spec as _normalize_denoise_spec
from .common import fetch_history as _fetch_history




def _get_forecast_methods_data_safe() -> Dict[str, Any]:
    """Safely fetch forecast methods metadata.

    Falls back to a minimal set of classical methods if discovery fails.
    Only 'method' and 'available' keys are required by this module.
    """
    try:
        from .forecast import get_forecast_methods_data as _get
        data = _get()
        if isinstance(data, dict) and 'methods' in data:
            return data
    except Exception:
        pass
    return {
        'methods': [
            {'method': 'naive', 'available': True},
            {'method': 'drift', 'available': True},
            {'method': 'seasonal_naive', 'available': True},
            {'method': 'theta', 'available': True},
            {'method': 'fourier_ols', 'available': True},
        ]
    }


def _bars_per_year(timeframe: str) -> float:
    """Approximate number of bars per year for a timeframe."""
    try:
        secs = TIMEFRAME_SECONDS.get(str(timeframe))
        if not secs or secs <= 0:
            return float('nan')
        return float((365.0 * 24.0 * 3600.0) / float(secs))
    except Exception:
        return float('nan')


def _compute_performance_metrics(
    returns: List[float],
    timeframe: str,
    horizon: int,
    slippage_bps: float,
) -> Dict[str, float]:
    """Compute portfolio-level performance statistics from per-trade returns."""

    metrics: Dict[str, float] = {}
    if not returns:
        return metrics

    arr = np.asarray([float(r) for r in returns if r is not None], dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return metrics

    bars_per_year = _bars_per_year(timeframe)
    trades_per_year = float(bars_per_year / max(1, int(horizon))) if math.isfinite(bars_per_year) else float('nan')

    avg_return = float(np.mean(arr))
    win_rate = float(np.mean(arr > 0.0)) if arr.size > 0 else float('nan')
    std_ret = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    sharpe = float('nan')
    if std_ret > 1e-12 and math.isfinite(trades_per_year) and trades_per_year > 0:
        sharpe = float((avg_return / std_ret) * math.sqrt(trades_per_year))

    equity = np.cumprod(1.0 + arr)
    peak = np.maximum.accumulate(equity)
    drawdowns = equity / np.where(peak == 0.0, 1.0, peak) - 1.0
    max_drawdown = float(abs(np.min(drawdowns))) if drawdowns.size > 0 else float('nan')

    cumulative_return = float(equity[-1] - 1.0) if equity.size > 0 else float('nan')
    years = float(arr.size / trades_per_year) if math.isfinite(trades_per_year) and trades_per_year > 0 else float('nan')
    annual_return = float('nan')
    if math.isfinite(years) and years > 0 and equity.size > 0 and equity[-1] > 0:
        try:
            annual_return = float(equity[-1] ** (1.0 / years) - 1.0)
        except Exception:
            annual_return = float('nan')
    calmar = float('nan')
    if max_drawdown > 0 and math.isfinite(max_drawdown) and math.isfinite(annual_return):
        calmar = float(annual_return / max_drawdown)

    metrics.update({
        "avg_return_per_trade": avg_return,
        "win_rate": win_rate,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
        "calmar_ratio": calmar,
        "cumulative_return": cumulative_return,
        "annual_return": annual_return,
        "num_trades": float(arr.size),
        "trades_per_year": trades_per_year,
        "slippage_bps": float(slippage_bps),
    })
    return metrics


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
    anchors: Optional[List[str]] = None,
    # Unified per-run tuning applied to all methods (unless overridden in params_per_method)
    params: Optional[Dict[str, Any]] = None,
    # Feature engineering for exogenous/multivariate models
    features: Optional[Dict[str, Any]] = None,
    dimred_method: Optional[str] = None,
    dimred_params: Optional[Dict[str, Any]] = None,
    slippage_bps: float = 0.0,
    trade_threshold: float = 0.0,
) -> Dict[str, Any]:
    """Rolling-origin backtest over historical anchors using the forecast tool.

    Parameters: symbol, timeframe, horizon, steps, spacing, methods?, params_per_method?, target, denoise?
    - Picks `steps` anchor points spaced `spacing` bars apart, each with `horizon` future bars for validation.
    - For each method, runs our `forecast` as-of that anchor and reports MAE/RMSE/directional accuracy.
    """
    try:
        __stage = 'start'
        if timeframe not in TIMEFRAME_MAP:
            return {"error": f"Invalid timeframe: {timeframe}. Valid options: {list(TIMEFRAME_MAP.keys())}"}
        mt5_tf = TIMEFRAME_MAP[timeframe]

        # Fetch sufficient history via shared helper; ensure enough bars for anchors
        if anchors and isinstance(anchors, (list, tuple)) and len(anchors) > 0:
            need = int(len(anchors)) * int(horizon) + 600
        else:
            need = int(steps) * int(spacing) + int(horizon) + 400
        try:
            df = _fetch_history(symbol, timeframe, int(need), as_of=None)
        except Exception as ex:
            return {"error": str(ex)}
        if len(df) < (int(horizon) + 50):
            return {"error": "Not enough closed bars for backtest"}

        # Determine anchor indices (explicit anchors or rolling from end)
        total = len(df)
        anchor_indices: List[int] = []
        if anchors and isinstance(anchors, (list, tuple)) and len(anchors) > 0:
            tvals = df['time'].astype(float).to_numpy()
            tstr = [_format_time_minimal_util(ts) for ts in tvals]
            idx_by_time = {s: i for i, s in enumerate(tstr)}
            for s in anchors:
                i = idx_by_time.get(str(s).strip())
                if i is not None and (i + int(horizon)) < total:
                    anchor_indices.append(int(i))
        else:
            pos = total - int(horizon) - 1
            for _ in range(int(steps)):
                if pos <= 1:
                    break
                anchor_indices.append(int(pos))
                pos -= int(spacing)
            anchor_indices = list(reversed(anchor_indices))
        if not anchor_indices:
            return {"error": "Failed to determine backtest anchors"}

        # Normalize methods input (allow comma or whitespace separated string)
        if isinstance(methods, str):
            txt = methods.strip()
            if "," in txt:
                methods = [s.strip() for s in txt.split(",") if s.strip()]
            else:
                methods = [s for s in txt.split() if s]

        # Default methods based on quantity
        if not methods:
            if quantity == 'volatility':
                methods = ['ewma', 'parkinson']
            else:
                methods_info = _get_forecast_methods_data_safe()
                avail = [m['method'] for m in methods_info.get('methods', []) if m.get('available')]
                preferred = ['naive', 'drift', 'seasonal_naive', 'theta', 'fourier_ols', 'sf_autoarima', 'sf_theta']
                methods = [m for m in preferred if m in avail]
                if not methods:
                    methods = [m for m in ('naive', 'drift', 'theta') if m in avail]
        params_map = dict(params_per_method or {})

        # Build ground-truth windows for each anchor
        closes = df['close'].astype(float).to_numpy()
        times = df['time'].astype(float).to_numpy()
        actual_windows: Dict[int, Tuple[List[float], List[float]]] = {}
        for idx in anchor_indices:
            if idx + int(horizon) >= len(closes):
                continue
            actual = closes[idx + 1: idx + 1 + int(horizon)].tolist()
            ts = times[idx + 1: idx + 1 + int(horizon)].tolist()
            actual_windows[idx] = (actual, ts)
        if not actual_windows:
            return {"error": "No valid validation windows found"}

        # Normalize denoise spec once for the whole run (uniform across methods)
        try:
            _dn_used = _normalize_denoise_spec(denoise, default_when='pre_ti') if denoise is not None else None
        except Exception:
            _dn_used = None

        # Run forecasts per method and compute metrics
        results: Dict[str, Any] = {}
        for method in methods:
            per_anchor = []
            for idx in anchor_indices:
                if idx not in actual_windows:
                    continue
                anchor_time = _format_time_minimal_util(times[idx])
                truth, ts = actual_windows[idx]
                try:
                    if quantity == 'volatility':
                        # Volatility forecast: allow proxy in params map (params_map[method].get('proxy'))
                        pm = params_map.get(method) or {}
                        proxy = pm.pop('proxy', None) if isinstance(pm, dict) else None
                        r = forecast_volatility(  # type: ignore
                            symbol=symbol,
                            timeframe=timeframe,
                            method=method,  # type: ignore
                            horizon=int(horizon),
                            as_of=anchor_time,
                            params=pm if isinstance(pm, dict) else None,
                            proxy=proxy,  # type: ignore
                            denoise=_dn_used,
                        )
                    else:
                        # Choose per-method params falling back to global params
                        pm = params_map.get(method)
                        if pm is None:
                            pm = params
                        r = forecast(
                            symbol=symbol,
                            timeframe=timeframe,
                            method=method,  # type: ignore[arg-type]
                            horizon=int(horizon),
                            as_of=anchor_time,
                            params=pm,
                            target=target,
                            denoise=_dn_used,
                            features=features,
                            dimred_method=dimred_method,
                            dimred_params=dimred_params,
                        )
                except Exception as ex:
                    per_anchor.append({"anchor": anchor_time, "success": False, "error": str(ex)})
                    continue
                if 'error' in r:
                    per_anchor.append({"anchor": anchor_time, "success": False, "error": r['error']})
                    continue
                if quantity == 'volatility':
                    # Compute realized horizon sigma from ground truth prices
                    act = np.array(truth, dtype=float)
                    r_act = np.diff(np.log(np.maximum(act, 1e-12))) if act.size >= 2 else np.array([], dtype=float)
                    realized_sigma = float(np.sqrt(np.sum(np.clip(r_act, -1e6, 1e6)**2))) if r_act.size > 0 else float('nan')
                    pred_sigma = float(r.get('horizon_sigma_return', float('nan')))
                    mae = float(abs(pred_sigma - realized_sigma)) if np.isfinite(pred_sigma) and np.isfinite(realized_sigma) else float('nan')
                    rmse = mae
                    per_anchor.append({
                        "anchor": anchor_time,
                        "success": np.isfinite(pred_sigma) and np.isfinite(realized_sigma),
                        "mae": mae,
                        "rmse": rmse,
                        "forecast_sigma": pred_sigma,
                        "realized_sigma": realized_sigma,
                    })
                else:
                    fc = r.get('forecast_price') if target == 'price' else r.get('forecast_return')
                    if not fc:
                        per_anchor.append({"anchor": anchor_time, "success": False, "error": "Empty forecast"})
                        continue
                    fcv = np.array(fc, dtype=float)
                    act = np.array(truth, dtype=float)
                    m = min(len(fcv), len(act))
                    if m <= 0:
                        per_anchor.append({"anchor": anchor_time, "success": False, "error": "No overlap"})
                        continue
                    mae = float(np.mean(np.abs(fcv[:m] - act[:m])))
                    rmse = float(np.sqrt(np.mean((fcv[:m] - act[:m])**2)))
                    if m > 1:
                        da = float(np.mean(np.sign(np.diff(fcv[:m])) == np.sign(np.diff(act[:m]))))
                    else:
                        da = float('nan')
                    entry_price = float(closes[idx]) if idx < len(closes) else float('nan')
                    exit_price = float(act[m-1]) if m > 0 else float('nan')
                    if target == 'return':
                        expected_move = float(np.nansum(fcv[:m]))
                    else:
                        expected_move = float((float(fcv[m-1]) - entry_price)) if math.isfinite(entry_price) else float('nan')
                    expected_return = float('nan')
                    if target == 'return':
                        expected_return = expected_move
                    elif math.isfinite(entry_price) and entry_price != 0.0:
                        expected_return = expected_move / entry_price
                    direction = 0
                    threshold = float(trade_threshold or 0.0)
                    if math.isfinite(expected_return):
                        if expected_return > threshold:
                            direction = 1
                        elif expected_return < -threshold:
                            direction = -1
                    position = 'flat'
                    if direction > 0:
                        position = 'long'
                    elif direction < 0:
                        position = 'short'
                    gross_return = float('nan')
                    net_return = float('nan')
                    if direction != 0 and math.isfinite(entry_price) and entry_price != 0.0 and math.isfinite(exit_price):
                        gross_return = direction * ((exit_price - entry_price) / entry_price)
                        slip = float(abs(slippage_bps) or 0.0) / 10000.0
                        net_return = gross_return - float(direction != 0) * 2.0 * slip
                        if net_return <= -0.999:
                            net_return = -0.999
                    elif direction == 0:
                        gross_return = 0.0
                        net_return = 0.0
                    per_anchor.append({
                        "anchor": anchor_time,
                        "success": True,
                        "mae": mae,
                        "rmse": rmse,
                        "directional_accuracy": da,
                        "forecast": [float(v) for v in fcv[:m].tolist()],
                        "actual": [float(v) for v in act[:m].tolist()],
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "expected_return": expected_return,
                        "position": position,
                        "trade_return_gross": gross_return,
                        "trade_return": net_return,
                    })
            # Aggregate
            ok = [x for x in per_anchor if x.get('success')]
            if ok:
                agg = {
                    "success": True,
                    "avg_mae": float(np.mean([x['mae'] for x in ok])),
                    "avg_rmse": float(np.mean([x['rmse'] for x in ok])),
                    "successful_tests": len(ok),
                    "num_tests": len(per_anchor),
                    "details": per_anchor,
                }
                if quantity != 'volatility':
                    da_vals = [x.get('directional_accuracy') for x in ok]
                    da_vals = [v for v in da_vals if v is not None and np.isfinite(v)]
                    if da_vals:
                        agg["avg_directional_accuracy"] = float(np.mean(da_vals))
                    trade_returns = [x.get('trade_return') for x in ok if x.get('trade_return') is not None]
                    trade_returns = [float(v) for v in trade_returns if v is not None and np.isfinite(v)]
                    metrics = _compute_performance_metrics(trade_returns, timeframe, int(horizon), float(slippage_bps)) if trade_returns else {}
                    if metrics:
                        agg["avg_trade_return"] = float(metrics.get("avg_return_per_trade", float('nan')))
                        agg["win_rate"] = float(metrics.get("win_rate", float('nan')))
                        agg["consistency"] = float(metrics.get("win_rate", float('nan')))
                        agg["metrics"] = metrics
                        agg["slippage_bps"] = float(slippage_bps)
                if _dn_used:
                    agg["denoise_used"] = _dn_used
                results[method] = agg
            else:
                results[method] = {
                    "success": False,
                    "successful_tests": 0,
                    "num_tests": len(per_anchor),
                    "details": per_anchor,
                    "slippage_bps": float(slippage_bps),
                }

        return {
            "success": True,
            "symbol": symbol,
            "timeframe": timeframe,
            "horizon": int(horizon),
            "steps": int(steps),
            "spacing": int(spacing),
            "methods": methods,
            "denoise_used": _dn_used,
            "slippage_bps": float(slippage_bps),
            "trade_threshold": float(trade_threshold or 0.0),
            "results": results,
        }
    except Exception as e:
        return {"error": f"Error in forecast_backtest: {str(e)}"}
