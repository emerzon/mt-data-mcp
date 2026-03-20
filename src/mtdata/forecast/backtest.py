from typing import Any, Dict, List, Optional, Tuple, Literal
import numpy as np
import math

from ..shared.constants import TIMEFRAME_MAP
from ..shared.schema import TimeframeLiteral, DenoiseSpec
from ..shared.validators import invalid_timeframe_error
from ..utils.utils import _format_time_minimal
from .volatility import forecast_volatility
from .forecast import forecast
from ..utils.denoise import normalize_denoise_spec as _normalize_denoise_spec
from .common import (
    bars_per_year as _bars_per_year,
    fetch_history as _fetch_history,
    log_returns_from_prices as _log_returns_from_prices,
    quantity_to_target as _quantity_to_target,
)


def _get_forecast_methods_data_safe() -> Dict[str, Any]:
    """Safely fetch forecast methods metadata.

    Falls back to a minimal set of classical methods if discovery fails.
    Only 'method' and 'available' keys are required by this module.
    """
    try:
        from .forecast_registry import get_forecast_methods_data as _get

        data = _get()
        if isinstance(data, dict) and "methods" in data:
            return data
    except Exception:
        pass
    return {
        "methods": [
            {"method": "naive", "available": True},
            {"method": "drift", "available": True},
            {"method": "seasonal_naive", "available": True},
            {"method": "theta", "available": True},
            {"method": "fourier_ols", "available": True},
        ]
    }


_MIN_ANNUALIZATION_TRADES = 30
_MIN_ANNUALIZATION_YEARS = 0.25


def _compute_performance_metrics(
    returns: List[float],
    timeframe: str,
    horizon: int,
    slippage_bps: float,
) -> Dict[str, Any]:
    """Compute portfolio-level performance statistics from per-trade returns."""

    metrics: Dict[str, Any] = {}
    if not returns:
        return metrics

    arr = np.asarray([float(r) for r in returns if r is not None], dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return metrics

    bars_per_year = _bars_per_year(timeframe)
    trades_per_year = (
        float(bars_per_year / max(1, int(horizon)))
        if math.isfinite(bars_per_year)
        else float("nan")
    )

    avg_return = float(np.mean(arr))
    win_rate = float(np.mean(arr > 0.0)) if arr.size > 0 else float("nan")
    std_ret = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    enough_trades = int(arr.size) >= int(_MIN_ANNUALIZATION_TRADES)
    sharpe = float("nan")
    if (
        enough_trades
        and std_ret > 1e-12
        and math.isfinite(trades_per_year)
        and trades_per_year > 0
    ):
        sharpe = float((avg_return / std_ret) * math.sqrt(trades_per_year))

    equity = np.cumprod(1.0 + arr)
    peak = np.maximum.accumulate(equity)
    drawdowns = equity / np.where(peak == 0.0, 1.0, peak) - 1.0
    max_drawdown = float(abs(np.min(drawdowns))) if drawdowns.size > 0 else float("nan")

    cumulative_return = float(equity[-1] - 1.0) if equity.size > 0 else float("nan")
    years = (
        float(arr.size / trades_per_year)
        if math.isfinite(trades_per_year) and trades_per_year > 0
        else float("nan")
    )
    annual_return = float("nan")
    if (
        enough_trades
        and math.isfinite(years)
        and years >= _MIN_ANNUALIZATION_YEARS
        and equity.size > 0
        and equity[-1] > 0
    ):
        try:
            annual_return = float(equity[-1] ** (1.0 / years) - 1.0)
        except Exception:
            annual_return = float("nan")
    calmar = float("nan")
    if (
        max_drawdown > 0
        and math.isfinite(max_drawdown)
        and math.isfinite(annual_return)
    ):
        calmar = float(annual_return / max_drawdown)

    def _finite_or_none(value: float) -> Optional[float]:
        try:
            value_f = float(value)
        except Exception:
            return None
        if not math.isfinite(value_f):
            return None
        return value_f

    metrics.update(
        {
            "avg_return_per_trade": avg_return,
            "win_rate": win_rate,
            "sharpe_ratio": _finite_or_none(sharpe),
            "max_drawdown": max_drawdown,
            "calmar_ratio": _finite_or_none(calmar),
            "cumulative_return": cumulative_return,
            "annual_return": _finite_or_none(annual_return),
            "num_trades": float(arr.size),
            "trades_per_year": trades_per_year,
            "slippage_bps": float(slippage_bps),
        }
    )
    if not enough_trades:
        metrics["sample_warning"] = (
            f"Only {int(arr.size)} trades. Annualized risk metrics "
            f"(Sharpe/Calmar/annual_return) are suppressed below {_MIN_ANNUALIZATION_TRADES} trades."
        )
        metrics["min_trades_for_annualization"] = float(_MIN_ANNUALIZATION_TRADES)
    return metrics


def forecast_backtest(
    symbol: str,
    timeframe: TimeframeLiteral = "H1",
    horizon: int = 12,
    steps: int = 5,
    spacing: int = 20,
    methods: Optional[List[str]] = None,
    params_per_method: Optional[Dict[str, Any]] = None,
    quantity: Literal["price", "return", "volatility"] = "price",  # type: ignore
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
    detail: Literal["compact", "full"] = "compact",  # type: ignore
) -> Dict[str, Any]:
    """Rolling-origin backtest over historical anchors using the forecast tool.

    Parameters: symbol, timeframe, horizon, steps, spacing, methods?, params_per_method?, quantity, denoise?
    - Picks `steps` anchor points spaced `spacing` bars apart, each with `horizon` future bars for validation.
    - For each method, runs our `forecast` as-of that anchor and reports MAE/RMSE/directional accuracy.
    """
    try:
        __stage = "start"
        detail_mode = str(detail or "compact").strip().lower()
        if detail_mode not in ("compact", "full"):
            detail_mode = "compact"
        include_paths = detail_mode == "full"
        if timeframe not in TIMEFRAME_MAP:
            return {"error": invalid_timeframe_error(timeframe, TIMEFRAME_MAP)}

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
            tvals = df["time"].astype(float).to_numpy()
            tstr = [_format_time_minimal(ts) for ts in tvals]
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
            if quantity == "volatility":
                methods = ["ewma", "parkinson"]
            else:
                methods_info = _get_forecast_methods_data_safe()
                avail = [
                    m["method"]
                    for m in methods_info.get("methods", [])
                    if m.get("available")
                ]
                preferred = [
                    "naive",
                    "drift",
                    "seasonal_naive",
                    "theta",
                    "fourier_ols",
                    "sf_autoarima",
                    "sf_theta",
                ]
                methods = [m for m in preferred if m in avail]
                if not methods:
                    methods = [m for m in ("naive", "drift", "theta") if m in avail]
        params_map = dict(params_per_method or {})
        target_mode = _quantity_to_target(quantity)

        # Build ground-truth windows for each anchor
        closes = df["close"].astype(float).to_numpy()
        times = df["time"].astype(float).to_numpy()
        actual_windows: Dict[int, Tuple[List[float], List[float]]] = {}
        for idx in anchor_indices:
            if idx + int(horizon) >= len(closes):
                continue
            if target_mode == "return" and quantity != "volatility":
                prev = np.maximum(closes[idx : idx + int(horizon)], 1e-12)
                nxt = np.maximum(closes[idx + 1 : idx + 1 + int(horizon)], 1e-12)
                with np.errstate(divide="ignore", invalid="ignore"):
                    actual = np.log(nxt / prev).tolist()
            else:
                actual = closes[idx + 1 : idx + 1 + int(horizon)].tolist()
            ts = times[idx + 1 : idx + 1 + int(horizon)].tolist()
            if len(actual) != int(horizon) or len(ts) != int(horizon):
                continue
            actual_windows[idx] = (actual, ts)
        if not actual_windows:
            return {"error": "No valid validation windows found"}

        # Normalize denoise spec once for the whole run (uniform across methods)
        try:
            _dn_used = (
                _normalize_denoise_spec(denoise, default_when="pre_ti")
                if denoise is not None
                else None
            )
        except Exception:
            _dn_used = None

        # Run forecasts per method and compute metrics
        results: Dict[str, Any] = {}
        for method in methods:
            per_anchor = []
            for idx in anchor_indices:
                if idx not in actual_windows:
                    continue
                anchor_time = _format_time_minimal(times[idx])
                truth, ts = actual_windows[idx]
                try:
                    if quantity == "volatility":
                        # Volatility forecast: allow proxy in params map (params_map[method].get('proxy'))
                        pm_raw = params_map.get(method) or {}
                        pm = dict(pm_raw) if isinstance(pm_raw, dict) else {}
                        proxy = pm.pop("proxy", None) if isinstance(pm, dict) else None
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
                            quantity=quantity,  # type: ignore[arg-type]
                            denoise=_dn_used,
                            features=features,
                            dimred_method=dimred_method,
                            dimred_params=dimred_params,
                        )
                except Exception as ex:
                    per_anchor.append(
                        {"anchor": anchor_time, "success": False, "error": str(ex)}
                    )
                    continue
                if "error" in r:
                    per_anchor.append(
                        {"anchor": anchor_time, "success": False, "error": r["error"]}
                    )
                    continue
                if quantity == "volatility":
                    # Compute realized horizon sigma from ground truth prices
                    act = np.array(truth, dtype=float)
                    r_act = (
                        _log_returns_from_prices(act)
                        if act.size >= 2
                        else np.array([], dtype=float)
                    )
                    realized_sigma = (
                        float(np.sqrt(np.mean(np.square(np.clip(r_act, -1e6, 1e6)))))
                        if r_act.size > 0
                        else float("nan")
                    )
                    pred_sigma = float(r.get("horizon_sigma_return", float("nan")))
                    mae = (
                        float(abs(pred_sigma - realized_sigma))
                        if np.isfinite(pred_sigma) and np.isfinite(realized_sigma)
                        else float("nan")
                    )
                    rmse = mae
                    per_anchor.append(
                        {
                            "anchor": anchor_time,
                            "success": np.isfinite(pred_sigma)
                            and np.isfinite(realized_sigma),
                            "mae": mae,
                            "rmse": rmse,
                            "forecast_sigma": pred_sigma,
                            "realized_sigma": realized_sigma,
                        }
                    )
                else:
                    if target_mode == "return":
                        fc = r.get("forecast_return") or r.get("forecast_price")
                    else:
                        fc = r.get("forecast_price")
                    if not fc:
                        per_anchor.append(
                            {
                                "anchor": anchor_time,
                                "success": False,
                                "error": "Empty forecast",
                            }
                        )
                        continue
                    fcv = np.array(fc, dtype=float)
                    act = np.array(truth, dtype=float)
                    m = min(len(fcv), len(act))
                    if m <= 0:
                        per_anchor.append(
                            {
                                "anchor": anchor_time,
                                "success": False,
                                "error": "No overlap",
                            }
                        )
                        continue
                    mae = float(np.mean(np.abs(fcv[:m] - act[:m])))
                    rmse = float(np.sqrt(np.mean((fcv[:m] - act[:m]) ** 2)))
                    if m > 1:
                        da = float(
                            np.mean(
                                np.sign(np.diff(fcv[:m])) == np.sign(np.diff(act[:m]))
                            )
                        )
                    else:
                        da = float("nan")
                    entry_price = (
                        float(closes[idx]) if idx < len(closes) else float("nan")
                    )
                    if target_mode == "return":
                        expected_move = float(np.nansum(fcv[:m]))
                    else:
                        expected_move = (
                            float((float(fcv[m - 1]) - entry_price))
                            if math.isfinite(entry_price)
                            else float("nan")
                        )
                    expected_return = float("nan")
                    if target_mode == "return":
                        try:
                            expected_return = float(math.exp(expected_move) - 1.0)
                        except Exception:
                            expected_return = float("nan")
                    elif math.isfinite(entry_price) and entry_price != 0.0:
                        expected_return = expected_move / entry_price
                    direction = 0
                    threshold = float(trade_threshold or 0.0)
                    if math.isfinite(expected_return):
                        if expected_return > threshold:
                            direction = 1
                        elif expected_return < -threshold:
                            direction = -1
                    position = "flat"
                    if direction > 0:
                        position = "long"
                    elif direction < 0:
                        position = "short"
                    gross_return = float("nan")
                    net_return = float("nan")
                    exit_price = float("nan")
                    exit_step = m - 1
                    if direction != 0:
                        if target_mode == "return":
                            try:
                                realized_path = np.array(act[:m], dtype=float)
                                if not np.all(np.isfinite(realized_path)):
                                    realized_path = np.nan_to_num(
                                        realized_path, nan=0.0, posinf=0.0, neginf=0.0
                                    )
                                cum_log = np.cumsum(realized_path)
                                forecast_target_log = float(np.nansum(fcv[:m]))
                                if (
                                    math.isfinite(forecast_target_log)
                                    and abs(forecast_target_log) > 0
                                ):
                                    if direction > 0:
                                        hit_idx = np.where(
                                            cum_log >= forecast_target_log
                                        )[0]
                                    else:
                                        hit_idx = np.where(
                                            cum_log <= forecast_target_log
                                        )[0]
                                    if hit_idx.size > 0:
                                        exit_step = int(hit_idx[0])
                                realized_log = (
                                    float(cum_log[exit_step]) if cum_log.size else 0.0
                                )
                                gross_return = direction * float(
                                    math.exp(realized_log) - 1.0
                                )
                                exit_idx = idx + exit_step + 1
                                exit_price = (
                                    float(closes[exit_idx])
                                    if exit_idx < len(closes)
                                    else float("nan")
                                )
                            except Exception:
                                gross_return = float("nan")
                            slip = float(abs(slippage_bps) or 0.0) / 10000.0
                            net_return = gross_return - 2.0 * slip
                            if net_return <= -0.999:
                                net_return = -0.999
                        elif math.isfinite(entry_price) and entry_price != 0.0:
                            try:
                                forecast_target_price = float(fcv[m - 1])
                                realized_prices = np.array(act[:m], dtype=float)
                                if math.isfinite(forecast_target_price):
                                    if direction > 0:
                                        hit_idx = np.where(
                                            realized_prices >= forecast_target_price
                                        )[0]
                                    else:
                                        hit_idx = np.where(
                                            realized_prices <= forecast_target_price
                                        )[0]
                                    if hit_idx.size > 0:
                                        exit_step = int(hit_idx[0])
                                exit_price = (
                                    float(realized_prices[exit_step])
                                    if realized_prices.size
                                    else float("nan")
                                )
                            except Exception:
                                exit_price = float("nan")
                            if math.isfinite(exit_price):
                                gross_return = direction * (
                                    (exit_price - entry_price) / entry_price
                                )
                                slip = float(abs(slippage_bps) or 0.0) / 10000.0
                                net_return = gross_return - 2.0 * slip
                                if net_return <= -0.999:
                                    net_return = -0.999
                    elif direction == 0:
                        gross_return = 0.0
                        net_return = 0.0
                    detail_row = {
                        "anchor": anchor_time,
                        "success": True,
                        "mae": mae,
                        "rmse": rmse,
                        "directional_accuracy": da,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "exit_step": int(exit_step) + 1 if m > 0 else 0,
                        "expected_return": expected_return,
                        "position": position,
                        "trade_return_gross": gross_return,
                        "trade_return": net_return,
                    }
                    if include_paths:
                        detail_row["forecast"] = [float(v) for v in fcv[:m].tolist()]
                        detail_row["actual"] = [float(v) for v in act[:m].tolist()]
                    else:
                        detail_row["horizon_used"] = int(m)
                        detail_row["forecast_end"] = (
                            float(fcv[m - 1]) if m > 0 else None
                        )
                        detail_row["actual_end"] = float(act[m - 1]) if m > 0 else None
                    per_anchor.append(detail_row)
            # Aggregate
            ok = [x for x in per_anchor if x.get("success")]
            if ok:
                agg = {
                    "success": True,
                    "avg_mae": float(np.mean([x["mae"] for x in ok])),
                    "avg_rmse": float(np.mean([x["rmse"] for x in ok])),
                    "successful_tests": len(ok),
                    "num_tests": len(per_anchor),
                    "details": per_anchor,
                }
                if quantity != "volatility":
                    da_vals = [x.get("directional_accuracy") for x in ok]
                    da_vals = [v for v in da_vals if v is not None and np.isfinite(v)]
                    if da_vals:
                        agg["avg_directional_accuracy"] = float(np.mean(da_vals))
                    trade_returns = [
                        x.get("trade_return")
                        for x in ok
                        if x.get("trade_return") is not None
                    ]
                    trade_returns = [
                        float(v)
                        for v in trade_returns
                        if v is not None and np.isfinite(v)
                    ]
                    metrics = (
                        _compute_performance_metrics(
                            trade_returns, timeframe, int(horizon), float(slippage_bps)
                        )
                        if trade_returns
                        else {}
                    )
                    if metrics:
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
            "slippage_bps": float(slippage_bps),
            "trade_threshold": float(trade_threshold or 0.0),
            "detail": detail_mode,
            "results": results,
        }
    except Exception as e:
        return {"error": f"Error in forecast_backtest: {str(e)}"}
