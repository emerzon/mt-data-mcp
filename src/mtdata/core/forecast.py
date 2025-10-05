
from typing import Any, Dict, Optional, List, Literal, Tuple, Set

from .schema import TimeframeLiteral, DenoiseSpec, ForecastMethodLiteral
from .server import mcp, _auto_connect_wrapper
from ..forecast.forecast import forecast as _forecast_impl
from ..forecast.backtest import forecast_backtest as _forecast_backtest_impl
from ..forecast.volatility import forecast_volatility as _forecast_volatility_impl
from ..forecast.forecast import get_forecast_methods_data as _get_forecast_methods_data
from ..forecast.tune import genetic_search_forecast_params as _genetic_search_impl
from ..forecast.common import fetch_history as _fetch_history, parse_kv_or_json as _parse_kv_or_json
from ..forecast.monte_carlo import simulate_gbm_mc as _simulate_gbm_mc, simulate_hmm_mc as _simulate_hmm_mc, summarize_paths as _summarize_paths
from ..forecast.monte_carlo import gbm_single_barrier_upcross_prob as _gbm_upcross_prob
from .constants import TIMEFRAME_SECONDS
BARRIER_GRID_PRESETS = {
    'scalp': {
        'tp_min': 0.08, 'tp_max': 0.60, 'tp_steps': 7,
        'sl_min': 0.20, 'sl_max': 1.20, 'sl_steps': 7,
    },
    'intraday': {
        'tp_min': 0.25, 'tp_max': 1.50, 'tp_steps': 7,
        'sl_min': 0.25, 'sl_max': 2.50, 'sl_steps': 9,
    },
    'swing': {
        'tp_min': 0.60, 'tp_max': 3.50, 'tp_steps': 7,
        'sl_min': 0.50, 'sl_max': 4.50, 'sl_steps': 8,
    },
    'position': {
        'tp_min': 1.00, 'tp_max': 8.00, 'tp_steps': 8,
        'sl_min': 0.75, 'sl_max': 6.00, 'sl_steps': 8,
    },
}

import MetaTrader5 as mt5
import numpy as _np

@mcp.tool()
@_auto_connect_wrapper
def forecast_generate(
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
    )


@mcp.tool()
@_auto_connect_wrapper
def forecast_backtest_run(
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
    slippage_bps: float = 0.0,
    trade_threshold: float = 0.0,
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
        slippage_bps=slippage_bps,
        trade_threshold=trade_threshold,
    )


@mcp.tool()
@_auto_connect_wrapper
def forecast_volatility_estimate(
    symbol: str,
    timeframe: TimeframeLiteral = "H1",
    horizon: int = 1,
    method: Literal['ewma','parkinson','gk','rs','yang_zhang','rolling_std','realized_kernel','har_rv','garch','egarch','gjr_garch','garch_t','egarch_t','gjr_garch_t','figarch','arima','sarima','ets','theta'] = 'ewma',  # type: ignore
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


@mcp.tool()
@_auto_connect_wrapper
def forecast_list_methods() -> Dict[str, Any]:
    """List forecast methods, availability, and parameter docs."""
    try:
        return _get_forecast_methods_data()
    except Exception as e:
        return {"error": f"Error listing forecast methods: {e}"}


@mcp.tool()
@_auto_connect_wrapper
def forecast_conformal_intervals(
    symbol: str,
    timeframe: TimeframeLiteral = "H1",
    method: ForecastMethodLiteral = "theta",
    horizon: int = 12,
    steps: int = 25,
    spacing: int = 10,
    alpha: float = 0.1,
    denoise: Optional[DenoiseSpec] = None,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Conformalized forecast intervals via rolling-origin calibration.

    - Calibrates per-step absolute residual quantiles using `steps` historical anchors (spaced by `spacing`).
    - Returns point forecast (from `method`) and conformal bands per step.
    """
    try:
        # 1) Rolling backtest to collect residuals
        bt = _forecast_backtest_impl(
            symbol=symbol,
            timeframe=timeframe,
            horizon=int(horizon),
            steps=int(steps),
            spacing=int(spacing),
            methods=[str(method)],
            denoise=denoise,
            params={str(method): dict(params or {})},
        )
        if 'error' in bt:
            return bt
        res = bt.get('results', {}).get(str(method))
        if not res or not res.get('details'):
            return {"error": "Conformal calibration failed: no backtest details"}
        # Build per-step residuals |y_hat_i - y_i|
        fh = int(horizon)
        errs = [[] for _ in range(fh)]
        for d in res['details']:
            fc = d.get('forecast'); act = d.get('actual')
            if not fc or not act:
                continue
            m = min(len(fc), len(act), fh)
            for i in range(m):
                try:
                    errs[i].append(abs(float(fc[i]) - float(act[i])))
                except Exception:
                    continue
        # Per-step quantiles
        import numpy as _np
        q = 1.0 - float(alpha)
        qerrs = [float(_np.quantile(_np.array(e, dtype=float), q)) if e else float('nan') for e in errs]

        # 2) Forecast now (latest)
        fc_now = _forecast_impl(
            symbol=symbol,
            timeframe=timeframe,
            method=method,  # type: ignore
            horizon=int(horizon),
            params=params,
            denoise=denoise,
        )
        if 'error' in fc_now:
            return fc_now
        yhat = fc_now.get('forecast_price') or []
        if not yhat:
            return {"error": "Empty point forecast for conformal intervals"}
        yhat_arr = _np.array(yhat, dtype=float)
        fh_eff = min(fh, yhat_arr.size)
        lo = _np.empty(fh_eff, dtype=float); hi = _np.empty(fh_eff, dtype=float)
        for i in range(fh_eff):
            e = qerrs[i] if i < len(qerrs) and _np.isfinite(qerrs[i]) else 0.0
            lo[i] = yhat_arr[i] - e
            hi[i] = yhat_arr[i] + e
        out = dict(fc_now)
        out['conformal'] = {
            'alpha': float(alpha),
            'calibration_steps': int(steps),
            'calibration_spacing': int(spacing),
            'per_step_q': [float(v) for v in qerrs],
        }
        out['lower_price'] = [float(v) for v in lo.tolist()]
        out['upper_price'] = [float(v) for v in hi.tolist()]
        out['ci_alpha'] = float(alpha)
        return out
    except Exception as e:
        return {"error": f"Error computing conformal forecast: {str(e)}"}


@mcp.tool()
@_auto_connect_wrapper
def forecast_tune_genetic(
    symbol: str,
    timeframe: TimeframeLiteral = "H1",
    method: Optional[str] = "theta",
    methods: Optional[List[str]] = None,
    horizon: int = 12,
    steps: int = 5,
    spacing: int = 20,
    search_space: Optional[Dict[str, Any]] = None,
    metric: str = "avg_rmse",
    mode: str = "min",
    population: int = 12,
    generations: int = 10,
    crossover_rate: float = 0.6,
    mutation_rate: float = 0.3,
    seed: int = 42,
    trade_threshold: float = 0.0,
    denoise: Optional[DenoiseSpec] = None,
    features: Optional[Dict[str, Any]] = None,
    dimred_method: Optional[str] = None,
    dimred_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Genetic search over method params to optimize a backtest metric.

    - search_space: dict or JSON like {param: {type, min, max, choices?, log?}}
    - metric: e.g., 'avg_rmse', 'avg_mae', 'avg_directional_accuracy'
    - mode: 'min' or 'max'
    """
    try:
        ss = _parse_kv_or_json(search_space)
        # Prefer multi-method search unless user pins a single method AND provides a flat space
        method_for_search: Optional[str] = method
        from ..forecast.tune import default_search_space as _default_ss
        if not isinstance(ss, dict) or not ss:
            # No space provided: default to a small multi-method space
            ss = _default_ss(method=None, methods=methods)
            method_for_search = None
        elif isinstance(methods, (list, tuple)) and len(methods) > 0:
            # Explicit methods list present: treat as multi-method search
            method_for_search = None
        return _genetic_search_impl(
            symbol=symbol,
            timeframe=timeframe,  # type: ignore[arg-type]
            method=str(method_for_search) if method_for_search is not None else None,
            methods=methods,
            horizon=int(horizon),
            steps=int(steps),
            spacing=int(spacing),
            search_space=ss,
            metric=str(metric),
            mode=str(mode),
            population=int(population),
            generations=int(generations),
            crossover_rate=float(crossover_rate),
            mutation_rate=float(mutation_rate),
            seed=int(seed),
            trade_threshold=float(trade_threshold),
            denoise=denoise,
            features=features,
            dimred_method=dimred_method,
            dimred_params=dimred_params,
        )
    except Exception as e:
        return {"error": f"Error in genetic tuning: {e}"}


@mcp.tool()
@_auto_connect_wrapper
def forecast_barrier_hit_probabilities(
    symbol: str,
    timeframe: TimeframeLiteral = "H1",
    horizon: int = 12,
    method: Literal['mc_gbm','hmm_mc'] = 'hmm_mc',  # type: ignore
    direction: Literal['long','short'] = 'long',  # trade direction context for TP/SL
    # Barrier specification (choose one style per side)
    tp_abs: Optional[float] = None,
    sl_abs: Optional[float] = None,
    tp_pct: Optional[float] = None,   # percent, e.g. 0.5 => +0.5%
    sl_pct: Optional[float] = None,   # percent, e.g. 0.5 => -0.5%
    tp_pips: Optional[float] = None,  # approximate pip mapping (10*point for FX with 5/3 digits)
    sl_pips: Optional[float] = None,
    params: Optional[Dict[str, Any]] = None,
    denoise: Optional[DenoiseSpec] = None,
) -> Dict[str, Any]:
    """Monte Carlo barrier analysis: probability of reaching TP/SL within horizon.

    - direction: 'long' means TP above and SL below last_price; 'short' means TP below and SL above.
    - method: 'mc_gbm' (GBM) or 'hmm_mc' (Gaussian HMM regimes)
    - Barriers can be absolute prices (tp_abs/sl_abs), percentage offsets (tp_pct/sl_pct),
      or pips (tp_pips/sl_pips). Percentage values are in percent points (0.5 => 0.5%).
    - Returns probabilities of hitting TP before SL, SL before TP, neither hit, and
      time-to-hit stats; also per-step cumulative hit curves.
    """
    try:
        if timeframe not in TIMEFRAME_SECONDS:
            return {"error": f"Invalid timeframe: {timeframe}"}
        p = _parse_kv_or_json(params)
        # Fetch enough history for calibration
        need = int(max(300, horizon + 100))
        df = _fetch_history(symbol, timeframe, need, as_of=None)
        if len(df) < 10:
            return {"error": "Insufficient history for simulation"}
        # Current price baseline
        last_price = float(df['close'].astype(float).iloc[-1])
        # Resolve pip size (approximate): use 10*point for 5/3-digit FX, else 1*point
        pip_size = None
        try:
            info = mt5.symbol_info(symbol)
            if info is not None:
                digits = int(getattr(info, 'digits', 0) or 0)
                point = float(getattr(info, 'point', 0.0) or 0.0)
                pip_size = float(point * (10.0 if digits in (3, 5) else 1.0)) if point > 0 else None
        except Exception:
            pip_size = None

        def _coerce_float(v: Any) -> Optional[float]:
            try:
                if v is None:
                    return None
                return float(str(v))
            except Exception:
                return None

        # Compute absolute TP/SL prices with explicit trade direction
        dir_long = str(direction).lower() == 'long'
        tp_price = _coerce_float(tp_abs)
        sl_price = _coerce_float(sl_abs)
        r_tp = _coerce_float(tp_pct)
        r_sl = _coerce_float(sl_pct)
        pp_tp = _coerce_float(tp_pips)
        pp_sl = _coerce_float(sl_pips)

        if tp_price is None or sl_price is None:
            # Derive from pct/pips if absolutes not provided
            if dir_long:
                if tp_price is None:
                    if r_tp is not None:
                        tp_price = last_price * (1.0 + (r_tp / 100.0))
                    elif pp_tp is not None and pip_size is not None:
                        tp_price = last_price + pp_tp * pip_size
                if sl_price is None:
                    if r_sl is not None:
                        sl_price = last_price * (1.0 - (r_sl / 100.0))
                    elif pp_sl is not None and pip_size is not None:
                        sl_price = last_price - pp_sl * pip_size
            else:  # short
                if tp_price is None:
                    if r_tp is not None:
                        tp_price = last_price * (1.0 - (r_tp / 100.0))
                    elif pp_tp is not None and pip_size is not None:
                        tp_price = last_price - pp_tp * pip_size
                if sl_price is None:
                    if r_sl is not None:
                        sl_price = last_price * (1.0 + (r_sl / 100.0))
                    elif pp_sl is not None and pip_size is not None:
                        sl_price = last_price + pp_sl * pip_size

        if tp_price is None or sl_price is None:
            return {"error": "Provide barriers via tp_abs/sl_abs or tp_pct/sl_pct or tp_pips/sl_pips"}

        # Ensure correct side relative to direction (adjust minimally if inverted)
        if dir_long:
            if tp_price <= last_price:
                tp_price = last_price * 1.000001
            if sl_price >= last_price:
                sl_price = last_price * 0.999999
        else:
            if tp_price >= last_price:
                tp_price = last_price * 0.999999
            if sl_price <= last_price:
                sl_price = last_price * 1.000001

        # Build input series (denoise optional)
        base_col = 'close'
        if denoise:
            try:
                from ..utils.denoise import _apply_denoise as _apply_denoise_util
                added = _apply_denoise_util(df, denoise, default_when='pre_ti')
                if f"{base_col}_dn" in added:
                    base_col = f"{base_col}_dn"
            except Exception:
                pass
        prices = df[base_col].astype(float).to_numpy()

        # Simulate paths
        sims = int(p.get('n_sims', p.get('sims', 2000)) or 2000)
        seed = int(p.get('seed', 42) or 42)
        if str(method).lower() == 'mc_gbm':
            sim = _simulate_gbm_mc(prices, horizon=int(horizon), n_sims=int(sims), seed=int(seed))
        elif str(method).lower() == 'hmm_mc':
            n_states = int(p.get('n_states', 2) or 2)
            sim = _simulate_hmm_mc(prices, horizon=int(horizon), n_states=int(n_states), n_sims=int(sims), seed=int(seed))
        else:
            return {"error": f"Unsupported method: {method}. Use 'mc_gbm' or 'hmm_mc'"}

        price_paths = _np.asarray(sim['price_paths'], dtype=float)
        S, H = price_paths.shape
        # First-hit computations
        tp_first = 0
        sl_first = 0
        both_tie = 0
        no_hit = 0
        t_hit_tp = []
        t_hit_sl = []
        # Per-step cumulative hit curves
        tp_any_by_t = _np.zeros(H, dtype=float)
        sl_any_by_t = _np.zeros(H, dtype=float)
        for s in range(S):
            path = price_paths[s]
            idx_tp = _np.argmax(path >= tp_price) if _np.any(path >= tp_price) else -1
            idx_sl = _np.argmax(path <= sl_price) if _np.any(path <= sl_price) else -1
            # Update cumulative
            if idx_tp >= 0:
                tp_any_by_t[idx_tp:] += 1.0
            if idx_sl >= 0:
                sl_any_by_t[idx_sl:] += 1.0
            # First hit logic
            if idx_tp < 0 and idx_sl < 0:
                no_hit += 1
                continue
            if idx_tp >= 0 and (idx_sl < 0 or idx_tp < idx_sl):
                tp_first += 1
                t_hit_tp.append(idx_tp + 1)  # 1-based bars-to-hit
            elif idx_sl >= 0 and (idx_tp < 0 or idx_sl < idx_tp):
                sl_first += 1
                t_hit_sl.append(idx_sl + 1)
            else:  # tie
                both_tie += 1
                t_hit_tp.append(idx_tp + 1)
                t_hit_sl.append(idx_sl + 1)

        S_f = float(S)
        prob_tp_first = (tp_first + 0.5 * both_tie) / S_f
        prob_sl_first = (sl_first + 0.5 * both_tie) / S_f
        prob_no_hit = no_hit / S_f
        tp_any_curve = (tp_any_by_t / S_f).tolist()
        sl_any_curve = (sl_any_by_t / S_f).tolist()

        def _stats(arr: list[int]) -> Dict[str, float]:
            if not arr:
                return {"mean": float('nan'), "median": float('nan')}
            a = _np.asarray(arr, dtype=float)
            return {"mean": float(a.mean()), "median": float(_np.median(a))}

        tf_secs = TIMEFRAME_SECONDS.get(timeframe, 0)
        tp_stats = _stats(t_hit_tp)
        sl_stats = _stats(t_hit_sl)
        def _finite_or_none(x: float) -> Optional[float]:
            try:
                return float(x) if _np.isfinite(x) else None
            except Exception:
                return None
        # Directional interpretation:
        # - For long: TP is above last_price, SL is below; prob_tp_first is long win probability.
        # - For short: TP is below last_price, SL is above; prob_tp_first is short win probability.
        edge = float(prob_tp_first - prob_sl_first)
        out = {
            "success": True,
            "symbol": symbol,
            "timeframe": timeframe,
            "method": method,
            "horizon": int(horizon),
            "direction": direction,
            "last_price": last_price,
            "tp_price": float(tp_price),
            "sl_price": float(sl_price),
            "prob_tp_first": float(prob_tp_first),
            "prob_sl_first": float(prob_sl_first),
            "prob_no_hit": float(prob_no_hit),
            "edge": float(edge),
            "tp_hit_prob_by_t": [float(v) for v in tp_any_curve],
            "sl_hit_prob_by_t": [float(v) for v in sl_any_curve],
            "time_to_tp_bars": tp_stats,
            "time_to_sl_bars": sl_stats,
            "time_to_tp_seconds": {k: _finite_or_none(v * tf_secs) for k, v in tp_stats.items()},
            "time_to_sl_seconds": {k: _finite_or_none(v * tf_secs) for k, v in sl_stats.items()},
            "params_used": {k: p[k] for k in p if k in {"n_sims", "seed", "n_states"}},
        }
        return out
    except Exception as e:
        return {"error": f"Error computing barrier probabilities: {str(e)}"}


@mcp.tool()
@_auto_connect_wrapper
def forecast_barrier_closed_form(
    symbol: str,
    timeframe: TimeframeLiteral = "H1",
    horizon: int = 12,
    direction: Literal['up','down'] = 'up',  # type: ignore
    barrier: float = 0.0,
    mu: Optional[float] = None,
    sigma: Optional[float] = None,
    denoise: Optional[DenoiseSpec] = None,
) -> Dict[str, Any]:
    """Closed-form single-barrier hit probability for GBM within horizon."""
    try:
        need = int(max(400, horizon + 100))
        df = _fetch_history(symbol, timeframe, need, as_of=None)
        if len(df) < 10:
            return {"error": "Insufficient history"}
        base_col = 'close'
        if denoise:
            try:
                from ..utils.denoise import _apply_denoise as _apply_denoise_util
                added = _apply_denoise_util(df, denoise, default_when='pre_ti')
                if f"{base_col}_dn" in added:
                    base_col = f"{base_col}_dn"
            except Exception:
                pass
        prices = _np.asarray(df[base_col].astype(float).to_numpy(), dtype=float)
        prices = prices[_np.isfinite(prices)]
        if prices.size < 5:
            return {"error": "Insufficient prices"}
        s0 = float(prices[-1])
        if barrier <= 0:
            return {"error": "Provide a positive barrier price"}
        tf_secs = TIMEFRAME_SECONDS.get(timeframe, 0)
        if not tf_secs:
            return {"error": f"Unsupported timeframe seconds for {timeframe}"}
        T = float(tf_secs * int(horizon)) / (365.0 * 24.0 * 3600.0)
        if mu is None or sigma is None:
            with _np.errstate(divide='ignore', invalid='ignore'):
                r = _np.diff(_np.log(_np.maximum(prices, 1e-12)))
            r = r[_np.isfinite(r)]
            if r.size < 5:
                return {"error": "Insufficient returns for calibration"}
            mu_hat = float(_np.mean(r)) * (365.0 * 24.0 * 3600.0 / tf_secs)
            sigma_hat = float(_np.std(r, ddof=1)) * (365.0 * 24.0 * 3600.0 / tf_secs) ** 0.5
            if mu is None:
                mu = mu_hat
            if sigma is None:
                sigma = sigma_hat
        log_drift = float(mu)
        sigma_val = float(sigma)
        if sigma_val <= 0:
            return {"error": "Sigma must be positive"}
        sigma_sq = sigma_val * sigma_val
        gbm_drift = log_drift + 0.5 * sigma_sq
        dir_lower = str(direction).lower()
        if dir_lower == 'down':
            s0_inv = 1.0 / s0
            b_inv = 1.0 / float(barrier)
            inv_drift = sigma_sq - gbm_drift
            prob = _gbm_upcross_prob(s0_inv, b_inv, float(inv_drift), sigma_val, float(T))
        else:
            prob = _gbm_upcross_prob(s0, float(barrier), float(gbm_drift), sigma_val, float(T))
        return {
            "success": True,
            "symbol": symbol,
            "timeframe": timeframe,
            "horizon": int(horizon),
            "direction": direction,
            "last_price": s0,
            "barrier": float(barrier),
            "mu_annual": float(gbm_drift),
            "log_drift_annual": float(log_drift),
            "sigma_annual": sigma_val,
            "prob_hit": float(prob),
        }
    except Exception as e:
        return {"error": f"Error computing closed-form barrier probability: {str(e)}"}


@mcp.tool()
@_auto_connect_wrapper
def forecast_barrier_optimize(
    symbol: str,
    timeframe: TimeframeLiteral = "H1",
    horizon: int = 12,
    method: Literal['mc_gbm','hmm_mc'] = 'hmm_mc',  # type: ignore
    direction: Literal['long','short'] = 'long',  # trade direction context for TP/SL
    mode: Literal['pct','pips'] = 'pct',  # type: ignore
    tp_min: float = 0.25,
    tp_max: float = 1.5,
    tp_steps: int = 7,
    sl_min: float = 0.25,
    sl_max: float = 2.5,
    sl_steps: int = 9,
    params: Optional[Dict[str, Any]] = None,
    denoise: Optional[DenoiseSpec] = None,
    objective: Literal['edge','prob_tp_first','kelly','ev','ev_uncond','kelly_uncond'] = 'edge',  # type: ignore
    return_grid: bool = True,
    top_k: Optional[int] = None,
    output: Literal['full','summary'] = 'full',  # type: ignore
    grid_style: Literal['fixed','volatility','ratio','preset'] = 'fixed',  # type: ignore
    preset: Optional[str] = None,
    vol_window: int = 250,
    vol_min_mult: float = 0.5,
    vol_max_mult: float = 4.0,
    vol_steps: int = 7,
    vol_sl_extra: float = 1.8,
    vol_floor_pct: float = 0.15,
    vol_floor_pips: float = 8.0,
    ratio_min: float = 0.5,
    ratio_max: float = 4.0,
    ratio_steps: int = 8,
    refine: bool = False,
    refine_radius: float = 0.3,
    refine_steps: int = 5,
) -> Dict[str, Any]:
    """Optimize TP/SL barriers with support for presets, volatility scaling, ratios, and two-stage refinement."""
    try:
        if timeframe not in TIMEFRAME_SECONDS:
            return {"error": f"Invalid timeframe: {timeframe}"}

        params_dict = _parse_kv_or_json(params)
        mode_val = str(mode).lower()
        objective_val = str(objective).lower()
        valid_objectives = {'edge', 'prob_tp_first', 'kelly', 'ev', 'kelly_uncond', 'ev_uncond'}
        if objective_val not in valid_objectives:
            objective_val = 'edge'

        grid_style_val = str(params_dict.get('grid_style', grid_style)).lower()
        if grid_style_val not in {'fixed', 'volatility', 'ratio', 'preset'}:
            grid_style_val = 'fixed'
        preset_candidate = params_dict.get('grid_preset', params_dict.get('preset', preset))
        preset_val = str(preset_candidate).lower() if isinstance(preset_candidate, str) and preset_candidate else None

        refine_flag = bool(params_dict.get('refine', refine))
        refine_radius_val = max(0.0, float(params_dict.get('refine_radius', refine_radius)))
        refine_steps_val = max(2, int(params_dict.get('refine_steps', refine_steps)))

        ratio_min_val = float(params_dict.get('ratio_min', ratio_min))
        ratio_max_val = float(params_dict.get('ratio_max', ratio_max))
        ratio_steps_val = max(2, int(params_dict.get('ratio_steps', ratio_steps)))
        if ratio_min_val <= 0:
            ratio_min_val = ratio_min
        if ratio_max_val < ratio_min_val:
            ratio_max_val = ratio_min_val

        vol_window_val = int(params_dict.get('vol_window', vol_window))
        vol_min_mult_val = float(params_dict.get('vol_min_mult', vol_min_mult))
        vol_max_mult_val = float(params_dict.get('vol_max_mult', vol_max_mult))
        vol_steps_val = max(2, int(params_dict.get('vol_steps', vol_steps)))
        vol_sl_extra_val = float(params_dict.get('vol_sl_extra', vol_sl_extra))
        vol_sl_multiplier_val = float(params_dict.get('vol_sl_multiplier', vol_sl_extra_val))
        vol_sl_steps_val = max(vol_steps_val, int(params_dict.get('vol_sl_steps', vol_steps_val + 2)))
        vol_floor_pct_val = float(params_dict.get('vol_floor_pct', vol_floor_pct))
        vol_floor_pips_val = float(params_dict.get('vol_floor_pips', vol_floor_pips))

        # Optional risk/reward filter applied across all grid styles
        rr_min_val = params_dict.get('rr_min')
        rr_max_val = params_dict.get('rr_max')
        try:
            rr_min_val = float(rr_min_val) if rr_min_val is not None else None
        except Exception:
            rr_min_val = None
        try:
            rr_max_val = float(rr_max_val) if rr_max_val is not None else None
        except Exception:
            rr_max_val = None
        if rr_min_val is not None and rr_min_val <= 0:
            rr_min_val = None
        if rr_max_val is not None and rr_max_val <= 0:
            rr_max_val = None

        tp_min_val = float(params_dict.get('tp_min', tp_min))
        tp_max_val = float(params_dict.get('tp_max', tp_max))
        tp_steps_val = max(1, int(params_dict.get('tp_steps', tp_steps)))
        sl_min_val = float(params_dict.get('sl_min', sl_min))
        sl_max_val = float(params_dict.get('sl_max', sl_max))
        sl_steps_val = max(1, int(params_dict.get('sl_steps', sl_steps)))

        need = int(max(300, horizon + 100))
        df = _fetch_history(symbol, timeframe, need, as_of=None)
        if len(df) < 10:
            return {"error": "Insufficient history for simulation"}
        last_price = float(df['close'].astype(float).iloc[-1])

        pip_size = None
        try:
            info = mt5.symbol_info(symbol)
            if info is not None:
                digits = int(getattr(info, 'digits', 0) or 0)
                point = float(getattr(info, 'point', 0.0) or 0.0)
                if point > 0:
                    pip_size = float(point * (10.0 if digits in (3, 5) else 1.0))
        except Exception:
            pip_size = None
        if mode_val == 'pips' and (pip_size is None or pip_size <= 0):
            return {"error": "Pip size unavailable for this symbol; use mode='pct' or provide absolute barriers."}

        base_col = 'close'
        if denoise:
            try:
                from ..utils.denoise import _apply_denoise as _apply_denoise_util
                added = _apply_denoise_util(df, denoise, default_when='pre_ti')
                if f"{base_col}_dn" in added:
                    base_col = f"{base_col}_dn"
            except Exception:
                pass
        prices = df[base_col].astype(float).to_numpy()

        sims = int(params_dict.get('n_sims', params_dict.get('sims', 4000)) or 4000)
        seed = int(params_dict.get('seed', 42) or 42)
        n_seeds = int(params_dict.get('n_seeds', 1) or 1)
        paths_list: List[_np.ndarray] = []
        method_name = str(method).lower()
        if method_name == 'mc_gbm':
            for offset in range(max(1, n_seeds)):
                sim = _simulate_gbm_mc(prices, horizon=int(horizon), n_sims=int(sims), seed=int(seed + offset))
                paths_list.append(_np.asarray(sim['price_paths'], dtype=float))
        elif method_name == 'hmm_mc':
            n_states = int(params_dict.get('n_states', 2) or 2)
            for offset in range(max(1, n_seeds)):
                sim = _simulate_hmm_mc(prices, horizon=int(horizon), n_states=int(n_states), n_sims=int(sims), seed=int(seed + offset))
                paths_list.append(_np.asarray(sim['price_paths'], dtype=float))
        else:
            return {"error": f"Unsupported method: {method}. Use 'mc_gbm' or 'hmm_mc'"}

        paths = _np.vstack(paths_list) if len(paths_list) > 1 else paths_list[0]
        S, H = paths.shape
        last_idx = H - 1

        def _linspace(a: float, b: float, n: int) -> _np.ndarray:
            try:
                return _np.linspace(float(a), float(b), int(max(1, n)))
            except Exception:
                return _np.array([float(a)])

        seen: Set[Tuple[int, int]] = set()
        base_candidates: List[Tuple[float, float]] = []

        def _push(tp_unit: float, sl_unit: float, bucket: List[Tuple[float, float]]) -> None:
            try:
                tp_val = float(tp_unit)
                sl_val = float(sl_unit)
            except (TypeError, ValueError):
                return
            if not _np.isfinite(tp_val) or not _np.isfinite(sl_val):
                return
            if tp_val <= 0 or sl_val <= 0:
                return
            key = (int(round(tp_val * 1e6)), int(round(sl_val * 1e6)))
            if key in seen:
                return
            seen.add(key)
            bucket.append((tp_val, sl_val))

        def _add_fixed(bucket: List[Tuple[float, float]], tp_a: float, tp_b: float, tp_n: int, sl_a: float, sl_b: float, sl_n: int) -> None:
            for tp_val in _linspace(tp_a, tp_b, tp_n):
                for sl_val in _linspace(sl_a, sl_b, sl_n):
                    _push(tp_val, sl_val, bucket)

        vol_context: Optional[Dict[str, Any]] = None

        if grid_style_val == 'preset':
            preset_key = preset_val or 'intraday'
            cfg = BARRIER_GRID_PRESETS.get(preset_key, BARRIER_GRID_PRESETS['intraday'])
            if mode_val == 'pct':
                _add_fixed(base_candidates, cfg['tp_min'], cfg['tp_max'], int(cfg['tp_steps']), cfg['sl_min'], cfg['sl_max'], int(cfg['sl_steps']))
            else:
                scale = (float(last_price) / float(pip_size)) / 100.0
                _add_fixed(base_candidates, cfg['tp_min'] * scale, cfg['tp_max'] * scale, int(cfg['tp_steps']), cfg['sl_min'] * scale, cfg['sl_max'] * scale, int(cfg['sl_steps']))
        elif grid_style_val == 'volatility':
            win = min(max(vol_window_val, 20), len(prices) - 1)
            sigma_pct = 0.0
            if win > 5:
                with _np.errstate(divide='ignore', invalid='ignore'):
                    r = _np.diff(_np.log(_np.maximum(prices[-(win + 1):], 1e-12)))
                r = r[_np.isfinite(r)]
                if r.size > 5:
                    sigma_step = float(_np.std(r, ddof=1))
                    if _np.isfinite(sigma_step):
                        sigma_pct = float(sigma_step * (_np.sqrt(max(1, horizon))) * 100.0)
            base_pct = max(sigma_pct, vol_floor_pct_val)
            if mode_val == 'pct':
                tp_base = base_pct
                sl_base = base_pct
            else:
                price_move = (base_pct / 100.0) * float(last_price)
                pip_unit = float(pip_size)
                base_pips = price_move / pip_unit if pip_unit > 0 else 0.0
                base_pips = max(base_pips, vol_floor_pips_val)
                tp_base = base_pips
                sl_base = base_pips
            tp_mults = _linspace(vol_min_mult_val, vol_max_mult_val, vol_steps_val)
            sl_mults = _linspace(vol_min_mult_val, vol_max_mult_val * vol_sl_multiplier_val, vol_sl_steps_val)
            for tp_m in tp_mults:
                for sl_m in sl_mults:
                    _push(tp_base * float(tp_m), sl_base * float(sl_m), base_candidates)
            vol_context = {
                'sigma_pct_horizon': sigma_pct,
                'base_unit': tp_base,
                'tp_multipliers': [float(v) for v in tp_mults.tolist()],
                'sl_multipliers': [float(v) for v in sl_mults.tolist()],
            }
        elif grid_style_val == 'ratio':
            for sl_val in _linspace(sl_min_val, sl_max_val, sl_steps_val):
                for ratio_val in _linspace(ratio_min_val, ratio_max_val, ratio_steps_val):
                    _push(float(sl_val * ratio_val), float(sl_val), base_candidates)
        else:
            _add_fixed(base_candidates, tp_min_val, tp_max_val, tp_steps_val, sl_min_val, sl_max_val, sl_steps_val)

        if not base_candidates:
            _add_fixed(base_candidates, tp_min_val, tp_max_val, tp_steps_val, sl_min_val, sl_max_val, sl_steps_val)

        spread_bps = float(params_dict.get('spread_bps', 0.0) or 0.0)
        fee_bps = float(params_dict.get('fee_bps', 0.0) or 0.0)
        slippage_bps = float(params_dict.get('slippage_bps', 0.0) or 0.0)
        total_bps = spread_bps + fee_bps + slippage_bps

        results: List[Dict[str, Any]] = []

        def _evaluate(tp_unit: float, sl_unit: float, source: str) -> Optional[Dict[str, Any]]:
            long_dir = str(direction).lower() == 'long'
            if mode_val == 'pct':
                if long_dir:
                    tp_price = float(last_price * (1.0 + tp_unit / 100.0))
                    sl_price = float(last_price * (1.0 - sl_unit / 100.0))
                    tp_dist = tp_price - float(last_price)
                    sl_dist = float(last_price) - sl_price
                else:
                    tp_price = float(last_price * (1.0 - tp_unit / 100.0))
                    sl_price = float(last_price * (1.0 + sl_unit / 100.0))
                    tp_dist = float(last_price) - tp_price
                    sl_dist = sl_price - float(last_price)
            else:
                if long_dir:
                    tp_price = float(last_price + tp_unit * float(pip_size))
                    sl_price = float(last_price - sl_unit * float(pip_size))
                    tp_dist = float(tp_unit * float(pip_size))
                    sl_dist = float(sl_unit * float(pip_size))
                else:
                    tp_price = float(last_price - tp_unit * float(pip_size))
                    sl_price = float(last_price + sl_unit * float(pip_size))
                    tp_dist = float(tp_unit * float(pip_size))
                    sl_dist = float(sl_unit * float(pip_size))
            if tp_dist <= 0 or sl_dist <= 0:
                return None

            tp_first = sl_first = both_tie = no_hit = 0
            t_hit_tp: List[int] = []
            t_hit_sl: List[int] = []
            pnl = _np.zeros(S, dtype=float)
            for idx in range(S):
                path = paths[idx]
                idx_tp = _np.argmax(path >= tp_price) if _np.any(path >= tp_price) else -1
                idx_sl = _np.argmax(path <= sl_price) if _np.any(path <= sl_price) else -1
                if idx_tp < 0 and idx_sl < 0:
                    no_hit += 1
                    pnl[idx] = float(path[last_idx] - float(last_price))
                    continue
                if idx_tp >= 0 and (idx_sl < 0 or idx_tp < idx_sl):
                    tp_first += 1
                    t_hit_tp.append(idx_tp + 1)
                    pnl[idx] = float(tp_dist)
                elif idx_sl >= 0 and (idx_tp < 0 or idx_sl < idx_tp):
                    sl_first += 1
                    t_hit_sl.append(idx_sl + 1)
                    pnl[idx] = float(-sl_dist)
                else:
                    both_tie += 1
                    t_hit_tp.append(idx_tp + 1)
                    t_hit_sl.append(idx_sl + 1)
                    pnl[idx] = float(0.5 * (tp_dist - sl_dist))

            if total_bps != 0.0:
                per_trade_cost = float(total_bps) * 1e-4 * float(last_price) * 2.0
                pnl -= per_trade_cost

            S_f = float(S)
            p_tp_first = (tp_first + 0.5 * both_tie) / S_f
            p_sl_first = (sl_first + 0.5 * both_tie) / S_f
            prob_hit_total = p_tp_first + p_sl_first
            prob_no_hit = no_hit / S_f
            prob_hit_any = 1.0 - prob_no_hit
            p_win_hit = (p_tp_first / prob_hit_total) if prob_hit_total > 0 else _np.nan
            reward_risk = float(tp_dist / sl_dist)
            # Enforce optional RR filter
            if (rr_min_val is not None and reward_risk < rr_min_val) or (rr_max_val is not None and reward_risk > rr_max_val):
                return None
            kelly = float(p_win_hit - (1.0 - p_win_hit) / reward_risk) if prob_hit_total > 0 and reward_risk > 0 and _np.isfinite(p_win_hit) else float('-inf')
            ev = float(p_win_hit * reward_risk - (1.0 - p_win_hit)) if prob_hit_total > 0 and reward_risk > 0 and _np.isfinite(p_win_hit) else float('-inf')
            edge = float(p_tp_first - p_sl_first)
            ev_uncond_raw = float(_np.mean(pnl))
            ev_uncond = float(ev_uncond_raw / sl_dist) if sl_dist > 0 else float('nan')
            p_win_uncond = float(_np.mean(pnl > 0.0))
            kelly_uncond = float(p_win_uncond - (1.0 - p_win_uncond) / reward_risk) if reward_risk > 0 else float('-inf')
            tp_med = float(_np.median(_np.asarray(t_hit_tp))) if t_hit_tp else float('nan')
            sl_med = float(_np.median(_np.asarray(t_hit_sl))) if t_hit_sl else float('nan')
            se_tp = float(_np.sqrt(max(p_tp_first * (1.0 - p_tp_first), 0.0) / S_f))
            se_sl = float(_np.sqrt(max(p_sl_first * (1.0 - p_sl_first), 0.0) / S_f))
            se_no = float(_np.sqrt(max(prob_no_hit * (1.0 - prob_no_hit), 0.0) / S_f))
            ev_std = float(_np.std(pnl / sl_dist if sl_dist > 0 else pnl, ddof=1)) if S > 1 else 0.0
            se_ev_uncond = float(ev_std / _np.sqrt(S_f)) if S > 1 else 0.0

            return {
                'tp': float(tp_unit),
                'sl': float(sl_unit),
                'tp_price': tp_price,
                'sl_price': sl_price,
                'reward_risk': float(reward_risk),
                'prob_tp_first': float(p_tp_first), 'prob_tp_first_se': se_tp,
                'prob_sl_first': float(p_sl_first), 'prob_sl_first_se': se_sl,
                'prob_no_hit': float(prob_no_hit), 'prob_no_hit_se': se_no,
                'prob_hit_any': float(prob_hit_any),
                'prob_win_given_hit': float(p_win_hit) if _np.isfinite(p_win_hit) else None,
                'edge': float(edge),
                'kelly': float(kelly),
                'ev': float(ev),
                'kelly_uncond': float(kelly_uncond),
                'ev_uncond': float(ev_uncond), 'ev_uncond_se': se_ev_uncond,
                'pnl_mean': float(ev_uncond_raw),
                'tp_median_bars': tp_med,
                'sl_median_bars': sl_med,
                'source': source,
            }

        for tp_unit, sl_unit in base_candidates:
            res = _evaluate(tp_unit, sl_unit, 'base')
            if res:
                results.append(res)

        refine_candidates: List[Tuple[float, float]] = []

        objective_map = {
            'edge': 'edge',
            'prob_tp_first': 'prob_tp_first',
            'kelly': 'kelly',
            'ev': 'ev',
            'kelly_uncond': 'kelly_uncond',
            'ev_uncond': 'ev_uncond',
        }
        score_key = objective_map.get(objective_val, 'edge')

        def _score_value(row: Dict[str, Any]) -> float:
            val = row.get(score_key, float('-inf'))
            if not _np.isfinite(val):
                return float('-inf')
            return float(val)

        if refine_flag and results:
            try:
                primary_best = max(results, key=_score_value)
            except ValueError:
                primary_best = None
            if primary_best and refine_radius_val > 0:
                base_tp = max(float(primary_best['tp']), 1e-6)
                base_sl = max(float(primary_best['sl']), 1e-6)
                tp_low = max(base_tp * (1.0 - refine_radius_val), base_tp * 0.25)
                tp_high = base_tp * (1.0 + refine_radius_val)
                sl_low = max(base_sl * (1.0 - refine_radius_val), base_sl * 0.25)
                sl_high = base_sl * (1.0 + refine_radius_val)
                if tp_high > tp_low and sl_high > sl_low:
                    for tp_val in _linspace(tp_low, tp_high, refine_steps_val):
                        for sl_val in _linspace(sl_low, sl_high, refine_steps_val):
                            _push(tp_val, sl_val, refine_candidates)
                    for tp_unit, sl_unit in refine_candidates:
                        res = _evaluate(tp_unit, sl_unit, 'refine')
                        if res:
                            results.append(res)

        if not results:
            return {"error": "No valid grid points computed"}

        best = max(results, key=_score_value)

        tracked_keys = {
            'n_sims', 'seed', 'n_states', 'grid_style', 'grid_preset', 'preset', 'vol_window', 'vol_min_mult', 'vol_max_mult', 'vol_steps',
            'vol_sl_extra', 'vol_sl_multiplier', 'vol_sl_steps', 'vol_floor_pct', 'vol_floor_pips', 'ratio_min', 'ratio_max', 'ratio_steps',
            'refine', 'refine_radius', 'refine_steps', 'spread_bps', 'fee_bps', 'slippage_bps', 'n_seeds'
        }
        params_used = {k: params_dict[k] for k in params_dict if k in tracked_keys}
        params_used.update({
            'grid_style': grid_style_val,
            'preset': preset_val,
            'refine': refine_flag,
            'refine_radius': refine_radius_val,
            'refine_steps': refine_steps_val,
        })

        payload: Dict[str, Any] = {
            'success': True,
            'symbol': symbol,
            'timeframe': timeframe,
            'method': method,
            'horizon': int(horizon),
            'direction': direction,
            'mode': mode_val,
            'last_price': last_price,
            'objective': objective_val,
            'grid_style': grid_style_val,
            'preset': preset_val,
            'refine_applied': bool(refine_candidates),
            'grid_points': len(results),
            'best': best,
            'params_used': params_used,
            'score_key': score_key,
        }
        if vol_context is not None:
            payload['volatility_context'] = vol_context

        if isinstance(top_k, int) and top_k > 0:
            top_sorted = sorted(results, key=_score_value, reverse=True)[:int(top_k)]
            payload['top'] = top_sorted
        if return_grid and output == 'full':
            payload['grid'] = results
        if output == 'summary':
            payload = {
                k: v for k, v in payload.items()
                if k in {
                    'success', 'symbol', 'timeframe', 'method', 'horizon', 'mode', 'last_price', 'objective',
                    'grid_style', 'preset', 'refine_applied', 'grid_points', 'best', 'top', 'params_used', 'score_key'
                } and (k != 'top' or 'top' in payload)
            }
        return payload
    except Exception as e:
        return {"error": f"Error optimizing barriers: {str(e)}"}
