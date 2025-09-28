
from typing import Any, Dict, Optional, List, Literal

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
import MetaTrader5 as mt5
import numpy as _np
from typing import Any, Optional, Dict, List, Literal

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
        if not isinstance(ss, dict) or not ss:
            # Build sensible defaults based on provided method(s)
            from ..forecast.tune import default_search_space as _default_ss
            ss = _default_ss(method=method, methods=methods)
        return _genetic_search_impl(
            symbol=symbol,
            timeframe=timeframe,  # type: ignore[arg-type]
            method=str(method) if method is not None else None,
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

        # Compute absolute TP/SL prices
        tp_price = _coerce_float(tp_abs)
        sl_price = _coerce_float(sl_abs)
        r_tp = _coerce_float(tp_pct)
        r_sl = _coerce_float(sl_pct)
        pp_tp = _coerce_float(tp_pips)
        pp_sl = _coerce_float(sl_pips)

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

        if tp_price is None or sl_price is None:
            return {"error": "Provide barriers via tp_abs/sl_abs or tp_pct/sl_pct or tp_pips/sl_pips"}
        if not (tp_price > last_price and sl_price < last_price):
            # Tolerate inverted sides but warn
            if tp_price <= last_price:
                tp_price = last_price * 1.000001
            if sl_price >= last_price:
                sl_price = last_price * 0.999999

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
        edge = float(prob_tp_first - prob_sl_first)
        out = {
            "success": True,
            "symbol": symbol,
            "timeframe": timeframe,
            "method": method,
            "horizon": int(horizon),
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
    """Closed-form single-barrier hit probability for GBM within horizon.

    If mu/sigma are omitted, calibrates from recent log-returns. For 'down',
    computes upcrossing on inverted price.
    """
    try:
        # Fetch recent history for calibration and last price
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
        # Time in years (approx) from number of bars
        tf_secs = TIMEFRAME_SECONDS.get(timeframe, 0)
        if not tf_secs:
            return {"error": f"Unsupported timeframe seconds for {timeframe}"}
        T = float(tf_secs * int(horizon)) / (365.0 * 24.0 * 3600.0)
        # Calibrate mu/sigma if not provided (drift and vol of log-returns)
        if mu is None or sigma is None:
            with _np.errstate(divide='ignore', invalid='ignore'):
                r = _np.diff(_np.log(_np.maximum(prices, 1e-12)))
            r = r[_np.isfinite(r)]
            if r.size < 5:
                return {"error": "Insufficient returns for calibration"}
            mu_hat = float(_np.mean(r)) * (365.0 * 24.0 * 3600.0 / tf_secs)  # per year
            sigma_hat = float(_np.std(r, ddof=1)) * (365.0 * 24.0 * 3600.0 / tf_secs) ** 0.5
            if mu is None:
                mu = mu_hat
            if sigma is None:
                sigma = sigma_hat
        # Direction handling: for down barrier, invert price and barrier
        if str(direction).lower() == 'down':
            # P(S hits down barrier) = P(1/S hits up barrier at 1/barrier) with adjusted drift
            s0_inv = 1.0 / s0
            b_inv = 1.0 / float(barrier)
            # For X_t = ln S_t, inversion changes drift sign on centered BM; use same GBM formula approximately
            prob = _gbm_upcross_prob(s0_inv, b_inv, -float(mu), float(sigma), float(T))
        else:
            prob = _gbm_upcross_prob(s0, float(barrier), float(mu), float(sigma), float(T))
        return {
            "success": True,
            "symbol": symbol,
            "timeframe": timeframe,
            "horizon": int(horizon),
            "direction": direction,
            "last_price": s0,
            "barrier": float(barrier),
            "mu_annual": float(mu),
            "sigma_annual": float(sigma),
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
    mode: Literal['pct','pips'] = 'pct',  # type: ignore
    tp_min: float = 0.2,
    tp_max: float = 1.0,
    tp_steps: int = 5,
    sl_min: float = 0.2,
    sl_max: float = 1.0,
    sl_steps: int = 5,
    params: Optional[Dict[str, Any]] = None,
    denoise: Optional[DenoiseSpec] = None,
    objective: Literal['edge','prob_tp_first','kelly','ev','ev_uncond','kelly_uncond'] = 'edge',  # type: ignore
    return_grid: bool = True,
    top_k: Optional[int] = None,
    output: Literal['full','summary'] = 'full',  # type: ignore
) -> Dict[str, Any]:
    """Optimize TP/SL barriers over a grid using Monte Carlo paths.

    - mode='pct' treats values as percent points (0.5 => 0.5%).
    - mode='pips' treats values as pips (approx pip=10*point for 5/3-digit FX).
    - Returns grid results and the best configuration by objective.
    Objectives:
      - edge: prob_tp_first - prob_sl_first
      - prob_tp_first: maximize TP-first probability
      - kelly: p - (1-p)/b, where p = P(TP before SL | hit), b = TP distance / SL distance
      - ev: expected value per unit risk = p*b - (1-p)
    """
    try:
        if timeframe not in TIMEFRAME_SECONDS:
            return {"error": f"Invalid timeframe: {timeframe}"}
        p = _parse_kv_or_json(params)
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
                pip_size = float(point * (10.0 if digits in (3, 5) else 1.0)) if point > 0 else None
        except Exception:
            pip_size = None
        if mode == 'pips' and pip_size is None:
            return {"error": "Pip size unavailable for this symbol; use mode='pct' or provide absolute barriers."}

        # Denoise optional
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

        # Simulate one set of paths for grid evaluation
        sims = int(p.get('n_sims', p.get('sims', 4000)) or 4000)
        seed = int(p.get('seed', 42) or 42)
        n_seeds = int(p.get('n_seeds', 1) or 1)
        paths_list = []
        if str(method).lower() == 'mc_gbm':
            for i in range(max(1, n_seeds)):
                sim = _simulate_gbm_mc(prices, horizon=int(horizon), n_sims=int(sims), seed=int(seed + i))
                paths_list.append(_np.asarray(sim['price_paths'], dtype=float))
        elif str(method).lower() == 'hmm_mc':
            n_states = int(p.get('n_states', 2) or 2)
            for i in range(max(1, n_seeds)):
                sim = _simulate_hmm_mc(prices, horizon=int(horizon), n_states=int(n_states), n_sims=int(sims), seed=int(seed + i))
                paths_list.append(_np.asarray(sim['price_paths'], dtype=float))
        else:
            return {"error": f"Unsupported method: {method}. Use 'mc_gbm' or 'hmm_mc'"}

        paths = _np.vstack(paths_list) if len(paths_list) > 1 else paths_list[0]
        S, H = paths.shape
        last_idx = H - 1

        # Build grids
        def _linspace(a: float, b: float, n: int) -> _np.ndarray:
            try:
                return _np.linspace(float(a), float(b), int(max(1, n)))
            except Exception:
                return _np.array([float(a)])
        tp_vals = _linspace(tp_min, tp_max, tp_steps)
        sl_vals = _linspace(sl_min, sl_max, sl_steps)

        results: List[Dict[str, Any]] = []
        for tp in tp_vals:
            for sl in sl_vals:
                # Convert to absolute prices
                if mode == 'pct':
                    tp_price = last_price * (1.0 + float(tp) / 100.0)
                    sl_price = last_price * (1.0 - float(sl) / 100.0)
                    tp_dist = tp_price - last_price
                    sl_dist = last_price - sl_price
                else:  # pips
                    tp_price = last_price + float(tp) * float(pip_size)
                    sl_price = last_price - float(sl) * float(pip_size)
                    tp_dist = float(tp) * float(pip_size)
                    sl_dist = float(sl) * float(pip_size)
                if tp_dist <= 0 or sl_dist <= 0:
                    continue

                tp_first = 0
                sl_first = 0
                both_tie = 0
                no_hit = 0
                t_hit_tp: List[int] = []
                t_hit_sl: List[int] = []
                # Per-path realized PnL (unconditional), in same units as distances
                pnl = _np.zeros(S, dtype=float)
                for sidx in range(S):
                    path = paths[sidx]
                    idx_tp = _np.argmax(path >= tp_price) if _np.any(path >= tp_price) else -1
                    idx_sl = _np.argmax(path <= sl_price) if _np.any(path <= sl_price) else -1
                    if idx_tp < 0 and idx_sl < 0:
                        no_hit += 1
                        # Close at horizon: PnL is end - start
                        pnl[sidx] = float(path[last_idx] - last_price)
                        continue
                    if idx_tp >= 0 and (idx_sl < 0 or idx_tp < idx_sl):
                        tp_first += 1
                        t_hit_tp.append(idx_tp + 1)
                        pnl[sidx] = float(tp_dist)
                    elif idx_sl >= 0 and (idx_tp < 0 or idx_sl < idx_tp):
                        sl_first += 1
                        t_hit_sl.append(idx_sl + 1)
                        pnl[sidx] = float(-sl_dist)
                    else:
                        both_tie += 1
                        t_hit_tp.append(idx_tp + 1)
                        t_hit_sl.append(idx_sl + 1)
                        # Split tie as average of TP/SL outcomes
                        pnl[sidx] = float(0.5 * (tp_dist - sl_dist))

                # Apply simple cost model if provided: entry+exit costs in price units
                spread_bps = float(p.get('spread_bps', 0.0) or 0.0)
                fee_bps = float(p.get('fee_bps', 0.0) or 0.0)
                slippage_bps = float(p.get('slippage_bps', 0.0) or 0.0)
                total_bps = (spread_bps + fee_bps + slippage_bps)
                if total_bps != 0.0:
                    # Convert bps to price units relative to last_price, cost paid on both entry and exit
                    per_trade_cost = float(total_bps) * 1e-4 * float(last_price) * 2.0
                    pnl -= per_trade_cost

                S_f = float(S)
                p_tp_first = (tp_first + 0.5 * both_tie) / S_f
                p_sl_first = (sl_first + 0.5 * both_tie) / S_f
                denom = p_tp_first + p_sl_first
                p_win = (p_tp_first / denom) if denom > 0 else 0.0
                b = float(tp_dist / sl_dist) if sl_dist > 0 else 0.0
                kelly = float(p_win - (1.0 - p_win) / b) if b > 0 else float('-inf')
                ev = float(p_win * b - (1.0 - p_win)) if b > 0 else float('-inf')
                edge = float(p_tp_first - p_sl_first)
                # Unconditional metrics including no-hit and costs
                ev_uncond_raw = float(_np.mean(pnl))
                # Normalize to "per unit risk" similar to ev if desired
                ev_uncond = float(ev_uncond_raw / sl_dist) if sl_dist > 0 else float('nan')
                p_win_uncond = float(_np.mean(pnl > 0.0))
                kelly_uncond = float(p_win_uncond - (1.0 - p_win_uncond) / b) if b > 0 else float('-inf')

                tp_med = float(_np.median(_np.asarray(t_hit_tp))) if t_hit_tp else float('nan')
                sl_med = float(_np.median(_np.asarray(t_hit_sl))) if t_hit_sl else float('nan')
                # Uncertainty estimates (SE) for probabilities and EV
                se_tp = float(_np.sqrt(max(p_tp_first * (1.0 - p_tp_first), 0.0) / S_f))
                se_sl = float(_np.sqrt(max(p_sl_first * (1.0 - p_sl_first), 0.0) / S_f))
                se_no = float(_np.sqrt(max((no_hit / S_f) * (1.0 - (no_hit / S_f)), 0.0) / S_f))
                # For EV_uncond, use sample std/sqrt(S)
                ev_std = float(_np.std(pnl / sl_dist if sl_dist > 0 else pnl, ddof=1)) if S > 1 else 0.0
                se_ev_uncond = float(ev_std / _np.sqrt(S_f)) if S > 1 else 0.0

                results.append({
                    'tp': float(tp), 'sl': float(sl),
                    'prob_tp_first': float(p_tp_first), 'prob_tp_first_se': se_tp,
                    'prob_sl_first': float(p_sl_first), 'prob_sl_first_se': se_sl,
                    'prob_no_hit': float(no_hit / S_f), 'prob_no_hit_se': se_no,
                    'edge': float(edge),
                    'kelly': float(kelly),
                    'ev': float(ev),
                    'kelly_uncond': float(kelly_uncond),
                    'ev_uncond': float(ev_uncond), 'ev_uncond_se': se_ev_uncond,
                    'tp_median_bars': tp_med,
                    'sl_median_bars': sl_med,
                })

        if not results:
            return {"error": "No valid grid points computed"}

        # Choose best by objective
        if objective == 'prob_tp_first':
            best = max(results, key=lambda r: r['prob_tp_first'])
        elif objective == 'kelly':
            best = max(results, key=lambda r: r['kelly'])
        elif objective == 'ev':
            best = max(results, key=lambda r: r['ev'])
        elif objective == 'kelly_uncond':
            best = max(results, key=lambda r: r['kelly_uncond'])
        elif objective == 'ev_uncond':
            best = max(results, key=lambda r: r['ev_uncond'])
        else:
            best = max(results, key=lambda r: r['edge'])

        payload: Dict[str, Any] = {
            'success': True,
            'symbol': symbol,
            'timeframe': timeframe,
            'method': method,
            'horizon': int(horizon),
            'mode': mode,
            'last_price': last_price,
            'objective': objective if objective in {'edge','prob_tp_first','kelly','ev','kelly_uncond','ev_uncond'} else 'edge',
            'grid_points': len(results),
            'best': best,
            'params_used': {k: p[k] for k in p if k in {"n_sims", "seed", "n_states"}},
        }
        # Optional top-k
        if isinstance(top_k, int) and top_k > 0:
            key = ('prob_tp_first' if objective == 'prob_tp_first' else ('kelly' if objective == 'kelly' else ('ev' if objective == 'ev' else 'edge')))
            top_sorted = sorted(results, key=lambda r: r.get(key, float('-inf')), reverse=True)[:int(top_k)]
            payload['top'] = top_sorted
        # Return grid conditionally
        if return_grid and output == 'full':
            payload['grid'] = results
        if output == 'summary':
            # Strip verbose fields
            payload = {k: v for k, v in payload.items() if k in {'success','symbol','timeframe','method','horizon','mode','last_price','objective','grid_points','best','top','params_used'} and (k != 'top' or 'top' in payload)}
        return payload
    except Exception as e:
        return {"error": f"Error optimizing barriers: {str(e)}"}
