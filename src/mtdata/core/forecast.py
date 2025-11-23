
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
    from ..forecast.barriers import forecast_barrier_hit_probabilities as _impl
    return _impl(
        symbol=symbol,
        timeframe=timeframe,
        horizon=horizon,
        method=method,
        direction=direction,
        tp_abs=tp_abs,
        sl_abs=sl_abs,
        tp_pct=tp_pct,
        sl_pct=sl_pct,
        tp_pips=tp_pips,
        sl_pips=sl_pips,
        params=params,
        denoise=denoise,
    )


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
    from ..forecast.barriers import forecast_barrier_closed_form as _impl
    return _impl(
        symbol=symbol,
        timeframe=timeframe,
        horizon=horizon,
        direction=direction,
        barrier=barrier,
        mu=mu,
        sigma=sigma,
        denoise=denoise,
    )


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
    from ..forecast.barriers import forecast_barrier_optimize as _impl
    return _impl(
        symbol=symbol,
        timeframe=timeframe,
        horizon=horizon,
        method=method,
        direction=direction,
        mode=mode,
        tp_min=tp_min,
        tp_max=tp_max,
        tp_steps=tp_steps,
        sl_min=sl_min,
        sl_max=sl_max,
        sl_steps=sl_steps,
        params=params,
        denoise=denoise,
        objective=objective,
        return_grid=return_grid,
        top_k=top_k,
        output=output,
        grid_style=grid_style,
        preset=preset,
        vol_window=vol_window,
        vol_min_mult=vol_min_mult,
        vol_max_mult=vol_max_mult,
        vol_steps=vol_steps,
        vol_sl_extra=vol_sl_extra,
        vol_floor_pct=vol_floor_pct,
        vol_floor_pips=vol_floor_pips,
        ratio_min=ratio_min,
        ratio_max=ratio_max,
        ratio_steps=ratio_steps,
        refine=refine,
        refine_radius=refine_radius,
        refine_steps=refine_steps,
    )
