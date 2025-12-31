from typing import Any, Dict, Optional, List, Literal, Tuple

from .schema import TimeframeLiteral, DenoiseSpec, ForecastLibraryLiteral, ForecastMethodLiteral
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
from functools import lru_cache
import difflib


@lru_cache(maxsize=1)
def _discover_sktime_forecasters() -> Dict[str, Tuple[str, str]]:
    """Return mapping of forecaster class name (lower) -> (class_name, dotted path).

    Best-effort: skips modules that fail to import (optional dependencies).
    Filters out test modules and private/internal classes.
    """
    try:
        import pkgutil
        import importlib
        import sktime.forecasting as _sf  # type: ignore
        from sktime.forecasting.base import BaseForecaster  # type: ignore
    except Exception:
        return {}

    mapping: Dict[str, Tuple[str, str]] = {}

    def _skip_module(mod_name: str) -> bool:
        parts = mod_name.split(".")
        if "tests" in parts:
            return True
        if any(p.startswith("test") for p in parts):
            return True
        return False

    for mod in pkgutil.walk_packages(getattr(_sf, "__path__", []), _sf.__name__ + "."):
        mod_name = getattr(mod, "name", None)
        if not isinstance(mod_name, str) or _skip_module(mod_name):
            continue
        try:
            m = importlib.import_module(mod_name)
        except Exception:
            continue
        for _, obj in vars(m).items():
            if not isinstance(obj, type):
                continue
            if obj is BaseForecaster:
                continue
            name = getattr(obj, "__name__", None)
            if not isinstance(name, str) or not name or name.startswith("_"):
                continue
            try:
                if not issubclass(obj, BaseForecaster):
                    continue
            except Exception:
                continue
            key = name.lower()
            if key not in mapping:
                mapping[key] = (name, f"{obj.__module__}.{name}")
    return mapping


def _normalize_forecaster_name(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def _resolve_sktime_forecaster(model: str) -> Optional[Tuple[str, str]]:
    """Resolve a user-provided model name to (class_name, dotted_path) using fuzzy matching."""
    model_s = str(model or "").strip()
    if not model_s:
        return None

    mapping = _discover_sktime_forecasters()
    if not mapping:
        return None

    # Exact match on class name (case-insensitive)
    exact = mapping.get(model_s.lower())
    if exact:
        return exact

    # Normalized match (e.g. "theta", "theta_forecaster", "ThetaForecaster")
    norm_map: Dict[str, Tuple[str, str]] = {}
    for _, (cls_name, dotted) in mapping.items():
        norm_map.setdefault(_normalize_forecaster_name(cls_name), (cls_name, dotted))

    qn = _normalize_forecaster_name(model_s)
    if qn in norm_map:
        return norm_map[qn]

    # Prefer startswith/contains in normalized space (more intuitive than difflib alone).
    starts = [v for k, v in norm_map.items() if k.startswith(qn)]
    if starts:
        return sorted(starts, key=lambda t: len(t[0]))[0]
    contains = [v for k, v in norm_map.items() if qn and qn in k]
    if contains:
        return sorted(contains, key=lambda t: len(t[0]))[0]

    # Fallback to difflib against normalized names.
    candidates = difflib.get_close_matches(qn, list(norm_map.keys()), n=1, cutoff=0.6)
    if candidates:
        return norm_map[candidates[0]]
    return None

@mcp.tool()
@_auto_connect_wrapper
def forecast_generate(
    symbol: str,
    timeframe: TimeframeLiteral = "H1",
    method: Optional[str] = None,
    library: Optional[ForecastLibraryLiteral] = None,
    model: Optional[str] = None,
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
    future_covariates: Optional[List[str]] = None,
    country: Optional[str] = None,
) -> Dict[str, Any]:
    """Fast forecasts for the next `horizon` bars using lightweight methods.

    Delegates to the implementation under `mtdata.forecast.forecast`.
    
    Features can include `future_covariates` like 'hour', 'dow', 'month', 'is_holiday' (requires holidays lib).
    """
    try:
        if int(horizon) <= 0:
            return {"error": "horizon must be a positive integer"}
    except Exception:
        return {"error": "horizon must be a positive integer"}
    # Resolve method selection:
    # - Backward compatible: `method` can still be provided directly.
    # - Preferred: (`library`, `model`) selects a method within an optional library without huge CLI enums.
    #
    # CLI compatibility: if `--library` is provided but the caller still passes `--method`,
    # interpret `method` as `model` unless `model` is already provided.
    if library is not None and model is None and method is not None:
        model = str(method)
        method = None

    resolved_method = (str(method).strip() if method is not None else "")
    p = dict(params or {})

    # Backward compatibility shorthands (method-only callers):
    # - sf_* -> statsforecast wrapper + model_name
    # - skt_* -> sktime wrapper + estimator lookup
    # - moirai -> sktime MOIRAIForecaster
    if resolved_method:
        m0 = resolved_method.strip()
        m0_l = m0.lower()
        if m0_l.startswith("sf_"):
            library = "statsforecast"
            model = m0[3:]
            resolved_method = ""
        elif m0_l.startswith("skt_"):
            library = "sktime"
            model = m0[4:]
            resolved_method = ""
        elif m0_l == "moirai":
            library = "sktime"
            model = "MOIRAIForecaster"
            resolved_method = ""

    if not resolved_method:
        lib = (str(library).strip().lower() if library is not None else "")
        mdl = (str(model).strip() if model is not None else "")

        if lib in ("", "native"):
            resolved_method = mdl or "theta"
        elif lib == "statsforecast":
            resolved_method = "statsforecast"
            if mdl:
                # Accept either AutoARIMA / autoarima; normalize to StatsForecast model_name.
                p.setdefault("model_name", mdl)
        elif lib == "sktime":
            # If model looks like a dotted estimator path, use generic `sktime` wrapper.
            if "." in mdl:
                resolved_method = "sktime"
                p.setdefault("estimator", mdl)
            else:
                # Resolve by closest matching forecaster class name.
                # Examples:
                # - "theta" -> ThetaForecaster
                # - "MOIRAI" -> MOIRAIForecaster
                query = mdl.strip() if mdl else "ThetaForecaster"
                found = _resolve_sktime_forecaster(query)
                if found:
                    _, dotted = found
                    resolved_method = "sktime"
                    p.setdefault("estimator", dotted)
                else:
                    # Fall back to the generic wrapper with an explicit dotted path if possible.
                    raise ValueError(
                        f"Unknown sktime forecaster '{query}'. "
                        "Run `python cli.py forecast_generate --library sktime` to list available forecasters."
                    )
        elif lib == "pretrained":
            resolved_method = mdl or "chronos2"
        elif lib == "mlforecast":
            resolved_method = "mlforecast"
            if mdl:
                p.setdefault("model", mdl)
        else:
            # Unknown library: fall back safely.
            resolved_method = mdl or "theta"

    features = features or {}
    if future_covariates:
        # If passed as list, join them for the features dict string/list support
        if isinstance(features, dict):
            features['future_covariates'] = future_covariates
    if country:
        if isinstance(features, dict):
            features['country'] = country

    return _forecast_impl(
        symbol=symbol,
        timeframe=timeframe,  # type: ignore[arg-type]
        method=str(resolved_method),  # type: ignore[arg-type]
        horizon=horizon,
        lookback=lookback,
        as_of=as_of,
        params=p,
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
def forecast_list_library_models(
    library: Literal["native", "statsforecast", "sktime", "pretrained", "mlforecast"],
) -> Dict[str, Any]:
    """List available model names within a forecast library.

    - statsforecast: lists `statsforecast.models.*` class names.
    - sktime: lists supported aliases plus notes for using dotted estimator paths.
    """
    lib = str(library).strip().lower()
    if lib == "native":
        # "Native" methods are mtdata-provided top-level algorithms (as opposed to
        # external-library model spaces like sktime/statsforecast).
        try:
            from mtdata.forecast.forecast_methods import FORECAST_METHODS as _METHODS
        except Exception:
            _METHODS = ()

        excluded = {"statsforecast", "sktime", "mlforecast", "chronos2", "chronos_bolt", "timesfm", "lag_llama"}
        models = [m for m in _METHODS if m not in excluded]
        return {
            "library": lib,
            "models": sorted(models),
            "usage": [
                "python cli.py forecast_generate SYMBOL --library native --model analog",
                "python cli.py forecast_generate SYMBOL --library native --model theta",
            ],
        }

    if lib == "statsforecast":
        try:
            from statsforecast import models as _models  # type: ignore
        except Exception as ex:
            return {"library": lib, "error": f"statsforecast import failed: {ex}"}
        names: List[str] = []
        for attr in dir(_models):
            if attr.startswith("_"):
                continue
            obj = getattr(_models, attr, None)
            if not isinstance(obj, type):
                continue
            if getattr(obj, "__module__", None) != getattr(_models, "__name__", None):
                continue
            if not any(callable(getattr(obj, a, None)) for a in ("fit", "forecast", "predict")):
                continue
            names.append(attr)
        names = sorted(set(names))
        return {
            "library": lib,
            "models": names,
            "usage": "python cli.py forecast_generate SYMBOL --library statsforecast --model AutoARIMA",
        }

    if lib == "sktime":
        mapping = _discover_sktime_forecasters()
        forecasters = sorted({v[0] for v in mapping.values()})
        return {
            "library": lib,
            "models": forecasters,
            "usage": [
                "python cli.py forecast_generate SYMBOL --library sktime --model theta",
                "python cli.py forecast_generate SYMBOL --library sktime --model ThetaForecaster",
                "python cli.py forecast_generate SYMBOL --library sktime --model sktime.forecasting.theta.ThetaForecaster --model-params \"sp=24\"",
            ],
            "note": "The --model value is matched to the closest available forecaster name; you can also pass a dotted class path. Constructor kwargs go in --model-params (or use --set model.<k>=<v>).",
        }

    if lib == "pretrained":
        # These are the pretrained adapters shipped with mtdata.
        pretrained = [
            {
                "model": "chronos2",
                "requires": ["chronos-forecasting>=2.0.0", "torch"],
                "notes": "Hugging Face model id via params.model_name (default: amazon/chronos-bolt-base for compatibility).",
            },
            {
                "model": "chronos_bolt",
                "requires": ["chronos-forecasting>=2.0.0", "torch"],
                "notes": "Same adapter as chronos2; different default naming.",
            },
            {
                "model": "timesfm",
                "requires": ["timesfm", "torch"],
                "notes": "Uses timesfm 2.x (GitHub) API; runs without downloading external weights.",
            },
            {
                "model": "lag_llama",
                "requires": ["lag-llama", "gluonts", "torch"],
                "notes": "May not be installable on Python 3.13 due to upstream pins; included for completeness.",
            },
        ]
        return {
            "library": lib,
            "models": pretrained,
            "usage": [
                "python cli.py forecast_generate SYMBOL --library pretrained --model chronos2",
                "python cli.py forecast_generate SYMBOL --library pretrained --model timesfm",
            ],
        }

    if lib == "mlforecast":
        return {
            "library": lib,
            "note": "Use `--model <dotted sklearn/lightgbm regressor class>` plus optional constructor kwargs in --model-params (or use --set model.<k>=<v>).",
            "usage": [
                "python cli.py forecast_generate SYMBOL --library mlforecast --model sklearn.ensemble.RandomForestRegressor --model-params \"n_estimators=200\"",
                "python cli.py forecast_generate SYMBOL --method mlf_rf",
            ],
        }

    return {"library": lib, "error": "Unsupported library (supported: native, statsforecast, sktime, pretrained, mlforecast)"}


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
def forecast_barrier_prob(
    symbol: str,
    timeframe: TimeframeLiteral = "H1",
    horizon: int = 12,
    method: Literal['mc', 'closed_form'] = 'mc',
    # MC params
    mc_method: Literal['mc_gbm','hmm_mc','garch','bootstrap'] = 'hmm_mc',  # type: ignore
    direction: Literal['long','short', 'up', 'down'] = 'long',  # type: ignore
    tp_abs: Optional[float] = None,
    sl_abs: Optional[float] = None,
    tp_pct: Optional[float] = None,
    sl_pct: Optional[float] = None,
    tp_pips: Optional[float] = None,
    sl_pips: Optional[float] = None,
    params: Optional[Dict[str, Any]] = None,
    denoise: Optional[DenoiseSpec] = None,
    # Closed form params
    barrier: float = 0.0,
    mu: Optional[float] = None,
    sigma: Optional[float] = None,
) -> Dict[str, Any]:
    """Calculate probability of price hitting TP/SL barriers using Monte Carlo or Closed Form methods.
    
    **REQUIRED**: symbol parameter must be provided
    
    Use Cases:
    ----------
    - Validate TP/SL levels before entering a trade
    - Assess probability of hitting profit target vs stop loss
    - Optimize barrier levels based on probability analysis
    
    Parameters:
    -----------
    symbol : str (REQUIRED)
        Trading symbol to analyze (e.g., "EURUSD", "BTCUSD")
    
    timeframe : str, optional (default="H1")
        Analysis timeframe: "M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1", "MN1"
    
    horizon : int, optional (default=12)
        Number of bars to forecast ahead
    
    method : str, optional (default="mc")
        Calculation method:
        - "mc": Monte Carlo simulation (more flexible, handles complex scenarios)
        - "closed_form": Analytical solution (faster, simpler assumptions)
    
    Monte Carlo Parameters (method="mc"):
    -------------------------------------
    mc_method : str, optional (default="hmm_mc")
        - "hmm_mc": Hidden Markov Model-based MC
        - "mc_gbm": Geometric Brownian Motion MC
        - "garch": GARCH(1,1) volatility model (requires 'arch' package)
        - "bootstrap": Circular block bootstrap (historical simulation)
    
    direction : str, optional (default="long")
        Trade direction: "long" / "short" (or "up" / "down" for closed_form)
    
    tp_abs : float, optional
        Absolute take profit price level
    
    sl_abs : float, optional
        Absolute stop loss price level
    
    tp_pct : float, optional
        Take profit as percentage (e.g., 2.0 for 2%)
    
    sl_pct : float, optional
        Stop loss as percentage
    
    tp_pips : float, optional
        Take profit in pips
    
    sl_pips : float, optional
        Stop loss in pips
    
    Closed Form Parameters (method="closed_form"):
    ----------------------------------------------
    barrier : float, optional (default=0.0)
        Target barrier level
    
    mu : float, optional
        Drift parameter (calculated if not provided)
    
    sigma : float, optional
        Volatility parameter (calculated if not provided)
    
    Returns:
    --------
    dict
        Probability analysis including:
        - success: bool
        - symbol: str
        - probabilities: dict with TP/SL hit probabilities
        - method_used: str
    
    Examples:
    ---------
    # Check probability of hitting TP vs SL (Monte Carlo)
    forecast_barrier_prob(
        symbol="EURUSD",
        method="mc",
        direction="long",
        tp_abs=1.1100,
        sl_abs=1.0950
    )
    
    # Use percentage-based barriers
    forecast_barrier_prob(
        symbol="EURUSD",
        direction="long",
        tp_pct=2.0,
        sl_pct=1.0
    )
    
    # Closed form calculation (faster)
    forecast_barrier_prob(
        symbol="GBPUSD",
        method="closed_form",
        direction="up",
        barrier=1.2700
    )
    """
    if method == 'mc':
        from ..forecast.barriers import forecast_barrier_hit_probabilities as _impl
        # Ensure direction is valid for MC
        d = str(direction).lower()
        if d not in ('long', 'short'):
             # fallback mapping
             d = 'long' if d == 'up' else 'short'

        return _impl(
            symbol=symbol,
            timeframe=timeframe,
            horizon=horizon,
            method=mc_method,
            direction=d, # type: ignore
            tp_abs=tp_abs,
            sl_abs=sl_abs,
            tp_pct=tp_pct,
            sl_pct=sl_pct,
            tp_pips=tp_pips,
            sl_pips=sl_pips,
            params=params,
            denoise=denoise,
        )
    elif method == 'closed_form':
        from ..forecast.barriers import forecast_barrier_closed_form as _impl
        # Map direction: long->up, short->down if user passed long/short
        d = str(direction).lower()
        if d == 'long': d = 'up'
        elif d == 'short': d = 'down'
        
        return _impl(
            symbol=symbol,
            timeframe=timeframe,
            horizon=horizon,
            direction=d, # type: ignore
            barrier=barrier,
            mu=mu,
            sigma=sigma,
            denoise=denoise,
        )
    else:
        return {"error": f"Unknown method: {method}"}


@mcp.tool()
@_auto_connect_wrapper
def forecast_barrier_optimize(
    symbol: str,
    timeframe: TimeframeLiteral = "H1",
    horizon: int = 12,
    method: Literal['mc_gbm','hmm_mc','garch','bootstrap'] = 'hmm_mc',  # type: ignore
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
