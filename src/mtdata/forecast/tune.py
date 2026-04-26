from __future__ import annotations

import math
import random
import threading
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple

from .backtest import forecast_backtest as _forecast_backtest
from .optimize import (
    build_comprehensive_search_space as _build_search_space,
)
from .optimize import (
    composite_fitness_score as _composite_fitness,
)
from .optimize import (
    extract_method_params_from_genotype as _extract_params,
)

# Sensible default search spaces per method (lightweight, CPU-friendly)
# These are intentionally conservative to keep runtime practical.
_DEFAULT_SPACES_METHOD_SCOPED: Dict[str, Dict[str, Any]] = {
    "_shared": {},
    # Classical fast methods
    "theta": {
        # Seasonality period in bars; intraday typical daily cycles ~ 24, weekly ~ 5 for D1
        "seasonality": {"type": "int", "min": 8, "max": 72},
    },
    "fourier_ols": {
        # Fourier period (bars) and number of harmonics; allow optional trend toggle
        "m": {"type": "int", "min": 8, "max": 96},
        "K": {"type": "int", "min": 1, "max": 6},
        "trend": {"type": "categorical", "choices": [True, False]},
    },
    "seasonal_naive": {
        # Period for repeating last seasonal value
        "seasonality": {"type": "int", "min": 5, "max": 96},
    },
    "naive": {},
    "drift": {},
    "ses": {
        # Smoothing level for Simple Exponential Smoothing
        "alpha": {"type": "float", "min": 0.05, "max": 0.95},
    },
    "holt": {
        # Damped trend on/off
        "damped": {"type": "categorical", "choices": [True, False]},
    },
    "holt_winters_add": {
        "seasonality": {"type": "int", "min": 8, "max": 72},
    },
    "holt_winters_mul": {
        "seasonality": {"type": "int", "min": 8, "max": 72},
    },
    "arima": {
        "p": {"type": "int", "min": 0, "max": 3},
        "d": {"type": "int", "min": 0, "max": 2},
        "q": {"type": "int", "min": 0, "max": 3},
    },
    "sarima": {
        "p": {"type": "int", "min": 0, "max": 3},
        "d": {"type": "int", "min": 0, "max": 2},
        "q": {"type": "int", "min": 0, "max": 3},
        "P": {"type": "int", "min": 0, "max": 2},
        "D": {"type": "int", "min": 0, "max": 1},
        "Q": {"type": "int", "min": 0, "max": 2},
        "seasonality": {"type": "int", "min": 4, "max": 48},
    },
    # Monte Carlo
    "mc_gbm": {
        "n_sims": {"type": "int", "min": 200, "max": 1000},
        "seed": {"type": "categorical", "choices": [13, 37, 42, 99]},
    },
    "hmm_mc": {
        "n_sims": {"type": "int", "min": 200, "max": 1000},
        "n_states": {"type": "int", "min": 2, "max": 4},
        "seed": {"type": "categorical", "choices": [13, 37, 42, 99]},
    },
    # NeuralForecast (lightweight ranges)
    "nhits": {
        "input_size": {"type": "int", "min": 64, "max": 256},
        "max_epochs": {"type": "int", "min": 10, "max": 50},
        "batch_size": {"type": "int", "min": 16, "max": 64},
        "learning_rate": {"type": "float", "min": 1e-4, "max": 1e-2, "log": True},
    },
    "nbeatsx": {
        "input_size": {"type": "int", "min": 64, "max": 256},
        "max_epochs": {"type": "int", "min": 10, "max": 50},
        "batch_size": {"type": "int", "min": 16, "max": 64},
        "learning_rate": {"type": "float", "min": 1e-4, "max": 1e-2, "log": True},
    },
    "tft": {
        "input_size": {"type": "int", "min": 64, "max": 256},
        "max_epochs": {"type": "int", "min": 10, "max": 50},
        "batch_size": {"type": "int", "min": 16, "max": 64},
        "learning_rate": {"type": "float", "min": 1e-4, "max": 1e-2, "log": True},
    },
    "patchtst": {
        "input_size": {"type": "int", "min": 64, "max": 256},
        "max_epochs": {"type": "int", "min": 10, "max": 50},
        "batch_size": {"type": "int", "min": 16, "max": 64},
        "learning_rate": {"type": "float", "min": 1e-4, "max": 1e-2, "log": True},
    },
    # StatsForecast
    "sf_autoarima": {
        "seasonality": {"type": "int", "min": 8, "max": 72},
        "stepwise": {"type": "categorical", "choices": [True, False]},
        "d": {"type": "int", "min": 0, "max": 2},
        "D": {"type": "int", "min": 0, "max": 1},
    },
    "sf_theta": {
        "seasonality": {"type": "int", "min": 8, "max": 72},
    },
    "sf_autoets": {
        "seasonality": {"type": "int", "min": 8, "max": 72},
    },
    "sf_seasonalnaive": {
        "seasonality": {"type": "int", "min": 5, "max": 96},
    },
    # MLForecast
    "mlf_rf": {
        "n_estimators": {"type": "int", "min": 100, "max": 500},
        "max_depth": {"type": "categorical", "choices": [None, 5, 10, 15, 20]},
    },
    "mlf_lightgbm": {
        "n_estimators": {"type": "int", "min": 100, "max": 500},
        "learning_rate": {"type": "float", "min": 0.01, "max": 0.2},
        "num_leaves": {"type": "int", "min": 15, "max": 63},
        "max_depth": {"type": "categorical", "choices": [-1, 6, 8, 12, 16]},
    },
    # Transformer family (point forecasts use context length primarily)
    "chronos_bolt": {
        "context_length": {"type": "int", "min": 64, "max": 320},
    },
    "chronos2": {
        "context_length": {"type": "int", "min": 64, "max": 320},
    },
    "timesfm": {
        "context_length": {"type": "int", "min": 64, "max": 320},
    },
    "lag_llama": {
        "context_length": {"type": "int", "min": 64, "max": 320},
    },
    # Ensemble (not implemented): placeholder
    "ensemble": {},
}


def default_search_space(method: Optional[str] = None, methods: Optional[List[str]] = None) -> Dict[str, Any]:
    """Return a sensible default search space.

    - Multiple methods: returns a method-scoped dict with sections for each listed method
      (falling back to shared defaults where available).
    - Single method: returns a flat parameter space for that method.
    - If neither provided, returns method-scoped defaults for a small common set.
    """
    if methods and isinstance(methods, (list, tuple)) and len(methods) > 0:
        out: Dict[str, Any] = {"_shared": dict(_DEFAULT_SPACES_METHOD_SCOPED.get("_shared", {}))}
        for m in methods:
            if m in _DEFAULT_SPACES_METHOD_SCOPED:
                out[m] = dict(_DEFAULT_SPACES_METHOD_SCOPED[m])
        # Ensure at least something is present
        if len(out) == 1:  # only _shared
            # add a couple of common ones if user passed unknowns
            for m in ("theta", "fourier_ols"):
                out[m] = dict(_DEFAULT_SPACES_METHOD_SCOPED[m])
        return out
    if method:
        sp = _DEFAULT_SPACES_METHOD_SCOPED.get(str(method))
        if isinstance(sp, dict) and sp:
            return dict(sp)
        # Fallback to a generic seasonality search for classical methods
        return {"seasonality": {"type": "int", "min": 8, "max": 48}}
    # Neither provided: return a compact method-scoped default
    return {
        "_shared": {},
        "theta": dict(_DEFAULT_SPACES_METHOD_SCOPED["theta"]),
        "fourier_ols": dict(_DEFAULT_SPACES_METHOD_SCOPED["fourier_ols"]),
    }


Metric = str


def _is_flat_search_space(sp: Dict[str, Any]) -> bool:
    return any(
        isinstance(v, dict) and ('type' in v or 'min' in v or 'max' in v or 'choices' in v)
        for v in sp.values()
    )


def _suggest_optuna_param(trial: Any, name: str, space: Dict[str, Any]) -> Any:
    t = str(space.get('type', 'float')).lower()
    if t == 'categorical':
        choices = list(space.get('choices') or [])
        if not choices:
            return None
        return trial.suggest_categorical(name, choices)
    if t == 'int':
        lo_raw = space.get('min', 0)
        hi_raw = space.get('max', lo_raw)
        try:
            lo = int(lo_raw)
        except Exception:
            lo = 0
        try:
            hi = int(hi_raw)
        except Exception:
            hi = lo
        if lo > hi:
            lo, hi = hi, lo
        return int(trial.suggest_int(name, lo, hi))
    lo_raw = space.get('min', 0.0)
    hi_raw = space.get('max', None)
    try:
        lo = float(lo_raw)
    except Exception:
        lo = 0.0
    if hi_raw is None:
        hi_raw = max(lo, 1.0)
    try:
        hi = float(hi_raw)
    except Exception:
        hi = max(lo, 1.0)
    if lo > hi:
        lo, hi = hi, lo
    if bool(space.get('log', False)) and lo > 0.0 and hi > 0.0:
        return float(trial.suggest_float(name, lo, hi, log=True))
    return float(trial.suggest_float(name, lo, hi))


def _eval_candidate(
    *,
    symbol: str,
    timeframe: str,
    method: Optional[str],
    horizon: int,
    steps: int,
    spacing: int,
    candidate_params: Dict[str, Any],
    metric: Metric,
    mode: str,
    denoise: Optional[Dict[str, Any]] = None,
    features: Optional[Dict[str, Any]] = None,
    dimred_method: Optional[str] = None,
    dimred_params: Optional[Dict[str, Any]] = None,
    trade_threshold: float = 0.0,
) -> Tuple[float, Dict[str, Any]]:
    """Run a backtest for a single candidate and return (score, result_dict).

    mode: 'min' or 'max'. score is direction-consistent (lower is better if mode == 'min').
    """
    # Allow method gene inside candidate
    sel_method = str(candidate_params.get('method')) if candidate_params.get('method') else (str(method) if method else None)
    if not sel_method:
        return (math.inf if mode == 'min' else -math.inf, {"error": "No method provided"})
    cand_only = {k: v for k, v in candidate_params.items() if k != 'method'}
    res = _forecast_backtest(
        symbol=symbol,
        timeframe=timeframe,  # type: ignore
        horizon=int(horizon),
        steps=int(steps),
        spacing=int(spacing),
        methods=[sel_method],
        params_per_method={sel_method: cand_only},
        denoise=denoise,
        features=features,
        dimred_method=dimred_method,
        dimred_params=dimred_params,
        trade_threshold=float(trade_threshold),
    )
    # Pull method aggregate
    r = res.get('results', {}).get(sel_method) if isinstance(res, dict) else None
    if not isinstance(r, dict) or not r.get('success'):
        return (math.inf if mode == 'min' else -math.inf, res)
    val = r.get(metric)
    try:
        score = float(val)
    except Exception:
        # Fallback to rmse/mae if metric missing
        val2 = r.get('avg_rmse', r.get('avg_mae'))
        try:
            score = float(val2)
        except Exception:
            score = math.inf if mode == 'min' else -math.inf
    return (score if mode == 'min' else -score, {'_sel_method': sel_method, **(res or {})})


def _sample_param(space: Dict[str, Any], rng: random.Random) -> Any:
    t = str(space.get('type', 'float')).lower()
    if t == 'categorical':
        choices = space.get('choices') or []
        return rng.choice(list(choices)) if choices else None
    if t == 'int':
        lo_raw = space.get('min', 0)
        hi_raw = space.get('max', lo_raw)
        try:
            lo = int(lo_raw)
        except Exception:
            lo = 0
        try:
            hi = int(hi_raw)
        except Exception:
            hi = lo
        if lo > hi:
            lo, hi = hi, lo
        return rng.randint(lo, hi)
    # float (optionally log)
    lo_raw = space.get('min', 0.0)
    hi_raw = space.get('max', None)
    try:
        lo = float(lo_raw)
    except Exception:
        lo = 0.0
    if hi_raw is None:
        hi_raw = max(lo, 1.0)
    try:
        hi = float(hi_raw)
    except Exception:
        hi = max(lo, 1.0)
    if lo > hi:
        lo, hi = hi, lo
    if bool(space.get('log', False)) and lo > 0 and hi > 0:
        import math as _m
        a = _m.log(lo); b = _m.log(hi)
        x = rng.random() * (b - a) + a
        return float(_m.exp(x))
    return rng.random() * (hi - lo) + lo


def _mutate_value(value: Any, space: Dict[str, Any], rng: random.Random, strength: float = 0.2) -> Any:
    t = str(space.get('type', 'float')).lower()
    if t == 'categorical':
        choices = list(space.get('choices') or [])
        if not choices:
            return value
        if len(choices) == 1:
            return choices[0]
        # pick a different choice
        cand = [c for c in choices if c != value]
        return rng.choice(cand) if cand else value
    if t == 'int':
        lo_raw = space.get('min', value)
        hi_raw = space.get('max', value)
        try:
            lo = int(lo_raw if lo_raw is not None else (value if value is not None else 0))
        except Exception:
            lo = 0
        try:
            hi = int(hi_raw if hi_raw is not None else lo)
        except Exception:
            hi = lo
        if value is None:
            value = int((lo + hi) // 2)
        span = max(1, int(round((hi - lo) * strength)))
        return max(lo, min(hi, int(value) + rng.randint(-span, span)))
    # float
    lo_raw = space.get('min', value)
    hi_raw = space.get('max', value)
    try:
        lo = float(lo_raw if lo_raw is not None else (value if value is not None else 0.0))
    except Exception:
        lo = 0.0
    try:
        hi = float(hi_raw if hi_raw is not None else (lo + 1.0))
    except Exception:
        hi = max(lo, 1.0)
    if value is None:
        try:
            value = float((lo + hi) * 0.5)
        except Exception:
            value = lo
    span = (hi - lo) * max(0.01, strength)
    return max(lo, min(hi, float(value) + (rng.random() * 2.0 - 1.0) * span))


def _crossover_for_method(
    a: Dict[str, Any],
    b: Dict[str, Any],
    param_spaces: Dict[str, Any],
    rng: random.Random,
) -> Dict[str, Any]:
    child: Dict[str, Any] = {}
    for k, spec in param_spaces.items():
        av = a.get(k)
        bv = b.get(k)
        child[k] = av if (rng.random() < 0.5) else bv
        if child[k] is None and av is None and bv is None:
            try:
                child[k] = _sample_param(spec or {}, rng)
            except Exception:
                child[k] = None
        # small blend for floats
        t = str(spec.get('type', 'float')).lower()
        if t not in ('categorical', 'int'):
            try:
                fa = float(av if av is not None else _sample_param(spec or {}, rng))
                fb = float(bv if bv is not None else _sample_param(spec or {}, rng))
                child[k] = fa * 0.5 + fb * 0.5
            except Exception:
                pass
    return child


def optuna_search_forecast_params(  # noqa: C901
    *,
    symbol: str,
    timeframe: str,
    method: Optional[str],
    methods: Optional[List[str]] = None,
    horizon: int = 12,
    steps: int = 5,
    spacing: int = 20,
    search_space: Optional[Dict[str, Any]] = None,
    metric: Metric = 'avg_rmse',
    mode: str = 'min',
    n_trials: int = 40,
    timeout: Optional[float] = None,
    n_jobs: int = 1,
    sampler: str = 'tpe',
    pruner: str = 'median',
    seed: int = 42,
    study_name: Optional[str] = None,
    storage: Optional[str] = None,
    denoise: Optional[Dict[str, Any]] = None,
    features: Optional[Dict[str, Any]] = None,
    dimred_method: Optional[str] = None,
    dimred_params: Optional[Dict[str, Any]] = None,
    trade_threshold: float = 0.0,
) -> Dict[str, Any]:
    """Optuna search for best params for a forecast method under backtest."""
    import optuna

    mode_val = str(mode or 'min').strip().lower()
    if mode_val not in {'min', 'max'}:
        mode_val = 'min'

    raw = dict(search_space or {})
    method_scoped = not _is_flat_search_space(raw)
    shared_key = '_shared'
    method_names_from_space: List[str] = []
    if method_scoped:
        method_names_from_space = [k for k in raw.keys() if k != shared_key]

    def _spaces_for(mname: Optional[str]) -> Dict[str, Any]:
        if not method_scoped:
            sp = dict(raw)
            sp.pop('method', None)
            return sp
        out: Dict[str, Any] = {}
        shared = raw.get(shared_key) if isinstance(raw.get(shared_key), dict) else {}
        if isinstance(shared, dict):
            out.update(shared)
        if mname and isinstance(raw.get(mname), dict):
            out.update(raw.get(mname))
        return out

    method_choices: List[str] = []
    if isinstance(methods, (list, tuple)) and methods:
        method_choices = list(methods)
    elif method_scoped and method_names_from_space:
        method_choices = list(method_names_from_space)

    has_method_gene = (
        (not method_scoped)
        and ('method' in raw)
        and isinstance(raw.get('method'), dict)
        and str(raw['method'].get('type', 'categorical')).lower() == 'categorical'
    )

    sampler_name = str(sampler or 'tpe').strip().lower()
    if sampler_name == 'random':
        sampler_obj = optuna.samplers.RandomSampler(seed=int(seed))
    elif sampler_name == 'cmaes':
        sampler_obj = optuna.samplers.CmaEsSampler(seed=int(seed))
    else:
        sampler_name = 'tpe'
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r".*multivariate.*experimental.*",
                category=Warning,
            )
            sampler_obj = optuna.samplers.TPESampler(seed=int(seed), multivariate=True)

    pruner_name = str(pruner or 'median').strip().lower()
    if pruner_name in {'none', 'off', 'disabled'}:
        pruner_obj = optuna.pruners.NopPruner()
    elif pruner_name == 'hyperband':
        pruner_obj = optuna.pruners.HyperbandPruner()
    elif pruner_name == 'percentile':
        pruner_obj = optuna.pruners.PercentilePruner(50.0)
    else:
        pruner_name = 'median'
        pruner_obj = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0)

    storage_val: Optional[str]
    if storage is None:
        storage_val = None
    else:
        storage_str = str(storage).strip()
        storage_val = storage_str or None
    study_name_val: Optional[str]
    if study_name is None:
        study_name_val = None
    else:
        name_str = str(study_name).strip()
        study_name_val = name_str or None
    load_if_exists = bool(storage_val and study_name_val)

    direction = 'minimize' if mode_val == 'min' else 'maximize'
    study = optuna.create_study(
        direction=direction,
        sampler=sampler_obj,
        pruner=pruner_obj,
        study_name=study_name_val,
        storage=storage_val,
        load_if_exists=load_if_exists,
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    history: List[Dict[str, Any]] = []
    trial_results: Dict[int, Dict[str, Any]] = {}
    trial_candidates: Dict[int, Dict[str, Any]] = {}
    lock = threading.Lock()

    best_score = math.inf if mode_val == 'min' else -math.inf
    best_params: Dict[str, Any] = {}
    best_result: Optional[Dict[str, Any]] = None

    def _objective(trial: Any) -> float:
        nonlocal best_score, best_params, best_result
        cand: Dict[str, Any] = {}

        sel_method = None
        if has_method_gene:
            try:
                sel_method = _suggest_optuna_param(trial, 'method', raw['method'])
                cand['method'] = sel_method
            except Exception:
                sel_method = None
        elif method_choices:
            sel_method = trial.suggest_categorical('method', list(method_choices))
            cand['method'] = sel_method
        else:
            sel_method = method
            if sel_method is not None:
                cand['method'] = sel_method

        pspaces = _spaces_for(str(sel_method) if sel_method else None)
        for k, spec in pspaces.items():
            cand[k] = _suggest_optuna_param(trial, k, spec or {})

        score, res = _eval_candidate(
            symbol=symbol,
            timeframe=timeframe,
            method=method,
            horizon=horizon,
            steps=steps,
            spacing=spacing,
            candidate_params=cand,
            metric=metric,
            mode=mode_val,
            denoise=denoise,
            features=features,
            dimred_method=dimred_method,
            dimred_params=dimred_params,
            trade_threshold=float(trade_threshold),
        )

        true_score = score if mode_val == 'min' else -score
        finite_score = float(true_score) if math.isfinite(true_score) else None
        objective_score = float(true_score)
        if not math.isfinite(objective_score):
            objective_score = 1e18 if mode_val == 'min' else -1e18

        with lock:
            hist_row: Dict[str, Any] = {"trial": int(trial.number), "score": float(objective_score), "params": dict(cand)}
            if isinstance(res, dict) and res.get('_sel_method'):
                hist_row['method'] = res.get('_sel_method')
            history.append(hist_row)
            if isinstance(res, dict):
                trial_results[int(trial.number)] = res
            trial_candidates[int(trial.number)] = dict(cand)

            if finite_score is not None:
                better = (
                    (mode_val == 'min' and finite_score < best_score)
                    or (mode_val != 'min' and finite_score > best_score)
                )
                if better:
                    best_score = finite_score
                    best_params = dict(cand)
                    best_result = res if isinstance(res, dict) else None

        return float(objective_score)

    timeout_val = None
    if timeout is not None:
        try:
            timeout_float = float(timeout)
            timeout_val = timeout_float if timeout_float > 0 else None
        except Exception:
            timeout_val = None
    n_jobs_val = max(1, int(n_jobs))
    n_trials_val = max(1, int(n_trials))
    study.optimize(_objective, n_trials=n_trials_val, timeout=timeout_val, n_jobs=n_jobs_val)

    if not best_params and len(study.trials) > 0:
        try:
            bt = study.best_trial
            best_params = dict(trial_candidates.get(int(bt.number), bt.params))
            best_score = float(bt.value)
            br = trial_results.get(int(bt.number))
            best_result = br if isinstance(br, dict) else None
        except Exception:
            pass

    payload: Dict[str, Any] = {
        "success": True,
        "best_score": float(best_score),
        "best_params": best_params,
        "metric": metric,
        "mode": mode_val,
        "optimizer": "optuna",
        "n_trials": int(n_trials_val),
        "timeout": float(timeout_val) if timeout_val is not None else None,
        "n_jobs": int(n_jobs_val),
        "sampler": sampler_name,
        "pruner": pruner_name,
        "seed": int(seed),
        "history_count": len(history),
    }
    if storage_val:
        payload["storage"] = storage_val
    if study_name_val:
        payload["study_name"] = study_name_val
    if best_result is not None:
        sel = best_result.get('_sel_method') if isinstance(best_result, dict) else None
        agg = None
        try:
            agg = best_result.get('results', {}).get(sel) if isinstance(best_result, dict) else None
        except Exception:
            agg = None
        payload["best_method"] = sel
        if isinstance(agg, dict):
            payload["best_result_summary"] = {"horizon": agg.get('horizon'), "result": agg}
    if history:
        payload["history_tail"] = history[-50:]
    return payload


def genetic_search_forecast_params(  # noqa: C901
    *,
    symbol: str,
    timeframe: str,
    method: Optional[str],
    methods: Optional[List[str]] = None,
    horizon: int = 12,
    steps: int = 5,
    spacing: int = 20,
    search_space: Optional[Dict[str, Any]] = None,
    metric: Metric = 'avg_rmse',
    mode: str = 'min',
    population: int = 12,
    generations: int = 10,
    crossover_rate: float = 0.6,
    mutation_rate: float = 0.3,
    seed: int = 42,
    denoise: Optional[Dict[str, Any]] = None,
    features: Optional[Dict[str, Any]] = None,
    dimred_method: Optional[str] = None,
    dimred_params: Optional[Dict[str, Any]] = None,
    trade_threshold: float = 0.0,
) -> Dict[str, Any]:
    """Genetic search for best params for a forecast method under backtest.

    - search_space: {param: {type: 'int'|'float'|'categorical', min, max, choices?, log?}}
    - metric: one of backtest aggregates (e.g., 'avg_rmse', 'avg_mae', 'avg_directional_accuracy')
    - mode: 'min' or 'max' (direction)
    """
    rng = random.Random(int(seed))
    raw = dict(search_space or {})

    # Detect method-scoped search space vs flat
    def _is_flat(sp: Dict[str, Any]) -> bool:
        return any(isinstance(v, dict) and ('type' in v or 'min' in v or 'max' in v or 'choices' in v) for v in sp.values())

    method_scoped = not _is_flat(raw)
    shared_key = '_shared'
    method_names_from_space: List[str] = []
    if method_scoped:
        method_names_from_space = [k for k in raw.keys() if k != shared_key]

    # Helper to get param spaces for a given method
    def _spaces_for(mname: Optional[str]) -> Dict[str, Any]:
        if not method_scoped:
            # Flat space; drop method gene if present
            sp = dict(raw)
            sp.pop('method', None)
            return sp
        # Merge shared + method-specific
        out: Dict[str, Any] = {}
        shared = raw.get(shared_key) if isinstance(raw.get(shared_key), dict) else {}
        if isinstance(shared, dict):
            out.update(shared)
        if mname and isinstance(raw.get(mname), dict):
            out.update(raw.get(mname))
        return out

    # Build method choices if searching over methods
    method_choices: List[str] = []
    if isinstance(methods, (list, tuple)) and methods:
        method_choices = list(methods)
    elif method_scoped and method_names_from_space:
        method_choices = list(method_names_from_space)

    # If there are method choices and no explicit 'method' gene in a flat space, we will sample it explicitly
    has_method_gene = (not method_scoped) and ('method' in raw) and str(raw['method'].get('type', 'categorical')).lower() == 'categorical' if 'method' in raw and isinstance(raw['method'], dict) else False

    # Initialize population
    pop: List[Dict[str, Any]] = []
    for _ in range(max(2, int(population))):
        cand: Dict[str, Any] = {}
        # Choose method if searching across methods
        sel_method = None
        if has_method_gene:
            try:
                sel_method = _sample_param(raw['method'], rng)
                cand['method'] = sel_method
            except Exception:
                sel_method = None
        elif method_choices:
            sel_method = rng.choice(method_choices)
            cand['method'] = sel_method
        else:
            sel_method = method  # fixed
        # Sample params for selected method
        pspaces = _spaces_for(str(sel_method) if sel_method else None)
        for k, spec in pspaces.items():
            cand[k] = _sample_param(spec or {}, rng)
        pop.append(cand)

    history: List[Dict[str, Any]] = []
    best_score = math.inf if mode == 'min' else -math.inf
    best_params: Dict[str, Any] = {}
    best_result: Optional[Dict[str, Any]] = None

    for gen in range(max(1, int(generations))):
        scored: List[Tuple[float, Dict[str, Any]]] = []
        for cand in pop:
            score, res = _eval_candidate(
                symbol=symbol,
                timeframe=timeframe,
                method=method,
                horizon=horizon,
                steps=steps,
                spacing=spacing,
                candidate_params=cand,
                metric=metric,
                mode=mode,
                denoise=denoise,
                features=features,
                dimred_method=dimred_method,
                dimred_params=dimred_params,
                trade_threshold=float(trade_threshold),
            )
            scored.append((score, cand))
            hist_entry = {"generation": gen, "score": float(score), "params": dict(cand)}
            if isinstance(res, dict) and res.get('_sel_method'):
                hist_entry['method'] = res.get('_sel_method')
            history.append(hist_entry)
            # Keep global best in true metric direction
            true_score = score if mode == 'min' else -score
            if (mode == 'min' and true_score < best_score) or (mode != 'min' and true_score > best_score):
                best_score = true_score
                best_params = dict(cand)
                best_result = res if isinstance(res, dict) else None

        # Selection (elitism: top 2)
        scored.sort(key=lambda t: t[0])  # ascending in adjusted score
        elites = [dict(scored[0][1]), dict(scored[1][1])] if len(scored) >= 2 else [dict(scored[0][1])]

        # Breed new population
        new_pop: List[Dict[str, Any]] = []
        new_pop.extend(elites)
        while len(new_pop) < len(pop):
            # Tournament selection
            a = rng.choice(scored)[1]
            b = rng.choice(scored)[1]
            # Decide child method
            child: Dict[str, Any] = {}
            if has_method_gene:
                # crossover method gene from parents (random pick)
                child_method = rng.choice([a.get('method'), b.get('method')])
                child['method'] = child_method
            elif method_choices:
                child_method = rng.choice([a.get('method'), b.get('method')]) or rng.choice(method_choices)
                child['method'] = child_method
            else:
                child_method = method
                if child_method is not None:
                    child['method'] = child_method
            # Crossover parameters relevant to chosen method
            pspaces = _spaces_for(str(child_method) if child_method else None)
            child.update(_crossover_for_method(a, b, pspaces, rng) if rng.random() < crossover_rate else {})
            # Mutation for parameters of chosen method
            for k, spec in pspaces.items():
                if rng.random() < mutation_rate:
                    child[k] = _mutate_value(child.get(k), spec or {}, rng)
            new_pop.append(child)
        pop = new_pop[: len(pop)]

    payload: Dict[str, Any] = {
        "success": True,
        "best_score": float(best_score),
        "best_params": best_params,
        "metric": metric,
        "mode": mode,
        "population": int(population),
        "generations": int(generations),
        "history_count": len(history),
    }
    if best_result is not None:
        sel = best_result.get('_sel_method') if isinstance(best_result, dict) else None
        agg = None
        try:
            agg = best_result.get('results', {}).get(sel) if isinstance(best_result, dict) else None
        except Exception:
            agg = None
        payload["best_method"] = sel
        if isinstance(agg, dict):
            payload["best_result_summary"] = {"horizon": agg.get('horizon'), "result": agg}
    # Optional: compact history preview (keeps payload small)
    try:
        tail_n = 50
        if isinstance(history, list) and history:
            payload["history_tail"] = history[-tail_n:]
    except Exception:
        pass
    return payload


def genetic_search_optimize_hints(  # noqa: C901
    *,
    symbol: str,
    timeframes: Optional[List[str]] = None,
    methods: Optional[List[str]] = None,
    horizon: int = 12,
    steps: int = 5,
    spacing: int = 20,
    search_space: Optional[Dict[str, Any]] = None,
    fitness_metric: str = "composite",
    fitness_weights: Optional[Dict[str, float]] = None,
    population: int = 20,
    generations: int = 15,
    crossover_rate: float = 0.6,
    mutation_rate: float = 0.3,
    seed: int = 42,
    max_search_time_seconds: Optional[float] = None,
    denoise: Optional[Dict[str, Any]] = None,
    features: Optional[Dict[str, Any]] = None,
    dimred_method: Optional[str] = None,
    dimred_params: Optional[Dict[str, Any]] = None,
    top_n: int = 5,
    include_feature_genes: bool = False,
) -> Dict[str, Any]:
    """Comprehensive genetic search for optimal forecast settings across timeframes, methods, and parameters.

    Searches across:
    - Timeframes (H1, H4, D1, W1, etc.)
    - Methods (fast + pretrained: theta, ARIMA, chronos, timesfm, etc.)
    - Method-specific parameters
    - Optional feature indicators (RSI, MACD, Bollinger Bands)

    Fitness: Composite score combining Sharpe ratio, win rate, inverse drawdown, and return,
    or single metric if fitness_metric != "composite".

    Args:
        symbol: Symbol to optimize for
        timeframes: Timeframes to search (default: ['H1', 'H4', 'D1', 'W1'])
        methods: Methods to search (default: fast + pretrained)
        horizon, steps, spacing: Backtest parameters
        search_space: Optional pre-built search space. If None, uses default from optimize module.
        fitness_metric: 'composite' or specific metric name ('avg_rmse', 'sharpe_ratio', etc.)
        fitness_weights: Custom weights for composite fitness (dict of metric: weight)
        population: Genetic population size
        generations: Number of generations
        crossover_rate, mutation_rate: Genetic parameters
        seed: Random seed
        max_search_time_seconds: Optional timeout for search
        denoise, features, dimred_method, dimred_params: Preprocessing options
        top_n: Number of top configurations to return
        include_feature_genes: If True, search over feature indicators

    Returns:
        Dict with 'success', 'hints' (list of top-N configs), 'search_summary', etc.
    """
    start_time = time.time()
    rng = random.Random(int(seed))

    # Build search space if not provided
    if search_space is None:
        search_space = _build_search_space(
            timeframes=timeframes,
            methods=methods,
            include_features=include_feature_genes,
        )
    else:
        # Ensure comprehensive space has required keys
        if '_method_spaces' not in search_space:
            from .tune import default_search_space as _default_spaces
            search_space['_method_spaces'] = _default_spaces(methods=methods)

    # Extract genetic gene specs
    tf_choices = search_space.get('timeframe', {}).get('choices', timeframes or ['H1', 'H4', 'D1'])
    method_choices = search_space.get('method', {}).get('choices', methods or ['theta'])
    method_spaces = search_space.get('_method_spaces', {})

    # Helper: create random individual (genotype)
    def _create_individual() -> Dict[str, Any]:
        individual: Dict[str, Any] = {
            'timeframe': rng.choice(list(tf_choices)),
            'method': rng.choice(list(method_choices)),
        }
        # Sample parameters for chosen method
        meth = individual['method']
        params_space = method_spaces.get(str(meth), {})
        for param_name, param_spec in params_space.items():
            individual[param_name] = _sample_param(param_spec or {}, rng)
        return individual

    # Helper: crossover two individuals
    def _crossover(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
        child: Dict[str, Any] = {
            'timeframe': rng.choice([a.get('timeframe'), b.get('timeframe')]),
            'method': rng.choice([a.get('method'), b.get('method')]),
        }
        meth = child['method']
        params_space = method_spaces.get(str(meth), {})
        for param_name in params_space.keys():
            av = a.get(param_name)
            bv = b.get(param_name)
            child[param_name] = av if rng.random() < 0.5 else bv
        return child

    # Helper: mutate an individual
    def _mutate(individual: Dict[str, Any]) -> Dict[str, Any]:
        mutant = dict(individual)
        # Mutate timeframe with small probability
        if rng.random() < mutation_rate * 0.3:
            mutant['timeframe'] = rng.choice(list(tf_choices))
        # Mutate method with small probability
        if rng.random() < mutation_rate * 0.3:
            mutant['method'] = rng.choice(list(method_choices))
        # Mutate parameters
        meth = mutant['method']
        params_space = method_spaces.get(str(meth), {})
        for param_name, param_spec in params_space.items():
            if rng.random() < mutation_rate:
                mutant[param_name] = _mutate_value(mutant.get(param_name), param_spec or {}, rng)
        return mutant

    # Helper: evaluate candidate
    def _extract_method_backtest_metrics(backtest_res: Dict[str, Any], method_name: str) -> Dict[str, Any]:
        if not isinstance(backtest_res, dict):
            return {}
        method_results = backtest_res.get('results', {}).get(method_name, {})
        if not isinstance(method_results, dict):
            return {}
        metrics = method_results.get('metrics')
        if isinstance(metrics, dict):
            merged = dict(metrics)
            for key in (
                'avg_mae',
                'avg_rmse',
                'avg_directional_accuracy',
                'successful_tests',
                'num_tests',
            ):
                if key in method_results and key not in merged:
                    merged[key] = method_results.get(key)
            return merged
        return method_results

    def _evaluate(individual: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        tf = str(individual.get('timeframe', 'H1'))
        method = str(individual.get('method', 'theta'))
        params = {k: v for k, v in individual.items() if k not in ('timeframe', 'method')}

        # Run backtest with this config
        score, res = _eval_candidate(
            symbol=symbol,
            timeframe=tf,
            method=method,
            horizon=horizon,
            steps=steps,
            spacing=spacing,
            candidate_params={'method': method, **params},
            metric=fitness_metric if fitness_metric != 'composite' else 'avg_rmse',
            mode='min' if fitness_metric != 'composite' else 'min',
            denoise=denoise,
            features=features,
            dimred_method=dimred_method,
            dimred_params=dimred_params,
            trade_threshold=0.0,
        )

        # Compute fitness
        if fitness_metric == 'composite':
            # Extract backtest metrics
            try:
                backtest_metrics = _extract_method_backtest_metrics(res, method)
            except Exception:
                backtest_metrics = {}
            fitness = _composite_fitness(backtest_metrics, weights=fitness_weights)
            # Convert to minimization score (1 - fitness to minimize)
            fitness_score = 1.0 - fitness
        else:
            # Single metric: use raw score from backtest
            fitness_score = float(score)

        return fitness_score, res

    # Initialize population
    population_size = max(2, int(population))
    pop: List[Tuple[Dict[str, Any], float, Dict[str, Any]]] = []

    for _ in range(population_size):
        ind = _create_individual()
        try:
            fitness, res = _evaluate(ind)
            pop.append((ind, fitness, res))
        except Exception as ex:
            # Failed evaluation; use worst score
            pop.append((ind, math.inf, {'error': str(ex)}))

        # Check timeout
        if max_search_time_seconds and (time.time() - start_time) > max_search_time_seconds:
            return {
                'success': False,
                'error': f'Search timeout after {time.time() - start_time:.1f}s',
                'hints': [],
            }

    history: List[Dict[str, Any]] = []
    best_overall = min(pop, key=lambda t: t[1])[1]
    # Generational loop
    for gen in range(max(1, int(generations))):
        # Sort by fitness
        pop.sort(key=lambda t: t[1])

        # Elitism: keep top 2
        new_pop = [pop[0], pop[1] if len(pop) > 1 else pop[0]]

        # Breed new population
        while len(new_pop) < population_size:
            # Tournament selection
            a_ind, a_fit, _ = rng.choice(pop)
            b_ind, b_fit, _ = rng.choice(pop)

            # Crossover
            if rng.random() < crossover_rate:
                child = _crossover(a_ind, b_ind)
            else:
                child = dict(a_ind if a_fit < b_fit else b_ind)

            # Mutation
            if rng.random() < mutation_rate:
                child = _mutate(child)

            # Evaluate child
            try:
                fitness, res = _evaluate(child)
                new_pop.append((child, fitness, res))
            except Exception as ex:
                new_pop.append((child, math.inf, {'error': str(ex)}))

            # Check timeout
            if max_search_time_seconds and (time.time() - start_time) > max_search_time_seconds:
                return {
                    'success': False,
                    'error': f'Search timeout after {time.time() - start_time:.1f}s',
                    'hints': [],
                }

        pop = new_pop[: population_size]

        # Track best
        gen_best = min(pop, key=lambda t: t[1])
        if gen_best[1] < best_overall:
            best_overall = gen_best[1]

        gen_summary = {
            'generation': gen,
            'best_score': float(gen_best[1]),
            'avg_score': float(sum(p[1] for p in pop) / len(pop)) if pop else float('nan'),
        }
        history.append(gen_summary)

    # Extract top-N candidates
    pop.sort(key=lambda t: t[1])
    top_configs: List[Dict[str, Any]] = []

    seen_configs = set()
    duplicate_results_filtered = 0
    for individual, fitness, backtest_res in pop:
        if len(top_configs) >= int(top_n):
            break
        tf, method, params = _extract_params(individual, search_space)
        config_key = (
            str(tf),
            str(method),
            tuple(sorted((str(key), repr(value)) for key, value in dict(params).items())),
        )
        if config_key in seen_configs:
            duplicate_results_filtered += 1
            continue
        seen_configs.add(config_key)

        # Build hint entry
        hint: Dict[str, Any] = {
            'rank': len(top_configs) + 1,
            'timeframe': tf,
            'method': method,
            'method_params': params,
            'fitness_score': 1.0 - fitness if fitness_metric == 'composite' else fitness,
        }

        # Extract backtest metrics
        try:
            method_metrics = _extract_method_backtest_metrics(backtest_res, method)
            if method_metrics:
                hint['backtest_metrics'] = {
                    'sharpe_ratio': method_metrics.get('sharpe_ratio'),
                    'win_rate': method_metrics.get('win_rate'),
                    'max_drawdown': method_metrics.get('max_drawdown'),
                    'avg_return_per_trade': method_metrics.get('avg_return_per_trade'),
                    'calmar_ratio': method_metrics.get('calmar_ratio'),
                    'annual_return': method_metrics.get('annual_return'),
                }
        except Exception:
            pass

        top_configs.append(hint)

    elapsed = time.time() - start_time
    result: Dict[str, Any] = {
        'success': True,
        'hints': top_configs,
        'search_summary': {
            'symbol': symbol,
            'population': int(population),
            'generations': int(generations),
            'elapsed_seconds': round(elapsed, 2),
            'fitness_metric': fitness_metric,
            'timeframes_searched': list(tf_choices),
            'methods_searched': list(method_choices),
            'total_evaluations': int(population) * int(generations),
            'unique_configs_returned': len(top_configs),
            'duplicate_results_filtered': int(duplicate_results_filtered),
        },
        'history_tail': history[-10:] if history else [],
    }

    return result
