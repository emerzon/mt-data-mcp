from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import math
import random

from .backtest import forecast_backtest as _forecast_backtest


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
        "rolling_agg": {"type": "categorical", "choices": ["mean", "min", "max", "std"]},
    },
    "mlf_lightgbm": {
        "n_estimators": {"type": "int", "min": 100, "max": 500},
        "learning_rate": {"type": "float", "min": 0.01, "max": 0.2},
        "num_leaves": {"type": "int", "min": 15, "max": 63},
        "max_depth": {"type": "categorical", "choices": [-1, 6, 8, 12, 16]},
        "rolling_agg": {"type": "categorical", "choices": ["mean", "min", "max", "std"]},
    },
    # Transformer family (point forecasts use context length primarily)
    "chronos_bolt": {
        "context_length": {"type": "int", "min": 64, "max": 320},
    },
    "chronos2": {
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


def genetic_search_forecast_params(
    *,
    symbol: str,
    timeframe: str,
    method: Optional[str],
    methods: Optional[List[str]] = None,
    horizon: int = 12,
    steps: int = 5,
    spacing: int = 20,
    search_space: Dict[str, Any] = {},
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
    # Optional: compact CSV of history
    try:
        import io, csv
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator='\n')
        w.writerow(["generation", "score", "params", "method"])
        for h in history:
            w.writerow([h.get("generation"), h.get("score"), h.get("params"), h.get("method", "")])
        payload["csv_header"] = "generation,score,params,method"
        payload["csv_data"] = buf.getvalue().strip()
    except Exception:
        pass
    return payload
