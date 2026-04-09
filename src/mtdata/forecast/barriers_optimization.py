import math
import traceback
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Set, Tuple

import numpy as np

from ..shared.constants import TIMEFRAME_SECONDS
from ..shared.schema import DenoiseSpec, TimeframeLiteral
from ..utils.barriers import (
    get_pip_size as _get_pip_size,
)
from ..utils.barriers import (
    normalize_trade_direction,
)
from ..utils.utils import _UNPARSED_BOOL, _parse_bool_like
from ..utils.utils import parse_kv_or_json as _parse_kv_or_json
from .barrier_stats import (
    bootstrap_metric_uncertainty as _bootstrap_uncertainty,
)
from .barrier_stats import (
    cross_seed_stability as _cross_seed_stability,
)
from .barrier_stats import (
    mc_convergence_diagnostic as _mc_convergence,
)
from .barrier_stats import (
    minimum_simulations_for_ci_width as _min_sims_for_ci,
)
from .barrier_stats import (
    sensitivity_analysis_single_parameter as _sensitivity_analysis,
)
from .barrier_stats import (
    statistical_power_analysis as _power_analysis,
)
from .barriers_shared import (
    BARRIER_GRID_PRESETS,
    DEGENERATE_OBJECTIVE_MIN_RESOLVE,
    _annotate_candidate_metrics,
    _auto_barrier_method,
    _binomial_se,
    _binomial_wilson_95,
    _brownian_bridge_hits,
    _build_actionability_payload,
    _build_selection_diagnostics,
    _candidate_is_viable,
    _candidate_status_reason,
    _get_live_reference_price,
    _least_negative_ref,
    _resolve_reference_prices,
    _safe_float,
    _scale_price_paths_to_reference,
    _sort_candidate_results,
)
from .common import fetch_history as _fetch_history
from .common import log_returns_from_prices as _log_returns_from_prices
from .monte_carlo import (
    simulate_bootstrap_mc as _simulate_bootstrap_mc,
)
from .monte_carlo import (
    simulate_garch_mc as _simulate_garch_mc,
)
from .monte_carlo import (
    simulate_gbm_mc as _simulate_gbm_mc,
)
from .monte_carlo import (
    simulate_heston_mc as _simulate_heston_mc,
)
from .monte_carlo import (
    simulate_hmm_mc as _simulate_hmm_mc,
)
from .monte_carlo import (
    simulate_jump_diffusion_mc as _simulate_jump_diffusion_mc,
)

_BARRIER_SEARCH_PROFILE_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "fast": {
        "n_sims": 1200,
        "n_trials": 24,
        "tp_steps": 4,
        "sl_steps": 4,
        "ratio_steps": 4,
        "vol_steps": 4,
        "refine": False,
    },
    "medium": {
        "n_sims": 4000,
        "n_trials": 63,
        "tp_steps": 7,
        "sl_steps": 9,
        "ratio_steps": 8,
        "vol_steps": 7,
        "refine": False,
    },
    "long": {
        "n_sims": 10000,
        "n_trials": 600,
        "tp_steps": 41,
        "sl_steps": 51,
        "ratio_steps": 24,
        "vol_steps": 18,
        "refine": True,
    },
}


@dataclass(frozen=True)
class _BarrierEvaluationContext:
    mode_val: str
    dir_long: bool
    last_price: float
    pip_size: float
    rr_min_val: Optional[float]
    rr_max_val: Optional[float]
    has_trading_costs: bool
    ev_deduct_cost: float
    cost_per_trade: float
    min_prob_win_val: Optional[float]
    max_prob_no_hit_val: Optional[float]
    min_prob_resolve_val: Optional[float]
    max_median_time_val: Optional[float]


@dataclass(frozen=True)
class _BarrierBridgeInputs:
    enabled: bool
    sigma: float
    log_paths: Optional[np.ndarray]
    uniform_tp: Optional[np.ndarray]
    uniform_sl: Optional[np.ndarray]


def _coerce_barrier_bool_flag(value: Any, default: bool = False) -> bool:
    parsed = _parse_bool_like(value)
    if parsed is _UNPARSED_BOOL:
        return bool(default)
    return bool(parsed)


def _resolve_barrier_search_profile_config(
    params_dict: Dict[str, Any],
    *,
    search_profile: Any,
    fast_defaults: Any,
) -> Tuple[str, Dict[str, Any]]:
    search_profile_requested = str(
        params_dict.get("search_profile", params_dict.get("profile", search_profile))
    ).strip().lower()
    if search_profile_requested not in _BARRIER_SEARCH_PROFILE_DEFAULTS:
        search_profile_requested = "medium"
    fast_defaults_requested = _coerce_barrier_bool_flag(
        params_dict.get("fast_defaults", fast_defaults),
        default=bool(fast_defaults),
    )
    search_profile_val = "fast" if fast_defaults_requested else search_profile_requested
    return search_profile_val, dict(_BARRIER_SEARCH_PROFILE_DEFAULTS[search_profile_val])


def _resolve_profile_param(
    params_dict: Dict[str, Any],
    profile_cfg: Dict[str, Any],
    *,
    param_key: str,
    arg_value: Any,
) -> Any:
    if param_key in params_dict:
        return params_dict[param_key]
    if arg_value is not None:
        return arg_value
    return profile_cfg[param_key]


def _candidate_barrier_prices(
    tp_unit: float,
    sl_unit: float,
    *,
    context: _BarrierEvaluationContext,
) -> Tuple[float, float]:
    if context.mode_val == "pct":
        if context.dir_long:
            tp_price = context.last_price * (1.0 + tp_unit / 100.0)
            sl_price = context.last_price * (1.0 - sl_unit / 100.0)
        else:
            tp_price = context.last_price * (1.0 - tp_unit / 100.0)
            sl_price = context.last_price * (1.0 + sl_unit / 100.0)
    else:
        if context.dir_long:
            tp_price = context.last_price + tp_unit * context.pip_size
            sl_price = context.last_price - sl_unit * context.pip_size
        else:
            tp_price = context.last_price - tp_unit * context.pip_size
            sl_price = context.last_price + sl_unit * context.pip_size
    return float(tp_price), float(sl_price)


def _candidate_barrier_geometry_is_valid(
    tp_price: float,
    sl_price: float,
    *,
    context: _BarrierEvaluationContext,
) -> bool:
    if not np.isfinite(tp_price) or not np.isfinite(sl_price):
        return False
    if np.isfinite(context.last_price) and context.last_price > 0.0:
        if tp_price <= 0.0 or sl_price <= 0.0:
            return False
        if context.dir_long:
            return sl_price < context.last_price < tp_price
        return tp_price < context.last_price < sl_price
    return True


def _candidate_hit_arrays(
    eval_paths: np.ndarray,
    *,
    tp_trigger: float,
    sl_trigger: float,
    context: _BarrierEvaluationContext,
    bridge_inputs: _BarrierBridgeInputs,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    _, horizon_total = eval_paths.shape
    if context.dir_long:
        hit_tp = eval_paths >= tp_trigger
        hit_sl = eval_paths <= sl_trigger
    else:
        hit_tp = eval_paths <= tp_trigger
        hit_sl = eval_paths >= sl_trigger
    if (
        bridge_inputs.enabled
        and bridge_inputs.log_paths is not None
        and bridge_inputs.uniform_tp is not None
        and bridge_inputs.uniform_sl is not None
    ):
        tp_dir = "up" if context.dir_long else "down"
        sl_dir = "down" if context.dir_long else "up"
        tp_bridge = _brownian_bridge_hits(
            bridge_inputs.log_paths,
            float(np.log(max(1e-12, tp_trigger))),
            bridge_inputs.sigma,
            direction=tp_dir,
            uniform=bridge_inputs.uniform_tp,
        )
        sl_bridge = _brownian_bridge_hits(
            bridge_inputs.log_paths,
            float(np.log(max(1e-12, sl_trigger))),
            bridge_inputs.sigma,
            direction=sl_dir,
            uniform=bridge_inputs.uniform_sl,
        )
        hit_tp = hit_tp | tp_bridge
        hit_sl = hit_sl | sl_bridge
    any_tp = hit_tp.any(axis=1)
    any_sl = hit_sl.any(axis=1)
    first_tp = hit_tp.argmax(axis=1)
    first_sl = hit_sl.argmax(axis=1)
    first_tp[~any_tp] = horizon_total
    first_sl[~any_sl] = horizon_total
    wins = first_tp < first_sl
    losses = first_sl < first_tp
    ties = (first_tp == first_sl) & (first_tp < horizon_total)
    return first_tp, first_sl, wins, losses, ties


def _barrier_return_fractions(
    net_reward: float,
    net_risk: float,
    *,
    tp_price: float,
    sl_price: float,
    context: _BarrierEvaluationContext,
) -> Tuple[float, float]:
    if context.mode_val == "pct":
        reward_frac = net_reward / 100.0
        risk_frac = net_risk / 100.0
    elif context.last_price > 0 and context.pip_size:
        unit_to_return = float(context.pip_size) / float(context.last_price)
        reward_frac = net_reward * unit_to_return
        risk_frac = net_risk * unit_to_return
    elif context.last_price > 0:
        reward_frac = abs(tp_price - context.last_price) / context.last_price
        risk_frac = abs(sl_price - context.last_price) / context.last_price
    else:
        reward_frac = 0.0
        risk_frac = 0.0
    reward_frac = max(reward_frac, -0.999)
    if risk_frac >= 1.0:
        risk_frac = 0.999
    return reward_frac, risk_frac


def _evaluate_barrier_candidate(
    tp_unit: float,
    sl_unit: float,
    eval_paths: np.ndarray,
    *,
    context: _BarrierEvaluationContext,
    bridge_inputs: _BarrierBridgeInputs,
) -> Tuple[Optional[Dict[str, Any]], bool]:
    sims_total, horizon_total = eval_paths.shape
    tp_price, sl_price = _candidate_barrier_prices(tp_unit, sl_unit, context=context)
    if not _candidate_barrier_geometry_is_valid(tp_price, sl_price, context=context):
        return None, True

    first_tp, first_sl, wins, losses, ties = _candidate_hit_arrays(
        eval_paths,
        tp_trigger=tp_price,
        sl_trigger=sl_price,
        context=context,
        bridge_inputs=bridge_inputs,
    )
    n_wins = int(wins.sum())
    n_losses = int(losses.sum())
    n_ties = int(ties.sum())

    prob_win = n_wins / sims_total
    prob_loss = n_losses / sims_total
    prob_tie = n_ties / sims_total
    prob_neutral = max(0.0, 1.0 - prob_win - prob_loss - prob_tie)
    prob_resolve = 1.0 - prob_neutral
    prob_tp_first = (n_wins + 0.5 * n_ties) / sims_total
    prob_sl_first = (n_losses + 0.5 * n_ties) / sims_total
    effective_prob_win = prob_tp_first
    effective_prob_loss = prob_sl_first

    risk = sl_unit
    reward = tp_unit
    rr = reward / risk if risk > 0 else 0
    if context.rr_min_val and rr < context.rr_min_val:
        return None, False
    if context.rr_max_val and rr > context.rr_max_val:
        return None, False

    net_reward = reward - context.ev_deduct_cost if context.has_trading_costs else reward
    net_risk = risk + context.ev_deduct_cost if context.has_trading_costs else risk
    net_rr = net_reward / net_risk if net_risk > 0 else 0.0

    ev_gross = effective_prob_win * reward - effective_prob_loss * risk
    ev_val = (
        effective_prob_win * net_reward - effective_prob_loss * net_risk
        if context.has_trading_costs
        else ev_gross
    )
    edge = prob_win - prob_loss
    win_lo, win_hi = _binomial_wilson_95(prob_win, int(sims_total))
    loss_lo, loss_hi = _binomial_wilson_95(prob_loss, int(sims_total))
    tie_lo, tie_hi = _binomial_wilson_95(prob_tie, int(sims_total))
    no_hit_lo, no_hit_hi = _binomial_wilson_95(prob_neutral, int(sims_total))

    kelly_val = 0.0
    if net_rr > 0:
        kelly_val = effective_prob_win - (effective_prob_loss / net_rr)

    active = effective_prob_win + effective_prob_loss
    if active > 0:
        prob_win_c = effective_prob_win / active
        prob_loss_c = effective_prob_loss / active
        ev_cond = prob_win_c * net_reward - prob_loss_c * net_risk
        kelly_cond = prob_win_c - (prob_loss_c / net_rr if net_rr > 0 else 0.0)
    else:
        ev_cond = 0.0
        kelly_cond = 0.0

    resolve_mask = (first_tp < horizon_total) | (first_sl < horizon_total)
    time_in_trade = np.minimum(np.minimum(first_tp, first_sl) + 1, horizon_total)
    t_res_mean_all = float(np.mean(time_in_trade)) if time_in_trade.size else None
    t_res_med_all = float(np.median(time_in_trade)) if time_in_trade.size else None
    if np.any(resolve_mask):
        resolve_times = np.minimum(first_tp, first_sl)[resolve_mask] + 1
        t_res_mean = float(np.mean(resolve_times)) if resolve_times.size else None
        t_res_med = float(np.median(resolve_times)) if resolve_times.size else None
    else:
        t_res_mean = None
        t_res_med = None

    ev_per_bar = 0.0
    if t_res_mean_all and t_res_mean_all > 0:
        ev_per_bar = ev_val / t_res_mean_all

    profit_factor = 0.0
    denom = effective_prob_loss * net_risk
    if denom > 0:
        profit_factor = (effective_prob_win * net_reward) / denom
    elif effective_prob_win > 0 and net_reward > 0:
        profit_factor = 1e9

    reward_frac, risk_frac = _barrier_return_fractions(
        net_reward,
        net_risk,
        tp_price=tp_price,
        sl_price=sl_price,
        context=context,
    )
    utility_val = (
        (effective_prob_win * math.log1p(reward_frac))
        + (effective_prob_loss * math.log1p(-risk_frac))
    )

    if context.min_prob_win_val is not None and effective_prob_win < context.min_prob_win_val:
        return None, False
    if context.max_prob_no_hit_val is not None and prob_neutral > context.max_prob_no_hit_val:
        return None, False
    if context.min_prob_resolve_val is not None and prob_resolve < context.min_prob_resolve_val:
        return None, False
    if context.max_median_time_val is not None:
        if t_res_med is None or t_res_med > context.max_median_time_val:
            return None, False

    t_hit_tp = first_tp[wins | ties] + 1
    t_hit_sl = first_sl[losses | ties] + 1
    t_tp_med = float(np.median(t_hit_tp)) if t_hit_tp.size else None
    t_sl_med = float(np.median(t_hit_sl)) if t_hit_sl.size else None

    result = {
        "tp": tp_unit,
        "sl": sl_unit,
        "rr": rr,
        "tp_price": float(tp_price),
        "sl_price": float(sl_price),
        "prob_win": prob_win,
        "prob_loss": prob_loss,
        "prob_tp_first": prob_tp_first,
        "prob_sl_first": prob_sl_first,
        "prob_no_hit": prob_neutral,
        "prob_tie": prob_tie,
        "prob_win_se": _binomial_se(prob_win, int(sims_total)),
        "prob_loss_se": _binomial_se(prob_loss, int(sims_total)),
        "prob_tie_se": _binomial_se(prob_tie, int(sims_total)),
        "prob_no_hit_se": _binomial_se(prob_neutral, int(sims_total)),
        "prob_win_ci95": {"low": float(win_lo), "high": float(win_hi)},
        "prob_loss_ci95": {"low": float(loss_lo), "high": float(loss_hi)},
        "prob_tie_ci95": {"low": float(tie_lo), "high": float(tie_hi)},
        "prob_no_hit_ci95": {"low": float(no_hit_lo), "high": float(no_hit_hi)},
        "prob_resolve": prob_resolve,
        "ev": ev_val,
        "ev_gross": ev_gross if context.has_trading_costs else None,
        "ev_net": ev_val if context.has_trading_costs else None,
        "ev_cond": ev_cond,
        "edge": edge,
        "kelly": kelly_val,
        "kelly_cond": kelly_cond,
        "ev_per_bar": ev_per_bar,
        "profit_factor": profit_factor,
        "utility": utility_val,
        "t_hit_tp_median": t_tp_med,
        "t_hit_sl_median": t_sl_med,
        "t_hit_tp_median_cond": t_tp_med,
        "t_hit_sl_median_cond": t_sl_med,
        "t_hit_resolve_mean": t_res_mean,
        "t_hit_resolve_median": t_res_med,
        "t_hit_resolve_mean_all": t_res_mean_all,
        "t_hit_resolve_median_all": t_res_med_all,
    }
    _annotate_candidate_metrics(result, cost_per_trade=context.cost_per_trade)
    return result, False


def _evaluate_barrier_bucket(
    bucket: List[Tuple[float, float]],
    eval_paths: np.ndarray,
    *,
    context: _BarrierEvaluationContext,
    bridge_inputs: _BarrierBridgeInputs,
    count_invalid: bool = True,
) -> Tuple[List[Dict[str, Any]], int]:
    rows: List[Dict[str, Any]] = []
    invalid_candidates = 0
    for tp_unit, sl_unit in bucket:
        row, is_invalid = _evaluate_barrier_candidate(
            tp_unit,
            sl_unit,
            eval_paths,
            context=context,
            bridge_inputs=bridge_inputs,
        )
        if row is not None:
            rows.append(row)
        elif count_invalid and is_invalid:
            invalid_candidates += 1
    return rows, invalid_candidates


def _rounded_ranked_barrier_value(value: Any, decimals: int = 6) -> Any:
    try:
        if value is None:
            return None
        num = float(value)
        if not np.isfinite(num):
            return str(value)
        return round(num, decimals)
    except Exception:
        return value


def _dedupe_ranked_barrier_candidates(
    ranked_candidates: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    deduped_ranked: List[Dict[str, Any]] = []
    seen_ranked: Set[Tuple[Any, ...]] = set()
    for row in ranked_candidates:
        if not isinstance(row, dict):
            continue

        row_key = (
            _rounded_ranked_barrier_value(row.get("tp"), 6),
            _rounded_ranked_barrier_value(row.get("sl"), 6),
            _rounded_ranked_barrier_value(row.get("tp_price"), 6),
            _rounded_ranked_barrier_value(row.get("sl_price"), 6),
            _rounded_ranked_barrier_value(row.get("ev"), 6),
            _rounded_ranked_barrier_value(row.get("edge"), 6),
            _rounded_ranked_barrier_value(row.get("kelly"), 6),
            _rounded_ranked_barrier_value(row.get("prob_tp_first"), 6),
            _rounded_ranked_barrier_value(row.get("prob_sl_first"), 6),
            _rounded_ranked_barrier_value(row.get("prob_no_hit"), 6),
        )
        if row_key in seen_ranked:
            continue
        seen_ranked.add(row_key)
        deduped_ranked.append(row)
    return deduped_ranked


def _select_barrier_candidate_views(
    ranked_candidates: List[Dict[str, Any]],
    *,
    cost_per_trade: float,
    viable_only_val: bool,
    concise_val: bool,
    top_k_val: Optional[int],
    return_grid: bool,
    output_mode: str,
) -> Dict[str, Any]:
    ranked_candidates = _dedupe_ranked_barrier_candidates(ranked_candidates)
    viable_candidates = [
        row
        for row in ranked_candidates
        if _candidate_is_viable(row, cost_per_trade=cost_per_trade)
    ]

    candidates = viable_candidates if viable_only_val else ranked_candidates
    if top_k_val is not None:
        candidates = candidates[:top_k_val]
    elif concise_val and not viable_candidates and len(candidates) > 5:
        candidates = candidates[:5]

    grid_out = candidates if (return_grid and not concise_val) else None
    if output_mode == "summary" and grid_out is not None:
        limit = top_k_val or min(10, len(grid_out))
        grid_out = grid_out[:limit]

    results_limit = min(10, len(candidates))
    if output_mode == "summary":
        if top_k_val is not None:
            results_limit = top_k_val
        elif concise_val:
            results_limit = min(5, len(candidates))
        else:
            results_limit = min(10, len(candidates))
    summary_results = candidates[:results_limit]

    viability_filtered_out = bool(viable_only_val and not viable_candidates and ranked_candidates)
    warning = None
    if not candidates:
        if viability_filtered_out:
            warning = "No viable TP/SL candidates satisfied the viability filter."
        else:
            warning = "No valid TP/SL candidates after applying grid generation and constraints."

    return {
        "ranked_candidates": ranked_candidates,
        "viable_candidates": viable_candidates,
        "candidates": candidates,
        "grid_out": grid_out,
        "summary_results": summary_results,
        "viability_filtered_out": viability_filtered_out,
        "warning": warning,
    }


def forecast_barrier_optimize(  # noqa: C901
    symbol: str,
    timeframe: TimeframeLiteral = "H1",
    horizon: int = 12,
    method: Literal['mc_gbm','mc_gbm_bb','hmm_mc','garch','bootstrap','heston','jump_diffusion','auto'] = 'hmm_mc',
    direction: Literal['long','short'] = 'long',
    mode: Literal['pct','pips'] = 'pct',
    tp_min: float = 0.25,
    tp_max: float = 1.5,
    tp_steps: Optional[int] = None,
    sl_min: float = 0.25,
    sl_max: float = 2.5,
    sl_steps: Optional[int] = None,
    params: Optional[Dict[str, Any]] = None,
    denoise: Optional[DenoiseSpec] = None,
    objective: Literal[
        'edge',
        'prob_tp_first',
        'prob_resolve',
        'kelly',
        'kelly_cond',
        'ev',
        'ev_cond',
        'ev_per_bar',
        'profit_factor',
        'min_loss_prob',
        'utility',
    ] = 'ev',
    return_grid: bool = True,
    top_k: Optional[int] = None,
    output: Literal['full','summary'] = 'summary',
    viable_only: bool = False,
    concise: bool = False,
    grid_style: Literal['fixed','volatility','ratio','preset'] = 'fixed',
    preset: Optional[str] = None,
    vol_window: int = 250,
    vol_min_mult: float = 0.5,
    vol_max_mult: float = 4.0,
    vol_steps: Optional[int] = None,
    vol_sl_extra: float = 1.8,
    vol_floor_pct: float = 0.15,
    vol_floor_pips: float = 8.0,
    ratio_min: float = 0.5,
    ratio_max: float = 4.0,
    ratio_steps: Optional[int] = None,
    refine: Optional[bool] = None,
    refine_radius: float = 0.3,
    refine_steps: int = 5,
    min_prob_win: Optional[float] = None,
    max_prob_no_hit: Optional[float] = None,
    max_median_time: Optional[float] = None,
    fast_defaults: bool = False,
    search_profile: Literal['fast', 'medium', 'long'] = 'medium',
    statistical_robustness: bool = False,
    target_ci_width: float = 0.05,
    n_seeds_stability: int = 3,
    enable_bootstrap: bool = False,
    n_bootstrap: int = 200,
    enable_convergence_check: bool = True,
    convergence_window: int = 100,
    convergence_threshold: float = 0.01,
    enable_power_analysis: bool = False,
    power_effect_size: float = 0.05,
    enable_sensitivity_analysis: bool = False,
    sensitivity_params: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Optimize TP/SL barriers over a grid of candidate levels.

    Unit conventions:
    - mode="pct": tp/sl are percentage *points* (e.g., tp=0.5 means +0.5%).
    - mode="pips": tp/sl are ticks (trade_tick_size units).

    Grid styles:
    - fixed/volatility generate tp/sl directly in the selected `mode`.
    - preset ranges are defined in pct terms and converted when `mode="pips"`.
    - ratio treats `ratio_min/max` as reward/risk = tp/sl (TP distance divided
      by SL distance), with SL sampled from `sl_min/max`.

    Metrics:
    - ev/ev_cond/ev_per_bar are reported in the same units as tp/sl (pct points
      or ticks). `ev_per_bar` divides by unconditional mean time-in-trade
      (bars), counting unresolved paths at horizon expiry.
    
    Statistical Robustness:
    - statistical_robustness: Enable comprehensive statistical analysis
    - target_ci_width: Target confidence interval width (default 0.05 = ±2.5%)
    - n_seeds_stability: Number of seeds for cross-seed stability analysis
    - enable_bootstrap: Enable bootstrap uncertainty estimation
    - n_bootstrap: Number of bootstrap samples
    - enable_convergence_check: Check MC convergence diagnostics
    - convergence_window: Window size for convergence check
    - convergence_threshold: Convergence threshold for probability change
    - enable_power_analysis: Statistical power analysis for probabilities
    - power_effect_size: Minimum detectable effect size for power analysis
    - enable_sensitivity_analysis: Sensitivity analysis for barrier parameters
    - sensitivity_params: List of parameters to analyze (default: ['tp', 'sl'])
    """
    try:
        if timeframe not in TIMEFRAME_SECONDS:
            return {"error": f"Invalid timeframe: {timeframe}"}
        try:
            horizon_val = int(horizon)
        except Exception:
            return {"error": f"Invalid horizon: {horizon}. Must be a positive integer."}
        if horizon_val <= 0:
            return {"error": f"Invalid horizon: {horizon_val}. Must be >= 1."}
        direction_norm, direction_error = normalize_trade_direction(direction)
        if direction_error:
            return {"error": direction_error}

        params_dict = _parse_kv_or_json(params)
        mode_val = str(mode).lower()
        if mode_val not in {'pct', 'pips'}:
            return {"error": f"Invalid mode: {mode}. Use 'pct' or 'pips'."}
        output_mode = str(output).strip().lower()
        if output_mode not in {'full', 'summary'}:
            output_mode = 'summary'

        search_profile_val, profile_cfg = _resolve_barrier_search_profile_config(
            params_dict,
            search_profile=search_profile,
            fast_defaults=fast_defaults,
        )

        viable_only_val = _coerce_barrier_bool_flag(
            params_dict.get('viable_only', viable_only),
            default=bool(viable_only),
        )
        concise_val = _coerce_barrier_bool_flag(
            params_dict.get('concise', concise),
            default=bool(concise),
        )
        if concise_val:
            output_mode = 'summary'
        objective_val = str(objective).lower()
        objective_requested = objective_val
        valid_objectives = {
            'edge',
            'prob_tp_first',
            'prob_resolve',
            'kelly',
            'kelly_cond',
            'ev',
            'ev_cond',
            'ev_per_bar',
            'profit_factor',
            'min_loss_prob',
            'utility',
        }
        if objective_val not in valid_objectives:
            objective_val = 'ev'
        objective_changed = objective_val != objective_requested

        optimizer_val = str(params_dict.get('optimizer', 'grid')).strip().lower()
        if optimizer_val not in {'grid', 'optuna'}:
            optimizer_val = 'grid'
        optuna_default_trials = int(profile_cfg['n_trials'])
        optuna_trials_val = max(1, int(params_dict.get('n_trials', optuna_default_trials)))
        optuna_timeout_raw = params_dict.get('timeout')
        try:
            optuna_timeout_val = float(optuna_timeout_raw) if optuna_timeout_raw is not None else None
            if optuna_timeout_val is not None and optuna_timeout_val <= 0:
                optuna_timeout_val = None
        except Exception:
            optuna_timeout_val = None
        optuna_n_jobs_val = max(1, int(params_dict.get('n_jobs', 1)))
        optuna_sampler_val = str(params_dict.get('sampler', 'tpe')).strip().lower()
        if optuna_sampler_val not in {'tpe', 'random', 'cmaes'}:
            optuna_sampler_val = 'tpe'
        optuna_pruner_val = str(params_dict.get('pruner', 'median')).strip().lower()
        if optuna_pruner_val not in {'median', 'none', 'hyperband', 'percentile'}:
            optuna_pruner_val = 'median'
        optuna_pareto_val = _coerce_barrier_bool_flag(params_dict.get('optuna_pareto', False), default=False)
        try:
            pareto_limit_val = int(params_dict.get('pareto_limit', 20))
        except Exception:
            pareto_limit_val = 20
        if pareto_limit_val <= 0:
            pareto_limit_val = 20
        optuna_pareto_objectives_raw = params_dict.get('optuna_pareto_objectives')

        def _normalize_optuna_direction(value: Any, default: str = 'maximize') -> str:
            v = str(value or default).strip().lower()
            if v in {'max', 'maximize', 'maximise'}:
                return 'maximize'
            if v in {'min', 'minimize', 'minimise'}:
                return 'minimize'
            return str(default)

        pareto_objectives: List[Tuple[str, str]] = [
            ('ev', 'maximize'),
            ('prob_loss', 'minimize'),
            ('t_hit_resolve_median', 'minimize'),
        ]
        if isinstance(optuna_pareto_objectives_raw, dict) and optuna_pareto_objectives_raw:
            tmp: List[Tuple[str, str]] = []
            for mk, mv in optuna_pareto_objectives_raw.items():
                metric_name = str(mk).strip()
                if not metric_name:
                    continue
                tmp.append((metric_name, _normalize_optuna_direction(mv, default='maximize')))
            if tmp:
                pareto_objectives = tmp

        if top_k is not None:
            try:
                top_k_val = int(top_k)
            except Exception:
                return {"error": f"Invalid top_k: {top_k}. Must be a positive integer."}
            if top_k_val <= 0:
                return {"error": f"Invalid top_k: {top_k_val}. Must be >= 1."}
        else:
            top_k_val = None

        grid_style_val = str(params_dict.get('grid_style', grid_style)).lower()
        if grid_style_val not in {'fixed', 'volatility', 'ratio', 'preset'}:
            grid_style_val = 'fixed'
        preset_candidate = params_dict.get('grid_preset', params_dict.get('preset', preset))
        preset_val = str(preset_candidate).lower() if isinstance(preset_candidate, str) and preset_candidate else None

        refine_default = _resolve_profile_param(
            params_dict,
            profile_cfg,
            param_key='refine',
            arg_value=refine,
        )
        refine_flag = _coerce_barrier_bool_flag(
            params_dict.get('refine', refine_default),
            default=bool(refine_default),
        )
        refine_radius_val = max(0.0, float(params_dict.get('refine_radius', refine_radius)))
        refine_steps_val = max(2, int(params_dict.get('refine_steps', refine_steps)))

        ratio_min_val = float(params_dict.get('ratio_min', ratio_min))
        ratio_max_val = float(params_dict.get('ratio_max', ratio_max))
        ratio_steps_default = int(
            _resolve_profile_param(
                params_dict,
                profile_cfg,
                param_key='ratio_steps',
                arg_value=ratio_steps,
            )
        )
        ratio_steps_val = max(2, int(params_dict.get('ratio_steps', ratio_steps_default)))
        if ratio_min_val <= 0:
            ratio_min_val = ratio_min
        if ratio_max_val < ratio_min_val:
            ratio_max_val = ratio_min_val

        vol_window_val = int(params_dict.get('vol_window', vol_window))
        vol_min_mult_val = float(params_dict.get('vol_min_mult', vol_min_mult))
        vol_max_mult_val = float(params_dict.get('vol_max_mult', vol_max_mult))
        vol_steps_default = int(
            _resolve_profile_param(
                params_dict,
                profile_cfg,
                param_key='vol_steps',
                arg_value=vol_steps,
            )
        )
        vol_steps_val = max(2, int(params_dict.get('vol_steps', vol_steps_default)))
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

        min_prob_win_val = params_dict.get('min_prob_win', min_prob_win)
        max_prob_no_hit_val = params_dict.get('max_prob_no_hit', max_prob_no_hit)
        max_median_time_val = params_dict.get('max_median_time', max_median_time)
        min_prob_resolve_val = params_dict.get('min_prob_resolve')
        try:
            min_prob_win_val = float(min_prob_win_val) if min_prob_win_val is not None else None
        except Exception:
            min_prob_win_val = None
        try:
            max_prob_no_hit_val = float(max_prob_no_hit_val) if max_prob_no_hit_val is not None else None
        except Exception:
            max_prob_no_hit_val = None
        try:
            max_median_time_val = float(max_median_time_val) if max_median_time_val is not None else None
        except Exception:
            max_median_time_val = None
        try:
            min_prob_resolve_val = float(min_prob_resolve_val) if min_prob_resolve_val is not None else None
        except Exception:
            min_prob_resolve_val = None
        if min_prob_win_val is not None:
            if not np.isfinite(min_prob_win_val):
                min_prob_win_val = None
            else:
                min_prob_win_val = max(0.0, min(1.0, min_prob_win_val))
        if max_prob_no_hit_val is not None:
            if not np.isfinite(max_prob_no_hit_val):
                max_prob_no_hit_val = None
            else:
                max_prob_no_hit_val = max(0.0, min(1.0, max_prob_no_hit_val))
        if max_median_time_val is not None:
            if not np.isfinite(max_median_time_val) or max_median_time_val <= 0:
                max_median_time_val = None
        if min_prob_resolve_val is not None:
            if not np.isfinite(min_prob_resolve_val):
                min_prob_resolve_val = None
            else:
                min_prob_resolve_val = max(0.0, min(1.0, min_prob_resolve_val))
        elif objective_val in {'profit_factor', 'min_loss_prob'}:
            min_prob_resolve_val = DEGENERATE_OBJECTIVE_MIN_RESOLVE

        tp_min_val = float(params_dict.get('tp_min', tp_min))
        tp_max_val = float(params_dict.get('tp_max', tp_max))
        tp_steps_default = int(
            _resolve_profile_param(
                params_dict,
                profile_cfg,
                param_key='tp_steps',
                arg_value=tp_steps,
            )
        )
        tp_steps_val = max(1, int(params_dict.get('tp_steps', tp_steps_default)))
        sl_min_val = float(params_dict.get('sl_min', sl_min))
        sl_max_val = float(params_dict.get('sl_max', sl_max))
        sl_steps_default = int(
            _resolve_profile_param(
                params_dict,
                profile_cfg,
                param_key='sl_steps',
                arg_value=sl_steps,
            )
        )
        sl_steps_val = max(1, int(params_dict.get('sl_steps', sl_steps_default)))
        
        statistical_robustness_requested = _coerce_barrier_bool_flag(
            params_dict.get('statistical_robustness', statistical_robustness),
            default=bool(statistical_robustness),
        )
        target_ci_width_val = float(params_dict.get('target_ci_width', target_ci_width))
        if not 0 < target_ci_width_val < 1:
            target_ci_width_val = 0.05
        n_seeds_stability_val = max(2, int(params_dict.get('n_seeds_stability', n_seeds_stability)))
        enable_bootstrap_val = _coerce_barrier_bool_flag(
            params_dict.get('enable_bootstrap', enable_bootstrap),
            default=bool(enable_bootstrap),
        )
        n_bootstrap_val = max(50, int(params_dict.get('n_bootstrap', n_bootstrap)))
        enable_convergence_check_val = _coerce_barrier_bool_flag(
            params_dict.get('enable_convergence_check', enable_convergence_check),
            default=bool(enable_convergence_check),
        )
        convergence_window_val = max(10, int(params_dict.get('convergence_window', convergence_window)))
        convergence_threshold_val = float(params_dict.get('convergence_threshold', convergence_threshold))
        if convergence_threshold_val <= 0:
            convergence_threshold_val = 0.01
        enable_power_analysis_val = _coerce_barrier_bool_flag(
            params_dict.get('enable_power_analysis', enable_power_analysis),
            default=bool(enable_power_analysis),
        )
        power_effect_size_val = float(params_dict.get('power_effect_size', power_effect_size))
        if power_effect_size_val <= 0:
            power_effect_size_val = 0.05
        enable_sensitivity_analysis_val = _coerce_barrier_bool_flag(
            params_dict.get('enable_sensitivity_analysis', enable_sensitivity_analysis),
            default=bool(enable_sensitivity_analysis),
        )
        sensitivity_params_requested = params_dict.get('sensitivity_params', sensitivity_params)
        if isinstance(sensitivity_params_requested, str):
            sensitivity_params_requested = [
                p.strip().lower() for p in sensitivity_params_requested.split(',') if p.strip()
            ]
        elif not isinstance(sensitivity_params_requested, list):
            sensitivity_params_requested = ['tp', 'sl']
        else:
            sensitivity_params_requested = [
                str(p).strip().lower() for p in sensitivity_params_requested if str(p).strip()
            ]

        need = int(max(2000, horizon_val + 100))
        df = _fetch_history(symbol, timeframe, need, as_of=None)
        if len(df) < 10:
            return {"error": "Insufficient history for simulation"}
        use_live_price_raw = params_dict.get('use_live_price', params_dict.get('live_price', True))
        if isinstance(use_live_price_raw, str):
            use_live_price = use_live_price_raw.strip().lower() not in {"0", "false", "no", "off"}
        else:
            use_live_price = bool(use_live_price_raw)
        last_price_close, last_price, last_price_source, _price_warning, price_error = _resolve_reference_prices(
            df['close'].astype(float).to_numpy(),
            symbol=symbol,
            direction=direction_norm,
            use_live_price=use_live_price,
            live_price_getter=_get_live_reference_price,
        )
        if price_error:
            return {"error": price_error}

        pip_size = _get_pip_size(symbol)
        if mode_val == 'pips' and (pip_size is None or pip_size <= 0):
            return {"error": "Tick size unavailable for this symbol; use mode='pct' or provide absolute barriers."}

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

        sims_default = int(profile_cfg['n_sims'])
        sims = int(params_dict.get('n_sims', params_dict.get('sims', sims_default)) or sims_default)
        if sims <= 0:
            return {"error": f"Invalid n_sims: {sims}. Must be >= 1."}
        seed_raw = params_dict.get('seed')
        seed = int(seed_raw) if seed_raw is not None else None
        request_seed_base = (
            int(seed)
            if seed is not None
            else int(np.random.default_rng().integers(0, np.iinfo(np.int32).max))
        )
        optuna_seed = int(request_seed_base)
        n_seeds = int(params_dict.get('n_seeds', 1) or 1)
        if n_seeds <= 0:
            return {"error": f"Invalid n_seeds: {n_seeds}. Must be >= 1."}
        
        if statistical_robustness_requested:
            min_sims_recommended = _min_sims_for_ci(
                target_width=target_ci_width_val,
                expected_prob=0.5,
                confidence=0.95,
                conservative=True,
            )
            if sims < min_sims_recommended:
                sims = min_sims_recommended

        spread_pips_val = float(params_dict.get('spread_pips', 0.0) or 0.0)
        spread_pct_val = float(params_dict.get('spread_pct', 0.0) or 0.0)
        commission_pct_val = float(params_dict.get('commission_pct', 0.0) or 0.0)
        slippage_pips_val = float(params_dict.get('slippage_pips', 0.0) or 0.0)
        slippage_pct_val = float(params_dict.get('slippage_pct', 0.0) or 0.0)

        if mode_val == 'pct':
            # pips → pct points:  pips * pip_size / price * 100
            pip_to_pct = (float(pip_size) / last_price * 100.0) if (pip_size and last_price > 0) else 0.0
            cost_spread = spread_pct_val + spread_pips_val * pip_to_pct
            cost_slippage = slippage_pct_val + slippage_pips_val * pip_to_pct
            cost_commission = commission_pct_val
        else:
            # pct → pips:  pct / 100 * price / pip_size
            pct_to_pips = (last_price / float(pip_size) / 100.0) if (pip_size and pip_size > 0 and last_price > 0) else 0.0
            cost_spread = spread_pips_val + spread_pct_val * pct_to_pips
            cost_slippage = slippage_pips_val + slippage_pct_val * pct_to_pips
            cost_commission = commission_pct_val * pct_to_pips
        dir_long = (direction_norm == 'long')

        # Costs are applied symmetrically to net payoffs for both directions.
        ev_deduct_cost = max(0.0, cost_spread + cost_slippage + cost_commission)

        # Keep old cost_per_trade for backwards compatibility in outputs
        cost_per_trade = max(0.0, cost_spread + cost_slippage + cost_commission)
        has_trading_costs = cost_per_trade > 0.0

        # Minimum barrier constraints
        min_barrier_multiplier = float(params_dict.get('min_barrier_multiplier', 2.0) if params_dict.get('min_barrier_multiplier') is not None else 2.0)
        if mode_val == 'pct':
            min_barrier_absolute = float(params_dict.get('min_barrier_pct', 0.0) or 0.0)
        else:
            min_barrier_absolute = float(params_dict.get('min_barrier_pips', 0.0) or 0.0)
        min_barrier_distance = max(min_barrier_absolute, min_barrier_multiplier * cost_spread)
        
        method_name = str(method).lower().strip()
        method_requested = method_name
        auto_reason = None
        supported_member_methods = ['mc_gbm', 'mc_gbm_bb', 'hmm_mc', 'garch', 'bootstrap', 'heston', 'jump_diffusion', 'auto']

        if method_name == 'ensemble':
            ensemble_methods_raw = params_dict.get('ensemble_methods', ['hmm_mc', 'garch', 'heston', 'jump_diffusion'])
            ensemble_methods: List[str] = []
            if isinstance(ensemble_methods_raw, str):
                ensemble_methods = [p.strip().lower() for p in ensemble_methods_raw.split(',') if p.strip()]
            elif isinstance(ensemble_methods_raw, (list, tuple)):
                for item in ensemble_methods_raw:
                    if isinstance(item, str) and item.strip():
                        ensemble_methods.append(item.strip().lower())
            if not ensemble_methods:
                ensemble_methods = ['hmm_mc', 'garch', 'heston', 'jump_diffusion']
            dedup_members: List[str] = []
            seen_members: Set[str] = set()
            for member_name in ensemble_methods:
                if member_name == 'ensemble':
                    continue
                if member_name not in supported_member_methods:
                    continue
                if member_name in seen_members:
                    continue
                seen_members.add(member_name)
                dedup_members.append(member_name)
            ensemble_methods = dedup_members
            if not ensemble_methods:
                return {"error": "Ensemble requires at least one valid member method."}

            ensemble_agg = str(params_dict.get('ensemble_agg', 'median')).strip().lower()
            if ensemble_agg not in {'median', 'weighted_mean'}:
                ensemble_agg = 'median'

            weight_map_raw = params_dict.get('ensemble_weights')
            ensemble_weight_map: Dict[str, float] = {}
            if isinstance(weight_map_raw, dict):
                for mk, mv in weight_map_raw.items():
                    try:
                        w = float(mv)
                    except Exception:
                        continue
                    if not np.isfinite(w) or w <= 0:
                        continue
                    ensemble_weight_map[str(mk).strip().lower()] = float(w)

            member_params = dict(params_dict)
            for extra_key in (
                'ensemble_methods',
                'ensemble_agg',
                'ensemble_weights',
                'ensemble_top_k',
                'ensemble_vote_metric',
            ):
                member_params.pop(extra_key, None)

            member_runs: List[Dict[str, Any]] = []
            member_errors: List[Dict[str, Any]] = []
            for member_method in ensemble_methods:
                member_out = forecast_barrier_optimize(
                    symbol=symbol,
                    timeframe=timeframe,
                    horizon=horizon_val,
                    method=member_method,
                    direction=direction_norm,  # type: ignore[arg-type]
                    mode=mode_val,  # type: ignore[arg-type]
                    tp_min=tp_min_val,
                    tp_max=tp_max_val,
                    tp_steps=tp_steps_val,
                    sl_min=sl_min_val,
                    sl_max=sl_max_val,
                    sl_steps=sl_steps_val,
                    params=member_params,
                    denoise=denoise,
                    objective=objective_val,  # type: ignore[arg-type]
                    return_grid=False,
                    top_k=1,
                    output='summary',  # type: ignore[arg-type]
                    viable_only=viable_only_val,
                    concise=concise_val,
                    grid_style=grid_style_val,  # type: ignore[arg-type]
                    preset=preset_val,
                    vol_window=vol_window_val,
                    vol_min_mult=vol_min_mult_val,
                    vol_max_mult=vol_max_mult_val,
                    vol_steps=vol_steps_val,
                    vol_sl_extra=vol_sl_extra_val,
                    vol_floor_pct=vol_floor_pct_val,
                    vol_floor_pips=vol_floor_pips_val,
                    ratio_min=ratio_min_val,
                    ratio_max=ratio_max_val,
                    ratio_steps=ratio_steps_val,
                    refine=refine_flag,
                    refine_radius=refine_radius_val,
                    refine_steps=refine_steps_val,
                    min_prob_win=min_prob_win_val,
                    max_prob_no_hit=max_prob_no_hit_val,
                    max_median_time=max_median_time_val,
                    fast_defaults=bool(search_profile_val == 'fast'),
                    search_profile=search_profile_val,  # type: ignore[arg-type]
                    statistical_robustness=bool(statistical_robustness_requested),
                    target_ci_width=target_ci_width_val,
                    n_seeds_stability=n_seeds_stability_val,
                    enable_bootstrap=bool(enable_bootstrap_val),
                    n_bootstrap=n_bootstrap_val,
                    enable_convergence_check=bool(enable_convergence_check_val),
                    convergence_window=convergence_window_val,
                    convergence_threshold=convergence_threshold_val,
                    enable_power_analysis=bool(enable_power_analysis_val),
                    power_effect_size=power_effect_size_val,
                    enable_sensitivity_analysis=bool(enable_sensitivity_analysis_val),
                    sensitivity_params=sensitivity_params_requested,
                )
                if not isinstance(member_out, dict) or not member_out.get('success'):
                    err_msg = None
                    if isinstance(member_out, dict):
                        err_msg = member_out.get('error')
                    if err_msg is None:
                        err_msg = f"Member method {member_method} failed"
                    member_errors.append({"method": member_method, "error": str(err_msg)})
                    continue
                best_row = member_out.get('best')
                if not isinstance(best_row, dict):
                    member_errors.append({"method": member_method, "error": "No best candidate returned"})
                    continue
                actual_best = dict(best_row)
                actual_best["member_method"] = str(member_method)
                actual_best["member_method_used"] = str(member_out.get('method', member_method))
                _annotate_candidate_metrics(actual_best, cost_per_trade=cost_per_trade)
                member_runs.append({
                    "method": member_method,
                    "method_used": member_out.get('method', member_method),
                    "best": actual_best,
                    "output": member_out,
                })

            if not member_runs:
                return {"error": "Ensemble failed: no successful member methods.", "member_errors": member_errors}

            n_total = len(ensemble_methods)
            n_succeeded = len(member_runs)
            n_failed = len(member_errors)
            ensemble_degraded = n_failed > n_total / 2
            ensemble_confidence = "high" if n_failed == 0 else ("medium" if n_succeeded > n_failed else "low")

            metric_keys = [
                'tp', 'sl', 'rr', 'tp_price', 'sl_price',
                'prob_win', 'prob_loss', 'prob_tp_first', 'prob_sl_first',
                'prob_no_hit', 'prob_tie', 'prob_resolve',
                'ev', 'ev_cond', 'edge', 'breakeven_win_rate', 'edge_vs_breakeven',
                'kelly', 'kelly_cond',
                'ev_per_bar', 'profit_factor', 'utility',
                't_hit_tp_median', 't_hit_sl_median',
                't_hit_resolve_mean', 't_hit_resolve_median',
            ]

            def _member_weight(row: Dict[str, Any]) -> float:
                member_key = str(row.get('method', '')).strip().lower()
                if member_key in ensemble_weight_map:
                    return float(ensemble_weight_map[member_key])
                return 1.0

            def _agg_metric(metric_name: str) -> Optional[float]:
                vals: List[float] = []
                wts: List[float] = []
                for row in member_runs:
                    best_row = row.get('best', {})
                    if not isinstance(best_row, dict):
                        continue
                    raw = best_row.get(metric_name)
                    try:
                        val = float(raw)
                    except Exception:
                        continue
                    if not np.isfinite(val):
                        continue
                    vals.append(float(val))
                    wts.append(_member_weight(row))
                if not vals:
                    return None
                if ensemble_agg == 'weighted_mean':
                    sw = float(sum(wts))
                    if sw > 0:
                        return float(sum(v * w for v, w in zip(vals, wts)) / sw)
                    return float(np.mean(np.asarray(vals, dtype=float)))
                return float(np.median(np.asarray(vals, dtype=float)))

            aggregate_metrics: Dict[str, Any] = {}
            for metric_name in metric_keys:
                val = _agg_metric(metric_name)
                if val is not None:
                    aggregate_metrics[metric_name] = val
            if 'rr' not in aggregate_metrics and aggregate_metrics.get('tp') and aggregate_metrics.get('sl'):
                try:
                    tp_val = float(aggregate_metrics['tp'])
                    sl_val = float(aggregate_metrics['sl'])
                    if sl_val > 0:
                        aggregate_metrics['rr'] = float(tp_val / sl_val)
                except Exception:
                    pass
            if 'prob_resolve' not in aggregate_metrics and aggregate_metrics.get('prob_no_hit') is not None:
                try:
                    aggregate_metrics['prob_resolve'] = float(1.0 - float(aggregate_metrics['prob_no_hit']))
                except Exception:
                    pass
            _annotate_candidate_metrics(aggregate_metrics, cost_per_trade=cost_per_trade)

            ranked_candidates = [
                dict(row.get('best', {}))
                for row in member_runs
                if isinstance(row.get('best'), dict)
            ]
            _sort_candidate_results(ranked_candidates, objective_val)
            viable_candidates = [
                row for row in ranked_candidates
                if _candidate_is_viable(row, cost_per_trade=cost_per_trade)
            ]
            if viable_only_val:
                candidates = viable_candidates
            else:
                candidates = ranked_candidates
            if top_k_val is not None:
                candidates = candidates[:top_k_val]
            elif concise_val and not viable_candidates and len(candidates) > 5:
                candidates = candidates[:5]

            grid_out = candidates if (return_grid and not concise_val) else None
            if output_mode == 'summary' and grid_out is not None:
                limit = top_k_val or min(10, len(grid_out))
                grid_out = grid_out[:limit]

            results_limit = min(10, len(candidates))
            if output_mode == 'summary':
                if top_k_val is not None:
                    results_limit = top_k_val
                elif concise_val:
                    results_limit = min(5, len(candidates))
                else:
                    results_limit = min(10, len(candidates))
            summary_results = candidates[:results_limit]

            member_prices = [
                float(r.get('output', {}).get('last_price'))
                for r in member_runs
                if isinstance(r.get('output', {}).get('last_price'), (int, float))
            ]
            member_close_prices = [
                float(r.get('output', {}).get('last_price_close'))
                for r in member_runs
                if isinstance(r.get('output', {}).get('last_price_close'), (int, float))
            ]
            out_last_price = float(np.median(np.asarray(member_prices, dtype=float))) if member_prices else float(last_price)
            out_last_price_close = (
                float(np.median(np.asarray(member_close_prices, dtype=float)))
                if member_close_prices else float(last_price_close)
            )

            viability_filtered_out = bool(viable_only_val and not viable_candidates and ranked_candidates)
            selected_best = candidates[0] if candidates else None
            if isinstance(selected_best, dict):
                _annotate_candidate_metrics(selected_best, cost_per_trade=cost_per_trade)
            viable = _candidate_is_viable(selected_best, cost_per_trade=cost_per_trade)
            viable_results_total = int(len(viable_candidates))
            status = "ok" if viable else ("non_viable" if viability_filtered_out else ("no_candidates" if not selected_best else "non_viable"))
            status_reason = None
            if viability_filtered_out:
                status_reason = "No viable ensemble candidate satisfied the viability filter."
            elif status == "no_candidates":
                status_reason = "No valid ensemble candidate was produced."
            elif status == "non_viable":
                status_reason = _candidate_status_reason(
                    selected_best,
                    cost_per_trade=cost_per_trade,
                )

            member_summaries: List[Dict[str, Any]] = []
            for row in member_runs:
                best_row = row.get('best', {})
                member_method = row.get('method')
                member_summaries.append({
                    "method": member_method,
                    "method_used": row.get('method_used'),
                    "ev": best_row.get('ev') if isinstance(best_row, dict) else None,
                    "ev_per_bar": best_row.get('ev_per_bar') if isinstance(best_row, dict) else None,
                    "prob_win": best_row.get('prob_win') if isinstance(best_row, dict) else None,
                    "prob_loss": best_row.get('prob_loss') if isinstance(best_row, dict) else None,
                    "prob_no_hit": best_row.get('prob_no_hit') if isinstance(best_row, dict) else None,
                    "rr": best_row.get('rr') if isinstance(best_row, dict) else None,
                    "edge_vs_breakeven": best_row.get('edge_vs_breakeven') if isinstance(best_row, dict) else None,
                    "phantom_profit_risk": best_row.get('phantom_profit_risk') if isinstance(best_row, dict) else None,
                    "tp": best_row.get('tp') if isinstance(best_row, dict) else None,
                    "sl": best_row.get('sl') if isinstance(best_row, dict) else None,
                    "selected": bool(
                        isinstance(selected_best, dict)
                        and str(selected_best.get("member_method")) == str(member_method)
                    ),
                })

            out = {
                "success": True,
                "symbol": symbol,
                "timeframe": timeframe,
                "method": "ensemble",
                "horizon": horizon_val,
                "direction": direction_norm,
                "mode": mode_val,
                "optimizer": optimizer_val,
                "last_price": out_last_price,
                "last_price_close": out_last_price_close,
                "last_price_source": "ensemble_members_median",
                "objective": objective_val,
                "search_profile": search_profile_val,
                "fast_defaults": bool(search_profile_val == 'fast'),
                "compute_profile": {
                    "profile": search_profile_val,
                    "n_sims": int(sims),
                    "n_trials": int(optuna_trials_val) if optimizer_val == 'optuna' else None,
                    "tp_steps": int(tp_steps_val),
                    "sl_steps": int(sl_steps_val),
                    "ratio_steps": int(ratio_steps_val),
                    "vol_steps": int(vol_steps_val),
                    "refine": bool(refine_flag),
                    "statistical_robustness": {
                        "enabled": bool(statistical_robustness_requested),
                        "target_ci_width": target_ci_width_val,
                        "n_seeds_stability": n_seeds_stability_val,
                        "bootstrap_enabled": bool(enable_bootstrap_val),
                        "n_bootstrap": n_bootstrap_val,
                        "convergence_check_enabled": bool(enable_convergence_check_val),
                        "convergence_window": convergence_window_val,
                        "convergence_threshold": convergence_threshold_val,
                        "power_analysis_enabled": bool(enable_power_analysis_val),
                        "power_effect_size": power_effect_size_val,
                        "sensitivity_analysis_enabled": bool(enable_sensitivity_analysis_val),
                    } if statistical_robustness_requested else None,
                },
                "results": summary_results,
                "results_total": len(candidates),
                "viable_results_total": viable_results_total,
                "best": selected_best,
                "viable": bool(viable),
                "least_negative": _least_negative_ref(selected_best) if (selected_best and not viable) else None,
                "grid": grid_out,
                "no_candidates": not bool(selected_best),
                "status": status,
                "status_reason": status_reason,
                "no_action": status != "ok",
                "ensemble": {
                    "methods": ensemble_methods,
                    "agg": ensemble_agg,
                    "weights": ensemble_weight_map if ensemble_weight_map else None,
                    "members": member_summaries,
                    "member_errors": member_errors,
                    "aggregate_metrics": aggregate_metrics or None,
                    "n_total": n_total,
                    "n_succeeded": n_succeeded,
                    "n_failed": n_failed,
                    "degraded": ensemble_degraded,
                    "confidence": ensemble_confidence,
                    "selected_member": (
                        {
                            "method": selected_best.get("member_method"),
                            "method_used": selected_best.get("member_method_used"),
                        }
                        if isinstance(selected_best, dict) else None
                    ),
                },
            }
            if objective_changed:
                out["objective_requested"] = objective_requested
                out["objective_used"] = objective_val
            if member_errors:
                if ensemble_degraded:
                    out["warning"] = (
                        f"Ensemble degraded: {n_failed}/{n_total} members failed "
                        f"(confidence={ensemble_confidence}). "
                        f"Results based on {n_succeeded} method(s) only — interpret with caution."
                    )
                elif n_succeeded == 1:
                    out["warning"] = (
                        f"{n_failed}/{n_total} ensemble member(s) failed. "
                        f"Only 1 method succeeded — ensemble averaging has no diversification benefit."
                    )
                else:
                    out["warning"] = f"{n_failed} ensemble member(s) failed."
            diagnostics: Dict[str, Any] = {}
            if selected_best:
                diagnostics = _build_selection_diagnostics(
                    selected_best,
                    cost_per_trade=cost_per_trade,
                )
                out.update(diagnostics)
            if statistical_robustness_requested and isinstance(selected_best, dict):
                selected_output = next(
                    (
                        row.get("output")
                        for row in member_runs
                        if isinstance(row.get("best"), dict)
                        and str(row.get("method")) == str(selected_best.get("member_method"))
                    ),
                    None,
                )
                if isinstance(selected_output, dict):
                    selected_stats = selected_output.get("statistical_robustness")
                    if isinstance(selected_stats, dict):
                        stats_copy = dict(selected_stats)
                        stats_copy["source"] = "selected_member"
                        stats_copy["member_method"] = str(selected_best.get("member_method"))
                        out["statistical_robustness"] = stats_copy
                    min_sims_member = selected_output.get("min_sims_recommended")
                    if min_sims_member is not None:
                        out["min_sims_recommended"] = min_sims_member
            if min_prob_resolve_val is not None:
                out["min_prob_resolve"] = float(min_prob_resolve_val)
            if has_trading_costs:
                out["trading_costs"] = {
                    "cost_per_trade": _safe_float(cost_per_trade),
                    "cost_unit": mode_val,
                    "spread_pips": _safe_float(spread_pips_val) if spread_pips_val else None,
                    "spread_pct": _safe_float(spread_pct_val) if spread_pct_val else None,
                    "commission_pct": _safe_float(commission_pct_val) if commission_pct_val else None,
                    "slippage_pips": _safe_float(slippage_pips_val) if slippage_pips_val else None,
                    "slippage_pct": _safe_float(slippage_pct_val) if slippage_pct_val else None,
                }
            if selected_best and not viable:
                out["advice"] = [
                    "Increase horizon to allow more time for barrier resolution.",
                    "Try the opposite direction and compare objective metrics.",
                    "Widen TP/SL search ranges or switch grid_style to volatility/ratio.",
                    "Skip this setup if edge and EV remain unattractive.",
                ]
            actionability_payload = _build_actionability_payload(
                status=status,
                status_reason=status_reason,
                row=selected_best,
                diagnostics=diagnostics,
                warning=out.get("warning"),
                ensemble_degraded=ensemble_degraded,
            )
            out.update(actionability_payload)
            out["no_action"] = not bool(actionability_payload.get("trade_gate_passed"))
            return out

        if method_name == 'auto':
            method_name, auto_reason = _auto_barrier_method(
                symbol, timeframe, prices, horizon=horizon_val
            )

        def _simulate_paths_for_seed_range(
            seed_base: Optional[int],
            seed_count: int,
        ) -> Tuple[
            np.ndarray,
            bool,
            float,
            Optional[np.ndarray],
            Optional[np.ndarray],
            Optional[np.ndarray],
        ]:
            local_paths_list: List[np.ndarray] = []
            local_bb_enabled = method_name == 'mc_gbm_bb'
            effective_seed_count = max(1, int(seed_count))
            local_seed_base = (
                int(seed_base)
                if seed_base is not None
                else int(np.random.default_rng().integers(0, np.iinfo(np.int32).max))
            )

            if method_name in ('mc_gbm', 'mc_gbm_bb'):
                for offset in range(effective_seed_count):
                    sim = _simulate_gbm_mc(
                        prices,
                        horizon=horizon_val,
                        n_sims=int(sims),
                        seed=int(local_seed_base + offset),
                    )
                    local_paths_list.append(np.asarray(sim['price_paths'], dtype=float))
            elif method_name == 'hmm_mc':
                n_states = int(params_dict.get('n_states', 2) or 2)
                for offset in range(effective_seed_count):
                    sim = _simulate_hmm_mc(
                        prices,
                        horizon=horizon_val,
                        n_states=int(n_states),
                        n_sims=int(sims),
                        seed=int(local_seed_base + offset),
                    )
                    local_paths_list.append(np.asarray(sim['price_paths'], dtype=float))
            elif method_name == 'garch':
                p_order = int(params_dict.get('p', 1))
                q_order = int(params_dict.get('q', 1))
                for offset in range(effective_seed_count):
                    sim = _simulate_garch_mc(
                        prices,
                        horizon=horizon_val,
                        n_sims=int(sims),
                        seed=int(local_seed_base + offset),
                        p_order=p_order,
                        q_order=q_order,
                    )
                    local_paths_list.append(np.asarray(sim['price_paths'], dtype=float))
            elif method_name == 'bootstrap':
                bs = params_dict.get('block_size')
                if bs:
                    bs = int(bs)
                for offset in range(effective_seed_count):
                    sim = _simulate_bootstrap_mc(
                        prices,
                        horizon=horizon_val,
                        n_sims=int(sims),
                        seed=int(local_seed_base + offset),
                        block_size=bs,
                    )
                    local_paths_list.append(np.asarray(sim['price_paths'], dtype=float))
            elif method_name == 'heston':
                for offset in range(effective_seed_count):
                    sim = _simulate_heston_mc(
                        prices,
                        horizon=horizon_val,
                        n_sims=int(sims),
                        seed=int(local_seed_base + offset),
                        kappa=params_dict.get('kappa'),
                        theta=params_dict.get('theta'),
                        xi=params_dict.get('xi'),
                        rho=params_dict.get('rho'),
                        v0=params_dict.get('v0'),
                    )
                    local_paths_list.append(np.asarray(sim['price_paths'], dtype=float))
            elif method_name == 'jump_diffusion':
                for offset in range(effective_seed_count):
                    sim = _simulate_jump_diffusion_mc(
                        prices,
                        horizon=horizon_val,
                        n_sims=int(sims),
                        seed=int(local_seed_base + offset),
                        jump_lambda=params_dict.get('jump_lambda', params_dict.get('lambda')),
                        jump_mu=params_dict.get('jump_mu'),
                        jump_sigma=params_dict.get('jump_sigma'),
                        jump_threshold=float(params_dict.get('jump_threshold', 3.0)),
                    )
                    local_paths_list.append(np.asarray(sim['price_paths'], dtype=float))
            else:
                raise ValueError(
                    f"Unsupported method: {method}. Use 'mc_gbm', 'mc_gbm_bb', "
                    f"'hmm_mc', 'garch', 'bootstrap', 'heston', 'jump_diffusion', "
                    f"'auto', or 'ensemble'."
                )

            local_paths = (
                np.vstack(local_paths_list)
                if len(local_paths_list) > 1
                else local_paths_list[0]
            )
            try:
                sim_anchor_price = float(prices[-1])
            except Exception:
                sim_anchor_price = float(last_price_close)
            local_paths = _scale_price_paths_to_reference(
                local_paths,
                simulated_anchor_price=sim_anchor_price,
                reference_price=last_price,
            )

            local_bb_sigma = 0.0
            local_bb_log_paths = None
            local_bb_uniform_tp = None
            local_bb_uniform_sl = None
            if local_bb_enabled:
                rets = _log_returns_from_prices(prices)
                rets = rets[np.isfinite(rets)]
                local_bb_sigma = float(np.std(rets, ddof=1)) if rets.size else 0.0
                if not np.isfinite(local_bb_sigma) or local_bb_sigma <= 0:
                    local_bb_enabled = False
                else:
                    local_sims_total, local_horizon = local_paths.shape
                    log_paths = np.log(np.clip(local_paths, 1e-12, None))
                    log_s0 = float(np.log(max(last_price, 1e-12)))
                    local_bb_log_paths = np.concatenate(
                        [np.full((local_sims_total, 1), log_s0), log_paths],
                        axis=1,
                    )
                    rng_bb = np.random.RandomState(int(local_seed_base) + 7)
                    local_bb_uniform_tp = rng_bb.rand(local_sims_total, local_horizon)
                    local_bb_uniform_sl = rng_bb.rand(local_sims_total, local_horizon)

            return (
                local_paths,
                local_bb_enabled,
                local_bb_sigma,
                local_bb_log_paths,
                local_bb_uniform_tp,
                local_bb_uniform_sl,
            )

        try:
            (
                paths,
                bb_enabled,
                bb_sigma,
                bb_log_paths,
                bb_uniform_tp,
                bb_uniform_sl,
            ) = _simulate_paths_for_seed_range(seed_base=request_seed_base, seed_count=int(n_seeds))
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as e:
            return {
                "error": f"Simulation failed ({method_name}): {e}",
                "error_type": "simulation_failure",
                "traceback_summary": traceback.format_exc()[-500:],
            }

        S, H = paths.shape

        def _linspace(a: float, b: float, n: int) -> np.ndarray:
            try:
                return np.linspace(float(a), float(b), int(max(1, n)))
            except Exception:
                return np.array([float(a)])

        seen: Set[Tuple[int, int]] = set()
        base_candidates: List[Tuple[float, float]] = []

        def _push(tp_unit: float, sl_unit: float, bucket: List[Tuple[float, float]]) -> None:
            try:
                tp_val = float(tp_unit)
                sl_val = float(sl_unit)
            except (TypeError, ValueError):
                return
            if not np.isfinite(tp_val) or not np.isfinite(sl_val):
                return
            if tp_val <= 0 or sl_val <= 0:
                return
            if tp_val < min_barrier_distance or sl_val < min_barrier_distance:
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

        if grid_style_val == 'preset':
            preset_key = preset_val or 'intraday'
            cfg = BARRIER_GRID_PRESETS.get(preset_key, BARRIER_GRID_PRESETS['intraday'])
            if mode_val == 'pct':
                _add_fixed(base_candidates, cfg['tp_min'], cfg['tp_max'], int(cfg['tp_steps']), cfg['sl_min'], cfg['sl_max'], int(cfg['sl_steps']))
            else:
                scale = (float(last_price) / float(pip_size)) / 100.0
                _add_fixed(base_candidates, cfg['tp_min'] * scale, cfg['tp_max'] * scale, int(cfg['tp_steps']), cfg['sl_min'] * scale, cfg['sl_max'] * scale, int(cfg['sl_steps']))
        
        elif grid_style_val == 'volatility':
            # Calculate simple volatility over window
            rets = _log_returns_from_prices(prices)
            rets = rets[np.isfinite(rets)]
            if rets.size > vol_window_val:
                rets = rets[-vol_window_val:]
            vol_per_bar = float(np.std(rets, ddof=1)) if rets.size > 1 else 0.0
            vol_horizon = vol_per_bar * np.sqrt(horizon_val)

            # Convert to percentage space for baseline
            vol_pct = vol_horizon * 100.0

            if mode_val == 'pct':
                tp_start = max(vol_floor_pct_val, vol_pct * vol_min_mult_val)
                tp_end = max(tp_start * 1.1, vol_pct * vol_max_mult_val)
                sl_start = max(vol_floor_pct_val, vol_pct * vol_min_mult_val * 0.8)
                _add_fixed(base_candidates, tp_start, tp_end, vol_steps_val, sl_start, sl_start * vol_sl_multiplier_val, vol_sl_steps_val)
            else:
                # Convert volatility to ticks and apply tick floor when in pips mode
                vol_pips = (vol_pct / 100.0) * (last_price / float(pip_size))
                tp_start = max(vol_floor_pips_val, vol_pips * vol_min_mult_val)
                tp_end = max(tp_start * 1.1, vol_pips * vol_max_mult_val)
                sl_start = max(vol_floor_pips_val, vol_pips * vol_min_mult_val * 0.8)
                _add_fixed(base_candidates, tp_start, tp_end, vol_steps_val, sl_start, sl_start * vol_sl_multiplier_val, vol_sl_steps_val)
            
        elif grid_style_val == 'ratio':
            # Fixed SL grid, TP derived from ratios
            sl_start = sl_min_val
            sl_end = sl_max_val
            for sl_val in _linspace(sl_start, sl_end, sl_steps_val):
                for r in _linspace(ratio_min_val, ratio_max_val, ratio_steps_val):
                    _push(sl_val * r, sl_val, base_candidates)
        
        else: # fixed
            _add_fixed(base_candidates, tp_min_val, tp_max_val, tp_steps_val, sl_min_val, sl_max_val, sl_steps_val)

        # Evaluate candidates
        results: List[Dict[str, Any]] = []
        optuna_meta: Optional[Dict[str, Any]] = None
        dir_long = direction_norm == 'long'
        invalid_barrier_candidates = 0
        eval_context = _BarrierEvaluationContext(
            mode_val=mode_val,
            dir_long=dir_long,
            last_price=float(last_price),
            pip_size=float(pip_size),
            rr_min_val=rr_min_val,
            rr_max_val=rr_max_val,
            has_trading_costs=has_trading_costs,
            ev_deduct_cost=float(ev_deduct_cost),
            cost_per_trade=float(cost_per_trade),
            min_prob_win_val=min_prob_win_val,
            max_prob_no_hit_val=max_prob_no_hit_val,
            min_prob_resolve_val=min_prob_resolve_val,
            max_median_time_val=max_median_time_val,
        )

        def _evaluate(
            bucket: List[Tuple[float, float]],
            eval_paths: np.ndarray,
            eval_bb_enabled: bool,
            eval_bb_sigma: float,
            eval_bb_log_paths: Optional[np.ndarray],
            eval_bb_uniform_tp: Optional[np.ndarray],
            eval_bb_uniform_sl: Optional[np.ndarray],
            count_invalid: bool = True,
        ) -> List[Dict[str, Any]]:
            nonlocal invalid_barrier_candidates
            rows, invalid_count = _evaluate_barrier_bucket(
                bucket,
                eval_paths,
                context=eval_context,
                bridge_inputs=_BarrierBridgeInputs(
                    enabled=eval_bb_enabled,
                    sigma=float(eval_bb_sigma),
                    log_paths=eval_bb_log_paths,
                    uniform_tp=eval_bb_uniform_tp,
                    uniform_sl=eval_bb_uniform_sl,
                ),
                count_invalid=count_invalid,
            )
            invalid_barrier_candidates += invalid_count
            return rows

        def _objective_convergence_inputs(
            eval_paths: np.ndarray,
            *,
            best_row: Dict[str, Any],
        ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[str]]:
            _, horizon_total = eval_paths.shape
            tp_trigger = _safe_float(best_row.get('tp_price'))
            sl_trigger = _safe_float(best_row.get('sl_price'))
            reward = _safe_float(best_row.get('tp'))
            risk = _safe_float(best_row.get('sl'))
            if tp_trigger is None or sl_trigger is None or reward is None or risk is None:
                return None, None, None

            hit_tp = (eval_paths >= tp_trigger) if dir_long else (eval_paths <= tp_trigger)
            hit_sl = (eval_paths <= sl_trigger) if dir_long else (eval_paths >= sl_trigger)
            any_tp = hit_tp.any(axis=1)
            any_sl = hit_sl.any(axis=1)

            first_tp = hit_tp.argmax(axis=1)
            first_sl = hit_sl.argmax(axis=1)
            first_tp[~any_tp] = horizon_total
            first_sl[~any_sl] = horizon_total

            wins = (first_tp < first_sl).astype(float)
            losses = (first_sl < first_tp).astype(float)
            ties = ((first_tp == first_sl) & (first_tp < horizon_total)).astype(float)
            resolves = wins + losses + ties

            trials = np.arange(1, eval_paths.shape[0] + 1, dtype=float)
            cum_wins = np.cumsum(wins)
            cum_losses = np.cumsum(losses)
            cum_ties = np.cumsum(ties)
            cum_resolves = np.cumsum(resolves)

            prob_win_series = cum_wins / trials
            prob_loss_series = cum_losses / trials
            prob_tp_first_series = (cum_wins + 0.5 * cum_ties) / trials
            prob_sl_first_series = (cum_losses + 0.5 * cum_ties) / trials
            prob_resolve_series = cum_resolves / trials

            net_reward = reward - cost_per_trade if has_trading_costs else reward
            net_risk = risk + cost_per_trade if has_trading_costs else risk
            net_rr = net_reward / net_risk if net_risk > 0 else 0.0

            reward_frac = 0.0
            risk_frac = 0.0
            if mode_val == 'pct':
                reward_frac = net_reward / 100.0
                risk_frac = net_risk / 100.0
            elif last_price > 0 and pip_size:
                unit_to_return = float(pip_size) / float(last_price)
                reward_frac = net_reward * unit_to_return
                risk_frac = net_risk * unit_to_return
            elif last_price > 0:
                reward_frac = abs(tp_trigger - last_price) / last_price
                risk_frac = abs(sl_trigger - last_price) / last_price
            reward_frac = max(reward_frac, -0.999)
            if risk_frac >= 1.0:
                risk_frac = 0.999

            event = f"selected_objective_{objective_val}"
            if objective_val == 'prob_tp_first':
                estimate_series = prob_tp_first_series
            elif objective_val == 'prob_resolve':
                estimate_series = prob_resolve_series
            elif objective_val == 'min_loss_prob':
                estimate_series = prob_loss_series
            elif objective_val == 'edge':
                estimate_series = prob_win_series - prob_loss_series
            elif objective_val == 'ev':
                estimate_series = (
                    prob_tp_first_series * net_reward
                    - prob_sl_first_series * net_risk
                )
            elif objective_val == 'ev_per_bar':
                time_in_trade = np.minimum(np.minimum(first_tp, first_sl) + 1, horizon_total).astype(float)
                mean_time_series = np.cumsum(time_in_trade) / trials
                ev_series = (
                    prob_tp_first_series * net_reward
                    - prob_sl_first_series * net_risk
                )
                estimate_series = np.divide(
                    ev_series,
                    mean_time_series,
                    out=np.zeros_like(ev_series),
                    where=mean_time_series > 0,
                )
            elif objective_val == 'kelly':
                estimate_series = (
                    prob_tp_first_series - (prob_sl_first_series / net_rr)
                    if net_rr > 0 else np.zeros_like(prob_tp_first_series)
                )
            elif objective_val == 'ev_cond':
                active = cum_wins + cum_losses + cum_ties
                active_mask = active > 0
                estimate_series = np.zeros_like(prob_tp_first_series)
                if np.any(active_mask):
                    win_c = np.divide(
                        cum_wins + 0.5 * cum_ties,
                        active,
                        out=np.zeros_like(active, dtype=float),
                        where=active_mask,
                    )
                    loss_c = np.divide(
                        cum_losses + 0.5 * cum_ties,
                        active,
                        out=np.zeros_like(active, dtype=float),
                        where=active_mask,
                    )
                    estimate_series[active_mask] = (
                        win_c[active_mask] * net_reward
                        - loss_c[active_mask] * net_risk
                    )
            elif objective_val == 'kelly_cond':
                active = cum_wins + cum_losses + cum_ties
                active_mask = active > 0
                estimate_series = np.zeros_like(prob_tp_first_series)
                if np.any(active_mask) and net_rr > 0:
                    win_c = np.divide(
                        cum_wins + 0.5 * cum_ties,
                        active,
                        out=np.zeros_like(active, dtype=float),
                        where=active_mask,
                    )
                    loss_c = np.divide(
                        cum_losses + 0.5 * cum_ties,
                        active,
                        out=np.zeros_like(active, dtype=float),
                        where=active_mask,
                    )
                    estimate_series[active_mask] = (
                        win_c[active_mask] - (loss_c[active_mask] / net_rr)
                    )
            elif objective_val == 'profit_factor':
                denom = prob_sl_first_series * net_risk
                estimate_series = np.zeros_like(prob_tp_first_series)
                valid = denom > 0
                estimate_series[valid] = (prob_tp_first_series[valid] * net_reward) / denom[valid]
                positive_no_loss = (~valid) & (prob_tp_first_series > 0) & (net_reward > 0)
                estimate_series[positive_no_loss] = 1e9
            elif objective_val == 'utility':
                estimate_series = (
                    prob_tp_first_series * math.log1p(reward_frac)
                    + prob_sl_first_series * math.log1p(-risk_frac)
                )
            else:
                estimate_series = prob_resolve_series

            return estimate_series * trials, trials, event

        pareto_front: Optional[List[Dict[str, Any]]] = None
        if optimizer_val == 'optuna':
            try:
                import optuna
                try:
                    from optuna.exceptions import (
                        ExperimentalWarning as _OptunaExperimentalWarning,
                    )
                except Exception:
                    _OptunaExperimentalWarning = Warning
            except Exception as ex:
                return {"error": f"Optuna optimizer requested but unavailable: {ex}"}

            def _suppress_optuna_experimental_warnings() -> None:
                warnings.simplefilter("ignore", _OptunaExperimentalWarning)
                warnings.filterwarnings("ignore", category=_OptunaExperimentalWarning)
                warnings.filterwarnings(
                    "ignore",
                    message=r".*multivariate.*experimental feature.*",
                )

            tp_vals = [float(tp) for tp, _ in base_candidates] if base_candidates else [float(tp_min_val), float(tp_max_val)]
            sl_vals = [float(sl) for _, sl in base_candidates] if base_candidates else [float(sl_min_val), float(sl_max_val)]
            tp_lo = max(1e-9, min_barrier_distance, float(min(tp_vals)))
            tp_hi = max(tp_lo, float(max(tp_vals)))
            sl_lo = max(1e-9, min_barrier_distance, float(min(sl_vals)))
            sl_hi = max(sl_lo, float(max(sl_vals)))
            rr_lo = max(1e-9, float(min(ratio_min_val, ratio_max_val)))
            rr_hi = max(rr_lo, float(max(ratio_min_val, ratio_max_val)))

            sampler_name = optuna_sampler_val
            if sampler_name == 'random':
                sampler_obj = optuna.samplers.RandomSampler(seed=optuna_seed)
            elif sampler_name == 'cmaes':
                sampler_obj = optuna.samplers.CmaEsSampler(seed=optuna_seed)
            else:
                sampler_name = 'tpe'
                with warnings.catch_warnings():
                    _suppress_optuna_experimental_warnings()
                    sampler_obj = optuna.samplers.TPESampler(seed=optuna_seed, multivariate=True)

            pruner_name = optuna_pruner_val
            if pruner_name in {'none'}:
                pruner_obj = optuna.pruners.NopPruner()
            elif pruner_name == 'hyperband':
                pruner_obj = optuna.pruners.HyperbandPruner()
            elif pruner_name == 'percentile':
                pruner_obj = optuna.pruners.PercentilePruner(50.0)
            else:
                pruner_name = 'median'
                pruner_obj = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0)

            optuna.logging.set_verbosity(optuna.logging.WARNING)
            sampled_rows: List[Dict[str, Any]] = []
            trial_rows: Dict[int, Dict[str, Any]] = {}

            if optuna_pareto_val:
                directions = [d for _, d in pareto_objectives]
                with warnings.catch_warnings():
                    _suppress_optuna_experimental_warnings()
                    study = optuna.create_study(directions=directions, sampler=sampler_obj, pruner=pruner_obj)

                def _bad_values() -> Tuple[float, ...]:
                    vals: List[float] = []
                    for _, d in pareto_objectives:
                        vals.append(-1e18 if d == 'maximize' else 1e18)
                    return tuple(vals)

                def _metric_value(row: Dict[str, Any], metric: str, direction_name: str) -> float:
                    raw = row.get(metric)
                    try:
                        value = float(raw)
                    except Exception:
                        return -1e18 if direction_name == 'maximize' else 1e18
                    if not np.isfinite(value):
                        return -1e18 if direction_name == 'maximize' else 1e18
                    return float(value)

                def _objective_trial(trial: Any) -> Tuple[float, ...]:
                    if grid_style_val == 'ratio':
                        sl_unit = float(trial.suggest_float('sl', sl_lo, sl_hi))
                        rr_unit = float(trial.suggest_float('rr', rr_lo, rr_hi))
                        tp_unit = sl_unit * rr_unit
                    else:
                        tp_unit = float(trial.suggest_float('tp', tp_lo, tp_hi))
                        sl_unit = float(trial.suggest_float('sl', sl_lo, sl_hi))

                    rows = _evaluate(
                        [(tp_unit, sl_unit)],
                        paths,
                        bb_enabled,
                        bb_sigma,
                        bb_log_paths,
                        bb_uniform_tp,
                        bb_uniform_sl,
                    )
                    if not rows:
                        return _bad_values()
                    row = rows[0]
                    sampled_rows.append(row)
                    trial_rows[int(trial.number)] = row
                    trial.set_user_attr('tp', float(row.get('tp', tp_unit)))
                    trial.set_user_attr('sl', float(row.get('sl', sl_unit)))
                    values = tuple(
                        _metric_value(row, metric_name, direction_name)
                        for metric_name, direction_name in pareto_objectives
                    )
                    trial.set_user_attr('objective_values', {
                        metric_name: float(values[idx]) for idx, (metric_name, _) in enumerate(pareto_objectives)
                    })
                    return values

                try:
                    with warnings.catch_warnings():
                        _suppress_optuna_experimental_warnings()
                        study.optimize(
                            _objective_trial,
                            n_trials=int(optuna_trials_val),
                            timeout=float(optuna_timeout_val) if optuna_timeout_val is not None else None,
                            n_jobs=int(optuna_n_jobs_val),
                        )
                except (ValueError, RuntimeError) as e:
                    return {
                        "error": f"Optuna optimization failed (pareto): {e}",
                        "error_type": "optuna_failure",
                        "traceback_summary": traceback.format_exc()[-500:],
                    }
                front: List[Dict[str, Any]] = []
                for trial in study.best_trials:
                    row = trial_rows.get(int(trial.number))
                    if not isinstance(row, dict):
                        continue
                    entry = dict(row)
                    values = list(trial.values) if isinstance(trial.values, (list, tuple)) else []
                    entry['trial'] = int(trial.number)
                    entry['objective_values'] = {
                        metric_name: float(values[idx]) if idx < len(values) else None
                        for idx, (metric_name, _) in enumerate(pareto_objectives)
                    }
                    front.append(entry)

                if front:
                    front.sort(
                        key=lambda x: (
                            -float(x.get('ev', -1e18)) if x.get('ev') is not None else 1e18,
                            float(x.get('prob_loss', 1e18)) if x.get('prob_loss') is not None else 1e18,
                            float(x.get('t_hit_resolve_median', 1e18)) if x.get('t_hit_resolve_median') is not None else 1e18,
                        )
                    )
                pareto_front = front[:int(pareto_limit_val)]
            else:
                maximize = objective_val != 'min_loss_prob'
                direction = 'maximize' if maximize else 'minimize'
                with warnings.catch_warnings():
                    _suppress_optuna_experimental_warnings()
                    study = optuna.create_study(direction=direction, sampler=sampler_obj, pruner=pruner_obj)

                def _objective_trial(trial: Any) -> float:
                    if grid_style_val == 'ratio':
                        sl_unit = float(trial.suggest_float('sl', sl_lo, sl_hi))
                        rr_unit = float(trial.suggest_float('rr', rr_lo, rr_hi))
                        tp_unit = sl_unit * rr_unit
                    else:
                        tp_unit = float(trial.suggest_float('tp', tp_lo, tp_hi))
                        sl_unit = float(trial.suggest_float('sl', sl_lo, sl_hi))

                    rows = _evaluate(
                        [(tp_unit, sl_unit)],
                        paths,
                        bb_enabled,
                        bb_sigma,
                        bb_log_paths,
                        bb_uniform_tp,
                        bb_uniform_sl,
                    )
                    if not rows:
                        return -1e18 if maximize else 1e18
                    row = rows[0]
                    sampled_rows.append(row)
                    trial_rows[int(trial.number)] = row
                    trial.set_user_attr('tp', float(row.get('tp', tp_unit)))
                    trial.set_user_attr('sl', float(row.get('sl', sl_unit)))
                    if objective_val == 'min_loss_prob':
                        return float(row.get('prob_loss', 1.0))
                    return float(row.get(objective_val, row.get('ev', -1e18)))

                try:
                    with warnings.catch_warnings():
                        _suppress_optuna_experimental_warnings()
                        study.optimize(
                            _objective_trial,
                            n_trials=int(optuna_trials_val),
                            timeout=float(optuna_timeout_val) if optuna_timeout_val is not None else None,
                            n_jobs=int(optuna_n_jobs_val),
                        )
                except (ValueError, RuntimeError) as e:
                    return {
                        "error": f"Optuna optimization failed: {e}",
                        "error_type": "optuna_failure",
                        "traceback_summary": traceback.format_exc()[-500:],
                    }

            dedup: Dict[Tuple[int, int], Dict[str, Any]] = {}
            for row in sampled_rows:
                try:
                    key = (int(round(float(row.get('tp', 0.0)) * 1e6)), int(round(float(row.get('sl', 0.0)) * 1e6)))
                except Exception:
                    continue
                cur = dedup.get(key)
                if cur is None:
                    dedup[key] = row
                    continue
                if objective_val == 'min_loss_prob':
                    if float(row.get('prob_loss', 1.0)) < float(cur.get('prob_loss', 1.0)):
                        dedup[key] = row
                else:
                    if float(row.get(objective_val, -1e18)) > float(cur.get(objective_val, -1e18)):
                        dedup[key] = row

            results.extend(dedup.values())
            optuna_meta = {
                "n_trials": int(optuna_trials_val),
                "completed_trials": int(len(study.trials)),
                "sampler": sampler_name,
                "pruner": pruner_name,
                "timeout": float(optuna_timeout_val) if optuna_timeout_val is not None else None,
                "n_jobs": int(optuna_n_jobs_val),
                "pareto": bool(optuna_pareto_val),
            }
            if optuna_pareto_val:
                optuna_meta["pareto_objectives"] = [
                    {"metric": metric_name, "direction": direction_name}
                    for metric_name, direction_name in pareto_objectives
                ]
        else:
            results.extend(
                _evaluate(
                    base_candidates,
                    paths,
                    bb_enabled,
                    bb_sigma,
                    bb_log_paths,
                    bb_uniform_tp,
                    bb_uniform_sl,
                )
            )

        _sort_candidate_results(results, objective_val)

        if refine_flag and results:
            best_seed = results[0]
            tp_c = best_seed['tp']
            sl_c = best_seed['sl']
            tp_a = max(1e-9, tp_c * (1.0 - refine_radius_val))
            tp_b = tp_c * (1.0 + refine_radius_val)
            sl_a = max(1e-9, sl_c * (1.0 - refine_radius_val))
            sl_b = sl_c * (1.0 + refine_radius_val)
            refine_candidates: List[Tuple[float, float]] = []
            _add_fixed(refine_candidates, tp_a, tp_b, refine_steps_val, sl_a, sl_b, refine_steps_val)
            results.extend(
                _evaluate(
                    refine_candidates,
                    paths,
                    bb_enabled,
                    bb_sigma,
                    bb_log_paths,
                    bb_uniform_tp,
                    bb_uniform_sl,
                )
            )
            _sort_candidate_results(results, objective_val)

        candidate_views = _select_barrier_candidate_views(
            list(results),
            cost_per_trade=cost_per_trade,
            viable_only_val=viable_only_val,
            concise_val=concise_val,
            top_k_val=top_k_val,
            return_grid=return_grid,
            output_mode=output_mode,
        )
        ranked_candidates = candidate_views["ranked_candidates"]
        viable_candidates = candidate_views["viable_candidates"]
        candidates = candidate_views["candidates"]
        grid_out = candidate_views["grid_out"]
        summary_results = candidate_views["summary_results"]
        viability_filtered_out = candidate_views["viability_filtered_out"]
        no_candidates = len(candidates) == 0
        warning = candidate_views["warning"]
            
        best = candidates[0] if candidates else None
        if isinstance(best, dict):
            _annotate_candidate_metrics(best, cost_per_trade=cost_per_trade)
        viable = _candidate_is_viable(best, cost_per_trade=cost_per_trade)
        viable_results_total = int(len(viable_candidates))
        status = "ok"
        status_reason = None
        if viability_filtered_out:
            status = "non_viable"
            status_reason = warning
        elif no_candidates:
            status = "no_candidates"
            status_reason = warning
        elif not viable:
            status = "non_viable"
            status_reason = _candidate_status_reason(best, cost_per_trade=cost_per_trade)

        out = {
            "success": True,
            "symbol": symbol,
            "timeframe": timeframe,
            "method": method_name,
            "horizon": horizon_val,
            "direction": direction_norm,
            "mode": mode_val,
            "optimizer": optimizer_val,
            "last_price": float(last_price),
            "last_price_close": float(last_price_close),
            "last_price_source": last_price_source,
            "objective": objective_val,
            "search_profile": search_profile_val,
            "fast_defaults": bool(search_profile_val == 'fast'),
            "compute_profile": {
                "profile": search_profile_val,
                "n_sims": int(sims),
                "n_trials": int(optuna_trials_val) if optimizer_val == 'optuna' else None,
                "tp_steps": int(tp_steps_val),
                "sl_steps": int(sl_steps_val),
                "ratio_steps": int(ratio_steps_val),
                "vol_steps": int(vol_steps_val),
                "refine": bool(refine_flag),
                "statistical_robustness": {
                    "enabled": bool(statistical_robustness_requested),
                    "target_ci_width": target_ci_width_val,
                    "n_seeds_stability": n_seeds_stability_val,
                    "bootstrap_enabled": bool(enable_bootstrap_val),
                    "n_bootstrap": n_bootstrap_val,
                    "convergence_check_enabled": bool(enable_convergence_check_val),
                    "convergence_window": convergence_window_val,
                    "convergence_threshold": convergence_threshold_val,
                    "power_analysis_enabled": bool(enable_power_analysis_val),
                    "power_effect_size": power_effect_size_val,
                    "sensitivity_analysis_enabled": bool(enable_sensitivity_analysis_val),
                } if statistical_robustness_requested else None,
            },
            "results": summary_results,
            "results_total": len(candidates),
            "viable_results_total": viable_results_total,
            "best": best,
            "viable": viable,
            "least_negative": _least_negative_ref(best) if (best is not None and not viable) else None,
            "grid": grid_out,
            "no_candidates": no_candidates,
            "status": status,
            "status_reason": status_reason,
            "no_action": status != "ok",
        }
        if optuna_meta is not None:
            out["optuna"] = optuna_meta
        if pareto_front is not None:
            out["pareto_front"] = pareto_front
            out["pareto_count"] = int(len(pareto_front))
        
        if statistical_robustness_requested and isinstance(best, dict) and len(candidates) > 0:
            statistical_analysis: Dict[str, Any] = {
                "minimum_simulations": {
                    "recommended": int(min_sims_recommended),
                    "used": int(sims),
                    "target_ci_width": target_ci_width_val,
                    "confidence": 0.95,
                }
            }

            if enable_power_analysis_val and best.get('prob_win') is not None:
                prob_win_val = float(best['prob_win'])
                power_result = _power_analysis(
                    base_prob=prob_win_val,
                    effect_size=power_effect_size_val,
                    n_sims=int(sims),
                    alpha=0.05,
                )
                if 'error' not in power_result:
                    statistical_analysis['power_analysis'] = power_result
            
            if enable_convergence_check_val and isinstance(best, dict):
                cumulative_metric, cumulative_trials, convergence_event = _objective_convergence_inputs(
                    paths,
                    best_row=best,
                )
                if (
                    cumulative_metric is not None
                    and cumulative_trials is not None
                    and convergence_event is not None
                ):
                    convergence_result = _mc_convergence(
                        cumulative_metric,
                        cumulative_trials,
                        window_size=convergence_window_val,
                        threshold=convergence_threshold_val,
                    )
                    convergence_result["event"] = convergence_event
                    convergence_result["objective"] = objective_val
                    convergence_result["tp_price"] = float(best.get('tp_price'))
                    convergence_result["sl_price"] = float(best.get('sl_price'))
                    statistical_analysis['convergence_diagnostic'] = convergence_result
            
            if enable_bootstrap_val:
                try:
                    tp_trigger = float(best.get('tp_price', 0))
                    sl_trigger = float(best.get('sl_price', 0))
                    if tp_trigger > 0 and sl_trigger > 0:
                        bootstrap_result = _bootstrap_uncertainty(
                            paths=paths,
                            tp_trigger=tp_trigger,
                            sl_trigger=sl_trigger,
                            direction=direction_norm,
                            entry_price=last_price,
                            reward=float(best.get('tp', 0.0)),
                            risk=float(best.get('sl', 0.0)),
                            cost_per_trade=float(cost_per_trade),
                            n_bootstrap=n_bootstrap_val,
                            seed=seed,
                        )
                        if bootstrap_result:
                            statistical_analysis['bootstrap_uncertainty'] = bootstrap_result
                except Exception:
                    pass
            
            if n_seeds_stability_val > 1:
                results_by_seed: Dict[int, Dict[str, Any]] = {}
                for seed_offset in range(1, min(n_seeds_stability_val, 5) + 1):
                    seed_key = int(request_seed_base + seed_offset)
                    try:
                        (
                            stability_paths,
                            stability_bb_enabled,
                            stability_bb_sigma,
                            stability_bb_log_paths,
                            stability_bb_uniform_tp,
                            stability_bb_uniform_sl,
                        ) = _simulate_paths_for_seed_range(seed_base=seed_key, seed_count=1)
                    except (ValueError, RuntimeError, np.linalg.LinAlgError):
                        continue
                    seed_rows = _evaluate(
                        [(float(best.get('tp', 0.0)), float(best.get('sl', 0.0)))],
                        stability_paths,
                        stability_bb_enabled,
                        stability_bb_sigma,
                        stability_bb_log_paths,
                        stability_bb_uniform_tp,
                        stability_bb_uniform_sl,
                        count_invalid=False,
                    )
                    if seed_rows:
                        results_by_seed[seed_key] = seed_rows[0]

                if len(results_by_seed) > 1:
                    stability_result = _cross_seed_stability(
                        results_by_seed=results_by_seed,
                        threshold_cv=0.10,
                    )
                    stability_result["seeds_attempted"] = int(min(n_seeds_stability_val, 5))
                    stability_result["seeds_succeeded"] = int(len(results_by_seed))
                    statistical_analysis['cross_seed_stability'] = stability_result
                else:
                    statistical_analysis['cross_seed_stability'] = {
                        "stable": False,
                        "error": "Need at least 2 successful seed re-runs for stability analysis",
                        "seeds_attempted": int(min(n_seeds_stability_val, 5)),
                        "seeds_succeeded": int(len(results_by_seed)),
                        "recommendation": "Retry with a supported stochastic method or fewer failure-prone seeds.",
                    }

            if enable_sensitivity_analysis_val and sensitivity_params_requested:
                base_result = {"best": best}
                sensitivity_results: Dict[str, Any] = {}

                def _local_sensitivity_values(base_value: float) -> List[float]:
                    values: List[float] = []
                    seen_values: Set[int] = set()
                    for multiplier in (0.8, 0.9, 1.0, 1.1, 1.2):
                        value = max(min_barrier_distance, float(base_value) * multiplier)
                        key = int(round(value * 1e6))
                        if key in seen_values:
                            continue
                        seen_values.add(key)
                        values.append(value)
                    return values

                def _evaluate_sensitivity(override: Dict[str, float]) -> Dict[str, Any]:
                    tp_unit = float(override.get('tp', best.get('tp', 0.0)))
                    sl_unit = float(override.get('sl', best.get('sl', 0.0)))
                    rows = _evaluate(
                        [(tp_unit, sl_unit)],
                        paths,
                        bb_enabled,
                        bb_sigma,
                        bb_log_paths,
                        bb_uniform_tp,
                        bb_uniform_sl,
                        count_invalid=False,
                    )
                    if not rows:
                        return {"success": False}
                    return {"success": True, "best": rows[0]}

                for param_name in sensitivity_params_requested:
                    if param_name not in {'tp', 'sl'}:
                        continue
                    base_value_raw = best.get(param_name)
                    if base_value_raw is None:
                        continue
                    try:
                        parameter_values = _local_sensitivity_values(float(base_value_raw))
                    except (TypeError, ValueError):
                        continue
                    sensitivity_result = _sensitivity_analysis(
                        base_result=base_result,
                        parameter_name=param_name,
                        parameter_values=parameter_values,
                        evaluate_fn=_evaluate_sensitivity,
                    )
                    if sensitivity_result.get("success"):
                        sensitivity_results[param_name] = sensitivity_result

                if sensitivity_results:
                    statistical_analysis["sensitivity_analysis"] = sensitivity_results
            
            if statistical_analysis:
                out['statistical_robustness'] = statistical_analysis
                out['min_sims_recommended'] = int(min_sims_recommended)
        
        diagnostics = {}
        if isinstance(best, dict):
            diagnostics = _build_selection_diagnostics(best, cost_per_trade=cost_per_trade)
            out.update(diagnostics)
        if warning is not None:
            out["warning"] = warning
        elif best is not None and not viable:
            out["advice"] = [
                "Increase horizon to allow more time for barrier resolution.",
                "Try the opposite direction and compare objective metrics.",
                "Widen TP/SL search ranges or switch grid_style to volatility/ratio.",
                "Skip this setup if edge and EV remain unattractive.",
            ]
        actionability_payload = _build_actionability_payload(
            status=status,
            status_reason=status_reason,
            row=best,
            diagnostics=diagnostics,
            warning=out.get("warning"),
        )
        out.update(actionability_payload)
        out["no_action"] = not bool(actionability_payload.get("trade_gate_passed"))
        if invalid_barrier_candidates > 0:
            out["barrier_sanity_filtered"] = int(invalid_barrier_candidates)
        if min_prob_resolve_val is not None:
            out["min_prob_resolve"] = float(min_prob_resolve_val)
        if objective_changed:
            out["objective_requested"] = objective_requested
            out["objective_used"] = objective_val
        if method_requested != method_name:
            out["method_requested"] = method_requested
            out["method_used"] = method_name
            if auto_reason:
                out["auto_reason"] = auto_reason
        if bb_enabled:
            out["bridge_correction"] = True
        if viable_only_val:
            out["viable_only"] = True
        if concise_val:
            out["concise"] = True
        if has_trading_costs:
            out["trading_costs"] = {
                "cost_per_trade": _safe_float(cost_per_trade),
                "cost_unit": mode_val,
                "spread_pips": _safe_float(spread_pips_val) if spread_pips_val else None,
                "spread_pct": _safe_float(spread_pct_val) if spread_pct_val else None,
                "commission_pct": _safe_float(commission_pct_val) if commission_pct_val else None,
                "slippage_pips": _safe_float(slippage_pips_val) if slippage_pips_val else None,
                "slippage_pct": _safe_float(slippage_pct_val) if slippage_pct_val else None,
            }
        return out

    except (KeyError, AttributeError, IndexError):
        raise
    except Exception as e:
        return {
            "error": f"Error optimizing barriers: {str(e)}",
            "error_type": type(e).__name__,
            "traceback_summary": traceback.format_exc()[-500:],
        }
