"""Statistical robustness utilities for barrier optimization.

This module provides advanced statistical analysis tools for Monte Carlo
barrier simulations, including:
- Minimum simulation requirements for target CI width
- Multiple confidence interval methods (Wilson, Agresti-Coull, Jeffreys)
- Convergence diagnostics for MC simulations
- Cross-seed stability analysis
- Bootstrap resampling for uncertainty estimation
- Sensitivity analysis for barrier parameters
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import math
import numpy as np
from scipy import stats as scipy_stats

BarrierPowerAnalysisValue = Union[float, int, str]


def minimum_simulations_for_ci_width(
    target_width: float = 0.05,
    expected_prob: float = 0.5,
    confidence: float = 0.95,
    conservative: bool = True,
) -> int:
    """Calculate minimum simulations needed for target confidence interval width.
    
    Uses Wilson score interval approximation: width ≈ 2*z*sqrt(p*(1-p)/n)
    
    Args:
        target_width: Desired CI width (e.g., 0.05 for ±2.5%)
        expected_prob: Expected probability (0.5 is most conservative)
        confidence: Confidence level (0.90, 0.95, 0.99)
        conservative: If True, use worst-case p=0.5 regardless of expected_prob
        
    Returns:
        Minimum number of simulations required
    """
    if not 0 < target_width < 1:
        raise ValueError(f"target_width must be in (0, 1), got {target_width}")
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    
    z = scipy_stats.norm.ppf(1 - (1 - confidence) / 2)
    p = 0.5 if conservative else float(np.clip(expected_prob, 0.01, 0.99))
    
    n = (2 * z) ** 2 * p * (1 - p) / (target_width ** 2)
    return max(int(math.ceil(n)), 100)


def _confidence_interval_wilson_proportion(
    p_hat: float,
    n_trials: int,
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """Wilson score interval from a probability estimate and trial count."""
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    if n_trials <= 0:
        return float("nan"), float("nan")

    p_hat = float(np.clip(p_hat, 0.0, 1.0))
    z = scipy_stats.norm.ppf(1 - (1 - confidence) / 2)
    z2 = z * z

    denom = 1.0 + z2 / n_trials
    center = (p_hat + z2 / (2.0 * n_trials)) / denom
    margin = (z / denom) * math.sqrt(
        (p_hat * (1.0 - p_hat) / n_trials) + (z2 / (4.0 * n_trials * n_trials))
    )

    lo = max(0.0, center - margin)
    hi = min(1.0, center + margin)
    return float(lo), float(hi)


def confidence_interval_wilson(
    successes: int,
    n_trials: int,
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """Wilson score interval for binomial proportion.
    
    Better than normal approximation for small n or extreme probabilities.
    
    Args:
        successes: Number of successes
        n_trials: Total number of trials
        confidence: Confidence level (default 0.95)
        
    Returns:
        (lower_bound, upper_bound) tuple
    """
    if n_trials <= 0:
        return float("nan"), float("nan")

    p_hat = float(np.clip(successes / n_trials, 0.0, 1.0))
    return _confidence_interval_wilson_proportion(p_hat, n_trials, confidence=confidence)


def confidence_interval_agresti_coull(
    successes: int,
    n_trials: int,
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """Agresti-Coull interval for binomial proportion.
    
    Simple adjustment to Wald interval with better coverage.
    
    Args:
        successes: Number of successes
        n_trials: Total number of trials
        confidence: Confidence level (default 0.95)
        
    Returns:
        (lower_bound, upper_bound) tuple
    """
    if n_trials <= 0:
        return float("nan"), float("nan")
    
    z = scipy_stats.norm.ppf(1 - (1 - confidence) / 2)
    z2 = z * z
    
    n_tilde = n_trials + z2
    p_tilde = (successes + z2 / 2) / n_tilde
    
    se = math.sqrt(p_tilde * (1 - p_tilde) / n_tilde)
    center = float(p_tilde)
    margin = z * se
    
    lo = max(0.0, center - margin)
    hi = min(1.0, center + margin)
    return float(lo), float(hi)


def confidence_interval_jeffreys(
    successes: int,
    n_trials: int,
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """Jeffreys interval for binomial proportion (Bayesian).
    
    Uses Beta(0.5, 0.5) prior. Good frequentist properties.
    
    Args:
        successes: Number of successes
        n_trials: Total number of trials
        confidence: Confidence level (default 0.95)
        
    Returns:
        (lower_bound, upper_bound) tuple
    """
    if n_trials <= 0:
        return float("nan"), float("nan")
    
    alpha = 1 - confidence
    successes_adj = float(successes) + 0.5
    failures_adj = float(n_trials - successes) + 0.5
    
    lo = scipy_stats.beta.ppf(alpha / 2, successes_adj, failures_adj)
    hi = scipy_stats.beta.ppf(1 - alpha / 2, successes_adj, failures_adj)
    
    return float(max(0.0, lo)), float(min(1.0, hi))


def confidence_interval_bootstrap(
    successes: int,
    n_trials: int,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: Optional[int] = None,
) -> Tuple[float, float]:
    """Bootstrap confidence interval for binomial proportion.
    
    Non-parametric approach, useful for complex estimators.
    
    Args:
        successes: Number of successes
        n_trials: Total number of trials
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (default 0.95)
        seed: Random seed for reproducibility
        
    Returns:
        (lower_bound, upper_bound) tuple
    """
    if n_trials <= 0 or n_bootstrap <= 0:
        return float("nan"), float("nan")
    
    rng = np.random.RandomState(seed)
    p_hat = successes / n_trials
    
    bootstrap_probs = rng.binomial(n_trials, p_hat, size=n_bootstrap) / n_trials
    alpha = 1 - confidence
    
    lo = float(np.percentile(bootstrap_probs, 100 * alpha / 2))
    hi = float(np.percentile(bootstrap_probs, 100 * (1 - alpha / 2)))
    
    return float(max(0.0, lo)), float(min(1.0, hi))


def mc_convergence_diagnostic(
    cumulative_successes: np.ndarray,
    cumulative_trials: np.ndarray,
    window_size: int = 50,
    threshold: float = 0.01,
) -> Dict[str, Any]:
    """Check if Monte Carlo simulation has converged.
    
    Analyzes the stability of the probability estimate over the last
    window_size simulations.
    
    Args:
        cumulative_successes: Array of cumulative success counts
        cumulative_trials: Array of cumulative trial counts
        window_size: Number of recent samples to check stability
        threshold: Convergence threshold for probability change
        
    Returns:
        Dictionary with convergence metrics:
        - converged: bool indicating if converged
        - current_estimate: current probability estimate
        - window_mean: mean estimate over window
        - window_std: std dev over window
        - max_change: max change in window
        - recommendation: string recommendation
    """
    if len(cumulative_successes) != len(cumulative_trials):
        raise ValueError("Arrays must have same length")
    if len(cumulative_trials) < window_size:
        return {
            "converged": False,
            "current_estimate": float("nan"),
            "window_mean": float("nan"),
            "window_std": float("nan"),
            "max_change": float("nan"),
            "recommendation": f"Need at least {window_size} samples for convergence check",
            "samples_collected": len(cumulative_trials),
            "samples_needed": window_size,
        }
    
    probs = cumulative_successes / cumulative_trials
    recent_probs = probs[-window_size:]
    
    window_mean = float(np.mean(recent_probs))
    window_std = float(np.std(recent_probs, ddof=1))
    max_change = float(np.max(np.abs(np.diff(recent_probs))))
    current_estimate = float(probs[-1])
    
    converged = (window_std < threshold) and (max_change < threshold)
    
    if not converged:
        if window_std > threshold:
            recommendation = (
                f"Increase simulations - high variance in recent window "
                f"(std={window_std:.4f} > {threshold})"
            )
        else:
            recommendation = (
                f"Increase simulations - large jumps detected "
                f"(max_change={max_change:.4f} > {threshold})"
            )
    else:
        recommendation = "Simulation appears converged"
    
    return {
        "converged": converged,
        "current_estimate": current_estimate,
        "window_mean": window_mean,
        "window_std": window_std,
        "max_change": max_change,
        "recommendation": recommendation,
        "samples_collected": len(cumulative_trials),
        "window_size": window_size,
    }


def cross_seed_stability(
    results_by_seed: Dict[int, Dict[str, Any]],
    metric_keys: Optional[List[str]] = None,
    threshold_cv: float = 0.10,
) -> Dict[str, Any]:
    """Analyze stability of results across different random seeds.
    
    Args:
        results_by_seed: Dictionary mapping seed -> result dict
        metric_keys: List of metrics to analyze (default: common metrics)
        threshold_cv: Coefficient of variation threshold for stability
        
    Returns:
        Dictionary with stability analysis:
        - stable: bool indicating overall stability
        - metrics: per-metric stability analysis
        - recommendation: string recommendation
    """
    if not results_by_seed:
        return {
            "stable": False,
            "error": "No results provided",
            "recommendation": "Run with multiple seeds for stability analysis",
        }
    
    if metric_keys is None:
        metric_keys = [
            'prob_win', 'prob_loss', 'prob_tp_first', 'prob_sl_first',
            'ev', 'edge', 'kelly', 'prob_no_hit'
        ]
    
    n_seeds = len(results_by_seed)
    metrics_analysis: Dict[str, Dict[str, Any]] = {}
    all_stable = True
    
    for metric in metric_keys:
        values = []
        for seed_result in results_by_seed.values():
            if not isinstance(seed_result, dict):
                continue
            val = seed_result.get(metric)
            if val is not None and np.isfinite(float(val)):
                values.append(float(val))
        
        if len(values) < 2:
            metrics_analysis[metric] = {
                "stable": False,
                "reason": "Insufficient data points",
                "n_values": len(values),
            }
            all_stable = False
            continue
        
        mean_val = float(np.mean(values))
        std_val = float(np.std(values, ddof=1))
        cv = std_val / abs(mean_val) if abs(mean_val) > 1e-10 else float('inf')
        
        stable = cv < threshold_cv
        if not stable:
            all_stable = False
        
        metrics_analysis[metric] = {
            "stable": stable,
            "mean": mean_val,
            "std": std_val,
            "cv": cv,
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "range": float(np.max(values) - np.min(values)),
            "n_seeds": n_seeds,
        }
    
    if all_stable:
        recommendation = (
            f"Results are stable across {n_seeds} seeds (CV < {threshold_cv:.0%}). "
            f"Confidence in estimates is high."
        )
    else:
        unstable_metrics = [
            m for m, analysis in metrics_analysis.items()
            if not analysis.get("stable", False)
        ]
        recommendation = (
            f"Results show instability across seeds for: {', '.join(unstable_metrics)}. "
            f"Consider increasing n_sims or using more seeds."
        )
    
    return {
        "stable": all_stable,
        "n_seeds": n_seeds,
        "threshold_cv": threshold_cv,
        "metrics": metrics_analysis,
        "recommendation": recommendation,
    }


def sensitivity_analysis_single_parameter(
    base_result: Dict[str, Any],
    parameter_name: str,
    parameter_values: List[float],
    evaluate_fn: Any,
    metric_keys: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Analyze sensitivity of results to a single parameter.
    
    Args:
        base_result: Base configuration result
        parameter_name: Name of parameter to vary
        parameter_values: List of parameter values to test
        evaluate_fn: Function to evaluate configuration
        metric_keys: Metrics to track (default: key metrics)
        
    Returns:
        Dictionary with sensitivity analysis results
    """
    if metric_keys is None:
        metric_keys = ['prob_win', 'prob_loss', 'ev', 'edge', 'kelly']
    
    results = []
    for param_val in parameter_values:
        try:
            result = evaluate_fn({parameter_name: param_val})
            if isinstance(result, dict) and result.get('success'):
                results.append({
                    parameter_name: param_val,
                    **{k: result.get('best', {}).get(k) for k in metric_keys}
                })
        except Exception:
            continue
    
    if not results:
        return {
            "success": False,
            "error": "No successful evaluations",
            "parameter": parameter_name,
            "values_tested": len(parameter_values),
        }
    
    sensitivity_metrics: Dict[str, Dict[str, Any]] = {}
    for metric in metric_keys:
        values = [r.get(metric) for r in results if r.get(metric) is not None]
        if len(values) >= 2:
            values_arr = np.array(values, dtype=float)
            sensitivity_metrics[metric] = {
                "min": float(np.nanmin(values_arr)),
                "max": float(np.nanmax(values_arr)),
                "range": float(np.nanmax(values_arr) - np.nanmin(values_arr)),
                "std": float(np.nanstd(values_arr, ddof=1)),
                "mean": float(np.nanmean(values_arr)),
                "elasticity": float(
                    (np.nanmax(values_arr) - np.nanmin(values_arr)) / 
                    (np.nanmean(values_arr) + 1e-10)
                ) if len(values) > 0 else 0.0,
            }
    
    base_values = {k: base_result.get('best', {}).get(k) for k in metric_keys}
    
    return {
        "success": True,
        "parameter": parameter_name,
        "values_tested": len(parameter_values),
        "base_configuration": base_values,
        "sensitivity_metrics": sensitivity_metrics,
        "results": results,
        "recommendation": _sensitivity_recommendation(sensitivity_metrics, parameter_name),
    }


def _sensitivity_recommendation(
    sensitivity_metrics: Dict[str, Dict[str, Any]],
    parameter_name: str,
) -> str:
    """Generate recommendation based on sensitivity analysis."""
    if not sensitivity_metrics:
        return "Insufficient data for sensitivity analysis"
    
    high_sensitivity = []
    for metric, analysis in sensitivity_metrics.items():
        if analysis.get('elasticity', 0) > 0.5:
            high_sensitivity.append(metric)
    
    if high_sensitivity:
        return (
            f"High sensitivity detected for {parameter_name} affecting: "
            f"{', '.join(high_sensitivity)}. "
            f"Parameter selection is critical - optimize carefully."
        )
    else:
        return (
            f"Low sensitivity to {parameter_name} - results are robust "
            f"to parameter variations in the tested range."
        )


def bootstrap_metric_uncertainty(
    paths: np.ndarray,
    tp_trigger: float,
    sl_trigger: float,
    direction: str = 'long',
    entry_price: Optional[float] = None,
    reward: Optional[float] = None,
    risk: Optional[float] = None,
    cost_per_trade: float = 0.0,
    n_bootstrap: int = 500,
    metrics: Optional[List[str]] = None,
    seed: Optional[int] = None,
) -> Dict[str, Dict[str, float]]:
    """Estimate uncertainty of barrier metrics via bootstrap resampling.
    
    Args:
        paths: Price paths array (n_sims x horizon)
        tp_trigger: Take profit trigger price
        sl_trigger: Stop loss trigger price
        direction: 'long' or 'short'
        entry_price: Entry/reference price used to define TP/SL distances
        reward: Optional TP distance in optimizer output units
        risk: Optional SL distance in optimizer output units
        cost_per_trade: Optional trading cost in the same units as reward/risk
        n_bootstrap: Number of bootstrap samples
        metrics: Metrics to estimate (default: all key metrics)
        seed: Random seed
        
    Returns:
        Dictionary mapping metric -> {mean, std, ci_low, ci_high}
    """
    n_sims, horizon = paths.shape
    
    if metrics is None:
        metrics = [
            'prob_win', 'prob_loss', 'prob_tie', 'prob_tp_first', 'prob_sl_first',
            'ev', 'ev_cond', 'edge', 'kelly', 'kelly_cond', 'prob_no_hit', 'prob_resolve'
        ]
    
    rng = np.random.RandomState(seed)
    bootstrap_estimates: Dict[str, List[float]] = {m: [] for m in metrics}
    
    dir_long = (direction == 'long')
    anchor_price = float(entry_price) if entry_price is not None else float(paths[0, 0])
    
    for _ in range(n_bootstrap):
        indices = rng.choice(n_sims, size=n_sims, replace=True)
        bootstrap_paths = paths[indices]
        
        if dir_long:
            hit_tp = (bootstrap_paths >= tp_trigger)
            hit_sl = (bootstrap_paths <= sl_trigger)
        else:
            hit_tp = (bootstrap_paths <= tp_trigger)
            hit_sl = (bootstrap_paths >= sl_trigger)
        
        any_tp = hit_tp.any(axis=1)
        any_sl = hit_sl.any(axis=1)
        
        first_tp = hit_tp.argmax(axis=1)
        first_sl = hit_sl.argmax(axis=1)
        first_tp[~any_tp] = horizon
        first_sl[~any_sl] = horizon
        
        wins = (first_tp < first_sl)
        losses = (first_sl < first_tp)
        ties = (first_tp == first_sl) & (first_tp < horizon)
        
        n_wins = wins.sum()
        n_losses = losses.sum()
        n_ties = ties.sum()
        
        prob_win = n_wins / n_sims
        prob_loss = n_losses / n_sims
        prob_tie = n_ties / n_sims
        prob_tp_first = (n_wins + 0.5 * n_ties) / n_sims
        prob_sl_first = (n_losses + 0.5 * n_ties) / n_sims
        prob_no_hit = 1.0 - (any_tp | any_sl).sum() / n_sims
        prob_resolve = 1.0 - prob_no_hit
        
        edge = prob_win - prob_loss

        if reward is not None and risk is not None:
            gross_reward = float(reward)
            gross_risk = float(risk)
        else:
            gross_reward = abs(tp_trigger - anchor_price)
            gross_risk = abs(sl_trigger - anchor_price)

        net_reward = gross_reward - max(0.0, float(cost_per_trade or 0.0))
        net_risk = gross_risk + max(0.0, float(cost_per_trade or 0.0))
        net_rr = net_reward / net_risk if net_risk > 0 else 0.0
        ev = prob_tp_first * net_reward - prob_sl_first * net_risk
        kelly = prob_tp_first - (prob_sl_first / net_rr) if net_rr > 0 else 0.0

        if prob_resolve > 0:
            prob_win_c = prob_tp_first / prob_resolve
            prob_loss_c = prob_sl_first / prob_resolve
            ev_cond = prob_win_c * net_reward - prob_loss_c * net_risk
            kelly_cond = prob_win_c - (prob_loss_c / net_rr) if net_rr > 0 else 0.0
        else:
            ev_cond = 0.0
            kelly_cond = 0.0
        
        for metric in metrics:
            val = locals().get(metric)
            if val is not None and np.isfinite(val):
                bootstrap_estimates[metric].append(float(val))
    
    results: Dict[str, Dict[str, float]] = {}
    for metric in metrics:
        values = bootstrap_estimates[metric]
        if values:
            values_arr = np.array(values)
            results[metric] = {
                "mean": float(np.mean(values_arr)),
                "std": float(np.std(values_arr, ddof=1)),
                "ci_low": float(np.percentile(values_arr, 2.5)),
                "ci_high": float(np.percentile(values_arr, 97.5)),
                "n_bootstrap": n_bootstrap,
            }
    
    return results


def statistical_power_analysis(
    base_prob: float,
    effect_size: float,
    n_sims: int,
    alpha: float = 0.05,
) -> Dict[str, BarrierPowerAnalysisValue]:
    """Calculate statistical power to detect an effect in barrier probabilities.
    
    Args:
        base_prob: Baseline probability (e.g., prob_win under null)
        effect_size: Minimum detectable effect (absolute difference)
        n_sims: Number of simulations
        alpha: Significance level (default 0.05)
        
    Returns:
        Dictionary with power analysis results
    """
    if not 0 < base_prob < 1:
        return {"error": "base_prob must be in (0, 1)"}
    if effect_size <= 0:
        return {"error": "effect_size must be positive"}
    if n_sims <= 0:
        return {"error": "n_sims must be positive"}
    
    p1 = base_prob
    p2 = base_prob + effect_size
    if p2 >= 1:
        return {
            "error": "Effect size too large - p2 >= 1",
            "p1": p1,
            "p2": p2,
        }
    
    p_pooled = (p1 + p2) / 2
    se_null = math.sqrt(p_pooled * (1 - p_pooled) * 2 / n_sims)
    se_alt = math.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / n_sims)
    
    z_alpha = scipy_stats.norm.ppf(1 - alpha / 2)
    z_beta = (abs(p2 - p1) - z_alpha * se_null) / se_alt
    
    power = scipy_stats.norm.cdf(z_beta)
    
    min_n_for_80_power = int(math.ceil(
        2 * p_pooled * (1 - p_pooled) * 
        (scipy_stats.norm.ppf(0.8) + z_alpha) ** 2 / 
        (effect_size ** 2)
    ))
    
    return {
        "power": float(power),
        "power_percent": float(power * 100),
        "base_probability": p1,
        "alternative_probability": p2,
        "effect_size": effect_size,
        "n_sims": n_sims,
        "alpha": alpha,
        "min_n_for_80_power": min_n_for_80_power,
        "interpretation": (
            f"{power*100:.1f}% power to detect effect size {effect_size:.3f} "
            f"with {n_sims} simulations"
            if power >= 0.8
            else f"Low power ({power*100:.1f}%). Need ~{min_n_for_80_power} sims for 80% power."
        ),
    }


def ensemble_ci_from_multiple_methods(
    method_results: List[Dict[str, Any]],
    metric: str = 'prob_win',
    confidence: float = 0.95,
) -> Dict[str, Any]:
    """Combine confidence intervals from multiple methods/seeds.
    
    Uses bootstrap-t method to combine estimates.
    
    Args:
        method_results: List of result dictionaries from different methods/seeds
        metric: Metric to combine
        confidence: Confidence level
        
    Returns:
        Combined estimate with confidence interval
    """
    if not method_results:
        return {"error": "No method results provided"}
    
    values = []
    for result in method_results:
        if not isinstance(result, dict):
            continue
        best = result.get('best', {})
        if not isinstance(best, dict):
            continue
        val = best.get(metric)
        if val is not None and np.isfinite(float(val)):
            values.append(float(val))
    
    if len(values) < 2:
        return {
            "metric": metric,
            "n_methods": len(values),
            "error": "Need at least 2 methods for ensemble CI",
        }
    
    values_arr = np.array(values)
    mean_val = float(np.mean(values_arr))
    std_val = float(np.std(values_arr, ddof=1))
    se_val = std_val / math.sqrt(len(values))
    
    t_crit = scipy_stats.t.ppf(1 - (1 - confidence) / 2, df=len(values) - 1)
    ci_low = mean_val - t_crit * se_val
    ci_high = mean_val + t_crit * se_val
    
    return {
        "metric": metric,
        "n_methods": len(values),
        "mean": mean_val,
        "std": std_val,
        "se": se_val,
        "ci_low": float(max(0.0, ci_low)),
        "ci_high": float(min(1.0, ci_high)),
        "confidence": confidence,
        "method": "ensemble_t_interval",
    }
