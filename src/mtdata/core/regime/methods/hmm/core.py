"""HMM (Hidden Markov Model) regime detection."""

from __future__ import annotations

import warnings
from typing import Any, Dict, Optional

import numpy as np

_ENTROPY_TREND_RELATIVE_CHANGE = 0.15
_ENTROPY_TREND_IMPROVING_MULTIPLIER = 1.0 - _ENTROPY_TREND_RELATIVE_CHANGE
_ENTROPY_TREND_DEGRADING_MULTIPLIER = 1.0 + _ENTROPY_TREND_RELATIVE_CHANGE


def _hmm_reliability_from_gamma(gamma: np.ndarray) -> Dict[str, Any]:
    """Estimate HMM reliability from marginal probabilities.

    ``entropy_trend`` compares the mean entropy in the second half of the
    series against the first half. Smaller relative moves stay ``"stable"`` so
    minor noise does not flip the trend label; a materially lower second-half
    entropy means the HMM is becoming more certain (``"improving"``), while a
    materially higher second-half entropy means the fit is becoming less
    certain (``"degrading"``).
    """
    n = int(gamma.shape[0])
    k = int(gamma.shape[1])
    if n == 0 or k == 0:
        return {"confidence": 0.0, "entropy_trend": "unknown"}

    entropies = -np.sum(np.where(gamma > 0, gamma * np.log(np.maximum(gamma, 1e-12)), 0.0), axis=1)
    max_ent = np.log(float(k))
    avg_entropy = float(np.mean(entropies)) if entropies.size else max_ent
    reliability = float(np.clip(1.0 - (avg_entropy / max(max_ent, 1e-6)), 0.0, 1.0))

    first_half = entropies[: n // 2] if n > 1 else entropies
    second_half = entropies[n // 2 :] if n > 1 else entropies
    trend = "stable"
    if first_half.size and second_half.size:
        if np.mean(second_half) < np.mean(first_half) * _ENTROPY_TREND_IMPROVING_MULTIPLIER:
            trend = "improving"
        elif np.mean(second_half) > np.mean(first_half) * _ENTROPY_TREND_DEGRADING_MULTIPLIER:
            trend = "degrading"

    return {"confidence": round(reliability, 4), "entropy_trend": trend}


def fit_temporal_gaussian_hmm_1d(
    x: np.ndarray,
    *,
    n_states: int = 2,
    max_iter: int = 80,
    tol: float = 1e-6,
    seed: Optional[int] = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Fit a true temporal Gaussian HMM and expose regime payload inputs.

    Returns ``(weights, mu, sigma, gamma, metadata)`` where ``gamma`` contains
    smoothed state posteriors for each observation and ``weights`` are the mean
    posterior occupancies. State order is canonicalized to ascending mean so the
    surrounding regime payload keeps its stable label semantics.
    """
    from mtdata.forecast.monte_carlo import (
        _is_hmm_degenerate,
        _normalize_probability_vector,
        _normalize_transition_matrix,
    )
    from hmmlearn.hmm import GaussianHMM

    x = np.asarray(x, dtype=float).reshape(-1)
    x = x[np.isfinite(x)]
    n_obs = int(x.size)
    requested_states = max(1, int(n_states))
    if n_obs < max(4, requested_states + 1):
        raise ValueError("Not enough observations for temporal Gaussian HMM calibration")

    x2 = x.reshape(-1, 1)
    model: Optional[GaussianHMM] = None
    fitted_states = requested_states
    for fitted_states in range(requested_states, 0, -1):
        model = GaussianHMM(
            n_components=fitted_states,
            covariance_type="diag",
            n_iter=int(max_iter),
            tol=float(tol),
            random_state=seed,
            init_params="stc",
        )
        quantiles = np.linspace(0.0, 1.0, fitted_states + 2, dtype=float)[1:-1]
        model.means_ = np.quantile(x, quantiles).reshape(fitted_states, 1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(x2)
        if fitted_states == 1 or not _is_hmm_degenerate(
            model, x2, fitted_states, n_obs
        ):
            break

    if model is None:
        raise RuntimeError("Temporal Gaussian HMM fit did not produce a model")

    gamma = np.asarray(model.predict_proba(x2), dtype=float)
    mu = np.asarray(model.means_, dtype=float).reshape(-1)
    covars = np.asarray(model.covars_, dtype=float).reshape(fitted_states, -1)[:, 0]
    sigma = np.sqrt(np.clip(covars, 1e-12, None))
    weights = _normalize_probability_vector(np.mean(gamma, axis=0))
    transition_matrix = _normalize_transition_matrix(
        np.asarray(model.transmat_, dtype=float)
    )
    initial_state_probabilities = _normalize_probability_vector(
        np.asarray(getattr(model, "startprob_", gamma[0]), dtype=float)
    )
    log_likelihood = float(model.score(x2) * n_obs)

    order = np.argsort(mu)
    gamma = gamma[:, order]
    weights = _normalize_probability_vector(weights[order])
    mu = mu[order]
    sigma = sigma[order]
    transition_matrix = transition_matrix[np.ix_(order, order)]
    initial_state_probabilities = _normalize_probability_vector(
        initial_state_probabilities[order]
    )
    metadata = {
        "backend": "gaussian_hmm",
        "fitted_n_states": int(fitted_states),
        "transition_matrix": transition_matrix,
        "initial_state_probabilities": initial_state_probabilities,
        "log_likelihood": log_likelihood,
    }
    return weights, mu, sigma, gamma, metadata


__all__ = ["_hmm_reliability_from_gamma", "fit_temporal_gaussian_hmm_1d"]
