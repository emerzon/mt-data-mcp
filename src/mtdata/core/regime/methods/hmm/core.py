"""HMM (Hidden Markov Model) regime detection."""
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


def _hmm_reliability_from_gamma(gamma: np.ndarray, states: np.ndarray) -> Dict[str, Any]:
    """Estimate HMM reliability from marginal probabilities."""
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
        if np.mean(second_half) < np.mean(first_half) * 0.85:
            trend = "improving"
        elif np.mean(second_half) > np.mean(first_half) * 1.15:
            trend = "degrading"

    return {"confidence": round(reliability, 4), "entropy_trend": trend}


__all__ = ["_hmm_reliability_from_gamma"]
