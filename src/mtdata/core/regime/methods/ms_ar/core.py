"""MS-AR (Markov-Switching AutoRegressive) regime detection."""
from typing import Any, Dict

import numpy as np


def _ms_ar_reliability_from_smoothed(
    smoothed_probs: np.ndarray,
    params_used: Dict[str, Any],
) -> Dict[str, Any]:
    """Estimate MS-AR reliability from smoothed marginal probabilities."""
    n = int(smoothed_probs.shape[0])
    k = int(smoothed_probs.shape[1])
    if n == 0 or k == 0:
        return {"confidence": 0.0, "notes": "empty_probs"}

    # Average assignment confidence (max prob per observation)
    max_probs = np.max(smoothed_probs, axis=1)
    avg_confidence = float(np.mean(max_probs))

    # Check for absorbing states (prob == 1.0)
    absorbing_count = int(np.sum(smoothed_probs >= 0.9999))

    # Transition matrix precision estimate
    notes = "ok"
    if avg_confidence < 0.55:
        notes = "low_confidence"
    elif absorbing_count > n * 0.5:
        notes = "many_absorbing"

    return {
        "confidence": round(float(np.clip(avg_confidence, 0.0, 1.0)), 4),
        "notes": notes,
        "k_regimes": int(params_used.get("k_regimes", k)),
        "order": int(params_used.get("order", 0)),
    }


__all__ = ["_ms_ar_reliability_from_smoothed"]
