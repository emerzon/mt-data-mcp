"""Regime detection package - refactored from monolithic regime.py."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _count_state_transitions(state: np.ndarray) -> int:
    """Count number of state transitions in array."""
    if state.size <= 1:
        return 0
    return int(np.sum(state[1:] != state[:-1]))


def _state_runs(state: np.ndarray) -> List[Dict[str, int]]:
    """Extract runs of consecutive states."""
    runs: List[Dict[str, int]] = []
    if state.size == 0:
        return runs
    start = 0
    current = int(state[0])
    for i in range(1, int(state.size)):
        value = int(state[i])
        if value == current:
            continue
        runs.append(
            {
                "start": int(start),
                "end": int(i - 1),
                "state": int(current),
                "length": int(i - start),
            }
        )
        start = i
        current = value
    runs.append(
        {
            "start": int(start),
            "end": int(state.size - 1),
            "state": int(current),
            "length": int(state.size - start),
        }
    )
    return runs


def _confirm_state_changes_causally(
    state: np.ndarray,
    min_regime_bars: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Confirm state changes using past observations only.

    A new raw state must persist for ``min_regime_bars`` consecutive rows
    before it becomes the emitted state. Earlier rows are never revised.
    """
    raw = np.asarray(state, dtype=int)
    if raw.size == 0:
        return raw.copy(), {
            "min_regime_bars": max(1, int(min_regime_bars)),
            "postprocess": "causal_confirmation",
            "transitions_before": 0,
            "transitions_after": 0,
        }
    required = max(1, int(min_regime_bars))
    emitted = np.empty_like(raw)
    confirmed = int(raw[0])
    candidate = confirmed
    candidate_count = 0
    emitted[0] = confirmed
    for idx in range(1, raw.size):
        observed = int(raw[idx])
        if observed == confirmed:
            candidate = confirmed
            candidate_count = 0
        else:
            if observed != candidate:
                candidate = observed
                candidate_count = 1
            else:
                candidate_count += 1
            if candidate_count >= required:
                confirmed = candidate
                candidate_count = 0
        emitted[idx] = confirmed
    before = _count_state_transitions(raw)
    after = _count_state_transitions(emitted)
    return emitted, {
        "min_regime_bars": required,
        "postprocess": "causal_confirmation",
        "smoothing_applied": bool(np.any(emitted != raw)),
        "transitions_before": before,
        "transitions_after": after,
        "pending_state": int(candidate) if candidate != confirmed else None,
        "pending_bars": int(candidate_count),
    }


def _normalize_state_probability_matrix(
    probs: Any,
    *,
    rows: int,
    requested_states: int,
) -> np.ndarray:
    """Return a rectangular state-probability matrix for payload serialization."""
    out = np.zeros((max(0, int(rows)), max(1, int(requested_states))), dtype=float)
    if not isinstance(probs, np.ndarray) or probs.ndim != 2 or probs.shape[0] != rows:
        return out

    cols = min(out.shape[1], probs.shape[1])
    out[:, :cols] = np.asarray(probs[:, :cols], dtype=float)
    return out


def _canonicalize_regime_labels(
    state: np.ndarray,
    probs: Optional[np.ndarray],
    series: np.ndarray,
) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
    """Reorder regime state labels by ascending mean of *series* per state.

    After HMM / MS-AR fitting and smoothing, state IDs are positional
    (depend on random init, fitting order, or smoothing elimination).
    This function produces a canonical numbering where state 0 always
    corresponds to the lowest mean value in *series* (typically the
    most bearish regime when series = returns).

    Also renumbers to eliminate gaps left by smoothing (e.g. {0,1,3} → {0,1,2}).

    Returns (state, probs, meta) where *meta* contains the old→new mapping.
    """
    state_arr = np.asarray(state, dtype=int)
    unique_states = np.unique(state_arr)

    def _remap_probability_columns(
        probs_value: Any,
        old_to_new_mapping: Dict[int, int],
    ) -> np.ndarray:
        probs_arr = np.asarray(probs_value, dtype=float)
        if probs_arr.ndim != 2:
            return probs_arr

        new_probs = np.zeros_like(probs_arr)
        for old_col, new_col in old_to_new_mapping.items():
            if 0 <= old_col < probs_arr.shape[1] and 0 <= new_col < probs_arr.shape[1]:
                new_probs[:, new_col] = probs_arr[:, old_col]

        row_sums = np.sum(new_probs, axis=1, keepdims=True)
        valid_rows = np.isfinite(row_sums[:, 0]) & (row_sums[:, 0] > 0)
        if np.any(valid_rows):
            new_probs[valid_rows] = new_probs[valid_rows] / row_sums[valid_rows]
        return new_probs

    if unique_states.size <= 1:
        # Single state — just ensure it's 0
        if unique_states.size == 1 and unique_states[0] != 0:
            old_state = int(unique_states[0])
            state_arr = np.zeros_like(state_arr)
            if probs is not None:
                probs = _remap_probability_columns(probs, {old_state: 0})
            return state_arr, probs, {
                "relabeled": True,
                "mapping": {str(old_state): 0},
            }
        return state_arr, probs, {"relabeled": False, "mapping": {}}

    series_arr = np.asarray(series, dtype=float)
    n = min(len(state_arr), len(series_arr))
    state_arr_trimmed = state_arr[:n]
    series_trimmed = series_arr[:n]

    # Compute mean series value per state
    means: Dict[int, float] = {}
    for s in unique_states:
        mask = state_arr_trimmed == s
        if mask.any():
            means[int(s)] = float(np.nanmean(series_trimmed[mask]))
        else:
            means[int(s)] = 0.0

    # Sort by mean value (ascending)
    sorted_states = sorted(means.keys(), key=lambda s: means[s])
    old_to_new = {old: new for new, old in enumerate(sorted_states)}

    # Check if already canonical
    is_identity = all(old == new for old, new in old_to_new.items())
    if is_identity:
        return state_arr, probs, {"relabeled": False, "mapping": {}}

    # Apply relabeling
    new_state = np.empty_like(state_arr)
    for old, new in old_to_new.items():
        new_state[state_arr == old] = new

    new_probs: Optional[np.ndarray] = None
    if probs is not None:
        new_probs = _remap_probability_columns(probs, old_to_new)

    mapping_str = {str(old): int(new) for old, new in old_to_new.items()}
    return new_state, new_probs, {"relabeled": True, "mapping": mapping_str}


__all__ = [
    "_count_state_transitions",
    "_state_runs",
    "_confirm_state_changes_causally",
    "_normalize_state_probability_matrix",
    "_canonicalize_regime_labels",
]
