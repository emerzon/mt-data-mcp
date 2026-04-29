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


def _smooth_short_state_runs(
    state: np.ndarray,
    probs: Optional[np.ndarray],
    min_regime_bars: int,
) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
    """Merge short state runs into neighboring regimes to reduce one-bar flicker."""
    state_arr = np.asarray(state, dtype=int).copy()
    probs_arr = np.asarray(probs, dtype=float).copy() if isinstance(probs, np.ndarray) else None
    min_bars = max(1, int(min_regime_bars))
    transitions_before = _count_state_transitions(state_arr)
    if min_bars <= 1 or state_arr.size < 2:
        return state_arr, probs_arr, {
            "min_regime_bars": int(min_bars),
            "smoothing_applied": False,
            "transitions_before": int(transitions_before),
            "transitions_after": int(transitions_before),
        }

    changed = False
    iteration_cap = min(max(1, int(state_arr.size)), 128)
    iterations_run = 0
    for _ in range(iteration_cap):
        runs = _state_runs(state_arr)
        short_candidates = [
            (idx, run)
            for idx, run in enumerate(runs)
            if int(run["length"]) < min_bars and len(runs) > 1
        ]
        if not short_candidates:
            break
        iterations_run += 1
        idx, run = min(
            short_candidates,
            key=lambda item: (int(item[1]["length"]), int(item[1]["start"])),
        )
        left = runs[idx - 1] if idx > 0 else None
        right = runs[idx + 1] if idx + 1 < len(runs) else None
        if left is None and right is None:
            break
        replacement = None
        if left is None:
            replacement = int(right["state"]) if right is not None else None
        elif right is None:
            replacement = int(left["state"])
        else:
            left_state = int(left["state"])
            right_state = int(right["state"])
            if left_state == right_state:
                replacement = left_state
            else:
                left_len = int(left["length"])
                right_len = int(right["length"])
                if left_len > right_len:
                    replacement = left_state
                elif right_len > left_len:
                    replacement = right_state
                else:
                    left_score = 0.0
                    right_score = 0.0
                    if (
                        probs_arr is not None
                        and probs_arr.ndim == 2
                        and probs_arr.shape[0] == state_arr.size
                    ):
                        start = int(run["start"])
                        end = int(run["end"]) + 1
                        if 0 <= left_state < probs_arr.shape[1]:
                            left_score = float(
                                np.nanmean(probs_arr[start:end, left_state])
                            )
                        if 0 <= right_state < probs_arr.shape[1]:
                            right_score = float(
                                np.nanmean(probs_arr[start:end, right_state])
                            )
                    replacement = left_state if left_score >= right_score else right_state

        if replacement is None or int(replacement) == int(run["state"]):
            break
        start = int(run["start"])
        end = int(run["end"]) + 1
        state_arr[start:end] = int(replacement)
        if (
            probs_arr is not None
            and probs_arr.ndim == 2
            and probs_arr.shape[0] == state_arr.size
            and 0 <= int(replacement) < probs_arr.shape[1]
        ):
            probs_arr[start:end, :] = 0.0
            probs_arr[start:end, int(replacement)] = 1.0
        changed = True

    transitions_after = _count_state_transitions(state_arr)
    remaining_short_runs = [
        run for run in _state_runs(state_arr) if int(run["length"]) < min_bars
    ]
    return state_arr, probs_arr, {
        "min_regime_bars": int(min_bars),
        "smoothing_applied": bool(changed),
        "transitions_before": int(transitions_before),
        "transitions_after": int(transitions_after),
        "iterations_run": int(iterations_run),
        "iteration_cap": int(iteration_cap),
        "min_regime_bars_satisfied": len(remaining_short_runs) == 0,
        "remaining_short_runs": len(remaining_short_runs),
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

    if unique_states.size <= 1:
        # Single state — just ensure it's 0
        if unique_states.size == 1 and unique_states[0] != 0:
            old_state = int(unique_states[0])
            state_arr = np.zeros_like(state_arr)
            if probs is not None:
                probs_arr = np.asarray(probs, dtype=float)
                if probs_arr.ndim == 2 and old_state < probs_arr.shape[1]:
                    probs_arr = probs_arr.copy()
                    probs_arr[:, 0] = probs_arr[:, old_state]
                probs = probs_arr
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
        probs_arr = np.asarray(probs, dtype=float)
        if probs_arr.ndim == 2 and probs_arr.shape[1] >= max(unique_states) + 1:
            new_probs = np.empty_like(probs_arr)
            col_map = {old: new for old, new in old_to_new.items()}
            for old_col, new_col in col_map.items():
                if old_col < probs_arr.shape[1] and new_col < probs_arr.shape[1]:
                    new_probs[:, new_col] = probs_arr[:, old_col]
            # Copy any extra columns (shouldn't exist normally)
            mapped_new = set(old_to_new.values())
            for c in range(probs_arr.shape[1]):
                if c not in mapped_new:
                    new_probs[:, c] = probs_arr[:, c]
        else:
            new_probs = probs_arr

    mapping_str = {str(old): int(new) for old, new in old_to_new.items()}
    return new_state, new_probs, {"relabeled": True, "mapping": mapping_str}


__all__ = [
    "_count_state_transitions",
    "_state_runs",
    "_smooth_short_state_runs",
    "_normalize_state_probability_matrix",
    "_canonicalize_regime_labels",
]
