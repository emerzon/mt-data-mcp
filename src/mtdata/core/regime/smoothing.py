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
        short_runs = [idx for idx, run in enumerate(runs) if int(run["length"]) < min_bars]
        if not short_runs:
            break
        iterations_run += 1
        pass_changed = False
        for idx in short_runs:
            run = runs[idx]
            left = runs[idx - 1] if idx > 0 else None
            right = runs[idx + 1] if idx + 1 < len(runs) else None
            if left is None and right is None:
                continue
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
                                left_score = float(np.nanmean(probs_arr[start:end, left_state]))
                            if 0 <= right_state < probs_arr.shape[1]:
                                right_score = float(np.nanmean(probs_arr[start:end, right_state]))
                        replacement = left_state if left_score >= right_score else right_state

            if replacement is None or int(replacement) == int(run["state"]):
                continue
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
            pass_changed = True
            changed = True
        if not pass_changed:
            break

    transitions_after = _count_state_transitions(state_arr)
    return state_arr, probs_arr, {
        "min_regime_bars": int(min_bars),
        "smoothing_applied": bool(changed),
        "transitions_before": int(transitions_before),
        "transitions_after": int(transitions_after),
        "iterations_run": int(iterations_run),
        "iteration_cap": int(iteration_cap),
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


__all__ = [
    "_count_state_transitions",
    "_state_runs",
    "_smooth_short_state_runs",
    "_normalize_state_probability_matrix",
]
