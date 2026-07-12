"""Canonical path-level barrier outcome classification."""

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np


@dataclass(frozen=True)
class BarrierPathOutcomes:
    """Mutually exclusive first-hit outcomes for simulated price paths."""

    first_tp: np.ndarray
    first_sl: np.ndarray
    wins: np.ndarray
    losses: np.ndarray
    ties: np.ndarray
    unresolved: np.ndarray
    time_in_trade: np.ndarray
    horizon: int


def evaluate_barrier_path_outcomes(
    paths: np.ndarray,
    *,
    tp_trigger: float,
    sl_trigger: float,
    direction: Literal["long", "short"],
    extra_tp_hits: Optional[np.ndarray] = None,
    extra_sl_hits: Optional[np.ndarray] = None,
) -> BarrierPathOutcomes:
    """Classify TP-first, SL-first, same-step, and unresolved paths.

    ``extra_*_hits`` allows intra-step detectors such as a Brownian bridge to
    contribute crossings while preserving one canonical first-hit policy.
    """
    eval_paths = np.asarray(paths, dtype=float)
    if eval_paths.ndim != 2:
        raise ValueError("Barrier paths must be a two-dimensional array.")
    sims_total, horizon = eval_paths.shape
    if sims_total <= 0 or horizon <= 0:
        raise ValueError("Barrier paths must contain at least one path and one step.")

    if direction == "long":
        hit_tp = eval_paths >= float(tp_trigger)
        hit_sl = eval_paths <= float(sl_trigger)
    elif direction == "short":
        hit_tp = eval_paths <= float(tp_trigger)
        hit_sl = eval_paths >= float(sl_trigger)
    else:
        raise ValueError("Barrier direction must be 'long' or 'short'.")

    for name, extra, target in (
        ("extra_tp_hits", extra_tp_hits, hit_tp),
        ("extra_sl_hits", extra_sl_hits, hit_sl),
    ):
        if extra is None:
            continue
        extra_arr = np.asarray(extra, dtype=bool)
        if extra_arr.shape != target.shape:
            raise ValueError(
                f"{name} must have shape {target.shape}, got {extra_arr.shape}."
            )
        target |= extra_arr

    any_tp = hit_tp.any(axis=1)
    any_sl = hit_sl.any(axis=1)
    first_tp = hit_tp.argmax(axis=1)
    first_sl = hit_sl.argmax(axis=1)
    first_tp[~any_tp] = horizon
    first_sl[~any_sl] = horizon

    wins = first_tp < first_sl
    losses = first_sl < first_tp
    ties = (first_tp == first_sl) & (first_tp < horizon)
    unresolved = ~(wins | losses | ties)
    time_in_trade = np.minimum(np.minimum(first_tp, first_sl) + 1, horizon)
    return BarrierPathOutcomes(
        first_tp=first_tp,
        first_sl=first_sl,
        wins=wins,
        losses=losses,
        ties=ties,
        unresolved=unresolved,
        time_in_trade=time_in_trade,
        horizon=int(horizon),
    )
