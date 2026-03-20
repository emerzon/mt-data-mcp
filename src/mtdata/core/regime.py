from typing import Any, Dict, Optional, List, Literal, Tuple
import logging
import time
import numpy as np

from ._mcp_instance import mcp
from .execution_logging import (
    infer_result_success,
    log_operation_finish,
    log_operation_start,
)
from .mt5_gateway import get_mt5_gateway, mt5_connection_error
from .schema import TimeframeLiteral, DenoiseSpec
from .constants import TIMEFRAME_SECONDS
from ..forecast.common import fetch_history as _fetch_history
from ..utils.utils import _format_time_minimal
from ..utils.denoise import _resolve_denoise_base_col
from ..utils.mt5 import ensure_mt5_connection_or_raise

logger = logging.getLogger(__name__)


_CRYPTO_SYMBOL_HINTS = (
    "BTC",
    "ETH",
    "XRP",
    "LTC",
    "BCH",
    "DOGE",
    "SOL",
    "ADA",
    "DOT",
    "AVAX",
    "BNB",
    "TRX",
    "LINK",
    "MATIC",
)

_MAX_SHORT_STATE_SMOOTHING_PASSES = 128


def _regime_connection_error() -> Optional[Dict[str, Any]]:
    return mt5_connection_error(
        get_mt5_gateway(ensure_connection_impl=ensure_mt5_connection_or_raise),
    )


def _count_state_transitions(state: np.ndarray) -> int:
    if state.size <= 1:
        return 0
    return int(np.sum(state[1:] != state[:-1]))


def _state_runs(state: np.ndarray) -> List[Dict[str, int]]:
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
) -> tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
    """Merge short state runs into neighboring regimes to reduce one-bar flicker."""
    state_arr = np.asarray(state, dtype=int).copy()
    probs_arr = (
        np.asarray(probs, dtype=float).copy() if isinstance(probs, np.ndarray) else None
    )
    min_bars = max(1, int(min_regime_bars))
    transitions_before = _count_state_transitions(state_arr)
    if min_bars <= 1 or state_arr.size < 2:
        return (
            state_arr,
            probs_arr,
            {
                "min_regime_bars": int(min_bars),
                "smoothing_applied": False,
                "transitions_before": int(transitions_before),
                "transitions_after": int(transitions_before),
            },
        )

    changed = False
    iteration_cap = min(max(1, int(state_arr.size)), _MAX_SHORT_STATE_SMOOTHING_PASSES)
    iterations_run = 0
    for _ in range(iteration_cap):
        runs = _state_runs(state_arr)
        short_runs = [
            idx for idx, run in enumerate(runs) if int(run["length"]) < min_bars
        ]
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
                                left_score = float(
                                    np.nanmean(probs_arr[start:end, left_state])
                                )
                            if 0 <= right_state < probs_arr.shape[1]:
                                right_score = float(
                                    np.nanmean(probs_arr[start:end, right_state])
                                )
                        replacement = (
                            left_state if left_score >= right_score else right_state
                        )

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
    return (
        state_arr,
        probs_arr,
        {
            "min_regime_bars": int(min_bars),
            "smoothing_applied": bool(changed),
            "transitions_before": int(transitions_before),
            "transitions_after": int(transitions_after),
            "iterations_run": int(iterations_run),
            "iteration_cap": int(iteration_cap),
        },
    )


def _is_probably_crypto_symbol(symbol: Any) -> bool:
    s = str(symbol or "").upper().strip()
    if not s:
        return False
    normalized = "".join(ch for ch in s if ch.isalnum())
    if not normalized:
        return False
    return any(token in normalized for token in _CRYPTO_SYMBOL_HINTS)


def _default_bocpd_hazard_lambda(symbol: Any, timeframe: Any) -> int:
    tf = str(timeframe or "H1").upper().strip() or "H1"
    tf_seconds = int(TIMEFRAME_SECONDS.get(tf, 3600))

    if _is_probably_crypto_symbol(symbol):
        if tf_seconds <= 900:  # <= M15
            return 48
        if tf_seconds <= 3600:  # <= H1
            return 72
        if tf_seconds <= 14400:  # <= H4
            return 96
        if tf_seconds <= 86400:  # <= D1
            return 128
        return 180

    return 250


def _default_bocpd_cp_threshold(symbol: Any, timeframe: Any) -> float:
    tf = str(timeframe or "H1").upper().strip() or "H1"
    tf_seconds = int(TIMEFRAME_SECONDS.get(tf, 3600))
    if _is_probably_crypto_symbol(symbol):
        if tf_seconds <= 3600:  # <= H1
            return 0.35
        if tf_seconds <= 14400:  # <= H4
            return 0.40
        return 0.45
    return 0.50


def _auto_calibrate_bocpd_params(
    returns: np.ndarray,
    symbol: Any,
    timeframe: Any,
) -> Tuple[int, float, Dict[str, Any]]:
    """Auto-calibrate BOCPD hazard/threshold from recent return distribution."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    base_lambda = int(_default_bocpd_hazard_lambda(symbol, timeframe))
    base_threshold = float(_default_bocpd_cp_threshold(symbol, timeframe))
    if r.size < 30:
        return (
            base_lambda,
            base_threshold,
            {
                "calibrated": False,
                "reason": "insufficient_points",
                "points": int(r.size),
                "base_hazard_lambda": int(base_lambda),
                "base_cp_threshold": float(base_threshold),
            },
        )

    mu = float(np.mean(r))
    sigma = float(np.std(r, ddof=0))
    sigma_safe = sigma if sigma > 1e-12 else 1e-12
    centered = r - mu
    z = centered / sigma_safe
    kurt = float(np.mean(z**4) - 3.0) if z.size > 0 else 0.0
    jump_share = float(np.mean(np.abs(z) >= 2.5)) if z.size > 0 else 0.0
    trend_strength = float(abs(mu) / sigma_safe)
    move_zscore = float(abs(np.sum(r)) / (sigma_safe * np.sqrt(max(1, int(r.size)))))

    vol_norm = float(np.clip(sigma / 0.003, 0.0, 3.0))
    kurt_norm = float(np.clip(max(0.0, kurt) / 6.0, 0.0, 2.0))
    jump_norm = float(np.clip(jump_share / 0.08, 0.0, 2.0))
    trend_norm = float(np.clip(trend_strength / 0.20, 0.0, 2.0))
    move_sig_norm = float(np.clip(move_zscore / 3.0, 0.0, 2.5))

    sensitivity = float(
        np.clip(
            1.0
            + 0.35 * vol_norm
            + 0.25 * kurt_norm
            + 0.25 * jump_norm
            - 0.20 * trend_norm
            + 0.60 * move_sig_norm,
            0.5,
            2.2,
        )
    )

    hazard_floor = max(12, int(round(base_lambda * 0.25)))
    hazard_cap = min(500, max(hazard_floor + 1, int(round(base_lambda * 1.80))))
    hazard_lambda = int(
        np.clip(int(round(base_lambda / sensitivity)), hazard_floor, hazard_cap)
    )

    cp_threshold = float(
        np.clip(
            base_threshold
            - 0.08 * (vol_norm / 3.0)
            - 0.06 * (jump_norm / 2.0)
            - 0.04 * (kurt_norm / 2.0)
            + 0.04 * (trend_norm / 2.0),
            0.15,
            0.75,
        )
    )
    if move_sig_norm > 0.0:
        cp_threshold = float(
            np.clip(cp_threshold - 0.08 * (move_sig_norm / 2.5), 0.15, 0.75)
        )

    diagnostics = {
        "calibrated": True,
        "points": int(r.size),
        "base_hazard_lambda": int(base_lambda),
        "base_cp_threshold": float(base_threshold),
        "asset_class_hint": "crypto" if _is_probably_crypto_symbol(symbol) else "other",
        "sigma": float(sigma),
        "kurtosis_excess": float(kurt),
        "jump_share_abs_z_ge_2_5": float(jump_share),
        "trend_strength": float(trend_strength),
        "move_zscore": float(move_zscore),
        "vol_norm": float(vol_norm),
        "kurt_norm": float(kurt_norm),
        "jump_norm": float(jump_norm),
        "trend_norm": float(trend_norm),
        "move_sig_norm": float(move_sig_norm),
        "sensitivity": float(sensitivity),
        "hazard_floor": int(hazard_floor),
        "hazard_cap": int(hazard_cap),
    }
    return int(hazard_lambda), float(cp_threshold), diagnostics


def _walkforward_quantile_threshold_calibration(
    series: np.ndarray,
    hazard_lambda: int,
    base_threshold: float,
    *,
    target_false_alarm_rate: float = 0.02,
    window: Optional[int] = None,
    step: Optional[int] = None,
    max_windows: int = 6,
    bootstrap_runs: int = 2,
    seed: int = 42,
) -> Tuple[float, Dict[str, Any]]:
    """Calibrate CP threshold from null BOCPD maxima over walk-forward windows."""
    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]
    diagnostics: Dict[str, Any] = {
        "mode": "walkforward_quantile",
        "calibrated": False,
        "points": int(x.size),
        "target_false_alarm_rate": float(np.clip(target_false_alarm_rate, 1e-4, 0.25)),
        "base_threshold": float(base_threshold),
    }
    if x.size < 120:
        diagnostics["reason"] = "insufficient_points"
        return float(base_threshold), diagnostics

    win = int(window) if window is not None else int(min(240, max(80, x.size // 3)))
    win = int(np.clip(win, 60, max(60, x.size)))
    stp = int(step) if step is not None else int(max(20, win // 3))
    stp = int(max(10, stp))
    max_w = int(max(1, max_windows))
    n_boot = int(max(1, bootstrap_runs))
    q = float(np.clip(1.0 - diagnostics["target_false_alarm_rate"], 0.50, 0.999))

    starts = list(range(0, max(1, x.size - win + 1), stp))
    if not starts:
        starts = [max(0, x.size - win)]
    starts = starts[-max_w:]

    try:
        from ..utils.regime import bocpd_gaussian

        rng = np.random.default_rng(int(seed))
        null_maxima: List[float] = []
        for s in starts:
            seg = x[int(s) : int(s + win)]
            if seg.size < 30:
                continue
            rl = int(min(1000, seg.size))
            for _ in range(n_boot):
                shuffled = rng.permutation(seg)
                r = bocpd_gaussian(
                    shuffled,
                    hazard_lambda=int(max(1, hazard_lambda)),
                    max_run_length=rl,
                )
                cp = np.asarray(r.get("cp_prob", []), dtype=float)
                cp = cp[np.isfinite(cp)]
                if cp.size:
                    null_maxima.append(float(np.nanmax(cp)))
        if not null_maxima:
            diagnostics["reason"] = "no_null_scores"
            return float(base_threshold), diagnostics

        null_q = float(np.quantile(np.asarray(null_maxima, dtype=float), q))
        calibrated = float(np.clip(max(float(base_threshold), null_q), 0.15, 0.90))
        diagnostics.update(
            {
                "calibrated": True,
                "window": int(win),
                "step": int(stp),
                "windows_used": int(len(starts)),
                "bootstrap_runs": int(n_boot),
                "null_scores_count": int(len(null_maxima)),
                "null_max_quantile": float(null_q),
                "quantile": float(q),
                "threshold_delta": float(calibrated - float(base_threshold)),
            }
        )
        return calibrated, diagnostics
    except Exception as ex:
        diagnostics["reason"] = "calibration_error"
        diagnostics["error"] = str(ex)
        return float(base_threshold), diagnostics


def _filter_bocpd_change_points(
    cp_prob: np.ndarray,
    threshold: float,
    *,
    min_distance_bars: int = 5,
    min_regime_bars: int = 5,
    confirm_bars: int = 1,
    confirm_relaxed_mult: float = 0.90,
    edge_multiplier: float = 1.08,
) -> Tuple[List[int], Dict[str, Any]]:
    """Filter CP candidates with confirmation, cooldown, and edge guards."""
    cp = np.asarray(cp_prob, dtype=float)
    cp = np.where(np.isfinite(cp), cp, np.nan)
    n = int(cp.size)
    raw_idx = [
        int(i)
        for i, v in enumerate(cp.tolist())
        if np.isfinite(v) and float(v) >= float(threshold)
    ]
    min_dist = int(max(1, min_distance_bars))
    min_regime = int(max(1, min_regime_bars))
    conf = int(max(1, confirm_bars))
    relaxed = float(threshold) * float(np.clip(confirm_relaxed_mult, 0.5, 1.0))
    edge_thr = float(threshold) * float(max(1.0, edge_multiplier))
    accepted: List[int] = []

    rejects = {
        "cooldown": 0,
        "left_boundary": 0,
        "confirmation": 0,
        "edge_threshold": 0,
        "edge_support": 0,
    }

    for idx in raw_idx:
        if idx < min_regime:
            rejects["left_boundary"] += 1
            continue
        if accepted and (idx - accepted[-1]) < min_dist:
            rejects["cooldown"] += 1
            continue

        bars_to_end = n - idx
        in_edge_zone = bars_to_end < min_regime
        if in_edge_zone:
            if not np.isfinite(cp[idx]) or float(cp[idx]) < edge_thr:
                rejects["edge_threshold"] += 1
                continue
            support_start = max(0, idx - conf + 1)
            support_window = cp[support_start : idx + 1]
            support_count = int(
                np.sum(np.asarray(support_window, dtype=float) >= relaxed)
            )
            need = int(min(conf, support_window.size))
            if support_count < need:
                rejects["edge_support"] += 1
                continue
        else:
            fwd = cp[idx : min(n, idx + conf)]
            support_count = int(np.sum(np.asarray(fwd, dtype=float) >= relaxed))
            need = int(min(conf, fwd.size))
            if support_count < need:
                rejects["confirmation"] += 1
                continue
        accepted.append(int(idx))

    diagnostics = {
        "raw_candidates_count": int(len(raw_idx)),
        "accepted_count": int(len(accepted)),
        "filtered_count": int(len(raw_idx) - len(accepted)),
        "min_distance_bars": int(min_dist),
        "min_regime_bars": int(min_regime),
        "confirm_bars": int(conf),
        "confirm_relaxed_mult": float(np.clip(confirm_relaxed_mult, 0.5, 1.0)),
        "edge_multiplier": float(max(1.0, edge_multiplier)),
        "reject_reasons": rejects,
    }
    return accepted, diagnostics


def _bocpd_reliability_score(
    cp_prob: np.ndarray,
    cp_indices: List[int],
    *,
    threshold: float,
    lookback: int,
    min_regime_bars: int,
    expected_false_alarm_rate: float,
    calibration_age_bars: int,
    threshold_calibrated: bool,
) -> Dict[str, Any]:
    """Estimate BOCPD reliability from margins, edge concentration, and calibration."""
    cp = np.asarray(cp_prob, dtype=float)
    cp = cp[np.isfinite(cp)]
    n = int(cp.size)
    if n == 0:
        return {
            "confidence": 0.0,
            "reliability_label": "low",
            "expected_false_alarm_rate": float(
                np.clip(expected_false_alarm_rate, 1e-4, 0.25)
            ),
            "calibration_age_bars": int(max(0, calibration_age_bars)),
            "threshold_margin": 0.0,
            "recent_cp_density": 0.0,
            "edge_cp_share": 0.0,
            "threshold_calibrated": bool(threshold_calibrated),
        }

    lb = int(max(1, min(int(lookback), n)))
    start = n - lb
    tail = cp[-lb:]
    cps_recent = [int(i) for i in cp_indices if int(i) >= start]
    edge_zone = int(max(1, min_regime_bars))
    edge_count = int(sum(1 for i in cps_recent if int(i) >= (n - edge_zone)))
    edge_share = float(edge_count / max(1, len(cps_recent))) if cps_recent else 0.0
    density = float(len(cps_recent) / float(lb))
    peak = float(np.nanmax(tail)) if tail.size else 0.0
    margin = float(max(0.0, peak - float(threshold)))

    target_fa = float(np.clip(expected_false_alarm_rate, 1e-4, 0.25))
    margin_factor = float(np.clip(margin / 0.15, 0.0, 1.0))
    edge_factor = float(np.clip(1.0 - edge_share, 0.0, 1.0))
    density_penalty = float(
        np.clip(abs(density - target_fa) / max(target_fa, 1e-6), 0.0, 1.0)
    )
    calibration_factor = 1.0 if bool(threshold_calibrated) else 0.6
    score = float(
        np.clip(
            0.45 * margin_factor
            + 0.30 * edge_factor
            + 0.15 * (1.0 - density_penalty)
            + 0.10 * calibration_factor,
            0.0,
            1.0,
        )
    )
    label = "high" if score >= 0.75 else ("medium" if score >= 0.45 else "low")
    return {
        "confidence": float(score),
        "reliability_label": label,
        "expected_false_alarm_rate": float(target_fa),
        "calibration_age_bars": int(max(0, calibration_age_bars)),
        "threshold_margin": float(margin),
        "recent_cp_density": float(density),
        "edge_cp_share": float(edge_share),
        "threshold_calibrated": bool(threshold_calibrated),
    }


def _consolidate_payload(
    payload: Dict[str, Any], method: str, output_mode: str, include_series: bool = False
) -> Dict[str, Any]:
    """Consolidate time series into regime segments and restructure payload."""
    try:
        times = payload.get("times")
        if not times or not isinstance(times, list):
            return payload

        # Prepare consolidation
        segments: List[Dict[str, Any]] = []

        # Extract states/regimes
        states = []
        probs = []

        if method == "bocpd":
            # For BOCPD, we define regimes by change points
            # We can create a 'regime_id' that increments at each CP
            # We also look at 'change_points' list in payload
            cps_idx = set()
            if "change_points" in payload and isinstance(
                payload["change_points"], list
            ):
                for cp in payload["change_points"]:
                    if isinstance(cp, dict) and "idx" in cp:
                        cps_idx.add(cp["idx"])

            curr_regime = 0
            # Reconstruct per-step state
            for i in range(len(times)):
                if i in cps_idx:
                    curr_regime += 1
                states.append(curr_regime)

            # Probs
            raw_probs = payload.get("cp_prob")
            if isinstance(raw_probs, list):
                probs = raw_probs
            else:
                probs = [0.0] * len(times)

        elif method in ("ms_ar", "hmm", "clustering"):
            raw_state = payload.get("state")
            if isinstance(raw_state, list):
                states = raw_state

            # Probs
            # structure is usually list of lists [ [p0, p1...], ... ]
            raw_probs = payload.get("state_probabilities")
            # We might just store the max prob or the prob of the current state?
            if isinstance(raw_probs, list) and raw_probs:
                if isinstance(raw_probs[0], list):
                    # Pick prob of selected state
                    for s, p_vec in zip(states, raw_probs):
                        if isinstance(p_vec, list) and 0 <= s < len(p_vec):
                            probs.append(p_vec[s])
                        else:
                            probs.append(None)
                else:
                    probs = raw_probs  # Should not happen based on current logic but safe fallback

        if not states or len(states) != len(times):
            # Fallback if creation failed
            return payload

        filtered_entries = []
        for idx, state in enumerate(states):
            try:
                state_value = int(state)
            except Exception:
                state_value = None
            if state_value is None or state_value < 0:
                continue
            prob_value = probs[idx] if idx < len(probs) else None
            filtered_entries.append((times[idx], state_value, prob_value))
        if filtered_entries and len(filtered_entries) != len(times):
            times = [entry[0] for entry in filtered_entries]
            states = [entry[1] for entry in filtered_entries]
            probs = [entry[2] for entry in filtered_entries]

        # Consolidate
        # Loop through
        curr_start = times[0]
        curr_state = states[0]
        curr_prob_sum = 0.0
        curr_count = 0

        i = 0
        while i < len(times):
            t = times[i]
            s = states[i]
            p = probs[i] if i < len(probs) and probs[i] is not None else 0.0

            # Check change (state change)
            # For BOCPD, 's' changes exactly at CP.
            if s != curr_state and curr_count > 0:
                # close segment
                avg_prob = curr_prob_sum / max(1, curr_count)
                segments.append(
                    {
                        "start": curr_start,
                        "end": times[i - 1] if i > 0 else curr_start,
                        "duration": curr_count,
                        "regime": curr_state,  # state ID or regime ID
                        "confidence": avg_prob,  # average prob of being in this state/regime (for HMM) or CP prob (BOCPD - meaningless inside segment usually)
                    }
                )
                # New segment
                curr_start = t
                curr_state = s
                curr_prob_sum = 0.0
                curr_count = 0

            curr_prob_sum += p
            curr_count += 1
            i += 1

        # Final segment
        if curr_count > 0:
            avg_prob = curr_prob_sum / max(1, curr_count)
            segments.append(
                {
                    "start": curr_start,
                    "end": times[-1],
                    "duration": curr_count,
                    "regime": curr_state,
                    "confidence": avg_prob,
                }
            )

        # Post-process segments for readability
        # For BOCPD, 'confidence' is avg cp_prob which is usually low except at edges.
        # Maybe we want the PEAK prob? or just drop it.
        # For HMM, 'confidence' is avg prob of that state.

        final_segments = []
        for i, seg in enumerate(segments):
            row = {
                "start": seg["start"],
                "end": seg["end"],
                "bars": seg["duration"],
                "regime": seg["regime"],
            }
            if method != "bocpd":
                row["avg_conf"] = round(seg["confidence"], 4)
            final_segments.append(row)

        # Restructure Payload
        # We want 'regimes' to be the MAIN table.
        # We want to hide raw series under 'series' if output='full'.

        new_payload = {
            "symbol": payload.get("symbol"),
            "timeframe": payload.get("timeframe"),
            "method": payload.get("method"),
            "success": True,
        }

        if "threshold" in payload:
            new_payload["threshold"] = payload["threshold"]
        if "reliability" in payload:
            new_payload["reliability"] = payload["reliability"]
        if "tuning_hint" in payload:
            new_payload["tuning_hint"] = payload["tuning_hint"]

        # Add consolidated table
        new_payload["regimes"] = final_segments

        # Handle raw series
        if output_mode == "full" and include_series:
            series_data = {}
            for k in [
                "times",
                "cp_prob",
                "state",
                "state_probabilities",
                "change_points",
            ]:
                if k in payload:
                    series_data[k] = payload[k]
            new_payload["series"] = series_data
        elif output_mode == "compact" and include_series:
            # Maybe keep tail of series in 'series'?
            series_data = {}
            for k in ["times", "cp_prob", "state"]:
                if k in payload:
                    series_data[k] = payload[k]  # Already truncated by caller?
            new_payload["series"] = series_data

        # Add params
        if "params_used" in payload:
            new_payload["params_used"] = payload["params_used"]

        return new_payload

    except Exception as e:
        # Fallback to original payload on error
        payload["consolidation_error"] = str(e)
        payload["success"] = False
        return payload


def _summary_only_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Return a minimal payload for `output='summary'` (no regimes/series)."""
    out: Dict[str, Any] = {
        "symbol": payload.get("symbol"),
        "timeframe": payload.get("timeframe"),
        "method": payload.get("method"),
        "target": payload.get("target"),
        "success": bool(payload.get("success", True)),
    }
    if "summary" in payload:
        out["summary"] = payload["summary"]
    if "reliability" in payload:
        out["reliability"] = payload["reliability"]
    if "params_used" in payload:
        out["params_used"] = payload["params_used"]
    if "threshold" in payload:
        out["threshold"] = payload["threshold"]
    if "tuning_hint" in payload:
        out["tuning_hint"] = payload["tuning_hint"]
    return out


@mcp.tool()
def regime_detect(
    symbol: str,
    timeframe: TimeframeLiteral = "H1",
    limit: int = 800,
    method: Literal["bocpd", "hmm", "ms_ar"] = "bocpd",  # type: ignore
    target: Literal["return", "price"] = "return",  # type: ignore
    params: Optional[Dict[str, Any]] = None,
    denoise: Optional[DenoiseSpec] = None,
    threshold: float = 0.5,
    output: Literal["full", "summary", "compact"] = "compact",  # type: ignore
    lookback: int = 300,
    include_series: bool = False,
    min_regime_bars: int = 5,
) -> Dict[str, Any]:
    """Detect regimes and/or change-points over the last `limit` bars.

    - method: 'bocpd' (Bayesian online change-point; Gaussian), 'hmm' (Gaussian mixture/HMM-lite), or 'ms_ar' (Markov-switching AR).
    - params (bocpd): optional `hazard_mode` = auto_default|auto_calibrated (defaults to auto_calibrated).
      Explicit `hazard_lambda` / `cp_threshold` always take precedence over auto selection.
      Optional robustness params:
        `cp_threshold_calibration_mode` (default `walkforward_quantile`),
        `threshold_target_false_alarm_rate`,
        `cp_confirm_bars` (default `1`, live-oriented),
        `min_cp_distance_bars`, `cp_edge_multiplier`.
    - include_series: If True, include raw time series data (probs, states) in output even if output='full'. Default False.
    - min_regime_bars: Merge short state runs (< this many bars) for state-based methods to reduce flicker.
    - output:
        - 'compact' (default): Returns recent consolidated 'regimes' and method summary.
        - 'full': Returns full consolidated 'regimes'. Raw 'series' included only if include_series=True.
        - 'summary': Returns stats only.
    """
    started_at = time.perf_counter()
    log_operation_start(
        logger,
        operation="regime_detect",
        symbol=symbol,
        timeframe=timeframe,
        method=method,
        target=target,
        output=output,
        limit=limit,
    )

    def _finish(result: Dict[str, Any]) -> Dict[str, Any]:
        log_operation_finish(
            logger,
            operation="regime_detect",
            started_at=started_at,
            success=infer_result_success(result),
            symbol=symbol,
            timeframe=timeframe,
            method=method,
            target=target,
            output=output,
            limit=limit,
        )
        return result

    connection_error = _regime_connection_error()
    if connection_error is not None:
        return _finish(connection_error)
    try:
        p = dict(params or {})
        try:
            min_regime_bars_val = int(p.get("min_regime_bars", min_regime_bars))
        except Exception:
            return _finish({"error": "min_regime_bars must be an integer >= 1."})
        if min_regime_bars_val < 1:
            return _finish({"error": "min_regime_bars must be >= 1."})
        df = _fetch_history(symbol, timeframe, int(max(limit, 50)), as_of=None)
        if len(df) < 10:
            return _finish({"error": "Insufficient history"})
        base_col = _resolve_denoise_base_col(
            df, denoise, base_col="close", default_when="pre_ti"
        )
        y = df[base_col].astype(float).to_numpy()
        times = df["time"].astype(float).to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            calibration_returns = np.diff(np.log(np.maximum(y, 1e-12)))
        calibration_returns = calibration_returns[np.isfinite(calibration_returns)]
        if target == "return":
            with np.errstate(divide="ignore", invalid="ignore"):
                x = np.diff(np.log(np.maximum(y, 1e-12)))
            x = x[np.isfinite(x)]
            t = times[1 : 1 + x.size]
        else:
            x = y[np.isfinite(y)]
            t = times[: x.size]

        if x.size < 2:
            return _finish({"error": "Insufficient finite observations after filter"})

        # format times
        t_fmt = [_format_time_minimal(tt) for tt in t]

        if method == "bocpd":
            from ..utils.regime import bocpd_gaussian

            hazard_mode = (
                str(p.get("hazard_mode", "auto_calibrated") or "auto_calibrated")
                .strip()
                .lower()
            )
            if hazard_mode in {"auto", "calibrated"}:
                hazard_mode = "auto_calibrated"
            if hazard_mode not in {"auto_default", "auto_calibrated"}:
                hazard_mode = "auto_calibrated"

            hazard_src = "params"
            threshold_src = "arg"
            calibration_info: Optional[Dict[str, Any]] = None
            threshold_calibration_info: Optional[Dict[str, Any]] = None

            auto_hazard = _default_bocpd_hazard_lambda(symbol, timeframe)
            auto_threshold = _default_bocpd_cp_threshold(symbol, timeframe)
            if hazard_mode == "auto_calibrated":
                auto_hazard, auto_threshold, calibration_info = (
                    _auto_calibrate_bocpd_params(
                        returns=calibration_returns, symbol=symbol, timeframe=timeframe
                    )
                )

            if "hazard_lambda" in p and p.get("hazard_lambda") is not None:
                hazard_lambda = int(p.get("hazard_lambda"))
            else:
                hazard_lambda = int(auto_hazard)
                hazard_src = (
                    "auto_calibrated"
                    if hazard_mode == "auto_calibrated"
                    else "auto_default"
                )
            if "cp_threshold" in p and p.get("cp_threshold") is not None:
                threshold_used = float(p.get("cp_threshold"))
                threshold_src = "params.cp_threshold"
            elif "threshold" in p and p.get("threshold") is not None:
                threshold_used = float(p.get("threshold"))
                threshold_src = "params.threshold"
            else:
                if abs(float(threshold) - 0.5) <= 1e-12:
                    threshold_used = float(auto_threshold)
                    threshold_src = (
                        "auto_calibrated"
                        if hazard_mode == "auto_calibrated"
                        else "auto_default"
                    )
                else:
                    threshold_used = float(threshold)
                    threshold_src = "arg"
            max_rl = int(p.get("max_run_length", min(1000, x.size)))
            threshold_cal_mode = (
                str(
                    p.get("cp_threshold_calibration_mode", "walkforward_quantile")
                    or "walkforward_quantile"
                )
                .strip()
                .lower()
            )
            if threshold_cal_mode in {"auto", "walkforward", "quantile"}:
                threshold_cal_mode = "walkforward_quantile"
            if (
                threshold_src in {"auto_calibrated", "auto_default"}
                and threshold_cal_mode == "walkforward_quantile"
            ):
                try:
                    target_fa = float(p.get("threshold_target_false_alarm_rate", 0.02))
                except Exception:
                    target_fa = 0.02
                try:
                    cal_window = (
                        int(p["threshold_calibration_window"])
                        if "threshold_calibration_window" in p
                        and p.get("threshold_calibration_window") is not None
                        else None
                    )
                except Exception:
                    cal_window = None
                try:
                    cal_step = (
                        int(p["threshold_calibration_step"])
                        if "threshold_calibration_step" in p
                        and p.get("threshold_calibration_step") is not None
                        else None
                    )
                except Exception:
                    cal_step = None
                try:
                    cal_max_windows = int(p.get("threshold_calibration_max_windows", 6))
                except Exception:
                    cal_max_windows = 6
                try:
                    cal_boot = int(p.get("threshold_calibration_bootstraps", 2))
                except Exception:
                    cal_boot = 2
                threshold_used, threshold_calibration_info = (
                    _walkforward_quantile_threshold_calibration(
                        series=x,
                        hazard_lambda=hazard_lambda,
                        base_threshold=threshold_used,
                        target_false_alarm_rate=target_fa,
                        window=cal_window,
                        step=cal_step,
                        max_windows=cal_max_windows,
                        bootstrap_runs=cal_boot,
                    )
                )
            res = bocpd_gaussian(x, hazard_lambda=hazard_lambda, max_run_length=max_rl)
            cp_prob = np.asarray(
                res.get("cp_prob", np.zeros_like(x, dtype=float)), dtype=float
            )
            raw_cp_idx = [
                int(i)
                for i, v in enumerate(cp_prob.tolist())
                if np.isfinite(v) and float(v) >= float(threshold_used)
            ]
            try:
                cp_confirm_bars = int(p.get("cp_confirm_bars", 1))
            except Exception:
                cp_confirm_bars = 1
            try:
                cp_confirm_relaxed_mult = float(p.get("cp_confirm_relaxed_mult", 0.90))
            except Exception:
                cp_confirm_relaxed_mult = 0.90
            if "cp_edge_multiplier" in p and p.get("cp_edge_multiplier") is not None:
                try:
                    cp_edge_multiplier = float(p.get("cp_edge_multiplier"))
                except Exception:
                    cp_edge_multiplier = 1.08
            else:
                # When threshold is already calibrated via walk-forward null quantiles,
                # avoid double-tightening the edge gate.
                if (
                    threshold_src in {"auto_calibrated", "auto_default"}
                    and isinstance(threshold_calibration_info, dict)
                    and bool(threshold_calibration_info.get("calibrated", False))
                ):
                    cp_edge_multiplier = 1.0
                else:
                    cp_edge_multiplier = 1.08
            try:
                min_cp_distance_bars = int(
                    p.get("min_cp_distance_bars", max(2, min_regime_bars_val))
                )
            except Exception:
                min_cp_distance_bars = max(2, min_regime_bars_val)
            cp_idx, cp_filter_meta = _filter_bocpd_change_points(
                cp_prob=cp_prob,
                threshold=float(threshold_used),
                min_distance_bars=int(max(1, min_cp_distance_bars)),
                min_regime_bars=int(max(1, min_regime_bars_val)),
                confirm_bars=int(max(1, cp_confirm_bars)),
                confirm_relaxed_mult=float(cp_confirm_relaxed_mult),
                edge_multiplier=float(cp_edge_multiplier),
            )
            cps = [
                {"idx": i, "time": t_fmt[i], "prob": float(cp_prob[i])} for i in cp_idx
            ]
            tuning_hint: Optional[str] = None
            if len(cps) == 0:
                if (
                    len(raw_cp_idx) > 0
                    and int(cp_filter_meta.get("filtered_count", 0)) > 0
                ):
                    tuning_hint = (
                        "Change-point candidates were filtered by robustness guards "
                        "(confirmation/cooldown/edge checks). Tune cp_confirm_bars, "
                        "min_cp_distance_bars, or cp_edge_multiplier if needed."
                    )
                else:
                    tuning_hint = (
                        "No change points detected. Try lowering threshold or reducing "
                        f"hazard_lambda (currently {hazard_lambda}); active threshold={threshold_used:.2f}."
                    )
            if isinstance(threshold_calibration_info, dict):
                expected_fa_rate = float(
                    threshold_calibration_info.get("target_false_alarm_rate", 0.02)
                )
                calibration_age_bars = int(
                    threshold_calibration_info.get(
                        "points",
                        calibration_info.get("points", 0)
                        if isinstance(calibration_info, dict)
                        else 0,
                    )
                )
                threshold_calibrated = bool(
                    threshold_calibration_info.get("calibrated", False)
                )
            else:
                expected_fa_rate = 0.02
                calibration_age_bars = int(
                    calibration_info.get("points", 0)
                    if isinstance(calibration_info, dict)
                    else 0
                )
                threshold_calibrated = False
            reliability = _bocpd_reliability_score(
                cp_prob=cp_prob,
                cp_indices=cp_idx,
                threshold=float(threshold_used),
                lookback=int(lookback),
                min_regime_bars=int(max(1, min_regime_bars_val)),
                expected_false_alarm_rate=float(expected_fa_rate),
                calibration_age_bars=int(calibration_age_bars),
                threshold_calibrated=bool(threshold_calibrated),
            )
            payload = {
                "success": True,
                "symbol": symbol,
                "timeframe": timeframe,
                "method": method,
                "target": target,
                "times": t_fmt,
                "cp_prob": [
                    float(v) for v in np.asarray(cp_prob, dtype=float).tolist()
                ],
                "change_points": cps,
                "threshold": float(threshold_used),
                "reliability": reliability,
                "params_used": {
                    "hazard_lambda": hazard_lambda,
                    "hazard_lambda_source": hazard_src,
                    "cp_threshold": float(threshold_used),
                    "cp_threshold_source": threshold_src,
                    "hazard_mode": hazard_mode,
                    "max_run_length": max_rl,
                    "cp_filter": cp_filter_meta,
                },
            }
            if isinstance(calibration_info, dict):
                payload["params_used"]["auto_calibration"] = calibration_info
            if isinstance(threshold_calibration_info, dict):
                payload["params_used"]["cp_threshold_calibration"] = (
                    threshold_calibration_info
                )
            if tuning_hint is not None:
                payload["tuning_hint"] = tuning_hint
            if output in ("summary", "compact"):
                n = min(int(lookback), len(cp_prob))
                tail = (
                    np.asarray(cp_prob[-n:], dtype=float)
                    if n > 0
                    else np.asarray(cp_prob, dtype=float)
                )
                recent_cps = [c for c in cps if c.get("idx", 0) >= (len(cp_prob) - n)]
                summary = {
                    "lookback": int(n),
                    "last_cp_prob": float(cp_prob[-1])
                    if len(cp_prob)
                    else float("nan"),
                    "max_cp_prob": float(np.nanmax(tail))
                    if tail.size
                    else float("nan"),
                    "mean_cp_prob": float(np.nanmean(tail))
                    if tail.size
                    else float("nan"),
                    "change_points_count": int(len(recent_cps)),
                    "raw_change_points_count": int(
                        sum(1 for idx in raw_cp_idx if int(idx) >= (len(cp_prob) - n))
                    ),
                    "filtered_change_points_count": int(
                        max(
                            0,
                            sum(
                                1
                                for idx in raw_cp_idx
                                if int(idx) >= (len(cp_prob) - n)
                            )
                            - int(len(recent_cps)),
                        )
                    ),
                    "recent_change_points": recent_cps[-5:],
                    "confidence": float(reliability.get("confidence", 0.0)),
                    "expected_false_alarm_rate": float(
                        reliability.get("expected_false_alarm_rate", expected_fa_rate)
                    ),
                    "calibration_age_bars": int(
                        reliability.get("calibration_age_bars", calibration_age_bars)
                    ),
                }
                if tuning_hint is not None:
                    summary["tuning_hint"] = tuning_hint
                payload["summary"] = summary
                if output == "summary":
                    return _finish(_summary_only_payload(payload))
                if output == "compact" and n > 0:
                    # Compact mode uses the tail of the series; remap CP indices so they
                    # remain consistent with the truncated `times` array used by consolidation.
                    tail_offset = len(t_fmt) - n
                    payload["times"] = t_fmt[-n:]
                    payload["cp_prob"] = payload["cp_prob"][-n:]
                    tail_cps: List[Dict[str, Any]] = []
                    for cp in payload.get("change_points", []):
                        if not isinstance(cp, dict):
                            continue
                        idx = cp.get("idx")
                        if isinstance(idx, int) and idx >= tail_offset:
                            cp_tail = dict(cp)
                            cp_tail["idx"] = idx - tail_offset
                            tail_cps.append(cp_tail)
                    payload["change_points"] = tail_cps

            return _finish(
                _consolidate_payload(
                    payload, method, output, include_series=include_series
                )
            )

        elif method == "ms_ar":
            try:
                from statsmodels.tsa.regime_switching.markov_regression import (
                    MarkovRegression,
                )  # type: ignore
            except Exception:
                return _finish(
                    {
                        "error": "statsmodels MarkovRegression not available. Install statsmodels."
                    }
                )
            k_regimes = int(p.get("k_regimes", 2))
            order = int(p.get("order", 0))
            try:
                mod = MarkovRegression(
                    endog=x,
                    k_regimes=max(2, k_regimes),
                    trend="c",
                    order=max(0, order),
                    switching_variance=True,
                )
                res = mod.fit(disp=False, maxiter=int(p.get("maxiter", 100)))
                smoothed = res.smoothed_marginal_probabilities
                if hasattr(smoothed, "values"):
                    smoothed = smoothed.values
                state = np.argmax(smoothed, axis=1)
                probs = smoothed
            except Exception as ex:
                return _finish({"error": f"MS-AR fitting error: {ex}"})
            payload = {
                "success": True,
                "symbol": symbol,
                "timeframe": timeframe,
                "method": method,
                "target": target,
                "times": t_fmt,
                "state": [int(s) for s in state.tolist()],
                "state_probabilities": [
                    [float(v) for v in row] for row in probs.tolist()
                ],
                "params_used": {"k_regimes": k_regimes, "order": order},
            }
            if output in ("summary", "compact"):
                n = min(int(lookback), len(state))
                st_tail = state[-n:] if n > 0 else state
                last_s = int(state[-1]) if len(state) else None
                unique, counts = np.unique(st_tail, return_counts=True)
                shares = {
                    int(k): float(c) / float(len(st_tail) or 1)
                    for k, c in zip(unique, counts)
                }
                summary = {
                    "lookback": int(n),
                    "last_state": last_s,
                    "state_shares": shares,
                }
                payload["summary"] = summary
                if output == "summary":
                    return _finish(_summary_only_payload(payload))
                if output == "compact" and n > 0:
                    payload["times"] = t_fmt[-n:]
                    payload["state"] = payload["state"][-n:]
                    payload["state_probabilities"] = payload["state_probabilities"][-n:]

            return _finish(
                _consolidate_payload(
                    payload, method, output, include_series=include_series
                )
            )

        elif method == "hmm":  # 'hmm' (mixture/HMM-lite)
            try:
                from ..forecast.monte_carlo import fit_gaussian_mixture_1d
            except Exception as ex:
                return _finish({"error": f"HMM-lite import error: {ex}"})
            n_states = int(p.get("n_states", 2))
            w, mu, sigma, gamma, _ = fit_gaussian_mixture_1d(
                x, n_states=max(2, n_states)
            )
            state = (
                np.argmax(gamma, axis=1)
                if gamma.ndim == 2 and gamma.shape[0] == x.size
                else np.zeros(x.size, dtype=int)
            )
            gamma_smoothed: Optional[np.ndarray] = (
                gamma if isinstance(gamma, np.ndarray) else None
            )
            state, gamma_smoothed, smoothing_meta = _smooth_short_state_runs(
                state=np.asarray(state, dtype=int),
                probs=gamma_smoothed,
                min_regime_bars=min_regime_bars_val,
            )
            gamma_for_payload = (
                gamma_smoothed
                if isinstance(gamma_smoothed, np.ndarray)
                else (gamma if isinstance(gamma, np.ndarray) else None)
            )
            payload = {
                "success": True,
                "symbol": symbol,
                "timeframe": timeframe,
                "method": method,
                "target": target,
                "times": t_fmt,
                "state": [int(s) for s in state.tolist()],
                "state_probabilities": [
                    [float(v) for v in row]
                    for row in (
                        gamma_for_payload.tolist()
                        if isinstance(gamma_for_payload, np.ndarray)
                        and gamma_for_payload.ndim == 2
                        else np.zeros((x.size, n_states)).tolist()
                    )
                ],
                "regime_params": {
                    "weights": [float(v) for v in w.tolist()],
                    "mu": [float(v) for v in mu.tolist()],
                    "sigma": [float(v) for v in sigma.tolist()],
                },
                "params_used": {
                    "n_states": int(n_states),
                    "min_regime_bars": int(min_regime_bars_val),
                    "smoothing_applied": bool(
                        smoothing_meta.get("smoothing_applied", False)
                    ),
                    "transitions_before": int(
                        smoothing_meta.get("transitions_before", 0)
                    ),
                    "transitions_after": int(
                        smoothing_meta.get("transitions_after", 0)
                    ),
                },
            }
            if output in ("summary", "compact"):
                n = min(int(lookback), len(state))
                st_tail = state[-n:] if n > 0 else state
                last_s = int(state[-1]) if len(state) else None
                unique, counts = np.unique(st_tail, return_counts=True)
                shares = {
                    int(k): float(c) / float(len(st_tail) or 1)
                    for k, c in zip(unique, counts)
                }
                order = np.argsort(sigma)
                ranks = {int(s): int(r) for r, s in enumerate(order)}
                summary = {
                    "lookback": int(n),
                    "last_state": last_s,
                    "state_shares": shares,
                    "state_sigma": {int(i): float(sigma[i]) for i in range(len(sigma))},
                    "state_order_by_sigma": ranks,
                    "transitions_before": int(
                        smoothing_meta.get("transitions_before", 0)
                    ),
                    "transitions_after": int(
                        smoothing_meta.get("transitions_after", 0)
                    ),
                    "smoothing_applied": bool(
                        smoothing_meta.get("smoothing_applied", False)
                    ),
                }
                payload["summary"] = summary
                if output == "summary":
                    return _finish(_summary_only_payload(payload))
                if output == "compact" and n > 0:
                    payload["times"] = t_fmt[-n:]
                    payload["state"] = payload["state"][-n:]
                    if (
                        isinstance(gamma_for_payload, np.ndarray)
                        and len(gamma_for_payload) >= n
                    ):
                        payload["state_probabilities"] = payload["state_probabilities"][
                            -n:
                        ]

            return _finish(
                _consolidate_payload(
                    payload, method, output, include_series=include_series
                )
            )

        elif method == "clustering":
            try:
                from .features import extract_rolling_features
                from sklearn.preprocessing import StandardScaler
                from sklearn.cluster import KMeans
                from sklearn.decomposition import PCA
            except ImportError as ex:
                return _finish({"error": f"Clustering dependencies missing: {ex}"})

            window_size = int(p.get("window_size", 20))
            k_regimes = int(p.get("k_regimes", 3))
            use_pca = bool(p.get("use_pca", True))
            n_components = int(p.get("n_components", 3))

            # Extract features (use 'return' or 'price'? 'return' is stationary, usually better)
            # x is already computed based on target input
            features_df = extract_rolling_features(x, window_size=window_size)

            # Align features with time
            # valid_indices are where features are not NaN
            valid_mask = ~features_df.isna().any(axis=1)
            X_valid = features_df.loc[valid_mask]

            if X_valid.empty:
                return _finish(
                    {
                        "error": "Not enough data for feature extraction (check window_size)"
                    }
                )

            # Normalize
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_valid)

            # PCA
            if use_pca and X_scaled.shape[1] > n_components:
                pca = PCA(n_components=min(n_components, X_scaled.shape[1]))
                X_final = pca.fit_transform(X_scaled)
            else:
                X_final = X_scaled

            # Cluster
            kmeans = KMeans(n_clusters=k_regimes, random_state=42, n_init="auto")
            labels = kmeans.fit_predict(X_final)

            # Map back to full length
            # features_df has same length as x (n), but with NaNs at start
            # We want to fill the result states.
            # 0-fill or forward fill or -1?
            # Let's say -1 for undefined.
            full_states = np.full(len(x), -1, dtype=int)
            full_states[valid_mask] = labels

            # Probabilities? KMeans doesn't give probs easily (distance based).
            # We can use 1.0 for the assigned cluster or distance-to-center logic.
            # Simplified: 1.0 for assigned.
            full_probs = np.zeros((len(x), k_regimes))
            full_probs[valid_mask, labels] = 1.0

            # Reconstruct payload
            payload = {
                "success": True,
                "symbol": symbol,
                "timeframe": timeframe,
                "method": method,
                "target": target,
                "times": t_fmt,
                "state": [int(s) for s in full_states.tolist()],
                "state_probabilities": [
                    [float(v) for v in row] for row in full_probs.tolist()
                ],
                "params_used": {
                    "k_regimes": k_regimes,
                    "window_size": window_size,
                    "use_pca": use_pca,
                    "n_components": n_components,
                },
            }

            # Summary stats
            if output in ("summary", "compact"):
                n_summary = min(int(lookback), len(full_states))
                st_tail = full_states[-n_summary:] if n_summary > 0 else full_states
                # Filter out -1
                st_tail_valid = st_tail[st_tail != -1]

                unique, counts = np.unique(st_tail_valid, return_counts=True)
                shares = {
                    int(k): float(c) / float(len(st_tail_valid) or 1)
                    for k, c in zip(unique, counts)
                }

                summary = {
                    "lookback": int(n_summary),
                    "last_state": int(full_states[-1]) if len(full_states) else None,
                    "state_shares": shares,
                }
                payload["summary"] = summary

                if output == "summary":
                    return _finish(_summary_only_payload(payload))
                if output == "compact" and n_summary > 0:
                    payload["times"] = t_fmt[-n_summary:]
                    payload["state"] = payload["state"][-n_summary:]
                    payload["state_probabilities"] = payload["state_probabilities"][
                        -n_summary:
                    ]

            return _finish(
                _consolidate_payload(
                    payload, method, output, include_series=include_series
                )
            )

    except Exception as e:
        return _finish({"error": f"Error detecting regimes: {str(e)}"})
