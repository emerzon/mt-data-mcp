"""BOCPD (Bayesian Online Change Point Detection) core algorithm."""
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


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
            "expected_false_alarm_rate": float(np.clip(expected_false_alarm_rate, 1e-4, 0.25)),
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
    density_penalty = float(np.clip(abs(density - target_fa) / max(target_fa, 1e-6), 0.0, 1.0))
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
        from .....utils.regime import bocpd_gaussian

        rng = np.random.default_rng(int(seed))
        null_maxima: List[float] = []
        for s in starts:
            seg = x[int(s): int(s + win)]
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
    raw_idx = [int(i) for i, v in enumerate(cp.tolist()) if np.isfinite(v) and float(v) >= float(threshold)]
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
            support_window = cp[support_start: idx + 1]
            support_count = int(np.sum(np.asarray(support_window, dtype=float) >= relaxed))
            need = int(min(conf, support_window.size))
            if support_count < need:
                rejects["edge_support"] += 1
                continue
        else:
            fwd = cp[idx: min(n, idx + conf)]
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


__all__ = [
    "_bocpd_reliability_score",
    "_walkforward_quantile_threshold_calibration",
    "_filter_bocpd_change_points",
]
