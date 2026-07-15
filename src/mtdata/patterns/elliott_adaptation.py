"""Deterministic, causal market adaptation for Elliott pivot detection."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp
from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from ..utils.denoise import denoise_series

PivotBuilder = Callable[[np.ndarray, float, int], Sequence[int]]


@dataclass(frozen=True)
class ElliottAdaptation:
    """Effective request-scoped pivot signal, scan pairs, and diagnostics."""

    pivot_signal: np.ndarray
    scan_pairs: List[Tuple[float, int]]
    diagnostics: Dict[str, Any]


def _robust_mad(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0
    median = float(np.median(finite))
    return float(1.4826 * np.median(np.abs(finite - median)))


def _volatility_statistics(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    *,
    window_bars: int,
) -> Dict[str, float]:
    n = int(close.size)
    start = max(0, n - max(2, int(window_bars)))
    c = np.asarray(close[start:], dtype=float)
    positive = np.isfinite(c) & (c > 0.0)
    if c.size < 3 or not np.all(positive):
        return {"return_mad_pct": 0.0, "median_true_range_pct": 0.0}
    log_returns = np.diff(np.log(c))
    return_mad_pct = float(100.0 * np.expm1(max(0.0, _robust_mad(log_returns))))

    tr_pct = np.asarray([], dtype=float)
    if high.size == n and low.size == n:
        h = np.asarray(high[start:], dtype=float)
        l = np.asarray(low[start:], dtype=float)
        previous = c[:-1]
        if h.size == c.size and l.size == c.size and previous.size:
            tr = np.maximum.reduce(
                [
                    h[1:] - l[1:],
                    np.abs(h[1:] - previous),
                    np.abs(l[1:] - previous),
                ]
            )
            with np.errstate(divide="ignore", invalid="ignore"):
                tr_pct = 100.0 * tr / np.abs(previous)
    median_tr_pct = float(np.nanmedian(tr_pct)) if tr_pct.size else 0.0
    if not np.isfinite(median_tr_pct):
        median_tr_pct = 0.0
    return {
        "return_mad_pct": return_mad_pct,
        "median_true_range_pct": max(0.0, median_tr_pct),
    }


def _adaptive_thresholds(stats: Dict[str, float]) -> Tuple[float, List[float]]:
    base = max(
        4.0 * float(stats.get("return_mad_pct") or 0.0),
        0.75 * float(stats.get("median_true_range_pct") or 0.0),
    )
    if not np.isfinite(base) or base <= 0.0:
        return 0.5, [0.5]
    base = float(np.clip(base, 0.05, 10.0))
    thresholds = sorted(
        {
            round(float(np.clip(base * multiplier, 0.03, 15.0)), 8)
            for multiplier in (0.6, 1.0, 1.8)
        }
    )
    return base, thresholds


def _apply_candidate_spec(close: np.ndarray, spec: Dict[str, Any]) -> np.ndarray:
    method = str(spec.get("method") or "none")
    if method == "none":
        return np.asarray(close, dtype=float).copy()
    log_close = pd.Series(np.log(np.asarray(close, dtype=float)))
    filtered = denoise_series(
        log_close,
        method=method,
        params=dict(spec.get("params") or {}),
        causality="causal",
    )
    return np.exp(filtered.to_numpy(dtype=float))


def _candidate_signals(
    close: np.ndarray, parameter_close: np.ndarray | None = None
) -> List[Tuple[str, Dict[str, Any], np.ndarray]]:
    parameter_values = (
        np.asarray(parameter_close, dtype=float)
        if isinstance(parameter_close, np.ndarray) and parameter_close.size
        else np.asarray(close, dtype=float)
    )
    log_close = pd.Series(np.log(parameter_values))
    returns = np.diff(log_close.to_numpy(dtype=float))
    measurement_var = max(_robust_mad(returns) ** 2, 1e-12)
    specs: List[Tuple[str, str, Dict[str, Any]]] = [
        ("none", "none", {}),
        ("ema_3", "ema", {"span": 3}),
        ("ema_5", "ema", {"span": 5}),
        ("median_3", "median", {"window": 3}),
        ("hampel_7", "hampel", {"window": 7, "n_sigmas": 3.0}),
        (
            "kalman_robust",
            "kalman",
            {
                "measurement_var": measurement_var,
                "process_var": 0.1 * measurement_var,
            },
        ),
    ]
    out: List[Tuple[str, Dict[str, Any], np.ndarray]] = []
    for label, method, params in specs:
        spec = {"method": method, "params": dict(params)}
        out.append((label, spec, _apply_candidate_spec(close, spec)))
    return out


def _pivot_retention_score(
    signal: np.ndarray,
    *,
    threshold_pct: float,
    min_distance: int,
    pivot_builder: PivotBuilder,
) -> float:
    n = int(signal.size)
    horizon = max(3, int(min_distance) * 2)
    scores: List[float] = []
    for fraction in (0.55, 0.65, 0.75, 0.85):
        short_end = int(round(n * fraction))
        long_end = min(n, short_end + horizon)
        if short_end < 12 or long_end <= short_end:
            continue
        short = [int(v) for v in pivot_builder(signal[:short_end], threshold_pct, min_distance)]
        long = [int(v) for v in pivot_builder(signal[:long_end], threshold_pct, min_distance)]
        if not short:
            continue
        tolerance = max(1, int(min_distance) // 2)
        weights = np.ones(len(short), dtype=float)
        weights[-min(2, len(short)) :] = 2.0
        retained = np.asarray(
            [any(abs(idx - other) <= tolerance for other in long) for idx in short],
            dtype=float,
        )
        scores.append(float(np.average(retained, weights=weights)))
    return float(np.mean(scores)) if scores else 0.0


def _pivot_lag_score(
    signal: np.ndarray,
    raw_close: np.ndarray,
    *,
    threshold_pct: float,
    min_distance: int,
    pivot_builder: PivotBuilder,
) -> Tuple[float, float]:
    pivots = [int(v) for v in pivot_builder(signal, threshold_pct, min_distance)]
    if len(pivots) < 3:
        return 0.0, float(min_distance)
    lags: List[int] = []
    radius = max(1, int(min_distance))
    for pos in range(1, len(pivots) - 1):
        idx = pivots[pos]
        left = max(0, idx - radius)
        right = min(raw_close.size, idx + radius + 1)
        if right <= left:
            continue
        is_peak = signal[idx] > signal[pivots[pos - 1]] and signal[idx] > signal[pivots[pos + 1]]
        local = raw_close[left:right]
        extreme = int(np.argmax(local) if is_peak else np.argmin(local)) + left
        lags.append(abs(idx - extreme))
    median_lag = float(np.median(lags)) if lags else float(min_distance)
    return float(exp(-median_lag / max(1.0, float(min_distance)))), median_lag


def _score_signal(
    signal: np.ndarray,
    raw_close: np.ndarray,
    *,
    threshold_pct: float,
    min_distance: int,
    pivot_builder: PivotBuilder,
) -> Dict[str, float]:
    raw_log = np.log(raw_close)
    signal_log = np.log(signal)
    return_scale = max(_robust_mad(np.diff(raw_log)), 1e-12)
    rmse = float(np.sqrt(np.mean(np.square(signal_log - raw_log))))
    fidelity = float(exp(-rmse / (3.0 * return_scale)))
    raw_roughness = float(np.median(np.abs(np.diff(raw_log, n=2)))) if raw_log.size >= 3 else 0.0
    signal_roughness = float(np.median(np.abs(np.diff(signal_log, n=2)))) if signal_log.size >= 3 else raw_roughness
    reduction = 0.0
    if raw_roughness > 1e-12:
        reduction = float(np.clip((raw_roughness - signal_roughness) / raw_roughness, 0.0, 0.5) / 0.5)
    stability = _pivot_retention_score(
        signal,
        threshold_pct=threshold_pct,
        min_distance=min_distance,
        pivot_builder=pivot_builder,
    )
    lag_score, median_lag = _pivot_lag_score(
        signal,
        raw_close,
        threshold_pct=threshold_pct,
        min_distance=min_distance,
        pivot_builder=pivot_builder,
    )
    score = 0.55 * stability + 0.20 * lag_score + 0.15 * fidelity + 0.10 * reduction
    return {
        "score": float(score),
        "stability": float(stability),
        "lag_score": float(lag_score),
        "median_lag_bars": float(median_lag),
        "fidelity": float(fidelity),
        "roughness_reduction": float(reduction),
    }


def _select_scan_pairs(
    signal: np.ndarray,
    thresholds: Sequence[float],
    distances: Sequence[int],
    *,
    pivot_builder: PivotBuilder,
) -> List[Tuple[float, int]]:
    n = max(1, int(signal.size))
    candidates: List[Tuple[float, int, float, Tuple[int, ...]]] = []
    for threshold in thresholds:
        for distance in distances:
            signature = tuple(
                int(v) for v in pivot_builder(signal, float(threshold), int(distance))
            )
            if len(signature) < 2:
                continue
            density = 100.0 * len(signature) / n
            candidates.append((float(threshold), int(distance), density, signature))
    chosen: List[Tuple[float, int]] = []
    used_signatures: set[Tuple[int, ...]] = set()
    for target in (12.0, 8.0, 5.0):
        ranked = sorted(
            candidates,
            key=lambda item: (
                abs(item[2] - target),
                abs(item[0] - float(np.median(thresholds))),
                item[1],
            ),
        )
        winner = next((item for item in ranked if item[3] not in used_signatures), None)
        if winner is None:
            continue
        chosen.append((winner[0], winner[1]))
        used_signatures.add(winner[3])
    return chosen


def resolve_elliott_adaptation(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    *,
    pivot_builder: PivotBuilder,
    scale_mode: str,
    adaptive_denoise: str,
    adaptive_window_bars: int,
    adaptive_min_improvement: float,
    min_distance: int,
    pivot_price_source: str,
    external_denoise_applied: bool = False,
    fallback_threshold_pct: float = 0.5,
) -> ElliottAdaptation:
    """Resolve automatic signal and scan pairs without using future bars."""
    raw = np.asarray(close, dtype=float)
    n = int(raw.size)
    diagnostics: Dict[str, Any] = {
        "mode": str(scale_mode),
        "causality": "causal_walk_forward",
        "calibration_bars": int(min(n, max(1, adaptive_window_bars))),
    }
    invalid_prices = not np.all(np.isfinite(raw)) or np.any(raw <= 0.0)
    if str(scale_mode).lower() != "auto" or n < 96 or invalid_prices:
        reason = (
            "fixed_mode"
            if str(scale_mode).lower() != "auto"
            else "invalid_adaptation_prices"
            if invalid_prices
            else "insufficient_adaptation_data"
        )
        diagnostics.update(
            {
                "adaptive": False,
                "fallback_reason": reason,
                "selected_filter": {"method": "none", "params": {}},
            }
        )
        return ElliottAdaptation(
            raw.copy(),
            [(float(fallback_threshold_pct), int(min_distance))],
            diagnostics,
        )

    stats = _volatility_statistics(
        raw, high, low, window_bars=int(adaptive_window_bars)
    )
    base_threshold, thresholds = _adaptive_thresholds(stats)
    distances = sorted(
        {
            max(2, int(min_distance) - 2),
            max(2, int(min_distance)),
            max(2, int(min_distance) + 3),
        }
    )
    selected_signal = raw.copy()
    selected_spec: Dict[str, Any] = {"method": "none", "params": {}}
    candidate_metrics: List[Dict[str, Any]] = []
    skip_reason: str | None = None
    denoise_mode = str(adaptive_denoise).strip().lower()
    if external_denoise_applied:
        skip_reason = "explicit_denoise_precedence"
        selected_spec = {"method": "external_explicit", "params": {}}
    elif str(pivot_price_source).strip().lower() == "ohlc":
        skip_reason = "raw_ohlc_geometry_preserved"
    elif denoise_mode == "off":
        skip_reason = "adaptive_denoise_disabled"
    else:
        calibration_start = max(0, n - int(adaptive_window_bars))
        calibration_raw = raw[calibration_start:]
        candidates = _candidate_signals(raw, calibration_raw)
        scored: List[Tuple[str, Dict[str, Any], np.ndarray, Dict[str, float]]] = []
        for label, spec, full_signal in candidates:
            signal = full_signal[calibration_start:]
            metrics = _score_signal(
                signal,
                calibration_raw,
                threshold_pct=base_threshold,
                min_distance=int(min_distance),
                pivot_builder=pivot_builder,
            )
            candidate_metrics.append({"candidate": label, **spec, **metrics})
            scored.append((label, spec, full_signal, metrics))
        raw_entry = scored[0]
        ranked = sorted(
            enumerate(scored),
            key=lambda item: (-float(item[1][3]["score"]), int(item[0])),
        )
        winner = ranked[0][1]
        raw_score = float(raw_entry[3]["score"])
        improves = float(winner[3]["score"]) >= raw_score + float(adaptive_min_improvement)
        stable_enough = float(winner[3]["stability"]) >= float(raw_entry[3]["stability"]) - 0.02
        lag_ok = float(winner[3]["median_lag_bars"]) <= float(min_distance)
        if winner[0] != "none" and improves and stable_enough and lag_ok:
            selected_spec = dict(winner[1])
            if denoise_mode == "auto":
                selected_signal = winner[2]
        elif winner[0] != "none":
            skip_reason = "candidate_did_not_clear_safety_margin"
        if denoise_mode == "diagnostic":
            selected_spec = {"method": "none", "params": {}}
            selected_signal = raw.copy()
            skip_reason = "diagnostic_only"

    scan_pairs = _select_scan_pairs(
        selected_signal,
        thresholds,
        distances,
        pivot_builder=pivot_builder,
    )
    if not scan_pairs:
        scan_pairs = [(base_threshold, int(min_distance))]
    diagnostics.update(
        {
            "adaptive": True,
            "volatility": stats,
            "base_threshold_pct": float(base_threshold),
            "thresholds_pct": [float(v) for v in thresholds],
            "distance_candidates": [int(v) for v in distances],
            "scan_pairs": [
                {"threshold_pct": float(threshold), "min_distance": int(distance)}
                for threshold, distance in scan_pairs
            ],
            "selected_filter": selected_spec,
            "denoise_mode": denoise_mode,
        }
    )
    if candidate_metrics:
        diagnostics["candidate_metrics"] = candidate_metrics
    if skip_reason:
        diagnostics["denoise_skip_reason"] = skip_reason
    return ElliottAdaptation(selected_signal, scan_pairs, diagnostics)


__all__ = ["ElliottAdaptation", "resolve_elliott_adaptation"]

