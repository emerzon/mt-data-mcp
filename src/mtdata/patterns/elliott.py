from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from ..utils.utils import to_float_np
from .common import PatternResultBase, interval_overlap_ratio as _interval_overlap_ratio


@dataclass
class ElliottWaveConfig:
    """Configuration for Elliott Wave detection."""

    # Legacy options (kept for API compatibility)
    min_prominence_pct: float = 0.5
    min_distance: int = 5

    # New options
    swing_threshold_pct: Optional[float] = None  # ZigZag swing threshold in percent
    gmm_components: int = 2  # Impulsive vs corrective clusters
    min_gmm_waves: int = 8
    wave_min_len: int = 3
    use_volume_confirmation: bool = True
    volume_confirm_min_ratio: float = 1.05
    volume_confirm_bonus: float = 0.06
    volume_confirm_penalty: float = 0.04
    impulse_fib_weights: List[float] = field(default_factory=lambda: [0.30, 0.35, 0.20, 0.15])
    impulse_rule_weight: float = 0.65
    impulse_cls_weight: float = 0.35
    correction_rule_weight: float = 0.60
    correction_cls_weight: float = 0.40
    use_regime_context: bool = True
    regime_window_bars: int = 160
    regime_trend_strength_threshold: float = 1.25
    regime_efficiency_trending_threshold: float = 0.35
    regime_alignment_bonus: float = 0.05
    regime_countertrend_penalty: float = 0.05
    min_impulse_bars: int = 30
    min_correction_bars: int = 20

    # Autotuning controls
    autotune: bool = True
    tune_thresholds: Optional[List[float]] = None
    tune_min_distance: Optional[List[int]] = None
    autotune_skip_repeated_pivots: bool = True
    autotune_early_stop_repeats: int = 3
    autotune_scenario_overlap_ratio: float = 0.9
    scan_timeframes: List[str] = field(default_factory=lambda: ["H1", "H4", "D1"])
    max_scan_timeframes: int = 3

    # Analyzer controls
    min_confidence: float = 0.0
    top_k: int = 10
    include_fallback_candidate: bool = True
    recent_bars: int = 3
    correction_min_c_vs_a: float = 0.25
    correction_exclusion_bar_tolerance: int = 1
    correction_exclusion_overlap_ratio: float = 0.9
    pattern_types: List[str] = field(default_factory=lambda: ["impulse", "correction"])


@dataclass
class ElliottWaveResult(PatternResultBase):
    """Result of an Elliott Wave pattern detection."""

    wave_type: str  # "Impulse", "Correction", or "Candidate"
    wave_sequence: List[int]
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ElliottRuleEvaluation:
    valid: bool
    fib_score: float
    metrics: Dict[str, float]
    violations: List[str] = field(default_factory=list)


@dataclass
class ElliottScenario:
    pivots: List[int]
    bullish: bool
    confidence: float
    cls_score: float
    rule_eval: ElliottRuleEvaluation
    threshold_used: float
    min_distance_used: int
    fallback_candidate: bool = False
    synthetic_terminal_pivot: bool = False
    wave_type: str = "Impulse"
    validated_wave_type: Optional[str] = None
    classification_available: bool = False
    pivot_confirmations: Optional[List[bool]] = None


def _normalize_pattern_types(config: ElliottWaveConfig) -> set[str]:
    raw = getattr(config, "pattern_types", None)
    if not raw:
        return {"impulse", "correction"}
    out: set[str] = set()
    for item in raw:
        s = str(item).strip().lower()
        if s in ("impulse", "impulses"):
            out.add("impulse")
        elif s in ("correction", "corrective", "abc"):
            out.add("correction")
    return out or {"impulse", "correction"}


def _zigzag_pivots_indices(close: np.ndarray, threshold_pct: float) -> Tuple[List[int], List[str]]:
    """Lightweight ZigZag pivot detection returning indices and directions."""

    n = int(close.size)
    if n <= 1:
        return [], []

    piv_idx: List[int] = []
    piv_dir: List[str] = []  # "up" or "down" for the swing ending at the pivot

    pivot_i = 0
    pivot_p = float(close[0])
    trend: Optional[str] = None
    last_ext_i = 0
    last_ext_p = float(close[0])
    pre_high_i = 0
    pre_high_p = float(close[0])
    pre_low_i = 0
    pre_low_p = float(close[0])

    for i in range(1, n):
        p = float(close[i])
        if not np.isfinite(p):
            continue

        if trend is None:
            if p >= pre_high_p:
                pre_high_p = p
                pre_high_i = i
            if p <= pre_low_p:
                pre_low_p = p
                pre_low_i = i

            up_change = (p - pre_low_p) / abs(pre_low_p) * 100.0 if pre_low_p != 0 else 0.0
            down_change = (pre_high_p - p) / abs(pre_high_p) * 100.0 if pre_high_p != 0 else 0.0
            can_start_up = up_change >= threshold_pct and pre_low_i < i
            can_start_down = down_change >= threshold_pct and pre_high_i < i

            if can_start_up and can_start_down:
                if pre_low_i > pre_high_i:
                    trend = "up"
                    pivot_i = pre_low_i
                    pivot_p = pre_low_p
                elif pre_high_i > pre_low_i:
                    trend = "down"
                    pivot_i = pre_high_i
                    pivot_p = pre_high_p
                elif up_change >= down_change:
                    trend = "up"
                    pivot_i = pre_low_i
                    pivot_p = pre_low_p
                else:
                    trend = "down"
                    pivot_i = pre_high_i
                    pivot_p = pre_high_p
            elif can_start_up:
                trend = "up"
                pivot_i = pre_low_i
                pivot_p = pre_low_p
            elif can_start_down:
                trend = "down"
                pivot_i = pre_high_i
                pivot_p = pre_high_p
            else:
                continue
            last_ext_i = i
            last_ext_p = p
            piv_idx.append(pivot_i)
            piv_dir.append(trend)

        if trend == "up":
            if p > last_ext_p:
                last_ext_p = p
                last_ext_i = i
            retr = (last_ext_p - p) / last_ext_p * 100.0 if last_ext_p != 0 else 0.0
            if retr >= threshold_pct:
                piv_idx.append(last_ext_i)
                piv_dir.append("up")
                trend = "down"
                pivot_i = i
                pivot_p = p
                last_ext_i = i
                last_ext_p = p
        else:
            if p < last_ext_p:
                last_ext_p = p
                last_ext_i = i
            retr = (p - last_ext_p) / abs(last_ext_p) * 100.0 if last_ext_p != 0 else 0.0
            if retr >= threshold_pct:
                piv_idx.append(last_ext_i)
                piv_dir.append("down")
                trend = "up"
                pivot_i = i
                pivot_p = p
                last_ext_i = i
                last_ext_p = p

    if len(piv_idx) == 0 or piv_idx[-1] != last_ext_i:
        piv_idx.append(last_ext_i)
        piv_dir.append("up" if trend == "up" else "down" if trend == "down" else "flat")

    return piv_idx, piv_dir


def _enforce_min_distance_on_pivots(pivots: List[int], close: np.ndarray, min_distance: int) -> List[int]:
    """Ensure pivots are monotonic and separated by at least min_distance bars."""

    if not pivots:
        return []

    md = int(max(1, min_distance))
    sorted_unique: List[int] = []
    for idx in pivots:
        i = int(idx)
        if i < 0 or i >= int(close.size):
            continue
        if not sorted_unique or i > sorted_unique[-1]:
            sorted_unique.append(i)
        elif i == sorted_unique[-1]:
            continue

    if not sorted_unique:
        return []
    if len(sorted_unique) <= 1:
        return sorted_unique

    out = [sorted_unique[0]]
    for idx in sorted_unique[1:]:
        prev = out[-1]
        if idx - prev >= md:
            out.append(idx)
            continue
        # When two pivots are too close, keep the one with a larger move from anchor.
        if len(out) == 1:
            out[-1] = idx
            continue
        anchor = out[-2]
        old_move = abs(float(close[prev]) - float(close[anchor]))
        new_move = abs(float(close[idx]) - float(close[anchor]))
        if new_move >= old_move:
            out[-1] = idx
    return out


def _segment_waves_from_pivots(pivots: List[int]) -> List[Tuple[int, int]]:
    waves: List[Tuple[int, int]] = []
    if len(pivots) < 2:
        return waves
    for i in range(len(pivots) - 1):
        s, e = pivots[i], pivots[i + 1]
        if e > s:
            waves.append((s, e))
    return waves


def _extract_wave_features_with_index(
    waves: List[Tuple[int, int]],
    close: np.ndarray,
) -> Tuple[np.ndarray, Dict[int, int]]:
    """Extract per-wave features and preserve the source wave index mapping."""
    features: List[List[float]] = []
    wave_index_map: Dict[int, int] = {}
    for wave_idx, (start, end) in enumerate(waves):
        if end <= start:
            continue
        s = float(close[start])
        e = float(close[end])
        if not np.isfinite(s) or not np.isfinite(e):
            continue

        duration = float(end - start)
        price_change = e - s
        slope = price_change / duration if duration > 0 else 0.0
        normalized_price_change = price_change / s if s != 0 else 0.0
        wave_index_map[wave_idx] = len(features)
        features.append([duration, normalized_price_change, slope])

    if not features:
        return np.empty((0, 3), dtype=float), {}
    return np.asarray(features, dtype=float), wave_index_map


def _extract_wave_features(waves: List[Tuple[int, int]], close: np.ndarray) -> np.ndarray:
    """Backward-compatible feature extractor used by existing tests/importers."""
    features, _ = _extract_wave_features_with_index(waves, close)
    return features


def _cluster_impulsive_score(cluster_mean: np.ndarray) -> float:
    price_change = abs(float(cluster_mean[1])) if cluster_mean.size > 1 else 0.0
    slope = abs(float(cluster_mean[2])) if cluster_mean.size > 2 else 0.0
    duration = abs(float(cluster_mean[0])) if cluster_mean.size > 0 else 0.0
    return float(price_change + 0.75 * slope + 0.1 * duration)


def _select_impulsive_cluster(cluster_means: np.ndarray) -> Optional[int]:
    if cluster_means.ndim != 2 or cluster_means.shape[0] < 1:
        return None
    scores = np.asarray(
        [_cluster_impulsive_score(cluster_means[i]) for i in range(cluster_means.shape[0])],
        dtype=float,
    )
    if scores.size == 0 or not np.all(np.isfinite(scores)):
        return None
    return int(np.argmax(scores))


def _cluster_direction_score(cluster_mean: np.ndarray) -> float:
    price_change = float(cluster_mean[1]) if cluster_mean.size > 1 else 0.0
    slope = float(cluster_mean[2]) if cluster_mean.size > 2 else 0.0
    return float(price_change + 0.75 * slope)


def _select_directional_cluster(cluster_means: np.ndarray, bullish: bool) -> Optional[int]:
    if cluster_means.ndim != 2 or cluster_means.shape[0] < 1:
        return None
    scores = np.asarray(
        [_cluster_direction_score(cluster_means[i]) for i in range(cluster_means.shape[0])],
        dtype=float,
    )
    if scores.size == 0 or not np.all(np.isfinite(scores)):
        return None
    return int(np.argmax(scores) if bullish else np.argmin(scores))


def _normalized_impulse_fib_weights(config: Optional[ElliottWaveConfig]) -> Tuple[float, float, float, float]:
    defaults = (0.30, 0.35, 0.20, 0.15)
    if config is None:
        return defaults
    raw = getattr(config, "impulse_fib_weights", None)
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        return defaults
    try:
        weights = [float(v) for v in raw]
    except Exception:
        return defaults
    if not all(np.isfinite(v) and v >= 0.0 for v in weights):
        return defaults
    total = float(sum(weights))
    if total <= 1e-12:
        return defaults
    return tuple(float(v / total) for v in weights)


def _classify_waves(
    features: np.ndarray,
    config: ElliottWaveConfig,
) -> Tuple[np.ndarray, Optional[GaussianMixture], Optional[StandardScaler], Optional[np.ndarray], Optional[int]]:
    """Classify waves with GMM; return cluster labels, model, scaler, probabilities and impulsive cluster id."""

    min_waves = max(int(config.gmm_components), int(max(1, getattr(config, "min_gmm_waves", 8))))
    if features.shape[0] < min_waves:
        return np.array([]), None, None, None, None

    try:
        feature_std = np.nanstd(features, axis=0)
        if not np.all(np.isfinite(feature_std)) or float(np.nanmax(np.abs(feature_std))) <= 1e-8:
            return np.array([]), None, None, None, None

        scaler = StandardScaler()
        scaled = scaler.fit_transform(features)
        if not np.all(np.isfinite(scaled)):
            return np.array([]), None, None, None, None

        gmm = GaussianMixture(n_components=config.gmm_components, random_state=42)
        gmm.fit(scaled)
        labels = gmm.predict(scaled)
        if np.unique(labels).size < int(config.gmm_components):
            return np.array([]), None, None, None, None
        probs = gmm.predict_proba(scaled)
        impulsive_cluster = _select_impulsive_cluster(gmm.means_)
        if impulsive_cluster is None:
            return np.array([]), None, None, None, None
        return labels.astype(int), gmm, scaler, probs, impulsive_cluster
    except Exception:
        return np.array([]), None, None, None, None


def _window_hit(v: float, lo: float, hi: float, taper: float = 0.2) -> float:
    rng = hi - lo
    if rng <= 0:
        return 0.0
    if lo <= v <= hi:
        return 1.0
    if v < lo:
        d = (lo - v) / (taper * rng)
        return float(max(0.0, 1.0 - d))
    d = (v - hi) / (taper * rng)
    return float(max(0.0, 1.0 - d))


def _evaluate_impulse_rules(
    c: np.ndarray,
    piv: List[int],
    bullish: bool,
    config: Optional[ElliottWaveConfig] = None,
) -> ElliottRuleEvaluation:
    """Validate core 5-wave impulse rules and compute Fibonacci alignment score."""

    if len(piv) != 6:
        return ElliottRuleEvaluation(valid=False, fib_score=0.0, metrics={}, violations=["pivot_count_not_6"])

    w = [float(c[i]) for i in piv]
    if not all(np.isfinite(v) for v in w):
        return ElliottRuleEvaluation(valid=False, fib_score=0.0, metrics={}, violations=["non_finite_prices"])

    L = [w[i + 1] - w[i] for i in range(5)]
    absL = [abs(x) for x in L]

    sgn = 1.0 if bullish else -1.0
    expected = [sgn, -sgn, sgn, -sgn, sgn]
    violations: List[str] = []

    if not all((L[i] > 0) if expected[i] > 0 else (L[i] < 0) for i in range(5)):
        violations.append("direction_sequence_invalid")

    # Rule 2: Wave 2 does not retrace beyond start of wave 1.
    if bullish and w[2] <= w[0]:
        violations.append("wave2_over_retrace")
    if (not bullish) and w[2] >= w[0]:
        violations.append("wave2_over_retrace")

    # Rule 3: Wave 3 is not the shortest among waves 1, 3, 5.
    if absL[2] <= min(absL[0], absL[4]):
        violations.append("wave3_shortest")

    # Rule 4: Wave 4 does not overlap Wave 1 territory.
    if bullish and w[4] <= w[1]:
        violations.append("wave4_overlap")
    if (not bullish) and w[4] >= w[1]:
        violations.append("wave4_overlap")

    r2 = absL[1] / absL[0] if absL[0] > 0 else 0.0
    r4 = absL[3] / absL[2] if absL[2] > 0 else 0.0
    e3 = absL[2] / absL[0] if absL[0] > 0 else 0.0
    rel53 = absL[4] / absL[0] if absL[0] > 0 else 0.0
    rel5_alt = absL[4] / abs(w[3] - w[1]) if (w[3] != w[1]) else 0.0

    s2 = _window_hit(r2, 0.382, 0.618)
    s4 = _window_hit(r4, 0.236, 0.382)
    s3 = _window_hit(e3, 1.618, 2.618)
    s5 = max(_window_hit(rel53, 0.8, 1.2), _window_hit(rel5_alt, 0.55, 0.75))
    w2, w3, w4, w5 = _normalized_impulse_fib_weights(config)
    fib_score = float(w2 * s2 + w3 * s3 + w4 * s4 + w5 * s5)
    direction = 1.0 if bullish else -1.0
    wave5_target_equal_wave1 = float(w[4] + direction * absL[0])
    wave5_target_0618_wave3 = float(w[4] + direction * 0.618 * absL[2])
    wave5_target_zone_low = float(min(wave5_target_equal_wave1, wave5_target_0618_wave3))
    wave5_target_zone_high = float(max(wave5_target_equal_wave1, wave5_target_0618_wave3))

    metrics = {
        "r2": float(r2),
        "r4": float(r4),
        "e3": float(e3),
        "rel53": float(rel53),
        "rel5_alt": float(rel5_alt),
        "fib_score": fib_score,
        "wave5_target_equal_wave1": wave5_target_equal_wave1,
        "wave5_target_0618_wave3": wave5_target_0618_wave3,
        "wave5_target_zone_low": wave5_target_zone_low,
        "wave5_target_zone_high": wave5_target_zone_high,
        "fib_weight_wave2": float(w2),
        "fib_weight_wave3": float(w3),
        "fib_weight_wave4": float(w4),
        "fib_weight_wave5": float(w5),
    }
    return ElliottRuleEvaluation(valid=(len(violations) == 0), fib_score=fib_score, metrics=metrics, violations=violations)


def _impulse_rules_and_score(c: np.ndarray, piv: List[int], bullish: bool) -> Tuple[bool, float, Dict[str, float]]:
    """Backward-compatible wrapper around rule evaluation."""

    ev = _evaluate_impulse_rules(c, piv, bullish)
    return ev.valid, ev.fib_score, ev.metrics


def _blend_confidence(rule_score: float, cls_score: float, *, classification_available: bool, rule_weight: float, cls_weight: float) -> float:
    if not classification_available:
        return float(min(1.0, max(0.0, rule_score)))
    rule_weight = max(0.0, float(rule_weight))
    cls_weight = max(0.0, float(cls_weight))
    total = float(max(1e-9, rule_weight + cls_weight))
    blended = (rule_weight * float(rule_score) + cls_weight * float(cls_score)) / total
    return float(min(1.0, max(0.0, blended)))


def _evaluate_correction_rules(
    c: np.ndarray,
    piv: List[int],
    bullish: bool,
    config: Optional[ElliottWaveConfig] = None,
) -> ElliottRuleEvaluation:
    """Validate a 3-wave corrective sequence (ABC) and compute score."""

    if len(piv) != 4:
        return ElliottRuleEvaluation(valid=False, fib_score=0.0, metrics={}, violations=["pivot_count_not_4"])

    w = [float(c[i]) for i in piv]
    if not all(np.isfinite(v) for v in w):
        return ElliottRuleEvaluation(valid=False, fib_score=0.0, metrics={}, violations=["non_finite_prices"])

    L = [w[i + 1] - w[i] for i in range(3)]
    absL = [abs(x) for x in L]
    sgn = 1.0 if bullish else -1.0
    expected = [sgn, -sgn, sgn]
    violations: List[str] = []

    if not all((L[i] > 0) if expected[i] > 0 else (L[i] < 0) for i in range(3)):
        violations.append("direction_sequence_invalid")

    # B should not fully retrace wave A start in a standard zigzag interpretation.
    if bullish and w[2] <= w[0]:
        violations.append("waveB_over_retrace")
    if (not bullish) and w[2] >= w[0]:
        violations.append("waveB_over_retrace")

    # C should have meaningful size versus A to avoid noise classifications.
    min_c_vs_a = float(getattr(config, "correction_min_c_vs_a", 0.25)) if config is not None else 0.25
    if absL[2] < min_c_vs_a * absL[0]:
        violations.append("waveC_too_short")

    b_retrace = absL[1] / absL[0] if absL[0] > 0 else 0.0
    c_vs_a = absL[2] / absL[0] if absL[0] > 0 else 0.0
    score_b = _window_hit(b_retrace, 0.382, 0.786)
    score_c = _window_hit(c_vs_a, 0.618, 1.618)
    fib_score = float(0.5 * (score_b + score_c))

    metrics = {
        "b_retrace": float(b_retrace),
        "c_vs_a": float(c_vs_a),
        "fib_score": fib_score,
    }
    return ElliottRuleEvaluation(valid=(len(violations) == 0), fib_score=fib_score, metrics=metrics, violations=violations)


def _classification_score_window(
    probs: Optional[np.ndarray],
    cluster_means: Optional[np.ndarray],
    k: int,
    bullish: bool,
    window_len: int,
    trend_slots: List[int],
    counter_slots: List[int],
    wave_index_map: Optional[Dict[int, int]] = None,
) -> float:
    if probs is None or cluster_means is None:
        return 0.5
    if cluster_means.ndim != 2 or cluster_means.shape[0] < 1:
        return 0.5

    cluster_for_trend = _select_directional_cluster(cluster_means, bullish)
    if cluster_for_trend is None:
        return 0.5
    if wave_index_map is not None:
        mapped_rows: List[int] = []
        for offset in range(window_len):
            mapped = wave_index_map.get(int(k + offset))
            if mapped is None:
                return 0.5
            mapped_rows.append(int(mapped))
        window_probs = probs[np.asarray(mapped_rows, dtype=int), cluster_for_trend]
    else:
        if probs.shape[0] < (k + window_len):
            return 0.5
        window_probs = probs[k : k + window_len, cluster_for_trend]
    if window_probs.shape[0] != window_len:
        return 0.5

    trend_vals = [float(window_probs[i]) for i in trend_slots if 0 <= int(i) < window_len]
    counter_vals = [float(window_probs[i]) for i in counter_slots if 0 <= int(i) < window_len]
    if not trend_vals and not counter_vals:
        return 0.5

    trend_score = float(sum(trend_vals) / len(trend_vals)) if trend_vals else 0.5
    counter_score = float(sum(counter_vals) / len(counter_vals)) if counter_vals else 0.5
    cls_score = 0.5 * trend_score + 0.5 * (1.0 - counter_score)
    return float(min(1.0, max(0.0, cls_score)))


def _result_sort_key(result: ElliottWaveResult) -> Tuple[float, int, str, Tuple[int, ...]]:
    return (
        -float(result.confidence),
        -int(result.end_index),
        str(result.wave_type).lower(),
        tuple(int(i) for i in result.wave_sequence),
    )


def _contiguous_pivot_slices(pivots: List[int], size: int) -> set[Tuple[int, ...]]:
    if size <= 0 or len(pivots) < size:
        return set()
    return {tuple(int(v) for v in pivots[i : i + size]) for i in range(len(pivots) - size + 1)}


def _pivot_sequence_is_near_match(
    candidate: List[int],
    reference: Tuple[int, ...],
    *,
    bar_tolerance: int,
    overlap_ratio: float,
) -> bool:
    if len(candidate) != len(reference) or not candidate:
        return False
    if all(abs(int(a) - int(b)) <= int(bar_tolerance) for a, b in zip(candidate, reference)):
        return True
    overlap = max(
        _interval_overlap_ratio(int(candidate[0]), int(candidate[-1]), int(reference[0]), int(reference[-1])),
        _interval_coverage_ratio(int(candidate[0]), int(candidate[-1]), int(reference[0]), int(reference[-1])),
    )
    return bool(overlap >= float(overlap_ratio))


def _scenario_signature(scenarios: List["ElliottScenario"]) -> Tuple[Tuple[str, bool, int, int, int], ...]:
    entries = {
        (
            str(scenario.wave_type).lower(),
            bool(scenario.bullish),
            int(len(scenario.pivots)),
            int(scenario.pivots[0]),
            int(scenario.pivots[-1]),
        )
        for scenario in scenarios
        if scenario.pivots
    }
    return tuple(sorted(entries))


def _scenario_signatures_overlap(
    current: Tuple[Tuple[str, bool, int, int, int], ...],
    prior: Tuple[Tuple[str, bool, int, int, int], ...],
    *,
    overlap_ratio: float,
) -> bool:
    if not current or not prior or len(current) != len(prior):
        return False
    unmatched = list(prior)
    for item in current:
        match_index: Optional[int] = None
        for idx, prior_item in enumerate(unmatched):
            if item[:3] != prior_item[:3]:
                continue
            overlap = max(
                _interval_overlap_ratio(int(item[3]), int(item[4]), int(prior_item[3]), int(prior_item[4])),
                _interval_coverage_ratio(int(item[3]), int(item[4]), int(prior_item[3]), int(prior_item[4])),
            )
            if overlap >= float(overlap_ratio):
                match_index = idx
                break
        if match_index is None:
            return False
        unmatched.pop(match_index)
    return not unmatched


def _elliott_result_key(result: ElliottWaveResult) -> Tuple[str, Tuple[int, ...]]:
    return str(result.wave_type), tuple(int(i) for i in result.wave_sequence)


def _upsert_elliott_result(
    results_by_key: Dict[Tuple[str, Tuple[int, ...]], ElliottWaveResult],
    result: ElliottWaveResult,
) -> None:
    key = _elliott_result_key(result)
    prior = results_by_key.get(key)
    if prior is None or float(result.confidence) > float(prior.confidence):
        results_by_key[key] = result


def _pivot_signature_for_settings(
    close: np.ndarray,
    threshold_pct: float,
    min_distance: int,
) -> Tuple[int, ...]:
    piv_idx, _ = _zigzag_pivots_indices(close, float(threshold_pct))
    piv_idx = _enforce_min_distance_on_pivots(piv_idx, close, int(max(1, min_distance)))
    return tuple(int(idx) for idx in piv_idx)


def _interval_coverage_ratio(a_start: int, a_end: int, b_start: int, b_end: int) -> float:
    lo = max(int(a_start), int(b_start))
    hi = min(int(a_end), int(b_end))
    inter = max(0, hi - lo + 1)
    span = max(0, int(a_end) - int(a_start) + 1)
    if span <= 0:
        return 0.0
    return float(inter) / float(span)


def _filter_overlapping_corrections(
    results: List[ElliottWaveResult],
    *,
    overlap_threshold: float = 0.8,
) -> List[ElliottWaveResult]:
    impulses = [result for result in results if str(result.wave_type).lower() == "impulse"]
    if not impulses:
        return results

    filtered: List[ElliottWaveResult] = []
    for result in results:
        if str(result.wave_type).lower() != "correction":
            filtered.append(result)
            continue
        overlaps_impulse = any(
            max(
                _interval_overlap_ratio(
                    int(result.start_index),
                    int(result.end_index),
                    int(impulse.start_index),
                    int(impulse.end_index),
                ),
                _interval_coverage_ratio(
                    int(result.start_index),
                    int(result.end_index),
                    int(impulse.start_index),
                    int(impulse.end_index),
                ),
            ) >= float(overlap_threshold)
            for impulse in impulses
        )
        if not overlaps_impulse:
            filtered.append(result)
    return filtered


class ElliottWaveAnalyzer:
    """Facade-style analyzer inspired by ta4j's scenario pipeline."""

    def __init__(self, close: np.ndarray, times: np.ndarray, config: ElliottWaveConfig):
        self.close = close
        self.times = times
        self.config = config

    def analyze_once(self, threshold_pct: float, min_distance: int) -> List[ElliottScenario]:
        piv_idx, _ = _zigzag_pivots_indices(self.close, float(threshold_pct))
        piv_idx = _enforce_min_distance_on_pivots(piv_idx, self.close, min_distance)
        if len(piv_idx) < 4:
            return []

        waves = _segment_waves_from_pivots(piv_idx)
        if not waves:
            return []

        features, wave_index_map = _extract_wave_features_with_index(waves, self.close)
        if features.shape[0] == 0:
            return []

        _, gmm, _, probs, _ = _classify_waves(features, self.config)
        cluster_means = gmm.means_ if gmm is not None else None
        classification_available = probs is not None and cluster_means is not None

        out: List[ElliottScenario] = []
        pattern_types = _normalize_pattern_types(self.config)
        total_waves = len(waves)
        min_wave_span = int(max(1, max(min_distance, self.config.wave_min_len)))
        correction_exclusions: List[Tuple[int, ...]] = []
        correction_bar_tolerance = max(0, int(getattr(self.config, "correction_exclusion_bar_tolerance", 1)))
        correction_overlap_ratio = float(
            max(0.0, min(1.0, getattr(self.config, "correction_exclusion_overlap_ratio", 0.9)))
        )

        if "impulse" in pattern_types:
            for k in range(0, total_waves - 4):
                piv_seq = [int(x) for x in piv_idx[k : k + 6]]
                if len(piv_seq) < 6:
                    continue

                if any((piv_seq[j + 1] - piv_seq[j]) < min_wave_span for j in range(5)):
                    continue

                bullish = bool(self.close[piv_seq[-1]] > self.close[piv_seq[0]])
                bearish = bool(self.close[piv_seq[-1]] < self.close[piv_seq[0]])
                if not (bullish or bearish):
                    continue

                rule_eval = _evaluate_impulse_rules(self.close, piv_seq, bullish=bullish, config=self.config)
                if not rule_eval.valid:
                    continue

                cls_score = _classification_score_window(
                    probs,
                    cluster_means,
                    k,
                    bullish,
                    window_len=5,
                    trend_slots=[0, 2, 4],
                    counter_slots=[1, 3],
                    wave_index_map=wave_index_map,
                )
                pivot_confirmations = [True] * len(piv_seq)
                if piv_seq and int(piv_seq[-1]) == int(piv_idx[-1]):
                    pivot_confirmations = [False] * len(piv_seq)
                confidence = _blend_confidence(
                    rule_eval.fib_score,
                    cls_score,
                    classification_available=classification_available,
                    rule_weight=float(getattr(self.config, "impulse_rule_weight", 0.65)),
                    cls_weight=float(getattr(self.config, "impulse_cls_weight", 0.35)),
                )
                if confidence < float(self.config.min_confidence):
                    continue

                out.append(
                    ElliottScenario(
                        pivots=piv_seq,
                        bullish=bullish,
                        confidence=confidence,
                        cls_score=cls_score,
                        rule_eval=rule_eval,
                        threshold_used=float(threshold_pct),
                        min_distance_used=int(min_distance),
                        wave_type="Impulse",
                        classification_available=classification_available,
                        pivot_confirmations=pivot_confirmations,
                    )
                )
                correction_exclusions.extend(_contiguous_pivot_slices(piv_seq, 4))

        if "correction" in pattern_types:
            for k in range(0, total_waves - 2):
                piv_seq = [int(x) for x in piv_idx[k : k + 4]]
                if len(piv_seq) < 4:
                    continue
                if any(
                    _pivot_sequence_is_near_match(
                        piv_seq,
                        excluded,
                        bar_tolerance=correction_bar_tolerance,
                        overlap_ratio=correction_overlap_ratio,
                    )
                    for excluded in correction_exclusions
                ):
                    continue

                if any((piv_seq[j + 1] - piv_seq[j]) < min_wave_span for j in range(3)):
                    continue

                bullish = bool(self.close[piv_seq[-1]] > self.close[piv_seq[0]])
                bearish = bool(self.close[piv_seq[-1]] < self.close[piv_seq[0]])
                if not (bullish or bearish):
                    continue

                rule_eval = _evaluate_correction_rules(
                    self.close,
                    piv_seq,
                    bullish=bullish,
                    config=self.config,
                )
                if not rule_eval.valid:
                    continue

                cls_score = _classification_score_window(
                    probs,
                    cluster_means,
                    k,
                    bullish,
                    window_len=3,
                    trend_slots=[0, 2],
                    counter_slots=[1],
                    wave_index_map=wave_index_map,
                )
                pivot_confirmations = [True] * len(piv_seq)
                if piv_seq and int(piv_seq[-1]) == int(piv_idx[-1]):
                    pivot_confirmations = [False] * len(piv_seq)
                confidence = _blend_confidence(
                    rule_eval.fib_score,
                    cls_score,
                    classification_available=classification_available,
                    rule_weight=float(getattr(self.config, "correction_rule_weight", 0.60)),
                    cls_weight=float(getattr(self.config, "correction_cls_weight", 0.40)),
                )
                if confidence < float(self.config.min_confidence):
                    continue

                out.append(
                    ElliottScenario(
                        pivots=piv_seq,
                        bullish=bullish,
                        confidence=confidence,
                        cls_score=cls_score,
                        rule_eval=rule_eval,
                        threshold_used=float(threshold_pct),
                        min_distance_used=int(min_distance),
                        wave_type="Correction",
                        classification_available=classification_available,
                        pivot_confirmations=pivot_confirmations,
                    )
                )

        return out

    def build_result(self, scenario: ElliottScenario) -> ElliottWaveResult:
        piv_seq = [int(x) for x in scenario.pivots]
        start_index = int(piv_seq[0])
        end_index = int(piv_seq[-1])
        if str(scenario.wave_type).lower() == "correction" and len(piv_seq) == 4:
            labels = ["S", "A", "B", "C"]
        else:
            labels = [f"W{j}" for j in range(len(piv_seq))]

        wave_points_labeled: List[Dict[str, Any]] = []
        for j, idx in enumerate(piv_seq):
            ti = PatternResultBase.resolve_time(self.times, idx)
            is_confirmed = True
            if isinstance(scenario.pivot_confirmations, list) and j < len(scenario.pivot_confirmations):
                is_confirmed = bool(scenario.pivot_confirmations[j])
            wave_points_labeled.append(
                {
                    "label": labels[j],
                    "index": int(idx),
                    "time": ti,
                    "price": float(self.close[idx]),
                    "is_confirmed": bool(is_confirmed),
                }
            )

        invalidation_level = float(self.close[piv_seq[0]]) if piv_seq else None
        sequence_direction = "bull" if scenario.bullish else "bear"

        details: Dict[str, Any] = {
            "wave_points": [float(self.close[i]) for i in piv_seq],
            "wave_points_labeled": wave_points_labeled,
            "bullish": bool(scenario.bullish),
            "trend": sequence_direction,
            "sequence_direction": sequence_direction,
            "pattern_family": str(scenario.wave_type).lower(),
            "fib_metrics": dict(scenario.rule_eval.metrics),
            "rule_violations": list(scenario.rule_eval.violations),
            "cls_score": float(scenario.cls_score),
            "tuned_threshold_pct": float(scenario.threshold_used),
            "min_distance_used": int(scenario.min_distance_used),
            "fallback_candidate": bool(scenario.fallback_candidate),
            "synthetic_terminal_pivot": bool(scenario.synthetic_terminal_pivot),
            "classification_available": bool(scenario.classification_available),
            "pattern_confirmed": bool(all(scenario.pivot_confirmations)) if isinstance(scenario.pivot_confirmations, list) else True,
            "has_unconfirmed_terminal_pivot": bool(
                isinstance(scenario.pivot_confirmations, list)
                and len(scenario.pivot_confirmations) > 0
                and not bool(scenario.pivot_confirmations[-1])
            ),
            "invalidation_level": invalidation_level,
        }
        if str(scenario.wave_type).lower() == "correction":
            details["correction_metrics"] = dict(scenario.rule_eval.metrics)
            details["prior_impulse_direction"] = "bear" if scenario.bullish else "bull"
            details["trend_context"] = "counter_trend"
        if str(scenario.wave_type).lower() == "impulse":
            wave5_targets: Dict[str, float] = {}
            for src_key, out_key in (
                ("wave5_target_equal_wave1", "equal_wave1"),
                ("wave5_target_0618_wave3", "wave3_0_618"),
                ("wave5_target_zone_low", "zone_low"),
                ("wave5_target_zone_high", "zone_high"),
            ):
                value = scenario.rule_eval.metrics.get(src_key)
                if value is None:
                    continue
                try:
                    wave5_targets[out_key] = float(value)
                except Exception:
                    continue
            if wave5_targets:
                details["wave5_targets"] = wave5_targets
        if scenario.fallback_candidate and scenario.validated_wave_type:
            details["candidate_validates_as"] = str(scenario.validated_wave_type).lower()

        return ElliottWaveResult(
            wave_type=scenario.wave_type,
            wave_sequence=piv_seq,
            confidence=float(scenario.confidence),
            start_index=start_index,
            end_index=end_index,
            start_time=PatternResultBase.resolve_time(self.times, start_index),
            end_time=PatternResultBase.resolve_time(self.times, end_index),
            details=details,
        )

    def build_fallback(self, threshold_base: float, min_distance: int) -> Optional[ElliottWaveResult]:
        if not bool(self.config.include_fallback_candidate):
            return None

        n = int(self.close.size)
        if n < 2:
            return None

        thr_cand = float(min(0.2, max(0.01, threshold_base)))
        piv_idx, _ = _zigzag_pivots_indices(self.close, thr_cand)
        piv_idx = _enforce_min_distance_on_pivots(piv_idx, self.close, max(1, min_distance))
        if len(piv_idx) < 1:
            return None

        synthetic_terminal_pivot = False
        if int(piv_idx[-1]) != int(n - 1):
            piv_idx = list(piv_idx) + [int(n - 1)]
            synthetic_terminal_pivot = True
        if len(piv_idx) < 2:
            return None

        pattern_types = _normalize_pattern_types(self.config)
        preferred_len = 6 if "impulse" in pattern_types else 4
        seq_len = min(preferred_len, len(piv_idx))
        piv_seq = [int(i) for i in piv_idx[-seq_len:]]
        bullish = bool(float(self.close[piv_seq[-1]]) > float(self.close[piv_seq[0]]))

        rule_eval = ElliottRuleEvaluation(valid=False, fib_score=0.0, metrics={}, violations=["fallback_candidate"])
        validated_wave_type: Optional[str] = None
        if len(piv_seq) == 6 and "impulse" in pattern_types:
            rule_eval = _evaluate_impulse_rules(self.close, piv_seq, bullish=bullish, config=self.config)
            if rule_eval.valid:
                validated_wave_type = "Impulse"
        elif len(piv_seq) == 4 and "correction" in pattern_types:
            rule_eval = _evaluate_correction_rules(
                self.close,
                piv_seq,
                bullish=bullish,
                config=self.config,
            )
            if rule_eval.valid:
                validated_wave_type = "Correction"

        conf_raw = 0.30 * float(rule_eval.fib_score) if len(piv_seq) in (4, 6) else 0.2
        conf = float(min(0.45, max(0.1, conf_raw)))

        scenario = ElliottScenario(
            pivots=piv_seq,
            bullish=bullish,
            confidence=conf,
            cls_score=0.5,
            rule_eval=rule_eval,
            threshold_used=thr_cand,
            min_distance_used=int(min_distance),
            fallback_candidate=True,
            synthetic_terminal_pivot=bool(synthetic_terminal_pivot),
            wave_type="Candidate",
            validated_wave_type=validated_wave_type,
            classification_available=False,
            pivot_confirmations=([True] * max(0, len(piv_seq) - 1)) + ([False] if piv_seq else []),
        )
        return self.build_result(scenario)


def detect_elliott_waves(df: pd.DataFrame, config: Optional[ElliottWaveConfig] = None) -> List[ElliottWaveResult]:
    if config is None:
        config = ElliottWaveConfig()

    if not isinstance(df, pd.DataFrame) or "close" not in df.columns:
        return []

    if "time" in df.columns:
        try:
            t = to_float_np(df["time"])
        except (TypeError, ValueError):
            t = np.asarray([], dtype=float)
    else:
        t = np.asarray([], dtype=float)
    c = to_float_np(df["close"])
    n = int(c.size)
    pattern_types = _normalize_pattern_types(config)
    min_pattern_pivots = 6 if "impulse" in pattern_types else 4
    min_floor = (
        int(getattr(config, "min_impulse_bars", 30))
        if min_pattern_pivots == 6
        else int(getattr(config, "min_correction_bars", 20))
    )
    min_required = max(min_pattern_pivots * max(1, int(config.min_distance)), min_floor)
    if n < min_required:
        return []

    analyzer = ElliottWaveAnalyzer(c, t, config)

    results_by_key: Dict[Tuple[str, Tuple[int, ...]], ElliottWaveResult] = {}

    if bool(getattr(config, "autotune", False)):
        thr_list = (
            config.tune_thresholds
            if isinstance(config.tune_thresholds, list) and len(config.tune_thresholds) > 0
            else [0.2, 0.3, 0.5, 0.75, 1.0]
        )
        md_base = int(max(1, config.min_distance))
        md_list = (
            config.tune_min_distance
            if isinstance(config.tune_min_distance, list) and len(config.tune_min_distance) > 0
            else sorted({max(1, md_base - 2), md_base, md_base + 2, md_base + 4})
        )

        seen_pivot_signatures: set[Tuple[int, ...]] = set()
        prior_scenario_signature: Tuple[Tuple[str, bool, int, int, int], ...] = tuple()
        repeated_scenario_runs = 0
        stop_after_repeats = max(0, int(getattr(config, "autotune_early_stop_repeats", 3)))
        scenario_overlap_ratio = float(max(0.0, min(1.0, getattr(config, "autotune_scenario_overlap_ratio", 0.9))))
        stop_autotune = False
        for thr_val in thr_list:
            try:
                thr_f = float(thr_val)
            except Exception:
                continue
            for md in md_list:
                try:
                    md_i = int(md)
                except Exception:
                    continue
                if bool(getattr(config, "autotune_skip_repeated_pivots", True)):
                    signature = _pivot_signature_for_settings(c, thr_f, md_i)
                    if len(signature) > 1 and signature in seen_pivot_signatures:
                        continue
                    if len(signature) > 1:
                        seen_pivot_signatures.add(signature)

                scenarios = analyzer.analyze_once(thr_f, md_i)
                if stop_after_repeats > 0:
                    current_signature = _scenario_signature(scenarios)
                    if current_signature and _scenario_signatures_overlap(
                        current_signature,
                        prior_scenario_signature,
                        overlap_ratio=scenario_overlap_ratio,
                    ):
                        repeated_scenario_runs += 1
                    else:
                        repeated_scenario_runs = 0
                    prior_scenario_signature = current_signature
                for scenario in scenarios:
                    result = analyzer.build_result(scenario)
                    _upsert_elliott_result(results_by_key, result)
                if stop_after_repeats > 0 and repeated_scenario_runs >= stop_after_repeats:
                    stop_autotune = True
                    break
            if stop_autotune:
                break
    else:
        thr = float(config.swing_threshold_pct if config.swing_threshold_pct is not None else config.min_prominence_pct)
        scenarios = analyzer.analyze_once(thr, int(max(1, config.min_distance)))
        for scenario in scenarios:
            _upsert_elliott_result(results_by_key, analyzer.build_result(scenario))

    recent_bars = int(max(1, getattr(config, "recent_bars", 3)))
    results = _filter_overlapping_corrections(list(results_by_key.values()))
    results.sort(key=_result_sort_key)
    has_recent = any(int(getattr(r, "end_index", -10)) >= int(n - recent_bars) for r in results)
    if not has_recent:
        thr_base = float(config.swing_threshold_pct if config.swing_threshold_pct is not None else config.min_prominence_pct)
        fallback = analyzer.build_fallback(thr_base, int(max(1, config.min_distance)))
        if fallback is not None:
            _upsert_elliott_result(results_by_key, fallback)
            results = _filter_overlapping_corrections(list(results_by_key.values()))

    results.sort(key=_result_sort_key)
    k = int(getattr(config, "top_k", 10))
    if k > 0:
        results = results[:k]
    return results
