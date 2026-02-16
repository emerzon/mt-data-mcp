from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from ..utils.utils import to_float_np
from .common import PatternResultBase


@dataclass
class ElliottWaveConfig:
    """Configuration for Elliott Wave detection."""

    # Legacy options (kept for API compatibility)
    min_prominence_pct: float = 0.5
    min_distance: int = 5

    # New options
    swing_threshold_pct: Optional[float] = None  # ZigZag swing threshold in percent
    gmm_components: int = 2  # Impulsive vs corrective clusters
    wave_min_len: int = 3

    # Autotuning controls
    autotune: bool = True
    tune_thresholds: Optional[List[float]] = None
    tune_min_distance: Optional[List[int]] = None

    # Analyzer controls
    min_confidence: float = 0.0
    top_k: int = 10
    include_fallback_candidate: bool = True
    recent_bars: int = 3
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
    wave_type: str = "Impulse"


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

    for i in range(1, n):
        p = float(close[i])
        if not np.isfinite(p):
            continue

        if trend is None:
            change = (p - pivot_p) / pivot_p * 100.0 if pivot_p != 0 else 0.0
            if abs(change) >= threshold_pct:
                trend = "up" if change > 0 else "down"
                last_ext_i = i
                last_ext_p = p
                piv_idx.append(pivot_i)
                piv_dir.append(trend)
            else:
                if p > last_ext_p:
                    last_ext_p = p
                    last_ext_i = i
                if p < last_ext_p and trend is None:
                    last_ext_p = p
                    last_ext_i = i
                continue

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


def _extract_wave_features(waves: List[Tuple[int, int]], close: np.ndarray) -> np.ndarray:
    """Extract per-wave features used for optional clustering."""

    features: List[List[float]] = []
    for start, end in waves:
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
        features.append([duration, normalized_price_change, slope])

    if not features:
        return np.empty((0, 3), dtype=float)
    return np.asarray(features, dtype=float)


def _classify_waves(
    features: np.ndarray,
    config: ElliottWaveConfig,
) -> Tuple[np.ndarray, Optional[GaussianMixture], Optional[StandardScaler], Optional[np.ndarray], Optional[int]]:
    """Classify waves with GMM; return labels, model, scaler, probabilities and impulsive cluster id."""

    if features.shape[0] < config.gmm_components:
        return np.array([]), None, None, None, None

    try:
        scaler = StandardScaler()
        scaled = scaler.fit_transform(features)
        if not np.all(np.isfinite(scaled)):
            return np.array([]), None, None, None, None

        gmm = GaussianMixture(n_components=config.gmm_components, random_state=42)
        gmm.fit(scaled)
        labels = gmm.predict(scaled)
        probs = gmm.predict_proba(scaled)
        impulsive_cluster = int(np.argmax(np.abs(gmm.means_[:, 1])))
        wave_types = (labels == impulsive_cluster).astype(int)
        return wave_types, gmm, scaler, probs, impulsive_cluster
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


def _evaluate_impulse_rules(c: np.ndarray, piv: List[int], bullish: bool) -> ElliottRuleEvaluation:
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
        violations.append("alternation_failed")

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
    fib_score = float(0.25 * (s2 + s3 + s4 + s5))

    metrics = {
        "r2": float(r2),
        "r4": float(r4),
        "e3": float(e3),
        "rel53": float(rel53),
        "rel5_alt": float(rel5_alt),
        "fib_score": fib_score,
    }
    return ElliottRuleEvaluation(valid=(len(violations) == 0), fib_score=fib_score, metrics=metrics, violations=violations)


def _impulse_rules_and_score(c: np.ndarray, piv: List[int], bullish: bool) -> Tuple[bool, float, Dict[str, float]]:
    """Backward-compatible wrapper around rule evaluation."""

    ev = _evaluate_impulse_rules(c, piv, bullish)
    return ev.valid, ev.fib_score, ev.metrics


def _evaluate_correction_rules(c: np.ndarray, piv: List[int], bullish: bool) -> ElliottRuleEvaluation:
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
        violations.append("alternation_failed")

    # B should not fully retrace wave A start in a standard zigzag interpretation.
    if bullish and w[2] <= w[0]:
        violations.append("waveB_over_retrace")
    if (not bullish) and w[2] >= w[0]:
        violations.append("waveB_over_retrace")

    # C should have meaningful size versus A to avoid noise classifications.
    if absL[2] < 0.5 * absL[0]:
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
) -> float:
    if probs is None or cluster_means is None:
        return 0.5
    if probs.shape[0] < (k + window_len):
        return 0.5
    if cluster_means.ndim != 2 or cluster_means.shape[0] < 1:
        return 0.5

    cluster_for_trend = int(np.argmax(cluster_means[:, 1])) if bullish else int(np.argmin(cluster_means[:, 1]))
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

        features = _extract_wave_features(waves, self.close)
        if features.shape[0] == 0:
            return []

        _, gmm, _, probs, _ = _classify_waves(features, self.config)
        cluster_means = gmm.means_ if gmm is not None else None

        out: List[ElliottScenario] = []
        pattern_types = _normalize_pattern_types(self.config)
        total_waves = len(waves)
        min_wave_span = int(max(1, max(min_distance, self.config.wave_min_len)))

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

                rule_eval = _evaluate_impulse_rules(self.close, piv_seq, bullish=bullish)
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
                )
                confidence = float(min(1.0, max(0.0, 0.65 * rule_eval.fib_score + 0.35 * cls_score)))
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
                    )
                )

        if "correction" in pattern_types:
            for k in range(0, total_waves - 2):
                piv_seq = [int(x) for x in piv_idx[k : k + 4]]
                if len(piv_seq) < 4:
                    continue

                if any((piv_seq[j + 1] - piv_seq[j]) < min_wave_span for j in range(3)):
                    continue

                bullish = bool(self.close[piv_seq[-1]] > self.close[piv_seq[0]])
                bearish = bool(self.close[piv_seq[-1]] < self.close[piv_seq[0]])
                if not (bullish or bearish):
                    continue

                rule_eval = _evaluate_correction_rules(self.close, piv_seq, bullish=bullish)
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
                )
                confidence = float(min(1.0, max(0.0, 0.60 * rule_eval.fib_score + 0.40 * cls_score)))
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
            try:
                ti = float(self.times[idx]) if self.times.size > idx else None
            except Exception:
                ti = None
            wave_points_labeled.append(
                {
                    "label": labels[j],
                    "index": int(idx),
                    "time": ti,
                    "price": float(self.close[idx]),
                }
            )

        invalidation_level = float(self.close[piv_seq[0]]) if piv_seq else None

        details: Dict[str, Any] = {
            "wave_points": [float(self.close[i]) for i in piv_seq],
            "wave_points_labeled": wave_points_labeled,
            "bullish": bool(scenario.bullish),
            "trend": ("bull" if scenario.bullish else "bear"),
            "pattern_family": str(scenario.wave_type).lower(),
            "fib_metrics": dict(scenario.rule_eval.metrics),
            "rule_violations": list(scenario.rule_eval.violations),
            "cls_score": float(scenario.cls_score),
            "tuned_threshold_pct": float(scenario.threshold_used),
            "min_distance_used": int(scenario.min_distance_used),
            "fallback_candidate": bool(scenario.fallback_candidate),
            "invalidation_level": invalidation_level,
        }
        if str(scenario.wave_type).lower() == "correction":
            details["correction_metrics"] = dict(scenario.rule_eval.metrics)

        return ElliottWaveResult(
            wave_type=scenario.wave_type,
            wave_sequence=piv_seq,
            confidence=float(scenario.confidence),
            start_index=start_index,
            end_index=end_index,
            start_time=float(self.times[start_index]) if self.times.size > start_index else None,
            end_time=float(self.times[end_index]) if self.times.size > end_index else None,
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

        if int(piv_idx[-1]) != int(n - 1):
            piv_idx = list(piv_idx) + [int(n - 1)]
        if len(piv_idx) < 2:
            return None

        pattern_types = _normalize_pattern_types(self.config)
        preferred_len = 6 if "impulse" in pattern_types else 4
        seq_len = min(preferred_len, len(piv_idx))
        piv_seq = [int(i) for i in piv_idx[-seq_len:]]
        bullish = bool(float(self.close[piv_seq[-1]]) > float(self.close[piv_seq[0]]))

        rule_eval = ElliottRuleEvaluation(valid=False, fib_score=0.0, metrics={}, violations=["fallback_candidate"])
        wave_type = "Candidate"
        if len(piv_seq) == 6 and "impulse" in pattern_types:
            rule_eval = _evaluate_impulse_rules(self.close, piv_seq, bullish=bullish)
            if rule_eval.valid:
                wave_type = "Impulse"
        elif len(piv_seq) == 4 and "correction" in pattern_types:
            rule_eval = _evaluate_correction_rules(self.close, piv_seq, bullish=bullish)
            if rule_eval.valid:
                wave_type = "Correction"

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
            wave_type=wave_type,
        )
        return self.build_result(scenario)


def detect_elliott_waves(df: pd.DataFrame, config: Optional[ElliottWaveConfig] = None) -> List[ElliottWaveResult]:
    if config is None:
        config = ElliottWaveConfig()

    if not isinstance(df, pd.DataFrame) or "close" not in df.columns:
        return []

    t = to_float_np(df.get("time", pd.Series(df.index).astype(np.int64) // 10**9))
    c = to_float_np(df["close"])
    n = int(c.size)
    pattern_types = _normalize_pattern_types(config)
    min_pattern_pivots = 6 if "impulse" in pattern_types else 4
    min_floor = 30 if min_pattern_pivots == 6 else 20
    min_required = max(min_pattern_pivots * max(1, int(config.min_distance)), min_floor)
    if n < min_required:
        return []

    analyzer = ElliottWaveAnalyzer(c, t, config)

    results: List[ElliottWaveResult] = []
    seen_keys = set()

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

                scenarios = analyzer.analyze_once(thr_f, md_i)
                for scenario in scenarios:
                    result = analyzer.build_result(scenario)
                    key = (result.wave_type, tuple(int(i) for i in result.wave_sequence))
                    if key in seen_keys:
                        for j in range(len(results)):
                            old_key = (results[j].wave_type, tuple(int(i) for i in results[j].wave_sequence))
                            if old_key == key and result.confidence > results[j].confidence:
                                results[j] = result
                                break
                        continue
                    seen_keys.add(key)
                    results.append(result)
    else:
        thr = float(config.swing_threshold_pct if config.swing_threshold_pct is not None else config.min_prominence_pct)
        scenarios = analyzer.analyze_once(thr, int(max(1, config.min_distance)))
        for scenario in scenarios:
            results.append(analyzer.build_result(scenario))

    results.sort(key=lambda r: (float(r.confidence), int(r.end_index)), reverse=True)

    recent_bars = int(max(1, getattr(config, "recent_bars", 3)))
    has_recent = any(int(getattr(r, "end_index", -10)) >= int(n - recent_bars) for r in results)
    if not has_recent:
        thr_base = float(config.swing_threshold_pct if config.swing_threshold_pct is not None else config.min_prominence_pct)
        fallback = analyzer.build_fallback(thr_base, int(max(1, config.min_distance)))
        if fallback is not None:
            fallback_key = (fallback.wave_type, tuple(int(i) for i in fallback.wave_sequence))
            if fallback_key not in seen_keys:
                results.append(fallback)
            else:
                for j in range(len(results)):
                    key_j = (results[j].wave_type, tuple(int(i) for i in results[j].wave_sequence))
                    if key_j == fallback_key and fallback.confidence > results[j].confidence:
                        results[j] = fallback
                        break

    results.sort(key=lambda r: (float(r.confidence), int(r.end_index)), reverse=True)
    k = int(getattr(config, "top_k", 10))
    if k > 0:
        results = results[:k]
    return results
