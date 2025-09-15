from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from ..utils.utils import to_float_np

@dataclass
class ElliottWaveConfig:
    """Configuration for Elliott Wave detection."""
    # Legacy options (kept for API compatibility)
    min_prominence_pct: float = 0.5
    min_distance: int = 5
    # New options
    swing_threshold_pct: Optional[float] = None  # ZigZag swing threshold in percent (fallbacks to min_prominence_pct)
    gmm_components: int = 2  # Impulsive vs. Corrective
    wave_min_len: int = 3
    # Autotuning controls
    autotune: bool = True
    tune_thresholds: Optional[List[float]] = None  # e.g., [0.2, 0.35, 0.5]
    tune_min_distance: Optional[List[int]] = None  # e.g., [4, 6, 8]

@dataclass
class ElliottWaveResult:
    """Result of an Elliott Wave pattern detection."""
    wave_type: str  # "Impulse" or "Correction"
    wave_sequence: List[int]
    confidence: float
    start_index: int
    end_index: int
    start_time: Optional[float]
    end_time: Optional[float]
    details: Dict[str, Any] = field(default_factory=dict)

def _zigzag_pivots_indices(close: np.ndarray, threshold_pct: float) -> Tuple[List[int], List[str]]:
    """Lightweight ZigZag pivot detection returning indices and directions.

    Ports the logic used by core.simplify ZigZag to avoid heavy imports and circular deps.
    threshold_pct is percent move required to confirm a swing.
    """
    n = int(close.size)
    if n <= 1:
        return [], []
    piv_idx: List[int] = []
    piv_dir: List[str] = []  # 'up' or 'down' for the completed swing ending at the pivot
    pivot_i = 0
    pivot_p = float(close[0])
    trend: Optional[str] = None
    last_ext_i = 0
    last_ext_p = float(close[0])
    for i in range(1, n):
        p = float(close[i])
        if trend is None:
            change = (p - pivot_p) / pivot_p * 100.0 if pivot_p != 0 else 0.0
            if abs(change) >= threshold_pct:
                trend = 'up' if change > 0 else 'down'
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
        if trend == 'up':
            if p > last_ext_p:
                last_ext_p = p
                last_ext_i = i
            retr = (last_ext_p - p) / last_ext_p * 100.0 if last_ext_p != 0 else 0.0
            if retr >= threshold_pct:
                piv_idx.append(last_ext_i)
                piv_dir.append('up')
                trend = 'down'
                pivot_i = i
                pivot_p = p
                last_ext_i = i
                last_ext_p = p
        else:  # down
            if p < last_ext_p:
                last_ext_p = p
                last_ext_i = i
            retr = (p - last_ext_p) / abs(last_ext_p) * 100.0 if last_ext_p != 0 else 0.0
            if retr >= threshold_pct:
                piv_idx.append(last_ext_i)
                piv_dir.append('down')
                trend = 'up'
                pivot_i = i
                pivot_p = p
                last_ext_i = i
                last_ext_p = p
    # Append final extreme if missing
    if len(piv_idx) == 0 or piv_idx[-1] != last_ext_i:
        piv_idx.append(last_ext_i)
        piv_dir.append('up' if trend == 'up' else 'down' if trend == 'down' else 'flat')
    return piv_idx, piv_dir

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
    """Extracts features for each wave."""
    features = []
    for start, end in waves:
        if end <= start: continue
        
        duration = end - start
        price_change = close[end] - close[start]
        slope = price_change / duration if duration > 0 else 0.0
        
        # Normalize price change by the start price
        normalized_price_change = price_change / close[start] if close[start] != 0 else 0.0
        
        features.append([duration, normalized_price_change, slope])
        
    return np.array(features)

def _classify_waves(features: np.ndarray, config: ElliottWaveConfig) -> Tuple[np.ndarray, Optional[GaussianMixture], Optional[StandardScaler], Optional[np.ndarray], Optional[int]]:
    """Classify waves with GMM; return labels, model, scaler, probabilities and impulsive cluster id."""
    if features.shape[0] < config.gmm_components:
        return np.array([]), None, None, None, None
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    gmm = GaussianMixture(n_components=config.gmm_components, random_state=42)
    gmm.fit(scaled)
    labels = gmm.predict(scaled)
    probs = gmm.predict_proba(scaled)
    # Choose cluster with largest absolute normalized move (feature index 1)
    impulsive_cluster = int(np.argmax(np.abs(gmm.means_[:, 1])))
    wave_types = (labels == impulsive_cluster).astype(int)
    return wave_types, gmm, scaler, probs, impulsive_cluster

def _impulse_rules_and_score(c: np.ndarray, piv: List[int], bullish: bool) -> Tuple[bool, float, Dict[str, float]]:
    """Validate Elliott 5-wave impulse rules and compute a Fibonacci-based score.

    piv: 6 pivot indices W0..W5. bullish=True for uptrend impulse, False for downtrend.
    Returns (valid, score, metrics).
    """
    if len(piv) != 6:
        return False, 0.0, {}
    w = [float(c[i]) for i in piv]
    # Segment lengths L1..L5
    L = [w[i + 1] - w[i] for i in range(5)]
    # Sign expectations
    sgn = 1.0 if bullish else -1.0
    expected = [sgn, -sgn, sgn, -sgn, sgn]
    if not all((L[i] > 0) if expected[i] > 0 else (L[i] < 0) for i in range(5)):
        return False, 0.0, {}
    absL = [abs(x) for x in L]
    # Rule: Wave 2 does not retrace beyond start of wave 1
    if bullish:
        if w[2] <= w[0]:
            return False, 0.0, {}
    else:
        if w[2] >= w[0]:
            return False, 0.0, {}
    # Rule: Wave 3 is not the shortest among 1,3,5
    if absL[2] < min(absL[0], absL[2], absL[4]):
        return False, 0.0, {}
    # Rule: Wave 4 does not overlap with territory of Wave 1 (cash indices exception ignored)
    # For bullish, require W4 > W1 high; for bearish, require W4 < W1 low
    if bullish:
        if w[4] <= w[1]:
            return False, 0.0, {}
    else:
        if w[4] >= w[1]:
            return False, 0.0, {}
    # Fibonacci heuristics scoring
    score = 0.0
    # Retracement ratios
    r2 = absL[1] / absL[0] if absL[0] > 0 else 0.0
    r4 = absL[3] / absL[2] if absL[2] > 0 else 0.0
    e3 = absL[2] / absL[0] if absL[0] > 0 else 0.0
    # Wave 5 relation to Wave 1 or 0.618*(W3-W1)
    rel53 = absL[4] / absL[0] if absL[0] > 0 else 0.0
    rel5_alt = absL[4] / abs((w[3] - w[1])) if (w[3] != w[1]) else 0.0
    def window_hit(v: float, lo: float, hi: float, w: float = 0.2) -> float:
        # Score 1.0 when inside [lo,hi]; taper linearly to 0 at +-w*range outside
        rng = hi - lo
        if rng <= 0:
            return 0.0
        if lo <= v <= hi:
            return 1.0
        if v < lo:
            d = (lo - v) / (w * rng)
            return max(0.0, 1.0 - d)
        d = (v - hi) / (w * rng)
        return max(0.0, 1.0 - d)
    s2 = window_hit(r2, 0.382, 0.618)
    s4 = window_hit(r4, 0.236, 0.382)
    s3 = window_hit(e3, 1.618, 2.618)
    s5 = max(window_hit(rel53, 0.8, 1.2), window_hit(rel5_alt, 0.55, 0.75))
    score = 0.25 * (s2 + s3 + s4 + s5)
    metrics = {
        'r2': r2,
        'r4': r4,
        'e3': e3,
        'rel53': rel53,
        'rel5_alt': rel5_alt,
        'fib_score': score,
    }
    return True, float(score), metrics

def detect_elliott_waves(df: pd.DataFrame, config: Optional[ElliottWaveConfig] = None) -> List[ElliottWaveResult]:
    if config is None:
        config = ElliottWaveConfig()

    if not isinstance(df, pd.DataFrame) or 'close' not in df.columns:
        return []

    t = to_float_np(df.get('time', pd.Series(df.index).astype(np.int64) // 10**9))
    c = to_float_np(df['close'])
    n = c.size
    min_required = max(6 * max(1, config.min_distance), 30)
    if n < min_required:
        return []

    def _run_once(thr_val: float, min_dist: int) -> List[ElliottWaveResult]:
        piv_idx, _piv_dir = _zigzag_pivots_indices(c, thr_val)
        if len(piv_idx) < 6:
            return []
        waves = _segment_waves_from_pivots(piv_idx)
        if not waves:
            return []
        features = _extract_wave_features(waves, c)
        if features.shape[0] == 0:
            return []
        wave_types, gmm, scaler, probs, impulsive_cluster = _classify_waves(features, config)
        if gmm is None or scaler is None or probs is None or impulsive_cluster is None:
            return []
        cluster_means = gmm.means_
        out: List[ElliottWaveResult] = []
        total_waves = len(waves)
        for k in range(0, total_waves - 4):
            piv_seq = piv_idx[k:k + 6]
            if len(piv_seq) < 6:
                continue
            # Duration filter
            durations_ok = True
            for j in range(5):
                if (piv_seq[j + 1] - piv_seq[j]) < max(1, min_dist):
                    durations_ok = False
                    break
            if not durations_ok:
                continue
            bullish = c[piv_seq[-1]] > c[piv_seq[0]]
            bearish = c[piv_seq[-1]] < c[piv_seq[0]]
            if not (bullish or bearish):
                continue
            valid, fib_score, metrics = _impulse_rules_and_score(c, piv_seq, bullish=bullish)
            if not valid:
                continue
            cluster_for_trend = int(np.argmax(cluster_means[:, 1])) if bullish else int(np.argmin(cluster_means[:, 1]))
            window_probs = probs[k:k + 5, cluster_for_trend]
            p13 = (window_probs[0] + window_probs[2] + window_probs[4]) / 3.0
            p24 = (window_probs[1] + window_probs[3]) / 2.0
            cls_score = 0.5 * p13 + 0.5 * (1.0 - p24)
            confidence = float(min(1.0, max(0.0, 0.6 * fib_score + 0.4 * cls_score)))
            start_index = int(piv_seq[0])
            end_index = int(piv_seq[-1])
            # Build labeled wave point metadata
            labels = [f"W{j}" for j in range(len(piv_seq))]
            wave_points_labeled = []
            for j, idx in enumerate(piv_seq):
                try:
                    ti = float(t[idx]) if t.size > int(idx) else None
                except Exception:
                    ti = None
                wave_points_labeled.append({
                    "label": labels[j],
                    "index": int(idx),
                    "time": ti,
                    "price": float(c[int(idx)])
                })

            out.append(ElliottWaveResult(
                wave_type="Impulse",
                wave_sequence=[int(x) for x in piv_seq],
                confidence=confidence,
                start_index=start_index,
                end_index=end_index,
                start_time=float(t[start_index]) if t.size > start_index else None,
                end_time=float(t[end_index]) if t.size > end_index else None,
                details={
                    "wave_points": [float(c[i]) for i in piv_seq],
                    "wave_points_labeled": wave_points_labeled,
                    "bullish": bool(bullish),
                    "trend": ("bull" if bullish else "bear"),
                    "fib_metrics": metrics,
                    "cls_probs_impulsive": [float(p) for p in window_probs.tolist()],
                    "tuned_threshold_pct": float(thr_val),
                    "min_distance_used": int(min_dist),
                }
            ))
        return out

    results: List[ElliottWaveResult] = []
    if getattr(config, 'autotune', False):
        thr_list = config.tune_thresholds if isinstance(config.tune_thresholds, list) and len(config.tune_thresholds) > 0 else [0.2, 0.3, 0.5, 0.75, 1.0]
        md_base = int(max(1, config.min_distance))
        md_list = config.tune_min_distance if isinstance(config.tune_min_distance, list) and len(config.tune_min_distance) > 0 else sorted({max(1, md_base - 2), md_base, md_base + 2, md_base + 4})
        seen_keys = set()
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
                cand = _run_once(thr_f, md_i)
                for r in cand:
                    key = tuple(int(i) for i in r.wave_sequence)
                    if key in seen_keys:
                        # Keep best confidence if duplicate
                        # Replace only if better
                        for j in range(len(results)):
                            if tuple(int(i) for i in results[j].wave_sequence) == key and r.confidence > results[j].confidence:
                                results[j] = r
                                break
                        continue
                    seen_keys.add(key)
                    results.append(r)
    else:
        thr = float(config.swing_threshold_pct if config.swing_threshold_pct is not None else config.min_prominence_pct)
        results = _run_once(thr, int(max(1, config.min_distance)))

    # Ensure we have a recent forming candidate: if no result ends near the latest bars,
    # produce a low-confidence fallback candidate from recent pivots.
    try:
        recent_bars = 3
        has_recent = any((int(getattr(r, 'end_index', -10)) >= int(n - recent_bars)) for r in results)
        if not has_recent:
            # Use a more permissive threshold for fallback to ensure pivots exist
            thr_base = float(config.swing_threshold_pct if config.swing_threshold_pct is not None else config.min_prominence_pct)
            thr_cand = float(min(0.2, max(0.01, thr_base)))
            piv_idx, _ = _zigzag_pivots_indices(c, thr_cand)
            if isinstance(piv_idx, list) and len(piv_idx) >= 1:
                # Ensure candidate extends to the most recent bar to mark it as forming
                if int(piv_idx[-1]) != int(n - 1):
                    piv_idx = list(piv_idx) + [int(n - 1)]
                if len(piv_idx) >= 2:
                    seq_len = min(6, len(piv_idx))
                    piv_seq = [int(i) for i in piv_idx[-seq_len:]]
                    start_index = int(piv_seq[0])
                    end_index = int(piv_seq[-1])
                    bullish = bool(float(c[end_index]) > float(c[start_index]))
                    # Try a lightweight fib-based confidence when we have 6 pivots; else use a small constant
                    fib_score = 0.0
                    fib_metrics: Dict[str, float] = {}
                    if len(piv_seq) == 6:
                        valid, fib_score, fib_metrics = _impulse_rules_and_score(c, piv_seq, bullish=bullish)
                        if not valid:
                            # still treat as candidate, but keep low confidence
                            fib_score = max(0.0, float(fib_score))
                    # Confidence: keep low; softly reflect fib_score if available
                    conf = float(min(0.5, max(0.1, 0.3 * float(fib_score) if len(piv_seq) == 6 else 0.2)))
                    # Labeled points for fallback candidate
                    labels = [f"W{j}" for j in range(len(piv_seq))]
                    wave_points_labeled = []
                    for j, idx in enumerate(piv_seq):
                        try:
                            ti = float(t[idx]) if t.size > int(idx) else None
                        except Exception:
                            ti = None
                        wave_points_labeled.append({
                            "label": labels[j],
                            "index": int(idx),
                            "time": ti,
                            "price": float(c[int(idx)])
                        })

                    details: Dict[str, Any] = {
                        "wave_points": [float(c[i]) for i in piv_seq],
                        "wave_points_labeled": wave_points_labeled,
                        "bullish": bool(bullish),
                        "trend": ("bull" if bullish else "bear"),
                        "fallback_candidate": True,
                        "pivot_count": int(len(piv_seq)),
                        "tuned_threshold_pct": float(thr_cand),
                    }
                    if fib_metrics:
                        details["fib_metrics"] = fib_metrics
                    results.append(ElliottWaveResult(
                        wave_type="Impulse",
                        wave_sequence=[int(x) for x in piv_seq],
                        confidence=conf,
                        start_index=start_index,
                        end_index=end_index,
                        start_time=float(t[start_index]) if t.size > start_index else None,
                        end_time=float(t[end_index]) if t.size > end_index else None,
                        details=details,
                    ))
    except Exception:
        # best-effort candidate; ignore any issues here
        pass

    results.sort(key=lambda r: (r.confidence, r.end_index), reverse=True)
    return results
