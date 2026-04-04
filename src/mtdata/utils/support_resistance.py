"""Weighted support/resistance detection from historical retests."""

from __future__ import annotations

from datetime import datetime, timezone
import math
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..utils.constants import TIMEFRAME_SECONDS as _TIMEFRAME_SECONDS

_METHOD_NAME = "weighted_retests"
_DEFAULT_REACTION_BARS = 6
_DEFAULT_ADX_PERIOD = 14
_DEFAULT_BOUNCE_WEIGHT = 0.8
_DEFAULT_ADX_WEIGHT = 0.35
_DEFAULT_SWING_REVERSAL_ATR = 1.3
_DEFAULT_EPISODE_TOUCH_DECAY = 0.25
_DEFAULT_ADAPTIVE_TOLERANCE_ATR_MULT = 0.25
_DEFAULT_ADAPTIVE_TOLERANCE_MIN_SCALE = 0.75
_DEFAULT_ADAPTIVE_TOLERANCE_MAX_SCALE = 1.85
_DEFAULT_ADAPTIVE_REACTION_MIN_SCALE = 0.67
_DEFAULT_ADAPTIVE_REACTION_MAX_SCALE = 1.6
_DEFAULT_ADAPTIVE_RECENT_WINDOW = 8
_DEFAULT_ZONE_STD_MULT = 1.5
_DEFAULT_ZONE_ATR_MULT = 0.2
_DEFAULT_BREAKOUT_BUFFER_ATR_MULT = 0.15
_DEFAULT_BREAKOUT_PENALTY_BASE = 0.35
_DEFAULT_BREAKOUT_PENALTY_ATR_MULT = 0.25
_DEFAULT_ROLE_REVERSAL_BONUS = 0.65
_DEFAULT_MTF_DEDUPE_FACTOR = 0.35
_DEFAULT_MTF_CONFIRMATION_BONUS = 0.2
_AUTO_TIMEFRAMES = ("M15", "H1", "H4", "D1")
_TIMEFRAME_WEIGHTS = {
    "M15": 0.9,
    "H1": 1.0,
    "H4": 1.15,
    "D1": 1.3,
}
def get_auto_support_resistance_timeframes() -> tuple[str, ...]:
    return _AUTO_TIMEFRAMES


def _to_epoch(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        if isinstance(value, (int, float, np.integer, np.floating)):
            out = float(value)
            return out if math.isfinite(out) else None
        if hasattr(value, "timestamp"):
            out = float(value.timestamp())
            return out if math.isfinite(out) else None
    except Exception:
        return None
    return None


def _format_time(timestamp: Optional[float]) -> Optional[str]:
    if isinstance(timestamp, str):
        cleaned = timestamp.strip()
        return cleaned or None
    if timestamp is None or not math.isfinite(float(timestamp)):
        return None
    try:
        return datetime.fromtimestamp(float(timestamp), tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return None


def _parse_output_time(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        out = float(value)
        return out if math.isfinite(out) else None
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return None
        try:
            return datetime.strptime(cleaned, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc).timestamp()
        except Exception:
            return None
    return None


def _to_numeric_array(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame.columns:
        return np.array([], dtype=float)
    series = pd.to_numeric(frame[column], errors="coerce")
    return series.to_numpy(dtype=float, copy=False)


def _last_finite(values: np.ndarray) -> Optional[float]:
    for value in values[::-1]:
        if math.isfinite(float(value)):
            return float(value)
    return None


def _weighted_average(items: List[tuple[float, float]]) -> Optional[float]:
    total_weight = 0.0
    total_value = 0.0
    for weight, value in items:
        weight_value = float(weight)
        item_value = float(value)
        if not math.isfinite(weight_value) or not math.isfinite(item_value) or weight_value <= 0.0:
            continue
        total_weight += weight_value
        total_value += weight_value * item_value
    if total_weight <= 0.0:
        return None
    return total_value / total_weight


def _compute_atr_and_adx(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    *,
    period: int,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(closes)
    if n == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    prev_close = np.concatenate(([closes[0]], closes[:-1]))
    tr = np.maximum.reduce(
        [
            highs - lows,
            np.abs(highs - prev_close),
            np.abs(lows - prev_close),
        ]
    )

    up_move = np.diff(highs, prepend=highs[0])
    down_move = -np.diff(lows, prepend=lows[0])
    plus_dm = np.where((up_move > down_move) & (up_move > 0.0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0.0), down_move, 0.0)

    alpha = 1.0 / float(max(1, int(period)))
    tr_s = pd.Series(tr, dtype=float)
    plus_s = pd.Series(plus_dm, dtype=float)
    minus_s = pd.Series(minus_dm, dtype=float)

    atr = tr_s.ewm(alpha=alpha, adjust=False, min_periods=max(1, int(period))).mean()
    plus_smoothed = plus_s.ewm(alpha=alpha, adjust=False, min_periods=max(1, int(period))).mean()
    minus_smoothed = minus_s.ewm(alpha=alpha, adjust=False, min_periods=max(1, int(period))).mean()

    with np.errstate(divide="ignore", invalid="ignore"):
        plus_di = 100.0 * plus_smoothed / atr
        minus_di = 100.0 * minus_smoothed / atr
        dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di)

    adx = dx.ewm(alpha=alpha, adjust=False, min_periods=max(1, int(period))).mean()
    return (
        atr.to_numpy(dtype=float, copy=False),
        adx.to_numpy(dtype=float, copy=False),
    )


def _resolve_adaptive_settings(
    closes: np.ndarray,
    atr: np.ndarray,
    *,
    base_tolerance_pct: float,
    base_reaction_bars: int,
) -> Dict[str, Any]:
    atr_pct_values: List[float] = []
    for close_value, atr_value in zip(closes, atr):
        close_float = abs(float(close_value))
        atr_float = float(atr_value)
        if close_float <= 1e-9 or not math.isfinite(close_float) or not math.isfinite(atr_float) or atr_float <= 0.0:
            continue
        atr_pct_values.append(atr_float / close_float)

    if not atr_pct_values:
        return {
            "adaptive_mode": "atr_regime",
            "effective_tolerance_pct": float(base_tolerance_pct),
            "effective_reaction_bars": int(base_reaction_bars),
            "current_atr_pct": None,
            "baseline_atr_pct": None,
            "volatility_ratio": 1.0,
        }

    recent_window = max(3, min(_DEFAULT_ADAPTIVE_RECENT_WINDOW, len(atr_pct_values)))
    baseline_atr_pct = float(np.nanmedian(np.asarray(atr_pct_values, dtype=float)))
    current_atr_pct = float(np.nanmedian(np.asarray(atr_pct_values[-recent_window:], dtype=float)))
    if not math.isfinite(baseline_atr_pct) or baseline_atr_pct <= 0.0:
        baseline_atr_pct = current_atr_pct if math.isfinite(current_atr_pct) and current_atr_pct > 0.0 else 0.0

    volatility_ratio = 1.0
    if baseline_atr_pct > 0.0 and math.isfinite(current_atr_pct):
        volatility_ratio = current_atr_pct / baseline_atr_pct
    if not math.isfinite(volatility_ratio) or volatility_ratio <= 0.0:
        volatility_ratio = 1.0

    clipped_ratio = min(max(volatility_ratio, 0.5), 2.5)
    tolerance_scale = min(
        max(math.sqrt(clipped_ratio), _DEFAULT_ADAPTIVE_TOLERANCE_MIN_SCALE),
        _DEFAULT_ADAPTIVE_TOLERANCE_MAX_SCALE,
    )
    reaction_scale = min(
        max(1.0 / math.sqrt(clipped_ratio), _DEFAULT_ADAPTIVE_REACTION_MIN_SCALE),
        _DEFAULT_ADAPTIVE_REACTION_MAX_SCALE,
    )
    atr_anchor = max(0.0, current_atr_pct * _DEFAULT_ADAPTIVE_TOLERANCE_ATR_MULT)
    base_tolerance_value = max(0.0, float(base_tolerance_pct))
    scaled_tolerance = base_tolerance_value * tolerance_scale
    if volatility_ratio >= 1.0:
        effective_tolerance_pct = max(scaled_tolerance, atr_anchor)
    else:
        # In compressed regimes keep the ATR floor informative, but never let it widen
        # the zone beyond the caller's base tolerance.
        compressed_anchor = atr_anchor * clipped_ratio
        effective_tolerance_pct = min(base_tolerance_value, max(scaled_tolerance, compressed_anchor))
    effective_tolerance_pct = min(0.05, max(0.0, effective_tolerance_pct))
    effective_reaction_bars = max(1, int(round(float(base_reaction_bars) * reaction_scale)))

    return {
        "adaptive_mode": "atr_regime",
        "effective_tolerance_pct": float(effective_tolerance_pct),
        "effective_reaction_bars": int(effective_reaction_bars),
        "current_atr_pct": float(current_atr_pct) if math.isfinite(current_atr_pct) else None,
        "baseline_atr_pct": float(baseline_atr_pct) if math.isfinite(baseline_atr_pct) and baseline_atr_pct > 0.0 else None,
        "volatility_ratio": float(volatility_ratio),
    }


def _build_test(
    *,
    test_type: str,
    index: int,
    level_value: float,
    highs: np.ndarray,
    lows: np.ndarray,
    atr: np.ndarray,
    adx: np.ndarray,
    epochs: List[Optional[float]],
    reaction_bars: int,
    decay_half_life_bars: int,
) -> Optional[Dict[str, Any]]:
    future_start = int(index) + 1
    future_stop = min(len(highs), future_start + int(reaction_bars))
    if future_start >= future_stop:
        return None

    if test_type == "support":
        favorable = float(np.nanmax(highs[future_start:future_stop]) - level_value)
    else:
        favorable = float(level_value - np.nanmin(lows[future_start:future_stop]))
    if not math.isfinite(favorable) or favorable <= 0.0:
        return None

    atr_value = float(atr[index]) if index < len(atr) and math.isfinite(float(atr[index])) else float("nan")
    if not math.isfinite(atr_value) or atr_value <= 0.0:
        local_range = float(highs[index] - lows[index]) if math.isfinite(float(highs[index] - lows[index])) else 0.0
        atr_value = max(local_range, abs(level_value) * 0.001, 1e-9)

    bounce_atr = max(0.0, favorable / atr_value)
    pretest_adx = 0.0
    if index > 0 and index - 1 < len(adx):
        adx_value = float(adx[index - 1])
        if math.isfinite(adx_value):
            pretest_adx = max(0.0, adx_value)
    elif index < len(adx):
        adx_value = float(adx[index])
        if math.isfinite(adx_value):
            pretest_adx = max(0.0, adx_value)

    age_bars = max(0, len(highs) - 1 - int(index))
    half_life = max(1, int(decay_half_life_bars))
    decay_weight = math.exp(-math.log(2.0) * float(age_bars) / float(half_life))

    retest_component = decay_weight
    bounce_component = decay_weight * _DEFAULT_BOUNCE_WEIGHT * math.log1p(bounce_atr)
    adx_component = decay_weight * _DEFAULT_ADX_WEIGHT * (min(pretest_adx, 60.0) / 25.0)
    score = retest_component + bounce_component + adx_component

    return {
        "type": test_type,
        "index": int(index),
        "value": float(level_value),
        "atr_value": float(atr_value),
        "timestamp": epochs[index] if index < len(epochs) else None,
        "age_bars": int(age_bars),
        "decay_weight": float(decay_weight),
        "bounce_atr": float(bounce_atr),
        "pretest_adx": float(pretest_adx),
        "retest_component": float(retest_component),
        "bounce_component": float(bounce_component),
        "adx_component": float(adx_component),
        "score": float(score),
    }


def _collect_local_extrema_candidates(highs: np.ndarray, lows: np.ndarray) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for index in range(1, len(highs) - 1):
        low_center = float(lows[index])
        low_prev = float(lows[index - 1])
        low_next = float(lows[index + 1])
        if all(math.isfinite(v) for v in (low_center, low_prev, low_next)) and low_center <= low_prev and low_center <= low_next:
            candidates.append({"type": "support", "index": int(index), "value": float(low_center)})

        high_center = float(highs[index])
        high_prev = float(highs[index - 1])
        high_next = float(highs[index + 1])
        if all(math.isfinite(v) for v in (high_center, high_prev, high_next)) and high_center >= high_prev and high_center >= high_next:
            candidates.append({"type": "resistance", "index": int(index), "value": float(high_center)})
    candidates.sort(key=lambda item: (int(item["index"]), 0 if str(item["type"]) == "support" else 1))
    return candidates


def _is_more_extreme_candidate(candidate: Dict[str, Any], current: Dict[str, Any]) -> bool:
    candidate_value = float(candidate["value"])
    current_value = float(current["value"])
    candidate_index = int(candidate["index"])
    current_index = int(current["index"])
    candidate_type = str(candidate["type"])
    if candidate_type == "support":
        return candidate_value < current_value or (
            math.isclose(candidate_value, current_value, rel_tol=0.0, abs_tol=1e-12) and candidate_index > current_index
        )
    return candidate_value > current_value or (
        math.isclose(candidate_value, current_value, rel_tol=0.0, abs_tol=1e-12) and candidate_index > current_index
    )


def _filter_swing_candidates(
    candidates: List[Dict[str, Any]],
    *,
    atr: np.ndarray,
) -> List[Dict[str, Any]]:
    if not candidates:
        return []

    accepted: List[Dict[str, Any]] = []
    fallback_atr = 1e-9
    finite_atr = [float(value) for value in atr if math.isfinite(float(value)) and float(value) > 0.0]
    if finite_atr:
        fallback_atr = float(np.nanmedian(np.asarray(finite_atr, dtype=float)))
        if not math.isfinite(fallback_atr) or fallback_atr <= 0.0:
            fallback_atr = 1e-9

    def _atr_for_index(index: int) -> float:
        if 0 <= index < len(atr):
            value = float(atr[index])
            if math.isfinite(value) and value > 0.0:
                return value
        return fallback_atr

    for candidate in candidates:
        if not accepted:
            accepted.append(dict(candidate))
            continue

        last = accepted[-1]
        if str(candidate["type"]) == str(last["type"]):
            if _is_more_extreme_candidate(candidate, last):
                candidate_index = int(candidate["index"])
                last_index = int(last["index"])
                extension_threshold = max(_atr_for_index(candidate_index), _atr_for_index(last_index), 1e-9) * _DEFAULT_SWING_REVERSAL_ATR
                extension = abs(float(candidate["value"]) - float(last["value"]))
                if math.isfinite(extension) and extension >= extension_threshold:
                    accepted.append(dict(candidate))
                else:
                    accepted[-1] = dict(candidate)
            continue

        candidate_index = int(candidate["index"])
        last_index = int(last["index"])
        candidate_atr = _atr_for_index(candidate_index)
        last_atr = _atr_for_index(last_index)
        reversal_threshold = max(candidate_atr, last_atr, fallback_atr, 1e-9) * _DEFAULT_SWING_REVERSAL_ATR
        swing_move = abs(float(candidate["value"]) - float(last["value"]))
        if math.isfinite(swing_move) and swing_move >= reversal_threshold:
            accepted.append(dict(candidate))

    return accepted


def _collect_tests(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    *,
    epochs: List[Optional[float]],
    reaction_bars: int,
    adx_period: int,
    decay_half_life_bars: int,
    atr: Optional[np.ndarray] = None,
    adx: Optional[np.ndarray] = None,
) -> List[Dict[str, Any]]:
    atr_values = atr
    adx_values = adx
    if atr_values is None or adx_values is None:
        atr_values, adx_values = _compute_atr_and_adx(highs, lows, closes, period=adx_period)
    tests: List[Dict[str, Any]] = []
    candidates = _collect_local_extrema_candidates(highs, lows)
    swing_candidates = _filter_swing_candidates(candidates, atr=atr_values)
    for candidate in swing_candidates:
        test = _build_test(
            test_type=str(candidate["type"]),
            index=int(candidate["index"]),
            level_value=float(candidate["value"]),
            highs=highs,
            lows=lows,
            atr=atr_values,
            adx=adx_values,
            epochs=epochs,
            reaction_bars=reaction_bars,
            decay_half_life_bars=decay_half_life_bars,
        )
        if test is not None:
            tests.append(test)

    return tests


def _cluster_tests(tests: List[Dict[str, Any]], *, tolerance_pct: float) -> List[Dict[str, Any]]:
    clusters: List[Dict[str, Any]] = []
    for test in sorted(tests, key=lambda item: float(item["value"])):
        value = float(test["value"])
        best_cluster: Optional[Dict[str, Any]] = None
        best_delta: Optional[float] = None
        for cluster in clusters:
            ref = float(cluster["value"])
            threshold = max(abs(ref), abs(value), 1e-9) * float(tolerance_pct)
            delta = abs(ref - value)
            if delta <= threshold and (best_delta is None or delta < best_delta):
                best_cluster = cluster
                best_delta = delta

        score = max(float(test["score"]), 1e-9)
        timestamp = test.get("timestamp")
        if best_cluster is None:
            clusters.append(
                {
                    "value": float(value),
                    "weight_sum": float(score),
                    "touches": 1,
                    "score_base": float(test["score"]),
                    "score": float(test["score"]),
                    "retest_score": float(test["retest_component"]),
                    "bounce_score": float(test["bounce_component"]),
                    "adx_score": float(test["adx_component"]),
                    "first_time": timestamp,
                    "last_time": timestamp,
                    "first_index": int(test["index"]),
                    "last_index": int(test["index"]),
                    "support_tests": 1 if test["type"] == "support" else 0,
                    "resistance_tests": 1 if test["type"] == "resistance" else 0,
                    "value_sq_sum": float(value * value * score),
                    "touch_min": float(value),
                    "touch_max": float(value),
                    "atr_metric_sum": float(test["atr_value"]) * score,
                    "atr_weight_sum": float(score),
                    "bounce_metric_sum": float(test["bounce_atr"]) * score,
                    "adx_metric_sum": float(test["pretest_adx"]) * score,
                    "metric_weight_sum": float(score),
                    "tests": [dict(test)],
                }
            )
            continue

        cluster = best_cluster
        new_weight = float(cluster["weight_sum"]) + score
        cluster["value"] = (float(cluster["value"]) * float(cluster["weight_sum"]) + value * score) / new_weight
        cluster["weight_sum"] = new_weight
        cluster["touches"] = int(cluster["touches"]) + 1
        cluster["score_base"] = float(cluster["score_base"]) + float(test["score"])
        cluster["score"] = float(cluster["score"]) + float(test["score"])
        cluster["retest_score"] = float(cluster["retest_score"]) + float(test["retest_component"])
        cluster["bounce_score"] = float(cluster["bounce_score"]) + float(test["bounce_component"])
        cluster["adx_score"] = float(cluster["adx_score"]) + float(test["adx_component"])
        cluster["first_index"] = min(int(cluster["first_index"]), int(test["index"]))
        cluster["last_index"] = max(int(cluster["last_index"]), int(test["index"]))
        cluster["support_tests"] = int(cluster["support_tests"]) + (1 if test["type"] == "support" else 0)
        cluster["resistance_tests"] = int(cluster["resistance_tests"]) + (1 if test["type"] == "resistance" else 0)
        cluster["value_sq_sum"] = float(cluster["value_sq_sum"]) + float(value * value * score)
        cluster["touch_min"] = min(float(cluster["touch_min"]), float(value))
        cluster["touch_max"] = max(float(cluster["touch_max"]), float(value))
        cluster["atr_metric_sum"] = float(cluster["atr_metric_sum"]) + float(test["atr_value"]) * score
        cluster["atr_weight_sum"] = float(cluster["atr_weight_sum"]) + float(score)
        cluster["bounce_metric_sum"] = float(cluster["bounce_metric_sum"]) + float(test["bounce_atr"]) * score
        cluster["adx_metric_sum"] = float(cluster["adx_metric_sum"]) + float(test["pretest_adx"]) * score
        cluster["metric_weight_sum"] = float(cluster["metric_weight_sum"]) + score
        cluster.setdefault("tests", []).append(dict(test))

        if timestamp is not None:
            if cluster["first_time"] is None or float(timestamp) < float(cluster["first_time"]):
                cluster["first_time"] = timestamp
            if cluster["last_time"] is None or float(timestamp) > float(cluster["last_time"]):
                cluster["last_time"] = timestamp

    return clusters


def _apply_episode_metrics(cluster: Dict[str, Any], *, episode_gap_bars: int) -> None:
    tests = sorted(cluster.get("tests", []), key=lambda item: (int(item.get("index", -1)), str(item.get("type", ""))))
    if not tests:
        cluster["episodes"] = 0
        cluster["support_episodes"] = 0
        cluster["resistance_episodes"] = 0
        cluster["episode_details"] = []
        return

    gap_value = max(1, int(episode_gap_bars))
    episodes: List[Dict[str, Any]] = []
    current_episode: Optional[Dict[str, Any]] = None

    for test in tests:
        test_index = int(test.get("index", -1))
        test_type = str(test.get("type", ""))
        if (
            current_episode is None
            or test_type != str(current_episode.get("type", ""))
            or test_index - int(current_episode.get("last_index", test_index)) > gap_value
        ):
            current_episode = {
                "type": test_type,
                "first_index": test_index,
                "last_index": test_index,
                "first_time": test.get("timestamp"),
                "last_time": test.get("timestamp"),
                "tests": [dict(test)],
            }
            episodes.append(current_episode)
            continue

        current_episode["last_index"] = test_index
        current_episode["last_time"] = test.get("timestamp")
        current_episode.setdefault("tests", []).append(dict(test))

    adjusted_retest = 0.0
    adjusted_bounce = 0.0
    adjusted_adx = 0.0
    adjusted_base = 0.0
    episode_details: List[Dict[str, Any]] = []
    support_episodes = 0
    resistance_episodes = 0

    for episode in episodes:
        episode_tests = sorted(
            episode.get("tests", []),
            key=lambda item: (float(item.get("score", 0.0)), int(item.get("index", -1))),
            reverse=True,
        )
        episode_score = 0.0
        episode_retest = 0.0
        episode_bounce = 0.0
        episode_adx = 0.0
        for rank, test in enumerate(episode_tests):
            weight = 1.0 if rank == 0 else _DEFAULT_EPISODE_TOUCH_DECAY
            episode_score += float(test.get("score", 0.0)) * weight
            episode_retest += float(test.get("retest_component", 0.0)) * weight
            episode_bounce += float(test.get("bounce_component", 0.0)) * weight
            episode_adx += float(test.get("adx_component", 0.0)) * weight

        adjusted_base += episode_score
        adjusted_retest += episode_retest
        adjusted_bounce += episode_bounce
        adjusted_adx += episode_adx
        episode_type = str(episode.get("type", ""))
        if episode_type == "support":
            support_episodes += 1
        elif episode_type == "resistance":
            resistance_episodes += 1
        episode_details.append(
            {
                "type": episode_type,
                "touches": len(episode_tests),
                "first_touch": _format_time(episode.get("first_time")),
                "last_touch": _format_time(episode.get("last_time")),
            }
        )

    cluster["episodes"] = len(episodes)
    cluster["support_episodes"] = int(support_episodes)
    cluster["resistance_episodes"] = int(resistance_episodes)
    cluster["episode_details"] = episode_details
    cluster["score_base"] = float(adjusted_base)
    cluster["retest_score"] = float(adjusted_retest)
    cluster["bounce_score"] = float(adjusted_bounce)
    cluster["adx_score"] = float(adjusted_adx)
    cluster["score"] = float(adjusted_base)


def _dominant_source(cluster: Dict[str, Any]) -> str:
    support_episodes = int(cluster.get("support_episodes", 0))
    resistance_episodes = int(cluster.get("resistance_episodes", 0))
    if support_episodes > resistance_episodes:
        return "support"
    if resistance_episodes > support_episodes:
        return "resistance"

    support_tests = int(cluster.get("support_tests", 0))
    resistance_tests = int(cluster.get("resistance_tests", 0))
    if support_tests > resistance_tests:
        return "support"
    if resistance_tests > support_tests:
        return "resistance"
    return "mixed"


def _cluster_atr_avg(cluster: Dict[str, Any]) -> Optional[float]:
    weight = float(cluster.get("atr_weight_sum", 0.0))
    if weight > 0.0:
        value = float(cluster.get("atr_metric_sum", 0.0)) / weight
        if math.isfinite(value) and value > 0.0:
            return value
    zone_low = cluster.get("zone_low")
    zone_high = cluster.get("zone_high")
    zone_width_atr = cluster.get("zone_width_atr")
    try:
        if zone_low is not None and zone_high is not None and zone_width_atr not in (None, 0):
            zone_width = float(zone_high) - float(zone_low)
            out = zone_width / float(zone_width_atr)
            if math.isfinite(out) and out > 0.0:
                return out
    except Exception:
        pass
    return None


def _resolve_zone(
    cluster: Dict[str, Any],
    *,
    tolerance_pct: float,
) -> Dict[str, Optional[float]]:
    try:
        zone_low = cluster.get("zone_low")
        zone_high = cluster.get("zone_high")
        if zone_low is not None and zone_high is not None:
            low = float(zone_low)
            high = float(zone_high)
            if math.isfinite(low) and math.isfinite(high) and high >= low:
                width = high - low
                zone_width_atr = cluster.get("zone_width_atr")
                return {
                    "zone_low": low,
                    "zone_high": high,
                    "zone_width": width,
                    "zone_width_atr": None if zone_width_atr is None else float(zone_width_atr),
                    "atr_avg": _cluster_atr_avg(cluster),
                }
    except Exception:
        pass

    value = float(cluster["value"])
    weight_sum = max(float(cluster.get("weight_sum", 0.0)), 1e-9)
    value_sq_sum = float(cluster.get("value_sq_sum", value * value * weight_sum))
    variance = max(0.0, (value_sq_sum / weight_sum) - (value * value))
    weighted_std = math.sqrt(variance)
    atr_avg = _cluster_atr_avg(cluster)
    tol_pad = max(abs(value), 1e-9) * max(float(tolerance_pct), 0.0) * 0.5
    atr_pad = (atr_avg or max(abs(value), 1e-9) * max(float(tolerance_pct), 0.0)) * _DEFAULT_ZONE_ATR_MULT
    dispersion_pad = weighted_std * _DEFAULT_ZONE_STD_MULT
    half_width = max(tol_pad, atr_pad, dispersion_pad, 1e-9)
    touch_min = float(cluster.get("touch_min", value))
    touch_max = float(cluster.get("touch_max", value))
    zone_low = min(touch_min, value - half_width)
    zone_high = max(touch_max, value + half_width)
    zone_width = max(0.0, zone_high - zone_low)
    zone_width_atr = None
    if atr_avg is not None and atr_avg > 0.0:
        zone_width_atr = zone_width / atr_avg
    return {
        "zone_low": zone_low,
        "zone_high": zone_high,
        "zone_width": zone_width,
        "zone_width_atr": zone_width_atr,
        "atr_avg": atr_avg,
    }


def _analyze_cluster_state(
    cluster: Dict[str, Any],
    *,
    closes: np.ndarray,
    epochs: List[Optional[float]],
    tolerance_pct: float,
) -> None:
    dominant_source = _dominant_source(cluster)
    zone = _resolve_zone(cluster, tolerance_pct=tolerance_pct)
    cluster["zone_low"] = zone["zone_low"]
    cluster["zone_high"] = zone["zone_high"]
    cluster["zone_width"] = zone["zone_width"]
    cluster["zone_width_atr"] = zone["zone_width_atr"]

    cluster["decisive_break_count"] = 0
    cluster["avg_breach_atr"] = None
    cluster["first_break_time"] = None
    cluster["first_break_index"] = None
    cluster["last_break_time"] = None
    cluster["last_break_index"] = None
    cluster["role_reversal_count"] = 0
    cluster["breakout_penalty"] = 0.0
    cluster["role_reversal_bonus"] = 0.0
    cluster["status"] = "active"

    if dominant_source not in {"support", "resistance"}:
        cluster["score"] = max(0.0, float(cluster.get("score_base", cluster.get("score", 0.0))))
        return

    zone_low = zone["zone_low"]
    zone_high = zone["zone_high"]
    atr_avg = zone["atr_avg"]
    if zone_low is None or zone_high is None:
        cluster["score"] = max(0.0, float(cluster.get("score_base", cluster.get("score", 0.0))))
        return

    zone_mid = 0.5 * (float(zone_low) + float(zone_high))
    buffer = max(
        (atr_avg or 0.0) * _DEFAULT_BREAKOUT_BUFFER_ATR_MULT,
        max(abs(zone_mid), 1e-9) * max(float(tolerance_pct), 0.0) * 0.25,
        1e-9,
    )
    analysis_start = min(int(test["index"]) for test in cluster.get("tests", []) or [{"index": 0}])
    break_indices: List[int] = []
    break_times: List[Optional[float]] = []
    breach_episode_max: List[float] = []
    in_break = False
    episode_max = 0.0

    for index in range(max(0, analysis_start), len(closes)):
        close = float(closes[index])
        if not math.isfinite(close):
            is_broken = False
            breach = 0.0
        elif dominant_source == "support":
            breach = max(0.0, float(zone_low) - close)
            is_broken = close < (float(zone_low) - buffer)
        else:
            breach = max(0.0, close - float(zone_high))
            is_broken = close > (float(zone_high) + buffer)

        breach_atr = 0.0
        if atr_avg is not None and atr_avg > 0.0:
            breach_atr = breach / atr_avg

        if is_broken:
            if not in_break:
                in_break = True
                break_indices.append(index)
                break_times.append(epochs[index] if index < len(epochs) else None)
                episode_max = breach_atr
            else:
                episode_max = max(episode_max, breach_atr)
        elif in_break:
            breach_episode_max.append(float(episode_max))
            in_break = False
            episode_max = 0.0

    if in_break:
        breach_episode_max.append(float(episode_max))

    breakout_count = len(break_indices)
    avg_breach_atr = float(np.mean(breach_episode_max)) if breach_episode_max else None
    first_break_index = break_indices[0] if break_indices else None
    first_break_time = break_times[0] if break_times else None
    last_break_index = break_indices[-1] if break_indices else None
    last_break_time = break_times[-1] if break_times else None

    role_reversal_tests = 0
    if breakout_count and first_break_index is not None:
        expected_new_role = "resistance" if dominant_source == "support" else "support"
        role_reversal_tests = sum(
            1
            for test in cluster.get("tests", [])
            if str(test.get("type")) == expected_new_role and int(test.get("index", -1)) > int(first_break_index)
        )

    breakout_penalty = 0.0
    if breakout_count:
        breakout_penalty = breakout_count * (
            _DEFAULT_BREAKOUT_PENALTY_BASE
            + _DEFAULT_BREAKOUT_PENALTY_ATR_MULT * min(float(avg_breach_atr or 0.0), 4.0)
        )
    role_reversal_bonus = 0.0
    if role_reversal_tests:
        role_reversal_bonus = min(1.5, _DEFAULT_ROLE_REVERSAL_BONUS * float(role_reversal_tests))

    cluster["decisive_break_count"] = int(breakout_count)
    cluster["avg_breach_atr"] = None if avg_breach_atr is None else float(avg_breach_atr)
    cluster["first_break_time"] = first_break_time
    cluster["first_break_index"] = first_break_index
    cluster["last_break_time"] = last_break_time
    cluster["last_break_index"] = last_break_index
    cluster["role_reversal_count"] = int(role_reversal_tests)
    cluster["breakout_penalty"] = float(breakout_penalty)
    cluster["role_reversal_bonus"] = float(role_reversal_bonus)

    if role_reversal_tests:
        expected_new_role = "resistance" if dominant_source == "support" else "support"
        cluster["status"] = f"role_reversed_{expected_new_role}"
    elif breakout_count:
        cluster["status"] = f"broken_{dominant_source}"
    else:
        cluster["status"] = "active"

    base_score = float(cluster.get("score_base", cluster.get("score", 0.0)))
    cluster["score"] = max(0.0, base_score - breakout_penalty + role_reversal_bonus)


def _format_level(cluster: Dict[str, Any], *, current_price: Optional[float], tolerance_pct: float) -> Dict[str, Any]:
    value = float(cluster["value"])
    zone = _resolve_zone(cluster, tolerance_pct=tolerance_pct)
    dominant_source = _dominant_source(cluster)
    level_type = dominant_source if dominant_source in {"support", "resistance"} else "support"
    distance = None
    distance_pct = None
    if current_price is not None and math.isfinite(current_price):
        level_type = "support" if value <= current_price else "resistance"
        distance = value - current_price
        distance_pct = abs(distance) / max(abs(current_price), 1e-9)

    metric_weight_sum = float(cluster.get("metric_weight_sum", 0.0))
    avg_bounce_atr = None
    avg_pretest_adx = None
    if metric_weight_sum > 0.0:
        avg_bounce_atr = float(cluster["bounce_metric_sum"]) / metric_weight_sum
        avg_pretest_adx = float(cluster["adx_metric_sum"]) / metric_weight_sum

    base_score = float(cluster.get("score_base", cluster.get("score", 0.0)))
    breakout_penalty = float(cluster.get("breakout_penalty", 0.0))
    role_reversal_bonus = float(cluster.get("role_reversal_bonus", 0.0))
    mtf_confirmation_bonus = float(cluster.get("mtf_confirmation_bonus", 0.0))
    total_score = max(0.0, base_score - breakout_penalty + role_reversal_bonus + mtf_confirmation_bonus)
    last_break_time = cluster.get("last_break_time")

    return {
        "type": level_type,
        "value": float(round(value, 6)),
        "touches": int(cluster["touches"]),
        "episodes": int(cluster.get("episodes", cluster.get("touches", 0))),
        "score": float(round(total_score, 4)),
        "distance": None if distance is None else float(round(distance, 6)),
        "distance_pct": None if distance_pct is None else float(round(distance_pct, 6)),
        "zone_low": None if zone["zone_low"] is None else float(round(float(zone["zone_low"]), 6)),
        "zone_high": None if zone["zone_high"] is None else float(round(float(zone["zone_high"]), 6)),
        "zone_width": None if zone["zone_width"] is None else float(round(float(zone["zone_width"]), 6)),
        "zone_width_atr": None if zone["zone_width_atr"] is None else float(round(float(zone["zone_width_atr"]), 4)),
        "first_touch": _format_time(cluster.get("first_time")),
        "last_touch": _format_time(cluster.get("last_time")),
        "dominant_source": dominant_source,
        "status": str(cluster.get("status", "active")),
        "source_tests": {
            "support": int(cluster.get("support_tests", 0)),
            "resistance": int(cluster.get("resistance_tests", 0)),
        },
        "source_episodes": {
            "support": int(cluster.get("support_episodes", cluster.get("support_tests", 0))),
            "resistance": int(cluster.get("resistance_episodes", cluster.get("resistance_tests", 0))),
        },
        "avg_bounce_atr": None if avg_bounce_atr is None else float(round(avg_bounce_atr, 4)),
        "avg_pretest_adx": None if avg_pretest_adx is None else float(round(avg_pretest_adx, 4)),
        "breakout_analysis": {
            "decisive_break_count": int(cluster.get("decisive_break_count", 0)),
            "avg_breach_atr": None
            if cluster.get("avg_breach_atr") is None
            else float(round(float(cluster["avg_breach_atr"]), 4)),
            "last_break_time": _format_time(last_break_time) if last_break_time is not None else None,
            "role_reversal_count": int(cluster.get("role_reversal_count", 0)),
        },
        "episode_details": list(cluster.get("episode_details", [])),
        "score_breakdown": {
            "base": float(round(base_score, 4)),
            "retests": float(round(float(cluster["retest_score"]), 4)),
            "bounce": float(round(float(cluster["bounce_score"]), 4)),
            "adx": float(round(float(cluster["adx_score"]), 4)),
            "breakout_penalty": float(round(breakout_penalty, 4)),
            "role_reversal_bonus": float(round(role_reversal_bonus, 4)),
            "mtf_confirmation_bonus": float(round(mtf_confirmation_bonus, 4)),
            "total": float(round(total_score, 4)),
        },
    }


def _timeframe_weight(timeframe: Any) -> float:
    return float(_TIMEFRAME_WEIGHTS.get(str(timeframe or "").upper(), 1.0))


def _timeframe_seconds(timeframe: Any) -> Optional[int]:
    return _TIMEFRAME_SECONDS.get(str(timeframe or "").upper())


def _timeframe_sort_key(timeframe: Any) -> tuple[int, str]:
    key = str(timeframe or "").upper()
    try:
        return (_AUTO_TIMEFRAMES.index(key), key)
    except ValueError:
        return (len(_AUTO_TIMEFRAMES), key)


def _pick_first_current_price(results: List[Dict[str, Any]]) -> Optional[float]:
    for result in results:
        try:
            value = result.get("current_price")
            if value is None:
                continue
            out = float(value)
            if math.isfinite(out):
                return out
        except Exception:
            continue
    return None


def _build_merge_signature(level: Dict[str, Any], timeframe: str) -> Optional[Dict[str, Any]]:
    first_touch = _parse_output_time(level.get("first_touch"))
    last_touch = _parse_output_time(level.get("last_touch"))
    if first_touch is None and last_touch is None:
        return None
    if first_touch is None:
        first_touch = last_touch
    if last_touch is None:
        last_touch = first_touch
    if first_touch is None or last_touch is None:
        return None
    start = min(float(first_touch), float(last_touch))
    end = max(float(first_touch), float(last_touch))
    timeframe_seconds = float(_timeframe_seconds(timeframe) or 3600)
    return {
        "timeframe": str(timeframe).upper(),
        "dominant_source": str(level.get("dominant_source") or level.get("type") or "mixed"),
        "window_start": start,
        "window_end": end,
        "pad_seconds": timeframe_seconds,
    }


def _signatures_match_same_event(existing: Dict[str, Any], incoming: Dict[str, Any]) -> bool:
    existing_source = str(existing.get("dominant_source") or "mixed")
    incoming_source = str(incoming.get("dominant_source") or "mixed")
    if existing_source != incoming_source and "mixed" not in {existing_source, incoming_source}:
        return False

    try:
        existing_start = float(existing["window_start"])
        existing_end = float(existing["window_end"])
        incoming_start = float(incoming["window_start"])
        incoming_end = float(incoming["window_end"])
    except Exception:
        return False

    if not all(math.isfinite(value) for value in (existing_start, existing_end, incoming_start, incoming_end)):
        return False

    gap = 0.0
    if existing_end < incoming_start:
        gap = incoming_start - existing_end
    elif incoming_end < existing_start:
        gap = existing_start - incoming_end
    pad = max(
        float(existing.get("pad_seconds", 0.0)),
        float(incoming.get("pad_seconds", 0.0)),
        60.0,
    )
    return gap <= (pad * 2.0)


def _cluster_has_same_event_signature(
    cluster: Dict[str, Any],
    *,
    signature: Optional[Dict[str, Any]],
    timeframe: str,
) -> bool:
    if signature is None:
        return False
    for existing in cluster.get("merge_signatures", []):
        if str(existing.get("timeframe") or "").upper() == str(timeframe).upper():
            continue
        if _signatures_match_same_event(existing, signature):
            return True
    return False


def merge_support_resistance_results(
    results: List[Dict[str, Any]],
    *,
    symbol: Optional[str] = None,
    timeframe: str = "auto",
    limit: Optional[int] = None,
    tolerance_pct: float = 0.0015,
    min_touches: int = 2,
    max_levels: int = 4,
    reaction_bars: int = _DEFAULT_REACTION_BARS,
    adx_period: int = _DEFAULT_ADX_PERIOD,
    decay_half_life_bars: Optional[int] = None,
) -> Dict[str, Any]:
    if not results:
        raise ValueError("No history available")

    adaptive_tolerance_pairs: List[tuple[float, float]] = []
    adaptive_reaction_pairs: List[tuple[float, float]] = []
    volatility_ratio_pairs: List[tuple[float, float]] = []
    current_atr_pairs: List[tuple[float, float]] = []
    baseline_atr_pairs: List[tuple[float, float]] = []
    for payload in results:
        tf = str(payload.get("timeframe") or "").upper()
        tf_weight = _timeframe_weight(tf)
        for key, bucket in (
            ("effective_tolerance_pct", adaptive_tolerance_pairs),
            ("effective_reaction_bars", adaptive_reaction_pairs),
            ("volatility_ratio", volatility_ratio_pairs),
            ("current_atr_pct", current_atr_pairs),
            ("baseline_atr_pct", baseline_atr_pairs),
        ):
            try:
                value = payload.get(key)
                if value is None:
                    continue
                numeric = float(value)
                if math.isfinite(numeric):
                    bucket.append((tf_weight, numeric))
            except Exception:
                continue

    merge_tolerance_value = _weighted_average(adaptive_tolerance_pairs)
    if merge_tolerance_value is None:
        merge_tolerance_value = float(tolerance_pct)
    merge_reaction_value = _weighted_average(adaptive_reaction_pairs)
    if merge_reaction_value is None:
        merge_reaction_value = float(reaction_bars)
    merge_volatility_ratio = _weighted_average(volatility_ratio_pairs)
    if merge_volatility_ratio is None:
        merge_volatility_ratio = 1.0
    merge_current_atr_pct = _weighted_average(current_atr_pairs)
    merge_baseline_atr_pct = _weighted_average(baseline_atr_pairs)

    clusters: List[Dict[str, Any]] = []
    for payload in results:
        tf = str(payload.get("timeframe") or "").upper()
        tf_weight = _timeframe_weight(tf)
        for level in payload.get("levels") or []:
            try:
                value = float(level["value"])
                raw_score = float(level.get("score", 0.0))
            except Exception:
                continue
            weighted_score = raw_score * tf_weight
            if not math.isfinite(value) or not math.isfinite(weighted_score) or weighted_score <= 0.0:
                continue

            best_cluster: Optional[Dict[str, Any]] = None
            best_delta: Optional[float] = None
            for cluster in clusters:
                ref = float(cluster["value"])
                threshold = max(abs(ref), abs(value), 1e-9) * float(merge_tolerance_value)
                delta = abs(ref - value)
                if delta <= threshold and (best_delta is None or delta < best_delta):
                    best_cluster = cluster
                    best_delta = delta

            breakdown = level.get("score_breakdown") if isinstance(level.get("score_breakdown"), dict) else {}
            source_tests = level.get("source_tests") if isinstance(level.get("source_tests"), dict) else {}
            source_episodes = level.get("source_episodes") if isinstance(level.get("source_episodes"), dict) else {}
            breakout_analysis = level.get("breakout_analysis") if isinstance(level.get("breakout_analysis"), dict) else {}
            first_touch = str(level.get("first_touch") or "").strip() or None
            last_touch = str(level.get("last_touch") or "").strip() or None
            touches = max(1, int(level.get("touches", 1)))
            episodes = max(1, int(level.get("episodes", touches)))
            avg_bounce = level.get("avg_bounce_atr")
            avg_adx = level.get("avg_pretest_adx")
            base_score = float(
                breakdown.get(
                    "base",
                    float(breakdown.get("retests", 0.0))
                    + float(breakdown.get("bounce", 0.0))
                    + float(breakdown.get("adx", 0.0)),
                )
            )
            breakout_penalty = float(breakdown.get("breakout_penalty", 0.0))
            role_reversal_bonus = float(breakdown.get("role_reversal_bonus", 0.0))
            decisive_break_count = int(breakout_analysis.get("decisive_break_count", 0))
            role_reversal_count = int(breakout_analysis.get("role_reversal_count", 0))
            avg_breach_atr = breakout_analysis.get("avg_breach_atr")
            zone_low = level.get("zone_low")
            zone_high = level.get("zone_high")
            zone_width_atr = level.get("zone_width_atr")
            status = str(level.get("status") or "").strip() or None
            merge_signature = _build_merge_signature(level, tf)
            mtf_confirmation_bonus = float(breakdown.get("mtf_confirmation_bonus", 0.0))

            if best_cluster is None:
                cluster = {
                    "value": float(value),
                    "weight_sum": float(weighted_score),
                    "touches": int(touches),
                    "episodes": int(episodes),
                    "score_base": float(base_score) * tf_weight,
                    "score": float(weighted_score),
                    "retest_score": float(breakdown.get("retests", raw_score)) * tf_weight,
                    "bounce_score": float(breakdown.get("bounce", 0.0)) * tf_weight,
                    "adx_score": float(breakdown.get("adx", 0.0)) * tf_weight,
                    "breakout_penalty": float(breakout_penalty) * tf_weight,
                    "role_reversal_bonus": float(role_reversal_bonus) * tf_weight,
                    "mtf_confirmation_bonus": float(mtf_confirmation_bonus),
                    "support_tests": int(source_tests.get("support", 0)),
                    "resistance_tests": int(source_tests.get("resistance", 0)),
                    "support_episodes": int(source_episodes.get("support", source_tests.get("support", 0))),
                    "resistance_episodes": int(source_episodes.get("resistance", source_tests.get("resistance", 0))),
                    "first_touch": first_touch,
                    "last_touch": last_touch,
                    "zone_low": None if zone_low is None else float(zone_low),
                    "zone_high": None if zone_high is None else float(zone_high),
                    "zone_width_atr_sum": 0.0,
                    "zone_weight_sum": 0.0,
                    "bounce_metric_sum": 0.0,
                    "adx_metric_sum": 0.0,
                    "metric_weight_sum": 0.0,
                    "decisive_break_count": int(decisive_break_count),
                    "role_reversal_count": int(role_reversal_count),
                    "avg_breach_atr_sum": 0.0,
                    "breakout_metric_weight_sum": 0.0,
                    "last_break_time": breakout_analysis.get("last_break_time"),
                    "status": status or "active",
                    "cross_timeframe_dedupe_count": 0,
                    "deduped_timeframes": set(),
                    "merge_signatures": [merge_signature] if merge_signature is not None else [],
                    "source_timeframes": {tf} if tf else set(),
                    "timeframe_scores": {tf: float(weighted_score)} if tf else {},
                    "timeframe_raw_scores": {tf: float(raw_score)} if tf else {},
                    "timeframe_touches": {tf: int(touches)} if tf else {},
                    "timeframe_episodes": {tf: int(episodes)} if tf else {},
                    "timeframe_weights": {tf: float(tf_weight)} if tf else {},
                    "timeframe_merge_modes": {tf: "full"} if tf else {},
                }
                if isinstance(avg_bounce, (int, float)) and math.isfinite(float(avg_bounce)):
                    cluster["bounce_metric_sum"] = float(avg_bounce) * float(weighted_score)
                    cluster["metric_weight_sum"] += float(weighted_score)
                if isinstance(avg_adx, (int, float)) and math.isfinite(float(avg_adx)):
                    cluster["adx_metric_sum"] = float(avg_adx) * float(weighted_score)
                    if cluster["metric_weight_sum"] <= 0.0:
                        cluster["metric_weight_sum"] += float(weighted_score)
                if zone_width_atr is not None:
                    try:
                        z_width_atr = float(zone_width_atr)
                        if math.isfinite(z_width_atr):
                            cluster["zone_width_atr_sum"] = z_width_atr * float(weighted_score)
                            cluster["zone_weight_sum"] = float(weighted_score)
                    except Exception:
                        pass
                if isinstance(avg_breach_atr, (int, float)) and math.isfinite(float(avg_breach_atr)):
                    cluster["avg_breach_atr_sum"] = float(avg_breach_atr) * max(float(decisive_break_count), 1.0)
                    cluster["breakout_metric_weight_sum"] = max(float(decisive_break_count), 1.0)
                clusters.append(cluster)
                continue

            cluster = best_cluster
            is_same_event = bool(
                tf
                and tf not in cluster.get("source_timeframes", set())
                and _cluster_has_same_event_signature(cluster, signature=merge_signature, timeframe=tf)
            )
            contribution_scale = _DEFAULT_MTF_DEDUPE_FACTOR if is_same_event else 1.0
            scaled_weighted_score = float(weighted_score) * contribution_scale
            confirmation_bonus = 0.0
            if is_same_event:
                prior_scores = [
                    float(cluster["timeframe_scores"].get(existing_tf, 0.0))
                    for existing_tf in cluster.get("source_timeframes", set())
                    if float(cluster["timeframe_scores"].get(existing_tf, 0.0)) > 0.0
                ]
                if prior_scores:
                    confirmation_bonus = min(float(weighted_score), max(prior_scores)) * _DEFAULT_MTF_CONFIRMATION_BONUS
            effective_weighted_score = scaled_weighted_score + confirmation_bonus
            new_weight = float(cluster["weight_sum"]) + effective_weighted_score
            cluster["value"] = (float(cluster["value"]) * float(cluster["weight_sum"]) + value * effective_weighted_score) / new_weight
            cluster["weight_sum"] = new_weight
            cluster["touches"] = int(cluster["touches"]) + (1 if is_same_event else int(touches))
            cluster["episodes"] = int(cluster.get("episodes", 0)) + (0 if is_same_event else int(episodes))
            cluster["score_base"] = float(cluster["score_base"]) + float(base_score) * tf_weight * contribution_scale
            cluster["score"] = float(cluster["score"]) + effective_weighted_score
            cluster["retest_score"] = float(cluster["retest_score"]) + float(breakdown.get("retests", raw_score)) * tf_weight * contribution_scale
            cluster["bounce_score"] = float(cluster["bounce_score"]) + float(breakdown.get("bounce", 0.0)) * tf_weight * contribution_scale
            cluster["adx_score"] = float(cluster["adx_score"]) + float(breakdown.get("adx", 0.0)) * tf_weight * contribution_scale
            cluster["breakout_penalty"] = float(cluster["breakout_penalty"]) + float(breakout_penalty) * tf_weight * contribution_scale
            cluster["role_reversal_bonus"] = float(cluster["role_reversal_bonus"]) + float(role_reversal_bonus) * tf_weight * contribution_scale
            cluster["mtf_confirmation_bonus"] = float(cluster.get("mtf_confirmation_bonus", 0.0)) + confirmation_bonus
            cluster["support_tests"] = int(cluster["support_tests"]) + (
                1 if is_same_event and int(source_tests.get("support", 0)) > 0 else int(source_tests.get("support", 0))
            )
            cluster["resistance_tests"] = int(cluster["resistance_tests"]) + (
                1 if is_same_event and int(source_tests.get("resistance", 0)) > 0 else int(source_tests.get("resistance", 0))
            )
            cluster["support_episodes"] = int(cluster.get("support_episodes", 0)) + (
                0 if is_same_event else int(source_episodes.get("support", source_tests.get("support", 0)))
            )
            cluster["resistance_episodes"] = int(cluster.get("resistance_episodes", 0)) + (
                0 if is_same_event else int(source_episodes.get("resistance", source_tests.get("resistance", 0)))
            )
            if first_touch is not None:
                current_first = cluster.get("first_touch")
                cluster["first_touch"] = first_touch if current_first is None else min(str(current_first), first_touch)
            if last_touch is not None:
                current_last = cluster.get("last_touch")
                cluster["last_touch"] = last_touch if current_last is None else max(str(current_last), last_touch)
            if zone_low is not None:
                low = float(zone_low)
                cluster["zone_low"] = low if cluster.get("zone_low") is None else min(float(cluster["zone_low"]), low)
            if zone_high is not None:
                high = float(zone_high)
                cluster["zone_high"] = high if cluster.get("zone_high") is None else max(float(cluster["zone_high"]), high)
            if tf:
                cluster["source_timeframes"].add(tf)
                cluster["timeframe_scores"][tf] = float(cluster["timeframe_scores"].get(tf, 0.0)) + effective_weighted_score
                cluster["timeframe_raw_scores"][tf] = float(cluster["timeframe_raw_scores"].get(tf, 0.0)) + float(raw_score)
                cluster["timeframe_touches"][tf] = int(cluster["timeframe_touches"].get(tf, 0)) + (1 if is_same_event else int(touches))
                cluster["timeframe_episodes"][tf] = int(cluster["timeframe_episodes"].get(tf, 0)) + int(episodes)
                cluster["timeframe_weights"][tf] = float(tf_weight)
                cluster["timeframe_merge_modes"][tf] = "deduped" if is_same_event else "full"
            if isinstance(avg_bounce, (int, float)) and math.isfinite(float(avg_bounce)):
                cluster["bounce_metric_sum"] = float(cluster["bounce_metric_sum"]) + float(avg_bounce) * scaled_weighted_score
                cluster["metric_weight_sum"] = float(cluster["metric_weight_sum"]) + scaled_weighted_score
            if isinstance(avg_adx, (int, float)) and math.isfinite(float(avg_adx)):
                cluster["adx_metric_sum"] = float(cluster["adx_metric_sum"]) + float(avg_adx) * scaled_weighted_score
                if not (isinstance(avg_bounce, (int, float)) and math.isfinite(float(avg_bounce))):
                    cluster["metric_weight_sum"] = float(cluster["metric_weight_sum"]) + scaled_weighted_score
            if zone_width_atr is not None:
                try:
                    z_width_atr = float(zone_width_atr)
                    if math.isfinite(z_width_atr):
                        cluster["zone_width_atr_sum"] = float(cluster["zone_width_atr_sum"]) + z_width_atr * scaled_weighted_score
                        cluster["zone_weight_sum"] = float(cluster["zone_weight_sum"]) + scaled_weighted_score
                except Exception:
                    pass
            if is_same_event:
                cluster["decisive_break_count"] = max(int(cluster["decisive_break_count"]), int(decisive_break_count))
                cluster["role_reversal_count"] = max(int(cluster["role_reversal_count"]), int(role_reversal_count))
                cluster["cross_timeframe_dedupe_count"] = int(cluster.get("cross_timeframe_dedupe_count", 0)) + 1
                if tf:
                    cluster["deduped_timeframes"].add(tf)
            else:
                cluster["decisive_break_count"] = int(cluster["decisive_break_count"]) + int(decisive_break_count)
                cluster["role_reversal_count"] = int(cluster["role_reversal_count"]) + int(role_reversal_count)
            if isinstance(avg_breach_atr, (int, float)) and math.isfinite(float(avg_breach_atr)):
                weight = max(float(decisive_break_count), 1.0) * contribution_scale
                cluster["avg_breach_atr_sum"] = float(cluster["avg_breach_atr_sum"]) + float(avg_breach_atr) * weight
                cluster["breakout_metric_weight_sum"] = float(cluster["breakout_metric_weight_sum"]) + weight
            last_break_time = breakout_analysis.get("last_break_time")
            if isinstance(last_break_time, str) and last_break_time.strip():
                current_break_time = cluster.get("last_break_time")
                cluster["last_break_time"] = last_break_time if current_break_time is None else max(str(current_break_time), last_break_time)
            if status and status.startswith("role_reversed_"):
                cluster["status"] = status
            elif status and cluster.get("status") != "role_reversed_support" and cluster.get("status") != "role_reversed_resistance":
                cluster["status"] = status
            if merge_signature is not None:
                cluster.setdefault("merge_signatures", []).append(merge_signature)

    usable_clusters = [cluster for cluster in clusters if int(cluster.get("episodes", cluster.get("touches", 0))) >= max(1, int(min_touches))]
    if not usable_clusters and clusters:
        usable_clusters = sorted(
            clusters,
            key=lambda cluster: (
                float(cluster.get("score", 0.0)),
                int(cluster.get("episodes", cluster.get("touches", 0))),
                int(cluster.get("touches", 0)),
            ),
            reverse=True,
        )[:1]

    current_price = _pick_first_current_price(results)
    formatted_levels: List[Dict[str, Any]] = []
    for cluster in usable_clusters:
        zone_weight_sum = float(cluster.get("zone_weight_sum", 0.0))
        if zone_weight_sum > 0.0:
            cluster["zone_width_atr"] = float(cluster["zone_width_atr_sum"]) / zone_weight_sum
        breakout_weight = float(cluster.get("breakout_metric_weight_sum", 0.0))
        if breakout_weight > 0.0:
            cluster["avg_breach_atr"] = float(cluster["avg_breach_atr_sum"]) / breakout_weight
        cluster["score"] = max(
            0.0,
            float(cluster.get("score_base", 0.0))
            - float(cluster.get("breakout_penalty", 0.0))
            + float(cluster.get("role_reversal_bonus", 0.0))
            + float(cluster.get("mtf_confirmation_bonus", 0.0)),
        )
        level = _format_level(cluster, current_price=current_price, tolerance_pct=float(merge_tolerance_value))
        level["touches"] = int(cluster.get("touches", level.get("touches", 0)))
        level["episodes"] = int(cluster.get("episodes", level.get("episodes", level.get("touches", 0))))
        level["first_touch"] = cluster.get("first_touch")
        level["last_touch"] = cluster.get("last_touch")
        timeframes = sorted(cluster.get("source_timeframes", set()), key=_timeframe_sort_key)
        level["source_timeframes"] = timeframes
        level["merge_details"] = {
            "cross_timeframe_dedupe_count": int(cluster.get("cross_timeframe_dedupe_count", 0)),
            "deduped_timeframes": sorted(cluster.get("deduped_timeframes", set()), key=_timeframe_sort_key),
        }
        level["timeframe_contributions"] = [
            {
                "timeframe": tf,
                "weight": float(cluster["timeframe_weights"].get(tf, _timeframe_weight(tf))),
                "raw_score": float(round(float(cluster["timeframe_raw_scores"].get(tf, 0.0)), 4)),
                "weighted_score": float(round(float(cluster["timeframe_scores"].get(tf, 0.0)), 4)),
                "touches": int(cluster["timeframe_touches"].get(tf, 0)),
                "episodes": int(cluster["timeframe_episodes"].get(tf, cluster["timeframe_touches"].get(tf, 0))),
                "merge_mode": str(cluster["timeframe_merge_modes"].get(tf, "full")),
            }
            for tf in timeframes
        ]
        formatted_levels.append(level)

    support_candidates = [dict(level) for level in formatted_levels if level.get("type") == "support"]
    resistance_candidates = [dict(level) for level in formatted_levels if level.get("type") == "resistance"]

    support_candidates.sort(key=lambda level: (-float(level.get("score", 0.0)), -float(level.get("value", 0.0))))
    resistance_candidates.sort(key=lambda level: (-float(level.get("score", 0.0)), float(level.get("value", 0.0))))
    for rank, level in enumerate(support_candidates, start=1):
        level["strength_rank"] = rank
    for rank, level in enumerate(resistance_candidates, start=1):
        level["strength_rank"] = rank

    max_levels_value = max(1, int(max_levels))
    supports = support_candidates[:max_levels_value]
    resistances = resistance_candidates[:max_levels_value]
    supports.sort(key=lambda level: (-float(level.get("value", 0.0)), -float(level.get("score", 0.0))))
    resistances.sort(key=lambda level: (float(level.get("value", 0.0)), -float(level.get("score", 0.0))))

    start_values = [
        str((payload.get("window") or {}).get("start") or "").strip()
        for payload in results
        if isinstance(payload.get("window"), dict) and str((payload.get("window") or {}).get("start") or "").strip()
    ]
    end_values = [
        str((payload.get("window") or {}).get("end") or "").strip()
        for payload in results
        if isinstance(payload.get("window"), dict) and str((payload.get("window") or {}).get("end") or "").strip()
    ]
    timeframes_analyzed = [
        str(payload.get("timeframe") or "").upper()
        for payload in sorted(results, key=lambda item: _timeframe_sort_key(item.get("timeframe")))
        if str(payload.get("timeframe") or "").strip()
    ]

    return {
        "success": True,
        "symbol": symbol if symbol is not None else results[0].get("symbol"),
        "timeframe": str(timeframe),
        "mode": "auto",
        "timeframes_analyzed": timeframes_analyzed,
        "timeframe_weights": {tf: _timeframe_weight(tf) for tf in timeframes_analyzed},
        "per_timeframe": [
            {
                "timeframe": str(payload.get("timeframe")),
                "supports": len(payload.get("supports") or []),
                "resistances": len(payload.get("resistances") or []),
                "current_price": payload.get("current_price"),
                "window": payload.get("window"),
                "effective_tolerance_pct": payload.get("effective_tolerance_pct"),
                "effective_reaction_bars": payload.get("effective_reaction_bars"),
                "volatility_ratio": payload.get("volatility_ratio"),
                "current_atr_pct": payload.get("current_atr_pct"),
                "baseline_atr_pct": payload.get("baseline_atr_pct"),
            }
            for payload in sorted(results, key=lambda item: _timeframe_sort_key(item.get("timeframe")))
        ],
        "limit": int(limit) if limit is not None else int(results[0].get("limit", 0) or 0),
        "method": _METHOD_NAME,
        "tolerance_pct": float(tolerance_pct),
        "effective_tolerance_pct": float(merge_tolerance_value),
        "min_touches": int(max(1, int(min_touches))),
        "qualification_basis": "episodes",
        "max_levels": int(max_levels_value),
        "reaction_bars": int(max(1, int(reaction_bars))),
        "effective_reaction_bars": int(max(1, int(round(merge_reaction_value)))),
        "adx_period": int(max(2, int(adx_period))),
        "adaptive_mode": "atr_regime",
        "volatility_ratio": float(merge_volatility_ratio),
        "current_atr_pct": None if merge_current_atr_pct is None else float(merge_current_atr_pct),
        "baseline_atr_pct": None if merge_baseline_atr_pct is None else float(merge_baseline_atr_pct),
        "decay_half_life_bars": None if decay_half_life_bars is None else int(decay_half_life_bars),
        "current_price": None if current_price is None else float(round(current_price, 6)),
        "window": {
            "start": min(start_values) if start_values else None,
            "end": max(end_values) if end_values else None,
        },
        "supports": supports,
        "resistances": resistances,
        "levels": supports + resistances,
    }


def compact_support_resistance_level(level: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(level, dict):
        return None
    out: Dict[str, Any] = {}
    for key in ("type", "value", "distance", "distance_pct", "touches", "episodes", "status", "score", "strength_rank"):
        value = level.get(key)
        if value is not None:
            out[str(key)] = value
    source_timeframes = level.get("source_timeframes")
    if isinstance(source_timeframes, list) and source_timeframes:
        out["source_timeframes"] = list(source_timeframes)
    dominant_source = level.get("dominant_source")
    if isinstance(dominant_source, str) and dominant_source.strip():
        out["dominant_source"] = dominant_source
    return out or None


def compact_support_resistance_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return payload

    out: Dict[str, Any] = {}
    for key in ("success", "symbol", "timeframe", "mode", "method", "current_price", "timeframes_analyzed"):
        value = payload.get(key)
        if value is not None:
            out[str(key)] = value

    window = payload.get("window")
    if isinstance(window, dict):
        window_out = {str(k): v for k, v in window.items() if v is not None}
        if window_out:
            out["window"] = window_out

    supports = payload.get("supports")
    resistances = payload.get("resistances")
    if isinstance(supports, list) or isinstance(resistances, list):
        counts: Dict[str, int] = {}
        if isinstance(supports, list):
            counts["support"] = len(supports)
        if isinstance(resistances, list):
            counts["resistance"] = len(resistances)
        if counts:
            counts["total"] = sum(counts.values())
            out["level_counts"] = counts

    nearest: Dict[str, Any] = {}
    if isinstance(supports, list) and supports:
        support_compact = compact_support_resistance_level(supports[0])
        if support_compact:
            nearest["support"] = support_compact
    if isinstance(resistances, list) and resistances:
        resistance_compact = compact_support_resistance_level(resistances[0])
        if resistance_compact:
            nearest["resistance"] = resistance_compact
    if nearest:
        out["nearest"] = nearest

    levels = payload.get("levels")
    if isinstance(levels, list):
        compact_levels = [
            compacted
            for compacted in (compact_support_resistance_level(level) for level in levels)
            if compacted
        ]
        if compact_levels:
            out["levels"] = compact_levels

    warnings = payload.get("warnings")
    if isinstance(warnings, list) and warnings:
        out["warnings"] = list(warnings)

    meta = payload.get("meta")
    if isinstance(meta, dict) and meta:
        out["meta"] = dict(meta)
    return out or dict(payload)


def compute_support_resistance_levels(
    frame: pd.DataFrame,
    *,
    symbol: Optional[str] = None,
    timeframe: Optional[str] = None,
    limit: Optional[int] = None,
    tolerance_pct: float = 0.0015,
    min_touches: int = 2,
    max_levels: int = 4,
    reaction_bars: int = _DEFAULT_REACTION_BARS,
    adx_period: int = _DEFAULT_ADX_PERIOD,
    decay_half_life_bars: Optional[int] = None,
) -> Dict[str, Any]:
    if frame is None or getattr(frame, "empty", True):
        raise ValueError("No history available")
    required_cols = ("high", "low", "close")
    missing = [column for column in required_cols if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")
    if len(frame) < 3:
        raise ValueError("Need at least 3 bars to compute support/resistance levels")

    tolerance_value = float(tolerance_pct)
    if tolerance_value < 0.0:
        raise ValueError("tolerance_pct must be non-negative")
    min_touches_value = max(1, int(min_touches))
    max_levels_value = max(1, int(max_levels))
    reaction_bars_value = max(1, int(reaction_bars))
    adx_period_value = max(2, int(adx_period))
    half_life_value = (
        max(5, int(decay_half_life_bars))
        if decay_half_life_bars is not None
        else max(20, int(round(len(frame) / 4.0)))
    )

    highs = _to_numeric_array(frame, "high")
    lows = _to_numeric_array(frame, "low")
    closes = _to_numeric_array(frame, "close")
    current_price = _last_finite(closes)
    atr, adx = _compute_atr_and_adx(highs, lows, closes, period=adx_period_value)
    adaptive_settings = _resolve_adaptive_settings(
        closes,
        atr,
        base_tolerance_pct=tolerance_value,
        base_reaction_bars=reaction_bars_value,
    )
    effective_tolerance_value = float(adaptive_settings["effective_tolerance_pct"])
    effective_reaction_bars = int(adaptive_settings["effective_reaction_bars"])

    epochs = [_to_epoch(value) for value in frame["time"].tolist()] if "time" in frame.columns else []
    window_start = next((value for value in epochs if value is not None), None)
    window_end = next((value for value in reversed(epochs) if value is not None), None)

    tests = _collect_tests(
        highs,
        lows,
        closes,
        epochs=epochs,
        reaction_bars=effective_reaction_bars,
        adx_period=adx_period_value,
        decay_half_life_bars=half_life_value,
        atr=atr,
        adx=adx,
    )
    clusters = _cluster_tests(tests, tolerance_pct=effective_tolerance_value)
    for cluster in clusters:
        _apply_episode_metrics(cluster, episode_gap_bars=max(1, effective_reaction_bars))
        _analyze_cluster_state(
            cluster,
            closes=closes,
            epochs=epochs,
            tolerance_pct=effective_tolerance_value,
        )
    usable_clusters = [cluster for cluster in clusters if int(cluster.get("episodes", cluster["touches"])) >= min_touches_value]
    if not usable_clusters and clusters:
        usable_clusters = sorted(
            clusters,
            key=lambda cluster: (
                float(cluster.get("score", 0.0)),
                int(cluster.get("episodes", cluster.get("touches", 0))),
                int(cluster.get("touches", 0)),
                int(cluster.get("last_index", -1)),
            ),
            reverse=True,
        )[:1]

    formatted_levels = [
        _format_level(cluster, current_price=current_price, tolerance_pct=effective_tolerance_value)
        for cluster in usable_clusters
    ]
    support_candidates = [dict(level) for level in formatted_levels if level.get("type") == "support"]
    resistance_candidates = [dict(level) for level in formatted_levels if level.get("type") == "resistance"]

    support_candidates.sort(key=lambda level: (-float(level.get("score", 0.0)), -float(level.get("value", 0.0))))
    for rank, level in enumerate(support_candidates, start=1):
        level["strength_rank"] = rank
    resistance_candidates.sort(key=lambda level: (-float(level.get("score", 0.0)), float(level.get("value", 0.0))))
    for rank, level in enumerate(resistance_candidates, start=1):
        level["strength_rank"] = rank

    supports = support_candidates[:max_levels_value]
    resistances = resistance_candidates[:max_levels_value]
    supports.sort(key=lambda level: (-float(level.get("value", 0.0)), -float(level.get("score", 0.0))))
    resistances.sort(key=lambda level: (float(level.get("value", 0.0)), -float(level.get("score", 0.0))))

    return {
        "success": True,
        "symbol": symbol,
        "timeframe": timeframe,
        "mode": "single",
        "timeframes_analyzed": [str(timeframe)] if timeframe is not None else [],
        "limit": int(limit) if limit is not None else len(frame),
        "method": _METHOD_NAME,
        "tolerance_pct": float(tolerance_value),
        "effective_tolerance_pct": float(effective_tolerance_value),
        "min_touches": int(min_touches_value),
        "qualification_basis": "episodes",
        "max_levels": int(max_levels_value),
        "reaction_bars": int(reaction_bars_value),
        "effective_reaction_bars": int(effective_reaction_bars),
        "adx_period": int(adx_period_value),
        "adaptive_mode": str(adaptive_settings["adaptive_mode"]),
        "volatility_ratio": float(adaptive_settings["volatility_ratio"]),
        "current_atr_pct": adaptive_settings["current_atr_pct"],
        "baseline_atr_pct": adaptive_settings["baseline_atr_pct"],
        "decay_half_life_bars": int(half_life_value),
        "current_price": None if current_price is None else float(round(current_price, 6)),
        "window": {
            "start": _format_time(window_start),
            "end": _format_time(window_end),
        },
        "supports": supports,
        "resistances": resistances,
        "levels": supports + resistances,
    }
