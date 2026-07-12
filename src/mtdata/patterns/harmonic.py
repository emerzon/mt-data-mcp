from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from ..utils.utils import to_float_np
from .common import PatternResultBase, fallback_local_extrema

_DEFAULT_PATTERN_TYPES = (
    "abcd",
    "gartley",
    "bat",
    "alternate_bat",
    "butterfly",
    "crab",
    "deep_crab",
    "shark",
    "cypher",
    "five_o",
)

_PATTERN_TYPE_ALIASES = {
    "abcd": "abcd",
    "ab=cd": "abcd",
    "gartley": "gartley",
    "bat": "bat",
    "alt_bat": "alternate_bat",
    "altbat": "alternate_bat",
    "alternate_bat": "alternate_bat",
    "alternatebat": "alternate_bat",
    "butterfly": "butterfly",
    "butter_fly": "butterfly",
    "crab": "crab",
    "deep_crab": "deep_crab",
    "deepcrab": "deep_crab",
    "shark": "shark",
    "cypher": "cypher",
    "cypper": "cypher",
    "five_o": "five_o",
    "five0": "five_o",
    "5o": "five_o",
    "5_0": "five_o",
    "5-0": "five_o",
}


@dataclass
class HarmonicDetectorConfig:
    max_bars: int = 1500
    min_input_bars: int = 80
    pattern_types: List[str] = field(default_factory=lambda: list(_DEFAULT_PATTERN_TYPES))
    ratio_tolerance: float = 0.06
    max_pivots: int = 24
    min_confidence: float = 0.45
    stop_buffer_pct: float = 0.2
    include_forming: bool = True
    recent_bars: int = 30
    max_pattern_age_bars: int = 300
    min_prominence_pct: float = 0.5
    min_distance: int = 5
    pivot_use_hl: bool = True
    pivot_use_atr_adaptive_prominence: bool = True
    pivot_use_atr_adaptive_distance: bool = True
    pivot_atr_period: int = 14
    pivot_atr_prominence_mult: float = 1.0
    pivot_atr_distance_mult: float = 0.2
    pivot_max_distance_scale: float = 3.0
    pivot_enable_fallback: bool = True
    pivot_fallback_order: int = 2


@dataclass
class HarmonicPatternResult(PatternResultBase):
    name: str
    status: str
    bias: str
    entry_price: float
    target_prices: List[float]
    invalidation_price: float
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _SwingPoint:
    index: int
    kind: str
    price: float


@dataclass(frozen=True)
class _PatternSpec:
    key: str
    display: str
    family: str
    ratios: Dict[str, Tuple[float, float]]


_XABCD_SPECS: Dict[str, _PatternSpec] = {
    "gartley": _PatternSpec(
        "gartley",
        "Gartley",
        "XABCD",
        {
            "xab": (0.618, 0.618),
            "abc": (0.382, 0.886),
            "bcd": (1.13, 1.618),
            "xad": (0.786, 0.786),
        },
    ),
    "bat": _PatternSpec(
        "bat",
        "Bat",
        "XABCD",
        {
            "xab": (0.382, 0.5),
            "abc": (0.382, 0.886),
            "bcd": (1.618, 2.618),
            "xad": (0.886, 0.886),
        },
    ),
    "alternate_bat": _PatternSpec(
        "alternate_bat",
        "Alternate Bat",
        "XABCD",
        {
            "xab": (0.382, 0.382),
            "abc": (0.382, 0.886),
            "bcd": (2.0, 3.618),
            "xad": (1.13, 1.13),
        },
    ),
    "butterfly": _PatternSpec(
        "butterfly",
        "Butterfly",
        "XABCD",
        {
            "xab": (0.786, 0.786),
            "abc": (0.382, 0.886),
            "bcd": (1.618, 2.24),
            "xad": (1.27, 1.618),
        },
    ),
    "crab": _PatternSpec(
        "crab",
        "Crab",
        "XABCD",
        {
            "xab": (0.382, 0.618),
            "abc": (0.382, 0.886),
            "bcd": (2.24, 3.618),
            "xad": (1.618, 1.618),
        },
    ),
    "deep_crab": _PatternSpec(
        "deep_crab",
        "Deep Crab",
        "XABCD",
        {
            "xab": (0.886, 0.886),
            "abc": (0.382, 0.886),
            "bcd": (2.0, 3.618),
            "xad": (1.618, 1.618),
        },
    ),
    "shark": _PatternSpec(
        "shark",
        "Shark",
        "XABCD",
        {
            "xab": (0.382, 0.618),
            "abc": (1.13, 1.618),
            "bcd": (1.618, 2.24),
            "xad": (0.886, 1.13),
        },
    ),
    "cypher": _PatternSpec(
        "cypher",
        "Cypher",
        "XABCD",
        {
            "xab": (0.382, 0.618),
            "abc": (1.13, 1.414),
            "xcd": (0.786, 0.786),
        },
    ),
    "five_o": _PatternSpec(
        "five_o",
        "5-0",
        "XABCD",
        {
            "xab": (1.13, 1.618),
            "abc": (1.618, 2.24),
            "bcd": (0.5, 0.5),
        },
    ),
}

_ABCD_SPEC = _PatternSpec(
    "abcd",
    "ABCD",
    "ABCD",
    {
        "abc": (0.382, 0.886),
        "bcd": (1.13, 2.618),
        "cd_ab": (0.9, 1.2),
    },
)


def _canonical_pattern_type(value: Any) -> Optional[str]:
    key = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    return _PATTERN_TYPE_ALIASES.get(key)


def _normalized_pattern_types(cfg: HarmonicDetectorConfig) -> List[str]:
    raw = getattr(cfg, "pattern_types", None)
    values: List[Any]
    if isinstance(raw, str):
        values = [part.strip() for part in raw.replace(";", ",").split(",") if part.strip()]
    elif isinstance(raw, (list, tuple, set)):
        values = list(raw)
    else:
        values = list(_DEFAULT_PATTERN_TYPES)

    out: List[str] = []
    for item in values:
        canonical = _canonical_pattern_type(item)
        if canonical is not None and canonical not in out:
            out.append(canonical)
    return out or list(_DEFAULT_PATTERN_TYPES)


def validate_harmonic_detector_config(cfg: HarmonicDetectorConfig) -> list[str]:
    warnings: list[str] = []
    for attr in (
        "max_bars",
        "min_input_bars",
        "max_pivots",
        "min_distance",
        "pivot_atr_period",
        "recent_bars",
    ):
        value = getattr(cfg, attr, None)
        if isinstance(value, (int, float)) and value <= 0:
            warnings.append(f"{attr} must be positive, got {value}")
    if cfg.min_input_bars > cfg.max_bars:
        warnings.append(
            f"min_input_bars ({cfg.min_input_bars}) exceeds max_bars ({cfg.max_bars})"
        )
    for attr in ("ratio_tolerance", "min_confidence", "stop_buffer_pct", "min_prominence_pct"):
        value = getattr(cfg, attr, None)
        if isinstance(value, (int, float)) and value < 0:
            warnings.append(f"{attr} must be non-negative, got {value}")

    raw_types = getattr(cfg, "pattern_types", None)
    if isinstance(raw_types, str):
        values = [part.strip() for part in raw_types.replace(";", ",").split(",") if part.strip()]
    elif isinstance(raw_types, (list, tuple, set)):
        values = list(raw_types)
    else:
        values = []
    invalid = [str(value) for value in values if _canonical_pattern_type(value) is None]
    if invalid:
        warnings.append(
            "pattern_types contains unsupported value(s): "
            + ", ".join(sorted(dict.fromkeys(invalid)))
        )
    return warnings


def _compute_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    h = np.asarray(high, dtype=float)
    l = np.asarray(low, dtype=float)
    c = np.asarray(close, dtype=float)
    n = min(h.size, l.size, c.size)
    if n <= 0:
        return np.asarray([], dtype=float)
    h = h[:n]
    l = l[:n]
    c = c[:n]
    prev_c = np.concatenate(([c[0]], c[:-1]))
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    win = max(2, int(period))
    try:
        return pd.Series(tr).rolling(win, min_periods=max(2, win // 2)).mean().to_numpy(dtype=float)
    except Exception:
        return tr.astype(float)


def _pivot_thresholds(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    cfg: HarmonicDetectorConfig,
) -> Tuple[float, int]:
    x = np.asarray(close, dtype=float)
    finite = x[np.isfinite(x)]
    base = float(np.median(finite)) if finite.size else 0.0
    if not np.isfinite(base) or abs(base) <= 1e-12:
        base = float(np.mean(finite)) if finite.size else 1.0
    prom_abs = abs(base) * (float(cfg.min_prominence_pct) / 100.0)
    min_dist = max(2, int(cfg.min_distance))

    if bool(cfg.pivot_use_atr_adaptive_prominence) or bool(cfg.pivot_use_atr_adaptive_distance):
        atr = _compute_atr(high, low, x, int(cfg.pivot_atr_period))
        finite_atr = atr[np.isfinite(atr) & (atr > 0.0)]
        if finite_atr.size > 0:
            atr_med = float(np.median(finite_atr))
            if bool(cfg.pivot_use_atr_adaptive_prominence):
                prom_abs = max(prom_abs, float(cfg.pivot_atr_prominence_mult) * atr_med)
            if bool(cfg.pivot_use_atr_adaptive_distance) and abs(base) > 1e-12:
                atr_pct = abs(atr_med / base) * 100.0
                scale = 1.0 + max(0.0, float(cfg.pivot_atr_distance_mult)) * atr_pct
                scale = min(float(max(1.0, cfg.pivot_max_distance_scale)), max(1.0, scale))
                min_dist = max(2, int(round(float(cfg.min_distance) * scale)))
    return float(max(1e-12, prom_abs)), int(min_dist)


def _detect_pivots(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    cfg: HarmonicDetectorConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(close, dtype=float)
    if x.size < max(5, int(cfg.min_distance) * 3):
        return np.asarray([], dtype=int), np.asarray([], dtype=int)
    h = np.asarray(high, dtype=float)
    l = np.asarray(low, dtype=float)
    if h.size != x.size or not np.isfinite(h).all():
        h = x
    if l.size != x.size or not np.isfinite(l).all():
        l = x

    prom_abs, min_dist = _pivot_thresholds(x, h, l, cfg)
    src_hi = h if bool(cfg.pivot_use_hl) else x
    src_lo = l if bool(cfg.pivot_use_hl) else x
    try:
        peaks, _ = find_peaks(src_hi, prominence=prom_abs, distance=min_dist)
        troughs, _ = find_peaks(-src_lo, prominence=prom_abs, distance=min_dist)
    except ValueError:
        return np.asarray([], dtype=int), np.asarray([], dtype=int)
    if bool(cfg.pivot_enable_fallback):
        if peaks.size < 2:
            peaks = fallback_local_extrema(
                src_hi,
                min_dist,
                max(1, int(cfg.pivot_fallback_order)),
                prefer_high=True,
            )
        if troughs.size < 2:
            troughs = fallback_local_extrema(
                src_lo,
                min_dist,
                max(1, int(cfg.pivot_fallback_order)),
                prefer_high=False,
            )
    return peaks.astype(int), troughs.astype(int)


def _prepare_inputs(
    df: pd.DataFrame,
    cfg: HarmonicDetectorConfig,
) -> Optional[Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]]:
    if not isinstance(df, pd.DataFrame) or "close" not in df.columns:
        return None
    if len(df) > int(cfg.max_bars):
        df = df.iloc[-int(cfg.max_bars) :].copy()

    close = to_float_np(df["close"])
    used_close_for_high = "high" not in df.columns
    used_close_for_low = "low" not in df.columns
    high = to_float_np(df["high"]) if not used_close_for_high else close
    low = to_float_np(df["low"]) if not used_close_for_low else close
    if high.size != close.size:
        used_close_for_high = True
        high = close
    if low.size != close.size:
        used_close_for_low = True
        low = close
    if used_close_for_high or used_close_for_low:
        import logging

        logging.getLogger(__name__).warning(
            "Harmonic pattern detection falling back to close for missing/mismatched "
            "high/low columns (high_fallback=%s, low_fallback=%s).",
            used_close_for_high,
            used_close_for_low,
        )
    n = int(close.size)
    if n < max(10, int(cfg.min_input_bars)):
        return None
    if "time" in df.columns:
        times = to_float_np(df["time"])
        if times.size != n:
            times = np.arange(n, dtype=float)
    else:
        times = np.arange(n, dtype=float)
    return df, times, close, high, low, n


def _build_swing_points(
    peaks: np.ndarray,
    troughs: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    cfg: HarmonicDetectorConfig,
) -> List[_SwingPoint]:
    raw: List[_SwingPoint] = []
    for idx in np.asarray(peaks, dtype=int).tolist():
        if 0 <= idx < high.size and np.isfinite(float(high[idx])):
            raw.append(_SwingPoint(index=int(idx), kind="high", price=float(high[idx])))
    for idx in np.asarray(troughs, dtype=int).tolist():
        if 0 <= idx < low.size and np.isfinite(float(low[idx])):
            raw.append(_SwingPoint(index=int(idx), kind="low", price=float(low[idx])))
    raw.sort(key=lambda p: (p.index, 0 if p.kind == "low" else 1))

    swings: List[_SwingPoint] = []
    for point in raw:
        if not swings:
            swings.append(point)
            continue
        prev = swings[-1]
        if point.index == prev.index:
            better = (
                point
                if (
                    point.kind == "high"
                    and (prev.kind != "high" or point.price > prev.price)
                    or point.kind == "low"
                    and (prev.kind != "low" or point.price < prev.price)
                )
                else prev
            )
            swings[-1] = better
            continue
        if point.kind == prev.kind:
            if (point.kind == "high" and point.price > prev.price) or (
                point.kind == "low" and point.price < prev.price
            ):
                swings[-1] = point
            continue
        swings.append(point)

    max_pivots = max(4, int(cfg.max_pivots))
    if len(swings) > max_pivots:
        swings = swings[-max_pivots:]
    return swings


def _finite_positive(value: float) -> bool:
    return bool(np.isfinite(float(value)) and float(value) > 1e-12)


def _ratios_xabcd(points: List[_SwingPoint]) -> Optional[Dict[str, float]]:
    x, a, b, c, d = [float(p.price) for p in points]
    xa = abs(a - x)
    ab = abs(b - a)
    bc = abs(c - b)
    cd = abs(d - c)
    xc = abs(c - x)
    if not all(_finite_positive(v) for v in (xa, ab, bc, cd)):
        return None
    ratios = {
        "xab": ab / xa,
        "abc": bc / ab,
        "bcd": cd / bc,
        "cd_ab": cd / ab,
        "xad": abs(d - a) / xa,
        "xd": abs(d - x) / xa,
    }
    if _finite_positive(xc):
        ratios["xcd"] = cd / xc
    return ratios


def _ratios_abcd(points: List[_SwingPoint]) -> Optional[Dict[str, float]]:
    a, b, c, d = [float(p.price) for p in points]
    ab = abs(b - a)
    bc = abs(c - b)
    cd = abs(d - c)
    ac = abs(c - a)
    if not all(_finite_positive(v) for v in (ab, bc, cd)):
        return None
    ratios = {
        "abc": bc / ab,
        "bcd": cd / bc,
        "cd_ab": cd / ab,
    }
    if _finite_positive(ac):
        ratios["acd"] = cd / ac
    return ratios


def _ratio_abs_tolerance(lo: float, hi: float, cfg: HarmonicDetectorConfig) -> float:
    basis = max(abs(float(lo)), abs(float(hi)))
    return max(1e-9, float(cfg.ratio_tolerance) * basis)


def _score_ratio(
    value: float,
    lo: float,
    hi: float,
    cfg: HarmonicDetectorConfig,
) -> Optional[float]:
    if not np.isfinite(float(value)):
        return None
    lo_f = float(min(lo, hi))
    hi_f = float(max(lo, hi))
    tol = _ratio_abs_tolerance(lo_f, hi_f, cfg)
    value_f = float(value)
    if abs(hi_f - lo_f) <= 1e-12:
        dist = abs(value_f - lo_f)
        if dist > tol:
            return None
        return float(max(0.0, 1.0 - dist / tol))
    # Peak at band midpoint so wide acceptance ranges do not inflate confidence.
    mid = 0.5 * (lo_f + hi_f)
    half = 0.5 * (hi_f - lo_f)
    if lo_f <= value_f <= hi_f:
        if half <= 1e-12:
            return 1.0
        return float(max(0.0, 1.0 - abs(value_f - mid) / half))
    dist = lo_f - value_f if value_f < lo_f else value_f - hi_f
    if dist > tol:
        return None
    return float(max(0.0, 1.0 - dist / tol))


def _score_spec(
    ratios: Dict[str, float],
    spec: _PatternSpec,
    cfg: HarmonicDetectorConfig,
) -> Optional[Tuple[float, Dict[str, float]]]:
    scores: Dict[str, float] = {}
    for key, bounds in spec.ratios.items():
        if key not in ratios:
            return None
        score = _score_ratio(float(ratios[key]), bounds[0], bounds[1], cfg)
        if score is None:
            return None
        scores[key] = float(score)
    if not scores:
        return None
    return float(sum(scores.values()) / len(scores)), scores


def _projection_values(
    points: List[_SwingPoint],
    spec: _PatternSpec,
    bullish: bool,
) -> List[float]:
    values: List[float] = []
    prices = [float(p.price) for p in points]
    if spec.family == "XABCD":
        x, a, b, c, _ = prices
        xa = abs(a - x)
        ab = abs(b - a)
        bc = abs(c - b)
        xc = abs(c - x)
    else:
        a, b, c, _ = prices
        xa = 0.0
        ab = abs(b - a)
        bc = abs(c - b)
        xc = 0.0
    for key, bounds in spec.ratios.items():
        if key not in {"xad", "xcd", "bcd", "cd_ab"}:
            continue
        ratio_values = [float(bounds[0])]
        if abs(float(bounds[1]) - float(bounds[0])) > 1e-12:
            ratio_values.append(float(bounds[1]))
        for ratio in ratio_values:
            projection: Optional[float] = None
            if key == "xad" and _finite_positive(xa):
                projection = a - ratio * xa if bullish else a + ratio * xa
            elif key == "xcd" and _finite_positive(xc):
                projection = c - ratio * xc if bullish else c + ratio * xc
            elif key == "bcd" and _finite_positive(bc):
                projection = c - ratio * bc if bullish else c + ratio * bc
            elif key == "cd_ab" and _finite_positive(ab):
                projection = c - ratio * ab if bullish else c + ratio * ab
            if projection is not None and np.isfinite(float(projection)):
                values.append(float(projection))
    return values


def _confidence_from_components(
    ratio_score: float,
    projections: List[float],
    entry_price: float,
    points: List[_SwingPoint],
    cfg: HarmonicDetectorConfig,
) -> float:
    confluence = 0.5
    if projections:
        width = max(projections) - min(projections)
        last_leg = abs(float(points[-2].price) - float(points[-1].price))
        scale = max(last_leg, abs(entry_price) * 0.005, 1e-12)
        confluence = max(0.0, min(1.0, 1.0 - width / scale))
    confidence = 0.8 * float(ratio_score) + 0.2 * float(confluence)
    return float(max(0.0, min(0.98, confidence)))


def _make_targets(
    points: List[_SwingPoint],
    bullish: bool,
    cfg: HarmonicDetectorConfig,
) -> Tuple[float, float, float]:
    entry = float(points[-1].price)
    c_price = float(points[-2].price)
    cd = abs(c_price - entry)
    if bullish:
        target_1 = entry + 0.382 * cd
        target_2 = entry + 0.618 * cd
        invalidation = min(entry, float(points[0].price)) - abs(entry) * float(cfg.stop_buffer_pct) / 100.0
    else:
        target_1 = entry - 0.382 * cd
        target_2 = entry - 0.618 * cd
        invalidation = max(entry, float(points[0].price)) + abs(entry) * float(cfg.stop_buffer_pct) / 100.0
    return float(target_1), float(target_2), float(invalidation)


def _build_result(
    points: List[_SwingPoint],
    spec: _PatternSpec,
    ratios: Dict[str, float],
    ratio_scores: Dict[str, float],
    confidence: float,
    projections: List[float],
    t: np.ndarray,
    n_bars: int,
    cfg: HarmonicDetectorConfig,
) -> HarmonicPatternResult:
    bullish = points[-1].kind == "low"
    bias = "bullish" if bullish else "bearish"
    direction_label = "Bullish" if bullish else "Bearish"
    pivot_labels = list("XABCD") if spec.family == "XABCD" else list("ABCD")
    pivot_indexes = {label: int(point.index) for label, point in zip(pivot_labels, points)}
    pivot_prices = {label: float(point.price) for label, point in zip(pivot_labels, points)}
    entry = float(points[-1].price)
    target_1, target_2, invalidation = _make_targets(points, bullish, cfg)
    prz_values = list(projections) if projections else [entry]
    prz_low = float(min(prz_values))
    prz_high = float(max(prz_values))
    prz_mid = float((prz_low + prz_high) / 2.0)
    bars_since_d = int(max(0, (n_bars - 1) - int(points[-1].index)))
    completion_freshness = (
        "recent" if bars_since_d <= int(cfg.recent_bars) else "historical"
    )

    details: Dict[str, Any] = {
        "pattern_family": "harmonic",
        "harmonic_pattern": spec.key,
        "bias": bias,
        "direction": bias,
        "pivot_labels": pivot_labels,
        "pivot_indexes": pivot_indexes,
        "pivot_prices": pivot_prices,
        "ratios": {key: float(round(value, 8)) for key, value in ratios.items()},
        "ratio_scores": {key: float(round(value, 6)) for key, value in ratio_scores.items()},
        "expected_ratios": {
            key: [float(bounds[0]), float(bounds[1])] for key, bounds in spec.ratios.items()
        },
        "prz_low": float(prz_low),
        "prz_mid": float(prz_mid),
        "prz_high": float(prz_high),
        "entry_price": float(entry),
        "target_price_1": float(target_1),
        "target_price_2": float(target_2),
        "invalidation_price": float(invalidation),
        "bars_since_completion_pivot": int(bars_since_d),
        "completion_freshness": completion_freshness,
    }
    return HarmonicPatternResult(
        name=f"{direction_label} {spec.display}",
        status="completed",
        confidence=confidence,
        start_index=int(points[0].index),
        end_index=int(points[-1].index),
        start_time=PatternResultBase.resolve_time(t, int(points[0].index)),
        end_time=PatternResultBase.resolve_time(t, int(points[-1].index)),
        bias=bias,
        entry_price=entry,
        target_prices=[float(target_1), float(target_2)],
        invalidation_price=float(invalidation),
        details=details,
    )


def _candidate_results(
    points: List[_SwingPoint],
    specs: List[_PatternSpec],
    t: np.ndarray,
    n_bars: int,
    cfg: HarmonicDetectorConfig,
) -> List[HarmonicPatternResult]:
    ratios = _ratios_xabcd(points) if len(points) == 5 else _ratios_abcd(points)
    if ratios is None:
        return []
    bullish = points[-1].kind == "low"
    out: List[HarmonicPatternResult] = []
    for spec in specs:
        scored = _score_spec(ratios, spec, cfg)
        if scored is None:
            continue
        ratio_score, ratio_scores = scored
        projections = _projection_values(points, spec, bullish)
        confidence = _confidence_from_components(
            ratio_score,
            projections,
            float(points[-1].price),
            points,
            cfg,
        )
        if confidence < float(cfg.min_confidence):
            continue
        out.append(
            _build_result(
                points,
                spec,
                ratios,
                ratio_scores,
                confidence,
                projections,
                t,
                n_bars,
                cfg,
            )
        )
    return out


def _dedupe_results(results: List[HarmonicPatternResult]) -> List[HarmonicPatternResult]:
    best_by_signature: Dict[Tuple[int, ...], HarmonicPatternResult] = {}
    for result in results:
        details = result.details if isinstance(result.details, dict) else {}
        pivot_indexes = details.get("pivot_indexes")
        if isinstance(pivot_indexes, dict):
            signature = tuple(int(v) for _, v in sorted(pivot_indexes.items()))
        else:
            signature = (int(result.start_index), int(result.end_index))
        current = best_by_signature.get(signature)
        if current is None or float(result.confidence) > float(current.confidence):
            best_by_signature[signature] = result
    deduped = list(best_by_signature.values())
    deduped.sort(key=lambda r: (int(r.end_index), float(r.confidence)), reverse=True)
    return deduped


def detect_harmonic_patterns(
    df: pd.DataFrame,
    cfg: Optional[HarmonicDetectorConfig] = None,
) -> List[HarmonicPatternResult]:
    if cfg is None:
        cfg = HarmonicDetectorConfig()
    prepared = _prepare_inputs(df, cfg)
    if prepared is None:
        return []
    _, t, close, high, low, n_bars = prepared
    peaks, troughs = _detect_pivots(close, high, low, cfg)
    swings = _build_swing_points(peaks, troughs, high, low, cfg)
    if len(swings) < 4:
        return []

    enabled = set(_normalized_pattern_types(cfg))
    xabcd_specs = [spec for key, spec in _XABCD_SPECS.items() if key in enabled]
    abcd_specs = [_ABCD_SPEC] if "abcd" in enabled else []

    results: List[HarmonicPatternResult] = []
    max_age = int(getattr(cfg, "max_pattern_age_bars", 0))
    for start in range(0, max(0, len(swings) - 4)):
        window = swings[start : start + 5]
        if len(window) == 5 and xabcd_specs:
            results.extend(_candidate_results(window, xabcd_specs, t, n_bars, cfg))
    for start in range(0, max(0, len(swings) - 3)):
        window = swings[start : start + 4]
        if len(window) == 4 and abcd_specs:
            results.extend(_candidate_results(window, abcd_specs, t, n_bars, cfg))

    if max_age > 0:
        cutoff = int(n_bars) - int(max_age)
        results = [r for r in results if int(r.end_index) >= cutoff]
    if not bool(getattr(cfg, "include_forming", True)):
        results = [r for r in results if str(r.status).lower() == "completed"]
    return _dedupe_results(results)
