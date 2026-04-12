from typing import List

import numpy as np

from ..common import PatternResultBase
from ..common import interval_overlap_ratio as _interval_overlap_ratio
from .config import ClassicDetectorConfig, ClassicPatternResult
from .utils import (
    _apply_breakout_confidence_bonus,
    _dtw_distance,
    _find_forward_level_breakout,
    _fit_line,
    _level_close,
    _paa,
    _result,
    _template_hs_variants,
    _tol_abs_from_close,
    _znorm,
)


def _dedupe_overlapping_patterns(
    results: List[ClassicPatternResult],
    *,
    overlap_threshold: float = 0.6,
) -> List[ClassicPatternResult]:
    deduped: List[ClassicPatternResult] = []
    ordered = sorted(
        results,
        key=lambda r: (
            float(r.confidence),
            int(r.end_index) - int(r.start_index),
            int(r.end_index),
        ),
        reverse=True,
    )
    for candidate in ordered:
        if any(
            candidate.name == prior.name
            and _interval_overlap_ratio(
                int(candidate.start_index),
                int(candidate.end_index),
                int(prior.start_index),
                int(prior.end_index),
            ) >= float(overlap_threshold)
            for prior in deduped
        ):
            continue
        deduped.append(candidate)
    deduped.sort(key=lambda r: (int(r.start_index), int(r.end_index)))
    return deduped


def _level_components(vals: np.ndarray, tol_pct: float) -> List[List[int]]:
    n = int(vals.size)
    if n <= 0:
        return []
    order = np.argsort(np.asarray(vals, dtype=float), kind="mergesort")
    components: List[List[int]] = []
    component = [int(order[0])]
    for pos in range(1, n):
        prev_idx = int(order[pos - 1])
        cur_idx = int(order[pos])
        if _level_close(float(vals[prev_idx]), float(vals[cur_idx]), tol_pct):
            component.append(cur_idx)
            continue
        component.sort()
        components.append(component)
        component = [cur_idx]
    component.sort()
    components.append(component)
    components.sort(key=lambda item: item[0])
    return components


def _neckline_quality_score(
    *,
    slope: float,
    r2: float,
    point_count: int,
    cfg: ClassicDetectorConfig,
) -> float:
    neck_penalty = max(0.0, 1.0 - min(1.0, abs(float(slope)) / max(1e-6, cfg.max_flat_slope * 5.0)))
    if int(point_count) <= 2:
        return float(neck_penalty)
    return float(0.5 * neck_penalty + 0.5 * max(0.0, min(1.0, float(r2))))

def detect_tops_bottoms(
    c: np.ndarray,
    peaks: np.ndarray,
    troughs: np.ndarray,
    t: np.ndarray,
    cfg: ClassicDetectorConfig
) -> List[ClassicPatternResult]:
    out: List[ClassicPatternResult] = []
    
    def group_levels(idxs: np.ndarray, name_top: str, name_triple: str, kind: str):
        if idxs.size < 2:
            return
        vals = c[idxs]
        tol_abs = _tol_abs_from_close(c, cfg.same_level_tol_pct)
        for cluster in _level_components(vals.astype(float), float(cfg.same_level_tol_pct)):
            if len(cluster) >= 2:
                name = name_triple if len(cluster) >= 3 else name_top
                ii = idxs[cluster]
                start_i, end_i = int(ii[0]), int(ii[-1])
                status = "forming"
                level = float(np.median(vals[cluster]))
                neckline = float(np.min(c[start_i:end_i + 1])) if kind == "top" else float(np.max(c[start_i:end_i + 1]))
                direction = "down" if kind == "top" else "up"
                breakout_look = max(int(cfg.completion_lookback_bars), int(max(1, cfg.breakout_lookahead)))
                break_i = _find_forward_level_breakout(
                    c,
                    end_i,
                    neckline,
                    direction,
                    breakout_look,
                    tol_abs,
                    tol_pct=float(cfg.same_level_tol_pct),
                )
                if break_i is not None:
                    status = "completed"
                out.append(ClassicPatternResult(
                    name=name,
                    status=status,
                    confidence=(
                        _apply_breakout_confidence_bonus((0.5 + 0.1 * (len(cluster) - 2)), cfg)
                        if break_i is not None
                        else float(min(1.0, (0.5 + 0.1 * (len(cluster) - 2))))
                    ),
                    start_index=start_i,
                    end_index=int(break_i if break_i is not None else end_i),
                    start_time=PatternResultBase.resolve_time(t, start_i),
                    end_time=PatternResultBase.resolve_time(t, int(break_i if break_i is not None else end_i)),
                    details={
                        "level": level,
                        "touches": int(len(cluster)),
                        "neckline": neckline,
                        "breakout_direction": direction if break_i is not None else None,
                        "breakout_index": int(break_i) if break_i is not None else None,
                        "bias": "bearish" if "Top" in name else "bullish",
                    },
                ))
    
    group_levels(peaks[-10:], "Double Top", "Triple Top", "top")
    group_levels(troughs[-10:], "Double Bottom", "Triple Bottom", "bottom")
    return out

def detect_head_shoulders(  # noqa: C901
    c: np.ndarray,
    peaks: np.ndarray,
    troughs: np.ndarray,
    t: np.ndarray,
    cfg: ClassicDetectorConfig
) -> List[ClassicPatternResult]:
    out: List[ClassicPatternResult] = []
    n = c.size
    if peaks.size < 3 or troughs.size < 2:
        return out
        
    tol_pct = float(cfg.same_level_tol_pct)
    breakout_look = max(int(cfg.completion_lookback_bars), int(max(1, cfg.breakout_lookahead)))
    pivot_cap = int(getattr(cfg, "head_shoulders_max_peak_candidates", 0))
    if pivot_cap <= 0:
        pivot_cap = max(6, int(cfg.max_pattern_pivots) * 2)
    peak_candidates = peaks[-pivot_cap:] if peaks.size > pivot_cap else peaks
    trough_candidates = troughs[-pivot_cap:] if troughs.size > pivot_cap else troughs
    peak_list = [int(v) for v in peak_candidates.tolist()]
    trough_list = [int(v) for v in trough_candidates.tolist()]

    def _append_pattern(
        *,
        head_idx: int,
        lsh: int,
        rsh: int,
        regular: bool,
        neck_idxs: np.ndarray,
        validation_source: List[int],
    ) -> None:
        if neck_idxs.size < 2:
            return
        head_price = float(c[head_idx])
        ls_p = float(c[lsh])
        rs_p = float(c[rsh])
        neck_x = neck_idxs.astype(float)
        neck_y = c[neck_idxs]
        slope, intercept, r2 = _fit_line(neck_x, neck_y)
        neck_idx_set = {int(v) for v in neck_idxs.tolist()}
        neck_validation_points = int(
            sum(
                1
                for idx in validation_source
                if lsh < int(idx) < rsh and int(idx) not in neck_idx_set
            )
        )

        left_span = head_idx - lsh
        right_span = rsh - head_idx
        span_ratio = left_span / float(max(1, right_span))
        if not (0.5 <= span_ratio <= 2.0):
            return

        sh_avg = (ls_p + rs_p) / 2.0
        if sh_avg == 0.0:
            return
        shoulder_eps = max(1e-9, abs(sh_avg) * 1e-9)
        head_prom = (
            (head_price - sh_avg) / abs(sh_avg) * 100.0
            if regular
            else (sh_avg - head_price) / abs(sh_avg) * 100.0
        )
        if head_prom < max(1.0, tol_pct):
            return
        left_neck_at_shoulder = float(slope * float(lsh) + intercept)
        right_neck_at_shoulder = float(slope * float(rsh) + intercept)
        if not np.isfinite(left_neck_at_shoulder) or not np.isfinite(right_neck_at_shoulder):
            return
        if regular:
            if (
                left_neck_at_shoulder >= (ls_p + shoulder_eps)
                or right_neck_at_shoulder >= (rs_p + shoulder_eps)
            ):
                return
        else:
            if (
                left_neck_at_shoulder <= (ls_p - shoulder_eps)
                or right_neck_at_shoulder <= (rs_p - shoulder_eps)
            ):
                return

        status = 'forming'
        name = 'Head and Shoulders' if regular else 'Inverse Head and Shoulders'
        broke = False
        end_i = int(rsh)
        for k in range(1, breakout_look + 1):
            i = rsh + k
            if i >= n:
                break
            neck_i = slope * i + intercept
            px = float(c[i])
            if regular and px < neck_i:
                status = 'completed'; broke = True; end_i = int(i); break
            if (not regular) and px > neck_i:
                status = 'completed'; broke = True; end_i = int(i); break

        sym_conf = max(0.0, 1.0 - abs(span_ratio - 1.0))
        sh_sim_conf = max(0.0, 1.0 - (abs(ls_p - rs_p) / max(1e-9, abs(sh_avg))))
        neck_quality = _neckline_quality_score(
            slope=float(slope),
            r2=float(r2),
            point_count=int(neck_idxs.size),
            cfg=cfg,
        )
        prom_conf = min(1.0, head_prom / (tol_pct * 2.0))
        base_conf = 0.25 * sym_conf + 0.35 * sh_sim_conf + 0.2 * neck_quality + 0.2 * prom_conf
        if broke:
            base_conf = min(1.0, base_conf + 0.1)

        if getattr(cfg, 'use_dtw_check', False):
            seg_start = max(0, int(lsh))
            seg_end = int(end_i if broke else rsh)
            seg = c[seg_start: seg_end + 1].astype(float)
            seg_n = _znorm(_paa(seg, int(getattr(cfg, 'dtw_paa_len', 80))))
            dist = min(
                _dtw_distance(seg_n, tpl)
                for tpl in _template_hs_variants(len(seg_n), inverse=not regular)
            )
            maxd = float(getattr(cfg, 'dtw_max_dist', 0.6))
            if not np.isfinite(dist):
                dist = maxd * 10.0
            if dist > (2.0 * maxd):
                return
            if dist > maxd:
                base_conf *= 0.7
            else:
                base_conf = min(1.0, base_conf + 0.1)

        details = {
            'left_shoulder': float(ls_p),
            'right_shoulder': float(rs_p),
            'head': float(head_price),
            'neckline_source': 'troughs' if regular else 'peaks',
            'neck_slope': float(slope),
            'neck_intercept': float(intercept),
            'neck_r2': float(r2),
            'neck_points': int(neck_idxs.size),
            'neck_validation_points': int(neck_validation_points),
            'bias': 'bearish' if regular else 'bullish',
        }
        out.append(ClassicPatternResult(
            name=name,
            status=status,
            confidence=float(base_conf),
            start_index=int(lsh),
            end_index=end_i,
            start_time=PatternResultBase.resolve_time(t, int(lsh)),
            end_time=PatternResultBase.resolve_time(t, int(end_i)),
            details=details,
        ))

    for head_idx in peak_list:
        if head_idx < 0 or head_idx >= n:
            continue
        ls_candidates = [pi for pi in peak_list if pi < head_idx]
        rs_candidates = [pi for pi in peak_list if pi > head_idx]
        if not ls_candidates or not rs_candidates:
            continue
        lsh = int(ls_candidates[-1])
        rsh = int(rs_candidates[0])
        if lsh < 0 or rsh >= n:
            continue
        head_price = float(c[head_idx])
        ls_p = float(c[lsh])
        rs_p = float(c[rsh])
        if not ((ls_p < head_price) and (rs_p < head_price)):
            continue
        if not _level_close(ls_p, rs_p, tol_pct * 1.5):
            continue
        nl1_candidates = [ti for ti in trough_list if lsh < ti < head_idx]
        nl2_candidates = [ti for ti in trough_list if head_idx < ti < rsh]
        if not nl1_candidates or not nl2_candidates:
            continue
        neck_idxs = np.asarray([int(nl1_candidates[-1]), int(nl2_candidates[0])], dtype=int)
        _append_pattern(
            head_idx=int(head_idx),
            lsh=int(lsh),
            rsh=int(rsh),
            regular=True,
            neck_idxs=neck_idxs,
            validation_source=trough_list,
        )

    for head_idx in trough_list:
        if head_idx < 0 or head_idx >= n:
            continue
        ls_candidates = [ti for ti in trough_list if ti < head_idx]
        rs_candidates = [ti for ti in trough_list if ti > head_idx]
        if not ls_candidates or not rs_candidates:
            continue
        lsh = int(ls_candidates[-1])
        rsh = int(rs_candidates[0])
        if lsh < 0 or rsh >= n:
            continue
        head_price = float(c[head_idx])
        ls_p = float(c[lsh])
        rs_p = float(c[rsh])
        if not ((ls_p > head_price) and (rs_p > head_price)):
            continue
        if not _level_close(ls_p, rs_p, tol_pct * 1.5):
            continue
        nl1_candidates = [pi for pi in peak_list if lsh < pi < head_idx]
        nl2_candidates = [pi for pi in peak_list if head_idx < pi < rsh]
        if not nl1_candidates or not nl2_candidates:
            continue
        neck_idxs = np.asarray([int(nl1_candidates[-1]), int(nl2_candidates[0])], dtype=int)
        _append_pattern(
            head_idx=int(head_idx),
            lsh=int(lsh),
            rsh=int(rsh),
            regular=False,
            neck_idxs=neck_idxs,
            validation_source=peak_list,
        )
    return _dedupe_overlapping_patterns(out)

def detect_rounding(
    c: np.ndarray,
    t: np.ndarray,
    cfg: ClassicDetectorConfig
) -> List[ClassicPatternResult]:
    out: List[ClassicPatternResult] = []
    n = c.size
    configured_windows = [int(v) for v in getattr(cfg, "rounding_window_sizes", []) if int(v) > 0]
    candidate_windows = configured_windows or [100, 150, int(cfg.rounding_window_bars), 300]
    valid_windows = sorted({min(int(w), n) for w in candidate_windows if min(int(w), n) >= 100})
    if not valid_windows:
        return out

    candidates: List[ClassicPatternResult] = []
    for W in valid_windows:
        seg = c[-W:]
        x = np.linspace(-1.0, 1.0, W)
        try:
            qa, qb, qc = np.polyfit(x, seg.astype(float), 2)
        except (TypeError, ValueError, np.linalg.LinAlgError):
            continue

        if not (np.isfinite(qa) and np.isfinite(qb) and np.isfinite(qc)):
            continue
        if abs(float(qa)) <= 1e-12:
            continue
        xv = -float(qb) / (2.0 * float(qa))
        if not (-0.55 <= xv <= 0.55):
            continue

        edge_n = max(6, W // 10)
        left_edge = float(np.mean(seg[:edge_n]))
        right_edge = float(np.mean(seg[-edge_n:]))
        if not _level_close(left_edge, right_edge, cfg.same_level_tol_pct * 2.0):
            continue

        peak = float(np.max(seg))
        trough = float(np.min(seg))
        amp_pct = abs(peak - trough) / max(1e-9, abs((peak + trough) / 2.0)) * 100.0
        if amp_pct < 2.0:
            continue

        tol_abs = _tol_abs_from_close(c, cfg.same_level_tol_pct)
        if qa > 0:
            name = "Rounding Bottom"
            status = "completed" if float(c[-1]) > (max(left_edge, right_edge) + tol_abs) else "forming"
            bias = "bullish"
        else:
            name = "Rounding Top"
            status = "completed" if float(c[-1]) < (min(left_edge, right_edge) - tol_abs) else "forming"
            bias = "bearish"

        conf = min(1.0, 0.5 + 0.3 * min(1.0, amp_pct / 12.0))
        candidate = _result(
            name,
            status,
            conf,
            int(n - W),
            int(n - 1),
            t,
            {
                "quad_a": float(qa),
                "quad_b": float(qb),
                "vertex_x_norm": float(xv),
                "left_edge": left_edge,
                "right_edge": right_edge,
                "amplitude_pct": float(amp_pct),
                "window_bars": int(W),
                "bias": bias,
            },
        )
        candidates.append(candidate)

    if not candidates:
        return out
    return _dedupe_overlapping_patterns(candidates, overlap_threshold=0.75)
