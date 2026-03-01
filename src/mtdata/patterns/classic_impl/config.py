from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List
from ..common import PatternResultBase

@dataclass
class ClassicDetectorConfig:
    # General
    max_bars: int = 1500
    # Pivot/zigzag parameters
    min_prominence_pct: float = 0.5  # peak/trough prominence in percent of price
    min_distance: int = 5            # minimum distance between pivots (bars)
    pivot_use_hl: bool = True        # use high/low (when available) for pivot extraction
    pivot_use_atr_adaptive_prominence: bool = True
    pivot_use_atr_adaptive_distance: bool = True
    pivot_atr_period: int = 14
    pivot_atr_prominence_mult: float = 1.0
    pivot_atr_distance_mult: float = 0.2
    pivot_max_distance_scale: float = 3.0
    # Trendline/line-fit
    max_flat_slope: float = 1e-4     # absolute slope to consider line flat (price units per bar)
    min_r2: float = 0.6              # minimum R^2 for line fit confidence
    # Levels tolerance (same-level checks)
    same_level_tol_pct: float = 0.4  # peaks/lows considered equal if within this percent
    # Pattern-specific
    min_touches: int = 2             # minimum touches to validate support/resistance boundary
    min_channel_touches: int = 4     # across both bounds
    max_consolidation_bars: int = 60 # for flags/pennants after pole
    min_pole_return_pct: float = 2.0 # minimum pole size (percent) before a flag/pennant
    breakout_lookahead: int = 8      # bars to consider breakout confirmation
    convergence_fallback_scale: float = 1.2  # fallback widening factor when no past window exists
    channel_parallel_slope_ratio: float = 0.15  # relative slope spread tolerated for channels
    pennant_parallel_slope_ratio: float = 0.2  # relative slope spread tolerated for flags/pennants
    # Confidence blending
    touch_weight: float = 0.35
    r2_weight: float = 0.35
    geometry_weight: float = 0.30
    # Robust fitting and shape checks
    use_robust_fit: bool = True     # use RANSAC for line fits when available
    ransac_residual_pct: float = 0.15  # residual threshold as fraction of median price
    ransac_min_samples: int = 2
    ransac_max_trials: int = 50
    use_dtw_check: bool = True      # optional DTW shape confirmation for select patterns
    dtw_paa_len: int = 80            # PAA downsampling length for DTW
    dtw_max_dist: float = 0.6        # acceptance threshold after z-norm
    # Output/completion controls
    include_aliases: bool = False    # include generic aliases like "Trend Line"/"Trend Channel"
    completion_confirm_bars: int = 2 # touches needed near the right edge to mark completed
    completion_lookback_bars: int = 5  # lookback window for completion confirmation
    auto_complete_stale_forming: bool = False  # backward-compat aging of old forming patterns
    include_lifecycle_metadata: bool = True
    # Optional confidence calibration map:
    # { "default": {"0.40": 0.35, "0.70": 0.62, "0.90": 0.82},
    #   "head and shoulders": {"0.50": 0.45, "0.80": 0.76} }
    calibrate_confidence: bool = False
    confidence_calibration_map: Dict[str, Any] = field(default_factory=dict)
    confidence_calibration_blend: float = 1.0


@dataclass
class ClassicPatternResult(PatternResultBase):
    name: str
    status: str  # "completed" | "forming"
    details: Dict[str, Any]
