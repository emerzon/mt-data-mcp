"""BOCPD (Bayesian Online Change Point Detection) calibration utilities."""
from typing import Any, Dict, Tuple
import numpy as np

from ...crypto import _is_probably_crypto_symbol
from .....shared.constants import TIMEFRAME_SECONDS

# Empirical BOCPD calibration defaults. Higher volatility, fatter tails, more
# jump activity, and larger cumulative moves should make the detector react
# faster, while stronger drift/trend should damp that sensitivity a bit to
# avoid over-firing on sustained directional moves.
_BOCPD_JUMP_ZSCORE_THRESHOLD = 2.5
_BOCPD_VOL_NORM_BASE = 0.003
_BOCPD_VOL_NORM_CAP = 3.0
_BOCPD_KURTOSIS_NORM_BASE = 6.0
_BOCPD_KURTOSIS_NORM_CAP = 2.0
_BOCPD_JUMP_SHARE_NORM_BASE = 0.08
_BOCPD_JUMP_SHARE_NORM_CAP = 2.0
_BOCPD_TREND_STRENGTH_NORM_BASE = 0.20
_BOCPD_TREND_STRENGTH_NORM_CAP = 2.0
_BOCPD_MOVE_ZSCORE_NORM_BASE = 3.0
_BOCPD_MOVE_ZSCORE_NORM_CAP = 2.5
_BOCPD_SENSITIVITY_VOL_WEIGHT = 0.35
_BOCPD_SENSITIVITY_KURT_WEIGHT = 0.25
_BOCPD_SENSITIVITY_JUMP_WEIGHT = 0.25
_BOCPD_SENSITIVITY_TREND_WEIGHT = -0.20
_BOCPD_SENSITIVITY_MOVE_WEIGHT = 0.60
_BOCPD_SENSITIVITY_MIN = 0.5
_BOCPD_SENSITIVITY_MAX = 2.2
_BOCPD_HAZARD_FLOOR_MIN = 12
_BOCPD_HAZARD_FLOOR_RATIO = 0.25
_BOCPD_HAZARD_CAP_MAX = 500
_BOCPD_HAZARD_CAP_RATIO = 1.80
_BOCPD_CP_THRESHOLD_VOL_WEIGHT = 0.08
_BOCPD_CP_THRESHOLD_JUMP_WEIGHT = 0.06
_BOCPD_CP_THRESHOLD_KURT_WEIGHT = 0.04
_BOCPD_CP_THRESHOLD_TREND_WEIGHT = 0.04
_BOCPD_CP_THRESHOLD_MOVE_WEIGHT = 0.08
_BOCPD_CP_THRESHOLD_MIN = 0.15
_BOCPD_CP_THRESHOLD_MAX = 0.75


def _default_bocpd_hazard_lambda(symbol: Any, timeframe: Any) -> int:
    """Get default hazard lambda based on symbol type and timeframe."""
    tf = str(timeframe or "H1").upper().strip() or "H1"
    tf_seconds = int(TIMEFRAME_SECONDS.get(tf, 3600))

    if _is_probably_crypto_symbol(symbol):
        if tf_seconds <= 900:   # <= M15
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
    """Get default CP threshold based on symbol type and timeframe."""
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
    """Auto-calibrate BOCPD hazard/threshold from recent return distribution.

    The private constants used below are empirical defaults that normalize
    recent volatility, tail risk, jump frequency, drift strength, and
    cumulative move significance onto a common scale before nudging the hazard
    lambda and CP threshold.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    base_lambda = int(_default_bocpd_hazard_lambda(symbol, timeframe))
    base_threshold = float(_default_bocpd_cp_threshold(symbol, timeframe))
    if r.size < 30:
        return base_lambda, base_threshold, {
            "calibrated": False,
            "reason": "insufficient_points",
            "points": int(r.size),
            "base_hazard_lambda": int(base_lambda),
            "base_cp_threshold": float(base_threshold),
        }

    mu = float(np.mean(r))
    sigma = float(np.std(r, ddof=0))
    sigma_safe = sigma if sigma > 1e-12 else 1e-12
    centered = r - mu
    z = centered / sigma_safe
    kurt = float(np.mean(z ** 4) - 3.0) if z.size > 0 else 0.0
    jump_share = float(np.mean(np.abs(z) >= _BOCPD_JUMP_ZSCORE_THRESHOLD)) if z.size > 0 else 0.0
    trend_strength = float(abs(mu) / sigma_safe)
    move_zscore = float(abs(np.sum(r)) / (sigma_safe * np.sqrt(max(1, int(r.size)))))

    vol_norm = float(np.clip(sigma / _BOCPD_VOL_NORM_BASE, 0.0, _BOCPD_VOL_NORM_CAP))
    kurt_norm = float(np.clip(max(0.0, kurt) / _BOCPD_KURTOSIS_NORM_BASE, 0.0, _BOCPD_KURTOSIS_NORM_CAP))
    jump_norm = float(np.clip(jump_share / _BOCPD_JUMP_SHARE_NORM_BASE, 0.0, _BOCPD_JUMP_SHARE_NORM_CAP))
    trend_norm = float(
        np.clip(
            trend_strength / _BOCPD_TREND_STRENGTH_NORM_BASE,
            0.0,
            _BOCPD_TREND_STRENGTH_NORM_CAP,
        )
    )
    move_sig_norm = float(np.clip(move_zscore / _BOCPD_MOVE_ZSCORE_NORM_BASE, 0.0, _BOCPD_MOVE_ZSCORE_NORM_CAP))

    sensitivity = float(
        np.clip(
            1.0
            + _BOCPD_SENSITIVITY_VOL_WEIGHT * vol_norm
            + _BOCPD_SENSITIVITY_KURT_WEIGHT * kurt_norm
            + _BOCPD_SENSITIVITY_JUMP_WEIGHT * jump_norm
            + _BOCPD_SENSITIVITY_TREND_WEIGHT * trend_norm
            + _BOCPD_SENSITIVITY_MOVE_WEIGHT * move_sig_norm,
            _BOCPD_SENSITIVITY_MIN,
            _BOCPD_SENSITIVITY_MAX,
        )
    )

    hazard_floor = max(_BOCPD_HAZARD_FLOOR_MIN, int(round(base_lambda * _BOCPD_HAZARD_FLOOR_RATIO)))
    hazard_cap = min(
        _BOCPD_HAZARD_CAP_MAX,
        max(hazard_floor + 1, int(round(base_lambda * _BOCPD_HAZARD_CAP_RATIO))),
    )
    hazard_lambda = int(np.clip(int(round(base_lambda / sensitivity)), hazard_floor, hazard_cap))

    cp_threshold = float(
        np.clip(
            base_threshold
            - _BOCPD_CP_THRESHOLD_VOL_WEIGHT * (vol_norm / _BOCPD_VOL_NORM_CAP)
            - _BOCPD_CP_THRESHOLD_JUMP_WEIGHT * (jump_norm / _BOCPD_JUMP_SHARE_NORM_CAP)
            - _BOCPD_CP_THRESHOLD_KURT_WEIGHT * (kurt_norm / _BOCPD_KURTOSIS_NORM_CAP)
            + _BOCPD_CP_THRESHOLD_TREND_WEIGHT * (trend_norm / _BOCPD_TREND_STRENGTH_NORM_CAP),
            _BOCPD_CP_THRESHOLD_MIN,
            _BOCPD_CP_THRESHOLD_MAX,
        )
    )
    if move_sig_norm > 0.0:
        cp_threshold = float(
            np.clip(
                cp_threshold - _BOCPD_CP_THRESHOLD_MOVE_WEIGHT * (move_sig_norm / _BOCPD_MOVE_ZSCORE_NORM_CAP),
                _BOCPD_CP_THRESHOLD_MIN,
                _BOCPD_CP_THRESHOLD_MAX,
            )
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


__all__ = [
    "_default_bocpd_hazard_lambda",
    "_default_bocpd_cp_threshold",
    "_auto_calibrate_bocpd_params",
]
