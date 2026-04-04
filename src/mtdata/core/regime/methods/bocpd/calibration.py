"""BOCPD (Bayesian Online Change Point Detection) calibration utilities."""
from typing import Any, Dict, Optional, Tuple
import numpy as np

from ...crypto import _is_probably_crypto_symbol
from .....shared.constants import TIMEFRAME_SECONDS


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
    """Auto-calibrate BOCPD hazard/threshold from recent return distribution."""
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
    jump_share = float(np.mean(np.abs(z) >= 2.5)) if z.size > 0 else 0.0
    trend_strength = float(abs(mu) / sigma_safe)
    move_zscore = float(abs(np.sum(r)) / (sigma_safe * np.sqrt(max(1, int(r.size)))))

    vol_norm = float(np.clip(sigma / 0.003, 0.0, 3.0))
    kurt_norm = float(np.clip(max(0.0, kurt) / 6.0, 0.0, 2.0))
    jump_norm = float(np.clip(jump_share / 0.08, 0.0, 2.0))
    trend_norm = float(np.clip(trend_strength / 0.20, 0.0, 2.0))
    move_sig_norm = float(np.clip(move_zscore / 3.0, 0.0, 2.5))

    sensitivity = float(
        np.clip(
            1.0
            + 0.35 * vol_norm
            + 0.25 * kurt_norm
            + 0.25 * jump_norm
            - 0.20 * trend_norm
            + 0.60 * move_sig_norm,
            0.5,
            2.2,
        )
    )

    hazard_floor = max(12, int(round(base_lambda * 0.25)))
    hazard_cap = min(500, max(hazard_floor + 1, int(round(base_lambda * 1.80))))
    hazard_lambda = int(np.clip(int(round(base_lambda / sensitivity)), hazard_floor, hazard_cap))

    cp_threshold = float(
        np.clip(
            base_threshold
            - 0.08 * (vol_norm / 3.0)
            - 0.06 * (jump_norm / 2.0)
            - 0.04 * (kurt_norm / 2.0)
            + 0.04 * (trend_norm / 2.0),
            0.15,
            0.75,
        )
    )
    if move_sig_norm > 0.0:
        cp_threshold = float(np.clip(cp_threshold - 0.08 * (move_sig_norm / 2.5), 0.15, 0.75))

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
