"""Risk-based position sizing helper.

Provides a standalone function to compute trade volume from risk parameters,
independent of the full risk-analyze use case. Ready for use by order-entry
paths that want to derive volume from risk percentage.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple


def _floor_volume_steps(raw: float, step: float) -> int:
    """Floor to nearest volume step count."""
    if step <= 0 or not math.isfinite(raw):
        return 0
    return int(math.floor(raw / step))


def _resolve_risk_tick_value(
    *,
    tick_value: float,
    tick_value_loss: Optional[float] = None,
) -> float:
    """Prefer the broker-reported loss tick value for downside-risk math."""
    try:
        loss_tick_value = float(tick_value_loss)  # type: ignore[arg-type]
    except Exception:
        loss_tick_value = float("nan")
    if math.isfinite(loss_tick_value) and loss_tick_value > 0:
        return loss_tick_value
    return float(tick_value)


def compute_risk_based_volume(
    *,
    equity: float,
    risk_pct: float,
    entry_price: float,
    stop_loss_price: float,
    direction: str,
    tick_value: float,
    tick_value_loss: Optional[float] = None,
    tick_size: float,
    volume_min: float,
    volume_max: float,
    volume_step: float,
) -> Tuple[Optional[float], Dict[str, Any]]:
    """Compute position volume from risk parameters.

    Returns ``(volume, metadata)`` where volume is ``None`` on error.
    The metadata dict always contains either ``"error"`` or ``"suggested_volume"``.
    """
    errors: List[str] = []
    risk_tick_value = _resolve_risk_tick_value(
        tick_value=tick_value,
        tick_value_loss=tick_value_loss,
    )

    if not math.isfinite(equity) or equity <= 0:
        errors.append("Equity must be positive and finite.")
    if not math.isfinite(risk_pct) or risk_pct <= 0:
        errors.append("risk_pct must be positive and finite.")
    if not math.isfinite(entry_price):
        errors.append("entry_price must be finite.")
    if not math.isfinite(stop_loss_price):
        errors.append("stop_loss_price must be finite.")
    if not math.isfinite(risk_tick_value) or risk_tick_value <= 0:
        errors.append("tick_value must be positive and finite.")
    if not math.isfinite(tick_size) or tick_size <= 0:
        errors.append("tick_size must be positive and finite.")

    if errors:
        return None, {"error": "; ".join(errors)}

    risk_amount = equity * (risk_pct / 100.0)

    if direction.lower() == "long":
        sl_distance_ticks = (entry_price - stop_loss_price) / tick_size
    else:
        sl_distance_ticks = (stop_loss_price - entry_price) / tick_size

    if sl_distance_ticks <= 0:
        return None, {
            "error": "Stop-loss distance must be positive (wrong side of entry?).",
            "sl_distance_ticks": round(sl_distance_ticks, 4),
        }

    raw_volume = risk_amount / (sl_distance_ticks * risk_tick_value)
    if not math.isfinite(raw_volume) or raw_volume <= 0:
        return None, {"error": "Calculated volume is invalid."}

    if not math.isfinite(volume_step) or volume_step <= 0:
        volume_step = max(volume_min, 0.01)

    volume_steps = _floor_volume_steps(raw_volume, volume_step)
    suggested = volume_steps * volume_step

    rounding_mode = "rounded_down_to_step"
    notes: List[str] = []

    if suggested < volume_min:
        suggested = volume_min
        rounding_mode = "clamped_to_min_volume"
        notes.append("Clamped up to broker minimum volume.")
    elif suggested > volume_max:
        suggested = volume_max
        rounding_mode = "clamped_to_max_volume"
        notes.append("Clamped down to broker maximum volume.")

    # Round to volume_step precision
    step_txt = f"{volume_step:.10f}".rstrip("0")
    step_decimals = len(step_txt.split(".")[1]) if "." in step_txt else 0
    if step_decimals > 0:
        suggested = float(f"{suggested:.{step_decimals}f}")
    else:
        suggested = float(round(suggested))

    actual_risk = sl_distance_ticks * risk_tick_value * suggested
    actual_risk_pct = (actual_risk / equity) * 100.0

    return suggested, {
        "suggested_volume": suggested,
        "raw_volume": round(raw_volume, 8),
        "risk_amount": round(risk_amount, 2),
        "actual_risk": round(actual_risk, 2),
        "actual_risk_pct": round(actual_risk_pct, 4),
        "sl_distance_ticks": round(sl_distance_ticks, 4),
        "risk_tick_value": round(risk_tick_value, 8),
        "volume_rounding": rounding_mode,
        "notes": notes,
    }
