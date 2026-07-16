"""Risk-based position sizing helper.

Provides a standalone function to compute trade volume from risk parameters,
independent of the full risk-analyze use case. Ready for use by order-entry
paths that want to derive volume from risk percentage.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from ...shared.market_units import price_delta_ticks
from ...utils.coercion import coerce_finite_float as _finite_float

DEFAULT_KELLY_FRACTION_MULTIPLIER = 0.5
DEFAULT_KELLY_MAX_RISK_PCT = 2.0


def _floor_volume_steps(raw: float, step: float) -> int:
    """Floor to nearest volume step count."""
    if step <= 0 or not math.isfinite(raw):
        return 0
    step_ratio = raw / step
    step_count = math.floor(step_ratio)
    if step_count < 0:
        return 0

    next_step_count = step_count + 1
    next_volume = float(next_step_count) * float(step)
    if next_volume >= raw:
        snap_tolerance = max(
            math.ulp(float(raw)) * 256.0,
            math.ulp(next_volume) * 256.0,
        )
        if next_volume - float(raw) <= snap_tolerance:
            step_count = next_step_count

    return int(step_count)


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


def compute_kelly_sizing_context(
    *,
    win_rate: Any,
    avg_win: Any,
    avg_loss: Any,
    fraction_multiplier: Any = DEFAULT_KELLY_FRACTION_MULTIPLIER,
    max_risk_pct: Any = DEFAULT_KELLY_MAX_RISK_PCT,
    desired_risk_pct: Any = None,
    source: Optional[str] = None,
) -> Tuple[Optional[float], Dict[str, Any]]:
    """Compute the effective risk percent implied by Kelly inputs.

    Returns ``(effective_risk_pct, metadata)``. Non-positive Kelly edge is a
    valid outcome and returns ``0.0`` with ``status="kelly_no_edge"``.
    """
    errors: List[str] = []
    win_rate_f = _finite_float(win_rate)
    avg_win_f = _finite_float(avg_win)
    avg_loss_f = _finite_float(avg_loss)
    multiplier_f = _finite_float(fraction_multiplier)
    max_risk_pct_f = _finite_float(max_risk_pct)
    desired_risk_pct_f = _finite_float(desired_risk_pct)

    if win_rate_f is None or not 0.0 <= win_rate_f <= 1.0:
        errors.append("kelly_win_rate must be finite and between 0 and 1.")
    if avg_win_f is None or avg_win_f <= 0:
        errors.append("kelly_avg_win must be positive and finite.")
    if avg_loss_f is None or avg_loss_f == 0:
        errors.append("kelly_avg_loss must be non-zero and finite.")
    if multiplier_f is None or multiplier_f < 0:
        errors.append("kelly_fraction_multiplier must be non-negative and finite.")
    if max_risk_pct_f is None or max_risk_pct_f <= 0:
        errors.append("kelly_max_risk_pct must be positive and finite.")
    if desired_risk_pct is not None and (
        desired_risk_pct_f is None or desired_risk_pct_f <= 0
    ):
        errors.append("desired_risk_pct must be positive and finite when supplied.")
    if errors:
        return None, {"error": "; ".join(errors)}

    loss_magnitude = abs(float(avg_loss_f))
    if loss_magnitude <= 0:
        return None, {"error": "kelly_avg_loss must be non-zero and finite."}

    odds = float(avg_win_f) / loss_magnitude
    if not math.isfinite(odds) or odds <= 0:
        return None, {"error": "Kelly odds must be positive and finite."}

    probability = float(win_rate_f)
    loss_probability = 1.0 - probability
    kelly_fraction = probability - (loss_probability / odds)
    half_kelly_fraction = kelly_fraction * 0.5
    applied_fraction = kelly_fraction * float(multiplier_f)
    uncapped_risk_pct = max(0.0, applied_fraction * 100.0)
    cap_risk_pct = float(max_risk_pct_f)
    if desired_risk_pct_f is not None:
        cap_risk_pct = min(cap_risk_pct, float(desired_risk_pct_f))
    effective_risk_pct = min(uncapped_risk_pct, cap_risk_pct)

    context: Dict[str, Any] = {
        "win_rate": probability,
        "avg_win_return": float(avg_win_f),
        "avg_loss_return": loss_magnitude,
        "avg_win_loss_ratio": odds,
        "kelly_fraction": kelly_fraction,
        "half_kelly_fraction": half_kelly_fraction,
        "kelly_fraction_multiplier": float(multiplier_f),
        "applied_kelly_fraction": applied_fraction,
        "uncapped_risk_pct": uncapped_risk_pct,
        "cap_risk_pct": cap_risk_pct,
        "effective_risk_pct": effective_risk_pct,
    }
    if source:
        context["source"] = source
    if desired_risk_pct_f is not None:
        context["desired_risk_pct_cap"] = float(desired_risk_pct_f)
    if kelly_fraction <= 0.0 or effective_risk_pct <= 0.0:
        context["status"] = "kelly_no_edge"
        context["effective_risk_pct"] = 0.0
        return 0.0, context
    return effective_risk_pct, context


def compute_risk_based_volume(  # noqa: C901
    *,
    equity: float,
    risk_pct: Optional[float] = None,
    entry_price: float,
    stop_loss_price: float,
    direction: str,
    tick_value: float,
    tick_value_loss: Optional[float] = None,
    tick_size: float,
    volume_min: float,
    volume_max: float,
    volume_step: float,
    strict_risk: bool = True,
    sizing_method: str = "fixed_fraction",
    kelly_win_rate: Optional[float] = None,
    kelly_avg_win: Optional[float] = None,
    kelly_avg_loss: Optional[float] = None,
    kelly_fraction_multiplier: float = DEFAULT_KELLY_FRACTION_MULTIPLIER,
    kelly_max_risk_pct: float = DEFAULT_KELLY_MAX_RISK_PCT,
    kelly_source: Optional[str] = None,
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
    method = str(sizing_method or "fixed_fraction").strip().lower()
    if method not in {"fixed_fraction", "kelly"}:
        errors.append("sizing_method must be 'fixed_fraction' or 'kelly'.")
    risk_pct_value = _finite_float(risk_pct)
    if method == "fixed_fraction" and (
        risk_pct_value is None or risk_pct_value <= 0
    ):
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

    if direction.lower() == "long":
        sl_distance_ticks = price_delta_ticks(entry_price, stop_loss_price, tick_size)
    else:
        sl_distance_ticks = price_delta_ticks(stop_loss_price, entry_price, tick_size)

    if sl_distance_ticks is None or sl_distance_ticks <= 0:
        return None, {
            "error": "Stop-loss distance must be positive (wrong side of entry?).",
            "sl_distance_ticks": sl_distance_ticks,
        }

    kelly_context: Optional[Dict[str, Any]] = None
    if method == "kelly":
        effective_risk_pct, kelly_context = compute_kelly_sizing_context(
            win_rate=kelly_win_rate,
            avg_win=kelly_avg_win,
            avg_loss=kelly_avg_loss,
            fraction_multiplier=kelly_fraction_multiplier,
            max_risk_pct=kelly_max_risk_pct,
            desired_risk_pct=risk_pct,
            source=kelly_source,
        )
        if effective_risk_pct is None:
            return None, kelly_context
        if effective_risk_pct <= 0.0:
            return 0.0, {
                "status": "kelly_no_edge",
                "strict_risk": bool(strict_risk),
                "suggested_volume": 0.0,
                "raw_volume": 0.0,
                "risk_amount": 0.0,
                "actual_risk": 0.0,
                "actual_risk_pct": 0.0,
                "requested_risk_pct": 0.0,
                "risk_pct_diff": 0.0,
                "risk_over_target": False,
                "risk_compliance": "kelly_no_positive_edge",
                "risk_overshoot_pct": 0.0,
                "risk_overshoot_currency": 0.0,
                "risk_over_target_reason": None,
                "sl_distance_ticks": round(sl_distance_ticks, 4),
                "risk_tick_value": round(risk_tick_value, 8),
                "volume_rounding": "kelly_no_edge",
                "notes": [
                    "Kelly sizing produced no positive edge; suggested volume is 0.0."
                ],
                "sizing_method": method,
                "kelly": kelly_context,
            }
        requested_risk_pct = float(effective_risk_pct)
    else:
        requested_risk_pct = float(risk_pct_value)

    risk_amount = equity * (requested_risk_pct / 100.0)

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
    risk_pct_diff = actual_risk_pct - requested_risk_pct
    risk_over_target = actual_risk_pct > (requested_risk_pct + 1e-9)
    overshoot_pct = max(0.0, actual_risk_pct - requested_risk_pct)
    overshoot_currency = max(0.0, actual_risk - risk_amount)
    risk_over_target_reason = None
    if risk_over_target:
        if rounding_mode == "clamped_to_min_volume":
            risk_over_target_reason = "min_volume_constraint"
        elif rounding_mode == "clamped_to_max_volume":
            risk_over_target_reason = "max_volume_constraint"
        elif rounding_mode == "rounded_down_to_step":
            risk_over_target_reason = "step_rounding_precision"
        else:
            risk_over_target_reason = "broker_volume_constraints"
        notes.append(
            "Actual risk still exceeds the requested level after broker volume constraints."
        )

    strict_risk_blocked = bool(
        strict_risk
        and risk_over_target
        and rounding_mode == "clamped_to_min_volume"
    )
    min_viable_volume = None
    min_viable_risk = None
    min_viable_risk_pct = None
    min_viable_overshoot_pct = None
    min_viable_overshoot_currency = None
    if strict_risk_blocked:
        min_viable_volume = suggested
        min_viable_risk = actual_risk
        min_viable_risk_pct = actual_risk_pct
        min_viable_overshoot_pct = overshoot_pct
        min_viable_overshoot_currency = overshoot_currency
        suggested = 0.0
        actual_risk = 0.0
        actual_risk_pct = 0.0
        rounding_mode = "blocked_by_min_volume_risk"
        notes.append(
            "Strict risk is enabled; no broker-accepted volume fits the requested risk."
        )

    risk_compliance = (
        "blocked_min_volume_exceeds_requested_risk"
        if strict_risk_blocked
        else (
            "exceeds_requested_risk"
            if risk_over_target
            else "within_requested_risk"
        )
    )
    meta: Dict[str, Any] = {
        **({"status": "blocked"} if strict_risk_blocked else {}),
        "strict_risk": bool(strict_risk),
        "suggested_volume": suggested,
        "raw_volume": round(raw_volume, 8),
        "risk_amount": round(risk_amount, 2),
        "actual_risk": round(actual_risk, 2),
        "actual_risk_pct": round(actual_risk_pct, 4),
        "requested_risk_pct": requested_risk_pct,
        "risk_pct_diff": round(risk_pct_diff, 4),
        "risk_over_target": risk_over_target,
        "risk_compliance": risk_compliance,
        "risk_overshoot_pct": round(overshoot_pct, 4),
        "risk_overshoot_currency": round(overshoot_currency, 2),
        "risk_over_target_reason": risk_over_target_reason,
        "sl_distance_ticks": round(sl_distance_ticks, 4),
        "risk_tick_value": round(risk_tick_value, 8),
        "volume_rounding": rounding_mode,
        "notes": notes,
    }
    if method == "kelly":
        meta["sizing_method"] = method
        meta["kelly"] = kelly_context
    if strict_risk_blocked:
        meta.update(
            {
                "min_viable_volume": min_viable_volume,
                "min_viable_risk": round(float(min_viable_risk or 0.0), 2),
                "min_viable_risk_pct": round(
                    float(min_viable_risk_pct or 0.0), 4
                ),
                "min_viable_risk_overshoot_pct": round(
                    float(min_viable_overshoot_pct or 0.0), 4
                ),
                "min_viable_risk_overshoot_currency": round(
                    float(min_viable_overshoot_currency or 0.0), 2
                ),
            }
        )
    return suggested, meta
