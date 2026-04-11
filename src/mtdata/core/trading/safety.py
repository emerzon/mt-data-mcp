"""Pre-trade safety rails.

Provides a configurable policy that gates order requests before they reach
the broker.  Every rule is opt-in: when no policy is supplied (or all fields
are ``None``), existing behavior is fully preserved.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class TradeSafetyPolicy(BaseModel):
    """Configurable pre-trade safety rails.

    Each field is ``None`` by default, meaning the check is disabled.
    """

    max_volume: Optional[float] = None
    """Reject orders whose volume exceeds this cap."""

    require_stop_loss: bool = False
    """Reject market orders that have no stop-loss."""

    max_deviation: Optional[int] = None
    """Cap the deviation/slippage value sent to the broker."""

    reduce_only: bool = False
    """Only allow close-direction orders (no new net exposure)."""


def _evaluate_safety_policy(
    policy: Optional[TradeSafetyPolicy],
    *,
    volume: Optional[float] = None,
    stop_loss: Optional[float] = None,
    deviation: Optional[int] = None,
    side: Optional[str] = None,
    existing_side: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Evaluate *policy* against the given order parameters.

    Returns ``None`` when the request passes all active rails.  Returns a
    structured error dict when a rail blocks the request.
    """
    if policy is None:
        return None

    violations: List[str] = []

    if policy.max_volume is not None and volume is not None:
        try:
            if math.isfinite(volume) and volume > policy.max_volume:
                violations.append(
                    f"Volume {volume} exceeds safety limit of {policy.max_volume}."
                )
        except (TypeError, ValueError):
            pass

    if policy.require_stop_loss:
        if stop_loss is None or not math.isfinite(stop_loss):
            violations.append("Safety policy requires a stop-loss on every order.")

    if policy.max_deviation is not None and deviation is not None:
        try:
            if int(deviation) > policy.max_deviation:
                violations.append(
                    f"Deviation {deviation} exceeds safety limit of {policy.max_deviation}."
                )
        except (TypeError, ValueError):
            pass

    if policy.reduce_only and side is not None:
        _OPPOSITE = {"BUY": "SELL", "SELL": "BUY"}
        if existing_side is None:
            violations.append(
                "Reduce-only policy: no existing position to reduce."
            )
        elif side.upper() != _OPPOSITE.get(existing_side.upper(), ""):
            violations.append(
                f"Reduce-only policy: order side {side} does not reduce "
                f"existing {existing_side} position."
            )

    if not violations:
        return None

    return {
        "error": "Order blocked by safety policy.",
        "violations": violations,
    }


# ---------------------------------------------------------------------------
# Account-level risk gate
# ---------------------------------------------------------------------------


class AccountRiskLimits(BaseModel):
    """Configurable account-level risk thresholds.

    All fields are ``None`` (disabled) by default.
    """

    min_margin_level_pct: Optional[float] = None
    """Reject if account margin level (%) is below this threshold."""

    max_floating_loss: Optional[float] = None
    """Reject if unrealized loss exceeds this absolute value (positive number)."""

    max_total_exposure_lots: Optional[float] = None
    """Reject if total open volume (lots) would exceed this after the new order."""


def _evaluate_account_risk_gate(
    limits: Optional[AccountRiskLimits],
    *,
    account_info: Any = None,
    new_volume: float = 0.0,
    existing_volume: float = 0.0,
) -> Optional[Dict[str, Any]]:
    """Evaluate *limits* against current account state.

    Returns ``None`` when the request passes all gates.  Returns a
    structured error dict when a gate blocks the request.
    """
    if limits is None:
        return None

    violations: List[str] = []

    if limits.min_margin_level_pct is not None and account_info is not None:
        margin_level = _safe_float_attr(account_info, "margin_level")
        if margin_level is not None and margin_level < limits.min_margin_level_pct:
            violations.append(
                f"Margin level {margin_level:.1f}% is below the "
                f"minimum threshold of {limits.min_margin_level_pct:.1f}%."
            )

    if limits.max_floating_loss is not None and account_info is not None:
        profit = _safe_float_attr(account_info, "profit")
        if profit is not None and profit < 0 and abs(profit) > limits.max_floating_loss:
            violations.append(
                f"Floating loss ${abs(profit):.2f} exceeds the "
                f"limit of ${limits.max_floating_loss:.2f}."
            )

    if limits.max_total_exposure_lots is not None:
        total_after = existing_volume + new_volume
        if total_after > limits.max_total_exposure_lots:
            violations.append(
                f"Total exposure {total_after:.2f} lots would exceed the "
                f"limit of {limits.max_total_exposure_lots:.2f} lots."
            )

    if not violations:
        return None

    return {
        "error": "Order blocked by account risk gate.",
        "violations": violations,
    }


def _safe_float_attr(obj: Any, name: str) -> Optional[float]:
    """Extract a float attribute safely, returning None on failure."""
    try:
        val = getattr(obj, name, None)
        if val is None:
            return None
        fv = float(val)
        return fv if math.isfinite(fv) else None
    except (TypeError, ValueError):
        return None
