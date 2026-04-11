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
