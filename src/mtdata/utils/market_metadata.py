from __future__ import annotations

import math
from typing import Any, Callable, Dict

from .freshness import (
    MAX_STANDARD_WEEKEND_DATA_AGE_SECONDS,
    QUOTE_STALE_SECONDS,
    closed_session_context,
    format_freshness_label,
)

TICK_VOLUME_SEMANTICS = "tick_volume_is_broker_tick_count_not_lots"

FRESHNESS_ANCHOR_QUERY_EXPECTED_END = "query_expected_end"
FRESHNESS_ANCHOR_WALL_CLOCK = "wall_clock"

FRESHNESS_METRIC_LAST_COMPLETED_BAR_AGE = "last_completed_bar_age_seconds"
FRESHNESS_METRIC_LAST_TICK_AGE = "last_tick_age_seconds"
FRESHNESS_METRIC_REQUESTED_RANGE_END_GAP = "requested_range_end_gap_seconds"


def attach_candle_volume_semantics(payload: Dict[str, Any]) -> None:
    volume_type = str(payload.get("volume_type") or "").strip().lower()
    volume_unit = str(payload.get("volume_unit") or "").strip().lower()
    if volume_type == "tick_count" or volume_unit in {
        "broker_tick_count",
        "mt5_tick_volume",
    }:
        payload["volume_semantics"] = TICK_VOLUME_SEMANTICS


def normalize_policy_relaxed(value: Any) -> bool:
    try:
        return bool(value)
    except Exception:
        return False


def tick_freshness_state(
    *,
    data_stale: Any,
    market_status: Any = None,
    market_status_reason: Any = None,
    freshness_policy_relaxed: Any = None,
) -> str:
    if normalize_policy_relaxed(freshness_policy_relaxed):
        status = str(market_status or "closed_or_idle").strip().lower()
        reason = str(market_status_reason or "").strip().lower()
        parts = [status.replace(" ", "_")] if status else ["closed_or_idle"]
        if reason:
            parts.append(reason.replace(" ", "_"))
        parts.append("snapshot")
        return "_".join(part for part in parts if part)
    return "stale" if bool(data_stale) else "fresh"


def build_tick_freshness_context(
    symbol: Any,
    *,
    tick_epoch: Any,
    now_epoch: Any,
    item: str = "tick",
    stale_after_seconds: Any = QUOTE_STALE_SECONDS,
    age_rounder: Callable[[float], Any] | None = None,
) -> Dict[str, Any]:
    try:
        current_epoch = float(now_epoch)
        latest_tick_epoch = float(tick_epoch)
    except Exception:
        return {}
    if not math.isfinite(current_epoch) or not math.isfinite(latest_tick_epoch):
        return {}

    age_seconds = max(0.0, current_epoch - latest_tick_epoch)
    try:
        threshold = max(0.0, float(stale_after_seconds))
    except Exception:
        threshold = float(QUOTE_STALE_SECONDS)

    closed_session = closed_session_context(
        symbol,
        now_epoch=current_epoch,
        item=item,
        data_age_seconds=age_seconds,
    )
    data_stale = age_seconds > threshold
    if closed_session and closed_session.get("freshness_policy_relaxed"):
        data_stale = False

    rounded_age = age_rounder(age_seconds) if age_rounder is not None else round(age_seconds, 3)
    stale_after: int | float
    if float(threshold).is_integer():
        stale_after = int(threshold)
    else:
        stale_after = threshold

    out: Dict[str, Any] = {
        "data_age_seconds": rounded_age,
        "data_age_anchor": FRESHNESS_ANCHOR_WALL_CLOCK,
        "data_age_metric": FRESHNESS_METRIC_LAST_TICK_AGE,
        "stale_after_seconds": stale_after,
        "data_stale": data_stale,
    }
    if closed_session:
        out.update(closed_session)
    out["freshness_basis"] = (
        f"weekend_relaxed_max_{int(MAX_STANDARD_WEEKEND_DATA_AGE_SECONDS // 86400)}d"
        if normalize_policy_relaxed(out.get("freshness_policy_relaxed"))
        else f"absolute_{stale_after}s"
    )

    freshness = format_freshness_label(
        data_stale=data_stale,
        market_status=out.get("market_status"),
        market_status_reason=out.get("market_status_reason"),
        age_seconds=age_seconds,
        item=item,
    )
    if freshness:
        out["freshness"] = freshness
    out["freshness_state"] = tick_freshness_state(
        data_stale=data_stale,
        market_status=out.get("market_status"),
        market_status_reason=out.get("market_status_reason"),
        freshness_policy_relaxed=out.get("freshness_policy_relaxed"),
    )
    return out
