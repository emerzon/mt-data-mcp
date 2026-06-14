from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Literal, Optional

from ..services.data_service import fetch_candles, fetch_ticks
from ..shared.constants import TIMEFRAME_SECONDS
from ..shared.schema import CompactStandardFullDetailLiteral, TimeframeLiteral
from ..utils.mt5 import (
    MT5ConnectionError,
    _symbol_ready_guard,
    ensure_mt5_connection_or_raise,
)
from ..utils.utils import _parse_start_datetime, _positive_float_attr
from ..utils.volume_profile import VolumeProfileConfig, compute_volume_profile
from ._mcp_instance import mcp
from .execution_logging import run_logged_operation
from .mt5_gateway import create_mt5_gateway
from .output_contract import normalize_output_extras

logger = logging.getLogger(__name__)

VolumeProfileSourceLiteral = Literal["auto", "ticks", "m1_bars"]
VolumeProfilePriceSourceLiteral = Literal["mid", "last", "bid", "ask"]
VolumeProfileVolumeSourceLiteral = Literal["auto", "real_volume", "tick_volume", "tick_count"]

_DEFAULT_MAX_TICK_WINDOW_DAYS = 7
_DEFAULT_MAX_TICKS = 200_000
_DEFAULT_MAX_M1_BARS = 20_000
_DEFAULT_PROFILE_LIMIT = 200
_MIN_TICK_PRICE_COVERAGE_RATIO = 0.5


def _utc_now_naive() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _positive_int_attr(obj: Any, *names: str) -> Optional[int]:
    value = _positive_float_attr(obj, *names)
    if value is None:
        return None
    return int(value)


def _window_days(start: Optional[str], end: Optional[str]) -> Optional[float]:
    start_dt = _parse_start_datetime(start) if start else None
    end_dt = _parse_start_datetime(end) if end else None
    if start and start_dt is None:
        return None
    if end and end_dt is None:
        return None
    if start_dt is None and end_dt is None:
        return None
    if end_dt is None:
        end_dt = _utc_now_naive()
    if start_dt is None:
        return None
    seconds = max(0.0, float((end_dt - start_dt).total_seconds()))
    return seconds / 86400.0


def _resolve_profile_window(
    *,
    start: Optional[str],
    end: Optional[str],
    timeframe: Optional[str],
    limit: Optional[int],
) -> Dict[str, Optional[str]]:
    if start:
        return {"start": start, "end": end}
    if timeframe is None and limit is None:
        return {"start": start, "end": end}
    if not timeframe:
        return {"error": "timeframe is required when limit is provided"}
    tf = str(timeframe).strip().upper()
    seconds = TIMEFRAME_SECONDS.get(tf)
    if seconds is None:
        return {"error": f"Invalid timeframe {timeframe!r}"}
    if limit is None:
        bars = _DEFAULT_PROFILE_LIMIT
    else:
        try:
            bars = int(limit)
        except (TypeError, ValueError):
            bars = 0
        if bars <= 0:
            return {
                "error": (
                    "limit must be a positive integer when timeframe is provided; "
                    f"omit limit to use the default {int(_DEFAULT_PROFILE_LIMIT)} bars."
                )
            }
    end_dt = _parse_start_datetime(end) if end else _utc_now_naive()
    if end and end_dt is None:
        return {"error": f"Could not parse end datetime {end!r}"}
    assert end_dt is not None
    start_dt = end_dt - timedelta(seconds=int(seconds) * bars)
    return {
        "start": start_dt.isoformat(sep=" ", timespec="seconds"),
        "end": end if end else end_dt.isoformat(sep=" ", timespec="seconds"),
    }


def _table_rows(payload: Dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("data")
    return rows if isinstance(rows, list) else []


def _observed_profile_window(
    rows: list[dict[str, Any]],
    *,
    fallback_start: Optional[str],
    fallback_end: Optional[str],
) -> Dict[str, Optional[str]]:
    times = [row.get("time") for row in rows if row.get("time") not in (None, "")]
    if not times:
        return {"start": fallback_start, "end": fallback_end}
    return {"start": str(times[0]), "end": str(times[-1])}


def _fetch_tick_rows(
    *,
    symbol: str,
    start: Optional[str],
    end: Optional[str],
    max_ticks: int,
) -> Dict[str, Any]:
    payload = fetch_ticks(
        symbol=symbol,
        limit=max(1, int(max_ticks)),
        start=start,
        end=end,
        format="full_rows",
    )
    if payload.get("error"):
        return payload
    rows = _table_rows(payload)
    return {
        "success": True,
        "source": "ticks",
        "rows": rows,
        "fetch_payload": payload,
        "diagnostics": {
            "tick_rows": int(len(rows)),
            "requested_max_ticks": int(max_ticks),
        },
    }


def _fetch_m1_rows(
    *,
    symbol: str,
    start: Optional[str],
    end: Optional[str],
    max_m1_bars: int,
) -> Dict[str, Any]:
    payload = fetch_candles(
        symbol=symbol,
        timeframe="M1",
        limit=max(1, int(max_m1_bars)),
        start=start,
        end=end,
        ohlcv="OHLCV",
        include_incomplete=False,
    )
    if payload.get("error"):
        return payload
    candles = _table_rows(payload)
    rows = []
    for candle in candles:
        if not isinstance(candle, dict):
            continue
        volume = candle.get("real_volume")
        try:
            real_volume = float(volume)
        except (TypeError, ValueError):
            real_volume = 0.0
        if real_volume <= 0.0:
            volume = candle.get("tick_volume")
        try:
            weight = float(volume)
        except (TypeError, ValueError):
            weight = 0.0
        prices = []
        for key in ("low", "close", "high"):
            try:
                value = float(candle.get(key))
            except (TypeError, ValueError):
                continue
            if math.isfinite(value) and value > 0.0:
                prices.append(value)
        if not prices:
            continue
        per_price_weight = weight / float(len(prices)) if weight > 0.0 else 0.0
        for price in prices:
            rows.append(
                {
                    "last": price,
                    "mid": price,
                    "bid": price,
                    "ask": price,
                    "tick_volume": per_price_weight,
                    "real_volume": real_volume / float(len(prices)) if real_volume > 0.0 else 0.0,
                }
            )
    return {
        "success": True,
        "source": "m1_bars",
        "rows": rows,
        "fetch_payload": payload,
        "diagnostics": {
            "m1_bars": int(len(candles)),
            "profile_rows": int(len(rows)),
            "requested_max_m1_bars": int(max_m1_bars),
            "approximation": "M1 bar volume split across low/close/high prices.",
        },
        "warnings": [
            "Volume profile used M1-bar approximation instead of raw ticks; intrabar volume location is estimated."
        ],
    }


def _row_finite_positive(row: Any, key: str) -> Optional[float]:
    if isinstance(row, dict):
        value = row.get(key)
    else:
        value = getattr(row, key, None)
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isfinite(numeric) and numeric > 0.0:
        return numeric
    return None


def _tick_price_quality(rows: list[dict[str, Any]], price_source: str) -> Dict[str, Any]:
    source = str(price_source or "mid").strip().lower()
    total = len(rows)
    valid = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        price = _row_finite_positive(row, source)
        if source == "mid" and price is None:
            bid = _row_finite_positive(row, "bid")
            ask = _row_finite_positive(row, "ask")
            if bid is not None and ask is not None:
                price = (bid + ask) / 2.0
        if price is not None:
            valid += 1
    ratio = (valid / total) if total else 0.0
    return {
        "price_source": source,
        "input_rows": int(total),
        "valid_price_rows": int(valid),
        "dropped_price_rows": int(max(0, total - valid)),
        "valid_price_ratio": round(ratio, 4),
    }


def _should_fallback_from_tick_prices(quality: Dict[str, Any]) -> bool:
    input_rows = int(quality.get("input_rows") or 0)
    if input_rows <= 0:
        return True
    valid_rows = int(quality.get("valid_price_rows") or 0)
    if valid_rows <= 0:
        return True
    ratio = float(quality.get("valid_price_ratio") or 0.0)
    return ratio < _MIN_TICK_PRICE_COVERAGE_RATIO


def _select_profile_rows(
    *,
    symbol: str,
    start: Optional[str],
    end: Optional[str],
    source: str,
    price_source: str,
    max_tick_window_days: int,
    max_ticks: int,
    max_m1_bars: int,
) -> Dict[str, Any]:
    source_value = str(source or "auto").strip().lower()
    if source_value not in {"auto", "ticks", "m1_bars"}:
        return {"error": "source must be one of: auto, ticks, m1_bars"}
    days = _window_days(start, end)
    use_m1 = source_value == "m1_bars"
    if source_value == "auto" and days is not None and days > float(max_tick_window_days):
        use_m1 = True

    if use_m1:
        selected = _fetch_m1_rows(
            symbol=symbol,
            start=start,
            end=end,
            max_m1_bars=max_m1_bars,
        )
        if days is not None and days > float(max_tick_window_days):
            diagnostics = selected.setdefault("diagnostics", {})
            if isinstance(diagnostics, dict):
                diagnostics["tick_window_days"] = round(days, 4)
                diagnostics["max_tick_window_days"] = int(max_tick_window_days)
                diagnostics["auto_fallback_reason"] = "requested window exceeds bounded tick window"
        return selected

    tick_result = _fetch_tick_rows(
        symbol=symbol,
        start=start,
        end=end,
        max_ticks=max_ticks,
    )
    if not tick_result.get("error"):
        rows = tick_result.get("rows")
        if not isinstance(rows, list):
            rows = []
        quality = _tick_price_quality(rows, price_source)
        diagnostics = tick_result.setdefault("diagnostics", {})
        if isinstance(diagnostics, dict):
            diagnostics["tick_price_quality"] = quality
        if source_value == "auto" and _should_fallback_from_tick_prices(quality):
            fallback = _fetch_m1_rows(
                symbol=symbol,
                start=start,
                end=end,
                max_m1_bars=max_m1_bars,
            )
            fallback_diagnostics = fallback.setdefault("diagnostics", {})
            if isinstance(fallback_diagnostics, dict):
                fallback_diagnostics["auto_fallback_reason"] = (
                    "tick price coverage below threshold"
                )
                fallback_diagnostics["tick_price_quality"] = quality
                fallback_diagnostics["min_tick_price_coverage_ratio"] = (
                    _MIN_TICK_PRICE_COVERAGE_RATIO
                )
            return fallback
        if isinstance(diagnostics, dict) and source_value == "auto":
            diagnostics["auto_source_reason"] = (
                "tick data within bounded window with adequate price coverage"
            )
        return tick_result
    if source_value == "ticks":
        return tick_result
    fallback = _fetch_m1_rows(
        symbol=symbol,
        start=start,
        end=end,
        max_m1_bars=max_m1_bars,
    )
    diagnostics = fallback.setdefault("diagnostics", {})
    if isinstance(diagnostics, dict):
        diagnostics["auto_fallback_reason"] = "tick fetch failed"
        diagnostics["tick_error"] = tick_result.get("error")
    return fallback


def _profile_detail_payload(profile: Dict[str, Any], detail: str) -> Dict[str, Any]:
    detail_value = str(detail or "compact").strip().lower()
    if detail_value in {"summary", "summary_only"}:
        detail_value = "compact"
    if detail_value not in {"compact", "standard", "full"}:
        detail_value = "compact"
    keys = [
        "success",
        "symbol",
        "source",
        "window",
        "price_source",
        "volume_kind",
        "bucket_size",
        "value_area_pct",
        "price_point",
        "price_digits",
        "total_volume",
        "poc",
        "vah",
        "val",
        "levels",
        "value_area",
        "diagnostics",
        "warnings",
        "as_of",
        "timezone",
        "data_age_seconds",
        "data_stale",
        "units",
    ]
    out = {key: profile[key] for key in keys if key in profile}
    value_area = out.get("value_area")
    if isinstance(value_area, dict):
        compact_value_area = dict(value_area)
        bucket_indexes = compact_value_area.get("bucket_indexes")
        if isinstance(bucket_indexes, list):
            compact_value_area["bucket_count"] = len(bucket_indexes)
        if detail_value != "full":
            compact_value_area.pop("bucket_indexes", None)
        out["value_area"] = compact_value_area
    if detail_value == "compact":
        out.pop("levels", None)
        out.pop("units", None)
        window = out.get("window")
        if isinstance(window, dict) and not any(
            value not in (None, "") for value in window.values()
        ):
            out.pop("window", None)
    else:
        out["detail"] = detail_value
    if detail_value == "standard":
        out["buckets"] = profile.get("buckets", [])[:50]
        out["bucket_note"] = "First 50 buckets returned; use detail='full' for all buckets."
    elif detail_value == "full":
        out["buckets"] = profile.get("buckets", [])
        fetch_payload = profile.get("fetch_payload")
        if isinstance(fetch_payload, dict):
            out["fetch_meta"] = {
                key: fetch_payload.get(key)
                for key in ("count", "start", "end", "timezone", "price_precision", "price_currency")
                if key in fetch_payload
            }
    return out


def _profile_freshness_meta(fetch_payload: Any) -> Dict[str, Any]:
    if not isinstance(fetch_payload, dict):
        return {}
    out: Dict[str, Any] = {}
    for target, source_names in (
        ("as_of", ("as_of", "data_fetched_at")),
        ("timezone", ("timezone",)),
        ("data_age_seconds", ("data_age_seconds", "data_freshness_seconds")),
        ("data_stale", ("data_stale",)),
    ):
        for name in source_names:
            value = fetch_payload.get(name)
            if value not in (None, "", [], {}):
                out[target] = value
                break
    if "data_age_seconds" not in out:
        meta = fetch_payload.get("meta")
        diagnostics = meta.get("diagnostics") if isinstance(meta, dict) else None
        freshness = diagnostics.get("freshness") if isinstance(diagnostics, dict) else None
        if isinstance(freshness, dict):
            age_seconds = freshness.get("data_freshness_seconds")
            if age_seconds is not None:
                out["data_age_seconds"] = age_seconds
            within_policy = freshness.get("last_bar_within_policy_window")
            relaxed_policy = bool(freshness.get("freshness_policy_relaxed"))
            if within_policy is not None:
                out["data_stale"] = bool(not bool(within_policy) and not relaxed_policy)
    return out


def _profile_units(profile: Dict[str, Any]) -> Dict[str, str]:
    volume_kind = str(profile.get("volume_kind") or "volume_weight").strip() or "volume_weight"
    return {
        "price": "absolute_price",
        "bucket_size": "absolute_price",
        "poc.price": "absolute_price",
        "vah.price": "absolute_price",
        "val.price": "absolute_price",
        "volume": volume_kind,
        "total_volume": volume_kind,
        "value_area.volume": volume_kind,
    }


def compute_volume_profile_payload(
    *,
    symbol: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    timeframe: Optional[TimeframeLiteral] = None,
    limit: Optional[int] = None,
    source: VolumeProfileSourceLiteral = "auto",
    price_source: VolumeProfilePriceSourceLiteral = "mid",
    volume_source: VolumeProfileVolumeSourceLiteral = "auto",
    bucket_size: Optional[float] = None,
    bucket_points: Optional[float] = None,
    bucket_count: Optional[int] = None,
    max_buckets: int = 120,
    value_area_pct: float = 0.70,
    reference_price: Optional[float] = None,
    max_tick_window_days: int = _DEFAULT_MAX_TICK_WINDOW_DAYS,
    max_ticks: int = _DEFAULT_MAX_TICKS,
    max_m1_bars: int = _DEFAULT_MAX_M1_BARS,
    detail: CompactStandardFullDetailLiteral = "compact",
) -> Dict[str, Any]:
    if limit is not None and not timeframe:
        return {
            "error": (
                "limit is a bar count and requires timeframe; "
                "use max_ticks to cap tick rows."
            )
        }
    window = _resolve_profile_window(
        start=start,
        end=end,
        timeframe=timeframe,
        limit=limit,
    )
    if window.get("error"):
        return {"error": window["error"]}
    resolved_start = window.get("start")
    resolved_end = window.get("end")
    create_mt5_gateway(ensure_connection_impl=ensure_mt5_connection_or_raise).ensure_connection()
    with _symbol_ready_guard(symbol) as (err, info):
        if err:
            return {"error": err}
        price_digits = _positive_int_attr(info, "digits")
        price_point = _positive_float_attr(info, "point", "trade_tick_size")
    selected = _select_profile_rows(
        symbol=symbol,
        start=resolved_start,
        end=resolved_end,
        source=source,
        price_source=price_source,
        max_tick_window_days=max_tick_window_days,
        max_ticks=max_ticks,
        max_m1_bars=max_m1_bars,
    )
    if selected.get("error"):
        return selected
    config = VolumeProfileConfig(
        price_source=price_source,
        volume_source=volume_source,
        bucket_size=bucket_size,
        bucket_points=bucket_points,
        bucket_count=bucket_count,
        max_buckets=max_buckets,
        value_area_pct=value_area_pct,
        price_point=price_point,
        price_digits=price_digits,
        reference_price=reference_price,
    )
    profile = compute_volume_profile(selected.get("rows", []), config)
    if profile.get("error"):
        profile["symbol"] = symbol
        profile["source"] = selected.get("source")
        profile["price_point"] = price_point
        profile["price_digits"] = price_digits
        profile["diagnostics"] = {
            **(selected.get("diagnostics") or {}),
            **(profile.get("diagnostics") or {}),
        }
        return profile
    profile["symbol"] = symbol
    profile["source"] = selected.get("source")
    profile["window"] = _observed_profile_window(
        selected.get("rows", []),
        fallback_start=resolved_start,
        fallback_end=resolved_end,
    )
    profile["diagnostics"] = {
        **(selected.get("diagnostics") or {}),
        **(profile.get("diagnostics") or {}),
    }
    fetch_payload = selected.get("fetch_payload")
    profile.update(_profile_freshness_meta(fetch_payload))
    profile["units"] = _profile_units(profile)
    if selected.get("warnings"):
        profile["warnings"] = list(selected.get("warnings") or [])
    profile["fetch_payload"] = fetch_payload
    return _profile_detail_payload(profile, detail)


@mcp.tool()
def volume_profile_levels(  # noqa: PLR0913
    symbol: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    timeframe: Optional[TimeframeLiteral] = None,
    limit: Optional[int] = None,
    source: VolumeProfileSourceLiteral = "auto",
    price_source: VolumeProfilePriceSourceLiteral = "mid",
    volume_source: VolumeProfileVolumeSourceLiteral = "auto",
    bucket_size: Optional[float] = None,
    bucket_points: Optional[float] = None,
    bucket_count: Optional[int] = None,
    max_buckets: int = 120,
    value_area_pct: float = 0.70,
    reference_price: Optional[float] = None,
    max_tick_window_days: int = _DEFAULT_MAX_TICK_WINDOW_DAYS,
    max_ticks: int = _DEFAULT_MAX_TICKS,
    max_m1_bars: int = _DEFAULT_MAX_M1_BARS,
    detail: CompactStandardFullDetailLiteral = "compact",
    extras: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute volume-profile POC, VAH, and VAL from ticks or M1-bar approximation.

    `source="auto"` uses bounded raw ticks for short windows and falls back to
    M1-bar approximation for larger windows. `limit` is always a bar count and
    requires `timeframe`; use `max_ticks` to cap tick rows. When `timeframe` is
    provided without `limit`, the window defaults to 200 bars. `price_source="mid"`
    is the safe default for FX symbols where tick `last` is often unavailable.
    """

    def _run() -> Dict[str, Any]:
        try:
            detail_value = str(detail or "compact").strip().lower()
            if normalize_output_extras(extras):
                detail_value = "full"
            return compute_volume_profile_payload(
                symbol=symbol,
                start=start,
                end=end,
                timeframe=timeframe,
                limit=limit,
                source=source,
                price_source=price_source,
                volume_source=volume_source,
                bucket_size=bucket_size,
                bucket_points=bucket_points,
                bucket_count=bucket_count,
                max_buckets=max_buckets,
                value_area_pct=value_area_pct,
                reference_price=reference_price,
                max_tick_window_days=max_tick_window_days,
                max_ticks=max_ticks,
                max_m1_bars=max_m1_bars,
                detail=detail_value,  # type: ignore[arg-type]
            )
        except MT5ConnectionError as exc:
            return {"error": str(exc)}
        except Exception as exc:
            return {"error": f"Error computing volume profile levels: {str(exc)}"}

    return run_logged_operation(
        logger,
        operation="volume_profile_levels",
        symbol=symbol,
        start=start,
        end=end,
        timeframe=timeframe,
        limit=limit,
        source=source,
        price_source=price_source,
        volume_source=volume_source,
        bucket_size=bucket_size,
        bucket_points=bucket_points,
        bucket_count=bucket_count,
        max_buckets=max_buckets,
        value_area_pct=value_area_pct,
        max_tick_window_days=max_tick_window_days,
        max_ticks=max_ticks,
        max_m1_bars=max_m1_bars,
        detail=detail,
        extras=extras,
        func=_run,
    )
