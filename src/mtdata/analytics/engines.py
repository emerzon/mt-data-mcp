"""Statistical engines for the advanced MT5-native analytics tools."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import dateparser
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, skew

from ..core.analytics_requests import (
    MarketMicrostructureRequest,
    MarketRelativeStrengthRequest,
    PortfolioRiskDecomposeRequest,
    StrategyCandidate,
    StrategyValidateRequest,
    TradeExecutionQualityRequest,
)
from ..shared.constants import TIMEFRAME_MAP, TIMEFRAME_SECONDS
from ..shared.market_units import forex_points_per_pip
from ..shared.symbols import (
    is_probably_crypto_symbol,
    is_probably_fx_session_symbol,
)
from ..utils.barriers import normalize_same_bar_policy
from ..utils.freshness import (
    closed_session_context,
    format_age_seconds,
    standard_weekend_window,
)
from ..utils.market_metadata import build_tick_freshness_context
from ..utils.quote import (
    compute_spread_metrics,
    enforce_quote_execution_readiness,
    resolve_quote_tick,
    tick_epoch,
    tick_value,
)
from ..utils.sessions import market_session_label, session_definition_for_clock
from ..utils.tick_flags import mt5_trade_event_mask
from ..utils.time import bar_close_epoch, format_datetime_utc, format_epoch_utc
from ..utils.utils import validate_historical_range


def _mapping(row: Any) -> Dict[str, Any]:
    if isinstance(row, dict):
        return dict(row)
    converter = getattr(row, "_asdict", None)
    if callable(converter):
        return dict(converter())
    return {name: getattr(row, name) for name in dir(row) if not name.startswith("_") and not callable(getattr(row, name, None))}


def _classify_trade_sides(
    trades: pd.DataFrame, prevailing_mid: pd.Series
) -> pd.Series:
    """Apply prevailing-quote classification, then the full-series tick rule."""
    sides = np.sign(trades["last"] - prevailing_mid.loc[trades.index])
    tick_sides = (
        np.sign(trades["last"].diff())
        .replace(0.0, np.nan)
        .ffill()
        .fillna(0.0)
    )
    zero = sides == 0
    sides.loc[zero] = tick_sides.loc[zero]
    return sides


def _portfolio_mark_context(gateway: Any, positions: List[Dict[str, Any]]) -> Dict[str, Any]:
    contexts: List[Dict[str, Any]] = []
    valid_times: List[float] = []
    symbol_counts: Dict[str, int] = {}
    for row in positions:
        symbol = str(row.get("symbol") or "").strip()
        if not symbol:
            continue
        symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1
    for symbol, position_count in symbol_counts.items():
        try:
            raw_tick = gateway.symbol_info_tick(symbol)
        except Exception:
            raw_tick = None
        query_epoch = datetime.now(timezone.utc).timestamp()
        tick, quote_source = resolve_quote_tick(
            gateway,
            symbol,
            raw_tick,
            now_epoch=query_epoch,
        )
        quote_epoch = tick_epoch(tick)
        observed_epoch = datetime.now(timezone.utc).timestamp()
        freshness = build_tick_freshness_context(
            symbol,
            tick_epoch=quote_epoch,
            now_epoch=observed_epoch,
        )
        if quote_epoch is not None:
            try:
                valid_times.append(float(quote_epoch))
            except (TypeError, ValueError):
                pass
        if not freshness:
            freshness = {
                "data_stale": None,
                "usable_for_live_trading": False,
                "freshness_state": "unknown",
                "freshness_reason": "missing_tick_timestamp",
            }
        freshness["symbol"] = symbol
        freshness["positions"] = position_count
        freshness["quote_time"] = format_epoch_utc(quote_epoch)
        freshness.update(quote_source)
        try:
            bid = float(tick_value(tick, "bid") or 0.0)
            ask = float(tick_value(tick, "ask") or 0.0)
        except (TypeError, ValueError):
            bid = ask = 0.0
        if not (ask > bid > 0.0):
            freshness["usable_for_live_trading"] = False
            freshness["freshness_state"] = "unusable_quote"
            freshness["freshness_reason"] = (
                "locked_quote" if ask == bid and bid > 0.0 else "invalid_quote"
            )
            freshness["quote_warning"] = (
                "A positive bid/ask spread is required for a live portfolio mark."
            )
        contexts.append(freshness)
    if not contexts:
        return {
            "valuation_time": None,
            "valuation_basis": "no_position_marks",
            "data_stale": None,
            "mark_freshness_status": "not_applicable",
            "mark_freshness": [],
        }
    live_ready = bool(contexts) and all(
        item.get("usable_for_live_trading") is True for item in contexts
    )
    stale_values = [item.get("data_stale") for item in contexts]
    data_stale = (
        True
        if any(value is True for value in stale_values)
        else False
        if all(value is False for value in stale_values)
        else None
    )
    return {
        "valuation_time": format_epoch_utc(min(valid_times)) if valid_times else None,
        "valuation_basis": (
            "live_position_marks_with_completed_bar_return_history"
            if live_ready
            else "stale_or_unverified_position_marks_with_completed_bar_return_history"
        ),
        "data_stale": data_stale,
        "usable_for_live_trading": live_ready,
        "mark_freshness": contexts,
    }


def _filtered_historical_returns(
    returns: pd.DataFrame,
    *,
    alpha: float,
) -> tuple[pd.DataFrame, pd.Series]:
    """Standardize each return by volatility known before that return."""
    ewma_std = returns.ewm(alpha=alpha, adjust=False).std()
    current_vol = ewma_std.iloc[-1].replace(0, np.nan)
    conditional_vol = ewma_std.shift(1).replace(0, np.nan)
    standardized = (
        returns.div(conditional_vol)
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    return standardized, current_vol


def _frame(rows: Any) -> pd.DataFrame:
    if rows is None:
        return pd.DataFrame()
    if isinstance(rows, pd.DataFrame):
        return rows.copy()
    if isinstance(rows, np.ndarray) and rows.dtype.names:
        return pd.DataFrame(rows)
    return pd.DataFrame([_mapping(row) for row in list(rows)])


def _parse_time(value: Optional[str], default: datetime) -> datetime:
    if not value:
        return default
    parsed = dateparser.parse(str(value), settings={"TIMEZONE": "UTC", "RETURN_AS_TIMEZONE_AWARE": True})
    if parsed is None:
        raise ValueError(f"Could not parse datetime: {value}")
    return parsed.astimezone(timezone.utc)


def _window(start: Optional[str], end: Optional[str], minutes_back: int) -> Tuple[datetime, datetime]:
    now = datetime.now(timezone.utc)
    to_dt = _parse_time(end, now)
    from_dt = _parse_time(start, to_dt - timedelta(minutes=int(minutes_back)))
    if from_dt >= to_dt:
        raise ValueError("start must be earlier than end")
    return from_dt, to_dt


def _finite(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _percentiles(values: Iterable[float]) -> Dict[str, Optional[float]]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if not len(arr):
        return {key: None for key in ("mean", "median", "p90", "p95", "p99", "max")}
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.quantile(arr, 0.90)),
        "p95": float(np.quantile(arr, 0.95)),
        "p99": float(np.quantile(arr, 0.99)),
        "max": float(np.max(arr)),
    }


def _bootstrap_mean_ci(values: Sequence[float], samples: int, seed: int = 42) -> Optional[List[float]]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 5:
        return None
    rng = np.random.default_rng(seed)
    block = max(1, int(round(math.sqrt(len(arr)))))
    means = []
    for _ in range(int(samples)):
        starts = rng.integers(0, len(arr), size=math.ceil(len(arr) / block))
        draw = np.concatenate([arr[(start + np.arange(block)) % len(arr)] for start in starts])[: len(arr)]
        means.append(float(np.mean(draw)))
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def _round_execution_stat(value: Any, *, significant_digits: int = 6) -> Any:
    """Remove binary float tails from derived execution statistics."""
    if value is None:
        return None
    numeric = float(value)
    if not math.isfinite(numeric) or numeric == 0.0:
        return numeric
    decimals = significant_digits - int(math.floor(math.log10(abs(numeric)))) - 1
    return round(numeric, decimals)


def _execution_percentiles(values: Iterable[float]) -> Dict[str, Optional[float]]:
    return {
        key: _round_execution_stat(value)
        for key, value in _percentiles(values).items()
    }


def _execution_duration_display(
    stats: Dict[str, Optional[float]],
) -> Dict[str, str]:
    """Format millisecond duration statistics for quick human inspection."""
    out: Dict[str, str] = {}
    for key, value in stats.items():
        if value is None:
            continue
        milliseconds = max(0.0, float(value))
        if milliseconds < 1000.0:
            display = f"{int(round(milliseconds))}ms"
        elif milliseconds < 60_000.0:
            seconds = milliseconds / 1000.0
            display = f"{seconds:.2f}".rstrip("0").rstrip(".") + "s"
        else:
            display = format_age_seconds(milliseconds / 1000.0)
        if display is not None:
            out[str(key)] = display
    return out


def _execution_bootstrap_mean_ci(
    values: Sequence[float],
    samples: int,
) -> Optional[List[float]]:
    interval = _bootstrap_mean_ci(values, samples)
    if interval is None:
        return None
    return [_round_execution_stat(value) for value in interval]


def _block_bootstrap_positive_mean_p_value(
    values: Sequence[float], samples: int, seed: int = 42
) -> Optional[float]:
    """One-sided p-value for positive mean under a centered block-bootstrap null."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 5:
        return None
    observed = float(np.mean(arr))
    centered = arr - observed
    rng = np.random.default_rng(seed)
    block = max(2, int(round(math.sqrt(len(arr)))))
    exceed = 0
    for _ in range(int(samples)):
        starts = rng.integers(0, len(centered), size=math.ceil(len(centered) / block))
        draw = np.concatenate(
            [centered[(start + np.arange(block)) % len(centered)] for start in starts]
        )[: len(centered)]
        exceed += int(float(np.mean(draw)) >= observed)
    return float((exceed + 1) / (int(samples) + 1))


def _tick_frame(gateway: Any, symbol: str, start: datetime, end: datetime, max_ticks: int) -> Tuple[pd.DataFrame, bool]:
    flags = getattr(gateway, "COPY_TICKS_ALL", 0)
    df = _frame(gateway.copy_ticks_range(symbol, start, end, flags))
    if df.empty:
        return pd.DataFrame(
            {
                column: pd.Series(dtype=float)
                for column in (
                    "epoch",
                    "bid",
                    "ask",
                    "last",
                    "volume",
                    "volume_real",
                    "flags",
                    "spread_valid",
                    "spread_quality",
                    "mid",
                    "spread",
                )
            }
        ), False
    time_msc = _finite(df.get("time_msc", pd.Series(index=df.index, dtype=float)))
    epoch = _finite(df.get("time", pd.Series(index=df.index, dtype=float)))
    df["epoch"] = np.where(time_msc > 0, time_msc / 1000.0, epoch)
    dedupe_columns = [
        column
        for column in ("epoch", "bid", "ask", "last", "volume", "volume_real", "flags")
        if column in df.columns
    ]
    df = (
        df[np.isfinite(df["epoch"])]
        .sort_values("epoch", kind="stable")
        .drop_duplicates(
            subset=dedupe_columns,
            keep="last",
        )
    )
    truncated = len(df) > int(max_ticks)
    if truncated:
        df = df.tail(int(max_ticks)).copy()
    for column in ("bid", "ask", "last", "volume", "volume_real", "flags"):
        if column not in df:
            df[column] = 0.0
        df[column] = _finite(df[column]).fillna(0.0)
    try:
        bid_flag = int(getattr(gateway, "TICK_FLAG_BID", 2) or 2)
    except (TypeError, ValueError):
        bid_flag = 2
    try:
        ask_flag = int(getattr(gateway, "TICK_FLAG_ASK", 4) or 4)
    except (TypeError, ValueError):
        ask_flag = 4
    flag_values = df["flags"].astype(np.int64)
    one_sided_update = ((flag_values & bid_flag) != 0) != (
        (flag_values & ask_flag) != 0
    )
    two_sided_quote = (df["bid"] > 0) & (df["ask"] > df["bid"])
    incomplete_one_sided_update = one_sided_update & ~two_sided_quote
    locked_quote = (df["bid"] > 0) & (df["ask"] == df["bid"])
    inverted_quote = (df["bid"] > 0) & (df["ask"] > 0) & (df["ask"] < df["bid"])
    df["spread_quality"] = np.select(
        [incomplete_one_sided_update, locked_quote, inverted_quote],
        ["one_sided_update", "locked", "inverted"],
        default="two_sided",
    )
    df.loc[(df["bid"] <= 0) | (df["ask"] <= 0), "spread_quality"] = "one_sided"
    df["spread_valid"] = two_sided_quote
    df["mid"] = np.where(
        two_sided_quote,
        (df["bid"] + df["ask"]) / 2.0,
        np.nan,
    )
    df["spread"] = np.where(np.isfinite(df["mid"]), df["ask"] - df["bid"], np.nan)
    return df.reset_index(drop=True), truncated


def _microstructure_latest_quote(
    gateway: Any,
    symbol: str,
    latest_tick: pd.Series,
    *,
    reconcile_live_quote: bool,
    now_epoch: float,
) -> Dict[str, Any]:
    """Return execution quote state while retaining the final raw update state."""
    raw_epoch = float(latest_tick["epoch"])
    raw_quality = str(latest_tick["spread_quality"])
    out: Dict[str, Any] = {
        "bid": latest_tick.get("bid"),
        "ask": latest_tick.get("ask"),
        "epoch": raw_epoch,
        "spread_quality": raw_quality,
        "quote_source": "mt5.copy_ticks_range",
        "quote_source_state": "latest_raw_update",
        "raw_update_quality": raw_quality,
        "raw_update_epoch": raw_epoch,
        "reconciled": False,
    }
    if not reconcile_live_quote or raw_quality == "two_sided":
        return out

    try:
        cached_tick = gateway.symbol_info_tick(symbol)
    except Exception:
        cached_tick = None
    resolved_tick, quote_source = resolve_quote_tick(
        gateway,
        symbol,
        cached_tick,
        now_epoch=now_epoch,
    )
    spread = compute_spread_metrics(
        tick_value(resolved_tick, "bid"),
        tick_value(resolved_tick, "ask"),
    )
    if spread.get("spread_quality") != "two_sided":
        return out

    resolved_epoch = tick_epoch(resolved_tick)
    out.update(
        {
            "bid": tick_value(resolved_tick, "bid"),
            "ask": tick_value(resolved_tick, "ask"),
            "epoch": raw_epoch if resolved_epoch is None else float(resolved_epoch),
            "spread_quality": "two_sided",
            "quote_source": quote_source.get("quote_source"),
            "quote_source_state": quote_source.get("quote_source_state"),
            "reconciled": True,
        }
    )
    return out


def analyze_microstructure(  # noqa: C901
    request: MarketMicrostructureRequest, gateway: Any
) -> Dict[str, Any]:
    try:
        symbol_info = gateway.symbol_info(request.symbol)
    except Exception:
        symbol_info = None
    if symbol_info is None:
        return {
            "error": f"Symbol '{request.symbol}' was not found by MT5.",
            "error_code": "symbol_not_found",
            "symbol": request.symbol,
            "remediation": (
                "Use symbols_list to find the broker's exact symbol name and suffix."
            ),
            "related_tools": ["symbols_list"],
        }
    start, end = _window(request.start, request.end, request.minutes_back)
    df, truncated = _tick_frame(gateway, request.symbol, start, end, request.max_ticks)
    completed_session_context = None
    session = closed_session_context(
        request.symbol,
        now_epoch=end.timestamp(),
        item="tick stream",
    )
    if (
        len(df) < 20
        and session is not None
        and request.start is None
        and request.end is None
    ):
        closure = standard_weekend_window(end)
        if closure is not None:
            completed_end = closure[0]
            completed_start = completed_end - timedelta(minutes=request.minutes_back)
            completed_df, completed_truncated = _tick_frame(
                gateway,
                request.symbol,
                completed_start,
                completed_end,
                request.max_ticks,
            )
            if len(completed_df) > len(df):
                start, end = completed_start, completed_end
                df, truncated = completed_df, completed_truncated
                last_epoch = float(df["epoch"].iloc[-1])
                completed_session_context = closed_session_context(
                    request.symbol,
                    now_epoch=datetime.now(timezone.utc).timestamp(),
                    item="tick stream",
                    data_age_seconds=max(
                        0.0,
                        datetime.now(timezone.utc).timestamp() - last_epoch,
                    ),
                )
    if len(df) < 20:
        last_tick_epoch = float(df["epoch"].iloc[-1]) if len(df) else None
        error_session = closed_session_context(
            request.symbol,
            now_epoch=datetime.now(timezone.utc).timestamp(),
            item="tick stream",
            data_age_seconds=(
                max(0.0, end.timestamp() - last_tick_epoch)
                if last_tick_epoch is not None
                else None
            ),
        )
        if error_session and error_session.get("market_status") == "closed":
            return {
                "error": "Market is closed and fewer than 20 recent usable ticks are available.",
                "error_code": "market_closed",
                "remediation": (
                    "Wait for the session to reopen or analyze a currently trading symbol."
                ),
                "ticks_available": int(len(df)),
                "last_tick_time": (
                    format_epoch_utc(last_tick_epoch)
                    if last_tick_epoch is not None
                    else None
                ),
                **error_session,
                "note": (
                    "Market is closed; fewer than 20 ticks were found in the "
                    "latest completed-session analysis window."
                ),
            }
        return {"error": "At least 20 usable ticks are required.", "error_code": "insufficient_data"}
    quote_mask = np.isfinite(df["mid"])
    flag_values = df["flags"].astype(np.int64)
    trade_mask = (flag_values & mt5_trade_event_mask(gateway)) != 0
    trade_mask &= df["last"] > 0
    real_mask = trade_mask & (df["volume_real"] > 0)
    trade_count = int(trade_mask.sum())
    real_share = float(real_mask.sum() / trade_count) if trade_count else 0.0
    tier = "trade_volume" if trade_count and real_share >= 0.80 else "trade_ticks" if trade_count else "quote_only"
    q = df.loc[quote_mask].copy()
    q["dt"] = q["epoch"].diff()
    q["mid_return"] = np.log(q["mid"]).diff()
    q["bid_revision"] = np.sign(q["bid"].diff())
    q["ask_revision"] = np.sign(q["ask"].diff())
    point = float(getattr(symbol_info, "point", 0.0) or 0.0)
    digits = int(getattr(symbol_info, "digits", 0) or 0)
    points_per_pip = forex_points_per_pip(
        request.symbol,
        path=str(getattr(symbol_info, "path", "") or ""),
        point=point,
        digits=digits,
    )
    revision_pressure = float(np.nanmean((q["bid_revision"] + q["ask_revision"]) / 2.0)) if len(q) > 1 else 0.0
    start_epoch = float(df["epoch"].iloc[0])
    duration = max(0.001, float(df["epoch"].iloc[-1] - start_epoch))
    requested_duration = max(0.0, float((end - start).total_seconds()))
    temporal_coverage_pct = (
        min(100.0, (duration / requested_duration) * 100.0)
        if requested_duration > 0.0
        else 100.0
    )
    bucket = ((df["epoch"] - start_epoch) // int(request.bucket_seconds)).astype(int)
    windows: List[Dict[str, Any]] = []
    for bucket_id, part in df.groupby(bucket):
        pq = part[np.isfinite(part["mid"])]
        bucket_start_epoch = float(part["epoch"].iloc[0])
        bucket_end_epoch = float(part["epoch"].iloc[-1])
        windows.append({
            "bucket": int(bucket_id),
            "start": format_epoch_utc(bucket_start_epoch),
            "end": format_epoch_utc(bucket_end_epoch),
            "start_epoch": bucket_start_epoch,
            "end_epoch": bucket_end_epoch,
            "ticks": int(len(part)),
            "ticks_per_second": float(len(part) / max(1.0, part["epoch"].iloc[-1] - part["epoch"].iloc[0])),
            "spread_median": float(pq["spread"].median()) if len(pq) else None,
            "spread_p95": float(pq["spread"].quantile(0.95)) if len(pq) else None,
            "mid_volatility": float(np.nanstd(np.log(pq["mid"]).diff())) if len(pq) > 2 else None,
        })
    windows.sort(key=lambda item: (-(item.get("spread_p95") or -1.0), -item["ticks"]))
    summary: Dict[str, Any] = {
        "feed_tier": tier,
        "ticks": int(len(df)),
        "duration_seconds": duration,
        "ticks_per_second": float(len(df) / duration),
        "spread": _percentiles(q["spread"]),
        "quote_gap_seconds": _percentiles(q["dt"].dropna()),
        "mid_realized_volatility": float(np.sqrt(np.nansum(np.square(q["mid_return"])))) if len(q) > 1 else None,
        "broker_quote_revision_imbalance": revision_pressure,
    }
    if point > 0:
        summary["spread_points"] = _percentiles(q["spread"] / point)
        if points_per_pip:
            summary["spread_pips"] = _percentiles(
                q["spread"] / (point * points_per_pip)
            )
    applicability = {
        "quote_metrics": bool(len(q) >= 20),
        "trade_direction_metrics": tier in {"trade_ticks", "trade_volume"},
        "volume_impact_metrics": tier == "trade_volume",
    }
    if trade_count:
        trades = df.loc[trade_mask].copy()
        prevailing_mid = df["mid"].ffill()
        trades["side"] = _classify_trade_sides(trades, prevailing_mid)
        summary["trade_count"] = trade_count
        summary["trade_count_imbalance"] = float(trades["side"].sum() / max(1, trade_count))
        if tier == "trade_volume":
            weights = trades["volume_real"].where(trades["volume_real"] > 0, np.nan)
            signed = weights * trades["side"]
            total = float(weights.sum())
            summary["signed_volume_imbalance"] = float(signed.sum() / total) if total > 0 else None
            summary["vwap"] = float((trades["last"] * weights).sum() / total) if total > 0 else None
            returns = np.log(trades["last"]).diff()
            dv = signed.fillna(0.0)
            valid = np.isfinite(returns) & np.isfinite(dv) & (dv != 0)
            if int(valid.sum()) >= 20:
                x = dv[valid].to_numpy(dtype=float)
                y = returns[valid].to_numpy(dtype=float)
                summary["broker_tick_signed_volume_impact_slope"] = float(np.dot(x, y) / np.dot(x, x)) if np.dot(x, x) > 0 else None
                summary["broker_tick_abs_return_per_real_volume"] = float(np.nanmean(np.abs(y) / np.maximum(np.abs(x), 1e-12)))
                summary["volume_impact_observations"] = int(valid.sum())
    p95 = summary["spread"].get("p95")
    event_windows = [item for item in windows if p95 is not None and item.get("spread_p95") is not None and item["spread_p95"] >= p95][:10]
    events = [
        {
            key: value
            for key, value in item.items()
            if key not in {"start_epoch", "end_epoch"}
        }
        for item in event_windows
    ]
    warnings = []
    if completed_session_context is not None:
        warnings.append(
            "Market is closed; metrics use the latest completed-session tick window."
        )
    if tier != "trade_volume":
        warnings.append("Real trade volume is insufficient; volume-impact metrics were omitted.")
    if truncated:
        warnings.append(
            "max_ticks truncated the requested window; every metric covers only "
            "the retained latest-tick tail described by data_quality."
        )
    if temporal_coverage_pct < 90.0:
        warnings.append(
            "Observed ticks span less than 90% of the requested elapsed window; "
            "interpret temporal comparisons with caution."
        )
    warnings.append(
        "Metrics describe the connected broker's tick feed and do not establish centralized market-wide order flow or liquidity."
    )
    data_quality = {
        "feed_tier": tier,
        "quote_coverage": float(quote_mask.mean()),
        "trade_tick_coverage": float(trade_mask.mean()),
        "real_volume_trade_coverage": real_share,
        "invalid_partial_quote_ticks": int(
            df["spread_quality"].isin(
                {"one_sided", "one_sided_update", "inverted"}
            ).sum()
        ),
        "locked_quote_ticks": int((df["spread_quality"] == "locked").sum()),
        "latest_raw_update_quality": str(df["spread_quality"].iloc[-1]),
        "truncated": truncated,
        "retained": "latest" if truncated else "complete_window",
        "requested_start": start.isoformat(),
        "requested_end": end.isoformat(),
        "requested_duration_seconds": requested_duration,
        "observed_duration_seconds": duration,
        "temporal_coverage_pct": temporal_coverage_pct,
        "observed_start_epoch": float(df["epoch"].iloc[0]),
        "observed_end_epoch": float(df["epoch"].iloc[-1]),
    }
    if request.detail in {"compact", "summary"}:
        spread_key = (
            "spread_pips"
            if "spread_pips" in summary
            else "spread_points"
            if "spread_points" in summary
            else "spread"
        )
        spread_unit = {
            "spread_pips": "fx_pips",
            "spread_points": "broker_points",
            "spread": "absolute_price",
        }[spread_key]
        spread_stats = summary[spread_key]
        if spread_key == "spread_pips":
            spread_series = q["spread"] / (point * float(points_per_pip))
        elif spread_key == "spread_points":
            spread_series = q["spread"] / point
        else:
            spread_series = q["spread"]
        spread_series = pd.to_numeric(spread_series, errors="coerce")
        latest_tick = df.iloc[-1]
        raw_update_epoch = float(latest_tick["epoch"])
        latest_quote = _microstructure_latest_quote(
            gateway,
            request.symbol,
            latest_tick,
            reconcile_live_quote=(
                request.start is None
                and request.end is None
                and completed_session_context is None
            ),
            now_epoch=datetime.now(timezone.utc).timestamp(),
        )
        latest_spread_quality = str(latest_quote["spread_quality"])
        latest_quote_epoch = float(latest_quote["epoch"])
        latest_spread = None
        if latest_spread_quality in {"two_sided", "locked"}:
            latest_absolute_spread = float(latest_quote["ask"]) - float(
                latest_quote["bid"]
            )
            if spread_key == "spread_pips":
                latest_spread = latest_absolute_spread / (point * float(points_per_pip))
            elif spread_key == "spread_points":
                latest_spread = latest_absolute_spread / point
            else:
                latest_spread = latest_absolute_spread
        recent_mask = (
            q["epoch"] >= raw_update_epoch - 300.0
        )
        recent_spreads = spread_series.loc[recent_mask].dropna()
        recent_median = (
            float(recent_spreads.median()) if len(recent_spreads) else None
        )
        window_median = spread_stats.get("median")
        latest_to_window_ratio = (
            float(latest_spread) / float(window_median)
            if latest_spread_quality == "two_sided"
            and latest_spread is not None
            and window_median is not None
            and float(window_median) > 0
            else None
        )
        spread_regime = (
            "locked_quote"
            if latest_spread_quality == "locked"
            else "unreliable_quote"
            if latest_spread_quality != "two_sided"
            else "wider_than_window"
            if latest_to_window_ratio is not None
            and latest_to_window_ratio >= 2.0 - 1e-9
            else "tighter_than_window"
            if latest_to_window_ratio is not None
            and latest_to_window_ratio <= 0.5 + 1e-9
            else "near_window_median"
            if latest_to_window_ratio is not None
            else "unknown"
        )
        if latest_quote.get("reconciled"):
            warnings.append(
                "The final raw tick-stream update was not executable; latest spread "
                "uses the canonical reconciled live quote while raw update quality "
                "remains in data_quality."
            )
        elif latest_spread_quality == "locked":
            warnings.append(
                "Latest analyzed quote is locked (bid equals ask); its zero "
                "spread is not usable for execution."
            )
        elif latest_spread_quality != "two_sided":
            warnings.append(
                "Latest analyzed quote is not a valid two-sided quote; do not "
                "use it for execution."
            )
        if spread_regime in {"wider_than_window", "tighter_than_window"}:
            warnings.append(
                "Latest analyzed spread differs materially from the full-window "
                "median; use latest and recent_5m_median for near-term execution context."
            )
        compact_result = {
            "success": True,
            "symbol": request.symbol,
            "summary": {
                "feed_tier": tier,
                "ticks": int(len(df)),
                "duration_seconds": duration,
                "ticks_per_second": float(len(df) / duration),
                "spread": {
                    "latest": _round_execution_stat(latest_spread),
                    "latest_as_of": format_epoch_utc(latest_quote_epoch),
                    "spread_valid": latest_spread_quality == "two_sided",
                    "spread_quality": latest_spread_quality,
                    "raw_update_quality": latest_quote["raw_update_quality"],
                    "raw_update_as_of": format_epoch_utc(
                        float(latest_quote["raw_update_epoch"])
                    ),
                    "recent_5m_median": _round_execution_stat(recent_median),
                    "window_median": _round_execution_stat(window_median),
                    "window_p95": _round_execution_stat(spread_stats.get("p95")),
                    "latest_to_window_median_ratio": _round_execution_stat(
                        latest_to_window_ratio
                    ),
                    "regime": spread_regime,
                    "unit": spread_unit,
                    "basis": (
                        "canonical_live_quote_against_historical_tick_window_distribution"
                        if latest_quote.get("reconciled")
                        else "historical_tick_window_distribution"
                    ),
                    "source": latest_quote.get("quote_source"),
                    "source_state": latest_quote.get("quote_source_state"),
                },
            },
            "observed_window": {
                "start": format_epoch_utc(float(df["epoch"].iloc[0])),
                "end": format_epoch_utc(float(df["epoch"].iloc[-1])),
            },
            "data_quality": {
                key: data_quality[key]
                for key in (
                    "quote_coverage",
                    "invalid_partial_quote_ticks",
                    "locked_quote_ticks",
                    "latest_raw_update_quality",
                    "truncated",
                    "retained",
                    "requested_start",
                    "requested_end",
                    "requested_duration_seconds",
                    "observed_duration_seconds",
                    "temporal_coverage_pct",
                )
            },
            "warnings": warnings,
        }
        if completed_session_context is not None:
            compact_result.update(completed_session_context)
        return compact_result
    result = {
        "success": True,
        "symbol": request.symbol,
        "timezone": "UTC",
        "summary": summary,
        "liquidity_events": events,
        **({"windows": windows} if request.detail == "full" else {}),
        "data_quality": data_quality,
        "method_applicability": applicability,
        "estimator_scope": {
            "market_scope": "connected_broker_tick_feed",
            "trade_sign_method": "prevailing_quote_then_tick_rule",
            "volume_source": "volume_real" if tier == "trade_volume" else None,
            "volume_unit": "broker_reported_real_volume" if tier == "trade_volume" else None,
        },
        "units": {
            "spread": "absolute_price",
            "spread_points": "broker_points",
            "spread_pips": "fx_pips_when_symbol_is_identifiable_as_forex",
            "quote_gap_seconds": "seconds",
            "broker_quote_revision_imbalance": "signed_fraction",
            "broker_tick_signed_volume_impact_slope": "log_return_per_broker_real_volume",
            "broker_tick_abs_return_per_real_volume": "absolute_log_return_per_broker_real_volume",
        },
        "warnings": warnings,
    }
    if completed_session_context is not None:
        result.update(completed_session_context)
    return result


def _deal_side(row: Dict[str, Any], gateway: Any) -> Optional[str]:
    value = row.get("type")
    text = str(value).lower()
    if text in {"buy", "0", str(getattr(gateway, "DEAL_TYPE_BUY", 0))}:
        return "buy"
    if text in {"sell", "1", str(getattr(gateway, "DEAL_TYPE_SELL", 1))}:
        return "sell"
    return None


def _order_type_label(value: Any, gateway: Any) -> str:
    order_types = (
        ("ORDER_TYPE_BUY", 0),
        ("ORDER_TYPE_SELL", 1),
        ("ORDER_TYPE_BUY_LIMIT", 2),
        ("ORDER_TYPE_SELL_LIMIT", 3),
        ("ORDER_TYPE_BUY_STOP", 4),
        ("ORDER_TYPE_SELL_STOP", 5),
        ("ORDER_TYPE_BUY_STOP_LIMIT", 6),
        ("ORDER_TYPE_SELL_STOP_LIMIT", 7),
        ("ORDER_TYPE_CLOSE_BY", 8),
    )
    for name, fallback in order_types:
        code = getattr(gateway, name, fallback)
        if value == code or str(value) == str(code):
            return name.removeprefix("ORDER_TYPE_")
    return "UNKNOWN"


def _execution_symbol_catalog(gateway: Any) -> Dict[str, Dict[str, Any]]:
    try:
        raw_symbols = list(gateway.symbols_get() or [])
    except Exception:
        return {}
    catalog: Dict[str, Dict[str, Any]] = {}
    for item in raw_symbols:
        row = _mapping(item)
        name = str(row.get("name") or getattr(item, "name", "") or "").strip()
        if name:
            catalog[name.casefold()] = {
                "name": name,
                "path": str(row.get("path") or getattr(item, "path", "") or ""),
            }
    return catalog


def _execution_session_calendar(
    symbol: str,
    *,
    gateway: Any,
    catalog: Dict[str, Dict[str, Any]],
) -> tuple[str, Optional[str]]:
    metadata = catalog.get(str(symbol).casefold(), {})
    path = str(metadata.get("path") or "")
    if not path:
        try:
            info = gateway.symbol_info(symbol)
        except Exception:
            info = None
        path = str(getattr(info, "path", "") or "")
    if is_probably_crypto_symbol(symbol):
        return "continuous_24_7", path or None
    if is_probably_fx_session_symbol(symbol, path=path):
        return "fx", path or None
    return "utc_hour_only", path or None


def _execution_session_definition(calendar: str) -> Dict[str, Any]:
    if calendar == "fx":
        return session_definition_for_clock("UTC", "fx")
    if calendar == "continuous_24_7":
        return {
            "basis": "continuous_market",
            "calendar": "continuous_24_7",
            "clock": "UTC",
            "continuous": "All UTC hours; no off-session bucket is applied.",
        }
    return {
        "basis": "utc_hour_only",
        "calendar": "utc_hour_only",
        "clock": "UTC",
        "note": (
            "No reliable venue calendar was available; use by_hour_utc and do "
            "not interpret named geographic sessions."
        ),
    }


def analyze_execution_quality(  # noqa: C901
    request: TradeExecutionQualityRequest, gateway: Any
) -> Dict[str, Any]:
    range_error = validate_historical_range(request.start, request.end)
    if range_error is not None:
        return range_error
    start, end = _window(request.start, request.end, request.minutes_back)
    account_currency = None
    account_info = getattr(gateway, "account_info", None)
    if callable(account_info):
        try:
            account_currency = str(
                getattr(account_info(), "currency", "") or ""
            ).strip() or None
        except Exception:
            account_currency = None
    symbol_catalog = _execution_symbol_catalog(gateway)
    resolved_symbol = None
    if request.symbol:
        from ..utils.mt5 import resolve_broker_symbol_name

        resolved_symbol = resolve_broker_symbol_name(request.symbol, gateway=gateway)
        exact = symbol_catalog.get(str(resolved_symbol).casefold())
        if symbol_catalog and exact is None:
            return {
                "error": f"Symbol {request.symbol!r} was not found by MT5.",
                "error_code": "symbol_not_found",
                "symbol": request.symbol,
                "remediation": (
                    "Use symbols_list to discover the exact broker symbol name."
                ),
            }
        if exact is not None:
            resolved_symbol = str(exact["name"])
        elif not symbol_catalog:
            try:
                symbol_info = gateway.symbol_info(resolved_symbol)
            except Exception:
                symbol_info = None
            if symbol_info is None:
                return {
                    "error": f"Symbol {request.symbol!r} was not found by MT5.",
                    "error_code": "symbol_not_found",
                    "symbol": request.symbol,
                    "remediation": (
                        "Use symbols_list to discover the exact broker symbol name."
                    ),
                }
    kwargs = {"group": resolved_symbol} if resolved_symbol else {}
    raw_deals = [
        _mapping(row)
        for row in (gateway.history_deals_get(start, end, **kwargs) or [])
    ]
    raw_orders = [
        _mapping(row)
        for row in (gateway.history_orders_get(start, end, **kwargs) or [])
    ]
    if resolved_symbol:
        deals = [
            row
            for row in raw_deals
            if str(row.get("symbol") or "").casefold()
            == resolved_symbol.casefold()
        ]
        orders = [
            row
            for row in raw_orders
            if not str(row.get("symbol") or "").strip()
            or str(row.get("symbol") or "").casefold()
            == resolved_symbol.casefold()
        ]
    else:
        deals = raw_deals
        orders = raw_orders
    order_by_ticket = {int(row.get("ticket") or 0): row for row in orders if row.get("ticket")}
    fills = []
    skipped = {
        "non_trade": 0,
        "filter": 0,
        "unbenchmarked": 0,
        "missing_markout": 0,
        "future_timestamp": 0,
    }
    eligible_deals = []
    for deal in deals:
        side = _deal_side(deal, gateway)
        volume = float(deal.get("volume") or 0.0)
        symbol = str(deal.get("symbol") or "").strip()
        if side is None or volume <= 0 or not symbol:
            skipped["non_trade"] += 1
            continue
        if request.side and side != request.side:
            skipped["filter"] += 1
            continue
        if request.magic is not None and int(deal.get("magic") or 0) != int(request.magic):
            skipped["filter"] += 1
            continue
        eligible_deals.append(deal)

    eligible_deals.sort(
        key=lambda row: (
            float(row.get("time_msc") or 0),
            int(row.get("ticket") or 0),
        ),
        reverse=True,
    )
    benchmark_sources = {
        "arrival_quote": 0,
        "pending_order_price": 0,
        "order_price": 0,
        "order_price_fallback": 0,
    }
    arrival_quote_observations = 0
    processed_candidates = 0
    observed_epoch = datetime.now(timezone.utc).timestamp()
    future_tolerance_seconds = 300.0
    for deal in eligible_deals:
        processed_candidates += 1
        side = _deal_side(deal, gateway)
        volume = float(deal.get("volume") or 0.0)
        symbol = str(deal.get("symbol") or "").strip()
        order = order_by_ticket.get(int(deal.get("order") or 0), {})
        fill_epoch = float(deal.get("time_msc") or 0) / 1000.0 or float(deal.get("time") or 0)
        if fill_epoch > observed_epoch + future_tolerance_seconds:
            skipped["future_timestamp"] += 1
            continue
        time_setup_msc = float(order.get("time_setup_msc") or 0.0)
        if not time_setup_msc and order.get("time_setup"):
            time_setup_msc = float(order["time_setup"]) * 1000.0
        setup_epoch = time_setup_msc / 1000.0 if time_setup_msc else None
        qstart = datetime.fromtimestamp(fill_epoch - request.quote_window_seconds, tz=timezone.utc)
        qend = datetime.fromtimestamp(fill_epoch + max(request.markout_seconds) + 5, tz=timezone.utc)
        ticks, _ = _tick_frame(gateway, symbol, qstart, qend, 50_000)
        before = ticks[(ticks["epoch"] <= fill_epoch) & np.isfinite(ticks["mid"])]
        fill_time_quote = None
        if len(before):
            fill_tick = before.iloc[-1]
            fill_time_quote = float(fill_tick["ask"] if side == "buy" else fill_tick["bid"])
        order_type_value = order.get("type")
        order_type_label = _order_type_label(order_type_value, gateway)
        market_order_types = {
            getattr(gateway, "ORDER_TYPE_BUY", 0),
            getattr(gateway, "ORDER_TYPE_SELL", 1),
        }
        # Historical gateways can omit the originating order type. In that
        # ambiguous case, retain the conservative market-fill benchmark rather
        # than silently treating the deal as a pending-order fill.
        is_market_order = (
            order_type_value is None or order_type_value in market_order_types
        )
        order_price = float(
            order.get("price_open") or order.get("price_current") or 0.0
        )
        arrival_quote = None
        arrival_quote_epoch = None
        benchmark_price = None
        benchmark_epoch = None
        benchmark_source = None
        if request.benchmark == "arrival_quote" and setup_epoch is not None:
            arrival_start = datetime.fromtimestamp(
                setup_epoch - request.quote_window_seconds,
                tz=timezone.utc,
            )
            arrival_end = datetime.fromtimestamp(setup_epoch, tz=timezone.utc)
            arrival_ticks, _ = _tick_frame(
                gateway, symbol, arrival_start, arrival_end, 50_000
            )
            arrival_before = arrival_ticks[
                (arrival_ticks["epoch"] <= setup_epoch)
                & np.isfinite(arrival_ticks["mid"])
            ]
            if len(arrival_before):
                latest = arrival_before.iloc[-1]
                arrival_quote = float(
                    latest["ask"] if side == "buy" else latest["bid"]
                )
                arrival_quote_epoch = float(latest["epoch"])
                arrival_quote_observations += 1
        if request.benchmark == "order_price":
            if order_price > 0:
                benchmark_price = order_price
                benchmark_source = "order_price"
        elif not is_market_order:
            if order_price > 0:
                benchmark_price = order_price
                benchmark_source = "pending_order_price"
        elif arrival_quote and arrival_quote > 0:
            benchmark_price = arrival_quote
            benchmark_epoch = arrival_quote_epoch
            benchmark_source = "arrival_quote"
        elif request.benchmark_fallback == "order_price" and order_price > 0:
            benchmark_price = order_price
            benchmark_source = "order_price_fallback"
        fill_price = float(deal.get("price") or 0.0)
        if not benchmark_price or fill_price <= 0:
            skipped["unbenchmarked"] += 1
            continue
        benchmark_sources[str(benchmark_source)] += 1
        sign = 1.0 if side == "buy" else -1.0
        slippage_bps = (
            sign * (fill_price - benchmark_price) / benchmark_price * 10_000.0
        )
        pending_arrival_shortfall_bps = (
            sign * (fill_price - arrival_quote) / arrival_quote * 10_000.0
            if not is_market_order and arrival_quote and arrival_quote > 0
            else None
        )
        markouts: Dict[str, Optional[float]] = {}
        for horizon in request.markout_seconds:
            candidates = ticks[(ticks["epoch"] >= fill_epoch + horizon) & (ticks["epoch"] <= fill_epoch + horizon + 5) & np.isfinite(ticks["mid"])]
            if len(candidates):
                markouts[str(horizon)] = float(sign * (float(candidates.iloc[0]["mid"]) - fill_price) / fill_price * 10_000.0)
            else:
                markouts[str(horizon)] = None
                skipped["missing_markout"] += 1
        initial_volume = float(order.get("volume_initial") or volume)
        order_to_fill_duration_ms = (
            max(
                0.0,
                float(deal.get("time_msc") or fill_epoch * 1000.0)
                - time_setup_msc,
            )
            if time_setup_msc
            else None
        )
        item = {
            "deal_ticket": deal.get("ticket"),
            "order_ticket": deal.get("order"),
            "position_id": deal.get("position_id"),
            "symbol": symbol,
            "side": side,
            "volume": volume,
            "fill_price": fill_price,
            "benchmark_price": benchmark_price,
            "benchmark_source": benchmark_source,
            "benchmark_epoch": benchmark_epoch,
            "benchmark_time": (
                format_epoch_utc(benchmark_epoch)
                if benchmark_epoch is not None
                else None
            ),
            "fill_time_quote": fill_time_quote,
            "slippage_bps": slippage_bps,
            "price_improved": slippage_bps < 0,
            **(
                {
                    "arrival_quote_price": arrival_quote,
                    "arrival_quote_epoch": arrival_quote_epoch,
                    "arrival_quote_time": format_epoch_utc(arrival_quote_epoch),
                    "arrival_implementation_shortfall_bps": (
                        pending_arrival_shortfall_bps
                    ),
                }
                if pending_arrival_shortfall_bps is not None
                and arrival_quote_epoch is not None
                else {}
            ),
            "order_to_fill_duration_ms": order_to_fill_duration_ms,
            "fill_timing_basis": (
                "market_fill_latency" if is_market_order else "pending_time_to_fill"
            ),
            "is_market_order": is_market_order,
            "deal_fill_ratio": min(1.0, volume / initial_volume) if initial_volume > 0 else None,
            "commission": float(deal.get("commission") or 0.0),
            "fee": float(deal.get("fee") or 0.0),
            "commission_fee_per_lot": max(
                0.0,
                -(
                    float(deal.get("commission") or 0.0)
                    + float(deal.get("fee") or 0.0)
                ),
            )
            / volume,
            "markout_bps": markouts,
            "fill_epoch": fill_epoch,
            "order_type": order_type_label,
            "order_type_code": order_type_value,
            "hour_utc": datetime.fromtimestamp(fill_epoch, tz=timezone.utc).hour,
        }
        session_calendar, symbol_path = _execution_session_calendar(
            symbol,
            gateway=gateway,
            catalog=symbol_catalog,
        )
        item["session_calendar"] = session_calendar
        item["session"] = None
        if symbol_path:
            item["symbol_path"] = symbol_path
        if session_calendar == "fx":
            item["session"] = market_session_label(
                datetime.fromtimestamp(fill_epoch, tz=timezone.utc),
                session_calendar="fx",
            )
        elif session_calendar == "continuous_24_7":
            item["session"] = "continuous"
        try:
            action = getattr(gateway, "ORDER_TYPE_BUY", 0) if side == "buy" else getattr(gateway, "ORDER_TYPE_SELL", 1)
            shortfall = gateway.order_calc_profit(
                action, symbol, volume, benchmark_price, fill_price
            )
            if shortfall is not None:
                item["execution_shortfall_currency_estimate"] = float(shortfall)
        except Exception:
            pass
        fills.append(item)
        if len(fills) >= request.limit:
            break
    fills.sort(
        key=lambda item: (
            float(item.get("fill_epoch") or 0),
            int(item.get("deal_ticket") or 0),
        )
    )
    market_order_fills = [item for item in fills if item.get("is_market_order")]
    non_market_order_fills = [item for item in fills if not item.get("is_market_order")]
    market_slippages = [
        float(item["slippage_bps"]) for item in market_order_fills
    ]
    pending_slippages = [
        float(item["slippage_bps"]) for item in non_market_order_fills
    ]
    pending_arrival_shortfalls = [
        float(item["arrival_implementation_shortfall_bps"])
        for item in non_market_order_fills
        if item.get("arrival_implementation_shortfall_bps") is not None
    ]
    if request.benchmark == "order_price":
        headline_fills = fills
        slippage_basis = "explicit_order_price_all_fills"
    elif market_order_fills:
        headline_fills = market_order_fills
        slippage_basis = (
            "market_arrival_quote_with_order_price_fallback"
            if any(
                item.get("benchmark_source") == "order_price_fallback"
                for item in market_order_fills
            )
            else "market_arrival_quote"
        )
    else:
        headline_fills = non_market_order_fills
        slippage_basis = "pending_order_price_no_market_fills"
    headline_slippages = [
        float(item["slippage_bps"]) for item in headline_fills
    ]
    order_fill_totals: Dict[Any, Dict[str, float]] = {}
    for item in fills:
        order_ticket = item.get("order_ticket")
        state = order_fill_totals.setdefault(
            order_ticket,
            {"filled_volume": 0.0, "initial_volume": 0.0},
        )
        state["filled_volume"] += float(item.get("volume") or 0.0)
        order = order_by_ticket.get(int(order_ticket or 0), {})
        state["initial_volume"] = max(
            state["initial_volume"],
            float(order.get("volume_initial") or item.get("volume") or 0.0),
        )
    partial_orders = sum(
        state["initial_volume"] > 0.0
        and state["filled_volume"] < state["initial_volume"] * 0.999
        for state in order_fill_totals.values()
    )
    summary = {
        "fills": len(fills),
        "orders": len({item["order_ticket"] for item in fills}),
        "market_order_fills": len(market_order_fills),
        "non_market_order_fills": len(non_market_order_fills),
        "slippage_basis": slippage_basis,
        "slippage_bps": _execution_percentiles(headline_slippages),
        "market_fill_slippage_bps": _execution_percentiles(market_slippages),
        "pending_fill_vs_order_bps": _execution_percentiles(pending_slippages),
        "pending_arrival_implementation_shortfall_bps": _execution_percentiles(
            pending_arrival_shortfalls
        ),
        "mean_slippage_ci_95": _execution_bootstrap_mean_ci(
            headline_slippages, 500
        ),
        "price_improvement_rate": _round_execution_stat(
            np.mean([item["price_improved"] for item in headline_fills])
        ) if headline_fills else None,
        "partial_fill_rate": _round_execution_stat(
            partial_orders / len(order_fill_totals)
        ) if order_fill_totals else None,
        "partial_orders": int(partial_orders),
        "orders_evaluated_for_partial_fills": len(order_fill_totals),
        "partial_fill_rate_basis": "orders_aggregated_from_deals",
        "market_fill_latency_ms": _execution_percentiles(
            item["order_to_fill_duration_ms"]
            for item in market_order_fills
            if item.get("order_to_fill_duration_ms") is not None
        ),
        "pending_time_to_fill_ms": _execution_percentiles(
            item["order_to_fill_duration_ms"]
            for item in non_market_order_fills
            if item.get("order_to_fill_duration_ms") is not None
        ),
        "order_to_fill_duration_ms": _execution_percentiles(
            item["order_to_fill_duration_ms"]
            for item in fills
            if item.get("order_to_fill_duration_ms") is not None
        ),
        "commission_fee_per_lot": _execution_percentiles(item["commission_fee_per_lot"] for item in fills),
    }
    duration_display = {
        name.removesuffix("_ms"): display
        for name in ("pending_time_to_fill_ms", "order_to_fill_duration_ms")
        if (
            display := _execution_duration_display(
                summary.get(name) if isinstance(summary.get(name), dict) else {}
            )
        )
    }
    if duration_display:
        summary["duration_display"] = duration_display
    for horizon in request.markout_seconds:
        summary.setdefault("markout_bps", {})[str(horizon)] = _execution_percentiles(item["markout_bps"].get(str(horizon)) for item in fills if item["markout_bps"].get(str(horizon)) is not None)
    breakdowns: Dict[str, List[Dict[str, Any]]] = {}
    if fills:
        fill_frame = pd.DataFrame(fills)
        for keys, label in ((["symbol", "side"], "by_symbol_side"), (["order_type"], "by_order_type"), (["session_calendar", "session"], "by_session"), (["hour_utc"], "by_hour_utc")):
            breakdowns[label] = []
            source_frame = fill_frame
            if label == "by_session":
                source_frame = fill_frame[fill_frame["session"].notna()]
            for group_key, items in source_frame.groupby(keys):
                labels = group_key if isinstance(group_key, tuple) else (group_key,)
                row = {name: value for name, value in zip(keys, labels)}
                row.update({"fills": len(items), "slippage_bps": _execution_percentiles(items["slippage_bps"])})
                if label == "by_order_type":
                    codes = [
                        value
                        for value in items["order_type_code"].dropna().unique().tolist()
                    ]
                    if len(codes) == 1:
                        row["order_type_code"] = codes[0]
                    row["order_to_fill_duration_ms"] = _execution_percentiles(
                        items["order_to_fill_duration_ms"]
                    )
                breakdowns[label].append(row)
    sample_start = format_epoch_utc(fills[0]["fill_epoch"]) if fills else None
    sample_end = format_epoch_utc(fills[-1]["fill_epoch"]) if fills else None
    benchmark_attempts = max(
        0, processed_candidates - skipped["future_timestamp"]
    )
    fallback_count = benchmark_sources["order_price_fallback"]
    warnings = []
    if fallback_count:
        warnings.append(
            f"{fallback_count} fill(s) used order price because no arrival quote was available."
        )
    if non_market_order_fills:
        warnings.append(
            "pending_time_to_fill_ms measures intentional limit/stop order wait, not "
            "broker execution latency; order_to_fill_duration_ms is a mixed duration."
        )
        if request.benchmark == "arrival_quote":
            warnings.append(
                "Pending fills use their order price for fill-quality slippage; "
                "setup-to-fill market movement is reported separately as "
                "pending_arrival_implementation_shortfall_bps."
            )
    if market_order_fills and non_market_order_fills and request.benchmark == "arrival_quote":
        warnings.append(
            "Headline slippage_bps and price_improvement_rate use market-order fills "
            "only; pending fill quality is reported separately."
        )
    if skipped["future_timestamp"]:
        warnings.append(
            f"Skipped {skipped['future_timestamp']} fill(s) whose broker timestamp "
            "was more than 5 minutes ahead of the observation clock."
        )
    session_calendars = sorted(
        {
            str(item.get("session_calendar"))
            for item in fills
            if item.get("session_calendar")
        }
    )
    if not session_calendars and resolved_symbol:
        fallback_calendar, _ = _execution_session_calendar(
            resolved_symbol,
            gateway=gateway,
            catalog=symbol_catalog,
        )
        session_calendars = [fallback_calendar]
    session_definitions = {
        calendar: _execution_session_definition(calendar)
        for calendar in session_calendars
    }
    if "utc_hour_only" in session_calendars:
        warnings.append(
            "One or more symbols have no reliable venue calendar; use by_hour_utc for those fills."
        )
    matched_symbols = sorted(
        {
            str(item.get("symbol"))
            for item in eligible_deals
            if item.get("symbol")
        }
    )
    return {
        "success": True,
        **(
            {
                "symbol_filter": {
                    "requested": request.symbol,
                    "resolved": resolved_symbol,
                    "match_mode": "exact",
                }
            }
            if request.symbol
            else {}
        ),
        **({"currency": account_currency} if account_currency else {}),
        "summary": summary,
        **({"breakdowns": breakdowns} if request.detail != "compact" else {}),
        **({"items": fills} if request.detail == "full" else {}),
        "sample_quality": {"status": "ok" if len(fills) >= request.min_sample else "insufficient", "minimum": request.min_sample, "observed": len(fills)},
        "data_quality": {
            "history_deals": len(deals),
            "history_orders": len(orders),
            "history_deals_before_exact_filter": len(raw_deals),
            "history_orders_before_exact_filter": len(raw_orders),
            "matched_symbols": matched_symbols,
            "eligible_trade_deals": len(eligible_deals),
            "processed_candidates": processed_candidates,
            "matched_fills": len(fills),
            "skipped": skipped,
            "benchmark": {
                "requested": request.benchmark,
                "fallback_policy": request.benchmark_fallback,
                "source_counts": benchmark_sources,
                "fallback_count": fallback_count,
                "arrival_quote_coverage": (
                    arrival_quote_observations / benchmark_attempts
                    if request.benchmark == "arrival_quote" and benchmark_attempts
                    else 0.0
                    if request.benchmark == "arrival_quote"
                    else None
                ),
            },
            **(
                {"session_definition": next(iter(session_definitions.values()))}
                if len(session_definitions) == 1
                else {"session_definitions": session_definitions}
            ),
        },
        "sample": {
            "selection_order": "latest_first",
            "display_order": "chronological",
            "total_eligible": len(eligible_deals),
            "sample_start": sample_start,
            "sample_end": sample_end,
            "truncated": processed_candidates < len(eligible_deals),
        },
        "timing_definition": {
            "market_fill_latency_ms": "market_order_setup_to_fill_elapsed_time",
            "pending_time_to_fill_ms": "pending_order_setup_to_fill_wait_duration_not_execution_latency",
            "order_to_fill_duration_ms": "all_order_setup_to_fill_mixed_duration_not_execution_latency",
        },
        "price_quality_definition": {
            "slippage_bps": slippage_basis,
            "market_fill_slippage_bps": "market_fill_vs_arrival_executable_quote",
            "pending_fill_vs_order_bps": "pending_fill_vs_submitted_order_price",
            "pending_arrival_implementation_shortfall_bps": (
                "pending_fill_vs_order_setup_executable_quote_not_broker_slippage"
            ),
        },
        "units": {
            "slippage_bps": "basis_points_positive_is_worse",
            "market_fill_slippage_bps": "basis_points_positive_is_worse",
            "pending_fill_vs_order_bps": "basis_points_positive_is_worse",
            "pending_arrival_implementation_shortfall_bps": (
                "basis_points_positive_is_worse"
            ),
            "markout_bps": "basis_points_positive_is_favorable",
            "market_fill_latency_ms": "milliseconds",
            "pending_time_to_fill_ms": "milliseconds",
            "order_to_fill_duration_ms": "milliseconds",
            "commission": "account_currency",
            "fee": "account_currency",
            "commission_fee_per_lot": "account_currency_per_broker_lot",
            "execution_shortfall_currency_estimate": "account_currency_positive_is_worse",
        },
        "warnings": warnings,
    }


def _rates(
    gateway: Any,
    symbol: str,
    timeframe: str,
    count: int,
    *,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    if start and end:
        from_dt, to_dt = _window(start, end, 1)
        raw = gateway.copy_rates_range(symbol, TIMEFRAME_MAP[timeframe], from_dt, to_dt)
    else:
        raw = gateway.copy_rates_from_pos(symbol, TIMEFRAME_MAP[timeframe], 0, int(count) + 2)
    df = _frame(raw)
    if df.empty or "close" not in df or "time" not in df:
        return pd.DataFrame()
    df = df.sort_values("time", kind="stable").drop_duplicates("time", keep="last")
    for column in ("open", "high", "low", "close", "tick_volume", "real_volume", "spread"):
        if column not in df:
            df[column] = 0.0
        df[column] = _finite(df[column])
    now = datetime.now(timezone.utc).timestamp()
    seconds = TIMEFRAME_SECONDS[timeframe]
    df = df[df["time"] + seconds <= now]
    return df.tail(int(count)).reset_index(drop=True)


def _builtin_signal(close: pd.Series, candidate: StrategyCandidate) -> pd.Series:
    params = candidate.params
    if candidate.strategy in {"sma_cross", "ema_cross"}:
        fast = int(params.get("fast_period", 10))
        slow = int(params.get("slow_period", 30))
        if fast >= slow:
            raise ValueError("fast_period must be less than slow_period")
        if candidate.strategy == "sma_cross":
            a = close.rolling(fast, min_periods=fast).mean()
            b = close.rolling(slow, min_periods=slow).mean()
        else:
            a = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
            b = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
        valid = a.notna() & b.notna()
        previous_valid = valid.shift(1, fill_value=False)
        crossed_above = valid & previous_valid & (a > b) & (a.shift(1) <= b.shift(1))
        crossed_below = valid & previous_valid & (a < b) & (a.shift(1) >= b.shift(1))
        return pd.Series(
            np.where(crossed_above, 1.0, np.where(crossed_below, -1.0, 0.0)),
            index=close.index,
        ).where(valid)
    length = int(params.get("rsi_length", 14))
    oversold = float(params.get("oversold", 30.0))
    overbought = float(params.get("overbought", 70.0))
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    rsi = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    valid = rsi.notna()
    previous = rsi.shift(1)
    entered_oversold = valid & previous.notna() & (rsi < oversold) & (previous >= oversold)
    entered_overbought = valid & previous.notna() & (rsi > overbought) & (previous <= overbought)
    return pd.Series(
        np.where(entered_oversold, 1.0, np.where(entered_overbought, -1.0, 0.0)),
        index=close.index,
    ).where(valid)


def _candidate_signal_definition(candidate: StrategyCandidate) -> str:
    if candidate.type == "forecast_threshold":
        return "forecast_threshold_anchor"
    if candidate.strategy in {"sma_cross", "ema_cross"}:
        return "cross_event"
    return "zone_entry_event"


_MAX_FORECAST_SIGNAL_ANCHORS = 200


def _forecast_signal(df: pd.DataFrame, candidate: StrategyCandidate, symbol: str, timeframe: str) -> pd.Series:
    from ..forecast.forecast import execute_forecast

    signal = pd.Series(np.nan, index=df.index, dtype=float)
    model_lookback = int(candidate.params.get("lookback", 200))
    params = {key: value for key, value in candidate.params.items() if key != "lookback"}
    eligible = list(range(model_lookback, len(df) - candidate.horizon, max(1, candidate.horizon)))
    if len(eligible) > _MAX_FORECAST_SIGNAL_ANCHORS:
        eligible = eligible[-_MAX_FORECAST_SIGNAL_ANCHORS:]
    for idx in eligible:
        history = df.iloc[: idx + 1].copy()
        try:
            result = execute_forecast(
                symbol=symbol,
                timeframe=timeframe,
                method=str(candidate.method),
                horizon=candidate.horizon,
                lookback=model_lookback,
                params=params,
                quantity="price",
                prefetched_df=history,
            )
            expected = result.get("expected_return")
            if expected is None:
                values = (
                    result.get("forecast_price")
                    or result.get("forecast")
                    or result.get("values")
                    or result.get("predictions")
                )
                if isinstance(values, list) and values:
                    expected = (float(values[-1]) - float(history["close"].iloc[-1])) / float(history["close"].iloc[-1])
            if expected is not None:
                value = float(expected)
                signal.iloc[idx] = 1.0 if value > candidate.long_above else -1.0 if value < candidate.short_below else 0.0
        except Exception:
            continue
    return signal


def _walk_forward_windows(
    start_bar: int,
    end_bar: int,
    *,
    n_splits: int,
    embargo: int,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    edges = np.linspace(
        int(start_bar),
        max(int(start_bar), int(end_bar) + 1),
        int(n_splits) + 2,
        dtype=int,
    )
    fold_windows: List[Tuple[int, int]] = []
    embargo_intervals: List[Tuple[int, int]] = []
    for fold in range(int(n_splits)):
        block_start = int(edges[fold + 1])
        test_start = block_start + int(embargo)
        test_end = int(edges[fold + 2]) - 1
        if embargo > 0:
            embargo_intervals.append((block_start, min(test_start - 1, test_end)))
        fold_windows.append((test_start, test_end))
    return fold_windows, embargo_intervals


def _barrier_returns(
    df: pd.DataFrame,
    signal: pd.Series,
    horizon: int,
    tp_pct: float,
    sl_pct: float,
    same_bar_policy: str = "sl_first",
) -> Tuple[np.ndarray, np.ndarray]:
    indices: List[int] = []
    outcomes: List[float] = []
    tp = float(tp_pct) / 100.0
    sl = float(sl_pct) / 100.0
    next_eligible_signal = 0
    for idx in range(len(df) - horizon):
        if idx < next_eligible_signal:
            continue
        direction = float(signal.iloc[idx]) if pd.notna(signal.iloc[idx]) else 0.0
        if direction == 0:
            continue
        entry_idx = idx + 1
        entry = float(df["open"].iloc[entry_idx])
        if not math.isfinite(entry) or entry <= 0.0:
            entry = float(df["close"].iloc[entry_idx])
        result = None
        for step in range(horizon):
            outcome_idx = entry_idx + step
            high = float(df["high"].iloc[outcome_idx])
            low = float(df["low"].iloc[outcome_idx])
            favorable = (high / entry - 1.0) if direction > 0 else (1.0 - low / entry)
            adverse = (1.0 - low / entry) if direction > 0 else (high / entry - 1.0)
            adverse_hit = adverse >= sl
            favorable_hit = favorable >= tp
            if adverse_hit and favorable_hit:
                if same_bar_policy == "tp_first":
                    result = tp
                elif same_bar_policy == "neutral":
                    result = 0.0
                else:
                    result = -sl
                break
            if adverse_hit:
                result = -sl
                break
            if favorable_hit:
                result = tp
                break
        if result is None:
            result = direction * (float(df["close"].iloc[idx + horizon]) / entry - 1.0)
        indices.append(idx)
        outcomes.append(float(result))
        # A persistent state is one position, not a fresh overlapping trade on
        # every bar.  The next entry may be considered only after this
        # position's full outcome window has ended.
        next_eligible_signal = idx + int(horizon)
    return np.asarray(indices, dtype=int), np.asarray(outcomes, dtype=float)


_MIN_HISTORICAL_SPREAD_COVERAGE = 0.9


def _observed_spread_bps(
    request: StrategyValidateRequest,
    gateway: Any,
    frame: pd.DataFrame,
) -> Tuple[Optional[float], str, bool, Dict[str, Any]]:
    if request.cost_model == "fixed":
        return (
            float(request.spread_bps),
            "explicit",
            True,
            {"basis": "request"},
        )
    spread_points = _finite(frame.get("spread", pd.Series(dtype=float)))
    close = _finite(frame.get("close", pd.Series(dtype=float)))
    try:
        point = float(getattr(gateway.symbol_info(request.symbol), "point", 0.0) or 0.0)
    except Exception:
        point = 0.0
    valid_mask = (
        np.isfinite(spread_points)
        & (spread_points > 0.0)
        & np.isfinite(close)
        & (close > 0.0)
    )
    observations = int(valid_mask.sum())
    total_bars = int(len(frame))
    coverage = float(observations / total_bars) if total_bars else 0.0
    window = {
        "basis": "historical_bar_spread",
        "start": (
            format_epoch_utc(float(frame["time"].iloc[0]))
            if total_bars and "time" in frame
            else None
        ),
        "end": (
            format_epoch_utc(float(frame["time"].iloc[-1]))
            if total_bars and "time" in frame
            else None
        ),
        "observations": observations,
        "bars": total_bars,
        "coverage_pct": round(coverage * 100.0, 2),
        "minimum_complete_coverage_pct": round(
            _MIN_HISTORICAL_SPREAD_COVERAGE * 100.0,
            2,
        ),
    }
    if observations and math.isfinite(point) and point > 0.0:
        spread_values = spread_points[valid_mask] * point / close[valid_mask] * 10_000.0
        return (
            float(np.median(spread_values)),
            "mt5_historical_bar_spread_median",
            coverage >= _MIN_HISTORICAL_SPREAD_COVERAGE,
            window,
        )
    return (
        None,
        "unavailable",
        False,
        window,
    )


def validate_strategies(  # noqa: C901
    request: StrategyValidateRequest, gateway: Any
) -> Dict[str, Any]:
    try:
        symbol_info = gateway.symbol_info(request.symbol)
    except Exception as exc:
        return {
            "success": False,
            "error": f"Could not validate symbol '{request.symbol}': {exc}",
            "error_code": "symbol_lookup_failed",
            "symbol": request.symbol,
            "remediation": "Check the MT5 connection and retry the symbol lookup.",
            "related_tools": ["symbols_list"],
        }
    if symbol_info is None:
        return {
            "success": False,
            "error": f"Symbol '{request.symbol}' was not found by MT5.",
            "error_code": "symbol_not_found",
            "symbol": request.symbol,
            "remediation": (
                "Use symbols_list to find the broker's exact symbol name and suffix."
            ),
            "related_tools": ["symbols_list"],
        }
    df = _rates(
        gateway,
        request.symbol,
        request.timeframe,
        request.lookback + request.barrier.horizon + 5,
        start=request.start,
        end=request.end,
    )
    if len(df) < 200:
        return {"error": "At least 200 completed bars are required.", "error_code": "insufficient_data"}
    spread_bps, spread_source, complete, spread_window = _observed_spread_bps(
        request,
        gateway,
        df,
    )
    if spread_bps is None:
        return {
            "success": False,
            "error": (
                "Transaction-cost spread is unavailable for the requested evaluation window. "
                "Provide spread_bps with cost_model='fixed' or use a window whose "
                "completed bars include historical spread observations."
            ),
            "error_code": "cost_model_unavailable",
            "cost_model": {
                "source": spread_source,
                "spread_bps": None,
                "window": spread_window,
                "complete": False,
            },
        }
    round_trip_bps = spread_bps + 2.0 * (request.commission_bps + request.slippage_bps)
    purge = int(request.purge_bars or 0)
    embargo = int(
        request.embargo_bars
        if request.embargo_bars is not None
        else request.barrier.horizon
    )
    labelable_end = len(df) - int(request.barrier.horizon) - 1
    fold_windows, embargo_intervals = _walk_forward_windows(
        0,
        labelable_end,
        n_splits=request.n_splits,
        embargo=embargo,
    )
    results = []
    for candidate in request.candidates:
        signal_definition = _candidate_signal_definition(candidate)
        signal = _builtin_signal(df["close"], candidate) if candidate.type == "builtin_strategy" else _forecast_signal(df, candidate, request.symbol, request.timeframe)
        candidate_fold_windows = fold_windows
        candidate_embargo_intervals = embargo_intervals
        valid_signal_bars = np.flatnonzero(signal.notna().to_numpy())
        signal_coverage = {
            "anchors_computed": int(len(valid_signal_bars)),
            "first_bar": int(valid_signal_bars[0]) if len(valid_signal_bars) else None,
            "last_bar": int(valid_signal_bars[-1]) if len(valid_signal_bars) else None,
            "anchor_limit": (
                _MAX_FORECAST_SIGNAL_ANCHORS
                if candidate.type == "forecast_threshold"
                else None
            ),
        }
        valid_signal_values = signal.iloc[valid_signal_bars].to_numpy(dtype=float)
        signal_counts = {
            "long": int(np.sum(valid_signal_values > 0.0)),
            "short": int(np.sum(valid_signal_values < 0.0)),
            "neutral": int(np.sum(valid_signal_values == 0.0)),
            "non_finite_or_unavailable": int(len(signal) - len(valid_signal_bars)),
        }
        if candidate.type == "forecast_threshold" and len(valid_signal_bars):
            candidate_fold_windows, candidate_embargo_intervals = _walk_forward_windows(
                int(valid_signal_bars[0]),
                labelable_end,
                n_splits=request.n_splits,
                embargo=embargo,
            )
        same_bar_policy = normalize_same_bar_policy(request.barrier.same_bar_policy)
        indices, gross = _barrier_returns(
            df,
            signal,
            request.barrier.horizon,
            request.barrier.tp_pct,
            request.barrier.sl_pct,
            same_bar_policy,
        )
        if len(indices) < request.n_splits * 5:
            if not len(valid_signal_bars):
                insufficient_reason = "forecast_unavailable_for_all_anchors"
            elif signal_counts["long"] + signal_counts["short"] == 0:
                insufficient_reason = "threshold_not_crossed"
            else:
                insufficient_reason = "too_few_non_overlapping_trades"
            results.append({
                "id": candidate.id,
                "evaluation_status": "insufficient_data",
                "signal_definition": signal_definition,
                "trades": int(len(indices)),
                "minimum_trades_required": int(request.n_splits * 5),
                "insufficient_data_reason": insufficient_reason,
                "signal_coverage": signal_coverage,
                "signal_counts": signal_counts,
            })
            continue
        fold_rows = []
        skipped_folds: List[Dict[str, Any]] = []
        all_net = []
        calibrated_probabilities: List[float] = []
        calibrated_labels: List[int] = []
        for fold, (test_start, test_end) in enumerate(candidate_fold_windows):
            if test_start > test_end:
                skipped_folds.append({"fold": fold + 1, "reason": "empty_test_window"})
                continue
            test_mask = (
                (indices >= test_start)
                & (indices + int(request.barrier.horizon) <= test_end)
            )
            test_indices = indices[test_mask]
            test_gross = gross[test_mask]
            if not len(test_indices):
                skipped_folds.append({"fold": fold + 1, "reason": "no_test_trades"})
                continue
            test = test_gross - round_trip_bps / 10_000.0
            train_mask = (
                indices + int(request.barrier.horizon) < int(test_start) - purge
            )
            embargo_excluded = np.zeros(len(indices), dtype=bool)
            for gap_start, gap_end in candidate_embargo_intervals:
                if gap_start >= test_start:
                    break
                embargo_excluded |= (indices >= gap_start) & (indices <= gap_end)
            train_mask &= ~embargo_excluded
            train_count = int(np.sum(train_mask))
            if train_count < 5:
                skipped_folds.append({
                    "fold": fold + 1,
                    "reason": "insufficient_training_trades",
                    "train_trades": train_count,
                })
                continue
            all_net.extend(test.tolist())
            if train_count >= 100:
                try:
                    from sklearn.linear_model import LogisticRegression

                    train_x = signal.iloc[indices[train_mask]].to_numpy(dtype=float).reshape(-1, 1)
                    train_net = gross[train_mask] - round_trip_bps / 10_000.0
                    train_y = (train_net > 0).astype(int)
                    test_x = signal.iloc[test_indices].to_numpy(dtype=float).reshape(-1, 1)
                    if len(np.unique(train_y)) > 1 and np.all(np.isfinite(train_x)) and np.all(np.isfinite(test_x)):
                        calibrator = LogisticRegression(random_state=42).fit(train_x, train_y)
                        calibrated_probabilities.extend(calibrator.predict_proba(test_x)[:, 1].tolist())
                        calibrated_labels.extend((test > 0).astype(int).tolist())
                except Exception:
                    pass
            fold_rows.append({
                "fold": fold + 1,
                "train_trades": train_count,
                "test_trades": int(len(test)),
                "test_start_bar": int(test_indices[0]),
                "test_end_bar": int(test_indices[-1]),
                "test_window_start_bar": int(test_start),
                "test_window_end_bar": int(test_end),
                "horizon_tail_excluded": int(request.barrier.horizon),
                "embargo_bars_excluded": int(embargo),
                "extra_purge_bars": int(purge),
                "net_expectancy": float(np.mean(test)),
                "win_rate": float(np.mean(test > 0)),
            })
        arr = np.asarray(all_net, dtype=float)
        if not len(arr):
            results.append({
                "id": candidate.id,
                "evaluation_status": "insufficient_data",
                "signal_definition": signal_definition,
                "trades": 0,
                "minimum_trades_required": int(request.n_splits * 5),
                "insufficient_data_reason": "no_evaluable_oos_folds",
                "signal_coverage": signal_coverage,
                "signal_counts": signal_counts,
                "skipped_folds": skipped_folds,
            })
            continue
        equity = np.cumprod(1.0 + np.clip(arr, -0.999, None))
        peaks = np.maximum.accumulate(equity)
        drawdown = equity / peaks - 1.0
        std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        per_trade_sharpe = float(np.mean(arr) / std) if std > 0 else 0.0
        sharpe = per_trade_sharpe if std > 0 else None
        mean_return_t_stat = (
            float(per_trade_sharpe * math.sqrt(len(arr))) if std > 0 else None
        )
        trials = max(1, len(request.candidates))
        gamma = 0.5772156649015329
        expected_max = 0.0
        if trials > 1:
            expected_max = (1.0 - gamma) * norm.ppf(1.0 - 1.0 / trials) + gamma * norm.ppf(1.0 - 1.0 / (trials * math.e))
            expected_max /= math.sqrt(max(1, len(arr)))
        moment_scale = float(np.std(arr))
        skewness = float(skew(arr, bias=False)) if len(arr) > 2 and moment_scale > 1e-12 else 0.0
        kurt = float(kurtosis(arr, fisher=False, bias=False)) if len(arr) > 3 and moment_scale > 1e-12 else 3.0
        psr_denom = math.sqrt(max(1e-12, 1.0 - skewness * per_trade_sharpe + ((kurt - 1.0) / 4.0) * per_trade_sharpe**2))
        deflated_probability = float(norm.cdf((per_trade_sharpe - expected_max) * math.sqrt(max(1, len(arr) - 1)) / psr_denom))
        expectancy_ci = _bootstrap_mean_ci(arr.tolist(), request.bootstrap_samples)
        mean_return_p_value = _block_bootstrap_positive_mean_p_value(
            arr.tolist(), request.bootstrap_samples
        )
        fold_expectancies = [item["net_expectancy"] for item in fold_rows]
        folds_evaluated = int(len(fold_rows))
        fold_coverage = float(folds_evaluated / request.n_splits)
        fold_stability = float(
            np.sum(np.asarray(fold_expectancies) > 0) / request.n_splits
        ) if fold_expectancies else 0.0
        base_rate_stability = {"status": "insufficient_data", "observations": len(calibrated_labels)}
        if calibrated_labels:
            probs = np.asarray(calibrated_probabilities, dtype=float)
            labels = np.asarray(calibrated_labels, dtype=float)
            distinct_probabilities = int(len(np.unique(np.round(probs, 12))))
            base_rate_stability = {
                "status": "available",
                "observations": len(labels),
                "method": "direction_group_train_base_rate",
                "base_rate_brier_score": float(np.mean((probs - labels) ** 2)),
                "weighted_base_rate_gap": float(abs(np.mean(probs) - np.mean(labels))),
                "distinct_probabilities": distinct_probabilities,
                "label_basis": "net_return_after_costs_positive",
                "interpretation": "Long/short win-rate stability across folds; not continuous-score calibration.",
            }
        results.append({
            "id": candidate.id,
            "type": candidate.type,
            "evaluation_status": (
                "complete" if folds_evaluated == request.n_splits else "partial"
            ),
            "signal_definition": signal_definition,
            "trades": int(len(arr)),
            "net_expectancy": float(np.mean(arr)),
            "expectancy_ci_95": expectancy_ci,
            "win_rate": float(np.mean(arr > 0)),
            "profit_factor": float(arr[arr > 0].sum() / abs(arr[arr < 0].sum())) if np.any(arr < 0) else None,
            "sharpe": sharpe,
            "mean_return_t_stat": mean_return_t_stat,
            "deflated_sharpe_probability": deflated_probability,
            "mean_return_p_value": mean_return_p_value,
            "max_drawdown": float(np.min(drawdown)),
            "fold_stability": fold_stability,
            "folds_requested": int(request.n_splits),
            "folds_evaluated": folds_evaluated,
            "fold_coverage": fold_coverage,
            "signal_coverage": signal_coverage,
            "signal_counts": signal_counts,
            "skipped_folds": skipped_folds,
            "same_bar_policy": same_bar_policy,
            "direction_base_rate_stability": base_rate_stability,
            **({"folds": fold_rows} if request.detail == "full" else {}),
        })
    eligible_p = sorted(
        [(idx, float(item["mean_return_p_value"])) for idx, item in enumerate(results) if item.get("mean_return_p_value") is not None],
        key=lambda pair: pair[1],
    )
    running = 0.0
    for rank, (idx, p_value) in enumerate(eligible_p):
        adjusted = min(1.0, p_value * (len(eligible_p) - rank))
        running = max(running, adjusted)
        results[idx]["holm_adjusted_p_value"] = running
    for item in results:
        if item.get("evaluation_status") not in {"complete", "partial"}:
            continue
        ci = item.get("expectancy_ci_95")
        fold_share = float(item.get("fold_stability") or 0.0)
        adjusted_p = item.get("holm_adjusted_p_value")
        criteria = {
            "cost_model_complete": bool(complete),
            "all_requested_folds_evaluated": bool(
                int(item.get("folds_evaluated") or 0) == request.n_splits
            ),
            "expectancy_ci_above_zero": bool(ci and float(ci[0]) > 0.0),
            "holm_adjusted_p_at_most_alpha": bool(
                adjusted_p is not None
                and float(adjusted_p) <= request.significance_alpha
            ),
            "positive_fold_share_at_least_minimum": bool(
                fold_share >= request.min_positive_fold_share
            ),
        }
        if all(criteria.values()):
            classification = "positive"
        elif ci and float(ci[1]) < 0.0:
            classification = "negative"
        else:
            classification = "inconclusive"
        item["evidence"] = {
            "classification": classification,
            "criteria": criteria,
            "provisional_positive_before_complete_costs": bool(
                not complete
                and all(
                    value
                    for name, value in criteria.items()
                    if name != "cost_model_complete"
                )
            ),
            "significance_alpha": float(request.significance_alpha),
            "minimum_positive_fold_share": float(request.min_positive_fold_share),
        }
    ranked = sorted(results, key=lambda item: (item.get("net_expectancy") is None, -(item.get("net_expectancy") or -1e9)))
    warnings_out: List[str] = []
    if not complete:
        warnings_out.append(
            "Historical spread coverage is below 90%; positive classification "
            "is disabled. Use cost_model='fixed' with an explicit spread_bps for "
            "a controlled complete-cost comparison."
        )
    for item in results:
        folds_evaluated = int(item.get("folds_evaluated") or 0)
        if item.get("evaluation_status") == "partial":
            warnings_out.append(
                f"Candidate {item.get('id')} evaluated {folds_evaluated} of "
                f"{request.n_splits} requested folds; positive classification is disabled."
            )
    return {
        "success": True,
        "symbol": request.symbol,
        "timeframe": request.timeframe,
        "rankings": ranked,
        "validation": {
            "protocol": "anchored_expanding_fixed_candidate_oos",
            "n_splits": request.n_splits,
            "outcome_horizon_bars": int(request.barrier.horizon),
            "extra_purge_bars": purge,
            "embargo_bars": embargo,
            "candidate_parameters_reestimated": False,
            "forecast_models_refit_per_anchor": any(
                item.type == "forecast_threshold" for item in request.candidates
            ),
            "forecast_signal_anchor_limit": _MAX_FORECAST_SIGNAL_ANCHORS,
            "same_bar_policy": request.barrier.same_bar_policy,
            "completed_candles_only": True,
            "signal_timing": "completed_bar_close",
            "execution_timing": "next_bar_open",
            "barrier_window": "entry_bar_through_horizon",
        },
        "cost_model": {"source": spread_source, "spread_bps": spread_bps, "commission_bps_per_side": request.commission_bps, "slippage_bps_per_side": request.slippage_bps, "round_trip_bps": round_trip_bps, "window": spread_window, "complete": complete},
        "data_quality": {"bars": len(df), "cost_model_complete": complete},
        "units": {
            "net_expectancy": "return_fraction_per_trade",
            "max_drawdown": "return_fraction",
            "sharpe": "mean_net_return_per_trade_divided_by_per_trade_standard_deviation",
            "mean_return_t_stat": "dimensionless_test_statistic",
            "trades": "non_overlapping_positions",
        },
        "warnings": warnings_out,
    }


def _position_side(row: Dict[str, Any], gateway: Any) -> str:
    value = row.get("type")
    return "buy" if str(value).lower() in {"buy", "0", str(getattr(gateway, "POSITION_TYPE_BUY", 0))} else "sell"


def _position_sensitivity(gateway: Any, row: Dict[str, Any]) -> Tuple[Optional[float], Optional[str]]:
    symbol = str(row.get("symbol") or "")
    volume = float(row.get("volume") or 0.0)
    side = _position_side(row, gateway)
    raw_tick = gateway.symbol_info_tick(symbol)
    tick, _ = resolve_quote_tick(
        gateway,
        symbol,
        raw_tick,
        now_epoch=datetime.now(timezone.utc).timestamp(),
    )
    price = float(getattr(tick, "bid" if side == "sell" else "ask", 0.0) or row.get("price_current") or 0.0)
    if not symbol or volume <= 0 or price <= 0:
        return None, "missing symbol, volume, or mark price"
    action = getattr(gateway, "ORDER_TYPE_BUY", 0) if side == "buy" else getattr(gateway, "ORDER_TYPE_SELL", 1)
    up = gateway.order_calc_profit(action, symbol, volume, price, price * 1.0001)
    down = gateway.order_calc_profit(action, symbol, volume, price, price * 0.9999)
    if up is None or down is None:
        return None, "order_calc_profit unavailable"
    up_sens = float(up) / 0.0001
    down_sens = float(down) / -0.0001
    scale = max(abs(up_sens), abs(down_sens), 1e-12)
    if abs(up_sens - down_sens) / scale > 0.05:
        return None, "nonlinear or asymmetric P&L response"
    return float((up_sens + down_sens) / 2.0), None


def _nearest_broker_volume(
    requested: float,
    *,
    minimum: Optional[float],
    maximum: Optional[float],
    step: Optional[float],
) -> Optional[float]:
    """Return the closest positive volume that satisfies known broker bounds."""
    if step is None:
        candidate = requested
        if minimum is not None:
            candidate = max(candidate, minimum)
        if maximum is not None:
            candidate = min(candidate, maximum)
        return float(f"{candidate:.10f}") if candidate > 0.0 else None

    nearby_steps = {
        math.floor(requested / step),
        math.ceil(requested / step),
        round(requested / step),
    }
    if minimum is not None:
        nearby_steps.add(math.ceil(minimum / step - 1e-12))
    if maximum is not None:
        nearby_steps.add(math.floor(maximum / step + 1e-12))
    candidates = []
    for count in nearby_steps:
        candidate = float(f"{float(count) * step:.10f}")
        if candidate <= 0.0:
            continue
        if minimum is not None and candidate < minimum - 1e-12:
            continue
        if maximum is not None and candidate > maximum + 1e-12:
            continue
        candidates.append(candidate)
    if not candidates:
        return None
    return min(candidates, key=lambda value: (abs(value - requested), value))


def _validate_proposed_trade(
    gateway: Any,
    proposed: Any,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Resolve a proposed trade and validate its volume before risk simulation."""
    from ..core.trading.validation import _validate_volume, coerce_finite_float
    from ..utils.mt5 import resolve_broker_symbol_name

    input_symbol = str(proposed.symbol)
    resolved_symbol = resolve_broker_symbol_name(input_symbol, gateway=gateway)
    try:
        symbol_info = gateway.symbol_info(resolved_symbol)
    except Exception:
        symbol_info = None
    if symbol_info is None:
        return None, {
            "error": f"Symbol {input_symbol!r} was not found by MT5.",
            "error_code": "symbol_not_found",
            "field": "proposed_trade.symbol",
            "symbol": input_symbol,
            "remediation": "Use symbols_list to discover the exact broker symbol name.",
        }

    volume, volume_error = _validate_volume(proposed.volume, symbol_info)
    if volume_error is not None:
        minimum = coerce_finite_float(getattr(symbol_info, "volume_min", None))
        maximum = coerce_finite_float(getattr(symbol_info, "volume_max", None))
        step = coerce_finite_float(getattr(symbol_info, "volume_step", None))
        minimum = minimum if minimum is not None and minimum > 0.0 else None
        maximum = maximum if maximum is not None and maximum > 0.0 else None
        step = step if step is not None and step > 0.0 else None
        requested = float(proposed.volume)
        nearest = _nearest_broker_volume(
            requested,
            minimum=minimum,
            maximum=maximum,
            step=step,
        )
        return None, {
            "error": (
                f"Invalid proposed_trade.volume for {resolved_symbol}: "
                f"{volume_error}."
            ),
            "error_code": "invalid_proposed_trade_volume",
            "field": "proposed_trade.volume",
            "symbol": resolved_symbol,
            "requested_volume": requested,
            "constraints": {
                "volume_min": minimum,
                "volume_max": maximum,
                "volume_step": step,
            },
            "nearest_valid_volume": nearest,
            "remediation": (
                "Choose a lot size within the broker range and aligned to "
                "volume_step."
            ),
        }

    return {
        "symbol": resolved_symbol,
        "symbol_input": input_symbol,
        "side": proposed.side,
        "volume": float(volume),
    }, None


def decompose_portfolio_risk(  # noqa: C901
    request: PortfolioRiskDecomposeRequest,
    gateway: Any,
) -> Dict[str, Any]:
    holding_periods = [
        f"{horizon} {request.timeframe} bar{'s' if horizon != 1 else ''}"
        for horizon in request.horizon_bars
    ]
    model_context: Dict[str, Any] = {
        "timeframe": request.timeframe,
        "horizon_bars": list(request.horizon_bars),
        "holding_periods": holding_periods,
        "lookback_requested": request.lookback,
        "confidence_levels": list(request.confidence),
        "simulations": request.simulations,
        "ewma_half_life": request.ewma_half_life,
        "random_seed": request.seed,
        "completion_policy": "allow_partial" if request.allow_partial else "fail_closed",
    }
    proposed = request.proposed_trade
    proposed_validated: Optional[Dict[str, Any]] = None
    if proposed is not None:
        proposed_validated, proposed_error = _validate_proposed_trade(
            gateway,
            proposed,
        )
        if proposed_error is not None:
            return proposed_error
    account = None
    try:
        account_info = getattr(gateway, "account_info", None)
        if callable(account_info):
            account = account_info()
    except Exception:
        account = None
    positions = [_mapping(row) for row in (gateway.positions_get() or [])]
    base_position_count = len(positions)
    if proposed_validated is not None:
        proposed_symbol = str(proposed_validated["symbol"])
        proposed_side = str(proposed_validated["side"])
        tick = gateway.symbol_info_tick(proposed_symbol)
        positions.append({
            "ticket": "proposed",
            "symbol": proposed_symbol,
            "type": getattr(gateway, "POSITION_TYPE_BUY", 0) if proposed_side == "buy" else getattr(gateway, "POSITION_TYPE_SELL", 1),
            "volume": proposed_validated["volume"],
            "price_current": getattr(tick, "ask" if proposed_side == "buy" else "bid", None),
            "proposed": True,
        })
    all_positions = list(positions)
    total_position_count = len(all_positions)
    requested_symbols = sorted(
        {
            str(row.get("symbol") or "")
            for row in all_positions
            if row.get("symbol")
        }
    )
    mark_context = _portfolio_mark_context(gateway, all_positions)
    model_context.update(mark_context)
    unusable_mark_symbols = {
        str(item.get("symbol") or "")
        for item in mark_context.get("mark_freshness", [])
        if item.get("usable_for_live_trading") is not True
    }
    mark_omissions = [
        {
            "symbol": str(item.get("symbol") or ""),
            "stage": "mark_freshness",
            "reason": item.get("freshness_reason") or "mark_not_live_ready",
            "freshness_state": item.get("freshness_state"),
            "data_age_seconds": item.get("data_age_seconds"),
            "quote_source": item.get("quote_source"),
        }
        for item in mark_context.get("mark_freshness", [])
        if item.get("usable_for_live_trading") is not True
    ]
    if mark_omissions and not request.allow_partial:
        return {
            "error": "One or more material position marks are not live-ready.",
            "error_code": "portfolio_mark_unusable",
            "failures": mark_omissions,
            "model_context": model_context,
            "remediation": (
                "Refresh MT5 quotes or set allow_partial=true to omit unsafe marks."
            ),
        }
    if mark_omissions:
        positions = [
            row
            for row in all_positions
            if str(row.get("symbol") or "") not in unusable_mark_symbols
        ]
        if not positions:
            return {
                "success": True,
                "empty": True,
                "partial_failure": True,
                "positions": base_position_count,
                "message": "No positions had a live-ready mark.",
                "summary": {
                    "positions": base_position_count,
                    "positions_after_proposed": total_position_count,
                    "symbols": 0,
                    "symbols_requested": len(requested_symbols),
                },
                "risk": [],
                "timeframe": request.timeframe,
                "holding_periods": holding_periods,
                "model_context": model_context,
                "data_quality": {
                    "allow_partial": True,
                    "mark_omissions": mark_omissions,
                    "symbols_requested": requested_symbols,
                    "symbols_modeled": [],
                    "symbols_omitted": requested_symbols,
                },
                "warnings": [
                    "All material positions were omitted because their marks were not live-ready."
                ],
            }
    if not positions:
        return {
            "success": True,
            "empty": True,
            "positions": 0,
            "message": "No open positions.",
            "summary": {"positions": 0},
            "risk": [],
            "timeframe": request.timeframe,
            "holding_periods": holding_periods,
            "model_context": model_context,
        }
    sensitivities: Dict[str, float] = {}
    proposed_sensitivity: Optional[Tuple[str, float]] = None
    failures = []
    for row in positions:
        sensitivity, error = _position_sensitivity(gateway, row)
        symbol = str(row.get("symbol") or "")
        if error or sensitivity is None:
            failures.append({"symbol": symbol, "ticket": row.get("ticket"), "reason": error})
            continue
        sensitivities[symbol] = sensitivities.get(symbol, 0.0) + sensitivity
        if row.get("proposed"):
            proposed_sensitivity = (symbol, float(sensitivity))
    if failures and not request.allow_partial:
        return {"error": "One or more material positions could not be priced safely.", "error_code": "portfolio_pricing_incomplete", "failures": failures}
    series = {}
    history_failures: List[Dict[str, Any]] = []
    for symbol in sensitivities:
        bars = _rates(gateway, symbol, request.timeframe, request.lookback + max(request.horizon_bars) + 5)
        if len(bars) >= 100:
            values = pd.Series(np.log(bars["close"]).diff().to_numpy(), index=bars["time"].to_numpy(), name=symbol).dropna()
            series[symbol] = values
        else:
            history_failures.append({
                "symbol": symbol,
                "stage": "return_history",
                "bars_available": int(len(bars)),
                "bars_required": 100,
                "reason": "insufficient completed return history",
            })
    if history_failures and not request.allow_partial:
        return {
            "error": "One or more material positions lacked sufficient return history.",
            "error_code": "portfolio_pricing_incomplete",
            "failures": history_failures,
        }
    if not series:
        return {
            "error": "No aligned return history was available.",
            "error_code": "insufficient_data",
            "failures": history_failures,
        }
    returns_available = pd.concat(series.values(), axis=1, join="inner").dropna()
    if len(returns_available) < 100:
        return {"error": "At least 100 aligned returns are required.", "error_code": "insufficient_data", "aligned_rows": len(returns_available)}
    returns_available.columns = list(series)
    # Extra leading observations are fetched only to warm up volatility and
    # multi-bar calculations. The requested lookback is the stable calibration
    # window and must not change when another horizon is added.
    returns = returns_available.tail(int(request.lookback)).copy()
    alpha = 1.0 - math.exp(math.log(0.5) / request.ewma_half_life)
    standardized, current_vol = _filtered_historical_returns(
        returns_available,
        alpha=alpha,
    )
    standardized = standardized.tail(int(request.lookback)).copy()
    ewma_vol = current_vol.copy()
    if request.method == "historical":
        standardized = returns.copy()
        current_vol = pd.Series(1.0, index=returns.columns)
    rng = np.random.default_rng(request.seed)
    risk_rows = []
    scenario_details: Dict[int, np.ndarray] = {}
    sensitivity_vec = np.asarray([sensitivities[column] for column in standardized.columns], dtype=float)
    for horizon in request.horizon_bars:
        max_start = len(standardized) - horizon
        if max_start < 1:
            continue
        starts = rng.integers(0, max_start + 1, size=request.simulations)
        scenario_returns = np.stack([standardized.iloc[start : start + horizon].sum(axis=0).to_numpy(dtype=float) for start in starts])
        scenario_returns = scenario_returns * current_vol.to_numpy(dtype=float)
        scenario_simple_returns = np.expm1(scenario_returns)
        component_pnl = scenario_simple_returns * sensitivity_vec
        pnl = component_pnl.sum(axis=1)
        base_pnl = pnl.copy()
        if proposed_sensitivity and proposed_sensitivity[0] in list(standardized.columns):
            proposed_idx = list(standardized.columns).index(proposed_sensitivity[0])
            base_pnl = pnl - component_pnl[:, proposed_idx]
        scenario_details[horizon] = pnl
        for confidence in request.confidence:
            cutoff = float(np.quantile(pnl, 1.0 - confidence))
            tail = pnl <= cutoff
            es_components = -np.mean(component_pnl[tail], axis=0) if np.any(tail) else np.zeros(len(sensitivity_vec))
            base_cutoff = float(np.quantile(base_pnl, 1.0 - confidence))
            base_tail = base_pnl <= base_cutoff
            base_es = float(max(0.0, -np.mean(base_pnl[base_tail]))) if np.any(base_tail) else None
            after_es = float(max(0.0, -np.mean(pnl[tail]))) if np.any(tail) else None
            risk_rows.append({
                "horizon_bars": horizon,
                "holding_period": (
                    f"{horizon} {request.timeframe} "
                    f"bar{'s' if horizon != 1 else ''}"
                ),
                "confidence": confidence,
                "calibration_observations": int(len(standardized)),
                "horizon_windows_available": int(max_start + 1),
                "var": float(max(0.0, -cutoff)),
                "expected_shortfall": after_es,
                **({"before_expected_shortfall": base_es, "incremental_expected_shortfall": (after_es - base_es) if after_es is not None and base_es is not None else None} if proposed_sensitivity else {}),
                "component_expected_shortfall": [
                    {"symbol": symbol, "value": float(value)} for symbol, value in zip(standardized.columns, es_components)
                ],
                "worst_simulated_pnl": float(np.min(pnl)),
            })
    exposure_abs = np.abs(sensitivity_vec)
    weights = exposure_abs / exposure_abs.sum() if exposure_abs.sum() else exposure_abs
    correlation = returns.corr()
    worst_historical = (np.expm1(returns) * sensitivity_vec).sum(axis=1)
    perfect_correlation = []
    for horizon in request.horizon_bars:
        if request.method == "filtered_historical":
            horizon_vol = ewma_vol * math.sqrt(float(horizon))
        else:
            horizon_vol = returns.rolling(int(horizon)).sum().std(ddof=1)
        horizon_vol = horizon_vol.reindex(standardized.columns).fillna(0.0)
        signed_loading = float(
            np.dot(sensitivity_vec, horizon_vol.to_numpy(dtype=float))
        )
        shock_direction = -1.0 if signed_loading >= 0.0 else 1.0
        perfect_correlation.append({
            "horizon_bars": int(horizon),
            "shock_sigma": 1.0,
            "common_factor_direction": shock_direction,
            "pnl": float(-abs(signed_loading)),
            "marginal_volatility": {
                str(symbol): float(value)
                for symbol, value in horizon_vol.items()
            },
        })
    stresses = {
        "volatility_double_worst_pnl": float(min(np.min(values) * 2.0 for values in scenario_details.values())),
        "perfect_positive_correlation_1sigma": perfect_correlation,
        "worst_historical_bar_pnl": float(worst_historical.min()),
    }
    proposed_context = None
    if proposed_validated is not None:
        try:
            proposed_symbol = str(proposed_validated["symbol"])
            proposed_side = str(proposed_validated["side"])
            proposed_volume = float(proposed_validated["volume"])
            tick = gateway.symbol_info_tick(proposed_symbol)
            action = getattr(gateway, "ORDER_TYPE_BUY", 0) if proposed_side == "buy" else getattr(gateway, "ORDER_TYPE_SELL", 1)
            price = float(getattr(tick, "ask" if proposed_side == "buy" else "bid"))
            margin = gateway.order_calc_margin(action, proposed_symbol, proposed_volume, price)
            proposed_context = {
                "symbol": proposed_symbol,
                "side": proposed_side,
                "volume": proposed_volume,
                "margin_required": float(margin) if margin is not None else None,
            }
            if proposed_validated["symbol_input"] != proposed_symbol:
                proposed_context["symbol_input"] = proposed_validated["symbol_input"]
        except Exception:
            proposed_context = {
                "symbol": proposed_validated["symbol"],
                "side": proposed_validated["side"],
                "volume": proposed_validated["volume"],
                "margin_required": None,
            }
    account_context = {
        key: value
        for key, value in {
            "currency": getattr(account, "currency", None),
            "equity": getattr(account, "equity", None),
        }.items()
        if value is not None
    }
    modeled_symbols = [str(column) for column in standardized.columns]
    omitted_symbols = sorted(set(requested_symbols) - set(modeled_symbols))
    warnings_out: List[str] = []
    if mark_omissions:
        warnings_out.append(
            "Some positions had non-live marks and were omitted because allow_partial=true."
        )
    if failures:
        warnings_out.append(
            "Some positions could not be priced and were omitted because allow_partial=true."
        )
    if history_failures:
        warnings_out.append(
            "Some priced symbols lacked sufficient return history and were omitted because allow_partial=true."
        )
    data_start = format_epoch_utc(float(returns.index[0]))
    data_end = format_epoch_utc(float(returns.index[-1]))
    model_context.update(
        {
            "aligned_returns": len(returns),
            "aligned_returns_available": len(returns_available),
            "warmup_returns_discarded": int(len(returns_available) - len(returns)),
            "data_start": data_start,
            "data_end": data_end,
        }
    )
    return {
        "success": True,
        "method": request.method,
        "timeframe": request.timeframe,
        "holding_periods": holding_periods,
        "model_context": model_context,
        **account_context,
        "summary": {"positions": base_position_count, "positions_after_proposed": total_position_count, "symbols": len(modeled_symbols), "symbols_requested": len(requested_symbols), "aligned_rows": len(returns), "concentration_hhi": float(np.sum(weights**2))},
        "risk": risk_rows,
        "stresses": stresses,
        "proposed_trade": proposed_context,
        "data_quality": {
            "pricing_failures": failures,
            "history_failures": history_failures,
            "mark_omissions": mark_omissions,
            "allow_partial": request.allow_partial,
            "symbols_requested": requested_symbols,
            "symbols_modeled": modeled_symbols,
            "symbols_omitted": omitted_symbols,
            "aligned_coverage": float(len(returns) / max(len(item) for item in series.values())),
        },
        "warnings": warnings_out,
        "units": {
            "var": "account_currency",
            "expected_shortfall": "account_currency",
            "sensitivity": "account_currency_per_1.0_return",
            "stresses": "account_currency",
        },
        **({"correlation": correlation.to_dict()} if request.detail == "full" else {}),
    }


def _robust_z(values: pd.Series) -> pd.Series:
    if values.empty:
        return values.astype(float)
    clipped = values.clip(values.quantile(0.05), values.quantile(0.95))
    median = clipped.median()
    mad = float(np.median(np.abs(clipped - median)))
    if mad <= 1e-12:
        std = float(clipped.std())
        return (clipped - median) / std if std > 0 else clipped * 0.0
    return (clipped - median) / (1.4826 * mad)


def _relative_strength_history_window(
    symbol: str,
    bars: pd.DataFrame,
    *,
    timeframe: str,
    now_epoch: float,
) -> Dict[str, Any]:
    if bars.empty or "time" not in bars:
        return {"bars_available": 0, "freshness": "unavailable"}
    start_epoch = float(bars["time"].iloc[0])
    latest_open_epoch = float(bars["time"].iloc[-1])
    latest_close_epoch = bar_close_epoch(latest_open_epoch, timeframe)
    signed_age_seconds = float(now_epoch) - latest_close_epoch
    age_seconds = max(0.0, signed_age_seconds)
    stale_after_seconds = max(1, int(TIMEFRAME_SECONDS[timeframe]) * 2)
    timestamp_in_future = signed_age_seconds < 0.0
    stale = timestamp_in_future or age_seconds > stale_after_seconds
    closed_session = closed_session_context(
        symbol,
        now_epoch=now_epoch,
        item="bar",
        data_age_seconds=None if timestamp_in_future else age_seconds,
    )
    policy_relaxed = bool(
        closed_session and closed_session.get("freshness_policy_relaxed")
    )
    if timestamp_in_future:
        freshness = "future_timestamp"
    elif stale and policy_relaxed:
        freshness = "closed_session_snapshot"
    elif stale:
        freshness = "stale"
    else:
        freshness = "fresh"
    context: Dict[str, Any] = {
        "bars_available": int(len(bars)),
        "history_start": format_epoch_utc(start_epoch),
        "latest_bar_open": format_epoch_utc(latest_open_epoch),
        "latest_bar_close": format_epoch_utc(latest_close_epoch),
        "latest_bar_age_seconds": round(age_seconds, 3),
        "stale_after_seconds": stale_after_seconds,
        "freshness": freshness,
    }
    if timestamp_in_future:
        context["timestamp_in_future"] = True
        context["timestamp_skew_seconds"] = round(-signed_age_seconds, 3)
    if closed_session:
        context["market_status"] = closed_session.get("market_status")
        context["market_status_reason"] = closed_session.get(
            "market_status_reason"
        )
        context["freshness_policy_relaxed"] = policy_relaxed
    return context


def _relative_strength_quote_status(quote_quality: Dict[str, Any]) -> str:
    if quote_quality.get("usable_for_live_trading") is True:
        return "live_ready"
    spread_quality = quote_quality.get("spread_quality")
    if spread_quality == "locked":
        return "locked_quote"
    if spread_quality not in (None, "two_sided"):
        return "invalid_quote"
    if isinstance(quote_quality.get("quote_source_conflict"), dict):
        return "conflicting_quote_sources"
    return str(
        quote_quality.get("freshness_state")
        or quote_quality.get("freshness_reason")
        or "not_live_ready"
    )


def _project_relative_strength_row(
    row: Dict[str, Any],
    *,
    detail: str,
) -> Dict[str, Any]:
    if detail == "full":
        return row

    quote_quality = row.get("quote_quality")
    data_window = row.get("data_window")
    out = {
        key: row[key]
        for key in ("symbol", "rank", "score", "rank_percentile")
        if key in row
    }
    out["quote_status"] = _relative_strength_quote_status(
        quote_quality if isinstance(quote_quality, dict) else {}
    )
    if isinstance(data_window, dict) and data_window.get("freshness") is not None:
        out["history_status"] = data_window["freshness"]
    if detail == "summary":
        return out

    for key in (
        "rank_stability",
        "raw_momentum",
        "residual_momentum",
        "spread_pct",
    ):
        if key in row:
            out[key] = row[key]
    if detail == "standard":
        for key in (
            "beta",
            "volatility",
            "tick_volume",
            "above_sma20",
            "above_sma50",
        ):
            if key in row:
                out[key] = row[key]
    return out


def rank_relative_strength(  # noqa: C901
    request: MarketRelativeStrengthRequest, gateway: Any
) -> Dict[str, Any]:
    raw_symbols = list(gateway.symbols_get() or [])
    explicit = {item.strip().upper() for item in str(request.symbols or "").split(",") if item.strip()}
    available_names = {
        str(_mapping(item).get("name") or getattr(item, "name", "")).upper()
        for item in raw_symbols
        if str(_mapping(item).get("name") or getattr(item, "name", "")).strip()
    }
    missing_explicit = sorted(explicit - available_names)
    selected = []
    for item in raw_symbols:
        row = _mapping(item)
        name = str(row.get("name") or getattr(item, "name", "")).upper()
        path = str(row.get("path") or getattr(item, "path", ""))
        visible = bool(row.get("visible", getattr(item, "visible", False)))
        if explicit and name not in explicit:
            continue
        if request.group and str(request.group).lower() not in path.lower():
            continue
        if request.universe == "visible" and not visible and not explicit:
            continue
        selected.append(name)
        if len(selected) >= request.max_symbols:
            break
    benchmark_symbol = request.benchmark.upper() if request.benchmark else None
    if benchmark_symbol and benchmark_symbol not in available_names:
        return {
            "error": f"Requested benchmark {benchmark_symbol!r} is unavailable.",
            "error_code": "benchmark_not_found",
            "benchmark": benchmark_symbol,
            "remediation": "Use symbols_list to verify the benchmark's broker symbol name.",
        }
    candidate_symbols = [
        symbol for symbol in selected if symbol != benchmark_symbol
    ]
    requested_candidates = (explicit & available_names) - (
        {benchmark_symbol} if benchmark_symbol else set()
    )
    omitted_explicit = sorted(requested_candidates - set(candidate_symbols))
    if omitted_explicit:
        return {
            "error": "The explicit candidate basket exceeds the selected symbol limit.",
            "error_code": "candidate_limit_exceeded",
            "missing_symbols": omitted_explicit,
            "remediation": "Increase max_symbols or submit a smaller explicit basket.",
        }
    if explicit and not candidate_symbols:
        return {
            "error": "None of the requested candidate symbols are available.",
            "error_code": "symbol_not_found",
            "missing_symbols": missing_explicit or sorted(explicit),
            "remediation": "Use symbols_list to discover broker symbol names and suffixes.",
        }
    data_symbols = list(candidate_symbols)
    if benchmark_symbol and benchmark_symbol not in data_symbols:
        data_symbols.append(benchmark_symbol)
    lookback = max(max(request.horizons) + request.volatility_lookback + 15, 100)
    analysis_started_at = datetime.now(timezone.utc)
    analysis_started_epoch = analysis_started_at.timestamp()
    histories: Dict[str, pd.DataFrame] = {}
    history_windows: Dict[str, Dict[str, Any]] = {}
    skipped = []
    for symbol in data_symbols:
        bars = _rates(gateway, symbol, request.timeframe, lookback)
        history_window = _relative_strength_history_window(
            symbol,
            bars,
            timeframe=request.timeframe,
            now_epoch=analysis_started_epoch,
        )
        history_window["bars_requested"] = int(lookback)
        history_windows[symbol] = history_window
        if len(bars) < int(lookback * 0.90):
            skipped.append(
                {
                    "symbol": symbol,
                    "reason": "history coverage below 90%",
                    "data_window": history_window,
                }
            )
            continue
        histories[symbol] = bars
    candidate_histories = {
        symbol: histories[symbol]
        for symbol in candidate_symbols
        if symbol in histories
    }
    missing_candidate_history = sorted(set(candidate_symbols) - set(candidate_histories))
    if benchmark_symbol and benchmark_symbol not in histories:
        return {
            "error": f"Requested benchmark {benchmark_symbol!r} lacks sufficient history.",
            "error_code": "benchmark_history_unavailable",
            "benchmark": benchmark_symbol,
            "skipped": skipped,
        }
    if len(candidate_histories) < 2:
        return {
            "error": "At least two available symbols with sufficient history are required.",
            "error_code": "insufficient_data",
            "missing_symbols": sorted(
                set(missing_explicit) | set(missing_candidate_history)
            ),
            "skipped": skipped,
        }
    quote_excluded_symbols: List[str] = []
    quote_contexts: Dict[str, Dict[str, Any]] = {}
    scoring_histories: Dict[str, pd.DataFrame] = {}
    for symbol, bars in candidate_histories.items():
        latest = bars.iloc[-1]
        try:
            raw_tick = gateway.symbol_info_tick(symbol)
        except Exception:
            raw_tick = None
        quote_query_epoch = datetime.now(timezone.utc).timestamp()
        tick, quote_source = resolve_quote_tick(
            gateway,
            symbol,
            raw_tick,
            now_epoch=quote_query_epoch,
        )
        quote_quality = build_tick_freshness_context(
            symbol,
            tick_epoch=tick_epoch(tick),
            now_epoch=datetime.now(timezone.utc).timestamp(),
        )
        quote_quality.update(quote_source)
        try:
            bid = float(tick_value(tick, "bid") or 0.0)
            ask = float(tick_value(tick, "ask") or 0.0)
        except (TypeError, ValueError):
            bid = ask = 0.0
        spread_pct = (
            (ask - bid) / ((ask + bid) / 2.0) * 100.0
            if ask > bid > 0.0
            else None
        )
        enforce_quote_execution_readiness(
            quote_quality,
            bid=bid,
            ask=ask,
            quote_source_conflict=quote_quality.get("quote_source_conflict"),
        )
        symbol_window = history_windows[symbol]
        if quote_quality.get("usable_for_live_trading") is not True:
            quote_excluded_symbols.append(symbol)
        if request.max_spread_pct is not None and (
            spread_pct is None or spread_pct > request.max_spread_pct
        ):
            skipped.append(
                {
                    "symbol": symbol,
                    "reason": (
                        "spread unavailable" if spread_pct is None else "spread filter"
                    ),
                    "quote_quality": quote_quality,
                    "data_window": symbol_window,
                }
            )
            continue
        tick_volume = int(latest.get("tick_volume") or 0)
        if request.min_tick_volume is not None and tick_volume < request.min_tick_volume:
            skipped.append(
                {
                    "symbol": symbol,
                    "reason": "tick-volume filter",
                    "data_window": symbol_window,
                }
            )
            continue
        quote_contexts[symbol] = {
            "quote_quality": quote_quality,
            "spread_pct": spread_pct,
            "tick_volume": tick_volume,
        }
        scoring_histories[symbol] = bars

    factor_histories = dict(scoring_histories)
    if benchmark_symbol and benchmark_symbol in histories:
        factor_histories[benchmark_symbol] = histories[benchmark_symbol]
    return_frames = []
    for symbol, bars in factor_histories.items():
        return_frames.append(pd.Series(np.log(bars["close"]).diff().to_numpy(), index=bars["time"].to_numpy(), name=symbol))
    returns = (
        pd.concat(return_frames, axis=1, join="outer")
        if return_frames
        else pd.DataFrame()
    )
    explicit_factor = returns[request.benchmark.upper()] if request.benchmark and request.benchmark.upper() in returns else None
    rows = []
    aligned_epoch_windows: Dict[str, tuple[float, float]] = {}
    score_parts: Dict[int, Dict[str, float]] = {h: {} for h in request.horizons}
    stability_parts: Dict[int, Dict[int, Dict[str, float]]] = {offset: {h: {} for h in request.horizons} for offset in (0, 5, 10)}
    for symbol, bars in scoring_histories.items():
        own = pd.Series(np.log(bars["close"]).diff().to_numpy(), index=bars["time"].to_numpy()).dropna()
        factor = explicit_factor if explicit_factor is not None else returns.drop(columns=[symbol], errors="ignore").mean(axis=1, skipna=True)
        aligned = pd.concat([own.rename("own"), factor.rename("factor")], axis=1, join="inner").dropna()
        symbol_window = dict(history_windows[symbol])
        if not aligned.empty:
            aligned_epoch_windows[symbol] = (
                float(aligned.index[0]),
                float(aligned.index[-1]),
            )
            symbol_window.update(
                {
                    "aligned_start": format_epoch_utc(float(aligned.index[0])),
                    "aligned_end": format_epoch_utc(float(aligned.index[-1])),
                    "aligned_observations": int(len(aligned)),
                }
            )
        if len(aligned) < request.volatility_lookback:
            skipped.append(
                {
                    "symbol": symbol,
                    "reason": "factor alignment below minimum",
                    "data_window": symbol_window,
                }
            )
            continue
        cov = aligned["own"].tail(request.volatility_lookback).cov(aligned["factor"].tail(request.volatility_lookback))
        variance = aligned["factor"].tail(request.volatility_lookback).var()
        beta = float(cov / variance) if variance and variance > 0 else 0.0
        residual = aligned["own"] - beta * aligned["factor"]
        vol = float(residual.tail(request.volatility_lookback).std())
        raw_momentum = {}
        residual_momentum = {}
        for horizon in request.horizons:
            raw_value = float(aligned["own"].tail(horizon).sum())
            residual_value = float(residual.tail(horizon).sum())
            raw_momentum[str(horizon)] = raw_value
            residual_momentum[str(horizon)] = residual_value
            score_parts[horizon][symbol] = residual_value / max(vol * math.sqrt(horizon), 1e-12)
            for offset in stability_parts:
                if len(residual) >= horizon + offset:
                    stability_parts[offset][horizon][symbol] = float(residual.iloc[: len(residual) - offset].tail(horizon).sum()) / max(vol * math.sqrt(horizon), 1e-12)
        latest = bars.iloc[-1]
        quote_context = quote_contexts[symbol]
        quote_quality = quote_context["quote_quality"]
        spread_pct = quote_context["spread_pct"]
        tick_volume = quote_context["tick_volume"]
        rows.append(
            {
                "symbol": symbol,
                "beta": beta,
                "volatility": vol,
                "raw_momentum": raw_momentum,
                "residual_momentum": residual_momentum,
                "spread_pct": spread_pct,
                "quote_quality": quote_quality,
                "tick_volume": tick_volume,
                "above_sma20": bool(
                    float(latest["close"])
                    > float(bars["close"].tail(20).mean())
                ),
                "above_sma50": bool(
                    float(latest["close"])
                    > float(bars["close"].tail(50).mean())
                ),
                "data_window": symbol_window,
            }
        )
    row_by_symbol = {row["symbol"]: row for row in rows}
    composite = pd.Series(0.0, index=list(row_by_symbol), dtype=float)
    for horizon, weight in zip(request.horizons, request.weights):
        values = pd.Series({symbol: value for symbol, value in score_parts[horizon].items() if symbol in row_by_symbol}, dtype=float)
        composite = composite.add(_robust_z(values) * weight, fill_value=0.0)
    ranked = composite.sort_values(ascending=False)
    offset_ranks: Dict[int, Dict[str, int]] = {}
    for offset, horizons_data in stability_parts.items():
        offset_score = pd.Series(0.0, index=list(row_by_symbol), dtype=float)
        for horizon, weight in zip(request.horizons, request.weights):
            values = pd.Series({symbol: value for symbol, value in horizons_data[horizon].items() if symbol in row_by_symbol}, dtype=float)
            offset_score = offset_score.add(_robust_z(values) * weight, fill_value=0.0)
        offset_ranks[offset] = {symbol: rank for rank, symbol in enumerate(offset_score.sort_values(ascending=False).index, start=1)}
    score_tie_tolerance = 1e-12
    previous_score: Optional[float] = None
    shared_rank = 0
    for position, (symbol, score) in enumerate(ranked.items(), start=1):
        numeric_score = float(score)
        if previous_score is None or abs(previous_score - numeric_score) > score_tie_tolerance:
            shared_rank = position
        previous_score = numeric_score
        row_by_symbol[symbol]["score"] = float(score)
        row_by_symbol[symbol]["rank"] = shared_rank
        if len(ranked) >= 10:
            row_by_symbol[symbol]["rank_percentile"] = float(
                1.0 - (shared_rank - 1) / max(1, len(ranked) - 1)
            )
        observed_ranks = [mapping[symbol] for mapping in offset_ranks.values() if symbol in mapping]
        row_by_symbol[symbol]["rank_stability"] = float(max(0.0, 1.0 - np.std(observed_ranks) / max(1.0, len(ranked) - 1)))
    ordered = [row_by_symbol[symbol] for symbol in ranked.index]
    latest_returns = {h: [row["raw_momentum"][str(h)] for row in ordered] for h in request.horizons}
    breadth = {
        "positive_by_horizon": {
            str(h): (
                float(np.mean(np.asarray(values) > 0)) if values else None
            )
            for h, values in latest_returns.items()
        },
        "advance_decline_balance": float(np.mean(np.sign(np.asarray(latest_returns[request.horizons[0]])))) if ordered else None,
        "dispersion": float(np.std(list(composite.values), ddof=1)) if len(composite) > 1 else 0.0,
        "above_sma20": float(np.mean([row["above_sma20"] for row in ordered])) if ordered else None,
        "above_sma50": float(np.mean([row["above_sma50"] for row in ordered])) if ordered else None,
    }
    returned_count = min(int(request.limit), len(ordered))
    leader_count = (returned_count + 1) // 2
    laggard_count = returned_count - leader_count
    leader_rows = ordered[:leader_count]
    laggard_rows = (
        list(reversed(ordered[-laggard_count:])) if laggard_count else []
    )
    selected_rankings = sorted(
        [*leader_rows, *laggard_rows],
        key=lambda row: int(row["rank"]),
    )
    output_leaders = [
        _project_relative_strength_row(row, detail=request.detail)
        for row in leader_rows
    ]
    output_laggards = [
        _project_relative_strength_row(row, detail=request.detail)
        for row in laggard_rows
    ]

    ranked_symbols = [str(row["symbol"]) for row in ordered]
    ranked_aligned_windows = [
        aligned_epoch_windows[symbol]
        for symbol in ranked_symbols
        if symbol in aligned_epoch_windows
    ]
    effective_common_window: Dict[str, Any] = {
        "start": None,
        "end": None,
        "timestamp_basis": "bar_open",
        "aligned_symbols": len(ranked_aligned_windows),
    }
    if ranked_aligned_windows:
        common_start = max(start for start, _ in ranked_aligned_windows)
        common_end = min(end for _, end in ranked_aligned_windows)
        effective_common_window.update(
            {
                "start": format_epoch_utc(common_start),
                "end": format_epoch_utc(common_end),
                "has_overlap": common_start <= common_end,
            }
        )

    ranked_latest_epochs = {
        symbol: bar_close_epoch(
            float(candidate_histories[symbol]["time"].iloc[-1]),
            request.timeframe,
        )
        for symbol in ranked_symbols
        if symbol in candidate_histories and not candidate_histories[symbol].empty
    }
    alignment_tolerance_seconds = int(TIMEFRAME_SECONDS[request.timeframe])
    endpoint_alignment: Dict[str, Any] = {
        "timestamp_basis": "bar_close",
        "tolerance_seconds": alignment_tolerance_seconds,
        "status": "unavailable",
        "span_seconds": None,
        "lagging_symbols": [],
    }
    if ranked_latest_epochs:
        earliest_endpoint = min(ranked_latest_epochs.values())
        latest_endpoint = max(ranked_latest_epochs.values())
        endpoint_span = max(0.0, latest_endpoint - earliest_endpoint)
        endpoint_alignment.update(
            {
                "earliest_bar_close": format_epoch_utc(earliest_endpoint),
                "latest_bar_close": format_epoch_utc(latest_endpoint),
                "span_seconds": round(endpoint_span, 3),
                "status": (
                    "aligned"
                    if endpoint_span == 0.0
                    else (
                        "mixed_within_tolerance"
                        if endpoint_span <= alignment_tolerance_seconds
                        else "incomparable"
                    )
                ),
                "comparable": endpoint_span <= alignment_tolerance_seconds,
                "lagging_symbols": sorted(
                    symbol
                    for symbol, endpoint in ranked_latest_epochs.items()
                    if endpoint < latest_endpoint
                ),
            }
        )

    tied_universe = bool(ordered) and (
        float(ranked.max()) - float(ranked.min()) <= score_tie_tolerance
    )
    ranking_withheld = bool(ordered) and endpoint_alignment.get("comparable") is False
    ranking_withheld = ranking_withheld or tied_universe
    published_leaders = [] if ranking_withheld else output_leaders
    published_laggards = [] if ranking_withheld else output_laggards
    published_rankings = [] if ranking_withheld else selected_rankings
    published_breadth: Dict[str, Any] = (
        {
            "status": (
                "withheld_incomparable_endpoints"
                if endpoint_alignment.get("comparable") is False
                else "withheld_tied_scores"
            ),
            "reason": (
                "Cross-sectional breadth requires latest completed bars within "
                "the endpoint-alignment tolerance."
                if endpoint_alignment.get("comparable") is False
                else "All composite scores are tied within the published tolerance."
            ),
        }
        if ranking_withheld
        else breadth
    )

    analysis_as_of = format_datetime_utc(datetime.now(timezone.utc), timespec="auto")
    result = {
        "success": True,
        "status": (
            "incomparable"
            if endpoint_alignment.get("comparable") is False
            else "tied"
            if tied_universe
            else "ranked" if ordered else "no_matches"
        ),
        "timeframe": request.timeframe,
        "analysis_as_of": analysis_as_of,
        "data_window": {
            "requested": {
                "lookback_bars": int(lookback),
                "horizons_bars": list(request.horizons),
                "volatility_lookback_bars": int(request.volatility_lookback),
            },
            "effective_common": effective_common_window,
            "endpoint_alignment": endpoint_alignment,
        },
        "universe_size": len(ordered),
        "returned_count": len(published_rankings),
        "applied_limit": int(request.limit),
        "ranking_selection": {
            "method": (
                "withheld_incomparable_endpoints"
                if endpoint_alignment.get("comparable") is False
                else "withheld_tied_scores"
                if tied_universe
                else "strongest_and_weakest_tails"
            ),
            "leader_count": len(published_leaders),
            "laggard_count": len(published_laggards),
            "rankings_order": "strongest_to_weakest",
        },
        "rank_quality": (
            "incomparable_endpoints"
            if endpoint_alignment.get("comparable") is False
            else "tied_scores"
            if tied_universe
            else "cross_sectional" if len(ordered) >= 10 else "illustrative_small_universe"
        ),
        "score_definition": {
            "method": "weighted_robust_z_of_volatility_scaled_residual_momentum",
            "horizons_bars": list(request.horizons),
            "weights": list(request.weights),
            "higher_is_stronger": True,
            "score_tie_tolerance": score_tie_tolerance,
        },
        "leaders": published_leaders,
        "laggards": published_laggards,
        "breadth": published_breadth,
        "factor": {
            "source": benchmark_symbol or "equal_weight_universe",
            "requested_source": benchmark_symbol,
        },
        "data_quality": {
            "selected_symbols": len(candidate_symbols),
            "data_symbols_fetched": len(histories),
            "ranked_symbols": len(published_rankings),
            "scored_symbols": len(ordered),
            "skipped": skipped,
            "missing_symbols": sorted(
                set(missing_explicit) | set(missing_candidate_history)
            ),
            "unavailable_symbols": missing_explicit,
            "history_unavailable_symbols": missing_candidate_history,
            "benchmark_excluded_from_ranking": benchmark_symbol
            if benchmark_symbol in selected
            else None,
            "minimum_history_coverage": 0.90,
            "endpoint_alignment": endpoint_alignment,
            "quote_not_live_ready_symbols": sorted(set(quote_excluded_symbols)),
            **(
                {"symbol_windows": history_windows}
                if request.detail == "full"
                else {}
            ),
        },
        "units": {"raw_momentum": "log_return_fraction", "residual_momentum": "log_return_fraction", "volatility": "per_bar_log_return_stddev", "score": "robust_z_composite", "rank_stability": "fraction_0_to_1", "tick_volume": "broker_tick_count"},
        **({"rankings": published_rankings} if request.detail == "full" else {}),
    }
    result_warnings: List[str] = []
    if missing_explicit:
        result_warnings.append(
            "Unavailable requested symbols were omitted: "
            + ", ".join(missing_explicit)
            + "."
        )
    if missing_candidate_history:
        result_warnings.append(
            "Requested symbols with insufficient history were omitted: "
            + ", ".join(missing_candidate_history)
            + "."
        )
    if endpoint_alignment.get("comparable") is False:
        result_warnings.append(
            "Candidate symbols do not share comparable latest-bar endpoints within "
            f"the {alignment_tolerance_seconds}s tolerance; no ranking was returned."
        )
    elif tied_universe:
        result_warnings.append(
            "All composite scores are tied within the score tolerance; no "
            "directional leader or laggard was returned."
        )
    quote_not_live = result["data_quality"]["quote_not_live_ready_symbols"]
    if quote_not_live:
        result_warnings.append(
            "Current quotes are not live-ready for these historically ranked symbols: "
            + ", ".join(quote_not_live)
            + "."
        )
    if result_warnings:
        result["warnings"] = result_warnings
    if not ordered:
        result["message"] = "No symbols matched the requested quote/volume filters."
    return result
