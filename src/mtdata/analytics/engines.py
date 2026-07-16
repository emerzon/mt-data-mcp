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
from ..utils.barriers import normalize_same_bar_policy
from ..utils.freshness import closed_session_context
from ..utils.tick_flags import mt5_trade_event_mask
from ..utils.time import format_epoch_utc


def _mapping(row: Any) -> Dict[str, Any]:
    if isinstance(row, dict):
        return dict(row)
    converter = getattr(row, "_asdict", None)
    if callable(converter):
        return dict(converter())
    return {name: getattr(row, name) for name in dir(row) if not name.startswith("_") and not callable(getattr(row, name, None))}


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
    quote_flag_mask = (
        getattr(gateway, "TICK_FLAG_BID", 2)
        | getattr(gateway, "TICK_FLAG_ASK", 4)
    )
    observed_quote_flags = (df["flags"].astype(np.int64) & quote_flag_mask) != 0
    complete_quote_update = (
        (df["flags"].astype(np.int64) & quote_flag_mask) == quote_flag_mask
    )
    valid_quote = (df["bid"] > 0) & (df["ask"] >= df["bid"])
    if bool(observed_quote_flags.any()):
        valid_quote &= complete_quote_update
    df["spread_valid"] = valid_quote
    df["mid"] = np.where(valid_quote, (df["bid"] + df["ask"]) / 2.0, np.nan)
    df["spread"] = np.where(np.isfinite(df["mid"]), df["ask"] - df["bid"], np.nan)
    return df.reset_index(drop=True), truncated


def analyze_microstructure(request: MarketMicrostructureRequest, gateway: Any) -> Dict[str, Any]:
    start, end = _window(request.start, request.end, request.minutes_back)
    df, truncated = _tick_frame(gateway, request.symbol, start, end, request.max_ticks)
    if len(df) < 20:
        last_tick_epoch = float(df["epoch"].iloc[-1]) if len(df) else None
        session = closed_session_context(
            request.symbol,
            now_epoch=end.timestamp(),
            item="tick stream",
            data_age_seconds=(
                max(0.0, end.timestamp() - last_tick_epoch)
                if last_tick_epoch is not None
                else None
            ),
        )
        if session and session.get("market_status") == "closed":
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
                **session,
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
    try:
        symbol_info = gateway.symbol_info(request.symbol)
    except Exception:
        symbol_info = None
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
        trades["side"] = np.sign(trades["last"] - prevailing_mid.loc[trades.index])
        zero = trades["side"] == 0
        trades.loc[zero, "side"] = np.sign(trades.loc[zero, "last"].diff()).fillna(0.0)
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
    if tier != "trade_volume":
        warnings.append("Real trade volume is insufficient; volume-impact metrics were omitted.")
    warnings.append(
        "Metrics describe the connected broker's tick feed and do not establish centralized market-wide order flow or liquidity."
    )
    data_quality = {
        "feed_tier": tier,
        "quote_coverage": float(quote_mask.mean()),
        "trade_tick_coverage": float(trade_mask.mean()),
        "real_volume_trade_coverage": real_share,
        "invalid_partial_quote_ticks": int((~df["spread_valid"]).sum()),
        "truncated": truncated,
        "requested_start": start.isoformat(),
        "requested_end": end.isoformat(),
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
        return {
            "success": True,
            "symbol": request.symbol,
            "summary": {
                "feed_tier": tier,
                "ticks": int(len(df)),
                "duration_seconds": duration,
                "ticks_per_second": float(len(df) / duration),
                "spread": {
                    "median": spread_stats.get("median"),
                    "p95": spread_stats.get("p95"),
                    "unit": spread_unit,
                },
            },
            "data_quality": {
                key: data_quality[key]
                for key in (
                    "quote_coverage",
                    "invalid_partial_quote_ticks",
                    "truncated",
                )
            },
            "warnings": warnings,
        }
    return {
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


def _deal_side(row: Dict[str, Any], gateway: Any) -> Optional[str]:
    value = row.get("type")
    text = str(value).lower()
    if text in {"buy", "0", str(getattr(gateway, "DEAL_TYPE_BUY", 0))}:
        return "buy"
    if text in {"sell", "1", str(getattr(gateway, "DEAL_TYPE_SELL", 1))}:
        return "sell"
    return None


def analyze_execution_quality(request: TradeExecutionQualityRequest, gateway: Any) -> Dict[str, Any]:
    start, end = _window(request.start, request.end, request.minutes_back)
    kwargs = {"group": f"*{request.symbol}*"} if request.symbol else {}
    deals = [_mapping(row) for row in (gateway.history_deals_get(start, end, **kwargs) or [])]
    orders = [_mapping(row) for row in (gateway.history_orders_get(start, end, **kwargs) or [])]
    order_by_ticket = {int(row.get("ticket") or 0): row for row in orders if row.get("ticket")}
    fills = []
    skipped = {"non_trade": 0, "filter": 0, "unbenchmarked": 0, "missing_markout": 0}
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
    benchmark_sources = {"arrival_quote": 0, "order_price": 0, "order_price_fallback": 0}
    processed_candidates = 0
    for deal in eligible_deals:
        processed_candidates += 1
        side = _deal_side(deal, gateway)
        volume = float(deal.get("volume") or 0.0)
        symbol = str(deal.get("symbol") or "").strip()
        order = order_by_ticket.get(int(deal.get("order") or 0), {})
        fill_epoch = float(deal.get("time_msc") or 0) / 1000.0 or float(deal.get("time") or 0)
        qstart = datetime.fromtimestamp(fill_epoch - request.quote_window_seconds, tz=timezone.utc)
        qend = datetime.fromtimestamp(fill_epoch + max(request.markout_seconds) + 5, tz=timezone.utc)
        ticks, _ = _tick_frame(gateway, symbol, qstart, qend, 50_000)
        before = ticks[(ticks["epoch"] <= fill_epoch) & np.isfinite(ticks["mid"])]
        arrival = None
        benchmark_source = None
        if request.benchmark == "arrival_quote" and len(before):
            latest = before.iloc[-1]
            arrival = float(latest["ask"] if side == "buy" else latest["bid"])
            benchmark_source = "arrival_quote"
        if request.benchmark == "order_price":
            candidate = float(order.get("price_open") or order.get("price_current") or 0.0)
            if candidate > 0:
                arrival = candidate
                benchmark_source = "order_price"
        elif (
            (not arrival or arrival <= 0)
            and request.benchmark_fallback == "order_price"
        ):
            candidate = float(order.get("price_open") or order.get("price_current") or 0.0)
            if candidate > 0:
                arrival = candidate
                benchmark_source = "order_price_fallback"
        fill_price = float(deal.get("price") or 0.0)
        if not arrival or fill_price <= 0:
            skipped["unbenchmarked"] += 1
            continue
        benchmark_sources[str(benchmark_source)] += 1
        sign = 1.0 if side == "buy" else -1.0
        slippage_bps = sign * (fill_price - arrival) / arrival * 10_000.0
        time_setup_msc = float(order.get("time_setup_msc") or 0.0)
        if not time_setup_msc and order.get("time_setup"):
            time_setup_msc = float(order["time_setup"]) * 1000.0
        markouts: Dict[str, Optional[float]] = {}
        for horizon in request.markout_seconds:
            candidates = ticks[(ticks["epoch"] >= fill_epoch + horizon) & (ticks["epoch"] <= fill_epoch + horizon + 5) & np.isfinite(ticks["mid"])]
            if len(candidates):
                markouts[str(horizon)] = float(sign * (float(candidates.iloc[0]["mid"]) - fill_price) / fill_price * 10_000.0)
            else:
                markouts[str(horizon)] = None
                skipped["missing_markout"] += 1
        initial_volume = float(order.get("volume_initial") or volume)
        order_type_value = order.get("type")
        market_order_types = {
            getattr(gateway, "ORDER_TYPE_BUY", 0),
            getattr(gateway, "ORDER_TYPE_SELL", 1),
        }
        is_market_order = order_type_value in market_order_types
        order_to_fill_ms = (
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
            "benchmark_price": arrival,
            "benchmark_source": benchmark_source,
            "slippage_bps": slippage_bps,
            "price_improved": slippage_bps < 0,
            "order_to_fill_ms": order_to_fill_ms,
            "is_market_order": is_market_order,
            "fill_ratio": min(1.0, volume / initial_volume) if initial_volume > 0 else None,
            "commission": float(deal.get("commission") or 0.0),
            "fee": float(deal.get("fee") or 0.0),
            "commission_fee_per_lot": (float(deal.get("commission") or 0.0) + float(deal.get("fee") or 0.0)) / volume,
            "markout_bps": markouts,
            "fill_epoch": fill_epoch,
            "order_type": str(order_type_value if order_type_value is not None else "unknown"),
            "hour_utc": datetime.fromtimestamp(fill_epoch, tz=timezone.utc).hour,
        }
        hour = int(item["hour_utc"])
        item["session"] = "asia" if hour < 7 else "london" if hour < 13 else "overlap" if hour < 17 else "new_york" if hour < 22 else "off"
        try:
            action = getattr(gateway, "ORDER_TYPE_BUY", 0) if side == "buy" else getattr(gateway, "ORDER_TYPE_SELL", 1)
            shortfall = gateway.order_calc_profit(action, symbol, volume, arrival, fill_price)
            if shortfall is not None:
                item["execution_shortfall_currency_estimate"] = float(-shortfall)
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
    slippages = [float(item["slippage_bps"]) for item in fills]
    market_order_fills = [item for item in fills if item.get("is_market_order")]
    non_market_order_fills = [item for item in fills if not item.get("is_market_order")]
    summary = {
        "fills": len(fills),
        "orders": len({item["order_ticket"] for item in fills}),
        "market_order_fills": len(market_order_fills),
        "non_market_order_fills": len(non_market_order_fills),
        "slippage_bps": _percentiles(slippages),
        "mean_slippage_ci_95": _bootstrap_mean_ci(slippages, 500),
        "price_improvement_rate": float(np.mean([item["price_improved"] for item in fills])) if fills else None,
        "partial_fill_rate": float(np.mean([(item.get("fill_ratio") or 1.0) < 0.999 for item in fills])) if fills else None,
        "market_order_latency_ms": _percentiles(
            item["order_to_fill_ms"]
            for item in market_order_fills
            if item.get("order_to_fill_ms") is not None
        ),
        "non_market_order_latency_ms": _percentiles(
            item["order_to_fill_ms"]
            for item in non_market_order_fills
            if item.get("order_to_fill_ms") is not None
        ),
        "order_to_fill_ms": _percentiles(
            item["order_to_fill_ms"]
            for item in fills
            if item.get("order_to_fill_ms") is not None
        ),
        "commission_fee_per_lot": _percentiles(item["commission_fee_per_lot"] for item in fills),
    }
    for horizon in request.markout_seconds:
        summary.setdefault("markout_bps", {})[str(horizon)] = _percentiles(item["markout_bps"].get(str(horizon)) for item in fills if item["markout_bps"].get(str(horizon)) is not None)
    breakdowns: Dict[str, List[Dict[str, Any]]] = {}
    if fills:
        fill_frame = pd.DataFrame(fills)
        for keys, label in ((["symbol", "side"], "by_symbol_side"), (["order_type"], "by_order_type"), (["session"], "by_session"), (["hour_utc"], "by_hour_utc")):
            breakdowns[label] = []
            for group_key, items in fill_frame.groupby(keys):
                labels = group_key if isinstance(group_key, tuple) else (group_key,)
                row = {name: value for name, value in zip(keys, labels)}
                row.update({"fills": len(items), "slippage_bps": _percentiles(items["slippage_bps"])})
                if label == "by_order_type":
                    row["order_to_fill_ms"] = _percentiles(items["order_to_fill_ms"])
                breakdowns[label].append(row)
    sample_start = format_epoch_utc(fills[0]["fill_epoch"]) if fills else None
    sample_end = format_epoch_utc(fills[-1]["fill_epoch"]) if fills else None
    benchmark_attempts = len(fills) + skipped["unbenchmarked"]
    fallback_count = benchmark_sources["order_price_fallback"]
    warnings = []
    if fallback_count:
        warnings.append(
            f"{fallback_count} fill(s) used order price because no arrival quote was available."
        )
    return {
        "success": True,
        "summary": summary,
        "breakdowns": breakdowns,
        **({"items": fills} if request.detail == "full" else {}),
        "sample_quality": {"status": "ok" if len(fills) >= request.min_sample else "insufficient", "minimum": request.min_sample, "observed": len(fills)},
        "data_quality": {
            "history_deals": len(deals),
            "history_orders": len(orders),
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
                    benchmark_sources["arrival_quote"] / benchmark_attempts
                    if benchmark_attempts
                    else None
                ),
            },
        },
        "sample": {
            "selection_order": "latest_first",
            "display_order": "chronological",
            "total_eligible": len(eligible_deals),
            "sample_start": sample_start,
            "sample_end": sample_end,
            "truncated": processed_candidates < len(eligible_deals),
        },
        "latency_definition": {
            "market_order_latency_ms": "market_order_setup_to_fill_elapsed_time",
            "non_market_order_latency_ms": "non_market_order_setup_to_fill_elapsed_time_including_pending_wait",
            "order_to_fill_ms": "all_order_setup_to_fill_elapsed_time_including_pending_wait",
        },
        "units": {"slippage_bps": "basis_points_positive_is_worse", "markout_bps": "basis_points_positive_is_favorable", "market_order_latency_ms": "milliseconds", "non_market_order_latency_ms": "milliseconds", "order_to_fill_ms": "milliseconds"},
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
        return pd.Series(np.where(a > b, 1.0, np.where(a < b, -1.0, 0.0)), index=close.index).where(a.notna() & b.notna())
    length = int(params.get("rsi_length", 14))
    oversold = float(params.get("oversold", 30.0))
    overbought = float(params.get("overbought", 70.0))
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    rsi = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    return pd.Series(np.where(rsi < oversold, 1.0, np.where(rsi > overbought, -1.0, 0.0)), index=close.index).where(rsi.notna())


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
                values = result.get("forecast") or result.get("values") or result.get("predictions")
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
    for idx in range(len(df) - horizon):
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
    return np.asarray(indices, dtype=int), np.asarray(outcomes, dtype=float)


def _observed_spread_bps(request: StrategyValidateRequest, gateway: Any) -> Tuple[float, str, bool]:
    if request.spread_bps is not None:
        return float(request.spread_bps), "explicit", request.cost_model == "fixed"
    now = datetime.now(timezone.utc)
    ticks, _ = _tick_frame(gateway, request.symbol, now - timedelta(hours=1), now, 10_000)
    valid = ticks[np.isfinite(ticks["mid"]) & (ticks["mid"] > 0)]
    if len(valid):
        return float(np.median(valid["spread"] / valid["mid"] * 10_000.0)), "mt5_tick_median", False
    return 0.0, "unavailable", False


def validate_strategies(  # noqa: C901
    request: StrategyValidateRequest, gateway: Any
) -> Dict[str, Any]:
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
    spread_bps, spread_source, complete = _observed_spread_bps(request, gateway)
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
            results.append({"id": candidate.id, "evaluation_status": "insufficient_data", "trades": int(len(indices))})
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
                    train_y = (gross[train_mask] > 0).astype(int)
                    test_x = signal.iloc[test_indices].to_numpy(dtype=float).reshape(-1, 1)
                    if len(np.unique(train_y)) > 1 and np.all(np.isfinite(train_x)) and np.all(np.isfinite(test_x)):
                        calibrator = LogisticRegression(random_state=42).fit(train_x, train_y)
                        calibrated_probabilities.extend(calibrator.predict_proba(test_x)[:, 1].tolist())
                        calibrated_labels.extend((test_gross > 0).astype(int).tolist())
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
            results.append({"id": candidate.id, "evaluation_status": "insufficient_data", "trades": 0})
            continue
        equity = np.cumprod(1.0 + np.clip(arr, -0.999, None))
        peaks = np.maximum.accumulate(equity)
        drawdown = equity / peaks - 1.0
        std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        sharpe = float(np.mean(arr) / std * math.sqrt(len(arr))) if std > 0 else None
        per_trade_sharpe = float(np.mean(arr) / std) if std > 0 else 0.0
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
        calibration = {"status": "insufficient_data", "observations": len(calibrated_labels)}
        if calibrated_labels:
            probs = np.asarray(calibrated_probabilities, dtype=float)
            labels = np.asarray(calibrated_labels, dtype=float)
            ece = 0.0
            for lower in np.linspace(0.0, 0.9, 10):
                mask = (probs >= lower) & (probs < lower + 0.1 if lower < 0.9 else probs <= 1.0)
                if np.any(mask):
                    ece += float(np.mean(mask)) * abs(float(np.mean(probs[mask])) - float(np.mean(labels[mask])))
            calibration = {"status": "calibrated", "observations": len(labels), "method": "sigmoid_train_only", "brier_score": float(np.mean((probs - labels) ** 2)), "expected_calibration_error": ece}
        results.append({
            "id": candidate.id,
            "type": candidate.type,
            "evaluation_status": "complete",
            "trades": int(len(arr)),
            "net_expectancy": float(np.mean(arr)),
            "expectancy_ci_95": expectancy_ci,
            "win_rate": float(np.mean(arr > 0)),
            "profit_factor": float(arr[arr > 0].sum() / abs(arr[arr < 0].sum())) if np.any(arr < 0) else None,
            "sharpe": sharpe,
            "deflated_sharpe_probability": deflated_probability,
            "mean_return_p_value": mean_return_p_value,
            "max_drawdown": float(np.min(drawdown)),
            "fold_stability": fold_stability,
            "folds_requested": int(request.n_splits),
            "folds_evaluated": folds_evaluated,
            "fold_coverage": fold_coverage,
            "signal_coverage": signal_coverage,
            "skipped_folds": skipped_folds,
            "same_bar_policy": same_bar_policy,
            "calibration": calibration,
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
        if item.get("evaluation_status") != "complete":
            continue
        ci = item.get("expectancy_ci_95")
        fold_share = float(item.get("fold_stability") or 0.0)
        adjusted_p = item.get("holm_adjusted_p_value")
        criteria = {
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
            "significance_alpha": float(request.significance_alpha),
            "minimum_positive_fold_share": float(request.min_positive_fold_share),
        }
    ranked = sorted(results, key=lambda item: (item.get("net_expectancy") is None, -(item.get("net_expectancy") or -1e9)))
    warnings_out = [] if complete else ["Observed spread was used, but commission/slippage completeness could not be proven from sufficient matched fills."]
    for item in results:
        folds_evaluated = int(item.get("folds_evaluated") or 0)
        if item.get("evaluation_status") == "complete" and folds_evaluated < request.n_splits:
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
        "cost_model": {"source": spread_source, "spread_bps": spread_bps, "commission_bps_per_side": request.commission_bps, "slippage_bps_per_side": request.slippage_bps, "round_trip_bps": round_trip_bps, "complete": complete},
        "data_quality": {"bars": len(df), "cost_model_complete": complete},
        "warnings": warnings_out,
    }


def _position_side(row: Dict[str, Any], gateway: Any) -> str:
    value = row.get("type")
    return "buy" if str(value).lower() in {"buy", "0", str(getattr(gateway, "POSITION_TYPE_BUY", 0))} else "sell"


def _position_sensitivity(gateway: Any, row: Dict[str, Any]) -> Tuple[Optional[float], Optional[str]]:
    symbol = str(row.get("symbol") or "")
    volume = float(row.get("volume") or 0.0)
    side = _position_side(row, gateway)
    tick = gateway.symbol_info_tick(symbol)
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


def decompose_portfolio_risk(request: PortfolioRiskDecomposeRequest, gateway: Any) -> Dict[str, Any]:
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
    account = None
    try:
        account_info = getattr(gateway, "account_info", None)
        if callable(account_info):
            account = account_info()
    except Exception:
        account = None
    positions = [_mapping(row) for row in (gateway.positions_get() or [])]
    base_position_count = len(positions)
    if request.proposed_trade:
        tick = gateway.symbol_info_tick(request.proposed_trade.symbol)
        positions.append({
            "ticket": "proposed",
            "symbol": request.proposed_trade.symbol,
            "type": getattr(gateway, "POSITION_TYPE_BUY", 0) if request.proposed_trade.side == "buy" else getattr(gateway, "POSITION_TYPE_SELL", 1),
            "volume": request.proposed_trade.volume,
            "price_current": getattr(tick, "ask" if request.proposed_trade.side == "buy" else "bid", None),
            "proposed": True,
        })
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
    returns = pd.concat(series.values(), axis=1, join="inner").dropna()
    if len(returns) < 100:
        return {"error": "At least 100 aligned returns are required.", "error_code": "insufficient_data", "aligned_rows": len(returns)}
    returns.columns = list(series)
    alpha = 1.0 - math.exp(math.log(0.5) / request.ewma_half_life)
    standardized, current_vol = _filtered_historical_returns(
        returns,
        alpha=alpha,
    )
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
        component_pnl = scenario_returns * sensitivity_vec
        pnl = component_pnl.sum(axis=1)
        base_pnl = pnl.copy()
        if proposed_sensitivity and proposed_sensitivity[0] in list(standardized.columns):
            proposed_idx = list(standardized.columns).index(proposed_sensitivity[0])
            base_pnl = pnl - scenario_returns[:, proposed_idx] * proposed_sensitivity[1]
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
    worst_historical = (returns * sensitivity_vec).sum(axis=1)
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
    proposed = request.proposed_trade
    proposed_context = None
    if proposed:
        try:
            tick = gateway.symbol_info_tick(proposed.symbol)
            action = getattr(gateway, "ORDER_TYPE_BUY", 0) if proposed.side == "buy" else getattr(gateway, "ORDER_TYPE_SELL", 1)
            price = float(getattr(tick, "ask" if proposed.side == "buy" else "bid"))
            margin = gateway.order_calc_margin(action, proposed.symbol, proposed.volume, price)
            proposed_context = {"symbol": proposed.symbol, "side": proposed.side, "volume": proposed.volume, "margin_required": float(margin) if margin is not None else None}
        except Exception:
            proposed_context = {"symbol": proposed.symbol, "margin_required": None}
    account_context = {
        key: value
        for key, value in {
            "currency": getattr(account, "currency", None),
            "equity": getattr(account, "equity", None),
        }.items()
        if value is not None
    }
    requested_symbols = sorted(
        {str(row.get("symbol") or "") for row in positions if row.get("symbol")}
    )
    modeled_symbols = [str(column) for column in standardized.columns]
    omitted_symbols = sorted(set(requested_symbols) - set(modeled_symbols))
    warnings_out: List[str] = []
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
            "data_start": data_start,
            "data_end": data_end,
            "as_of": data_end,
        }
    )
    return {
        "success": True,
        "method": request.method,
        "timeframe": request.timeframe,
        "holding_periods": holding_periods,
        "model_context": model_context,
        **account_context,
        "summary": {"positions": base_position_count, "positions_after_proposed": len(positions), "symbols": len(modeled_symbols), "symbols_requested": len(requested_symbols), "aligned_rows": len(returns), "concentration_hhi": float(np.sum(weights**2))},
        "risk": risk_rows,
        "stresses": stresses,
        "proposed_trade": proposed_context,
        "data_quality": {
            "pricing_failures": failures,
            "history_failures": history_failures,
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
    clipped = values.clip(values.quantile(0.05), values.quantile(0.95))
    median = clipped.median()
    mad = float(np.median(np.abs(clipped - median)))
    if mad <= 1e-12:
        std = float(clipped.std())
        return (clipped - median) / std if std > 0 else clipped * 0.0
    return (clipped - median) / (1.4826 * mad)


def rank_relative_strength(request: MarketRelativeStrengthRequest, gateway: Any) -> Dict[str, Any]:
    raw_symbols = list(gateway.symbols_get() or [])
    explicit = {item.strip().upper() for item in str(request.symbols or "").split(",") if item.strip()}
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
    if request.benchmark and request.benchmark.upper() not in selected:
        selected.append(request.benchmark.upper())
    lookback = max(max(request.horizons) + request.volatility_lookback + 15, 100)
    histories: Dict[str, pd.DataFrame] = {}
    skipped = []
    for symbol in selected:
        bars = _rates(gateway, symbol, request.timeframe, lookback)
        if len(bars) < int(lookback * 0.90):
            skipped.append({"symbol": symbol, "reason": "history coverage below 90%"})
            continue
        histories[symbol] = bars
    if len(histories) < 2:
        return {"error": "At least two symbols with sufficient history are required.", "error_code": "insufficient_data", "skipped": skipped}
    return_frames = []
    for symbol, bars in histories.items():
        return_frames.append(pd.Series(np.log(bars["close"]).diff().to_numpy(), index=bars["time"].to_numpy(), name=symbol))
    returns = pd.concat(return_frames, axis=1, join="outer")
    explicit_factor = returns[request.benchmark.upper()] if request.benchmark and request.benchmark.upper() in returns else None
    rows = []
    score_parts: Dict[int, Dict[str, float]] = {h: {} for h in request.horizons}
    stability_parts: Dict[int, Dict[int, Dict[str, float]]] = {offset: {h: {} for h in request.horizons} for offset in (0, 5, 10)}
    for symbol, bars in histories.items():
        own = pd.Series(np.log(bars["close"]).diff().to_numpy(), index=bars["time"].to_numpy()).dropna()
        factor = explicit_factor if explicit_factor is not None else returns.drop(columns=[symbol], errors="ignore").mean(axis=1, skipna=True)
        aligned = pd.concat([own.rename("own"), factor.rename("factor")], axis=1, join="inner").dropna()
        if len(aligned) < request.volatility_lookback:
            skipped.append({"symbol": symbol, "reason": "factor alignment below minimum"})
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
        tick = gateway.symbol_info_tick(symbol)
        bid = float(getattr(tick, "bid", 0.0) or 0.0)
        ask = float(getattr(tick, "ask", 0.0) or 0.0)
        spread_pct = ((ask - bid) / ((ask + bid) / 2.0) * 100.0) if ask >= bid > 0 else None
        tick_volume = int(latest.get("tick_volume") or 0)
        if request.max_spread_pct is not None and (spread_pct is None or spread_pct > request.max_spread_pct):
            skipped.append({"symbol": symbol, "reason": "spread filter"})
            continue
        if request.min_tick_volume is not None and tick_volume < request.min_tick_volume:
            skipped.append({"symbol": symbol, "reason": "tick-volume filter"})
            continue
        rows.append({"symbol": symbol, "beta": beta, "volatility": vol, "raw_momentum": raw_momentum, "residual_momentum": residual_momentum, "spread_pct": spread_pct, "tick_volume": tick_volume, "above_sma20": bool(float(latest["close"]) > float(bars["close"].tail(20).mean())), "above_sma50": bool(float(latest["close"]) > float(bars["close"].tail(50).mean()))})
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
    for rank, (symbol, score) in enumerate(ranked.items(), start=1):
        row_by_symbol[symbol]["score"] = float(score)
        row_by_symbol[symbol]["rank"] = rank
        if len(ranked) >= 10:
            row_by_symbol[symbol]["rank_percentile"] = float(
                1.0 - (rank - 1) / max(1, len(ranked) - 1)
            )
        observed_ranks = [mapping[symbol] for mapping in offset_ranks.values() if symbol in mapping]
        row_by_symbol[symbol]["rank_stability"] = float(max(0.0, 1.0 - np.std(observed_ranks) / max(1.0, len(ranked) - 1)))
    ordered = [row_by_symbol[symbol] for symbol in ranked.index]
    latest_returns = {h: [row["raw_momentum"][str(h)] for row in ordered] for h in request.horizons}
    breadth = {
        "positive_by_horizon": {str(h): float(np.mean(np.asarray(values) > 0)) for h, values in latest_returns.items()},
        "advance_decline_balance": float(np.mean(np.sign(np.asarray(latest_returns[request.horizons[0]])))) if ordered else None,
        "dispersion": float(np.std(list(composite.values), ddof=1)) if len(composite) > 1 else 0.0,
        "above_sma20": float(np.mean([row["above_sma20"] for row in ordered])) if ordered else None,
        "above_sma50": float(np.mean([row["above_sma50"] for row in ordered])) if ordered else None,
    }
    leader_count = min(request.limit, (len(ordered) + 1) // 2)
    laggard_count = min(request.limit, len(ordered) - leader_count)
    return {
        "success": True,
        "timeframe": request.timeframe,
        "universe_size": len(ordered),
        "rank_quality": (
            "cross_sectional" if len(ordered) >= 10 else "illustrative_small_universe"
        ),
        "score_definition": {
            "method": "weighted_robust_z_of_volatility_scaled_residual_momentum",
            "horizons_bars": list(request.horizons),
            "weights": list(request.weights),
            "higher_is_stronger": True,
        },
        "leaders": ordered[:leader_count],
        "laggards": (
            list(reversed(ordered[-laggard_count:])) if laggard_count else []
        ),
        "breadth": breadth,
        "factor": {"source": request.benchmark.upper() if request.benchmark else "equal_weight_universe"},
        "data_quality": {"selected_symbols": len(selected), "ranked_symbols": len(ordered), "skipped": skipped, "minimum_history_coverage": 0.90},
        "units": {"raw_momentum": "log_return_fraction", "residual_momentum": "log_return_fraction", "volatility": "per_bar_log_return_stddev", "score": "robust_z_composite", "rank_stability": "fraction_0_to_1", "tick_volume": "broker_tick_count"},
        **({"all_rankings": ordered} if request.detail == "full" else {}),
    }
