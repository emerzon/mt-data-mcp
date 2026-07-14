
import json
import logging
import math
import re
import time
import warnings
from datetime import datetime, timedelta
from datetime import timezone as dt_timezone
from numbers import Real
from typing import Any, Dict, List, Literal, Optional, Tuple

import pandas as pd

from ..bootstrap.settings import mt5_config
from ..core.error_envelope import build_error_payload
from ..core.output_contract import normalize_output_detail
from ..shared.constants import (
    DEFAULT_ROW_LIMIT,
    FETCH_RETRY_ATTEMPTS,
    FETCH_RETRY_DELAY,
    SANITY_BARS_TOLERANCE,
    SIMPLIFY_DEFAULT_METHOD,
    SIMPLIFY_DEFAULT_MODE,
    SIMPLIFY_DEFAULT_POINTS_RATIO_FROM_LIMIT,
    TI_NAN_WARMUP_FACTOR,
    TI_NAN_WARMUP_MIN_ADD,
    TICKS_LOOKBACK_DAYS,
    TIMEFRAME_MAP,
    TIMEFRAME_SECONDS,
)
from ..shared.market_units import forex_points_per_pip
from ..shared.schema import DenoiseSpec, IndicatorSpec, SimplifySpec, TimeframeLiteral
from ..shared.validators import invalid_timeframe_error
from ..utils.denoise import (
    _apply_denoise as _apply_denoise_util,
)
from ..utils.denoise import (
    _consume_denoise_warnings,
)
from ..utils.denoise import (
    normalize_denoise_spec as _normalize_denoise_spec,
)
from ..utils.freshness import closed_session_context
from ..utils.indicators import (
    _apply_ta_indicators,
    _estimate_warmup_bars,
    _find_unknown_ta_indicators,
    _parse_ti_specs,
)
from ..utils.market_metadata import (
    FRESHNESS_ANCHOR_QUERY_EXPECTED_END,
    FRESHNESS_ANCHOR_WALL_CLOCK,
    FRESHNESS_METRIC_LAST_COMPLETED_BAR_AGE,
    FRESHNESS_METRIC_REQUESTED_RANGE_END_GAP,
    build_tick_freshness_context,
)

# Imports from utils
from ..utils.mt5 import (
    _mt5_copy_rates_from,
    _mt5_copy_rates_from_pos,
    _mt5_copy_rates_range,
    _mt5_copy_ticks_range,
    _rates_to_df,
    _symbol_ready_guard,
    describe_mt5_time_normalization,
    get_cached_mt5_time_alignment,
    get_symbol_info_cached,
    mt5,
    resolve_broker_symbol_name,
)
from ..utils.ohlcv import validate_and_clean_ohlcv_frame

# Simplify entrypoint and helpers.
from ..utils.simplify import (
    _choose_simplify_points,
    _lttb_select_indices,
    _select_indices_for_timeseries,
    _simplify_dataframe_rows_ext,
)
from ..utils.tick_flags import is_mt5_trade_event
from ..utils.time import (
    _format_datetime_minute_explicit,
    _format_time_explicit,
    _format_time_explicit_local,
    _resolve_client_tz,
    format_epoch_utc,
)
from ..utils.utils import (
    _format_numeric_rows_from_df,
    _normalize_ohlcv_arg,
    _parse_start_datetime,
    _table_from_rows,
    _utc_epoch_seconds,
    coerce_scalar,
)

logger = logging.getLogger(__name__)


_TICK_SUMMARY_MIN_ANALYTIC_TICKS = 20
_ONE_SIDED_TICK_WARNING_RATIO = 0.50
_DATE_FORMAT_HINT = (
    "Accepted examples: '2026-01-15', '2026-01-15 14:30', "
    "'2026-01-15T14:30:00Z', 'yesterday', '2 days ago', 'last Friday'."
)
_CANDLE_PRICE_COLUMNS = frozenset({"open", "high", "low", "close"})
_TICK_PRICE_COLUMNS = frozenset({"bid", "ask", "mid", "spread", "last"})
_TICK_PRICE_STAT_KEYS = frozenset(
    {
        "first",
        "last",
        "low",
        "high",
        "mean",
        "std",
        "stderr",
        "change",
        "median",
        "q25",
        "q75",
    }
)
_TICK_ROW_UNITS = {
    "time_epoch": "unix_seconds",
    "bid": "absolute_price",
    "ask": "absolute_price",
    "last": "absolute_price",
    "mid": "absolute_price",
    "spread": "absolute_price",
    "spread_points": "broker_points",
    "spread_pips": "pips",
    "spread_pct": "percentage_points (1.0 = 1%)",
    "tick_gap_ms": "milliseconds",
    "volume": "last_trade_volume",
    "volume_real": "last_trade_volume_real",
}
def _format_mt5_last_error() -> str:
    try:
        err = mt5.last_error()
    except Exception as exc:
        return str(exc)
    if isinstance(err, tuple) and len(err) == 2:
        code, message = err
        return f"({code}, {message!r})"
    return str(err)


def _symbol_price_digits(*infos: Any) -> int:
    for info in infos:
        try:
            digits_raw = getattr(info, "digits", None)
        except Exception:
            digits_raw = None
        if isinstance(digits_raw, (int, float)):
            return max(0, int(digits_raw))
    return 0


def _symbol_price_currency(*infos: Any) -> Optional[str]:
    for info in infos:
        for attr in ("currency_profit", "currency_margin"):
            try:
                value = getattr(info, attr, None)
            except Exception:
                value = None
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _symbol_candle_price_basis(*infos: Any) -> str:
    for info in infos:
        try:
            chart_mode = getattr(info, "chart_mode", None)
        except Exception:
            chart_mode = None
        if isinstance(chart_mode, str):
            normalized = chart_mode.strip().lower()
            if "bid" in normalized:
                return "bid"
            if "last" in normalized:
                return "last_trade"
        if isinstance(chart_mode, (int, float)) and not isinstance(chart_mode, bool):
            if int(chart_mode) == 0:
                return "bid"
            if int(chart_mode) == 1:
                return "last_trade"
    return "broker_chart_price"


def _symbol_price_point(*infos: Any) -> Optional[float]:
    for info in infos:
        try:
            point_raw = getattr(info, "point", None)
        except Exception:
            point_raw = None
        if isinstance(point_raw, (int, float)):
            point = float(point_raw)
            if math.isfinite(point) and point > 0.0:
                return point
    return None


def _symbol_path(*infos: Any) -> str:
    for info in infos:
        try:
            path = getattr(info, "path", None)
        except Exception:
            path = None
        if isinstance(path, str) and path.strip():
            return path.strip()
    return ""


def _round_price_value(value: Any, digits: int) -> Any:
    if digits <= 0 or value is None or isinstance(value, bool):
        return value
    if not isinstance(value, (int, float)):
        return value
    numeric = float(value)
    if not math.isfinite(numeric):
        return value
    return round(numeric, digits)


def _round_row_price_columns(
    rows: List[List[Any]],
    headers: List[str],
    *,
    digits: int,
    price_columns: frozenset[str],
) -> List[List[Any]]:
    if digits <= 0:
        return rows
    price_indexes = [
        idx for idx, header in enumerate(headers) if str(header) in price_columns
    ]
    if not price_indexes:
        return rows
    rounded_rows: List[List[Any]] = []
    for row in rows:
        rounded = list(row)
        for idx in price_indexes:
            if idx < len(rounded):
                rounded[idx] = _round_price_value(rounded[idx], digits)
        rounded_rows.append(rounded)
    return rounded_rows


_PRICE_INDICATOR_PREFIXES = (
    "ALMA_",
    "BBL_",
    "BBM_",
    "BBU_",
    "DEMA_",
    "EMA_",
    "HMA_",
    "KAMA_",
    "SMA_",
    "TEMA_",
    "VWAP",
    "VWMA_",
    "WMA_",
)


def _price_indicator_columns(columns: List[str]) -> List[str]:
    out: List[str] = []
    for column in columns:
        name = str(column or "").strip().upper()
        if name.startswith(_PRICE_INDICATOR_PREFIXES):
            out.append(str(column))
    return out


def _round_tick_price_payload(out: Dict[str, Any], digits: int) -> None:
    if digits <= 0:
        return
    stats = out.get("stats")
    if isinstance(stats, dict):
        for name in ("bid", "ask", "mid", "spread", "last"):
            values = stats.get(name)
            if not isinstance(values, dict):
                continue
            for key in _TICK_PRICE_STAT_KEYS:
                if key in values:
                    stat_digits = digits
                    if name == "spread" and key in {"mean", "std", "stderr"}:
                        stat_digits = max(digits + 2, digits)
                    values[key] = _round_price_value(values[key], stat_digits)
    last_quote = out.get("last_quote")
    if isinstance(last_quote, dict):
        for key in ("bid", "ask", "mid", "spread"):
            if key in last_quote:
                last_quote[key] = _round_price_value(last_quote[key], digits)
    if isinstance(stats, dict):
        for volume_key in ("volume", "volume_real"):
            volume_stats = stats.get(volume_key)
            if isinstance(volume_stats, dict):
                for key in ("vwap_mid", "vwap_last"):
                    if key in volume_stats:
                        volume_stats[key] = _round_price_value(volume_stats[key], digits)


def _tick_units_for_headers(headers: List[str]) -> Dict[str, str]:
    return {
        key: unit
        for key, unit in _TICK_ROW_UNITS.items()
        if key in headers
    }


def _tick_spread_points(spread: Any, price_point: Optional[float]) -> Optional[float]:
    if price_point is None or price_point <= 0.0:
        return None
    spread_value = _finite_or_none(spread)
    if spread_value is None:
        return None
    return round(spread_value / price_point, 4)


def _tick_spread_pct(spread: Any, mid: Any) -> Optional[float]:
    spread_value = _finite_or_none(spread)
    mid_value = _finite_or_none(mid)
    if spread_value is None or mid_value is None or mid_value <= 0.0:
        return None
    return round((spread_value / mid_value) * 100.0, 6)


def _describe_rate_fetch_error(symbol: str, *, info_before: Any = None) -> str:
    if info_before is None:
        try:
            info_before = get_symbol_info_cached(symbol)
        except Exception:
            info_before = None

    error_text = _format_mt5_last_error()
    if info_before is None:
        return (
            f"Symbol '{symbol}' was not found or is not available in MT5. "
            f"Use symbols_list(search_term='{symbol}') to find broker-specific names and suffixes."
        )
    return f"Failed to get rates for {symbol}: {error_text}"


def _build_no_data_error_with_context(
    symbol: str,
    timeframe: TimeframeLiteral,
    mt5_timeframe: int,
    start_datetime: Optional[str],
    end_datetime: Optional[str],
) -> Dict[str, Any]:
    """Build a detailed error payload when no data is available for the requested range."""
    error_msg = "No data available"
    details: Dict[str, Any] = {}
    
    # Add requested range to context if provided
    if start_datetime or end_datetime:
        details["requested_range"] = {
            k: v for k, v in [("start", start_datetime), ("end", end_datetime)]
            if v is not None
        }
    
    # Try to sample available bars for this timeframe to suggest a usable range.
    try:
        available_bars = _mt5_copy_rates_from_pos(symbol, mt5_timeframe, 0, 100_000)

        if available_bars is not None and len(available_bars) > 0:
            times: List[float] = []
            for bar in available_bars:
                try:
                    epoch = float(bar["time"])
                except Exception:
                    continue
                if math.isfinite(epoch):
                    times.append(epoch)
            if not times:
                raise ValueError("available bars have no finite timestamps")
            first_epoch = min(times)
            last_epoch = max(times)
            first_time = datetime.fromtimestamp(first_epoch, tz=dt_timezone.utc)
            last_time = datetime.fromtimestamp(last_epoch, tz=dt_timezone.utc)

            details["available_range"] = {
                "earliest": _format_time_explicit(first_epoch),
                "latest": _format_time_explicit(last_epoch),
            }

            # Provide a suggestion based on the mismatch
            if start_datetime:
                try:
                    req_start, _ = _parse_fetch_datetime_arg(start_datetime)
                    if req_start is not None and req_start.tzinfo is None:
                        req_start = req_start.replace(tzinfo=dt_timezone.utc)
                    elif req_start is not None:
                        req_start = req_start.astimezone(dt_timezone.utc)
                    if req_start and req_start > last_time:
                        error_msg = f"No data available - requested start date is after latest available data ({_format_time_explicit(last_epoch)})"
                        details["suggestion"] = f"Use start='{_format_time_explicit(last_epoch)}' or earlier"
                    elif req_start and req_start < first_time:
                        error_msg = f"No data available - requested date range is before earliest available data ({_format_time_explicit(first_epoch)})"
                        details["suggestion"] = f"Use start='{_format_time_explicit(first_epoch)}' or later"
                except Exception:
                    pass
    except Exception:
        # Silently ignore any errors when trying to get available range
        pass
    
    return build_error_payload(
        error_msg,
        code="data_fetch_candles_no_data",
        operation="data_fetch_candles",
        details=details or None,
    )


def _indicator_param_syntax_error(ti_spec: Optional[str]) -> Optional[str]:
    if not ti_spec:
        return None
    for name, _args, _kwargs in _parse_ti_specs(ti_spec):
        if re.fullmatch(r"[+-]?\d+(?:\.\d+)?", str(name or "").strip()):
            return "Indicator params must use parentheses, e.g. sma(20), not sma,20."
    return None


def _resolve_live_bar_reference_epoch(symbol: Optional[str], timeframe: str) -> float:
    """Use wall-clock time when classifying whether the latest bar is live."""
    del symbol, timeframe
    system_epoch = _utc_epoch_seconds(datetime.now(dt_timezone.utc))
    return float(system_epoch)


def _is_last_bar_forming(
    rates_or_df: Any,
    timeframe: str,
    *,
    current_time_epoch: Optional[float] = None,
) -> bool:
    """Return True if the last bar in *rates_or_df* is still forming."""
    try:
        seconds_per_bar = TIMEFRAME_SECONDS.get(timeframe, 3600)
        current_time = (
            float(current_time_epoch)
            if current_time_epoch is not None and math.isfinite(float(current_time_epoch))
            else float(_utc_epoch_seconds(datetime.now(dt_timezone.utc)))
        )
        if isinstance(rates_or_df, pd.DataFrame):
            if len(rates_or_df) == 0 or '__epoch' not in rates_or_df.columns:
                return False
            last_epoch = float(rates_or_df['__epoch'].iloc[-1])
        else:
            if rates_or_df is None or len(rates_or_df) == 0:
                return False
            last_epoch = float(rates_or_df[-1]["time"])
        return 0 <= current_time - last_epoch < seconds_per_bar
    except Exception:
        return False


def _drop_incomplete_tail(
    rates: Any,
    timeframe: str,
    *,
    current_time_epoch: Optional[float] = None,
) -> Any:
    """Drop the last bar from *rates* if it is still forming."""
    if (
        rates is not None
        and len(rates) > 0
        and _is_last_bar_forming(rates, timeframe, current_time_epoch=current_time_epoch)
    ):
        return rates[:-1]
    return rates


def _drop_incomplete_tail_df(
    df: pd.DataFrame,
    timeframe: str,
    *,
    current_time_epoch: Optional[float] = None,
) -> Tuple[pd.DataFrame, bool]:
    """Drop the last row from *df* if it is still forming.  Returns (df, trimmed)."""
    if len(df) > 0 and _is_last_bar_forming(df, timeframe, current_time_epoch=current_time_epoch):
        return df.iloc[:-1], True
    return df, False


def _build_candle_freshness_diagnostics(
    *,
    last_bar_epoch: Any,
    expected_end_epoch: Any,
    freshness_cutoff_epoch: Any,
) -> Dict[str, Any]:
    def _coerce_epoch(value: Any) -> Optional[float]:
        try:
            epoch = float(value)
        except Exception:
            return None
        if not math.isfinite(epoch):
            return None
        return epoch

    last_epoch = _coerce_epoch(last_bar_epoch)
    expected_epoch = _coerce_epoch(expected_end_epoch)
    cutoff_epoch = _coerce_epoch(freshness_cutoff_epoch)
    data_freshness_seconds: Optional[float] = None
    last_bar_within_policy_window: Optional[bool] = None

    if last_epoch is not None and expected_epoch is not None:
        data_freshness_seconds = round(
            max(0.0, float(expected_epoch - last_epoch)),
            3,
        )
    if last_epoch is not None and cutoff_epoch is not None:
        last_bar_within_policy_window = bool(last_epoch >= cutoff_epoch)

    return {
        "last_bar_epoch": last_epoch,
        "expected_end_epoch": expected_epoch,
        "freshness_cutoff_epoch": cutoff_epoch,
        "data_freshness_seconds": data_freshness_seconds,
        "last_bar_within_policy_window": last_bar_within_policy_window,
    }


def _relax_live_completed_bar_freshness(
    *,
    symbol: str,
    rates: Any,
    timeframe: TimeframeLiteral,
    expected_end_ts: float,
    start_datetime: Optional[str],
    end_datetime: Optional[str],
    freshness_meta: Dict[str, Any],
) -> bool:
    if start_datetime or end_datetime:
        return False
    if _is_last_bar_forming(
        rates,
        timeframe,
        current_time_epoch=float(expected_end_ts),
    ):
        return False
    closed_session = closed_session_context(
        symbol,
        now_epoch=expected_end_ts,
        item="bar",
        data_age_seconds=freshness_meta.get("data_freshness_seconds"),
    )
    if not closed_session or not bool(
        closed_session.get("freshness_policy_relaxed")
    ):
        return False
    freshness_meta["freshness_policy_relaxed"] = True
    freshness_meta["market_session_status"] = closed_session.get(
        "market_status"
    )
    freshness_meta["market_session_reason"] = closed_session.get(
        "market_status_reason"
    )
    freshness_meta["market_session_source"] = closed_session.get(
        "market_status_source"
    )
    freshness_meta["freshness_note"] = closed_session.get("note")
    return True


def _fetch_rates_with_warmup(  # noqa: C901
    symbol: str,
    mt5_timeframe: int,
    timeframe: TimeframeLiteral,
    candles: int,
    warmup_bars: int,
    start_datetime: Optional[str],
    end_datetime: Optional[str],
    *,
    include_incomplete: bool = False,
    retry: bool = True,
    sanity_check: bool = True,
    diagnostics: Optional[Dict[str, Any]] = None,
):
    """Fetch MT5 rates with optional warmup, retry, and end-bar sanity checks."""
    extra_bars = 0 if include_incomplete else 1
    if diagnostics is not None:
        diagnostics.pop("freshness", None)
    if start_datetime and end_datetime:
        seconds_per_bar, timeframe_error = _resolve_fetch_timeframe_seconds(timeframe)
        if timeframe_error:
            return None, timeframe_error
        from_date, from_date_error = _parse_fetch_datetime_arg(start_datetime)
        to_date, to_date_error = _parse_fetch_datetime_arg(end_datetime)
        if from_date_error or to_date_error:
            return None, from_date_error or to_date_error
        if from_date > to_date:
            return None, "start_datetime must be before end_datetime"
        future_error = _future_start_error(start_datetime, from_date, seconds_per_bar)
        if future_error:
            return None, future_error
        from_date_internal = from_date - timedelta(seconds=seconds_per_bar * (warmup_bars + extra_bars))
        expected_end_ts = _utc_epoch_seconds(to_date)

        def _fetch():
            return _mt5_copy_rates_range(symbol, mt5_timeframe, from_date_internal, to_date)

    elif start_datetime:
        from_date, from_date_error = _parse_fetch_datetime_arg(start_datetime)
        if from_date_error:
            return None, from_date_error
        seconds_per_bar, timeframe_error = _resolve_fetch_timeframe_seconds(timeframe)
        if timeframe_error:
            return None, timeframe_error
        future_error = _future_start_error(start_datetime, from_date, seconds_per_bar)
        if future_error:
            return None, future_error
        to_date = datetime.now(dt_timezone.utc)
        expected_end_ts = _utc_epoch_seconds(to_date)

        def _fetch():
            return _mt5_copy_rates_from(
                symbol,
                mt5_timeframe,
                to_date,
                candles + warmup_bars + extra_bars,
            )

    elif end_datetime:
        to_date, to_date_error = _parse_fetch_datetime_arg(end_datetime)
        if to_date_error:
            return None, to_date_error
        seconds_per_bar, timeframe_error = _resolve_fetch_timeframe_seconds(timeframe)
        if timeframe_error:
            return None, timeframe_error
        expected_end_ts = _utc_epoch_seconds(to_date)

        def _fetch():
            return _mt5_copy_rates_from(symbol, mt5_timeframe, to_date, candles + warmup_bars + extra_bars)

    else:
        utc_now = datetime.now(dt_timezone.utc)
        seconds_per_bar, timeframe_error = _resolve_fetch_timeframe_seconds(timeframe)
        if timeframe_error:
            return None, timeframe_error
        expected_end_ts = _utc_epoch_seconds(utc_now)

        def _fetch():
            return _mt5_copy_rates_from(symbol, mt5_timeframe, utc_now, candles + warmup_bars + extra_bars)

    attempts = FETCH_RETRY_ATTEMPTS if retry else 1
    rates = None
    stale_last_t: Optional[float] = None
    freshness_cutoff: Optional[float] = None
    for idx in range(attempts):
        rates = _fetch()
        if rates is not None and len(rates) > 0:
            last_t = rates[-1]["time"]
            freshness_cutoff = expected_end_ts - seconds_per_bar * (SANITY_BARS_TOLERANCE + extra_bars)
            freshness_meta = _build_candle_freshness_diagnostics(
                last_bar_epoch=last_t,
                expected_end_epoch=expected_end_ts,
                freshness_cutoff_epoch=freshness_cutoff,
            )
            if start_datetime or end_datetime:
                freshness_meta["data_freshness_anchor"] = (
                    FRESHNESS_ANCHOR_QUERY_EXPECTED_END
                )
                freshness_meta["data_freshness_metric"] = (
                    FRESHNESS_METRIC_REQUESTED_RANGE_END_GAP
                )
            else:
                freshness_meta["data_freshness_anchor"] = FRESHNESS_ANCHOR_WALL_CLOCK
                freshness_meta["data_freshness_metric"] = (
                    FRESHNESS_METRIC_LAST_COMPLETED_BAR_AGE
                )
            if diagnostics is not None:
                diagnostics["freshness"] = freshness_meta
            if not sanity_check:
                break
            if bool(freshness_meta.get("last_bar_within_policy_window")):
                stale_last_t = None
                break
            if _relax_live_completed_bar_freshness(
                symbol=symbol,
                rates=rates,
                timeframe=timeframe,
                expected_end_ts=expected_end_ts,
                start_datetime=start_datetime,
                end_datetime=end_datetime,
                freshness_meta=freshness_meta,
            ):
                stale_last_t = None
                break
            stale_last_t = float(last_t)
        if retry and idx < (attempts - 1):
            time.sleep(FETCH_RETRY_DELAY)
    if (
        sanity_check
        and stale_last_t is not None
        and freshness_cutoff is not None
        and rates is not None
        and len(rates) > 0
    ):
        return None, (
            f"Data appears stale for {symbol} {timeframe}: latest completed bar is "
            f"from {_format_time_explicit(stale_last_t)}. Market may be closed; "
            "set allow_stale=true to retrieve the latest "
            "available completed historical bars."
        )
    return rates, None


def _parse_fetch_datetime_arg(value: str) -> tuple[Optional[datetime], Optional[str]]:
    parsed = _parse_start_datetime(value)
    if parsed is None:
        return None, f"Could not parse date {value!r}. {_DATE_FORMAT_HINT}"
    return parsed, None


def _future_start_error(
    start_datetime: str, from_date: datetime, seconds_per_bar: int
) -> Optional[str]:
    """Return an error when the requested start is in the future.

    A future ``start`` yields no historical bars; MT5 silently returns recent
    bars that are then trimmed away, producing an opaque empty success. Reject
    it explicitly (like reversed dates) so callers get an actionable signal.
    A one-bar + clock-skew tolerance avoids false positives near the live bar.
    """
    try:
        from_epoch = _utc_epoch_seconds(from_date)
        tolerance = max(int(seconds_per_bar), 300)
        if from_epoch > time.time() + tolerance:
            return (
                f"start datetime {start_datetime} is in the future; "
                "no historical data is available for future dates."
            )
    except Exception:
        return None
    return None


def _resolve_fetch_timeframe_seconds(timeframe: TimeframeLiteral) -> tuple[Optional[int], Optional[str]]:
    seconds_per_bar = TIMEFRAME_SECONDS.get(timeframe)
    if not seconds_per_bar:
        return None, f"Unable to determine timeframe seconds for {timeframe}"
    return int(seconds_per_bar), None


def _collect_candle_time_alignment(
    symbol: str,
    *,
    timeframe: TimeframeLiteral,
    start_datetime: Optional[str],
    end_datetime: Optional[str],
) -> Optional[Dict[str, Any]]:
    broker_time_check_enabled = bool(
        getattr(mt5_config, "broker_time_check_enabled", False)
    )
    if not broker_time_check_enabled or start_datetime or end_datetime:
        return None
    broker_time_check_ttl_seconds = int(
        getattr(mt5_config, "broker_time_check_ttl_seconds", 60) or 60
    )
    probe_timeframe = "M1" if timeframe != "M1" else timeframe
    try:
        return get_cached_mt5_time_alignment(
            symbol=symbol,
            probe_timeframe=probe_timeframe,
            ttl_seconds=broker_time_check_ttl_seconds,
        )
    except Exception as exc:
        return {
            "symbol": str(symbol),
            "probe_timeframe": probe_timeframe,
            "status": "unavailable",
            "reason": "inspection_failed",
            "error": str(exc),
        }


def _format_rate_times(epoch_series: pd.Series, *, use_client_tz: bool) -> pd.Series:
    epochs = pd.to_numeric(epoch_series, errors="coerce")
    dt_series = pd.to_datetime(epochs, unit="s", utc=True, errors="coerce")

    if use_client_tz:
        try:
            target_tz = mt5_config.get_client_tz()
            if target_tz is None:
                target_tz = datetime.now().astimezone().tzinfo
            if target_tz is not None:
                dt_series = dt_series.dt.tz_convert(target_tz)
        except Exception:
            pass

    formatted = dt_series.map(
        lambda value: (
            _format_datetime_minute_explicit(value.to_pydatetime())
            if pd.notna(value)
            else None
        )
    )
    if bool(formatted.isna().any()):
        formatter = _format_time_explicit_local if use_client_tz else _format_time_explicit
        fallback = epochs.map(lambda value: formatter(float(value)) if pd.notna(value) else None)
        formatted = formatted.where(~formatted.isna(), fallback)
    return formatted


def _timezone_label(*, use_client_tz: bool, client_tz: Any) -> str:
    if not use_client_tz:
        return "UTC"
    if client_tz is None:
        return "local"
    return getattr(client_tz, "key", None) or getattr(client_tz, "zone", None) or str(client_tz)


def _build_rates_df(rates: Any, use_client_tz: bool) -> pd.DataFrame:
    """Normalize raw MT5 rates into a DataFrame with epoch and display time columns."""
    df = _rates_to_df(rates)
    df['__epoch'] = df['time']
    df["time"] = _format_rate_times(df["time"], use_client_tz=use_client_tz)
    if 'volume' not in df.columns and 'tick_volume' in df.columns:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df['volume'] = df['tick_volume']
    return df


def _tick_field_value(tick: Any, name: str) -> Any:
    if tick is None:
        return None
    try:
        return tick[name]
    except Exception:
        pass
    try:
        return getattr(tick, name)
    except Exception:
        pass
    try:
        return tick.get(name)
    except Exception:
        return None


def _fetch_ticks_range_with_retry(
    symbol: str,
    from_date: datetime,
    to_date: datetime,
) -> Any:
    ticks = None
    for _ in range(FETCH_RETRY_ATTEMPTS):
        ticks = _mt5_copy_ticks_range(symbol, from_date, to_date, mt5.COPY_TICKS_ALL)
        if ticks is not None and len(ticks) > 0:
            break
        time.sleep(FETCH_RETRY_DELAY)
    return ticks


def _fetch_recent_ticks_backwards(
    symbol: str,
    *,
    to_date: datetime,
    limit: int,
    min_from_date: Optional[datetime] = None,
) -> Any:
    """Fetch the most recent ticks in bounded backward ranges to avoid huge queries."""
    if limit <= 0:
        return []
    if min_from_date is not None:
        min_is_aware = min_from_date.tzinfo is not None and min_from_date.utcoffset() is not None
        to_is_aware = to_date.tzinfo is not None and to_date.utcoffset() is not None
        if min_is_aware != to_is_aware:
            to_date = to_date.replace(tzinfo=min_from_date.tzinfo if min_is_aware else None)

    chunk_days = 1
    max_lookback_days = max(max(1, int(TICKS_LOOKBACK_DAYS)), 30)
    cursor_end = to_date
    lookback_days_used = 0
    saw_response = False
    collected: List[Any] = []

    while True:
        chunk_from = cursor_end - timedelta(days=chunk_days)
        if min_from_date is not None and chunk_from < min_from_date:
            chunk_from = min_from_date

        ticks_candidate = _fetch_ticks_range_with_retry(symbol, chunk_from, cursor_end)
        if ticks_candidate is not None:
            saw_response = True
            if len(ticks_candidate) > 0:
                collected = list(ticks_candidate) + collected
                if len(collected) > limit:
                    collected = collected[-limit:]
                if len(collected) >= limit:
                    break

        if min_from_date is not None:
            if chunk_from <= min_from_date:
                break
        else:
            lookback_days_used += chunk_days
            if lookback_days_used >= max_lookback_days:
                break

        cursor_end = chunk_from - timedelta(microseconds=1)

    if collected:
        return collected
    if saw_response:
        return []
    return None


def _trim_df_to_target(
    df: pd.DataFrame,
    start_datetime: Optional[str],
    end_datetime: Optional[str],
    candles: int,
    *,
    copy_rows: bool = True,
) -> pd.DataFrame:
    if start_datetime and end_datetime:
        from_dt = _parse_start_datetime(start_datetime)
        to_dt = _parse_start_datetime(end_datetime)
        if not from_dt or not to_dt:
            out = df.iloc[0:0]
            return out.copy() if copy_rows else out
        target_from = _utc_epoch_seconds(from_dt)
        target_to = _utc_epoch_seconds(to_dt)
        out = df.loc[(df['__epoch'] >= target_from) & (df['__epoch'] <= target_to)]
    elif start_datetime:
        from_dt = _parse_start_datetime(start_datetime)
        if not from_dt:
            out = df.iloc[0:0]
            return out.copy() if copy_rows else out
        target_from = _utc_epoch_seconds(from_dt)
        out = df.loc[df['__epoch'] >= target_from]
        if len(out) > candles:
            out = out.iloc[-candles:]
    elif end_datetime:
        to_dt = _parse_start_datetime(end_datetime)
        if not to_dt:
            out = df.iloc[0:0]
            return out.copy() if copy_rows else out
        target_to = _utc_epoch_seconds(to_dt)
        out = df.loc[df['__epoch'] <= target_to]
        if len(out) > candles:
            out = out.iloc[-candles:]
    else:
        out = df.iloc[-candles:] if len(df) > candles else df
    return out.copy() if copy_rows else out


def _normalize_indicator_spec(indicators: Optional[List[IndicatorSpec]]) -> Optional[str]:
    """Normalize indicator input into the compact internal string format."""
    if indicators is None:
        return None

    source: Any = indicators
    if isinstance(source, str):
        payload = source.strip()
        if payload.startswith('[') or payload.startswith('{'):
            try:
                source = json.loads(payload)
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid indicator JSON: {exc}") from exc

    if isinstance(source, (list, tuple)):
        parts: List[str] = []
        for item in source:
            if isinstance(item, dict) and 'name' in item:
                name = str(item.get('name'))
                params = item.get('params') or []
                if isinstance(params, (list, tuple)) and len(params) > 0:
                    args_str = ",".join(str(coerce_scalar(str(param))) for param in params)
                    parts.append(f"{name}({args_str})")
                elif isinstance(params, dict) and len(params) > 0:
                    args_str = ",".join(
                        f"{str(key).strip()}={coerce_scalar(str(param))}"
                        for key, param in params.items()
                        if str(key).strip()
                    )
                    parts.append(f"{name}({args_str})" if args_str else name)
                else:
                    parts.append(name)
            else:
                parts.append(str(item))
        return ",".join(parts)

    return str(source)


def _normalize_indicator_spec_for_display(ti_spec: Optional[str]) -> str:
    text = str(ti_spec or "").strip()
    if not text:
        return ""
    return re.sub(r"(?<![\d.])([+-]?\d+)\.0(?!\d)", r"\1", text)


def _display_indicator_column_name(column: str) -> str:
    text = str(column or "")
    if text.startswith("ATRr_"):
        text = "ATR_" + text[len("ATRr_") :]
    text = re.sub(r"^MACD([a-z])_", r"MACD_\1_", text)
    text = _normalize_indicator_spec_for_display(text)
    return text.replace(".", "_").lower()


def _normalize_indicator_columns_for_display(
    df: pd.DataFrame,
    columns: List[str],
) -> List[str]:
    if not columns:
        return []

    rename_map: Dict[str, str] = {}
    normalized: List[str] = []
    for column in columns:
        old_name = str(column)
        new_name = _display_indicator_column_name(old_name)
        normalized_name = old_name
        if new_name != old_name:
            if old_name in df.columns and new_name not in df.columns and new_name not in rename_map.values():
                rename_map[old_name] = new_name
                normalized_name = new_name
        normalized.append(normalized_name)

    if rename_map:
        df.rename(columns=rename_map, inplace=True)
    return normalized


def _extend_unique_headers(headers: List[str], columns: List[str]) -> None:
    for column in columns:
        if column not in headers:
            headers.append(column)


def _build_candle_headers(
    rates: Any,
    ohlcv: Optional[str],
    *,
    include_spread: bool = False,
) -> List[str]:
    """Build the initial candle header set before transforms add derived columns."""
    def _volume_values(field: str) -> List[int]:
        values: List[int] = []
        for rate in rates:
            try:
                value = rate[field]
            except (IndexError, KeyError, TypeError, ValueError):
                value = 0
            try:
                values.append(int(value))
            except (TypeError, ValueError, OverflowError):
                values.append(0)
        return values

    tick_volumes = _volume_values("tick_volume")
    real_volumes = _volume_values("real_volume")

    has_tick_volume = len(set(tick_volumes)) > 1 or any(value != 0 for value in tick_volumes)
    has_real_volume = len(set(real_volumes)) > 1 or any(value != 0 for value in real_volumes)
    requested = _normalize_ohlcv_arg(ohlcv)

    headers = ["time"]
    if requested is not None:
        if "O" in requested:
            headers.append("open")
        if "H" in requested:
            headers.append("high")
        if "L" in requested:
            headers.append("low")
        if "C" in requested:
            headers.append("close")
        if "V" in requested:
            headers.append("tick_volume")
        if include_spread:
            headers.append("spread")
        return headers

    headers.extend(["open", "high", "low", "close"])
    if has_tick_volume:
        headers.append("tick_volume")
    if has_real_volume:
        headers.append("real_volume")
    if include_spread:
        headers.append("spread")
    return headers


def _candle_volume_metadata(headers: List[str]) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    units: Dict[str, str] = {}
    if "tick_volume" in headers:
        meta["volume_type"] = "tick_count"
        meta["volume_note"] = (
            "MT5 tick_volume is broker tick count for the bar, not exchange traded volume."
        )
        units["tick_volume"] = "broker_tick_count"
    if "real_volume" in headers:
        meta["real_volume_type"] = "traded_volume"
        units["real_volume"] = "traded_volume"
    if units:
        meta["units"] = units
    return meta


def _candle_time_convention_metadata(timeframe: str) -> Dict[str, str]:
    tf = str(timeframe or "").strip().upper()
    if tf in {"D1", "W1", "MN1"}:
        return {
            "bar_time_convention": "bar_open_time",
            "bar_time_note": (
                "MT5 daily, weekly, and monthly candle time is the broker/server "
                "bar open time; it may not be UTC midnight."
            ),
        }
    return {"bar_time_convention": "bar_open_time"}


def _validate_ohlcv_selection(ohlcv: Optional[str]) -> Optional[str]:
    if ohlcv is None or str(ohlcv).strip() == "":
        return None
    if _normalize_ohlcv_arg(ohlcv) is not None:
        return None
    return (
        "Invalid ohlcv value. Use all, ohlcv, ohlc, close/price, compact "
        "letters from o/h/l/c/v, or comma-separated names such as "
        "open,high,low,close,volume."
    )


def _append_denoise_application(
    denoise_apps: List[Dict[str, Any]],
    source_spec: Any,
    *,
    default_when: str,
    default_causality: str,
    default_keep_original: bool,
    added_columns: List[str],
    overwritten_columns: List[str],
) -> None:
    if not added_columns and not overwritten_columns:
        return
    try:
        denoise_meta = dict(source_spec or {})
        columns = denoise_meta.get('columns', 'close')
        keep_original = bool(denoise_meta.get('keep_original', default_keep_original))
        denoise_apps.append(
            {
                'method': str(denoise_meta.get('method', 'none')).lower(),
                'when': str(denoise_meta.get('when', default_when)).lower(),
                'causality': str(denoise_meta.get('causality', default_causality)),
                'keep_original': keep_original,
                'columns': columns,
                'params': denoise_meta.get('params') or {},
                'added_columns': added_columns,
                'overwrote_columns': overwritten_columns,
            }
        )
    except Exception:
        pass


def _latest_indicator_values_missing(df: pd.DataFrame, columns: List[str]) -> bool:
    required_columns = _indicator_columns_required_for_completeness(columns)
    if not required_columns or len(df) <= 0:
        return False
    for column in required_columns:
        if column not in df.columns:
            return True
        value = df[column].iloc[-1]
        try:
            if pd.isna(value):
                return True
        except Exception:
            if value is None:
                return True
    return False


def _apply_pre_ti_denoise(
    df: pd.DataFrame,
    headers: List[str],
    denoise: Optional[DenoiseSpec],
    denoise_apps: List[Dict[str, Any]],
) -> None:
    if not denoise:
        return

    normalized = _normalize_denoise_spec(denoise, default_when='pre_ti')
    added_columns: List[str] = []
    if normalized and str(normalized.get('when', 'pre_ti')).lower() == 'pre_ti':
        added_columns = _apply_denoise_util(df, normalized, default_when='pre_ti')
        last_application = df.attrs.get("denoise_last_application")
        overwritten_columns = (
            list(last_application.get("overwrote_columns") or [])
            if isinstance(last_application, dict)
            else []
        )
        _extend_unique_headers(headers, added_columns)
        _append_denoise_application(
            denoise_apps,
            normalized,
            default_when='pre_ti',
            default_causality='causal',
            default_keep_original=True,
            added_columns=added_columns,
            overwritten_columns=overwritten_columns,
        )


def _apply_indicator_stage(
    df: pd.DataFrame,
    headers: List[str],
    ti_spec: Optional[str],
    denoise: Optional[DenoiseSpec],
) -> List[str]:
    ti_cols: List[str] = []
    if not ti_spec:
        return ti_cols

    columns_before = {str(column) for column in df.columns}
    reported_columns = [str(column) for column in _apply_ta_indicators(df, ti_spec)]
    created_columns = [
        str(column) for column in df.columns if str(column) not in columns_before
    ]
    ti_cols = list(dict.fromkeys([*reported_columns, *created_columns]))
    ti_cols = _normalize_indicator_columns_for_display(df, ti_cols)
    _extend_unique_headers(headers, ti_cols)

    if denoise and ti_cols:
        dn_base = _normalize_denoise_spec(denoise, default_when='post_ti')
        if dn_base and bool(dn_base.get('apply_to_ti') or dn_base.get('ti')):
            dn_ti = dict(dn_base)
            dn_ti['columns'] = list(ti_cols)
            dn_ti.setdefault('when', 'post_ti')
            dn_ti.setdefault('keep_original', False)
            _apply_denoise_util(df, dn_ti, default_when='post_ti')

    return ti_cols


def _indicator_columns_required_for_completeness(columns: List[str]) -> List[str]:
    """Return indicator columns that must be populated on every output row.

    pandas-ta-classic's Supertrend long and short bands are regime-specific:
    ``SUPERTl`` is null in short regimes and ``SUPERTs`` is null in long regimes.
    Those nulls express which band is inactive, rather than a warmup failure.
    """
    required: List[str] = []
    for column in columns:
        family = str(column or "").split("_", 1)[0].lower()
        if family in {"supertl", "superts"}:
            continue
        required.append(column)
    return required


def _indicator_columns_with_missing_values(
    df: pd.DataFrame,
    ti_cols: List[str],
) -> List[str]:
    missing_cols: List[str] = []
    for col in _indicator_columns_required_for_completeness(ti_cols):
        if col not in df.columns:
            continue
        try:
            if bool(df[col].isna().any()):
                missing_cols.append(str(col))
        except Exception:
            continue
    return missing_cols


def _drop_incomplete_indicator_rows(
    df: pd.DataFrame,
    ti_cols: List[str],
) -> Tuple[pd.DataFrame, int, List[str]]:
    required_cols = _indicator_columns_required_for_completeness(ti_cols)
    existing_cols = [col for col in required_cols if col in df.columns]
    if not existing_cols or len(df) == 0:
        return df, 0, []

    missing_cols = _indicator_columns_with_missing_values(df, ti_cols)
    if not missing_cols:
        return df, 0, []

    missing_mask = df[existing_cols].isna().any(axis=1)
    dropped_rows = int(missing_mask.sum())
    if dropped_rows <= 0:
        return df, 0, []

    return df.loc[~missing_mask].copy(), dropped_rows, missing_cols


def _apply_post_ti_denoise(
    df: pd.DataFrame,
    headers: List[str],
    denoise: Optional[DenoiseSpec],
    denoise_apps: List[Dict[str, Any]],
) -> None:
    if not denoise:
        return
    if not (isinstance(denoise, dict) and denoise.get('when') not in (None, "")):
        return

    normalized = _normalize_denoise_spec(denoise, default_when='post_ti')
    added_columns: List[str] = []
    if normalized and str(normalized.get('when', 'post_ti')).lower() == 'post_ti':
        added_columns = _apply_denoise_util(df, normalized, default_when='post_ti')
        last_application = df.attrs.get("denoise_last_application")
        overwritten_columns = (
            list(last_application.get("overwrote_columns") or [])
            if isinstance(last_application, dict)
            else []
        )
        _extend_unique_headers(headers, added_columns)
        _append_denoise_application(
            denoise_apps,
            normalized,
            default_when='post_ti',
            default_causality='causal',
            default_keep_original=True,
            added_columns=added_columns,
            overwritten_columns=overwritten_columns,
        )


def _rebuild_candle_indicator_window(
    rates: Any,
    *,
    use_client_tz: bool,
    denoise: Optional[DenoiseSpec],
    ti_spec: Optional[str],
    headers: List[str],
) -> Tuple[pd.DataFrame, List[str]]:
    """Rebuild the warmup window and re-run the pre-indicator stages."""
    df = _build_rates_df(rates, use_client_tz)
    if denoise:
        normalized = _normalize_denoise_spec(denoise, default_when='pre_ti')
        if normalized and str(normalized.get('when', 'pre_ti')).lower() == 'pre_ti':
            _apply_denoise_util(df, normalized, default_when='pre_ti')
    ti_cols = _apply_indicator_stage(df, headers, ti_spec, denoise)
    return df, ti_cols


def _collect_session_gaps(
    df: pd.DataFrame,
    *,
    timeframe: TimeframeLiteral,
    use_client_tz: bool,
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    session_gaps: List[Dict[str, Any]] = []
    expected_bar_seconds = float(TIMEFRAME_SECONDS.get(timeframe, 0) or 0)
    if expected_bar_seconds <= 0 or '__epoch' not in df.columns or len(df) <= 1:
        return session_gaps, None

    try:
        epochs = pd.to_numeric(df['__epoch'], errors='coerce').to_numpy(dtype=float)
        threshold = expected_bar_seconds * 1.5
        for index in range(1, len(epochs)):
            prev_t = float(epochs[index - 1])
            curr_t = float(epochs[index])
            if not (math.isfinite(prev_t) and math.isfinite(curr_t)):
                continue

            gap_seconds = float(curr_t - prev_t)
            if gap_seconds <= threshold:
                continue

            if use_client_tz:
                from_disp = _format_time_explicit_local(prev_t)
                to_disp = _format_time_explicit_local(curr_t)
            else:
                from_disp = _format_time_explicit(prev_t)
                to_disp = _format_time_explicit(curr_t)

            missing_bars_est = max(1, int(round(gap_seconds / expected_bar_seconds)) - 1)
            prev_dt = datetime.fromtimestamp(prev_t, tz=dt_timezone.utc)
            curr_dt = datetime.fromtimestamp(curr_t, tz=dt_timezone.utc)
            crosses_weekend = (
                prev_dt.weekday() >= 5
                or curr_dt.weekday() >= 5
                or ((curr_t - prev_t) >= (36.0 * 3600.0))
            )
            gap_context = "weekend/session break" if crosses_weekend else "session break"
            session_gaps.append(
                {
                    "from": from_disp,
                    "to": to_disp,
                    "gap_seconds": gap_seconds,
                    "expected_bar_seconds": expected_bar_seconds,
                    "missing_bars_est": int(missing_bars_est),
                    "context": gap_context,
                }
            )
    except Exception as exc:
        logger.warning("Session gap diagnostics unavailable: %s", exc)
        return session_gaps, "Session gap diagnostics unavailable."

    return session_gaps, None


def _annotate_candle_gap_rows(
    payload: Dict[str, Any],
    session_gaps: List[Dict[str, Any]],
) -> None:
    rows = payload.get("data")
    if not isinstance(rows, list) or not session_gaps:
        return
    gaps_by_to = {
        str(gap.get("to")): {
            "gap_seconds": gap.get("gap_seconds"),
            "missing_bars_est": gap.get("missing_bars_est"),
            "context": gap.get("context"),
        }
        for gap in session_gaps
        if isinstance(gap, dict) and gap.get("to") not in (None, "")
    }
    if not gaps_by_to:
        return
    for row in rows:
        if not isinstance(row, dict):
            continue
        gap = gaps_by_to.get(str(row.get("time")))
        if gap:
            row["gap_before"] = {
                key: value for key, value in gap.items() if value not in (None, "")
            }


def _format_candle_times(
    df: pd.DataFrame,
    headers: List[str],
    *,
    time_as_epoch: bool,
    use_client_tz: bool,
    client_tz: Any,
) -> None:
    if 'time' not in headers or len(df) <= 0:
        return

    epochs = pd.to_numeric(df['__epoch'], errors='coerce').astype(float)
    if time_as_epoch:
        df['time'] = epochs
        df.attrs['_tz_used_name'] = 'UTC'
        return

    tz_used_name = 'UTC'
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        time_values = pd.to_datetime(epochs, unit='s', utc=True)
        if use_client_tz:
            tz_used_name = getattr(client_tz, 'zone', None) or str(client_tz)
            time_values = time_values.dt.tz_convert(client_tz)
        df['time'] = time_values.map(lambda value: _format_datetime_minute_explicit(value.to_pydatetime()))
    df.attrs['_tz_used_name'] = tz_used_name


def _normalize_simplify_spec(
    simplify: Optional[SimplifySpec],
    *,
    limit: int,
    fallback_rows: int,
) -> Optional[Dict[str, Any]]:
    if simplify is None:
        return None

    simplify_eff = dict(simplify)
    simplify_eff['mode'] = str(simplify_eff.get('mode', SIMPLIFY_DEFAULT_MODE)).lower().strip()
    has_points = any(
        key in simplify_eff and simplify_eff[key] is not None
        for key in ("points", "target_points", "max_points", "ratio")
    )
    if has_points:
        return simplify_eff

    try:
        default_pts = max(3, int(round(int(limit) * SIMPLIFY_DEFAULT_POINTS_RATIO_FROM_LIMIT)))
    except Exception:
        default_pts = max(3, int(round(fallback_rows * SIMPLIFY_DEFAULT_POINTS_RATIO_FROM_LIMIT)))
    simplify_eff['points'] = default_pts
    return simplify_eff


def _public_simplify_meta(meta: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(meta, dict):
        return None
    out: Dict[str, Any] = {}
    for key in ("method", "mode", "points", "ratio"):
        value = meta.get(key)
        if value is not None:
            out[key] = value
    return out or None


def fetch_candles(  # noqa: C901
    symbol: str,
    timeframe: TimeframeLiteral = "H1",
    limit: int = DEFAULT_ROW_LIMIT,
    start: Optional[str] = None,
    end: Optional[str] = None,
    ohlcv: Optional[str] = None,
    indicators: Optional[List[IndicatorSpec]] = None,
    denoise: Optional[DenoiseSpec] = None,
    simplify: Optional[SimplifySpec] = None,
    time_as_epoch: bool = False,
    *,
    include_spread: bool = False,
    include_incomplete: bool = False,
    allow_stale: bool = False,
) -> Dict[str, Any]:
    """Return historical candles as tabular data."""
    try:
        symbol = resolve_broker_symbol_name(symbol)
        query_started_at = time.perf_counter()
        # Backward/compat mappings to internal variable names used in implementation
        candles = int(limit)
        if candles <= 0:
            return {"error": "limit must be greater than 0."}
        start_datetime = start
        end_datetime = end
        ti = indicators
        # Validate timeframe using the shared map
        if timeframe not in TIMEFRAME_MAP:
            return {"error": invalid_timeframe_error(timeframe, TIMEFRAME_MAP)}
        mt5_timeframe = TIMEFRAME_MAP[timeframe]
        ohlcv_error = _validate_ohlcv_selection(ohlcv)
        if ohlcv_error is not None:
            return {"error": ohlcv_error}
        
        # Ensure symbol is ready; remember original visibility to restore later
        _info_before = get_symbol_info_cached(symbol)
        with _symbol_ready_guard(symbol, info_before=_info_before) as (err, _info):
            if err:
                return {"error": err}
            price_digits = _symbol_price_digits(_info, _info_before)
            price_currency = _symbol_price_currency(_info, _info_before)
            price_basis = _symbol_candle_price_basis(_info, _info_before)

            try:
                ti_spec = _normalize_indicator_spec(ti)
            except ValueError as exc:
                return {"error": str(exc)}
            indicator_syntax_error = _indicator_param_syntax_error(ti_spec)
            if indicator_syntax_error:
                return {"error": indicator_syntax_error}
            # Determine warmup bars if technical indicators requested
            unknown_indicators = _find_unknown_ta_indicators(ti_spec or "")
            if unknown_indicators:
                return {
                    "error": (
                        "Unknown indicator(s): "
                        + ", ".join(unknown_indicators)
                        + ". Parameters use name(params) syntax, e.g. rsi(14) or "
                        "macd(12,26,9); use indicators_list to view valid indicator names."
                    )
                }
            warmup_bars = _estimate_warmup_bars(ti_spec)
            rate_fetch_diagnostics: Dict[str, Any] = {}
            freshness_diagnostics: Optional[Dict[str, Any]] = None
            historical_bounds_requested = bool(start_datetime or end_datetime)

            rates, rates_error = _fetch_rates_with_warmup(
                symbol,
                mt5_timeframe,
                timeframe,
                candles,
                warmup_bars,
                start_datetime,
                end_datetime,
                include_incomplete=include_incomplete,
                retry=True,
                sanity_check=not bool(allow_stale) and not historical_bounds_requested,
                diagnostics=rate_fetch_diagnostics,
            )
            freshness_diagnostics = rate_fetch_diagnostics.get("freshness")
            time_normalization = describe_mt5_time_normalization(symbol=symbol)
            if rates_error:
                error_payload: Dict[str, Any] = {"error": rates_error}
                if isinstance(freshness_diagnostics, dict):
                    error_payload["details"] = {
                        "diagnostics": {
                            "freshness": dict(freshness_diagnostics),
                        },
                    }
                return error_payload
        # visibility handled by _symbol_ready_guard
        
        if rates is None:
            return {"error": _describe_rate_fetch_error(symbol, info_before=_info_before)}

        # Generate tabular format with dynamic column filtering
        if len(rates) == 0:
            return _build_no_data_error_with_context(
                symbol, timeframe, mt5_timeframe, start_datetime, end_datetime
            )
        raw_bars_fetched = int(len(rates))
        live_bar_reference_epoch = _resolve_live_bar_reference_epoch(symbol, timeframe)
        initial_incomplete_trimmed = False
        if not include_incomplete:
            rates_before_trim = int(len(rates))
            rates = _drop_incomplete_tail(
                rates,
                timeframe,
                current_time_epoch=live_bar_reference_epoch,
            )
            initial_incomplete_trimmed = int(len(rates)) < rates_before_trim
        if len(rates) == 0:
            return _build_no_data_error_with_context(
                symbol, timeframe, mt5_timeframe, start_datetime, end_datetime
            )
        headers = _build_candle_headers(
            rates,
            ohlcv,
            include_spread=include_spread,
        )
        
        # Construct DataFrame to support indicators and consistent output
        client_tz = _resolve_client_tz()
        _use_ctz = client_tz is not None
        df = _build_rates_df(rates, _use_ctz)
        quality_rows_removed = 0
        ohlcv_warnings: List[str] = []
        try:
            rows_before_quality = int(len(df))
            df, new_ohlcv_warnings = validate_and_clean_ohlcv_frame(df, epoch_col="__epoch")
        except ValueError as exc:
            return {"error": str(exc)}
        quality_rows_removed += max(0, rows_before_quality - int(len(df)))
        ohlcv_warnings.extend(new_ohlcv_warnings)
        if len(df) == 0:
            return {"error": f"No valid candle data available for {symbol}"}

        # Track denoise metadata if applied
        denoise_apps: List[Dict[str, Any]] = []
        denoise_warnings: List[str] = []
        ti_warnings: List[str] = []
        _apply_pre_ti_denoise(df, headers, denoise, denoise_apps)
        denoise_warnings.extend(_consume_denoise_warnings(df))
        ti_cols = _apply_indicator_stage(df, headers, ti_spec, denoise)
        denoise_warnings.extend(_consume_denoise_warnings(df))

        # Filter out warmup region to return the intended target window only
        df = _trim_df_to_target(df, start_datetime, end_datetime, candles, copy_rows=True)
        rows_after_target_trim = int(len(df))
        warmup_retry_meta: Dict[str, Any] = {
            "applied": False,
            "warmup_bars": int(warmup_bars),
        }
        indicator_rows_dropped = 0

        # If TI requested, check for NaNs and retry once with increased warmup
        if ti_spec and ti_cols:
            try:
                if _indicator_columns_with_missing_values(df, ti_cols):
                    # Increase warmup and refetch once
                    warmup_bars_retry = max(int(warmup_bars * TI_NAN_WARMUP_FACTOR), warmup_bars + TI_NAN_WARMUP_MIN_ADD)
                    rates_retry, rates_retry_error = _fetch_rates_with_warmup(
                        symbol,
                        mt5_timeframe,
                        timeframe,
                        candles,
                        warmup_bars_retry,
                        start_datetime,
                        end_datetime,
                        include_incomplete=include_incomplete,
                        retry=True,
                        sanity_check=not bool(allow_stale) and not historical_bounds_requested,
                    )
                    retry_applied = rates_retry is not None and len(rates_retry) > 0
                    warmup_retry_meta = {
                        "applied": bool(retry_applied),
                        "warmup_bars": int(warmup_bars_retry),
                        "raw_bars_fetched": int(len(rates_retry)) if rates_retry is not None else 0,
                    }
                    if rates_retry_error:
                        warmup_retry_meta["error"] = str(rates_retry_error)
                        ti_warnings.append(
                            "Indicator warmup retry failed: "
                            f"{rates_retry_error}. Indicator values may be incomplete."
                        )
                    # Rebuild df and indicators with the larger window
                    if retry_applied:
                        df, ti_cols = _rebuild_candle_indicator_window(
                            rates_retry,
                            use_client_tz=_use_ctz,
                            denoise=denoise,
                            ti_spec=ti_spec,
                            headers=headers,
                        )
                        denoise_warnings.extend(_consume_denoise_warnings(df))
                        try:
                            rows_before_quality = int(len(df))
                            df, retry_ohlcv_warnings = validate_and_clean_ohlcv_frame(df, epoch_col="__epoch")
                        except ValueError as exc:
                            return {"error": str(exc)}
                        quality_rows_removed += max(0, rows_before_quality - int(len(df)))
                        for warning_text in retry_ohlcv_warnings:
                            if warning_text not in ohlcv_warnings:
                                ohlcv_warnings.append(warning_text)
                        if len(df) == 0:
                            return {"error": f"No valid candle data available for {symbol}"}
                        # Re-trim to target window
                        df = _trim_df_to_target(df, start_datetime, end_datetime, candles, copy_rows=False)
                        rows_after_target_trim = int(len(df))
            except Exception as exc:
                warmup_retry_meta["error"] = str(exc)
                logger.warning("Indicator warmup retry failed", exc_info=True)
                ti_warnings.append(
                    f"Indicator warmup retry failed: {exc}. Indicator values may be incomplete."
                )

        if ti_spec and ti_cols:
            df, dropped_rows, missing_indicator_cols = _drop_incomplete_indicator_rows(df, ti_cols)
            if dropped_rows:
                indicator_rows_dropped += int(dropped_rows)
                warmup_retry_meta["incomplete_rows_dropped"] = int(dropped_rows)
                warmup_retry_meta["incomplete_indicator_columns"] = list(missing_indicator_cols)
                if len(df) == 0:
                    warning_text = (
                        f"Dropped {dropped_rows} candle rows with incomplete indicator values; "
                        "no complete indicator rows remain."
                    )
                    if warning_text not in ti_warnings:
                        ti_warnings.append(warning_text)
                    return {
                        "success": False,
                        "error_code": "data_fetch_candles_incomplete_indicators",
                        "error": (
                            f"No complete indicator rows available for {symbol} {timeframe}; "
                            "increase limit, reduce indicator lookback, or allow a larger "
                            "historical warmup window."
                        ),
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "indicator_columns": list(missing_indicator_cols),
                        "warnings": list(ti_warnings),
                        "meta": {
                            "diagnostics": {
                                "query": {
                                    "warmup_retry": warmup_retry_meta,
                                },
                            },
                        },
                    }
                ti_warnings.append(
                    f"Dropped {dropped_rows} candle rows with incomplete indicator values after warmup."
                )
                rows_after_target_trim = int(len(df))

        # Authoritative incomplete-tail trim: covers the initial fetch *and*
        # the TI-retry rebuild path, applied before any non-causal transforms.
        _trimmed_incomplete = False
        if not include_incomplete:
            df, _trimmed_incomplete = _drop_incomplete_tail_df(
                df,
                timeframe,
                current_time_epoch=live_bar_reference_epoch,
            )

        # Optional post-TI denoising (adds new columns by default)
        _apply_post_ti_denoise(df, headers, denoise, denoise_apps)
        denoise_warnings.extend(_consume_denoise_warnings(df))

        # Ensure headers are unique and exist in df
        headers = [h for h in headers if h in df.columns]

        # Detect large time discontinuities (e.g., closed session windows) and
        # surface them explicitly so users can interpret forecast/analysis gaps.
        session_gaps, session_gap_warning = _collect_session_gaps(df, timeframe=timeframe, use_client_tz=_use_ctz)
        expected_bar_seconds = float(TIMEFRAME_SECONDS.get(timeframe, 0) or 0)

        # Reformat time consistently across rows for display, unless caller
        # explicitly requests numeric UTC epoch seconds.
        _format_candle_times(
            df,
            headers,
            time_as_epoch=time_as_epoch,
            use_client_tz=_use_ctz,
            client_tz=client_tz,
        )

        # Optionally reduce number of rows for readability/output size
        original_rows = len(df)
        simplify_eff = _normalize_simplify_spec(simplify, limit=limit, fallback_rows=original_rows)
        df, simplify_meta = _simplify_dataframe_rows_ext(df, headers, simplify_eff if simplify_eff is not None else simplify)
        # If simplify changed representation, respect returned headers
        if simplify_meta is not None and 'headers' in simplify_meta and isinstance(simplify_meta['headers'], list):
            headers = [h for h in simplify_meta['headers'] if isinstance(h, str)]

        # Assemble rows from (possibly reduced) DataFrame for selected headers
        tail_is_forming = _is_last_bar_forming(
            df,
            timeframe,
            current_time_epoch=live_bar_reference_epoch,
        )
        ti_added_cols = [str(c) for c in ti_cols if isinstance(c, str)]
        price_indicator_cols = _price_indicator_columns(ti_added_cols)
        rows = _format_numeric_rows_from_df(df, headers, stringify=False)
        rows = _round_row_price_columns(
            rows,
            headers,
            digits=price_digits,
            price_columns=frozenset([*_CANDLE_PRICE_COLUMNS, *price_indicator_cols]),
        )
        as_of_epoch = time.time()
        query_latency_ms = round((time.perf_counter() - query_started_at) * 1000.0, 3)
        query_mode = "range" if (start_datetime or end_datetime) else "latest"
        broker_time_check_result = _collect_candle_time_alignment(
            symbol,
            timeframe=timeframe,
            start_datetime=start_datetime,
            end_datetime=end_datetime,
        )
        # Build tabular payload
        payload = _table_from_rows(headers, rows)
        # `candles` is the domain-specific row count for this tool; avoid
        # duplicating the generic `count` field in public output.
        payload.pop("count", None)
        if time_as_epoch:
            for row in payload.get("data", []) or []:
                if isinstance(row, dict) and "time" in row:
                    try:
                        row["time"] = float(row["time"])
                    except Exception:
                        pass
        
        candles_returned = int(len(df))
        candles_requested = int(candles)
        candles_excluded = max(0, candles_requested - candles_returned)
        incomplete_candles_skipped = int(bool(initial_incomplete_trimmed)) + int(bool(_trimmed_incomplete))
        has_forming_candle = bool(initial_incomplete_trimmed or _trimmed_incomplete or tail_is_forming)
        forming_candle_included = bool(include_incomplete and tail_is_forming)
        forming_candle_skipped = bool(incomplete_candles_skipped and not include_incomplete)
        latest_indicator_missing = _latest_indicator_values_missing(df, ti_added_cols)
        if forming_candle_included:
            forming_candle_status = "included"
        elif forming_candle_skipped:
            forming_candle_status = "skipped"
        elif has_forming_candle:
            forming_candle_status = "detected"
        else:
            forming_candle_status = "none"
        remaining_after_forming = max(0, candles_excluded - incomplete_candles_skipped)
        indicator_excluded = min(int(indicator_rows_dropped), remaining_after_forming)
        remaining_after_indicator = max(0, remaining_after_forming - indicator_excluded)
        quality_excluded = min(int(quality_rows_removed), remaining_after_indicator)
        remaining_excluded = max(0, remaining_after_indicator - quality_excluded)
        window_shortfall = remaining_excluded if (start_datetime or end_datetime) else 0
        source_shortfall = max(0, remaining_excluded - window_shortfall)
        candle_excluded_total = (
            incomplete_candles_skipped
            + indicator_excluded
            + quality_excluded
            + window_shortfall
            + source_shortfall
        )
        candle_counts = {
            "requested": candles_requested,
            "returned": candles_returned,
            "excluded": {
                "forming_bar": incomplete_candles_skipped,
                "indicator_warmup": indicator_excluded,
                "quality_filtered": quality_excluded,
                "window_or_source_shortfall": window_shortfall + source_shortfall,
                "total": candle_excluded_total,
            },
        }
        volume_metadata = _candle_volume_metadata(headers)
        latest_bar_epoch = None
        first_bar_time = None
        latest_bar_time = None
        data_rows = payload.get("data")
        if isinstance(data_rows, list) and data_rows:
            first_row = data_rows[0]
            latest_row = data_rows[-1]
            if isinstance(first_row, dict):
                first_bar_time = first_row.get("time")
            if isinstance(latest_row, dict):
                latest_bar_time = latest_row.get("time")
        try:
            if len(df) > 0 and "__epoch" in df.columns:
                latest_bar_epoch = float(df["__epoch"].iloc[-1])
        except Exception:
            latest_bar_epoch = None

        payload.update({
            "success": True,
            "symbol": symbol,
            "timeframe": timeframe,
            "candles": candles_returned,
            "requested_limit": candles_requested,
            "returned_count": candles_returned,
            "as_of": format_epoch_utc(as_of_epoch),
            **time_normalization,
            **volume_metadata,
            **_candle_time_convention_metadata(timeframe),
            "candles_requested": candles_requested,
            "candles_excluded": candles_excluded,
            "candle_counts": candle_counts,
            "incomplete_candles_skipped": incomplete_candles_skipped,
            "has_forming_candle": has_forming_candle,
            "forming_candle_status": forming_candle_status,
            "forming_candle_included": forming_candle_included,
            "forming_candle_skipped": forming_candle_skipped,
            "meta": {
                "diagnostics": {
                    "query": {
                        "mode": query_mode,
                        "include_spread": bool(include_spread),
                        "include_incomplete": bool(include_incomplete),
                        "latency_ms": query_latency_ms,
                        "requested_bars": candles_requested,
                        "warmup_bars": int(warmup_bars),
                        "raw_bars_fetched": raw_bars_fetched,
                        "rows_after_target_trim": rows_after_target_trim,
                        "indicator_rows_dropped": int(indicator_rows_dropped),
                        "quality_rows_removed": int(quality_rows_removed),
                        "cache_status": "unknown",
                        "warmup_retry": warmup_retry_meta,
                    },
                    "indicators": {
                        "requested": bool(ti_spec),
                        "spec": _normalize_indicator_spec_for_display(ti_spec),
                        "added_columns": ti_added_cols,
                    },
                    "session_gaps": {
                        "expected_bar_seconds": float(expected_bar_seconds) if expected_bar_seconds > 0 else None,
                    },
                    "time_normalization": dict(time_normalization),
                },
            },
        })
        if broker_time_check_result is not None:
            payload["meta"]["diagnostics"]["mt5_time_alignment"] = (
                dict(broker_time_check_result)
            )
        if price_indicator_cols and price_digits > 0:
            rounding_meta = {
                "price_columns": price_indicator_cols,
                "price_precision": int(price_digits),
                "policy": "symbol_price_precision",
            }
            payload["indicator_rounding"] = rounding_meta
            payload["meta"]["diagnostics"]["indicators"]["rounding"] = rounding_meta
        data_window = {
            "start": first_bar_time,
            "end": latest_bar_time,
            "requested_limit": candles_requested,
            "returned_count": candles_returned,
            "latest_bar_complete": not forming_candle_included,
        }
        if latest_bar_epoch is not None and query_mode != "range":
            latest_bar_age_epoch = float(latest_bar_epoch)
            latest_bar_age_metric = "latest_bar_open_age_seconds"
            if not forming_candle_included and expected_bar_seconds > 0:
                latest_bar_age_epoch = float(latest_bar_epoch) + float(expected_bar_seconds)
                latest_bar_age_metric = "latest_completed_bar_close_age_seconds"
            data_window["latest_bar_age_seconds"] = round(
                max(0.0, float(as_of_epoch) - latest_bar_age_epoch),
                3,
            )
            data_window["latest_bar_age_metric"] = latest_bar_age_metric
        payload["data_window"] = {
            key: value
            for key, value in data_window.items()
            if value is not None
        }
        if ohlcv not in (None, ""):
            payload["ohlcv_filter_applied"] = True
            payload["ohlcv_filter"] = str(ohlcv).strip()
        if forming_candle_included:
            data_rows = payload.get("data")
            if isinstance(data_rows, list) and data_rows:
                payload["forming_candle_index"] = len(data_rows) - 1
        if query_mode == "range":
            query_applied: Dict[str, Any] = {
                "mode": query_mode,
                "timeframe": timeframe,
                "limit": candles_requested,
            }
            if start_datetime not in (None, ""):
                query_applied["start"] = start_datetime
            if end_datetime not in (None, ""):
                query_applied["end"] = end_datetime
            payload["query_applied"] = query_applied
        if price_currency:
            payload["price_currency"] = price_currency
        payload["price_basis"] = price_basis
        if incomplete_candles_skipped and not include_incomplete:
            if ti_spec and latest_indicator_missing:
                payload["hint"] = (
                    "Latest forming candle was skipped. Set include_incomplete=true only if you need "
                    "that bar; increase limit if requested indicators need more warmup context."
                )
            else:
                payload["hint"] = "Set include_incomplete=true to include the latest forming candle."
        if isinstance(freshness_diagnostics, dict):
            payload["meta"]["diagnostics"]["freshness"] = dict(freshness_diagnostics)
        if include_spread:
            payload["spread_unit"] = "broker_points"
            payload["spread_note"] = (
                "Native MT5 candle spread is reported in broker points. If a fallback "
                "estimate is applied, spread_unit is changed to price."
            )
        if session_gap_warning:
            payload["meta"]["diagnostics"]["session_gaps"]["warning"] = session_gap_warning
        payload["timezone"] = _timezone_label(use_client_tz=_use_ctz, client_tz=client_tz)
        if simplify_meta is not None:
            payload["simplified"] = True
            payload["simplify"] = _public_simplify_meta(simplify_meta) or {"applied": True}
            simplify_method = str(simplify_meta.get("method") or "").strip().lower()
            simplify_reduced_rows = int(original_rows) > int(len(df))
            if simplify_reduced_rows and simplify_method in {"lttb", "rdp", "pla", "apca"}:
                payload["series_type"] = "downsampled_visualization"
                payload["equal_interval"] = False
                payload["analysis_compatible"] = False
                payload.setdefault("warnings", []).append(
                    "Simplified candle rows are visualization samples with irregular time gaps; "
                    "do not use them as equal-interval OHLC input for indicators or forecasts."
                )
        # Attach denoise applications metadata if any
        if denoise_apps:
            payload['denoise'] = {'applications': denoise_apps}
            payload['denoise_status'] = 'applied'
        elif denoise:
            payload['denoise_status'] = 'skipped'
            payload['denoise_applied'] = False
            if denoise_warnings:
                payload['denoise_status_reason'] = denoise_warnings[0]
        if denoise_apps or ti_spec:
            denoise_stages = {
                str(app.get("when") or "").lower()
                for app in denoise_apps
                if isinstance(app, dict)
            }
            pipeline = ["fetch_ohlcv"]
            if "pre_ti" in denoise_stages:
                pipeline.append("denoise_pre_ti")
            if ti_spec:
                pipeline.append("indicators")
                payload["indicator_input"] = (
                    "pre_ti_denoised_ohlcv"
                    if "pre_ti" in denoise_stages
                    else "raw_ohlcv"
                )
            if "post_ti" in denoise_stages:
                pipeline.append("denoise_post_ti")
            payload["processing_pipeline"] = pipeline
        if denoise_warnings:
            warns = payload.get('warnings')
            if not isinstance(warns, list):
                warns = []
            for warning_text in denoise_warnings:
                if warning_text not in warns:
                    warns.append(warning_text)
            payload['warnings'] = warns
        if ohlcv_warnings:
            warns = payload.get('warnings')
            if not isinstance(warns, list):
                warns = []
            for warning_text in ohlcv_warnings:
                if warning_text not in warns:
                    warns.append(warning_text)
            payload['warnings'] = warns
        if ti_warnings:
            warns = payload.get('warnings')
            if not isinstance(warns, list):
                warns = []
            for warning_text in ti_warnings:
                if warning_text not in warns:
                    warns.append(warning_text)
            payload['warnings'] = warns
        if session_gaps:
            payload['session_gaps'] = session_gaps
            _annotate_candle_gap_rows(payload, session_gaps)
            warns = payload.get('warnings')
            if not isinstance(warns, list):
                warns = []
            warns.append(
                "Detected session gaps larger than expected bar spacing ({secs:.0f}s).".format(
                    secs=expected_bar_seconds,
                )
            )
            try:
                first_gap = session_gaps[0]
                warns.append(
                    "Example gap: {from_} -> {to} ({missing} missing bars, likely {context}).".format(
                        from_=str(first_gap.get("from")),
                        to=str(first_gap.get("to")),
                        missing=int(first_gap.get("missing_bars_est") or 0),
                        context=str(first_gap.get("context") or "session break"),
                    )
                )
            except Exception:
                pass
            payload['warnings'] = warns
        elif session_gap_warning:
            warns = payload.get('warnings')
            if not isinstance(warns, list):
                warns = []
            warns.append(session_gap_warning)
            payload['warnings'] = warns

        # If include_spread requested but spread data is missing or all zero, try fallback estimate from recent ticks.
        if include_spread:
            data_rows = payload.get("data", []) or []
            has_spread_values = False
            spread_all_zero = True
            spread_value_count = 0
            spread_zero_count = 0
            spread_idx = None
            try:
                if "spread" in headers:
                    spread_idx = headers.index("spread")
            except Exception:
                spread_idx = None
            for row in data_rows:
                if isinstance(row, dict):
                    if "spread" in row and row.get("spread") is not None:
                        has_spread_values = True
                        spread_value_count += 1
                        try:
                            if float(row.get("spread", 0)) == 0.0:
                                spread_zero_count += 1
                            else:
                                spread_all_zero = False
                        except Exception:
                            has_spread_values = True
                            spread_all_zero = False
                elif isinstance(row, (list, tuple)):
                    if spread_idx is not None and spread_idx < len(row):
                        val = row[spread_idx]
                        if val is not None:
                            has_spread_values = True
                            spread_value_count += 1
                            try:
                                if float(val) == 0.0:
                                    spread_zero_count += 1
                                else:
                                    spread_all_zero = False
                            except Exception:
                                has_spread_values = True
                                spread_all_zero = False
            spread_mostly_zero = (
                has_spread_values
                and spread_value_count >= 3
                and spread_zero_count / float(spread_value_count) >= 0.75
            )
            if not has_spread_values or spread_all_zero or spread_mostly_zero:
                data_rows = _remove_unavailable_spread_from_candle_rows(
                    data_rows,
                    spread_idx=spread_idx,
                )
                payload["data"] = data_rows
                payload["spread_historical_available"] = False
                payload.pop("spread_unit", None)
                payload.pop("spread_note", None)
                if spread_mostly_zero and not spread_all_zero:
                    payload.setdefault("warnings", []).append(
                        "include_spread native candle spread is zero for most bars; "
                        "using an estimated spread because MT5 candle spread appears sparse."
                    )
                try:
                    tick_stats = fetch_ticks(symbol, limit=5000, format="stats")
                    spread_stats = tick_stats.get("stats", {}).get("spread")
                    est_mean = None
                    if isinstance(spread_stats, dict):
                        est_mean = spread_stats.get("mean") or spread_stats.get("median") or spread_stats.get("first")
                    estimate_source = "tick_stats"
                    live_spread = _live_tick_spread(symbol)
                    if est_mean is not None:
                        est_mean = float(est_mean)
                        if live_spread is not None and live_spread > 0.0:
                            diff_ratio = abs(float(live_spread) - est_mean) / max(float(live_spread), abs(est_mean), 1e-12)
                            payload.setdefault("meta", {}).setdefault("diagnostics", {}).setdefault("spread_estimate", {})["live_spread"] = live_spread
                            payload["meta"]["diagnostics"]["spread_estimate"]["live_diff_ratio"] = diff_ratio
                            if diff_ratio > 0.5:
                                payload.setdefault("warnings", []).append(
                                    "include_spread tick-stat estimate differed from live spread "
                                    f"by {diff_ratio:.0%}; live ticker spread ({live_spread:g}) applied."
                                )
                                est_mean = float(live_spread)
                                estimate_source = "live_ticker_crosscheck"
                                payload["spread_accuracy"] = "tick_stats_replaced_by_live"
                        payload.setdefault("warnings", []).append(
                            "include_spread requested but per-bar spread unavailable; a single "
                            f"reference spread from {estimate_source} ({est_mean:g}) is returned "
                            "at payload level and is not per-bar historical spread."
                        )
                        payload["spread_reference"] = {
                            "value": est_mean,
                            "unit": "price",
                            "source": estimate_source,
                            "basis": "single_reference_not_per_bar_historical",
                        }
                        payload.setdefault("meta", {}).setdefault("diagnostics", {}).setdefault("spread_estimate", {})["estimated_mean"] = est_mean
                        payload["meta"]["diagnostics"]["spread_estimate"]["source"] = estimate_source
                        payload["meta"]["diagnostics"]["spread_estimate"]["unit"] = "price"
                        payload["meta"]["diagnostics"]["spread_estimate"]["tick_stats"] = spread_stats
                    else:
                        # Fallback to live ticker
                        if live_spread is not None:
                            est_mean = float(live_spread)
                            payload.setdefault("warnings", []).append(
                                "include_spread requested but spread unavailable; "
                                f"current live ticker spread ({est_mean:g}) is returned at payload "
                                "level and is not per-bar historical spread."
                            )
                            payload["spread_reference"] = {
                                "value": est_mean,
                                "unit": "price",
                                "source": "live_ticker",
                                "basis": "single_reference_not_per_bar_historical",
                            }
                            payload.setdefault("meta", {}).setdefault("diagnostics", {}).setdefault("spread_estimate", {})["estimated_mean"] = est_mean
                            payload["meta"]["diagnostics"]["spread_estimate"]["source"] = "live_ticker"
                            payload["meta"]["diagnostics"]["spread_estimate"]["unit"] = "price"
                except Exception:
                    payload.setdefault("warnings", []).append("include_spread requested but spread unavailable; no fallback available.")

        return payload
    except Exception as e:
        return {
            "error": f"Error getting rates: {type(e).__name__}: {e}",
            "error_detail": {
                "operation": "fetch_candles",
                "symbol": symbol,
                "timeframe": timeframe,
                "start": str(start) if start else None,
                "end": str(end) if end else None,
            },
        }


def _live_tick_spread(symbol: str) -> Optional[float]:
    try:
        tick = mt5.symbol_info_tick(symbol)
    except Exception:
        tick = None
    bid = getattr(tick, "bid", None) if tick is not None else None
    ask = getattr(tick, "ask", None) if tick is not None else None
    try:
        bid_f = float(bid)
        ask_f = float(ask)
    except Exception:
        return None
    if not math.isfinite(bid_f) or not math.isfinite(ask_f):
        return None
    spread = max(0.0, ask_f - bid_f)
    return spread if spread > 0.0 else None


def _remove_unavailable_spread_from_candle_rows(
    data_rows: list[Any],
    *,
    spread_idx: int | None,
) -> list[Any]:
    for i, row in enumerate(data_rows):
        if isinstance(row, dict):
            row.pop("spread", None)
            row.pop("spread_source", None)
        else:
            row_list = list(row)
            if spread_idx is not None and spread_idx < len(row_list):
                row_list[spread_idx] = None
            data_rows[i] = row_list
    return data_rows


def _mt5_tick_flag_value(name: str, default: int) -> int:
    try:
        return int(getattr(mt5, name))
    except (TypeError, ValueError, AttributeError):
        return int(default)


def _tick_flag_definitions() -> tuple[tuple[int, str, str], ...]:
    return (
        (
            _mt5_tick_flag_value("TICK_FLAG_BID", 2),
            "bid",
            "Bid price changed in this snapshot.",
        ),
        (
            _mt5_tick_flag_value("TICK_FLAG_ASK", 4),
            "ask",
            "Ask price changed in this snapshot.",
        ),
        (
            _mt5_tick_flag_value("TICK_FLAG_LAST", 8),
            "last",
            "Last traded price changed in this snapshot.",
        ),
        (
            _mt5_tick_flag_value("TICK_FLAG_VOLUME", 16),
            "volume",
            "Last-trade volume changed in this snapshot.",
        ),
        (
            _mt5_tick_flag_value("TICK_FLAG_BUY", 32),
            "buy",
            "Last trade was buyer-initiated.",
        ),
        (
            _mt5_tick_flag_value("TICK_FLAG_SELL", 64),
            "sell",
            "Last trade was seller-initiated.",
        ),
        (
            _mt5_tick_flag_value("TICK_FLAG_VOLUME_REAL", 1024),
            "volume_real",
            "Last-trade real volume changed in this snapshot.",
        ),
    )


def _decode_tick_flags(flag_value: int) -> List[str]:
    try:
        remaining = int(flag_value)
    except (TypeError, ValueError):
        return []
    labels: List[str] = []
    for bit, label, _description in _tick_flag_definitions():
        if bit > 0 and remaining & bit:
            labels.append(label)
            remaining &= ~bit
    while remaining > 0:
        bit = remaining & -remaining
        labels.append(f"unknown_{bit}")
        remaining &= ~bit
    return labels


def _observed_tick_flags_decoded(flags: List[int]) -> Dict[str, List[str]]:
    return {
        str(flag): _decode_tick_flags(flag)
        for flag in sorted(set(int(value) for value in flags if int(value) != 0))
    }


def _finite_or_none(value: Any) -> Optional[float]:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _json_safe_payload(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe_payload(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe_payload(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe_payload(item) for item in value]
    if isinstance(value, Real) and not isinstance(value, bool):
        number = float(value)
        if not math.isfinite(number):
            return None
    try:
        if pd.isna(value) and not isinstance(value, (str, bytes)):
            return None
    except Exception:
        pass
    return value


def _compact_tick_summary(out: Dict[str, Any]) -> Dict[str, Any]:
    spread = out.get("stats", {}).get("spread")
    compact_spread: Dict[str, Any] = {}
    if isinstance(spread, dict):
        available = spread.get("available")
        try:
            spread_unavailable = available is not None and not bool(available)
        except Exception:
            spread_unavailable = False
        if spread_unavailable:
            compact_spread["available"] = False
        else:
            for source_key, target_key in (
                ("low", "low"),
                ("high", "high"),
                ("mean", "mean"),
            ):
                value = spread.get(source_key)
                if value is not None:
                    compact_spread[target_key] = value
    compact: Dict[str, Any] = {
        "success": bool(out.get("success")),
        "symbol": out.get("symbol"),
        "count": out.get("count"),
        "start": out.get("start"),
        "end": out.get("end"),
        "duration_seconds": out.get("duration_seconds"),
        "tick_rate_per_second": out.get("tick_rate_per_second"),
        "tick_count": out.get("tick_count", out.get("count")),
        "trade_event_count": out.get("trade_event_count"),
        "quote_update_count": out.get("quote_update_count"),
        "timezone": out.get("timezone"),
        "stats": {"spread": compact_spread},
    }
    if out.get("price_precision") is not None:
        compact["price_precision"] = out.get("price_precision")
    if out.get("price_point") is not None:
        compact["price_point"] = out.get("price_point")
    if out.get("price_currency") is not None:
        compact["price_currency"] = out.get("price_currency")
    for key in (
        "time_basis",
        "raw_time_basis",
        "time_normalization",
        "broker_server_tz",
        "session_utc_offset_seconds",
    ):
        if out.get(key) is not None:
            compact[key] = out.get(key)
    for key in (
        "freshness",
        "data_age_seconds",
        "data_stale",
        "market_status",
        "market_status_reason",
        "market_status_source",
        "freshness_policy_relaxed",
        "note",
    ):
        if out.get(key) is not None:
            compact[key] = out.get(key)
    if isinstance(out.get("last_quote"), dict):
        compact["last_quote"] = dict(out["last_quote"])
    if isinstance(out.get("data_quality"), dict):
        compact["data_quality"] = dict(out["data_quality"])
    return compact


def fetch_ticks(  # noqa: C901
    symbol: str,
    limit: int = DEFAULT_ROW_LIMIT,
    start: Optional[str] = None,
    end: Optional[str] = None,
    simplify: Optional[SimplifySpec] = None,
    time_as_epoch: bool = False,
    format: Literal["summary", "stats", "rows", "full_rows"] = "summary",
) -> Dict[str, Any]:
    """Fetch tick data and return either a summary (default) or raw rows.

    Parameters
    ----------
    format : {"summary","stats","rows","full_rows"}
        - "summary" (default): compact descriptive statistics over the fetched
          ticks. Samples below 20 ticks report spread stats only with a sample
          adequacy note; larger samples include bid/ask/mid, plus last and
          volume when available.
        - "stats": more detailed stats (includes extra distribution moments and
          quantiles).
        - "rows": return tick rows as structured data.
        - "full_rows": return rows with per-tick epoch, mid, spread, and gap fields.
    """
    try:
        symbol = resolve_broker_symbol_name(symbol)
        effective_limit = int(limit)
        if effective_limit <= 0:
            return {"error": "limit must be greater than 0."}
        # Ensure symbol is ready; remember original visibility to restore later
        _info_before = get_symbol_info_cached(symbol)
        with _symbol_ready_guard(symbol, info_before=_info_before) as (err, _info):
            if err:
                return {"error": err}
            price_digits = _symbol_price_digits(_info, _info_before)
            price_currency = _symbol_price_currency(_info, _info_before)
            price_point = _symbol_price_point(_info, _info_before)
            points_per_pip = (
                forex_points_per_pip(
                    symbol,
                    path=_symbol_path(_info, _info_before),
                    point=price_point,
                    digits=price_digits,
                )
                if price_point is not None
                else None
            )
            time_normalization = describe_mt5_time_normalization(symbol=symbol)

            # Normalized params only. This is an output shape selector, not the
            # shared compact/full detail enum.
            output_mode = str(format or "summary").strip().lower()
            output_mode = {
                "raw": "rows",
                "ticks": "rows",
            }.get(output_mode, output_mode)
            if start:
                from_date = _parse_start_datetime(start)
                if not from_date:
                    return {"error": f"Could not parse start date {start!r}. {_DATE_FORMAT_HINT}"}
                if end:
                    to_date = _parse_start_datetime(end)
                    if not to_date:
                        return {"error": f"Could not parse end date {end!r}. {_DATE_FORMAT_HINT}"}
                    if from_date > to_date:
                        return {"error": "start must be before or equal to end."}
                    ticks = _fetch_ticks_range_with_retry(symbol, from_date, to_date)
                    if ticks is not None and effective_limit and len(ticks) > effective_limit:
                        ticks = ticks[-effective_limit:]
                else:
                    ticks = _fetch_recent_ticks_backwards(
                        symbol,
                        to_date=datetime.now(dt_timezone.utc),
                        limit=effective_limit,
                        min_from_date=from_date,
                    )
            else:
                # Get recent ticks from current time (now)
                to_date = datetime.now(dt_timezone.utc)
                ticks = _fetch_recent_ticks_backwards(
                    symbol,
                    to_date=to_date,
                    limit=effective_limit,
                )
        # visibility handled by _symbol_ready_guard
        
        if ticks is None:
            return {"error": f"Failed to get ticks for {symbol}: {mt5.last_error()}"}
        
        # Generate tabular format with dynamic column filtering
        if len(ticks) == 0:
            return {"error": "No tick data available"}

        if output_mode not in ("summary", "stats", "rows", "full_rows"):
            return {
                "error": (
                    f"Invalid format: {format}. "
                    "Use 'summary', 'stats', 'rows', or 'full_rows'."
                )
            }

        def _tick_epoch_seconds(tick: Any) -> float:
            time_msc = _tick_field_value(tick, "time_msc")
            try:
                time_msc_value = float(time_msc)
            except (TypeError, ValueError):
                time_msc_value = float("nan")
            if math.isfinite(time_msc_value) and time_msc_value > 0.0:
                return time_msc_value / 1000.0
            return float(_tick_field_value(tick, "time"))

        # Extract shared tick columns once so summary/stats, simplification,
        # and row rendering can all reuse the same values.
        _epochs: List[float] = []
        bids: List[float] = []
        asks: List[float] = []
        effective_bids: List[Optional[float]] = []
        effective_asks: List[Optional[float]] = []
        lasts: List[float] = []
        flags: List[int] = []
        volumes: List[float] = []
        volumes_real: List[float] = []
        trade_events: List[bool] = []
        quote_types: List[str] = []
        for tick in ticks:
            _epochs.append(_tick_epoch_seconds(tick))
            bid_value = _finite_or_none(_tick_field_value(tick, "bid"))
            ask_value = _finite_or_none(_tick_field_value(tick, "ask"))
            bid = float("nan") if bid_value is None else bid_value
            ask = float("nan") if ask_value is None else ask_value
            flag_value = int(_tick_field_value(tick, "flags") or 0)
            bids.append(bid)
            asks.append(ask)
            last_value = _finite_or_none(_tick_field_value(tick, "last"))
            lasts.append(
                float("nan")
                if last_value is None or last_value <= 0.0
                else last_value
            )
            flags.append(flag_value)
            effective_bids.append(None if bid_value is None else bid)
            effective_asks.append(None if ask_value is None else ask)
            if bid_value is None and ask_value is not None:
                quote_types.append("ask_only")
            elif ask_value is None and bid_value is not None:
                quote_types.append("bid_only")
            elif bid_value is None and ask_value is None:
                quote_types.append("no_quote")
            else:
                quote_types.append("bid_ask")
            try:
                volume_value = float(_tick_field_value(tick, "volume"))
            except (TypeError, ValueError):
                volume_value = float("nan")
            try:
                volume_real_value = float(_tick_field_value(tick, "volume_real"))
            except (TypeError, ValueError):
                volume_real_value = float("nan")
            volumes.append(volume_value)
            volumes_real.append(volume_real_value)
            trade_events.append(
                is_mt5_trade_event(flag_value, mt5)
                and (
                    (last_value is not None and last_value > 0.0)
                    or (math.isfinite(volume_value) and volume_value > 0.0)
                    or (
                        math.isfinite(volume_real_value)
                        and volume_real_value > 0.0
                    )
                )
            )

        has_last = any(math.isfinite(value) for value in lasts)
        finite_volumes = [v for v in volumes if math.isfinite(v)]
        has_volume = bool(finite_volumes) and (
            len(set(finite_volumes)) > 1 or any(v != 0.0 for v in finite_volumes)
        )
        has_flags = len(set(flags)) > 1 or any(v != 0 for v in flags)
        has_real_volume = any(math.isfinite(v) and v != 0.0 for v in volumes_real)
        incomplete_quote_count = sum(
            1
            for bid, ask in zip(effective_bids, effective_asks, strict=False)
            if bid is None or ask is None
        )
        bid_update_flag = _mt5_tick_flag_value("TICK_FLAG_BID", 2)
        ask_update_flag = _mt5_tick_flag_value("TICK_FLAG_ASK", 4)
        quote_update_mask = bid_update_flag | ask_update_flag
        has_quote_update_flags = any(flag & quote_update_mask for flag in flags)
        quote_update_types: List[str] = []
        spread_valid_flags: List[bool] = []
        for flag, quote_type in zip(flags, quote_types, strict=False):
            bid_updated = bool(flag & bid_update_flag)
            ask_updated = bool(flag & ask_update_flag)
            if bid_updated and ask_updated:
                update_type = "bid_ask_update"
            elif bid_updated:
                update_type = "bid_only_update"
            elif ask_updated:
                update_type = "ask_only_update"
            else:
                update_type = "update_flags_unavailable"
            quote_update_types.append(update_type)
            spread_valid_flags.append(
                quote_type == "bid_ask"
                and (
                    (bid_updated and ask_updated)
                    if has_quote_update_flags
                    else True
                )
            )
        one_sided_update_count = sum(
            update_type in {"bid_only_update", "ask_only_update"}
            for update_type in quote_update_types
        )
        zero_spread_count = sum(
            bool(valid)
            and bid is not None
            and ask is not None
            and float(ask) == float(bid)
            for valid, bid, ask in zip(
                spread_valid_flags,
                effective_bids,
                effective_asks,
                strict=False,
            )
        )

        full_rows = output_mode == "full_rows"

        # Keep row schemas stable; compact public output prunes unused fields.
        headers = ["time"]
        if full_rows:
            headers.append("time_epoch")
        headers.extend(["bid", "ask"])
        include_quote_type = any(value != "bid_ask" for value in quote_types)
        if include_quote_type:
            headers.append("quote_type")
        if full_rows:
            headers.extend(["quote_update_type", "spread_valid"])
            headers.extend(["mid", "spread"])
            if price_point is not None:
                headers.append("spread_points")
            if points_per_pip is not None:
                headers.append("spread_pips")
            headers.extend(["spread_pct", "tick_gap_ms"])
        headers.extend(["last", "volume", "volume_real", "flags", "flags_decoded"])

        # Choose a consistent millisecond time format for tick rows.
        # Low-level tick fetch helpers have already normalized epochs to UTC.
        client_tz = _resolve_client_tz()
        _use_ctz = client_tz is not None

        def _format_tick_time(epoch: float) -> str:
            if time_as_epoch:
                return int(epoch) if float(epoch).is_integer() else float(epoch)
            try:
                dt = datetime.fromtimestamp(float(epoch), tz=dt_timezone.utc)
            except Exception:
                return str(epoch)
            if _use_ctz:
                try:
                    dt = dt.astimezone(client_tz)
                except Exception:
                    dt = dt.astimezone()
            millis = int(round(float(dt.microsecond) / 1000.0))
            if millis >= 1000:
                dt = dt + timedelta(seconds=1)
                millis = 0
            offset = dt.strftime("%z")
            if offset == "+0000":
                offset = "Z"
            elif len(offset) == 5 and offset[0] in {"+", "-"}:
                offset = f"{offset[:3]}:{offset[3:]}"
            return f"{dt.strftime('%Y-%m-%dT%H:%M:%S')}.{millis:03d}{offset}"

        original_count = len(ticks)
        simplify_eff = _normalize_simplify_spec(simplify, limit=limit, fallback_rows=original_count)
        simplify_present = (simplify_eff is not None) or (simplify is not None)
        simplify_used = simplify_eff if simplify_eff is not None else simplify
        simplify_mode = (
            str((simplify_used or {}).get("mode", SIMPLIFY_DEFAULT_MODE)).lower().strip()
            if simplify_present
            else SIMPLIFY_DEFAULT_MODE
        )

        df_ticks = pd.DataFrame({
            "__epoch": _epochs,
            "bid": effective_bids,
            "ask": effective_asks,
        })
        if full_rows:
            tick_gap_ms: List[Optional[float]] = [None]
            for idx in range(1, len(_epochs)):
                tick_gap_ms.append(float((_epochs[idx] - _epochs[idx - 1]) * 1000.0))
            df_ticks["time_epoch"] = [
                int(epoch) if float(epoch).is_integer() else float(epoch)
                for epoch in _epochs
            ]
            df_ticks["mid"] = (df_ticks["bid"] + df_ticks["ask"]) / 2.0
            df_ticks["spread"] = df_ticks["ask"] - df_ticks["bid"]
            if price_point is not None:
                df_ticks["spread_points"] = df_ticks["spread"] / price_point
            if price_point is not None and points_per_pip is not None:
                df_ticks["spread_pips"] = (
                    df_ticks["spread"] / price_point / points_per_pip
                )
            df_ticks["spread_pct"] = (df_ticks["spread"] / df_ticks["mid"]) * 100.0
            df_ticks["tick_gap_ms"] = tick_gap_ms
        df_ticks["last"] = lasts
        df_ticks["volume"] = volumes
        df_ticks["volume_real"] = volumes_real
        df_ticks["flags"] = flags
        df_ticks["trade_event"] = trade_events
        df_ticks["spread_valid"] = spread_valid_flags
        df_ticks["quote_update_type"] = quote_update_types
        if include_quote_type:
            df_ticks["quote_type"] = quote_types
        df_ticks["flags_decoded"] = [
            _decode_tick_flags(flag_value) for flag_value in flags
        ]
        df_ticks["time"] = [_format_tick_time(e) for e in _epochs]

        def _add_tick_data_quality(payload: Dict[str, Any]) -> None:
            if (
                incomplete_quote_count <= 0
                and one_sided_update_count <= 0
                and zero_spread_count <= 0
            ):
                return
            incomplete_ratio = incomplete_quote_count / max(1, original_count)
            quote_type_counts = {
                kind: quote_types.count(kind)
                for kind in sorted(set(quote_types))
            }
            complete_ticks = int(quote_type_counts.get("bid_ask", 0))
            incomplete_ticks = int(original_count - complete_ticks)
            payload["data_quality"] = {
                "incomplete_quote_ticks": int(incomplete_quote_count),
                "complete_ticks": complete_ticks,
                "incomplete_ticks": incomplete_ticks,
                "total_ticks": int(original_count),
                "incomplete_quote_ratio": round(incomplete_ratio, 4),
                "spread_ticks_excluded": int(incomplete_quote_count),
                "one_sided_updates": int(one_sided_update_count),
                "valid_spread_ticks": int(sum(spread_valid_flags)),
                "zero_spread_ticks": int(zero_spread_count),
                "warning_ratio": _ONE_SIDED_TICK_WARNING_RATIO,
                "quote_type_counts": quote_type_counts,
            }
            if (
                incomplete_ratio < _ONE_SIDED_TICK_WARNING_RATIO
                and one_sided_update_count <= 0
                and zero_spread_count <= 0
            ):
                payload["data_quality"]["incomplete_quote_status"] = "info"
                return
            payload["data_quality"]["incomplete_quote_status"] = "warning"
            warnings_list = payload.get("warnings")
            if not isinstance(warnings_list, list):
                warnings_list = []
            warning = (
                "Spread statistics exclude incomplete and one-sided quote updates; "
                "zero-spread counts include only coherent two-sided updates."
            )
            if warning not in warnings_list:
                warnings_list.append(warning)
            payload["warnings"] = warnings_list

        def _add_tick_last_quality(payload: Dict[str, Any]) -> None:
            if has_last:
                return
            payload["last_unavailable"] = True
            warnings_list = payload.get("warnings")
            if not isinstance(warnings_list, list):
                warnings_list = []
            warning = "Broker tick data did not provide a usable last price; last is null."
            if warning not in warnings_list:
                warnings_list.append(warning)
            payload["warnings"] = warnings_list

        def _add_tick_context_fields(payload: Dict[str, Any]) -> None:
            last_quote = payload.get("last_quote")
            if isinstance(last_quote, dict) and price_point is not None:
                spread_value = _finite_or_none(last_quote.get("spread"))
                if spread_value is not None:
                    last_quote["spread_points"] = round(spread_value / price_point, 4)
                    if points_per_pip is not None:
                        last_quote["spread_pips"] = round(
                            spread_value / price_point / points_per_pip,
                            4,
                        )
            if isinstance(last_quote, dict):
                spread_pct = _tick_spread_pct(
                    last_quote.get("spread"),
                    last_quote.get("mid"),
                )
                if spread_pct is not None:
                    last_quote["spread_pct"] = spread_pct
            payload.update(time_normalization)
            if start or end or not _epochs:
                return
            latest_tick_epoch = float(_epochs[-1])
            freshness_context = build_tick_freshness_context(
                symbol,
                tick_epoch=latest_tick_epoch,
                now_epoch=time.time(),
                item="tick",
                age_rounder=lambda value: round(value, 3),
            )
            payload.update(freshness_context)

        def _compact_summary_from_ticks() -> Dict[str, Any]:
            df_stats = df_ticks.copy()
            df_stats["mid"] = (df_stats["bid"] + df_stats["ask"]) / 2.0
            df_stats["spread"] = df_stats["ask"] - df_stats["bid"]
            start_epoch = float(df_stats["__epoch"].iloc[0])
            end_epoch = float(df_stats["__epoch"].iloc[-1])
            duration_seconds = float(max(0.0, end_epoch - start_epoch))
            tick_rate_per_second = (
                float(len(df_stats) / duration_seconds) if duration_seconds > 0 else None
            )
            spread = pd.to_numeric(
                df_stats["spread"].where(df_stats["spread_valid"]),
                errors="coerce",
            ).dropna()
            out: Dict[str, Any] = {
                "success": True,
                "symbol": symbol,
                "count": int(len(df_stats)),
                "start": str(df_stats["time"].iloc[0]),
                "end": str(df_stats["time"].iloc[-1]),
                "duration_seconds": duration_seconds,
                "tick_rate_per_second": tick_rate_per_second,
                "tick_count": int(len(df_stats)),
                "trade_event_count": int(sum(trade_events)),
                "quote_update_count": int(
                    sum(flag & quote_update_mask != 0 for flag in flags)
                ),
                "timezone": _timezone_label(use_client_tz=_use_ctz, client_tz=client_tz),
                "stats": {
                    "spread": {
                        "low": float(spread.min()),
                        "high": float(spread.max()),
                        "mean": float(spread.mean()),
                    }
                },
                "last_quote": {
                    "bid": float(df_stats["bid"].iloc[-1]),
                    "ask": float(df_stats["ask"].iloc[-1]),
                    "mid": float(df_stats["mid"].iloc[-1]),
                    "spread": float(df_stats["spread"].iloc[-1]),
                },
            }
            if price_digits > 0:
                out["price_precision"] = int(price_digits)
            if price_point is not None:
                out["price_point"] = price_point
            if price_currency:
                out["price_currency"] = price_currency
            _add_tick_data_quality(out)
            _add_tick_last_quality(out)
            _add_tick_context_fields(out)
            _round_tick_price_payload(out, price_digits)
            return _json_safe_payload(_compact_tick_summary(out))

        def _add_tick_summary_fields(payload: Dict[str, Any]) -> None:
            summary = _compact_summary_from_ticks()
            for key, value in summary.items():
                if key not in ("success", "symbol", "count", "timezone"):
                    payload[key] = value

        if output_mode in ("summary", "stats"):
            detailed_stats = output_mode == "stats"

            def _series_stats(s: pd.Series, *, total_count: int) -> Dict[str, Any]:
                vals = pd.to_numeric(s, errors="coerce")
                vals = vals[pd.notna(vals)].astype(float)
                n = int(vals.shape[0])
                if n <= 0:
                    out = {
                        "available": False,
                        "first": float("nan"),
                        "last": float("nan"),
                        "low": float("nan"),
                        "high": float("nan"),
                        "mean": float("nan"),
                        "std": float("nan"),
                        "stderr": float("nan"),
                        "kurtosis": float("nan"),
                        "change": float("nan"),
                        "change_pct": float("nan"),
                    }
                    if detailed_stats:
                        out["median"] = float("nan")
                        out["skew"] = float("nan")
                        out["q25"] = float("nan")
                        out["q75"] = float("nan")
                    if detailed_stats or n != int(total_count):
                        out["count"] = n
                    return _json_safe_payload(out)
                first = float(vals.iloc[0])
                last = float(vals.iloc[-1])
                low = float(vals.min())
                high = float(vals.max())
                mean = float(vals.mean())
                std = float(vals.std(ddof=0)) if n > 0 else float("nan")
                stderr = float(std / math.sqrt(n)) if n > 0 else float("nan")
                kurtosis = float(vals.kurtosis()) if n >= 4 else None
                change = float(last - first)
                change_pct = float((change / first) * 100.0) if first != 0.0 else float("nan")
                out = {
                    "first": first,
                    "last": last,
                    "low": low,
                    "high": high,
                    "mean": mean,
                    "std": std,
                    "stderr": stderr,
                    "kurtosis": kurtosis,
                    "change": change,
                    "change_pct": change_pct,
                }
                if detailed_stats:
                    out["median"] = float(vals.median())
                    out["skew"] = float(vals.skew()) if n >= 3 else float("nan")
                    out["q25"] = float(vals.quantile(0.25))
                    out["q75"] = float(vals.quantile(0.75))
                if detailed_stats or n != int(total_count):
                    out["count"] = n
                return _json_safe_payload(out)

            df_stats = df_ticks.copy()
            df_stats["mid"] = (df_stats["bid"] + df_stats["ask"]) / 2.0
            df_stats["spread"] = (df_stats["ask"] - df_stats["bid"]).where(
                df_stats["spread_valid"]
            )

            start_epoch = float(df_stats["__epoch"].iloc[0])
            end_epoch = float(df_stats["__epoch"].iloc[-1])
            duration_seconds = float(max(0.0, end_epoch - start_epoch))
            tick_rate_per_second = (
                float(len(df_stats) / duration_seconds) if duration_seconds > 0 else None
            )

            timezone = _timezone_label(use_client_tz=_use_ctz, client_tz=client_tz)

            out: Dict[str, Any] = {
                "success": True,
                "symbol": symbol,
                "output": "stats" if detailed_stats else "summary",
                "count": int(len(df_stats)),
                "start": str(df_stats["time"].iloc[0]),
                "end": str(df_stats["time"].iloc[-1]),
                "duration_seconds": duration_seconds,
                "tick_rate_per_second": tick_rate_per_second,
                "timezone": timezone,
                "stats": {
                    "bid": _series_stats(df_stats["bid"], total_count=len(df_stats)),
                    "ask": _series_stats(df_stats["ask"], total_count=len(df_stats)),
                    "mid": _series_stats(df_stats["mid"], total_count=len(df_stats)),
                    "spread": _series_stats(df_stats["spread"], total_count=len(df_stats)),
                },
            }
            out["last_quote"] = {
                "bid": float(df_stats["bid"].iloc[-1]),
                "ask": float(df_stats["ask"].iloc[-1]),
                "mid": float(df_stats["mid"].iloc[-1]),
                "spread": float(df_stats["spread"].iloc[-1]),
            }
            if duration_seconds <= 0:
                out["tick_rate_note"] = "< 1s window"
            small_summary_sample = (
                not detailed_stats
                and int(len(df_stats)) < _TICK_SUMMARY_MIN_ANALYTIC_TICKS
            )
            if not detailed_stats:
                out["sample_adequacy"] = not small_summary_sample
                if small_summary_sample:
                    out["sample_adequacy_note"] = (
                        f"Small sample ({len(df_stats)} ticks"
                        f" in {duration_seconds:g}s) - spread stats only."
                    )
                    out["sample_min_ticks"] = _TICK_SUMMARY_MIN_ANALYTIC_TICKS
                    out["stats"] = {"spread": out["stats"]["spread"]}
            spread_stats = out.get("stats", {}).get("spread")
            if isinstance(spread_stats, dict):
                try:
                    spread_first = float(spread_stats.get("first"))
                    spread_change_pct = spread_stats.get("change_pct")
                    spread_change_pct_f = float(spread_change_pct) if spread_change_pct is not None else float("nan")
                    if spread_first == 0.0 and not math.isfinite(spread_change_pct_f):
                        spread_stats["change_pct"] = None
                        out["spread_change_pct_note"] = "first spread was zero"
                except Exception:
                    pass
            if price_digits > 0:
                out["price_precision"] = int(price_digits)
            if price_point is not None:
                out["price_point"] = price_point
            if price_currency:
                out["price_currency"] = price_currency
            units = _tick_units_for_headers(headers)
            if units and detailed_stats:
                out["units"] = units
            if has_last and not small_summary_sample:
                out["stats"]["last"] = _series_stats(df_stats["last"], total_count=len(df_stats))

            trade_event_mask = df_stats["trade_event"].astype(bool)
            trade_event_count = int(trade_event_mask.sum())
            out["tick_count"] = int(len(df_stats))
            out["trade_event_count"] = trade_event_count
            out["quote_update_count"] = int(
                sum(flag & quote_update_mask != 0 for flag in flags)
            )
            if detailed_stats:
                out["stats"]["tick_count"] = {
                    "kind": "tick_count",
                    "sum": int(len(df_stats)),
                    "per_second": tick_rate_per_second,
                }

            volume_kind: Optional[str] = None
            vol_vals = pd.Series(index=df_stats.index, dtype=float)
            real_trade_volume = df_stats["volume_real"].where(trade_event_mask)
            snapshot_trade_volume = df_stats["volume"].where(trade_event_mask)
            if bool((real_trade_volume.fillna(0.0) > 0.0).any()):
                volume_kind = "volume_real"
                vol_vals = real_trade_volume
            elif bool((snapshot_trade_volume.fillna(0.0) > 0.0).any()):
                volume_kind = "volume"
                vol_vals = snapshot_trade_volume

            if volume_kind is not None:
                vol_vals_num = pd.to_numeric(vol_vals, errors="coerce").astype(float)
                vol_sum = float(vol_vals_num.fillna(0.0).sum())
                vol_nonzero_count = int((vol_vals_num.fillna(0.0) != 0.0).sum())
                vol_out: Dict[str, Any] = {
                    "kind": volume_kind,
                    "sum": vol_sum,
                    "per_second": (
                        float(vol_sum / duration_seconds) if duration_seconds > 0 else None
                    ),
                    "per_trade_event": float(vol_sum / float(trade_event_count or 1)),
                    "nonzero_share": float(vol_nonzero_count) / float(trade_event_count or 1),
                }
                try:
                    mean_v = float(vol_vals_num.mean())
                    std_v = float(vol_vals_num.std(ddof=0))
                    vol_out["cv"] = (
                        float(std_v / mean_v) if (mean_v != 0.0 and not math.isnan(mean_v)) else float("nan")
                    )
                except Exception:
                    pass

                if vol_sum > 0.0:
                    try:
                        top_n = min(10, int(len(vol_vals_num)))
                        if top_n > 0:
                            vol_top = vol_vals_num.fillna(0.0).sort_values(ascending=False).iloc[:top_n]
                            vol_out["top10_share"] = float(vol_top.sum() / vol_sum)
                    except Exception:
                        pass
                    try:
                        q95 = float(vol_vals_num.quantile(0.95))
                        spikes = vol_vals_num[vol_vals_num >= q95]
                        vol_out["spike95_count"] = int(spikes.shape[0])
                        vol_out["spike95_share"] = float(spikes.fillna(0.0).sum() / vol_sum)
                    except Exception:
                        pass
                    try:
                        w = vol_vals_num.fillna(0.0)
                        vol_out["vwap_mid"] = float((df_stats["mid"] * w).sum() / vol_sum)
                        if has_last:
                            vol_out["vwap_last"] = float((df_stats["last"] * w).sum() / vol_sum)
                    except Exception:
                        pass

                    try:
                        dmid = df_stats["mid"].diff().abs()
                        corr_df = pd.DataFrame(
                            {"volume": vol_vals_num, "abs_mid_change": dmid}
                        ).dropna()
                        if (
                            int(corr_df.shape[0]) >= 3
                            and int(corr_df["volume"].nunique()) > 1
                            and int(corr_df["abs_mid_change"].nunique()) > 1
                        ):
                            vol_out["corr_abs_mid_change"] = float(
                                corr_df["volume"].corr(corr_df["abs_mid_change"])
                            )
                    except Exception:
                        pass

                    try:
                        n_v = int(vol_vals_num.shape[0])
                        if n_v >= 4:
                            half = max(1, int(n_v // 2))
                            first_mean = float(vol_vals_num.iloc[:half].mean())
                            second_mean = float(vol_vals_num.iloc[half:].mean())
                            vol_out["half_ratio"] = (
                                float(second_mean / first_mean) if first_mean != 0.0 else float("nan")
                            )
                    except Exception:
                        pass

                if detailed_stats:
                    vol_out["dist"] = _series_stats(
                        vol_vals_num, total_count=trade_event_count
                    )
                if not small_summary_sample:
                    out["stats"][volume_kind] = vol_out

            _add_tick_data_quality(out)
            _add_tick_last_quality(out)
            _add_tick_context_fields(out)
            _round_tick_price_payload(out, price_digits)
            return _json_safe_payload(
                out if detailed_stats else _compact_tick_summary(out)
            )

        # If simplify mode requests approximation or resampling, use shared path
        if simplify_present and simplify_mode in ('approximate', 'resample'):
            df_out, simplify_meta = _simplify_dataframe_rows_ext(df_ticks, headers, simplify_used)
            rows = _format_numeric_rows_from_df(df_out, headers, stringify=False)
            rows = _round_row_price_columns(
                rows,
                headers,
                digits=price_digits,
                price_columns=_TICK_PRICE_COLUMNS,
            )
            table_payload = _table_from_rows(headers, rows)
            payload = {
                "success": True,
                "symbol": symbol,
                "count": len(rows),
            }
            payload.update(table_payload)
            payload["timezone"] = _timezone_label(use_client_tz=_use_ctz, client_tz=client_tz)
            if price_point is not None:
                payload["price_point"] = price_point
            if price_currency:
                payload["price_currency"] = price_currency
            units = _tick_units_for_headers(headers)
            if units:
                payload["units"] = units
            payload["tick_count"] = int(original_count)
            payload["trade_event_count"] = int(sum(trade_events))
            payload["quote_update_count"] = int(
                sum(flag & quote_update_mask != 0 for flag in flags)
            )
            _add_tick_summary_fields(payload)
            if has_flags:
                payload["flags_legend"] = _observed_tick_flags_decoded(flags)
            if simplify_meta is not None and original_count > len(rows):
                payload["simplified"] = True
                meta = dict(simplify_meta)
                meta["columns"] = [
                    c
                    for c in ["bid", "ask"]
                    + (["last"] if has_last else [])
                    + (["volume"] if has_volume else [])
                    + (["volume_real"] if has_real_volume else [])
                ]
                payload["simplify"] = meta
            _add_tick_data_quality(payload)
            _add_tick_last_quality(payload)
            _add_tick_context_fields(payload)
            return _json_safe_payload(payload)
        # Optional simplification based on a chosen y-series
        select_indices = list(range(original_count))
        _simp_method_used: Optional[str] = None
        _simp_params_meta: Optional[Dict[str, Any]] = None
        if simplify_present and original_count > 3:
            try:
                # Always represent available bid/ask/last and volume columns.
                cols: List[str] = ['bid', 'ask']
                if has_last:
                    cols.append('last')
                if has_volume:
                    cols.append('volume')
                if has_real_volume:
                    cols.append('volume_real')
                n_out = _choose_simplify_points(original_count, simplify_used)
                per = max(3, int(round(n_out / max(1, len(cols)))))
                idx_set: set = set([0, original_count - 1])
                params_accum: Dict[str, Any] = {}
                method_used_overall = None
                extracted_columns: Dict[str, List[float]] = {
                    "bid": bids,
                    "ask": asks,
                    "last": lasts,
                    "volume": volumes,
                    "volume_real": volumes_real,
                }
                series_by_col: Dict[str, List[float]] = {c: extracted_columns[c] for c in cols}
                for c in cols:
                    series = series_by_col[c]
                    sub_spec = dict(simplify)
                    sub_spec['points'] = per
                    idxs, method_used, params_meta = _select_indices_for_timeseries(_epochs, series, sub_spec)
                    method_used_overall = method_used
                    for i in idxs:
                        if 0 <= int(i) < original_count:
                            idx_set.add(int(i))
                    try:
                        if params_meta:
                            for k2, v2 in params_meta.items():
                                params_accum.setdefault(k2, v2)
                    except Exception:
                        pass
                union_idxs = sorted(idx_set)
                # Build composite metric for refinement/top-up
                mins: Dict[str, float] = {}
                ranges: Dict[str, float] = {}
                for c in cols:
                    vals = series_by_col[c]
                    if vals:
                        mn, mx = min(vals), max(vals)
                        ranges[c] = max(1e-12, mx - mn)
                        mins[c] = mn
                    else:
                        ranges[c] = 1.0
                        mins[c] = 0.0
                comp: List[float] = []
                for i in range(original_count):
                    s = 0.0
                    for c in cols:
                        vv = (series_by_col[c][i] - mins[c]) / ranges[c]
                        s += abs(vv)
                    comp.append(s)
                if len(union_idxs) > n_out:
                    refined = _lttb_select_indices(_epochs, comp, n_out)
                    select_indices = sorted(set(int(i) for i in refined if 0 <= i < original_count))
                elif len(union_idxs) < n_out:
                    refined = _lttb_select_indices(_epochs, comp, n_out)
                    merged = sorted(set(union_idxs).union(refined))
                    if len(merged) > n_out:
                        keep = set([0, original_count - 1])
                        candidates = [(comp[i], i) for i in merged if i not in keep]
                        candidates.sort(reverse=True)
                        for _, i in candidates:
                            keep.add(i)
                            if len(keep) >= n_out:
                                break
                        select_indices = sorted(keep)
                    else:
                        select_indices = merged
                else:
                    select_indices = union_idxs
                _simp_method_used = method_used_overall or str((simplify_used or {}).get('method', SIMPLIFY_DEFAULT_METHOD)).lower()
                _simp_params_meta = params_accum
            except Exception:
                select_indices = list(range(original_count))

        rows = []
        for i in select_indices:
            time_value = _format_tick_time(_epochs[i])
            values = [time_value]
            if full_rows:
                epoch_value = _epochs[i]
                values.append(int(epoch_value) if float(epoch_value).is_integer() else float(epoch_value))
            values.extend(
                [
                    _round_price_value(effective_bids[i], price_digits),
                    _round_price_value(effective_asks[i], price_digits),
                ]
            )
            if include_quote_type:
                values.append(quote_types[i])
            if full_rows:
                values.extend([quote_update_types[i], spread_valid_flags[i]])
                bid_value = effective_bids[i]
                ask_value = effective_asks[i]
                mid = (
                    (bid_value + ask_value) / 2.0
                    if bid_value is not None and ask_value is not None
                    else None
                )
                spread = (
                    ask_value - bid_value
                    if bid_value is not None and ask_value is not None
                    else None
                )
                spread_points = _tick_spread_points(spread, price_point)
                spread_pct = _tick_spread_pct(spread, mid)
                gap_ms = None if i <= 0 else float((_epochs[i] - _epochs[i - 1]) * 1000.0)
                values.extend(
                    [
                        _round_price_value(mid, price_digits),
                        _round_price_value(spread, price_digits),
                    ]
                )
                if price_point is not None:
                    values.append(spread_points)
                if points_per_pip is not None:
                    values.append(
                        round(spread_points / points_per_pip, 4)
                        if spread_points is not None
                        else None
                    )
                values.extend([spread_pct, gap_ms])
            values.append(_round_price_value(_finite_or_none(lasts[i]), price_digits))
            values.append(_finite_or_none(volumes[i]))
            values.append(_finite_or_none(volumes_real[i]))
            values.append(int(flags[i]) if flags[i] is not None else 0)
            values.append(_decode_tick_flags(flags[i]))
            rows.append(values)

        table_payload = _table_from_rows(headers, rows)
        payload = {
            "success": True,
            "symbol": symbol,
            "count": len(rows),
        }
        payload.update(table_payload)
        payload["timezone"] = _timezone_label(use_client_tz=_use_ctz, client_tz=client_tz)
        if price_point is not None:
            payload["price_point"] = price_point
        if price_currency:
            payload["price_currency"] = price_currency
        units = _tick_units_for_headers(headers)
        if units:
            payload["units"] = units
        payload["tick_count"] = int(original_count)
        payload["trade_event_count"] = int(sum(trade_events))
        payload["quote_update_count"] = int(
            sum(flag & quote_update_mask != 0 for flag in flags)
        )
        _add_tick_summary_fields(payload)
        if has_flags:
            payload["flags_legend"] = _observed_tick_flags_decoded(flags)
        _add_tick_data_quality(payload)
        _add_tick_last_quality(payload)
        _add_tick_context_fields(payload)
        if simplify_present and original_count > len(rows):
            payload["simplified"] = True
            meta = {
                "method": (_simp_method_used or str((simplify_used or {}).get('method', SIMPLIFY_DEFAULT_METHOD)).lower()),
                "original_rows": original_count,
                "multi_column": True,
                "columns": [
                    c
                    for c in ["bid", "ask"]
                    + (["last"] if has_last else [])
                    + (["volume"] if has_volume else [])
                    + (["volume_real"] if has_real_volume else [])
                ],
            }
            try:
                if _simp_params_meta:
                    meta.update(_simp_params_meta)
                else:
                    # Return key params if present
                    for key in ("epsilon", "max_error", "segments", "points", "ratio"):
                        if key in (simplify or {}):
                            meta[key] = (simplify or {})[key]
            except Exception:
                pass
            payload["simplify"] = meta
        return _json_safe_payload(payload)
    except Exception as e:
        return {"error": f"Error getting ticks: {str(e)}"}
