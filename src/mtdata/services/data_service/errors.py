
import errno
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

from ...bootstrap.settings import mt5_config
from ...core.error_envelope import build_error_payload
from ...core.output_contract import normalize_output_detail
from ...shared.constants import (
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
from ...shared.market_units import forex_points_per_pip
from ...shared.schema import DenoiseSpec, IndicatorSpec, SimplifySpec, TimeframeLiteral
from ...shared.validators import invalid_timeframe_error
from ...utils.coercion import round_finite
from ...utils.denoise import (
    DenoiseCausalityError,
    consume_denoise_warnings,
)
from ...utils.denoise import (
    apply_denoise as apply_denoise_util,
)
from ...utils.denoise import (
    normalize_denoise_spec as _normalize_denoise_spec,
)
from ...utils.freshness import closed_session_context, is_standard_weekend_closure
from ...utils.indicators import (
    _apply_ta_indicators,
    _estimate_warmup_bars,
    _find_unknown_ta_indicators,
    _parse_ti_specs,
)
from ...utils.market_metadata import (
    FRESHNESS_ANCHOR_QUERY_EXPECTED_END,
    FRESHNESS_ANCHOR_WALL_CLOCK,
    FRESHNESS_METRIC_LAST_COMPLETED_BAR_AGE,
    FRESHNESS_METRIC_REQUESTED_RANGE_END_GAP,
    TICK_VOLUME_COMPARISON_NOTE,
    TICK_VOLUME_EVENT_BASIS,
    TICK_VOLUME_TAPE_EQUIVALENT,
    build_tick_freshness_context,
)

# Imports from utils
from ...utils.mt5 import (
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
from ...utils.mt5 import (
    symbol_candle_price_basis as _symbol_candle_price_basis,
)
from ...utils.mt5 import (
    symbol_path as _symbol_path,
)
from ...utils.mt5 import (
    symbol_price_currency as _symbol_price_currency,
)
from ...utils.mt5 import (
    symbol_price_digits as _symbol_price_digits,
)
from ...utils.mt5 import (
    symbol_price_point as _symbol_price_point,
)
from ...utils.ohlcv import validate_and_clean_ohlcv_frame
from ...utils.quote import (
    canonical_quote_midpoint,
    canonical_quote_spread,
    enforce_quote_execution_readiness,
    resolve_quote_tick,
    tick_epoch,
)
from ...utils.quote import tick_value as _tick_field_value

# Simplify entrypoint and helpers.
from ...utils.simplify import (
    _choose_simplify_points,
    _lttb_select_indices,
    _select_indices_for_timeseries,
    _simplify_dataframe_rows_ext,
)
from ...utils.tick_flags import is_mt5_trade_event
from ...utils.time import (
    _format_datetime_minute_explicit,
    _format_time_explicit,
    _format_time_explicit_local,
    _localize_broker_calendar_time,
    _resolve_client_tz,
    bar_close_epoch,
    format_datetime_utc,
    format_epoch_utc,
)
from ...utils.utils import (
    _calendar_period_bounds,
    _format_numeric_rows_from_df,
    _iana_timezone_datetime_issue,
    _is_calendar_period_expression,
    _normalize_ohlcv_arg,
    _parse_end_datetime,
    _parse_start_datetime,
    _table_from_rows,
    _utc_epoch_seconds,
    coerce_scalar,
)

_DATE_FORMAT_HINT = (
    "Accepted examples: '2026-01-15', '2026-01-15 14:30', "
    "'2026-01-15T14:30:00Z', '2026-01-15 09:30 America/New_York', "
    "'yesterday', '2 days ago', 'last Friday'."
)


def _format_mt5_last_error() -> str:
    try:
        err = mt5.last_error()
    except Exception as exc:
        return str(exc)
    if isinstance(err, tuple) and len(err) == 2:
        code, message = err
        return f"({code}, {message!r})"
    return str(err)


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


def _bounded_weekend_no_data_context(
    symbol: str,
    start_datetime: Optional[str],
    end_datetime: Optional[str],
) -> Dict[str, Any]:
    if not start_datetime or not end_datetime:
        return {}
    try:
        start_utc, _ = _parse_fetch_datetime_arg(start_datetime)
        end_utc, _ = _parse_fetch_datetime_arg(end_datetime, end_bound=True)
        if start_utc is None or end_utc is None:
            return {}
        start_utc = (
            start_utc.replace(tzinfo=dt_timezone.utc)
            if start_utc.tzinfo is None
            else start_utc.astimezone(dt_timezone.utc)
        )
        end_utc = (
            end_utc.replace(tzinfo=dt_timezone.utc)
            if end_utc.tzinfo is None
            else end_utc.astimezone(dt_timezone.utc)
        )
        duration = end_utc - start_utc
        if duration.total_seconds() < 0 or duration > timedelta(days=3):
            return {}
        midpoint = start_utc + duration / 2
        if not (
            is_standard_weekend_closure(start_utc)
            and (
                is_standard_weekend_closure(end_utc)
                or is_standard_weekend_closure(midpoint)
            )
        ):
            return {}
        session = closed_session_context(
            symbol,
            now_epoch=midpoint.timestamp(),
            item="candles",
        )
        if not session or session.get("market_status_reason") != "weekend":
            return {}
    except Exception:
        return {}

    return {
        "no_data_reason": "market_closed_weekend",
        "market_status": "closed",
        "market_status_reason": "weekend",
        "market_status_source": "standard_weekend_hours",
        "note": (
            f"The requested range falls entirely within standard weekend closure "
            f"hours for {symbol}; no candles are expected."
        ),
        "suggestion": "Choose a range containing an open trading session.",
    }


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
    details.update(
        _bounded_weekend_no_data_context(symbol, start_datetime, end_datetime)
    )
    
    # Fetch only the latest available bar. Error construction must remain cheap;
    # discovering the terminal's full historical floor would require an
    # unbounded history read.
    try:
        available_bars = _mt5_copy_rates_from_pos(symbol, mt5_timeframe, 0, 1)

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
            last_epoch = max(times)
            last_time = datetime.fromtimestamp(last_epoch, tz=dt_timezone.utc)

            details["available_range"] = {
                "latest": _format_time_explicit(last_epoch),
                "earliest": None,
                "earliest_status": "not_scanned",
            }

            # Provide a suggestion based on the mismatch
            if start_datetime:
                try:
                    req_start, _ = _parse_fetch_datetime_arg(
                        start_datetime,
                        timeframe=timeframe,
                    )
                    if req_start is not None and req_start.tzinfo is None:
                        req_start = req_start.replace(tzinfo=dt_timezone.utc)
                    elif req_start is not None:
                        req_start = req_start.astimezone(dt_timezone.utc)
                    if req_start and req_start > last_time:
                        error_msg = f"No data available - requested start date is after latest available data ({_format_time_explicit(last_epoch)})"
                        details["suggestion"] = f"Use start='{_format_time_explicit(last_epoch)}' or earlier"
                except Exception:
                    pass
    except Exception:
        # Silently ignore any errors when trying to get available range
        pass
    
    payload = build_error_payload(
        error_msg,
        code="data_fetch_candles_no_data",
        operation="data_fetch_candles",
        details=details or None,
    )
    if start_datetime or end_datetime:
        payload["query_applied"] = _candle_query_applied(
            timeframe=timeframe,
            start=start_datetime,
            end=end_datetime,
            limit=None,
        )
    return payload


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


from .candles import _candle_query_applied, _parse_fetch_datetime_arg
