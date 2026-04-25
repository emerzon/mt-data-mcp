
import json
import logging
import math
import re
import time
import warnings
from datetime import datetime, timedelta
from datetime import timezone as dt_timezone
from typing import Any, Dict, List, Literal, Optional, Tuple

import pandas as pd

from ..bootstrap.settings import mt5_config
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
    TIME_DISPLAY_FORMAT,
    TIMEFRAME_MAP,
    TIMEFRAME_SECONDS,
)

# Imports from core (schema, constants, server utils)
# Imports from core (schema, constants)
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
from ..utils.indicators import (
    _apply_ta_indicators_util,
    _estimate_warmup_bars_util,
    _find_unknown_ta_indicators_util,
    _parse_ti_specs,
)

# Imports from utils
from ..utils.mt5 import (
    _mt5_copy_rates_from,
    _mt5_copy_rates_from_pos,
    _mt5_copy_rates_range,
    _mt5_copy_ticks_range,
    _mt5_epoch_to_utc as _mt5_epoch_to_utc_compat,
    _rates_to_df,
    _symbol_ready_guard,
    get_cached_mt5_time_alignment,
    get_symbol_info_cached,
    mt5,
)
from ..utils.ohlcv import validate_and_clean_ohlcv_frame

# Simplify entrypoint and helpers.
from ..utils.simplify import (
    _choose_simplify_points,
    _lttb_select_indices,
    _select_indices_for_timeseries,
    _simplify_dataframe_rows_ext,
)
from ..utils.utils import (
    _coerce_scalar,
    _format_numeric_rows_from_df,
    _format_time_minimal,
    _format_time_minimal_local,
    _maybe_strip_year,
    _normalize_ohlcv_arg,
    _parse_start_datetime,
    _resolve_client_tz,
    _style_time_format,
    _table_from_rows,
    _time_format_from_epochs,
    _utc_epoch_seconds,
)

logger = logging.getLogger(__name__)


def _mt5_epoch_to_utc(value: float) -> float:
    """Backward-compatible patch target; MT5 reads are normalized upstream."""
    return _mt5_epoch_to_utc_compat(value)


_AUTO_TIME_ALIGNMENT_MIN_SHIFT_SECONDS = 1800
_AUTO_TIME_ALIGNMENT_MAX_SHIFT_SECONDS = 18 * 3600
_TICK_SUMMARY_MIN_ANALYTIC_TICKS = 20


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
        return f"Symbol '{symbol}' was not found or is not available in MT5."
    return f"Failed to get rates for {symbol}: {error_text}"


def _build_no_data_error_with_context(
    symbol: str,
    timeframe: TimeframeLiteral,
    mt5_timeframe: int,
    start_datetime: Optional[str],
    end_datetime: Optional[str],
) -> Dict[str, Any]:
    """Build a detailed error message when no data is available for the requested range.
    
    Returns a dict with 'error' key and optional 'details' with context.
    """
    error_msg = "No data available"
    details: Dict[str, Any] = {}
    
    # Add requested range to context if provided
    if start_datetime or end_datetime:
        details["requested_range"] = {
            k: v for k, v in [("start", start_datetime), ("end", end_datetime)]
            if v is not None
        }
    
    # Try to get first and last available bars for this timeframe to suggest range
    try:
        first_bar = _mt5_copy_rates_from_pos(symbol, mt5_timeframe, 0, 1)
        last_bar = _mt5_copy_rates_from_pos(symbol, mt5_timeframe, -1, 1)
        
        if first_bar is not None and len(first_bar) > 0 and last_bar is not None and len(last_bar) > 0:
            first_time = datetime.fromtimestamp(first_bar[0]['time'], tz=dt_timezone.utc)
            last_time = datetime.fromtimestamp(last_bar[0]['time'], tz=dt_timezone.utc)
            
            details["available_range"] = {
                "earliest": _format_time_minimal(first_bar[0]['time']),
                "latest": _format_time_minimal(last_bar[0]['time']),
            }
            
            # Provide a suggestion based on the mismatch
            if start_datetime:
                try:
                    req_start, _ = _parse_fetch_datetime_arg(start_datetime)
                    if req_start and req_start > last_time:
                        error_msg = f"No data available - requested start date is after latest available data ({_format_time_minimal(last_bar[0]['time'])})"
                        details["suggestion"] = f"Use start='{_format_time_minimal(last_bar[0]['time'])}' or earlier"
                    elif req_start and req_start < first_time:
                        error_msg = f"No data available - requested date range is before earliest available data ({_format_time_minimal(first_bar[0]['time'])})"
                        details["suggestion"] = f"Use start='{_format_time_minimal(first_bar[0]['time'])}' or later"
                except Exception:
                    pass
    except Exception:
        # Silently ignore any errors when trying to get available range
        pass
    
    result: Dict[str, Any] = {"error": error_msg}
    if details:
        result["details"] = details
    return result


def _indicator_param_syntax_error(ti_spec: Optional[str]) -> Optional[str]:
    if not ti_spec:
        return None
    for name, _args, _kwargs in _parse_ti_specs(ti_spec):
        if re.fullmatch(r"[+-]?\d+(?:\.\d+)?", str(name or "").strip()):
            return "Indicator params must use parentheses, e.g. sma(20), not sma,20."
    return None


def _resolve_live_bar_reference_epoch(symbol: Optional[str], timeframe: str) -> float:
    """Prefer a fresh broker tick timestamp when classifying the live bar."""
    seconds_per_bar = int(TIMEFRAME_SECONDS.get(timeframe, 3600) or 3600)
    system_epoch = _utc_epoch_seconds(datetime.now(dt_timezone.utc))
    symbol_name = str(symbol or "").strip()
    if not symbol_name:
        return float(system_epoch)
    try:
        tick = mt5.symbol_info_tick(symbol_name)
        tick_time = getattr(tick, "time", None) if tick is not None else None
        if tick_time is None:
            return float(system_epoch)
        tick_epoch = float(tick_time)
        if not math.isfinite(tick_epoch):
            return float(system_epoch)
        freshness_limit = float(max(seconds_per_bar, 300))
        if abs(float(system_epoch) - tick_epoch) <= freshness_limit:
            return tick_epoch
    except Exception:
        pass
    return float(system_epoch)


def _is_last_candle_open(
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
            if not rates_or_df or len(rates_or_df) == 0:
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
        and _is_last_candle_open(rates, timeframe, current_time_epoch=current_time_epoch)
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
    if len(df) > 0 and _is_last_candle_open(df, timeframe, current_time_epoch=current_time_epoch):
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
        data_freshness_seconds = float(expected_epoch - last_epoch)
    if last_epoch is not None and cutoff_epoch is not None:
        last_bar_within_policy_window = bool(last_epoch >= cutoff_epoch)

    return {
        "last_bar_epoch": last_epoch,
        "expected_end_epoch": expected_epoch,
        "freshness_cutoff_epoch": cutoff_epoch,
        "data_freshness_seconds": data_freshness_seconds,
        "last_bar_within_policy_window": last_bar_within_policy_window,
    }


def _fetch_rates_with_warmup(
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
    auto_shift_seconds = _resolve_live_rate_auto_shift_seconds(
        symbol=symbol,
        timeframe=timeframe,
        start_datetime=start_datetime,
        end_datetime=end_datetime,
    )
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
        to_date = from_date + timedelta(seconds=seconds_per_bar * (candles + 2 + extra_bars))
        from_date_internal = from_date - timedelta(seconds=seconds_per_bar * (warmup_bars + extra_bars))
        expected_end_ts = _utc_epoch_seconds(to_date)

        def _fetch():
            return _mt5_copy_rates_range(symbol, mt5_timeframe, from_date_internal, to_date)

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
            if auto_shift_seconds:
                rates = _shift_rate_times(rates, auto_shift_seconds)
            if not sanity_check:
                break
            last_t = rates[-1]["time"]
            freshness_cutoff = expected_end_ts - seconds_per_bar * (SANITY_BARS_TOLERANCE + extra_bars)
            freshness_meta = _build_candle_freshness_diagnostics(
                last_bar_epoch=last_t,
                expected_end_epoch=expected_end_ts,
                freshness_cutoff_epoch=freshness_cutoff,
            )
            if diagnostics is not None:
                diagnostics["freshness"] = freshness_meta
            if bool(freshness_meta.get("last_bar_within_policy_window")):
                stale_last_t = None
                break
            if (
                not start_datetime
                and not end_datetime
                and not include_incomplete
                and not _is_last_candle_open(
                    rates,
                    timeframe,
                    current_time_epoch=float(expected_end_ts),
                )
            ):
                freshness_meta["freshness_policy_relaxed"] = (
                    "latest_completed_bar_for_live_request"
                )
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
            f"from {_format_time_minimal(stale_last_t)}. Market may be closed; "
            "set allow_stale=true to retrieve the latest "
            "available completed historical bars."
        )
    return rates, None


def _parse_fetch_datetime_arg(value: str) -> tuple[Optional[datetime], Optional[str]]:
    parsed = _parse_start_datetime(value)
    if parsed is None:
        return None, "Invalid date format. Try '2025-08-29', '2025-08-29 14:30', 'yesterday 14:00'."
    return parsed, None


def _resolve_fetch_timeframe_seconds(timeframe: TimeframeLiteral) -> tuple[Optional[int], Optional[str]]:
    seconds_per_bar = TIMEFRAME_SECONDS.get(timeframe)
    if not seconds_per_bar:
        return None, f"Unable to determine timeframe seconds for {timeframe}"
    return int(seconds_per_bar), None


def _resolve_live_rate_auto_shift_seconds(
    *,
    symbol: str,
    timeframe: TimeframeLiteral,
    start_datetime: Optional[str],
    end_datetime: Optional[str],
) -> int:
    if start_datetime or end_datetime:
        return 0

    try:
        if mt5_config.get_server_tz() is not None:
            return 0
    except Exception:
        pass

    try:
        configured_offset_seconds = int(mt5_config.get_time_offset_seconds())
    except Exception:
        configured_offset_seconds = 0
    if configured_offset_seconds != 0:
        return 0

    try:
        alignment = get_cached_mt5_time_alignment(
            symbol=symbol,
            probe_timeframe=timeframe,
            ttl_seconds=int(getattr(mt5_config, "broker_time_check_ttl_seconds", 60) or 60),
        )
    except Exception:
        return 0

    if not isinstance(alignment, dict) or not bool(alignment.get("offset_inference_reliable")):
        return 0

    try:
        inferred_offset_seconds = int(alignment.get("inferred_offset_seconds"))
    except Exception:
        return 0

    if not (
        _AUTO_TIME_ALIGNMENT_MIN_SHIFT_SECONDS
        <= abs(inferred_offset_seconds)
        <= _AUTO_TIME_ALIGNMENT_MAX_SHIFT_SECONDS
    ):
        return 0

    return -inferred_offset_seconds


def _shift_rate_times(rates: Any, shift_seconds: int) -> Any:
    if rates is None or int(shift_seconds) == 0:
        return rates

    shift_value = int(shift_seconds)
    try:
        names = getattr(getattr(rates, "dtype", None), "names", None)
    except Exception:
        names = None

    if names and "time" in names:
        try:
            shifted_rates = rates.copy()
            shifted_rates["time"] = shifted_rates["time"] + shift_value
        except Exception:
            return rates
        return shifted_rates

    if isinstance(rates, list):
        if not any(isinstance(row, dict) and "time" in row for row in rates):
            return rates
        shifted_rows = []
        for row in rates:
            if not isinstance(row, dict):
                shifted_rows.append(row)
                continue
            if "time" not in row:
                shifted_rows.append(row)
                continue
            shifted_row = dict(row)
            try:
                shifted_row["time"] = float(shifted_row["time"]) + shift_value
            except Exception:
                pass
            shifted_rows.append(shifted_row)
        return shifted_rows
    return rates


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

    formatted = dt_series.dt.strftime(TIME_DISPLAY_FORMAT)
    if bool(formatted.isna().any()):
        formatter = _format_time_minimal_local if use_client_tz else _format_time_minimal
        fallback = epochs.map(lambda value: formatter(float(value)) if pd.notna(value) else None)
        formatted = formatted.where(~formatted.isna(), fallback)
    return formatted


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
            out = out.iloc[:candles]
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
                    args_str = ",".join(str(_coerce_scalar(str(param))) for param in params)
                    parts.append(f"{name}({args_str})")
                elif isinstance(params, dict) and len(params) > 0:
                    args_str = ",".join(
                        f"{str(key).strip()}={_coerce_scalar(str(param))}"
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
        return "ATR_" + text[len("ATRr_") :]
    return text


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
    tick_volumes = [int(rate["tick_volume"]) for rate in rates]
    real_volumes = [int(rate["real_volume"]) for rate in rates]

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


def _append_denoise_application(
    denoise_apps: List[Dict[str, Any]],
    source_spec: Any,
    *,
    default_when: str,
    default_causality: str,
    default_keep_original: bool,
    added_columns: List[str],
) -> None:
    try:
        denoise_meta = dict(source_spec or {})
        denoise_apps.append(
            {
                'method': str(denoise_meta.get('method', 'none')).lower(),
                'when': str(denoise_meta.get('when', default_when)).lower(),
                'causality': str(denoise_meta.get('causality', default_causality)),
                'keep_original': bool(denoise_meta.get('keep_original', default_keep_original)),
                'columns': denoise_meta.get('columns', 'close'),
                'params': denoise_meta.get('params') or {},
                'added_columns': added_columns,
            }
        )
    except Exception:
        pass


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
        _extend_unique_headers(headers, added_columns)

    _append_denoise_application(
        denoise_apps,
        denoise,
        default_when='pre_ti',
        default_causality='causal',
        default_keep_original=False,
        added_columns=added_columns,
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

    ti_cols = _apply_ta_indicators_util(df, ti_spec)
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


def _apply_post_ti_denoise(
    df: pd.DataFrame,
    headers: List[str],
    denoise: Optional[DenoiseSpec],
    denoise_apps: List[Dict[str, Any]],
) -> None:
    if not denoise:
        return

    normalized = _normalize_denoise_spec(denoise, default_when='post_ti')
    added_columns: List[str] = []
    if normalized and str(normalized.get('when', 'post_ti')).lower() == 'post_ti':
        added_columns = _apply_denoise_util(df, normalized, default_when='post_ti')
        _extend_unique_headers(headers, added_columns)

    _append_denoise_application(
        denoise_apps,
        normalized,
        default_when='post_ti',
        default_causality='zero_phase',
        default_keep_original=True,
        added_columns=added_columns,
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
                from_disp = _format_time_minimal_local(prev_t)
                to_disp = _format_time_minimal_local(curr_t)
            else:
                from_disp = _format_time_minimal(prev_t)
                to_disp = _format_time_minimal(curr_t)

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

    fmt = TIME_DISPLAY_FORMAT
    tz_used_name = 'UTC'
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        time_values = pd.to_datetime(epochs, unit='s', utc=True)
        if use_client_tz:
            tz_used_name = getattr(client_tz, 'zone', None) or str(client_tz)
            time_values = time_values.dt.tz_convert(client_tz)
        df['time'] = time_values.dt.strftime(fmt)
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
        
        # Ensure symbol is ready; remember original visibility to restore later
        _info_before = get_symbol_info_cached(symbol)
        with _symbol_ready_guard(symbol, info_before=_info_before) as (err, _info):
            if err:
                return {"error": err}

            try:
                ti_spec = _normalize_indicator_spec(ti)
            except ValueError as exc:
                return {"error": str(exc)}
            indicator_syntax_error = _indicator_param_syntax_error(ti_spec)
            if indicator_syntax_error:
                return {"error": indicator_syntax_error}
            # Determine warmup bars if technical indicators requested
            unknown_indicators = _find_unknown_ta_indicators_util(ti_spec or "")
            if unknown_indicators:
                return {
                    "error": (
                        "Unknown indicator(s): "
                        + ", ".join(unknown_indicators)
                        + ". Use indicators_list to view valid indicator names."
                    )
                }
            warmup_bars = _estimate_warmup_bars_util(ti_spec)
            rate_fetch_diagnostics: Dict[str, Any] = {}
            freshness_diagnostics: Optional[Dict[str, Any]] = None

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
                sanity_check=not bool(allow_stale),
                diagnostics=rate_fetch_diagnostics,
            )
            freshness_diagnostics = rate_fetch_diagnostics.get("freshness")
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

        # If TI requested, check for NaNs and retry once with increased warmup
        if ti_spec and ti_cols:
            try:
                if df[ti_cols].isna().any().any():
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
                        sanity_check=not bool(allow_stale),
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
        last_candle_open = _is_last_candle_open(
            df,
            timeframe,
            current_time_epoch=live_bar_reference_epoch,
        )
        rows = _format_numeric_rows_from_df(df, headers, stringify=False)
        query_latency_ms = round((time.perf_counter() - query_started_at) * 1000.0, 3)
        query_mode = "range" if (start_datetime or end_datetime) else "latest"
        ti_added_cols = [str(c) for c in ti_cols if isinstance(c, str)]
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
        has_forming_candle = bool(initial_incomplete_trimmed or _trimmed_incomplete or last_candle_open)

        payload.update({
            "success": True,
            "symbol": symbol,
            "timeframe": timeframe,
            "candles": candles_returned,
            "candles_requested": candles_requested,
            "candles_excluded": candles_excluded,
            "last_candle_open": last_candle_open,
            "incomplete_candles_skipped": incomplete_candles_skipped,
            "has_forming_candle": has_forming_candle,
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
                },
            },
        })
        if incomplete_candles_skipped and not include_incomplete:
            payload["hint"] = "Set include_incomplete=true to include the latest forming candle."
        if isinstance(freshness_diagnostics, dict):
            payload["meta"]["diagnostics"]["freshness"] = dict(freshness_diagnostics)
        if session_gap_warning:
            payload["meta"]["diagnostics"]["session_gaps"]["warning"] = session_gap_warning
        if not _use_ctz:
            payload["timezone"] = "UTC"
        if simplify_meta is not None:
            payload["simplified"] = True
            payload["simplify"] = _public_simplify_meta(simplify_meta) or {"applied": True}
        # Attach denoise applications metadata if any
        if denoise_apps:
            payload['denoise'] = {'applications': denoise_apps}
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
        return payload
    except Exception as e:
        return {"error": f"Error getting rates: {str(e)}"}


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
            "Bid price changed or is present in the tick.",
        ),
        (
            _mt5_tick_flag_value("TICK_FLAG_ASK", 4),
            "ask",
            "Ask price changed or is present in the tick.",
        ),
        (
            _mt5_tick_flag_value("TICK_FLAG_LAST", 8),
            "last",
            "Last traded price changed or is present in the tick.",
        ),
        (
            _mt5_tick_flag_value("TICK_FLAG_VOLUME", 16),
            "volume",
            "Tick volume changed or is present in the tick.",
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
            "Real volume changed or is present in the tick.",
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


def _compact_tick_summary(out: Dict[str, Any]) -> Dict[str, Any]:
    spread = out.get("stats", {}).get("spread")
    compact_spread: Dict[str, Any] = {}
    if isinstance(spread, dict):
        if spread.get("available") is False:
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
        "count": out.get("count"),
        "start": out.get("start"),
        "end": out.get("end"),
        "duration_seconds": out.get("duration_seconds"),
        "tick_rate_per_second": out.get("tick_rate_per_second"),
        "timezone": out.get("timezone"),
        "stats": {"spread": compact_spread},
    }
    return compact


def fetch_ticks(  # noqa: C901
    symbol: str,
    limit: int = DEFAULT_ROW_LIMIT,
    start: Optional[str] = None,
    end: Optional[str] = None,
    simplify: Optional[SimplifySpec] = None,
    format: Literal["summary", "stats", "rows"] = "summary",
) -> Dict[str, Any]:
    """Fetch tick data and return either a summary (default) or raw rows.

    Parameters
    ----------
    format : {"summary","stats","rows"}
        - "summary" (default): compact descriptive statistics over the fetched
          ticks. Samples below 20 ticks report spread stats only with a sample
          adequacy note; larger samples include bid/ask/mid, plus last and
          volume when available.
        - "stats": more detailed stats (includes extra distribution moments and
          quantiles).
        - "rows": return tick rows as structured data.
    """
    try:
        effective_limit = int(limit)
        if effective_limit <= 0:
            return {"error": "limit must be greater than 0."}
        # Ensure symbol is ready; remember original visibility to restore later
        _info_before = get_symbol_info_cached(symbol)
        with _symbol_ready_guard(symbol, info_before=_info_before) as (err, _info):
            if err:
                return {"error": err}
            price_digits = 0
            try:
                digits_raw = getattr(_info, "digits", None)
                if isinstance(digits_raw, (int, float)):
                    price_digits = max(0, int(digits_raw))
            except Exception:
                price_digits = 0

            # Normalized params only
            output_mode = normalize_output_detail(
                format,
                default="summary",
                aliases={
                    "raw": "rows",
                    "ticks": "rows",
                },
            )
            if start:
                from_date = _parse_start_datetime(start)
                if not from_date:
                    return {"error": "Invalid date format. Try examples like '2025-08-29', '2025-08-29 14:30', 'yesterday 14:00', '2 days ago'."}
                if end:
                    to_date = _parse_start_datetime(end)
                    if not to_date:
                        return {"error": "Invalid 'end' date format. Try '2025-08-29 14:30' or 'yesterday 18:00'."}
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

        if output_mode not in ("summary", "stats", "rows"):
            return {
                "error": (
                    f"Invalid format: {format}. "
                    "Use 'summary', 'stats', or 'rows'."
                )
            }

        def _tick_field(tick: Any, name: str) -> Any:
            return _tick_field_value(tick, name)

        needs_stats = output_mode in ("summary", "stats")

        # Extract shared tick columns once so summary/stats, simplification,
        # and row rendering can all reuse the same values.
        _epochs: List[float] = []
        bids: List[float] = []
        asks: List[float] = []
        lasts: List[float] = []
        flags: List[int] = []
        volumes: List[float] = []
        volumes_real: List[float] = []
        for tick in ticks:
            _epochs.append(float(_tick_field(tick, "time")))
            bids.append(float(_tick_field(tick, "bid")))
            asks.append(float(_tick_field(tick, "ask")))
            lasts.append(float(_tick_field(tick, "last") or 0.0))
            flags.append(int(_tick_field(tick, "flags") or 0))
            try:
                volumes.append(float(_tick_field(tick, "volume")))
            except (TypeError, ValueError):
                volumes.append(float("nan"))
            try:
                volumes_real.append(float(_tick_field(tick, "volume_real")))
            except (TypeError, ValueError):
                volumes_real.append(float("nan"))

        has_last = len(set(lasts)) > 1 or any(v != 0 for v in lasts)
        finite_volumes = [v for v in volumes if math.isfinite(v)]
        has_volume = bool(finite_volumes) and (
            len(set(finite_volumes)) > 1 or any(v != 0.0 for v in finite_volumes)
        )
        has_flags = len(set(flags)) > 1 or any(v != 0 for v in flags)
        has_real_volume = any(math.isfinite(v) and v != 0.0 for v in volumes_real)

        # Build header dynamically (time, bid, ask are always included)
        headers = ["time", "bid", "ask"]
        if has_last:
            headers.append("last")
        if has_volume:
            headers.append("volume")
        if has_real_volume:
            headers.append("volume_real")
        if has_flags:
            headers.append("flags")

        # Choose a consistent time format for all rows (strip year if constant).
        # Low-level tick fetch helpers already normalize MT5 times to UTC.
        client_tz = _resolve_client_tz()
        _use_ctz = client_tz is not None
        fmt: Optional[str] = None
        if not _use_ctz:
            fmt = _time_format_from_epochs(_epochs)
            fmt = _maybe_strip_year(fmt, _epochs)
            fmt = _style_time_format(fmt)

        def _format_tick_time(epoch: float) -> str:
            if _use_ctz:
                return _format_time_minimal_local(epoch)
            return datetime.fromtimestamp(epoch, tz=dt_timezone.utc).strftime(fmt)

        original_count = len(ticks)
        simplify_eff = _normalize_simplify_spec(simplify, limit=limit, fallback_rows=original_count)
        simplify_present = (simplify_eff is not None) or (simplify is not None)
        simplify_used = simplify_eff if simplify_eff is not None else simplify
        simplify_mode = (
            str((simplify_used or {}).get("mode", SIMPLIFY_DEFAULT_MODE)).lower().strip()
            if simplify_present
            else SIMPLIFY_DEFAULT_MODE
        )

        df_ticks: Optional[pd.DataFrame] = None
        if output_mode in ("summary", "stats") or (
            simplify_present and simplify_mode in ("approximate", "resample")
        ):
            df_ticks = pd.DataFrame({
                "__epoch": _epochs,
                "bid": bids,
                "ask": asks,
            })
            if has_last:
                df_ticks["last"] = lasts
            if has_volume:
                df_ticks["volume"] = volumes
            if has_flags:
                df_ticks["flags"] = flags
            df_ticks["time"] = [_format_tick_time(e) for e in _epochs]

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
                    return out
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
                return out

            df_stats = df_ticks.copy()
            df_stats["mid"] = (df_stats["bid"] + df_stats["ask"]) / 2.0
            df_stats["spread"] = (df_stats["ask"] - df_stats["bid"])

            start_epoch = float(df_stats["__epoch"].iloc[0])
            end_epoch = float(df_stats["__epoch"].iloc[-1])
            duration_seconds = float(max(0.0, end_epoch - start_epoch))
            tick_rate_per_second = (
                float(len(df_stats) / duration_seconds) if duration_seconds > 0 else None
            )

            timezone = "UTC"
            if _use_ctz:
                try:
                    timezone = str(client_tz)
                except Exception:
                    timezone = "local"

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
            if has_last and not small_summary_sample:
                out["stats"]["last"] = _series_stats(df_stats["last"], total_count=len(df_stats))

            volume_kind = "tick_volume"
            vol_vals = pd.Series([1.0] * int(len(df_stats)), dtype=float)
            if has_real_volume and len(volumes_real) == len(df_stats):
                volume_kind = "real_volume"
                vol_vals = pd.Series(volumes_real, dtype=float)

            if volume_kind == "real_volume":
                vol_vals_num = pd.to_numeric(vol_vals, errors="coerce").astype(float)
                vol_sum = float(vol_vals_num.fillna(0.0).sum())
                vol_nonzero_count = int((vol_vals_num.fillna(0.0) != 0.0).sum())
                vol_out: Dict[str, Any] = {
                    "kind": volume_kind,
                    "sum": vol_sum,
                    "per_second": (
                        float(vol_sum / duration_seconds) if duration_seconds > 0 else None
                    ),
                    "per_tick": float(vol_sum / float(len(df_stats) or 1)),
                    "nonzero_share": float(vol_nonzero_count) / float(len(df_stats) or 1),
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
                        if int(corr_df.shape[0]) >= 3:
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
                    vol_out["dist"] = _series_stats(vol_vals_num, total_count=len(df_stats))
                if not small_summary_sample:
                    out["stats"]["volume"] = vol_out
            else:
                if detailed_stats:
                    out["stats"]["volume"] = {
                        "kind": volume_kind,
                        "per_second": tick_rate_per_second,
                        "sum": int(len(df_stats)),
                    }
                elif not small_summary_sample:
                    out["stats"]["volume"] = {"kind": volume_kind}

            return out if detailed_stats else _compact_tick_summary(out)

        # If simplify mode requests approximation or resampling, use shared path
        if simplify_present and simplify_mode in ('approximate', 'resample'):
            df_out, simplify_meta = _simplify_dataframe_rows_ext(df_ticks, headers, simplify_used)
            rows = _format_numeric_rows_from_df(df_out, headers, stringify=False)
            payload = _table_from_rows(headers, rows)
            payload.update({
                "success": True,
                "symbol": symbol,
                "count": len(rows),
            })
            if not _use_ctz:
                payload["timezone"] = "UTC"
            if has_flags:
                payload["flags_decoded"] = _observed_tick_flags_decoded(flags)
            if simplify_meta is not None and original_count > len(rows):
                payload["simplified"] = True
                meta = dict(simplify_meta)
                meta["columns"] = [c for c in ["bid","ask"] + (["last"] if has_last else []) + (["volume"] if has_volume else [])]
                payload["simplify"] = meta
            return payload
        # Optional simplification based on a chosen y-series
        select_indices = list(range(original_count))
        _simp_method_used: Optional[str] = None
        _simp_params_meta: Optional[Dict[str, Any]] = None
        if simplify_present and original_count > 3:
            try:
                # Always represent all available numeric columns (bid/ask/(last)/(volume))
                cols: List[str] = ['bid', 'ask']
                if has_last:
                    cols.append('last')
                if has_volume:
                    cols.append('volume')
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
            time_str = _format_tick_time(_epochs[i])
            values = [time_str, bids[i], asks[i]]
            if has_last:
                values.append(lasts[i])
            if has_volume:
                values.append(volumes[i])
            if has_real_volume:
                values.append(volumes_real[i])
            if has_flags:
                values.append(flags[i])
            rows.append(values)

        payload = _table_from_rows(headers, rows)
        payload.update({
            "success": True,
            "symbol": symbol,
            "count": len(rows),
        })
        if not _use_ctz:
            payload["timezone"] = "UTC"
        if has_flags:
            payload["flags_decoded"] = _observed_tick_flags_decoded(flags)
        if simplify_present and original_count > len(rows):
            payload["simplified"] = True
            meta = {
                "method": (_simp_method_used or str((simplify_used or {}).get('method', SIMPLIFY_DEFAULT_METHOD)).lower()),
                "original_rows": original_count,
                "multi_column": True,
                "columns": [c for c in ["bid","ask"] + (["last"] if has_last else []) + (["volume"] if has_volume else [])],
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
        return payload
    except Exception as e:
        return {"error": f"Error getting ticks: {str(e)}"}

