
from datetime import datetime, timedelta, timezone as dt_timezone
import logging
import math
from typing import Any, Dict, Optional, List, Literal
import pandas as pd
import warnings
import json
import time

# Imports from core (schema, constants, server utils)

# Imports from core (schema, constants)
from ..core.schema import TimeframeLiteral, IndicatorSpec, DenoiseSpec, SimplifySpec
from ..core.constants import (
    TIMEFRAME_MAP, TIMEFRAME_SECONDS, FETCH_RETRY_ATTEMPTS, FETCH_RETRY_DELAY,
    SANITY_BARS_TOLERANCE, TI_NAN_WARMUP_FACTOR, TI_NAN_WARMUP_MIN_ADD,
    SIMPLIFY_DEFAULT_MODE, SIMPLIFY_DEFAULT_POINTS_RATIO_FROM_LIMIT, TICKS_LOOKBACK_DAYS,
    DEFAULT_ROW_LIMIT
)

# Imports from utils
from ..utils.mt5 import (
    _mt5_copy_rates_from, _mt5_copy_rates_range, _mt5_copy_ticks_from,
    _mt5_copy_ticks_range, _mt5_epoch_to_utc, _ensure_symbol_ready, get_symbol_info_cached
)
from ..utils.utils import (
    _table_from_rows, _format_time_minimal, _format_time_minimal_local,
    _resolve_client_tz, _time_format_from_epochs, _maybe_strip_year,
    _style_time_format, _format_numeric_rows_from_df, _parse_start_datetime,    
    _coerce_scalar, _normalize_ohlcv_arg
)
from ..utils.indicators import _estimate_warmup_bars_util, _apply_ta_indicators_util
from ..utils.denoise import _apply_denoise as _apply_denoise_util, normalize_denoise_spec as _normalize_denoise_spec

# Imports from core (simplify - to be refactored later, but for now import from core or utils?)
# The plan says "Refactor core/simplify.py" is next.
# For now, I will import from core.simplify to keep it working, or better, import the utils directly if possible.
# core/simplify.py delegates to utils/simplify.py mostly.
# But `_simplify_dataframe_rows_ext` is in core/simplify.py.
from ..core.simplify import _simplify_dataframe_rows_ext, _choose_simplify_points, _select_indices_for_timeseries, _lttb_select_indices

import MetaTrader5 as mt5

logger = logging.getLogger(__name__)


def _fetch_rates_with_warmup(
    symbol: str,
    mt5_timeframe: int,
    timeframe: TimeframeLiteral,
    candles: int,
    warmup_bars: int,
    start_datetime: Optional[str],
    end_datetime: Optional[str],
    *,
    retry: bool = True,
    sanity_check: bool = True,
):
    """Fetch MT5 rates with optional warmup, retry, and end-bar sanity checks."""
    if start_datetime and end_datetime:
        from_date = _parse_start_datetime(start_datetime)
        to_date = _parse_start_datetime(end_datetime)
        if not from_date or not to_date:
            return None, "Invalid date format. Try '2025-08-29', '2025-08-29 14:30', 'yesterday 14:00'."
        if from_date > to_date:
            return None, "start_datetime must be before end_datetime"
        seconds_per_bar = TIMEFRAME_SECONDS.get(timeframe, 60)
        from_date_internal = from_date - timedelta(seconds=seconds_per_bar * warmup_bars)
        expected_end_ts = to_date.timestamp()

        def _fetch():
            return _mt5_copy_rates_range(symbol, mt5_timeframe, from_date_internal, to_date)

    elif start_datetime:
        from_date = _parse_start_datetime(start_datetime)
        if not from_date:
            return None, "Invalid date format. Try '2025-08-29', '2025-08-29 14:30', 'yesterday 14:00'."
        seconds_per_bar = TIMEFRAME_SECONDS.get(timeframe)
        if not seconds_per_bar:
            return None, f"Unable to determine timeframe seconds for {timeframe}"
        to_date = from_date + timedelta(seconds=seconds_per_bar * (candles + 2))
        from_date_internal = from_date - timedelta(seconds=seconds_per_bar * warmup_bars)
        expected_end_ts = to_date.timestamp()

        def _fetch():
            return _mt5_copy_rates_range(symbol, mt5_timeframe, from_date_internal, to_date)

    elif end_datetime:
        to_date = _parse_start_datetime(end_datetime)
        if not to_date:
            return None, "Invalid date format. Try '2025-08-29', '2025-08-29 14:30', 'yesterday 14:00'."
        seconds_per_bar = TIMEFRAME_SECONDS.get(timeframe, 60)
        expected_end_ts = to_date.timestamp()

        def _fetch():
            return _mt5_copy_rates_from(symbol, mt5_timeframe, to_date, candles + warmup_bars)

    else:
        utc_now = datetime.utcnow()
        seconds_per_bar = TIMEFRAME_SECONDS.get(timeframe, 60)
        expected_end_ts = utc_now.timestamp()

        def _fetch():
            return _mt5_copy_rates_from(symbol, mt5_timeframe, utc_now, candles + warmup_bars)

    attempts = FETCH_RETRY_ATTEMPTS if retry else 1
    rates = None
    for idx in range(attempts):
        rates = _fetch()
        if rates is not None and len(rates) > 0:
            if not sanity_check:
                break
            last_t = rates[-1]["time"]
            if last_t >= (expected_end_ts - seconds_per_bar * SANITY_BARS_TOLERANCE):
                break
        if retry and idx < (attempts - 1):
            time.sleep(FETCH_RETRY_DELAY)
    return rates, None


def _build_rates_df(rates: Any, use_client_tz: bool) -> pd.DataFrame:
    """Normalize raw MT5 rates into a DataFrame with epoch and display time columns."""
    df = pd.DataFrame(rates)
    try:
        if 'time' in df.columns:
            df['time'] = df['time'].astype(float).apply(_mt5_epoch_to_utc)
    except Exception:
        pass
    df['__epoch'] = df['time']
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df["time"] = df["time"].apply(_format_time_minimal_local if use_client_tz else _format_time_minimal)
    if 'volume' not in df.columns and 'tick_volume' in df.columns:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df['volume'] = df['tick_volume']
    return df


def _trim_df_to_target(
    df: pd.DataFrame,
    start_datetime: Optional[str],
    end_datetime: Optional[str],
    candles: int,
    *,
    copy_rows: bool = True,
) -> pd.DataFrame:
    if start_datetime and end_datetime:
        target_from = _parse_start_datetime(start_datetime).timestamp()
        target_to = _parse_start_datetime(end_datetime).timestamp()
        out = df.loc[(df['__epoch'] >= target_from) & (df['__epoch'] <= target_to)]
    elif start_datetime:
        target_from = _parse_start_datetime(start_datetime).timestamp()
        out = df.loc[df['__epoch'] >= target_from]
        if len(out) > candles:
            out = out.iloc[:candles]
    elif end_datetime:
        out = df.iloc[-candles:] if len(df) > candles else df
    else:
        out = df.iloc[-candles:] if len(df) > candles else df
    return out.copy() if copy_rows else out


def fetch_candles(
    symbol: str,
    timeframe: TimeframeLiteral = "H1",
    limit: int = DEFAULT_ROW_LIMIT,
    start: Optional[str] = None,
    end: Optional[str] = None,
    ohlcv: Optional[str] = None,
    indicators: Optional[List[IndicatorSpec]] = None,
    denoise: Optional[DenoiseSpec] = None,
    simplify: Optional[SimplifySpec] = None,
) -> Dict[str, Any]:
    """Return historical candles as tabular data."""
    try:
        # Backward/compat mappings to internal variable names used in implementation
        candles = int(limit)
        start_datetime = start
        end_datetime = end
        ti = indicators
        # Validate timeframe using the shared map
        if timeframe not in TIMEFRAME_MAP:
            return {"error": f"Invalid timeframe: {timeframe}. Valid options: {list(TIMEFRAME_MAP.keys())}"}
        mt5_timeframe = TIMEFRAME_MAP[timeframe]
        
        # Ensure symbol is ready; remember original visibility to restore later
        _info_before = get_symbol_info_cached(symbol)
        _was_visible = bool(_info_before.visible) if _info_before is not None else None
        err = _ensure_symbol_ready(symbol)
        if err:
            return {"error": err}
        
        try:
            # Normalize TI spec from structured list, JSON string, or compact string for internal processing
            ti_spec = None
            if ti is not None:
                source = ti
                # Accept JSON string input for robustness
                if isinstance(source, str):
                    s = source.strip()
                    if (s.startswith('[') and s.endswith(']')) or (s.startswith('{') and s.endswith('}')):
                        try:
                            source = json.loads(s)
                        except (json.JSONDecodeError, TypeError, ValueError):
                            source = ti  # leave as original string if parse fails
                if isinstance(source, (list, tuple)):
                    parts = []
                    for item in source:
                        if isinstance(item, dict) and 'name' in item:
                            nm = str(item.get('name'))
                            params = item.get('params') or []
                            if isinstance(params, (list, tuple)) and len(params) > 0:
                                args_str = ",".join(str(_coerce_scalar(str(p))) for p in params)
                                parts.append(f"{nm}({args_str})")
                            else:
                                parts.append(nm)
                        else:
                            parts.append(str(item))
                    ti_spec = ",".join(parts)
                else:
                    # Already a compact indicator string like "rsi(14),ema(20)"
                    ti_spec = str(source)
            # Determine warmup bars if technical indicators requested
            warmup_bars = _estimate_warmup_bars_util(ti_spec)

            rates, rates_error = _fetch_rates_with_warmup(
                symbol,
                mt5_timeframe,
                timeframe,
                candles,
                warmup_bars,
                start_datetime,
                end_datetime,
                retry=True,
                sanity_check=True,
            )
            if rates_error:
                return {"error": rates_error}
        finally:
            # Restore original visibility if we changed it
            if _was_visible is False:
                try:
                    mt5.symbol_select(symbol, False)
                except Exception as ex:
                    logger.debug("Failed to restore symbol visibility for %s: %s", symbol, ex)
        
        if rates is None:
            return {"error": f"Failed to get rates for {symbol}: {mt5.last_error()}"}

        # Generate tabular format with dynamic column filtering
        if len(rates) == 0:
            return {"error": "No data available"}
        
        # Check which optional columns have meaningful data (at least one non-zero/different value)
        tick_volumes = [int(rate["tick_volume"]) for rate in rates]
        real_volumes = [int(rate["real_volume"]) for rate in rates]
        
        has_tick_volume = len(set(tick_volumes)) > 1 or any(v != 0 for v in tick_volumes)
        has_real_volume = len(set(real_volumes)) > 1 or any(v != 0 for v in real_volumes)
        
        # Determine requested columns (O,H,L,C,V) if provided
        requested: Optional[set] = _normalize_ohlcv_arg(ohlcv)
        
        # Build header dynamically
        headers = ["time"]
        if requested is not None:
            # Include only requested subset
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
        else:
            # Default: OHLC always; include extras if meaningful
            headers.extend(["open", "high", "low", "close"])
            if has_tick_volume:
                headers.append("tick_volume")
            if has_real_volume:
                headers.append("real_volume")
        
        # Construct DataFrame to support indicators and consistent output
        client_tz = _resolve_client_tz()
        _use_ctz = client_tz is not None
        df = _build_rates_df(rates, _use_ctz)

        # Track denoise metadata if applied
        denoise_apps: List[Dict[str, Any]] = []
        # Optional pre-TI denoising (in-place by default)
        if denoise:
            _dn_pre = _normalize_denoise_spec(denoise, default_when='pre_ti')
            added_dn_pre: List[str] = []
            if _dn_pre and str(_dn_pre.get('when', 'pre_ti')).lower() == 'pre_ti':
                added_dn_pre = _apply_denoise_util(df, _dn_pre, default_when='pre_ti')
                for c in added_dn_pre:
                    if c not in headers:
                        headers.append(c)
            try:
                dn = dict(denoise)
                denoise_apps.append({
                    'method': str(dn.get('method','none')).lower(),
                    'when': str(dn.get('when','pre_ti')).lower(),
                    'causality': str(dn.get('causality', 'causal')),
                    'keep_original': bool(dn.get('keep_original', False)),
                    'columns': dn.get('columns','close'),
                    'params': dn.get('params') or {},
                    'added_columns': added_dn_pre,
                })
            except Exception:
                pass

        # Apply technical indicators if requested (dynamic)
        ti_cols: List[str] = []
        if ti_spec:
            ti_cols = _apply_ta_indicators_util(df, ti_spec)
            headers.extend([c for c in ti_cols if c not in headers])
            # Optional: denoise TI columns as well when requested
            if denoise and ti_cols:
                dn_base = _normalize_denoise_spec(denoise, default_when='post_ti')
                if dn_base and bool(dn_base.get('apply_to_ti') or dn_base.get('ti')):
                    dn_ti = dict(dn_base)
                    dn_ti['columns'] = list(ti_cols)
                    dn_ti.setdefault('when', 'post_ti')
                    dn_ti.setdefault('keep_original', False)
                    _apply_denoise_util(df, dn_ti, default_when='post_ti')

        # Build final header list when not using OHLCV subset
        if requested is None:
            # headers already includes OHLC and optional extras
            pass

        # Filter out warmup region to return the intended target window only
        df = _trim_df_to_target(df, start_datetime, end_datetime, candles, copy_rows=True)

        # If TI requested, check for NaNs and retry once with increased warmup
        if ti_spec and ti_cols:
            try:
                if df[ti_cols].isna().any().any():
                    # Increase warmup and refetch once
                    warmup_bars_retry = max(int(warmup_bars * TI_NAN_WARMUP_FACTOR), warmup_bars + TI_NAN_WARMUP_MIN_ADD)
                    rates_retry, _ = _fetch_rates_with_warmup(
                        symbol,
                        mt5_timeframe,
                        timeframe,
                        candles,
                        warmup_bars_retry,
                        start_datetime,
                        end_datetime,
                        retry=False,
                        sanity_check=False,
                    )
                    # Rebuild df and indicators with the larger window
                    if rates_retry is not None and len(rates_retry) > 0:
                        df = _build_rates_df(rates_retry, _use_ctz)
                        # Optional pre-TI denoising on retried window
                        if denoise:
                            _dn_pre2 = _normalize_denoise_spec(denoise, default_when='pre_ti')
                            if _dn_pre2 and str(_dn_pre2.get('when', 'pre_ti')).lower() == 'pre_ti':
                                _apply_denoise_util(df, _dn_pre2, default_when='pre_ti')
                        # Re-apply indicators and re-extend headers
                        ti_cols = _apply_ta_indicators_util(df, ti_spec)
                        headers.extend([c for c in ti_cols if c not in headers])
                        # Optional: denoise TI columns on retried window
                        if denoise and ti_cols:
                            dn_base2 = _normalize_denoise_spec(denoise, default_when='post_ti')
                            if dn_base2 and bool(dn_base2.get('apply_to_ti') or dn_base2.get('ti')):
                                dn_ti2 = dict(dn_base2)
                                dn_ti2['columns'] = list(ti_cols)
                                dn_ti2.setdefault('when', 'post_ti')
                                dn_ti2.setdefault('keep_original', False)
                                _apply_denoise_util(df, dn_ti2, default_when='post_ti')
                        # Re-trim to target window
                        df = _trim_df_to_target(df, start_datetime, end_datetime, candles, copy_rows=False)
            except Exception:
                pass

        # Optional post-TI denoising (adds new columns by default)
        if denoise:
            _dn_post = _normalize_denoise_spec(denoise, default_when='post_ti')
            added_dn = []
            if _dn_post and str(_dn_post.get('when', 'post_ti')).lower() == 'post_ti':
                added_dn = _apply_denoise_util(df, _dn_post, default_when='post_ti')
            for c in added_dn:
                if c not in headers:
                    headers.append(c)
            try:
                dn = _dn_post or {}
                denoise_apps.append({
                    'method': str(dn.get('method','none')).lower(),
                    'when': 'post_ti',
                    'causality': str(dn.get('causality', 'zero_phase')),
                    'keep_original': bool(dn.get('keep_original', True)),
                    'columns': dn.get('columns','close'),
                    'params': dn.get('params') or {},
                    'added_columns': added_dn,
                })
            except Exception:
                pass

        # Ensure headers are unique and exist in df
        headers = [h for h in headers if h in df.columns]

        # Reformat time consistently across rows for display
        if 'time' in headers and len(df) > 0:
            epochs_list = df['__epoch'].tolist()
            fmt = _time_format_from_epochs(epochs_list)
            fmt = _maybe_strip_year(fmt, epochs_list)
            fmt = _style_time_format(fmt)
            tz_used_name = 'UTC'
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if _use_ctz:
                    tz_used_name = getattr(client_tz, 'zone', None) or str(client_tz)
                    df['time'] = [
                        datetime.fromtimestamp(t, tz=dt_timezone.utc).astimezone(client_tz).strftime(fmt)
                        for t in epochs_list
                    ]
                else:
                    df['time'] = [
                        datetime.utcfromtimestamp(t).strftime(fmt)
                        for t in epochs_list
                    ]
            df.__dict__['_tz_used_name'] = tz_used_name

        # Optionally reduce number of rows for readability/output size
        original_rows = len(df)
        simplify_eff = None
        if simplify is not None:
            simplify_eff = dict(simplify)
            # Default mode
            simplify_eff['mode'] = str(simplify_eff.get('mode', SIMPLIFY_DEFAULT_MODE)).lower().strip()
            # If no explicit points/ratio provided, default to 10% of requested limit
            has_points = any(k in simplify_eff and simplify_eff[k] is not None for k in ("points","target_points","max_points","ratio"))
            if not has_points:
                try:
                    default_pts = max(3, int(round(int(limit) * SIMPLIFY_DEFAULT_POINTS_RATIO_FROM_LIMIT)))
                except Exception:
                    default_pts = max(3, int(round(original_rows * SIMPLIFY_DEFAULT_POINTS_RATIO_FROM_LIMIT)))
                simplify_eff['points'] = default_pts
        df, simplify_meta = _simplify_dataframe_rows_ext(df, headers, simplify_eff if simplify_eff is not None else simplify)
        # If simplify changed representation, respect returned headers
        if simplify_meta is not None and 'headers' in simplify_meta and isinstance(simplify_meta['headers'], list):
            headers = [h for h in simplify_meta['headers'] if isinstance(h, str)]

        # Assemble rows from (possibly reduced) DataFrame for selected headers
        rows = _format_numeric_rows_from_df(df, headers)

        # Build tabular payload
        payload = _table_from_rows(headers, rows)
        
        # Determine if the last candle is open or closed
        last_candle_open = False
        if len(df) > 0 and '__epoch' in df.columns:
            last_epoch = float(df['__epoch'].iloc[-1])
            seconds_per_bar = TIMEFRAME_SECONDS.get(timeframe, 3600)
            current_time = datetime.utcnow().timestamp()
            
            # A candle is "open" if current time is within its timeframe window
            time_since_candle_start = current_time - last_epoch
            last_candle_open = 0 <= time_since_candle_start < seconds_per_bar
        
        payload.update({
            "success": True,
            "symbol": symbol,
            "timeframe": timeframe,
            "candles": len(df),
            "last_candle_open": last_candle_open,
        })
        if not _use_ctz:
            payload["timezone"] = "UTC"
        if simplify_meta is not None:
            payload["simplified"] = True
            payload["simplify"] = simplify_meta
            payload["simplify"]["timeframe"] = timeframe
            payload["simplify"]["original_candles"] = original_rows
        # Attach denoise applications metadata if any
        if denoise_apps:
            payload['denoise'] = {
                'applied': True,
                'applications': denoise_apps,
            }
        return payload
    except Exception as e:
        return {"error": f"Error getting rates: {str(e)}"}

def fetch_ticks(
    symbol: str,
    limit: int = DEFAULT_ROW_LIMIT,
    start: Optional[str] = None,
    end: Optional[str] = None,
    simplify: Optional[SimplifySpec] = None,
    output: Literal["summary", "stats", "rows"] = "summary",
) -> Dict[str, Any]:
    """Fetch tick data and return either a summary (default) or raw rows.

    Parameters
    ----------
    output : {"summary","stats","rows"}
        - "summary" (default): compact descriptive statistics over the fetched
          ticks (bid/ask/mid, plus last and volume; volume uses real volume when
          available, otherwise tick_volume).
        - "stats": more detailed stats (includes extra distribution moments and
          quantiles).
        - "rows": return tick rows as structured data.
    """
    try:
        # Ensure symbol is ready; remember original visibility to restore later
        _info_before = get_symbol_info_cached(symbol)
        _was_visible = bool(_info_before.visible) if _info_before is not None else None
        err = _ensure_symbol_ready(symbol)
        if err:
            return {"error": err}
        
        try:
            # Normalized params only
            effective_limit = int(limit)
            output_mode = str(output or "summary").lower().strip()
            if start:
                from_date = _parse_start_datetime(start)
                if not from_date:
                    return {"error": "Invalid date format. Try examples like '2025-08-29', '2025-08-29 14:30', 'yesterday 14:00', '2 days ago'."}
                if end:
                    to_date = _parse_start_datetime(end)
                    if not to_date:
                        return {"error": "Invalid 'end' date format. Try '2025-08-29 14:30' or 'yesterday 18:00'."}
                    ticks = None
                    for _ in range(FETCH_RETRY_ATTEMPTS):
                        ticks = _mt5_copy_ticks_range(symbol, from_date, to_date, mt5.COPY_TICKS_ALL)
                        if ticks is not None and len(ticks) > 0:
                            break
                        time.sleep(FETCH_RETRY_DELAY)
                    if ticks is not None and effective_limit and len(ticks) > effective_limit:
                        ticks = ticks[-effective_limit:]
                else:
                    ticks = None
                    for _ in range(FETCH_RETRY_ATTEMPTS):
                        ticks = _mt5_copy_ticks_from(symbol, from_date, effective_limit, mt5.COPY_TICKS_ALL)
                        if ticks is not None and len(ticks) > 0:
                            break
                        time.sleep(FETCH_RETRY_DELAY)
            else:
                # Get recent ticks from current time (now)
                to_date = datetime.utcnow()
                from_date = to_date - timedelta(days=TICKS_LOOKBACK_DAYS)  # look back a configurable window
                ticks = None
                for _ in range(FETCH_RETRY_ATTEMPTS):
                    ticks = _mt5_copy_ticks_range(symbol, from_date, to_date, mt5.COPY_TICKS_ALL)
                    if ticks is not None and len(ticks) > 0:
                        break
                    time.sleep(FETCH_RETRY_DELAY)
                if ticks is not None and effective_limit and len(ticks) > effective_limit:
                    ticks = ticks[-effective_limit:]  # Get the last ticks
        finally:
            # Restore original visibility if we changed it
            if _was_visible is False:
                try:
                    mt5.symbol_select(symbol, False)
                except Exception:
                    pass
        
        if ticks is None:
            return {"error": f"Failed to get ticks for {symbol}: {mt5.last_error()}"}
        
        # Generate tabular format with dynamic column filtering
        if len(ticks) == 0:
            return {"error": "No tick data available"}

        if output_mode not in ("summary", "stats", "rows", "raw", "ticks"):
            return {"error": f"Invalid output mode: {output}. Use 'summary', 'stats', or 'rows'."}

        # Check which optional columns have meaningful data (for row output)
        lasts = [float(tick["last"]) for tick in ticks]
        flags = [int(tick["flags"]) for tick in ticks]
        volume_field_exists = True
        try:
            volumes = [float(tick["volume"]) for tick in ticks]
        except Exception:
            volumes = []
            volume_field_exists = False

        volume_real_field_exists = True
        try:
            volumes_real = [float(tick["volume_real"]) for tick in ticks]
        except Exception:
            volumes_real = []
            volume_real_field_exists = False

        has_last = len(set(lasts)) > 1 or any(v != 0 for v in lasts)
        has_volume = volume_field_exists and (len(set(volumes)) > 1 or any(v != 0 for v in volumes))
        has_flags = len(set(flags)) > 1 or any(v != 0 for v in flags)
        has_real_volume = volume_real_field_exists and any(math.isfinite(v) and v != 0.0 for v in volumes_real)
        
        # Build header dynamically (time, bid, ask are always included)
        headers = ["time", "bid", "ask"]
        if has_last:
            headers.append("last")
        if has_volume:
            headers.append("volume")
        if has_flags:
            headers.append("flags")
        
        # Build data rows with matching columns and escape properly
        # Choose a consistent time format for all rows (strip year if constant)
        # Normalize tick times to UTC
        _epochs = [_mt5_epoch_to_utc(float(t["time"])) for t in ticks]
        client_tz = _resolve_client_tz()
        _use_ctz = client_tz is not None
        if not _use_ctz:
            fmt = _time_format_from_epochs(_epochs)
            fmt = _maybe_strip_year(fmt, _epochs)
            fmt = _style_time_format(fmt)
        # Build a DataFrame of ticks to support non-select simplify modes
        def _tick_field(t, name: str):
            try:
                # numpy.void structured array element
                return t[name]
            except Exception:
                pass
            try:
                # namedtuple-like from symbol_info_tick
                return getattr(t, name)
            except Exception:
                pass
            try:
                # dict-like
                return t.get(name)
            except Exception:
                return None

        df_ticks = pd.DataFrame({
            "__epoch": _epochs,
            "bid": [float(_tick_field(t, "bid")) for t in ticks],
            "ask": [float(_tick_field(t, "ask")) for t in ticks],
        })
        if has_last:
            df_ticks["last"] = [float(_tick_field(t, "last")) for t in ticks]
        if has_volume:
            df_ticks["volume"] = [float(_tick_field(t, "volume")) for t in ticks]
        if has_flags:
            df_ticks["flags"] = [int(_tick_field(t, "flags")) for t in ticks]
        # Add display time column
        if _use_ctz:
            df_ticks["time"] = [
                _format_time_minimal_local(e) for e in _epochs
            ]
        else:
            df_ticks["time"] = [
                datetime.utcfromtimestamp(e).strftime(fmt) for e in _epochs     
            ]

        if output_mode in ("summary", "stats"):
            detailed_stats = output_mode == "stats"

            def _series_stats(s: pd.Series, *, total_count: int) -> Dict[str, Any]:
                vals = pd.to_numeric(s, errors="coerce")
                vals = vals[pd.notna(vals)].astype(float)
                n = int(vals.shape[0])
                if n <= 0:
                    return {"available": False}
                first = float(vals.iloc[0])
                last = float(vals.iloc[-1])
                low = float(vals.min())
                high = float(vals.max())
                mean = float(vals.mean())
                std = float(vals.std(ddof=0)) if n > 0 else float("nan")
                stderr = float(std / math.sqrt(n)) if n > 0 else float("nan")
                kurtosis = float(vals.kurtosis()) if n >= 4 else float("nan")
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

            start_epoch = float(df_stats["__epoch"].iloc[0])
            end_epoch = float(df_stats["__epoch"].iloc[-1])
            duration_seconds = float(max(0.0, end_epoch - start_epoch))
            tick_rate_per_second = (
                float(len(df_stats) / duration_seconds) if duration_seconds > 0 else float("nan")
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
                "start_epoch": start_epoch,
                "end_epoch": end_epoch,
                "duration_seconds": duration_seconds,
                "tick_rate_per_second": tick_rate_per_second,
                "timezone": timezone,
                "stats": {
                    "bid": _series_stats(df_stats["bid"], total_count=len(df_stats)),
                    "ask": _series_stats(df_stats["ask"], total_count=len(df_stats)),
                    "mid": _series_stats(df_stats["mid"], total_count=len(df_stats)),
                },
            }
            if has_last:
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
                        float(vol_sum / duration_seconds) if duration_seconds > 0 else float("nan")
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
                out["stats"]["volume"] = vol_out
            else:
                out["stats"]["volume"] = (
                    {
                        "kind": volume_kind,
                        "per_second": tick_rate_per_second,
                        "sum": int(len(df_stats)),
                    }
                    if detailed_stats
                    else {"kind": volume_kind}
                )
            return out

        # If simplify mode requests approximation or resampling, use shared path
        original_count = len(df_ticks)
        simplify_eff = None
        if simplify is not None:
            simplify_eff = dict(simplify)
            simplify_eff['mode'] = str(simplify_eff.get('mode', SIMPLIFY_DEFAULT_MODE)).lower().strip()
            has_points = any(k in simplify_eff and simplify_eff[k] is not None for k in ("points","target_points","max_points","ratio"))
            if not has_points:
                try:
                    default_pts = max(3, int(round(int(limit) * SIMPLIFY_DEFAULT_POINTS_RATIO_FROM_LIMIT)))
                except Exception:
                    default_pts = max(3, int(round(original_count * SIMPLIFY_DEFAULT_POINTS_RATIO_FROM_LIMIT)))
                simplify_eff['points'] = default_pts
        simplify_present = (simplify_eff is not None) or (simplify is not None)
        simplify_used = simplify_eff if simplify_eff is not None else simplify
        _mode = str((simplify_used or {}).get('mode', SIMPLIFY_DEFAULT_MODE)).lower().strip() if simplify_present else SIMPLIFY_DEFAULT_MODE
        if simplify_present and _mode in ('approximate', 'resample'):
            df_out, simplify_meta = _simplify_dataframe_rows_ext(df_ticks, headers, simplify_used)
            rows = _format_numeric_rows_from_df(df_out, headers)
            payload = _table_from_rows(headers, rows)
            payload.update({
                "success": True,
                "symbol": symbol,
                "count": len(rows),
            })
            if not _use_ctz:
                payload["timezone"] = "UTC"
            if simplify_meta is not None and original_count > len(rows):
                payload["simplified"] = True
                meta = dict(simplify_meta)
                meta["columns"] = [c for c in ["bid","ask"] + (["last"] if has_last else []) + (["volume"] if has_volume else [])]
                payload["simplify"] = meta
            return payload
        # Optional simplification based on a chosen y-series
        original_count = len(ticks)
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
                for c in cols:
                    series: List[float] = []
                    for t in ticks:
                        v = _tick_field(t, c)
                        try:
                            series.append(float(v))
                        except Exception:
                            series.append(float('nan'))
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
                    vals = []
                    for t in ticks:
                        try:
                            vals.append(float(_tick_field(t, c)))
                        except Exception:
                            vals.append(0.0)
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
                        try:
                            vv = (float(_tick_field(ticks[i], c)) - mins[c]) / ranges[c]
                        except Exception:
                            vv = 0.0
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
            tick = ticks[i]
            if _use_ctz:
                time_str = _format_time_minimal_local(_epochs[i])
            else:
                time_str = datetime.utcfromtimestamp(_epochs[i]).strftime(fmt)
            values = [time_str, str(tick['bid']), str(tick['ask'])]
            if has_last:
                values.append(str(tick['last']))
            if has_volume:
                values.append(str(tick['volume']))
            if has_flags:
                values.append(str(tick['flags']))
            rows.append(values)

        payload = _table_from_rows(headers, rows)
        payload.update({
            "success": True,
            "symbol": symbol,
            "count": len(rows),
        })
        if not _use_ctz:
            payload["timezone"] = "UTC"
        if simplify_present and original_count > len(rows):
            payload["simplified"] = True
            meta = {
                "method": (_simp_method_used or str((simplify_used or {}).get('method', SIMPLIFY_DEFAULT_METHOD)).lower()),
                "original_rows": original_count,
                "returned_rows": len(rows),
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
            # Normalize points to actual returned rows
            meta["points"] = len(rows)
            payload["simplify"] = meta
        return payload
    except Exception as e:
        return {"error": f"Error getting ticks: {str(e)}"}
