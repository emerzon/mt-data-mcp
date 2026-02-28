from datetime import datetime, timezone
import logging
from typing import Any, Dict, List, Optional, Tuple
import warnings

import pandas as pd
from ..utils.utils import (
    _table_from_rows,
    _format_time_minimal_local,
    _use_client_tz,
    _time_format_from_epochs,
    _maybe_strip_year,
    _style_time_format,
)

logger = logging.getLogger(__name__)
ta: Any = None
mt5: Any = None
TIMEFRAME_MAP: Optional[Dict[str, Any]] = None
_mt5_copy_rates_from: Any = None
_rates_to_df: Any = None
_symbol_ready_guard: Any = None


def _normalize_candlestick_name(pattern_name: str) -> str:
    nm = str(pattern_name).strip()
    if nm.lower().startswith('cdl_'):
        nm = nm[len('cdl_'):]
    return nm.replace('_', '').replace(' ', '').lower()


def _parse_min_strength(min_strength: float) -> float:
    try:
        thr = float(min_strength)
    except (TypeError, ValueError) as exc:
        raise ValueError("min_strength must be a float between 0.0 and 1.0.") from exc
    if not (0.0 <= thr <= 1.0):
        raise ValueError("min_strength must be between 0.0 and 1.0.")
    return thr


def _ensure_candlestick_runtime() -> None:
    global ta, mt5, TIMEFRAME_MAP, _mt5_copy_rates_from, _rates_to_df, _symbol_ready_guard

    if ta is None:
        try:
            import pandas_ta as ta_mod  # type: ignore
        except ModuleNotFoundError:
            try:
                import pandas_ta_classic as ta_mod  # type: ignore
            except ModuleNotFoundError as e:
                raise ModuleNotFoundError(
                    "pandas_ta not found. Install 'pandas-ta-classic' (or 'pandas-ta')."
                ) from e
        ta = ta_mod

    if mt5 is None:
        try:
            import MetaTrader5 as mt5_mod  # type: ignore
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "MetaTrader5 not found. Install 'MetaTrader5' to use candlestick detection."
            ) from e
        mt5 = mt5_mod

    if TIMEFRAME_MAP is None:
        from ..core.constants import TIMEFRAME_MAP as timeframe_map

        TIMEFRAME_MAP = timeframe_map

    if _mt5_copy_rates_from is None or _rates_to_df is None or _symbol_ready_guard is None:
        from ..utils.mt5 import (
            _mt5_copy_rates_from as copy_rates_from,
            _rates_to_df as rates_to_df,
            _symbol_ready_guard as symbol_ready_guard,
        )

        _mt5_copy_rates_from = copy_rates_from
        _rates_to_df = rates_to_df
        _symbol_ready_guard = symbol_ready_guard


def _is_candlestick_allowed(
    pattern_name: str,
    *,
    robust_only: bool,
    robust_set: set[str],
    whitelist_set: Optional[set[str]],
) -> bool:
    nm = _normalize_candlestick_name(pattern_name)
    if whitelist_set is not None and nm not in whitelist_set:
        return False
    if robust_only and nm not in robust_set:
        return False
    return True


def detect_candlestick_patterns(
    *,
    symbol: str,
    timeframe: str,
    limit: int,
    min_strength: float,
    min_gap: int,
    robust_only: bool,
    whitelist: Optional[str],
    top_k: int,
) -> Dict[str, Any]:
    try:
        _ensure_candlestick_runtime()
    except ModuleNotFoundError as exc:
        return {"error": str(exc)}
    if timeframe not in TIMEFRAME_MAP:
        return {"error": f"Invalid timeframe: {timeframe}. Valid options: {list(TIMEFRAME_MAP.keys())}"}
    try:
        thr = _parse_min_strength(min_strength)
    except ValueError as exc:
        return {"error": str(exc)}

    mt5_timeframe = TIMEFRAME_MAP[timeframe]

    with _symbol_ready_guard(symbol) as (err, _info):
        if err:
            return {"error": err}
        utc_now = datetime.now(timezone.utc)
        rates = _mt5_copy_rates_from(symbol, mt5_timeframe, utc_now, limit)

    if rates is None:
        return {"error": f"Failed to get rates for {symbol}: {mt5.last_error()}"}
    if len(rates) == 0:
        return {"error": "No candle data available"}

    df = _rates_to_df(rates)
    epochs = [float(t) for t in df['time'].tolist()] if 'time' in df.columns else []
    _use_ctz = _use_client_tz()
    if _use_ctz:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df['time'] = df['time'].apply(_format_time_minimal_local)
    else:
        time_fmt = _time_format_from_epochs(epochs) if epochs else "%Y-%m-%d %H:%M"
        time_fmt = _maybe_strip_year(time_fmt, epochs)
        time_fmt = _style_time_format(time_fmt)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df['time'] = df['time'].apply(
                lambda t: datetime.fromtimestamp(float(t), tz=timezone.utc).strftime(time_fmt)
            )

    for col in ['open', 'high', 'low', 'close']:
        if col not in df.columns:
            return {"error": f"Missing '{col}' data from rates"}

    try:
        temp = df.copy()
        temp['__epoch'] = [float(e) for e in epochs]
        temp.index = pd.to_datetime(temp['__epoch'], unit='s')
    except Exception:
        temp = df.copy()

    pattern_methods: List[str] = []
    try:
        for attr in dir(temp.ta):
            if not attr.startswith('cdl_'):
                continue
            func = getattr(temp.ta, attr, None)
            if callable(func):
                pattern_methods.append(attr)
    except Exception:
        logger.warning("Failed to enumerate candlestick pattern detectors from pandas_ta.", exc_info=True)

    if not pattern_methods:
        return {"error": "No candlestick pattern detectors (cdl_*) found in pandas_ta."}

    before_cols = set(temp.columns)
    for name in sorted(pattern_methods):
        try:
            method = getattr(temp.ta, name)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                method(append=True)
        except Exception:
            logger.warning("Candlestick pattern detector '%s' failed.", name, exc_info=True)
            continue

    pattern_cols = [c for c in temp.columns if c not in before_cols and c.lower().startswith('cdl_')]
    if not pattern_cols:
        return {"error": "No candle patterns produced any outputs."}

    rows: List[List[Any]] = []
    _robust_whitelist = {
        'engulfing', 'harami', '3inside', '3outside', 'eveningstar', 'morningstar',
        'darkcloudcover', 'piercing', 'inside', 'outside', 'hikkake'
    }
    parsed_whitelist: Optional[set[str]] = None
    if whitelist and isinstance(whitelist, str):
        try:
            parts = [p.strip() for p in whitelist.split(',') if p.strip()]
            if parts:
                parsed_whitelist = {_normalize_candlestick_name(p) for p in parts}
        except Exception:
            pass

    try:
        gap = max(0, int(min_gap))
    except Exception:
        gap = 3
    last_pick_idx = -10**9
    _deprioritize = {
        'shortline', 'longline', 'spinningtop', 'highwave',
        'marubozu', 'closingmarubozu', 'doji', 'gravestonedoji', 'longleggeddoji', 'rickshawman'
    }

    df_tail = df
    temp_tail = temp

    for i in range(len(temp_tail)):
        hits: List[Tuple[str, float]] = []
        for col in pattern_cols:
            try:
                val = float(temp_tail.iloc[i][col])
            except Exception:
                continue
            if abs(val) >= (thr * 100.0):
                name = col
                if name.lower().startswith('cdl_'):
                    name = name[len('cdl_'):]
                if _is_candlestick_allowed(
                    name,
                    robust_only=bool(robust_only),
                    robust_set=_robust_whitelist,
                    whitelist_set=parsed_whitelist,
                ):
                    hits.append((name, val))
        if not hits:
            continue
        if i - last_pick_idx < gap:
            continue
        non_dep = [(n, v) for (n, v) in hits if _normalize_candlestick_name(n) not in _deprioritize]
        pool = non_dep if non_dep else hits
        try:
            k = max(1, int(top_k))
        except Exception:
            k = 1
        picks = sorted(pool, key=lambda x: abs(x[1]), reverse=True)[:k]
        t_val = str(df_tail.iloc[i].get('time')) if 'time' in df_tail.columns else ''
        for name, value in picks:
            label_core = name.replace('_', ' ').strip().upper()
            dir_title = 'Bullish' if value > 0 else 'Bearish'
            rows.append([t_val, f"{dir_title} {label_core}" if label_core else dir_title])
        last_pick_idx = i

    headers = ["time", "pattern"]
    payload = _table_from_rows(headers, rows)
    payload.update({
        "success": True,
        "symbol": symbol,
        "timeframe": timeframe,
        "candles": int(limit),
        "mode": "candlestick",
    })
    if not _use_ctz:
        payload["timezone"] = "UTC"
    return payload
