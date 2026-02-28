from datetime import datetime, timezone
import logging
from typing import Any, Dict, List, Optional, Tuple
import warnings

import numpy as np
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
_CANDLESTICK_PATTERN_METHOD_CACHE: Optional[Tuple[str, ...]] = None


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


def _discover_candlestick_pattern_methods(ta_accessor: Any) -> Tuple[str, ...]:
    methods: List[str] = []
    for attr in dir(ta_accessor):
        if not attr.startswith('cdl_'):
            continue
        func = getattr(ta_accessor, attr, None)
        if callable(func):
            methods.append(attr)
    return tuple(sorted(methods))


def _get_candlestick_pattern_methods(temp: pd.DataFrame) -> List[str]:
    global _CANDLESTICK_PATTERN_METHOD_CACHE

    if _CANDLESTICK_PATTERN_METHOD_CACHE is None:
        try:
            _CANDLESTICK_PATTERN_METHOD_CACHE = _discover_candlestick_pattern_methods(temp.ta)
        except Exception:
            logger.warning("Failed to enumerate candlestick pattern detectors from pandas_ta.", exc_info=True)
            return []
    return list(_CANDLESTICK_PATTERN_METHOD_CACHE)


def _extract_candlestick_rows(
    df_tail: pd.DataFrame,
    temp_tail: pd.DataFrame,
    pattern_cols: List[str],
    *,
    threshold: float,
    robust_only: bool,
    robust_set: set[str],
    whitelist_set: Optional[set[str]],
    min_gap: int,
    top_k: int,
    deprioritize: set[str],
) -> List[List[Any]]:
    if not pattern_cols:
        return []

    base_names = np.asarray(
        [col[len('cdl_'):] if col.lower().startswith('cdl_') else col for col in pattern_cols],
        dtype=object,
    )
    normalized_names = np.asarray([_normalize_candlestick_name(name) for name in base_names], dtype=object)
    allowed_mask = np.asarray(
        [
            _is_candlestick_allowed(
                str(name),
                robust_only=bool(robust_only),
                robust_set=robust_set,
                whitelist_set=whitelist_set,
            )
            for name in base_names
        ],
        dtype=bool,
    )
    if not bool(np.any(allowed_mask)):
        return []

    try:
        values = (
            temp_tail.loc[:, pattern_cols]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(dtype=float, copy=False)
        )
    except Exception:
        values = temp_tail.loc[:, pattern_cols].to_numpy(dtype=float, copy=True)

    active_mask = np.isfinite(values) & (np.abs(values) >= float(threshold) * 100.0)
    active_mask &= allowed_mask[np.newaxis, :]
    if not bool(np.any(active_mask)):
        return []

    rows: List[List[Any]] = []
    gap = max(0, int(min_gap))
    k = max(1, int(top_k))
    last_pick_idx = -10**9
    non_dep_mask = np.asarray([str(name) not in deprioritize for name in normalized_names], dtype=bool)

    if 'time' in df_tail.columns:
        time_vals = df_tail['time'].astype(str).to_numpy(dtype=object, copy=False)
    else:
        time_vals = np.full(len(df_tail), "", dtype=object)

    candidate_rows = np.flatnonzero(np.any(active_mask, axis=1))
    for i in candidate_rows.tolist():
        if i - last_pick_idx < gap:
            continue
        hit_idx = np.flatnonzero(active_mask[i])
        if hit_idx.size == 0:
            continue
        pool_idx = hit_idx[non_dep_mask[hit_idx]]
        if pool_idx.size == 0:
            pool_idx = hit_idx
        order = np.argsort(-np.abs(values[i, pool_idx]), kind="mergesort")[:k]
        chosen_idx = pool_idx[order]
        t_val = str(time_vals[i])
        for col_idx in chosen_idx.tolist():
            name = str(base_names[col_idx])
            value = float(values[i, col_idx])
            label_core = name.replace('_', ' ').strip().upper()
            dir_title = 'Bullish' if value > 0 else 'Bearish'
            rows.append([t_val, f"{dir_title} {label_core}" if label_core else dir_title])
        last_pick_idx = i
    return rows


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

    pattern_methods = _get_candlestick_pattern_methods(temp)
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
    try:
        k = max(1, int(top_k))
    except Exception:
        k = 1
    _deprioritize = {
        'shortline', 'longline', 'spinningtop', 'highwave',
        'marubozu', 'closingmarubozu', 'doji', 'gravestonedoji', 'longleggeddoji', 'rickshawman'
    }

    rows = _extract_candlestick_rows(
        df,
        temp,
        pattern_cols,
        threshold=thr,
        robust_only=bool(robust_only),
        robust_set=_robust_whitelist,
        whitelist_set=parsed_whitelist,
        min_gap=gap,
        top_k=k,
        deprioritize=_deprioritize,
    )

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
