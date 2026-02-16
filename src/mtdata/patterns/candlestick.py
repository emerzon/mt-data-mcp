from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import warnings

import pandas as pd
import MetaTrader5 as mt5

from ..core.constants import TIMEFRAME_MAP
from ..utils.mt5 import _mt5_copy_rates_from, _rates_to_df, _symbol_ready_guard
from ..utils.utils import (
    _table_from_rows,
    _format_time_minimal_local,
    _use_client_tz,
    _time_format_from_epochs,
    _maybe_strip_year,
    _style_time_format,
)


def _is_candlestick_allowed(
    pattern_name: str,
    *,
    robust_only: bool,
    robust_set: set[str],
    whitelist_set: Optional[set[str]],
) -> bool:
    nm = str(pattern_name).replace('_', '').replace(' ', '').lower()
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
        if timeframe not in TIMEFRAME_MAP:
            return {"error": f"Invalid timeframe: {timeframe}. Valid options: {list(TIMEFRAME_MAP.keys())}"}
        mt5_timeframe = TIMEFRAME_MAP[timeframe]

        with _symbol_ready_guard(symbol) as (err, _info):
            if err:
                return {"error": err}
            utc_now = datetime.utcnow()
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
                df['time'] = df['time'].apply(lambda t: datetime.utcfromtimestamp(float(t)).strftime(time_fmt))

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
            pass

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
                continue

        pattern_cols = [c for c in temp.columns if c not in before_cols and c.lower().startswith('cdl_')]
        if not pattern_cols:
            return {"error": "No candle patterns produced any outputs."}

        rows: List[List[Any]] = []
        try:
            thr = float(min_strength)
        except Exception:
            thr = 0.95
        if thr > 1.0:
            thr = thr / 100.0
        thr = max(0.0, min(1.0, thr))

        _robust_whitelist = {
            'engulfing', 'harami', '3inside', '3outside', 'eveningstar', 'morningstar',
            'darkcloudcover', 'piercing', 'inside', 'outside', 'hikkake'
        }
        parsed_whitelist: Optional[set[str]] = None
        if whitelist and isinstance(whitelist, str):
            try:
                parts = [p.strip() for p in whitelist.split(',') if p.strip()]
                if parts:
                    parsed_whitelist = {p.replace('_', '').replace(' ', '').lower() for p in parts}
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
            non_dep = [(n, v) for (n, v) in hits if n.split('_')[0].lower() not in _deprioritize]
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
    except Exception as exc:
        return {"error": f"Error detecting candlestick patterns: {exc}"}
