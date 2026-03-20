from datetime import datetime, timezone
import logging
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
from ..core.patterns_support import (
    _config_bool,
    _config_float,
    _config_int,
    _infer_market_regime,
    _resolve_volume_series,
    _round_value,
    _volume_window_mean,
)
from .common import data_quality_warnings
from ..shared.constants import TIMEFRAME_SECONDS
from ..shared.validators import invalid_timeframe_error
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
_CANDLESTICK_PATTERN_METHOD_CACHE_KEY: Optional[str] = None
_CANDLESTICK_PATTERN_METHOD_CACHE_LOCK = Lock()
_CANDLESTICK_RUNTIME_LOCK = Lock()
_ROBUST_CANDLESTICK_WHITELIST = {
    'engulfing', 'harami', '3inside', '3outside', 'eveningstar', 'morningstar',
    'darkcloudcover', 'piercing', 'inside', 'outside', 'hikkake'
}
_CANDLESTICK_PATTERN_BAR_SPANS = {
    '2crows': 3,
    'counterattack': 2,
    'darkcloudcover': 2,
    'engulfing': 2,
    'harami': 2,
    'haramicross': 2,
    'hikkake': 3,
    'hikkakemod': 3,
    'inside': 2,
    'outside': 2,
    'piercing': 2,
    'tasukigap': 2,
    '3blackcrows': 3,
    '3inside': 3,
    '3outside': 3,
    '3starsinsouth': 3,
    '3whitesoldiers': 3,
    'advanceblock': 3,
    'deliberation': 3,
    'eveningstar': 3,
    'gapsidesidewhite': 3,
    'identical3crows': 3,
    'morningstar': 3,
    'sticksandwich': 3,
    'tristar': 3,
    'unique3river': 3,
    'xsidegap3methods': 3,
    'breakaway': 5,
    'ladderbottom': 5,
    'mathold': 5,
    'risefall3methods': 5,
}
_DEPRIORITIZED_CANDLESTICK_PATTERNS = {
    'shortline', 'longline', 'spinningtop', 'highwave',
    'marubozu', 'closingmarubozu', 'doji', 'gravestonedoji', 'longleggeddoji', 'rickshawman'
}


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


def _candlestick_base_strength(
    pattern_name: str,
    *,
    robust_set: set[str],
    deprioritize: set[str],
) -> float:
    normalized = _normalize_candlestick_name(pattern_name)
    base = 0.75
    if normalized in robust_set:
        base += 0.15
    if normalized in deprioritize:
        base -= 0.20
    return float(max(0.0, min(1.0, base)))


def _candlestick_strength_score(
    pattern_name: str,
    raw_signal: float,
    *,
    robust_set: set[str],
    deprioritize: set[str],
) -> float:
    raw = abs(float(raw_signal))
    if not np.isfinite(raw) or raw <= 0.0:
        return 0.0
    span_bars = _candlestick_span_bars(pattern_name)
    base = _candlestick_base_strength(
        pattern_name,
        robust_set=robust_set,
        deprioritize=deprioritize,
    )
    span_bonus = min(0.10, 0.05 * max(0, span_bars - 1))
    raw_signal_bonus = min(0.20, 0.20 * max(0.0, raw - 100.0) / 100.0)
    return float(max(0.0, min(1.0, base + span_bonus + raw_signal_bonus)))


def _candlestick_span_bars(pattern_name: str) -> int:
    return int(_CANDLESTICK_PATTERN_BAR_SPANS.get(_normalize_candlestick_name(pattern_name), 1))


def _ensure_candlestick_runtime() -> None:
    global ta, mt5, TIMEFRAME_MAP, _mt5_copy_rates_from, _rates_to_df, _symbol_ready_guard

    if (
        ta is not None
        and mt5 is not None
        and TIMEFRAME_MAP is not None
        and _mt5_copy_rates_from is not None
        and _rates_to_df is not None
        and _symbol_ready_guard is not None
    ):
        return

    with _CANDLESTICK_RUNTIME_LOCK:
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
                from ..utils.mt5 import mt5 as mt5_mod
            except ModuleNotFoundError as e:
                raise ModuleNotFoundError(
                    "MetaTrader5 not found. Install 'MetaTrader5' to use candlestick detection."
                ) from e
            mt5 = mt5_mod

        if TIMEFRAME_MAP is None:
            from ..shared.constants import TIMEFRAME_MAP as timeframe_map

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


def _candlestick_accessor_cache_key(ta_accessor: Any) -> str:
    accessor_type = type(ta_accessor)
    return f"{accessor_type.__module__}.{accessor_type.__qualname__}"


def _get_candlestick_pattern_methods(temp: pd.DataFrame) -> List[str]:
    global _CANDLESTICK_PATTERN_METHOD_CACHE, _CANDLESTICK_PATTERN_METHOD_CACHE_KEY

    cache_key = _candlestick_accessor_cache_key(temp.ta)

    if _CANDLESTICK_PATTERN_METHOD_CACHE is not None and _CANDLESTICK_PATTERN_METHOD_CACHE_KEY == cache_key:
        return list(_CANDLESTICK_PATTERN_METHOD_CACHE)

    with _CANDLESTICK_PATTERN_METHOD_CACHE_LOCK:
        if _CANDLESTICK_PATTERN_METHOD_CACHE is not None and _CANDLESTICK_PATTERN_METHOD_CACHE_KEY == cache_key:
            return list(_CANDLESTICK_PATTERN_METHOD_CACHE)
        try:
            _CANDLESTICK_PATTERN_METHOD_CACHE = _discover_candlestick_pattern_methods(temp.ta)
            _CANDLESTICK_PATTERN_METHOD_CACHE_KEY = cache_key
        except Exception:
            logger.warning("Failed to enumerate candlestick pattern detectors from pandas_ta.", exc_info=True)
            _CANDLESTICK_PATTERN_METHOD_CACHE = None
            _CANDLESTICK_PATTERN_METHOD_CACHE_KEY = None
            return []
    return list(_CANDLESTICK_PATTERN_METHOD_CACHE)


def _filter_candlestick_pattern_methods(
    pattern_methods: List[str],
    *,
    robust_only: bool,
    robust_set: set[str],
    whitelist_set: Optional[set[str]],
) -> List[str]:
    return [
        name
        for name in pattern_methods
        if _is_candlestick_allowed(
            str(name),
            robust_only=bool(robust_only),
            robust_set=robust_set,
            whitelist_set=whitelist_set,
        )
    ]


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
    include_metrics: bool = False,
    start_index: int = 0,
) -> List[List[Any]]:
    if not pattern_cols:
        return []

    base_names = np.asarray(
        [col[len('cdl_'):] if col.lower().startswith('cdl_') else col for col in pattern_cols],
        dtype=object,
    )
    normalized_names = np.asarray([_normalize_candlestick_name(name) for name in base_names], dtype=object)
    try:
        values = (
            temp_tail.loc[:, pattern_cols]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(dtype=float, copy=False)
        )
    except Exception:
        values = temp_tail.loc[:, pattern_cols].to_numpy(dtype=float, copy=True)

    strength_values = np.zeros_like(values, dtype=float)
    for col_idx, name in enumerate(base_names.tolist()):
        base_strength = _candlestick_base_strength(
            str(name),
            robust_set=robust_set,
            deprioritize=deprioritize,
        )
        span_bars = _candlestick_span_bars(str(name))
        span_bonus = min(0.10, 0.05 * max(0, span_bars - 1))
        raw_bonus = np.clip((np.abs(values[:, col_idx]) - 100.0) / 100.0, 0.0, 1.0) * 0.20
        strength_values[:, col_idx] = np.clip(base_strength + span_bonus + raw_bonus, 0.0, 1.0)

    active_mask = np.isfinite(values) & (np.abs(values) > 0.0) & (strength_values >= float(threshold))
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
    close_vals: Optional[np.ndarray] = None
    if include_metrics and 'close' in df_tail.columns:
        try:
            close_vals = pd.to_numeric(df_tail['close'], errors="coerce").to_numpy(dtype=float, copy=False)
        except Exception:
            close_vals = None

    start_idx = max(0, int(start_index))
    candidate_rows = np.flatnonzero(np.any(active_mask, axis=1))
    if start_idx > 0:
        candidate_rows = candidate_rows[candidate_rows >= start_idx]
    for i in candidate_rows.tolist():
        if i - last_pick_idx < gap:
            continue
        hit_idx = np.flatnonzero(active_mask[i])
        if hit_idx.size == 0:
            continue
        pool_idx = hit_idx[non_dep_mask[hit_idx]]
        if pool_idx.size == 0:
            pool_idx = hit_idx
        order = np.lexsort(
            (
                -np.abs(values[i, pool_idx]),
                -strength_values[i, pool_idx],
            )
        )[:k]
        chosen_idx = pool_idx[order]
        t_val = str(time_vals[i])
        for col_idx in chosen_idx.tolist():
            name = str(base_names[col_idx])
            value = float(values[i, col_idx])
            label_core = name.replace('_', ' ').strip().upper()
            dir_title = 'Bullish' if value > 0 else 'Bearish'
            if include_metrics:
                span_bars = _candlestick_span_bars(name)
                start_bar_idx = max(0, int(i - span_bars + 1))
                start_time = str(time_vals[start_bar_idx])
                end_time = str(time_vals[i])
                direction = "bullish" if value > 0 else "bearish"
                strength = float(strength_values[i, col_idx])
                raw_signal: Any
                if abs(value - round(value)) <= 1e-9:
                    raw_signal = int(round(value))
                else:
                    raw_signal = float(value)
                price = float(close_vals[i]) if close_vals is not None and np.isfinite(close_vals[i]) else None
                rows.append(
                    [
                        end_time,
                        f"{dir_title} {label_core}" if label_core else dir_title,
                        direction,
                        strength,
                        raw_signal,
                        price,
                        start_time,
                        end_time,
                        int(span_bars),
                        int(start_bar_idx),
                        int(i),
                    ]
                )
            else:
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
    last_n_bars: Optional[int] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    try:
        _ensure_candlestick_runtime()
    except ModuleNotFoundError as exc:
        return {"error": str(exc)}
    if timeframe not in TIMEFRAME_MAP:
        return {"error": invalid_timeframe_error(timeframe, TIMEFRAME_MAP or {})}
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
    warnings_out = data_quality_warnings(
        df,
        timeframe_seconds=float(TIMEFRAME_SECONDS.get(timeframe, 0) or 0),
    )
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

    parsed_whitelist: Optional[set[str]] = None
    if whitelist and isinstance(whitelist, str):
        try:
            parts = [p.strip() for p in whitelist.split(',') if p.strip()]
            if parts:
                parsed_whitelist = {_normalize_candlestick_name(p) for p in parts}
        except Exception:
            pass

    pattern_methods = _filter_candlestick_pattern_methods(
        pattern_methods,
        robust_only=bool(robust_only),
        robust_set=_ROBUST_CANDLESTICK_WHITELIST,
        whitelist_set=parsed_whitelist,
    )
    if not pattern_methods:
        return {"error": "No candlestick detectors match the requested filters."}

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

    try:
        gap = max(0, int(min_gap))
    except Exception:
        gap = 3
    try:
        k = max(1, int(top_k))
    except Exception:
        k = 1
    last_n_val: Optional[int] = None
    if last_n_bars is not None:
        try:
            last_n_val = int(last_n_bars)
        except Exception:
            return {"error": "last_n_bars must be a positive integer."}
        if last_n_val <= 0:
            return {"error": "last_n_bars must be >= 1."}
    start_index = 0
    if last_n_val is not None and len(df) > last_n_val:
        start_index = int(len(df) - last_n_val)
    rows = _extract_candlestick_rows(
        df,
        temp,
        pattern_cols,
        threshold=thr,
        robust_only=bool(robust_only),
        robust_set=_ROBUST_CANDLESTICK_WHITELIST,
        whitelist_set=parsed_whitelist,
        min_gap=gap,
        top_k=k,
        deprioritize=_DEPRIORITIZED_CANDLESTICK_PATTERNS,
        include_metrics=True,
        start_index=start_index,
    )

    headers = [
        "time",
        "pattern",
        "direction",
        "confidence",
        "raw_signal",
        "price",
        "start_time",
        "end_time",
        "n_bars",
        "start_index",
        "end_index",
    ]
    payload = _table_from_rows(headers, rows)
    _enrich_candlestick_payload(payload, df, config)
    payload.update({
        "success": True,
        "symbol": symbol,
        "timeframe": timeframe,
        "candles": int(limit),
        "mode": "candlestick",
        "min_strength": float(thr),
        "strength_scale": "semantic_pattern_conviction_v2",
        "signal_scale": "pandas_ta_signal_x100",
    })
    if warnings_out:
        payload["warnings"] = warnings_out
    if last_n_val is not None:
        payload["last_n_bars"] = int(last_n_val)
    if not _use_ctz:
        payload["timezone"] = "UTC"
    return payload


def _enrich_candlestick_payload(payload: Dict[str, Any], df: pd.DataFrame, config: Optional[Dict[str, Any]]) -> None:
    rows = payload.get("data")
    if not isinstance(rows, list) or not isinstance(df, pd.DataFrame) or len(df) <= 0:
        return
    regime_context = _infer_market_regime(df, config)
    volume, volume_source = _resolve_volume_series(df)
    for row in rows:
        if not isinstance(row, dict):
            continue
        _attach_candlestick_volume_confirmation(row, volume, volume_source, config)
        _attach_candlestick_regime_context(row, regime_context, config)


def _attach_candlestick_volume_confirmation(
    row: Dict[str, Any],
    volume: Optional[np.ndarray],
    volume_source: Optional[str],
    config: Optional[Dict[str, Any]],
) -> None:
    payload: Dict[str, Any] = {
        "mode": "signal_window",
        "status": "disabled" if not _config_bool(config, "use_volume_confirmation", True) else "unavailable",
        "volume_source": volume_source,
    }
    if payload["status"] == "disabled":
        row["volume_confirmation"] = payload
        return
    if volume is None or volume_source is None:
        row["volume_confirmation"] = payload
        return
    try:
        end_index = int(row.get("end_index"))
        start_index = int(row.get("start_index"))
    except Exception:
        row["volume_confirmation"] = payload
        return
    breakout_bars = _config_int(config, "volume_confirm_breakout_bars", 2, minimum=1)
    lookback_bars = _config_int(config, "volume_confirm_lookback_bars", 20, minimum=breakout_bars + 1)
    min_ratio = _config_float(config, "volume_confirm_min_ratio", 1.10, minimum=1.0)
    bonus = _config_float(config, "volume_confirm_bonus", 0.08, minimum=0.0)
    penalty = _config_float(config, "volume_confirm_penalty", 0.06, minimum=0.0)
    signal_end = max(int(start_index), int(end_index))
    pattern_start = min(int(start_index), int(end_index))
    signal_start = max(0, min(pattern_start, int(signal_end - breakout_bars + 1)))
    baseline_end = int(signal_start - 1)
    baseline_start = max(0, int(baseline_end - lookback_bars + 1))
    signal_avg = _volume_window_mean(volume, signal_start, signal_end)
    baseline_avg = _volume_window_mean(volume, baseline_start, baseline_end)
    ratio = (
        float(signal_avg) / float(baseline_avg)
        if signal_avg is not None and baseline_avg is not None and baseline_avg > 0
        else None
    )
    payload["lookback_bars"] = int(lookback_bars)
    payload["breakout_bars"] = int(breakout_bars)
    if baseline_avg is not None:
        payload["baseline_avg_volume"] = _round_value(baseline_avg)
    if signal_avg is not None:
        payload["signal_avg_volume"] = _round_value(signal_avg)
    if ratio is not None and np.isfinite(ratio):
        payload["signal_to_baseline_ratio"] = _round_value(ratio)
    confidence_delta = 0.0
    if ratio is None:
        payload["status"] = "unavailable"
    else:
        reject_ratio = (1.0 / float(min_ratio)) if min_ratio > 0 else 0.0
        if ratio >= float(min_ratio):
            payload["status"] = "confirmed"
            confidence_delta = float(bonus)
        elif ratio <= float(reject_ratio):
            payload["status"] = "rejected"
            confidence_delta = -float(penalty)
        else:
            payload["status"] = "neutral"
    if abs(confidence_delta) > 1e-12:
        payload["confidence_delta"] = _round_value(confidence_delta)
        row["confidence"] = float(max(0.0, min(1.0, float(row.get("confidence", 0.0)) + confidence_delta)))
    row["volume_confirmation"] = payload


def _attach_candlestick_regime_context(
    row: Dict[str, Any],
    regime_context: Optional[Dict[str, Any]],
    config: Optional[Dict[str, Any]],
) -> None:
    payload: Dict[str, Any] = {
        "status": "disabled" if not _config_bool(config, "use_regime_context", True) else "unavailable",
    }
    if payload["status"] == "disabled":
        row["regime_context"] = payload
        return
    if not isinstance(regime_context, dict):
        row["regime_context"] = payload
        return
    payload.update(regime_context)
    bias = str(row.get("direction") or "").strip().lower()
    if bias not in {"bullish", "bearish"}:
        payload["status"] = "not_directional"
        row["regime_context"] = payload
        return
    payload["pattern_bias"] = bias
    bonus = _config_float(config, "regime_alignment_bonus", 0.05, minimum=0.0)
    penalty = _config_float(config, "regime_countertrend_penalty", 0.05, minimum=0.0)
    confidence_delta = 0.0
    if payload.get("state") == "trending" and payload.get("direction") in {"bullish", "bearish"}:
        if bias == payload.get("direction"):
            payload["status"] = "aligned"
            payload["alignment"] = "aligned"
            confidence_delta = float(bonus)
        else:
            payload["status"] = "countertrend"
            payload["alignment"] = "countertrend"
            confidence_delta = -float(penalty)
    else:
        payload["status"] = "context_only"
        payload["alignment"] = "neutral"
    if abs(confidence_delta) > 1e-12:
        payload["confidence_delta"] = _round_value(confidence_delta)
        row["confidence"] = float(max(0.0, min(1.0, float(row.get("confidence", 0.0)) + confidence_delta)))
    row["regime_context"] = payload
