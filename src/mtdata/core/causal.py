"""Causal signal discovery tools."""

from __future__ import annotations

import contextlib
import io
import logging
import math
import time
import warnings
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

from ..shared.constants import TIME_DISPLAY_FORMAT, TIMEFRAME_MAP, TIMEFRAME_SECONDS
from ..shared.schema import DetailLiteral, TimeframeLiteral
from ..utils.mt5 import (
    _ensure_symbol_ready,
    _mt5_copy_rates_from,
    _mt5_copy_rates_range,
    ensure_mt5_connection_or_raise,
    mt5,
)
from ..utils.symbol import (
    _extract_group_path as _extract_group_path_util,
)
from ..utils.symbol import (
    _normalize_group_path_query,
)
from ..utils.utils import _parse_start_datetime
from ._mcp_instance import mcp
from .execution_logging import run_logged_operation
from .mt5_gateway import create_mt5_gateway, mt5_connection_error
from .output_contract import normalize_output_verbosity_detail

logger = logging.getLogger(__name__)

_CORRELATION_METHOD_ALIASES: Dict[str, str] = {
    "pearson": "pearson",
    "linear": "pearson",
    "spearman": "spearman",
    "rank": "spearman",
    "rank_corr": "spearman",
    "rank_correlation": "spearman",
}

_TRANSFORM_ALIASES: Dict[str, str] = {
    "log_return": "log_return",
    "logret": "log_return",
    "log-returns": "log_return",
    "pct": "pct",
    "return": "pct",
    "pct_change": "pct",
    "diff": "diff",
    "difference": "diff",
    "first_diff": "diff",
    "level": "level",
    "none": "level",
    "raw": "level",
    "price": "level",
    "log": "log_level",
    "log_level": "log_level",
    "log-price": "log_level",
    "log_price": "log_level",
}

_COINTEGRATION_TRANSFORM_ALIASES: Dict[str, str] = {
    "level": "level",
    "raw": "level",
    "price": "level",
    "log": "log_level",
    "log_level": "log_level",
    "log-price": "log_level",
    "log_price": "log_level",
}

_COINTEGRATION_TREND_ALIASES: Dict[str, str] = {
    "c": "c",
    "const": "c",
    "constant": "c",
    "ct": "ct",
    "trend": "ct",
    "ctt": "ctt",
    "quadratic": "ctt",
    "n": "n",
    "none": "n",
    "no_const": "n",
}


# Human-readable legends for output interpretation
_TRANSFORM_LEGEND: Dict[str, Dict[str, str]] = {
    "log_return": {
        "description": "Logarithmic returns (continuously compounded)",
        "formula": "ln(close_t / close_t-1)",
        "use_case": "Stationary, time-additive returns; preferred for multi-horizon analysis",
    },
    "pct": {
        "description": "Simple percentage change (unit fraction)",
        "formula": "(close_t - close_t-1) / close_t-1",
        "use_case": "Intuitive simple returns; 0.01 corresponds to a 1% gain",
    },
    "diff": {
        "description": "First difference (absolute change)",
        "formula": "close_t - close_t-1",
        "use_case": "Removes trends for stationary analysis; preserves scale",
    },
    "level": {
        "description": "Raw price levels (no transformation)",
        "formula": "close_t",
        "use_case": "Direct price analysis; required for cointegration tests",
    },
    "log_level": {
        "description": "Natural log of price levels",
        "formula": "ln(close_t)",
        "use_case": "Price-level analysis with reduced scale effects",
    },
}

_CORRELATION_METHOD_LEGEND: Dict[str, Dict[str, str]] = {
    "pearson": {
        "description": "Pearson linear correlation",
        "measures": "Linear relationship strength and direction",
        "range": "-1 (perfect negative) to +1 (perfect positive)",
        "sensitive_to": "Outliers and non-linear relationships",
    },
    "spearman": {
        "description": "Spearman rank correlation",
        "measures": "Monotonic relationship (rank-based)",
        "range": "-1 (perfect negative) to +1 (perfect positive)",
        "sensitive_to": "Non-linear but monotonic relationships; robust to outliers",
    },
}

_COINTEGRATION_TREND_LEGEND: Dict[str, Dict[str, str]] = {
    "c": {
        "description": "Constant only",
        "interpretation": "Tests for cointegration with non-zero mean but no trend",
    },
    "ct": {
        "description": "Constant and linear trend",
        "interpretation": "Tests for cointegration allowing for deterministic linear trend",
    },
    "ctt": {
        "description": "Constant and quadratic trend",
        "interpretation": "Tests for cointegration allowing for curved deterministic trends",
    },
    "n": {
        "description": "No deterministic terms",
        "interpretation": "Tests for cointegration around zero (rarely appropriate for prices)",
    },
}

_CAUSAL_DISCOVER_REQUEST_KEYS = frozenset(
    {
        "symbols_input",
        "symbols_expanded",
        "group_input",
        "group_resolved",
        "timeframe",
        "limit",
        "offset",
        "window_bars",
        "start",
        "end",
        "max_lag",
        "significance",
        "include_incomplete",
        "transform",
        "normalize",
        "detail",
    }
)

_CORRELATION_REQUEST_KEYS = frozenset(
    {
        "symbols_input",
        "symbols_expanded",
        "group_input",
        "group_resolved",
        "timeframe",
        "limit",
        "offset",
        "window_bars",
        "start",
        "end",
        "method",
        "transform",
        "min_overlap",
        "include_incomplete",
        "detail",
    }
)

_COINTEGRATION_REQUEST_KEYS = frozenset(
    {
        "symbols_input",
        "symbols_expanded",
        "group_input",
        "group_resolved",
        "timeframe",
        "limit",
        "offset",
        "window_bars",
        "start",
        "end",
        "transform",
        "method",
        "trend",
        "k_ar_diff",
        "significance",
        "min_overlap",
        "include_incomplete",
        "detail",
    }
)

_CROSS_CORRELATION_REQUEST_KEYS = frozenset(
    {
        "symbols_input",
        "timeframe",
        "window_bars",
        "start",
        "end",
        "max_lag",
        "method",
        "transform",
        "min_overlap",
        "bootstrap_samples",
        "include_incomplete",
        "detail",
    }
)


def _min_overlap_exceeds_window_message(*, min_overlap: int, window_bars: int) -> str:
    return (
        f"min_overlap ({int(min_overlap)}) cannot exceed window_bars ({int(window_bars)}). "
        "Reduce min_overlap or increase window_bars."
    )


def _causal_connection_error() -> Dict[str, Any] | None:
    return mt5_connection_error(
        create_mt5_gateway(
            adapter=mt5,
            ensure_connection_impl=ensure_mt5_connection_or_raise,
        )
    )


def _parse_symbols(value: Optional[str]) -> List[str]:
    items: List[str] = []
    for chunk in str(value or "").replace(";", ",").split(","):
        name = chunk.strip()
        if name:
            items.append(name)
    return list(dict.fromkeys(items))  # dedupe preserving order


def _visible_group_members(
    all_symbols: Any,
    group_path: str,
) -> List[str]:
    members: List[str] = []
    for sym in all_symbols or []:
        if not getattr(sym, "visible", True):
            continue
        if _extract_group_path_util(sym) == group_path:
            members.append(sym.name)
    return list(dict.fromkeys(members))


def _expand_symbols_for_group(
    anchor: str, gateway: Any = None
) -> tuple[List[str], str | None, str | None]:
    """Return visible group members for anchor along with the group path."""
    mt5_gateway = gateway or create_mt5_gateway(
        adapter=mt5,
        ensure_connection_impl=ensure_mt5_connection_or_raise,
    )
    info = mt5_gateway.symbol_info(anchor)
    if info is None:
        return [], f"Symbol {anchor} not found", None
    group_path = _extract_group_path_util(info)
    all_symbols = mt5_gateway.symbols_get()
    if all_symbols is None:
        return [], f"Failed to load symbol list: {mt5_gateway.last_error()}", group_path
    members = _visible_group_members(all_symbols, group_path)
    if anchor not in members:
        members.insert(0, anchor)
    deduped = list(dict.fromkeys(members))
    if len(deduped) < 2:
        return (
            deduped,
            f"Symbol group {group_path} has fewer than two visible instruments",
            group_path,
        )
    return deduped, None, group_path


def _expand_symbols_for_group_path(
    query: str, gateway: Any = None
) -> tuple[List[str], str | None, str | None]:
    """Return visible group members for an explicit MT5 group path query."""
    mt5_gateway = gateway or create_mt5_gateway(
        adapter=mt5,
        ensure_connection_impl=ensure_mt5_connection_or_raise,
    )
    group_query = str(query or "").strip()
    if not group_query:
        return [], "Group path must not be empty.", None

    all_symbols = mt5_gateway.symbols_get()
    if all_symbols is None:
        return [], f"Failed to load symbol list: {mt5_gateway.last_error()}", None

    groups: Dict[str, List[str]] = {}
    for sym in all_symbols:
        group_path = _extract_group_path_util(sym)
        if not group_path:
            continue
        if not getattr(sym, "visible", True):
            continue
        groups.setdefault(group_path, []).append(sym.name)

    if not groups:
        return [], "No visible MT5 symbol groups are available.", None

    query_lower = _normalize_group_path_query(group_query).lower()
    exact_matches = [
        group_path
        for group_path in groups
        if _normalize_group_path_query(group_path).lower() == query_lower
    ]
    matched_paths = exact_matches or [
        group_path
        for group_path in groups
        if query_lower in _normalize_group_path_query(group_path).lower()
    ]
    matched_paths = list(dict.fromkeys(matched_paths))
    if not matched_paths:
        return (
            [],
            f"Group '{group_query}' was not found among visible MT5 symbol groups.",
            None,
        )
    if len(matched_paths) > 1:
        preview = ", ".join(sorted(matched_paths)[:5])
        suffix = ", ..." if len(matched_paths) > 5 else ""
        return (
            [],
            (
                f"Group '{group_query}' matched multiple visible MT5 symbol groups: "
                f"{preview}{suffix}"
            ),
            None,
        )

    group_path = matched_paths[0]
    members = _visible_group_members(all_symbols, group_path)
    if len(members) < 2:
        return (
            members,
            f"Symbol group {group_path} has fewer than two visible instruments",
            group_path,
        )
    return members, None, group_path


def _resolve_history_window(
    start: Optional[str],
    end: Optional[str],
) -> Tuple[Optional[datetime], Optional[datetime], Optional[str]]:
    start_dt = _parse_start_datetime(start) if start else None
    if start and start_dt is None:
        return None, None, "Invalid start time."
    end_dt = _parse_start_datetime(end) if end else None
    if end and end_dt is None:
        return None, None, "Invalid end time."
    if start_dt is not None and end_dt is None:
        end_dt = datetime.now(timezone.utc).replace(tzinfo=None)
    if start_dt is not None and end_dt is not None and start_dt > end_dt:
        return None, None, "start must be before or equal to end."
    return start_dt, end_dt, None


def _fetch_series(
    symbol: str,
    timeframe,
    count: int,
    retries: int = 3,
    pause: float = 0.25,
    *,
    start: Optional[str] = None,
    end: Optional[str] = None,
    timeframe_key: Optional[str] = None,
    include_incomplete: bool = False,
) -> Tuple[pd.Series, str | None]:
    """Fetch close prices, excluding the current forming bar by default."""
    err = _ensure_symbol_ready(symbol)
    if err:
        return pd.Series(dtype=float), err
    start_dt, end_dt, window_error = _resolve_history_window(start, end)
    if window_error:
        return pd.Series(dtype=float), window_error

    for _attempt in range(retries):
        if start_dt is not None:
            data = _mt5_copy_rates_range(symbol, timeframe, start_dt, end_dt)
        elif end_dt is not None:
            data = _mt5_copy_rates_from(
                symbol, timeframe, end_dt, count + (0 if include_incomplete else 1)
            )
        else:
            utc_now = datetime.now(timezone.utc)
            data = _mt5_copy_rates_from(
                symbol, timeframe, utc_now, count + (0 if include_incomplete else 1)
            )
        if data is None or len(data) == 0:
            time.sleep(pause)
            continue
        try:
            df = pd.DataFrame(data)
        except Exception:
            df = pd.DataFrame(list(data))
        if df.empty or "time" not in df or "close" not in df:
            time.sleep(pause)
            continue
        df = df.sort_values("time")
        timeframe_name = str(timeframe_key or "").strip().upper()
        if not timeframe_name:
            timeframe_name = next(
                (name for name, value in TIMEFRAME_MAP.items() if value == timeframe),
                "",
            )
        bar_seconds = int(TIMEFRAME_SECONDS.get(timeframe_name, 0) or 0)
        forming_trimmed = False
        last_is_forming = False
        if bar_seconds > 0 and not df.empty:
            last_open_epoch = float(df.iloc[-1]["time"])
            now_epoch = datetime.now(timezone.utc).timestamp()
            last_is_forming = last_open_epoch + bar_seconds > now_epoch
            if not include_incomplete and last_is_forming:
                df = df.iloc[:-1]
                forming_trimmed = True
                last_is_forming = False
        if df.empty:
            time.sleep(pause)
            continue
        if start_dt is None and len(df) > count:
            df = df.tail(count)
        series = pd.Series(
            df["close"].to_numpy(dtype=float),
            index=pd.to_datetime(df["time"], unit="s"),
        )
        series = series[~series.index.duplicated(keep="last")]
        series.attrs["include_incomplete"] = bool(include_incomplete)
        series.attrs["forming_candle_skipped"] = bool(forming_trimmed)
        series.attrs["forming_candle_included"] = bool(last_is_forming)
        series.attrs["latest_bar_complete"] = not bool(last_is_forming)
        return series, None
    return pd.Series(dtype=float), f"Failed to fetch data for {symbol}" + (
        f" after {retries} retries" if retries > 1 else ""
    )


def _fetch_series_for_window(
    symbol: str,
    timeframe,
    count: int,
    *,
    start: Optional[str] = None,
    end: Optional[str] = None,
    timeframe_key: Optional[str] = None,
    include_incomplete: bool = False,
) -> Tuple[pd.Series, str | None]:
    if start or end:
        return _fetch_series(
            symbol,
            timeframe,
            count,
            start=start,
            end=end,
            timeframe_key=timeframe_key,
            include_incomplete=include_incomplete,
        )
    return _fetch_series(
        symbol,
        timeframe,
        count,
        timeframe_key=timeframe_key,
        include_incomplete=include_incomplete,
    )


def _bar_completion_context(
    series_map: Dict[str, pd.Series], *, include_incomplete: bool
) -> Dict[str, Any]:
    forming_included = any(
        bool(series.attrs.get("forming_candle_included"))
        for series in series_map.values()
    )
    forming_skipped = any(
        bool(series.attrs.get("forming_candle_skipped"))
        for series in series_map.values()
    )
    if forming_included:
        status = "included"
    elif forming_skipped:
        status = "skipped"
    else:
        status = "none"
    return {
        "include_incomplete": bool(include_incomplete),
        "latest_bar_complete": not forming_included,
        "forming_candle_status": status,
    }


def _transform_frame(frame: pd.DataFrame, transform: str) -> pd.DataFrame:
    transform = transform.strip().lower()
    if transform in ("log_return", "logret", "log-returns"):
        # log return r_t = ln(p_t) - ln(p_{t-1})
        clean = frame.astype(float).where(frame > 0)
        clean = clean.mask(clean <= 0)
        log_prices = np.log(clean)
        log_prices = log_prices.replace([np.inf, -np.inf], np.nan)
        frame = log_prices.diff()
    elif transform in ("log_level", "log", "log-price", "log_price"):
        clean = frame.astype(float).where(frame > 0)
        clean = clean.mask(clean <= 0)
        frame = np.log(clean).replace([np.inf, -np.inf], np.nan)
    elif transform in ("pct", "return", "pct_change"):
        frame = frame.pct_change()
    elif transform in ("diff", "difference", "first_diff"):
        frame = frame.diff()
    else:
        # default no transform
        return frame
    # Keep pairwise-complete rows for each tested symbol pair later.
    return frame.dropna(how="all")


def _normalize_correlation_method(value: str) -> str | None:
    text = str(value).strip().lower()
    if not text:
        return None
    return _CORRELATION_METHOD_ALIASES.get(text)


def _normalize_transform_name(value: str) -> str | None:
    text = str(value).strip().lower()
    if not text:
        return None
    return _TRANSFORM_ALIASES.get(text)


def _normalize_cointegration_transform(value: str) -> str | None:
    text = str(value).strip().lower()
    if not text:
        return None
    return _COINTEGRATION_TRANSFORM_ALIASES.get(text)


def _normalize_cointegration_trend(value: str) -> str | None:
    text = str(value).strip().lower()
    if not text:
        return None
    return _COINTEGRATION_TREND_ALIASES.get(text)


def _causal_transform_reason(tool: str, transform: str) -> str:
    transform_value = str(transform or "").strip().lower()
    if tool == "cointegration_test":
        if transform_value == "log_level":
            return "Cointegration tests price-level relationships; log_level preserves levels while reducing scale effects."
        return "Cointegration tests price-level relationships, so level-style transforms are used."
    if transform_value == "log_return":
        return "Return transforms compare co-movement and predictive links without shared price-scale effects."
    if transform_value == "pct":
        return "Percentage returns compare relative movement across different price scales."
    if transform_value == "diff":
        return "First differences remove level drift before pairwise relationship tests."
    return "Level transform keeps raw price levels; use for level relationships, not return co-movement."


def _pair_transform_comparability(tool: str, transform: str) -> Dict[str, List[str]]:
    """Describe which pair analytics defaults answer the same transformed question."""
    transform_value = str(transform or "").strip().lower()
    if transform_value in {"log_return", "pct", "diff"}:
        comparable = [
            name
            for name in (
                "correlation_matrix(default=log_return)",
                "causal_discover_signals(default=log_return)",
                "trade_var_cvar_calculate(default=log_return)",
            )
            if not name.startswith(str(tool or ""))
        ]
        return {
            "comparable_with": comparable,
            "not_comparable_with": ["cointegration_test(default=log_level)"],
        }
    comparable = ["cointegration_test(default=log_level)"]
    if str(tool or "") == "cointegration_test":
        comparable = []
    return {
        "comparable_with": comparable,
        "not_comparable_with": [
            "correlation_matrix(default=log_return)",
            "causal_discover_signals(default=log_return)",
            "trade_var_cvar_calculate(default=log_return)",
        ],
    }


def _pair_transform_guidance(
    tool: str,
    transform: str,
    *,
    detail: str,
) -> Dict[str, Any]:
    if detail in {"compact", "summary"}:
        return {}
    return {
        "transform_reason": _causal_transform_reason(tool, transform),
        **_pair_transform_comparability(tool, transform),
    }


def _standardize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame

    cols = list(frame.columns)
    numeric = frame.astype(float)
    means = numeric.mean(axis=0, skipna=True)
    stds = numeric.std(axis=0, ddof=0, skipna=True)
    standardized = (numeric - means) / stds.replace(0.0, np.nan)
    standardized = standardized.reindex(columns=cols)

    # Preserve prior semantics for constant columns.
    for col in cols:
        series = frame[col]
        std = float(series.std(ddof=0))
        if not math.isfinite(std) or std == 0.0:
            standardized[col] = series
    return standardized


def _transform_cointegration_frame(frame: pd.DataFrame, transform: str) -> pd.DataFrame:
    mode = _normalize_cointegration_transform(transform)
    numeric = frame.astype(float)
    if mode == "log_level":
        clean = numeric.where(numeric > 0)
        clean = clean.mask(clean <= 0)
        logged = np.log(clean)
        logged = logged.replace([np.inf, -np.inf], np.nan)
        return logged.dropna(how="all")
    return numeric


def _build_pairwise_frame(
    series_map: Dict[str, pd.Series],
    symbols: List[str],
) -> pd.DataFrame:
    aligned_map = {
        symbol: series_map[symbol]
        for symbol in symbols
        if isinstance(series_map.get(symbol), pd.Series)
    }
    if len(aligned_map) < 2:
        return pd.DataFrame()
    return pd.concat(aligned_map, axis=1, join="outer").sort_index()


def _format_sample_time(value: Any) -> str:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.strftime(TIME_DISPLAY_FORMAT)


def _pairwise_analysis_context(rows: List[Dict[str, Any]], *, timeframe: Any) -> Dict[str, Any]:
    context: Dict[str, Any] = {
        "timeframe": str(timeframe),
        "timezone": "UTC",
    }
    starts = [
        str(row.get("period_start"))
        for row in rows
        if row.get("period_start") not in (None, "")
    ]
    ends = [
        str(row.get("period_end"))
        for row in rows
        if row.get("period_end") not in (None, "")
    ]
    samples = [
        int(row["samples"])
        for row in rows
        if row.get("samples") is not None
    ]
    if starts:
        context["period_start"] = min(starts)
    if ends:
        context["period_end"] = max(ends)
    if samples:
        samples_min = min(samples)
        samples_max = max(samples)
        if samples_min == samples_max:
            context["samples"] = samples_min
        else:
            context["samples_min"] = samples_min
            context["samples_max"] = samples_max
    return context


def _correlation_fisher_ci(
    correlation: float, samples: int, *, z: float = 1.959963984540054
) -> tuple[Optional[float], Optional[float]]:
    """Fisher z-transform 95% CI for a correlation coefficient.

    Returns (None, None) when the CI is undefined (n<=3 or |r|>=1).
    """
    try:
        r = float(correlation)
        n = int(samples)
    except (TypeError, ValueError):
        return None, None
    if n <= 3 or not math.isfinite(r) or abs(r) >= 1.0:
        return None, None
    try:
        zr = math.atanh(r)
        se = 1.0 / math.sqrt(n - 3)
        lo = math.tanh(zr - z * se)
        hi = math.tanh(zr + z * se)
    except (ValueError, ZeroDivisionError):
        return None, None
    return round(lo, 6), round(hi, 6)


def _pairwise_period_alignment(
    rows: List[Dict[str, Any]],
    *,
    timeframe: Any,
) -> tuple[Dict[str, Any], Optional[str]]:
    starts = _sample_timestamps(row.get("period_start") for row in rows)
    ends = _sample_timestamps(row.get("period_end") for row in rows)
    if len(starts) < 2 and len(ends) < 2:
        return {}, None

    timeframe_key = str(timeframe or "").strip().upper()
    bar_seconds = float(TIMEFRAME_SECONDS.get(timeframe_key, 0) or 0)
    threshold = pd.Timedelta(seconds=bar_seconds) if bar_seconds > 0.0 else pd.Timedelta(0)
    start_span = (max(starts) - min(starts)) if len(starts) >= 2 else pd.Timedelta(0)
    end_span = (max(ends) - min(ends)) if len(ends) >= 2 else pd.Timedelta(0)
    if start_span <= threshold and end_span <= threshold:
        return {}, None

    context = {
        "period_scope": "pairwise_union",
        "pair_windows_aligned": False,
    }
    warning = (
        f"Pair sample windows differ by more than one {timeframe_key or 'timeframe'} bar; "
        "compare each row's period_start/period_end instead of treating context period as a shared window."
    )
    return context, warning


def _sample_timestamps(values: Any) -> List[pd.Timestamp]:
    timestamps: List[pd.Timestamp] = []
    for value in values:
        if value in (None, ""):
            continue
        try:
            timestamp = pd.Timestamp(value)
        except Exception:
            continue
        if pd.isna(timestamp):
            continue
        if timestamp.tzinfo is not None:
            timestamp = timestamp.tz_convert("UTC").tz_localize(None)
        timestamps.append(timestamp)
    return timestamps


def _rank_correlation_pairs(
    frame: pd.DataFrame,
    symbols: List[str],
    *,
    method: str,
    window_bars: int,
    min_overlap: int,
) -> tuple[List[Dict[str, Any]], Dict[str, int], Dict[str, int]]:
    rows: List[Dict[str, Any]] = []
    pair_overlaps: Dict[str, int] = {}
    skipped = {
        "min_overlap": 0,
        "nonfinite": 0,
    }

    for idx, left in enumerate(symbols):
        if left not in frame.columns:
            continue
        for right in symbols[idx + 1 :]:
            if right not in frame.columns:
                continue
            subset_all = frame[[left, right]].dropna(how="any")
            overlap_rows = int(len(subset_all))
            pair_overlaps[f"{left}-{right}"] = overlap_rows
            if overlap_rows < min_overlap:
                skipped["min_overlap"] += 1
                continue
            subset = subset_all.tail(window_bars)
            corr = subset[left].corr(subset[right], method=method)
            if corr is None or not math.isfinite(float(corr)):
                skipped["nonfinite"] += 1
                continue
            corr_f = float(corr)
            corr_rounded = round(corr_f, 6)
            ci95_low, ci95_high = _correlation_fisher_ci(corr_f, int(len(subset)))
            period_start = _format_sample_time(subset.index[0])
            period_end = _format_sample_time(subset.index[-1])
            rows.append(
                {
                    "left": left,
                    "right": right,
                    "correlation": corr_rounded,
                    "ci95_low": ci95_low,
                    "ci95_high": ci95_high,
                    "abs_correlation": round(abs(corr_f), 6),
                    "samples": int(len(subset)),
                    "period_start": period_start,
                    "period_end": period_end,
                    "calculation_samples": int(len(subset)),
                    "overlap_rows": overlap_rows,
                    "available_overlap_rows": overlap_rows,
                    "window_requested": int(window_bars),
                    "window_actual": int(len(subset)),
                    "window_truncated": bool(len(subset) < overlap_rows),
                    "relationship": (
                        "positive"
                        if corr_f > 0
                        else "negative"
                        if corr_f < 0
                        else "flat"
                    ),
                }
            )

    rows.sort(
        key=lambda item: (
            -float(item["abs_correlation"]),
            -int(item["samples"]),
            str(item["left"]),
            str(item["right"]),
        )
    )
    return rows, pair_overlaps, skipped


def _compact_correlation_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "symbol1": row.get("left"),
            "symbol2": row.get("right"),
            "correlation": row.get("correlation"),
            "ci95_low": row.get("ci95_low"),
            "ci95_high": row.get("ci95_high"),
            "samples": row.get("samples"),
            "period_start": row.get("period_start"),
            "period_end": row.get("period_end"),
        }
        for row in rows
    ]


def _normalize_output_limit(limit: Optional[int]) -> tuple[int | None, str | None]:
    if limit is None:
        return None, None
    try:
        value = int(limit)
    except (TypeError, ValueError):
        return None, "limit must be a positive integer."
    if value < 1:
        return None, "limit must be a positive integer."
    return value, None


def _normalize_output_offset(offset: int) -> tuple[int, str | None]:
    try:
        value = int(offset)
    except (TypeError, ValueError):
        return 0, "offset must be a non-negative integer."
    if value < 0:
        return 0, "offset must be a non-negative integer."
    return value, None


def _limit_pair_rows(
    rows: List[Dict[str, Any]],
    limit: int | None,
    offset: int = 0,
) -> tuple[List[Dict[str, Any]], bool, Dict[str, Any]]:
    total = int(len(rows))
    start = min(max(0, int(offset)), total)
    if limit is None:
        page = rows[start:]
    else:
        page = rows[start : start + int(limit)]
    has_more = bool(start + len(page) < total)
    truncated = bool(start > 0 or has_more)
    pagination = {
        "total_count": total,
        "offset": int(start),
        "limit": limit,
        "has_more": has_more,
    }
    return page, truncated, pagination


def _public_pair_row(row: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in row.items():
        if key == "left":
            out["symbol1"] = value
        elif key == "right":
            out["symbol2"] = value
        else:
            out[key] = value
    return out


def _build_correlation_matrix(
    symbols: List[str],
    rows: List[Dict[str, Any]],
) -> Dict[str, Dict[str, float | None]]:
    matrix: Dict[str, Dict[str, float | None]] = {
        str(symbol): {str(other): None for other in symbols} for symbol in symbols
    }
    for symbol in symbols:
        matrix[str(symbol)][str(symbol)] = 1.0
    for row in rows:
        left = str(row["left"])
        right = str(row["right"])
        corr_f = float(row["correlation"])
        matrix[left][right] = corr_f
        matrix[right][left] = corr_f
    return matrix


def _pair_highlight_ref(
    row: Dict[str, Any],
    *,
    metrics: tuple[str, ...],
) -> Dict[str, Any]:
    left = str(row.get("left") or "")
    right = str(row.get("right") or "")
    out: Dict[str, Any] = {
        "pair": f"{left}-{right}",
        "symbol1": left,
        "symbol2": right,
    }
    for key in metrics:
        if key in row:
            out[key] = row.get(key)
    if "samples" in row:
        out["samples"] = row.get("samples")
    return out


def _correlation_highlight_ref(
    item_index: int,
    row: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "item": int(item_index),
        "correlation": row.get("correlation"),
    }


def _build_correlation_summary(
    rows: List[Dict[str, Any]],
    *,
    top_n: int = 5,
) -> Dict[str, List[Dict[str, Any]]]:
    limit = max(1, int(top_n))
    if len(rows) <= limit:
        return {}
    indexed_rows = list(enumerate(rows))
    positive = sorted(
        [item for item in indexed_rows if float(item[1]["correlation"]) > 0.0],
        key=lambda item: (
            -float(item[1]["correlation"]),
            -int(item[1]["samples"]),
            str(item[1]["left"]),
            str(item[1]["right"]),
        ),
    )
    negative = sorted(
        [item for item in indexed_rows if float(item[1]["correlation"]) < 0.0],
        key=lambda item: (
            float(item[1]["correlation"]),
            -int(item[1]["samples"]),
            str(item[1]["left"]),
            str(item[1]["right"]),
        ),
    )
    return {
        "strongest_absolute": [
            _correlation_highlight_ref(index, row)
            for index, row in indexed_rows[:limit]
        ],
        "strongest_positive": [
            _correlation_highlight_ref(index, row)
            for index, row in positive[:limit]
        ],
        "strongest_negative": [
            _correlation_highlight_ref(index, row)
            for index, row in negative[:limit]
        ],
    }


def _format_pair_overlap_details(
    pair_overlaps: Dict[str, int],
    minimum_required: int,
) -> List[str]:
    return [
        f"{pair_key}: {int(rows)} rows (minimum {int(minimum_required)} required)"
        for pair_key, rows in sorted(
            pair_overlaps.items(), key=lambda kv: (kv[1], kv[0])
        )
    ]


def _critical_values_dict(values: Any) -> Dict[str, float | None]:
    arr = (
        np.asarray(values, dtype=float).reshape(-1)
        if values is not None
        else np.array([], dtype=float)
    )
    labels = ("1%", "5%", "10%")
    return {
        label: (
            float(arr[idx])
            if idx < arr.size and math.isfinite(float(arr[idx]))
            else None
        )
        for idx, label in enumerate(labels)
    }


def _fit_cointegration_hedge(
    dependent: pd.Series,
    hedge: pd.Series,
    *,
    trend: str,
) -> tuple[float | None, float | None, np.ndarray | None]:
    y = dependent.to_numpy(dtype=float)
    x = hedge.to_numpy(dtype=float)
    if y.size != x.size or y.size < 2:
        return None, None, None
    time_index = np.arange(1.0, float(x.size) + 1.0)
    if trend == "n":
        design = x.reshape(-1, 1)
    elif trend == "c":
        design = np.column_stack([x, np.ones(x.size, dtype=float)])
    elif trend == "ct":
        design = np.column_stack([x, np.ones(x.size, dtype=float), time_index])
    elif trend == "ctt":
        design = np.column_stack(
            [x, np.ones(x.size, dtype=float), time_index, time_index**2]
        )
    else:
        return None, None, None
    coeffs, *_ = np.linalg.lstsq(design, y, rcond=None)
    if coeffs.size < 1 or not math.isfinite(float(coeffs[0])):
        return None, None, None
    beta = float(coeffs[0])
    intercept = 0.0 if trend == "n" else float(coeffs[1]) if coeffs.size > 1 else 0.0
    spread = y - design @ coeffs
    return beta, intercept, spread


def _evaluate_cointegration_pair(
    subset: pd.DataFrame,
    left: str,
    right: str,
    *,
    trend: str,
    significance: float,
    coint_func: Any,
) -> tuple[Dict[str, Any] | None, List[Dict[str, Any]]]:
    failures: List[Dict[str, Any]] = []
    best_row: Dict[str, Any] | None = None

    # Engle-Granger is orientation-sensitive. Use the caller's stable pair
    # ordering instead of testing both directions and cherry-picking min(p).
    for dependent, hedge in ((left, right),):
        try:
            test_stat, p_value, critical_values = coint_func(
                subset[dependent],
                subset[hedge],
                trend=trend,
            )
        except Exception as ex:
            failures.append(
                {
                    "left": left,
                    "right": right,
                    "dependent": dependent,
                    "hedge": hedge,
                    "error": str(ex),
                    "error_type": type(ex).__name__,
                }
            )
            continue

        if p_value is None or not math.isfinite(float(p_value)):
            failures.append(
                {
                    "left": left,
                    "right": right,
                    "dependent": dependent,
                    "hedge": hedge,
                    "error": "Cointegration test returned a non-finite p-value.",
                    "error_type": "NonFinitePValue",
                }
            )
            continue

        hedge_ratio, intercept, spread = _fit_cointegration_hedge(
            subset[dependent],
            subset[hedge],
            trend=trend,
        )
        if hedge_ratio is None or spread is None:
            failures.append(
                {
                    "left": left,
                    "right": right,
                    "dependent": dependent,
                    "hedge": hedge,
                    "error": "Failed to estimate hedge ratio for the candidate spread.",
                    "error_type": "HedgeFitError",
                }
            )
            continue

        spread_last = float(spread[-1]) if spread.size else None
        spread_mean = float(np.mean(spread)) if spread.size else float("nan")
        spread_std = float(np.std(spread, ddof=0)) if spread.size else float("nan")
        spread_zscore = None
        if (
            spread_last is not None
            and math.isfinite(spread_last)
            and math.isfinite(spread_std)
            and spread_std > 0.0
            and math.isfinite(spread_mean)
        ):
            spread_zscore = float((spread_last - spread_mean) / spread_std)

        row = {
            "left": left,
            "right": right,
            "dependent": dependent,
            "hedge": hedge,
            "test_stat": float(test_stat) if math.isfinite(float(test_stat)) else None,
            "p_value": float(p_value),
            "critical_values": _critical_values_dict(critical_values),
            "hedge_ratio": float(hedge_ratio),
            "intercept": float(intercept),
            "spread_last": spread_last,
            "spread_zscore": spread_zscore,
            "samples": int(len(subset)),
            "period_start": _format_sample_time(subset.index[0]),
            "period_end": _format_sample_time(subset.index[-1]),
            "cointegrated": bool(float(p_value) < significance),
            "relationship": "cointegrated"
            if float(p_value) < significance
            else "no_cointegration",
            "orientation_policy": "left_dependent",
        }
        if best_row is None or float(row["p_value"]) < float(best_row["p_value"]):
            best_row = row

    return best_row, failures


def _apply_holm_pair_correction(
    rows: List[Dict[str, Any]],
    *,
    significance: float,
) -> None:
    """Apply a family-wise Holm correction to pairwise test results in place."""
    ordered = sorted(
        enumerate(rows),
        key=lambda item: (float(item[1]["p_value"]), item[0]),
    )
    family_size = len(ordered)
    running_adjusted = 0.0
    for rank, (_, row) in enumerate(ordered):
        raw = float(row["p_value"])
        adjusted = min(1.0, raw * float(family_size - rank))
        running_adjusted = max(running_adjusted, adjusted)
        row["p_value_raw"] = raw
        row["p_value"] = float(running_adjusted)
        row["p_value_correction"] = "holm_across_pairs"
        row["significance_basis"] = "p_value_holm_adjusted"
        row["significance_threshold"] = float(significance)
        row["pair_tests_run"] = int(family_size)
        row["cointegrated"] = bool(running_adjusted < significance)
        row["relationship"] = (
            "cointegrated" if running_adjusted < significance else "no_cointegration"
        )


def _build_cointegration_summary(
    rows: List[Dict[str, Any]],
    *,
    top_n: int = 5,
) -> Dict[str, List[Dict[str, Any]]]:
    limit = max(1, int(top_n))
    cointegrated = [row for row in rows if bool(row.get("cointegrated"))]
    summary: Dict[str, List[Dict[str, Any]]] = {}
    if len(rows) > limit:
        summary["best_pairs"] = [
            _pair_highlight_ref(
                row,
                metrics=("p_value", "test_stat", "cointegrated"),
            )
            for row in rows[:limit]
        ]
    if cointegrated and (len(rows) > limit or len(cointegrated) < len(rows)):
        summary["cointegrated_pairs"] = [
            _pair_highlight_ref(
                row,
                metrics=("p_value", "test_stat", "cointegrated"),
            )
            for row in cointegrated[:limit]
        ]
    return summary


def _johansen_rank(statistics: np.ndarray, critical_values: np.ndarray, column: int) -> int:
    rank = 0
    for index, statistic in enumerate(np.asarray(statistics, dtype=float)):
        if index >= len(critical_values):
            break
        if float(statistic) > float(critical_values[index][column]):
            rank = index + 1
        else:
            break
    return int(rank)


def _lagged_pair_values(
    left: np.ndarray,
    right: np.ndarray,
    lag: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Align arrays so a positive lag means left leads right."""
    if lag > 0:
        return left[:-lag], right[lag:]
    if lag < 0:
        shift = abs(int(lag))
        return left[shift:], right[:-shift]
    return left, right


def _correlation_value(left: np.ndarray, right: np.ndarray, method: str) -> float:
    if left.size < 2 or right.size < 2:
        return float("nan")
    if float(np.std(left, ddof=0)) <= 1e-15 or float(np.std(right, ddof=0)) <= 1e-15:
        return float("nan")
    if method == "spearman":
        from scipy.stats import spearmanr

        value = spearmanr(left, right, nan_policy="omit").statistic
        return float(value)
    return float(np.corrcoef(left, right)[0, 1])


def _block_bootstrap_correlation_ci(
    left: np.ndarray,
    right: np.ndarray,
    *,
    method: str,
    samples: int,
    block_size: int,
    confidence: float = 0.95,
) -> tuple[Optional[float], Optional[float]]:
    n = int(min(left.size, right.size))
    if n < 8 or samples < 20:
        return None, None
    block = max(2, min(int(block_size), n))
    rng = np.random.default_rng(42)
    values: List[float] = []
    max_start = max(1, n - block + 1)
    blocks_needed = int(math.ceil(n / float(block)))
    for _ in range(int(samples)):
        indices: List[int] = []
        for _block in range(blocks_needed):
            start_idx = int(rng.integers(0, max_start))
            indices.extend(range(start_idx, min(start_idx + block, n)))
        take = np.asarray(indices[:n], dtype=int)
        value = _correlation_value(left[take], right[take], method)
        if math.isfinite(value):
            values.append(value)
    if len(values) < 20:
        return None, None
    tail = (1.0 - float(confidence)) / 2.0
    low, high = np.quantile(np.asarray(values), [tail, 1.0 - tail])
    return float(low), float(high)


def _format_summary(
    rows: List[Dict[str, object]],
    symbols: List[str],
    transform: str,
    alpha: float,
    group_hint: str | None = None,
) -> str:
    if not rows:
        return "No valid pairings available for causal discovery."
    rows_sorted = sorted(
        rows, key=lambda item: (item["p_value"], item["effect"], item["cause"])
    )
    header = [
        f"Causal signal discovery (transform={transform}, alpha={alpha:.4f})",
        f"Symbols analysed: {', '.join(symbols)}",
        "",
        "Effect <- Cause | Lag | p-value | Samples | Conclusion",
        "-----------------------------------------------",
    ]
    if group_hint:
        header.insert(1, f"Group: {group_hint}")
    lines = header
    for row in rows_sorted:
        conclusion = "causal" if row["p_value"] < alpha else "no-link"
        lines.append(
            f"{row['effect']} <- {row['cause']} | {row['lag']} | {row['p_value']:.4f} | {row['samples']} | {conclusion}"
        )
    lines.append("")
    lines.append(
        "Lag refers to the history length of the cause series used in the best-performing test (ssr_ftest)."
    )
    lines.append(
        "Displayed p-values use Bonferroni correction across tested lags and "
        "all successfully tested directed pairs."
    )
    lines.append("Results are pairwise and do not imply full causal graphs.")
    return "\n".join(lines)


def _compact_causal_pair_rows(
    rows: List[Dict[str, Any]], *, limit: int = 20
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows[: max(0, int(limit))]:
        out.append(
            {
                "effect": row.get("effect"),
                "cause": row.get("cause"),
                "lag": row.get("lag"),
                "p_value": row.get("p_value"),
                "p_value_raw": row.get("p_value_raw"),
                "p_value_correction": row.get("p_value_correction"),
                "significance_basis": row.get("significance_basis"),
                "significance_threshold": row.get("significance_threshold"),
                "significant": bool(row.get("significant")),
            }
        )
    return out


def _causal_error(
    message: str,
    *,
    code: str,
    meta: Dict[str, Any],
    warnings: List[str] | None = None,
    details: List[str] | None = None,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "success": False,
        "error": str(message),
        "error_code": str(code),
        "meta": _causal_contract_meta(meta),
    }
    if warnings:
        out["warnings"] = warnings
    if details:
        out["details"] = details
    return out


def _causal_contract_meta(
    meta: Dict[str, Any],
    *,
    legends: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    meta_in = dict(meta or {})
    tool_name = str(meta_in.pop("_tool", "") or "").strip()
    request_keys_raw = meta_in.pop("_request_keys", ())
    request_keys = {
        str(key)
        for key in (
            request_keys_raw
            if isinstance(request_keys_raw, (set, frozenset, list, tuple))
            else ()
        )
    }

    request: Dict[str, Any] = {}
    stats: Dict[str, Any] = {}
    for key, value in meta_in.items():
        if value is None:
            continue
        if key in request_keys:
            request[key] = value
        else:
            stats[key] = value

    out: Dict[str, Any] = {
        "tool": tool_name,
        "request": request,
        "runtime": {},
    }
    if stats:
        out["stats"] = stats
    if legends:
        out["legends"] = legends
    return out


def _format_overlap_details(
    symbol_rows: Dict[str, int],
    aligned_rows: int,
    minimum_required: int,
) -> str:
    parts: List[str] = []
    for symbol, count in symbol_rows.items():
        parts.append(f"{symbol}: {int(count)} rows")
    parts.append(
        f"aligned: {int(aligned_rows)} (minimum {int(minimum_required)} required)"
    )
    return ", ".join(parts)


def _pair_overlap_counts(
    series_map: Dict[str, pd.Series], symbols: List[str]
) -> Dict[str, int]:
    overlaps: Dict[str, int] = {}
    for i, left in enumerate(symbols):
        left_series = series_map.get(left)
        if not isinstance(left_series, pd.Series):
            continue
        left_idx = left_series.dropna().index
        for right in symbols[i + 1 :]:
            right_series = series_map.get(right)
            if not isinstance(right_series, pd.Series):
                continue
            right_idx = right_series.dropna().index
            key = f"{left}-{right}"
            overlaps[key] = int(len(left_idx.intersection(right_idx)))
    return overlaps


def _build_overlap_frame(
    series_map: Dict[str, pd.Series],
    symbols: List[str],
    limit: int,
) -> pd.DataFrame:
    aligned_map = {
        symbol: series_map[symbol]
        for symbol in symbols
        if isinstance(series_map.get(symbol), pd.Series)
    }
    if len(aligned_map) < 2:
        return pd.DataFrame()
    return pd.concat(aligned_map, axis=1, join="inner").tail(limit)


def _pair_overlap_symbols(
    pair_key: str, symbols: List[str] | None = None
) -> tuple[str, str]:
    text = str(pair_key)
    if symbols:
        ordered = sorted(
            {str(symbol) for symbol in symbols if symbol}, key=len, reverse=True
        )
        for left in ordered:
            prefix = f"{left}-"
            if not text.startswith(prefix):
                continue
            right = text[len(prefix) :]
            if right in ordered:
                return left, right
    left, _, right = text.partition("-")
    return left, right


def _build_alignment_detail(
    symbol_rows: Dict[str, int],
    pair_overlaps: Dict[str, int],
    aligned_rows: int,
    minimum_required: int,
) -> Optional[Dict[str, Any]]:
    if not symbol_rows or not pair_overlaps:
        return None
    min_symbol_rows = int(min(symbol_rows.values()))
    if min_symbol_rows <= 0:
        return None
    shrinkage_ratio = float(aligned_rows) / float(min_symbol_rows)
    if aligned_rows >= minimum_required and shrinkage_ratio >= 0.90:
        return None
    bottleneck_pair = min(pair_overlaps.items(), key=lambda kv: kv[1])
    return {
        "pair_overlaps": pair_overlaps,
        "bottleneck_pair": str(bottleneck_pair[0]),
        "bottleneck_rows": int(bottleneck_pair[1]),
        "aligned_rows": int(aligned_rows),
        "min_symbol_rows": int(min_symbol_rows),
        "shrinkage_ratio": float(shrinkage_ratio),
    }


def _format_alignment_detail_summary(detail: Dict[str, Any]) -> str:
    pair_overlaps = detail.get("pair_overlaps")
    if not isinstance(pair_overlaps, dict):
        return ""
    pair_str = ", ".join(f"{k}: {int(v)}" for k, v in pair_overlaps.items())
    bottleneck_pair = str(detail.get("bottleneck_pair") or "")
    bottleneck_rows = detail.get("bottleneck_rows")
    suffix = ""
    if bottleneck_pair and bottleneck_rows is not None:
        suffix = f"; bottleneck={bottleneck_pair} ({int(bottleneck_rows)} rows)"
    return f"pair_overlaps: {pair_str}{suffix}"


@mcp.tool()
def causal_discover_signals(  # noqa: C901
    symbols: Optional[str] = None,
    group: Optional[str] = None,
    timeframe: TimeframeLiteral = "H1",
    limit: Optional[int] = None,
    offset: int = 0,
    window_bars: int = 500,
    start: Optional[str] = None,
    end: Optional[str] = None,
    max_lag: int = 5,
    significance: float = 0.05,
    include_incomplete: bool = False,
    transform: str = "log_return",
    normalize: bool = True,
    detail: DetailLiteral = "compact",
) -> Dict[str, Any]:
    """Run Granger-style causal discovery on MT5 symbols.

    Args:
        symbols: Comma-separated MT5 symbols; provide one symbol to auto-expand
            its group. Optional when using `group`.
        group: Explicit MT5 group path (for example "Forex\\Majors"). Mutually
            exclusive with `symbols`.
        timeframe: MT5 timeframe key (e.g. "M15", "H1").
        limit: Optional maximum number of returned causal rows.
        window_bars: Maximum overlapping transformed samples analysed per pair
            after applying any time window.
        start: Optional UTC-compatible start date/time for the analysis window.
        end: Optional UTC-compatible end date/time; end-only anchors recent history.
        max_lag: Maximum lag order for tests (>=1).
        significance: Family-wise alpha level for reporting causal links after
            Bonferroni correction across tested lags and directed pairs.
        include_incomplete: Include the current forming candle. Defaults to false
            so statistical tests use completed bars only.
        transform: Preprocessing transform: "log_return", "pct", "diff", "level", or "log_level".
        normalize: Z-score columns before testing to stabilise scale.
        detail: "compact" returns significant links plus top pair summaries; "full"
            returns every tested pair in items.
    """

    def _run() -> Dict[str, Any]:  # noqa: C901
        requested_detail = str(detail or "compact").strip().lower()
        if requested_detail not in {"compact", "standard", "summary", "full"}:
            return _causal_error(
                "detail must be one of: compact, standard, summary, full.",
                code="invalid_detail",
                meta={"detail": requested_detail},
            )
        detail_mode = normalize_output_verbosity_detail(detail, default="compact")
        meta: Dict[str, Any] = {
            "_tool": "causal_discover_signals",
            "_request_keys": _CAUSAL_DISCOVER_REQUEST_KEYS,
            "timeframe": str(timeframe),
            "limit": limit,
            "offset": int(offset),
            "window_bars": int(window_bars),
            "start": start,
            "end": end,
            "max_lag": int(max_lag),
            "significance": float(significance),
            "include_incomplete": bool(include_incomplete),
            "transform": str(transform),
            "normalize": bool(normalize),
            "detail": detail_mode,
        }
        transform_value = _normalize_transform_name(transform)
        if transform_value is None:
            return _causal_error(
                "Invalid transform. Valid options: log_return, pct, diff, level, log_level",
                code="invalid_transform",
                meta=meta,
            )
        meta["transform"] = transform_value
        if not math.isfinite(float(significance)) or not (
            0.0 < float(significance) < 1.0
        ):
            return _causal_error(
                "significance must be a finite fraction strictly between 0 and 1 (for example, 0.05 for 5%).",
                code="invalid_input",
                meta=meta,
            )
        connection_error = _causal_connection_error()
        if connection_error is not None:
            return _causal_error(
                str(connection_error.get("error") or "Failed to connect to MetaTrader5."),
                code=str(connection_error.get("error_code") or "mt5_connection_error"),
                meta=meta,
            )
        mt5_gateway = create_mt5_gateway(
            adapter=mt5,
            ensure_connection_impl=ensure_mt5_connection_or_raise,
        )

        try:
            from statsmodels.tsa.stattools import grangercausalitytests
        except Exception:
            return _causal_error(
                "statsmodels is required for causal discovery. Please install 'statsmodels'.",
                code="dependency_missing",
                meta=meta,
            )

        symbol_list = _parse_symbols(symbols)
        if symbol_list:
            meta["symbols_input"] = list(symbol_list)
        if group is not None:
            meta["group_input"] = str(group)
        group_hint: str | None = None
        requested_anchor = (
            symbol_list[0] if group is None and len(symbol_list) == 1 else None
        )
        if group and symbol_list:
            return _causal_error(
                "Provide either symbols or group for causal discovery, not both.",
                code="invalid_input",
                meta=meta,
            )
        if group:
            expanded, err, group_path = _expand_symbols_for_group_path(
                group,
                gateway=mt5_gateway,
            )
            if err:
                return _causal_error(
                    err,
                    code="symbol_group_error",
                    meta=meta,
                )
            symbol_list = expanded
            group_hint = group_path
            meta["group_resolved"] = group_path
            meta["symbols_expanded"] = list(symbol_list)
        elif not symbol_list:
            return _causal_error(
                "Provide at least one symbol or MT5 group for causal discovery.",
                code="invalid_input",
                meta=meta,
            )
        elif len(symbol_list) == 1:
            expanded, err, group_path = _expand_symbols_for_group(
                symbol_list[0], gateway=mt5_gateway
            )
            if err:
                return _causal_error(
                    err,
                    code="symbol_group_error",
                    meta=meta,
                )
            symbol_list = expanded
            group_hint = group_path
            meta["symbols_expanded"] = list(symbol_list)

        if len(symbol_list) < 2:
            return _causal_error(
                "Provide at least two symbols for causal discovery (e.g. 'EURUSD,GBPUSD').",
                code="invalid_input",
                meta=meta,
            )

        tf = TIMEFRAME_MAP.get(timeframe)
        if tf is None:
            valid = ", ".join(sorted(TIMEFRAME_MAP.keys()))
            return _causal_error(
                f"Invalid timeframe '{timeframe}'. Valid options: {valid}",
                code="invalid_timeframe",
                meta=meta,
            )

        if max_lag < 1:
            return _causal_error(
                "max_lag must be at least 1.",
                code="invalid_input",
                meta=meta,
            )

        output_limit, limit_error = _normalize_output_limit(limit)
        if limit_error is not None:
            return _causal_error(
                limit_error,
                code="invalid_input",
                meta=meta,
            )
        output_offset, offset_error = _normalize_output_offset(offset)
        if offset_error is not None:
            return _causal_error(
                offset_error,
                code="invalid_input",
                meta=meta,
            )
        if window_bars < 2:
            return _causal_error(
                "window_bars must be at least 2.",
                code="invalid_input",
                meta=meta,
            )

        fetch_count = max(int(window_bars) + max_lag + 10, 200)
        meta["fetch_count"] = int(fetch_count)
        series_map: Dict[str, pd.Series] = {}
        errors: List[str] = []
        for symbol_name in symbol_list:
            series, err = _fetch_series_for_window(
                symbol_name,
                tf,
                fetch_count,
                start=start,
                end=end,
                timeframe_key=str(timeframe),
                include_incomplete=bool(include_incomplete),
            )
            if err:
                errors.append(err)
            else:
                series_map[symbol_name] = series

        if errors and not series_map:
            return _causal_error(
                errors[0],
                code="data_fetch_failed",
                meta=meta,
                details=errors,
            )
        warnings_out: List[str] = []
        if errors:
            warnings_out.extend(errors)
            symbol_list = [s for s in symbol_list if s in series_map]

        if requested_anchor and requested_anchor not in series_map:
            details_out = []
            expanded_symbols = meta.get("symbols_expanded")
            if isinstance(expanded_symbols, list) and expanded_symbols:
                details_out.append(
                    f"Expanded group: {', '.join(str(sym) for sym in expanded_symbols)}"
                )
            return _causal_error(
                f"Requested symbol {requested_anchor} could not be fetched from its auto-expanded group.",
                code="anchor_symbol_missing",
                meta=meta,
                warnings=warnings_out,
                details=details_out or None,
            )

        if len(series_map) < 2:
            return _causal_error(
                "Not enough valid symbol data fetched to run causal discovery.",
                code="insufficient_symbols",
                meta=meta,
                warnings=warnings_out,
            )

        symbol_rows: Dict[str, int] = {
            str(sym): int(len(series_map.get(sym, pd.Series(dtype=float))))
            for sym in symbol_list
            if sym in series_map
        }
        if symbol_rows:
            meta["symbol_rows"] = symbol_rows

        pair_overlaps = _pair_overlap_counts(series_map, symbol_list)
        if pair_overlaps:
            meta["pair_overlaps"] = pair_overlaps

        joint_frame = _build_overlap_frame(series_map, symbol_list, int(window_bars))
        frame = _build_pairwise_frame(series_map, symbol_list)
        meta["symbols_used"] = list(frame.columns)
        meta["alignment_mode"] = "pairwise"
        min_required_samples = int(max_lag + 6)
        meta["minimum_samples_required"] = int(min_required_samples)
        # Retain joint overlap as a basket diagnostic only. Granger execution
        # below uses each pair's own overlap and window.
        meta["samples_aligned_raw"] = int(len(joint_frame))
        alignment_detail = _build_alignment_detail(
            symbol_rows=symbol_rows,
            pair_overlaps=pair_overlaps,
            aligned_rows=int(len(joint_frame)),
            minimum_required=min_required_samples,
        )
        if alignment_detail is not None:
            meta["alignment_detail"] = alignment_detail
        if int(window_bars) < min_required_samples:
            details_out = []
            if alignment_detail is not None:
                align_summary = _format_alignment_detail_summary(alignment_detail)
                if align_summary:
                    details_out.append(align_summary)
            return _causal_error(
                f"Insufficient pairwise observations after applying window_bars={int(window_bars)}; "
                f"minimum required is {min_required_samples}. Increase --window-bars to at least "
                f"{min_required_samples} or reduce max_lag (currently {int(max_lag)}).",
                code="insufficient_overlap",
                meta=meta,
                warnings=warnings_out,
                details=details_out or None,
            )
        usable_pair_overlaps = {
            pair: int(samples)
            for pair, samples in pair_overlaps.items()
            if int(samples) >= min_required_samples
        }
        if requested_anchor and not any(
            requested_anchor in _pair_overlap_symbols(pair, symbol_list)
            for pair in usable_pair_overlaps
        ):
            return _causal_error(
                f"Requested symbol {requested_anchor} had no usable pairwise overlap in its auto-expanded group.",
                code="insufficient_overlap",
                meta=meta,
                warnings=warnings_out,
            )
        if frame.empty or not usable_pair_overlaps:
            details_out = [
                _format_overlap_details(
                    symbol_rows=symbol_rows,
                    aligned_rows=int(len(joint_frame)),
                    minimum_required=min_required_samples,
                )
            ]
            if alignment_detail is not None:
                align_summary = _format_alignment_detail_summary(alignment_detail)
                if align_summary:
                    details_out.append(align_summary)
            return _causal_error(
                "Insufficient pairwise overlapping data between symbols to run tests.",
                code="insufficient_overlap",
                meta=meta,
                warnings=warnings_out,
                details=details_out,
            )

        frame = frame.dropna(how="all")
        transformed = _transform_frame(frame, transform_value)
        if transformed.empty:
            return _causal_error(
                "Transform produced insufficient samples for testing. Try using more history or a different transform.",
                code="insufficient_samples",
                meta=meta,
                warnings=warnings_out,
            )

        rows: List[Dict[str, object]] = []
        pair_attempts = 0
        pair_success = 0
        tested_directions: List[Dict[str, str]] = []
        pair_failures: List[Dict[str, Any]] = []
        pair_skips: List[Dict[str, Any]] = []
        maximum_allowable_lags: List[int] = []
        for effect in transformed.columns:
            for cause in transformed.columns:
                if effect == cause:
                    continue
                subset = (
                    transformed[[effect, cause]]
                    .dropna(how="any")
                    .tail(int(window_bars))
                )
                if normalize and not subset.empty:
                    subset = _standardize_frame(subset).dropna(how="any")
                if len(subset) <= max_lag + 2:
                    pair_skips.append(
                        {
                            "effect": effect,
                            "cause": cause,
                            "samples": int(len(subset)),
                            "reason": "insufficient_pairwise_samples",
                        }
                    )
                    continue
                pair_attempts += 1
                maximum_allowable_lag = max(
                    0, int((len(subset) - 1) / 3) - 1
                )
                maximum_allowable_lags.append(maximum_allowable_lag)
                if int(max_lag) > maximum_allowable_lag:
                    if len(pair_failures) < 10:
                        pair_failures.append(
                            {
                                "effect": effect,
                                "cause": cause,
                                "samples": int(len(subset)),
                                "requested_max_lag": int(max_lag),
                                "maximum_allowable_lag": maximum_allowable_lag,
                                "error": (
                                    "Insufficient observations for requested max_lag; "
                                    f"maximum allowable lag is {maximum_allowable_lag}."
                                ),
                                "error_type": "InsufficientObservations",
                            }
                        )
                    continue
                try:
                    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
                        warnings.simplefilter("ignore", category=FutureWarning)
                        tests = grangercausalitytests(
                            subset[[effect, cause]],
                            maxlag=max_lag,
                            verbose=False,
                        )
                except Exception as ex:
                    if len(pair_failures) < 10:
                        pair_failures.append(
                            {
                                "effect": effect,
                                "cause": cause,
                                "samples": int(len(subset)),
                                "error": str(ex),
                                "error_type": type(ex).__name__,
                            }
                        )
                    continue
                pair_success += 1
                tested_directions.append(
                    {"cause": str(cause), "effect": str(effect)}
                )
                best_lag = None
                best_p_raw = None
                tested_lags = 0
                for lag, result in tests.items():
                    stat, pvalue, *_ = result[0]["ssr_ftest"]
                    if math.isnan(pvalue):
                        continue
                    tested_lags += 1
                    if best_p_raw is None or pvalue < best_p_raw:
                        best_p_raw = float(pvalue)
                        best_lag = int(lag)
                if best_p_raw is None or best_lag is None:
                    continue
                lag_correction_factor = max(1, int(tested_lags))
                best_p = float(min(1.0, best_p_raw * lag_correction_factor))
                period_start = _format_sample_time(subset.index[0])
                period_end = _format_sample_time(subset.index[-1])
                rows.append(
                    {
                        "effect": effect,
                        "cause": cause,
                        "lag": best_lag,
                        "p_value": best_p,
                        "p_value_raw": float(best_p_raw),
                        "p_value_correction": "bonferroni",
                        "significance_basis": "p_value_bonferroni_adjusted",
                        "significance_threshold": float(significance),
                        "lag_tests_run": lag_correction_factor,
                        "samples": len(subset),
                        "period_start": period_start,
                        "period_end": period_end,
                        "significant": bool(best_p < significance),
                    }
                )
        if requested_anchor and not any(
            row.get("effect") == requested_anchor or row.get("cause") == requested_anchor
            for row in rows
        ):
            return _causal_error(
                f"Requested symbol {requested_anchor} had no usable pairwise overlap in its auto-expanded group.",
                code="insufficient_overlap",
                meta=meta,
                warnings=warnings_out,
                details=pair_skips or None,
            )
        pair_correction_factor = max(1, len(rows))
        for row in rows:
            lag_adjusted = float(row["p_value"])
            row["p_value_lag_adjusted"] = lag_adjusted
            row["p_value"] = float(min(1.0, lag_adjusted * pair_correction_factor))
            row["pair_tests_run"] = pair_correction_factor
            row["p_value_correction"] = "bonferroni_across_lags_and_pairs"
            row["significance_basis"] = "p_value_global_bonferroni_adjusted"
            row["significant"] = bool(float(row["p_value"]) < significance)
        rows_sorted = sorted(
            rows, key=lambda item: (item["p_value"], item["effect"], item["cause"])
        )
        significant_rows = [
            row for row in rows_sorted if bool(row.get("significant"))
        ]
        pair_sample_counts = [int(row["samples"]) for row in rows]
        undirected_pairs_tested = len(
            {
                tuple(sorted((item["cause"], item["effect"])))
                for item in tested_directions
            }
        )
        meta.update(
            {
                "group_hint": group_hint,
                "symbols_used": list(transformed.columns),
                "pairwise_samples_min": min(pair_sample_counts) if pair_sample_counts else 0,
                "pairwise_samples_max": max(pair_sample_counts) if pair_sample_counts else 0,
                "pairs_attempted": int(pair_attempts),
                "pairs_tested": int(pair_success),
                "pairs_failed": int(max(pair_attempts - pair_success, 0)),
                "pairs_skipped": int(len(pair_skips)),
                "p_value_correction": "bonferroni_across_lags_and_pairs",
                "pair_correction_factor": pair_correction_factor,
                "maximum_allowable_lag": min(maximum_allowable_lags)
                if maximum_allowable_lags
                else 0,
            }
        )
        if pair_failures:
            meta["pair_failures"] = pair_failures
            warnings_out.append(
                f"{max(pair_attempts - pair_success, 0)} pairwise Granger tests failed."
            )
        if pair_skips:
            meta["pair_skips"] = pair_skips[:20]
            warnings_out.append(
                f"{len(pair_skips)} directed pairs were skipped for insufficient pairwise samples."
            )
        if pair_success == 0:
            return _causal_error(
                "No Granger tests completed. Reduce max_lag or increase window_bars.",
                code="no_tests_completed",
                meta=meta,
                warnings=warnings_out,
                details=pair_failures or pair_skips or None,
            )
        rows_for_output = (
            rows_sorted
            if detail_mode in {"standard", "full"}
            else significant_rows
        )
        output_rows, output_truncated, pagination = _limit_pair_rows(
            rows_for_output,
            output_limit,
            output_offset,
        )
        meta["output_truncated"] = output_truncated
        out: Dict[str, Any] = {
            "success": True,
            "result": "links_found" if significant_rows else "no_links_found",
            "transform": transform_value,
            **_pair_transform_guidance(
                "causal_discover_signals",
                transform_value,
                detail=requested_detail,
            ),
            "items": output_rows,
            "count": int(len(output_rows)),
            "pairs_tested": int(pair_success),
            "pairs_tested_basis": "directed_granger_tests",
            "directed_tests": int(pair_success),
            "undirected_pairs": int(undirected_pairs_tested),
            "tested_directions": tested_directions[:20],
            **pagination,
            "context": {
                **_pairwise_analysis_context(rows_sorted, timeframe=timeframe),
                "limit": output_limit,
                "window_bars": int(window_bars),
                "start": start,
                "end": end,
                "transform": transform_value,
                "max_lag": int(max_lag),
                "significance": float(significance),
                **_bar_completion_context(
                    series_map, include_incomplete=bool(include_incomplete)
                ),
            },
            "summary": {
                "significance": float(significance),
                "significance_basis": "p_value_global_bonferroni_adjusted",
                "significance_threshold": float(significance),
                "counts": {
                    "pairs_tested": int(pair_success),
                    "directed_tests": int(pair_success),
                    "undirected_pairs": int(undirected_pairs_tested),
                    "significant_links": int(len(significant_rows)),
                }
            },
            "meta": _causal_contract_meta(
                meta,
                legends={
                    "transform": _TRANSFORM_LEGEND,
                    "note_p_value": "Lower p-values indicate stronger evidence of causality. Values < significance threshold indicate significant Granger-causal relationship.",
                    "note_lag": "Optimal lag order (bars) at which past values of 'cause' best predict current 'effect'",
                    "note_p_value_correction": "Displayed p-values use Bonferroni correction across tested lags and all successfully tested directed pairs.",
                },
            ),
        }
        if output_truncated:
            out["truncated"] = True
        if warnings_out:
            out["warnings"] = warnings_out
        if rows_sorted and detail_mode == "full":
            out["pairs"] = _compact_causal_pair_rows(rows_sorted, limit=20)
        if not rows_sorted:
            out["result"] = "no_tests_run"
            out["message"] = (
                "No causal relationships detected (insufficient data or all tests failed)."
            )
        elif not significant_rows:
            out["message"] = (
                "No statistically significant causal links detected at the selected threshold."
            )
            near_threshold = min(1.0, float(significance) * 2.0)
            near_misses = [
                {
                    "effect": row.get("effect"),
                    "cause": row.get("cause"),
                    "lag": row.get("lag"),
                    "p_value": row.get("p_value"),
                }
                for row in rows_sorted
                if float(row.get("p_value", 1.0)) <= near_threshold
            ][:3]
            if near_misses:
                out["near_misses"] = near_misses
            out["hint"] = (
                "For exploration, try higher significance, larger max_lag, or larger window_bars."
            )
        return out

    return run_logged_operation(
        logger,
        operation="causal_discover_signals",
        symbols=symbols,
        group=group,
        timeframe=timeframe,
        limit=limit,
        window_bars=window_bars,
        start=start,
        end=end,
        max_lag=max_lag,
        detail=detail,
        func=_run,
    )


@mcp.tool()
def correlation_matrix(  # noqa: C901
    symbols: Optional[str] = None,
    group: Optional[str] = None,
    timeframe: TimeframeLiteral = "H1",
    limit: Optional[int] = None,
    offset: int = 0,
    window_bars: int = 500,
    start: Optional[str] = None,
    end: Optional[str] = None,
    method: str = "pearson",
    transform: str = "log_return",
    min_overlap: int = 30,
    include_incomplete: bool = False,
    detail: DetailLiteral = "compact",
) -> Dict[str, Any]:
    """Calculate pairwise symbol correlations from MT5 price history.

    When a single symbol is provided, the tool automatically expands to include
    all related symbols from its MT5 group (e.g., EURUSD → EURUSD, GBPUSD, 
    USDCHF, USDJPY, USDCAD, AUDUSD). This enables correlation analysis across
    related pairs. To analyze correlations for specific symbols only, provide
    multiple symbols explicitly or use the `group` parameter.

    Args:
        symbols: Comma-separated MT5 symbols; a single symbol auto-expands to
            its entire MT5 group (e.g. "EURUSD" → all Forex majors).
            Optional when using `group`.
        group: Explicit MT5 group path (for example "Forex\\Majors"). Mutually
            exclusive with `symbols`.
        timeframe: MT5 timeframe key (e.g. "M15", "H1").
        limit: Optional maximum number of ranked pair rows returned.
        window_bars: Maximum number of overlapping transformed samples used per
            pair after applying any time window.
        start: Optional UTC-compatible start date/time for the analysis window.
        end: Optional UTC-compatible end date/time; end-only anchors recent history.
        method: Correlation method: "pearson" or "spearman".
        transform: Preprocessing transform: "log_return", "pct", "diff", "level", or "log_level".
        min_overlap: Minimum overlapping transformed samples required per pair.
        include_incomplete: Include the current forming candle. Defaults to false.
        detail: "compact" keeps canonical pair rows and counts; "standard" adds
            highlight indexes; "full" also includes the derived matrix view.
    """

    def _run() -> Dict[str, Any]:  # noqa: C901
        meta: Dict[str, Any] = {
            "_tool": "correlation_matrix",
            "_request_keys": _CORRELATION_REQUEST_KEYS,
            "timeframe": str(timeframe),
            "limit": limit,
            "offset": int(offset),
            "window_bars": int(window_bars),
            "start": start,
            "end": end,
            "method": str(method),
            "transform": str(transform),
            "min_overlap": int(min_overlap),
            "include_incomplete": bool(include_incomplete),
            "detail": str(detail or "compact"),
        }
        connection_error = _causal_connection_error()
        if connection_error is not None:
            return _causal_error(
                str(connection_error.get("error") or "Failed to connect to MetaTrader5."),
                code=str(connection_error.get("error_code") or "mt5_connection_error"),
                meta=meta,
            )
        mt5_gateway = create_mt5_gateway(
            adapter=mt5,
            ensure_connection_impl=ensure_mt5_connection_or_raise,
        )

        symbol_list = _parse_symbols(symbols)
        if symbol_list:
            meta["symbols_input"] = list(symbol_list)
        if group is not None:
            meta["group_input"] = str(group)
        requested_anchor = (
            symbol_list[0] if group is None and len(symbol_list) == 1 else None
        )
        group_hint: str | None = None
        if group and symbol_list:
            return _causal_error(
                "Provide either symbols or group for correlation analysis, not both.",
                code="invalid_input",
                meta=meta,
            )
        if group:
            expanded, err, group_path = _expand_symbols_for_group_path(
                group,
                gateway=mt5_gateway,
            )
            if err:
                return _causal_error(
                    err,
                    code="symbol_group_error",
                    meta=meta,
                )
            symbol_list = expanded
            group_hint = group_path
            meta["group_resolved"] = group_path
            meta["symbols_expanded"] = list(symbol_list)
        elif not symbol_list:
            return _causal_error(
                "Provide at least one symbol or MT5 group for correlation analysis.",
                code="invalid_input",
                meta=meta,
            )
        elif len(symbol_list) == 1:
            expanded, err, group_path = _expand_symbols_for_group(
                symbol_list[0],
                gateway=mt5_gateway,
            )
            if err:
                return _causal_error(
                    err,
                    code="symbol_group_error",
                    meta=meta,
                )
            symbol_list = expanded
            group_hint = group_path
            meta["symbols_expanded"] = list(symbol_list)

        if len(symbol_list) < 2:
            return _causal_error(
                "Provide at least two symbols for correlation analysis (e.g. 'EURUSD,GBPUSD').",
                code="invalid_input",
                meta=meta,
            )

        tf = TIMEFRAME_MAP.get(timeframe)
        if tf is None:
            valid = ", ".join(sorted(TIMEFRAME_MAP.keys()))
            return _causal_error(
                f"Invalid timeframe '{timeframe}'. Valid options: {valid}",
                code="invalid_timeframe",
                meta=meta,
            )

        output_limit, limit_error = _normalize_output_limit(limit)
        if limit_error is not None:
            return _causal_error(
                limit_error,
                code="invalid_input",
                meta=meta,
            )
        output_offset, offset_error = _normalize_output_offset(offset)
        if offset_error is not None:
            return _causal_error(
                offset_error,
                code="invalid_input",
                meta=meta,
            )

        if window_bars < 2:
            return _causal_error(
                "window_bars must be at least 2.",
                code="invalid_input",
                meta=meta,
            )

        if min_overlap < 2:
            return _causal_error(
                "min_overlap must be at least 2.",
                code="invalid_input",
                meta=meta,
            )
        if window_bars < min_overlap:
            return _causal_error(
                _min_overlap_exceeds_window_message(
                    min_overlap=min_overlap,
                    window_bars=window_bars,
                ),
                code="invalid_input",
                meta=meta,
            )

        method_value = _normalize_correlation_method(method)
        if method_value is None:
            return _causal_error(
                "Invalid method. Valid options: pearson, spearman",
                code="invalid_method",
                meta=meta,
            )
        meta["method"] = method_value

        transform_value = _normalize_transform_name(transform)
        if transform_value is None:
            return _causal_error(
                "Invalid transform. Valid options: log_return, pct, diff, level, log_level",
                code="invalid_transform",
                meta=meta,
            )
        meta["transform"] = transform_value
        requested_detail = str(detail or "compact").strip().lower()
        detail_mode = normalize_output_verbosity_detail(detail, default="compact")
        if requested_detail not in {"compact", "standard", "summary", "full"}:
            return _causal_error(
                "detail must be one of: compact, standard, summary, full.",
                code="invalid_input",
                meta=meta,
            )
        meta["detail"] = detail_mode

        fetch_count = max(int(window_bars) + 10, int(min_overlap) + 10, 200)
        meta["fetch_count"] = int(fetch_count)
        series_map: Dict[str, pd.Series] = {}
        errors: List[str] = []
        for symbol_name in symbol_list:
            series, err = _fetch_series_for_window(
                symbol_name,
                tf,
                fetch_count,
                start=start,
                end=end,
                timeframe_key=str(timeframe),
                include_incomplete=bool(include_incomplete),
            )
            if err:
                errors.append(err)
            else:
                series_map[symbol_name] = series

        if errors and not series_map:
            return _causal_error(
                errors[0],
                code="data_fetch_failed",
                meta=meta,
                details=errors,
            )

        warnings_out: List[str] = []
        if errors:
            warnings_out.extend(errors)
            symbol_list = [symbol for symbol in symbol_list if symbol in series_map]

        if requested_anchor and requested_anchor not in series_map:
            details_out = []
            expanded_symbols = meta.get("symbols_expanded")
            if isinstance(expanded_symbols, list) and expanded_symbols:
                details_out.append(
                    f"Expanded group: {', '.join(str(sym) for sym in expanded_symbols)}"
                )
            return _causal_error(
                f"Requested symbol {requested_anchor} could not be fetched from its auto-expanded group.",
                code="anchor_symbol_missing",
                meta=meta,
                warnings=warnings_out,
                details=details_out or None,
            )

        if len(series_map) < 2:
            return _causal_error(
                "Not enough valid symbol data fetched to calculate correlations.",
                code="insufficient_symbols",
                meta=meta,
                warnings=warnings_out,
            )

        symbol_rows: Dict[str, int] = {
            str(symbol): int(len(series_map.get(symbol, pd.Series(dtype=float))))
            for symbol in symbol_list
            if symbol in series_map
        }
        if symbol_rows:
            meta["symbol_rows"] = symbol_rows

        frame = _build_pairwise_frame(series_map, symbol_list)
        if frame.empty:
            return _causal_error(
                "Not enough valid symbol data fetched to calculate correlations.",
                code="insufficient_symbols",
                meta=meta,
                warnings=warnings_out,
            )

        try:
            transformed = _transform_frame(frame, transform_value)
            transformed = transformed.dropna(axis=1, how="all")
            symbols_used = [
                symbol for symbol in symbol_list if symbol in transformed.columns
            ]
            transformed = transformed.reindex(columns=symbols_used)
        except (TypeError, ValueError) as exc:
            return _causal_error(
                "Correlation preprocessing failed. Ensure fetched series contain numeric price data with unique symbol columns.",
                code="invalid_input",
                meta=meta,
                warnings=warnings_out,
                details=[str(exc)],
            )
        meta["group_hint"] = group_hint
        meta["symbols_used"] = list(symbols_used)

        transformed_rows = {
            str(symbol): int(transformed[symbol].dropna().shape[0])
            for symbol in symbols_used
        }
        if transformed_rows:
            meta["symbol_rows_after_transform"] = transformed_rows

        if len(symbols_used) < 2:
            return _causal_error(
                "Not enough symbols retained after transform to calculate correlations.",
                code="insufficient_symbols",
                meta=meta,
                warnings=warnings_out,
            )

        rows, pair_overlaps, skipped = _rank_correlation_pairs(
            transformed,
            symbols_used,
            method=method_value,
            window_bars=int(window_bars),
            min_overlap=int(min_overlap),
        )
        output_rows_raw, output_truncated, pagination = _limit_pair_rows(
            rows,
            output_limit,
            output_offset,
        )
        meta.update(
            {
                "pairs_attempted": int(
                    max(len(symbols_used) * (len(symbols_used) - 1) // 2, 0)
                ),
                "pairs_computed": int(len(rows)),
                "output_truncated": output_truncated,
                "pairs_skipped_min_overlap": int(skipped["min_overlap"]),
                "pairs_skipped_nonfinite": int(skipped["nonfinite"]),
                "pair_overlaps": pair_overlaps,
            }
        )

        if not rows:
            return _causal_error(
                "No symbol pairs had enough overlapping transformed samples to compute correlations.",
                code="insufficient_overlap",
                meta=meta,
                warnings=warnings_out,
                details=_format_pair_overlap_details(pair_overlaps, int(min_overlap))
                or None,
            )

        output_rows = (
            [_public_pair_row(row) for row in output_rows_raw]
            if detail_mode == "full"
            else _compact_correlation_rows(output_rows_raw)
        )
        highlights = (
            _build_correlation_summary(output_rows_raw)
            if detail_mode in {"standard", "full"}
            else {}
        )
        context = {
            **_pairwise_analysis_context(rows, timeframe=timeframe),
            "limit": output_limit,
            "window_bars": int(window_bars),
            "start": start,
            "end": end,
            "transform": transform_value,
            "min_overlap": int(min_overlap),
            **_bar_completion_context(
                series_map, include_incomplete=bool(include_incomplete)
            ),
        }
        alignment_context, alignment_warning = _pairwise_period_alignment(
            rows,
            timeframe=timeframe,
        )
        if alignment_context:
            context.update(alignment_context)
        if alignment_warning:
            warnings_out.append(alignment_warning)
        out: Dict[str, Any] = {
            "success": True,
            "transform": transform_value,
            **_pair_transform_guidance(
                "correlation_matrix",
                transform_value,
                detail=requested_detail,
            ),
            "items": output_rows,
            "count": int(len(output_rows_raw)),
            **pagination,
            "context": context,
            "summary": {
                "counts": {
                    "pairs": int(len(output_rows_raw)),
                },
                "highlights": highlights,
            },
            "meta": _causal_contract_meta(meta),
        }
        if detail_mode in {"standard", "full"}:
            out["context"]["transform_note"] = (
                "Correlation defaults to log_return; cointegration defaults to log_level because it tests price-level relationships."
            )
        if output_truncated:
            out["truncated"] = True
        if detail_mode == "full":
            out["matrix"] = _build_correlation_matrix(symbols_used, output_rows_raw)
        if warnings_out:
            out["warnings"] = warnings_out
        return out

    return run_logged_operation(
        logger,
        operation="correlation_matrix",
        symbols=symbols,
        group=group,
        timeframe=timeframe,
        limit=limit,
        window_bars=window_bars,
        start=start,
        end=end,
        method=method,
        transform=transform,
        min_overlap=min_overlap,
        include_incomplete=include_incomplete,
        detail=detail,
        func=_run,
    )


@mcp.tool()
def cross_correlation(  # noqa: C901
    symbols: str,
    timeframe: TimeframeLiteral = "H1",
    window_bars: int = 500,
    start: Optional[str] = None,
    end: Optional[str] = None,
    max_lag: int = 20,
    method: str = "pearson",
    transform: str = "log_return",
    min_overlap: int = 50,
    bootstrap_samples: int = 300,
    include_incomplete: bool = False,
    detail: DetailLiteral = "compact",
) -> Dict[str, Any]:
    """Measure lead-lag correlation for an explicit pair of MT5 symbols.

    Positive lag means the first symbol leads the second by that many bars;
    negative lag means the second symbol leads the first.
    """

    def _run() -> Dict[str, Any]:  # noqa: C901
        meta: Dict[str, Any] = {
            "_tool": "cross_correlation",
            "_request_keys": _CROSS_CORRELATION_REQUEST_KEYS,
            "symbols_input": symbols,
            "timeframe": timeframe,
            "window_bars": int(window_bars),
            "start": start,
            "end": end,
            "max_lag": int(max_lag),
            "method": method,
            "transform": transform,
            "min_overlap": int(min_overlap),
            "bootstrap_samples": int(bootstrap_samples),
            "include_incomplete": bool(include_incomplete),
            "detail": detail,
        }
        connection_error = _causal_connection_error()
        if connection_error is not None:
            return _causal_error(
                str(connection_error.get("error") or "Failed to connect to MetaTrader5."),
                code=str(connection_error.get("error_code") or "mt5_connection_error"),
                meta=meta,
            )
        symbol_list = _parse_symbols(symbols)
        if len(symbol_list) != 2:
            return _causal_error(
                "cross_correlation requires exactly two comma-separated symbols.",
                code="invalid_input",
                meta=meta,
            )
        tf = TIMEFRAME_MAP.get(timeframe)
        if tf is None:
            return _causal_error(
                f"Invalid timeframe '{timeframe}'.",
                code="invalid_timeframe",
                meta=meta,
            )
        if int(window_bars) < 10 or int(min_overlap) < 5:
            return _causal_error(
                "window_bars must be >= 10 and min_overlap must be >= 5.",
                code="invalid_input",
                meta=meta,
            )
        if int(max_lag) < 0 or int(max_lag) >= int(window_bars):
            return _causal_error(
                "max_lag must be >= 0 and less than window_bars.",
                code="invalid_input",
                meta=meta,
            )
        if not 20 <= int(bootstrap_samples) <= 2000:
            return _causal_error(
                "bootstrap_samples must be between 20 and 2000.",
                code="invalid_input",
                meta=meta,
            )
        method_value = _normalize_correlation_method(method)
        if method_value is None:
            return _causal_error(
                "Invalid method. Valid options: pearson, spearman",
                code="invalid_method",
                meta=meta,
            )
        transform_value = _normalize_transform_name(transform)
        if transform_value is None:
            return _causal_error(
                "Invalid transform. Valid options: log_return, pct, diff, level, log_level",
                code="invalid_transform",
                meta=meta,
            )
        detail_mode = normalize_output_verbosity_detail(detail, default="compact")
        fetch_count = max(int(window_bars) + int(max_lag) + 10, int(min_overlap) + 10)
        series_map: Dict[str, pd.Series] = {}
        errors: List[str] = []
        for symbol_name in symbol_list:
            series, fetch_error = _fetch_series_for_window(
                symbol_name,
                tf,
                fetch_count,
                start=start,
                end=end,
                timeframe_key=str(timeframe),
                include_incomplete=bool(include_incomplete),
            )
            if fetch_error:
                errors.append(fetch_error)
            else:
                series_map[symbol_name] = series
        if errors:
            return _causal_error(
                errors[0],
                code="data_fetch_failed",
                meta=meta,
                details=errors,
            )
        frame = _build_pairwise_frame(series_map, symbol_list)
        transformed = _transform_frame(frame, transform_value)
        aligned = transformed[symbol_list].dropna(how="any").tail(int(window_bars))
        if len(aligned) < int(min_overlap):
            return _causal_error(
                f"Only {len(aligned)} overlapping samples are available; min_overlap={min_overlap}.",
                code="insufficient_overlap",
                meta=meta,
            )
        left_values = aligned[symbol_list[0]].to_numpy(dtype=float)
        right_values = aligned[symbol_list[1]].to_numpy(dtype=float)
        rows: List[Dict[str, Any]] = []
        for lag in range(-int(max_lag), int(max_lag) + 1):
            lag_left, lag_right = _lagged_pair_values(left_values, right_values, lag)
            value = _correlation_value(lag_left, lag_right, method_value)
            if not math.isfinite(value) or lag_left.size < int(min_overlap):
                continue
            rows.append(
                {
                    "lag": int(lag),
                    "correlation": round(float(value), 6),
                    "samples": int(lag_left.size),
                }
            )
        if not rows:
            return _causal_error(
                "No lag had enough finite overlapping samples.",
                code="insufficient_overlap",
                meta=meta,
            )
        best = max(rows, key=lambda row: abs(float(row["correlation"])))
        selected_left, selected_right = _lagged_pair_values(
            left_values,
            right_values,
            int(best["lag"]),
        )
        block_size = max(2, int(round(math.sqrt(selected_left.size))))
        lag_tests = len(rows)
        familywise_confidence = 0.95
        per_lag_confidence = 1.0 - ((1.0 - familywise_confidence) / lag_tests)
        ci_low, ci_high = _block_bootstrap_correlation_ci(
            selected_left,
            selected_right,
            method=method_value,
            samples=int(bootstrap_samples),
            block_size=block_size,
            confidence=per_lag_confidence,
        )
        best_item = dict(best)
        best_item.update(
            {
                "leader": symbol_list[0] if int(best["lag"]) > 0 else symbol_list[1] if int(best["lag"]) < 0 else None,
                "follower": symbol_list[1] if int(best["lag"]) > 0 else symbol_list[0] if int(best["lag"]) < 0 else None,
                "ci_low": round(ci_low, 6) if ci_low is not None else None,
                "ci_high": round(ci_high, 6) if ci_high is not None else None,
                "significant": bool(ci_low is not None and ci_high is not None and (ci_low > 0.0 or ci_high < 0.0)),
            }
        )
        out: Dict[str, Any] = {
            "success": True,
            "symbols": symbol_list,
            "timeframe": timeframe,
            "transform": transform_value,
            "method": method_value,
            "best": best_item,
            "lag_convention": "positive lag means the first symbol leads the second",
            "context": {
                "window_bars": int(window_bars),
                "samples_aligned": int(len(aligned)),
                "max_lag": int(max_lag),
                "bootstrap_samples": int(bootstrap_samples),
                "bootstrap_block_size": int(block_size),
                "lag_tests": int(lag_tests),
                "ci_familywise_confidence": familywise_confidence,
                "ci_per_lag_confidence": round(per_lag_confidence, 8),
                "significance_correction": "bonferroni_across_lags",
                **_bar_completion_context(
                    series_map, include_incomplete=bool(include_incomplete)
                ),
            },
            "meta": _causal_contract_meta(meta),
        }
        if detail_mode == "full":
            out["items"] = rows
            out["count"] = len(rows)
        return out

    return run_logged_operation(
        logger,
        operation="cross_correlation",
        symbols=symbols,
        timeframe=timeframe,
        window_bars=window_bars,
        max_lag=max_lag,
        include_incomplete=include_incomplete,
        method=method,
        transform=transform,
        func=_run,
    )


@mcp.tool()
def cointegration_test(  # noqa: C901
    symbols: Optional[str] = None,
    group: Optional[str] = None,
    timeframe: TimeframeLiteral = "H1",
    limit: Optional[int] = None,
    offset: int = 0,
    window_bars: int = 500,
    start: Optional[str] = None,
    end: Optional[str] = None,
    transform: str = "log_level",
    method: Literal["engle_granger", "johansen"] = "engle_granger",
    trend: str = "c",
    k_ar_diff: int = 1,
    significance: float = 0.05,
    min_overlap: int = 80,
    include_incomplete: bool = False,
    detail: DetailLiteral = "compact",
) -> Dict[str, Any]:
    """Run Engle-Granger pair tests or a multivariate Johansen rank test.

    When a single symbol is provided, the tool automatically expands to include
    all related symbols from its MT5 group (e.g., EURUSD → EURUSD, GBPUSD, 
    USDCHF, USDJPY, USDCAD, AUDUSD). This enables cointegration analysis across
    related pairs. To test cointegration for specific symbols only, provide
    multiple symbols explicitly or use the `group` parameter.

    Args:
        symbols: Comma-separated MT5 symbols; a single symbol auto-expands to
            its entire MT5 group (e.g. "EURUSD" → all Forex majors).
            Optional when using `group`.
        group: Explicit MT5 group path (for example "Forex\\Majors"). Mutually
            exclusive with `symbols`.
        timeframe: MT5 timeframe key (e.g. "M15", "H1").
        limit: Optional maximum number of ranked pair rows returned.
        window_bars: Maximum number of overlapping transformed samples used per
            pair after applying any time window.
        start: Optional UTC-compatible start date/time for the analysis window.
        end: Optional UTC-compatible end date/time; end-only anchors recent history.
        transform: Price transform: "log_level" or "level".
        method: "engle_granger" for pairwise tests or "johansen" for one
            multivariate cointegration-rank test across all retained symbols.
        trend: Deterministic trend term for the test: "c", "ct", "ctt", or "n".
        k_ar_diff: Number of lagged differences for the Johansen test.
        significance: Alpha threshold for reporting cointegrated pairs.
            Johansen supports only 0.01, 0.05, or 0.1 because its critical
            value tables contain only those three levels.
        min_overlap: Minimum overlapping transformed samples required per pair;
            values above window_bars are capped to window_bars with a warning.
        include_incomplete: Include the current forming candle. Defaults to false.
        detail: "compact" keeps pair results concise; "full" adds overlap/window
            diagnostics and legends.
    """

    def _run() -> Dict[str, Any]:  # noqa: C901
        window_bars_value = int(window_bars)
        min_overlap_value = int(min_overlap)
        warnings_out: List[str] = []
        meta: Dict[str, Any] = {
            "_tool": "cointegration_test",
            "_request_keys": _COINTEGRATION_REQUEST_KEYS,
            "timeframe": str(timeframe),
            "limit": limit,
            "offset": int(offset),
            "window_bars": window_bars_value,
            "start": start,
            "end": end,
            "transform": str(transform),
            "method": str(method),
            "trend": str(trend),
            "k_ar_diff": int(k_ar_diff),
            "significance": float(significance),
            "min_overlap": min_overlap_value,
            "include_incomplete": bool(include_incomplete),
            "detail": str(detail or "compact"),
        }
        connection_error = _causal_connection_error()
        if connection_error is not None:
            return _causal_error(
                str(connection_error.get("error") or "Failed to connect to MetaTrader5."),
                code=str(connection_error.get("error_code") or "mt5_connection_error"),
                meta=meta,
            )
        mt5_gateway = create_mt5_gateway(
            adapter=mt5,
            ensure_connection_impl=ensure_mt5_connection_or_raise,
        )

        try:
            from statsmodels.tsa.stattools import coint
            from statsmodels.tsa.vector_ar.vecm import coint_johansen
        except Exception:
            return _causal_error(
                "statsmodels is required for cointegration testing. Please install 'statsmodels'.",
                code="dependency_missing",
                meta=meta,
            )

        method_value = str(method or "engle_granger").strip().lower()
        if method_value not in {"engle_granger", "johansen"}:
            return _causal_error(
                "Invalid method. Valid options: engle_granger, johansen",
                code="invalid_method",
                meta=meta,
            )
        meta["method"] = method_value
        if int(k_ar_diff) < 0:
            return _causal_error(
                "k_ar_diff must be >= 0.",
                code="invalid_input",
                meta=meta,
            )

        symbol_list = _parse_symbols(symbols)
        if symbol_list:
            meta["symbols_input"] = list(symbol_list)
        if group is not None:
            meta["group_input"] = str(group)
        requested_anchor = (
            symbol_list[0] if group is None and len(symbol_list) == 1 else None
        )
        group_hint: str | None = None
        if group and symbol_list:
            return _causal_error(
                "Provide either symbols or group for cointegration testing, not both.",
                code="invalid_input",
                meta=meta,
            )
        if group:
            expanded, err, group_path = _expand_symbols_for_group_path(
                group,
                gateway=mt5_gateway,
            )
            if err:
                return _causal_error(
                    err,
                    code="symbol_group_error",
                    meta=meta,
                )
            symbol_list = expanded
            group_hint = group_path
            meta["group_resolved"] = group_path
            meta["symbols_expanded"] = list(symbol_list)
        elif not symbol_list:
            return _causal_error(
                "Provide at least one symbol or MT5 group for cointegration testing.",
                code="invalid_input",
                meta=meta,
            )
        elif len(symbol_list) == 1:
            expanded, err, group_path = _expand_symbols_for_group(
                symbol_list[0],
                gateway=mt5_gateway,
            )
            if err:
                return _causal_error(
                    err,
                    code="symbol_group_error",
                    meta=meta,
                )
            symbol_list = expanded
            group_hint = group_path
            meta["symbols_expanded"] = list(symbol_list)

        if len(symbol_list) < 2:
            return _causal_error(
                "Provide at least two symbols for cointegration testing (e.g. 'EURUSD,GBPUSD').",
                code="invalid_input",
                meta=meta,
            )

        tf = TIMEFRAME_MAP.get(timeframe)
        if tf is None:
            valid = ", ".join(sorted(TIMEFRAME_MAP.keys()))
            return _causal_error(
                f"Invalid timeframe '{timeframe}'. Valid options: {valid}",
                code="invalid_timeframe",
                meta=meta,
            )

        output_limit, limit_error = _normalize_output_limit(limit)
        if limit_error is not None:
            return _causal_error(
                limit_error,
                code="invalid_input",
                meta=meta,
            )
        output_offset, offset_error = _normalize_output_offset(offset)
        if offset_error is not None:
            return _causal_error(
                offset_error,
                code="invalid_input",
                meta=meta,
            )

        if window_bars_value < 2:
            return _causal_error(
                "window_bars must be at least 2.",
                code="invalid_input",
                meta=meta,
            )

        if min_overlap_value < 2:
            return _causal_error(
                "min_overlap must be at least 2.",
                code="invalid_input",
                meta=meta,
            )
        if window_bars_value < min_overlap_value:
            requested_min_overlap = min_overlap_value
            min_overlap_value = window_bars_value
            meta["min_overlap_requested"] = requested_min_overlap
            meta["min_overlap"] = min_overlap_value
            warnings_out.append(
                f"min_overlap adjusted from {requested_min_overlap} to {min_overlap_value} "
                "to match window_bars."
            )

        if not (0.0 < float(significance) < 1.0):
            return _causal_error(
                "significance must be between 0 and 1.",
                code="invalid_input",
                meta=meta,
            )

        requested_detail = str(detail or "compact").strip().lower()
        detail_mode = normalize_output_verbosity_detail(detail, default="compact")
        if requested_detail not in {"compact", "standard", "summary", "full"}:
            return _causal_error(
                "detail must be one of: compact, standard, summary, full.",
                code="invalid_input",
                meta=meta,
            )
        meta["detail"] = detail_mode

        transform_value = _normalize_cointegration_transform(transform)
        if transform_value is None:
            return _causal_error(
                "Invalid transform. Valid options: log_level, level",
                code="invalid_transform",
                meta=meta,
            )
        meta["transform"] = transform_value

        trend_value = _normalize_cointegration_trend(trend)
        if trend_value is None:
            return _causal_error(
                "Invalid trend. Valid options: c, ct, ctt, n",
                code="invalid_trend",
                meta=meta,
            )
        meta["trend"] = trend_value
        if method_value == "johansen" and trend_value == "ctt":
            return _causal_error(
                "Johansen supports trend values n, c, or ct; ctt is only available for Engle-Granger.",
                code="invalid_trend",
                meta=meta,
            )
        if method_value == "johansen" and float(significance) not in {0.01, 0.05, 0.1}:
            return _causal_error(
                "Johansen significance must be one of: 0.01, 0.05, 0.1.",
                code="invalid_input",
                meta=meta,
            )

        fetch_count = max(window_bars_value + 10, min_overlap_value + 10, 200)
        meta["fetch_count"] = int(fetch_count)
        series_map: Dict[str, pd.Series] = {}
        errors: List[str] = []
        for symbol_name in symbol_list:
            series, err = _fetch_series_for_window(
                symbol_name,
                tf,
                fetch_count,
                start=start,
                end=end,
                timeframe_key=str(timeframe),
                include_incomplete=bool(include_incomplete),
            )
            if err:
                errors.append(err)
            else:
                series_map[symbol_name] = series

        if errors and not series_map:
            return _causal_error(
                errors[0],
                code="data_fetch_failed",
                meta=meta,
                warnings=warnings_out,
                details=errors,
            )

        if errors:
            warnings_out.extend(errors)
            symbol_list = [symbol for symbol in symbol_list if symbol in series_map]

        if requested_anchor and requested_anchor not in series_map:
            details_out = []
            expanded_symbols = meta.get("symbols_expanded")
            if isinstance(expanded_symbols, list) and expanded_symbols:
                details_out.append(
                    f"Expanded group: {', '.join(str(sym) for sym in expanded_symbols)}"
                )
            return _causal_error(
                f"Requested symbol {requested_anchor} could not be fetched from its auto-expanded group.",
                code="anchor_symbol_missing",
                meta=meta,
                warnings=warnings_out,
                details=details_out or None,
            )

        if len(series_map) < 2:
            return _causal_error(
                "Not enough valid symbol data fetched to run cointegration tests.",
                code="insufficient_symbols",
                meta=meta,
                warnings=warnings_out,
            )

        symbol_rows: Dict[str, int] = {
            str(symbol): int(len(series_map.get(symbol, pd.Series(dtype=float))))
            for symbol in symbol_list
            if symbol in series_map
        }
        if symbol_rows:
            meta["symbol_rows"] = symbol_rows

        frame = _build_pairwise_frame(series_map, symbol_list)
        if frame.empty:
            return _causal_error(
                "Not enough valid symbol data fetched to run cointegration tests.",
                code="insufficient_symbols",
                meta=meta,
                warnings=warnings_out,
            )

        transformed = _transform_cointegration_frame(frame, transform_value)
        transformed = transformed.dropna(axis=1, how="all")
        symbols_used = [
            symbol for symbol in symbol_list if symbol in transformed.columns
        ]
        transformed = transformed.reindex(columns=symbols_used)
        meta["group_hint"] = group_hint
        meta["symbols_used"] = list(symbols_used)

        transformed_rows = {
            str(symbol): int(transformed[symbol].dropna().shape[0])
            for symbol in symbols_used
        }
        if transformed_rows:
            meta["symbol_rows_after_transform"] = transformed_rows

        if len(symbols_used) < 2:
            return _causal_error(
                "Not enough symbols retained after transform to run cointegration tests.",
                code="insufficient_symbols",
                meta=meta,
                warnings=warnings_out,
            )

        if method_value == "johansen":
            complete = transformed[symbols_used].dropna(how="any")
            available_overlap = int(len(complete))
            if available_overlap < min_overlap_value:
                return _causal_error(
                    f"Only {available_overlap} complete multivariate observations are available; min_overlap={min_overlap_value}.",
                    code="insufficient_overlap",
                    meta=meta,
                    warnings=warnings_out,
                )
            sample = complete.tail(window_bars_value)
            if int(len(sample)) <= int(k_ar_diff) + len(symbols_used) + 1:
                return _causal_error(
                    "The Johansen test needs more observations than symbols plus lagged differences.",
                    code="insufficient_overlap",
                    meta=meta,
                    warnings=warnings_out,
                )
            det_order = {"n": -1, "c": 0, "ct": 1}[trend_value]
            try:
                johansen = coint_johansen(
                    sample.to_numpy(dtype=float),
                    det_order,
                    int(k_ar_diff),
                )
            except Exception as exc:
                return _causal_error(
                    "Johansen cointegration test failed.",
                    code="test_failed",
                    meta=meta,
                    warnings=warnings_out,
                    details=[str(exc)],
                )
            significance_column = {0.1: 0, 0.05: 1, 0.01: 2}[float(significance)]
            trace_rank = _johansen_rank(
                johansen.trace_stat,
                johansen.trace_stat_crit_vals,
                significance_column,
            )
            max_eig_rank = _johansen_rank(
                johansen.max_eig_stat,
                johansen.max_eig_stat_crit_vals,
                significance_column,
            )
            selected_rank = min(trace_rank, max_eig_rank)
            rank_rows: List[Dict[str, Any]] = []
            for rank_index in range(len(symbols_used)):
                rank_rows.append(
                    {
                        "rank_null": int(rank_index),
                        "trace_statistic": round(float(johansen.trace_stat[rank_index]), 6),
                        "trace_critical_value": round(float(johansen.trace_stat_crit_vals[rank_index][significance_column]), 6),
                        "max_eigen_statistic": round(float(johansen.max_eig_stat[rank_index]), 6),
                        "max_eigen_critical_value": round(float(johansen.max_eig_stat_crit_vals[rank_index][significance_column]), 6),
                    }
                )
            vectors: List[Dict[str, Any]] = []
            for vector_index in range(min(int(selected_rank), int(johansen.evec.shape[1]))):
                coefficients = johansen.evec[:, vector_index]
                scale = float(np.max(np.abs(coefficients)))
                if scale <= 0.0 or not np.isfinite(scale):
                    scale = 1.0
                vectors.append(
                    {
                        "vector": int(vector_index + 1),
                        "coefficients": {
                            symbol_name: round(float(coefficients[idx] / scale), 8)
                            for idx, symbol_name in enumerate(symbols_used)
                        },
                    }
                )
            out: Dict[str, Any] = {
                "success": True,
                "method": "johansen",
                "transform": transform_value,
                "symbols": symbols_used,
                "cointegration_rank": int(selected_rank),
                "trace_rank": int(trace_rank),
                "max_eigen_rank": int(max_eig_rank),
                "cointegrated": bool(selected_rank > 0),
                "items": rank_rows,
                "count": len(rank_rows),
                "cointegrating_vectors": vectors,
                "context": {
                    **_pairwise_analysis_context([], timeframe=timeframe),
                    "window_bars": window_bars_value,
                    "samples": int(len(sample)),
                    "available_overlap_rows": available_overlap,
                    "transform": transform_value,
                    "trend": trend_value,
                    "det_order": int(det_order),
                    "k_ar_diff": int(k_ar_diff),
                    "significance": float(significance),
                },
                "summary": {
                    "selected_rank": int(selected_rank),
                    "independent_stochastic_trends": int(max(0, len(symbols_used) - selected_rank)),
                    "rank_agreement": bool(trace_rank == max_eig_rank),
                },
                "meta": _causal_contract_meta(meta),
            }
            if warnings_out:
                out["warnings"] = warnings_out
            return out

        rows: List[Dict[str, Any]] = []
        pair_overlaps: Dict[str, int] = {}
        pair_failures: List[Dict[str, Any]] = []
        pairs_skipped_min_overlap = 0

        for idx, left in enumerate(symbols_used):
            for right in symbols_used[idx + 1 :]:
                subset_all = transformed[[left, right]].dropna(how="any")
                overlap_rows = int(len(subset_all))
                pair_overlaps[f"{left}-{right}"] = overlap_rows
                if overlap_rows < min_overlap_value:
                    pairs_skipped_min_overlap += 1
                    continue
                subset = subset_all.tail(window_bars_value)
                row, failures = _evaluate_cointegration_pair(
                    subset,
                    left,
                    right,
                    trend=trend_value,
                    significance=float(significance),
                    coint_func=coint,
                )
                if row is not None:
                    if detail_mode == "full":
                        row["overlap_rows"] = overlap_rows
                        row["aligned_observations"] = overlap_rows
                        row["available_overlap_rows"] = overlap_rows
                        row["calculation_samples"] = int(len(subset))
                        row["window_requested"] = window_bars_value
                        row["window_actual"] = int(len(subset))
                        row["window_truncated"] = bool(len(subset) < overlap_rows)
                    rows.append(row)
                if failures:
                    for failure in failures:
                        if len(pair_failures) < 10:
                            pair_failures.append(failure)

        _apply_holm_pair_correction(rows, significance=float(significance))

        rows.sort(
            key=lambda item: (
                float(item["p_value"]),
                -int(item["samples"]),
                str(item["left"]),
                str(item["right"]),
            )
        )
        output_rows_raw, output_truncated, pagination = _limit_pair_rows(
            rows,
            output_limit,
            output_offset,
        )
        meta.update(
            {
                "pairs_attempted": int(
                    max(len(symbols_used) * (len(symbols_used) - 1) // 2, 0)
                ),
                "pairs_tested": int(len(rows)),
                "output_truncated": output_truncated,
                "pairs_failed": int(len(pair_failures)),
                "pairs_skipped_min_overlap": int(pairs_skipped_min_overlap),
                "p_value_correction": "holm_across_pairs",
                "pair_tests_run": int(len(rows)),
            }
        )
        if detail_mode == "full":
            meta["pair_overlaps"] = pair_overlaps
            meta["window_interpretation"] = (
                "calculation_samples/window_actual is the capped sample count used in each test; "
                "aligned_observations/available_overlap_rows is the full pairwise overlap before "
                "the limit window is applied."
            )
        if pair_failures:
            meta["pair_failures"] = pair_failures
            warnings_out.append(
                f"{len(pair_failures)} orientation-level cointegration fits failed; see meta['pair_failures']."
            )

        if not rows:
            error_code = "insufficient_overlap"
            error_message = "No symbol pairs had enough overlapping transformed samples to run cointegration tests."
            details = (
                _format_pair_overlap_details(pair_overlaps, min_overlap_value) or None
            )
            if pair_failures and any(
                rows_count >= min_overlap_value for rows_count in pair_overlaps.values()
            ):
                error_code = "test_failed"
                error_message = (
                    "Cointegration tests failed for all eligible symbol pairs."
                )
            return _causal_error(
                error_message,
                code=error_code,
                meta=meta,
                warnings=warnings_out,
                details=details,
            )

        cointegrated_count = int(
            sum(1 for row in rows if bool(row.get("cointegrated")))
        )
        # Build transform legend for cointegration (uses different transforms)
        cointegration_transform_legend = {
            "level": _TRANSFORM_LEGEND["level"],
            "log_level": {
                "description": "Natural log of price levels",
                "formula": "ln(close_t)",
                "use_case": "Reduces scale effects while preserving cointegration relationships; common for price ratios",
            },
        }

        out: Dict[str, Any] = {
            "success": True,
            "transform": transform_value,
            **_pair_transform_guidance(
                "cointegration_test",
                transform_value,
                detail=requested_detail,
            ),
            "items": [_public_pair_row(row) for row in output_rows_raw],
            "count": int(len(output_rows_raw)),
            **pagination,
            "summary": {
                "counts": {
                    "pairs": int(len(output_rows_raw)),
                    "cointegrated": int(
                        sum(1 for row in output_rows_raw if bool(row.get("cointegrated")))
                    ),
                },
                "highlights": _build_cointegration_summary(output_rows_raw),
            },
            "context": {
                **_pairwise_analysis_context(rows, timeframe=timeframe),
                "limit": output_limit,
                "window_bars": window_bars_value,
                "start": start,
                "end": end,
                "transform": transform_value,
                "trend": trend_value,
                "min_overlap": min_overlap_value,
                **_bar_completion_context(
                    series_map, include_incomplete=bool(include_incomplete)
                ),
            },
            "meta": _causal_contract_meta(
                meta,
                legends=(
                    {
                        "transform": cointegration_transform_legend,
                        "trend": _COINTEGRATION_TREND_LEGEND,
                        "cointegration": {
                            "description": "Long-term equilibrium relationship between non-stationary price series",
                            "cointegrated_true": "Series share a common stochastic drift - deviations are mean-reverting",
                            "cointegrated_false": "No statistically significant long-term relationship detected",
                            "test_statistic": "Engle-Granger test statistic; more negative = stronger evidence of cointegration",
                            "critical_values": "Thresholds at 1%, 5%, 10% significance levels; test statistic < critical value indicates cointegration",
                        },
                        "hedge_ratio": "Units of quote symbol needed to hedge one unit of base symbol in a pairs trade",
                    }
                    if detail_mode == "full"
                    else None
                ),
            ),
        }
        if detail_mode in {"standard", "full"}:
            out["context"]["transform_note"] = (
                "Cointegration defaults to log_level; correlation defaults to log_return because it measures co-movement in returns."
            )
        if output_truncated:
            out["truncated"] = True
        if warnings_out:
            out["warnings"] = warnings_out
        if cointegrated_count == 0:
            out["message"] = (
                "No statistically significant cointegrated pairs detected at the selected threshold."
            )
        return out

    return run_logged_operation(
        logger,
        operation="cointegration_test",
        symbols=symbols,
        group=group,
        timeframe=timeframe,
        limit=limit,
        window_bars=window_bars,
        start=start,
        end=end,
        transform=transform,
        method=method,
        trend=trend,
        k_ar_diff=k_ar_diff,
        significance=significance,
        include_incomplete=include_incomplete,
        min_overlap=min_overlap,
        detail=detail,
        func=_run,
    )
