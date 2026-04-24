"""Causal signal discovery tools."""

from __future__ import annotations

import contextlib
import io
import logging
import math
import time
import warnings
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..shared.parameter_contracts import normalize_symbol_selector_aliases
from ..shared.schema import CompactFullDetailLiteral, TimeframeLiteral
from ..utils.mt5 import (
    _ensure_symbol_ready,
    _mt5_copy_rates_from,
    ensure_mt5_connection_or_raise,
    mt5,
)
from ..utils.symbol import _extract_group_path as _extract_group_path_util
from ._mcp_instance import mcp
from .constants import TIMEFRAME_MAP
from .execution_logging import run_logged_operation
from .mt5_gateway import get_mt5_gateway, mt5_connection_error

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
        "symbol_input",
        "symbols_input",
        "symbols_expanded",
        "timeframe",
        "limit",
        "max_lag",
        "significance",
        "transform",
        "normalize",
    }
)

_CORRELATION_REQUEST_KEYS = frozenset(
    {
        "symbol_input",
        "symbols_input",
        "symbols_expanded",
        "group_input",
        "group_resolved",
        "timeframe",
        "limit",
        "method",
        "transform",
        "min_overlap",
        "detail",
    }
)

_COINTEGRATION_REQUEST_KEYS = frozenset(
    {
        "symbol_input",
        "symbols_input",
        "symbols_expanded",
        "group_input",
        "group_resolved",
        "timeframe",
        "limit",
        "transform",
        "trend",
        "significance",
        "min_overlap",
    }
)


def _causal_connection_error() -> Dict[str, Any] | None:
    return mt5_connection_error(
        get_mt5_gateway(
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
    mt5_gateway = gateway or get_mt5_gateway(
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
    mt5_gateway = gateway or get_mt5_gateway(
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

    query_lower = group_query.lower()
    exact_matches = [
        group_path for group_path in groups if group_path.lower() == query_lower
    ]
    matched_paths = exact_matches or [
        group_path for group_path in groups if query_lower in group_path.lower()
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


def _fetch_series(
    symbol: str, timeframe, count: int, retries: int = 3, pause: float = 0.25
) -> Tuple[pd.Series, str | None]:
    """Fetch recent close prices for a symbol."""
    err = _ensure_symbol_ready(symbol)
    if err:
        return pd.Series(dtype=float), err

    for _attempt in range(retries):
        utc_now = datetime.now(timezone.utc)
        data = _mt5_copy_rates_from(symbol, timeframe, utc_now, count)
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
        if len(df) > count:
            df = df.tail(count)
        series = pd.Series(
            df["close"].to_numpy(dtype=float),
            index=pd.to_datetime(df["time"], unit="s"),
        )
        series = series[~series.index.duplicated(keep="last")]
        return series, None
    return pd.Series(dtype=float), f"Failed to fetch data for {symbol}" + (
        f" after {retries} retries" if retries > 1 else ""
    )


def _transform_frame(frame: pd.DataFrame, transform: str) -> pd.DataFrame:
    transform = transform.strip().lower()
    if transform in ("log_return", "logret", "log-returns"):
        # log return r_t = ln(p_t) - ln(p_{t-1})
        clean = frame.astype(float).where(frame > 0)
        clean = clean.mask(clean <= 0)
        log_prices = np.log(clean)
        log_prices = log_prices.replace([np.inf, -np.inf], np.nan)
        frame = log_prices.diff()
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


def _rank_correlation_pairs(
    frame: pd.DataFrame,
    symbols: List[str],
    *,
    method: str,
    limit: int,
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
            subset = subset_all.tail(limit)
            corr = subset[left].corr(subset[right], method=method)
            if corr is None or not math.isfinite(float(corr)):
                skipped["nonfinite"] += 1
                continue
            corr_f = float(corr)
            rows.append(
                {
                    "left": left,
                    "right": right,
                    "correlation": corr_f,
                    "abs_correlation": abs(corr_f),
                    "samples": int(len(subset)),
                    "overlap_rows": overlap_rows,
                    "window_requested": int(limit),
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


def _build_correlation_summary(
    rows: List[Dict[str, Any]],
    *,
    top_n: int = 5,
) -> Dict[str, List[Dict[str, Any]]]:
    limit = max(1, int(top_n))
    if len(rows) <= limit:
        return {}
    positive = sorted(
        [row for row in rows if float(row["correlation"]) > 0.0],
        key=lambda item: (
            -float(item["correlation"]),
            -int(item["samples"]),
            str(item["left"]),
            str(item["right"]),
        ),
    )
    negative = sorted(
        [row for row in rows if float(row["correlation"]) < 0.0],
        key=lambda item: (
            float(item["correlation"]),
            -int(item["samples"]),
            str(item["left"]),
            str(item["right"]),
        ),
    )
    return {
        "strongest_absolute": rows[:limit],
        "strongest_positive": positive[:limit],
        "strongest_negative": negative[:limit],
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
    if trend == "n":
        design = x.reshape(-1, 1)
    else:
        design = np.column_stack([x, np.ones(x.size, dtype=float)])
    coeffs, *_ = np.linalg.lstsq(design, y, rcond=None)
    if coeffs.size < 1 or not math.isfinite(float(coeffs[0])):
        return None, None, None
    beta = float(coeffs[0])
    intercept = 0.0 if trend == "n" else float(coeffs[1]) if coeffs.size > 1 else 0.0
    spread = y - (beta * x + intercept)
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

    for dependent, hedge in ((left, right), (right, left)):
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
            "cointegrated": bool(float(p_value) < significance),
            "relationship": "cointegrated"
            if float(p_value) < significance
            else "no_cointegration",
        }
        if best_row is None or float(row["p_value"]) < float(best_row["p_value"]):
            best_row = row

    return best_row, failures


def _build_cointegration_summary(
    rows: List[Dict[str, Any]],
    *,
    top_n: int = 5,
) -> Dict[str, List[Dict[str, Any]]]:
    limit = max(1, int(top_n))
    cointegrated = [row for row in rows if bool(row.get("cointegrated"))]
    return {
        "best_pairs": rows[:limit],
        "cointegrated_pairs": cointegrated[:limit],
    }


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
        "Displayed p-values use Bonferroni correction across tested lags for each pair."
    )
    lines.append("Results are pairwise and do not imply full causal graphs.")
    return "\n".join(lines)


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


def _select_prune_symbol(
    symbols: List[str],
    pair_overlaps: Dict[str, int],
    *,
    bottleneck_pair: str,
    preserve_symbol: str | None = None,
) -> str | None:
    candidates = [
        symbol
        for symbol in _pair_overlap_symbols(bottleneck_pair, symbols)
        if symbol in symbols
    ]
    if len(candidates) < 2:
        return None
    if preserve_symbol in candidates and len(candidates) > 1:
        prunable = [symbol for symbol in candidates if symbol != preserve_symbol]
        if prunable:
            candidates = prunable
    anchor = symbols[0] if symbols else None
    totals: Dict[str, int] = {symbol: 0 for symbol in candidates}
    for pair_key, overlap_rows in pair_overlaps.items():
        pair_symbols = _pair_overlap_symbols(pair_key, symbols)
        for symbol in candidates:
            if symbol in pair_symbols:
                totals[symbol] += int(overlap_rows)
    candidates.sort(
        key=lambda symbol: (
            totals.get(symbol, 0),
            1 if symbol == anchor else 0,
            -symbols.index(symbol),
        )
    )
    return candidates[0]


def _prune_symbols_for_overlap(
    series_map: Dict[str, pd.Series],
    symbols: List[str],
    *,
    limit: int,
    minimum_required: int,
    preserve_symbol: str | None = None,
) -> tuple[List[str], pd.DataFrame, Dict[str, Any] | None]:
    active_symbols = [symbol for symbol in symbols if symbol in series_map]
    frame = _build_overlap_frame(series_map, active_symbols, limit)
    if len(active_symbols) < 3 or len(frame) >= minimum_required:
        return active_symbols, frame, None

    dropped_symbols: List[str] = []
    iterations: List[Dict[str, Any]] = []
    while len(active_symbols) > 2 and len(frame) < minimum_required:
        pair_overlaps = _pair_overlap_counts(series_map, active_symbols)
        if not pair_overlaps:
            break
        bottleneck_pair, bottleneck_rows = min(
            pair_overlaps.items(), key=lambda kv: kv[1]
        )
        drop_symbol = _select_prune_symbol(
            active_symbols,
            pair_overlaps,
            bottleneck_pair=str(bottleneck_pair),
            preserve_symbol=preserve_symbol,
        )
        if not drop_symbol:
            break
        aligned_before = int(len(frame))
        active_symbols = [symbol for symbol in active_symbols if symbol != drop_symbol]
        frame = _build_overlap_frame(series_map, active_symbols, limit)
        dropped_symbols.append(drop_symbol)
        iterations.append(
            {
                "dropped_symbol": drop_symbol,
                "bottleneck_pair": str(bottleneck_pair),
                "bottleneck_rows": int(bottleneck_rows),
                "aligned_rows_before": aligned_before,
                "aligned_rows_after": int(len(frame)),
                "remaining_symbols": list(active_symbols),
            }
        )

    if not dropped_symbols:
        return active_symbols, frame, None
    return (
        active_symbols,
        frame,
        {
            "initial_symbols": list(symbols),
            "dropped_symbols": dropped_symbols,
            "kept_symbols": list(active_symbols),
            "aligned_rows_after_pruning": int(len(frame)),
            "minimum_required": int(minimum_required),
            "iterations": iterations,
        },
    )


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
    symbol: Optional[str] = None,
    timeframe: TimeframeLiteral = "H1",
    limit: int = 500,
    max_lag: int = 5,
    significance: float = 0.05,
    transform: str = "log_return",
    normalize: bool = True,
) -> Dict[str, Any]:
    """Run Granger-style causal discovery on MT5 symbols.

    Args:
        symbols: Comma-separated MT5 symbols; provide one symbol to auto-expand its group.
        symbol: Compatibility alias for `symbols`.
        timeframe: MT5 timeframe key (e.g. "M15", "H1").
        limit: Number of recent bars to analyse per symbol.
        max_lag: Maximum lag order for tests (>=1).
        significance: Alpha level for reporting causal links.
        transform: Preprocessing transform: "log_return", "pct", "diff", or "level".
        normalize: Z-score columns before testing to stabilise scale.
    """

    symbol_alias = symbol

    def _run() -> Dict[str, Any]:  # noqa: C901
        meta: Dict[str, Any] = {
            "_tool": "causal_discover_signals",
            "_request_keys": _CAUSAL_DISCOVER_REQUEST_KEYS,
            "timeframe": str(timeframe),
            "limit": int(limit),
            "max_lag": int(max_lag),
            "significance": float(significance),
            "transform": str(transform),
            "normalize": bool(normalize),
        }
        connection_error = _causal_connection_error()
        if connection_error is not None:
            return _causal_error(
                str(connection_error.get("error") or "Failed to connect to MetaTrader5."),
                code=str(connection_error.get("error_code") or "mt5_connection_error"),
                meta=meta,
            )
        mt5_gateway = get_mt5_gateway(
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

        symbol_list, selector_meta, selector_error = normalize_symbol_selector_aliases(
            symbol=symbol_alias,
            symbols=symbols,
            parse_selector=_parse_symbols,
        )
        meta.update(selector_meta)
        if selector_error is not None:
            return _causal_error(
                selector_error,
                code="invalid_input",
                meta=meta,
            )
        group_hint: str | None = None
        requested_anchor = symbol_list[0] if len(symbol_list) == 1 else None
        if not symbol_list:
            return _causal_error(
                "Provide at least one symbol for causal discovery (e.g. 'EURUSD' or 'EURUSD,GBPUSD').",
                code="invalid_input",
                meta=meta,
            )
        if len(symbol_list) == 1:
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

        fetch_count = max(limit + max_lag + 10, 200)
        meta["fetch_count"] = int(fetch_count)
        series_map: Dict[str, pd.Series] = {}
        errors: List[str] = []
        for symbol_name in symbol_list:
            series, err = _fetch_series(symbol_name, tf, fetch_count)
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

        frame = _build_overlap_frame(series_map, symbol_list, limit)
        meta["symbols_used"] = (
            list(frame.columns)
            if isinstance(frame, pd.DataFrame)
            else list(series_map.keys())
        )
        min_required_samples = int(max_lag + 6)
        meta["minimum_samples_required"] = int(min_required_samples)
        meta["samples_aligned_raw"] = int(len(frame))
        alignment_detail = _build_alignment_detail(
            symbol_rows=symbol_rows,
            pair_overlaps=pair_overlaps,
            aligned_rows=int(len(frame)),
            minimum_required=min_required_samples,
        )
        if alignment_detail is not None:
            meta["alignment_detail"] = alignment_detail
        overlap_pruning = None
        if frame.empty or len(frame) <= max_lag + 5:
            pruned_symbols, pruned_frame, overlap_pruning = _prune_symbols_for_overlap(
                series_map,
                symbol_list,
                limit=limit,
                minimum_required=min_required_samples,
                preserve_symbol=requested_anchor,
            )
            if overlap_pruning is not None:
                symbol_list = pruned_symbols
                frame = pruned_frame
                meta["overlap_pruning"] = overlap_pruning
                meta["pruned_symbols"] = list(
                    overlap_pruning.get("dropped_symbols") or []
                )
                meta["symbols_used"] = list(symbol_list)
                meta["samples_aligned_raw_after_pruning"] = int(len(frame))
                pair_overlaps_after = _pair_overlap_counts(series_map, symbol_list)
                if pair_overlaps_after:
                    meta["pair_overlaps_after_pruning"] = pair_overlaps_after
                symbol_rows_after = {
                    str(sym): int(symbol_rows.get(sym, 0))
                    for sym in symbol_list
                    if sym in symbol_rows
                }
                alignment_detail_after = _build_alignment_detail(
                    symbol_rows=symbol_rows_after,
                    pair_overlaps=pair_overlaps_after,
                    aligned_rows=int(len(frame)),
                    minimum_required=min_required_samples,
                )
                if alignment_detail_after is not None:
                    meta["alignment_detail_after_pruning"] = alignment_detail_after
                dropped_text = ", ".join(overlap_pruning.get("dropped_symbols") or [])
                kept_text = ", ".join(symbol_list)
                warnings_out.append(
                    f"Dropped {dropped_text} due to insufficient overlap; continuing with {kept_text}."
                )
        if frame.empty or len(frame) <= max_lag + 5:
            details_out = [
                _format_overlap_details(
                    symbol_rows=symbol_rows,
                    aligned_rows=int(len(frame)),
                    minimum_required=min_required_samples,
                )
            ]
            if alignment_detail is not None:
                align_summary = _format_alignment_detail_summary(alignment_detail)
                if align_summary:
                    details_out.append(align_summary)
            if overlap_pruning is not None:
                dropped_text = ", ".join(overlap_pruning.get("dropped_symbols") or [])
                kept_text = ", ".join(overlap_pruning.get("kept_symbols") or [])
                details_out.append(
                    "Auto-pruning dropped "
                    f"{dropped_text}; kept {kept_text}; aligned after pruning: {int(len(frame))}."
                )
            return _causal_error(
                "Insufficient overlapping data between symbols to run tests.",
                code="insufficient_overlap",
                meta=meta,
                warnings=warnings_out,
                details=details_out,
            )

        frame = frame.dropna(how="any")
        meta["samples_aligned_clean"] = int(len(frame))
        if frame.empty or len(frame) <= max_lag + 5:
            details_out = [
                _format_overlap_details(
                    symbol_rows=symbol_rows,
                    aligned_rows=int(len(frame)),
                    minimum_required=min_required_samples,
                )
            ]
            if alignment_detail is not None:
                align_summary = _format_alignment_detail_summary(alignment_detail)
                if align_summary:
                    details_out.append(align_summary)
            return _causal_error(
                "Insufficient clean samples after alignment.",
                code="insufficient_samples",
                meta=meta,
                warnings=warnings_out,
                details=details_out,
            )

        transformed = _transform_frame(frame, transform)
        if transformed.empty or len(transformed) <= max_lag + 2:
            return _causal_error(
                "Transform produced insufficient samples for testing. Try using more history or a different transform.",
                code="insufficient_samples",
                meta=meta,
                warnings=warnings_out,
            )

        if normalize:
            transformed = _standardize_frame(transformed)
            transformed = transformed.dropna(how="any")
            if transformed.empty or len(transformed) <= max_lag + 2:
                return _causal_error(
                    "Normalization resulted in insufficient samples.",
                    code="insufficient_samples",
                    meta=meta,
                    warnings=warnings_out,
                )

        rows: List[Dict[str, object]] = []
        pair_attempts = 0
        pair_success = 0
        pair_failures: List[Dict[str, Any]] = []
        for effect in transformed.columns:
            for cause in transformed.columns:
                if effect == cause:
                    continue
                subset = transformed[[effect, cause]].dropna(how="any")
                if len(subset) <= max_lag + 2:
                    continue
                pair_attempts += 1
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
                rows.append(
                    {
                        "effect": effect,
                        "cause": cause,
                        "lag": best_lag,
                        "p_value": best_p,
                        "p_value_raw": float(best_p_raw),
                        "p_value_correction": "bonferroni",
                        "lag_tests_run": lag_correction_factor,
                        "samples": len(subset),
                        "significant": bool(best_p < significance),
                    }
                )
        rows_sorted = sorted(
            rows, key=lambda item: (item["p_value"], item["effect"], item["cause"])
        )
        meta.update(
            {
                "group_hint": group_hint,
                "symbols_used": list(transformed.columns),
                "samples_aligned": int(len(transformed)),
                "pairs_attempted": int(pair_attempts),
                "pairs_tested": int(pair_success),
                "pairs_failed": int(max(pair_attempts - pair_success, 0)),
                "p_value_correction": "bonferroni_across_lags",
            }
        )
        if pair_failures:
            meta["pair_failures"] = pair_failures
            warnings_out.append(
                f"{max(pair_attempts - pair_success, 0)} pairwise Granger tests failed; see meta['pair_failures']."
            )
        data: Dict[str, Any] = {
            "items": rows_sorted,
        }
        out: Dict[str, Any] = {
            "success": True,
            "data": data,
            "summary": {
                "counts": {
                    "links": int(len(rows_sorted)),
                }
            },
            "meta": _causal_contract_meta(
                meta,
                legends={
                    "transform": _TRANSFORM_LEGEND,
                    "note_p_value": "Lower p-values indicate stronger evidence of causality. Values < significance threshold indicate significant Granger-causal relationship.",
                    "note_lag": "Optimal lag order (bars) at which past values of 'cause' best predict current 'effect'",
                    "note_p_value_correction": "Displayed p-values use Bonferroni correction across tested lags for each pair.",
                },
            ),
        }
        if warnings_out:
            out["warnings"] = warnings_out
        if not rows_sorted:
            out["message"] = (
                "No causal relationships detected (insufficient data or all tests failed)."
            )
        return out

    return run_logged_operation(
        logger,
        operation="causal_discover_signals",
        symbols=symbols,
        symbol=symbol,
        timeframe=timeframe,
        limit=limit,
        max_lag=max_lag,
        func=_run,
    )


@mcp.tool()
def correlation_matrix(  # noqa: C901
    symbols: Optional[str] = None,
    symbol: Optional[str] = None,
    group: Optional[str] = None,
    timeframe: TimeframeLiteral = "H1",
    limit: int = 500,
    method: str = "pearson",
    transform: str = "log_return",
    min_overlap: int = 30,
    detail: CompactFullDetailLiteral = "full",
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
        symbol: Compatibility alias for `symbols`. Optional when using `group`.
        group: Explicit MT5 group path (for example "Forex\\Majors"). Mutually
            exclusive with `symbols`.
        timeframe: MT5 timeframe key (e.g. "M15", "H1").
        limit: Maximum number of overlapping transformed samples used per pair.
        method: Correlation method: "pearson" or "spearman".
        transform: Preprocessing transform: "log_return", "pct", "diff", or "level".
        min_overlap: Minimum overlapping transformed samples required per pair.
        detail: "compact" keeps canonical pair rows plus highlights; "full" also
            includes the derived matrix view.
    """

    symbol_alias = symbol

    def _run() -> Dict[str, Any]:  # noqa: C901
        meta: Dict[str, Any] = {
            "_tool": "correlation_matrix",
            "_request_keys": _CORRELATION_REQUEST_KEYS,
            "timeframe": str(timeframe),
            "limit": int(limit),
            "method": str(method),
            "transform": str(transform),
            "min_overlap": int(min_overlap),
            "detail": str(detail or "full"),
        }
        connection_error = _causal_connection_error()
        if connection_error is not None:
            return _causal_error(
                str(connection_error.get("error") or "Failed to connect to MetaTrader5."),
                code=str(connection_error.get("error_code") or "mt5_connection_error"),
                meta=meta,
            )
        mt5_gateway = get_mt5_gateway(
            adapter=mt5,
            ensure_connection_impl=ensure_mt5_connection_or_raise,
        )

        symbol_list, selector_meta, selector_error = normalize_symbol_selector_aliases(
            symbol=symbol_alias,
            symbols=symbols,
            parse_selector=_parse_symbols,
        )
        meta.update(selector_meta)
        if group is not None:
            meta["group_input"] = str(group)
        if selector_error is not None:
            return _causal_error(
                selector_error,
                code="invalid_input",
                meta=meta,
            )
        requested_anchor = (
            symbol_list[0] if group is None and len(symbol_list) == 1 else None
        )
        group_hint: str | None = None
        if group and symbol_list:
            return _causal_error(
                "Provide either symbol/symbols or group for correlation analysis, not both.",
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

        if limit < 2:
            return _causal_error(
                "limit must be at least 2.",
                code="invalid_input",
                meta=meta,
            )

        if min_overlap < 2:
            return _causal_error(
                "min_overlap must be at least 2.",
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
                "Invalid transform. Valid options: log_return, pct, diff, level",
                code="invalid_transform",
                meta=meta,
            )
        meta["transform"] = transform_value
        detail_mode = str(detail or "full").strip().lower()
        if detail_mode not in {"compact", "full"}:
            return _causal_error(
                "detail must be 'compact' or 'full'.",
                code="invalid_input",
                meta=meta,
            )
        meta["detail"] = detail_mode

        fetch_count = max(int(limit) + 10, int(min_overlap) + 10, 200)
        meta["fetch_count"] = int(fetch_count)
        series_map: Dict[str, pd.Series] = {}
        errors: List[str] = []
        for symbol_name in symbol_list:
            series, err = _fetch_series(symbol_name, tf, fetch_count)
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
            limit=int(limit),
            min_overlap=int(min_overlap),
        )
        meta.update(
            {
                "pairs_attempted": int(
                    max(len(symbols_used) * (len(symbols_used) - 1) // 2, 0)
                ),
                "pairs_computed": int(len(rows)),
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

        data_out: Dict[str, Any] = {
            "items": rows,
        }
        if detail_mode == "full":
            data_out["matrix"] = _build_correlation_matrix(symbols_used, rows)

        out: Dict[str, Any] = {
            "success": True,
            "data": data_out,
            "summary": {
                "counts": {
                    "pairs": int(len(rows)),
                },
                "highlights": _build_correlation_summary(rows),
            },
            "meta": _causal_contract_meta(
                meta,
                legends={
                    "transform": _TRANSFORM_LEGEND,
                    "correlation_method": _CORRELATION_METHOD_LEGEND,
                    "relationship": {
                        "positive": "Symbols tend to move in the same direction",
                        "negative": "Symbols tend to move in opposite directions",
                        "flat": "No consistent directional relationship (correlation near zero)",
                        "strength_weak": "|correlation| < 0.3",
                        "strength_moderate": "0.3 <= |correlation| < 0.7",
                        "strength_strong": "|correlation| >= 0.7",
                    },
                },
            ),
        }
        if warnings_out:
            out["warnings"] = warnings_out
        return out

    return run_logged_operation(
        logger,
        operation="correlation_matrix",
        symbols=symbols,
        symbol=symbol,
        group=group,
        timeframe=timeframe,
        limit=limit,
        method=method,
        transform=transform,
        min_overlap=min_overlap,
        detail=detail,
        func=_run,
    )


@mcp.tool()
def cointegration_test(  # noqa: C901
    symbols: Optional[str] = None,
    symbol: Optional[str] = None,
    group: Optional[str] = None,
    timeframe: TimeframeLiteral = "H1",
    limit: int = 500,
    transform: str = "log_level",
    trend: str = "c",
    significance: float = 0.05,
    min_overlap: int = 80,
) -> Dict[str, Any]:
    """Run pairwise Engle-Granger cointegration tests on MT5 symbols.

    When a single symbol is provided, the tool automatically expands to include
    all related symbols from its MT5 group (e.g., EURUSD → EURUSD, GBPUSD, 
    USDCHF, USDJPY, USDCAD, AUDUSD). This enables cointegration analysis across
    related pairs. To test cointegration for specific symbols only, provide
    multiple symbols explicitly or use the `group` parameter.

    Args:
        symbols: Comma-separated MT5 symbols; a single symbol auto-expands to
            its entire MT5 group (e.g. "EURUSD" → all Forex majors).
            Optional when using `group`.
        symbol: Compatibility alias for `symbols`. Optional when using `group`.
        group: Explicit MT5 group path (for example "Forex\\Majors"). Mutually
            exclusive with `symbols`.
        timeframe: MT5 timeframe key (e.g. "M15", "H1").
        limit: Maximum number of overlapping transformed samples used per pair.
        transform: Price transform: "log_level" or "level".
        trend: Deterministic trend term for the test: "c", "ct", "ctt", or "n".
        significance: Alpha threshold for reporting cointegrated pairs.
        min_overlap: Minimum overlapping transformed samples required per pair.
    """

    symbol_alias = symbol

    def _run() -> Dict[str, Any]:  # noqa: C901
        meta: Dict[str, Any] = {
            "_tool": "cointegration_test",
            "_request_keys": _COINTEGRATION_REQUEST_KEYS,
            "timeframe": str(timeframe),
            "limit": int(limit),
            "transform": str(transform),
            "trend": str(trend),
            "significance": float(significance),
            "min_overlap": int(min_overlap),
        }
        connection_error = _causal_connection_error()
        if connection_error is not None:
            return _causal_error(
                str(connection_error.get("error") or "Failed to connect to MetaTrader5."),
                code=str(connection_error.get("error_code") or "mt5_connection_error"),
                meta=meta,
            )
        mt5_gateway = get_mt5_gateway(
            adapter=mt5,
            ensure_connection_impl=ensure_mt5_connection_or_raise,
        )

        try:
            from statsmodels.tsa.stattools import coint
        except Exception:
            return _causal_error(
                "statsmodels is required for cointegration testing. Please install 'statsmodels'.",
                code="dependency_missing",
                meta=meta,
            )

        symbol_list, selector_meta, selector_error = normalize_symbol_selector_aliases(
            symbol=symbol_alias,
            symbols=symbols,
            parse_selector=_parse_symbols,
        )
        meta.update(selector_meta)
        if group is not None:
            meta["group_input"] = str(group)
        if selector_error is not None:
            return _causal_error(
                selector_error,
                code="invalid_input",
                meta=meta,
            )
        requested_anchor = (
            symbol_list[0] if group is None and len(symbol_list) == 1 else None
        )
        group_hint: str | None = None
        if group and symbol_list:
            return _causal_error(
                "Provide either symbol/symbols or group for cointegration testing, not both.",
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

        if limit < 2:
            return _causal_error(
                "limit must be at least 2.",
                code="invalid_input",
                meta=meta,
            )

        if min_overlap < 2:
            return _causal_error(
                "min_overlap must be at least 2.",
                code="invalid_input",
                meta=meta,
            )

        if not (0.0 < float(significance) < 1.0):
            return _causal_error(
                "significance must be between 0 and 1.",
                code="invalid_input",
                meta=meta,
            )

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

        fetch_count = max(int(limit) + 10, int(min_overlap) + 10, 200)
        meta["fetch_count"] = int(fetch_count)
        series_map: Dict[str, pd.Series] = {}
        errors: List[str] = []
        for symbol in symbol_list:
            series, err = _fetch_series(symbol, tf, fetch_count)
            if err:
                errors.append(err)
            else:
                series_map[symbol] = series

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

        rows: List[Dict[str, Any]] = []
        pair_overlaps: Dict[str, int] = {}
        pair_failures: List[Dict[str, Any]] = []
        pairs_skipped_min_overlap = 0

        for idx, left in enumerate(symbols_used):
            for right in symbols_used[idx + 1 :]:
                subset_all = transformed[[left, right]].dropna(how="any")
                overlap_rows = int(len(subset_all))
                pair_overlaps[f"{left}-{right}"] = overlap_rows
                if overlap_rows < int(min_overlap):
                    pairs_skipped_min_overlap += 1
                    continue
                subset = subset_all.tail(int(limit))
                row, failures = _evaluate_cointegration_pair(
                    subset,
                    left,
                    right,
                    trend=trend_value,
                    significance=float(significance),
                    coint_func=coint,
                )
                if row is not None:
                    row["overlap_rows"] = overlap_rows
                    row["window_requested"] = int(limit)
                    row["window_actual"] = int(len(subset))
                    row["window_truncated"] = bool(len(subset) < overlap_rows)
                    rows.append(row)
                if failures:
                    for failure in failures:
                        if len(pair_failures) < 10:
                            pair_failures.append(failure)

        rows.sort(
            key=lambda item: (
                float(item["p_value"]),
                -int(item["samples"]),
                str(item["left"]),
                str(item["right"]),
            )
        )
        meta.update(
            {
                "pairs_attempted": int(
                    max(len(symbols_used) * (len(symbols_used) - 1) // 2, 0)
                ),
                "pairs_tested": int(len(rows)),
                "pairs_failed": int(len(pair_failures)),
                "pairs_skipped_min_overlap": int(pairs_skipped_min_overlap),
                "pair_overlaps": pair_overlaps,
            }
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
                _format_pair_overlap_details(pair_overlaps, int(min_overlap)) or None
            )
            if pair_failures and any(
                rows_count >= int(min_overlap) for rows_count in pair_overlaps.values()
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
            "data": {
                "items": rows,
            },
            "summary": {
                "counts": {
                    "pairs": int(len(rows)),
                    "cointegrated": cointegrated_count,
                },
                "highlights": _build_cointegration_summary(rows),
            },
            "meta": _causal_contract_meta(
                meta,
                legends={
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
                },
            ),
        }
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
        symbol=symbol,
        group=group,
        timeframe=timeframe,
        limit=limit,
        transform=transform,
        trend=trend,
        significance=significance,
        min_overlap=min_overlap,
        func=_run,
    )
