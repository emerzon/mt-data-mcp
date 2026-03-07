"""Causal signal discovery tools."""

from __future__ import annotations

import math
import logging
import time
import warnings
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

import pandas as pd
import numpy as np

from ._mcp_instance import mcp
from .constants import TIMEFRAME_MAP
from .execution_logging import infer_result_success, log_operation_finish, log_operation_start
from .mt5_gateway import create_mt5_gateway
from ..utils.mt5 import (
    MT5ConnectionError,
    _ensure_symbol_ready,
    _mt5_copy_rates_from,
    ensure_mt5_connection_or_raise,
    mt5,
)
from ..utils.symbol import _extract_group_path as _extract_group_path_util

logger = logging.getLogger(__name__)


def _causal_connection_error() -> Dict[str, Any] | None:
    mt5_gateway = _get_mt5_gateway()
    try:
        mt5_gateway.ensure_connection()
    except MT5ConnectionError as exc:
        return {"error": str(exc)}
    return None


def _get_mt5_gateway():
    return create_mt5_gateway(adapter=mt5, ensure_connection_impl=ensure_mt5_connection_or_raise)


def _parse_symbols(value: str) -> List[str]:
    items: List[str] = []
    for chunk in value.replace(";", ",").split(","):
        name = chunk.strip()
        if name:
            items.append(name)
    return list(dict.fromkeys(items))  # dedupe preserving order


def _expand_symbols_for_group(anchor: str, gateway: Any = None) -> tuple[List[str], str | None, str | None]:
    """Return visible group members for anchor along with the group path."""
    mt5_gateway = gateway or _get_mt5_gateway()
    info = mt5_gateway.symbol_info(anchor)
    if info is None:
        return [], f"Symbol {anchor} not found", None
    group_path = _extract_group_path_util(info)
    all_symbols = mt5_gateway.symbols_get()
    if all_symbols is None:
        return [], f"Failed to load symbol list: {mt5_gateway.last_error()}", group_path
    members: list[str] = []
    for sym in all_symbols:
        if not getattr(sym, 'visible', True) and sym.name != anchor:
            continue
        if _extract_group_path_util(sym) == group_path:
            members.append(sym.name)
    if anchor not in members:
        members.insert(0, anchor)
    deduped = list(dict.fromkeys(members))
    if len(deduped) < 2:
        return deduped, f"Symbol group {group_path} has fewer than two visible instruments", group_path
    return deduped, None, group_path



def _fetch_series(symbol: str, timeframe, count: int, retries: int = 3, pause: float = 0.25) -> Tuple[pd.Series, str | None]:
    """Fetch recent close prices for a symbol."""
    err = _ensure_symbol_ready(symbol)
    if err:
        return pd.Series(dtype=float), err

    for attempt in range(retries):
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
        series = pd.Series(df["close"].to_numpy(dtype=float), index=pd.to_datetime(df["time"], unit="s"))
        return series, None
    return pd.Series(dtype=float), f"Failed to fetch data for {symbol}" + (f" after {retries} retries" if retries > 1 else "")


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


def _format_summary(rows: List[Dict[str, object]], symbols: List[str], transform: str, alpha: float, group_hint: str | None = None) -> str:
    if not rows:
        return "No valid pairings available for causal discovery."
    rows_sorted = sorted(rows, key=lambda item: (item["p_value"], item["effect"], item["cause"]))
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
    lines.append("Lag refers to the history length of the cause series used in the best-performing test (ssr_ftest).")
    lines.append("Use the p-value to gauge strength; results are pairwise and do not imply full causal graphs.")
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
        "meta": meta,
    }
    if warnings:
        out["warnings"] = warnings
    if details:
        out["details"] = details
    return out


def _format_overlap_details(
    symbol_rows: Dict[str, int],
    aligned_rows: int,
    minimum_required: int,
) -> str:
    parts: List[str] = []
    for symbol, count in symbol_rows.items():
        parts.append(f"{symbol}: {int(count)} rows")
    parts.append(f"aligned: {int(aligned_rows)} (minimum {int(minimum_required)} required)")
    return ", ".join(parts)


def _pair_overlap_counts(series_map: Dict[str, pd.Series], symbols: List[str]) -> Dict[str, int]:
    overlaps: Dict[str, int] = {}
    for i, left in enumerate(symbols):
        left_series = series_map.get(left)
        if not isinstance(left_series, pd.Series):
            continue
        left_idx = left_series.dropna().index
        for right in symbols[i + 1:]:
            right_series = series_map.get(right)
            if not isinstance(right_series, pd.Series):
                continue
            right_idx = right_series.dropna().index
            key = f"{left}-{right}"
            overlaps[key] = int(len(left_idx.intersection(right_idx)))
    return overlaps


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
def causal_discover_signals(
    symbols: str,
    timeframe: str = "H1",
    limit: int = 500,
    max_lag: int = 5,
    significance: float = 0.05,
    transform: str = "log_return",
    normalize: bool = True,
) -> Dict[str, Any]:
    """Run Granger-style causal discovery on MT5 symbols.

    Args:
        symbols: Comma-separated MT5 symbols; provide one symbol to auto-expand its group.
        timeframe: MT5 timeframe key (e.g. "M15", "H1").
        limit: Number of recent bars to analyse per symbol.
        max_lag: Maximum lag order for tests (>=1).
        significance: Alpha level for reporting causal links.
        transform: Preprocessing transform: "log_return", "pct", "diff", or "level".
        normalize: Z-score columns before testing to stabilise scale.
    """
    started_at = time.perf_counter()
    log_operation_start(
        logger,
        operation="causal_discover_signals",
        symbols=symbols,
        timeframe=timeframe,
        limit=limit,
        max_lag=max_lag,
    )

    def _finish(result: Dict[str, Any]) -> Dict[str, Any]:
        log_operation_finish(
            logger,
            operation="causal_discover_signals",
            started_at=started_at,
            success=infer_result_success(result),
            symbols=symbols,
            timeframe=timeframe,
            limit=limit,
            max_lag=max_lag,
        )
        return result

    connection_error = _causal_connection_error()
    if connection_error is not None:
        return _finish(connection_error)
    mt5_gateway = _get_mt5_gateway()
    meta: Dict[str, Any] = {
        "timeframe": str(timeframe),
        "limit": int(limit),
        "max_lag": int(max_lag),
        "significance": float(significance),
        "transform": str(transform),
        "normalize": bool(normalize),
    }

    try:
        from statsmodels.tsa.stattools import grangercausalitytests
    except Exception:
        return _finish(_causal_error(
            "statsmodels is required for causal discovery. Please install 'statsmodels'.",
            code="dependency_missing",
            meta=meta,
        ))

    symbol_list = _parse_symbols(symbols)
    meta["symbols_input"] = list(symbol_list)
    group_hint: str | None = None
    if not symbol_list:
        return _finish(_causal_error(
            "Provide at least one symbol for causal discovery (e.g. 'EURUSD' or 'EURUSD,GBPUSD').",
            code="invalid_input",
            meta=meta,
        ))
    if len(symbol_list) == 1:
        expanded, err, group_path = _expand_symbols_for_group(symbol_list[0], gateway=mt5_gateway)
        if err:
            return _finish(_causal_error(
                err,
                code="symbol_group_error",
                meta=meta,
            ))
        symbol_list = expanded
        group_hint = group_path

    if len(symbol_list) < 2:
        return _finish(_causal_error(
            "Provide at least two symbols for causal discovery (e.g. 'EURUSD,GBPUSD').",
            code="invalid_input",
            meta=meta,
        ))

    tf = TIMEFRAME_MAP.get(timeframe)
    if tf is None:
        valid = ", ".join(sorted(TIMEFRAME_MAP.keys()))
        return _finish(_causal_error(
            f"Invalid timeframe '{timeframe}'. Valid options: {valid}",
            code="invalid_timeframe",
            meta=meta,
        ))

    if max_lag < 1:
        return _finish(_causal_error(
            "max_lag must be at least 1.",
            code="invalid_input",
            meta=meta,
        ))

    fetch_count = max(limit + max_lag + 10, 200)
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
        return _finish(_causal_error(
            errors[0],
            code="data_fetch_failed",
            meta=meta,
            details=errors,
        ))
    warnings_out: List[str] = []
    if errors:
        # Note missing symbols but continue with others
        warnings_out.extend(errors)
        symbol_list = [s for s in symbol_list if s in series_map]

    if len(series_map) < 2:
        return _finish(_causal_error(
            "Not enough valid symbol data fetched to run causal discovery.",
            code="insufficient_symbols",
            meta=meta,
            warnings=warnings_out,
        ))

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

    frame = pd.concat(series_map, axis=1, join="inner").tail(limit)
    meta["symbols_used"] = list(frame.columns) if isinstance(frame, pd.DataFrame) else list(series_map.keys())
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
        return _finish(_causal_error(
            "Insufficient overlapping data between symbols to run tests.",
            code="insufficient_overlap",
            meta=meta,
            warnings=warnings_out,
            details=details_out,
        ))

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
        return _finish(_causal_error(
            "Insufficient clean samples after alignment.",
            code="insufficient_samples",
            meta=meta,
            warnings=warnings_out,
            details=details_out,
        ))

    transformed = _transform_frame(frame, transform)
    if transformed.empty or len(transformed) <= max_lag + 2:
        return _finish(_causal_error(
            "Transform produced insufficient samples for testing. Try using more history or a different transform.",
            code="insufficient_samples",
            meta=meta,
            warnings=warnings_out,
        ))

    if normalize:
        transformed = _standardize_frame(transformed)
        transformed = transformed.dropna(how="any")
        if transformed.empty or len(transformed) <= max_lag + 2:
            return _finish(_causal_error(
                "Normalization resulted in insufficient samples.",
                code="insufficient_samples",
                meta=meta,
                warnings=warnings_out,
            ))  # rare but possible

    rows: List[Dict[str, object]] = []
    pair_attempts = 0
    pair_success = 0
    for effect in transformed.columns:
        for cause in transformed.columns:
            if effect == cause:
                continue
            subset = transformed[[effect, cause]].dropna(how="any")
            if len(subset) <= max_lag + 2:
                continue
            pair_attempts += 1
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=FutureWarning)
                    tests = grangercausalitytests(subset[[effect, cause]], maxlag=max_lag, verbose=False)
            except Exception:
                continue
            pair_success += 1
            best_lag = None
            best_p = None
            for lag, result in tests.items():
                stat, pvalue, *_ = result[0]["ssr_ftest"]
                if math.isnan(pvalue):
                    continue
                if best_p is None or pvalue < best_p:
                    best_p = float(pvalue)
                    best_lag = int(lag)
            if best_p is None or best_lag is None:
                continue
            rows.append(
                {
                    "effect": effect,
                    "cause": cause,
                    "lag": best_lag,
                    "p_value": best_p,
                    "samples": len(subset),
                    "significant": bool(best_p < significance),
                }
            )
    rows_sorted = sorted(rows, key=lambda item: (item["p_value"], item["effect"], item["cause"]))
    meta.update({
        "group_hint": group_hint,
        "symbols_used": list(transformed.columns),
        "samples_aligned": int(len(transformed)),
        "pairs_attempted": int(pair_attempts),
        "pairs_tested": int(pair_success),
    })
    data: Dict[str, Any] = {
        "links": rows_sorted,
        "count_links": int(len(rows_sorted)),
        "summary_text": _format_summary(rows_sorted, list(transformed.columns), transform, significance, group_hint=group_hint),
    }
    out: Dict[str, Any] = {
        "success": True,
        "data": data,
        "meta": meta,
    }
    if warnings_out:
        out["warnings"] = warnings_out
    if not rows_sorted:
        out["message"] = "No causal relationships detected (insufficient data or all tests failed)."
    return _finish(out)
