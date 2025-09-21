"""Causal signal discovery tools."""

from __future__ import annotations

import math
import time
from datetime import datetime
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
import MetaTrader5 as mt5

from .constants import TIMEFRAME_MAP
from .server import mcp, _auto_connect_wrapper, _ensure_symbol_ready
from ..utils.mt5 import _mt5_copy_rates_from
from ..utils.symbol import _extract_group_path as _extract_group_path_util


def _parse_symbols(value: str) -> List[str]:
    items: List[str] = []
    for chunk in value.replace(";", ",").split(","):
        name = chunk.strip()
        if name:
            items.append(name)
    return list(dict.fromkeys(items))  # dedupe preserving order


def _expand_symbols_for_group(anchor: str) -> tuple[List[str], str | None, str | None]:
    """Return visible group members for anchor along with the group path."""
    info = mt5.symbol_info(anchor)
    if info is None:
        return [], f"Symbol {anchor} not found", None
    group_path = _extract_group_path_util(info)
    all_symbols = mt5.symbols_get()
    if all_symbols is None:
        return [], f"Failed to load symbol list: {mt5.last_error()}", group_path
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
        utc_now = datetime.utcnow()
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
    return frame.dropna(how="any")


def _standardize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    standardized = frame.copy()
    for col in frame.columns:
        series = standardized[col]
        std = float(series.std(ddof=0))
        mean = float(series.mean())
        if not math.isfinite(std) or std == 0.0:
            continue
        standardized[col] = (series - mean) / std
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


@mcp.tool()
@_auto_connect_wrapper
def causal_discover_signals(
    symbols: str,
    timeframe: str = "H1",
    limit: int = 500,
    max_lag: int = 5,
    significance: float = 0.05,
    transform: str = "log_return",
    normalize: bool = True,
) -> str:
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
    try:
        from statsmodels.tsa.stattools import grangercausalitytests
    except Exception:
        return "statsmodels is required for causal discovery. Please install 'statsmodels'."

    symbol_list = _parse_symbols(symbols)
    group_hint: str | None = None
    if not symbol_list:
        return "Provide at least one symbol for causal discovery (e.g. 'EURUSD' or 'EURUSD,GBPUSD')."
    if len(symbol_list) == 1:
        expanded, err, group_path = _expand_symbols_for_group(symbol_list[0])
        if err:
            return err
        symbol_list = expanded
        group_hint = group_path

    if len(symbol_list) < 2:
        return "Provide at least two symbols for causal discovery (e.g. 'EURUSD,GBPUSD')."

    tf = TIMEFRAME_MAP.get(timeframe)
    if tf is None:
        valid = ", ".join(sorted(TIMEFRAME_MAP.keys()))
        return f"Invalid timeframe '{timeframe}'. Valid options: {valid}"

    if max_lag < 1:
        return "max_lag must be at least 1."

    fetch_count = max(limit + max_lag + 10, 200)
    series_map: Dict[str, pd.Series] = {}
    errors: List[str] = []
    for symbol in symbol_list:
        series, err = _fetch_series(symbol, tf, fetch_count)
        if err:
            errors.append(err)
        else:
            series_map[symbol] = series

    if errors and not series_map:
        return errors[0]
    if errors:
        # Note missing symbols but continue with others
        symbol_list = [s for s in symbol_list if s in series_map]

    if len(series_map) < 2:
        return "Not enough valid symbol data fetched to run causal discovery."

    frame = pd.concat(series_map, axis=1, join="inner").tail(limit)
    if frame.empty or len(frame) <= max_lag + 5:
        return "Insufficient overlapping data between symbols to run tests."

    frame = frame.dropna(how="any")
    if frame.empty or len(frame) <= max_lag + 5:
        return "Insufficient clean samples after alignment."

    transformed = _transform_frame(frame, transform)
    if transformed.empty or len(transformed) <= max_lag + 2:
        return "Transform produced insufficient samples for testing. Try using more history or a different transform."

    if normalize:
        transformed = _standardize_frame(transformed)
        transformed = transformed.dropna(how="any")
        if transformed.empty or len(transformed) <= max_lag + 2:
            return "Normalization resulted in insufficient samples."  # rare but possible

    rows: List[Dict[str, object]] = []
    for effect in transformed.columns:
        for cause in transformed.columns:
            if effect == cause:
                continue
            subset = transformed[[effect, cause]].dropna(how="any")
            if len(subset) <= max_lag + 2:
                continue
            try:
                tests = grangercausalitytests(subset[[effect, cause]], maxlag=max_lag, verbose=False)
            except Exception:
                continue
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
                }
            )

    if not rows:
        return "No causal relationships detected (insufficient data or all tests failed)."

    return _format_summary(rows, list(transformed.columns), transform, significance, group_hint=group_hint)
