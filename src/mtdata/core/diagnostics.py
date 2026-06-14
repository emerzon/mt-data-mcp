"""Time-series diagnostic MCP tools."""

from __future__ import annotations

import logging
import math
import warnings
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, periodogram

from ..forecast.common import bars_per_year
from ..shared.constants import TIMEFRAME_MAP
from ..shared.schema import CompactFullDetailLiteral, TimeframeLiteral
from ..utils.mt5 import (
    _ensure_symbol_ready,
    _mt5_copy_rates_from,
    ensure_mt5_connection_or_raise,
    mt5,
)
from ._mcp_instance import mcp
from .execution_logging import run_logged_operation
from .mt5_gateway import create_mt5_gateway
from .output_contract import normalize_output_verbosity_detail

logger = logging.getLogger(__name__)


def _fetch_diagnostic_bars(symbol: str, timeframe: str, lookback: int) -> tuple[pd.DataFrame, str | None]:
    tf = TIMEFRAME_MAP.get(str(timeframe or "").strip().upper())
    if tf is None:
        return pd.DataFrame(), f"Invalid timeframe '{timeframe}'."
    symbol_error = _ensure_symbol_ready(symbol)
    if symbol_error:
        return pd.DataFrame(), symbol_error
    rates = _mt5_copy_rates_from(
        symbol,
        tf,
        datetime.now(timezone.utc),
        max(2, int(lookback)),
    )
    if rates is None or len(rates) == 0:
        return pd.DataFrame(), f"Failed to fetch data for {symbol}."
    try:
        frame = pd.DataFrame(rates)
    except Exception:
        frame = pd.DataFrame(list(rates))
    required = {"time", "close"}
    if frame.empty or not required.issubset(frame.columns):
        return pd.DataFrame(), "Fetched bars do not contain time and close fields."
    frame = frame.sort_values("time").drop_duplicates("time", keep="last")
    return frame.tail(max(2, int(lookback))).reset_index(drop=True), None


def _diagnostic_series(frame: pd.DataFrame, target: str) -> pd.Series:
    target_value = str(target or "close").strip().lower()
    close = pd.to_numeric(frame["close"], errors="coerce")
    if target_value == "close":
        values = close
    elif target_value == "log_price":
        values = np.log(close.where(close > 0))
    elif target_value == "return":
        values = close.pct_change(fill_method=None)
    elif target_value == "log_return":
        values = np.log(close.where(close > 0)).diff()
    elif target_value == "diff":
        values = close.diff()
    else:
        raise ValueError("target must be one of: close, log_price, return, log_return, diff.")
    return pd.Series(values, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()


def _critical_values(values: Any) -> Dict[str, float]:
    if not isinstance(values, dict):
        return {}
    return {
        str(key): round(float(value), 6)
        for key, value in values.items()
        if value is not None and math.isfinite(float(value))
    }


def _clean_stationarity_warning(text: Any) -> str:
    """Translate raw statsmodels/scipy stationarity warnings into plain guidance."""
    raw = str(getattr(text, "message", text)).strip()
    low = raw.lower()
    if "p-value" in low and ("look-up table" in low or "lookup table" in low or "outside of the range" in low):
        if "smaller" in low:
            direction = "smaller than the reported value"
        elif "greater" in low or "larger" in low:
            direction = "greater than the reported value"
        else:
            direction = "outside the reported range"
        return (
            "KPSS p-value is approximate: the test statistic falls outside the "
            f"lookup table, so the actual p-value is {direction}."
        )
    return raw


@mcp.tool()
def stationarity_test(
    symbol: str,
    timeframe: TimeframeLiteral = "H1",
    lookback: int = 500,
    target: Literal["close", "log_price", "return", "log_return", "diff"] = "log_return",
    tests: str = "adf,kpss,pp",
    trend: Literal["c", "ct"] = "c",
    significance: float = 0.05,
    detail: CompactFullDetailLiteral = "compact",
) -> Dict[str, Any]:
    """Test an MT5 time series for stationarity using ADF, KPSS, and optional PP."""

    def _run() -> Dict[str, Any]:
        if int(lookback) < 20:
            return {"error": "lookback must be at least 20."}
        if not 0.0 < float(significance) < 1.0:
            return {"error": "significance must be between 0 and 1."}
        requested = [part.strip().lower() for part in str(tests or "").split(",") if part.strip()]
        requested = list(dict.fromkeys(requested))
        invalid = [name for name in requested if name not in {"adf", "kpss", "pp"}]
        if not requested or invalid:
            return {"error": "tests must contain one or more of: adf, kpss, pp."}
        detail_mode = normalize_output_verbosity_detail(detail, default="compact")
        gateway = create_mt5_gateway(adapter=mt5, ensure_connection_impl=ensure_mt5_connection_or_raise)
        gateway.ensure_connection()
        frame, fetch_error = _fetch_diagnostic_bars(symbol, timeframe, int(lookback))
        if fetch_error:
            return {"error": fetch_error}
        try:
            series = _diagnostic_series(frame, target)
        except ValueError as exc:
            return {"error": str(exc)}
        if len(series) < 20 or float(series.std(ddof=0)) <= 1e-15:
            return {"error": "At least 20 non-constant finite observations are required."}

        rows: List[Dict[str, Any]] = []
        warnings_out: List[str] = []
        alpha = float(significance)
        if "adf" in requested:
            from statsmodels.tsa.stattools import adfuller

            regression = "ct" if trend == "ct" else "c"
            result = adfuller(series.to_numpy(), regression=regression, autolag="AIC")
            rows.append(
                {
                    "test": "adf",
                    "statistic": round(float(result[0]), 6),
                    "p_value": round(float(result[1]), 6),
                    "lags": int(result[2]),
                    "samples": int(result[3]),
                    "stationary": bool(float(result[1]) < alpha),
                    "null_hypothesis": "unit_root",
                    **({"critical_values": _critical_values(result[4])} if detail_mode == "full" else {}),
                }
            )
        if "kpss" in requested:
            from statsmodels.tsa.stattools import kpss

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = kpss(series.to_numpy(), regression=trend, nlags="auto")
            rows.append(
                {
                    "test": "kpss",
                    "statistic": round(float(result[0]), 6),
                    "p_value": round(float(result[1]), 6),
                    "lags": int(result[2]),
                    "samples": int(len(series)),
                    "stationary": bool(float(result[1]) >= alpha),
                    "null_hypothesis": "stationary",
                    **({"critical_values": _critical_values(result[3])} if detail_mode == "full" else {}),
                }
            )
            warnings_out.extend(_clean_stationarity_warning(item) for item in caught)
        if "pp" in requested:
            try:
                from arch.unitroot import PhillipsPerron
            except ImportError:
                warnings_out.append("Phillips-Perron skipped because optional package 'arch' is not installed.")
            else:
                result = PhillipsPerron(series.to_numpy(), trend=trend)
                rows.append(
                    {
                        "test": "pp",
                        "statistic": round(float(result.stat), 6),
                        "p_value": round(float(result.pvalue), 6),
                        "lags": int(result.lags),
                        "samples": int(result.nobs),
                        "stationary": bool(float(result.pvalue) < alpha),
                        "null_hypothesis": "unit_root",
                        **({"critical_values": _critical_values(result.critical_values)} if detail_mode == "full" else {}),
                    }
                )

        votes = [bool(row["stationary"]) for row in rows]
        stationary_votes = int(sum(votes))
        conclusion = (
            "inconclusive"
            if not votes
            else "stationary"
            if stationary_votes == len(votes)
            else "non_stationary"
            if stationary_votes == 0
            else "mixed"
        )
        out: Dict[str, Any] = {
            "success": True,
            "symbol": symbol,
            "timeframe": timeframe,
            "target": target,
            "significance": alpha,
            "conclusion": conclusion,
            "stationary_votes": stationary_votes,
            "tests_completed": len(rows),
            "items": rows,
            "samples": int(len(series)),
        }
        if warnings_out:
            out["warnings"] = list(dict.fromkeys(warnings_out))
        if detail_mode == "full":
            out["interpretation"] = (
                "ADF and PP reject a unit-root null when stationary; KPSS fails to reject a stationarity null when stationary."
            )
        return out

    return run_logged_operation(
        logger,
        operation="stationarity_test",
        symbol=symbol,
        timeframe=timeframe,
        lookback=lookback,
        target=target,
        func=_run,
    )


@mcp.tool()
def seasonality_detect(
    symbol: str,
    timeframe: TimeframeLiteral = "H1",
    lookback: int = 1000,
    target: Literal["close", "log_price", "return", "log_return", "diff"] = "log_return",
    min_period: int = 2,
    max_period: Optional[int] = None,
    min_cycles: int = 3,
    top_n: int = 5,
    detail: CompactFullDetailLiteral = "compact",
) -> Dict[str, Any]:
    """Detect dominant seasonal periods using autocorrelation and spectral power."""

    def _run() -> Dict[str, Any]:
        if int(lookback) < 30:
            return {"error": "lookback must be at least 30."}
        if int(min_period) < 2 or int(min_cycles) < 2 or int(top_n) < 1:
            return {"error": "min_period >= 2, min_cycles >= 2, and top_n >= 1 are required."}
        gateway = create_mt5_gateway(adapter=mt5, ensure_connection_impl=ensure_mt5_connection_or_raise)
        gateway.ensure_connection()
        frame, fetch_error = _fetch_diagnostic_bars(symbol, timeframe, int(lookback))
        if fetch_error:
            return {"error": fetch_error}
        try:
            series = _diagnostic_series(frame, target)
        except ValueError as exc:
            return {"error": str(exc)}
        n = int(len(series))
        upper = min(int(max_period) if max_period is not None else max(2, n // int(min_cycles)), n // int(min_cycles))
        if upper < int(min_period) or n < 30:
            return {"error": "Insufficient samples for the requested period and cycle constraints."}
        values = series.to_numpy(dtype=float)
        centered = values - float(np.mean(values))
        variance = float(np.dot(centered, centered))
        if variance <= 1e-15:
            return {"error": "Cannot detect seasonality in a constant series."}
        periods = np.arange(int(min_period), upper + 1, dtype=int)
        acf_scores = np.asarray(
            [float(np.dot(centered[lag:], centered[:-lag]) / variance) for lag in periods],
            dtype=float,
        )
        frequencies, powers = periodogram(centered, detrend="linear", scaling="spectrum")
        spectral_by_period: Dict[int, float] = {}
        positive = frequencies > 0
        for frequency, power in zip(frequencies[positive], powers[positive]):
            period = int(round(1.0 / float(frequency)))
            if int(min_period) <= period <= upper:
                spectral_by_period[period] = max(spectral_by_period.get(period, 0.0), float(power))
        total_spectral_power = float(np.sum(powers[positive]))
        positive_acf = np.maximum(acf_scores, 0.0)
        peak_idx, _ = find_peaks(positive_acf)
        candidates = set(int(periods[index]) for index in peak_idx)
        candidates.update(sorted(spectral_by_period, key=spectral_by_period.get, reverse=True)[: max(int(top_n) * 3, 5)])
        if not candidates:
            candidates.update(int(value) for value in periods[np.argsort(positive_acf)[-max(int(top_n), 1) :]])
        rows: List[Dict[str, Any]] = []
        for period in candidates:
            acf_value = float(acf_scores[period - int(min_period)])
            spectral_strength = (
                float(spectral_by_period.get(period, 0.0) / total_spectral_power)
                if total_spectral_power > 0
                else 0.0
            )
            acf_strength = max(0.0, min(1.0, acf_value))
            score = 0.55 * acf_strength + 0.45 * spectral_strength
            rows.append(
                {
                    "period_bars": int(period),
                    "score": round(score, 6),
                    "acf": round(acf_value, 6),
                    "spectral_strength": round(spectral_strength, 6),
                    "cycles_observed": round(n / float(period), 2),
                }
            )
        rows.sort(key=lambda row: (-float(row["score"]), int(row["period_bars"])))
        rows = rows[: int(top_n)]
        out: Dict[str, Any] = {
            "success": True,
            "symbol": symbol,
            "timeframe": timeframe,
            "target": target,
            "samples": n,
            "search_range_bars": {"min": int(min_period), "max": upper},
            "items": rows,
            "count": len(rows),
            "dominant_period_bars": rows[0]["period_bars"] if rows else None,
            "score_formula": "0.55*acf_strength + 0.45*spectral_power_fraction; range 0-1, higher = stronger seasonality",
        }
        if normalize_output_verbosity_detail(detail, default="compact") == "full":
            out["method"] = {
                "acf_weight": 0.55,
                "periodogram_weight": 0.45,
                "spectral_component": "candidate_power / total_positive_frequency_power",
                "minimum_cycles": int(min_cycles),
            }
        return out

    return run_logged_operation(
        logger,
        operation="seasonality_detect",
        symbol=symbol,
        timeframe=timeframe,
        lookback=lookback,
        target=target,
        func=_run,
    )


def _robust_scores(values: pd.Series, method: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").astype(float)

    def _zero_scale_scores(center: float) -> pd.Series:
        deviations = (numeric - center).abs()
        return deviations.where(deviations == 0.0, np.inf)

    method_value = str(method or "mad").strip().lower()
    if method_value == "zscore":
        scale = float(numeric.std(ddof=0))
        center = float(numeric.mean())
        return (numeric - center).abs() / scale if scale > 0 else _zero_scale_scores(center)
    if method_value == "iqr":
        q1, q3 = numeric.quantile([0.25, 0.75])
        scale = float(q3 - q1)
        center = float(numeric.median())
        return (numeric - center).abs() / scale if scale > 0 else _zero_scale_scores(center)
    if method_value != "mad":
        raise ValueError("method must be one of: mad, iqr, zscore.")
    center = float(numeric.median())
    mad = float((numeric - center).abs().median())
    return 0.67448975 * (numeric - center).abs() / mad if mad > 0 else _zero_scale_scores(center)


@mcp.tool()
def outliers_detect(
    symbol: str,
    timeframe: TimeframeLiteral = "H1",
    lookback: int = 500,
    fields: str = "return,volume,range",
    method: Literal["mad", "iqr", "zscore"] = "mad",
    threshold: float = 3.5,
    limit: int = 50,
    detail: CompactFullDetailLiteral = "compact",
) -> Dict[str, Any]:
    """Detect anomalous MT5 bars using robust return, volume, and range scores."""

    def _run() -> Dict[str, Any]:
        if int(lookback) < 20 or int(limit) < 1 or float(threshold) <= 0:
            return {"error": "lookback >= 20, limit >= 1, and threshold > 0 are required."}
        requested = [part.strip().lower() for part in str(fields or "").split(",") if part.strip()]
        requested = list(dict.fromkeys(requested))
        if not requested or any(field not in {"return", "volume", "range"} for field in requested):
            return {"error": "fields must contain one or more of: return, volume, range."}
        gateway = create_mt5_gateway(adapter=mt5, ensure_connection_impl=ensure_mt5_connection_or_raise)
        gateway.ensure_connection()
        frame, fetch_error = _fetch_diagnostic_bars(symbol, timeframe, int(lookback))
        if fetch_error:
            return {"error": fetch_error}
        close = pd.to_numeric(frame["close"], errors="coerce")
        data: Dict[str, pd.Series] = {"return": close.pct_change(fill_method=None).abs()}
        if "volume" in requested:
            volume_col = "real_volume" if "real_volume" in frame and float(pd.to_numeric(frame["real_volume"], errors="coerce").fillna(0).sum()) > 0 else "tick_volume"
            if volume_col not in frame:
                return {"error": "Fetched bars do not contain volume data."}
            data["volume"] = pd.to_numeric(frame[volume_col], errors="coerce")
        if "range" in requested:
            if not {"high", "low"}.issubset(frame.columns):
                return {"error": "Fetched bars do not contain high and low fields."}
            data["range"] = (pd.to_numeric(frame["high"], errors="coerce") - pd.to_numeric(frame["low"], errors="coerce")).abs()
        score_frame = pd.DataFrame(index=frame.index)
        for field in requested:
            try:
                score_frame[field] = _robust_scores(data[field], method)
            except ValueError as exc:
                return {"error": str(exc)}
        max_scores = score_frame.max(axis=1, skipna=True)
        flagged = frame.loc[max_scores >= float(threshold)].copy()
        flagged["_score"] = max_scores.loc[flagged.index]
        flagged = flagged.sort_values("_score", ascending=False).head(int(limit))
        rows: List[Dict[str, Any]] = []
        full = normalize_output_verbosity_detail(detail, default="compact") == "full"
        for index, bar in flagged.iterrows():
            raw_field_scores = {
                field: float(score_frame.at[index, field])
                for field in requested
            }
            field_scores = {
                field: round(score if math.isfinite(score) else float(threshold), 4)
                for field, score in raw_field_scores.items()
                if not math.isnan(score)
            }
            raw_score = float(bar["_score"])
            item: Dict[str, Any] = {
                "time": datetime.fromtimestamp(float(bar["time"]), tz=timezone.utc).isoformat().replace("+00:00", "Z"),
                "score": round(raw_score if math.isfinite(raw_score) else float(threshold), 4),
                "fields": [
                    field
                    for field, score in raw_field_scores.items()
                    if not math.isnan(score) and score >= float(threshold)
                ],
            }
            if full:
                item.update(
                    {
                        "field_scores": field_scores,
                        "open": float(bar.get("open")),
                        "high": float(bar.get("high")),
                        "low": float(bar.get("low")),
                        "close": float(bar.get("close")),
                    }
                )
            rows.append(item)
        return {
            "success": True,
            "symbol": symbol,
            "timeframe": timeframe,
            "method": method,
            "threshold": float(threshold),
            "score_meaning": f"robust {method} deviation magnitude per bar; score >= threshold ({float(threshold)}) flags an outlier",
            "fields_analyzed": requested,
            "samples": int(len(frame)),
            "outliers_total": int((max_scores >= float(threshold)).sum()),
            "items": rows,
            "count": len(rows),
            "truncated": bool(int((max_scores >= float(threshold)).sum()) > len(rows)),
        }

    return run_logged_operation(
        logger,
        operation="outliers_detect",
        symbol=symbol,
        timeframe=timeframe,
        lookback=lookback,
        method=method,
        func=_run,
    )


@mcp.tool()
def volatility_term_structure(
    symbol: str,
    timeframe: TimeframeLiteral = "H1",
    lookback: int = 1000,
    horizons: str = "1,5,10,20,60",
    percentiles: str = "10,25,50,75,90",
    annualize: bool = True,
    detail: CompactFullDetailLiteral = "compact",
) -> Dict[str, Any]:
    """Compute current realized volatility and historical cones at multiple horizons."""

    def _run() -> Dict[str, Any]:
        if int(lookback) < 30:
            return {"error": "lookback must be at least 30."}
        try:
            horizon_values = sorted(
                set(int(part.strip()) for part in str(horizons).split(",") if part.strip())
            )
            percentile_values = sorted(
                set(float(part.strip()) for part in str(percentiles).split(",") if part.strip())
            )
        except Exception:
            return {"error": "horizons and percentiles must be comma-separated numbers."}
        if not horizon_values or any(value < 1 for value in horizon_values):
            return {"error": "horizons must contain positive integers."}
        if any(value <= 0.0 or value >= 100.0 for value in percentile_values):
            return {"error": "percentiles must be strictly between 0 and 100."}
        if max(horizon_values) >= int(lookback):
            return {"error": "Each horizon must be smaller than lookback."}
        gateway = create_mt5_gateway(adapter=mt5, ensure_connection_impl=ensure_mt5_connection_or_raise)
        gateway.ensure_connection()
        frame, fetch_error = _fetch_diagnostic_bars(symbol, timeframe, int(lookback))
        if fetch_error:
            return {"error": fetch_error}
        close = pd.to_numeric(frame["close"], errors="coerce")
        returns = np.log(close.where(close > 0)).diff().replace([np.inf, -np.inf], np.nan)
        factor = math.sqrt(bars_per_year(timeframe)) if annualize else 1.0
        if not math.isfinite(factor) or factor <= 0.0:
            factor = 1.0
        rows: List[Dict[str, Any]] = []
        for horizon in horizon_values:
            realized = returns.pow(2).rolling(window=int(horizon), min_periods=int(horizon)).mean().pow(0.5) * factor
            distribution = realized.dropna()
            if distribution.empty:
                continue
            current = float(distribution.iloc[-1])
            percentile_rank = float((distribution <= current).mean() * 100.0)
            cone = {
                f"p{int(value) if float(value).is_integer() else value:g}": round(
                    float(np.percentile(distribution.to_numpy(dtype=float), value)),
                    8,
                )
                for value in percentile_values
            }
            rows.append(
                {
                    "horizon_bars": int(horizon),
                    "current_volatility": round(current, 8),
                    "percentile_rank": round(percentile_rank, 2),
                    "cone": cone,
                    "samples": int(len(distribution)),
                }
            )
        if not rows:
            return {"error": "Insufficient finite returns for the requested horizons."}
        out: Dict[str, Any] = {
            "success": True,
            "symbol": symbol,
            "timeframe": timeframe,
            "annualized": bool(annualize),
            "unit": "annualized_decimal_volatility" if annualize else "per_bar_decimal_volatility",
            "cone_methodology": "percentiles of the historical distribution of rolling realized volatility at each horizon; percentile_rank shows where current vol sits in that distribution",
            "items": rows,
            "count": len(rows),
        }
        if annualize:
            out["bars_per_year"] = round(float(bars_per_year(timeframe)), 4)
            out["annualization_basis"] = "252_trading_days_24h_intraday"
        if normalize_output_verbosity_detail(detail, default="compact") == "full":
            out["method"] = "rolling_root_mean_square_log_return"
            out["lookback"] = int(lookback)
        return out

    return run_logged_operation(
        logger,
        operation="volatility_term_structure",
        symbol=symbol,
        timeframe=timeframe,
        lookback=lookback,
        func=_run,
    )
