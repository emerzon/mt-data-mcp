import logging
import math
from typing import Any, Dict, List, Literal, Optional

import numpy as np

from ..forecast.common import fetch_history as _fetch_history
from ..shared.schema import DenoiseSpec, DetailLiteral, TimeframeLiteral
from ..utils.barriers import (
    barrier_prices_are_valid as _barrier_prices_are_valid,
)
from ..utils.barriers import (
    build_barrier_kwargs_from as _build_barrier_kwargs_from,
)
from ..utils.barriers import (
    get_pip_size as _get_pip_size,
)
from ..utils.barriers import (
    normalize_same_bar_policy,
    validate_barrier_unit_family_exclusivity,
)
from ..utils.barriers import (
    normalize_trade_direction as _normalize_trade_direction,
)
from ..utils.barriers import (
    resolve_barrier_prices as _resolve_barrier_prices,
)
from ..utils.coercion import coerce_finite_float, round_finite
from ..utils.denoise import resolve_denoise_base_col
from ..utils.mt5 import (
    MT5ConnectionError,
    ensure_mt5_connection_or_raise,
    symbol_price_digits,
)
from ..utils.time import _format_time_minimal
from ._mcp_instance import mcp
from .execution_logging import run_logged_operation
from .mt5_gateway import create_mt5_gateway
from .output_contract import normalize_output_detail

logger = logging.getLogger(__name__)
_COMPACT_LABEL_SAMPLE_SIZE = 10
_DEFAULT_LABEL_HORIZON = 12
_DEFAULT_LABEL_LOOKBACK = 50
_DEFAULT_LABEL_LIMIT = 50


def _label_outcome(label: int) -> str:
    if label == 1:
        return "tp"
    if label == -1:
        return "sl"
    return "neutral"


def _neutral_barrier_pct_range(max_move_pct: Any) -> Optional[List[float]]:
    try:
        max_move = float(max_move_pct)
    except Exception:
        return None
    if not math.isfinite(max_move) or max_move <= 0.0:
        return None
    low = max_move * 0.4
    high = max_move * 0.8
    return [round(low, 4), round(max(high, low), 4)]


def _round_label_price(value: Any, *, digits: int) -> Optional[float]:
    if int(digits) <= 0:
        return coerce_finite_float(value)
    return round_finite(value, digits, on_invalid="none")


def _triple_barrier_sample_row(
    *,
    idx: int,
    closes: np.ndarray,
    t_entry: List[str],
    labels: List[int],
    hold: List[int],
    tp_times: List[Optional[str]],
    sl_times: List[Optional[str]],
    direction_value: str,
    pip_size: float,
    barrier_kwargs: Dict[str, Any],
    price_digits: int = 0,
    same_bar_flags: Optional[List[bool]] = None,
) -> Dict[str, Any]:
    label = int(labels[idx])
    row: Dict[str, Any] = {
        "entry_time": t_entry[idx],
        "label": label,
        "outcome": (
            "same_bar_neutral"
            if same_bar_flags and same_bar_flags[idx] and label == 0
            else _label_outcome(label)
        ),
        "holding_bars": hold[idx],
        "tp_time": tp_times[idx],
        "sl_time": sl_times[idx],
        "same_bar": bool(same_bar_flags and same_bar_flags[idx]),
    }
    try:
        entry_price = float(closes[idx])
        if math.isfinite(entry_price):
            row["entry_price"] = _round_label_price(entry_price, digits=price_digits)
            tp_price, sl_price = _resolve_barrier_prices(
                price=entry_price,
                direction=direction_value,
                pip_size=pip_size,
                adjust_inverted=False,
                **barrier_kwargs,
            )
            if tp_price is not None:
                row["tp_price"] = _round_label_price(tp_price, digits=price_digits)
            if sl_price is not None:
                row["sl_price"] = _round_label_price(sl_price, digits=price_digits)
    except Exception as exc:
        row["barrier_error"] = str(exc) or exc.__class__.__name__
    return row


def _first_true_offsets(mask: np.ndarray) -> np.ndarray:
    if mask.size == 0:
        return np.array([], dtype=int)
    hits = np.any(mask, axis=1)
    offsets = np.argmax(mask, axis=1).astype(int, copy=False) + 1
    offsets[~hits] = -1
    return offsets


def _build_triple_barrier_outputs(
    *,
    closes: np.ndarray,
    highs: Optional[np.ndarray],
    lows: Optional[np.ndarray],
    times: np.ndarray,
    horizon: int,
    label_on: str,
    direction_value: str,
    pip_size: float,
    barrier_kwargs: Dict[str, Any],
    same_bar_policy: str = "sl_first",
) -> tuple[
    List[int],
    List[int],
    List[str],
    List[Optional[str]],
    List[Optional[str]],
    List[bool],
    List[float],
    List[float],
    int,
]:
    max_entry_index = len(closes) - int(horizon)
    if max_entry_index <= 0:
        return [], [], [], [], [], [], [], [], 0

    entry_prices = closes[:max_entry_index]
    valid_price_mask = np.isfinite(entry_prices) & (entry_prices > 0.0)
    tp_levels = np.full(max_entry_index, np.nan, dtype=float)
    sl_levels = np.full(max_entry_index, np.nan, dtype=float)
    valid_barrier_mask = np.zeros(max_entry_index, dtype=bool)

    for idx in np.flatnonzero(valid_price_mask):
        price = float(entry_prices[idx])
        tp_price, sl_price = _resolve_barrier_prices(
            price=price,
            direction=direction_value,
            pip_size=pip_size,
            adjust_inverted=False,
            **barrier_kwargs,
        )
        if tp_price is None or sl_price is None:
            continue
        if not _barrier_prices_are_valid(
            price=price,
            direction=direction_value,
            tp_price=tp_price,
            sl_price=sl_price,
        ):
            continue
        tp_levels[idx] = float(tp_price)
        sl_levels[idx] = float(sl_price)
        valid_barrier_mask[idx] = True

    valid_entry_mask = valid_price_mask & valid_barrier_mask
    skipped_entries = int(max_entry_index - np.count_nonzero(valid_entry_mask))

    high_values = highs if highs is not None else closes
    low_values = lows if lows is not None else closes

    if label_on == "close":
        close_windows = np.lib.stride_tricks.sliding_window_view(
            closes[1:], int(horizon)
        )
        if direction_value == "long":
            tp_hits = close_windows >= tp_levels[:, None]
            sl_hits = close_windows <= sl_levels[:, None]
        else:
            tp_hits = close_windows <= tp_levels[:, None]
            sl_hits = close_windows >= sl_levels[:, None]
    else:
        high_windows = np.lib.stride_tricks.sliding_window_view(
            high_values[1:], int(horizon)
        )
        low_windows = np.lib.stride_tricks.sliding_window_view(
            low_values[1:], int(horizon)
        )
        if direction_value == "long":
            tp_hits = high_windows >= tp_levels[:, None]
            sl_hits = low_windows <= sl_levels[:, None]
        else:
            tp_hits = low_windows <= tp_levels[:, None]
            sl_hits = high_windows >= sl_levels[:, None]

    hit_tp = _first_true_offsets(tp_hits)
    hit_sl = _first_true_offsets(sl_hits)

    labels: List[int] = []
    hold: List[int] = []
    entries: List[str] = []
    tp_times: List[Optional[str]] = []
    sl_times: List[Optional[str]] = []
    same_bar_flags: List[bool] = []
    max_favorable_moves_pct: List[float] = []
    max_adverse_moves_pct: List[float] = []

    for idx in np.flatnonzero(valid_entry_mask):
        entry_price = float(entry_prices[idx])
        future_high = np.asarray(high_values[idx + 1 : idx + int(horizon) + 1], dtype=float)
        future_low = np.asarray(low_values[idx + 1 : idx + int(horizon) + 1], dtype=float)
        finite_high = future_high[np.isfinite(future_high)]
        finite_low = future_low[np.isfinite(future_low)]
        window_high = float(np.max(finite_high)) if finite_high.size else entry_price
        window_low = float(np.min(finite_low)) if finite_low.size else entry_price
        if direction_value == "long":
            favorable_move = max(0.0, (window_high - entry_price) / entry_price * 100.0)
            adverse_move = max(0.0, (entry_price - window_low) / entry_price * 100.0)
        else:
            favorable_move = max(0.0, (entry_price - window_low) / entry_price * 100.0)
            adverse_move = max(0.0, (window_high - entry_price) / entry_price * 100.0)
        max_favorable_moves_pct.append(favorable_move)
        max_adverse_moves_pct.append(adverse_move)

        tp_offset = int(hit_tp[idx])
        sl_offset = int(hit_sl[idx])
        is_same_bar = bool(tp_offset > 0 and tp_offset == sl_offset)
        same_bar_flags.append(is_same_bar)
        if tp_offset < 0 and sl_offset < 0:
            labels.append(0)
            hold.append(int(horizon))
            tp_times.append(None)
            sl_times.append(None)
        elif tp_offset > 0 and (sl_offset < 0 or tp_offset < sl_offset):
            labels.append(1)
            hold.append(tp_offset)
            tp_times.append(_format_time_minimal(times[idx + tp_offset]))
            sl_times.append(None)
        elif is_same_bar and same_bar_policy == "tp_first":
            labels.append(1)
            hold.append(tp_offset)
            tp_times.append(_format_time_minimal(times[idx + tp_offset]))
            sl_times.append(_format_time_minimal(times[idx + sl_offset]))
        elif is_same_bar and same_bar_policy == "neutral":
            labels.append(0)
            hold.append(tp_offset)
            tp_times.append(_format_time_minimal(times[idx + tp_offset]))
            sl_times.append(_format_time_minimal(times[idx + sl_offset]))
        elif sl_offset > 0 and (tp_offset < 0 or sl_offset <= tp_offset):
            labels.append(-1)
            hold.append(sl_offset)
            tp_times.append(None)
            sl_times.append(_format_time_minimal(times[idx + sl_offset]))
        entries.append(_format_time_minimal(times[idx]))

    return (
        labels,
        hold,
        entries,
        tp_times,
        sl_times,
        same_bar_flags,
        max_favorable_moves_pct,
        max_adverse_moves_pct,
        skipped_entries,
    )


@mcp.tool()
def labels_triple_barrier(
    symbol: str,
    timeframe: TimeframeLiteral = "H1",
    limit: int = _DEFAULT_LABEL_LIMIT,
    horizon: int = _DEFAULT_LABEL_HORIZON,
    tp_abs: Optional[float] = None,
    sl_abs: Optional[float] = None,
    tp_pct: Optional[float] = None,
    sl_pct: Optional[float] = None,
    tp_ticks: Optional[float] = None,
    sl_ticks: Optional[float] = None,
    denoise: Optional[DenoiseSpec] = None,
    direction: Literal["long", "short"] = "long",  # type: ignore
    label_on: Literal["close", "high_low"] = "high_low",  # type: ignore
    same_bar_policy: Literal["sl_first", "tp_first", "neutral"] = "sl_first",  # type: ignore
    detail: DetailLiteral = "compact",
    lookback: int = _DEFAULT_LABEL_LOOKBACK,
) -> Dict[str, Any]:
    """Label each bar with triple-barrier outcomes using future path up to `horizon` bars.

    Barriers:
      - Absolute prices: tp_abs/sl_abs
      - Percent offsets: tp_pct/sl_pct (0.5 => 0.5%)
      - Ticks: tp_ticks/sl_ticks (trade_tick_size from symbol info)
      Use exactly one barrier unit family per call; mixed units are rejected.

    label_on='high_low' considers intrabar extremes for barrier hits; 'close' uses closes only.
    same_bar_policy explicitly resolves bars that touch both barriers; the default
    is conservative SL-first because the intrabar ordering is unknowable.
    direction='long' or 'short' controls which side is treated as TP/SL.
    Outputs label: +1 (TP first), -1 (SL first), 0 (neither by horizon), and holding_bars until decision.
    """

    def _run() -> Dict[str, Any]:
        try:
            mt5_gateway = create_mt5_gateway(
                ensure_connection_impl=ensure_mt5_connection_or_raise
            )
            mt5_gateway.ensure_connection()
            symbol_info = mt5_gateway.symbol_info(symbol)
            price_digits = symbol_price_digits(symbol_info) if symbol_info else 0
            trade_tick_size = None
            if symbol_info is not None:
                try:
                    trade_tick_size = float(getattr(symbol_info, "trade_tick_size", 0.0) or 0.0)
                except Exception:
                    trade_tick_size = None
            direction_value, direction_error = _normalize_trade_direction(direction)
            if direction_error or direction_value is None:
                return {"error": direction_error or "Invalid direction."}
            try:
                same_bar_policy_value = normalize_same_bar_policy(same_bar_policy)
            except ValueError as exc:
                return {"error": str(exc)}
            warnings_out: List[str] = []
            raw_detail = str(detail or "").strip().lower()
            if raw_detail not in {"full", "standard", "summary", "compact"}:
                return {
                    "error": (
                        "Invalid detail level. Use 'compact', 'standard', 'full', or 'summary'."
                    )
                }
            output_mode = normalize_output_detail(detail)
            if output_mode not in {"full", "standard", "summary", "compact"}:
                return {
                    "error": (
                        "Invalid detail level. Use 'compact', 'standard', 'full', or 'summary'."
                    )
                }
            barrier_values = {
                "tp_abs": tp_abs,
                "sl_abs": sl_abs,
                "tp_pct": tp_pct,
                "sl_pct": sl_pct,
                "tp_ticks": tp_ticks,
                "sl_ticks": sl_ticks,
            }
            try:
                barrier_values = validate_barrier_unit_family_exclusivity(barrier_values)
            except ValueError as exc:
                return {"error": str(exc)}
            horizon_bars = int(horizon)
            if horizon_bars <= 0:
                return {"error": "horizon must be greater than 0."}
            requested_lookback = max(1, int(lookback))
            sample_limit = max(1, int(limit))
            history_bars_requested = int(requested_lookback + horizon_bars)
            df = _fetch_history(symbol, timeframe, history_bars_requested, as_of=None)
            history_bars_fetched = int(len(df))
            if history_bars_fetched > history_bars_requested:
                df = df.tail(history_bars_requested).copy()
            history_bars_used = int(len(df))
            if len(df) < horizon_bars + 2:
                return {"error": "Insufficient history for labeling"}
            base_col = resolve_denoise_base_col(
                df, denoise, base_col="close", default_when="pre_ti"
            )
            closes = df[base_col].astype(float).to_numpy()
            highs = (
                df["high"].astype(float).to_numpy() if "high" in df.columns else None
            )
            lows = df["low"].astype(float).to_numpy() if "low" in df.columns else None
            times = df["time"].astype(float).to_numpy()

            pip_size = _get_pip_size(symbol)

            N = len(closes)
            barrier_kwargs = _build_barrier_kwargs_from(barrier_values)
            max_entry_index = N - horizon_bars
            sample_entry_price = next(
                (
                    float(closes[idx])
                    for idx in range(max(0, max_entry_index))
                    if math.isfinite(float(closes[idx])) and float(closes[idx]) > 0.0
                ),
                None,
            )
            if sample_entry_price is None:
                return {
                    "error": "No valid positive entry prices available for labeling."
                }
            sample_tp, sample_sl = _resolve_barrier_prices(
                price=sample_entry_price,
                direction=direction_value,
                pip_size=pip_size,
                adjust_inverted=False,
                **barrier_kwargs,
            )
            if sample_tp is None or sample_sl is None:
                if tp_abs is not None or sl_abs is not None:
                    return {
                        "error": (
                            "Invalid absolute TP/SL levels for the entry price. "
                            "tp_abs/sl_abs are price levels; use tp_pct/sl_pct or "
                            "tp_ticks/sl_ticks for offset-style barriers."
                        )
                    }
                return {
                    "error": (
                        "Missing barriers. Provide either tp_pct and sl_pct, "
                        "tp_abs and sl_abs, or tp_ticks and sl_ticks."
                    ),
                    "error_code": "barrier_parameters_missing",
                    "remediation": (
                        "Choose explicit TP/SL barriers scaled to the symbol's volatility. "
                        "Run forecast_volatility_estimate to read the per-bar sigma, then set "
                        "barriers to a multiple of it (e.g. ~1-3x the per-bar sigma scaled to "
                        "the horizon), or use forecast_barrier_optimize to tune from history. "
                        "Fixed values like tp_pct=0.5 are not volatility-aware and may be hit "
                        "within a bar (or never), producing near-random labels."
                    ),
                    "related_tools": [
                        "forecast_volatility_estimate",
                        "forecast_barrier_optimize",
                    ],
                    "examples": [
                        "forecast_volatility_estimate(symbol='EURUSD', timeframe='H1')  # find per-bar sigma first",
                        "labels_triple_barrier(symbol='EURUSD', tp_ticks=50, sl_ticks=50)",
                    ],
                }
            if not _barrier_prices_are_valid(
                price=sample_entry_price,
                direction=direction_value,
                tp_price=sample_tp,
                sl_price=sample_sl,
            ):
                if tp_abs is not None or sl_abs is not None:
                    if direction_value == "long":
                        constraint = "tp_abs must be above entry_price and sl_abs must be below entry_price"
                    else:
                        constraint = "tp_abs must be below entry_price and sl_abs must be above entry_price"
                    direction_hint = {
                        "direction": direction_value,
                        "entry_price": round(float(sample_entry_price), 8),
                        "constraint": constraint,
                        "tp_abs": tp_abs,
                        "sl_abs": sl_abs,
                        "resolved_tp": round(float(sample_tp), 8) if sample_tp is not None else None,
                        "resolved_sl": round(float(sample_sl), 8) if sample_sl is not None else None,
                    }
                    offset_hint = None
                    abs_values = [
                        abs(float(value))
                        for value in (tp_abs, sl_abs)
                        if value is not None and math.isfinite(float(value))
                    ]
                    if abs_values and max(abs_values) < abs(float(sample_entry_price)) * 0.2:
                        offset_hint = (
                            "The absolute levels are far from the entry price; if these are offsets, "
                            "use tp_pct/sl_pct or tp_ticks/sl_ticks instead."
                        )
                        direction_hint["offset_hint"] = offset_hint
                    return {
                        "error": (
                            "Invalid absolute TP/SL levels for the entry price: "
                            f"{constraint}. entry_price≈{sample_entry_price:.8g}, "
                            f"tp_abs={tp_abs}, sl_abs={sl_abs}. "
                            "Use tp_pct/sl_pct or tp_ticks/sl_ticks for offset-style barriers."
                        ),
                        "direction_hint": direction_hint,
                        **({"offset_hint": offset_hint} if offset_hint else {}),
                    }
                return {
                    "error": "Resolved TP/SL barriers are invalid for the entry price."
                }
            (
                labels,
                hold,
                t_entry,
                tp_times,
                sl_times,
                same_bar_flags,
                max_favorable_moves_pct,
                max_adverse_moves_pct,
                skipped_entries,
            ) = _build_triple_barrier_outputs(
                closes=closes,
                highs=highs,
                lows=lows,
                times=times,
                horizon=horizon_bars,
                label_on=label_on,
                direction_value=direction_value,
                pip_size=pip_size,
                barrier_kwargs=barrier_kwargs,
                same_bar_policy=same_bar_policy_value,
            )
            rows_before_labeling = int(N)
            labelable_rows = int(max(0, max_entry_index))
            rows_after_labeling = int(len(labels))
            horizon_trimmed = int(max(0, rows_before_labeling - labelable_rows))
            horizon_trim_fraction = (
                float(horizon_trimmed) / float(rows_before_labeling)
                if rows_before_labeling > 0
                else 0.0
            )
            labeling_coverage = {
                "rows_before_labeling": rows_before_labeling,
                "labelable_rows_before_invalid_skips": labelable_rows,
                "rows_after_labeling": rows_after_labeling,
                "horizon_trimmed": horizon_trimmed,
                "horizon_trim_fraction": round(horizon_trim_fraction, 4),
                "invalid_entry_skipped": int(skipped_entries),
            }
            if rows_after_labeling < requested_lookback:
                warnings_out.append(
                    f"Only {rows_after_labeling} labeled row(s) were available for "
                    f"lookback={requested_lookback}; each label needs {horizon_bars} "
                    "future bar(s). Increase lookback if you need a larger labeled window."
                )

            payload: Dict[str, Any] = {
                "success": True,
                "symbol": symbol,
                "timeframe": timeframe,
                "direction": direction_value,
                "horizon": horizon_bars,
                "same_bar_policy": same_bar_policy_value,
                "rows_before_labeling": rows_before_labeling,
                "rows_after_labeling": rows_after_labeling,
                "horizon_trimmed": horizon_trimmed,
                "labeling_coverage": labeling_coverage,
                "history_bars_requested": history_bars_requested,
                "history_bars_fetched": history_bars_fetched,
                "history_bars_used": history_bars_used,
                "sample_limit": sample_limit,
                "entries": t_entry,
                "labels": labels,
                "outcomes": [
                    "same_bar_neutral" if same_bar and label == 0 else _label_outcome(label)
                    for label, same_bar in zip(labels, same_bar_flags)
                ],
                "holding_bars": hold,
                "tp_time": tp_times,
                "sl_time": sl_times,
                "same_bar": same_bar_flags,
            }
            if price_digits > 0:
                payload["price_precision"] = int(price_digits)
            if trade_tick_size is not None and trade_tick_size > 0:
                payload["trade_tick_size"] = trade_tick_size
            if output_mode == "full":
                payload["label_legend"] = {
                    "1": {
                        "code": 1,
                        "label": "tp_first",
                        "description": "Take-profit barrier hit before stop-loss (profitable outcome)",
                    },
                    "-1": {
                        "code": -1,
                        "label": "sl_first",
                        "description": "Stop-loss barrier hit before take-profit (loss outcome)",
                    },
                    "0": {
                        "code": 0,
                        "label": "hold",
                        "description": "Neither barrier hit within horizon bars (neutral/timeout outcome)",
                    },
                }
            elif output_mode in {"compact", "standard"}:
                payload["label_key"] = {"1": "tp_first", "-1": "sl_first", "0": "hold"}
            if warnings_out:
                payload["warnings"] = list(warnings_out)
            if skipped_entries > 0:
                payload.setdefault("warnings", []).append(
                    f"Skipped {int(skipped_entries)} entries with invalid or non-positive entry prices."
                )
                payload["skipped_entries"] = int(skipped_entries)
            if output_mode in ("summary", "compact", "standard"):
                import numpy as _np

                n = min(requested_lookback, len(labels))
                lab_tail = labels[-n:] if n > 0 else labels
                hold_tail = hold[-n:] if n > 0 else hold
                recommended_lookback = max(horizon_bars * 4, 30)
                bars_insufficient_for_horizon = int(n) <= horizon_bars * 2
                sample_quality = {
                    "status": "low" if int(n) < recommended_lookback else "ok",
                    "lookback": int(n),
                    "requested_lookback": requested_lookback,
                    "history_bars_requested": history_bars_requested,
                    "history_bars_used": history_bars_used,
                    "minimum_recommended": int(recommended_lookback),
                    "bars_insufficient_for_horizon": bool(bars_insufficient_for_horizon),
                }
                if int(n) < recommended_lookback:
                    sample_quality["reason"] = (
                        f"Only {int(n)} labeled rows are summarized; "
                        f"{recommended_lookback}+ is recommended for horizon={horizon_bars}."
                    )
                    warnings_out.append(
                        "Summary lookback is small relative to horizon; label counts may be unstable. "
                        f"Use lookback>={recommended_lookback} for a basic read."
                    )
                counts = {
                    "tp": int(sum(1 for v in lab_tail if v == 1)),
                    "sl": int(sum(1 for v in lab_tail if v == -1)),
                    "neutral": int(sum(1 for v in lab_tail if v == 0)),
                }
                med_hold = (
                    float(_np.median(_np.array(hold_tail, dtype=float)))
                    if hold_tail
                    else float("nan")
                )
                summary = {
                    "lookback": int(n),
                    "counts": counts,
                    "neutral_rate": (
                        round(float(counts["neutral"] / n), 6) if n else None
                    ),
                    "barrier_resolution_rate": (
                        round(float((counts["tp"] + counts["sl"]) / n), 6)
                        if n
                        else None
                    ),
                    "tp_rate": round(float(counts["tp"] / n), 6) if n else None,
                    "sl_rate": round(float(counts["sl"] / n), 6) if n else None,
                    "median_holding_bars": med_hold,
                    "sample_quality": sample_quality,
                }
                if n and counts["neutral"] / n >= 0.8:
                    warnings_out.append(
                        "At least 80% of summarized labels are neutral timeouts. "
                        "Tighten the barriers or increase horizon to produce more hits."
                    )
                favorable_tail = max_favorable_moves_pct[-n:] if n > 0 else max_favorable_moves_pct
                adverse_tail = max_adverse_moves_pct[-n:] if n > 0 else max_adverse_moves_pct
                if favorable_tail or adverse_tail:
                    summary["max_observed_move_pct"] = {
                        "favorable": round(float(max(favorable_tail or [0.0])), 6),
                        "adverse": round(float(max(adverse_tail or [0.0])), 6),
                    }
                if counts["tp"] == 0 and counts["sl"] == 0 and counts["neutral"] > 0:
                    summary["explanation"] = (
                        "All labels are neutral because no price path hit TP or SL within "
                        "the horizon. Label 0 means a timeout/hold outcome, not a "
                        "calculation failure; consider tightening barriers or increasing "
                        "horizon if you need more barrier hits."
                    )
                    moves = summary.get("max_observed_move_pct")
                    if isinstance(moves, dict):
                        tp_range = _neutral_barrier_pct_range(moves.get("favorable"))
                        sl_range = _neutral_barrier_pct_range(moves.get("adverse"))
                        if tp_range or sl_range:
                            summary["suggested_pct_barriers"] = {
                                key: value
                                for key, value in {
                                    "tp_pct": tp_range,
                                    "sl_pct": sl_range,
                                }.items()
                                if value is not None
                            }
                            summary["suggestion_basis"] = (
                                "Ranges are 40-80% of the max observed favorable/adverse move "
                                "inside the summary lookback; use forecast_barrier_optimize for "
                                "objective-specific tuning."
                            )
                if output_mode == "summary":
                    out = {
                        "success": True,
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "direction": direction_value,
                        "horizon": horizon_bars,
                        "rows_before_labeling": rows_before_labeling,
                        "rows_after_labeling": rows_after_labeling,
                        "horizon_trimmed": horizon_trimmed,
                        "labeling_coverage": labeling_coverage,
                        "history_bars_requested": history_bars_requested,
                        "history_bars_fetched": history_bars_fetched,
                        "history_bars_used": history_bars_used,
                        "sample_limit": sample_limit,
                        "sample_quality_status": sample_quality["status"],
                        "summary": summary,
                    }
                    if price_digits > 0:
                        out["price_precision"] = int(price_digits)
                    if trade_tick_size is not None and trade_tick_size > 0:
                        out["trade_tick_size"] = trade_tick_size
                    if warnings_out:
                        out["warnings"] = list(warnings_out)
                    if skipped_entries > 0:
                        out.setdefault("warnings", []).append(
                            f"Skipped {int(skipped_entries)} entries with invalid or non-positive entry prices."
                        )
                        out["skipped_entries"] = int(skipped_entries)
                    return out
                payload["sample_quality_status"] = sample_quality["status"]
                payload["summary"] = summary
                sample_n = min(n, sample_limit)
                sample_indices = list(
                    range(max(0, len(labels) - sample_n), len(labels))
                )

                def _sample_rows(indices: List[int]) -> List[Dict[str, Any]]:
                    return [
                        _triple_barrier_sample_row(
                            idx=idx,
                            closes=closes,
                            t_entry=t_entry,
                            labels=labels,
                            hold=hold,
                            tp_times=tp_times,
                            sl_times=sl_times,
                            same_bar_flags=same_bar_flags,
                            direction_value=direction_value,
                            pip_size=pip_size,
                            barrier_kwargs=barrier_kwargs,
                            price_digits=price_digits,
                        )
                        for idx in indices
                    ]

                if output_mode == "compact":
                    outcome_indices = [
                        len(labels) - len(lab_tail) + idx
                        for idx, value in enumerate(lab_tail)
                        if value != 0
                    ]
                    if outcome_indices:
                        compact_sample_limit = min(sample_limit, _COMPACT_LABEL_SAMPLE_SIZE)
                        sample_indices = outcome_indices[-compact_sample_limit:]
                        payload["sample_basis"] = "outcomes"
                    else:
                        sample_n = min(n, sample_limit, _COMPACT_LABEL_SAMPLE_SIZE)
                        sample_indices = list(
                            range(max(0, len(labels) - sample_n), len(labels))
                        )
                        payload["sample_basis"] = "recent"
                    payload["sample_size"] = int(len(sample_indices))
                    non_neutral_count = int(counts["tp"] + counts["sl"])
                    if (
                        n > 0
                        and non_neutral_count > 0
                        and non_neutral_count / n < 0.10
                    ):
                        neutral_pct = (float(counts["neutral"]) / float(n)) * 100.0
                        payload["sample_context"] = {
                            "neutral": int(counts["neutral"]),
                            "total": int(n),
                            "neutral_pct": round(neutral_pct, 2),
                            "non_neutral": non_neutral_count,
                            "tp": int(counts["tp"]),
                            "sl": int(counts["sl"]),
                            "note": (
                                f"{counts['neutral']} of {n} labels are neutral; "
                                f"showing {len(sample_indices)} non-neutral rows."
                            ),
                        }
                    if len(sample_indices) < n:
                        payload["sample_note"] = (
                            "data rows show non-neutral outcomes in the lookback window when present; "
                            f"otherwise the most recent {len(sample_indices)} observations."
                        )
                    payload["data"] = _sample_rows(sample_indices)
                    for key in (
                        "rows_before_labeling",
                        "rows_after_labeling",
                        "horizon_trimmed",
                        "sample_quality_status",
                    ):
                        payload.pop(key, None)
                    compact_sample_quality = summary.get("sample_quality")
                    if isinstance(compact_sample_quality, dict):
                        compact_sample_quality = dict(compact_sample_quality)
                        compact_sample_quality.pop("history_bars_requested", None)
                        compact_sample_quality.pop("history_bars_used", None)
                        summary["sample_quality"] = compact_sample_quality
                    for key in (
                        "entries",
                        "labels",
                        "outcomes",
                        "holding_bars",
                        "tp_time",
                        "sl_time",
                        "same_bar",
                    ):
                        payload.pop(key, None)
                elif output_mode == "standard":
                    payload["sample_basis"] = "recent"
                    sample_indices = sample_indices[-min(n, sample_limit):]
                    payload["sample_size"] = int(len(sample_indices))
                    payload["data_note"] = (
                        "data rows cover the recent summary lookback window."
                    )
                    payload["data"] = _sample_rows(sample_indices)
                    for key in (
                        "entries",
                        "labels",
                        "outcomes",
                        "holding_bars",
                        "tp_time",
                        "sl_time",
                        "same_bar",
                    ):
                        payload.pop(key, None)
            return payload
        except MT5ConnectionError as exc:
            return {"error": str(exc)}
        except Exception as exc:
            return {"error": f"Error computing triple-barrier labels: {str(exc)}"}

    return run_logged_operation(
        logger,
        operation="labels_triple_barrier",
        symbol=symbol,
        timeframe=timeframe,
        direction=direction,
        horizon=horizon,
        detail=detail,
        func=_run,
    )

