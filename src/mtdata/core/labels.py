import logging
import math
from typing import Any, Dict, List, Literal, Optional

import numpy as np

from ..forecast.common import fetch_history as _fetch_history
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
    normalize_trade_direction as _normalize_trade_direction,
)
from ..utils.barriers import (
    resolve_barrier_prices as _resolve_barrier_prices,
)
from ..utils.barriers import validate_barrier_unit_family_exclusivity
from ..utils.denoise import _resolve_denoise_base_col
from ..utils.mt5 import MT5ConnectionError, ensure_mt5_connection_or_raise
from ..utils.utils import _format_time_minimal
from ._mcp_instance import mcp
from .execution_logging import run_logged_operation
from .mt5_gateway import create_mt5_gateway
from .output_contract import normalize_output_detail
from ..shared.schema import DenoiseSpec, DetailLiteral, TimeframeLiteral

logger = logging.getLogger(__name__)
_COMPACT_LABEL_SAMPLE_SIZE = 10


def _label_outcome(label: int) -> str:
    if label == 1:
        return "tp"
    if label == -1:
        return "sl"
    return "neutral"


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
) -> tuple[
    List[int],
    List[int],
    List[str],
    List[Optional[str]],
    List[Optional[str]],
    List[float],
    List[float],
    int,
]:
    max_entry_index = len(closes) - int(horizon)
    if max_entry_index <= 0:
        return [], [], [], [], [], 0

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
        elif sl_offset > 0 and (tp_offset < 0 or sl_offset <= tp_offset):
            labels.append(-1)
            hold.append(sl_offset)
            tp_times.append(None)
            sl_times.append(_format_time_minimal(times[idx + sl_offset]))
        else:
            labels.append(0)
            hold.append(min(tp_offset, sl_offset))
            tp_times.append(_format_time_minimal(times[idx + tp_offset]))
            sl_times.append(_format_time_minimal(times[idx + sl_offset]))
        entries.append(_format_time_minimal(times[idx]))

    return (
        labels,
        hold,
        entries,
        tp_times,
        sl_times,
        max_favorable_moves_pct,
        max_adverse_moves_pct,
        skipped_entries,
    )


@mcp.tool()
def labels_triple_barrier(
    symbol: str,
    timeframe: TimeframeLiteral = "H1",
    limit: int = 1200,
    horizon: int = 12,
    tp_abs: Optional[float] = None,
    sl_abs: Optional[float] = None,
    tp_pct: Optional[float] = None,
    sl_pct: Optional[float] = None,
    tp_ticks: Optional[float] = None,
    sl_ticks: Optional[float] = None,
    denoise: Optional[DenoiseSpec] = None,
    direction: Literal["long", "short"] = "long",  # type: ignore
    label_on: Literal["close", "high_low"] = "high_low",  # type: ignore
    detail: DetailLiteral = "compact",
    lookback: int = 300,
) -> Dict[str, Any]:
    """Label each bar with triple-barrier outcomes using future path up to `horizon` bars.

    Barriers:
      - Absolute prices: tp_abs/sl_abs
      - Percent offsets: tp_pct/sl_pct (0.5 => 0.5%)
      - Ticks: tp_ticks/sl_ticks (trade_tick_size from symbol info)

    label_on='high_low' considers intrabar extremes for barrier hits; 'close' uses closes only.
    When both TP and SL are touched in the same high/low bar, the result is treated
    conservatively as SL-first because the intrabar ordering is unknowable.
    direction='long' or 'short' controls which side is treated as TP/SL.
    Outputs label: +1 (TP first), -1 (SL first), 0 (neither by horizon), and holding_bars until decision.
    """

    def _run() -> Dict[str, Any]:
        try:
            create_mt5_gateway(
                ensure_connection_impl=ensure_mt5_connection_or_raise
            ).ensure_connection()
            direction_value, direction_error = _normalize_trade_direction(direction)
            if direction_error or direction_value is None:
                return {"error": direction_error or "Invalid direction."}
            warnings_out: List[str] = []
            output_mode = normalize_output_detail(detail)
            if output_mode == "standard":
                output_mode = "compact"
            if output_mode not in {"full", "summary", "compact"}:
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
            df = _fetch_history(
                symbol, timeframe, int(max(limit, horizon + 50)), as_of=None
            )
            if len(df) < horizon + 2:
                return {"error": "Insufficient history for labeling"}
            base_col = _resolve_denoise_base_col(
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
            max_entry_index = N - int(horizon)
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
                return {
                    "error": (
                        "Missing barriers. Provide either tp_pct and sl_pct, "
                        "tp_abs and sl_abs, or tp_ticks and sl_ticks."
                    )
                }
            if not _barrier_prices_are_valid(
                price=sample_entry_price,
                direction=direction_value,
                tp_price=sample_tp,
                sl_price=sample_sl,
            ):
                if tp_abs is not None or sl_abs is not None:
                    return {
                        "error": (
                            "Invalid absolute TP/SL levels for the entry price. "
                            "tp_abs/sl_abs are absolute price levels, not offsets; "
                            "use tp_ticks/sl_ticks or tp_pct/sl_pct for offset-style barriers."
                        )
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
                max_favorable_moves_pct,
                max_adverse_moves_pct,
                skipped_entries,
            ) = _build_triple_barrier_outputs(
                closes=closes,
                highs=highs,
                lows=lows,
                times=times,
                horizon=int(horizon),
                label_on=label_on,
                direction_value=direction_value,
                pip_size=pip_size,
                barrier_kwargs=barrier_kwargs,
            )

            payload: Dict[str, Any] = {
                "success": True,
                "symbol": symbol,
                "timeframe": timeframe,
                "direction": direction_value,
                "horizon": int(horizon),
                "entries": t_entry,
                "labels": labels,
                "outcomes": [_label_outcome(label) for label in labels],
                "holding_bars": hold,
                "tp_time": tp_times,
                "sl_time": sl_times,
            }
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
            elif output_mode == "compact":
                payload["label_key"] = {"1": "tp_first", "-1": "sl_first", "0": "hold"}
            if warnings_out:
                payload["warnings"] = list(warnings_out)
            if skipped_entries > 0:
                payload.setdefault("warnings", []).append(
                    f"Skipped {int(skipped_entries)} entries with invalid or non-positive entry prices."
                )
                payload["skipped_entries"] = int(skipped_entries)
            if output_mode in ("summary", "compact"):
                import numpy as _np

                n = min(int(lookback), len(labels))
                lab_tail = labels[-n:] if n > 0 else labels
                hold_tail = hold[-n:] if n > 0 else hold
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
                    "median_holding_bars": med_hold,
                }
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
                if output_mode == "summary":
                    out = {
                        "success": True,
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "direction": direction_value,
                        "horizon": int(horizon),
                        "summary": summary,
                    }
                    if warnings_out:
                        out["warnings"] = list(warnings_out)
                    if skipped_entries > 0:
                        out.setdefault("warnings", []).append(
                            f"Skipped {int(skipped_entries)} entries with invalid or non-positive entry prices."
                        )
                        out["skipped_entries"] = int(skipped_entries)
                    return out
                payload["summary"] = summary
                sample_n = n
                sample_indices = list(range(max(0, len(labels) - sample_n), len(labels)))
                if output_mode == "compact":
                    outcome_indices = [
                        len(labels) - len(lab_tail) + idx
                        for idx, value in enumerate(lab_tail)
                        if value != 0
                    ]
                    if outcome_indices:
                        sample_indices = outcome_indices[-_COMPACT_LABEL_SAMPLE_SIZE:]
                        payload["sample_basis"] = "outcomes"
                    else:
                        sample_n = min(n, _COMPACT_LABEL_SAMPLE_SIZE)
                        sample_indices = list(range(max(0, len(labels) - sample_n), len(labels)))
                        payload["sample_basis"] = "recent"
                    payload["sample_size"] = int(len(sample_indices))
                    if len(sample_indices) < n:
                        payload["sample_note"] = (
                            "entries, labels, and timing arrays show non-neutral outcomes in the lookback window when present; "
                            f"otherwise the most recent {len(sample_indices)} observations."
                        )
                if sample_indices:
                    payload["entries"] = [t_entry[idx] for idx in sample_indices]
                    payload["labels"] = [labels[idx] for idx in sample_indices]
                    payload["outcomes"] = [_label_outcome(labels[idx]) for idx in sample_indices]
                    payload["holding_bars"] = [hold[idx] for idx in sample_indices]
                    payload["tp_time"] = [tp_times[idx] for idx in sample_indices]
                    payload["sl_time"] = [sl_times[idx] for idx in sample_indices]
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
