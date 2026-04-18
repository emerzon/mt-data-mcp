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
from ..utils.denoise import _resolve_denoise_base_col
from ..utils.mt5 import MT5ConnectionError, ensure_mt5_connection_or_raise
from ..utils.utils import _format_time_minimal
from ._mcp_instance import mcp
from .execution_logging import run_logged_operation
from .mt5_gateway import get_mt5_gateway
from .output_contract import normalize_output_detail
from .schema import DenoiseSpec, TimeframeLiteral

logger = logging.getLogger(__name__)
_COMPACT_LABEL_SAMPLE_SIZE = 10


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
    List[int], List[int], List[str], List[Optional[str]], List[Optional[str]], int
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
            adjust_inverted=True,
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
        high_values = highs if highs is not None else closes
        low_values = lows if lows is not None else closes
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

    for idx in np.flatnonzero(valid_entry_mask):
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

    return labels, hold, entries, tp_times, sl_times, skipped_entries


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
    tp_pips: Optional[float] = None,
    sl_pips: Optional[float] = None,
    denoise: Optional[DenoiseSpec] = None,
    direction: Literal["long", "short"] = "long",  # type: ignore
    label_on: Literal["close", "high_low"] = "high_low",  # type: ignore
    detail: Literal["full", "summary", "compact", "summary_only"] = "compact",  # type: ignore
    summary_only: bool = False,
    lookback: int = 300,
) -> Dict[str, Any]:
    """Label each bar with triple-barrier outcomes using future path up to `horizon` bars.

    Barriers:
      - Absolute prices: tp_abs/sl_abs
      - Percent offsets: tp_pct/sl_pct (0.5 => 0.5%)
      - Ticks: tp_pips/sl_pips (trade_tick_size from symbol info)

    label_on='high_low' considers intrabar extremes for barrier hits; 'close' uses closes only.
    When both TP and SL are touched in the same high/low bar, the result is treated
    conservatively as SL-first because the intrabar ordering is unknowable.
    direction='long' or 'short' controls which side is treated as TP/SL.
    Outputs label: +1 (TP first), -1 (SL first), 0 (neither by horizon), and holding_bars until decision.
    """

    def _run() -> Dict[str, Any]:
        try:
            get_mt5_gateway(
                ensure_connection_impl=ensure_mt5_connection_or_raise
            ).ensure_connection()
            direction_value, direction_error = _normalize_trade_direction(direction)
            if direction_error or direction_value is None:
                return {"error": direction_error or "Invalid direction."}
            output_mode = normalize_output_detail(
                detail,
                aliases={"summary_only": "summary"},
            )
            if isinstance(summary_only, str):
                summary_only_flag = summary_only.strip().lower() in {
                    "1",
                    "true",
                    "yes",
                    "y",
                    "on",
                }
            else:
                summary_only_flag = bool(summary_only)
            if summary_only_flag:
                output_mode = "summary"
            if output_mode not in {"full", "summary", "compact"}:
                return {
                    "error": "Invalid detail level. Use 'compact', 'full', 'summary', or 'summary_only'."
                }
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
            barrier_kwargs = _build_barrier_kwargs_from(
                {
                    "tp_abs": tp_abs,
                    "sl_abs": sl_abs,
                    "tp_pct": tp_pct,
                    "sl_pct": sl_pct,
                    "tp_pips": tp_pips,
                    "sl_pips": sl_pips,
                }
            )
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
                adjust_inverted=True,
                **barrier_kwargs,
            )
            if sample_tp is None or sample_sl is None:
                return {
                    "error": (
                        "Provide barriers via tp_abs/sl_abs or tp_pct/sl_pct or "
                        "tp_pips/sl_pips. If you need help choosing values, run "
                        "forecast_barrier_optimize first."
                    )
                }
            if not _barrier_prices_are_valid(
                price=sample_entry_price,
                direction=direction_value,
                tp_price=sample_tp,
                sl_price=sample_sl,
            ):
                return {
                    "error": "Resolved TP/SL barriers are invalid for the entry price."
                }
            labels, hold, t_entry, tp_times, sl_times, skipped_entries = (
                _build_triple_barrier_outputs(
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
            )

            # Build label legend for interpretability
            label_legend = {
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

            payload: Dict[str, Any] = {
                "success": True,
                "symbol": symbol,
                "timeframe": timeframe,
                "direction": direction_value,
                "horizon": int(horizon),
                "label_legend": label_legend,
                "entries": t_entry,
                "labels": labels,
                "holding_bars": hold,
                "tp_time": tp_times,
                "sl_time": sl_times,
            }
            if skipped_entries > 0:
                payload["warnings"] = [
                    f"Skipped {int(skipped_entries)} entries with invalid or non-positive entry prices."
                ]
                payload["skipped_entries"] = int(skipped_entries)
            if output_mode in ("summary", "compact"):
                import numpy as _np

                n = min(int(lookback), len(labels))
                lab_tail = labels[-n:] if n > 0 else labels
                hold_tail = hold[-n:] if n > 0 else hold
                counts = {
                    "pos": int(sum(1 for v in lab_tail if v == 1)),
                    "neg": int(sum(1 for v in lab_tail if v == -1)),
                    "neut": int(sum(1 for v in lab_tail if v == 0)),
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
                if output_mode == "summary":
                    out = {
                        "success": True,
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "direction": direction_value,
                        "horizon": int(horizon),
                        "summary": summary,
                    }
                    if skipped_entries > 0:
                        out["warnings"] = [
                            f"Skipped {int(skipped_entries)} entries with invalid or non-positive entry prices."
                        ]
                        out["skipped_entries"] = int(skipped_entries)
                    return out
                payload["summary"] = summary
                sample_n = n
                if output_mode == "compact":
                    sample_n = min(n, _COMPACT_LABEL_SAMPLE_SIZE)
                    payload["sample_size"] = int(sample_n)
                    if sample_n < n:
                        payload["sample_note"] = (
                            f"entries, labels, and timing arrays show the most recent {sample_n} observations."
                        )
                if sample_n > 0:
                    payload["entries"] = t_entry[-sample_n:]
                    payload["labels"] = lab_tail[-sample_n:]
                    payload["holding_bars"] = hold_tail[-sample_n:]
                    payload["tp_time"] = tp_times[-sample_n:]
                    payload["sl_time"] = sl_times[-sample_n:]
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
