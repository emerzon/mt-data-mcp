from typing import Any, Dict, Optional, Literal, List
import logging
import math

from ._mcp_instance import mcp
from .execution_logging import run_logged_operation
from .mt5_gateway import get_mt5_gateway
from ..utils.mt5 import MT5ConnectionError, ensure_mt5_connection_or_raise
from .schema import TimeframeLiteral, DenoiseSpec
from ..forecast.common import fetch_history as _fetch_history
from ..utils.utils import _format_time_minimal
from ..utils.denoise import _resolve_denoise_base_col
from ..utils.barriers import (
    get_pip_size as _get_pip_size,
    resolve_barrier_prices as _resolve_barrier_prices,
    build_barrier_kwargs_from as _build_barrier_kwargs_from,
    barrier_prices_are_valid as _barrier_prices_are_valid,
    normalize_trade_direction as _normalize_trade_direction,
)

logger = logging.getLogger(__name__)


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
    label_on: Literal['close','high_low'] = 'high_low',  # type: ignore
    output: Literal['full','summary','compact','summary_only'] = 'full',  # type: ignore
    summary_only: bool = False,
    lookback: int = 300,
) -> Dict[str, Any]:
    """Label each bar with triple-barrier outcomes using future path up to `horizon` bars.

    Barriers:
      - Absolute prices: tp_abs/sl_abs
      - Percent offsets: tp_pct/sl_pct (0.5 => 0.5%)
      - Ticks: tp_pips/sl_pips (trade_tick_size from symbol info)

    label_on='high_low' considers intrabar extremes for barrier hits; 'close' uses closes only.
    direction='long' or 'short' controls which side is treated as TP/SL.
    Outputs label: +1 (TP first), -1 (SL first), 0 (neither by horizon), and holding_bars until decision.
    """
    def _run() -> Dict[str, Any]:
        try:
            get_mt5_gateway(ensure_connection_impl=ensure_mt5_connection_or_raise).ensure_connection()
            direction_value, direction_error = _normalize_trade_direction(direction)
            if direction_error or direction_value is None:
                return {"error": direction_error or "Invalid direction."}
            output_mode = str(output).strip().lower()
            if output_mode == 'summary_only':
                output_mode = 'summary'
            if isinstance(summary_only, str):
                summary_only_flag = summary_only.strip().lower() in {"1", "true", "yes", "y", "on"}
            else:
                summary_only_flag = bool(summary_only)
            if summary_only_flag:
                output_mode = 'summary'
            if output_mode not in {'full', 'summary', 'compact'}:
                return {"error": "Invalid output mode. Use 'full', 'summary', 'compact', or summary_only=True."}
            df = _fetch_history(symbol, timeframe, int(max(limit, horizon + 50)), as_of=None)
            if len(df) < horizon + 2:
                return {"error": "Insufficient history for labeling"}
            base_col = _resolve_denoise_base_col(df, denoise, base_col='close', default_when='pre_ti')
            closes = df[base_col].astype(float).to_numpy()
            highs = df['high'].astype(float).to_numpy() if 'high' in df.columns else None
            lows = df['low'].astype(float).to_numpy() if 'low' in df.columns else None
            times = df['time'].astype(float).to_numpy()

            pip_size = _get_pip_size(symbol)

            N = len(closes)
            labels: List[int] = []
            hold: List[int] = []
            t_entry: List[str] = []
            tp_times: List[Optional[str]] = []
            sl_times: List[Optional[str]] = []
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
                return {"error": "No valid positive entry prices available for labeling."}
            sample_tp, sample_sl = _resolve_barrier_prices(
                price=sample_entry_price,
                direction=direction_value,
                pip_size=pip_size,
                adjust_inverted=True,
                **barrier_kwargs,
            )
            if sample_tp is None or sample_sl is None:
                return {"error": "Provide barriers via tp_abs/sl_abs or tp_pct/sl_pct or tp_pips/sl_pips"}
            if not _barrier_prices_are_valid(
                price=sample_entry_price,
                direction=direction_value,
                tp_price=sample_tp,
                sl_price=sample_sl,
            ):
                return {"error": "Resolved TP/SL barriers are invalid for the entry price."}
            skipped_entries = 0

            for i in range(0, max_entry_index):
                p0 = float(closes[i])
                if not math.isfinite(p0) or p0 <= 0.0:
                    skipped_entries += 1
                    continue
                tp, sl = _resolve_barrier_prices(
                    price=p0,
                    direction=direction_value,
                    pip_size=pip_size,
                    adjust_inverted=True,
                    **barrier_kwargs,
                )
                if tp is None or sl is None:
                    skipped_entries += 1
                    continue
                if not _barrier_prices_are_valid(
                    price=p0,
                    direction=direction_value,
                    tp_price=tp,
                    sl_price=sl,
                ):
                    skipped_entries += 1
                    continue

                hit_tp = -1
                hit_sl = -1
                for k in range(1, int(horizon) + 1):
                    idx = i + k
                    if idx >= N:
                        break
                    if label_on == 'close':
                        x = closes[idx]
                        tp_hit = x >= tp if direction_value == "long" else x <= tp
                        sl_hit = x <= sl if direction_value == "long" else x >= sl
                        if tp_hit and hit_tp < 0:
                            hit_tp = k
                        if sl_hit and hit_sl < 0:
                            hit_sl = k
                    else:
                        h = highs[idx] if highs is not None else closes[idx]
                        lw = lows[idx] if lows is not None else closes[idx]
                        tp_hit = h >= tp if direction_value == "long" else lw <= tp
                        sl_hit = lw <= sl if direction_value == "long" else h >= sl
                        if tp_hit and hit_tp < 0:
                            hit_tp = k
                        if sl_hit and hit_sl < 0:
                            hit_sl = k
                    if hit_tp > 0 or hit_sl > 0:
                        break
                if hit_tp < 0 and hit_sl < 0:
                    labels.append(0); hold.append(int(horizon)); tp_times.append(None); sl_times.append(None)
                elif hit_tp > 0 and (hit_sl < 0 or hit_tp < hit_sl):
                    labels.append(1); hold.append(hit_tp); tp_times.append(_format_time_minimal(times[i+hit_tp])); sl_times.append(None)
                elif hit_sl > 0 and (hit_tp < 0 or hit_sl < hit_tp):
                    labels.append(-1); hold.append(hit_sl); tp_times.append(None); sl_times.append(_format_time_minimal(times[i+hit_sl]))
                else:
                    labels.append(0); hold.append(min(hit_tp, hit_sl)); tp_times.append(_format_time_minimal(times[i+hit_tp])); sl_times.append(_format_time_minimal(times[i+hit_sl]))
                t_entry.append(_format_time_minimal(times[i]))

            payload: Dict[str, Any] = {
                "success": True,
                "symbol": symbol,
                "timeframe": timeframe,
                "direction": direction_value,
                "horizon": int(horizon),
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
            if output_mode in ('summary','compact'):
                import numpy as _np
                n = min(int(lookback), len(labels))
                lab_tail = labels[-n:] if n > 0 else labels
                hold_tail = hold[-n:] if n > 0 else hold
                counts = {"pos": int(sum(1 for v in lab_tail if v == 1)),
                          "neg": int(sum(1 for v in lab_tail if v == -1)),
                          "neut": int(sum(1 for v in lab_tail if v == 0))}
                med_hold = float(_np.median(_np.array(hold_tail, dtype=float))) if hold_tail else float('nan')
                summary = {"lookback": int(n), "counts": counts, "median_holding_bars": med_hold}
                if output_mode == 'summary':
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
                if n > 0:
                    payload["entries"] = t_entry[-n:]
                    payload["labels"] = lab_tail
                    payload["holding_bars"] = hold_tail
                    payload["tp_time"] = tp_times[-n:]
                    payload["sl_time"] = sl_times[-n:]
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
        output=output,
        func=_run,
    )
