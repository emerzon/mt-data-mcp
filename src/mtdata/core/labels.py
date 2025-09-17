from typing import Any, Dict, Optional, Literal
import numpy as np
import pandas as pd
import MetaTrader5 as mt5

from .server import mcp, _auto_connect_wrapper
from .schema import TimeframeLiteral, DenoiseSpec
from ..forecast.common import fetch_history as _fetch_history
from ..utils.utils import _format_time_minimal as _format_time_minimal_util
from ..utils.denoise import _apply_denoise as _apply_denoise_util


@mcp.tool()
@_auto_connect_wrapper
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
    label_on: Literal['close','high_low'] = 'high_low',  # type: ignore
    output: Literal['full','summary','compact'] = 'full',  # type: ignore
    lookback: int = 300,
) -> Dict[str, Any]:
    """Label each bar with triple-barrier outcomes using future path up to `horizon` bars.

    Barriers:
      - Absolute prices: tp_abs/sl_abs
      - Percent offsets: tp_pct/sl_pct (0.5 => 0.5%)
      - Pips: tp_pips/sl_pips (pip≈10×point for 5/3-digit FX)

    label_on='high_low' considers intrabar extremes for barrier hits; 'close' uses closes only.
    Outputs label: +1 (TP first), -1 (SL first), 0 (neither by horizon), and holding_bars until decision.
    """
    try:
        df = _fetch_history(symbol, timeframe, int(max(limit, horizon + 50)), as_of=None)
        if len(df) < horizon + 2:
            return {"error": "Insufficient history for labeling"}
        base_col = 'close'
        if denoise:
            try:
                added = _apply_denoise_util(df, denoise, default_when='pre_ti')
                if f"{base_col}_dn" in added:
                    base_col = f"{base_col}_dn"
            except Exception:
                pass
        closes = df[base_col].astype(float).to_numpy()
        highs = df['high'].astype(float).to_numpy() if 'high' in df.columns else None
        lows = df['low'].astype(float).to_numpy() if 'low' in df.columns else None
        times = df['time'].astype(float).to_numpy()

        # Pip size heuristic
        pip_size = None
        try:
            info = mt5.symbol_info(symbol)
            if info is not None:
                digits = int(getattr(info, 'digits', 0) or 0)
                point = float(getattr(info, 'point', 0.0) or 0.0)
                pip_size = float(point * (10.0 if digits in (3,5) else 1.0)) if point > 0 else None
        except Exception:
            pip_size = None

        N = len(closes)
        labels: List[int] = []
        hold: List[int] = []
        t_entry: List[str] = []
        tp_times: List[Optional[str]] = []
        sl_times: List[Optional[str]] = []

        for i in range(0, N - int(horizon) - 1):
            p0 = float(closes[i])
            # Compute absolute barriers
            tp = tp_abs; sl = sl_abs
            if tp is None:
                if tp_pct is not None:
                    tp = p0 * (1.0 + float(tp_pct) / 100.0)
                elif tp_pips is not None and pip_size is not None:
                    tp = p0 + float(tp_pips) * float(pip_size)
            if sl is None:
                if sl_pct is not None:
                    sl = p0 * (1.0 - float(sl_pct) / 100.0)
                elif sl_pips is not None and pip_size is not None:
                    sl = p0 - float(sl_pips) * float(pip_size)
            if tp is None or sl is None:
                return {"error": "Provide barriers via tp_abs/sl_abs or tp_pct/sl_pct or tp_pips/sl_pips"}

            l = 0
            hit_tp = -1
            hit_sl = -1
            for k in range(1, int(horizon) + 1):
                idx = i + k
                if idx >= N:
                    break
                if label_on == 'close':
                    x = closes[idx]
                    if x >= tp and hit_tp < 0:
                        hit_tp = k
                    if x <= sl and hit_sl < 0:
                        hit_sl = k
                else:
                    h = highs[idx] if highs is not None else closes[idx]
                    lw = lows[idx] if lows is not None else closes[idx]
                    if h >= tp and hit_tp < 0:
                        hit_tp = k
                    if lw <= sl and hit_sl < 0:
                        hit_sl = k
                if hit_tp > 0 or hit_sl > 0:
                    break
            # Decide label and holding
            if hit_tp < 0 and hit_sl < 0:
                labels.append(0); hold.append(int(horizon)); tp_times.append(None); sl_times.append(None)
            elif hit_tp > 0 and (hit_sl < 0 or hit_tp < hit_sl):
                labels.append(1); hold.append(hit_tp); tp_times.append(_format_time_minimal_util(times[i+hit_tp])); sl_times.append(None)
            elif hit_sl > 0 and (hit_tp < 0 or hit_sl < hit_tp):
                labels.append(-1); hold.append(hit_sl); tp_times.append(None); sl_times.append(_format_time_minimal_util(times[i+hit_sl]))
            else:
                # tie -> neutral label, track first hit
                labels.append(0); hold.append(min(hit_tp, hit_sl)); tp_times.append(_format_time_minimal_util(times[i+hit_tp])); sl_times.append(_format_time_minimal_util(times[i+hit_sl]))
            t_entry.append(_format_time_minimal_util(times[i]))

        payload: Dict[str, Any] = {
            "success": True,
            "symbol": symbol,
            "timeframe": timeframe,
            "horizon": int(horizon),
            "entries": t_entry,
            "labels": labels,
            "holding_bars": hold,
            "tp_time": tp_times,
            "sl_time": sl_times,
            "params_used": {
                "tp_abs": tp_abs, "sl_abs": sl_abs, "tp_pct": tp_pct, "sl_pct": sl_pct, "tp_pips": tp_pips, "sl_pips": sl_pips,
                "label_on": label_on,
            }
        }
        if output in ('summary','compact'):
            import numpy as _np
            n = min(int(lookback), len(labels))
            lab_tail = labels[-n:] if n > 0 else labels
            hold_tail = hold[-n:] if n > 0 else hold
            # Basic stats
            counts = {"pos": int(sum(1 for v in lab_tail if v == 1)),
                      "neg": int(sum(1 for v in lab_tail if v == -1)),
                      "neut": int(sum(1 for v in lab_tail if v == 0))}
            med_hold = float(_np.median(_np.array(hold_tail, dtype=float))) if hold_tail else float('nan')
            summary = {"lookback": int(n), "counts": counts, "median_holding_bars": med_hold}
            if output == 'summary':
                return {"success": True, "symbol": symbol, "timeframe": timeframe, "horizon": int(horizon), "summary": summary}
            # compact: include tail only
            payload["summary"] = summary
            if n > 0:
                payload["entries"] = t_entry[-n:]
                payload["labels"] = lab_tail
                payload["holding_bars"] = hold_tail
                payload["tp_time"] = tp_times[-n:]
                payload["sl_time"] = sl_times[-n:]
        return payload
    except Exception as e:
        return {"error": f"Error computing triple-barrier labels: {str(e)}"}
