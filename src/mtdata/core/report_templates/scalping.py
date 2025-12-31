from typing import Any, Dict, Optional
from ..schema import DenoiseSpec
from .basic import template_basic
from ..report_utils import merge_params, market_snapshot, apply_market_gates, attach_multi_timeframes
from ...utils.barriers import get_pip_size as _get_pip_size


def template_scalping(
    symbol: str,
    horizon: int,
    denoise: Optional[DenoiseSpec],
    params: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    user_keys = set(params.keys()) if isinstance(params, dict) else set()
    p = merge_params(params, {
        'context_limit': 400,
        'context_tail': 60,
        'backtest_steps': 20,
        'backtest_spacing': 10,
        'backtest_rmse_tolerance': 0.10,
        'patterns_limit': 200,
        'mode': 'pips',
        'grid_style': 'fixed',
        'tp_min': 2.0, 'tp_max': 15.0, 'tp_steps': 6,
        'sl_min': 4.0, 'sl_max': 30.0, 'sl_steps': 7,
        'top_k': 5,
        # Barrier optimization defaults
        'objective': 'ev_uncond',
        'params': {'spread_bps': 1.0, 'slippage_bps': 0.5, 'rr_min': 0.7, 'rr_max': 1.6},
    })
    # Choose default timeframe for scalping if not provided
    if 'timeframe' not in p:
        p['timeframe'] = 'M5'
    snap = market_snapshot(symbol)
    if str(p.get('mode', '')).lower() == 'pips':
        last_price = None
        spread_ticks = None
        if isinstance(snap, dict) and not snap.get('error'):
            bid = snap.get('bid')
            ask = snap.get('ask')
            try:
                if bid is not None and ask is not None:
                    last_price = (float(bid) + float(ask)) / 2.0
                elif bid is not None:
                    last_price = float(bid)
                elif ask is not None:
                    last_price = float(ask)
            except Exception:
                last_price = None
            try:
                spread_ticks = float(snap.get('spread_ticks')) if snap.get('spread_ticks') is not None else None
            except Exception:
                spread_ticks = None
        tick_size = _get_pip_size(symbol)
        if last_price and tick_size and last_price > 1000:
            def _set_default(key: str, value: float) -> None:
                if key in user_keys:
                    return
                p[key] = value

            if spread_ticks and spread_ticks > 0:
                tp_min_ticks = max(spread_ticks * 2.0, 50.0)
                tp_max_ticks = max(spread_ticks * 6.0, tp_min_ticks * 1.5)
                sl_min_ticks = max(spread_ticks * 2.5, 50.0)
                sl_max_ticks = max(spread_ticks * 8.0, sl_min_ticks * 1.5)
            else:
                tp_min_ticks = (last_price * 0.0001) / tick_size
                tp_max_ticks = (last_price * 0.0006) / tick_size
                sl_min_ticks = (last_price * 0.00015) / tick_size
                sl_max_ticks = (last_price * 0.0010) / tick_size

            _set_default('tp_min', tp_min_ticks)
            _set_default('tp_max', tp_max_ticks)
            _set_default('sl_min', sl_min_ticks)
            _set_default('sl_max', sl_max_ticks)

    base = template_basic(symbol, horizon, denoise, p)
    base.setdefault('sections', {})['market'] = snap
    gates = apply_market_gates(snap if isinstance(snap, dict) else {}, p)
    if gates:
        base['sections']['execution_gates'] = gates
    attach_multi_timeframes(
        base,
        symbol,
        denoise,
        extra_timeframes=p.get('extra_timeframes') or ['M1','M5','M15','D1'],
        pivot_timeframes=p.get('pivot_timeframes') or ['D1']
    )
    return base
