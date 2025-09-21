from typing import Any, Dict, Optional
from ..schema import DenoiseSpec
from .basic import template_basic
from ..report_utils import merge_params, market_snapshot, apply_market_gates, attach_multi_timeframes


def template_scalping(
    symbol: str,
    horizon: int,
    denoise: Optional[DenoiseSpec],
    params: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    p = merge_params(params, {
        'context_limit': 400,
        'context_tail': 60,
        'backtest_steps': 20,
        'backtest_spacing': 10,
        'backtest_rmse_tolerance': 0.10,
        'patterns_limit': 200,
        'mode': 'pips',
        'tp_min': 2.0, 'tp_max': 15.0, 'tp_steps': 6,
        'sl_min': 4.0, 'sl_max': 30.0, 'sl_steps': 7,
        'top_k': 5,
    })
    # Choose default timeframe for scalping if not provided
    if 'timeframe' not in p:
        p['timeframe'] = 'M5'
    base = template_basic(symbol, horizon, denoise, p)
    snap = market_snapshot(symbol)
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
