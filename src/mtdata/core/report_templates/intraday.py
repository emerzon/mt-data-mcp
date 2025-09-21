from typing import Any, Dict, Optional
from ..schema import DenoiseSpec
from .basic import template_basic
from ..report_utils import merge_params, market_snapshot, apply_market_gates, attach_multi_timeframes


def template_intraday(
    symbol: str,
    horizon: int,
    denoise: Optional[DenoiseSpec],
    params: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    p = merge_params(params, {
        'context_limit': 300,
        'context_tail': 50,
        'backtest_steps': 25,
        'backtest_spacing': 10,
        'backtest_rmse_tolerance': 0.05,
        'patterns_limit': 150,
        'mode': 'pct',
        'tp_min': 0.2, 'tp_max': 1.0, 'tp_steps': 5,
        'sl_min': 0.2, 'sl_max': 1.0, 'sl_steps': 5,
        'top_k': 5,
    })
    if 'timeframe' not in p:
        p['timeframe'] = 'H1'
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
        extra_timeframes=p.get('extra_timeframes') or ['M15','H1','H4','D1'],
        pivot_timeframes=p.get('pivot_timeframes') or ['D1','W1']
    )
    return base
