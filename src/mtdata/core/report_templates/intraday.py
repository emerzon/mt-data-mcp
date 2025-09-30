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
        # Dynamic defaults for volatility grid
        'grid_style': 'volatility',
        'vol_window': 250,
        'vol_min_mult': 0.6,
        'vol_max_mult': 2.2,
        'vol_sl_multiplier': 1.7,
        'vol_sl_steps': 9,
        'vol_floor_pct': 0.2,
        # Risk/reward filter defaults
        'rr_min': 0.8,
        'rr_max': 2.0,
        'refine': True,
        'refine_radius': 0.35,
        'refine_steps': 5,
        'tp_min': 0.25, 'tp_max': 1.5, 'tp_steps': 7,
        'sl_min': 0.25, 'sl_max': 2.5, 'sl_steps': 9,
        'top_k': 5,
        # Barrier optimization defaults
        'objective': 'ev_uncond',
        'params': {'spread_bps': 1.0, 'slippage_bps': 0.5, 'rr_min': 0.8, 'rr_max': 2.0},
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
