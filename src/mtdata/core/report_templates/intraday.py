from typing import Any, Dict, Optional
from ..schema import DenoiseSpec
from .common import build_report_with_market
from ..report_utils import merge_params


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
        'objective': 'ev',
        'params': {'spread_bps': 1.0, 'slippage_bps': 0.5, 'rr_min': 0.8, 'rr_max': 2.0},
    })
    if 'timeframe' not in p:
        p['timeframe'] = 'H1'
    return build_report_with_market(symbol, horizon, denoise, p, default_extra=['M15','H1','H4','D1'], default_pivots=['D1','W1'])
