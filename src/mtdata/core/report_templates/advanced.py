from typing import Any, Dict, Optional
from ..schema import DenoiseSpec
from .basic import template_basic, _get_raw_result


def template_advanced(
    symbol: str,
    horizon: int,
    denoise: Optional[DenoiseSpec],
    params: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    # Ensure a timeframe for subcalls
    p = dict(params or {})
    tf = str(p.get('timeframe', 'H1'))
    p['timeframe'] = tf
    
    base = template_basic(symbol, horizon, denoise, p)
    
    if isinstance(base, str):
        # If base is a string (error), return it
        return {'error': f'template_basic returned string: {base}'}
    elif not isinstance(base, dict):
        return {'error': f'template_basic returned unexpected type: {type(base)}'}

    # Rest of the function continues as before...

    # Regime summaries
    from ..regime import regime_detect
    p = dict(params or {})
    bocpd = _get_raw_result(regime_detect,
        symbol=symbol,
        timeframe=tf,
        limit=int(p.get('regime_limit', 1500)),
        method='bocpd', threshold=float(p.get('cp_threshold', 0.6)), output='summary', lookback=int(p.get('regime_lookback', 300))
    )
    hmm = _get_raw_result(regime_detect,
        symbol=symbol,
        timeframe=tf,
        limit=int(p.get('regime_limit', 1500)),
        method='hmm', params={'n_states': int(p.get('hmm_states', 3))}, output='compact', lookback=int(p.get('regime_lookback', 300))
    )
    base.setdefault('sections', {})['regime'] = {
        'bocpd': bocpd if 'error' in bocpd else {'summary': bocpd.get('summary')},
        'hmm': hmm if 'error' in hmm else {'summary': hmm.get('summary')},
    }

    # HAR-RV volatility summary
    from ..forecast import forecast_volatility_estimate
    har = _get_raw_result(forecast_volatility_estimate, symbol=symbol, timeframe=tf, horizon=int(horizon), method='har_rv', params={'rv_timeframe': 'M5', 'days': 150, 'window_w': 5, 'window_m': 22})
    if 'error' in har:
        base['sections']['volatility_har_rv'] = {'error': har['error']}
    else:
        base['sections']['volatility_har_rv'] = {
            'sigma_bar_return': har.get('sigma_bar_return'),
            'horizon_sigma_return': har.get('horizon_sigma_return'),
        }

    # Conformal intervals around chosen method
    try:
        best_method = base.get('sections', {}).get('backtest', {}).get('best_method', {}).get('method')
    except Exception:
        best_method = None
    if best_method:
        from ..forecast import forecast_conformal_intervals
        conf = _get_raw_result(forecast_conformal_intervals,
            symbol=symbol,
            timeframe=tf,
            method=best_method,
            horizon=int(horizon),
            steps=int(p.get('conformal_steps', 25)),
            spacing=int(p.get('conformal_spacing', 10)),
            alpha=float(p.get('conformal_alpha', 0.1)),
        )
        if 'error' in conf:
            base['sections']['forecast_conformal'] = {'error': conf['error'], 'method': best_method}
        else:
            base['sections']['forecast_conformal'] = {
                'method': best_method,
                'lower_price': conf.get('lower_price'),
                'upper_price': conf.get('upper_price'),
                'per_step_q': conf.get('conformal', {}).get('per_step_q'),
                'alpha': conf.get('ci_alpha'),
            }

    return base
