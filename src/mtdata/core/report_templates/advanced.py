import math
from typing import Any, Dict, List, Optional

from ...shared.schema import DenoiseSpec
from ..report.utils import (
    current_only_section_omission as _current_only_section_omission,
)
from ..report.utils import (
    is_bounded_report_window as _is_bounded_report_window,
)
from ..report.utils import (
    report_section_enabled,
)
from .basic import (
    _first_volatility_value,
    _get_raw_result,
    template_basic,
)


def _finite_number_list(value: Any) -> List[float]:
    if not isinstance(value, list):
        return []
    numbers: List[float] = []
    for item in value:
        try:
            number = float(item)
        except (TypeError, ValueError):
            return []
        if not math.isfinite(number):
            return []
        numbers.append(number)
    return numbers


def _conformal_report_section(
    payload: Dict[str, Any],
    *,
    method: str,
    horizon: int,
) -> Dict[str, Any]:
    lower = _finite_number_list(payload.get("lower_price"))
    upper = _finite_number_list(payload.get("upper_price"))
    forecast = _finite_number_list(payload.get("forecast_price"))
    conformal = payload.get("conformal")
    conformal = conformal if isinstance(conformal, dict) else {}
    per_step_q = _finite_number_list(conformal.get("per_step_q"))
    times = payload.get("forecast_time")
    times = times if isinstance(times, list) else []
    expected = int(horizon)
    if not (
        len(lower) == expected
        and len(upper) == expected
        and len(forecast) == expected
        and len(times) == expected
        and len(per_step_q) == expected
    ):
        return {
            "status": "error",
            "method": method,
            "error": (
                "Conformal interval output was incomplete: expected one timestamp, "
                "point forecast, lower bound, upper bound, and calibration quantile "
                f"for each of {expected} steps."
            ),
        }
    intervals: List[Dict[str, Any]] = []
    for index, (time_value, point, low, high, quantile) in enumerate(
        zip(times, forecast, lower, upper, per_step_q, strict=True),
        start=1,
    ):
        if time_value in (None, "") or not low <= point <= high:
            return {
                "status": "error",
                "method": method,
                "error": (
                    "Conformal interval output contained a missing timestamp or an "
                    f"unordered bound at step {index}."
                ),
            }
        intervals.append(
            {
                "step": index,
                "time": time_value,
                "forecast": point,
                "lower_price": low,
                "upper_price": high,
                "per_step_q": quantile,
            }
        )
    return {
        "method": method,
        "intervals": intervals,
        "lower_price": lower,
        "upper_price": upper,
        "per_step_q": per_step_q,
        "ci_alpha": payload.get("ci_alpha"),
        "nominal_confidence_level": payload.get("nominal_confidence_level"),
        "empirical_coverage": payload.get("empirical_coverage"),
        "coverage_status": payload.get("coverage_status"),
        "conformal": conformal,
    }


def template_advanced(
    symbol: str,
    horizon: int,
    denoise: Optional[DenoiseSpec],
    params: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    # Ensure a timeframe for subcalls
    p = dict(params or {})
    tf = str(p.get('timeframe', 'H1'))
    start = p.get('start')
    end = p.get('end')
    p['timeframe'] = tf
    p.setdefault('barrier_method', 'hmm_mc')
    p.setdefault('search_profile', 'medium')
    p.setdefault('forecast_ci_alpha', 0.1)
    p.setdefault('patterns_mode', 'classic')
    p.setdefault('patterns_extra_modes', ['elliott'])
    
    base = template_basic(symbol, horizon, denoise, p)
    
    if isinstance(base, str):
        # If base is a string (error), return it
        return {'error': f'template_basic returned string: {base}'}
    elif not isinstance(base, dict):
        return {'error': f'template_basic returned unexpected type: {type(base)}'}

    # Rest of the function continues as before...

    p = dict(params or {})
    if report_section_enabled(p, 'regime'):
        # Regime summaries
        from ..regime import regime_detect
        bocpd = _get_raw_result(regime_detect,
            symbol=symbol,
            timeframe=tf,
            fetch_limit=int(p.get('regime_limit', 1500)),
            start=start,
            end=end,
            method='bocpd', threshold=float(p.get('cp_threshold', 0.6)), detail='summary', lookback=int(p.get('regime_lookback', 300)),
            denoise=denoise,
        )
        hmm = _get_raw_result(regime_detect,
            symbol=symbol,
            timeframe=tf,
            fetch_limit=int(p.get('regime_limit', 1500)),
            start=start,
            end=end,
            method='hmm', params={'n_states': int(p.get('hmm_states', 3))}, detail='compact', lookback=int(p.get('regime_lookback', 300)),
            denoise=denoise,
        )
        base.setdefault('sections', {})['regime'] = {
            'bocpd': bocpd if 'error' in bocpd else {'summary': bocpd.get('summary')},
            'hmm': hmm if 'error' in hmm else {'summary': hmm.get('summary')},
        }

    # HAR-RV volatility summary
    if report_section_enabled(p, 'volatility_har_rv'):
        from ..forecast import forecast_volatility_estimate
        har = _get_raw_result(
            forecast_volatility_estimate,
            symbol=symbol,
            timeframe=tf,
            horizon=int(horizon),
            method='har_rv',
            start=start,
            end=end,
            params={'rv_timeframe': 'M5', 'days': 150, 'window_w': 5, 'window_m': 22},
            denoise=denoise,
        )
        if 'error' in har:
            base['sections']['volatility_har_rv'] = {'error': har['error']}
        else:
            base['sections']['volatility_har_rv'] = {
                'volatility_per_bar': _first_volatility_value(
                    har,
                    ('volatility_per_bar', 'sigma_bar_price'),
                ),
                'volatility_horizon': _first_volatility_value(
                    har,
                    ('volatility_horizon', 'horizon_sigma_price'),
                ),
            }

    # Conformal intervals around chosen method
    try:
        best_method = base.get('sections', {}).get('backtest', {}).get('best_method', {}).get('method')
    except Exception:
        best_method = None
    if not best_method:
        forecast_section = base.get('sections', {}).get('forecast')
        if isinstance(forecast_section, dict):
            best_method = forecast_section.get('method')
    forecast_section = base.get('sections', {}).get('forecast')
    has_native_interval = (
        isinstance(forecast_section, dict)
        and forecast_section.get('lower_price') not in (None, "", [], {})
        and forecast_section.get('upper_price') not in (None, "", [], {})
    )
    if not report_section_enabled(p, 'forecast_conformal'):
        pass
    elif _is_bounded_report_window(start, end):
        base['sections']['forecast_conformal'] = _current_only_section_omission(
            'forecast_conformal', start=start, end=end
        )
    elif has_native_interval:
        base['sections']['forecast_conformal'] = {
            'status': 'omitted',
            'reason': 'native_forecast_interval_available',
            'method': best_method,
        }
    elif best_method:
        from ..forecast import forecast_conformal_intervals
        conformal_spacing = max(int(horizon), int(p.get('conformal_spacing', 10)))
        conf = _get_raw_result(forecast_conformal_intervals,
            symbol=symbol,
            timeframe=tf,
            method=best_method,
            horizon=int(horizon),
            steps=int(p.get('conformal_steps', 25)),
            spacing=conformal_spacing,
            ci_alpha=float(p.get('conformal_alpha', 0.1)),
            denoise=denoise,
            detail='full',
        )
        if 'error' in conf:
            base['sections']['forecast_conformal'] = {'error': conf['error'], 'method': best_method}
        else:
            base['sections']['forecast_conformal'] = _conformal_report_section(
                conf,
                method=str(best_method),
                horizon=int(horizon),
            )
    elif p.get('_report_section_controls_active'):
        base['sections']['forecast_conformal'] = {
            'error': 'No backtest-selected method was available for conformal intervals.'
        }

    return base
