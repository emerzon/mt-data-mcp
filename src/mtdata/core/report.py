from typing import Any, Dict, Optional, List, Literal, Union

from .server import mcp, _auto_connect_wrapper
from .schema import DenoiseSpec
from .report_utils import render_enhanced_report, format_number, _get_indicator_value

TemplateName = Literal['basic','advanced','scalping','intraday','swing','position']


def _report_error_text(message: Any) -> str:
    text = str(message).strip()
    if not text:
        text = 'Unknown error.'
    return f"error: {text}\n"


def _report_error_payload(message: Any) -> Dict[str, Any]:
    text = str(message).strip()
    if not text:
        text = 'Unknown error.'
    return {"error": text}


@mcp.tool()
@_auto_connect_wrapper
def report_generate(
    symbol: str,
    horizon: Optional[int] = None,
    template: TemplateName = 'basic',
    denoise: Optional[DenoiseSpec] = None,
    params: Optional[Dict[str, Any]] = None,
    output: Literal['toon', 'markdown'] = 'toon',
) -> Union[str, Dict[str, Any]]:
    """Generate a consolidated, information-dense analysis report with compact multi-format output.

    - template: 'basic' (context, pivot, EWMA vol, backtest->best forecast, MC barrier grid, patterns)
                'advanced' (adds regimes, HAR-RV, conformal),
                or style-specific ('scalping' | 'intraday' | 'swing' | 'position').
    - params: optional dict to tune steps/spacing, grids, and optionally override timeframe per template via 'timeframe'.
    - denoise: pass-through to candle fetching (e.g., {method:'ema', params:{alpha:0.2}, columns:['close']}).
    - output: 'toon' (structured TOON) or 'markdown' (rendered report text).
    """
    try:
        output_mode = str(output or 'toon').strip().lower()
        name = (template or 'basic').lower().strip()
        p = dict(params or {})

        try:
            from .report_templates import (
                template_basic as _t_basic,
                template_advanced as _t_advanced,
                template_scalping as _t_scalping,
                template_intraday as _t_intraday,
                template_swing as _t_swing,
                template_position as _t_position,
            )
        except Exception as ex:
            if output_mode == 'markdown':
                return _report_error_text(f"Failed to import report templates: {ex}")
            return _report_error_payload(f"Failed to import report templates: {ex}")

        default_horizon = {
            'basic': 12,
            'advanced': 12,
            'scalping': 8,
            'intraday': 12,
            'swing': 24,
            'position': 30,
        }
        if isinstance(p.get('horizon'), (int, float)):
            eff_horizon = int(p.get('horizon'))
        elif horizon is not None and int(horizon) > 0:
            eff_horizon = int(horizon)
        else:
            eff_horizon = default_horizon.get(name, 12)

        if name == 'basic':
            rep = _t_basic(symbol, eff_horizon, denoise, p)
        elif name == 'advanced':
            rep = _t_advanced(symbol, eff_horizon, denoise, p)
        elif name == 'scalping':
            rep = _t_scalping(symbol, eff_horizon, denoise, p)
        elif name == 'intraday':
            rep = _t_intraday(symbol, eff_horizon, denoise, p)
        elif name == 'swing':
            rep = _t_swing(symbol, eff_horizon, denoise, p)
        elif name == 'position':
            rep = _t_position(symbol, eff_horizon, denoise, p)
        else:
            msg = f"Unknown template: {template}. Use one of basic, advanced, scalping, intraday, swing, position."
            if output_mode == 'markdown':
                return _report_error_text(msg)
            return _report_error_payload(msg)

        if not isinstance(rep, dict):
            msg = 'Report template returned an unexpected payload.'
            if output_mode == 'markdown':
                return _report_error_text(msg)
            return _report_error_payload(msg)
        if rep.get('error'):
            msg = rep.get('error')
            if output_mode == 'markdown':
                return _report_error_text(msg)
            return _report_error_payload(msg)

        summ: List[str] = []
        try:
            ctx = rep.get('sections', {}).get('context', {})
            last = ctx.get('last_snapshot') or {}
            price = last.get('close')
            ema20 = _get_indicator_value(last, 'EMA_20')
            ema50 = _get_indicator_value(last, 'EMA_50')
            rsi = _get_indicator_value(last, 'RSI_14')
            if price is not None:
                summ.append(f"close={format_number(price)}")
            if ema20 is not None and ema50 is not None:
                trend_note = (
                    'trend: above EMAs'
                    if float(price or 0) > float(ema20) > float(ema50)
                    else 'trend: mixed'
                )
                summ.append(trend_note)
            if rsi is not None:
                summ.append(f"RSI={format_number(rsi)}")
        except Exception:
            pass
        try:
            piv = rep.get('sections', {}).get('pivot', {})
            lev_rows = piv.get('levels')
            methods_meta = piv.get('methods')
            chosen_method = None
            if isinstance(methods_meta, list):
                for meta in methods_meta:
                    if not isinstance(meta, dict):
                        continue
                    name = str(meta.get('method') or '').strip()
                    if name:
                        chosen_method = name
                        break
            chosen_method = chosen_method or 'classic'
            available_methods: List[str] = []
            if isinstance(lev_rows, list):
                for row in lev_rows:
                    if not isinstance(row, dict):
                        continue
                    for key in row.keys():
                        if key == 'level':
                            continue
                        key_str = str(key)
                        if key_str not in available_methods:
                            available_methods.append(key_str)
            if available_methods and chosen_method not in available_methods:
                chosen_method = available_methods[0]

            def _pivot_lookup(level_key: str):
                target = level_key.lower()
                alt = 'pivot' if target == 'pp' else None
                if not isinstance(lev_rows, list):
                    return None
                for row in lev_rows:
                    if not isinstance(row, dict):
                        continue
                    lvl_name = str(row.get('level') or '').strip().lower()
                    if lvl_name == target or (alt and lvl_name == alt):
                        return row.get(chosen_method)
                return None

            pp = _pivot_lookup('PP')
            r1 = _pivot_lookup('R1')
            s1 = _pivot_lookup('S1')
            if pp is not None and r1 is not None and s1 is not None:
                summ.append(f"pivot {chosen_method} PP={format_number(pp)} (R1={format_number(r1)}, S1={format_number(s1)})")
        except Exception:
            pass
        try:
            vol = rep.get('sections', {}).get('volatility', {})
            if isinstance(vol, dict):
                hs = vol.get('horizon_sigma_price') or vol.get('horizon_sigma_return')
                if hs is not None:
                    summ.append(f"h{eff_horizon} sigma={format_number(hs)}")
        except Exception:
            pass
        try:
            fc = rep.get('sections', {}).get('forecast', {})
            if isinstance(fc, dict) and 'method' in fc:
                summ.append(f"forecast={fc.get('method')}")
        except Exception:
            pass
        try:
            bar = rep.get('sections', {}).get('barriers', {})
            if isinstance(bar, dict) and any(k in bar for k in ('long','short')):
                for dname in ('long','short'):
                    sub = bar.get(dname)
                    if not isinstance(sub, dict):
                        continue
                    best = sub.get('best') if isinstance(sub, dict) else None
                    if not best:
                        continue
                    tp = best.get('tp'); sl = best.get('sl'); edge = best.get('edge')
                    details: List[str] = []
                    details.append(f"dir={dname}")
                    if tp is not None:
                        details.append(f"tp={format_number(tp)}%")
                    if sl is not None:
                        details.append(f"sl={format_number(sl)}%")
                    if edge is not None:
                        details.append(f"edge={format_number(edge)}")
                    if details:
                        summ.append("barrier best " + ' '.join(details))
            else:
                best = bar.get('best') if isinstance(bar, dict) else None
                direction = bar.get('direction') if isinstance(bar, dict) else None
                if best:
                    tp = best.get('tp')
                    sl = best.get('sl')
                    edge = best.get('edge')
                    details: List[str] = []
                    if direction:
                        details.append(f"dir={str(direction)}")
                    if tp is not None:
                        details.append(f"tp={format_number(tp)}%")
                    if sl is not None:
                        details.append(f"sl={format_number(sl)}%")
                    if edge is not None:
                        details.append(f"edge={format_number(edge)}")
                    if details:
                        summ.append("barrier best " + ' '.join(details))
        except Exception:
            pass
        rep['summary'] = summ

        if output_mode == 'markdown':
            return render_enhanced_report(rep)
        return rep
    except Exception as exc:
        msg = f"Error generating report: {exc}"
        if str(output or '').strip().lower() == 'markdown':
            return _report_error_text(msg)
        return _report_error_payload(msg)
