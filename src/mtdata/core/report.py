from typing import Any, Dict, Optional, List, Literal

from .server import mcp, _auto_connect_wrapper
from .schema import DenoiseSpec
from .report_utils import render_markdown

TemplateName = Literal['basic','simple','advanced','scalping','intraday','swing','position']


@mcp.tool()
@_auto_connect_wrapper
def generate_report(
    symbol: str,
    horizon: Optional[int] = None,
    template: TemplateName = 'basic',
    denoise: Optional[DenoiseSpec] = None,
    params: Optional[Dict[str, Any]] = None,
    output: Literal['summary','full'] = 'summary',
) -> Dict[str, Any]:
    """Generate a consolidated, information‑dense analysis report in Markdown.

    - template: 'basic'/'simple' (context, pivot, EWMA vol, backtest→best forecast, MC barrier grid, patterns)
                'advanced' (adds regimes, HAR‑RV, conformal),
                or style‑specific ('scalping' | 'intraday' | 'swing' | 'position').
    - params: optional dict to tune steps/spacing, grids, and (optionally) override timeframe per template via 'timeframe'.
    - denoise: pass‑through to candle fetching (e.g., {method:'ema', params:{alpha:0.2}, columns:['close']}).
    """
    try:
        name = (template or 'basic').lower().strip()
        p = dict(params or {})

        # Import template builders
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
            return {"error": f"Failed to import report templates: {ex}"}

        default_horizon = {
            'basic': 12,
            'simple': 12,
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

        # Build report using selected template (templates infer timeframe as needed)
        if name in ('basic', 'simple'):
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
            return {"error": f"Unknown template: {template}. Use one of basic, advanced, scalping, intraday, swing, position."}

        # Build a compact summary for quick scan
        summ: List[str] = []
        try:
            ctx = rep.get('sections', {}).get('context', {})
            last = ctx.get('last_snapshot') or {}
            price = last.get('close')
            ema20 = last.get('EMA_20') if 'EMA_20' in last else last.get('ema_20')
            ema50 = last.get('EMA_50') if 'EMA_50' in last else last.get('ema_50')
            rsi = last.get('RSI_14') if 'RSI_14' in last else last.get('rsi_14')
            if price is not None:
                summ.append(f"close≈{price}")
            if ema20 is not None and ema50 is not None:
                summ.append("trend: above EMAs" if float(price or 0) > float(ema20) > float(ema50) else "trend: mixed")
            if rsi is not None:
                summ.append(f"RSI≈{rsi}")
        except Exception:
            pass
        try:
            piv = rep.get('sections', {}).get('pivot', {})
            lev = piv.get('levels') or {}
            if lev:
                pp = lev.get('PP')
                r1 = lev.get('R1'); s1 = lev.get('S1')
                if pp is not None and r1 is not None and s1 is not None:
                    summ.append(f"pivot PP={pp} (R1={r1}, S1={s1})")
        except Exception:
            pass
        try:
            vol = rep.get('sections', {}).get('volatility', {})
            if isinstance(vol, dict):
                hs = vol.get('horizon_sigma_price') or vol.get('horizon_sigma_return')
                if hs is not None:
                    summ.append(f"h{eff_horizon} σ≈{hs}")
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
            best = bar.get('best') if isinstance(bar, dict) else None
            if best:
                summ.append(f"barrier best tp={best.get('tp')}% sl={best.get('sl')}% edge={best.get('edge')}")
        except Exception:
            pass
        rep['summary'] = summ
        rep['success'] = True

        # Render Markdown
        markdown = render_markdown(rep)
        return {
            'success': True,
            'markdown': markdown,
            'report': rep,
        }
    except Exception as e:
        return {"error": f"Error generating report: {str(e)}"}
