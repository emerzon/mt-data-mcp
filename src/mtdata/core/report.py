from typing import Any, Dict, Optional, List, Literal, Union
import time
import warnings

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


def _append_diagnostic_warning(report: Dict[str, Any], message: str) -> None:
    text = str(message or "").strip()
    if not text:
        return
    diagnostics = report.get("diagnostics")
    if not isinstance(diagnostics, dict):
        diagnostics = {}
    warnings_list = diagnostics.get("warnings")
    if not isinstance(warnings_list, list):
        warnings_list = []
    if text not in warnings_list:
        warnings_list.append(text)
    diagnostics["warnings"] = warnings_list
    report["diagnostics"] = diagnostics


@mcp.tool()
@_auto_connect_wrapper
def report_generate(
    symbol: str,
    horizon: Optional[int] = None,
    template: TemplateName = 'basic',
    timeframe: Optional[str] = None,
    methods: Optional[Union[str, List[str]]] = None,
    denoise: Optional[DenoiseSpec] = None,
    params: Optional[Dict[str, Any]] = None,
    output: Literal['toon', 'markdown'] = 'toon',
) -> Union[str, Dict[str, Any]]:
    """Generate a consolidated, information-dense analysis report with compact multi-format output.

    - template: 'basic' (context, pivot, EWMA vol, backtest->best forecast, MC barrier grid, patterns)
                'advanced' (adds regimes, HAR-RV, conformal),
                or style-specific ('scalping' | 'intraday' | 'swing' | 'position').
    - params: optional dict to tune steps/spacing, grids, and optionally override timeframe per template via 'timeframe' or methods via 'methods'.
    - denoise: pass-through to candle fetching (e.g., {method:'ema', params:{alpha:0.2}, columns:['close']}).
    - output: 'toon' (structured TOON) or 'markdown' (rendered report text).
    """
    try:
        started = time.perf_counter()
        output_mode = str(output or 'toon').strip().lower()
        name = (template or 'basic').lower().strip()
        p = dict(params or {})
        if timeframe:
            p['timeframe'] = str(timeframe)
        if methods is not None:
            p['methods'] = methods

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

        captured_warnings: List[str] = []
        with warnings.catch_warnings(record=True) as warning_records:
            warnings.simplefilter("always")
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
        for warning_obj in warning_records:
            try:
                warning_text = str(warning_obj.message).strip()
            except Exception:
                warning_text = ""
            if warning_text:
                captured_warnings.append(warning_text)

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
        if captured_warnings:
            for warning_text in captured_warnings:
                _append_diagnostic_warning(rep, warning_text)

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
            calc_basis = piv.get("calculation_basis") if isinstance(piv.get("calculation_basis"), dict) else {}
            session_boundary = calc_basis.get("session_boundary")
            display_tz = calc_basis.get("display_timezone") or piv.get("timezone")
            context_parts: List[str] = []
            if session_boundary:
                context_parts.append(f"session={session_boundary}")
            if display_tz:
                context_parts.append(f"display_tz={display_tz}")
            if context_parts:
                summ.append("pivot context " + " ".join(context_parts))
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
                method_name = str(fc.get('method'))
                forecast_line = f"forecast={method_name}"
                values = None
                for key in ('forecast_price', 'forecast_return', 'forecast_series', 'forecast'):
                    candidate = fc.get(key)
                    if isinstance(candidate, list) and candidate:
                        values = candidate
                        break
                if isinstance(values, list) and len(values) >= 3:
                    nums: List[float] = []
                    for v in values:
                        try:
                            nums.append(float(v))
                        except Exception:
                            nums = []
                            break
                    if nums and len(nums) >= 3:
                        first = nums[0]
                        span = max(nums) - min(nums)
                        tol = max(1e-9, abs(first) * 1e-6)
                        if span <= tol:
                            forecast_line += " (flat)"
                            _append_diagnostic_warning(
                                rep,
                                "Selected forecast appears degenerate (near-constant values across horizon).",
                            )
                summ.append(forecast_line)
                timing_parts: List[str] = []
                last_obs = fc.get('last_observation_time')
                start_time = fc.get('forecast_start_time')
                anchor = fc.get('forecast_anchor')
                if last_obs:
                    timing_parts.append(f"last_obs={last_obs}")
                if start_time:
                    timing_parts.append(f"start={start_time}")
                if anchor:
                    timing_parts.append(f"anchor={anchor}")
                if timing_parts:
                    summ.append("forecast timing: " + " ".join(timing_parts))
        except Exception:
            pass
        try:
            backtest_sec = rep.get('sections', {}).get('backtest', {})
            criteria = backtest_sec.get('selection_criteria') if isinstance(backtest_sec, dict) else None
            best_payload = backtest_sec.get('best_method') if isinstance(backtest_sec, dict) else None
            if isinstance(criteria, dict):
                primary = str(criteria.get('primary_metric') or 'avg_rmse')
                tie_breaker = str(criteria.get('tie_breaker') or 'avg_directional_accuracy')
                tol_pct = criteria.get('rmse_tolerance_pct')
                if tol_pct is None:
                    tol_raw = criteria.get('rmse_tolerance')
                    try:
                        tol_pct = float(tol_raw) * 100.0 if tol_raw is not None else None
                    except Exception:
                        tol_pct = None
                line = f"forecast selection: min {primary}"
                if tol_pct is not None:
                    line += f", tie-window={format_number(tol_pct)}%"
                line += f", tie-break={tie_breaker}"
                min_da = criteria.get('min_directional_accuracy')
                if min_da is not None:
                    line += f", min-dir-acc>={format_number(min_da)}"
                if isinstance(best_payload, dict):
                    initial = best_payload.get('initial_method')
                    chosen = best_payload.get('method')
                    if initial and chosen and str(initial) != str(chosen):
                        line += ", fallback=degenerate-initial-forecast"
                summ.append(line)
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
                    tp = best.get('tp')
                    sl = best.get('sl')
                    ev = best.get('ev')
                    edge = best.get('edge')
                    details: List[str] = []
                    details.append(f"dir={dname}")
                    if tp is not None:
                        details.append(f"tp={format_number(tp)}%")
                    if sl is not None:
                        details.append(f"sl={format_number(sl)}%")
                    if ev is not None:
                        details.append(f"ev={format_number(ev)}")
                    if edge is not None:
                        details.append(f"edge={format_number(edge)}")
                    try:
                        if ev is not None and edge is not None:
                            ev_num = float(ev)
                            edge_num = float(edge)
                            if (ev_num > 0 and edge_num < 0) or (ev_num < 0 and edge_num > 0):
                                details.append("ev_edge_conflict=true")
                                details.append("ev_edge_conflict_reason=ev and edge have opposite signs")
                    except Exception:
                        pass
                    if details:
                        summ.append("barrier best " + ' '.join(details))
            else:
                best = bar.get('best') if isinstance(bar, dict) else None
                direction = bar.get('direction') if isinstance(bar, dict) else None
                if best:
                    tp = best.get('tp')
                    sl = best.get('sl')
                    ev = best.get('ev')
                    edge = best.get('edge')
                    details: List[str] = []
                    if direction:
                        details.append(f"dir={str(direction)}")
                    if tp is not None:
                        details.append(f"tp={format_number(tp)}%")
                    if sl is not None:
                        details.append(f"sl={format_number(sl)}%")
                    if ev is not None:
                        details.append(f"ev={format_number(ev)}")
                    if edge is not None:
                        details.append(f"edge={format_number(edge)}")
                    try:
                        if ev is not None and edge is not None:
                            ev_num = float(ev)
                            edge_num = float(edge)
                            if (ev_num > 0 and edge_num < 0) or (ev_num < 0 and edge_num > 0):
                                details.append("ev_edge_conflict=true")
                                details.append("ev_edge_conflict_reason=ev and edge have opposite signs")
                    except Exception:
                        pass
                    if details:
                        summ.append("barrier best " + ' '.join(details))
        except Exception:
            pass
        rep['summary'] = summ
        diagnostics = rep.get("diagnostics")
        if not isinstance(diagnostics, dict):
            diagnostics = {}
        diagnostics["execution_time_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
        rep["diagnostics"] = diagnostics

        if output_mode == 'markdown':
            return render_enhanced_report(rep)
        return rep
    except Exception as exc:
        msg = f"Error generating report: {exc}"
        if str(output or '').strip().lower() == 'markdown':
            return _report_error_text(msg)
        return _report_error_payload(msg)
