import math
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

from ...utils.barriers import get_pip_size as _get_pip_size
from ...utils.constants import TIME_DISPLAY_FORMAT
from ..tool_calling import call_tool_sync_raw
from .shared import (
    _as_float,
    _format_decimal,
    _format_probability,
    _format_series_preview,
    _format_signed,
    _format_state_shares,
    _format_table,
    _get_indicator_value,
    _indicator_key_variants,
    format_number,
)


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime(TIME_DISPLAY_FORMAT)


def parse_table_tail(data: Any, tail: int = 1) -> List[Dict[str, Any]]:
    """Return the last N rows from a tabular payload (list[dict] or {data|bars: ...})."""
    try:
        if isinstance(data, dict):
            rows_obj = data.get('data')
            if not isinstance(rows_obj, list):
                rows_obj = data.get('bars')
        else:
            rows_obj = data
        if not isinstance(rows_obj, list):
            return []
        tail_i = int(tail or 0)
        rows_in = [r for r in rows_obj if isinstance(r, dict)]
        if tail_i > 0:
            rows_in = rows_in[-tail_i:]

        def _coerce(v: Any) -> Any:
            if v is None or isinstance(v, (int, float, bool)):
                return v
            if not isinstance(v, str):
                return v
            s = v.strip()
            if s == "":
                return ""
            if s.lower() in ("nan", "inf", "+inf", "-inf"):
                try:
                    return float(s)
                except Exception:
                    return v
            if '.' in s or 'e' in s.lower():
                try:
                    return float(s)
                except Exception:
                    return v
            if s.lstrip('-').isdigit():
                try:
                    return int(s)
                except Exception:
                    return v
            return v

        out: List[Dict[str, Any]] = []
        for row in rows_in:
            out.append({str(k): _coerce(val) for k, val in row.items()})
        return out
    except Exception:
        return []


def extract_candle_freshness_diagnostics(data: Any) -> Optional[Dict[str, Any]]:
    try:
        if not isinstance(data, dict):
            return None
        for container_key in ('meta', 'details'):
            container = data.get(container_key)
            if not isinstance(container, dict):
                continue
            diagnostics = container.get('diagnostics')
            if not isinstance(diagnostics, dict):
                continue
            freshness = diagnostics.get('freshness')
            if isinstance(freshness, dict) and freshness:
                return dict(freshness)
        return None
    except Exception:
        return None


def attach_candle_freshness_diagnostics(payload: Dict[str, Any], data: Any) -> Dict[str, Any]:
    try:
        out = dict(payload) if isinstance(payload, dict) else {}
        freshness = out.get('freshness')
        if isinstance(freshness, dict) and freshness:
            return out
        extracted = extract_candle_freshness_diagnostics(data)
        if extracted:
            out['freshness'] = extracted
        return out
    except Exception:
        return dict(payload) if isinstance(payload, dict) else {}


def pick_best_forecast_method(
    bt: Dict[str, Any],
    rmse_tolerance: float = 0.05,
    min_directional_accuracy: Optional[float] = None,
) -> Optional[Tuple[str, Dict[str, Any]]]:
    try:
        results = bt.get('results') if isinstance(bt, dict) else None
        if not isinstance(results, dict) or not results:
            return None
        min_da: Optional[float] = None
        if min_directional_accuracy is not None:
            try:
                candidate = float(min_directional_accuracy)
            except Exception:
                candidate = float("nan")
            if math.isfinite(candidate):
                min_da = max(0.0, min(1.0, candidate))
        entries: List[Tuple[str, Dict[str, Any], float, Optional[float], int]] = []
        for m, res in results.items():
            if not isinstance(res, dict):
                continue
            if not res.get('success'):
                continue
            try:
                rmse = float(res.get('avg_rmse', float('inf')))
            except Exception:
                rmse = float('inf')
            try:
                da_val = res.get('avg_directional_accuracy')
                da = float(da_val) if da_val is not None else float('nan')
            except Exception:
                da = float('nan')
            ok = int(res.get('successful_tests', 0))
            if not (rmse == rmse):
                continue
            entries.append((m, res, rmse, da if da == da else None, ok))
        if min_da is not None:
            entries = [e for e in entries if e[3] is not None and float(e[3]) >= float(min_da)]
        if not entries:
            return None
        entries.sort(key=lambda x: x[2])
        best_rmse = entries[0][2]
        tol = max(0.0, float(rmse_tolerance))
        limit = best_rmse * (1.0 + tol)
        candidates = [e for e in entries if e[2] <= limit]
        with_dir = [e for e in candidates if e[3] is not None]
        if with_dir:
            with_dir.sort(key=lambda x: (-(x[3] or 0.0), x[2], -x[4]))
            chosen = with_dir[0]
        else:
            entries.sort(key=lambda x: (x[2], -x[4]))
            chosen = entries[0]
        return (chosen[0], chosen[1])
    except Exception:
        return None


def summarize_barrier_grid(grid: Dict[str, Any], top_k: int = 3) -> Dict[str, Any]:
    try:
        best = grid.get('best') if isinstance(grid, dict) else None
        top = grid.get('top') if isinstance(grid, dict) else None
        if not top and isinstance(grid.get('results'), list):
            items = grid['results']
            try:
                items = sorted(items, key=lambda x: float(x.get('score', x.get('edge', -1e9))), reverse=True)
            except Exception:
                pass
            top = items[:top_k]
        out: Dict[str, Any] = {}
        direction = grid.get('direction') if isinstance(grid, dict) else None
        if isinstance(best, dict):
            best_out = {
                'tp': best.get('tp'),
                'sl': best.get('sl'),
                'objective': best.get('objective') or grid.get('objective'),
                'edge': best.get('edge'),
                'kelly': best.get('kelly'),
                'ev': best.get('ev'),
                'prob_tp_first': best.get('prob_tp_first'),
                'prob_sl_first': best.get('prob_sl_first'),
                'prob_no_hit': best.get('prob_no_hit'),
                'median_time_to_tp': best.get('median_time_to_tp'),
                'tp_price': best.get('tp_price'),
                'sl_price': best.get('sl_price'),
            }
            try:
                ev_val = best_out.get('ev')
                edge_val = best_out.get('edge')
                if ev_val is not None and edge_val is not None:
                    ev_f = float(ev_val)
                    edge_f = float(edge_val)
                    if (ev_f > 0.0 and edge_f < 0.0) or (ev_f < 0.0 and edge_f > 0.0):
                        best_out['ev_edge_conflict'] = True
                        best_out['ev_edge_conflict_reason'] = "ev and edge have opposite signs"
            except Exception:
                pass
            out['best'] = best_out
            if direction:
                out['direction'] = direction
            if bool(best_out.get('ev_edge_conflict')):
                out['ev_edge_conflict'] = True
                out['ev_edge_conflict_reason'] = "ev and edge have opposite signs"
                out['caution'] = (
                    "EV and edge signs conflict for the selected candidate; inspect win probability "
                    "and break-even threshold before trading."
                )
        if isinstance(top, list):
            def _round_metric(value: Any, decimals: int) -> Any:
                try:
                    if value is None:
                        return None
                    num = float(value)
                    if not math.isfinite(num):
                        return str(value)
                    return round(num, decimals)
                except Exception:
                    return value

            def _row_key(row: Dict[str, Any]) -> Tuple[Any, ...]:
                return (
                    _round_metric(row.get('tp'), 4),
                    _round_metric(row.get('sl'), 4),
                    _round_metric(row.get('tp_price'), 6),
                    _round_metric(row.get('sl_price'), 6),
                    _round_metric(row.get('edge'), 4),
                    _round_metric(row.get('kelly'), 4),
                    _round_metric(row.get('ev'), 4),
                    _round_metric(row.get('prob_tp_first'), 4),
                    _round_metric(row.get('prob_sl_first'), 4),
                    _round_metric(row.get('prob_no_hit'), 4),
                )

            trimmed = []
            seen_keys: set = set()
            for it in top:
                if not isinstance(it, dict):
                    continue
                key = _row_key(it)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                trimmed.append({
                    'tp': it.get('tp'), 'sl': it.get('sl'),
                    'tp_price': it.get('tp_price'), 'sl_price': it.get('sl_price'),
                    'edge': it.get('edge'), 'kelly': it.get('kelly'), 'ev': it.get('ev'),
                    'prob_tp_first': it.get('prob_tp_first'), 'prob_sl_first': it.get('prob_sl_first'), 'prob_no_hit': it.get('prob_no_hit'),
                })
                if len(trimmed) >= int(top_k):
                    break
            if trimmed:
                out['top'] = trimmed
        for key in ("caution", "ev_edge_conflict", "ev_edge_conflict_reason", "selection_warnings"):
            value = grid.get(key)
            if value in (None, [], {}):
                continue
            out[key] = value
        return out or {"note": "no grid summary"}
    except Exception:
        return {"note": "no grid summary"}


def merge_params(base: Optional[Dict[str, Any]], extra: Dict[str, Any], override: bool = False) -> Dict[str, Any]:
    p = dict(base or {})
    for k, v in extra.items():
        if override or k not in p:
            p[k] = v
    return p


def market_snapshot(symbol: str, timezone: str = 'UTC') -> Dict[str, Any]:
    try:
        from ..market_depth import market_depth_fetch as _fetch_market_depth
        try:
            dom = _fetch_market_depth(symbol=symbol, __cli_raw=True)
        except TypeError:
            dom = _fetch_market_depth(symbol=symbol)
        bid = None; ask = None; spread = None
        top_buy_vol = None; top_sell_vol = None; total_buy_vol = None; total_sell_vol = None
        if isinstance(dom, dict) and dom.get('success'):
            t = dom.get('type')
            data = dom.get('data') or {}
            if t == 'tick_data':
                bid = data.get('bid'); ask = data.get('ask')
                if bid is not None and ask is not None:
                    try:
                        spread = float(ask) - float(bid)
                    except Exception:
                        spread = None
            elif t == 'full_depth':
                buys = data.get('buy_orders') or []
                sells = data.get('sell_orders') or []
                if buys:
                    try:
                        top_buy_vol = float(buys[0].get('volume') or 0.0)
                    except Exception:
                        top_buy_vol = None
                if sells:
                    try:
                        top_sell_vol = float(sells[0].get('volume') or 0.0)
                    except Exception:
                        top_sell_vol = None
                try:
                    total_buy_vol = float(sum(float(b.get('volume') or 0.0) for b in buys)) if buys else None
                except Exception:
                    total_buy_vol = None
                try:
                    total_sell_vol = float(sum(float(s.get('volume') or 0.0) for s in sells)) if sells else None
                except Exception:
                    total_sell_vol = None
        tick_size = _get_pip_size(symbol)
        pip_size = None
        if tick_size is not None:
            try:
                tick_size = float(tick_size)
            except Exception:
                tick_size = None
        if tick_size is not None and tick_size > 0:
            pip_size = tick_size
            if math.isclose(tick_size, 0.00001, rel_tol=0.0, abs_tol=1e-12) or math.isclose(
                tick_size,
                0.001,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                pip_size = tick_size * 10.0
        spread_ticks = None
        if tick_size and spread is not None:
            try:
                spread_ticks = float(spread) / float(tick_size) if tick_size > 0 else None
            except Exception:
                spread_ticks = None
        spread_pips = None
        if pip_size and spread is not None:
            try:
                spread_pips = float(spread) / float(pip_size) if pip_size > 0 else None
            except Exception:
                spread_pips = None
        return {
            'bid': bid,
            'ask': ask,
            'spread': spread,
            'tick_size': tick_size,
            'spread_ticks': spread_ticks,
            'spread_pips': spread_pips,
            'dom_top_buy_vol': top_buy_vol,
            'dom_top_sell_vol': top_sell_vol,
            'dom_total_buy_vol': total_buy_vol,
            'dom_total_sell_vol': total_sell_vol,
        }
    except Exception as e:
        return {'error': str(e)}


def apply_market_gates(section: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    gate = {}
    try:
        max_ticks = params.get('spread_max_ticks')
        max_pips = params.get('spread_max_pips')
        if max_ticks is not None and isinstance(section, dict):
            sp = section.get('spread_ticks')
            if sp is not None:
                gate['spread_ok'] = bool(float(sp) <= float(max_ticks))
                gate['spread_ticks'] = float(sp)
                gate['spread_max_ticks'] = float(max_ticks)
        elif max_pips is not None and isinstance(section, dict):
            sp = section.get('spread_pips')
            if sp is not None:
                gate['spread_ok'] = bool(float(sp) <= float(max_pips))
                gate['spread_pips'] = float(sp)
                gate['spread_max_pips'] = float(max_pips)
    except Exception:
        pass
    return gate


def context_for_tf(symbol: str, timeframe: str, denoise: Optional[Dict[str, Any]], limit: int = 200, tail: int = 30, *, _fetch_cache: Optional[Dict[Tuple[str, str], Optional[Dict[str, Any]]]] = None) -> Optional[Dict[str, Any]]:
    cache_key = (symbol.upper(), timeframe.upper())
    if _fetch_cache is not None and cache_key in _fetch_cache:
        return _fetch_cache[cache_key]
    try:
        from ..data import data_fetch_candles as _fetch_candles
        indicators = "ema(20),ema(50),ema(200),rsi(14),macd(12,26,9)"
        res = call_tool_sync_raw(
            _fetch_candles,
            symbol=symbol,
            timeframe=timeframe,
            limit=int(limit),
            indicators=indicators,
            denoise=denoise,
            cli_raw=True,
        )

        if not isinstance(res, dict) or res.get('error'):
            if _fetch_cache is not None:
                _fetch_cache[cache_key] = None
            return None
        freshness = extract_candle_freshness_diagnostics(res)
        rows = parse_table_tail(res, tail=int(tail))

        if not rows:
            if _fetch_cache is not None:
                _fetch_cache[cache_key] = None
            return None
        last = rows[-1]
        out = {
            'close': last.get('close'),
            'EMA_20': _get_indicator_value(last, 'EMA_20'),
            'EMA_50': _get_indicator_value(last, 'EMA_50'),
            'RSI_14': _get_indicator_value(last, 'RSI_14'),
            'MACD': _get_indicator_value(last, 'MACD_12_26_9'),
        }

        # Compute trend compact data for MTF matrix
        try:
            from ..report_templates.basic import _compute_compact_trend
            compact = _compute_compact_trend(rows)
            if compact:
                out['trend_compact'] = compact
        except Exception:
            # If trend compact calculation fails, continue without it
            pass

        # Add individual indicator values for MTF matrix
        if rows:
            last_row = rows[-1]
            out['rsi'] = _get_indicator_value(last_row, 'RSI_14')
            out['macd'] = _get_indicator_value(last_row, 'MACD_12_26_9')
            out['macd_signal'] = _get_indicator_value(last_row, 'MACDs_12_26_9')
            out['ema20'] = _get_indicator_value(last_row, 'EMA_20')
            out['ema50'] = _get_indicator_value(last_row, 'EMA_50')
            out['ema200'] = _get_indicator_value(last_row, 'EMA_200')
            out['price'] = last_row.get('close')
        if freshness:
            out['freshness'] = freshness

        if _fetch_cache is not None:
            _fetch_cache[cache_key] = out
        return out
    except Exception:
        if _fetch_cache is not None:
            _fetch_cache[cache_key] = None
        return None


def _extract_base_timeframe(report: Dict[str, Any]) -> Optional[str]:
    """Try to infer the base timeframe from report metadata or context."""
    base_tf = None
    try:
        meta = report.get('meta') if isinstance(report, dict) else None
        if isinstance(meta, dict) and meta.get('timeframe'):
            base_tf = str(meta.get('timeframe')).upper()
    except Exception:
        base_tf = None
    if base_tf is None:
        try:
            sections = report.get('sections') if isinstance(report, dict) else None
            context = sections.get('context') if isinstance(sections, dict) else None
            if isinstance(context, dict) and context.get('timeframe'):
                base_tf = str(context.get('timeframe')).upper()
        except Exception:
            base_tf = None
    return base_tf


def _collect_timeframe_section_entries(section: Any) -> Dict[str, Any]:
    entries: Dict[str, Any] = {}
    if not isinstance(section, dict):
        return entries
    for key, value in section.items():
        tf_key = str(key).upper()
        if not tf_key or tf_key.startswith("__"):
            continue
        entries[tf_key] = value
    return entries


def attach_multi_timeframes(report: Dict[str, Any], symbol: str, denoise: Optional[Dict[str, Any]], extra_timeframes: List[str], pivot_timeframes: Optional[List[str]] = None, *, _fetch_cache: Optional[Dict[Tuple[str, str], Optional[Dict[str, Any]]]] = None) -> None:
    sections = report.setdefault('sections', {})
    contexts: Dict[str, Any] = {}
    trend_mtf: Dict[str, Any] = {}
    base_tf = _extract_base_timeframe(report)
    existing_contexts = _collect_timeframe_section_entries(sections.get('contexts_multi'))
    context_section = sections.get('context')
    existing_trend_mtf = (
        _collect_timeframe_section_entries(context_section.get('trend_mtf'))
        if isinstance(context_section, dict)
        else {}
    )

    for tf in extra_timeframes or []:
        tf_str = str(tf).upper()
        if base_tf and tf_str == base_tf:
            continue
        existing_context = existing_contexts.get(tf_str)
        existing_trend = existing_trend_mtf.get(tf_str)
        if existing_context is not None:
            snap = existing_context
            if isinstance(existing_trend, dict):
                trend_mtf[str(tf)] = existing_trend.copy()
        else:
            snap = context_for_tf(symbol, tf, denoise, limit=200, tail=30, _fetch_cache=_fetch_cache)
        if snap:
            snap_for_contexts = snap
            if isinstance(snap, dict):
                # Keep contexts_multi focused on indicator/price snapshots; trend_compact
                # is represented in context.trend_mtf to avoid duplicating the same blob.
                snap_for_contexts = dict(snap)
                snap_for_contexts.pop('trend_compact', None)
                snap_for_contexts.pop('trend_compact_legend', None)
                snap_for_contexts.pop('trend_compact_explained', None)
            if not isinstance(snap_for_contexts, dict) or any(v is not None for v in snap_for_contexts.values()):
                contexts[str(tf)] = snap_for_contexts

            # Extract trend compact data for MTF matrix
            if existing_context is None and isinstance(snap, dict):
                trend_compact = snap.get('trend_compact')
                if isinstance(trend_compact, dict):
                    trend_mtf[str(tf)] = trend_compact.copy()

    if contexts:
        sections['contexts_multi'] = contexts

    # Attach compact trend info for the main context section
    if trend_mtf or contexts:
        if 'context' in sections and isinstance(sections['context'], dict):
            sections['context']['trend_mtf'] = trend_mtf
        else:
            sections['context'] = {'trend_mtf': trend_mtf}

    base_pivot_tf = None
    try:
        base_pivot = report.setdefault('sections', {}).get('pivot')
        if isinstance(base_pivot, dict) and base_pivot.get('timeframe'):
            base_pivot_tf = str(base_pivot.get('timeframe')).upper()
    except Exception:
        base_pivot_tf = None

    if pivot_timeframes:
        filtered_tfs: List[str] = []
        for tfp in pivot_timeframes:
            tfp_str = str(tfp).upper()
            if base_pivot_tf and tfp_str == base_pivot_tf:
                continue
            filtered_tfs.append(str(tfp))
        pivs: Dict[str, Any] = {}
        existing_pivots = _collect_timeframe_section_entries(sections.get('pivot_multi'))
        if filtered_tfs:
            try:
                from ..pivot import pivot_compute_points as _compute_pivot_points
                for tfp in filtered_tfs:
                    tfp_key = str(tfp).upper()
                    existing_pivot = existing_pivots.get(tfp_key)
                    if isinstance(existing_pivot, dict):
                        pivs[str(tfp)] = dict(existing_pivot)
                        continue
                    res = _compute_pivot_points(symbol=symbol, timeframe=tfp)
                    if isinstance(res, dict) and not res.get('error'):
                        pivs[str(tfp)] = {
                            'levels': res.get('levels'),
                            'methods': res.get('methods'),
                            'period': res.get('period'),
                            'timeframe': tfp,
                            'calculation_basis': res.get('calculation_basis'),
                            'timezone': res.get('timezone'),
                        }
            except Exception:
                pivs = {}
        if pivs:
            if base_pivot_tf:
                pivs['__base_timeframe__'] = base_pivot_tf
            sections['pivot_multi'] = pivs


def attach_report_timeframes(
    report: Dict[str, Any],
    symbol: str,
    denoise: Optional[Dict[str, Any]],
    params: Optional[Dict[str, Any]],
    *,
    default_extra: List[str],
    default_pivots: Optional[List[str]] = None,
    _fetch_cache: Optional[Dict[Tuple[str, str], Optional[Dict[str, Any]]]] = None,
) -> None:
    extra = (params or {}).get('extra_timeframes') or default_extra
    pivots = (params or {}).get('pivot_timeframes') or default_pivots
    attach_multi_timeframes(
        report,
        symbol,
        denoise,
        extra_timeframes=extra,
        pivot_timeframes=pivots,
        _fetch_cache=_fetch_cache,
    )


def attach_market_and_timeframes(
    report: Dict[str, Any],
    symbol: str,
    denoise: Optional[Dict[str, Any]],
    params: Optional[Dict[str, Any]],
    *,
    default_extra: List[str],
    default_pivots: Optional[List[str]] = None,
    snapshot: Optional[Dict[str, Any]] = None,
    _fetch_cache: Optional[Dict[Tuple[str, str], Optional[Dict[str, Any]]]] = None,
) -> Dict[str, Any]:
    snap = snapshot if snapshot is not None else market_snapshot(symbol)
    report.setdefault('sections', {})['market'] = snap
    gates = apply_market_gates(snap if isinstance(snap, dict) else {}, params or {})
    if gates:
        report['sections']['execution_gates'] = gates
    attach_report_timeframes(
        report,
        symbol,
        denoise,
        params,
        default_extra=default_extra,
        default_pivots=default_pivots,
        _fetch_cache=_fetch_cache,
    )
    return snap
def _needs_yaml_quotes(text: str) -> bool:
    if text == '':
        return True
    if text != text.strip():
        return True
    if text[0] in {'-', '?', ':', '#', '!', '*', '&', '%', '@', '`', '{', '}', '[', ']', ',', '|', '>'}:
        return True
    for ch in (':', ',', '#'):
        if ch in text:
            return True
    return any(c in text for c in ('\n', '\r', '\t'))


def _escape_yaml_string(text: str) -> str:
    escaped = text.replace('\\', '\\\\').replace('"', '\\"').replace('\r', ' ').replace('\n', ' ')
    return escaped


def _compact_scalar(value: Any) -> str:
    if value is None:
        return 'null'
    if isinstance(value, bool):
        return format_number(value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return format_number(value)
    text = str(value)
    if _needs_yaml_quotes(text):
        return f'"{_escape_yaml_string(text)}"'
    return text


def _compact_yaml(value: Any, indent: int = 0) -> str:
    prefix = '  ' * indent
    if isinstance(value, dict):
        if not value:
            return f"{prefix}{{}}"
        lines: List[str] = []
        for key, val in value.items():
            key_str = str(key)
            if isinstance(val, (dict, list)):
                lines.append(f"{prefix}{key_str}:")
                lines.append(_compact_yaml(val, indent + 1))
            else:
                lines.append(f"{prefix}{key_str}: {_compact_scalar(val)}")
        return "\n".join(lines)
    if isinstance(value, list):
        if not value:
            return f"{prefix}[]"
        lines: List[str] = []
        for item in value:
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}-")
                lines.append(_compact_yaml(item, indent + 1))
            else:
                lines.append(f"{prefix}- {_compact_scalar(item)}")
        return "\n".join(lines)
    return f"{prefix}{_compact_scalar(value)}"


def _compact_table_value(value: Any) -> str:
    if value is None:
        text = 'null'
    elif isinstance(value, bool):
        text = format_number(value)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        text = format_number(value)
    else:
        text = str(value)
    text = text.replace('\r', ' ').replace('\n', ' ')
    if any(ch in text for ch in (',', '"')):
        text = '"' + text.replace('"', '""') + '"'
    return text



from .rendering import (
    _build_pivot_context_line,
    _render_backtest_section,
    _render_barriers_section,
    _render_context_section,
    _render_contexts_multi_section,
    _render_execution_gates_section,
    _render_forecast_conformal_section,
    _render_forecast_section,
    _render_generic_section,
    _render_market_section,
    _render_patterns_section,
    _render_pivot_multi_section,
    _render_pivot_section,
    _render_regime_section,
    _render_volatility_har_section,
    _render_volatility_section,
    render_enhanced_report,
)
