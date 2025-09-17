from typing import Any, Dict, Optional, List, Tuple
from datetime import datetime
import math


def now_utc_iso() -> str:
    try:
        return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    except Exception:
        return str(datetime.utcnow())


def parse_csv_tail(csv_header: Optional[str], csv_data: Optional[str], tail: int = 1) -> List[Dict[str, Any]]:
    try:
        if not csv_header or not csv_data:
            return []
        cols = [c.strip() for c in csv_header.split(',')]
        lines = [ln for ln in csv_data.split('\n') if ln.strip()]
        if tail > 0:
            lines = lines[-int(tail):]
        out: List[Dict[str, Any]] = []
        for ln in lines:
            vals = [v.strip() for v in ln.split(',')]
            row: Dict[str, Any] = {}
            for i, c in enumerate(cols):
                v = vals[i] if i < len(vals) else ''
                try:
                    if v == '' or v is None:
                        row[c] = v
                    elif '.' in v or 'e' in v.lower():
                        row[c] = float(v)
                    elif v.lstrip('-').isdigit():
                        row[c] = int(v)
                    else:
                        row[c] = v
                except Exception:
                    row[c] = v
            out.append(row)
        return out
    except Exception:
        return []


def pick_best_forecast_method(bt: Dict[str, Any], rmse_tolerance: float = 0.05) -> Optional[Tuple[str, Dict[str, Any]]]:
    try:
        results = bt.get('results') if isinstance(bt, dict) else None
        if not isinstance(results, dict) or not results:
            return None
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
        if isinstance(best, dict):
            out['best'] = {
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
            }
        if isinstance(top, list):
            trimmed = []
            for it in top[:top_k]:
                if not isinstance(it, dict):
                    continue
                trimmed.append({
                    'tp': it.get('tp'), 'sl': it.get('sl'),
                    'edge': it.get('edge'), 'kelly': it.get('kelly'), 'ev': it.get('ev'),
                    'prob_tp_first': it.get('prob_tp_first'), 'prob_sl_first': it.get('prob_sl_first'), 'prob_no_hit': it.get('prob_no_hit'),
                })
            if trimmed:
                out['top'] = trimmed
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
        from .market_depth import market_depth_fetch as _fetch_market_depth
        import MetaTrader5 as _mt5
        dom = _fetch_market_depth(symbol=symbol, timezone=timezone)
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
        pip = None
        try:
            info = _mt5.symbol_info(symbol)
            if info is not None:
                digits = int(getattr(info, 'digits', 0) or 0)
                point = float(getattr(info, 'point', 0.0) or 0.0)
                pip = float(point * (10.0 if digits in (3, 5) else 1.0)) if point > 0 else None
        except Exception:
            pip = None
        spread_pips = None
        if pip and spread is not None:
            try:
                spread_pips = float(spread) / float(pip) if pip > 0 else None
            except Exception:
                spread_pips = None
        return {
            'bid': bid,
            'ask': ask,
            'spread': spread,
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
        max_pips = params.get('spread_max_pips')
        if max_pips is not None and isinstance(section, dict):
            sp = section.get('spread_pips')
            if sp is not None:
                gate['spread_ok'] = bool(float(sp) <= float(max_pips))
                gate['spread_pips'] = float(sp)
                gate['spread_max_pips'] = float(max_pips)
    except Exception:
        pass
    return gate


def context_for_tf(symbol: str, timeframe: str, denoise: Optional[Dict[str, Any]], limit: int = 200, tail: int = 30) -> Optional[Dict[str, Any]]:
    try:
        from .data import data_fetch_candles as _fetch_candles
        indicators = "ema(20),ema(50),rsi(14),macd(12,26,9)"
        res = _fetch_candles(symbol=symbol, timeframe=timeframe, limit=int(limit), indicators=indicators, denoise=denoise, timezone='UTC')
        if 'error' in res:
            return None
        rows = parse_csv_tail(res.get('csv_header'), res.get('csv_data'), tail=int(tail))
        if not rows:
            return None
        last = rows[-1]
        out = {
            'close': last.get('close'),
            'EMA_20': last.get('EMA_20') or last.get('ema_20'),
            'EMA_50': last.get('EMA_50') or last.get('ema_50'),
            'RSI_14': last.get('RSI_14') or last.get('rsi_14'),
            'MACD': last.get('MACD') or last.get('macd'),
        }
        return out
    except Exception:
        return None


def attach_multi_timeframes(report: Dict[str, Any], symbol: str, denoise: Optional[Dict[str, Any]], extra_timeframes: List[str], pivot_timeframes: Optional[List[str]] = None) -> None:
    contexts: Dict[str, Any] = {}
    for tf in extra_timeframes or []:
        snap = context_for_tf(symbol, tf, denoise, limit=200, tail=30)
        if snap:
            contexts[str(tf)] = snap
    if contexts:
        report.setdefault('sections', {})['contexts_multi'] = contexts
    if pivot_timeframes:
        pivs: Dict[str, Any] = {}
        try:
            from .pivot import pivot_compute_points as _compute_pivot_points
            for tfp in pivot_timeframes:
                res = _compute_pivot_points(symbol=symbol, timeframe=tfp)
                if isinstance(res, dict) and not res.get('error'):
                    pivs[str(tfp)] = {'levels': res.get('levels'), 'period': res.get('period')}
        except Exception:
            pivs = {}
        if pivs:
            report.setdefault('sections', {})['pivot_multi'] = pivs


def format_number(value: Any) -> str:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(num):
        return str(value)
    abs_num = abs(num)
    if abs_num >= 1000:
        return f"{num:,.2f}"
    if abs_num >= 100:
        return f"{num:,.2f}"
    if abs_num >= 1:
        return f"{num:.4f}"
    return f"{num:.6f}"

def render_markdown(report: Dict[str, Any]) -> str:
    meta = report.get('meta', {}) if isinstance(report, dict) else {}
    symbol = meta.get('symbol', '')
    timeframe = meta.get('timeframe', '')
    horizon = meta.get('horizon', '')
    generated = meta.get('generated_at', '')
    title = f"# Trade Report: {symbol} ({timeframe}) – horizon {horizon}"
    lines: List[str] = [title.strip()]
    if generated:
        lines.append(f"_Generated at: {generated}_")
    lines.append("")

    summary = report.get('summary') if isinstance(report, dict) else None
    if isinstance(summary, list) and summary:
        lines.append("## Snapshot")
        for item in summary:
            lines.append(f"- {item}")
        lines.append("")

    sections = report.get('sections', {}) if isinstance(report, dict) else {}

    # Context
    ctx = sections.get('context') if isinstance(sections, dict) else None
    if isinstance(ctx, dict):
        last = ctx.get('last_snapshot') if isinstance(ctx.get('last_snapshot'), dict) else {}
        notes = ctx.get('notes')
        lines.append("## Market Context")
        if last:
            close = last.get('close')
            ema20 = last.get('EMA_20') or last.get('ema_20')
            ema50 = last.get('EMA_50') or last.get('ema_50')
            rsi = last.get('RSI_14') or last.get('rsi_14')
            macd = last.get('MACD') or last.get('macd')
            if close is not None:
                lines.append(f"- Close: `{format_number(close)}`")
            if ema20 is not None or ema50 is not None:
                lines.append(f"- EMAs: 20=`{format_number(ema20)}`  |  50=`{format_number(ema50)}`")
            if rsi is not None:
                lines.append(f"- RSI(14): `{format_number(rsi)}`")
            if macd is not None:
                lines.append(f"- MACD: `{format_number(macd)}`")
        if notes:
            lines.append(f"- {notes}")
        lines.append("")

    # Pivot
    pivot = sections.get('pivot') if isinstance(sections, dict) else None
    if isinstance(pivot, dict) and not pivot.get('error'):
        levels = pivot.get('levels', {})
        lines.append("## Pivot Levels")
        if isinstance(levels, dict) and levels:
            for name, value in levels.items():
                lines.append(f"- {name}: `{format_number(value)}`")
        lines.append("")

    # Higher‑TF pivots
    piv_multi = sections.get('pivot_multi') if isinstance(sections, dict) else None
    if isinstance(piv_multi, dict) and piv_multi:
        lines.append("## Higher‑TF Pivots")
        for tf, obj in piv_multi.items():
            if not isinstance(obj, dict):
                continue
            lev = obj.get('levels') or {}
            if isinstance(lev, dict) and lev:
                lines.append(f"- {tf}:")
                for name, value in lev.items():
                    lines.append(f"  - {name}: `{format_number(value)}`")
        lines.append("")

    # Volatility
    vol = sections.get('volatility') if isinstance(sections, dict) else None
    if isinstance(vol, dict) and not vol.get('error'):
        lines.append("## Volatility (EWMA)")
        sigma_price = vol.get('horizon_sigma_price')
        sigma_ret = vol.get('horizon_sigma_return')
        if sigma_price is not None:
            lines.append(f"- Horizon σ (price): `{format_number(sigma_price)}`")
        if sigma_ret is not None:
            lines.append(f"- Horizon σ (return): `{format_number(sigma_ret)}`")
        lines.append("")

    # Advanced volatility
    vol_har = sections.get('volatility_har_rv') if isinstance(sections, dict) else None
    if isinstance(vol_har, dict) and not vol_har.get('error'):
        lines.append("## Volatility (HAR-RV)")
        sigma_ret = vol_har.get('horizon_sigma_return')
        if sigma_ret is not None:
            lines.append(f"- Horizon σ (return): `{format_number(sigma_ret)}`")
        lines.append("")

    # Backtest summary
    backtest = sections.get('backtest') if isinstance(sections, dict) else None
    if isinstance(backtest, dict):
        lines.append("## Backtest Overview")
        if backtest.get('error'):
            lines.append(f"- Error: {backtest.get('error')}")
        else:
            best = backtest.get('best_method', {})
            if isinstance(best, dict) and best:
                stats = best.get('stats', {}) if isinstance(best.get('stats'), dict) else {}
                da = stats.get('avg_directional_accuracy')
                lines.append(f"- Best method: **{best.get('method')}**")
                if stats:
                    lines.append(f"  - RMSE: `{format_number(stats.get('avg_rmse'))}`; MAE: `{format_number(stats.get('avg_mae'))}`; DirAcc: `{format_number(da)}`")
            ranking = backtest.get('ranking') if isinstance(backtest.get('ranking'), list) else []
            if ranking:
                lines.append("- Top methods:")
                for entry in ranking:
                    if not isinstance(entry, dict):
                        continue
                    lines.append(
                        f"  - {entry.get('method')}: RMSE=`{format_number(entry.get('avg_rmse'))}` | DirAcc=`{format_number(entry.get('avg_directional_accuracy'))}`"
                    )
        lines.append("")

    # Forecast
    forecast = sections.get('forecast') if isinstance(sections, dict) else None
    if isinstance(forecast, dict):
        lines.append("## Forecast")
        if forecast.get('error'):
            lines.append(f"- Error: {forecast.get('error')}")
        else:
            lines.append(f"- Method: **{forecast.get('method')}**")
            fp = forecast.get('forecast_price') or []
            if isinstance(fp, list) and fp:
                lines.append(f"- First horizon value: `{format_number(fp[0])}`")
                lines.append(f"- Last horizon value: `{format_number(fp[-1])}`")
            lower = forecast.get('lower_price')
            upper = forecast.get('upper_price')
            if isinstance(lower, list) and isinstance(upper, list) and lower and upper:
                lines.append(f"- Confidence band (t0): `{format_number(lower[0])}` – `{format_number(upper[0])}`")
        lines.append("")

    # Conformal (advanced)
    conf = sections.get('forecast_conformal') if isinstance(sections, dict) else None
    if isinstance(conf, dict) and not conf.get('error'):
        lines.append("## Conformal Intervals")
        lines.append(f"- Method: **{conf.get('method')}** (α={conf.get('alpha')})")
        low = conf.get('lower_price')
        up = conf.get('upper_price')
        if isinstance(low, list) and isinstance(up, list) and low and up:
            lines.append(f"- Band (t0): `{format_number(low[0])}` – `{format_number(up[0])}`")
        lines.append("")

    # Barrier optimization
    barriers = sections.get('barriers') if isinstance(sections, dict) else None
    if isinstance(barriers, dict) and not barriers.get('error'):
        lines.append("## Barrier Optimization")
        best = barriers.get('best') if isinstance(barriers.get('best'), dict) else None
        if best:
            lines.append(
                f"- Best edge: TP={format_number(best.get('tp'))}% | SL={format_number(best.get('sl'))}% | Edge=`{format_number(best.get('edge'))}` | Kelly=`{format_number(best.get('kelly'))}`"
            )
        top = barriers.get('top') if isinstance(barriers.get('top'), list) else []
        if top:
            lines.append("- Top combos:")
            for entry in top:
                if not isinstance(entry, dict):
                    continue
                lines.append(
                    f"  - TP={format_number(entry.get('tp'))}% | SL={format_number(entry.get('sl'))}% | Edge=`{format_number(entry.get('edge'))}`"
                )
        lines.append("")

    # Patterns
    patterns = sections.get('patterns') if isinstance(sections, dict) else None
    if isinstance(patterns, dict) and not patterns.get('error'):
        recent = patterns.get('recent') if isinstance(patterns.get('recent'), list) else []
        if recent:
            lines.append("## Recent Candlestick Patterns")
            for row in recent:
                if not isinstance(row, dict):
                    continue
                lines.append(f"- {row.get('time', 'time')} → {row.get('pattern', 'pattern')}")
            lines.append("")

    # Execution & Market
    market = sections.get('market') if isinstance(sections, dict) else None
    gates = sections.get('execution_gates') if isinstance(sections, dict) else None
    if isinstance(market, dict) or isinstance(gates, dict):
        lines.append("## Execution & Market")
        if isinstance(market, dict):
            bid = market.get('bid'); ask = market.get('ask')
            sp = market.get('spread'); spp = market.get('spread_pips')
            if bid is not None and ask is not None:
                lines.append(f"- Bid/Ask: `{format_number(bid)}` / `{format_number(ask)}`")
            if sp is not None:
                lines.append(f"- Spread: `{format_number(sp)}` ({format_number(spp)} pips)")
            tb = market.get('dom_top_buy_vol'); ts = market.get('dom_top_sell_vol')
            if tb is not None or ts is not None:
                lines.append(f"- DOM top vol: buy=`{format_number(tb)}` | sell=`{format_number(ts)}`")
        if isinstance(gates, dict) and gates:
            ok = gates.get('spread_ok')
            if ok is not None:
                status = "OK" if ok else "BLOCKED"
                lines.append(f"- Spread gate: {status} (spread={format_number(gates.get('spread_pips'))} pips; max={format_number(gates.get('spread_max_pips'))} pips)")
        lines.append("")

    # Regime (advanced)
    regime = sections.get('regime') if isinstance(sections, dict) else None
    if isinstance(regime, dict):
        lines.append("## Regime Signals")
        for key, block in regime.items():
            if not isinstance(block, dict):
                continue
            if block.get('error'):
                lines.append(f"- {key}: {block.get('error')}")
            else:
                summary = block.get('summary')
                if isinstance(summary, dict):
                    for sk, sv in summary.items():
                        lines.append(f"- {key} {sk}: {sv}")
        lines.append("")

    # Multi‑TF Context
    ctx_multi = sections.get('contexts_multi') if isinstance(sections, dict) else None
    if isinstance(ctx_multi, dict) and ctx_multi:
        lines.append("## Multi‑Timeframe Context")
        for tf, snap in ctx_multi.items():
            if not isinstance(snap, dict):
                continue
            close = snap.get('close'); e20 = snap.get('EMA_20'); e50 = snap.get('EMA_50'); rsi = snap.get('RSI_14')
            lines.append(
                f"- {tf}: Close=`{format_number(close)}` | EMA20=`{format_number(e20)}` | EMA50=`{format_number(e50)}` | RSI=`{format_number(rsi)}`"
            )
        lines.append("")

    return "\n".join(lines).strip() + "\n"
