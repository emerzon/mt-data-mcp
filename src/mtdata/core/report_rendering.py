"""Report rendering helpers and section formatters."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Tuple

from .report_shared import (
    _as_float,
    _format_decimal,
    _format_probability,
    _format_series_preview,
    _format_signed,
    _format_state_shares,
    _format_table,
    _get_indicator_value,
    format_number,
)


def _report_time_label(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value
    try:
        epoch = float(value)
    except Exception:
        return None
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def render_enhanced_report(report: Dict[str, Any]) -> str:
    """Render a markdown report with a consistent basic-style layout."""
    if not isinstance(report, dict):
        return "error: invalid report payload\n"

    output_lines: List[str] = []

    status_block = _render_sections_status(report.get("sections_status"))
    if status_block:
        output_lines.extend(status_block)
        output_lines.append("")

    sections = report.get("sections", {})
    if not isinstance(sections, dict):
        sections = {}

    rendered_keys: List[str] = []
    for key, formatter in _SECTION_RENDERERS:
        payload = sections.get(key)
        if payload is None:
            continue
        block = formatter(payload) if payload else []
        rendered_keys.append(key)
        if block:
            output_lines.extend(block)
            output_lines.append("")

    for key in sorted(sections.keys()):
        if key in rendered_keys:
            continue
        payload = sections[key]
        if isinstance(payload, dict) and set(payload.keys()) <= {"error"}:
            continue
        block = _render_generic_section(key, payload)
        if block:
            output_lines.extend(block)
            output_lines.append("")

    rendered = "\n".join(line.rstrip() for line in output_lines).rstrip()
    return rendered + "\n" if rendered else ""


def _render_sections_status(data: Any) -> List[str]:
    if not isinstance(data, dict):
        return []
    summary = data.get("summary")
    sections = data.get("sections")
    if not isinstance(summary, dict) or not isinstance(sections, dict):
        return []
    lines = ["## Section Status"]
    summary_row = [
        [
            str(int(summary.get("ok", 0))),
            str(int(summary.get("partial", 0))),
            str(int(summary.get("error", 0))),
        ]
    ]
    lines.extend(_format_table(["OK", "Partial", "Error"], summary_row, name="summary"))
    section_rows: List[List[str]] = []
    for name in sorted(sections.keys()):
        section_rows.append([str(name), str(sections[name])])
    if section_rows:
        lines.extend(
            _format_table(["Section", "Status"], section_rows, name="sections")
        )
    return lines


def _render_context_section(data: Any) -> List[str]:
    if not isinstance(data, dict):
        return []
    lines: List[str] = ["## Market Context"]

    if data.get("error"):
        lines.append(f"error: {data.get('error')}")
        return lines

    metrics: List[List[str | None]] = []
    tf_ref = data.get("timeframe")
    if tf_ref:
        metrics.append(["Timeframe", str(tf_ref)])
    snap = (
        data.get("last_snapshot") if isinstance(data.get("last_snapshot"), dict) else {}
    )
    price = snap.get("close")
    if price is not None:
        metrics.append(["Close", _format_decimal(price, 5)])
    ema20 = _get_indicator_value(snap, "EMA_20")
    ema50 = _get_indicator_value(snap, "EMA_50")
    if ema20 is not None and ema50 is not None:
        trend_state = "Above EMAs"
        p_val = _as_float(price)
        e20 = _as_float(ema20)
        e50 = _as_float(ema50)
        if not (
            p_val is not None
            and e20 is not None
            and e50 is not None
            and p_val > e20 > e50
        ):
            trend_state = "Mixed slope"
        metrics.append(
            [
                "EMA trend",
                f"{trend_state} (EMA20 {_format_decimal(ema20, 5)}, EMA50 {_format_decimal(ema50, 5)})",
            ]
        )
    rsi = _get_indicator_value(snap, "RSI_14")
    if rsi is not None:
        rsi_val = _as_float(rsi)
        if rsi_val is not None:
            if rsi_val >= 70:
                tag = "overbought"
            elif rsi_val <= 30:
                tag = "oversold"
            else:
                tag = "neutral"
            metrics.append(["RSI(14)", f"{_format_decimal(rsi, 2)} ({tag})"])
        else:
            metrics.append(["RSI(14)", _format_decimal(rsi, 2)])
    trend = (
        data.get("trend_compact")
        if isinstance(data.get("trend_compact"), dict)
        else None
    )
    if trend:
        slopes = trend.get("s") or []
        if slopes:
            slope_vals = ", ".join(_format_signed(s / 100.0) for s in slopes[:3])
            metrics.append(["Slope (ATR adj 5/20/60)", slope_vals])
        r2_vals = trend.get("r") or []
        if r2_vals:
            r2_fmt = ", ".join(f"{max(0, min(100, int(r)))}%" for r in r2_vals[:3])
            metrics.append(["Fit quality (R^2 5/20/60)", r2_fmt])
        atr_bps = trend.get("v")
        if atr_bps is not None:
            metrics.append(["ATR as bps", str(int(atr_bps))])
        squeeze = trend.get("q")
        if squeeze is not None:
            metrics.append(["Squeeze percentile", f"{int(squeeze)}%"])
        regime_map = {
            0: "neutral",
            1: "uptrend",
            2: "downtrend",
            3: "breakout up",
            4: "breakout down",
        }
        regime = trend.get("g")
        if regime in regime_map:
            metrics.append(["Regime signal", regime_map[int(regime)]])
        bars_high = trend.get("h")
        bars_low = trend.get("l")
        if bars_high is not None or bars_low is not None:
            metrics.append(
                [
                    "Bars since swing high/low",
                    f"{bars_high if bars_high is not None else 'n/a'} / {bars_low if bars_low is not None else 'n/a'}",
                ]
            )
    if metrics:
        lines.extend(_format_table(["Metric", "Value"], metrics, name="metrics"))
    note = str(data.get("notes", "")).strip()
    if note:
        lines.append(f"- Note: {note}")
    return lines


def _render_contexts_multi_section(data: Any) -> List[str]:
    if not isinstance(data, dict):
        return []
    rows: List[List[str | None]] = []
    for tf in sorted(data.keys()):
        snap = data[tf]
        if not isinstance(snap, dict):
            continue
        trend = (
            snap.get("trend_compact")
            if isinstance(snap.get("trend_compact"), dict)
            else None
        )
        slope_val = None
        atr_bps = None
        if trend:
            slopes = trend.get("s") or []
            if slopes:
                slope_val = slopes[0] / 100.0 if slopes[0] is not None else None
            atr_bps = trend.get("v")
        close_val = snap.get("close")
        ema20_val = snap.get("ema20") or _get_indicator_value(snap, "EMA_20")
        ema50_val = snap.get("ema50") or _get_indicator_value(snap, "EMA_50")
        ema200_val = snap.get("ema200") or _get_indicator_value(snap, "EMA_200")
        rsi_val = snap.get("rsi") or _get_indicator_value(snap, "RSI_14")
        rows.append(
            [
                str(tf),
                _format_decimal(close_val, 5),
                _format_decimal(ema20_val, 5),
                _format_decimal(ema50_val, 5),
                _format_decimal(ema200_val, 5),
                _format_decimal(rsi_val, 2),
                _format_signed(slope_val),
                str(int(atr_bps)) if atr_bps is not None else None,
            ]
        )
    rows = [row for row in rows if any(cell not in (None, "n/a") for cell in row[1:])]
    if not rows:
        return []
    lines = ["## Multi-Timeframe Context"]
    lines.extend(
        _format_table(
            ["TF", "Close", "EMA20", "EMA50", "EMA200", "RSI", "Slope(5)", "ATR bps"],
            rows,
            name="timeframes",
        )
    )
    return lines


def _render_pivot_section(data: Any) -> List[str]:
    if not isinstance(data, dict):
        return []
    levels = data.get("levels")
    if not isinstance(levels, list) or not levels:
        return []
    timeframe = str(data.get("timeframe") or "").upper()
    period = data.get("period")
    title_parts: List[str] = []
    if timeframe:
        title_parts.append(timeframe)
    if isinstance(period, dict):
        start = period.get("start")
        end = period.get("end")
        if start and end:
            title_parts.append(f"{start}->{end}")
    elif period:
        title_parts.append(str(period))
    title = "## Pivot Levels"
    if title_parts:
        title += " (" + ", ".join(title_parts) + ")"
    lines = [title]
    methods = []
    method_meta = data.get("methods")
    if isinstance(method_meta, list):
        for item in method_meta:
            if isinstance(item, dict):
                name = item.get("method")
                if name and name not in methods:
                    methods.append(str(name))
    if not methods:
        for row in levels:
            if isinstance(row, dict):
                for key in row.keys():
                    if key != "level" and key not in methods:
                        methods.append(key)
    headers = ["Level"] + [m.title() for m in methods]
    table_rows: List[List[str | None]] = []
    for row in levels:
        if not isinstance(row, dict):
            continue
        level_name = str(row.get("level") or "").upper()
        row_vals = [level_name or "n/a"]
        for method in methods:
            value = row.get(method)
            row_vals.append(format_number(value) if value is not None else None)
        table_rows.append(row_vals)
    lines.extend(_format_table(headers, table_rows, name="levels"))
    context_line = _build_pivot_context_line(data)
    if context_line:
        lines.append(context_line)
    return lines


def _build_pivot_context_line(data: Dict[str, Any]) -> str | None:
    parts: List[str] = []
    calc = data.get("calculation_basis")
    if isinstance(calc, dict):
        source_bar = calc.get("source_bar")
        if source_bar:
            parts.append(f"source={source_bar}")
        session_boundary = calc.get("session_boundary")
        if session_boundary:
            parts.append(f"session={session_boundary}")
        display_tz = calc.get("display_timezone")
        if display_tz:
            parts.append(f"display_tz={display_tz}")
    timezone_hint = data.get("timezone")
    if timezone_hint and not any(part.startswith("display_tz=") for part in parts):
        parts.append(f"timezone={timezone_hint}")
    if not parts:
        return None
    return "- Context: " + "; ".join(str(part) for part in parts if part)


def _render_pivot_multi_section(data: Any) -> List[str]:
    if not isinstance(data, dict):
        return []
    base_tf = (
        str(data.get("__base_timeframe__", "")).upper()
        if "__base_timeframe__" in data
        else ""
    )
    lines = ["## Multi-Timeframe Pivots"]
    for tf in sorted(data.keys()):
        if str(tf).startswith("__"):
            continue
        if base_tf and str(tf).upper() == base_tf:
            continue
        piv = data[tf]
        if not isinstance(piv, dict):
            continue
        levels = piv.get("levels")
        if not isinstance(levels, list) or not levels:
            continue
        methods = []
        method_meta = piv.get("methods")
        if isinstance(method_meta, list):
            for item in method_meta:
                if isinstance(item, dict):
                    name = item.get("method")
                    if name and name not in methods:
                        methods.append(str(name))
        if not methods:
            for row in levels:
                if isinstance(row, dict):
                    for key in row.keys():
                        if key != "level" and key not in methods:
                            methods.append(key)
        if not methods:
            continue
        rows: List[List[str]] = []
        for row in levels:
            if not isinstance(row, dict):
                continue
            level = str(row.get("level") or "").upper()
            row_vals = [level or "n/a"]
            for method in methods:
                value = row.get(method)
                row_vals.append(format_number(value) if value is not None else None)
            rows.append(row_vals)
        if rows:
            lines.append(f"### {tf}")
            context_line = _build_pivot_context_line(piv)
            if context_line:
                lines.append(context_line)
            lines.extend(
                _format_table(
                    ["Level"] + [m.title() for m in methods], rows, name="levels"
                )
            )
            lines.append("")
    return [line for line in lines if line != ""]


def _render_volatility_section(data: Any) -> List[str]:
    if not isinstance(data, dict):
        return []
    matrix = data.get("matrix")
    methods = data.get("methods")
    if not isinstance(matrix, list) or not matrix:
        return []
    if not isinstance(methods, list) or not methods:
        methods = [
            key
            for key in matrix[0].keys()
            if key not in {"horizon", "avg"}
            and not str(key).endswith("_note")
            and not str(key).endswith("_bar")
            and not str(key).endswith("_err")
        ]
    headers = ["Horizon"] + [m.upper() for m in methods]
    if any("avg" in row for row in matrix):
        headers.append("AVG")
    rows: List[List[str]] = []
    for row in matrix:
        if not isinstance(row, dict):
            continue
        horizon = row.get("horizon")
        line = [str(int(horizon)) if horizon is not None else "n/a"]
        for method in methods:
            val = row.get(method)
            line.append(format_number(val) if val is not None else "n/a")
        if "avg" in row:
            line.append(
                format_number(row.get("avg")) if row.get("avg") is not None else "n/a"
            )
        rows.append(line)
    if not rows:
        return []
    lines = ["## Volatility Snapshot", "*values are horizon sigma (returns)*"]
    lines.extend(_format_table(headers, rows, name="estimates"))
    return lines


def _render_forecast_section(data: Any) -> List[str]:
    if not isinstance(data, dict):
        return []
    lines = ["## Forecast"]
    method = data.get("method")
    if method:
        lines.append(f"- Method: {method}")
    last_observation = _report_time_label(
        data.get("last_observation_time", data.get("last_observation_epoch"))
    )
    forecast_start = _report_time_label(
        data.get("forecast_start_time", data.get("forecast_start_epoch"))
    )
    if last_observation is not None:
        lines.append(f"- Last observation: {last_observation}")
    if forecast_start is not None:
        lines.append(f"- Forecast start: {forecast_start}")
    if data.get("forecast_anchor") is not None:
        lines.append(f"- Forecast anchor: {data.get('forecast_anchor')}")
    if data.get("forecast_start_gap_bars") is not None:
        lines.append(
            f"- Forecast start gap (bars): {format_number(data.get('forecast_start_gap_bars'))}"
        )
    if data.get("forecast_step_seconds") is not None:
        lines.append(
            f"- Forecast step seconds: {format_number(data.get('forecast_step_seconds'))}"
        )
    if data.get("forecast_price") is not None:
        series_preview = _format_series_preview(data.get("forecast_price"), decimals=6)
        if series_preview is not None:
            lines.append(f"- Forecast price: {series_preview}")
        else:
            lines.append(f"- Forecast price: {format_number(data['forecast_price'])}")
    is_return = str(data.get("quantity", "")).lower() == "return" or isinstance(
        data.get("forecast_return"), list
    )
    if is_return:
        lower = data.get("lower_return", data.get("lower"))
        upper = data.get("upper_return", data.get("upper"))
    else:
        lower = data.get("lower_price")
        upper = data.get("upper_price")
    interval_label = "Return interval" if is_return else "Interval"
    if lower is not None or upper is not None:
        if isinstance(lower, list) or isinstance(upper, list):
            low_preview = (
                _format_series_preview(lower, decimals=6)
                if isinstance(lower, list)
                else format_number(lower)
            )
            up_preview = (
                _format_series_preview(upper, decimals=6)
                if isinstance(upper, list)
                else format_number(upper)
            )
            lines.append(f"- {interval_label}: {low_preview} to {up_preview}")
        else:
            lines.append(
                f"- {interval_label}: {format_number(lower)} to {format_number(upper)}"
            )
    if data.get("trend"):
        lines.append(f"- Trend: {data['trend']}")
    if data.get("ci_alpha") is not None:
        lines.append(f"- CI alpha: {format_number(data['ci_alpha'])}")
    return lines


def _render_barriers_section(data: Any) -> List[str]:
    if not isinstance(data, dict):
        return []
    if any(key in data for key in ("long", "short")):
        mode = str(data.get("mode", "pct")).lower()
        unit_lbl = "%"
        if mode == "pips":
            unit_lbl = "(ticks)"
        elif mode == "bps":
            unit_lbl = "(bps)"

        lines: List[str] = ["## Barrier Analytics"]
        headers = [
            "Direction",
            f"TP {unit_lbl}",
            f"SL {unit_lbl}",
            "TP lvl",
            "SL lvl",
            "Edge",
            "Kelly",
            "EV",
            "TP hit %",
            "SL hit %",
            "No-hit %",
        ]
        rows: List[List[str | None]] = []
        negative_edge = False
        conflict_dirs: List[str] = []
        for dir_name in ("long", "short"):
            sub = data.get(dir_name)
            if not isinstance(sub, dict):
                continue
            best = sub.get("best") if isinstance(sub.get("best"), dict) else None
            if not best:
                continue
            try:
                edge_val = float(best.get("edge"))
                if math.isfinite(edge_val) and edge_val < 0:
                    negative_edge = True
            except (TypeError, ValueError):
                pass
            conflict_flag = bool(best.get("ev_edge_conflict"))
            if not conflict_flag:
                try:
                    ev_val = float(best.get("ev"))
                    edge_val = float(best.get("edge"))
                    conflict_flag = (
                        math.isfinite(ev_val)
                        and math.isfinite(edge_val)
                        and (
                            (ev_val > 0.0 and edge_val < 0.0)
                            or (ev_val < 0.0 and edge_val > 0.0)
                        )
                    )
                except (TypeError, ValueError):
                    conflict_flag = False
            if conflict_flag:
                conflict_dirs.append(dir_name)
            rows.append(
                [
                    dir_name,
                    _format_decimal(best.get("tp"), 3),
                    _format_decimal(best.get("sl"), 3),
                    _format_decimal(best.get("tp_price"), 5),
                    _format_decimal(best.get("sl_price"), 5),
                    _format_decimal(best.get("edge"), 3),
                    _format_decimal(best.get("kelly"), 3),
                    _format_decimal(best.get("ev"), 3),
                    _format_probability(best.get("prob_tp_first")),
                    _format_probability(best.get("prob_sl_first")),
                    _format_probability(best.get("prob_no_hit")),
                ]
            )
        if rows:
            lines.extend(_format_table(headers, rows, name="candidates"))
            if negative_edge:
                lines.append(
                    "- Warning: best candidate has negative edge (expected value)."
                )
            for direction in conflict_dirs:
                lines.append(
                    f"- CAUTION ({direction}): EV and edge have opposite signs; reward/risk skew may mask low win probability."
                )
            section_caution = data.get("caution")
            if isinstance(section_caution, str) and section_caution.strip():
                lines.append(f"- CAUTION: {section_caution.strip()}")
            section_note = data.get("note")
            if isinstance(section_note, str) and section_note.strip():
                lines.append(f"- Note: {section_note.strip()}")
            return lines
        return []
    lines = ["## Barrier Analytics"]
    if isinstance(data.get("direction"), str):
        lines.append(f"- Direction: {data.get('direction')}")
    best = data.get("best") if isinstance(data.get("best"), dict) else None
    if best:
        negative_edge = False
        ev_edge_conflict = bool(best.get("ev_edge_conflict"))
        try:
            edge_val = float(best.get("edge"))
            if math.isfinite(edge_val) and edge_val < 0:
                negative_edge = True
        except (TypeError, ValueError):
            negative_edge = False
        if not ev_edge_conflict:
            try:
                ev_val = float(best.get("ev"))
                edge_val = float(best.get("edge"))
                ev_edge_conflict = (
                    math.isfinite(ev_val)
                    and math.isfinite(edge_val)
                    and (
                        (ev_val > 0.0 and edge_val < 0.0)
                        or (ev_val < 0.0 and edge_val > 0.0)
                    )
                )
            except (TypeError, ValueError):
                ev_edge_conflict = False
        headers = [
            "TP %",
            "SL %",
            "TP lvl",
            "SL lvl",
            "Edge",
            "Kelly",
            "EV",
            "TP hit %",
            "SL hit %",
            "No-hit %",
        ]
        row = [
            _format_decimal(best.get("tp"), 3),
            _format_decimal(best.get("sl"), 3),
            _format_decimal(best.get("tp_price"), 5),
            _format_decimal(best.get("sl_price"), 5),
            _format_decimal(best.get("edge"), 3),
            _format_decimal(best.get("kelly"), 3),
            _format_decimal(best.get("ev"), 3),
            _format_probability(best.get("prob_tp_first")),
            _format_probability(best.get("prob_sl_first")),
            _format_probability(best.get("prob_no_hit")),
        ]
        lines.extend(_format_table(headers, [row], name="best"))
        if negative_edge:
            lines.append(
                "- Warning: best candidate has negative edge (expected value)."
            )
        if ev_edge_conflict:
            lines.append(
                "- CAUTION: EV and edge have opposite signs; reward/risk skew may mask low win probability."
            )
    top = data.get("top")
    if isinstance(top, list) and top:
        headers = ["Rank", "TP %", "SL %", "Edge", "Kelly", "EV"]
        rows: List[List[str | None]] = []
        for idx, row_data in enumerate(top, start=1):
            if not isinstance(row_data, dict):
                continue
            rows.append(
                [
                    str(idx),
                    _format_decimal(row_data.get("tp"), 3),
                    _format_decimal(row_data.get("sl"), 3),
                    _format_decimal(row_data.get("edge"), 3),
                    _format_decimal(row_data.get("kelly"), 3),
                    _format_decimal(row_data.get("ev"), 3),
                ]
            )
        if rows:
            lines.append("")
            lines.extend(_format_table(headers, rows, name="top"))
    section_note = data.get("note")
    if isinstance(section_note, str) and section_note.strip():
        lines.append(f"- Note: {section_note.strip()}")
    return lines if len(lines) > 1 else []


def _render_market_section(data: Any) -> List[str]:
    if not isinstance(data, dict):
        return []
    lines = ["## Market Snapshot"]
    if data.get("error"):
        lines.append(f"- Error: {data.get('error')}")
        return lines
    entries = []
    for label, key in [("Bid", "bid"), ("Ask", "ask"), ("Spread", "spread")]:
        val = data.get(key)
        if val is not None:
            entries.append(f"- {label}: {format_number(val)}")
    if data.get("tick_size") is not None:
        entries.append(f"- Tick size: {format_number(data.get('tick_size'))}")
    if data.get("spread_ticks") is not None:
        entries.append(f"- Spread (ticks): {format_number(data.get('spread_ticks'))}")
    elif data.get("spread_pips") is not None:
        entries.append(f"- Spread (pips): {format_number(data.get('spread_pips'))}")
    depth = data.get("depth")
    if isinstance(depth, dict):
        buy = depth.get("total_buy")
        sell = depth.get("total_sell")
        if buy is not None or sell is not None:
            entries.append(
                f"- DOM volume (buy/sell): {format_number(buy)} / {format_number(sell)}"
            )
    if not entries:
        return []
    lines.extend(entries)
    return lines


def _render_backtest_section(data: Any) -> List[str]:
    if not isinstance(data, dict):
        return []
    ranking = data.get("ranking")
    if not isinstance(ranking, list) or not ranking:
        return []
    rows: List[List[str]] = []
    for row in ranking:
        if not isinstance(row, dict):
            continue
        rows.append(
            [
                str(row.get("method") or ""),
                format_number(row.get("avg_rmse")),
                format_number(row.get("avg_mae")),
                format_number(row.get("avg_directional_accuracy")),
                str(row.get("successful_tests") or "0"),
            ]
        )
    lines = ["## Backtest Ranking"]
    lines.extend(
        _format_table(
            ["Method", "RMSE", "MAE", "DirAcc", "Tests"], rows, name="results"
        )
    )
    return lines


def _render_patterns_section(data: Any) -> List[str]:
    if not isinstance(data, dict):
        return []
    recent = data.get("recent")
    if not isinstance(recent, list) or not recent:
        return []
    lines = ["## Recent Patterns"]
    for row in recent:
        if not isinstance(row, dict):
            continue
        pattern = row.get("pattern") or row.get("Pattern") or row.get("name")
        time_val = row.get("time") or row.get("Time")
        direction = row.get("direction") or row.get("Direction")
        desc_parts: List[str] = []
        if pattern:
            desc_parts.append(str(pattern))
        if direction:
            desc_parts.append(str(direction))
        if time_val:
            desc_parts.append(str(time_val))
        if desc_parts:
            lines.append("- " + " | ".join(desc_parts))
    return lines if len(lines) > 1 else []


def _render_regime_section(data: Any) -> List[str]:
    if not isinstance(data, dict):
        return []
    lines = ["## Regime Signals"]

    def _format_regime_summary(name: str, summary: Any) -> str | None:
        if summary is None:
            return None
        if isinstance(summary, dict):
            parts: List[str] = []
            if name == "bocpd":
                for key in ("last_cp_prob", "max_cp_prob", "mean_cp_prob"):
                    val = summary.get(key)
                    if val is not None:
                        parts.append(f"{key}={format_number(val)}")
                cp_count = summary.get("change_points_count")
                if cp_count is not None:
                    parts.append(f"change_points={int(cp_count)}")
            else:
                last_state = summary.get("last_state")
                if last_state is not None:
                    parts.append(f"last_state={last_state}")
                shares = _format_state_shares(summary.get("state_shares"))
                if shares:
                    parts.append(f"shares={shares}")
                order = summary.get("state_order_by_sigma")
                if isinstance(order, dict) and order:
                    parts.append(f"order_by_sigma={order}")
            return ", ".join(parts) if parts else str(summary)
        return str(summary)

    bocpd = data.get("bocpd")
    if isinstance(bocpd, dict):
        summary = _format_regime_summary("bocpd", bocpd.get("summary"))
        if summary:
            lines.append(f"- BOCPD: {summary}")
        elif bocpd.get("error"):
            lines.append(f"- BOCPD error: {bocpd['error']}")
    hmm = data.get("hmm")
    if isinstance(hmm, dict):
        summary = _format_regime_summary("hmm", hmm.get("summary"))
        if summary:
            lines.append(f"- HMM: {summary}")
        elif hmm.get("error"):
            lines.append(f"- HMM error: {hmm['error']}")
    return lines if len(lines) > 1 else []


def _render_execution_gates_section(data: Any) -> List[str]:
    if not isinstance(data, dict) or not data:
        return []
    lines = ["## Execution Gates"]
    for key, value in data.items():
        label = key.replace("_", " ").title()
        if isinstance(value, bool):
            lines.append(f"- {label}: {'yes' if value else 'no'}")
        else:
            lines.append(f"- {label}: {format_number(value)}")
    return lines


def _render_volatility_har_section(data: Any) -> List[str]:
    if not isinstance(data, dict):
        return []
    lines = ["## HAR-RV Volatility"]
    if data.get("sigma_bar_return") is not None:
        lines.append(f"- Bar sigma: {format_number(data['sigma_bar_return'])}")
    if data.get("horizon_sigma_return") is not None:
        lines.append(f"- Horizon sigma: {format_number(data['horizon_sigma_return'])}")
    return lines


def _render_forecast_conformal_section(data: Any) -> List[str]:
    if not isinstance(data, dict):
        return []
    lines = ["## Conformal Intervals"]
    if data.get("method"):
        lines.append(f"- Method: {data['method']}")
    is_return = str(data.get("quantity", "")).lower() == "return" or isinstance(
        data.get("forecast_return"), list
    )
    if is_return:
        lower = data.get("lower_return", data.get("lower"))
        upper = data.get("upper_return", data.get("upper"))
    else:
        lower = data.get("lower_price")
        upper = data.get("upper_price")
    interval_label = "Return interval" if is_return else "Interval"
    if lower is not None or upper is not None:
        if isinstance(lower, list) or isinstance(upper, list):
            low_preview = (
                _format_series_preview(lower, decimals=6)
                if isinstance(lower, list)
                else format_number(lower)
            )
            up_preview = (
                _format_series_preview(upper, decimals=6)
                if isinstance(upper, list)
                else format_number(upper)
            )
            lines.append(f"- {interval_label}: {low_preview} to {up_preview}")
        else:
            lines.append(
                f"- {interval_label}: {format_number(lower)} to {format_number(upper)}"
            )
    per_step = data.get("per_step_q")
    if isinstance(per_step, list) and per_step:
        sliced = per_step[: min(5, len(per_step))]

        def _format_quantile_value(value: Any) -> str:
            if value is None:
                return "null"
            if isinstance(value, str):
                text = value.strip()
                if text == "" or text.lower() in {"nan", "inf", "+inf", "-inf"}:
                    return "null"
            try:
                num = float(value)
            except (TypeError, ValueError):
                text = str(value).strip()
                return text if text else "null"
            if not math.isfinite(num):
                return "null"
            return format_number(num)

        formatted = [
            f"q{idx}={_format_quantile_value(item)}"
            for idx, item in enumerate(sliced, start=1)
        ]
        lines.append(f"- First step quantiles: {', '.join(formatted)}")
    if data.get("ci_alpha") is not None:
        lines.append(f"- CI alpha: {format_number(data['ci_alpha'])}")
    return lines


def _render_generic_section(name: str, payload: Any) -> List[str]:
    if not payload:
        return []
    title = name.replace("_", " ").title()
    lines = [f"## {title}"]
    if isinstance(payload, dict):
        for key in sorted(payload.keys()):
            val = payload[key]
            if isinstance(val, (list, tuple)):
                preview = ", ".join(str(item) for item in list(val)[:5])
                lines.append(f"- {key}: {preview}{'...' if len(val) > 5 else ''}")
            elif isinstance(val, dict):
                nested = ", ".join(
                    f"{nested_key}={nested_val}"
                    for nested_key, nested_val in list(val.items())[:5]
                )
                lines.append(f"- {key}: {nested}{'...' if len(val) > 5 else ''}")
            else:
                lines.append(f"- {key}: {format_number(val)}")
    elif isinstance(payload, (list, tuple)):
        for item in payload[:20]:
            lines.append(f"- {item}")
        if len(payload) > 20:
            lines.append("- ...")
    else:
        lines.append(str(payload))
    return lines


_SECTION_RENDERERS: List[Tuple[str, Callable[[Any], List[str]]]] = [
    ("context", _render_context_section),
    ("contexts_multi", _render_contexts_multi_section),
    ("pivot", _render_pivot_section),
    ("pivot_multi", _render_pivot_multi_section),
    ("volatility", _render_volatility_section),
    ("forecast", _render_forecast_section),
    ("barriers", _render_barriers_section),
    ("market", _render_market_section),
    ("backtest", _render_backtest_section),
    ("patterns", _render_patterns_section),
    ("regime", _render_regime_section),
    ("execution_gates", _render_execution_gates_section),
    ("volatility_har_rv", _render_volatility_har_section),
    ("forecast_conformal", _render_forecast_conformal_section),
]
