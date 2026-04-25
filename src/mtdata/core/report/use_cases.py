from __future__ import annotations

import logging
import time
import warnings
from datetime import datetime, timezone
from typing import Any, Dict, List

from ..execution_logging import log_operation_exception, run_logged_operation
from .requests import ReportGenerateRequest

logger = logging.getLogger(__name__)


def _normalize_report_detail(value: Any, *, default: str = "compact") -> str:
    normalized = str(default if value is None else value).strip().lower()
    if normalized in {"summary", "summary_only"}:
        return "compact"
    if normalized == "full":
        return "full"
    if normalized == "standard":
        return "standard"
    return "compact"


def _report_time_label(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value
    try:
        epoch = float(value)
    except Exception:
        return None
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _has_payload_error(payload: Any) -> bool:
    if isinstance(payload, dict):
        err = payload.get("error")
        if isinstance(err, str) and err.strip():
            return True
        return any(_has_payload_error(value) for key, value in payload.items() if key != "error")
    if isinstance(payload, list):
        return any(_has_payload_error(item) for item in payload)
    return False


def _has_payload_content(payload: Any) -> bool:
    if isinstance(payload, dict):
        return any(_has_payload_content(value) for key, value in payload.items() if key != "error")
    if isinstance(payload, list):
        return any(_has_payload_content(item) for item in payload)
    return payload not in (None, "")


def _collect_payload_errors(payload: Any, *, path: str = "") -> List[Dict[str, str]]:
    errors: List[Dict[str, str]] = []
    if isinstance(payload, dict):
        err = payload.get("error")
        if isinstance(err, str) and err.strip():
            errors.append({"path": path or "error", "message": err.strip()})
        for key, value in payload.items():
            if key == "error":
                continue
            child_path = f"{path}.{key}" if path else str(key)
            errors.extend(_collect_payload_errors(value, path=child_path))
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            child_path = f"{path}[{index}]" if path else f"[{index}]"
            errors.extend(_collect_payload_errors(value, path=child_path))
    return errors


def _build_sections_status(sections: Dict[str, Any]) -> Dict[str, Any]:
    statuses: Dict[str, str] = {}
    details: Dict[str, Dict[str, Any]] = {}
    summary = {"ok": 0, "partial": 0, "error": 0}
    for name, payload in sections.items():
        has_error = _has_payload_error(payload)
        has_content = _has_payload_content(payload)
        errors = _collect_payload_errors(payload)
        if has_error and has_content:
            status = "partial"
        elif has_error:
            status = "error"
        else:
            status = "ok"
        statuses[str(name)] = status
        summary[status] += 1
        if status != "ok":
            details[str(name)] = {
                "status": status,
                "reason": (
                    "section contains usable data plus one or more nested errors"
                    if status == "partial"
                    else "section contains errors and no usable data"
                ),
                "errors": errors,
            }
    out: Dict[str, Any] = {
        "summary": summary,
        "sections": statuses,
        "definitions": {
            "ok": "section returned usable data and no nested errors",
            "partial": "section returned usable data but one or more nested sub-results failed",
            "error": "section returned no usable data because it failed",
        },
    }
    if details:
        out["details"] = details
    return out


def _prioritize_report_payload(report: Dict[str, Any]) -> Dict[str, Any]:
    preferred_keys = (
        "success",
        "completeness",
        "summary",
        "sections_status",
        "sections",
        "diagnostics",
    )
    ordered: Dict[str, Any] = {}
    for key in preferred_keys:
        if key in report:
            ordered[key] = report[key]
    for key, value in report.items():
        if key not in ordered:
            ordered[key] = value
    return ordered


def _compact_report_payload(
    report: Dict[str, Any],
    *,
    symbol: str,
    template: str,
) -> Dict[str, Any]:
    compact: Dict[str, Any] = {
        "success": bool(report.get("success", True)),
        "symbol": symbol,
        "template": template,
        "detail": "compact",
    }
    completeness = report.get("completeness")
    if completeness not in (None, "", [], {}):
        compact["completeness"] = completeness
    for key in ("summary", "sections_status"):
        value = report.get(key)
        if value not in (None, "", [], {}):
            compact[key] = value
    diagnostics = report.get("diagnostics")
    if isinstance(diagnostics, dict):
        warnings_list = diagnostics.get("warnings")
        if warnings_list not in (None, "", [], {}):
            compact["warnings"] = warnings_list
    return compact


def run_report_generate(  # noqa: C901
    request: ReportGenerateRequest,
    *,
    render_report: Any,
    format_number: Any,
    get_indicator_value: Any,
    report_error_text: Any,
    report_error_payload: Any,
    append_diagnostic_warning: Any,
) -> str | Dict[str, Any]:
    output_mode = str(request.format or "toon").strip().lower()
    if output_mode == "structured":
        output_mode = "toon"
    template_name = (request.template or "basic").lower().strip()
    detail_value = _normalize_report_detail(getattr(request, "detail", "compact"))

    def _run() -> str | Dict[str, Any]:  # noqa: C901
        started_at = time.perf_counter()

        try:
            name = template_name
            params = dict(request.params or {})
            if request.timeframe:
                params["timeframe"] = str(request.timeframe)
            if request.methods is not None:
                params["methods"] = request.methods

            try:
                from ..report_templates import (
                    template_advanced as _t_advanced,
                )
                from ..report_templates import (
                    template_basic as _t_basic,
                )
                from ..report_templates import (
                    template_intraday as _t_intraday,
                )
                from ..report_templates import (
                    template_minimal as _t_minimal,
                )
                from ..report_templates import (
                    template_position as _t_position,
                )
                from ..report_templates import (
                    template_scalping as _t_scalping,
                )
                from ..report_templates import (
                    template_swing as _t_swing,
                )
            except Exception as ex:
                if output_mode == "markdown":
                    return report_error_text(f"Failed to import report templates: {ex}")
                return report_error_payload(f"Failed to import report templates: {ex}")

            default_horizon = {
                "basic": 12,
                "minimal": 12,
                "advanced": 12,
                "scalping": 8,
                "intraday": 12,
                "swing": 24,
                "position": 30,
            }
            if isinstance(params.get("horizon"), (int, float)):
                eff_horizon = int(params.get("horizon"))
            elif request.horizon is not None and int(request.horizon) > 0:
                eff_horizon = int(request.horizon)
            else:
                eff_horizon = default_horizon.get(name, 12)

            captured_warnings: List[str] = []
            with warnings.catch_warnings(record=True) as warning_records:
                warnings.simplefilter("always")
                if name == "basic":
                    rep = _t_basic(request.symbol, eff_horizon, request.denoise, params)
                elif name == "minimal":
                    rep = _t_minimal(request.symbol, eff_horizon, request.denoise, params)
                elif name == "advanced":
                    rep = _t_advanced(request.symbol, eff_horizon, request.denoise, params)
                elif name == "scalping":
                    rep = _t_scalping(request.symbol, eff_horizon, request.denoise, params)
                elif name == "intraday":
                    rep = _t_intraday(request.symbol, eff_horizon, request.denoise, params)
                elif name == "swing":
                    rep = _t_swing(request.symbol, eff_horizon, request.denoise, params)
                elif name == "position":
                    rep = _t_position(request.symbol, eff_horizon, request.denoise, params)
                else:
                    msg = (
                        f"Unknown template: {request.template}. "
                        "Use one of basic, minimal, advanced, scalping, intraday, swing, position."
                    )
                    if output_mode == "markdown":
                        return report_error_text(msg)
                    return report_error_payload(msg)

            for warning_obj in warning_records:
                try:
                    warning_text = str(warning_obj.message).strip()
                except Exception:
                    warning_text = ""
                if warning_text:
                    captured_warnings.append(warning_text)

            if not isinstance(rep, dict):
                msg = "Report template returned an unexpected payload."
                if output_mode == "markdown":
                    return report_error_text(msg)
                return report_error_payload(msg)
            if rep.get("error"):
                msg = rep.get("error")
                if output_mode == "markdown":
                    return report_error_text(msg)
                return report_error_payload(msg)
            if captured_warnings:
                for warning_text in captured_warnings:
                    append_diagnostic_warning(rep, warning_text)

            summ: List[str] = []
            try:
                ctx = rep.get("sections", {}).get("context", {})
                last = ctx.get("last_snapshot") or {}
                price = last.get("close")
                ema20 = get_indicator_value(last, "EMA_20")
                ema50 = get_indicator_value(last, "EMA_50")
                rsi = get_indicator_value(last, "RSI_14")
                if price is not None:
                    summ.append(f"close={format_number(price)}")
                if price is not None and ema20 is not None and ema50 is not None:
                    trend_note = (
                        "trend: above EMAs"
                        if float(price or 0) > float(ema20) > float(ema50)
                        else "trend: mixed"
                    )
                    summ.append(trend_note)
                if rsi is not None:
                    summ.append(f"RSI={format_number(rsi)}")
            except Exception:
                pass

            try:
                piv = rep.get("sections", {}).get("pivot", {})
                lev_rows = piv.get("levels")
                methods_meta = piv.get("methods")
                chosen_method = None
                if isinstance(methods_meta, list):
                    for meta in methods_meta:
                        if not isinstance(meta, dict):
                            continue
                        method_name = str(meta.get("method") or "").strip()
                        if method_name:
                            chosen_method = method_name
                            break
                chosen_method = chosen_method or "classic"
                available_methods: List[str] = []
                if isinstance(lev_rows, list):
                    for row in lev_rows:
                        if not isinstance(row, dict):
                            continue
                        for key in row.keys():
                            if key == "level":
                                continue
                            key_str = str(key)
                            if key_str not in available_methods:
                                available_methods.append(key_str)
                if available_methods and chosen_method not in available_methods:
                    chosen_method = available_methods[0]

                def _pivot_lookup(level_key: str):
                    target = level_key.lower()
                    alt = "pivot" if target == "pp" else None
                    if not isinstance(lev_rows, list):
                        return None
                    for row in lev_rows:
                        if not isinstance(row, dict):
                            continue
                        lvl_name = str(row.get("level") or "").strip().lower()
                        if lvl_name == target or (alt and lvl_name == alt):
                            return row.get(chosen_method)
                    return None

                pp = _pivot_lookup("PP")
                r1 = _pivot_lookup("R1")
                s1 = _pivot_lookup("S1")
                if pp is not None and r1 is not None and s1 is not None:
                    summ.append(
                        f"pivot {chosen_method} PP={format_number(pp)} "
                        f"(R1={format_number(r1)}, S1={format_number(s1)})"
                    )
                calc_basis = (
                    piv.get("calculation_basis")
                    if isinstance(piv.get("calculation_basis"), dict)
                    else {}
                )
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
                vol = rep.get("sections", {}).get("volatility", {})
                if isinstance(vol, dict):
                    hs = vol.get("horizon_sigma_price") or vol.get("horizon_sigma_return")
                    if hs is not None:
                        summ.append(f"h{eff_horizon} sigma={format_number(hs)}")
            except Exception:
                pass

            try:
                fc = rep.get("sections", {}).get("forecast", {})
                if isinstance(fc, dict) and "method" in fc:
                    method_name = str(fc.get("method"))
                    forecast_line = f"forecast={method_name}"
                    values = None
                    for key in ("forecast_price", "forecast_return", "forecast_series", "forecast"):
                        candidate = fc.get(key)
                        if isinstance(candidate, list) and candidate:
                            values = candidate
                            break
                    if isinstance(values, list) and len(values) >= 3:
                        nums: List[float] = []
                        for value in values:
                            try:
                                nums.append(float(value))
                            except Exception:
                                nums = []
                                break
                        if nums and len(nums) >= 3:
                            first = nums[0]
                            span = max(nums) - min(nums)
                            tol = max(1e-9, abs(first) * 1e-6)
                            if span <= tol:
                                forecast_line += " (flat)"
                                append_diagnostic_warning(
                                    rep,
                                    "Selected forecast appears degenerate (near-constant values across horizon).",
                                )
                    summ.append(forecast_line)
                    timing_parts: List[str] = []
                    last_obs = _report_time_label(
                        fc.get("last_observation_time", fc.get("last_observation_epoch"))
                    )
                    start_time = _report_time_label(
                        fc.get("forecast_start_time", fc.get("forecast_start_epoch"))
                    )
                    anchor = fc.get("forecast_anchor")
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
                backtest_sec = rep.get("sections", {}).get("backtest", {})
                criteria = backtest_sec.get("selection_criteria") if isinstance(backtest_sec, dict) else None
                best_payload = backtest_sec.get("best_method") if isinstance(backtest_sec, dict) else None
                if isinstance(criteria, dict):
                    primary = str(criteria.get("primary_metric") or "avg_rmse")
                    tie_breaker = str(criteria.get("tie_breaker") or "avg_directional_accuracy")
                    tol_pct = criteria.get("rmse_tolerance_pct")
                    if tol_pct is None:
                        tol_raw = criteria.get("rmse_tolerance")
                        try:
                            tol_pct = float(tol_raw) * 100.0 if tol_raw is not None else None
                        except Exception:
                            tol_pct = None
                    line = f"forecast selection: min {primary}"
                    if tol_pct is not None:
                        line += f", tie-window={format_number(tol_pct)}%"
                    line += f", tie-break={tie_breaker}"
                    min_da = criteria.get("min_directional_accuracy")
                    if min_da is not None:
                        line += f", min-dir-acc>={format_number(min_da)}"
                    if isinstance(best_payload, dict):
                        initial = best_payload.get("initial_method")
                        chosen = best_payload.get("method")
                        if initial and chosen and str(initial) != str(chosen):
                            line += ", fallback=degenerate-initial-forecast"
                    summ.append(line)
            except Exception:
                pass

            try:
                bar = rep.get("sections", {}).get("barriers", {})
                if isinstance(bar, dict) and any(k in bar for k in ("long", "short")):
                    for dname in ("long", "short"):
                        sub = bar.get(dname)
                        if not isinstance(sub, dict):
                            continue
                        best = sub.get("best") if isinstance(sub, dict) else None
                        if not best:
                            continue
                        tp = best.get("tp")
                        sl = best.get("sl")
                        ev = best.get("ev")
                        edge = best.get("edge")
                        details: List[str] = [f"dir={dname}"]
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
                            summ.append("barrier best " + " ".join(details))
                else:
                    best = bar.get("best") if isinstance(bar, dict) else None
                    direction = bar.get("direction") if isinstance(bar, dict) else None
                    if best:
                        tp = best.get("tp")
                        sl = best.get("sl")
                        ev = best.get("ev")
                        edge = best.get("edge")
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
                            summ.append("barrier best " + " ".join(details))
            except Exception:
                pass

            rep["summary"] = summ
            sections = rep.get("sections")
            if isinstance(sections, dict):
                sections_status = _build_sections_status(sections)
                rep["sections_status"] = sections_status
                summary_counts = sections_status.get("summary", {})
                error_count = int(summary_counts.get("error", 0))
                partial_count = int(summary_counts.get("partial", 0))
                rep["completeness"] = (
                    "failed"
                    if error_count > 0
                    else "partial"
                    if partial_count > 0
                    else "complete"
                )
                rep["success"] = bool(error_count == 0)
            diagnostics = rep.get("diagnostics")
            if not isinstance(diagnostics, dict):
                diagnostics = {}
            diagnostics["execution_time_ms"] = round((time.perf_counter() - started_at) * 1000.0, 3)
            rep["diagnostics"] = diagnostics
            rep["symbol"] = request.symbol
            rep["template"] = template_name
            rep["detail"] = detail_value
            rep = _prioritize_report_payload(rep)

            if output_mode == "markdown":
                return render_report(rep)
            if detail_value == "compact":
                return _compact_report_payload(rep, symbol=request.symbol, template=template_name)
            if detail_value == "standard":
                rep = dict(rep)
                rep.pop("diagnostics", None)
                rep["detail"] = "standard"
            return rep
        except Exception as exc:
            log_operation_exception(
                logger,
                operation="report_generate",
                started_at=started_at,
                exc=exc,
                symbol=request.symbol,
                template=template_name,
                format=output_mode,
            )
            msg = f"Error generating report: {exc}"
            if output_mode == "markdown":
                return report_error_text(msg)
            return report_error_payload(msg)

    return run_logged_operation(
        logger,
        operation="report_generate",
        symbol=request.symbol,
        template=template_name,
        format=output_mode,
        func=_run,
    )
