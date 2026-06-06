from __future__ import annotations

import logging
import math
import time
import warnings
from datetime import datetime, timezone
from typing import Any, Dict, List

from ..execution_logging import log_operation_exception, run_logged_operation
from ..output_contract import normalize_output_detail
from .requests import ReportGenerateRequest

logger = logging.getLogger(__name__)

_BARRIER_EV_EDGE_CONFLICT_NOTE = (
    "Expected value and break-even edge disagree; treat this barrier setup as "
    "lower-confidence and review win probability, payoff skew, and no-hit share."
)


def _round_report_barrier_metric(name: str, value: Any) -> Any:
    try:
        numeric = float(value)
    except Exception:
        return value
    if not math.isfinite(numeric):
        return value
    if name in {"tp_pct", "sl_pct"}:
        return round(numeric, 2)
    if name in {"ev", "edge", "edge_vs_breakeven"}:
        return round(numeric, 3)
    return value


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


def _is_report_error_noise(message: str) -> bool:
    return message.strip().lower() in {"", "no value", "none", "null"}


def _collect_payload_errors(payload: Any, *, path: str = "") -> List[Dict[str, str]]:
    errors: List[Dict[str, str]] = []
    if isinstance(payload, dict):
        err = payload.get("error")
        if isinstance(err, str):
            message = err.strip()
            if not _is_report_error_noise(message):
                errors.append({"path": path or "error", "message": message})
        for key, value in payload.items():
            if key == "error":
                continue
            child_path = f"{path}.{key}" if path else str(key)
            errors.extend(_collect_payload_errors(value, path=child_path))
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            child_path = f"{path}[{index}]" if path else f"[{index}]"
            errors.extend(_collect_payload_errors(value, path=child_path))
    deduped: List[Dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in errors:
        key = (item.get("path", ""), item.get("message", ""))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _is_user_facing_report_warning(warning_obj: Any) -> bool:
    category = getattr(warning_obj, "category", None)
    if isinstance(category, type) and issubclass(
        category,
        (DeprecationWarning, PendingDeprecationWarning, ImportWarning),
    ):
        return False
    try:
        warning_text = str(warning_obj.message).strip()
    except Exception:
        warning_text = ""
    if "torchao." in warning_text or "will be removed in a future release" in warning_text:
        return False
    return bool(warning_text)


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
    summary["total"] = len(statuses)
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
        "as_of",
        "timezone",
        "summary_structured",
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


def _valid_timezone_label(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    label = value.strip()
    return label or None


def _infer_report_timezone(report: Dict[str, Any]) -> str:
    for value in (
        report.get("timezone"),
        report.get("display_timezone"),
        (report.get("meta") or {}).get("timezone")
        if isinstance(report.get("meta"), dict)
        else None,
        (report.get("meta") or {}).get("display_timezone")
        if isinstance(report.get("meta"), dict)
        else None,
    ):
        label = _valid_timezone_label(value)
        if label:
            return label

    sections = report.get("sections")
    if isinstance(sections, dict):
        for section_name in ("context", "forecast", "market", "pivot"):
            section = sections.get(section_name)
            if not isinstance(section, dict):
                continue
            for key in ("timezone", "display_timezone"):
                label = _valid_timezone_label(section.get(key))
                if label:
                    return label
            calc_basis = section.get("calculation_basis")
            if isinstance(calc_basis, dict):
                label = _valid_timezone_label(calc_basis.get("display_timezone"))
                if label:
                    return label

        for section_name in ("contexts_multi", "pivot_multi"):
            section = sections.get(section_name)
            if not isinstance(section, dict):
                continue
            for item in section.values():
                if not isinstance(item, dict):
                    continue
                for key in ("timezone", "display_timezone"):
                    label = _valid_timezone_label(item.get(key))
                    if label:
                        return label
                calc_basis = item.get("calculation_basis")
                if isinstance(calc_basis, dict):
                    label = _valid_timezone_label(calc_basis.get("display_timezone"))
                    if label:
                        return label
    return "UTC"


def _attach_report_timezone(report: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(report, dict) or report.get("error"):
        return report
    if _valid_timezone_label(report.get("timezone")):
        return report
    out = dict(report)
    out["timezone"] = _infer_report_timezone(out)
    return out


def _add_barrier_conflict_notes(summary_structured: Any) -> Any:
    if not isinstance(summary_structured, dict):
        return summary_structured
    barriers = summary_structured.get("barriers")
    if not isinstance(barriers, dict):
        return summary_structured

    changed = False
    barriers_out: Dict[str, Any] = {}
    for name, entry in barriers.items():
        if isinstance(entry, dict) and bool(entry.get("ev_edge_conflict")):
            entry = dict(entry)
            entry.setdefault("trading_note", _BARRIER_EV_EDGE_CONFLICT_NOTE)
            changed = True
        barriers_out[name] = entry
    if not changed:
        return summary_structured

    out = dict(summary_structured)
    out["barriers"] = barriers_out
    return out


_COMPACT_SUMMARY_STRUCTURED_KEYS = (
    "market",
    "forecast",
    "backtest",
    "barriers",
    "patterns",
    "pivot",
    "volatility",
    "template_focus",
)


def _compact_summary_structured(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    out: Dict[str, Any] = {}
    for key in _COMPACT_SUMMARY_STRUCTURED_KEYS:
        section = value.get(key)
        if section not in (None, "", [], {}):
            out[key] = section
    if not out:
        return value
    omitted = [str(key) for key in value if key not in out]
    if omitted:
        out["omitted_sections"] = omitted
        out["show_full_hint"] = "Use detail=standard or detail=full for omitted report sections."
    return out


def _compact_report_payload(
    report: Dict[str, Any],
    *,
    symbol: str,
    template: str,
) -> Dict[str, Any]:
    def _barrier_conflict_warnings(summary_structured: Any) -> List[str]:
        if not isinstance(summary_structured, dict):
            return []
        barriers = summary_structured.get("barriers")
        if not isinstance(barriers, dict):
            return []
        directions: List[str] = []
        for name, entry in barriers.items():
            if not isinstance(entry, dict) or not bool(entry.get("ev_edge_conflict")):
                continue
            direction = entry.get("direction") or name
            if direction not in (None, ""):
                directions.append(str(direction))
        if not directions:
            return []
        if len(directions) == 1:
            joined = directions[0]
        elif len(directions) == 2:
            joined = f"{directions[0]} and {directions[1]}"
        else:
            joined = ", ".join(directions[:-1]) + f", and {directions[-1]}"
        return [f"Barrier EV/edge conflict detected for {joined} direction(s)."]

    compact: Dict[str, Any] = {
        "success": bool(report.get("success", True)),
        "symbol": symbol,
        "template": template,
        "detail": "compact",
    }
    timezone_label = _valid_timezone_label(report.get("timezone"))
    if timezone_label:
        compact["timezone"] = timezone_label
    if report.get("as_of") not in (None, ""):
        compact["as_of"] = report.get("as_of")
    completeness = report.get("completeness")
    if completeness not in (None, "", [], {}):
        compact["completeness"] = completeness
    assessment = report.get("overall_assessment")
    if assessment not in (None, "", [], {}):
        compact["overall_assessment"] = assessment
    elif report.get("executive_summary") not in (None, "", [], {}):
        compact["executive_summary"] = report.get("executive_summary")
    for key in ("section_controls",):
        value = report.get(key)
        if value not in (None, "", [], {}):
            compact[key] = value
    for key in ("summary_structured",):
        value = report.get(key)
        if value not in (None, "", [], {}):
            if key == "summary_structured":
                value = _add_barrier_conflict_notes(value)
                value = _compact_summary_structured(value)
            compact[key] = value
    if "summary_structured" not in compact:
        summary = report.get("summary")
        if summary not in (None, "", [], {}):
            compact["summary"] = summary
    diagnostics = report.get("diagnostics")
    warnings_out: List[Any] = []
    if isinstance(diagnostics, dict):
        warnings_list = diagnostics.get("warnings")
        if warnings_list not in (None, "", [], {}):
            if isinstance(warnings_list, list):
                warnings_out.extend(warnings_list)
            else:
                warnings_out.append(warnings_list)
    for warning in _barrier_conflict_warnings(compact.get("summary_structured")):
        if warning not in warnings_out:
            warnings_out.append(warning)
    if warnings_out:
        compact["warnings"] = warnings_out
    return compact


def _compact_report_top_patterns(patterns_section: Any, *, limit: int = 3) -> List[Dict[str, Any]]:
    if not isinstance(patterns_section, dict):
        return []
    rows = patterns_section.get("recent")
    if not isinstance(rows, list):
        return []
    compact: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        item = {
            key: row[key]
            for key in (
                "pattern",
                "name",
                "type",
                "direction",
                "signal",
                "confidence",
                "score",
                "time",
            )
            if row.get(key) not in (None, "", [], {})
        }
        if item:
            compact.append(item)
        if len(compact) >= max(1, int(limit)):
            break
    return compact


def _section_timeframes(section: Any) -> List[str]:
    if not isinstance(section, dict):
        return []
    return [
        str(key)
        for key, value in section.items()
        if key != "__base_timeframe__" and isinstance(value, dict)
    ]


def _split_report_section_names(value: Any) -> Optional[List[str]]:
    if value is None:
        return None
    if isinstance(value, str):
        names = [item.strip() for item in value.replace(",", " ").split() if item.strip()]
        return names or None
    if isinstance(value, (list, tuple)):
        names = [str(item).strip() for item in value if str(item).strip()]
        return names or None
    return None


def _apply_report_section_controls(
    report: Dict[str, Any],
    *,
    include_sections: Any = None,
    max_sections: Optional[int] = None,
    summary_only: bool = False,
) -> None:
    sections = report.get("sections")
    if not isinstance(sections, dict):
        return

    original_names = list(sections.keys())
    if summary_only:
        selected_names: List[str] = []
        missing_requested: List[str] = []
    else:
        requested_names = _split_report_section_names(include_sections)
        if requested_names:
            requested_lookup = {name.casefold(): name for name in original_names}
            selected_names = []
            missing_requested = []
            for requested in requested_names:
                actual = requested_lookup.get(requested.casefold())
                if actual is None:
                    missing_requested.append(requested)
                elif actual not in selected_names:
                    selected_names.append(actual)
        else:
            selected_names = list(original_names)
            missing_requested = []

        if max_sections is not None:
            selected_names = selected_names[: max(0, int(max_sections))]

    report["sections"] = {name: sections[name] for name in selected_names if name in sections}
    omitted_names = [name for name in original_names if name not in selected_names]
    if omitted_names or missing_requested or summary_only or max_sections is not None or include_sections:
        report["section_controls"] = {
            "summary_only": bool(summary_only),
            "included_sections": selected_names,
            "included_count": len(selected_names),
            "omitted_sections": omitted_names,
            "omitted_count": len(omitted_names),
        }
        if max_sections is not None:
            report["section_controls"]["max_sections"] = int(max_sections)
        if missing_requested:
            report["section_controls"]["missing_requested_sections"] = missing_requested


def _report_section_names_by_status(
    sections_status: Any,
    status: str,
) -> List[str]:
    if not isinstance(sections_status, dict):
        return []
    sections = sections_status.get("sections")
    if not isinstance(sections, dict):
        return []
    names: List[str] = []
    for name, payload in sections.items():
        if isinstance(payload, str) and payload.lower() == status:
            names.append(str(name))
        elif isinstance(payload, dict) and str(payload.get("status") or "").lower() == status:
            names.append(str(name))
    return names


def _build_overall_report_assessment(report: Dict[str, Any]) -> Dict[str, Any]:
    sections_status = report.get("sections_status")
    summary = sections_status.get("summary", {}) if isinstance(sections_status, dict) else {}
    total = int(summary.get("total", 0) or 0)
    errors = int(summary.get("error", 0) or 0)
    partial = int(summary.get("partial", 0) or 0)
    ok = int(summary.get("ok", 0) or 0)

    failed_sections = _report_section_names_by_status(sections_status, "error")
    partial_sections = _report_section_names_by_status(sections_status, "partial")

    if total <= 0:
        confidence = "low"
        recommended_action = "rerun_with_full_detail"
        summary_text = "No report sections were available for assessment."
    elif errors > 0:
        confidence = "low" if errors >= max(1, total // 3) else "medium"
        recommended_action = "use_with_caution"
        summary_text = "Report is usable only with caution because one or more sections failed."
    elif partial > 0:
        confidence = "medium"
        recommended_action = "review_partial_sections"
        summary_text = "Report is mostly usable, but partial sections reduce confidence."
    else:
        confidence = "high" if ok >= 3 else "medium"
        recommended_action = "review_key_levels_and_risk"
        summary_text = "Report sections completed successfully; review levels, forecast, and risk context before acting."

    assessment: Dict[str, Any] = {
        "is_trade_signal": False,
        "recommended_action": recommended_action,
        "confidence": confidence,
        "summary": summary_text,
        "section_health": {
            "ok": ok,
            "partial": partial,
            "error": errors,
            "total": total,
        },
    }
    if failed_sections:
        assessment["failed_sections"] = failed_sections[:6]
    if partial_sections:
        assessment["partial_sections"] = partial_sections[:6]
    return assessment


def _build_report_executive_summary(
    report: Dict[str, Any],
    *,
    symbol: str,
    template: str,
) -> Dict[str, Any]:
    assessment = report.get("overall_assessment")
    if not isinstance(assessment, dict):
        assessment = {}
    summary_structured = report.get("summary_structured")
    if not isinstance(summary_structured, dict):
        summary_structured = {}
    out: Dict[str, Any] = {
        "symbol": symbol,
        "template": template,
        "is_trade_signal": bool(assessment.get("is_trade_signal", False)),
        "recommended_action": assessment.get("recommended_action"),
        "confidence": assessment.get("confidence"),
        "completeness": report.get("completeness"),
    }
    section_health = assessment.get("section_health")
    if isinstance(section_health, dict):
        out["section_health"] = section_health
    for key in ("context", "backtest", "barriers", "patterns", "template_focus"):
        value = summary_structured.get(key)
        if value not in (None, "", [], {}):
            out[key] = value
    sections_with_issues = report.get("sections_with_issues")
    if sections_with_issues not in (None, "", [], {}):
        out["sections_with_issues"] = sections_with_issues
    return {key: value for key, value in out.items() if value not in (None, "", [], {})}


def _report_template_focus(
    *,
    template: str,
    report: Dict[str, Any],
    horizon: int,
) -> Dict[str, Any]:
    sections = report.get("sections")
    if not isinstance(sections, dict):
        return {}
    profile_by_template = {
        "basic": "balanced",
        "advanced": "regime_volatility",
        "scalping": "short_horizon_spread",
        "intraday": "intraday_mtf",
        "swing": "swing_mtf",
        "position": "higher_timeframe_mtf",
    }
    focus: Dict[str, Any] = {
        "profile": profile_by_template.get(template, template),
        "horizon": int(horizon),
    }
    meta = report.get("meta")
    if isinstance(meta, dict) and meta.get("timeframe") not in (None, "", [], {}):
        focus["base_timeframe"] = meta.get("timeframe")
    context_tfs = _section_timeframes(sections.get("contexts_multi"))
    if context_tfs:
        focus["context_timeframes"] = context_tfs
    pivot_tfs = _section_timeframes(sections.get("pivot_multi"))
    if pivot_tfs:
        focus["pivot_timeframes"] = pivot_tfs
    if isinstance(sections.get("market"), dict):
        focus["market_snapshot"] = True
    regime = sections.get("regime")
    if isinstance(regime, dict):
        methods = [
            str(key)
            for key, value in regime.items()
            if isinstance(value, dict) and key not in {"error", "warning"}
        ]
        if methods:
            focus["regime_methods"] = methods
    if isinstance(sections.get("volatility_har_rv"), dict):
        focus["extra_volatility"] = "har_rv"
    return focus


def run_report_generate(  # noqa: C901
    request: ReportGenerateRequest,
    *,
    format_number: Any,
    get_indicator_value: Any,
    report_error_payload: Any,
    append_diagnostic_warning: Any,
) -> str | Dict[str, Any]:
    template_name = (request.template or "basic").lower().strip()
    detail_value = normalize_output_detail(
        getattr(request, "detail", "compact"),
        aliases={"summary": "compact", "summary_only": "compact"},
    )

    def _run() -> str | Dict[str, Any]:  # noqa: C901
        started_at = time.perf_counter()

        try:
            name = template_name
            params = dict(request.params or {})
            if request.timeframe:
                params["timeframe"] = str(request.timeframe)
            if request.start:
                params["start"] = request.start
            if request.end:
                params["end"] = request.end
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
                    return report_error_payload(msg)

            for warning_obj in warning_records:
                if not _is_user_facing_report_warning(warning_obj):
                    continue
                try:
                    warning_text = str(warning_obj.message).strip()
                except Exception:
                    warning_text = ""
                if warning_text:
                    captured_warnings.append(warning_text)

            if not isinstance(rep, dict):
                msg = "Report template returned an unexpected payload."
                return report_error_payload(msg)
            if rep.get("error"):
                msg = rep.get("error")
                return report_error_payload(msg)
            if captured_warnings:
                for warning_text in captured_warnings:
                    append_diagnostic_warning(rep, warning_text)

            summ: List[str] = []
            summary_structured: Dict[str, Any] = {}
            try:
                ctx = rep.get("sections", {}).get("context", {})
                last = ctx.get("last_snapshot") or {}
                price = last.get("close")
                ema20 = get_indicator_value(last, "EMA_20")
                ema50 = get_indicator_value(last, "EMA_50")
                rsi = get_indicator_value(last, "RSI_14")
                market_summary: Dict[str, Any] = {}
                if price is not None:
                    summ.append(f"close={format_number(price)}")
                    market_summary["close"] = price
                if price is not None and ema20 is not None and ema50 is not None:
                    trend_note = (
                        "above EMAs"
                        if float(price or 0) > float(ema20) > float(ema50)
                        else "mixed"
                    )
                    summ.append(f"trend: {trend_note}")
                    market_summary["trend"] = trend_note
                if rsi is not None:
                    summ.append(f"RSI={format_number(rsi)}")
                    market_summary["rsi"] = rsi
                if market_summary:
                    summary_structured["market"] = market_summary
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
                    if isinstance(lev_rows, dict):
                        for candidate in (level_key, level_key.upper(), level_key.lower()):
                            if candidate in lev_rows:
                                return lev_rows.get(candidate)
                        if alt:
                            for candidate in (alt, alt.upper(), alt.lower()):
                                if candidate in lev_rows:
                                    return lev_rows.get(candidate)
                        return None
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
                pivot_summary: Dict[str, Any] = {"method": chosen_method}
                if pp is not None and r1 is not None and s1 is not None:
                    summ.append(
                        f"pivot {chosen_method} PP={format_number(pp)} "
                        f"(R1={format_number(r1)}, S1={format_number(s1)})"
                    )
                    pivot_summary.update({"PP": pp, "R1": r1, "S1": s1})
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
                    pivot_summary["session_boundary"] = session_boundary
                if display_tz:
                    context_parts.append(f"display_tz={display_tz}")
                    pivot_summary["display_timezone"] = display_tz
                if context_parts:
                    summ.append("pivot context " + " ".join(context_parts))
                if len(pivot_summary) > 1:
                    summary_structured["pivot"] = pivot_summary
            except Exception:
                pass

            try:
                vol = rep.get("sections", {}).get("volatility", {})
                if isinstance(vol, dict):
                    hs = vol.get("horizon_sigma_price") or vol.get("horizon_sigma_return")
                    vol_method = vol.get("method")
                    if hs is None:
                        matrix = vol.get("matrix")
                        if isinstance(matrix, list):
                            for row in matrix:
                                if not isinstance(row, dict):
                                    continue
                                if int(row.get("horizon") or 0) != int(eff_horizon):
                                    continue
                                hs = row.get("avg")
                                if hs is None:
                                    for key, value in row.items():
                                        if key in {"horizon", "avg"} or str(key).endswith(("_bar", "_err", "_note")):
                                            continue
                                        if isinstance(value, (int, float)):
                                            hs = value
                                            vol_method = str(key)
                                            break
                                if hs is not None and vol_method is None:
                                    methods = vol.get("methods")
                                    if isinstance(methods, list) and methods:
                                        vol_method = str(methods[0])
                                break
                    if hs is not None:
                        summ.append(f"h{eff_horizon} sigma={format_number(hs)}")
                        summary_structured["volatility"] = {
                            "horizon": eff_horizon,
                            "sigma": hs,
                        }
                        if vol_method:
                            summary_structured["volatility"]["method"] = vol_method
            except Exception:
                pass

            try:
                fc = rep.get("sections", {}).get("forecast", {})
                if isinstance(fc, dict) and "method" in fc:
                    method_name = str(fc.get("method"))
                    forecast_line = f"forecast={method_name}"
                    forecast_summary: Dict[str, Any] = {"method": method_name}
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
                                forecast_summary["flat"] = True
                                append_diagnostic_warning(
                                    rep,
                                    "Selected forecast appears degenerate (near-constant values across horizon).",
                                )
                            forecast_summary["first"] = nums[0]
                            forecast_summary["last"] = nums[-1]
                            forecast_summary["min"] = min(nums)
                            forecast_summary["max"] = max(nums)
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
                        forecast_summary["last_observation"] = last_obs
                    if start_time:
                        timing_parts.append(f"start={start_time}")
                        forecast_summary["start"] = start_time
                    if anchor:
                        timing_parts.append(f"anchor={anchor}")
                        forecast_summary["anchor"] = anchor
                    if timing_parts:
                        summ.append("forecast timing: " + " ".join(timing_parts))
                    summary_structured["forecast"] = forecast_summary
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
                    forecast_summary = summary_structured.setdefault("forecast", {})
                    if isinstance(forecast_summary, dict):
                        forecast_summary["selection"] = {
                            "primary_metric": primary,
                            "tie_breaker": tie_breaker,
                        }
                        if tol_pct is not None:
                            forecast_summary["rmse_tolerance_pct"] = tol_pct
                        if min_da is not None:
                            forecast_summary["min_directional_accuracy"] = min_da
                        if isinstance(best_payload, dict):
                            initial = best_payload.get("initial_method")
                            chosen = best_payload.get("method")
                            if initial is not None:
                                forecast_summary["initial_method"] = initial
                            if chosen is not None:
                                forecast_summary["chosen_method"] = chosen
                    summ.append(line)
                if isinstance(best_payload, dict):
                    best_method = best_payload.get("method")
                    stats = best_payload.get("stats")
                    backtest_summary: Dict[str, Any] = {}
                    if best_method not in (None, ""):
                        backtest_summary["best_method"] = best_method
                    if isinstance(stats, dict):
                        backtest_summary["stats"] = {
                            key: stats[key]
                            for key in (
                                "avg_rmse",
                                "avg_mae",
                                "avg_directional_accuracy",
                                "successful_tests",
                            )
                            if stats.get(key) is not None
                        }
                    if backtest_summary:
                        summary_structured["backtest"] = backtest_summary
            except Exception:
                pass

            try:
                bar = rep.get("sections", {}).get("barriers", {})
                barriers_summary: Dict[str, Any] = {}
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
                        edge_vs_breakeven = best.get("edge_vs_breakeven")
                        details: List[str] = [f"dir={dname}"]
                        barrier_entry: Dict[str, Any] = {}
                        if tp is not None:
                            tp_out = _round_report_barrier_metric("tp_pct", tp)
                            details.append(f"tp={format_number(tp_out)}%")
                            barrier_entry["tp_pct"] = tp_out
                        if sl is not None:
                            sl_out = _round_report_barrier_metric("sl_pct", sl)
                            details.append(f"sl={format_number(sl_out)}%")
                            barrier_entry["sl_pct"] = sl_out
                        if ev is not None:
                            ev_out = _round_report_barrier_metric("ev", ev)
                            details.append(f"ev={format_number(ev_out)}")
                            barrier_entry["ev"] = ev_out
                        if edge is not None:
                            edge_out = _round_report_barrier_metric("edge", edge)
                            details.append(f"edge={format_number(edge_out)}")
                            barrier_entry["edge"] = edge_out
                        if edge_vs_breakeven is not None:
                            edge_be_out = _round_report_barrier_metric(
                                "edge_vs_breakeven",
                                edge_vs_breakeven,
                            )
                            details.append(
                                f"edge_vs_breakeven={format_number(edge_be_out)}"
                            )
                            barrier_entry["edge_vs_breakeven"] = edge_be_out
                        try:
                            conflict_metric = (
                                "edge_vs_breakeven" if edge_vs_breakeven is not None else "edge"
                            )
                            conflict_value = edge_vs_breakeven if edge_vs_breakeven is not None else edge
                            if ev is not None and conflict_value is not None:
                                ev_num = float(ev)
                                edge_num = float(conflict_value)
                                if (ev_num > 0 and edge_num < 0) or (ev_num < 0 and edge_num > 0):
                                    reason = f"ev and {conflict_metric} have opposite signs"
                                    details.append("ev_edge_conflict=true")
                                    details.append(f"ev_edge_conflict_reason={reason}")
                                    barrier_entry["ev_edge_conflict"] = True
                                    barrier_entry["conflict_reason"] = reason
                                    barrier_entry["trading_note"] = _BARRIER_EV_EDGE_CONFLICT_NOTE
                        except Exception:
                            pass
                        if details:
                            summ.append("barrier best " + " ".join(details))
                        if barrier_entry:
                            barriers_summary[dname] = barrier_entry
                else:
                    best = bar.get("best") if isinstance(bar, dict) else None
                    direction = bar.get("direction") if isinstance(bar, dict) else None
                    if best:
                        tp = best.get("tp")
                        sl = best.get("sl")
                        ev = best.get("ev")
                        edge = best.get("edge")
                        edge_vs_breakeven = best.get("edge_vs_breakeven")
                        details: List[str] = []
                        barrier_entry: Dict[str, Any] = {}
                        if direction:
                            details.append(f"dir={str(direction)}")
                            barrier_entry["direction"] = str(direction)
                        if tp is not None:
                            tp_out = _round_report_barrier_metric("tp_pct", tp)
                            details.append(f"tp={format_number(tp_out)}%")
                            barrier_entry["tp_pct"] = tp_out
                        if sl is not None:
                            sl_out = _round_report_barrier_metric("sl_pct", sl)
                            details.append(f"sl={format_number(sl_out)}%")
                            barrier_entry["sl_pct"] = sl_out
                        if ev is not None:
                            ev_out = _round_report_barrier_metric("ev", ev)
                            details.append(f"ev={format_number(ev_out)}")
                            barrier_entry["ev"] = ev_out
                        if edge is not None:
                            edge_out = _round_report_barrier_metric("edge", edge)
                            details.append(f"edge={format_number(edge_out)}")
                            barrier_entry["edge"] = edge_out
                        if edge_vs_breakeven is not None:
                            edge_be_out = _round_report_barrier_metric(
                                "edge_vs_breakeven",
                                edge_vs_breakeven,
                            )
                            details.append(
                                f"edge_vs_breakeven={format_number(edge_be_out)}"
                            )
                            barrier_entry["edge_vs_breakeven"] = edge_be_out
                        try:
                            conflict_metric = (
                                "edge_vs_breakeven" if edge_vs_breakeven is not None else "edge"
                            )
                            conflict_value = edge_vs_breakeven if edge_vs_breakeven is not None else edge
                            if ev is not None and conflict_value is not None:
                                ev_num = float(ev)
                                edge_num = float(conflict_value)
                                if (ev_num > 0 and edge_num < 0) or (ev_num < 0 and edge_num > 0):
                                    reason = f"ev and {conflict_metric} have opposite signs"
                                    details.append("ev_edge_conflict=true")
                                    details.append(f"ev_edge_conflict_reason={reason}")
                                    barrier_entry["ev_edge_conflict"] = True
                                    barrier_entry["conflict_reason"] = reason
                                    barrier_entry["trading_note"] = _BARRIER_EV_EDGE_CONFLICT_NOTE
                        except Exception:
                            pass
                        if details:
                            summ.append("barrier best " + " ".join(details))
                        if barrier_entry:
                            barriers_summary["best"] = barrier_entry
                if barriers_summary:
                    summary_structured["barriers"] = barriers_summary
            except Exception:
                pass

            try:
                top_patterns = _compact_report_top_patterns(
                    rep.get("sections", {}).get("patterns", {})
                )
                if top_patterns:
                    summary_structured["patterns"] = {"recent": top_patterns}
            except Exception:
                pass

            try:
                if detail_value == "compact":
                    template_focus = _report_template_focus(
                        template=template_name,
                        report=rep,
                        horizon=eff_horizon,
                    )
                    if template_focus:
                        summary_structured["template_focus"] = template_focus
            except Exception:
                pass

            rep["summary"] = summ
            if summary_structured:
                rep["summary_structured"] = summary_structured
            _apply_report_section_controls(
                rep,
                include_sections=request.include_sections,
                max_sections=request.max_sections,
                summary_only=bool(request.summary_only),
            )
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
                sections_with_issues: Dict[str, List[str]] = {}
                partial_section_names = _report_section_names_by_status(sections_status, "partial")
                error_section_names = _report_section_names_by_status(sections_status, "error")
                if partial_section_names:
                    sections_with_issues["partial"] = partial_section_names
                if error_section_names:
                    sections_with_issues["error"] = error_section_names
                if sections_with_issues:
                    rep["sections_with_issues"] = sections_with_issues
                rep["overall_assessment"] = _build_overall_report_assessment(rep)
                rep["executive_summary"] = _build_report_executive_summary(
                    rep,
                    symbol=request.symbol,
                    template=template_name,
                )
            diagnostics = rep.get("diagnostics")
            if not isinstance(diagnostics, dict):
                diagnostics = {}
            diagnostics["execution_time_ms"] = round((time.perf_counter() - started_at) * 1000.0, 3)
            rep["diagnostics"] = diagnostics
            rep["symbol"] = request.symbol
            rep["template"] = template_name
            rep["detail"] = detail_value
            rep["as_of"] = (
                datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
            )
            rep = _attach_report_timezone(rep)
            rep = _prioritize_report_payload(rep)

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
            )
            msg = f"Error generating report: {exc}"
            return report_error_payload(msg)

    return run_logged_operation(
        logger,
        operation="report_generate",
        symbol=request.symbol,
        template=template_name,
        func=_run,
    )
