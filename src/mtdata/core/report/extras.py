"""Optional report sections shared by style templates."""

from __future__ import annotations

from typing import Any, Callable, Dict, List

from .utils import (
    current_only_section_omission,
    is_bounded_report_window,
    report_section_enabled,
)

ToolCaller = Callable[..., Dict[str, Any]]


def _empty(value: Any) -> bool:
    return value in (None, "", [], {})


def extract_report_pattern_rows(payload: Any, *, limit: int = 5) -> List[Dict[str, Any]]:
    """Prefer compact `top_patterns`, then legacy `recent_patterns` / table rows."""
    if not isinstance(payload, dict):
        return []
    for key in ("top_patterns", "recent_patterns", "recent"):
        rows = payload.get(key)
        if isinstance(rows, list) and rows:
            return [row for row in rows[: max(1, int(limit))] if isinstance(row, dict)]
    from .utils import parse_table_tail

    tail = parse_table_tail(payload, tail=max(20, int(limit)))
    return [row for row in tail[-max(1, int(limit)) :] if isinstance(row, dict)]


def compact_report_pattern_row(row: Dict[str, Any]) -> Dict[str, Any]:
    item: Dict[str, Any] = {}
    for key in (
        "pattern",
        "name",
        "type",
        "direction",
        "signal",
        "confidence",
        "match_score",
        "score",
        "time",
        "timeframe",
        "status",
    ):
        value = row.get(key)
        if not _empty(value):
            item[key] = value
    if "name" not in item:
        label = row.get("pattern") or row.get("type")
        if not _empty(label):
            item["name"] = label
    if "match_score" not in item:
        score = row.get("confidence")
        if score is None:
            score = row.get("score")
        if not _empty(score):
            item["match_score"] = score
    return item


def summarize_confluence_payload(payload: Dict[str, Any], *, limit: int = 5) -> Dict[str, Any]:
    if payload.get("error"):
        return {"error": payload.get("error")}
    levels: List[Dict[str, Any]] = []
    raw_levels = payload.get("levels")
    if isinstance(raw_levels, list):
        for row in raw_levels:
            if not isinstance(row, dict):
                continue
            item = {
                key: row[key]
                for key in ("price", "role", "score", "distance_pct", "source_families")
                if not _empty(row.get(key))
            }
            if item:
                levels.append(item)
            if len(levels) >= max(1, int(limit)):
                break
    out: Dict[str, Any] = {"levels": levels}
    for key in ("reference_price", "coverage_note", "level_scan_note"):
        if not _empty(payload.get(key)):
            out[key] = payload.get(key)
    return out


def summarize_volume_profile_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    if payload.get("error"):
        return {"error": payload.get("error")}
    out: Dict[str, Any] = {}
    for key in (
        "poc",
        "vah",
        "val",
        "value_area",
        "profile_source",
        "volume_kind",
        "is_synthetic",
    ):
        if not _empty(payload.get(key)):
            out[key] = payload.get(key)
    return out or {"error": "Volume profile returned no usable levels."}


def summarize_news_payload(payload: Dict[str, Any], *, limit: int = 3) -> Dict[str, Any]:
    if payload.get("error"):
        return {"error": payload.get("error")}
    out: Dict[str, Any] = {}
    for bucket in ("upcoming_events", "related_news", "impact_news"):
        rows = payload.get(bucket)
        if not isinstance(rows, list) or not rows:
            continue
        items: List[Dict[str, Any]] = []
        for row in rows[: max(1, int(limit))]:
            if not isinstance(row, dict):
                continue
            item = {
                key: row[key]
                for key in ("title", "headline", "when", "published_at", "impact", "time")
                if not _empty(row.get(key))
            }
            if item:
                items.append(item)
        if items:
            out[bucket] = items
    if payload.get("status") not in (None, ""):
        out["status"] = payload.get("status")
    return out or {"status": payload.get("status") or "no_results"}


def summarize_session_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    if payload.get("error"):
        return {"error": payload.get("error")}
    keep = (
        "status",
        "state",
        "mode",
        "tradable",
        "is_open",
        "session",
        "next_open",
        "next_close",
        "timezone",
        "symbol",
        "reason",
        "message",
    )
    out = {key: payload[key] for key in keep if not _empty(payload.get(key))}
    symbol_status = payload.get("symbol_status")
    if isinstance(symbol_status, dict):
        nested = summarize_session_payload(symbol_status)
        if nested and "error" not in nested:
            out.update({key: value for key, value in nested.items() if key not in out})
    return out or {"status": payload.get("status") or "unknown"}


def summarize_temporal_payload(payload: Dict[str, Any], *, limit: int = 6) -> Dict[str, Any]:
    if payload.get("error"):
        return {"error": payload.get("error")}
    out: Dict[str, Any] = {}
    for key in ("group_by", "timezone", "session_calendar", "summary"):
        if not _empty(payload.get(key)):
            out[key] = payload.get(key)
    rows = payload.get("data")
    if not isinstance(rows, list):
        rows = payload.get("groups")
    if isinstance(rows, list) and rows:
        trimmed: List[Dict[str, Any]] = []
        for row in rows[: max(1, int(limit))]:
            if not isinstance(row, dict):
                continue
            item = {
                key: row[key]
                for key in (
                    "group",
                    "session",
                    "label",
                    "count",
                    "avg_return",
                    "win_rate",
                    "volatility",
                )
                if not _empty(row.get(key))
            }
            if item:
                trimmed.append(item)
        if trimmed:
            out["groups"] = trimmed
    return out or {"error": "Temporal analysis returned no grouped rows."}


def attach_optional_report_sections(
    report: Dict[str, Any],
    *,
    call: ToolCaller,
    symbol: str,
    timeframe: str,
    params: Dict[str, Any],
    start: Any,
    end: Any,
) -> None:
    """Attach confluence, volume profile, session, news, and temporal sections."""
    sections = report.setdefault("sections", {})
    bounded = is_bounded_report_window(start, end)

    if report_section_enabled(params, "confluence"):
        from ..pivot import confluence_levels

        confluence = call(
            confluence_levels,
            symbol=symbol,
            pivot_timeframe=str(params.get("confluence_pivot_timeframe") or "D1"),
            sr_timeframe=str(params.get("confluence_sr_timeframe") or "auto"),
            lookback=int(params.get("confluence_lookback", 200)),
            start=start,
            end=end,
            max_levels=int(params.get("confluence_max_levels", 5)),
            detail="compact",
        )
        if "error" in confluence:
            sections["confluence"] = {"error": confluence.get("error")}
        else:
            sections["confluence"] = summarize_confluence_payload(
                confluence,
                limit=int(params.get("confluence_max_levels", 5)),
            )

    if report_section_enabled(params, "volume_profile"):
        from ..volume_profile import volume_profile_levels

        profile_window = (
            {"start": start, "end": end}
            if start is not None
            else {
                "end": end,
                "timeframe": timeframe,
                "lookback": int(params.get("volume_profile_lookback", 200)),
            }
        )
        profile = call(
            volume_profile_levels,
            symbol=symbol,
            **profile_window,
            detail="compact",
        )
        if "error" in profile:
            sections["volume_profile"] = {"error": profile.get("error")}
        else:
            sections["volume_profile"] = summarize_volume_profile_payload(profile)

    if report_section_enabled(params, "session"):
        if bounded:
            sections["session"] = current_only_section_omission(
                "session", start=start, end=end
            )
        else:
            from ..market_status import market_status

            session = call(market_status, symbol=symbol, detail="compact")
            if "error" in session:
                sections["session"] = {"error": session.get("error")}
            else:
                sections["session"] = summarize_session_payload(session)

    if report_section_enabled(params, "news"):
        if bounded:
            sections["news"] = current_only_section_omission("news", start=start, end=end)
        else:
            from ..news import news

            headlines = call(news, symbol=symbol, detail="compact", limit=8)
            if "error" in headlines:
                sections["news"] = {"error": headlines.get("error")}
            else:
                sections["news"] = summarize_news_payload(headlines)

    if report_section_enabled(params, "temporal"):
        from ..temporal import temporal_analyze

        temporal = call(
            temporal_analyze,
            symbol=symbol,
            timeframe=timeframe,
            group_by=str(params.get("temporal_group_by") or "session"),
            start=start,
            end=end,
            detail="compact",
        )
        if "error" in temporal:
            sections["temporal"] = {"error": temporal.get("error")}
        else:
            sections["temporal"] = summarize_temporal_payload(temporal)
