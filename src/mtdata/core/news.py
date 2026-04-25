"""Unified news MCP tool."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from ..services.unified_news import fetch_unified_news
from ._mcp_instance import mcp
from .execution_logging import run_logged_operation
from .output_contract import resolve_output_detail
from .schema import CompactFullDetailLiteral

logger = logging.getLogger(__name__)

_NEWS_COMPACT_TOP_LEVEL_KEYS = frozenset(
    {
        "instrument",
        "sources_used",
        "source_details",
        "matching",
        "general_count",
        "related_count",
        "impact_count",
        "upcoming_count",
        "recent_count",
    }
)
_NEWS_BUCKET_KEYS = frozenset(
    {"general_news", "related_news", "impact_news", "upcoming_events", "recent_events"}
)
_NEWS_BUCKET_COUNT_KEYS = {
    "general_news": "general_count",
    "related_news": "related_count",
    "impact_news": "impact_count",
    "upcoming_events": "upcoming_count",
    "recent_events": "recent_count",
}
_NEWS_COMPACT_ITEM_DROP_KEYS = frozenset(
    {
        "provider",
        "priority",
        "relevance_score",
        "importance_score",
        "metadata",
        "url",
        "category",
    }
)


def _news_datetime_utc(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        published_at = value
    else:
        text = str(value or "").strip()
        if not text:
            return None
        try:
            published_at = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None

    if published_at.tzinfo is None:
        return published_at.replace(tzinfo=timezone.utc)
    return published_at.astimezone(timezone.utc)


def _news_time_utc_text(value: datetime) -> str:
    published_at = value.astimezone(timezone.utc).replace(microsecond=0)
    if published_at.second:
        return published_at.strftime("%Y-%m-%d %H:%M:%S UTC")
    return published_at.strftime("%Y-%m-%d %H:%M UTC")


def _news_relative_time_text(value: datetime) -> Optional[str]:
    published_at = value.astimezone(timezone.utc)

    now = datetime.now(timezone.utc)
    delta_seconds = int(round((now - published_at).total_seconds()))
    if delta_seconds < 0:
        return None
    if delta_seconds < 60:
        return "just now"

    seconds = abs(delta_seconds)
    if seconds < 3600:
        amount = max(1, seconds // 60)
        unit = "minute"
    elif seconds < 86400:
        amount = max(1, seconds // 3600)
        unit = "hour"
    elif seconds < 604800:
        amount = max(1, seconds // 86400)
        unit = "day"
    elif seconds < 2592000:
        amount = max(1, seconds // 604800)
        unit = "week"
    else:
        amount = max(1, seconds // 2592000)
        unit = "month"

    plural = "" if amount == 1 else "s"
    return f"{amount} {unit}{plural} ago"


def _news_compact_time_field(
    published_at_value: Any,
    *,
    metadata_relative_time: Optional[str] = None,
) -> tuple[Optional[str], Optional[str]]:
    published_at = _news_datetime_utc(published_at_value)
    if published_at is not None and published_at > datetime.now(timezone.utc):
        return "time_utc", _news_time_utc_text(published_at)
    if metadata_relative_time:
        return "relative_time", metadata_relative_time
    if published_at is None:
        return None, None
    relative_time = _news_relative_time_text(published_at)
    if relative_time:
        return "relative_time", relative_time
    return "time_utc", _news_time_utc_text(published_at)


def _strip_news_compact_item_fields(value: Any, *, bucket_name: Optional[str] = None) -> Any:
    if not isinstance(value, dict):
        return value

    existing_time_utc = value.get("time_utc")
    if isinstance(existing_time_utc, str) and existing_time_utc.strip():
        time_field_name = "time_utc"
        time_field_value = existing_time_utc.strip()
    else:
        existing_relative_time = value.get("relative_time")
        if isinstance(existing_relative_time, str) and existing_relative_time.strip():
            time_field_name = "relative_time"
            time_field_value = existing_relative_time.strip()
        else:
            metadata_relative_time = None
            metadata = value.get("metadata")
            if isinstance(metadata, dict):
                metadata_relative = metadata.get("relative_time")
                if isinstance(metadata_relative, str) and metadata_relative.strip():
                    metadata_relative_time = metadata_relative.strip()
            time_field_name, time_field_value = _news_compact_time_field(
                value.get("published_at"),
                metadata_relative_time=metadata_relative_time,
            )

    out = {}
    title = value.get("title")
    if title is not None:
        out["title"] = title
    source = value.get("source")
    if source not in (None, ""):
        out["source"] = source
    kind = value.get("kind")
    if kind not in (None, ""):
        out["kind"] = kind
    published_at = value.get("published_at")
    if published_at not in (None, ""):
        out["published_at"] = published_at
    if time_field_name and time_field_value:
        out[time_field_name] = time_field_value
    for key, subvalue in value.items():
        key_text = str(key)
        if key_text in {
            "title",
            "source",
            "kind",
            "published_at",
            "relative_time",
            "time_utc",
        }:
            continue
        if key_text in _NEWS_COMPACT_ITEM_DROP_KEYS:
            continue
        if key_text == "summary" and subvalue is None:
            continue
        out[key] = subvalue
    return out


def normalize_news_output(
    result: Dict[str, Any],
    *,
    detail: Any = None,
) -> Dict[str, Any]:
    if not isinstance(result, dict):
        return dict(result)

    detail_mode = resolve_output_detail(detail=detail)
    if detail_mode == "full":
        return dict(result)

    out: Dict[str, Any] = {}
    for key, subvalue in result.items():
        key_text = str(key)
        if key_text in _NEWS_COMPACT_TOP_LEVEL_KEYS:
            continue
        if key_text == "symbol" and subvalue is None:
            continue
        if key_text in _NEWS_BUCKET_KEYS and isinstance(subvalue, list):
            out[key] = [
                _strip_news_compact_item_fields(item, bucket_name=key_text)
                for item in subvalue
            ]
            continue
        out[key] = subvalue
    return out


def _apply_news_limit(result: Dict[str, Any], *, limit: Optional[int]) -> Dict[str, Any]:
    if limit is None:
        return result
    out = dict(result)
    for key in _NEWS_BUCKET_KEYS:
        value = out.get(key)
        if isinstance(value, list) and len(value) > limit:
            out[key] = value[:limit]
            count_key = _NEWS_BUCKET_COUNT_KEYS.get(key)
            if count_key in out:
                out[count_key] = limit
    return out


@mcp.tool()
def news(
    symbol: Optional[str] = None,
    detail: CompactFullDetailLiteral = "compact",
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Fetch important general news and, optionally, symbol-relevant news.

    With no symbol, returns the most important recent general news from all
    available sources.

    With a symbol, returns two buckets:
    - `general_news`: important recent market-wide items.
    - `related_news`: items relevant to the instrument, including direct symbol
      news when available, asset-specific market snapshots, and macro events
      whose text and metadata suggest likely impact on the instrument.
    - `impact_news`: high-importance systemic headlines, such as war or energy
      shocks, that may matter even when they are not direct lexical matches.
    - `upcoming_events`: future economic-calendar items relevant to the
      instrument, surfaced separately so scheduled releases are easy to spot.
    - `recent_events`: the latest relevant economic releases, surfaced
      separately so actual values are easy to scan.

    Matching uses symbol aliases, asset-class terms, MT5 symbol metadata, and a
    lightweight cosine-similarity score over headline/event text.

    Parameters
    ----------
    symbol : str, optional
        Instrument to contextualize the news for, such as `AAPL`, `EURUSD`, or
        `BTCUSD`.
    detail : {"compact", "full"}, optional
        Response detail level. `compact` (default) keeps the current concise
        buckets, while `full` preserves the richer source, matching, and item
        metadata payloads.
    limit : int, optional
        Maximum number of items to return per news bucket. Omit to keep the
        source-selected bucket sizes.

    Returns
    -------
    dict
        Unified response containing:
        - `instrument`: inferred symbol context when `symbol` is provided
        - `general_news`: important recent general news
        - `related_news`: symbol-relevant news and events
        - `impact_news`: high-importance systemic market headlines
        - `upcoming_events`: future scheduled events relevant to the instrument
        - `recent_events`: latest relevant scheduled releases for the instrument
        - `source_details`: per-source candidate and selected counts
        - `matching`: summary of the relevance model
    """

    detail_mode = resolve_output_detail(detail=detail)
    limit_value: Optional[int] = None
    if limit is not None:
        try:
            limit_value = int(limit)
        except (TypeError, ValueError):
            return {"error": "limit must be a positive integer."}
        if limit_value < 1:
            return {"error": "limit must be a positive integer."}

    def _run() -> Dict[str, Any]:
        return _apply_news_limit(
            normalize_news_output(
                fetch_unified_news(symbol=symbol),
                detail=detail_mode,
            ),
            limit=limit_value,
        )

    return run_logged_operation(
        logger,
        operation="news",
        symbol=symbol,
        detail=detail_mode,
        limit=limit_value,
        func=_run,
    )
