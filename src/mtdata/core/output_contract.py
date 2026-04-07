from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from .runtime_metadata import build_runtime_timezone_meta

_VERBOSE_ONLY_KEYS = frozenset({"meta", "diagnostics", "debug", "debug_info"})
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
_NEWS_BUCKET_KEYS = frozenset({"general_news", "related_news", "impact_news", "upcoming_events", "recent_events"})
_NEWS_COMPACT_ITEM_DROP_KEYS = frozenset(
    {
        "provider",
        "priority",
        "relevance_score",
        "importance_score",
        "metadata",
        "url",
        "source",
        "category",
    }
)


def _strip_verbose_only_fields(value: Any) -> Any:
    if isinstance(value, dict):
        out = {}
        for key, subvalue in value.items():
            if str(key) in _VERBOSE_ONLY_KEYS:
                continue
            out[key] = _strip_verbose_only_fields(subvalue)
        return out
    if isinstance(value, list):
        return [_strip_verbose_only_fields(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_strip_verbose_only_fields(item) for item in value)
    return value


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
    if time_field_name and time_field_value:
        out[time_field_name] = time_field_value
    for key, subvalue in value.items():
        key_text = str(key)
        if key_text in {"title", "published_at", "relative_time", "time_utc"}:
            continue
        if key_text in _NEWS_COMPACT_ITEM_DROP_KEYS:
            continue
        if key_text == "kind" and str(bucket_name or "").strip() in {"upcoming_events", "recent_events"}:
            continue
        if key_text == "kind" and str(subvalue).strip().lower() == "headline":
            continue
        if key_text == "summary" and subvalue is None:
            continue
        out[key] = subvalue
    return out


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


def _strip_news_compact_fields(result: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for key, subvalue in result.items():
        key_text = str(key)
        if key_text in _NEWS_COMPACT_TOP_LEVEL_KEYS:
            continue
        if key_text == "symbol" and subvalue is None:
            continue
        if key_text in _NEWS_BUCKET_KEYS and isinstance(subvalue, list):
            out[key] = [_strip_news_compact_item_fields(item, bucket_name=key_text) for item in subvalue]
            continue
        out[key] = subvalue
    return out


def ensure_common_meta(
    result: Any,
    *,
    tool_name: Optional[str] = None,
    mt5_config: Any = None,
) -> Any:
    """Attach the shared output contract metadata without interface-specific wrappers."""
    if not isinstance(result, dict):
        return result

    out = dict(result)
    out.pop("cli_meta", None)

    meta_in = out.get("meta")
    meta = dict(meta_in) if isinstance(meta_in, dict) else {}

    normalized_tool = str(tool_name or "").strip()
    if normalized_tool and not str(meta.get("tool") or "").strip():
        meta["tool"] = normalized_tool

    runtime_in = meta.get("runtime")
    runtime = dict(runtime_in) if isinstance(runtime_in, dict) else {}
    if not isinstance(runtime.get("timezone"), dict):
        runtime["timezone"] = build_runtime_timezone_meta(
            out,
            mt5_config=mt5_config,
            include_local=False,
            include_now=False,
        )
    if runtime:
        meta["runtime"] = runtime

    if meta:
        out["meta"] = meta
    return out


def apply_output_verbosity(
    result: Any,
    *,
    verbose: bool = False,
    tool_name: Optional[str] = None,
    mt5_config: Any = None,
) -> Any:
    """Normalize shared verbose-only output sections across transports."""
    if not isinstance(result, dict):
        return result

    out = dict(result)
    out.pop("cli_meta", None)

    if not verbose:
        compact = _strip_verbose_only_fields(out)
        if str(tool_name or "").strip().lower() == "news":
            return _strip_news_compact_fields(compact)
        return compact

    return ensure_common_meta(
        out,
        tool_name=tool_name,
        mt5_config=mt5_config,
    )
