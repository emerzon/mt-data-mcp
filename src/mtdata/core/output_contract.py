from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Optional

from ..shared.schema import CANONICAL_OUTPUT_DETAIL_ALIASES
from ..utils.utils import _UNPARSED_BOOL, _parse_bool_like
from .runtime_metadata import build_runtime_timezone_meta

_MISSING = object()
_VERBOSE_ONLY_KEYS = frozenset(
    {
        "meta",
        "diagnostics",
        "debug",
        "debug_info",
        "collection_kind",
        "collection_contract_version",
    }
)


@dataclass(frozen=True)
class OutputContractState:
    """Resolved shared output contract state."""

    detail: str

    @property
    def shape_detail(self) -> str:
        return normalize_output_verbosity_detail(self.detail)

    @property
    def verbose(self) -> bool:
        return self.shape_detail == "full"

    @property
    def transport_verbose(self) -> bool:
        return self.shape_detail == "full"


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


def _iter_verbosity_sources(source: Any) -> Iterator[Any]:
    yield source
    if not isinstance(source, dict):
        return
    for candidate in source.values():
        if any(hasattr(candidate, field) for field in ("verbose", "detail")):
            yield candidate


def _read_verbosity_field(source: Any, field: str) -> Any:
    if isinstance(source, dict):
        return source.get(field, _MISSING)
    return getattr(source, field, _MISSING)


def _coerce_optional_verbose_flag(value: Any) -> Optional[bool]:
    if value is _MISSING or value is None:
        return None
    parsed = _parse_bool_like(value, allow_none=True)
    if parsed is _UNPARSED_BOOL:
        return bool(value)
    if parsed is None:
        return None
    return bool(parsed)


def _resolve_requested_detail_value(source: Any, *, detail: Any = _MISSING) -> Any:
    if detail is not _MISSING and detail is not None:
        return detail
    for candidate in _iter_verbosity_sources(source):
        detail_value = _read_verbosity_field(candidate, "detail")
        if detail_value is _MISSING or detail_value is None:
            continue
        return detail_value
    for candidate in _iter_verbosity_sources(source):
        verbose_value = _coerce_optional_verbose_flag(_read_verbosity_field(candidate, "verbose"))
        if verbose_value is None:
            continue
        return "full" if verbose_value else "compact"
    return None


def _normalize_detail_token(value: Any, *, default: str) -> str:
    return str(default if value is None else value).strip().lower()


def _normalize_detail_aliases(aliases: Optional[Mapping[str, str]]) -> dict[str, str]:
    if not aliases:
        return {}

    normalized: dict[str, str] = dict(CANONICAL_OUTPUT_DETAIL_ALIASES)
    for key, value in aliases.items():
        normalized[_normalize_detail_token(key, default="")] = _normalize_detail_token(
            value,
            default="",
        )
    return normalized


def _iter_contract_sources(
    source: Any,
    *,
    detail: Any = _MISSING,
    verbose: Any = _MISSING,
) -> tuple[Any, ...]:
    explicit: dict[str, Any] = {}
    if detail is not _MISSING:
        explicit["detail"] = detail
    if verbose is not _MISSING:
        explicit["verbose"] = verbose

    candidates: list[Any] = []
    if explicit:
        candidates.append(explicit)
    candidates.extend(_iter_verbosity_sources(source))
    return tuple(candidates)


def normalize_output_detail(
    value: Any,
    *,
    default: str = "compact",
    aliases: Optional[Mapping[str, str]] = None,
) -> str:
    """Normalize detail-like values while preserving tool-specific legacy modes."""
    normalized = _normalize_detail_token(value, default=default)
    normalized_aliases = _normalize_detail_aliases(aliases)
    if not normalized_aliases:
        return normalized
    return normalized_aliases.get(normalized, normalized)


def normalize_output_verbosity_detail(value: Any, *, default: str = "compact") -> str:
    """Resolve the shared compact/full verbosity contract."""
    normalized = normalize_output_detail(value, default=default)
    return "full" if normalized == "full" else "compact"


def resolve_output_detail(
    *,
    detail: Any = None,
    default: str = "compact",
) -> str:
    """Resolve the shared compact/full detail mode."""
    return normalize_output_verbosity_detail(detail, default=default)


def resolve_output_contract(
    source: Any = _MISSING,
    *,
    detail: Any = _MISSING,
    verbose: Any = _MISSING,
    default_detail: str = "compact",
    aliases: Optional[Mapping[str, str]] = None,
) -> OutputContractState:
    """Resolve shared detail state."""
    if detail is not _MISSING and detail is not None:
        requested_detail_value = detail
    elif verbose is not _MISSING:
        verbose_value = _coerce_optional_verbose_flag(verbose)
        requested_detail_value = "full" if verbose_value is True else "compact"
    else:
        requested_detail_value = _resolve_requested_detail_value(source, detail=detail)
    return OutputContractState(
        detail=normalize_output_detail(
            requested_detail_value,
            default=default_detail,
            aliases=aliases,
        ),
    )


def resolve_requested_output_verbosity(source: Any, *, default: bool = False) -> bool:
    """Resolve whether the shared output contract should keep full detail."""
    return resolve_output_contract(source, default_detail="full" if default else "compact").verbose


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


def attach_collection_contract(
    result: Any,
    *,
    collection_kind: str,
    rows: Any = None,
    series: Any = None,
    groups: Any = None,
    include_contract_meta: bool = True,
) -> Any:
    """Add normalized collection fields while preserving legacy payload shape."""
    if not isinstance(result, dict) or result.get("error"):
        return result

    def _already_present(collection: Any) -> bool:
        if collection is None:
            return False
        data = out.get("data")
        if data == collection:
            return True
        if isinstance(data, dict):
            table = data.get("table")
            if isinstance(table, dict) and table.get("rows") == collection:
                return True
            if data.get("items") == collection:
                return True
        return False

    out = dict(result)
    canonical_source = None
    if include_contract_meta:
        out.setdefault("collection_kind", str(collection_kind))
        out.setdefault("collection_contract_version", "collection.v1")
    if rows is not None and (include_contract_meta or not _already_present(rows)):
        out.setdefault("rows", rows)
        canonical_source = canonical_source or "rows"
    if series is not None and (include_contract_meta or not _already_present(series)):
        out.setdefault("series", series)
        canonical_source = canonical_source or "series"
    if groups is not None:
        out.setdefault("groups", groups)
        canonical_source = canonical_source or "groups"
    if include_contract_meta and canonical_source:
        out.setdefault("canonical_source", canonical_source)
    return out


def apply_output_verbosity(
    result: Any,
    *,
    detail: str = "compact",
    tool_name: Optional[str] = None,
    mt5_config: Any = None,
) -> Any:
    """Normalize shared detail-only output sections across transports."""
    if not isinstance(result, dict):
        return result

    out = dict(result)
    out.pop("cli_meta", None)

    if resolve_output_detail(detail=detail) != "full":
        return _strip_verbose_only_fields(out)

    return ensure_common_meta(
        out,
        tool_name=tool_name,
        mt5_config=mt5_config,
    )
