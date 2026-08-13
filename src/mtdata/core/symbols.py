
import logging
import math
import re
import time
from difflib import SequenceMatcher
from typing import Annotated, Any, Dict, List, Literal, Optional

from pydantic import Field

from ..shared.constants import (
    DEFAULT_ROW_LIMIT,
    TIMEFRAME_MAP,
    TIMEFRAME_SECONDS,
)
from ..shared.market_units import forex_points_per_pip
from ..shared.schema import DetailLiteral, TimeframeLiteral
from ..shared.symbols import FOREX_CURRENCY_CODES as _FOREX_CURRENCY_CODES
from ..shared.validators import invalid_timeframe_error
from ..utils.freshness import (
    QUOTE_STALE_SECONDS,
    closed_session_context,
    format_age_seconds,
    format_freshness_label,
)
from ..utils.market_metadata import (
    FRESHNESS_ANCHOR_WALL_CLOCK,
    FRESHNESS_METRIC_LAST_COMPLETED_BAR_AGE,
    TICK_VOLUME_SEMANTICS,
    build_tick_freshness_context,
)
from ..utils.mt5 import (
    MT5ConnectionError,
    _mt5_copy_rates_from_pos,
    _symbol_ready_guard,
    account_currency_from_gateway,
    ensure_mt5_connection_or_raise,
    mt5,
    resolve_broker_symbol_name,
)
from ..utils.mt5_enums import decode_mt5_bitmask_labels, decode_mt5_enum_label
from ..utils.quote import (
    compute_spread_metrics,
    resolve_quote_tick,
    tick_epoch,
    tick_value,
)
from ..utils.symbol import (
    _extract_group_path as _extract_group_path_util,
)
from ..utils.symbol import (
    _normalize_group_path_query,
    symbol_suggestions_from_gateway,
)
from ..utils.time import (
    _format_time_explicit,
    _format_time_explicit_local,
    _resolve_client_tz,
    bar_close_epoch,
)
from ..utils.utils import (
    _normalize_limit,
    _table_from_rows,
)
from ._mcp_instance import mcp
from .error_envelope import build_error_payload
from .execution_logging import run_logged_operation
from .mt5_gateway import create_mt5_gateway
from .output_contract import (
    attach_collection_contract,
    build_pagination_meta,
    normalize_output_detail,
    normalize_output_verbosity_detail,
    resolve_output_contract,
)
from .runtime_metadata import build_mt5_source_provenance

logger = logging.getLogger(__name__)


def _clean_broker_text(value: Any) -> Any:
    """Replace invalid Unicode surrogate code points in broker metadata."""
    if not isinstance(value, str):
        return value
    return re.sub(r"[\ud800-\udfff]", "\ufffd", value)
_MARKET_SCAN_STALE_BAR_SECONDS = 7 * 24 * 60 * 60
_MARKET_SCAN_STALE_QUOTE_SECONDS = QUOTE_STALE_SECONDS
_TOP_MARKETS_MAX_CANDIDATES = 250
_MARKET_SCAN_MAX_CANDIDATES = _TOP_MARKETS_MAX_CANDIDATES
_FOREX_SEARCH_PAIR_PRIORITY = {
    pair: idx
    for idx, pair in enumerate(
        (
            "EURUSD",
            "GBPUSD",
            "USDJPY",
            "USDCHF",
            "AUDUSD",
            "USDCAD",
            "NZDUSD",
            "EURGBP",
            "EURJPY",
            "EURCHF",
            "EURAUD",
            "EURCAD",
            "GBPJPY",
            "GBPCHF",
        )
    )
}
_SYMBOL_DEFAULT_CATEGORY_PRIORITY = {
    "indices": 2,
    "commodities": 3,
    "crypto": 4,
    "stocks": 5,
    "etfs": 6,
    "bonds": 7,
    "other": 8,
}


def _case_insensitive_sort_key(value: Any) -> tuple[str, str]:
    text = str(value or "").strip()
    return text.casefold(), text


def _normalize_symbol_search_term(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    pair_match = re.fullmatch(r"([A-Za-z]{2,5})\s*/\s*([A-Za-z]{2,5})", text)
    if pair_match:
        return f"{pair_match.group(1)}{pair_match.group(2)}".upper()
    return text


def _nonempty_symbol_string(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _client_timezone_label(client_tz: Any) -> str:
    if client_tz is None:
        return "UTC"
    return (
        getattr(client_tz, "key", None)
        or getattr(client_tz, "zone", None)
        or str(client_tz)
    )


_SYMBOL_DESCRIBE_PRICE_FIELDS = frozenset(
    {
        "bidlow",
        "bidhigh",
        "asklow",
        "askhigh",
        "session_open",
        "session_close",
    }
)

_SYMBOL_DESCRIBE_FIELDS: tuple[str, ...] = (
    "name",
    "description",
    "path",
    "bank",
    "currency_base",
    "currency_profit",
    "currency_margin",
    "digits",
    "point",
    "bidlow",
    "bidhigh",
    "asklow",
    "askhigh",
    "price_change",
    "session_open",
    "session_close",
    "price_change_pct",
    "trade_mode",
    "trade_exemode",
    "trade_calc_mode",
    "order_mode",
    "expiration_mode",
    "filling_mode",
    "trade_contract_size",
    "trade_tick_size",
    "trade_tick_value",
    "trade_tick_value_profit",
    "trade_tick_value_loss",
    "margin_initial",
    "margin_maintenance",
    "trade_stops_level",
    "trade_freeze_level",
    "volume_min",
    "volume_max",
    "volume_step",
    "volume_limit",
    "swap_mode",
    "swap_long",
    "swap_short",
    "swap_rollover3days",
    "spread_float",
    "ticks_bookdepth",
    "time",
    "select",
)

_SYMBOL_DESCRIBE_COMPACT_DIRECT_FIELDS: tuple[str, ...] = (
    "name",
    "description",
    "currency_base",
    "currency_base_inferred",
    "currency_base_warning",
    "currency_profit",
    "time",
    "quote_status",
    "quote_source",
    "quote_source_state",
    "quote_source_conflict",
    "data_stale",
    "data_age_seconds",
    "stale_after_seconds",
    "freshness",
    "freshness_state",
    "freshness_reason",
    "usable_for_live_trading",
    "usable_for_live_trading_basis",
    "timestamp_ahead_of_wall_clock",
    "timestamp_in_future",
    "timestamp_skew_seconds",
    "timestamp_skew_tolerance_seconds",
    "timestamp_warning",
    "market_status",
    "market_status_reason",
    "note",
    "warning",
    "price_change_pct",
    "price_change_pct_unit",
    "price_change_basis",
    "price_change_period",
    "digits",
    "point",
    "trade_contract_size",
    "trade_tick_size",
    "trade_tick_value",
    "trade_stops_level",
    "trade_freeze_level",
    "volume_min",
    "volume_max",
    "volume_step",
    "volume_limit",
    "trade_mode_label",
    "trade_exemode_label",
    "trade_calc_mode_label",
    "order_mode_labels",
    "filling_mode_labels",
    "spread_is_floating",
    "swap_mode_label",
    "swap_long",
    "swap_short",
)

_SYMBOL_DESCRIBE_SUMMARY_DIRECT_FIELDS: tuple[str, ...] = (
    "name",
    "description",
    "currency_base",
    "currency_base_inferred",
    "currency_base_warning",
    "currency_profit",
    "time",
    "freshness",
    "freshness_state",
    "freshness_reason",
    "usable_for_live_trading",
    "quote_source",
    "quote_source_state",
    "market_status",
    "market_status_reason",
    "note",
    "warning",
    "price_change_pct",
    "price_change_pct_unit",
    "price_change_basis",
    "price_change_period",
    "trade_mode_label",
    "order_mode_labels",
)

_COMMON_CRYPTO_BASES = (
    "BTC",
    "ETH",
    "SOL",
    "XRP",
    "LTC",
    "BCH",
    "ADA",
    "DOT",
    "DOGE",
    "BNB",
    "AVAX",
    "LINK",
    "XLM",
    "TRX",
    "UNI",
    "USDC",
    "USDT",
)

_SYMBOL_SEARCH_MODES = frozenset(
    {"auto", "name", "description", "group", "exact", "all"}
)

_SYMBOL_SEARCH_FIELDS: Dict[str, List[str]] = {
    "auto": ["symbol", "description", "group"],
    "name": ["symbol"],
    "description": ["description"],
    "group": ["group"],
    "exact": ["symbol"],
    "all": ["symbol", "description", "group"],
}

_SYMBOL_SEARCH_REASON_RANK: Dict[str, int] = {
    "exact_name": 0,
    "name_prefix": 1,
    "name_contains": 2,
    "description_contains": 3,
    "group_contains": 4,
    "matched": 9,
}

_SYMBOL_CATEGORY_ALIASES = {
    "fx": "forex",
    "forex": "forex",
    "crypto": "crypto",
    "cryptos": "crypto",
    "index": "indices",
    "indices": "indices",
    "commodity": "commodities",
    "commodities": "commodities",
    "stock": "stocks",
    "stocks": "stocks",
    "equity": "stocks",
    "equities": "stocks",
    "bond": "bonds",
    "bonds": "bonds",
    "etf": "etfs",
    "etfs": "etfs",
}


def _normalize_symbol_category_filter(value: Optional[str]) -> Optional[str]:
    text = str(value or "").strip().lower().replace("-", "_")
    if not text:
        return None
    return _SYMBOL_CATEGORY_ALIASES.get(text)


def _symbol_name_letters(symbol: Any) -> str:
    return re.sub(r"[^A-Z]", "", str(getattr(symbol, "name", "") or "").upper())


def _symbol_forex_pair(symbol: Any) -> Optional[str]:
    pair = _symbol_name_letters(symbol)[:6]
    if (
        len(pair) == 6
        and pair[:3] in _FOREX_CURRENCY_CODES
        and pair[3:] in _FOREX_CURRENCY_CODES
    ):
        return pair
    return None


def _symbol_crypto_match(symbol: Any, text: str) -> bool:
    tokens = set(re.findall(r"[a-z0-9]+", text.casefold()))
    if tokens.intersection(base.casefold() for base in _COMMON_CRYPTO_BASES):
        return True
    if tokens.intersection({"crypto", "cryptos", "cryptocurrency", "cryptocurrencies"}):
        return True

    letters = _symbol_name_letters(symbol)
    quote_tokens = set(_FOREX_CURRENCY_CODES).union({"USDT", "USDC"})
    for base in _COMMON_CRYPTO_BASES:
        base_text = str(base or "").upper()
        if not base_text or not letters.startswith(base_text):
            continue
        suffix = letters[len(base_text):]
        if suffix in quote_tokens:
            return True
    return False


def _symbol_category(symbol: Any) -> str:
    name = str(getattr(symbol, "name", "") or "")
    path = str(_extract_group_path_util(symbol) or "")
    description = str(getattr(symbol, "description", "") or "")
    text = f"{name} {path} {description}".casefold()
    pair_prefix = _symbol_forex_pair(symbol)

    if pair_prefix or "forex" in path.casefold():
        return "forex"
    if any(token in text for token in ("bond", "treasury", "bund", "gilt")):
        return "bonds"
    if "etf" in text:
        return "etfs"
    if any(token in text for token in ("stock", "stocks", "share", "shares", "equity")):
        return "stocks"
    if _symbol_crypto_match(symbol, text):
        return "crypto"
    if any(token in text for token in ("index", "indices", "nasdaq", "dow", "dax")):
        return "indices"
    if any(
        token in text
        for token in (
            "commodity",
            "commodities",
            "metal",
            "metals",
            "energy",
            "energies",
            "gold",
            "silver",
            "oil",
            "brent",
            "copper",
            "platinum",
            "palladium",
            "xau",
            "xag",
            "xpt",
            "xpd",
            "xcu",
        )
    ):
        return "commodities"
    return "other"


def _symbol_group_matches(symbol: Any, group_filter: Optional[str]) -> bool:
    if not group_filter:
        return True
    group_path = _normalize_group_path_query(_extract_group_path_util(symbol))
    return group_filter.casefold() in group_path.casefold()


def _symbol_currency_matches(symbol: Any, currency_filter: Optional[str]) -> bool:
    if not currency_filter:
        return True
    target = currency_filter.upper()
    for attr in ("currency_base", "currency_profit", "currency_margin"):
        value = str(getattr(symbol, attr, "") or "").upper()
        if value == target:
            return True
    return target in _symbol_name_letters(symbol)


def _symbol_search_match_reason(symbol: Any, search_term: str, search_mode: str) -> str:
    query = search_term.casefold()
    name = str(getattr(symbol, "name", "") or "")
    description = str(getattr(symbol, "description", "") or "")
    group = str(_extract_group_path_util(symbol) or "")
    name_folded = name.casefold()

    if search_mode == "exact":
        return "exact_name"
    if search_mode in {"auto", "all", "name"} and name_folded == query:
        return "exact_name"
    if search_mode in {"auto", "all", "name"}:
        if name_folded.startswith(query):
            return "name_prefix"
        if query in name_folded:
            return "name_contains"
    if search_mode in {"auto", "all", "description"} and query in description.casefold():
        return "description_contains"
    if search_mode in {"auto", "all", "group"} and query in group.casefold():
        return "group_contains"
    return "matched"


def _symbol_search_sort_key(
    symbol: Any,
    search_term: str,
    search_mode: str,
) -> tuple[int, int, str, str]:
    reason = _symbol_search_match_reason(symbol, search_term, search_mode)
    forex_rank = _symbol_search_forex_rank(symbol, search_term)
    reason_rank = _SYMBOL_SEARCH_REASON_RANK.get(reason, 9)
    if forex_rank < 100:
        reason_rank = min(reason_rank, _SYMBOL_SEARCH_REASON_RANK["name_prefix"])
    return (
        reason_rank,
        forex_rank,
        *_case_insensitive_sort_key(getattr(symbol, "name", "")),
    )


def _symbol_search_forex_rank(symbol: Any, search_term: str) -> int:
    query = re.sub(r"[^A-Z]", "", str(search_term or "").upper())
    if len(query) != 3 or query not in _FOREX_CURRENCY_CODES:
        return 100
    pair = _symbol_forex_pair(symbol)
    if not pair or query not in (pair[:3], pair[3:]):
        return 100
    return _FOREX_SEARCH_PAIR_PRIORITY.get(pair, 50)


def _symbol_default_list_sort_key(symbol: Any) -> tuple[int, int, str, str]:
    """Rank a no-filter symbol overview without fetching live market data."""
    pair = _symbol_forex_pair(symbol)
    if pair in _FOREX_SEARCH_PAIR_PRIORITY:
        return 0, _FOREX_SEARCH_PAIR_PRIORITY[pair], *_case_insensitive_sort_key(
            getattr(symbol, "name", "")
        )
    if pair:
        return 1, 0, *_case_insensitive_sort_key(getattr(symbol, "name", ""))
    category_rank = _SYMBOL_DEFAULT_CATEGORY_PRIORITY.get(_symbol_category(symbol), 9)
    return category_rank, 0, *_case_insensitive_sort_key(getattr(symbol, "name", ""))


def _symbol_top_match(
    rows: List[Dict[str, Any]], search_term: str
) -> Optional[Dict[str, Any]]:
    if not rows or not isinstance(rows[0], dict):
        return None
    row = rows[0]
    reason = row.get("match_reason")
    if reason not in {"exact_name", "name_prefix"}:
        return None
    if reason == "name_prefix":
        query = re.sub(r"[^A-Z]", "", str(search_term or "").upper())

        def _semantic_rank(candidate: Dict[str, Any]) -> int:
            symbol_name = re.sub(
                r"[^A-Z]", "", str(candidate.get("symbol") or "").upper()
            )
            if query in _FOREX_CURRENCY_CODES and len(symbol_name) >= 6:
                return _FOREX_SEARCH_PAIR_PRIORITY.get(symbol_name[:6], 50)
            return 0

        first_rank = _semantic_rank(row)
        if any(
            candidate.get("match_reason") == reason
            and _semantic_rank(candidate) == first_rank
            for candidate in rows[1:]
        ):
            return None
    out = {
        "symbol": row.get("symbol"),
        "match_reason": reason,
    }
    group = row.get("group")
    if group not in (None, ""):
        out["group"] = group
    return out


def _symbol_search_context(search_term: str, search_mode: str) -> Dict[str, Any]:
    context: Dict[str, Any] = {
        "term": search_term,
        "mode": search_mode,
        "fields": _SYMBOL_SEARCH_FIELDS.get(
            search_mode,
            ["symbol", "description", "group"],
        ),
        "match": (
            "case_insensitive_equality"
            if search_mode == "exact"
            else "case_insensitive_substring"
        ),
    }
    if search_mode in {"auto", "all"}:
        context["ranking"] = [
            "exact_name",
            "name_prefix",
            "name_contains",
            "description_contains",
            "group_contains",
        ]
    return context


def _symbol_search_normalized_from(
    raw_search_term: Optional[str],
    search_term: Optional[str],
) -> Optional[str]:
    raw = str(raw_search_term or "").strip()
    normalized = str(search_term or "").strip()
    if raw and normalized and raw != normalized:
        return raw
    return None


def _symbol_search_context_for_request(
    search_term: str,
    search_mode: str,
    *,
    raw_search_term: Optional[str],
) -> Dict[str, Any]:
    context = _symbol_search_context(search_term, search_mode)
    normalized_from = _symbol_search_normalized_from(raw_search_term, search_term)
    if normalized_from:
        context["normalized_from"] = normalized_from
    return context


def _symbol_search_suggestions(
    all_symbols: List[Any],
    search_term: str,
    *,
    limit: int = 5,
) -> List[Dict[str, Any]]:
    query = str(search_term or "").strip().casefold()
    if not query:
        return []
    scored: List[tuple[float, str, Any]] = []
    for symbol in all_symbols:
        name = str(getattr(symbol, "name", "") or "")
        if not name:
            continue
        name_folded = name.casefold()
        if query in name_folded:
            score = 1.0
        else:
            score = SequenceMatcher(None, query, name_folded).ratio()
        if score >= 0.55:
            scored.append((score, name, symbol))
    scored.sort(key=lambda item: (-item[0], *_case_insensitive_sort_key(item[1])))
    return [
        _symbol_suggestion_from_info(symbol)
        for _, _, symbol in scored[: max(1, int(limit))]
    ]


def _symbols_empty_search_context(
    all_symbols: List[Any],
    search_term: str,
    search_mode: str,
) -> Dict[str, Any]:
    context: Dict[str, Any] = {
        "message": (
            f"No symbols matched '{search_term}'. "
            "Try search_mode=all for broader matching or check spelling."
        ),
        "universe_size": len(all_symbols),
    }
    suggestions = _symbol_search_suggestions(all_symbols, search_term)
    if suggestions:
        context["suggestions"] = suggestions
    if search_mode == "all":
        context["message"] = (
            f"No symbols matched '{search_term}'. "
            "Check spelling or inspect broker groups with list_mode=groups."
        )
    return context


def _symbols_from_groups(
    groups: Dict[str, List[Any]],
    group_names: List[str],
) -> List[Any]:
    matched: List[Any] = []
    for group_name in group_names:
        matched.extend(groups[group_name])
    return matched


def _match_symbols_for_search(
    all_symbols: List[Any],
    search_term: str,
    search_mode: str,
) -> List[Any]:
    search_upper = search_term.upper()
    groups: Dict[str, List[Any]] = {}
    symbol_name_matches: List[Any] = []
    description_matches: List[Any] = []
    all_field_matches: List[Any] = []

    for symbol in all_symbols:
        group_path = _extract_group_path_util(symbol)
        groups.setdefault(group_path, []).append(symbol)

        symbol_name = str(getattr(symbol, "name", "") or "")
        description = str(getattr(symbol, "description", "") or "")
        name_hit = search_upper in symbol_name.upper()
        description_hit = search_upper in description.upper()
        group_hit = search_upper in str(group_path or "").upper()

        if search_mode == "exact":
            if symbol_name.upper() == search_upper:
                symbol_name_matches.append(symbol)
        elif name_hit:
            symbol_name_matches.append(symbol)
        if description_hit:
            description_matches.append(symbol)
        if name_hit or description_hit or group_hit:
            all_field_matches.append(symbol)

    matching_groups = [
        group_name
        for group_name in groups.keys()
        if search_upper in group_name.upper()
    ]

    if search_mode in {"exact", "name"}:
        return symbol_name_matches
    if search_mode == "description":
        return description_matches
    if search_mode == "group":
        return _symbols_from_groups(groups, matching_groups)
    if search_mode in {"auto", "all"}:
        return all_field_matches


_COMMON_QUOTE_CURRENCIES = (
    "USD",
    "USDT",
    "USDC",
    "EUR",
    "GBP",
    "JPY",
    "CHF",
    "AUD",
    "CAD",
    "NZD",
)


def _copy_symbol_describe_field(
    out: Dict[str, Any],
    source: Dict[str, Any],
    field: str,
) -> bool:
    if field not in source:
        return False
    value = source.get(field)
    if value is None:
        return False
    if isinstance(value, str) and value == "":
        return False
    if isinstance(value, list) and not value:
        return False
    out[field] = value
    return True


def _normalize_spread_float_field(payload: Dict[str, Any]) -> None:
    if "spread_float" not in payload:
        return
    value = payload.pop("spread_float")
    if isinstance(value, bool):
        payload["spread_is_floating"] = value
    elif value is not None:
        payload["spread_is_floating"] = bool(value)


def _infer_symbol_base_from_name(symbol_name: Any, quote_currency: Any) -> Optional[str]:
    name = re.sub(r"[^A-Z0-9]", "", str(symbol_name or "").upper())
    quote = str(quote_currency or "").strip().upper()
    if not name or quote not in _COMMON_QUOTE_CURRENCIES:
        return None
    if not name.endswith(quote):
        return None
    base = name[: -len(quote)]
    for crypto_base in _COMMON_CRYPTO_BASES:
        if base == crypto_base or base.endswith(crypto_base):
            return crypto_base
    return None


def _add_symbol_currency_diagnostics(symbol_data: Dict[str, Any]) -> None:
    currency_base = str(symbol_data.get("currency_base") or "").strip().upper()
    currency_profit = str(symbol_data.get("currency_profit") or "").strip().upper()
    if not currency_base or not currency_profit or currency_base != currency_profit:
        return
    inferred_base = _infer_symbol_base_from_name(
        symbol_data.get("name") or symbol_data.get("symbol"),
        currency_profit,
    )
    if not inferred_base or inferred_base == currency_base:
        return
    symbol_data["currency_base_inferred"] = inferred_base
    symbol_data["currency_base_warning"] = (
        "MT5 reports identical currency_base and currency_profit; verify broker metadata."
    )


def _compact_symbol_describe_payload(symbol_data: Dict[str, Any]) -> Dict[str, Any]:
    compact: Dict[str, Any] = {}
    for field in _SYMBOL_DESCRIBE_COMPACT_DIRECT_FIELDS:
        _copy_symbol_describe_field(compact, symbol_data, field)

    raw_margin_fields = {
        "margin_initial": "broker_margin_initial_raw",
        "margin_maintenance": "broker_margin_maintenance_raw",
    }
    for source, target in raw_margin_fields.items():
        if source in symbol_data:
            compact[target] = symbol_data[source]
    if any(target in compact for target in raw_margin_fields.values()):
        compact["margin_fields_note"] = (
            "Raw broker symbol template values, not cash required for an order. "
            "Actual margin depends on account leverage and instrument rules; use "
            "trade_place dry-run for an order-specific margin estimate."
        )

    _apply_symbol_currency_diagnostics(compact)

    if "time_epoch" in symbol_data:
        compact["time_epoch"] = symbol_data["time_epoch"]
    return compact


def _summary_symbol_describe_payload(symbol_data: Dict[str, Any]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    for field in _SYMBOL_DESCRIBE_SUMMARY_DIRECT_FIELDS:
        _copy_symbol_describe_field(summary, symbol_data, field)

    _apply_symbol_currency_diagnostics(summary)
    return summary


def _apply_symbol_currency_diagnostics(payload: Dict[str, Any]) -> None:
    inferred_base = payload.get("currency_base_inferred")
    reported_base = payload.get("currency_base")
    if inferred_base and payload.get("currency_base_warning"):
        payload["currency_base_reported"] = reported_base
        payload["currency_base_source"] = "reported_by_mt5"
        payload["currency_base_inference_source"] = "inferred_from_symbol_name"


def _attach_symbol_currency_anomaly_summary(
    payload: Dict[str, Any],
    *,
    anomalies: List[Dict[str, Any]],
) -> None:
    anomaly_count = len(anomalies)
    if anomaly_count <= 0:
        return
    payload["currency_metadata_anomaly_count"] = int(anomaly_count)
    payload["currency_metadata_anomalies"] = anomalies[:20]
    if anomaly_count > 20:
        payload["currency_metadata_anomalies_truncated"] = True
    named_symbols = ", ".join(
        str(item.get("symbol"))
        for item in anomalies[:5]
        if item.get("symbol")
    )
    payload["warnings"] = [
        f"{int(anomaly_count)} symbol(s) have broker currency metadata that "
        f"conflicts with the symbol name: {named_symbols}."
    ]
    payload["trust"] = "verify_broker_metadata"


def _symbol_session_type(
    *,
    name: Any,
    group: Any = None,
    description: Any = None,
) -> Optional[str]:
    text = " ".join(
        str(value or "").upper()
        for value in (name, group, description)
        if value not in (None, "")
    )
    if any(token in text for token in ("-24", "24HR", "24/5", "24H")):
        return "extended_24h"
    if "." in str(name or "") and any(token in text for token in ("STOCK", "CFD")):
        return "regular"
    return None


def _symbol_suggestion_from_info(symbol_info: Any) -> Dict[str, Any]:
    group = _extract_group_path_util(symbol_info)
    description = getattr(symbol_info, "description", None)
    suggestion: Dict[str, Any] = {
        "symbol": _clean_broker_text(getattr(symbol_info, "name", None)),
        "group": _clean_broker_text(group),
    }
    if description not in (None, ""):
        suggestion["description"] = _clean_broker_text(description)
    session_type = _symbol_session_type(
        name=getattr(symbol_info, "name", None),
        group=group,
        description=description,
    )
    if session_type is not None:
        suggestion["session_type"] = session_type
    return {key: value for key, value in suggestion.items() if value not in (None, "")}


def _symbol_list_optional_attr(symbol_info: Any, attr: str) -> Any:
    try:
        if attr not in dir(symbol_info):
            return None
        value = getattr(symbol_info, attr)
    except Exception:
        return None
    if callable(value) or value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return _clean_broker_text(value)


def _find_symbol_suggestions(
    mt5_gateway: Any,
    query: str,
    *,
    limit: int = 5,
) -> List[Dict[str, Any]]:
    return symbol_suggestions_from_gateway(
        mt5_gateway,
        query,
        limit=limit,
    )


@mcp.tool()
def symbols_list(  # noqa: C901
    search_term: Optional[str] = None,
    limit: Annotated[int, Field(ge=1)] = DEFAULT_ROW_LIMIT,
    offset: int = 0,
    list_mode: Literal["symbols", "groups"] = "symbols",  # type: ignore
    universe: Optional[Literal["visible", "all"]] = None,  # type: ignore
    group: Optional[str] = None,
    currency: Optional[str] = None,
    category: Optional[str] = None,
    search_mode: Literal[  # type: ignore
        "auto",
        "name",
        "description",
        "group",
        "exact",
        "all",
    ] = "auto",
    detail: DetailLiteral = "compact",  # type: ignore
) -> Dict[str, Any]:
    """List symbols or symbol groups.

    Search is case-insensitive. Slashed pairs such as EUR/USD are normalized to
    broker-style concatenated symbols such as EURUSD. Auto mode searches symbol,
    description, and group fields, then ranks exact/prefix/name matches before
    description and group matches.

    Without a search term, omitting universe lists Market Watch symbols. When a
    search term is present, omitting universe searches the broker catalog. An
    explicit universe="visible" or universe="all" is always honored. The
    unfiltered overview ranks FX majors, other FX pairs, then broad asset
    categories before symbol name. Use group, currency, and category to filter
    the resulting symbol set.
    """
    raw_search_term = str(search_term or "").strip() or None
    normalized_search_term = _normalize_symbol_search_term(search_term)
    group_filter = _normalize_group_path_query(group) if group else None
    currency_filter = str(currency or "").strip().upper() or None
    category_filter = _normalize_symbol_category_filter(category)
    detail_mode = normalize_output_detail(detail, default="compact")
    search_mode_value = str(search_mode or "auto").strip().lower()
    universe_value = (
        str(universe).strip().lower() if universe is not None else None
    )
    effective_universe = universe_value or (
        "all" if normalized_search_term else "visible"
    )

    def _run() -> Dict[str, Any]:  # noqa: C901
        try:
            if limit is not None:
                try:
                    if int(limit) <= 0:
                        return {"error": "limit must be a positive integer when provided."}
                except (TypeError, ValueError):
                    return {"error": "limit must be a positive integer when provided."}
            mt5_gateway = create_mt5_gateway(
                adapter=mt5,
                ensure_connection_impl=ensure_mt5_connection_or_raise,
            )
            mt5_gateway.ensure_connection()
            source = build_mt5_source_provenance(mt5_gateway)
            mode = str(list_mode or "symbols").strip().lower()
            if mode not in ("symbols", "groups"):
                return {"error": "list_mode must be 'symbols' or 'groups'."}
            if universe_value is not None and universe_value not in {"visible", "all"}:
                return {"error": "universe must be 'visible' or 'all'."}
            if category and not category_filter:
                return {
                    "error": (
                        "category must be one of forex, crypto, indices, "
                        "commodities, stocks, bonds, or etfs."
                    )
                }
            if search_mode_value not in _SYMBOL_SEARCH_MODES:
                return {
                    "error": (
                        "search_mode must be one of auto, name, description, "
                        "group, exact, or all."
                    )
                }
            if mode == "groups":
                if search_mode_value in {"name", "description"}:
                    return {
                        "error": (
                            "list_mode='groups' searches group paths only; use "
                            "search_mode=auto, group, exact, or all."
                        )
                    }
                return _list_symbol_groups(
                    search_term=normalized_search_term,
                    limit=limit,
                    offset=offset,
                    mt5_gateway=mt5_gateway,
                    detail=detail_mode,
                    universe=effective_universe,
                    group=group_filter,
                    currency=currency_filter,
                    category=category_filter,
                    search_mode=search_mode_value,
                    source=source,
                )

            matched_symbols = []
            search_universe: List[Any] = []
            all_symbols = mt5_gateway.symbols_get()
            if all_symbols is None:
                return {"error": f"Failed to get symbols: {mt5_gateway.last_error()}"}
            all_symbols_list = list(all_symbols)
            broker_symbol_count = len(all_symbols_list)
            visible_count = sum(
                1
                for symbol in all_symbols_list
                if bool(getattr(symbol, "visible", False))
            )
            if normalized_search_term:
                search_universe = all_symbols_list
                matched_symbols = _match_symbols_for_search(
                    search_universe,
                    normalized_search_term,
                    search_mode_value,
                )
            else:
                matched_symbols = all_symbols_list

            if normalized_search_term:
                matched_symbols = sorted(
                    matched_symbols,
                    key=lambda symbol: _symbol_search_sort_key(
                        symbol,
                        normalized_search_term,
                        search_mode_value,
                    ),
                )
            else:
                matched_symbols = sorted(
                    matched_symbols,
                    key=_symbol_default_list_sort_key,
                )
            only_visible = effective_universe == "visible"
            symbol_list = []
            for symbol in matched_symbols:
                if only_visible and not symbol.visible:
                    continue
                if not _symbol_group_matches(symbol, group_filter):
                    continue
                if not _symbol_currency_matches(symbol, currency_filter):
                    continue
                symbol_category = _symbol_category(symbol)
                if category_filter and symbol_category != category_filter:
                    continue
                row = {
                    "symbol": _clean_broker_text(symbol.name),
                    "group": _clean_broker_text(_extract_group_path_util(symbol)),
                    "description": _clean_broker_text(symbol.description),
                    "in_marketwatch": bool(getattr(symbol, "visible", False)),
                    "session_type": _symbol_session_type(
                        name=symbol.name,
                        group=_extract_group_path_util(symbol),
                        description=symbol.description,
                    ),
                }
                if category_filter:
                    row["category"] = symbol_category
                if normalized_search_term:
                    row["match_reason"] = _symbol_search_match_reason(
                        symbol,
                        normalized_search_term,
                        search_mode_value,
                    )
                for attr in (
                    "currency_base",
                    "currency_profit",
                    "digits",
                ):
                    value = _symbol_list_optional_attr(symbol, attr)
                    if value is not None:
                        row[attr] = value
                spread_is_floating = _symbol_list_optional_attr(symbol, "spread_float")
                if spread_is_floating is not None:
                    row["spread_is_floating"] = bool(spread_is_floating)
                _add_symbol_currency_diagnostics(row)
                _apply_symbol_currency_diagnostics(row)
                symbol_list.append(row)

            limit_value = _normalize_limit(limit)
            try:
                offset_value = int(offset or 0)
            except Exception:
                return {"error": "offset must be a non-negative integer."}
            if offset_value < 0:
                return {"error": "offset must be >= 0."}
            total_count = len(symbol_list)
            currency_anomalies = [
                {
                    "symbol": row.get("symbol"),
                    "field": "currency_base",
                    "issue": "reported_base_matches_profit_but_name_implies_different_base",
                    "reported": row.get("currency_base_reported")
                    or row.get("currency_base"),
                    "inferred": row.get("currency_base_inferred"),
                    "currency_profit": row.get("currency_profit"),
                }
                for row in symbol_list
                if row.get("currency_base_warning")
            ]
            filters = {}
            if group_filter:
                filters["group"] = group_filter
            if currency_filter:
                filters["currency"] = currency_filter
            if category_filter:
                filters["category"] = category_filter
            top_match = (
                _symbol_top_match(symbol_list, normalized_search_term)
                if normalized_search_term
                else None
            )
            ambiguous_match = bool(
                normalized_search_term
                and not top_match
                and len(symbol_list) > 1
                and symbol_list[0].get("match_reason") == "name_prefix"
                and symbol_list[1].get("match_reason") == "name_prefix"
            )
            if offset_value:
                symbol_list = symbol_list[offset_value:]
            if limit_value:
                symbol_list = symbol_list[:limit_value]
            if detail_mode == "summary":
                out = {
                    "success": True,
                    "list_mode": "symbols",
                    "count": len(symbol_list),
                    "search_term": normalized_search_term,
                    "search_mode": search_mode_value,
                    "universe": effective_universe,
                }
                if filters:
                    out["filters"] = filters
                sample = []
                for row in symbol_list[:5]:
                    sample.append(
                        {
                            key: row.get(key)
                            for key in (
                                "symbol",
                                "group",
                                "description",
                                "category",
                                "match_reason",
                                "currency_base",
                                "currency_base_reported",
                                "currency_base_inferred",
                                "currency_base_source",
                                "currency_base_inference_source",
                                "currency_base_warning",
                            )
                            if row.get(key) is not None
                        }
                    )
                if sample:
                    out["sample"] = sample
                    out["sample_count"] = len(sample)
                if normalized_search_term:
                    out["search"] = _symbol_search_context_for_request(
                        normalized_search_term,
                        search_mode_value,
                        raw_search_term=raw_search_term,
                    )
                    if top_match:
                        out["top_match"] = top_match
                    elif ambiguous_match:
                        out["ambiguous_match"] = True
                        out["match_candidates"] = [
                            row.get("symbol") for row in symbol_list[:5]
                        ]
                    out["universe_size"] = len(search_universe)
                    if total_count == 0:
                        out.update(
                            _symbols_empty_search_context(
                                search_universe,
                                normalized_search_term,
                                search_mode_value,
                            )
                        )
                elif effective_universe == "visible":
                    out["visible_count"] = visible_count
                    out["broker_symbol_count"] = broker_symbol_count
                    if broker_symbol_count > visible_count:
                        out["note"] = (
                            "Showing visible Market Watch symbols only "
                            f"({visible_count} of {broker_symbol_count}); "
                            "use universe=all or search_term for the broker catalog."
                        )
                if not normalized_search_term:
                    out["sort"] = "market_overview"
                out["pagination"] = build_pagination_meta(
                    total=total_count,
                    returned=len(symbol_list),
                    offset=offset_value,
                    limit=limit_value,
                )
                if currency_filter:
                    out["currency_filter_basis"] = "broker_reported_currency"
                _attach_symbol_currency_anomaly_summary(
                    out,
                    anomalies=currency_anomalies,
                )
                out["source"] = source
                return out
            headers = ["symbol", "group", "description"]
            if category_filter:
                headers.append("category")
            if normalized_search_term:
                headers.append("match_reason")
            for optional_header in (
                "currency_base",
                "currency_base_reported",
                "currency_base_inferred",
                "currency_base_source",
                "currency_base_inference_source",
                "currency_base_warning",
                "currency_profit",
                "digits",
                "spread_is_floating",
            ):
                if any(s.get(optional_header) is not None for s in symbol_list):
                    headers.append(optional_header)
            if any(s.get("session_type") for s in symbol_list):
                headers.append("session_type")
            if detail_mode in {"standard", "full"}:
                headers.append("in_marketwatch")
            rows = [[s.get(header) for header in headers] for s in symbol_list]
            result = _table_from_rows(headers, rows)
            result["universe"] = effective_universe
            if filters:
                result["filters"] = filters
            if normalized_search_term:
                result["search"] = _symbol_search_context_for_request(
                    normalized_search_term,
                    search_mode_value,
                    raw_search_term=raw_search_term,
                )
                if top_match:
                    result["top_match"] = top_match
                elif ambiguous_match:
                    result["ambiguous_match"] = True
                    result["match_candidates"] = [
                        row.get("symbol") for row in symbol_list[:5]
                    ]
                result["universe_size"] = len(search_universe)
                if total_count == 0:
                    result.update(
                        _symbols_empty_search_context(
                            search_universe,
                            normalized_search_term,
                            search_mode_value,
                            )
                        )
            elif effective_universe == "visible":
                result["visible_count"] = visible_count
                result["broker_symbol_count"] = broker_symbol_count
                if broker_symbol_count > visible_count:
                    result["note"] = (
                        "Showing visible Market Watch symbols only "
                        f"({visible_count} of {broker_symbol_count}); "
                        "use universe=all or search_term for the broker catalog."
                    )
            if not normalized_search_term:
                result["sort"] = "market_overview"
            result["pagination"] = build_pagination_meta(
                total=total_count,
                returned=len(symbol_list),
                offset=offset_value,
                limit=limit_value,
            )
            if currency_filter:
                result["currency_filter_basis"] = "broker_reported_currency"
            _attach_symbol_currency_anomaly_summary(
                result,
                anomalies=currency_anomalies,
            )
            result["source"] = source
            return attach_collection_contract(
                result,
                collection_kind="table",
                rows=result.get("data"),
                include_contract_meta=False,
            )
        except MT5ConnectionError as exc:
            return {"error": str(exc)}
        except Exception as exc:
            return {"error": f"Error getting symbols: {str(exc)}"}

    return run_logged_operation(
        logger,
        operation="symbols_list",
        search_term=normalized_search_term,
        limit=limit,
        offset=offset,
        list_mode=list_mode,
        universe=effective_universe,
        group=group_filter,
        currency=currency_filter,
        category=category_filter,
        search_mode=search_mode_value,
        detail=detail_mode,
        func=_run,
    )

def _list_symbol_groups(
    search_term: Optional[str] = None,
    limit: Annotated[int, Field(ge=1)] = DEFAULT_ROW_LIMIT,
    offset: int = 0,
    mt5_gateway: Any = None,
    detail: DetailLiteral = "compact",  # type: ignore
    universe: str = "visible",
    group: Optional[str] = None,
    currency: Optional[str] = None,
    category: Optional[str] = None,
    search_mode: str = "auto",
    source: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """List group paths as a tabular result with a single column: group."""
    try:
        gateway = mt5_gateway or create_mt5_gateway(
            adapter=mt5,
            ensure_connection_impl=ensure_mt5_connection_or_raise,
        )
        # Get all symbols first
        all_symbols = gateway.symbols_get()
        if all_symbols is None:
            return {"error": f"Failed to get symbols: {gateway.last_error()}"}
        
        universe_value = str(universe or "visible").strip().lower()
        group_filter = _normalize_group_path_query(group) if group else None
        currency_filter = str(currency or "").strip().upper() or None
        category_filter = _normalize_symbol_category_filter(category)
        search_mode_value = str(search_mode or "auto").strip().lower()
        filtered_symbols = [
            symbol
            for symbol in list(all_symbols)
            if (universe_value == "all" or bool(getattr(symbol, "visible", False)))
            and _symbol_group_matches(symbol, group_filter)
            and _symbol_currency_matches(symbol, currency_filter)
            and (not category_filter or _symbol_category(symbol) == category_filter)
        ]

        # Collect unique groups and compact discovery metadata.
        groups = {}
        for symbol in filtered_symbols:
            group_path = _extract_group_path_util(symbol)
            if group_path not in groups:
                groups[group_path] = {
                    "symbol_count": 0,
                    "visible_count": 0,
                    "sample_symbols": [],
                }
            group_meta = groups[group_path]
            group_meta["symbol_count"] += 1
            if bool(getattr(symbol, "visible", False)):
                group_meta["visible_count"] += 1
            symbol_name = str(getattr(symbol, "name", "") or "").strip()
            if symbol_name and len(group_meta["sample_symbols"]) < 3:
                group_meta["sample_symbols"].append(symbol_name)
        
        # Filter by search term if provided
        filtered_items = list(groups.items())
        if search_term:
            q = search_term.strip().lower()
            if search_mode_value == "exact":
                filtered_items = [
                    (key, value)
                    for key, value in filtered_items
                    if q == str(key or "").lower()
                ]
            else:
                filtered_items = [
                    (key, value)
                    for key, value in filtered_items
                    if q in str(key or "").lower()
                ]

        group_search_note = (
            f"No group path matches '{search_term}'. Group listing filters by group "
            "path, not symbol name; omit list_mode=groups to search symbols by name."
            if search_term and not filtered_items
            else None
        )

        # Sort groups by count (most symbols first)
        filtered_items.sort(
            key=lambda item: (
                -item[1]["symbol_count"],
                *_case_insensitive_sort_key(item[0]),
            )
        )

        # Apply limit
        limit_value = _normalize_limit(limit)
        try:
            offset_value = int(offset or 0)
        except Exception:
            return {"error": "offset must be a non-negative integer."}
        if offset_value < 0:
            return {"error": "offset must be >= 0."}
        total_count = len(filtered_items)
        if offset_value:
            filtered_items = filtered_items[offset_value:]
        if limit_value:
            filtered_items = filtered_items[:limit_value]

        detail_mode = normalize_output_detail(detail, default="compact")
        if detail_mode == "summary":
            out = {
                "success": True,
                "list_mode": "groups",
                "count": len(filtered_items),
                "search_term": search_term,
                "search_mode": search_mode_value,
                "universe": universe_value,
                "pagination": build_pagination_meta(
                    total=total_count,
                    returned=len(filtered_items),
                    offset=offset_value,
                    limit=limit_value,
                ),
            }
            if group_search_note:
                out["note"] = group_search_note
            filters = {
                key: value
                for key, value in {
                    "group": group_filter,
                    "currency": currency_filter,
                    "category": category_filter,
                }.items()
                if value
            }
            if filters:
                out["filters"] = filters
            out["source"] = source or build_mt5_source_provenance(gateway)
            return out
        rows = [
            [
                name,
                meta["symbol_count"],
                meta["visible_count"],
                meta["sample_symbols"],
            ]
            for name, meta in filtered_items
        ]
        result = _table_from_rows(
            ["group", "symbol_count", "visible_count", "sample_symbols"],
            rows,
        )
        result["list_mode"] = "groups"
        result["universe"] = universe_value
        result["search_mode"] = search_mode_value
        filters = {
            key: value
            for key, value in {
                "group": group_filter,
                "currency": currency_filter,
                "category": category_filter,
            }.items()
            if value
        }
        if filters:
            result["filters"] = filters
        result["source"] = source or build_mt5_source_provenance(gateway)
        result["pagination"] = build_pagination_meta(
            total=total_count,
            returned=len(filtered_items),
            offset=offset_value,
            limit=limit_value,
        )
        if group_search_note:
            result["note"] = group_search_note
        return attach_collection_contract(
            result,
            collection_kind="table",
            rows=result.get("data"),
            include_contract_meta=False,
        )
    except Exception as e:
        return {"error": f"Error getting symbol groups: {str(e)}"}

@mcp.tool()
def symbols_describe(  # noqa: C901
    symbol: str,
    detail: DetailLiteral = "compact",
) -> Dict[str, Any]:
    """Return symbol information as JSON for `symbol`.
    
    Parameters:
    -----------
    symbol : str
        Trading symbol (e.g., "EURUSD")
    detail : str, optional (default="compact")
        Output verbosity level:
        - "summary": Symbol identity, currencies, quote freshness, and session/trade labels
        - "compact": Essential identity, status, volume, and contract fields
        - "standard": Same concise field set as compact for this single-symbol metadata tool
        - "full": Complete metadata including all trading modes, swap details, and session times
    Returns:
    --------
    dict
        Symbol identifier plus requested detail fields
    """
    def _run() -> Dict[str, Any]:  # noqa: C901
        try:
            contract = resolve_output_contract(
                detail=detail,
                default_detail="compact",
            )
            mt5_gateway = create_mt5_gateway(
                adapter=mt5,
                ensure_connection_impl=ensure_mt5_connection_or_raise,
            )
            mt5_gateway.ensure_connection()
            resolved_symbol = resolve_broker_symbol_name(
                symbol,
                gateway=mt5_gateway,
            )
            symbol_info = mt5_gateway.symbol_info(resolved_symbol)
            if symbol_info is None:
                suggestions = _find_symbol_suggestions(mt5_gateway, symbol)
                details: Dict[str, Any] = {
                    "symbol": symbol,
                    "search_hint": f"Use symbols_list(search_term='{symbol}') to browse matching broker symbols.",
                }
                details["did_you_mean"] = suggestions
                return build_error_payload(
                    f"Symbol '{symbol}' not found in MT5 terminal.",
                    code="symbol_not_found",
                    operation="symbols_describe",
                    details=details,
                )

            enum_specs = {
                "trade_mode": {"prefixes": ("SYMBOL_TRADE_MODE_",), "bitmask": False},
                "trade_exemode": {"prefixes": ("SYMBOL_TRADE_EXECUTION_",), "bitmask": False},
                "trade_calc_mode": {"prefixes": ("SYMBOL_CALC_MODE_",), "bitmask": False},
                "swap_mode": {"prefixes": ("SYMBOL_SWAP_MODE_",), "bitmask": False},
                "filling_mode": {"prefixes": ("ORDER_FILLING_", "SYMBOL_FILLING_"), "bitmask": True},
                "expiration_mode": {"prefixes": ("SYMBOL_EXPIRATION_",), "bitmask": True},
                "order_mode": {"prefixes": ("SYMBOL_ORDER_",), "bitmask": True},
            }

            client_tz = _resolve_client_tz()
            if client_tz is None:
                time_formatter = _format_time_explicit
            else:
                time_formatter = _format_time_explicit_local

            symbol_data = {}
            quote_timestamp_available: Optional[bool] = None
            available_attrs = set(dir(symbol_info))
            for attr in _SYMBOL_DESCRIBE_FIELDS:
                if attr not in available_attrs:
                    continue
                try:
                    value = getattr(symbol_info, attr)
                except Exception:
                    continue
                if callable(value):
                    continue
                if value is None:
                    continue
                if isinstance(value, str) and value == "":
                    continue
                if isinstance(value, str):
                    value = _clean_broker_text(value)
                if attr == "time":
                    try:
                        from ..utils.mt5 import _mt5_epoch_to_utc

                        epoch = float(value)
                        if not math.isfinite(epoch) or epoch <= 0.0:
                            quote_timestamp_available = False
                            symbol_data.update(
                                {
                                    "quote_status": "unavailable",
                                    "data_stale": True,
                                    "freshness": "unavailable, no quote timestamp",
                                    "warning": (
                                        "MT5 has not initialized quote data for this symbol."
                                    ),
                                }
                            )
                            continue
                        utc_epoch = _mt5_epoch_to_utc(epoch)
                        quote_timestamp_available = True
                        symbol_data["quote_status"] = "available"
                        if contract.shape_detail == "full":
                            symbol_data["time_epoch"] = utc_epoch
                        symbol_data["time"] = time_formatter(utc_epoch)
                        symbol_data.update(
                            {
                                key: value
                                for key, value in _quote_staleness_fields(
                                    utc_epoch,
                                    symbol=(
                                        _nonempty_symbol_string(
                                            getattr(symbol_info, "name", None)
                                        )
                                        or symbol
                                    ),
                                ).items()
                                if key
                                in {
                                    "data_age_seconds",
                                    "data_age",
                                    "data_stale",
                                    "freshness",
                                    "stale_after_seconds",
                                    "market_status",
                                    "market_status_reason",
                                    "market_status_source",
                                    "note",
                                    "warning",
                                }
                            }
                        )
                    except Exception:
                        if contract.shape_detail == "full":
                            symbol_data["time_epoch"] = value
                        symbol_data["time"] = str(value)
                else:
                    if attr in _SYMBOL_DESCRIBE_PRICE_FIELDS:
                        digits = max(0, int(getattr(symbol_info, "digits", 0) or 0))
                        value = _market_scan_round(_market_scan_float(value), digits=digits)
                    symbol_data[attr] = value

                spec = enum_specs.get(attr)
                if not spec:
                    continue
                prefixes = spec.get("prefixes", ())
                is_bitmask = bool(spec.get("bitmask"))
                if is_bitmask:
                    labels = []
                    for prefix in prefixes:
                        labels = decode_mt5_bitmask_labels(mt5_gateway, value, prefix=prefix)
                        if labels:
                            break
                    if labels:
                        symbol_data[f"{attr}_labels"] = labels
                        symbol_data[f"{attr}_label"] = ", ".join(labels)
                else:
                    label = None
                    for prefix in prefixes:
                        label = decode_mt5_enum_label(mt5_gateway, value, prefix=prefix)
                        if label:
                            break
                    if label:
                        symbol_data[f"{attr}_label"] = label

            if quote_timestamp_available is False:
                symbol_data.setdefault("quote_status", "unavailable")
                symbol_data.setdefault("data_stale", True)
                symbol_data.setdefault("freshness", "unavailable, no quote timestamp")
                symbol_data.setdefault(
                    "warning",
                    "MT5 has not initialized quote data for this symbol.",
                )
                symbol_data.pop("price_change", None)

            # ``symbol_info.time`` is a cached terminal metadata field and can
            # disagree with the executable tick stream. Reconcile the quote in
            # the same way as market_ticker before publishing freshness.
            try:
                raw_tick = mt5_gateway.symbol_info_tick(resolved_symbol)
            except Exception:
                raw_tick = None
            tick_query_epoch = float(time.time())
            resolved_tick, quote_source = resolve_quote_tick(
                mt5_gateway,
                resolved_symbol,
                raw_tick,
                now_epoch=tick_query_epoch,
                stale_after_seconds=_MARKET_SCAN_STALE_QUOTE_SECONDS,
            )
            resolved_tick_epoch = tick_epoch(resolved_tick)
            if resolved_tick_epoch is not None:
                quote_timestamp_available = True
                for key in (
                    "data_age_seconds",
                    "data_age",
                    "data_stale",
                    "freshness",
                    "freshness_state",
                    "freshness_reason",
                    "usable_for_live_trading",
                    "usable_for_live_trading_basis",
                    "stale_after_seconds",
                    "market_status",
                    "market_status_reason",
                    "market_status_source",
                    "note",
                    "warning",
                    "timestamp_ahead_of_wall_clock",
                    "timestamp_in_future",
                    "timestamp_skew_seconds",
                    "timestamp_skew_tolerance_seconds",
                    "timestamp_warning",
                ):
                    symbol_data.pop(key, None)
                symbol_data["quote_status"] = "available"
                if contract.shape_detail == "full":
                    symbol_data["time_epoch"] = resolved_tick_epoch
                symbol_data["time"] = time_formatter(resolved_tick_epoch)
                symbol_data.update(
                    _quote_staleness_fields(
                        resolved_tick_epoch,
                        symbol=resolved_symbol,
                    )
                )
                symbol_data.update(quote_source)
                if isinstance(symbol_data.get("quote_source_conflict"), dict):
                    symbol_data["usable_for_live_trading"] = False
                    symbol_data["usable_for_live_trading_basis"] = (
                        "quote_age_market_session_spread_and_source_agreement"
                    )

            price_change_value = (
                _market_scan_float(symbol_data.get("price_change"))
                if quote_timestamp_available is not False
                else None
            )
            if price_change_value is not None:
                symbol_data["price_change_pct"] = _market_scan_round(
                    price_change_value,
                    digits=6,
                )
                symbol_data["price_change_pct_unit"] = "percent (1.0 = 1%)"
                symbol_data["price_change_basis"] = "broker_symbol_info_price_change"
                symbol_data["price_change_period"] = "broker_defined_unspecified"
            elif quote_timestamp_available is not False:
                session_open = _market_scan_float(symbol_data.get("session_open"))
                session_close = _market_scan_float(symbol_data.get("session_close"))
                if (
                    session_open is not None
                    and session_close is not None
                    and abs(session_open) > 1e-12
                ):
                    symbol_data["price_change_pct"] = _market_scan_round(
                        ((session_close - session_open) / abs(session_open)) * 100.0,
                        digits=6,
                    )
                    symbol_data["price_change_pct_unit"] = "percent (1.0 = 1%)"
                    symbol_data["price_change_basis"] = "session_open_to_session_close"
                    symbol_data["price_change_period"] = "broker_current_session"
            symbol_data.pop("price_change", None)

            _normalize_spread_float_field(symbol_data)
            _add_symbol_currency_diagnostics(symbol_data)
            if contract.detail == "summary":
                symbol_data = _summary_symbol_describe_payload(symbol_data)
            elif contract.shape_detail == "compact":
                symbol_data = _compact_symbol_describe_payload(symbol_data)

            symbol_name = _nonempty_symbol_string(symbol_data.pop("name", None))
            payload = {
                "success": True,
                "symbol": symbol_name or _nonempty_symbol_string(symbol) or symbol,
                "timezone": _client_timezone_label(client_tz),
                "details": symbol_data,
                "source": build_mt5_source_provenance(mt5_gateway),
            }
            if resolved_symbol != str(symbol or "").strip():
                payload["symbol_input"] = str(symbol)
            warning = symbol_data.get("currency_base_warning")
            if warning not in (None, ""):
                payload["warnings"] = [warning]
                payload["trust"] = "verify_broker_metadata"
            return payload
        except MT5ConnectionError as exc:
            return build_error_payload(
                str(exc),
                code="mt5_connection_error",
                operation="symbols_describe",
            )
        except Exception as exc:
            return build_error_payload(
                f"Error getting symbol info: {str(exc)}",
                code="symbols_describe_failed",
                operation="symbols_describe",
            )

    return run_logged_operation(
        logger,
        operation="symbols_describe",
        symbol=symbol,
        detail=detail,
        func=_run,
    )


def _market_scan_is_tradable(symbol: Any) -> bool:
    disabled_trade_mode = getattr(mt5, "SYMBOL_TRADE_MODE_DISABLED", None)
    if disabled_trade_mode is None:
        return True
    return getattr(symbol, "trade_mode", None) != disabled_trade_mode


def _market_scan_base_row(symbol: Any) -> Dict[str, Any]:
    return {
        "symbol": _clean_broker_text(getattr(symbol, "name", None)),
        "group": _clean_broker_text(_extract_group_path_util(symbol)),
        "asset_class": _symbol_category(symbol),
        "description": _clean_broker_text(getattr(symbol, "description", None)),
    }


def _market_scan_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return out


def _market_scan_bar_int(value: Any) -> Optional[int]:
    try:
        out = int(value)
    except Exception:
        return None
    return out


def _market_scan_round(value: Optional[float], digits: int = 6) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), max(0, int(digits)))


def _market_scan_points(value: Optional[float]) -> Optional[int | float]:
    rounded = _market_scan_round(value, digits=4)
    if rounded is None:
        return None
    nearest = round(float(rounded))
    if math.isclose(float(rounded), float(nearest), abs_tol=1e-9):
        return int(nearest)
    return rounded


def _market_scan_stale_bar_seconds(timeframe: Optional[str]) -> int:
    seconds = TIMEFRAME_SECONDS.get(str(timeframe or "").strip().upper())
    if seconds:
        return max(1, int(seconds) * 2)
    return int(_MARKET_SCAN_STALE_BAR_SECONDS)


def _market_scan_freshness_fields(
    bar_time: Optional[float],
    *,
    timeframe: Optional[str] = None,
    symbol: Any = None,
) -> Dict[str, Any]:
    if bar_time is None:
        return {}
    try:
        now_epoch = float(time.time())
        close_epoch = bar_close_epoch(bar_time, str(timeframe))
        age_seconds = max(0.0, now_epoch - close_epoch)
    except Exception:
        return {}
    stale_after_seconds = _market_scan_stale_bar_seconds(timeframe)
    data_stale = age_seconds > stale_after_seconds
    symbol_name = getattr(symbol, "name", symbol)
    session_symbol = (
        None if _symbol_category(symbol) == "crypto" else symbol_name
    )
    closed_session = closed_session_context(
        session_symbol,
        now_epoch=now_epoch,
        item="bar",
        data_age_seconds=age_seconds,
    )
    fields: Dict[str, Any] = {
        "data_freshness_seconds": _market_scan_round(age_seconds, digits=3),
        "data_freshness_anchor": FRESHNESS_ANCHOR_WALL_CLOCK,
        "data_freshness_metric": FRESHNESS_METRIC_LAST_COMPLETED_BAR_AGE,
        "stale_after_seconds": stale_after_seconds,
        "bar_age_hours": _market_scan_round(age_seconds / 3600.0, digits=3),
        "data_stale": data_stale,
        "history_policy_ok": not data_stale and not bool(closed_session),
        "freshness": format_freshness_label(
            data_stale=data_stale,
            age_seconds=age_seconds,
            item="bar",
        ),
    }
    if closed_session:
        fields.update(closed_session)
        policy_relaxed = bool(closed_session.get("freshness_policy_relaxed"))
        fields["freshness"] = format_freshness_label(
            data_stale=fields.get("data_stale"),
            market_status=fields.get("market_status") if policy_relaxed else None,
            market_status_reason=(
                fields.get("market_status_reason") if policy_relaxed else None
            ),
            age_seconds=age_seconds,
            item="bar",
        )
    elif not data_stale:
        age_text = format_age_seconds(age_seconds)
        fields["freshness"] = f"latest completed bar, {age_text} ago"
    if fields["data_stale"]:
        fields["stale_warning"] = (
            "Completed bar data may be stale; latest bar is "
            f"{fields['bar_age_hours']} hours old."
        )
    return fields


def _quote_staleness_fields(
    tick_time: Optional[float],
    *,
    symbol: Any = None,
) -> Dict[str, Any]:
    if tick_time is None:
        return {}
    try:
        now_epoch = float(time.time())
        signed_age_seconds = now_epoch - float(tick_time)
    except Exception:
        return {}
    fields = build_tick_freshness_context(
        symbol,
        tick_epoch=tick_time,
        now_epoch=now_epoch,
        item="tick",
        stale_after_seconds=_MARKET_SCAN_STALE_QUOTE_SECONDS,
        age_rounder=lambda value: _market_scan_round(value, digits=3),
    )
    fields["data_age"] = (
        f"{format_age_seconds(-signed_age_seconds)} ahead of wall clock"
        if signed_age_seconds < 0.0
        else format_age_seconds(signed_age_seconds)
    )
    if fields.get("timestamp_in_future"):
        fields["warning"] = fields.get("timestamp_warning")
    elif fields["data_stale"]:
        fields["warning"] = (
            "Live quote timestamp is older than "
            f"{int(_MARKET_SCAN_STALE_QUOTE_SECONDS)} seconds."
        )
    return fields


_MARKET_SCAN_BAR_FRESHNESS_FIELDS = {
    "data_freshness_seconds": "bar_age_seconds",
    "data_freshness_anchor": "bar_freshness_anchor",
    "data_freshness_metric": "bar_freshness_metric",
    "stale_after_seconds": "bar_stale_after_seconds",
    "bar_age_hours": "bar_age_hours",
    "data_stale": "bar_stale",
    "history_policy_ok": "history_policy_ok",
    "freshness": "bar_freshness",
    "market_status": "bar_market_status",
    "market_status_reason": "bar_market_status_reason",
    "freshness_policy_relaxed": "bar_freshness_policy_relaxed",
    "stale_warning": "bar_stale_warning",
}


def _market_scan_bar_freshness_fields(
    bar_time: Optional[float],
    *,
    timeframe: Optional[str] = None,
    symbol: Any = None,
) -> Dict[str, Any]:
    """Return bar-policy fields that cannot overwrite quote safety fields."""
    fields = _market_scan_freshness_fields(
        bar_time,
        timeframe=timeframe,
        symbol=symbol,
    )
    return {
        output_name: fields[input_name]
        for input_name, output_name in _MARKET_SCAN_BAR_FRESHNESS_FIELDS.items()
        if input_name in fields
    }


def _market_scan_quote_freshness_fields(
    tick_time: Optional[float],
    *,
    symbol: Any = None,
) -> Dict[str, Any]:
    if tick_time is None:
        return {}
    return {
        "tick_time": _format_time_explicit(tick_time),
        **_quote_staleness_fields(tick_time, symbol=symbol),
    }


def _market_scan_points_per_pip(symbol: Any, *, point: float, digits: int) -> Optional[float]:
    return forex_points_per_pip(
        str(getattr(symbol, "name", "") or ""),
        path=str(getattr(symbol, "path", "") or ""),
        point=point,
        digits=digits,
    )


def _build_market_scan_spread_row(
    symbol: Any,
    mt5_gateway: Any,
    *,
    spread_cost_currency: Optional[str] = None,
) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    raw_tick = mt5_gateway.symbol_info_tick(symbol.name)
    tick, quote_source = resolve_quote_tick(
        mt5_gateway,
        symbol.name,
        raw_tick,
        now_epoch=time.time(),
        stale_after_seconds=_MARKET_SCAN_STALE_QUOTE_SECONDS,
    )
    if tick is None:
        return None, f"Failed to get tick data: {mt5_gateway.last_error()}"

    bid = _market_scan_float(tick_value(tick, "bid"))
    ask = _market_scan_float(tick_value(tick, "ask"))
    tick_time = tick_epoch(tick)
    point = _market_scan_float(getattr(symbol, "point", 0.0)) or 0.0
    tick_size = _market_scan_float(getattr(symbol, "trade_tick_size", 0.0)) or 0.0
    trade_tick_value = (
        _market_scan_float(getattr(symbol, "trade_tick_value", 0.0)) or 0.0
    )
    digits = max(0, int(getattr(symbol, "digits", 0) or 0))

    points_per_pip = _market_scan_points_per_pip(symbol, point=point, digits=digits)
    spread_metrics = compute_spread_metrics(
        bid,
        ask,
        point=point,
        points_per_pip=points_per_pip,
        tick_size=tick_size,
        tick_value_money=trade_tick_value,
        account_currency=spread_cost_currency,
    )
    spread_quality = spread_metrics["spread_quality"]
    if spread_quality == "one_sided":
        return None, "Bid/ask quote is unavailable."
    if spread_quality == "inverted":
        return None, "Bid/ask quote is invalid."
    spread_valid = spread_metrics["spread_valid"]
    spread_abs = spread_metrics["spread"]
    mid = spread_metrics["mid"]
    spread_points = spread_metrics["spread_points"]
    spread_pips = spread_metrics["spread_pips"]
    spread_pct = spread_metrics["spread_pct"]
    spread_cost_per_lot = spread_metrics["spread_cost_per_lot"]
    pricing_basis = spread_metrics["pricing_basis"]

    quote_freshness = _market_scan_quote_freshness_fields(
        tick_time,
        symbol=symbol.name,
    )
    if not spread_valid:
        quality_warning = "Locked quote (bid equals ask) is not usable for live trading."
        existing_warning = quote_freshness.get("warning")
        quote_freshness["warning"] = (
            f"{existing_warning} {quality_warning}"
            if existing_warning
            else quality_warning
        )
        quote_freshness["usable_for_live_trading"] = False
        quote_freshness["usable_for_live_trading_basis"] = (
            "quote_age_market_session_and_positive_spread"
        )

    row = _market_scan_base_row(symbol)
    row.update(
        {
            "bid": _market_scan_round(bid, digits=digits),
            "ask": _market_scan_round(ask, digits=digits),
            "mid": _market_scan_round(mid, digits=digits + 1),
            "quote_as_of": (
                _format_time_explicit(tick_time) if tick_time is not None else None
            ),
            **quote_source,
            **quote_freshness,
            "spread": _market_scan_round(spread_abs, digits=digits),
            "spread_points": _market_scan_points(spread_points),
            "spread_pips": _market_scan_round(spread_pips, digits=4),
            "spread_pct": _market_scan_round(spread_pct, digits=6),
            "spread_cost_per_lot": _market_scan_round(spread_cost_per_lot, digits=6),
            "spread_valid": spread_valid,
            "spread_quality": spread_quality,
            "pricing_basis": pricing_basis,
        }
    )
    if isinstance(row.get("quote_source_conflict"), dict):
        row["usable_for_live_trading"] = False
        row["usable_for_live_trading_basis"] = (
            "quote_age_market_session_spread_and_source_agreement"
        )
    if spread_cost_per_lot is not None and spread_cost_currency:
        row["spread_cost_currency"] = spread_cost_currency
    return row, None


def _market_scan_completed_rates(
    symbol: str,
    *,
    timeframe: str,
    mt5_timeframe: Any,
    count: int,
) -> Any:
    requested = max(1, int(count))
    rates = _mt5_copy_rates_from_pos(symbol, mt5_timeframe, 0, requested + 1)
    if rates is None or len(rates) < 1:
        return rates
    latest_time = _market_scan_float(rates[-1]["time"])
    if (
        latest_time is not None
        and bar_close_epoch(latest_time, timeframe) > time.time()
    ):
        rates = rates[:-1]
    if len(rates) > requested:
        rates = rates[-requested:]
    return rates


def _build_market_scan_bar_row(
    symbol: Any,
    timeframe: str,
    mt5_timeframe: Any,
) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    rates = _market_scan_completed_rates(
        symbol.name,
        timeframe=timeframe,
        mt5_timeframe=mt5_timeframe,
        count=2,
    )
    if rates is None or len(rates) < 2:
        return None, f"At least two completed {timeframe} bars are required."

    latest_bar = rates[-1]
    previous_bar = rates[-2]
    open_price = _market_scan_float(latest_bar["open"])
    close_price = _market_scan_float(latest_bar["close"])
    previous_close = _market_scan_float(previous_bar["close"])
    if open_price is None or close_price is None:
        return None, "Completed bar is missing open/close prices."
    if previous_close is None:
        return None, "Previous completed bar is missing its close price."
    if previous_close == 0:
        return None, "Previous completed bar close price is zero."

    digits = max(0, int(getattr(symbol, "digits", 0) or 0))
    bar_time = _market_scan_float(latest_bar["time"])
    tick_volume = _market_scan_bar_int(latest_bar["tick_volume"])
    real_volume = _market_scan_bar_int(latest_bar["real_volume"])
    row = _market_scan_base_row(symbol)
    row.update(
        {
            "timeframe": timeframe,
            "data_source": f"{timeframe}_bars",
            "time": _format_time_explicit(bar_time) if bar_time is not None else None,
            **_market_scan_bar_freshness_fields(
                bar_time,
                timeframe=timeframe,
                symbol=symbol,
            ),
            "previous_close": _market_scan_round(previous_close, digits=digits),
            "open": _market_scan_round(open_price, digits=digits),
            "close": _market_scan_round(close_price, digits=digits),
            "price_currency": str(
                getattr(symbol, "currency_profit", "") or ""
            ).strip()
            or None,
            "price_basis": "mt5_latest_completed_bar_close",
            "price_point": _market_scan_float(getattr(symbol, "point", None)),
            "tick_volume": tick_volume,
            "real_volume": real_volume,
            "price_change_pct": _market_scan_round(
                ((close_price - previous_close) / previous_close) * 100.0,
                digits=6,
            ),
            "price_change_basis": "previous_completed_close_to_latest_completed_close",
            "price_change_period": {
                "bars": 1,
                "timeframe": timeframe,
                "bar_state": "completed",
            },
            "gap_pct": _market_scan_round(
                ((open_price - previous_close) / previous_close) * 100.0,
                digits=6,
            ),
            "gap_basis": "previous_completed_close_to_latest_completed_open",
        }
    )
    return row, None


def _market_scan_table(
    headers: List[str],
    rows: List[Dict[str, Any]],
    *,
    include_contract_meta: bool = True,
) -> Dict[str, Any]:
    ordered_rows = [[row.get(header) for header in headers] for row in rows]
    result = _table_from_rows(headers, ordered_rows)
    return attach_collection_contract(
        result,
        collection_kind="table",
        rows=result.get("data"),
        include_contract_meta=include_contract_meta,
    )


def _market_scan_contract_table(
    headers: List[str],
    rows: List[Dict[str, Any]],
    *,
    include_columns: bool = True,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "rows": [dict(row) for row in rows],
        "row_count": int(len(rows)),
    }
    if include_columns:
        columns = [str(header) for header in headers]
        seen = set(columns)
        for row in rows:
            if not isinstance(row, dict):
                continue
            for key in row.keys():
                column = str(key)
                if column not in seen:
                    columns.append(column)
                    seen.add(column)
        out["columns"] = columns
    return out


def _project_market_scan_rows(
    headers: List[str],
    rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    projected: List[Dict[str, Any]] = []
    for row in rows:
        out = {header: row.get(header) for header in headers}
        projected.append(out)
    return projected


def _compact_market_scan_projection(
    headers: List[str],
    rows: List[Dict[str, Any]],
) -> tuple[List[str], Dict[str, Any]]:
    """Prune empty columns and hoist metadata repeated across scan rows."""
    projected_headers = [
        header
        for header in headers
        if any(row.get(header) is not None for row in rows)
    ]
    shared: Dict[str, Any] = {}
    if len(rows) > 1:
        for header in (
            "price_basis",
            "bar_market_status_reason",
            "bar_freshness_policy_relaxed",
        ):
            values = [row.get(header) for row in rows]
            first = values[0]
            if first is not None and all(value == first for value in values[1:]):
                shared[header] = first
                projected_headers = [
                    candidate
                    for candidate in projected_headers
                    if candidate != header
                ]

        timestamp_warnings = [
            str(row.get("timestamp_warning"))
            for row in rows
            if row.get("timestamp_warning") not in (None, "")
        ]
        unique_warnings = list(dict.fromkeys(timestamp_warnings))
        if len(timestamp_warnings) > len(unique_warnings):
            shared["warnings"] = unique_warnings
            projected_headers = [
                header
                for header in projected_headers
                if header != "timestamp_warning"
            ]
    return projected_headers, shared


_MARKET_SCAN_UNITS = {
    "bid": "price",
    "ask": "price",
    "mid": "price",
    "close": "price",
    "previous_close": "price",
    "price_change_pct": "percent (1.0 = 1%)",
    "gap_pct": "percent (1.0 = 1%)",
    "tick_volume": "broker_tick_count",
    "real_volume": "traded_volume",
    "spread_points": "broker_points",
    "spread_pips": "pips",
    "spread_pct": "percent (1.0 = 1%)",
    "spread_cost_per_lot": "currency_per_lot_estimate",
    "rsi": "0_100",
    "sma_distance_pct": "percent (1.0 = 1%)",
    "price_point": "broker_price_increment",
    "data_age_seconds": "seconds",
    "data_freshness_seconds": "seconds",
    "stale_after_seconds": "seconds",
    "bar_age_seconds": "seconds",
    "bar_stale_after_seconds": "seconds",
    "bar_age_hours": "hours",
}


def _market_scan_units_for_rows(rows: List[Dict[str, Any]]) -> Dict[str, str]:
    seen_fields = {
        str(key)
        for row in rows
        if isinstance(row, dict)
        for key, value in row.items()
        if value is not None
    }
    units = {
        key: unit
        for key, unit in _MARKET_SCAN_UNITS.items()
        if key in seen_fields
    }
    if "spread_pips" in units:
        has_non_fx_spread_row = any(
            isinstance(row, dict)
            and row.get("spread_points") is not None
            and row.get("spread_pips") is None
            for row in rows
        )
        if has_non_fx_spread_row:
            units["spread_pips"] = "pips (forex_only; omitted when not applicable)"
    return units


def _attach_top_markets_units(
    out: Dict[str, Any],
    *row_groups: List[Dict[str, Any]],
) -> None:
    rows = [
        row
        for group in row_groups
        for row in group
        if isinstance(row, dict)
    ]
    units = _market_scan_units_for_rows(rows)
    if units:
        out["units"] = units
    _attach_market_scan_volume_semantics(out, units)


def _attach_market_scan_volume_semantics(
    out: Dict[str, Any],
    units: Dict[str, str],
) -> None:
    if units.get("tick_volume") == "broker_tick_count":
        out["volume_type"] = "tick_volume"
        out["volume_semantics"] = TICK_VOLUME_SEMANTICS


def _market_scan_contract_meta(
    *,
    request: Dict[str, Any],
    stats: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "tool": "market_scan",
        "request": {
            key: value for key, value in request.items() if value is not None
        },
        "runtime": {},
    }
    if stats:
        out["stats"] = {
            key: value for key, value in stats.items() if value is not None
        }
    return out


def _market_scan_error(
    message: str,
    *,
    code: str,
    request: Dict[str, Any],
    stats: Optional[Dict[str, Any]] = None,
    details: Any = None,
    warnings: Optional[List[str]] = None,
) -> Dict[str, Any]:
    out = build_error_payload(
        message,
        code=str(code),
        operation="market_scan",
    )
    out["meta"] = _market_scan_contract_meta(request=request, stats=stats)
    if details not in (None, [], {}):
        out["details"] = details
    if warnings:
        out["warnings"] = warnings
    return out


def _market_scan_freshness_summary(
    rows: List[Dict[str, Any]],
    *,
    include_stale_symbols: bool = True,
) -> Dict[str, Any]:
    if not rows:
        return {}
    def _row_stale(row: Dict[str, Any]) -> bool:
        return bool(row.get("bar_stale")) or bool(
            row.get("quote_stale", row.get("data_stale"))
        )

    stale_count = sum(1 for row in rows if _row_stale(row))
    stale_bar_count = sum(1 for row in rows if bool(row.get("bar_stale")))
    unsafe_quote_count = sum(
        1
        for row in rows
        if bool(row.get("quote_stale", row.get("data_stale")))
        or row.get(
            "quote_usable_for_live_trading",
            row.get("usable_for_live_trading"),
        ) is False
    )
    row_count = len(rows)
    if stale_count == row_count:
        freshness = "stale"
    elif stale_count:
        freshness = f"mixed, {stale_count}/{row_count} stale"
    else:
        freshness = "fresh"

    out: Dict[str, Any] = {
        "freshness": freshness,
        "stale_rows": int(stale_count),
        "freshness_basis": "conservative_quote_or_bar",
        "stale_bar_rows": int(stale_bar_count),
        "unsafe_quote_rows": int(unsafe_quote_count),
    }
    if stale_count and include_stale_symbols:
        out["stale_symbols"] = [
            str(row.get("symbol"))
            for row in rows
            if _row_stale(row) and str(row.get("symbol") or "").strip()
        ]
    row_times = [
        str(row.get("time") or "").strip()
        for row in rows
        if str(row.get("time") or "").strip()
    ]
    if row_times:
        out["data_as_of"] = max(row_times)

    now_epoch = time.time()
    closed_count = sum(
        1
        for row in rows
        if closed_session_context(row.get("symbol"), now_epoch=now_epoch)
    )
    if closed_count == row_count:
        out["session_status"] = "closed_weekend"
        if stale_count == 0:
            out["freshness"] = "closed_weekend_snapshot"
    elif closed_count:
        out["session_status"] = f"mixed, {closed_count}/{row_count} closed_weekend"
        if stale_count == 0:
            out["freshness"] = f"mixed, {closed_count}/{row_count} closed_weekend_snapshot"
    return out


def _namespace_market_scan_quote_freshness(row: Dict[str, Any]) -> None:
    """Keep quote freshness distinct from the bar that supplies scan prices."""
    field_map = {
        "tick_time": "quote_time",
        "data_age_seconds": "quote_age_seconds",
        "data_age_anchor": "quote_age_anchor",
        "data_age_metric": "quote_age_metric",
        "stale_after_seconds": "quote_stale_after_seconds",
        "data_stale": "quote_stale",
        "freshness_reason": "quote_freshness_reason",
        "timestamp_ahead_of_wall_clock": "quote_timestamp_ahead_of_wall_clock",
        "timestamp_in_future": "quote_timestamp_in_future",
        "timestamp_skew_seconds": "quote_timestamp_skew_seconds",
        "timestamp_skew_tolerance_seconds": "quote_timestamp_skew_tolerance_seconds",
        "timestamp_warning": "quote_timestamp_warning",
        "warning": "quote_warning",
        "freshness": "quote_freshness",
        "usable_for_live_trading": "quote_usable_for_live_trading",
        "usable_for_live_trading_basis": "quote_usable_for_live_trading_basis",
    }
    for source, target in field_map.items():
        if source in row:
            row[target] = row.pop(source)
    row["price_as_of"] = row.get("time")
    row["price_freshness"] = row.get("bar_freshness")


def _market_scan_quote_exclusion_reason(row: Dict[str, Any]) -> str:
    freshness_reason = str(row.get("quote_freshness_reason") or "").strip().lower()
    if freshness_reason and freshness_reason != "live_quote":
        return freshness_reason
    spread_quality = str(row.get("spread_quality") or "").strip().lower()
    if spread_quality and spread_quality != "two_sided":
        return f"quote_{spread_quality}"
    if row.get("quote_stale") is True:
        return "stale_quote"
    if freshness_reason:
        return freshness_reason
    return "quote_not_usable_for_live_trading"


_TOP_MARKETS_COMPACT_BASE_HEADERS = [
    "symbol",
    "group",
    "asset_class",
    "timeframe",
    "data_source",
    "time",
    "data_stale",
    "freshness",
    "spread_valid",
    "spread_quality",
    "usable_for_live_trading",
]

_TOP_MARKETS_COMPACT_SPREAD_HEADERS = [
    "quote_as_of",
    "bid",
    "ask",
    "mid",
    "spread_pct",
    "spread_points",
    "spread_pips",
]

_TOP_MARKETS_COMPACT_BAR_HEADERS = [
    "bar_stale",
    "bar_freshness",
    "close",
    "quote_as_of",
    "bid",
    "ask",
    "mid",
    "tick_volume",
    "price_change_pct",
]

_TOP_MARKETS_COMPACT_HEADERS = [
    *_TOP_MARKETS_COMPACT_BASE_HEADERS,
    "bar_stale",
    "bar_freshness",
    "close",
    "quote_as_of",
    "bid",
    "ask",
    "mid",
    "spread_pct",
    "spread_points",
    "tick_volume",
    "price_change_pct",
]

_TOP_MARKETS_FULL_BASE_HEADERS = [
    "symbol",
    "group",
    "asset_class",
    "description",
    "timeframe",
    "data_source",
    "time",
    "data_age_seconds",
    "data_age_anchor",
    "data_age_metric",
    "stale_after_seconds",
    "data_stale",
    "freshness_reason",
    "timestamp_ahead_of_wall_clock",
    "timestamp_in_future",
    "timestamp_skew_seconds",
    "timestamp_skew_tolerance_seconds",
    "timestamp_warning",
    "freshness",
    "warning",
    "spread_valid",
    "spread_quality",
    "usable_for_live_trading",
]

_TOP_MARKETS_COMPACT_VOLUME_HEADERS = [
    "symbol",
    "group",
    "asset_class",
    "timeframe",
    "data_source",
    "time",
    "bar_stale",
    "bar_freshness",
    "close",
    "tick_volume",
    "price_change_pct",
]

_TOP_MARKETS_FULL_SPREAD_HEADERS = [
    "quote_as_of",
    "bid",
    "ask",
    "mid",
    "spread",
    "spread_points",
    "spread_pct",
    "spread_cost_per_lot",
    "spread_cost_currency",
    "pricing_basis",
]

_TOP_MARKETS_FULL_BAR_HEADERS = [
    "bar_age_seconds",
    "bar_freshness_anchor",
    "bar_freshness_metric",
    "bar_stale_after_seconds",
    "bar_age_hours",
    "bar_stale",
    "bar_market_status",
    "bar_market_status_reason",
    "bar_freshness_policy_relaxed",
    "bar_freshness",
    "bar_stale_warning",
    "previous_close",
    "open",
    "close",
    "quote_as_of",
    "bid",
    "ask",
    "mid",
    "tick_volume",
    "real_volume",
    "price_change_pct",
]

_TOP_MARKETS_FULL_HEADERS = [
    *_TOP_MARKETS_FULL_BASE_HEADERS,
    *_TOP_MARKETS_FULL_SPREAD_HEADERS,
    *_TOP_MARKETS_FULL_BAR_HEADERS,
]


def _top_markets_headers(metric: str, *, detail_mode: str) -> List[str]:
    if metric == "spread":
        if detail_mode == "compact":
            return [
                *_TOP_MARKETS_COMPACT_BASE_HEADERS,
                *_TOP_MARKETS_COMPACT_SPREAD_HEADERS,
            ]
        return [
            *_TOP_MARKETS_FULL_BASE_HEADERS,
            *_TOP_MARKETS_FULL_SPREAD_HEADERS,
        ]
    if metric == "volume" and detail_mode == "compact":
        return list(_TOP_MARKETS_COMPACT_VOLUME_HEADERS)
    if detail_mode == "compact":
        return [
            *_TOP_MARKETS_COMPACT_BASE_HEADERS,
            *_TOP_MARKETS_COMPACT_BAR_HEADERS,
        ]
    return [
        *_TOP_MARKETS_FULL_BASE_HEADERS,
        *_TOP_MARKETS_FULL_BAR_HEADERS,
    ]


def _top_markets_all_headers(*, detail_mode: str) -> List[str]:
    compact_headers = [
        "rank_category",
        "rank",
        *_TOP_MARKETS_COMPACT_HEADERS,
    ]
    if detail_mode == "compact":
        return compact_headers
    return [
        "rank_category",
        "rank",
        *_TOP_MARKETS_FULL_HEADERS,
    ]


def _ranked_top_market_rows(
    ranking: str,
    rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    return [
        {
            "rank_category": ranking,
            "rank": rank,
            **row,
        }
        for rank, row in enumerate(rows, start=1)
    ]


def _top_market_data_source(metric: str, timeframe: str) -> str:
    return "live_tick" if metric == "spread" else f"{timeframe}_bars"


def _top_market_data_time_key(metric: str) -> str:
    return "tick_time" if metric == "spread" else "time"


def _top_market_rows_with_data_context(
    metric: str,
    rows: List[Dict[str, Any]],
    *,
    timeframe: str,
) -> List[Dict[str, Any]]:
    data_source = _top_market_data_source(metric, timeframe)
    data_time_key = _top_market_data_time_key(metric)
    normalized: List[Dict[str, Any]] = []
    for row in rows:
        mapped = dict(row)
        mapped["data_source"] = data_source
        mapped["time"] = row.get(data_time_key)
        normalized.append(mapped)
    return normalized


def _parse_market_scan_symbols(symbols: Optional[str]) -> List[str]:
    text = str(symbols or "").replace(";", ",").replace("\n", ",")
    parsed: List[str] = []
    seen: set[str] = set()
    for chunk in text.split(","):
        name = chunk.strip()
        if not name:
            continue
        upper = name.upper()
        if upper in seen:
            continue
        seen.add(upper)
        parsed.append(name)
    return parsed


def _market_scan_group_query_variants(value: Any) -> set[str]:
    normalized = _normalize_group_path_query(str(value or "")).casefold()
    tokens = [
        token
        for token in re.split(r"[^a-z0-9]+", normalized)
        if token
    ]

    def _singular(token: str) -> str:
        if token.endswith("ies") and len(token) > 3:
            return token[:-3] + "y"
        if token.endswith("s") and len(token) > 1:
            return token[:-1]
        return token

    variants = {
        normalized,
        re.sub(r"[^a-z0-9]+", "", normalized),
    }
    if tokens:
        variants.add("".join(tokens))
        variants.add("".join(_singular(token) for token in tokens))
        variants.update(tokens)
        variants.update(_singular(token) for token in tokens)
    return {variant for variant in variants if variant}


def _market_scan_group_matches_query(group_path: str, requested: str) -> bool:
    group_variants = _market_scan_group_query_variants(group_path)
    requested_variants = _market_scan_group_query_variants(requested)
    if group_variants.intersection(requested_variants):
        return True
    return any(
        requested_variant in group_variant
        for requested_variant in requested_variants
        for group_variant in group_variants
        if len(requested_variant) >= 3
    )


def _resolve_market_scan_group_path(
    all_symbols: List[Any],
    group: str,
) -> tuple[List[str], Optional[str]]:
    requested = str(group or "").strip()
    if not requested:
        return [], "group must not be empty."

    groups: Dict[str, str] = {}
    for symbol in all_symbols:
        group_path = str(_extract_group_path_util(symbol) or "").strip()
        if not group_path:
            continue
        groups.setdefault(_normalize_group_path_query(group_path).lower(), group_path)

    requested_lower = _normalize_group_path_query(requested).lower()
    exact = groups.get(requested_lower)
    if exact is not None:
        return [exact], None

    partial_matches = sorted(
        (
            value for value in groups.values()
            if _market_scan_group_matches_query(value, requested_lower)
        ),
        key=_case_insensitive_sort_key,
    )
    if len(partial_matches) == 1:
        return [partial_matches[0]], None
    if partial_matches:
        return partial_matches, None
    available = sorted(groups.values(), key=_case_insensitive_sort_key)
    if available:
        preview = ", ".join(available[:5])
        suffix = ", ..." if len(available) > 5 else ""
        return [], f"No symbol group matched '{requested}'. Available groups: {preview}{suffix}"
    return [], f"No symbol group matched '{requested}'."


def _select_market_scan_symbols(
    all_symbols: List[Any],
    *,
    symbols: Optional[str] = None,
    group: Optional[str],
    universe: str,
) -> tuple[List[Any], Dict[str, Any], Optional[str]]:
    requested_names = _parse_market_scan_symbols(symbols)
    selection_meta: Dict[str, Any] = {}
    if requested_names:
        selection_meta["symbols_input"] = list(requested_names)
    if requested_names and group:
        return [], selection_meta, "Provide either symbols or group, not both."

    tradable_symbols = [symbol for symbol in all_symbols if _market_scan_is_tradable(symbol)]

    if requested_names:
        by_upper: Dict[str, Any] = {}
        for tradable_symbol in tradable_symbols:
            name = str(getattr(tradable_symbol, "name", "") or "").strip()
            if not name:
                continue
            by_upper.setdefault(name.upper(), tradable_symbol)
        selected: List[Any] = []
        missing: List[str] = []
        for name in requested_names:
            symbol_obj = by_upper.get(name.upper())
            if symbol_obj is None:
                missing.append(name)
                continue
            selected.append(symbol_obj)
        if not selected:
            return [], selection_meta, "None of the requested symbols matched the MT5 symbol list."
        return selected, {
            **selection_meta,
            "scope": "symbols",
            "requested_symbols": requested_names,
            "missing_symbols": missing,
        }, None

    if group:
        resolved_groups, group_error = _resolve_market_scan_group_path(tradable_symbols, group)
        if group_error or not resolved_groups:
            return [], {}, group_error
        resolved_group_set = {
            _normalize_group_path_query(str(group_path).strip()).lower()
            for group_path in resolved_groups
        }
        selected = sorted(
            [
                symbol for symbol in tradable_symbols
                if (
                    _normalize_group_path_query(
                        str(_extract_group_path_util(symbol) or "").strip()
                    ).lower()
                    in resolved_group_set
                )
                and (universe == "all" or bool(getattr(symbol, "visible", False)))
            ],
            key=lambda symbol: _case_insensitive_sort_key(getattr(symbol, "name", "")),
        )
        return selected, {
            **selection_meta,
            "scope": "group",
            "group": resolved_groups[0] if len(resolved_groups) == 1 else str(group).strip(),
            "groups": resolved_groups,
        }, None

    selected = sorted(
        [
            symbol for symbol in tradable_symbols
            if universe == "all" or bool(getattr(symbol, "visible", False))
        ],
        key=lambda symbol: _case_insensitive_sort_key(getattr(symbol, "name", "")),
    )
    return selected, {**selection_meta, "scope": "universe"}, None


def _market_scan_compute_rsi(closes: List[float], length: int) -> Optional[float]:
    if length <= 0 or len(closes) < (length + 1):
        return None

    gains: List[float] = []
    losses: List[float] = []
    for prev_close, close in zip(closes[:-1], closes[1:], strict=False):
        delta = float(close - prev_close)
        gains.append(max(delta, 0.0))
        losses.append(max(-delta, 0.0))

    avg_gain = sum(gains[:length]) / float(length)
    avg_loss = sum(losses[:length]) / float(length)
    for gain, loss in zip(gains[length:], losses[length:], strict=False):
        avg_gain = ((avg_gain * float(length - 1)) + float(gain)) / float(length)
        avg_loss = ((avg_loss * float(length - 1)) + float(loss)) / float(length)

    if avg_loss <= 0.0:
        if avg_gain <= 0.0:
            return 50.0
        return 100.0

    relative_strength = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + relative_strength))


def _build_market_scan_signal_row(
    symbol: Any,
    *,
    timeframe: str,
    mt5_timeframe: Any,
    lookback: int,
    rsi_length: int,
    sma_period: int,
    include_rsi: bool,
    include_sma: bool,
) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    rsi_warmup_bars = (
        max(50, int(rsi_length) * 25) if include_rsi else 0
    )
    fetch_count = max(int(lookback), rsi_warmup_bars)
    rates = _market_scan_completed_rates(
        symbol.name,
        timeframe=timeframe,
        mt5_timeframe=mt5_timeframe,
        count=fetch_count,
    )
    if rates is None or len(rates) < 2:
        return None, f"At least two completed {timeframe} bars are required."

    latest_bar = rates[-1]
    previous_bar = rates[-2]
    open_price = _market_scan_float(latest_bar["open"])
    close_price = _market_scan_float(latest_bar["close"])
    previous_close = _market_scan_float(previous_bar["close"])
    if open_price is None or close_price is None:
        return None, "Completed bar is missing open/close prices."
    if previous_close is None:
        return None, "Previous completed bar is missing its close price."
    if previous_close == 0:
        return None, "Previous completed bar close price is zero."

    close_values: List[float] = []
    for bar in rates:
        close_value = _market_scan_float(bar["close"])
        if close_value is not None:
            close_values.append(close_value)

    digits = max(0, int(getattr(symbol, "digits", 0) or 0))
    bar_time = _market_scan_float(latest_bar["time"])
    tick_volume = _market_scan_bar_int(latest_bar["tick_volume"])
    real_volume = _market_scan_bar_int(latest_bar["real_volume"])
    sma_value = None
    if include_sma and len(close_values) >= max(1, int(sma_period)):
        sma_window = close_values[-int(sma_period):]
        sma_value = float(sum(sma_window) / len(sma_window))
    rsi_values = (
        close_values[-rsi_warmup_bars:]
        if include_rsi and rsi_warmup_bars
        else close_values
    )
    rsi_value = (
        _market_scan_compute_rsi(rsi_values, int(rsi_length))
        if include_rsi
        else None
    )
    sma_distance_pct = None
    if sma_value is not None and sma_value != 0:
        sma_distance_pct = ((close_price - sma_value) / sma_value) * 100.0

    row = _market_scan_base_row(symbol)
    row.update(
        {
            "timeframe": timeframe,
            "data_source": f"{timeframe}_bars",
            "time": _format_time_explicit(bar_time) if bar_time is not None else None,
            **_market_scan_bar_freshness_fields(
                bar_time,
                timeframe=timeframe,
                symbol=symbol,
            ),
            "previous_close": _market_scan_round(previous_close, digits=digits),
            "open": _market_scan_round(open_price, digits=digits),
            "close": _market_scan_round(close_price, digits=digits),
            "price_currency": str(
                getattr(symbol, "currency_profit", "") or ""
            ).strip()
            or None,
            "price_basis": "mt5_latest_completed_bar_close",
            "price_point": _market_scan_float(getattr(symbol, "point", None)),
            "tick_volume": tick_volume,
            "real_volume": real_volume,
            "price_change_pct": _market_scan_round(
                ((close_price - previous_close) / previous_close) * 100.0,
                digits=6,
            ),
            "price_change_basis": "previous_completed_close_to_latest_completed_close",
            "price_change_period": {
                "bars": 1,
                "timeframe": timeframe,
                "bar_state": "completed",
            },
            "rsi_warmup_bars": rsi_warmup_bars if include_rsi else None,
            "gap_pct": _market_scan_round(
                ((open_price - previous_close) / previous_close) * 100.0,
                digits=6,
            ),
            "gap_basis": "previous_completed_close_to_latest_completed_open",
        }
    )
    if include_rsi:
        row["rsi"] = _market_scan_round(rsi_value, digits=4)
    if include_sma:
        row["sma_value"] = _market_scan_round(sma_value, digits=digits)
        row["sma_distance_pct"] = _market_scan_round(sma_distance_pct, digits=6)
    return row, None


def _market_scan_missing_required_metric(
    row: Dict[str, Any],
    *,
    rank_by: str,
    rsi_above: Optional[float],
    rsi_below: Optional[float],
    price_vs_sma: Optional[str],
    max_spread_pct: Optional[float],
    min_tick_volume: Optional[int],
    min_price_change_pct: Optional[float],
    max_price_change_pct: Optional[float],
    min_gap_pct: Optional[float] = None,
    max_gap_pct: Optional[float] = None,
    rsi_length: int,
    sma_period: int,
) -> Optional[str]:
    requirements: List[tuple[str, str]] = []
    if rank_by in {"abs_price_change_pct", "price_change_pct"}:
        requirements.append(("price_change_pct", "price-change data is unavailable."))
    elif rank_by == "gap_pct":
        requirements.append(("gap_pct", "Gap data is unavailable."))
    elif rank_by == "tick_volume":
        requirements.append(("tick_volume", "Tick-volume data is unavailable."))
    elif rank_by == "rsi":
        requirements.append(("rsi", f"Not enough history to compute RSI({int(rsi_length)})."))
    elif rank_by == "spread_pct":
        requirements.append(("spread_pct", "Spread data is unavailable."))

    if min_price_change_pct is not None or max_price_change_pct is not None:
        requirements.append(("price_change_pct", "price-change data is unavailable."))
    if min_gap_pct is not None or max_gap_pct is not None:
        requirements.append(("gap_pct", "Gap data is unavailable."))
    if max_spread_pct is not None:
        requirements.append(("spread_pct", "Spread data is unavailable."))
    if min_tick_volume is not None:
        requirements.append(("tick_volume", "Tick-volume data is unavailable."))
    if rsi_above is not None or rsi_below is not None:
        requirements.append(("rsi", f"Not enough history to compute RSI({int(rsi_length)})."))
    if price_vs_sma is not None:
        requirements.append(("sma_value", f"Not enough history to compute SMA({int(sma_period)})."))

    for key, message in requirements:
        if row.get(key) is None:
            return message
    return None


def _market_scan_row_matches_filters(
    row: Dict[str, Any],
    *,
    min_price_change_pct: Optional[float],
    max_price_change_pct: Optional[float],
    max_spread_pct: Optional[float],
    min_tick_volume: Optional[int],
    rsi_below: Optional[float],
    rsi_above: Optional[float],
    price_vs_sma: Optional[str],
    min_gap_pct: Optional[float] = None,
    max_gap_pct: Optional[float] = None,
) -> bool:
    price_change_pct = _market_scan_float(row.get("price_change_pct"))
    gap_pct = _market_scan_float(row.get("gap_pct"))
    spread_pct = _market_scan_float(row.get("spread_pct"))
    tick_volume = _market_scan_bar_int(row.get("tick_volume"))
    rsi_value = _market_scan_float(row.get("rsi"))
    close_price = _market_scan_float(row.get("close"))
    sma_value = _market_scan_float(row.get("sma_value"))

    if min_price_change_pct is not None and (price_change_pct is None or price_change_pct < float(min_price_change_pct)):
        return False
    if max_price_change_pct is not None and (price_change_pct is None or price_change_pct > float(max_price_change_pct)):
        return False
    if min_gap_pct is not None and (gap_pct is None or gap_pct < float(min_gap_pct)):
        return False
    if max_gap_pct is not None and (gap_pct is None or gap_pct > float(max_gap_pct)):
        return False
    if max_spread_pct is not None and (
        row.get("spread_valid") is not True
        or spread_pct is None
        or spread_pct > float(max_spread_pct)
    ):
        return False
    if min_tick_volume is not None and (tick_volume is None or tick_volume < int(min_tick_volume)):
        return False
    if rsi_below is not None and (rsi_value is None or rsi_value > float(rsi_below)):
        return False
    if rsi_above is not None and (rsi_value is None or rsi_value < float(rsi_above)):
        return False
    if price_vs_sma == "above" and (close_price is None or sma_value is None or close_price <= sma_value):
        return False
    if price_vs_sma == "below" and (close_price is None or sma_value is None or close_price >= sma_value):
        return False
    return True


def _market_scan_sort_rows(
    rows: List[Dict[str, Any]],
    *,
    rank_by: str,
    rank_order: str,
    rsi_above: Optional[float],
    rsi_below: Optional[float],
) -> None:
    order = _market_scan_effective_rank_order(
        rank_by,
        rank_order=rank_order,
        rsi_above=rsi_above,
        rsi_below=rsi_below,
    )

    if rank_by == "abs_price_change_pct":
        rows.sort(
            key=lambda row: (
                bool(row.get("bar_stale")),
                row.get("price_change_pct") is None,
                (
                    abs(float(row.get("price_change_pct") or 0.0))
                    if order == "asc"
                    else -abs(float(row.get("price_change_pct") or 0.0))
                ),
                row.get("symbol") or "",
            )
        )
        return

    missing_value = float("inf") if order == "asc" else 0.0

    rows.sort(
        key=lambda row: (
            rank_by == "spread_pct" and row.get("spread_valid") is not True,
            bool(row.get("bar_stale")),
            row.get(rank_by) is None,
            (
                float(row.get(rank_by) if row.get(rank_by) is not None else missing_value)
                if order == "asc"
                else -(float(row.get(rank_by) or 0.0))
            ),
            row.get("symbol") or "",
        )
    )


_RANK_BY_ALIASES = {
    "abs_price_change": "abs_price_change_pct",
    "price_change": "price_change_pct",
    "spread": "spread_pct",
}

_SYMBOLS_TOP_MARKETS_INTERNAL_RANK_BY = {
    "abs_price_change_pct": "abs_price_change",
    "price_change_pct": "price_change",
    "tick_volume": "volume",
    "spread_pct": "spread",
}


_MARKET_SCAN_RANK_BY_CHOICES = (
    "abs_price_change_pct",
    "abs_price_change",
    "price_change_pct",
    "price_change",
    "tick_volume",
    "rsi",
    "spread_pct",
    "spread",
)


def _normalize_market_scan_rank_by(value: Any) -> tuple[str, Optional[str]]:
    raw_value = str(value or "abs_price_change_pct").strip().lower()
    return _RANK_BY_ALIASES.get(raw_value, raw_value), raw_value


def _normalize_market_scan_rank_order(value: Any) -> tuple[str, Optional[str]]:
    raw_value = str(value or "auto").strip().lower()
    aliases = {"ascending": "asc", "descending": "desc"}
    return aliases.get(raw_value, raw_value), raw_value


def _market_scan_effective_rank_order(
    rank_by: str,
    *,
    rank_order: str,
    rsi_above: Optional[float] = None,
    rsi_below: Optional[float] = None,
) -> str:
    order = str(rank_order or "auto").strip().lower()
    if order != "auto":
        return order
    if rank_by == "spread_pct":
        return "asc"
    if rank_by == "rsi" and rsi_below is not None and rsi_above is None:
        return "asc"
    return "desc"


def _market_scan_ranking_label(
    rank_by: str,
    *,
    rank_order: str,
    rsi_above: Optional[float] = None,
    rsi_below: Optional[float] = None,
) -> str:
    order = _market_scan_effective_rank_order(
        rank_by,
        rank_order=rank_order,
        rsi_above=rsi_above,
        rsi_below=rsi_below,
    )
    if rank_by == "abs_price_change_pct":
        return (
            "smallest_abs_price_change_pct"
            if order == "asc"
            else "largest_abs_price_change_pct"
        )
    if rank_by == "price_change_pct":
        return "lowest_price_change_pct" if order == "asc" else "highest_price_change_pct"
    if rank_by == "gap_pct":
        return "lowest_gap_pct" if order == "asc" else "highest_gap_pct"
    if rank_by == "tick_volume":
        return "lowest_tick_volume" if order == "asc" else "highest_tick_volume"
    if rank_by == "spread_pct":
        return "lowest_spread_pct" if order == "asc" else "highest_spread_pct"
    if rank_by == "rsi":
        return "lowest_rsi" if order == "asc" else "highest_rsi"
    return str(rank_by)


@mcp.tool()
def symbols_top_markets(  # noqa: C901
    rank_by: Literal[
        "all",
        "spread",
        "price_change",
        "spread_pct",
        "tick_volume",
        "price_change_pct",
        "abs_price_change",
        "abs_price_change_pct",
    ] = "abs_price_change_pct",  # type: ignore
    limit: Annotated[int, Field(ge=1)] = 10,
    universe: Literal["visible", "all"] = "visible",  # type: ignore
    timeframe: TimeframeLiteral = "H1",
    group: Optional[str] = None,
    category: Optional[str] = None,
    candidate_offset: Annotated[int, Field(ge=0)] = 0,
    candidate_limit: Annotated[
        Optional[int], Field(ge=1, le=_TOP_MARKETS_MAX_CANDIDATES)
    ] = None,
    detail: DetailLiteral = "compact",
) -> Dict[str, Any]:
    """Quick MT5 market overview ranked by spread, tick volume, or price change.

    Defaults to visible tradable symbols for responsiveness. Set `group` or
    `category` (forex, crypto, indices, commodities, stocks, bonds, etfs) to keep
    rankings inside a comparable asset universe. Set `universe="all"` to include
    hidden tradable symbols too; that mode is slower because MT5 may need to
    activate quotes for instruments that are not already visible. Defaults to a
    single absolute-price-change leaderboard; set `rank_by="price_change"` for
    gainers only or `rank_by="all"` for spread, volume, signed price-change,
    and absolute price-change leaderboards. Volume and price-change rankings use
    the most recent completed bar on `timeframe`. Uses compact leaderboard rows by default. Set
    `detail="full"` for the expanded row shape and collection metadata. Use
    `candidate_limit` and `candidate_offset` to scan a deterministic partition
    of a category/group larger than 250 symbols; merge each partition's top-N
    rows by the selected ranking value to obtain the global top-N. Use
    `market_scan` instead when you need explicit symbol inputs, RSI/SMA filters,
    or a single flat scanner table. Locked or invalid quotes are marked unsafe
    and rank after valid two-sided quotes in spread leaderboards.
    """

    detail_mode = normalize_output_verbosity_detail(detail, default="compact")

    def _run() -> Dict[str, Any]:  # noqa: C901
        try:
            if limit is not None:
                try:
                    if int(limit) <= 0:
                        return {"error": "limit must be a positive integer when provided."}
                except (TypeError, ValueError):
                    return {"error": "limit must be a positive integer when provided."}
            raw_rank_by_value = str(rank_by or "abs_price_change_pct").strip().lower()
            if raw_rank_by_value == "volume":
                return {
                    "error": (
                        "rank_by='volume' is ambiguous; use rank_by='tick_volume' "
                        "for broker tick counts."
                    )
                }
            rank_by_value = _RANK_BY_ALIASES.get(
                raw_rank_by_value,
                raw_rank_by_value,
            )
            rank_kind = _SYMBOLS_TOP_MARKETS_INTERNAL_RANK_BY.get(
                rank_by_value,
                rank_by_value,
            )
            if rank_kind not in {
                "all",
                "spread",
                "volume",
                "price_change",
                "abs_price_change",
            }:
                return {
                    "error": (
                        "rank_by must be one of: all, spread/spread_pct, "
                        "tick_volume, price_change/price_change_pct, "
                        "abs_price_change/abs_price_change_pct."
                    )
                }

            universe_value = str(universe or "visible").strip().lower()
            if universe_value not in {"visible", "all"}:
                return {"error": "universe must be 'visible' or 'all'."}
            group_filter = _normalize_group_path_query(group) if group else None
            category_filter = _normalize_symbol_category_filter(category)
            if category and not category_filter:
                return {
                    "error": (
                        "category must be one of forex, crypto, indices, "
                        "commodities, stocks, bonds, or etfs."
                    )
                }

            timeframe_value = str(timeframe or "H1").strip().upper()
            needs_bar_data = rank_kind in {
                "all",
                "volume",
                "price_change",
                "abs_price_change",
            }
            if needs_bar_data and timeframe_value not in TIMEFRAME_MAP:
                return {"error": invalid_timeframe_error(timeframe_value, TIMEFRAME_MAP)}
            mt5_timeframe = TIMEFRAME_MAP.get(timeframe_value)

            mt5_gateway = create_mt5_gateway(
                adapter=mt5,
                ensure_connection_impl=ensure_mt5_connection_or_raise,
            )
            mt5_gateway.ensure_connection()
            source = build_mt5_source_provenance(mt5_gateway)
            spread_cost_currency = (
                account_currency_from_gateway(mt5_gateway)
                if rank_kind != "volume"
                else None
            )

            raw_symbols = mt5_gateway.symbols_get()
            if raw_symbols is None:
                return {"error": f"Failed to get symbols: {mt5_gateway.last_error()}"}
            all_symbols = list(raw_symbols)

            selected_symbols = [
                symbol
                for symbol in all_symbols
                if _market_scan_is_tradable(symbol)
                and (universe_value == "all" or bool(getattr(symbol, "visible", False)))
            ]
            filters: Dict[str, Any] = {}
            if group_filter:
                resolved_groups, group_error = _resolve_market_scan_group_path(
                    selected_symbols,
                    group_filter,
                )
                if group_error or not resolved_groups:
                    return {"error": group_error or f"No symbol group matched '{group_filter}'."}
                resolved_group_set = {
                    _normalize_group_path_query(group_path).lower()
                    for group_path in resolved_groups
                }
                selected_symbols = [
                    symbol
                    for symbol in selected_symbols
                    if _normalize_group_path_query(
                        str(_extract_group_path_util(symbol) or "")
                    ).lower()
                    in resolved_group_set
                ]
                filters["group"] = (
                    resolved_groups[0]
                    if len(resolved_groups) == 1
                    else str(group_filter).strip()
                )
                if len(resolved_groups) > 1:
                    filters["groups"] = resolved_groups
            if category_filter:
                selected_symbols = [
                    symbol
                    for symbol in selected_symbols
                    if _symbol_category(symbol) == category_filter
                ]
                filters["category"] = category_filter
            selected_symbols = sorted(
                selected_symbols,
                key=lambda symbol: _case_insensitive_sort_key(getattr(symbol, "name", "")),
            )

            try:
                candidate_offset_value = int(candidate_offset or 0)
            except (TypeError, ValueError):
                return {"error": "candidate_offset must be a non-negative integer."}
            if candidate_offset_value < 0:
                return {"error": "candidate_offset must be >= 0."}
            candidate_limit_value: Optional[int] = None
            if candidate_limit is not None:
                try:
                    candidate_limit_value = int(candidate_limit)
                except (TypeError, ValueError):
                    return {
                        "error": (
                            "candidate_limit must be an integer between 1 and "
                            f"{_TOP_MARKETS_MAX_CANDIDATES}."
                        )
                    }
                if not 1 <= candidate_limit_value <= _TOP_MARKETS_MAX_CANDIDATES:
                    return {
                        "error": (
                            "candidate_limit must be between 1 and "
                            f"{_TOP_MARKETS_MAX_CANDIDATES}."
                        )
                    }

            candidate_total = len(selected_symbols)
            partition_requested = bool(
                candidate_limit_value is not None or candidate_offset_value
            )
            if candidate_total > _TOP_MARKETS_MAX_CANDIDATES and not partition_requested:
                return {
                    "error": (
                        f"The filtered universe contains {candidate_total} candidates, "
                        f"above the safe synchronous cap of {_TOP_MARKETS_MAX_CANDIDATES}. "
                        f"Scan it deterministically with candidate_limit="
                        f"{_TOP_MARKETS_MAX_CANDIDATES} and candidate_offset=0, then "
                        "increment candidate_offset until candidate_page.has_more is false. "
                        "Merge the partition top-N rows by the requested ranking value."
                    ),
                    "error_code": "candidate_universe_too_large",
                    "candidate_count": candidate_total,
                    "candidate_cap": _TOP_MARKETS_MAX_CANDIDATES,
                    "filters": filters,
                    "retry": {
                        "candidate_limit": _TOP_MARKETS_MAX_CANDIDATES,
                        "candidate_offset": 0,
                    },
                }

            if partition_requested:
                page_size = candidate_limit_value or _TOP_MARKETS_MAX_CANDIDATES
                selected_symbols = selected_symbols[
                    candidate_offset_value : candidate_offset_value + page_size
                ]

            limit_value = _normalize_limit(limit) or 10
            started_at = time.perf_counter()

            spread_rows: List[Dict[str, Any]] = []
            volume_rows: List[Dict[str, Any]] = []
            price_change_rows: List[Dict[str, Any]] = []
            metric_issues: Dict[str, List[Dict[str, str]]] = {
                "spread": [],
                "volume": [],
                "price_change": [],
            }
            metric_skips: Dict[str, int] = {
                "spread": 0,
                "volume": 0,
                "price_change": 0,
            }

            def _record_issue(metric_name: str, symbol_name: str, reason: str) -> None:
                metric_skips[metric_name] += 1
                if len(metric_issues[metric_name]) < 10:
                    metric_issues[metric_name].append(
                        {"symbol": symbol_name, "reason": reason}
                    )

            def _collect_for_symbol(symbol: Any) -> None:
                symbol_name = str(getattr(symbol, "name", "") or "")

                spread_row = None
                if rank_kind != "volume":
                    spread_row, spread_error = _build_market_scan_spread_row(
                        symbol,
                        mt5_gateway,
                        spread_cost_currency=spread_cost_currency,
                    )
                    if spread_error and rank_kind in {"all", "spread"}:
                        _record_issue("spread", symbol_name, spread_error)
                    elif spread_row is not None and rank_kind in {"all", "spread"}:
                        spread_rows.append(spread_row)

                if needs_bar_data and mt5_timeframe is not None:
                    bar_row, bar_error = _build_market_scan_bar_row(
                        symbol,
                        timeframe=timeframe_value,
                        mt5_timeframe=mt5_timeframe,
                    )
                    if bar_error:
                        if rank_kind in {"all", "volume"}:
                            _record_issue("volume", symbol_name, bar_error)
                        if rank_kind in {
                            "all",
                            "price_change",
                            "abs_price_change",
                        }:
                            _record_issue("price_change", symbol_name, bar_error)
                    elif bar_row is not None:
                        combined_bar_row = dict(spread_row or {})
                        combined_bar_row.update(bar_row)
                        if rank_kind in {"all", "volume"}:
                            volume_rows.append(dict(combined_bar_row))
                        if rank_kind in {
                            "all",
                            "price_change",
                            "abs_price_change",
                        }:
                            price_change_rows.append(dict(combined_bar_row))

            for symbol in selected_symbols:
                symbol_name = str(getattr(symbol, "name", "") or "")
                is_hidden = not bool(getattr(symbol, "visible", False))
                if universe_value == "all" and is_hidden:
                    with _symbol_ready_guard(symbol_name, info_before=symbol) as (err, _):
                        if err:
                            if rank_kind in {"all", "spread"}:
                                _record_issue("spread", symbol_name, err)
                            if rank_kind in {"all", "volume"}:
                                _record_issue("volume", symbol_name, err)
                            if rank_kind in {
                                "all",
                                "price_change",
                                "abs_price_change",
                            }:
                                _record_issue("price_change", symbol_name, err)
                            continue
                        _collect_for_symbol(symbol)
                    continue
                _collect_for_symbol(symbol)

            spread_rows.sort(
                key=lambda row: (
                    row.get("spread_valid") is not True,
                    bool(row.get("data_stale")),
                    row.get("spread_pct") is None,
                    row.get("spread_pct") if row.get("spread_pct") is not None else float("inf"),
                    row.get("symbol") or "",
                )
            )
            volume_rows.sort(
                key=lambda row: (
                    bool(row.get("data_stale")) or bool(row.get("bar_stale")),
                    row.get("tick_volume") is None,
                    -(row.get("tick_volume") or 0),
                    row.get("symbol") or "",
                )
            )
            price_change_rows.sort(
                key=lambda row: (
                    bool(row.get("data_stale")) or bool(row.get("bar_stale")),
                    row.get("price_change_pct") is None,
                    (
                        -abs(float(row.get("price_change_pct") or 0.0))
                        if rank_kind == "abs_price_change"
                        else -(row.get("price_change_pct") or 0.0)
                    ),
                    row.get("symbol") or "",
                )
            )
            abs_price_change_rows = [dict(row) for row in price_change_rows]
            abs_price_change_rows.sort(
                key=lambda row: (
                    bool(row.get("data_stale")) or bool(row.get("bar_stale")),
                    row.get("price_change_pct") is None,
                    -abs(float(row.get("price_change_pct") or 0.0)),
                    row.get("symbol") or "",
                )
            )

            evaluated_counts = {
                "spread": len(spread_rows),
                "volume": len(volume_rows),
                "price_change": len(price_change_rows),
                "abs_price_change": len(abs_price_change_rows),
            }

            spread_rows = _top_market_rows_with_data_context(
                "spread",
                spread_rows[:limit_value],
                timeframe=timeframe_value,
            )
            volume_rows = _top_market_rows_with_data_context(
                "volume",
                volume_rows[:limit_value],
                timeframe=timeframe_value,
            )
            price_change_rows = _top_market_rows_with_data_context(
                "price_change",
                price_change_rows[:limit_value],
                timeframe=timeframe_value,
            )
            abs_price_change_rows = _top_market_rows_with_data_context(
                "price_change",
                abs_price_change_rows[:limit_value],
                timeframe=timeframe_value,
            )

            def _scope_fields(
                metric_name: str,
                rows: List[Dict[str, Any]],
            ) -> Dict[str, Any]:
                available_count = int(evaluated_counts[metric_name])
                returned_count = int(len(rows))
                fields: Dict[str, Any] = {
                    "requested_limit": int(limit_value),
                    "universe_size": int(len(selected_symbols)),
                    "available_count": available_count,
                }
                if detail_mode == "full":
                    fields["returned_count"] = returned_count
                if returned_count < int(limit_value):
                    fields["note"] = (
                        f"Requested {int(limit_value)} rows but only "
                        f"{available_count} symbols provided {metric_name} data "
                        f"in the {universe_value} universe."
                    )
                return fields

            scan_meta = {"success": True, "source": source}
            if filters:
                scan_meta["filters"] = filters
            if partition_requested:
                scan_meta["ranking_scope"] = "candidate_partition"
                scan_meta["candidate_page"] = build_pagination_meta(
                    total=candidate_total,
                    returned=len(selected_symbols),
                    offset=candidate_offset_value,
                    limit=candidate_limit_value or _TOP_MARKETS_MAX_CANDIDATES,
                )
                scan_meta["candidate_page"]["aggregation_required"] = bool(
                    candidate_total > len(selected_symbols)
                )
                scan_meta["candidate_page"]["aggregation_note"] = (
                    "Merge top-N rows from every candidate partition by the selected "
                    "ranking value to compute the global leaderboard."
                )
            if detail_mode == "full":
                scan_meta.update(
                    {
                        "rank_by": rank_by_value,
                        "rank_by_input": raw_rank_by_value
                        if raw_rank_by_value != rank_by_value
                        else None,
                        "limit": limit_value,
                        "universe": universe_value,
                        "detail": detail_mode,
                        "timeframe": timeframe_value if needs_bar_data else None,
                        "timeframe_requested": timeframe_value,
                        "timeframe_used": timeframe_value if needs_bar_data else None,
                        "scanned_symbols": len(selected_symbols),
                        "query_latency_ms": round(
                            (time.perf_counter() - started_at) * 1000.0,
                            3,
                        ),
                    }
                )

            if rank_kind == "spread":
                out = _market_scan_table(
                    _top_markets_headers("spread", detail_mode=detail_mode),
                    spread_rows,
                    include_contract_meta=detail_mode == "full",
                )
                out.update(scan_meta)
                out["ranking"] = "lowest_spread"
                out.update(_scope_fields("spread", spread_rows))
                out.update(
                    _market_scan_freshness_summary(
                        spread_rows,
                        include_stale_symbols=detail_mode == "full",
                    )
                )
                _attach_top_markets_units(out, spread_rows)
                if detail_mode == "full":
                    out["evaluated_symbols"] = evaluated_counts["spread"]
                    out["skipped_symbols"] = metric_skips["spread"]
                    out["skipped_examples"] = metric_issues["spread"]
                return out

            if rank_kind == "volume":
                out = _market_scan_table(
                    _top_markets_headers("volume", detail_mode=detail_mode),
                    volume_rows,
                    include_contract_meta=detail_mode == "full",
                )
                out.update(scan_meta)
                out["ranking"] = "highest_tick_volume"
                out.update(_scope_fields("volume", volume_rows))
                out.update(
                    _market_scan_freshness_summary(
                        volume_rows,
                        include_stale_symbols=detail_mode == "full",
                    )
                )
                _attach_top_markets_units(out, volume_rows)
                if detail_mode == "full":
                    out["evaluated_symbols"] = evaluated_counts["volume"]
                    out["skipped_symbols"] = metric_skips["volume"]
                    out["skipped_examples"] = metric_issues["volume"]
                return out

            if rank_kind in {"price_change", "abs_price_change"}:
                out = _market_scan_table(
                    _top_markets_headers("price_change", detail_mode=detail_mode),
                    price_change_rows,
                    include_contract_meta=detail_mode == "full",
                )
                out.update(scan_meta)
                out["ranking"] = (
                    "largest_abs_price_change_pct"
                    if rank_kind == "abs_price_change"
                    else "highest_price_change_pct"
                )
                out["price_change_basis"] = (
                    "previous_completed_close_to_latest_completed_close"
                )
                out["price_change_period"] = {
                    "bars": 1,
                    "timeframe": timeframe,
                    "bar_state": "completed",
                }
                out.update(_scope_fields("price_change", price_change_rows))
                out.update(
                    _market_scan_freshness_summary(
                        price_change_rows,
                        include_stale_symbols=detail_mode == "full",
                    )
                )
                _attach_top_markets_units(out, price_change_rows)
                if detail_mode == "full":
                    out["evaluated_symbols"] = evaluated_counts["price_change"]
                    out["skipped_symbols"] = metric_skips["price_change"]
                    out["skipped_examples"] = metric_issues["price_change"]
                return attach_collection_contract(
                    out,
                    collection_kind="table",
                    rows=out.get("data"),
                    include_contract_meta=detail_mode == "full",
                )

            all_rows = [
                *_ranked_top_market_rows("lowest_spread", spread_rows),
                *_ranked_top_market_rows("highest_tick_volume", volume_rows),
                *_ranked_top_market_rows("highest_price_change_pct", price_change_rows),
                *_ranked_top_market_rows(
                    "largest_abs_price_change_pct",
                    abs_price_change_rows,
                ),
            ]
            out = _market_scan_table(
                _top_markets_all_headers(detail_mode=detail_mode),
                all_rows,
                include_contract_meta=detail_mode == "full",
            )
            out.update(scan_meta)
            out["ranking"] = "all"
            out["price_change_basis"] = (
                "previous_completed_close_to_latest_completed_close"
            )
            out["price_change_period"] = {
                "bars": 1,
                "timeframe": timeframe,
                "bar_state": "completed",
            }
            out["rank_categories"] = [
                "lowest_spread",
                "highest_tick_volume",
                "highest_price_change_pct",
                "largest_abs_price_change_pct",
            ]
            out["rank_by_categories"] = {
                "spread": "lowest_spread",
                "spread_pct": "lowest_spread",
                "tick_volume": "highest_tick_volume",
                "price_change": "highest_price_change_pct",
                "price_change_pct": "highest_price_change_pct",
                "abs_price_change": "largest_abs_price_change_pct",
                "abs_price_change_pct": "largest_abs_price_change_pct",
            }
            out["requested_limit"] = int(limit_value)
            out["universe_size"] = int(len(selected_symbols))
            out["returned_counts"] = {
                "lowest_spread": len(spread_rows),
                "highest_tick_volume": len(volume_rows),
                "highest_price_change_pct": len(price_change_rows),
                "largest_abs_price_change_pct": len(abs_price_change_rows),
            }
            out["available_counts"] = {
                "lowest_spread": evaluated_counts["spread"],
                "highest_tick_volume": evaluated_counts["volume"],
                "highest_price_change_pct": evaluated_counts["price_change"],
                "largest_abs_price_change_pct": evaluated_counts[
                    "abs_price_change"
                ],
            }
            out.update(
                _market_scan_freshness_summary(
                    all_rows,
                    include_stale_symbols=detail_mode == "full",
                )
            )
            ranking_rows = {
                "lowest_spread": spread_rows,
                "highest_tick_volume": volume_rows,
                "highest_price_change_pct": price_change_rows,
                "largest_abs_price_change_pct": abs_price_change_rows,
            }
            ranking_context: Dict[str, Dict[str, Any]] = {}
            ranking_times: List[str] = []
            for ranking_name, rows_for_ranking in ranking_rows.items():
                freshness_context = _market_scan_freshness_summary(
                    rows_for_ranking,
                    include_stale_symbols=False,
                )
                context = {
                    key: freshness_context[key]
                    for key in (
                        "data_as_of",
                        "freshness",
                        "stale_rows",
                        "stale_bar_rows",
                        "unsafe_quote_rows",
                    )
                    if key in freshness_context
                }
                if rows_for_ranking:
                    context["data_source"] = rows_for_ranking[0].get("data_source")
                    context["timeframe"] = rows_for_ranking[0].get("timeframe")
                ranking_time = str(context.get("data_as_of") or "").strip()
                if ranking_time:
                    ranking_times.append(ranking_time)
                ranking_context[ranking_name] = context
            out["ranking_context"] = ranking_context
            if ranking_times:
                out["data_as_of_range"] = {
                    "oldest": min(ranking_times),
                    "newest": max(ranking_times),
                }
                out["data_as_of"] = max(ranking_times)
                out["data_as_of_basis"] = "latest_source_timestamp_across_rankings"
            if detail_mode == "full":
                out["scan_stats"] = {
                    "spread": {
                        "evaluated_symbols": evaluated_counts["spread"],
                        "skipped_symbols": metric_skips["spread"],
                        "skipped_examples": metric_issues["spread"],
                    },
                    "volume": {
                        "evaluated_symbols": evaluated_counts["volume"],
                        "skipped_symbols": metric_skips["volume"],
                        "skipped_examples": metric_issues["volume"],
                    },
                    "price_change": {
                        "evaluated_symbols": evaluated_counts["price_change"],
                        "skipped_symbols": metric_skips["price_change"],
                        "skipped_examples": metric_issues["price_change"],
                    },
                    "abs_price_change": {
                        "evaluated_symbols": evaluated_counts["abs_price_change"],
                        "skipped_symbols": metric_skips["price_change"],
                        "skipped_examples": metric_issues["price_change"],
                    },
                }
            _attach_top_markets_units(out, all_rows)
            return out
        except MT5ConnectionError as exc:
            return {"error": str(exc)}
        except Exception as exc:
            return {"error": f"Error scanning top markets: {str(exc)}"}

    return run_logged_operation(
        logger,
        operation="symbols_top_markets",
        rank_by=rank_by,
        limit=limit,
        universe=universe,
        timeframe=timeframe,
        detail=detail_mode,
        func=_run,
    )


_MARKET_SCAN_PRESETS: Dict[str, Dict[str, Any]] = {
    "oversold": {"rsi_below": 30.0, "min_tick_volume": 1000, "rank_by": "rsi"},
    "overbought": {"rsi_above": 70.0, "min_tick_volume": 1000, "rank_by": "rsi"},
    "high_volume": {"rank_by": "tick_volume"},
    "tight_spread": {"max_spread_pct": 0.01, "min_tick_volume": 500, "rank_by": "spread_pct"},
    "gap_up": {"min_gap_pct": 2.0, "rank_by": "gap_pct"},
    "gap_down": {"max_gap_pct": -2.0, "rank_by": "gap_pct"},
}


@mcp.tool()
def market_scan(  # noqa: C901
    symbols: Optional[str] = None,
    group: Optional[str] = None,
    preset: Optional[str] = None,
    limit: Annotated[int, Field(ge=1)] = 10,
    offset: int = 0,
    universe: Literal["visible", "all"] = "visible",  # type: ignore
    timeframe: TimeframeLiteral = "H1",
    detail: DetailLiteral = "compact",
    lookback: int = 100,
    rsi_length: int = 14,
    sma_period: int = 20,
    min_price_change_pct: Optional[float] = None,
    max_price_change_pct: Optional[float] = None,
    max_spread_pct: Optional[float] = None,
    min_tick_volume: Optional[int] = None,
    rsi_below: Optional[float] = None,
    rsi_above: Optional[float] = None,
    price_vs_sma: Optional[Literal["above", "below"]] = None,  # type: ignore
    rank_by: Literal["abs_price_change_pct", "abs_price_change", "price_change_pct", "price_change", "tick_volume", "rsi", "spread_pct", "spread"] = "abs_price_change_pct",  # type: ignore
    rank_order: Literal["auto", "asc", "desc", "ascending", "descending"] = "auto",  # type: ignore
    quote_usable_only: Optional[bool] = None,
) -> Dict[str, Any]:
    """Filtered MT5 market scanner with one flat table and technical filters.

    Pass `symbols` for one instrument or a comma-separated list. `price_change_pct`
    is the return from the previous completed bar close to the latest completed
    bar close. `data` is the canonical flat row payload. Compact detail is the
    default; use
    `detail="full"` when you also want the explicit `columns` ordering hint
    for compatibility. Broad scans use the visible universe; `universe="all"`
    must be combined with `symbols` or `group` to avoid unbounded hidden-symbol
    activation. Use `symbols_top_markets` for a quick all-market overview with
    separate spread, volume, and mover leaderboards. `quote_usable_only` defaults
    to true for spread rankings and the `tight_spread` preset, excluding stale,
    future, locked, inverted, and one-sided quotes before pagination. Set it
    explicitly for other rankings. Locked or invalid quotes cannot satisfy a
    maximum-spread filter.
    """

    detail_mode = normalize_output_verbosity_detail(detail, default="compact")
    preset_value = str(preset or "").strip().lower().replace("-", "_")
    preset_error = None
    preset_config = _MARKET_SCAN_PRESETS.get(preset_value) if preset_value else None
    min_gap_pct: Optional[float] = None
    max_gap_pct: Optional[float] = None
    if preset_value and preset_config is None:
        preset_error = (
            "preset must be one of: "
            + ", ".join(sorted(_MARKET_SCAN_PRESETS))
            + "."
        )
    elif preset_config:
        min_gap_pct = preset_config.get("min_gap_pct")
        max_gap_pct = preset_config.get("max_gap_pct")
        if min_price_change_pct is None and "min_price_change_pct" in preset_config:
            min_price_change_pct = preset_config["min_price_change_pct"]
        if max_price_change_pct is None and "max_price_change_pct" in preset_config:
            max_price_change_pct = preset_config["max_price_change_pct"]
        if max_spread_pct is None and "max_spread_pct" in preset_config:
            max_spread_pct = preset_config["max_spread_pct"]
        if min_tick_volume is None and "min_tick_volume" in preset_config:
            min_tick_volume = preset_config["min_tick_volume"]
        if rsi_below is None and "rsi_below" in preset_config:
            rsi_below = preset_config["rsi_below"]
        if rsi_above is None and "rsi_above" in preset_config:
            rsi_above = preset_config["rsi_above"]
        if rank_by in {None, "abs_price_change_pct"} and "rank_by" in preset_config:
            rank_by = preset_config["rank_by"]

    normalized_rank_by, _ = _normalize_market_scan_rank_by(rank_by)
    quote_usable_only_value = (
        normalized_rank_by == "spread_pct" or preset_value == "tight_spread"
        if quote_usable_only is None
        else bool(quote_usable_only)
    )

    def _run() -> Dict[str, Any]:  # noqa: C901
        request: Dict[str, Any] = {
            "symbols": symbols,
            "group": group,
            "preset": preset_value or None,
            "limit": limit,
            "offset": offset,
            "universe": universe,
            "timeframe": timeframe,
            "detail": detail_mode,
            "lookback": lookback,
            "rank_by": rank_by,
            "rank_order": rank_order,
            "quote_usable_only": quote_usable_only_value,
            "filters": {
                key: value
                for key, value in {
                    "min_price_change_pct": min_price_change_pct,
                    "max_price_change_pct": max_price_change_pct,
                    "min_gap_pct": min_gap_pct,
                    "max_gap_pct": max_gap_pct,
                    "max_spread_pct": max_spread_pct,
                    "min_tick_volume": min_tick_volume,
                    "rsi_below": rsi_below,
                    "rsi_above": rsi_above,
                    "price_vs_sma": price_vs_sma,
                    "rsi_length": rsi_length,
                    "sma_period": sma_period,
                }.items()
                if value is not None
            },
        }
        try:
            if preset_error:
                return _market_scan_error(
                    preset_error,
                    code="invalid_input",
                    request=request,
                )

            universe_value = str(universe or "visible").strip().lower()
            request["universe"] = universe_value
            if universe_value not in {"visible", "all"}:
                return _market_scan_error(
                    "universe must be 'visible' or 'all'.",
                    code="invalid_input",
                    request=request,
                )

            symbols_value = str(symbols or "").strip()
            symbols_filter = symbols_value or None
            if universe_value == "all" and not symbols_filter and not group:
                return _market_scan_error(
                    (
                        "market_scan universe='all' requires symbols or group "
                        "to bound the scan. Use universe='visible' for broad scans "
                        "or symbols_top_markets for quick all-market leaderboards."
                    ),
                    code="invalid_input",
                    request=request,
                )

            timeframe_value = str(timeframe or "H1").strip().upper()
            request["timeframe"] = timeframe_value
            if timeframe_value not in TIMEFRAME_MAP:
                return _market_scan_error(
                    invalid_timeframe_error(timeframe_value, TIMEFRAME_MAP),
                    code="invalid_timeframe",
                    request=request,
                )
            mt5_timeframe = TIMEFRAME_MAP[timeframe_value]

            rank_by_value, rank_by_input = _normalize_market_scan_rank_by(rank_by)
            request["rank_by"] = rank_by_value
            if rank_by_input != rank_by_value:
                request["rank_by_input"] = rank_by_input
            if rank_by_value not in {"abs_price_change_pct", "price_change_pct", "gap_pct", "tick_volume", "rsi", "spread_pct"}:
                return _market_scan_error(
                    (
                        "rank_by must be one of: "
                        f"{', '.join(_MARKET_SCAN_RANK_BY_CHOICES)}."
                    ),
                    code="invalid_input",
                    request=request,
                )
            rank_order_value, rank_order_input = _normalize_market_scan_rank_order(rank_order)
            request["rank_order"] = rank_order_value
            if rank_order_input != rank_order_value:
                request["rank_order_input"] = rank_order_input
            if rank_order_value not in {"auto", "asc", "desc"}:
                return _market_scan_error(
                    "rank_order must be one of: auto, asc, desc, ascending, descending.",
                    code="invalid_input",
                    request=request,
                )

            price_vs_sma_value = None
            if price_vs_sma is not None:
                price_vs_sma_value = str(price_vs_sma).strip().lower()
                request["filters"] = {
                    **dict(request.get("filters", {})),
                    "price_vs_sma": price_vs_sma_value,
                }
                if price_vs_sma_value not in {"above", "below"}:
                    return _market_scan_error(
                        "price_vs_sma must be 'above' or 'below'.",
                        code="invalid_input",
                        request=request,
                    )

            try:
                lookback_value = int(lookback)
                rsi_length_value = int(rsi_length)
                sma_period_value = int(sma_period)
            except Exception:
                return _market_scan_error(
                    "lookback, rsi_length, and sma_period must be integers.",
                    code="invalid_input",
                    request=request,
                )
            request["lookback"] = lookback_value
            request["filters"] = {
                **dict(request.get("filters", {})),
                "rsi_length": rsi_length_value,
                "sma_period": sma_period_value,
            }
            if lookback_value < 2:
                return _market_scan_error(
                    "lookback must be at least 2.",
                    code="invalid_input",
                    request=request,
                )
            if rsi_length_value < 1:
                return _market_scan_error(
                    "rsi_length must be at least 1.",
                    code="invalid_input",
                    request=request,
                )
            if sma_period_value < 1:
                return _market_scan_error(
                    "sma_period must be at least 1.",
                    code="invalid_input",
                    request=request,
                )

            required_lookback = 2
            if rank_by_value == "rsi" or rsi_above is not None or rsi_below is not None:
                required_lookback = max(required_lookback, rsi_length_value + 1)
            if price_vs_sma_value is not None:
                required_lookback = max(required_lookback, sma_period_value)
            if lookback_value < required_lookback:
                return _market_scan_error(
                    (
                        f"lookback={lookback_value} is too small for the requested filters; "
                        f"need at least {required_lookback} bars."
                    ),
                    code="invalid_input",
                    request=request,
                )

            if limit is None:
                limit_value = 10
            else:
                try:
                    limit_value = int(limit)
                except Exception:
                    limit_value = 0
                if limit_value <= 0:
                    return _market_scan_error(
                        "limit must be a positive integer.",
                        code="invalid_input",
                        request=request,
                    )
            request["limit"] = limit_value
            try:
                offset_value = int(offset or 0)
            except Exception:
                return _market_scan_error(
                    "offset must be a non-negative integer.",
                    code="invalid_input",
                    request=request,
                )
            if offset_value < 0:
                return _market_scan_error(
                    "offset must be >= 0.",
                    code="invalid_input",
                    request=request,
                )
            request["offset"] = offset_value

            for filter_name, filter_value in (
                ("min_price_change_pct", min_price_change_pct),
                ("max_price_change_pct", max_price_change_pct),
                ("max_spread_pct", max_spread_pct),
                ("rsi_below", rsi_below),
                ("rsi_above", rsi_above),
            ):
                if filter_value is None:
                    continue
                numeric = _market_scan_float(filter_value)
                invalid = numeric is None
                if filter_name == "max_spread_pct":
                    invalid = invalid or numeric < 0.0
                else:
                    invalid = invalid or not 0.0 <= numeric <= 100.0
                if invalid:
                    if filter_name == "max_spread_pct":
                        expected = "a non-negative percentage"
                    elif filter_name in {"rsi_below", "rsi_above"}:
                        expected = "between 0 and 100"
                    else:
                        expected = "a finite number"
                    return _market_scan_error(
                        f"{filter_name} must be {expected}.",
                        code="invalid_input",
                        request=request,
                    )
            min_change = _market_scan_float(min_price_change_pct)
            max_change = _market_scan_float(max_price_change_pct)
            if (
                min_change is not None
                and max_change is not None
                and min_change > max_change
            ):
                return _market_scan_error(
                    "min_price_change_pct must be <= max_price_change_pct.",
                    code="contradictory_filters",
                    request=request,
                )
            rsi_floor = _market_scan_float(rsi_above)
            rsi_ceiling = _market_scan_float(rsi_below)
            if (
                rsi_floor is not None
                and rsi_ceiling is not None
                and rsi_floor >= rsi_ceiling
            ):
                return _market_scan_error(
                    "rsi_above must be < rsi_below.",
                    code="contradictory_filters",
                    request=request,
                )
            if min_tick_volume is not None:
                try:
                    min_tick_volume_value = int(min_tick_volume)
                except Exception:
                    min_tick_volume_value = -1
                if min_tick_volume_value < 0:
                    return _market_scan_error(
                        "min_tick_volume must be a non-negative integer.",
                        code="invalid_input",
                        request=request,
                    )

            include_rsi = (
                detail_mode != "compact"
                or rank_by_value == "rsi"
                or rsi_above is not None
                or rsi_below is not None
            )
            include_sma = detail_mode != "compact" or price_vs_sma_value is not None
            signal_lookback = lookback_value if (include_rsi or include_sma) else 2

            mt5_gateway = create_mt5_gateway(
                adapter=mt5,
                ensure_connection_impl=ensure_mt5_connection_or_raise,
            )
            mt5_gateway.ensure_connection()
            spread_cost_currency = account_currency_from_gateway(mt5_gateway)

            raw_symbols = mt5_gateway.symbols_get()
            if raw_symbols is None:
                return _market_scan_error(
                    f"Failed to get symbols: {mt5_gateway.last_error()}",
                    code="data_fetch_failed",
                    request=request,
                )
            all_symbols = list(raw_symbols)

            selected_symbols, selection_meta, selection_error = _select_market_scan_symbols(
                all_symbols,
                symbols=symbols_filter,
                group=group,
                universe=universe_value,
            )
            if selection_error:
                error_code = "invalid_input"
                if group and selection_meta.get("symbols_input") is None:
                    error_code = "symbol_group_error"
                return _market_scan_error(
                    selection_error,
                    code=error_code,
                    request=request,
                )

            if selection_meta.get("symbols_input") is not None:
                request["symbols_input"] = selection_meta.get("symbols_input")
            request["scope"] = selection_meta.get("scope")
            if selection_meta.get("group") is not None:
                request["group"] = selection_meta.get("group")
            if selection_meta.get("groups") is not None:
                request["groups"] = selection_meta.get("groups")
            if selection_meta.get("requested_symbols") is not None:
                request["requested_symbols"] = selection_meta.get("requested_symbols")
            if selection_meta.get("missing_symbols") is not None:
                request["missing_symbols"] = selection_meta.get("missing_symbols")
            if len(selected_symbols) > _MARKET_SCAN_MAX_CANDIDATES:
                return _market_scan_error(
                    (
                        f"The filtered universe contains {len(selected_symbols)} candidates, "
                        f"above the safe synchronous cap of {_MARKET_SCAN_MAX_CANDIDATES}. "
                        "Narrow the exact scan with symbols or group."
                    ),
                    code="candidate_universe_too_large",
                    request=request,
                    stats={
                        "candidate_count": len(selected_symbols),
                        "candidate_cap": _MARKET_SCAN_MAX_CANDIDATES,
                    },
                )
            started_at = time.perf_counter()
            matched_rows: List[Dict[str, Any]] = []
            skipped_examples: List[Dict[str, str]] = []
            skipped_symbols = 0
            evaluated_symbols = 0
            quote_eligibility_excluded = 0
            quote_eligibility_reasons: Dict[str, int] = {}
            quote_eligibility_examples: List[Dict[str, str]] = []

            def _record_issue(symbol_name: str, reason: str) -> None:
                nonlocal skipped_symbols
                skipped_symbols += 1
                if len(skipped_examples) < 10:
                    skipped_examples.append({"symbol": symbol_name, "reason": reason})

            for missing_symbol in selection_meta.get("missing_symbols", []):
                _record_issue(missing_symbol, "Requested symbol not found.")

            def _evaluate_symbol(symbol_obj: Any) -> None:
                nonlocal evaluated_symbols, quote_eligibility_excluded
                symbol_name = str(getattr(symbol_obj, "name", "") or "")

                spread_row, spread_error = _build_market_scan_spread_row(
                    symbol_obj,
                    mt5_gateway,
                    spread_cost_currency=spread_cost_currency,
                )
                if spread_error or spread_row is None:
                    _record_issue(symbol_name, spread_error or "Spread data is unavailable.")
                    return

                signal_row, signal_error = _build_market_scan_signal_row(
                    symbol_obj,
                    timeframe=timeframe_value,
                    mt5_timeframe=mt5_timeframe,
                    lookback=signal_lookback,
                    rsi_length=rsi_length_value,
                    sma_period=sma_period_value,
                    include_rsi=include_rsi,
                    include_sma=include_sma,
                )
                if signal_error or signal_row is None:
                    _record_issue(symbol_name, signal_error or "Bar data is unavailable.")
                    return

                row = dict(spread_row)
                row.update(signal_row)
                _namespace_market_scan_quote_freshness(row)
                metric_error = _market_scan_missing_required_metric(
                    row,
                    rank_by=rank_by_value,
                    rsi_above=rsi_above,
                    rsi_below=rsi_below,
                    price_vs_sma=price_vs_sma_value,
                    max_spread_pct=max_spread_pct,
                    min_tick_volume=min_tick_volume,
                    min_price_change_pct=min_price_change_pct,
                    max_price_change_pct=max_price_change_pct,
                    min_gap_pct=min_gap_pct,
                    max_gap_pct=max_gap_pct,
                    rsi_length=rsi_length_value,
                    sma_period=sma_period_value,
                )
                if metric_error:
                    _record_issue(symbol_name, metric_error)
                    return

                evaluated_symbols += 1
                if (
                    quote_usable_only_value
                    and row.get("quote_usable_for_live_trading") is not True
                ):
                    reason = _market_scan_quote_exclusion_reason(row)
                    quote_eligibility_excluded += 1
                    quote_eligibility_reasons[reason] = (
                        quote_eligibility_reasons.get(reason, 0) + 1
                    )
                    if len(quote_eligibility_examples) < 10:
                        quote_eligibility_examples.append(
                            {"symbol": symbol_name, "reason": reason}
                        )
                    return
                if not _market_scan_row_matches_filters(
                    row,
                    min_price_change_pct=min_price_change_pct,
                    max_price_change_pct=max_price_change_pct,
                    max_spread_pct=max_spread_pct,
                    min_tick_volume=min_tick_volume,
                    rsi_below=rsi_below,
                    rsi_above=rsi_above,
                    price_vs_sma=price_vs_sma_value,
                    min_gap_pct=min_gap_pct,
                    max_gap_pct=max_gap_pct,
                ):
                    return
                matched_rows.append(row)

            for symbol_obj in selected_symbols:
                symbol_name = str(getattr(symbol_obj, "name", "") or "")
                is_hidden = not bool(getattr(symbol_obj, "visible", False))
                if is_hidden:
                    with _symbol_ready_guard(symbol_name, info_before=symbol_obj) as (err, _):
                        if err:
                            _record_issue(symbol_name, err)
                            continue
                        _evaluate_symbol(symbol_obj)
                    continue
                _evaluate_symbol(symbol_obj)

            effective_rank_order = _market_scan_effective_rank_order(
                rank_by_value,
                rank_order=rank_order_value,
                rsi_above=rsi_above,
                rsi_below=rsi_below,
            )
            _market_scan_sort_rows(
                matched_rows,
                rank_by=rank_by_value,
                rank_order=effective_rank_order,
                rsi_above=rsi_above,
                rsi_below=rsi_below,
            )
            total_matches = len(matched_rows)
            limited_rows = matched_rows[offset_value : offset_value + limit_value]

            full_headers = [
                "symbol",
                "group",
                "asset_class",
                "description",
                "timeframe",
                "time",
                "price_as_of",
                "price_freshness",
                "quote_time",
                "quote_as_of",
                "bid",
                "ask",
                "mid",
                "quote_age_seconds",
                "quote_age_anchor",
                "quote_age_metric",
                "quote_stale_after_seconds",
                "quote_stale",
                "quote_freshness_reason",
                "quote_timestamp_in_future",
                "quote_timestamp_skew_seconds",
                "quote_timestamp_warning",
                "quote_warning",
                "quote_freshness",
                "quote_usable_for_live_trading",
                "spread_valid",
                "spread_quality",
                "bar_age_seconds",
                "bar_freshness_anchor",
                "bar_freshness_metric",
                "bar_stale_after_seconds",
                "bar_age_hours",
                "bar_stale",
                "bar_market_status",
                "bar_market_status_reason",
                "bar_freshness_policy_relaxed",
                "bar_freshness",
                "bar_stale_warning",
                "previous_close",
                "close",
                "price_currency",
                "price_basis",
                "price_point",
                "price_change_pct",
                "gap_pct",
                "tick_volume",
                "spread_pct",
                "spread_cost_per_lot",
                "spread_cost_currency",
                "rsi",
                "sma_value",
                "sma_distance_pct",
            ]
            compact_headers = [
                "symbol",
                "group",
                "asset_class",
                "timeframe",
                "data_source",
                "time",
                "price_as_of",
                "price_freshness",
                "quote_time",
                "quote_as_of",
                "bid",
                "ask",
                "mid",
                "quote_stale",
                "quote_freshness_reason",
                "quote_timestamp_in_future",
                "quote_timestamp_warning",
                "quote_usable_for_live_trading",
                "spread_valid",
                "spread_quality",
                "bar_stale",
                "bar_market_status",
                "bar_market_status_reason",
                "bar_freshness_policy_relaxed",
                "bar_freshness",
                "close",
                "price_currency",
                "price_basis",
                "price_point",
                "price_change_pct",
                "gap_pct",
                "tick_volume",
                "spread_pct",
                "spread_points",
                "spread_pips",
            ]
            if include_rsi:
                compact_headers.append("rsi")
            if include_sma:
                compact_headers.append("sma_distance_pct")
            optional_compact_headers = {
                "quote_timestamp_in_future",
                "quote_timestamp_warning",
                "bar_market_status",
                "bar_market_status_reason",
                "bar_freshness_policy_relaxed",
            }
            compact_headers = [
                header
                for header in compact_headers
                if header not in optional_compact_headers
                or any(row.get(header) is not None for row in limited_rows)
            ]
            compact_shared_fields: Dict[str, Any] = {}
            if detail_mode == "compact":
                compact_headers, compact_shared_fields = (
                    _compact_market_scan_projection(compact_headers, limited_rows)
                )
            headers = compact_headers if detail_mode == "compact" else full_headers
            output_rows = (
                _project_market_scan_rows(headers, limited_rows)
                if detail_mode == "compact"
                else limited_rows
            )
            request["filters"] = {
                key: value
                for key, value in {
                    "min_price_change_pct": min_price_change_pct,
                    "max_price_change_pct": max_price_change_pct,
                    "min_gap_pct": min_gap_pct,
                    "max_gap_pct": max_gap_pct,
                    "max_spread_pct": max_spread_pct,
                    "min_tick_volume": min_tick_volume,
                    "rsi_below": rsi_below,
                    "rsi_above": rsi_above,
                    "price_vs_sma": price_vs_sma_value,
                    "rsi_length": rsi_length_value
                    if (rsi_above is not None or rsi_below is not None or rank_by_value == "rsi")
                    else None,
                    "sma_period": sma_period_value
                    if price_vs_sma_value is not None
                    else None,
                }.items()
                if value is not None
            }
            stats = {
                "scanned_symbols": len(selected_symbols),
                "evaluated_symbols": evaluated_symbols,
                "matched_symbols": total_matches,
                "filtered_out_symbols": max(0, evaluated_symbols - total_matches),
                "skipped_symbols": skipped_symbols,
                "skipped_examples": skipped_examples,
                "quote_eligibility_excluded_symbols": quote_eligibility_excluded,
                "query_latency_ms": round((time.perf_counter() - started_at) * 1000.0, 3),
            }
            table_payload = _market_scan_contract_table(
                headers,
                output_rows,
                include_columns=detail_mode == "full",
            )
            freshness_summary = _market_scan_freshness_summary(
                limited_rows,
                include_stale_symbols=detail_mode == "full",
            )
            stale_rows = int(freshness_summary.get("stale_rows") or 0)
            returned_count = int(table_payload["row_count"])
            if total_matches > returned_count:
                message = (
                    f"Showing {returned_count} of {int(total_matches)} symbols matching "
                    "the requested market scan filters."
                )
            elif total_matches > 0:
                message = (
                    f"Returned all {returned_count} symbol(s) matching the requested "
                    "market scan filters."
                )
            else:
                message = "No symbols matched the requested market scan filters."
            if stale_rows:
                message = (
                    f"{message} Returned rows: "
                    f"{stale_rows}/{int(table_payload['row_count'])} stale."
                )
            out: Dict[str, Any] = {
                "success": True,
                "status": "ok" if total_matches > 0 else "no_matches",
                "message": message,
                "data": table_payload["rows"],
                "count": table_payload["row_count"],
                "rank_by": rank_by_value,
                "rank_order": effective_rank_order,
                "ranking": _market_scan_ranking_label(
                    rank_by_value,
                    rank_order=effective_rank_order,
                    rsi_above=rsi_above,
                    rsi_below=rsi_below,
                ),
                "quote_usable_only": quote_usable_only_value,
                "quote_eligibility": {
                    "basis": "quote_usable_for_live_trading",
                    "required": quote_usable_only_value,
                    "excluded_symbols": int(quote_eligibility_excluded),
                    "excluded_reasons": dict(sorted(quote_eligibility_reasons.items())),
                    "excluded_examples": quote_eligibility_examples,
                },
                "price_change_basis": "previous_completed_close_to_latest_completed_close",
                "price_change_period": {
                    "bars": 1,
                    "timeframe": timeframe,
                    "bar_state": "completed",
                },
                "gap_basis": "previous_completed_close_to_latest_completed_open",
                "pagination": build_pagination_meta(
                    total=total_matches,
                    returned=table_payload["row_count"],
                    offset=offset_value,
                    limit=limit_value,
                ),
                "universe_size": int(len(selected_symbols)),
                "summary": {
                    "counts": {
                        "scanned_symbols": int(stats["scanned_symbols"]),
                        "evaluated_symbols": int(stats["evaluated_symbols"]),
                        "filtered_out_symbols": int(stats["filtered_out_symbols"]),
                        "skipped_symbols": int(stats["skipped_symbols"]),
                        "quote_eligibility_excluded_symbols": int(
                            quote_eligibility_excluded
                        ),
                    }
                },
                "meta": _market_scan_contract_meta(request=request, stats=stats),
            }
            if rank_order_value != effective_rank_order:
                out["rank_order_requested"] = rank_order_value
            compact_warnings = compact_shared_fields.pop("warnings", None)
            out.update(compact_shared_fields)
            if compact_warnings:
                out["warnings"] = list(compact_warnings)
            if preset_value:
                out["preset"] = preset_value
                out["preset_filters"] = {
                    key: value
                    for key, value in dict(preset_config or {}).items()
                    if key != "rank_by"
                }
                out["preset_rank_by"] = rank_by_value
            if detail_mode == "full":
                out["returned_count"] = int(table_payload["row_count"])
                out["summary"]["counts"]["matched_symbols"] = int(
                    stats["matched_symbols"]
                )
            missing_symbols = list(selection_meta.get("missing_symbols") or [])
            if missing_symbols:
                out["missing_symbols"] = missing_symbols
                out.setdefault("warnings", []).append(
                    "Requested symbol(s) not found and excluded from the scan: "
                    + ", ".join(missing_symbols)
                    + "."
                )
            out.update(freshness_summary)
            units = _market_scan_units_for_rows(table_payload["rows"])
            if units:
                out["units"] = units
            _attach_market_scan_volume_semantics(out, units)
            if "columns" in table_payload:
                out["columns"] = table_payload["columns"]
            if len(selected_symbols) < int(limit_value):
                out["note"] = (
                    f"Requested {int(limit_value)} rows but only "
                    f"{len(selected_symbols)} symbols were available in the "
                    f"{universe_value} universe."
                )
            if total_matches == 0:
                out["summary"]["empty"] = True
            return attach_collection_contract(
                out,
                collection_kind="table",
                rows=output_rows,
                include_contract_meta=detail_mode == "full",
            )
        except MT5ConnectionError as exc:
            return _market_scan_error(
                str(exc),
                code="mt5_connection_error",
                request=request,
            )
        except Exception as exc:
            return _market_scan_error(
                f"Error running market scan: {str(exc)}",
                code="market_scan_failed",
                request=request,
            )

    return run_logged_operation(
        logger,
        operation="market_scan",
        symbols=symbols,
        group=group,
        limit=limit,
        offset=offset,
        universe=universe,
        timeframe=timeframe,
        func=_run,
    )
