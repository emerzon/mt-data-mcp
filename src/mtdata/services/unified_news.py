"""Unified news aggregation service with context-aware relevance ranking."""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import IntEnum
from typing import Any, Dict, Iterable, List, Optional, Protocol, runtime_checkable

from .finviz_service import (
    get_crypto_performance,
    get_economic_calendar,
    get_forex_performance,
    get_futures_performance,
    get_general_news,
    get_stock_news,
)
from .news_service import get_mt5_news
from ..utils.mt5 import get_symbol_info_cached

logger = logging.getLogger(__name__)

_DEFAULT_BUCKET_SIZE = 10
_CURRENCY_CODES = {"USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD"}
_CURRENCY_TERMS = {
    "USD": ["usd", "dollar", "us dollar", "fed", "fomc", "treasury", "cpi", "pce", "payrolls", "nfp"],
    "EUR": ["eur", "euro", "ecb", "lagarde", "eurozone"],
    "GBP": ["gbp", "pound", "sterling", "boe", "bank of england", "uk"],
    "JPY": ["jpy", "yen", "boj", "bank of japan", "japan", "ueda"],
    "AUD": ["aud", "aussie", "rba", "australia", "iron ore", "china"],
    "CAD": ["cad", "loonie", "boc", "canada", "crude", "oil"],
    "CHF": ["chf", "franc", "snb", "swiss", "switzerland"],
    "NZD": ["nzd", "kiwi", "rbnz", "new zealand"],
}
_CRYPTO_TERMS = {
    "BTC": ["btc", "bitcoin", "crypto", "etf", "stablecoin", "halving", "miner"],
    "ETH": ["eth", "ethereum", "crypto", "staking", "layer 2", "defi"],
    "SOL": ["sol", "solana", "crypto"],
    "XRP": ["xrp", "ripple", "crypto", "sec"],
    "DOGE": ["doge", "dogecoin", "crypto"],
    "BNB": ["bnb", "binance", "crypto"],
}
_COMMODITY_TERMS = {
    "XAU": ["xau", "gold", "bullion", "real yields", "safe haven"],
    "XAG": ["xag", "silver", "bullion", "industrial metals"],
    "WTI": ["wti", "oil", "crude", "inventory", "opec", "energy"],
    "BRENT": ["brent", "oil", "crude", "inventory", "opec", "energy"],
    "NG": ["natgas", "natural gas", "lng", "storage"],
}
_INDEX_TERMS = {
    "SPX": ["spx", "s&p 500", "us stocks", "earnings", "fed", "risk sentiment"],
    "NAS": ["nasdaq", "tech", "yields", "ai", "earnings"],
    "DJI": ["dow", "industrials", "us stocks", "fed"],
    "DAX": ["dax", "germany", "europe", "ecb"],
}
_CRYPTO_QUOTES = {"USD", "USDT", "USDC", "BTC", "ETH"}
_COMMODITY_BASES = {"XAU", "XAG", "WTI", "BRENT", "XBR", "NG", "NGAS", "NATGAS"}
_KNOWN_CRYPTO_BASES = set(_CRYPTO_TERMS)
_KNOWN_INDEX_HINTS = {
    "SPX500": "SPX",
    "US500": "SPX",
    "NAS100": "NAS",
    "USTEC": "NAS",
    "US30": "DJI",
    "DJ30": "DJI",
    "GER40": "DAX",
    "DE40": "DAX",
    "UK100": "FTSE",
    "JP225": "NIKKEI",
}
_CRYPTO_BASE_PREFIXES = {
    "DOGE": "DOGE",
    "BTC": "BTC",
    "ETH": "ETH",
    "SOL": "SOL",
    "XRP": "XRP",
    "BNB": "BNB",
}
_COMMODITY_BASE_PREFIXES = {
    "NATGAS": "NG",
    "BRENT": "BRENT",
    "NGAS": "NG",
    "XBR": "BRENT",
    "XAU": "XAU",
    "XAG": "XAG",
    "WTI": "WTI",
}
_INDEX_ALIASES = {
    "SPX": ["SPX500", "US500", "SP500", "ES"],
    "NAS": ["NAS100", "USTEC", "NDX", "NQ"],
    "DJI": ["US30", "DJ30", "YM", "DOW"],
    "DAX": ["GER40", "DE40", "DAX40", "FDAX"],
}
_INDEX_EXPOSURE_CURRENCIES = {
    "SPX": "USD",
    "NAS": "USD",
    "DJI": "USD",
    "DAX": "EUR",
    "FTSE": "GBP",
    "NIKKEI": "JPY",
}
_EQUITY_DESCRIPTION_STOPWORDS = {
    "class",
    "company",
    "corp",
    "corporation",
    "group",
    "holdings",
    "inc",
    "limited",
    "ltd",
    "ordinary",
    "plc",
    "shares",
}
_MACRO_EVENT_TERMS = (
    "cpi",
    "pce",
    "payroll",
    "nfp",
    "fomc",
    "fed",
    "ecb",
    "boe",
    "boj",
    "inflation",
    "interest rate",
    "rate decision",
    "rate cut",
    "rate hike",
    "gdp",
    "pmi",
    "jobs",
    "jobless",
    "unemployment",
    "retail sales",
    "ism",
)
_GENERAL_IMPORTANCE_TERMS = {
    "fed": 1.8,
    "fomc": 1.8,
    "ecb": 1.4,
    "boe": 1.4,
    "boj": 1.4,
    "cpi": 1.8,
    "inflation": 1.6,
    "payroll": 1.5,
    "nfp": 1.5,
    "pce": 1.4,
    "gdp": 1.2,
    "recession": 1.2,
    "tariff": 1.2,
    "war": 1.0,
    "sanction": 1.0,
    "earnings": 1.3,
    "guidance": 1.2,
    "etf": 1.2,
    "opec": 1.4,
    "rate cut": 1.5,
    "rate hike": 1.5,
    "decision": 0.8,
}


class NewsPriority(IntEnum):
    """Relative priority assigned to a normalized news item."""

    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class NewsItem:
    """Normalized cross-source news representation."""

    title: str
    provider: str
    source: str
    kind: str = "headline"
    published_at: Optional[datetime] = None
    url: Optional[str] = None
    summary: Optional[str] = None
    category: Optional[str] = None
    priority: NewsPriority = NewsPriority.MEDIUM
    metadata: Dict[str, Any] = field(default_factory=dict)
    relevance_score: float = 0.0
    importance_score: float = 0.0

    def search_text(self) -> str:
        parts = [self.title, self.summary or "", self.category or "", self.source, self.provider]
        for key in ("search_text", "event_for", "market", "ticker"):
            value = self.metadata.get(key)
            if isinstance(value, str) and value:
                parts.append(value)
        return " ".join(part for part in parts if part)

    def dedupe_key(self) -> str:
        if self.url:
            return self.url.strip().lower()
        title = re.sub(r"\s+", " ", self.title.strip().lower())
        return f"{self.provider}:{title}"

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "title": self.title,
            "provider": self.provider,
            "source": self.source,
            "kind": self.kind,
            "published_at": self.published_at.isoformat() if self.published_at else None,
            "url": self.url,
            "summary": self.summary,
            "category": self.category,
            "priority": self.priority.name,
            "relevance_score": round(float(self.relevance_score), 4),
            "importance_score": round(float(self.importance_score), 4),
        }
        if self.metadata:
            payload["metadata"] = self.metadata
        return payload


@dataclass(frozen=True)
class InstrumentContext:
    """Normalized instrument identity and exposure hints."""

    symbol: str
    asset_class: str
    base_asset: Optional[str]
    quote_asset: Optional[str]
    aliases: tuple[str, ...]
    terms: tuple[str, ...]
    description: Optional[str] = None
    metadata_hints: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "asset_class": self.asset_class,
            "base_asset": self.base_asset,
            "quote_asset": self.quote_asset,
            "aliases": list(self.aliases),
            "terms": list(self.terms),
            "description": self.description,
            "metadata_hints": self.metadata_hints,
        }


@runtime_checkable
class NewsSource(Protocol):
    """Protocol for pluggable news providers."""

    name: str

    def is_available(self) -> bool:
        """Return whether the source can currently be queried."""
        ...

    def fetch_general_candidates(self, limit: int) -> List[NewsItem]:
        """Fetch general market headlines or events."""
        ...

    def fetch_related_candidates(self, context: InstrumentContext, limit: int) -> List[NewsItem]:
        """Fetch source-specific candidates related to the given instrument."""
        ...


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _normalize_symbol(symbol: Optional[str]) -> Optional[str]:
    text = _safe_text(symbol).upper()
    return text or None


def _compact_token(value: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "", value.upper())


def _tokenize(value: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", value.lower())


def _unique_preserve_order(items: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for item in items:
        text = item.strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


def _maybe_parse_datetime(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    text = _safe_text(value)
    if not text:
        return None
    for fmt in (
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d",
        "%b %d '%y",
        "%b %d %Y",
    ):
        try:
            parsed = datetime.strptime(text.replace("Z", "+0000"), fmt)
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    try:
        from dateutil import parser as date_parser

        parsed = date_parser.parse(text)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _parse_relative_time(value: str) -> Optional[datetime]:
    text = _safe_text(value).lower()
    if not text:
        return None
    now = datetime.now(timezone.utc)
    if text == "just now":
        return now
    if text == "yesterday":
        return now - timedelta(days=1)
    match = re.search(r"(\d+)\s+(minute|hour|day)s?\s+ago", text)
    if not match:
        return None
    amount = int(match.group(1))
    unit = match.group(2)
    if unit == "minute":
        return now - timedelta(minutes=amount)
    if unit == "hour":
        return now - timedelta(hours=amount)
    return now - timedelta(days=amount)


def _safe_symbol_metadata(symbol: str) -> Dict[str, str]:
    try:
        info = get_symbol_info_cached(symbol)
    except Exception:
        return {}
    if info is None:
        return {}
    metadata: Dict[str, str] = {}
    for attr in ("path", "description", "currency_base", "currency_profit", "currency_margin", "basis"):
        value = getattr(info, attr, "")
        if isinstance(value, str) and value.strip():
            metadata[attr] = value.strip()
    return metadata


def _match_prefixed_base(compact: str, prefixes: Dict[str, str]) -> tuple[Optional[str], Optional[str]]:
    for prefix in sorted(prefixes, key=len, reverse=True):
        if compact.startswith(prefix):
            return prefixes[prefix], compact[len(prefix):] or None
    return None, None


def _classify_instrument(symbol: str) -> InstrumentContext:
    symbol_norm = _normalize_symbol(symbol)
    if not symbol_norm:
        raise ValueError("symbol is required")

    symbol_metadata = _safe_symbol_metadata(symbol_norm)
    path_text = _safe_text(symbol_metadata.get("path")).lower()
    desc_text = _safe_text(symbol_metadata.get("description")).lower()
    compact = _compact_token(symbol_norm)
    base_asset: Optional[str] = None
    quote_asset: Optional[str] = None
    asset_class = "equity"

    if "/" in symbol_norm:
        parts = [part for part in symbol_norm.split("/") if part]
        if len(parts) == 2:
            base_asset, quote_asset = parts[0], parts[1]
    elif len(compact) >= 6 and compact[:3].isalpha() and compact[3:6].isalpha():
        base_asset, quote_asset = compact[:3], compact[3:6]

    metadata_base = _safe_text(symbol_metadata.get("currency_base") or symbol_metadata.get("basis")).upper()
    metadata_quote = _safe_text(symbol_metadata.get("currency_profit") or symbol_metadata.get("currency_margin")).upper()
    if metadata_base and not base_asset:
        base_asset = metadata_base
    if metadata_quote and not quote_asset:
        quote_asset = metadata_quote

    if any(term in path_text or term in desc_text for term in ("crypto", "bitcoin", "ethereum")):
        asset_class = "crypto"
    elif any(term in path_text or term in desc_text for term in ("forex", "fx")):
        asset_class = "forex"
    elif any(term in path_text or term in desc_text for term in ("index", "indices")):
        asset_class = "index"
    elif any(term in path_text or term in desc_text for term in ("metal", "commodity", "energy", "oil", "gold")):
        asset_class = "commodity"
    elif base_asset in _COMMODITY_BASES:
        asset_class = "commodity"
        base_asset = _COMMODITY_BASE_PREFIXES.get(base_asset, base_asset)
    elif base_asset in _KNOWN_CRYPTO_BASES or quote_asset in _CRYPTO_QUOTES and base_asset in _KNOWN_CRYPTO_BASES:
        asset_class = "crypto"
    elif base_asset in _CURRENCY_CODES and quote_asset in _CURRENCY_CODES:
        asset_class = "forex"
    elif compact in _KNOWN_INDEX_HINTS:
        asset_class = "index"
        base_asset = _KNOWN_INDEX_HINTS[compact]
        quote_asset = None
    else:
        crypto_base, crypto_quote = _match_prefixed_base(compact, _CRYPTO_BASE_PREFIXES)
        if crypto_base:
            asset_class = "crypto"
            base_asset = crypto_base
            if crypto_quote:
                quote_asset = crypto_quote
        else:
            commodity_base, commodity_quote = _match_prefixed_base(compact, _COMMODITY_BASE_PREFIXES)
            if commodity_base:
                asset_class = "commodity"
                base_asset = commodity_base
                if commodity_quote:
                    quote_asset = commodity_quote

    aliases = [symbol_norm, compact]
    if base_asset and base_asset in _INDEX_ALIASES:
        aliases.extend(_INDEX_ALIASES[base_asset])
    if base_asset and quote_asset:
        aliases.append(f"{base_asset}/{quote_asset}")
    if asset_class == "crypto" and base_asset:
        aliases.append(base_asset)
    if asset_class == "commodity" and base_asset == "BRENT":
        aliases.append("XBR")
    if asset_class == "commodity" and base_asset == "NG":
        aliases.extend(["NGAS", "NATGAS"])

    terms: List[str] = [alias.lower() for alias in aliases]
    if base_asset in _CURRENCY_TERMS:
        terms.extend(_CURRENCY_TERMS[base_asset])
    if quote_asset in _CURRENCY_TERMS:
        terms.extend(_CURRENCY_TERMS[quote_asset])
    if base_asset in _CRYPTO_TERMS:
        terms.extend(_CRYPTO_TERMS[base_asset])
    if base_asset in _COMMODITY_TERMS:
        terms.extend(_COMMODITY_TERMS[base_asset])
    if base_asset in _INDEX_TERMS:
        terms.extend(_INDEX_TERMS[base_asset])
    if asset_class == "crypto":
        terms.extend(["risk sentiment", "liquidity", "etf", "inflation", "fed"])
    if asset_class == "equity":
        terms.extend(["earnings", "guidance", "analyst"])
    if asset_class == "forex":
        terms.extend(["forex", "fx", "rates", "central bank"])
    if asset_class == "commodity":
        terms.extend(["commodity", "inventory", "supply", "demand"])
    if asset_class == "index":
        terms.extend(["index", "risk sentiment", "breadth", "earnings"])

    if desc_text:
        desc_terms = [token for token in _tokenize(desc_text) if len(token) > 3][:6]
        terms.extend(desc_terms)

    return InstrumentContext(
        symbol=symbol_norm,
        asset_class=asset_class,
        base_asset=base_asset,
        quote_asset=quote_asset,
        aliases=tuple(_unique_preserve_order(aliases)),
        terms=tuple(_unique_preserve_order(terms)),
        description=desc_text or None,
        metadata_hints={key: value for key, value in symbol_metadata.items() if value},
    )


def _is_macro_sensitive_event(item: NewsItem) -> bool:
    if item.kind != "economic_event":
        return False
    text = item.search_text().lower()
    return any(term in text for term in _MACRO_EVENT_TERMS)


def _has_symbol_specific_evidence(item: NewsItem, context: InstrumentContext) -> bool:
    direct_symbol = _safe_text(item.metadata.get("direct_symbol")).upper()
    if direct_symbol == context.symbol:
        return True

    snapshot_ticker = _safe_text(item.metadata.get("ticker")).upper()
    if snapshot_ticker:
        compact_ticker = _compact_token(snapshot_ticker)
        if compact_ticker and any(compact_ticker == _compact_token(alias) for alias in context.aliases):
            return True

    text = item.search_text().lower()
    compact_text = _compact_token(text)
    for alias in context.aliases:
        alias_compact = _compact_token(alias)
        if len(alias_compact) >= 4 and alias_compact in compact_text:
            return True

    if context.asset_class == "equity" and context.description:
        tokens = set(_tokenize(text))
        desc_tokens = {
            token for token in _tokenize(context.description)
            if len(token) > 3 and token not in _EQUITY_DESCRIPTION_STOPWORDS
        }
        if desc_tokens & tokens:
            return True

    return False


def _cosine_similarity(lhs_tokens: Iterable[str], rhs_tokens: Iterable[str]) -> float:
    lhs = Counter(lhs_tokens)
    rhs = Counter(rhs_tokens)
    if not lhs or not rhs:
        return 0.0
    dot = sum(lhs[token] * rhs.get(token, 0) for token in lhs)
    if dot <= 0:
        return 0.0
    lhs_norm = math.sqrt(sum(value * value for value in lhs.values()))
    rhs_norm = math.sqrt(sum(value * value for value in rhs.values()))
    if lhs_norm <= 0 or rhs_norm <= 0:
        return 0.0
    return dot / (lhs_norm * rhs_norm)


def _score_importance(item: NewsItem) -> float:
    score = float(item.priority)
    text = item.search_text().lower()
    for term, weight in _GENERAL_IMPORTANCE_TERMS.items():
        if term in text:
            score += weight
    if item.kind == "economic_event":
        impact = _safe_text(item.metadata.get("impact")).lower()
        if impact == "high":
            score += 1.5
        elif impact == "medium":
            score += 0.75
    if item.kind == "direct_symbol":
        score += 1.0
    if item.kind == "market_snapshot":
        score += 0.6
    if item.published_at is not None:
        age_hours = max(0.0, (datetime.now(timezone.utc) - item.published_at).total_seconds() / 3600.0)
        score += max(0.0, 1.25 - min(age_hours, 48.0) / 48.0)
    return score


def _score_relevance(item: NewsItem, context: InstrumentContext) -> tuple[float, List[str]]:
    text = item.search_text().lower()
    compact_text = _compact_token(text)
    tokens = _tokenize(text)
    matched_terms: List[str] = []
    score = 0.0
    macro_sensitive_event = _is_macro_sensitive_event(item)

    direct_symbol = _safe_text(item.metadata.get("direct_symbol")).upper()
    if direct_symbol and direct_symbol == context.symbol:
        score += 4.0
        matched_terms.append(context.symbol)

    snapshot_ticker = _safe_text(item.metadata.get("ticker")).upper()
    if snapshot_ticker:
        compact_ticker = _compact_token(snapshot_ticker)
        if compact_ticker and any(compact_ticker == _compact_token(alias) for alias in context.aliases):
            score += 3.0
            matched_terms.append(snapshot_ticker)

    event_for = _safe_text(item.metadata.get("event_for")).upper()
    if event_for and event_for in {context.base_asset or "", context.quote_asset or ""}:
        score += 2.0
        matched_terms.append(event_for)
    elif (
        item.kind == "economic_event"
        and macro_sensitive_event
        and event_for
        and event_for == _INDEX_EXPOSURE_CURRENCIES.get(context.base_asset or "")
    ):
        score += 1.4
        matched_terms.append(event_for)

    for alias in context.aliases:
        alias_compact = _compact_token(alias)
        if alias_compact and alias_compact in compact_text:
            score += 1.4
            matched_terms.append(alias)

    for term in context.terms:
        term_norm = term.lower().strip()
        if not term_norm:
            continue
        if " " in term_norm:
            if term_norm in text:
                score += 0.9
                matched_terms.append(term_norm)
        elif term_norm in tokens:
            score += 0.5
            matched_terms.append(term_norm)

    similarity = _cosine_similarity(_tokenize(" ".join(context.terms)), tokens)
    score += similarity * 2.0

    matched_terms = _unique_preserve_order(matched_terms)
    return score, matched_terms


def _dedupe_items(items: Iterable[NewsItem]) -> List[NewsItem]:
    deduped: Dict[str, NewsItem] = {}
    for item in items:
        key = item.dedupe_key()
        existing = deduped.get(key)
        if existing is None or (item.importance_score, item.relevance_score) > (
            existing.importance_score,
            existing.relevance_score,
        ):
            deduped[key] = item
    return list(deduped.values())


class FinvizNewsSource:
    """Finviz-backed news provider and market context provider."""

    name = "finviz"

    def is_available(self) -> bool:
        return True

    def fetch_general_candidates(self, limit: int) -> List[NewsItem]:
        try:
            result = get_general_news(news_type="news", limit=limit, page=1)
            if not result.get("success"):
                return []
            items = result.get("items", [])
            out: List[NewsItem] = []
            for rank, item in enumerate(items):
                out.append(
                    NewsItem(
                        title=_safe_text(item.get("Title")),
                        provider=self.name,
                        source=_safe_text(item.get("Source")) or "Finviz",
                        kind="headline",
                        published_at=_maybe_parse_datetime(item.get("Date")),
                        url=_safe_text(item.get("Link")) or None,
                        category="market_news",
                        priority=NewsPriority.MEDIUM,
                        metadata={"source_rank": rank},
                    )
                )
            return out
        except Exception:
            logger.exception("Error fetching Finviz general candidates")
            return []

    def fetch_related_candidates(self, context: InstrumentContext, limit: int) -> List[NewsItem]:
        items: List[NewsItem] = []
        if context.asset_class == "equity":
            items.extend(self._fetch_direct_equity_news(context.symbol, limit))
        if context.asset_class == "crypto":
            items.extend(self._fetch_market_snapshots(get_crypto_performance(), context, rows_key="coins", market="crypto"))
        elif context.asset_class == "forex":
            items.extend(self._fetch_market_snapshots(get_forex_performance(), context, rows_key="pairs", market="forex"))
        elif context.asset_class in {"commodity", "index"}:
            items.extend(self._fetch_market_snapshots(get_futures_performance(), context, rows_key="futures", market="futures"))
        items.extend(self._fetch_economic_candidates(limit=max(limit, 12)))
        return items

    def _fetch_direct_equity_news(self, symbol: str, limit: int) -> List[NewsItem]:
        try:
            result = get_stock_news(symbol, limit=limit, page=1)
            if not result.get("success"):
                return []
            out: List[NewsItem] = []
            for rank, item in enumerate(result.get("news", [])):
                out.append(
                    NewsItem(
                        title=_safe_text(item.get("Title")),
                        provider=self.name,
                        source=_safe_text(item.get("Source")) or "Finviz",
                        kind="direct_symbol",
                        published_at=_maybe_parse_datetime(item.get("Date")),
                        url=_safe_text(item.get("Link")) or None,
                        category="symbol_news",
                        priority=NewsPriority.HIGH,
                        metadata={"direct_symbol": symbol, "source_rank": rank},
                    )
                )
            return out
        except Exception:
            logger.exception("Error fetching direct Finviz equity news for %s", symbol)
            return []

    def _fetch_market_snapshots(
        self,
        payload: Dict[str, Any],
        context: InstrumentContext,
        *,
        rows_key: str,
        market: str,
    ) -> List[NewsItem]:
        if not payload.get("success"):
            return []
        rows = payload.get(rows_key) or []
        scored_rows: List[tuple[float, Dict[str, Any]]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            ticker = _safe_text(row.get("Ticker") or row.get("Symbol") or row.get("Name"))
            search_text = " ".join(_safe_text(value) for value in row.values())
            row_score, _matched = _score_relevance(
                NewsItem(
                    title=ticker or market,
                    provider=self.name,
                    source=f"Finviz {market.title()}",
                    kind="market_snapshot",
                    category=market,
                    metadata={"ticker": ticker, "search_text": search_text, "market": market},
                ),
                context,
            )
            if row_score > 0:
                scored_rows.append((row_score, row))
        scored_rows.sort(key=lambda item: item[0], reverse=True)

        out: List[NewsItem] = []
        for row_score, row in scored_rows[:3]:
            ticker = _safe_text(row.get("Ticker") or row.get("Symbol") or row.get("Name"))
            summary_parts = []
            for key in ("Price", "Change", "Perf Day", "Perf Week", "Perf WTD"):
                value = row.get(key)
                if value not in (None, ""):
                    summary_parts.append(f"{key}: {value}")
            observed_at = datetime.now(timezone.utc)
            out.append(
                NewsItem(
                    title=f"{ticker or market.upper()} market snapshot",
                    provider=self.name,
                    source=f"Finviz {market.title()}",
                    kind="market_snapshot",
                    published_at=observed_at,
                    summary=", ".join(summary_parts) or None,
                    category=market,
                    priority=NewsPriority.HIGH,
                    metadata={
                        "ticker": ticker,
                        "market": market,
                        "search_text": " ".join(_safe_text(value) for value in row.values()),
                        "snapshot_time_inferred": True,
                        "snapshot_score": round(row_score, 4),
                    },
                )
            )
        return out

    def _fetch_economic_candidates(self, limit: int) -> List[NewsItem]:
        try:
            result = get_economic_calendar(limit=limit, page=1)
            if not result.get("success"):
                return []
            out: List[NewsItem] = []
            for rank, item in enumerate(result.get("items", [])):
                release = _safe_text(item.get("Release")) or "Economic event"
                event_for = _safe_text(item.get("For"))
                impact = _safe_text(item.get("Impact")).lower()
                summary_parts = []
                for key in ("Actual", "Expected", "Prior", "Category", "Reference"):
                    value = _safe_text(item.get(key))
                    if value:
                        summary_parts.append(f"{key}: {value}")
                priority = {
                    "high": NewsPriority.HIGH,
                    "medium": NewsPriority.MEDIUM,
                    "low": NewsPriority.LOW,
                }.get(impact, NewsPriority.MEDIUM)
                out.append(
                    NewsItem(
                        title=f"{release}{f' ({event_for})' if event_for else ''}",
                        provider=self.name,
                        source="Finviz Economic Calendar",
                        kind="economic_event",
                        published_at=_maybe_parse_datetime(item.get("Datetime")),
                        summary=" | ".join(summary_parts) or None,
                        category="economic_calendar",
                        priority=priority,
                        metadata={
                            "event_for": event_for,
                            "impact": impact,
                            "source_rank": rank,
                            "search_text": " ".join(
                                _safe_text(item.get(key))
                                for key in ("Release", "For", "Category", "Reference")
                            ),
                        },
                    )
                )
            return out
        except Exception:
            logger.exception("Error fetching Finviz economic candidates")
            return []


class MT5NewsSource:
    """MT5 local news provider."""

    name = "mt5"

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path
        self._available: Optional[bool] = None

    def is_available(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            result = get_mt5_news(news_db_path=self.db_path, limit=1)
            self._available = bool(result.get("success"))
        except Exception:
            self._available = False
        return self._available

    def fetch_general_candidates(self, limit: int) -> List[NewsItem]:
        return self._fetch_news(limit=limit)

    def fetch_related_candidates(self, context: InstrumentContext, limit: int) -> List[NewsItem]:
        return self._fetch_news(limit=max(limit, 20))

    def _fetch_news(self, limit: int) -> List[NewsItem]:
        if not self.is_available():
            return []
        try:
            result = get_mt5_news(news_db_path=self.db_path, limit=limit)
            if not result.get("success"):
                return []
            out: List[NewsItem] = []
            for rank, item in enumerate(result.get("news", [])):
                raw_priority = item.get("priority")
                priority = NewsPriority.HIGH if isinstance(raw_priority, int) and raw_priority > 0 else NewsPriority.MEDIUM
                out.append(
                    NewsItem(
                        title=_safe_text(item.get("subject")),
                        provider=self.name,
                        source=_safe_text(item.get("source") or item.get("category")) or "MT5",
                        kind="headline",
                        published_at=_parse_relative_time(_safe_text(item.get("relative_time"))),
                        category=_safe_text(item.get("category")) or None,
                        priority=priority,
                        metadata={
                            "relative_time": _safe_text(item.get("relative_time")) or None,
                            "mt5_priority": raw_priority,
                            "flags": item.get("flags"),
                            "source_rank": rank,
                            "search_text": " ".join(
                                _safe_text(item.get(key)) for key in ("subject", "category", "source")
                            ),
                        },
                    )
                )
            return out
        except Exception:
            logger.exception("Error fetching MT5 news candidates")
            return []


class NewsAggregator:
    """Aggregates general and instrument-related news across providers."""

    def __init__(self) -> None:
        self._sources: Dict[str, NewsSource] = {}
        self.register_source(FinvizNewsSource())
        self.register_source(MT5NewsSource())

    def register_source(self, source: NewsSource) -> None:
        self._sources[source.name] = source

    def get_available_sources(self) -> List[str]:
        return [name for name, source in self._sources.items() if source.is_available()]

    def fetch_news(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        bucket_size = _DEFAULT_BUCKET_SIZE
        candidate_limit = bucket_size * 3
        symbol_norm = _normalize_symbol(symbol)
        context = _classify_instrument(symbol_norm) if symbol_norm else None
        selected_sources = {name: src for name, src in self._sources.items() if src.is_available()}
        if not selected_sources:
            return {
                "success": False,
                "error": "No news sources available",
                "general_news": [],
                "related_news": [],
            }

        general_candidates: List[NewsItem] = []
        related_candidates: List[NewsItem] = []
        source_details: Dict[str, Dict[str, Any]] = {}

        for name, source in selected_sources.items():
            try:
                general_items = source.fetch_general_candidates(candidate_limit)
                general_candidates.extend(general_items)
                related_items: List[NewsItem] = []
                if context is not None:
                    related_items = source.fetch_related_candidates(context, candidate_limit)
                    related_candidates.extend(related_items)
                source_details[name] = {
                    "success": True,
                    "general_candidates": len(general_items),
                    "related_candidates": len(related_items),
                }
            except Exception as exc:
                logger.exception("Error collecting news from %s", name)
                source_details[name] = {"success": False, "error": str(exc)}

        general_pool = _dedupe_items(general_candidates)
        for item in general_pool:
            item.importance_score = _score_importance(item)
        general_pool.sort(
            key=lambda item: (
                item.importance_score,
                item.published_at or datetime.min.replace(tzinfo=timezone.utc),
                -int(item.metadata.get("source_rank", 0)),
            ),
            reverse=True,
        )

        related_news: List[NewsItem] = []
        if context is not None:
            related_pool = _dedupe_items(list(related_candidates) + list(general_pool))
            filtered_related: List[NewsItem] = []
            for item in related_pool:
                item.importance_score = _score_importance(item)
                item.relevance_score, matched_terms = _score_relevance(item, context)
                if matched_terms:
                    item.metadata["matched_terms"] = matched_terms
                if (
                    item.kind == "economic_event"
                    and context.asset_class in {"crypto", "equity", "index"}
                    and not _is_macro_sensitive_event(item)
                    and not _has_symbol_specific_evidence(item, context)
                ):
                    continue
                if (
                    context.asset_class == "equity"
                    and item.kind != "direct_symbol"
                    and not _has_symbol_specific_evidence(item, context)
                ):
                    continue
                threshold = 0.55 if item.kind != "direct_symbol" else 0.2
                if item.relevance_score >= threshold:
                    filtered_related.append(item)
            filtered_related.sort(
                key=lambda item: (
                    item.relevance_score,
                    item.importance_score,
                    item.published_at or datetime.min.replace(tzinfo=timezone.utc),
                ),
                reverse=True,
            )
            related_news = filtered_related[:bucket_size]

            related_keys = {item.dedupe_key() for item in related_news}
            general_pool = [item for item in general_pool if item.dedupe_key() not in related_keys]

        general_news = general_pool[:bucket_size]
        selected_general_counts = Counter(item.provider for item in general_news)
        selected_related_counts = Counter(item.provider for item in related_news)
        for name, details in source_details.items():
            if not details.get("success"):
                continue
            details["selected_general"] = selected_general_counts.get(name, 0)
            details["selected_related"] = selected_related_counts.get(name, 0)
            details["selected_total"] = details["selected_general"] + details["selected_related"]
        return {
            "success": True,
            "symbol": context.symbol if context is not None else None,
            "instrument": context.to_dict() if context is not None else None,
            "sources_used": list(selected_sources.keys()),
            "source_details": source_details,
            "matching": {
                "model": "keyword_plus_cosine_similarity",
                "notes": [
                    "Ranks direct symbol mentions, asset-class terms, macro exposures, and text cosine similarity.",
                    "Related news may include market snapshots and economic calendar events when they plausibly impact the instrument.",
                ],
            },
            "general_news": [item.to_dict() for item in general_news],
            "related_news": [item.to_dict() for item in related_news],
            "general_count": len(general_news),
            "related_count": len(related_news),
        }


_news_aggregator: Optional[NewsAggregator] = None


def get_news_aggregator() -> NewsAggregator:
    """Return the shared news aggregator instance."""

    global _news_aggregator
    if _news_aggregator is None:
        _news_aggregator = NewsAggregator()
    return _news_aggregator


def fetch_unified_news(symbol: Optional[str] = None) -> Dict[str, Any]:
    """Fetch general news and, when requested, instrument-related news."""

    aggregator = get_news_aggregator()
    return aggregator.fetch_news(symbol=symbol)


def register_news_source(source: NewsSource) -> None:
    """Register a custom news source for future extension."""

    get_news_aggregator().register_source(source)


def get_available_news_sources() -> List[str]:
    """Return the currently available source names."""

    return get_news_aggregator().get_available_sources()
