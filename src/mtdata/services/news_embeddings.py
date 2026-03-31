"""Optional Qwen embedding reranking support for unified news."""

from __future__ import annotations

import logging
import math
import os
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Iterable, Optional, Sequence

from ..bootstrap.settings import news_embeddings_config

if TYPE_CHECKING:
    from .unified_news import InstrumentContext, NewsItem

logger = logging.getLogger(__name__)

_EMBEDDING_SERVICE: Optional["NewsEmbeddingService"] = None


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _normalize_text_key(value: str) -> str:
    return " ".join(value.strip().split())


def _cosine_similarity(lhs: Sequence[float], rhs: Sequence[float]) -> float:
    if not lhs or not rhs or len(lhs) != len(rhs):
        return 0.0
    dot = 0.0
    lhs_norm = 0.0
    rhs_norm = 0.0
    for left, right in zip(lhs, rhs):
        lval = float(left)
        rval = float(right)
        dot += lval * rval
        lhs_norm += lval * lval
        rhs_norm += rval * rval
    if lhs_norm <= 0.0 or rhs_norm <= 0.0:
        return 0.0
    return dot / (math.sqrt(lhs_norm) * math.sqrt(rhs_norm))


def format_query_text(context: "InstrumentContext") -> str:
    parts = [
        f"symbol: {context.symbol}",
        f"asset class: {context.asset_class}",
    ]
    if context.base_asset:
        parts.append(f"base asset: {context.base_asset}")
    if context.quote_asset:
        parts.append(f"quote asset: {context.quote_asset}")
    if context.aliases:
        parts.append(f"aliases: {', '.join(alias for alias in context.aliases if alias)}")
    if context.description:
        parts.append(f"description: {context.description}")
    if context.terms:
        parts.append(f"terms: {', '.join(term for term in context.terms if term)}")
    return " | ".join(parts)


def format_document_text(item: "NewsItem") -> str:
    title = _safe_text(item.title) or "none"
    text_parts = [
        _safe_text(item.summary),
        _safe_text(item.category),
        _safe_text(item.source),
        _safe_text(item.provider),
    ]
    for key in ("event_for", "ticker", "market", "search_text"):
        value = item.metadata.get(key)
        if isinstance(value, str) and value.strip():
            text_parts.append(value.strip())
    text_body = " | ".join(part for part in text_parts if part) or "none"
    return f"title: {title} | text: {text_body}"


class NewsEmbeddingService:
    """Lazy embedding backend with small text caches for query/document vectors."""

    def __init__(self) -> None:
        self._model: Any = None
        self._load_attempted = False
        self._load_error: Optional[str] = None
        self._query_cache: OrderedDict[str, tuple[float, ...]] = OrderedDict()
        self._document_cache: OrderedDict[str, tuple[float, ...]] = OrderedDict()

    @property
    def enabled(self) -> bool:
        return True

    @property
    def top_n(self) -> int:
        return int(news_embeddings_config.top_n)

    @property
    def weight(self) -> float:
        return float(news_embeddings_config.weight)

    @property
    def cache_size(self) -> int:
        return int(news_embeddings_config.cache_size)

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "available": self.is_available(),
            "model": news_embeddings_config.model_name,
            "query_prompt": "query",
            "top_n": self.top_n,
            "weight": self.weight,
            "truncate_dim": news_embeddings_config.truncate_dim,
            "cache_size": self.cache_size,
            "load_error": self._load_error,
        }

    def is_available(self) -> bool:
        return self._ensure_model() is not None

    def score_documents(self, context: "InstrumentContext", items: Iterable["NewsItem"]) -> dict[str, float]:
        model = self._ensure_model()
        if model is None:
            return {}
        query_text = format_query_text(context)
        query_embedding = self._get_query_embedding(query_text)
        if query_embedding is None:
            return {}

        scores: dict[str, float] = {}
        for item in items:
            document_text = format_document_text(item)
            document_embedding = self._get_document_embedding(document_text)
            if document_embedding is None:
                continue
            scores[item.dedupe_key()] = _cosine_similarity(query_embedding, document_embedding)
        return scores

    def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model
        if self._load_attempted:
            return None

        self._load_attempted = True
        self._load_error = None
        token_name = _safe_text(news_embeddings_config.hf_token_env_var)
        if token_name and token_name != "HF_TOKEN":
            token_value = _safe_text(os.getenv(token_name))
            if token_value and not os.getenv("HF_TOKEN"):
                os.environ["HF_TOKEN"] = token_value
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:
            self._load_error = f"sentence-transformers unavailable: {exc}"
            logger.info("Unified news embeddings unavailable: %s", self._load_error)
            return None

        try:
            kwargs: dict[str, Any] = {}
            if news_embeddings_config.truncate_dim is not None:
                kwargs["truncate_dim"] = int(news_embeddings_config.truncate_dim)
            self._model = SentenceTransformer(news_embeddings_config.model_name, **kwargs)
            return self._model
        except Exception as exc:
            self._load_error = str(exc)
            logger.warning("Failed to load news embedding model %s: %s", news_embeddings_config.model_name, exc)
            return None

    def _get_query_embedding(self, text: str) -> Optional[tuple[float, ...]]:
        return self._cache_or_compute(self._query_cache, text, mode="query")

    def _get_document_embedding(self, text: str) -> Optional[tuple[float, ...]]:
        return self._cache_or_compute(self._document_cache, text, mode="document")

    def _cache_or_compute(
        self,
        cache: OrderedDict[str, tuple[float, ...]],
        text: str,
        *,
        mode: str,
    ) -> Optional[tuple[float, ...]]:
        key = _normalize_text_key(text)
        cached = cache.get(key)
        if cached is not None:
            cache.move_to_end(key)
            return cached

        vector = self._encode_text(key, mode=mode)
        if vector is None:
            return None
        cache[key] = vector
        cache.move_to_end(key)
        while self.cache_size >= 0 and len(cache) > self.cache_size > 0:
            cache.popitem(last=False)
        if self.cache_size == 0:
            cache.clear()
        return vector

    def _encode_text(self, text: str, *, mode: str) -> Optional[tuple[float, ...]]:
        model = self._ensure_model()
        if model is None:
            return None
        try:
            if mode == "query":
                vector = model.encode(text, prompt_name="query", normalize_embeddings=True)
            else:
                vector = model.encode(text, normalize_embeddings=True)
        except Exception as exc:
            self._load_error = f"encoding failed: {exc}"
            logger.warning("Failed to encode %s embedding: %s", mode, exc)
            return None
        if isinstance(vector, Sequence) and vector and isinstance(vector[0], Sequence):
            vector = vector[0]
        return tuple(float(value) for value in vector)


def get_news_embedding_service() -> NewsEmbeddingService:
    """Return the shared optional embedding service instance."""

    global _EMBEDDING_SERVICE
    if _EMBEDDING_SERVICE is None:
        _EMBEDDING_SERVICE = NewsEmbeddingService()
    return _EMBEDDING_SERVICE
