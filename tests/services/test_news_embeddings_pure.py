from __future__ import annotations

from mtdata.bootstrap.settings import news_embeddings_config
from mtdata.services.news_embeddings import (
    NewsEmbeddingService,
    _cosine_similarity,
    format_document_text,
    format_query_text,
)
from mtdata.services.unified_news import InstrumentContext, NewsItem


def _sample_context() -> InstrumentContext:
    return InstrumentContext(
        symbol="NAS100",
        asset_class="index",
        base_asset="NAS",
        quote_asset=None,
        aliases=("NAS100", "NQ"),
        terms=("nasdaq", "tech", "earnings"),
        description="Nasdaq 100 index",
        metadata_hints={},
    )


def test_format_query_text_includes_symbol_aliases_and_terms() -> None:
    text = format_query_text(_sample_context())

    assert "symbol: NAS100" in text
    assert "asset class: index" in text
    assert "aliases: NAS100, NQ" in text
    assert "terms: nasdaq, tech, earnings" in text


def test_format_document_text_uses_title_and_selected_metadata() -> None:
    item = NewsItem(
        title="Nasdaq futures climb ahead of CPI",
        provider="finviz",
        source="Finviz Futures",
        summary="NQ gains 0.8% before the US inflation release.",
        category="futures",
        metadata={"ticker": "NQ", "market": "futures", "event_for": "USD"},
    )

    text = format_document_text(item)

    assert text.startswith("title: Nasdaq futures climb ahead of CPI | text: ")
    assert "NQ gains 0.8%" in text
    assert "USD" in text
    assert "NQ" in text


def test_embedding_service_is_optional_until_model_is_available() -> None:
    service = NewsEmbeddingService()
    service._load_attempted = True

    assert service.is_available() is False
    assert service.status()["enabled"] is True
    assert service.status()["model"] == "Qwen/Qwen3-Embedding-0.6B"


def test_embedding_service_returns_empty_scores_when_backend_is_unavailable() -> None:
    service = NewsEmbeddingService()
    service._load_attempted = True
    item = NewsItem(title="test", provider="p", source="s")

    scores = service.score_documents(_sample_context(), [item])

    assert scores == {}


def test_embedding_service_caches_vectors(monkeypatch) -> None:
    class FakeModel:
        def __init__(self) -> None:
            self.query_calls = 0
            self.document_calls = 0

        def encode(self, text: str, prompt_name=None, normalize_embeddings=False, show_progress_bar=True):
            assert normalize_embeddings is True
            assert show_progress_bar is False
            if prompt_name == "query":
                self.query_calls += 1
                return [1.0, 0.0]
            self.document_calls += 1
            return [0.5, 0.5]

    service = NewsEmbeddingService()
    fake_model = FakeModel()
    monkeypatch.setattr(news_embeddings_config, "cache_size", 8)
    monkeypatch.setattr(service, "_ensure_model", lambda: fake_model)

    item = NewsItem(title="Nasdaq futures climb", provider="finviz", source="Finviz")

    first = service.score_documents(_sample_context(), [item])
    second = service.score_documents(_sample_context(), [item])

    assert first == second
    assert fake_model.query_calls == 1
    assert fake_model.document_calls == 1


def test_cosine_similarity_returns_zero_for_mismatched_lengths() -> None:
    assert _cosine_similarity([1.0], [1.0, 2.0]) == 0.0
