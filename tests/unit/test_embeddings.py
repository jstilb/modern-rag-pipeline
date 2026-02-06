"""Tests for embedding providers."""

import numpy as np

from src.rag.embeddings import MockEmbeddingProvider, create_embedding_provider
from src.rag.config import RAGConfig, RunMode


class TestMockEmbeddingProvider:
    def test_embed_texts(self) -> None:
        provider = MockEmbeddingProvider(dimensions=128)
        result = provider.embed_texts(["hello world", "test document"])
        assert result.is_ok()
        embeddings = result.unwrap()
        assert len(embeddings) == 2
        assert len(embeddings[0]) == 128

    def test_embed_query(self) -> None:
        provider = MockEmbeddingProvider(dimensions=128)
        result = provider.embed_query("test query")
        assert result.is_ok()
        embedding = result.unwrap()
        assert len(embedding) == 128

    def test_deterministic(self) -> None:
        provider = MockEmbeddingProvider(dimensions=128)
        r1 = provider.embed_query("hello world")
        r2 = provider.embed_query("hello world")
        assert r1.unwrap() == r2.unwrap()

    def test_similar_texts_similar_embeddings(self) -> None:
        provider = MockEmbeddingProvider(dimensions=128)
        # Texts sharing words should have higher similarity
        r1 = provider.embed_query("machine learning algorithms")
        r2 = provider.embed_query("machine learning models")
        r3 = provider.embed_query("cooking recipes pasta")

        e1 = np.array(r1.unwrap())
        e2 = np.array(r2.unwrap())
        e3 = np.array(r3.unwrap())

        sim_12 = np.dot(e1, e2)
        sim_13 = np.dot(e1, e3)

        # Similar texts should be more similar than dissimilar
        assert sim_12 > sim_13

    def test_unit_vectors(self) -> None:
        provider = MockEmbeddingProvider(dimensions=128)
        result = provider.embed_query("test")
        embedding = np.array(result.unwrap())
        norm = np.linalg.norm(embedding)
        assert abs(norm - 1.0) < 1e-6

    def test_dimensions_property(self) -> None:
        provider = MockEmbeddingProvider(dimensions=256)
        assert provider.dimensions == 256


class TestCreateEmbeddingProvider:
    def test_mock_mode(self) -> None:
        config = RAGConfig(mode=RunMode.MOCK)
        provider = create_embedding_provider(config)
        assert isinstance(provider, MockEmbeddingProvider)

    def test_production_mode(self) -> None:
        config = RAGConfig(mode=RunMode.PRODUCTION)
        provider = create_embedding_provider(config)
        # Should create OpenAI provider (won't test actual API calls)
        assert provider is not None
