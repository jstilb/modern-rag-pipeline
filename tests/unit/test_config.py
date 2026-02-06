"""Tests for RAG configuration."""

from src.rag.config import (
    ChunkingMethod,
    MockConfig,
    RAGConfig,
    RetrievalMethod,
    RunMode,
)


class TestRAGConfig:
    def test_default_mode_is_mock(self) -> None:
        config = RAGConfig()
        assert config.mode == RunMode.MOCK

    def test_default_chunking_method(self) -> None:
        config = RAGConfig()
        assert config.chunking_method == ChunkingMethod.RECURSIVE

    def test_default_retrieval_method(self) -> None:
        config = RAGConfig()
        assert config.retrieval_method == RetrievalMethod.HYBRID

    def test_custom_config(self) -> None:
        config = RAGConfig(
            mode=RunMode.PRODUCTION,
            chunk_size=1024,
            top_k=10,
        )
        assert config.mode == RunMode.PRODUCTION
        assert config.chunk_size == 1024
        assert config.top_k == 10

    def test_hybrid_weights_sum(self) -> None:
        config = RAGConfig()
        assert config.semantic_weight + config.keyword_weight == pytest.approx(1.0)


class TestMockConfig:
    def test_default_is_mock(self) -> None:
        config = MockConfig.default()
        assert config.mode == RunMode.MOCK

    def test_with_overrides(self) -> None:
        config = MockConfig.with_overrides(chunk_size=128)
        assert config.mode == RunMode.MOCK
        assert config.chunk_size == 128


import pytest
