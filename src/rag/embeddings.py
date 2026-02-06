"""Embedding providers with dependency injection for mock mode.

Supports:
- OpenAI embeddings (production)
- Mock embeddings (demo/testing - deterministic, no API keys)
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod

import numpy as np

from src.rag.config import RAGConfig, RunMode
from src.rag.result import Err, Ok, Result


class EmbeddingProvider(ABC):
    """Abstract embedding provider interface."""

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> Result[list[list[float]], str]:
        """Generate embeddings for a list of texts."""
        ...

    @abstractmethod
    def embed_query(self, query: str) -> Result[list[float], str]:
        """Generate embedding for a single query."""
        ...

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Return the embedding dimensions."""
        ...


class MockEmbeddingProvider(EmbeddingProvider):
    """Deterministic mock embeddings for testing and demos.

    Generates consistent embeddings based on text content hashing.
    Documents with similar words produce similar vectors.
    """

    def __init__(self, dimensions: int = 384) -> None:
        self._dimensions = dimensions

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed_texts(self, texts: list[str]) -> Result[list[list[float]], str]:
        try:
            embeddings = [self._generate_embedding(text) for text in texts]
            return Ok(embeddings)
        except Exception as e:
            return Err(f"Mock embedding failed: {e}")

    def embed_query(self, query: str) -> Result[list[float], str]:
        try:
            embedding = self._generate_embedding(query)
            return Ok(embedding)
        except Exception as e:
            return Err(f"Mock query embedding failed: {e}")

    def _generate_embedding(self, text: str) -> list[float]:
        """Generate a deterministic embedding from text content.

        Uses word-level hashing to create embeddings where
        texts sharing words will have higher cosine similarity.
        """
        # Create a seed from the text for reproducibility
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        seed = int(text_hash[:8], 16)
        rng = np.random.RandomState(seed)

        # Base random vector
        base = rng.randn(self._dimensions).astype(np.float64)

        # Add word-level features for similarity
        words = set(text.lower().split())
        for word in words:
            word_hash = hashlib.md5(word.encode()).hexdigest()
            word_seed = int(word_hash[:8], 16)
            word_rng = np.random.RandomState(word_seed)
            word_vec = word_rng.randn(self._dimensions).astype(np.float64)
            base += word_vec * 0.3

        # Normalize to unit vector
        norm = np.linalg.norm(base)
        if norm > 0:
            base = base / norm

        return base.tolist()


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI API embedding provider for production use."""

    def __init__(self, config: RAGConfig) -> None:
        self._config = config
        self._dimensions = config.embedding_dimensions

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed_texts(self, texts: list[str]) -> Result[list[list[float]], str]:
        try:
            from langchain_openai import OpenAIEmbeddings

            embeddings_model = OpenAIEmbeddings(
                model=self._config.embedding_model,
                openai_api_key=self._config.openai_api_key,
            )
            embeddings = embeddings_model.embed_documents(texts)
            return Ok(embeddings)
        except ImportError:
            return Err("langchain-openai not installed")
        except Exception as e:
            return Err(f"OpenAI embedding failed: {e}")

    def embed_query(self, query: str) -> Result[list[float], str]:
        try:
            from langchain_openai import OpenAIEmbeddings

            embeddings_model = OpenAIEmbeddings(
                model=self._config.embedding_model,
                openai_api_key=self._config.openai_api_key,
            )
            embedding = embeddings_model.embed_query(query)
            return Ok(embedding)
        except ImportError:
            return Err("langchain-openai not installed")
        except Exception as e:
            return Err(f"OpenAI query embedding failed: {e}")


def create_embedding_provider(config: RAGConfig) -> EmbeddingProvider:
    """Factory function to create the appropriate embedding provider."""
    if config.mode == RunMode.MOCK:
        return MockEmbeddingProvider(dimensions=config.embedding_dimensions)
    return OpenAIEmbeddingProvider(config)
