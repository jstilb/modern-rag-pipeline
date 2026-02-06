"""Configuration management for the RAG pipeline.

Supports three modes:
- Production: Real LLM and embedding APIs
- Mock: Deterministic fake responses for demos and testing
- Hybrid: Real embeddings with mock LLM (cost-effective testing)
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class RunMode(str, Enum):
    """Pipeline execution mode."""

    PRODUCTION = "production"
    MOCK = "mock"
    HYBRID = "hybrid"


class ChunkingMethod(str, Enum):
    """Available chunking strategies."""

    FIXED = "fixed"
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"
    SLIDING_WINDOW = "sliding_window"


class RetrievalMethod(str, Enum):
    """Available retrieval strategies."""

    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"


class RAGConfig(BaseSettings):
    """Main RAG pipeline configuration.

    All settings can be overridden via environment variables with the RAG_ prefix.
    Example: RAG_MODE=mock, RAG_OPENAI_API_KEY=sk-...
    """

    model_config = {"env_prefix": "RAG_"}

    # Core mode
    mode: RunMode = Field(default=RunMode.MOCK, description="Pipeline execution mode")

    # LLM settings
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    llm_model: str = Field(default="gpt-4o-mini", description="LLM model name")
    llm_temperature: float = Field(default=0.1, description="LLM temperature")
    llm_max_tokens: int = Field(default=1024, description="Max tokens for LLM response")

    # Embedding settings
    embedding_model: str = Field(
        default="text-embedding-3-small", description="Embedding model name"
    )
    embedding_dimensions: int = Field(default=384, description="Embedding vector dimensions")

    # ChromaDB settings
    chroma_persist_dir: str = Field(default=".chroma_data", description="ChromaDB persistence dir")
    chroma_collection: str = Field(default="rag_documents", description="ChromaDB collection name")

    # Chunking settings
    chunking_method: ChunkingMethod = Field(
        default=ChunkingMethod.RECURSIVE, description="Chunking strategy"
    )
    chunk_size: int = Field(default=512, description="Target chunk size in tokens")
    chunk_overlap: int = Field(default=64, description="Overlap between chunks in tokens")

    # Retrieval settings
    retrieval_method: RetrievalMethod = Field(
        default=RetrievalMethod.HYBRID, description="Retrieval strategy"
    )
    top_k: int = Field(default=5, description="Number of documents to retrieve")
    semantic_weight: float = Field(
        default=0.7, description="Weight for semantic search in hybrid mode"
    )
    keyword_weight: float = Field(
        default=0.3, description="Weight for keyword search in hybrid mode"
    )

    # API settings
    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default=8000, description="API port")


class MockConfig:
    """Configuration presets for mock/demo mode.

    Returns deterministic responses without requiring any API keys.
    Useful for testing, demos, and CI/CD pipelines.
    """

    @staticmethod
    def default() -> RAGConfig:
        """Create a default mock configuration."""
        return RAGConfig(mode=RunMode.MOCK)

    @staticmethod
    def with_overrides(**kwargs: object) -> RAGConfig:
        """Create mock config with specific overrides."""
        defaults = {"mode": RunMode.MOCK}
        defaults.update(kwargs)
        return RAGConfig(**defaults)  # type: ignore[arg-type]
