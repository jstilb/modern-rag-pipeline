"""Document models for the RAG pipeline.

Defines the core data structures for documents flowing through
the ingestion, chunking, retrieval, and generation stages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4


@dataclass(frozen=True, slots=True)
class Document:
    """A source document before chunking."""

    content: str
    source: str
    metadata: dict[str, str | int | float | bool] = field(default_factory=dict)
    doc_id: str = field(default_factory=lambda: str(uuid4()))
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def __post_init__(self) -> None:
        if not self.content.strip():
            raise ValueError("Document content cannot be empty")
        if not self.source.strip():
            raise ValueError("Document source cannot be empty")


@dataclass(frozen=True, slots=True)
class Chunk:
    """A chunk produced from a document after splitting."""

    content: str
    doc_id: str
    chunk_index: int
    source: str
    metadata: dict[str, str | int | float | bool] = field(default_factory=dict)
    chunk_id: str = field(default_factory=lambda: str(uuid4()))
    token_count: int = 0

    def __post_init__(self) -> None:
        if not self.content.strip():
            raise ValueError("Chunk content cannot be empty")


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """A chunk returned from retrieval with a relevance score."""

    chunk: Chunk
    score: float
    retrieval_method: str

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"Score must be between 0 and 1, got {self.score}")


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """The final output of the RAG pipeline."""

    answer: str
    query: str
    retrieved_chunks: list[RetrievedChunk]
    model: str
    latency_ms: float
    token_usage: dict[str, int] = field(default_factory=dict)
    generation_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    context_window_used: Optional[int] = None
