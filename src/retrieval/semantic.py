"""Semantic retrieval using vector similarity search.

Uses embedding-based cosine similarity to find the most
relevant document chunks for a given query.
"""

from __future__ import annotations

from typing import Optional

from src.rag.document import RetrievedChunk
from src.rag.result import Result
from src.retrieval.store import VectorStore


class SemanticRetriever:
    """Retrieves documents using embedding-based similarity search."""

    def __init__(self, store: VectorStore) -> None:
        self._store = store

    def retrieve(
        self, query: str, top_k: Optional[int] = None
    ) -> Result[list[RetrievedChunk], str]:
        """Retrieve the top-k most similar chunks for a query."""
        return self._store.search(query, top_k=top_k)
