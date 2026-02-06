"""Keyword-based retrieval using BM25 ranking.

Complements semantic search by catching exact keyword matches
that embedding models might miss (e.g., acronyms, proper nouns).
"""

from __future__ import annotations

from typing import Optional

from rank_bm25 import BM25Okapi

from src.rag.document import Chunk, RetrievedChunk
from src.rag.result import Err, Ok, Result


class KeywordRetriever:
    """Retrieves documents using BM25 keyword matching."""

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._bm25: Optional[BM25Okapi] = None
        self._tokenized_corpus: list[list[str]] = []

    def index(self, chunks: list[Chunk]) -> Result[int, str]:
        """Index chunks for BM25 retrieval."""
        if not chunks:
            return Ok(0)

        try:
            self._chunks = chunks
            self._tokenized_corpus = [
                chunk.content.lower().split() for chunk in chunks
            ]
            self._bm25 = BM25Okapi(self._tokenized_corpus)
            return Ok(len(chunks))
        except Exception as e:
            return Err(f"BM25 indexing failed: {e}")

    def retrieve(
        self, query: str, top_k: Optional[int] = None
    ) -> Result[list[RetrievedChunk], str]:
        """Retrieve the top-k chunks using BM25 scoring."""
        if self._bm25 is None or not self._chunks:
            return Ok([])

        k = top_k or 5

        try:
            tokenized_query = query.lower().split()
            scores = self._bm25.get_scores(tokenized_query)

            # Normalize scores to [0, 1]
            max_score = max(scores) if max(scores) > 0 else 1.0
            normalized = scores / max_score

            # Get top-k indices
            top_indices = sorted(
                range(len(scores)), key=lambda i: scores[i], reverse=True
            )[:k]

            retrieved: list[RetrievedChunk] = []
            for idx in top_indices:
                if normalized[idx] > 0.0:
                    retrieved.append(
                        RetrievedChunk(
                            chunk=self._chunks[idx],
                            score=float(normalized[idx]),
                            retrieval_method="keyword",
                        )
                    )

            return Ok(retrieved)
        except Exception as e:
            return Err(f"BM25 retrieval failed: {e}")

    def clear(self) -> None:
        """Clear the BM25 index."""
        self._chunks = []
        self._bm25 = None
        self._tokenized_corpus = []
