"""Hybrid retrieval combining semantic and keyword search.

Uses Reciprocal Rank Fusion (RRF) to merge results from both
retrieval strategies, getting the best of both approaches:
- Semantic: Understands meaning, handles paraphrasing
- Keyword: Catches exact matches, acronyms, proper nouns
"""

from __future__ import annotations

from dataclasses import replace
from typing import Optional

from src.rag.config import RAGConfig
from src.rag.document import Chunk, RetrievedChunk
from src.rag.result import Err, Ok, Result
from src.retrieval.keyword import KeywordRetriever
from src.retrieval.semantic import SemanticRetriever


class HybridRetriever:
    """Combines semantic and keyword retrieval using Reciprocal Rank Fusion."""

    def __init__(
        self,
        semantic: SemanticRetriever,
        keyword: KeywordRetriever,
        config: RAGConfig,
    ) -> None:
        self._semantic = semantic
        self._keyword = keyword
        self._config = config

    def retrieve(
        self, query: str, top_k: Optional[int] = None
    ) -> Result[list[RetrievedChunk], str]:
        """Retrieve documents using hybrid search with RRF fusion."""
        k = top_k or self._config.top_k

        # Fetch more candidates from each retriever for better fusion
        candidate_k = k * 3

        # Get semantic results
        sem_result = self._semantic.retrieve(query, top_k=candidate_k)
        if sem_result.is_err():
            return Err(f"Semantic retrieval failed: {sem_result.error}")  # type: ignore[union-attr]

        # Get keyword results
        kw_result = self._keyword.retrieve(query, top_k=candidate_k)
        if kw_result.is_err():
            return Err(f"Keyword retrieval failed: {kw_result.error}")  # type: ignore[union-attr]

        semantic_chunks = sem_result.unwrap()
        keyword_chunks = kw_result.unwrap()

        # Fuse results using RRF
        fused = self._reciprocal_rank_fusion(
            semantic_chunks,
            keyword_chunks,
            semantic_weight=self._config.semantic_weight,
            keyword_weight=self._config.keyword_weight,
        )

        return Ok(fused[:k])

    @staticmethod
    def _reciprocal_rank_fusion(
        semantic_results: list[RetrievedChunk],
        keyword_results: list[RetrievedChunk],
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3,
        k: int = 60,
    ) -> list[RetrievedChunk]:
        """Merge ranked lists using Reciprocal Rank Fusion.

        RRF score = sum(weight / (k + rank)) across lists.
        Robust to score scale differences between retrievers.
        """
        # Build score map keyed by chunk content (dedup by content)
        scores: dict[str, float] = {}
        chunk_map: dict[str, RetrievedChunk] = {}

        for rank, rc in enumerate(semantic_results):
            content_key = rc.chunk.content
            rrf_score = semantic_weight / (k + rank + 1)
            scores[content_key] = scores.get(content_key, 0.0) + rrf_score
            if content_key not in chunk_map:
                chunk_map[content_key] = rc

        for rank, rc in enumerate(keyword_results):
            content_key = rc.chunk.content
            rrf_score = keyword_weight / (k + rank + 1)
            scores[content_key] = scores.get(content_key, 0.0) + rrf_score
            if content_key not in chunk_map:
                chunk_map[content_key] = rc

        # Sort by fused score
        sorted_keys = sorted(scores.keys(), key=lambda c: scores[c], reverse=True)

        # Normalize scores to [0, 1]
        max_score = scores[sorted_keys[0]] if sorted_keys else 1.0

        result: list[RetrievedChunk] = []
        for content_key in sorted_keys:
            original = chunk_map[content_key]
            normalized_score = scores[content_key] / max_score if max_score > 0 else 0.0
            result.append(
                RetrievedChunk(
                    chunk=original.chunk,
                    score=min(1.0, normalized_score),
                    retrieval_method="hybrid",
                )
            )

        return result
