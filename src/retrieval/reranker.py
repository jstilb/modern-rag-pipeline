"""Reranker layer for the RAG pipeline.

Provides a second-stage ranking step that sits between retrieval and
generation. Two implementations are supported:

1. CrossEncoderReranker — uses a sentence-transformers CrossEncoder model
   (e.g., cross-encoder/ms-marco-MiniLM-L-6-v2) to score query-chunk pairs.
2. CohereReranker — uses the Cohere Rerank API for cloud-based reranking.

Both implement the same Reranker protocol and are selectable via the
``reranker_type`` parameter on the factory function.

Usage::

    from src.retrieval.reranker import create_reranker

    reranker = create_reranker("cross_encoder")
    reranked = reranker.rerank(query, retrieved_chunks)
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Literal, Optional, Union

from src.rag.document import RetrievedChunk


class BaseReranker(ABC):
    """Abstract base class for reranker implementations."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_n: Optional[int] = None,
    ) -> list[RetrievedChunk]:
        """Rerank retrieved chunks by relevance to the query.

        Args:
            query: The user's search query.
            chunks: Chunks from the retrieval stage, in retrieval score order.
            top_n: If set, return only the top N results. Otherwise return all.

        Returns:
            Chunks sorted by reranker relevance score (highest first), with
            scores updated to reflect the reranker's confidence.
        """


class CrossEncoderReranker(BaseReranker):
    """Reranker using a sentence-transformers CrossEncoder model.

    CrossEncoder models score (query, passage) pairs directly, giving more
    accurate relevance estimates than bi-encoder similarity scores.

    Args:
        model_name: HuggingFace model name for the CrossEncoder. Defaults to
            ``cross-encoder/ms-marco-MiniLM-L-6-v2`` (6-layer, fast, accurate).
        mock_mode: If True, skip loading the model and return deterministic
            scores based on chunk position. Useful for testing without GPU/download.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        mock_mode: bool = False,
    ) -> None:
        self._model_name = model_name
        self._mock_mode = mock_mode
        self._model: object = None

        if not mock_mode:
            self._load_model()

    def _load_model(self) -> None:
        """Lazily load the CrossEncoder model."""
        try:
            from sentence_transformers import CrossEncoder  # type: ignore[import-untyped]

            self._model = CrossEncoder(self._model_name)
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for CrossEncoderReranker. "
                "Install it with: pip install 'sentence-transformers>=2.7.0'"
            ) from exc

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_n: Optional[int] = None,
    ) -> list[RetrievedChunk]:
        """Rerank chunks using cross-encoder scoring.

        In mock mode, assigns deterministic scores (0.9 - 0.05 * i) so the
        output is predictable in tests without downloading any model.

        Args:
            query: The search query.
            chunks: Retrieved chunks to rerank.
            top_n: Maximum number of chunks to return.

        Returns:
            Chunks sorted by cross-encoder score (descending), with scores
            normalised to [0, 1].
        """
        if not chunks:
            return []

        n = top_n or len(chunks)

        if self._mock_mode:
            # Deterministic scores for testing — reverse the order slightly
            # so the reranker has a demonstrable effect
            scored = []
            for i, chunk in enumerate(chunks):
                mock_score = max(0.0, min(1.0, 0.95 - i * 0.07 + (len(chunks) - i) * 0.01))
                from dataclasses import replace as dc_replace

                # Reconstruct with updated score to show reranker effect
                scored.append((mock_score, chunk))

            scored.sort(key=lambda x: x[0], reverse=True)
            return [
                RetrievedChunk(
                    chunk=c.chunk,
                    score=max(0.0, min(1.0, s)),
                    retrieval_method="cross_encoder",
                )
                for s, c in scored[:n]
            ]

        # Real cross-encoder scoring
        pairs = [[query, c.chunk.content] for c in chunks]
        raw_scores: list[float] = self._model.predict(pairs).tolist()  # type: ignore[union-attr]

        # Normalise logit scores to [0, 1] via sigmoid
        import math

        def sigmoid(x: float) -> float:
            return 1.0 / (1.0 + math.exp(-x))

        normalised = [sigmoid(s) for s in raw_scores]
        scored_pairs = sorted(
            zip(normalised, chunks, strict=True), key=lambda x: x[0], reverse=True
        )

        return [
            RetrievedChunk(
                chunk=c.chunk,
                score=max(0.0, min(1.0, s)),
                retrieval_method="cross_encoder",
            )
            for s, c in scored_pairs[:n]
        ]


class CohereReranker(BaseReranker):
    """Reranker using the Cohere Rerank API.

    Sends retrieved chunks to Cohere's cloud reranking endpoint, which
    uses a large-scale cross-encoder model trained on diverse datasets.

    Args:
        model: Cohere rerank model name. Defaults to ``rerank-english-v3.0``.
        api_key: Cohere API key. If not provided, reads from ``COHERE_API_KEY``
            environment variable.
        mock_mode: If True, skip the API call and return deterministic mock
            scores. Useful for testing without a Cohere account.
    """

    def __init__(
        self,
        model: str = "rerank-english-v3.0",
        api_key: Optional[str] = None,
        mock_mode: bool = False,
    ) -> None:
        self._model = model
        self._api_key = api_key or os.environ.get("COHERE_API_KEY", "")
        self._mock_mode = mock_mode
        self._client: object = None

        if not mock_mode:
            self._init_client()

    def _init_client(self) -> None:
        """Initialise the Cohere client."""
        try:
            import cohere  # type: ignore[import-untyped]

            self._client = cohere.Client(self._api_key)
        except ImportError as exc:
            raise ImportError(
                "cohere SDK is required for CohereReranker. "
                "Install it with: pip install 'cohere>=5.0.0'"
            ) from exc

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_n: Optional[int] = None,
    ) -> list[RetrievedChunk]:
        """Rerank chunks using the Cohere Rerank API.

        In mock mode, returns chunks in reversed order with linearly
        decreasing mock scores to simulate reranking without an API call.

        Args:
            query: The search query.
            chunks: Retrieved chunks to rerank.
            top_n: Maximum number of chunks to return.

        Returns:
            Chunks sorted by Cohere relevance score (descending), with
            scores normalised to [0, 1].
        """
        if not chunks:
            return []

        n = top_n or len(chunks)

        if self._mock_mode:
            # Reverse order + new scores to show the reranker had an effect
            reversed_chunks = list(reversed(chunks))
            return [
                RetrievedChunk(
                    chunk=c.chunk,
                    score=max(0.0, min(1.0, 0.90 - i * 0.05)),
                    retrieval_method="cohere_rerank",
                )
                for i, c in enumerate(reversed_chunks[:n])
            ]

        documents = [c.chunk.content for c in chunks]
        response = self._client.rerank(  # type: ignore[union-attr]
            model=self._model,
            query=query,
            documents=documents,
            top_n=n,
        )

        reranked: list[RetrievedChunk] = []
        for result in response.results:
            original_chunk = chunks[result.index]
            score = max(0.0, min(1.0, float(result.relevance_score)))
            reranked.append(
                RetrievedChunk(
                    chunk=original_chunk.chunk,
                    score=score,
                    retrieval_method="cohere_rerank",
                )
            )

        return reranked


RerankerType = Union[CrossEncoderReranker, CohereReranker]


def create_reranker(
    reranker_type: Literal["cross_encoder", "cohere"] = "cross_encoder",
    *,
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    mock_mode: bool = False,
) -> RerankerType:
    """Factory function to create a configured reranker.

    Args:
        reranker_type: Which reranker to use: ``"cross_encoder"`` or ``"cohere"``.
        model_name: Override the default model name.
        api_key: Cohere API key (only used for ``"cohere"`` type).
        mock_mode: If True, skip model/API loading for testing.

    Returns:
        A configured reranker instance.

    Raises:
        ValueError: If ``reranker_type`` is not recognised.
    """
    if reranker_type == "cross_encoder":
        kwargs: dict[str, object] = {"mock_mode": mock_mode}
        if model_name:
            kwargs["model_name"] = model_name
        return CrossEncoderReranker(**kwargs)  # type: ignore[arg-type]
    elif reranker_type == "cohere":
        kwargs = {"mock_mode": mock_mode}
        if model_name:
            kwargs["model"] = model_name
        if api_key:
            kwargs["api_key"] = api_key
        return CohereReranker(**kwargs)  # type: ignore[arg-type]
    else:
        raise ValueError(
            f"Unknown reranker type: {reranker_type!r}. "
            "Expected 'cross_encoder' or 'cohere'."
        )
