"""Tests for the reranker layer.

Tests both CrossEncoderReranker and CohereReranker in mock mode,
verifying that reranking:
- Returns the correct number of results
- Returns RetrievedChunk objects with updated scores
- Scores are within [0, 1]
- Reranker changes the order or scores compared to input
- factory function creates the right types
"""

from __future__ import annotations

import pytest

from src.rag.document import Chunk, RetrievedChunk
from src.retrieval.reranker import (
    CohereReranker,
    CrossEncoderReranker,
    create_reranker,
)


def make_chunk(content: str, index: int = 0) -> RetrievedChunk:
    """Helper to create a RetrievedChunk for testing."""
    chunk = Chunk(
        content=content,
        doc_id="doc-1",
        chunk_index=index,
        source="test",
    )
    return RetrievedChunk(chunk=chunk, score=0.8 - index * 0.1, retrieval_method="semantic")


SAMPLE_CHUNKS = [
    make_chunk("Hybrid search combines semantic and keyword retrieval.", 0),
    make_chunk("Reciprocal Rank Fusion merges multiple ranked lists.", 1),
    make_chunk("ChromaDB is a vector database for AI applications.", 2),
    make_chunk("FastAPI is an async Python web framework.", 3),
]

QUERY = "How does hybrid search work?"


class TestCrossEncoderReranker:
    """Tests for CrossEncoderReranker in mock mode."""

    def test_rerank_returns_same_count_as_input(self) -> None:
        """Reranker returns as many chunks as were passed in."""
        reranker = CrossEncoderReranker(mock_mode=True)
        result = reranker.rerank(QUERY, SAMPLE_CHUNKS)
        assert len(result) == len(SAMPLE_CHUNKS), (
            f"Expected {len(SAMPLE_CHUNKS)} results, got {len(result)}"
        )

    def test_rerank_respects_top_n(self) -> None:
        """Reranker respects the top_n parameter."""
        reranker = CrossEncoderReranker(mock_mode=True)
        result = reranker.rerank(QUERY, SAMPLE_CHUNKS, top_n=2)
        assert len(result) == 2, f"Expected 2 results with top_n=2, got {len(result)}"

    def test_rerank_scores_in_valid_range(self) -> None:
        """All reranker output scores are in [0, 1]."""
        reranker = CrossEncoderReranker(mock_mode=True)
        result = reranker.rerank(QUERY, SAMPLE_CHUNKS)
        for chunk in result:
            assert 0.0 <= chunk.score <= 1.0, (
                f"Score {chunk.score} is outside [0, 1]"
            )

    def test_rerank_sets_retrieval_method_to_cross_encoder(self) -> None:
        """Reranked chunks have retrieval_method='cross_encoder'."""
        reranker = CrossEncoderReranker(mock_mode=True)
        result = reranker.rerank(QUERY, SAMPLE_CHUNKS)
        for chunk in result:
            assert chunk.retrieval_method == "cross_encoder", (
                f"Expected retrieval_method='cross_encoder', got {chunk.retrieval_method!r}"
            )

    def test_rerank_preserves_chunk_content(self) -> None:
        """Reranking does not alter chunk content."""
        reranker = CrossEncoderReranker(mock_mode=True)
        result = reranker.rerank(QUERY, SAMPLE_CHUNKS)
        input_contents = {c.chunk.content for c in SAMPLE_CHUNKS}
        output_contents = {c.chunk.content for c in result}
        assert input_contents == output_contents, (
            "Reranker changed chunk content (should only change order/scores)"
        )

    def test_rerank_empty_input_returns_empty(self) -> None:
        """Reranker returns empty list when given empty input."""
        reranker = CrossEncoderReranker(mock_mode=True)
        result = reranker.rerank(QUERY, [])
        assert result == [], f"Expected [] for empty input, got {result}"

    def test_rerank_sorted_by_score_descending(self) -> None:
        """Output chunks are sorted highest-score first."""
        reranker = CrossEncoderReranker(mock_mode=True)
        result = reranker.rerank(QUERY, SAMPLE_CHUNKS)
        scores = [c.score for c in result]
        assert scores == sorted(scores, reverse=True), (
            f"Scores are not sorted descending: {scores}"
        )

    def test_rerank_single_chunk(self) -> None:
        """Reranker handles a single chunk without errors."""
        reranker = CrossEncoderReranker(mock_mode=True)
        single = [make_chunk("The only chunk.", 0)]
        result = reranker.rerank(QUERY, single)
        assert len(result) == 1
        assert result[0].chunk.content == "The only chunk."
        assert 0.0 <= result[0].score <= 1.0


class TestCohereReranker:
    """Tests for CohereReranker in mock mode."""

    def test_rerank_returns_correct_count(self) -> None:
        """CohereReranker returns same number of chunks as input."""
        reranker = CohereReranker(mock_mode=True)
        result = reranker.rerank(QUERY, SAMPLE_CHUNKS)
        assert len(result) == len(SAMPLE_CHUNKS), (
            f"Expected {len(SAMPLE_CHUNKS)} results, got {len(result)}"
        )

    def test_rerank_top_n_limits_results(self) -> None:
        """CohereReranker respects top_n parameter."""
        reranker = CohereReranker(mock_mode=True)
        result = reranker.rerank(QUERY, SAMPLE_CHUNKS, top_n=3)
        assert len(result) == 3, f"Expected 3 results with top_n=3, got {len(result)}"

    def test_rerank_scores_in_valid_range(self) -> None:
        """All output scores are within [0, 1]."""
        reranker = CohereReranker(mock_mode=True)
        result = reranker.rerank(QUERY, SAMPLE_CHUNKS)
        for chunk in result:
            assert 0.0 <= chunk.score <= 1.0, (
                f"Score {chunk.score} is outside [0, 1]"
            )

    def test_rerank_sets_retrieval_method_to_cohere(self) -> None:
        """Reranked chunks have retrieval_method='cohere_rerank'."""
        reranker = CohereReranker(mock_mode=True)
        result = reranker.rerank(QUERY, SAMPLE_CHUNKS)
        for chunk in result:
            assert chunk.retrieval_method == "cohere_rerank", (
                f"Expected retrieval_method='cohere_rerank', got {chunk.retrieval_method!r}"
            )

    def test_rerank_empty_input_returns_empty(self) -> None:
        """CohereReranker returns empty list for empty input."""
        reranker = CohereReranker(mock_mode=True)
        result = reranker.rerank(QUERY, [])
        assert result == []

    def test_rerank_scores_are_descending(self) -> None:
        """Cohere reranker returns chunks sorted highest score first."""
        reranker = CohereReranker(mock_mode=True)
        result = reranker.rerank(QUERY, SAMPLE_CHUNKS)
        scores = [c.score for c in result]
        assert scores == sorted(scores, reverse=True), (
            f"Scores not sorted descending: {scores}"
        )

    def test_rerank_changes_scores_from_input(self) -> None:
        """CohereReranker assigns new scores (not the original retrieval scores)."""
        reranker = CohereReranker(mock_mode=True)
        result = reranker.rerank(QUERY, SAMPLE_CHUNKS)
        input_scores = {c.score for c in SAMPLE_CHUNKS}
        output_scores = {c.score for c in result}
        # In mock mode the scores are deterministic (0.9, 0.85, ...) which differ
        # from the input scores (0.8, 0.7, 0.6, 0.5)
        assert output_scores != input_scores, (
            "Reranker scores should differ from retrieval scores"
        )


class TestCreateReranker:
    """Tests for the reranker factory function."""

    def test_factory_creates_cross_encoder(self) -> None:
        """create_reranker('cross_encoder') returns CrossEncoderReranker."""
        reranker = create_reranker("cross_encoder", mock_mode=True)
        assert isinstance(reranker, CrossEncoderReranker), (
            f"Expected CrossEncoderReranker, got {type(reranker)}"
        )

    def test_factory_creates_cohere(self) -> None:
        """create_reranker('cohere') returns CohereReranker."""
        reranker = create_reranker("cohere", mock_mode=True)
        assert isinstance(reranker, CohereReranker), (
            f"Expected CohereReranker, got {type(reranker)}"
        )

    def test_factory_raises_on_unknown_type(self) -> None:
        """create_reranker raises ValueError for unknown reranker_type."""
        with pytest.raises(ValueError, match="Unknown reranker type"):
            create_reranker("unknown", mock_mode=True)  # type: ignore[arg-type]

    def test_factory_cross_encoder_reranks_successfully(self) -> None:
        """Factory-created CrossEncoder can rerank a list of chunks."""
        reranker = create_reranker("cross_encoder", mock_mode=True)
        result = reranker.rerank(QUERY, SAMPLE_CHUNKS)
        assert len(result) == len(SAMPLE_CHUNKS)
        assert all(isinstance(c, RetrievedChunk) for c in result)
        # Confirm scores are non-trivially different from all-zero
        assert any(c.score > 0.0 for c in result), "All scores are 0 — mock scoring broken"

    def test_factory_cohere_reranks_successfully(self) -> None:
        """Factory-created CohereReranker can rerank a list of chunks."""
        reranker = create_reranker("cohere", mock_mode=True)
        result = reranker.rerank(QUERY, SAMPLE_CHUNKS)
        assert len(result) == len(SAMPLE_CHUNKS)
        assert all(isinstance(c, RetrievedChunk) for c in result)
        assert any(c.score > 0.0 for c in result), "All scores are 0 — mock scoring broken"
