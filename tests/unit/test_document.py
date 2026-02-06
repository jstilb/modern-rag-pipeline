"""Tests for document models."""

import pytest
from hypothesis import given, strategies as st

from src.rag.document import Chunk, Document, GenerationResult, RetrievedChunk


class TestDocument:
    def test_create_document(self) -> None:
        doc = Document(content="Hello world", source="test.txt")
        assert doc.content == "Hello world"
        assert doc.source == "test.txt"
        assert doc.doc_id  # UUID generated
        assert doc.created_at  # Timestamp generated

    def test_empty_content_raises(self) -> None:
        with pytest.raises(ValueError, match="content cannot be empty"):
            Document(content="", source="test.txt")

    def test_whitespace_content_raises(self) -> None:
        with pytest.raises(ValueError, match="content cannot be empty"):
            Document(content="   ", source="test.txt")

    def test_empty_source_raises(self) -> None:
        with pytest.raises(ValueError, match="source cannot be empty"):
            Document(content="Hello", source="")

    def test_metadata_default(self) -> None:
        doc = Document(content="Hello", source="test.txt")
        assert doc.metadata == {}

    def test_metadata_custom(self) -> None:
        doc = Document(
            content="Hello", source="test.txt", metadata={"key": "value"}
        )
        assert doc.metadata == {"key": "value"}

    @given(st.text(min_size=1, max_size=100).filter(lambda x: x.strip()))
    def test_valid_content_accepted(self, content: str) -> None:
        doc = Document(content=content, source="test.txt")
        assert doc.content == content


class TestChunk:
    def test_create_chunk(self) -> None:
        chunk = Chunk(
            content="chunk text",
            doc_id="doc-1",
            chunk_index=0,
            source="test.txt",
        )
        assert chunk.content == "chunk text"
        assert chunk.chunk_index == 0

    def test_empty_content_raises(self) -> None:
        with pytest.raises(ValueError, match="content cannot be empty"):
            Chunk(content="", doc_id="doc-1", chunk_index=0, source="test.txt")


class TestRetrievedChunk:
    def test_valid_score(self) -> None:
        chunk = Chunk(
            content="text", doc_id="doc-1", chunk_index=0, source="test.txt"
        )
        rc = RetrievedChunk(chunk=chunk, score=0.85, retrieval_method="semantic")
        assert rc.score == 0.85

    def test_score_out_of_range(self) -> None:
        chunk = Chunk(
            content="text", doc_id="doc-1", chunk_index=0, source="test.txt"
        )
        with pytest.raises(ValueError, match="Score must be between"):
            RetrievedChunk(chunk=chunk, score=1.5, retrieval_method="semantic")

    def test_negative_score(self) -> None:
        chunk = Chunk(
            content="text", doc_id="doc-1", chunk_index=0, source="test.txt"
        )
        with pytest.raises(ValueError, match="Score must be between"):
            RetrievedChunk(chunk=chunk, score=-0.1, retrieval_method="semantic")

    @given(st.floats(min_value=0.0, max_value=1.0))
    def test_valid_scores_accepted(self, score: float) -> None:
        chunk = Chunk(
            content="text", doc_id="doc-1", chunk_index=0, source="test.txt"
        )
        rc = RetrievedChunk(chunk=chunk, score=score, retrieval_method="test")
        assert rc.score == score


class TestGenerationResult:
    def test_create_result(self) -> None:
        result = GenerationResult(
            answer="The answer is 42",
            query="What is the answer?",
            retrieved_chunks=[],
            model="mock",
            latency_ms=10.5,
        )
        assert result.answer == "The answer is 42"
        assert result.generation_id  # UUID generated
