"""Integration tests for the full RAG pipeline."""

import uuid

import pytest

from src.rag.config import RAGConfig, RunMode, ChunkingMethod, RetrievalMethod
from src.rag.document import Document
from src.rag.pipeline import RAGPipeline


def unique_config(**overrides: object) -> RAGConfig:
    """Create config with a unique collection name to avoid test pollution."""
    defaults = {
        "mode": RunMode.MOCK,
        "chroma_collection": f"test_{uuid.uuid4().hex[:8]}",
    }
    defaults.update(overrides)
    return RAGConfig(**defaults)  # type: ignore[arg-type]


SAMPLE_DOCS = [
    Document(
        content=(
            "Python is a high-level programming language known for its "
            "readability and simplicity. It supports multiple paradigms "
            "including procedural, object-oriented, and functional programming. "
            "Python is widely used in data science, machine learning, "
            "web development, and automation."
        ),
        source="python-overview.md",
    ),
    Document(
        content=(
            "TypeScript is a typed superset of JavaScript that compiles to "
            "plain JavaScript. It adds static type checking, interfaces, "
            "and other features that help catch errors at compile time. "
            "TypeScript is popular for large-scale web applications."
        ),
        source="typescript-overview.md",
    ),
    Document(
        content=(
            "Rust is a systems programming language focused on safety, "
            "speed, and concurrency. It prevents memory errors without "
            "garbage collection through its ownership system. Rust is used "
            "for performance-critical applications and systems programming."
        ),
        source="rust-overview.md",
    ),
]


class TestPipelineEndToEnd:
    def test_ingest_and_query(self) -> None:
        config = unique_config(chunk_size=50, top_k=3)
        pipeline = RAGPipeline(config)

        # Ingest
        result = pipeline.ingest(SAMPLE_DOCS)
        assert result.is_ok()
        assert result.unwrap() > 0
        assert pipeline.document_count > 0

        # Query
        query_result = pipeline.query("What is Python?")
        assert query_result.is_ok()

        gen = query_result.unwrap()
        assert len(gen.answer) > 0
        assert gen.query == "What is Python?"
        assert len(gen.retrieved_chunks) > 0
        assert gen.latency_ms > 0

    def test_different_chunking_methods(self) -> None:
        for method in ChunkingMethod:
            config = unique_config(
                chunking_method=method,
                chunk_size=50,
            )
            pipeline = RAGPipeline(config)
            result = pipeline.ingest(SAMPLE_DOCS)
            assert result.is_ok(), f"Failed with chunking method {method}"
            assert result.unwrap() > 0

    def test_different_retrieval_methods(self) -> None:
        for method in RetrievalMethod:
            config = unique_config(
                retrieval_method=method,
                chunk_size=50,
                top_k=2,
            )
            pipeline = RAGPipeline(config)
            pipeline.ingest(SAMPLE_DOCS)

            result = pipeline.query("programming language")
            assert result.is_ok(), f"Failed with retrieval method {method}"

    def test_empty_ingest(self) -> None:
        config = unique_config()
        pipeline = RAGPipeline(config)
        result = pipeline.ingest([])
        assert result.is_ok()
        assert result.unwrap() == 0

    def test_query_before_ingest(self) -> None:
        config = unique_config()
        pipeline = RAGPipeline(config)
        result = pipeline.query("test question")
        assert result.is_ok()
        gen = result.unwrap()
        assert "don't have enough context" in gen.answer.lower()

    def test_multiple_ingestions(self) -> None:
        config = unique_config(chunk_size=50)
        pipeline = RAGPipeline(config)

        r1 = pipeline.ingest([SAMPLE_DOCS[0]])
        assert r1.is_ok()
        count1 = pipeline.document_count

        r2 = pipeline.ingest([SAMPLE_DOCS[1]])
        assert r2.is_ok()
        count2 = pipeline.document_count
        assert count2 > count1

    def test_clear(self) -> None:
        config = unique_config(chunk_size=50)
        pipeline = RAGPipeline(config)

        pipeline.ingest(SAMPLE_DOCS)
        assert pipeline.document_count > 0

        result = pipeline.clear()
        assert result.is_ok()
        assert pipeline.document_count == 0

    def test_top_k_parameter(self) -> None:
        config = unique_config(chunk_size=30, top_k=10)
        pipeline = RAGPipeline(config)
        pipeline.ingest(SAMPLE_DOCS)

        result = pipeline.query("programming", top_k=2)
        assert result.is_ok()
        gen = result.unwrap()
        assert len(gen.retrieved_chunks) <= 2


class TestPipelineHybridRetrieval:
    def test_hybrid_search_combines_results(self) -> None:
        config = unique_config(
            retrieval_method=RetrievalMethod.HYBRID,
            chunk_size=50,
            top_k=3,
        )
        pipeline = RAGPipeline(config)
        pipeline.ingest(SAMPLE_DOCS)

        result = pipeline.query("Python data science")
        assert result.is_ok()
        gen = result.unwrap()
        assert len(gen.retrieved_chunks) > 0

        # Verify chunks have hybrid retrieval method
        methods = {rc.retrieval_method for rc in gen.retrieved_chunks}
        assert "hybrid" in methods
