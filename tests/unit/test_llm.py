"""Tests for LLM providers."""

from src.rag.config import RAGConfig, RunMode
from src.rag.document import Chunk, RetrievedChunk
from src.rag.llm import MockLLMProvider, create_llm_provider


def make_retrieved_chunk(content: str, score: float = 0.9) -> RetrievedChunk:
    chunk = Chunk(
        content=content,
        doc_id="doc-1",
        chunk_index=0,
        source="test.txt",
    )
    return RetrievedChunk(chunk=chunk, score=score, retrieval_method="test")


class TestMockLLMProvider:
    def test_generate_with_context(self) -> None:
        llm = MockLLMProvider()
        chunks = [
            make_retrieved_chunk("Machine learning is a field of AI."),
            make_retrieved_chunk("It focuses on learning from data."),
        ]
        result = llm.generate("What is ML?", chunks)
        assert result.is_ok()
        answer = result.unwrap()
        assert len(answer) > 0

    def test_generate_without_context(self) -> None:
        llm = MockLLMProvider()
        result = llm.generate("What is ML?", [])
        assert result.is_ok()
        answer = result.unwrap()
        assert "don't have enough context" in answer.lower()

    def test_deterministic(self) -> None:
        llm = MockLLMProvider()
        chunks = [make_retrieved_chunk("Test content")]
        r1 = llm.generate("question", chunks)
        r2 = llm.generate("question", chunks)
        assert r1.unwrap() == r2.unwrap()


class TestCreateLLMProvider:
    def test_mock_mode(self) -> None:
        config = RAGConfig(mode=RunMode.MOCK)
        provider = create_llm_provider(config)
        assert isinstance(provider, MockLLMProvider)

    def test_hybrid_mode_uses_mock_llm(self) -> None:
        config = RAGConfig(mode=RunMode.HYBRID)
        provider = create_llm_provider(config)
        assert isinstance(provider, MockLLMProvider)
