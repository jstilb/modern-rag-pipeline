"""Tests for retrieval components."""

import uuid

from src.rag.config import RAGConfig, RunMode
from src.rag.document import Chunk, Document
from src.rag.embeddings import MockEmbeddingProvider
from src.retrieval.keyword import KeywordRetriever
from src.retrieval.store import VectorStore


def make_chunks(n: int = 5) -> list[Chunk]:
    """Create n test chunks with distinct content."""
    topics = [
        "Machine learning algorithms process data to find patterns",
        "Neural networks consist of layers of interconnected nodes",
        "Natural language processing handles text and speech data",
        "Computer vision systems analyze images and video content",
        "Reinforcement learning agents learn through trial and error",
        "Deep learning models require large amounts of training data",
        "Transfer learning reuses pretrained model knowledge",
        "Data preprocessing is essential for model performance",
    ]
    return [
        Chunk(
            content=topics[i % len(topics)],
            doc_id=f"doc-{i}",
            chunk_index=i,
            source=f"source-{i}.md",
            token_count=len(topics[i % len(topics)].split()),
        )
        for i in range(n)
    ]


class TestKeywordRetriever:
    def test_index_and_retrieve(self) -> None:
        retriever = KeywordRetriever()
        chunks = make_chunks(5)

        result = retriever.index(chunks)
        assert result.is_ok()
        assert result.unwrap() == 5

        search_result = retriever.retrieve("machine learning algorithms", top_k=3)
        assert search_result.is_ok()
        retrieved = search_result.unwrap()
        assert len(retrieved) > 0
        assert retrieved[0].retrieval_method == "keyword"

    def test_retrieve_empty_index(self) -> None:
        retriever = KeywordRetriever()
        result = retriever.retrieve("test query")
        assert result.is_ok()
        assert len(result.unwrap()) == 0

    def test_index_empty_list(self) -> None:
        retriever = KeywordRetriever()
        result = retriever.index([])
        assert result.is_ok()
        assert result.unwrap() == 0

    def test_clear(self) -> None:
        retriever = KeywordRetriever()
        retriever.index(make_chunks(3))
        retriever.clear()
        result = retriever.retrieve("test")
        assert result.is_ok()
        assert len(result.unwrap()) == 0

    def test_score_normalization(self) -> None:
        retriever = KeywordRetriever()
        retriever.index(make_chunks(5))
        result = retriever.retrieve("machine learning", top_k=5)
        assert result.is_ok()
        for rc in result.unwrap():
            assert 0.0 <= rc.score <= 1.0


class TestVectorStore:
    def test_add_and_search(self) -> None:
        provider = MockEmbeddingProvider(dimensions=128)
        config = RAGConfig(
            mode=RunMode.MOCK,
            chroma_collection=f"test_{uuid.uuid4().hex[:8]}",
        )
        store = VectorStore(provider, config)

        chunks = make_chunks(3)
        result = store.add_chunks(chunks)
        assert result.is_ok()
        assert result.unwrap() == 3
        assert store.count == 3

        search_result = store.search("machine learning", top_k=2)
        assert search_result.is_ok()
        retrieved = search_result.unwrap()
        assert len(retrieved) <= 2

    def test_add_empty_list(self) -> None:
        provider = MockEmbeddingProvider(dimensions=128)
        config = RAGConfig(
            mode=RunMode.MOCK,
            chroma_collection=f"test_{uuid.uuid4().hex[:8]}",
        )
        store = VectorStore(provider, config)

        result = store.add_chunks([])
        assert result.is_ok()
        assert result.unwrap() == 0

    def test_search_empty_store(self) -> None:
        provider = MockEmbeddingProvider(dimensions=128)
        config = RAGConfig(
            mode=RunMode.MOCK,
            chroma_collection=f"test_{uuid.uuid4().hex[:8]}",
        )
        store = VectorStore(provider, config)

        result = store.search("test query")
        assert result.is_ok()
        assert len(result.unwrap()) == 0

    def test_clear(self) -> None:
        provider = MockEmbeddingProvider(dimensions=128)
        config = RAGConfig(
            mode=RunMode.MOCK,
            chroma_collection=f"test_{uuid.uuid4().hex[:8]}",
        )
        store = VectorStore(provider, config)

        store.add_chunks(make_chunks(3))
        assert store.count == 3

        result = store.clear()
        assert result.is_ok()
        assert store.count == 0
