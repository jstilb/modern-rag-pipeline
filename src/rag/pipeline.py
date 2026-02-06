"""Main RAG pipeline orchestrating ingestion, retrieval, and generation.

The pipeline is the primary entry point for the RAG system. It:
1. Ingests documents by chunking and indexing them
2. Retrieves relevant chunks for a query
3. Generates an answer using the retrieved context

All external dependencies are injected, enabling mock mode
for demos and testing without API keys.
"""

from __future__ import annotations

import time
from typing import Optional

from src.chunking.strategies import (
    ChunkingStrategy,
    FixedSizeChunker,
    RecursiveChunker,
    SemanticChunker,
    SlidingWindowChunker,
)
from src.rag.config import ChunkingMethod, RAGConfig, RetrievalMethod, RunMode
from src.rag.document import Chunk, Document, GenerationResult, RetrievedChunk
from src.rag.embeddings import EmbeddingProvider, create_embedding_provider
from src.rag.llm import LLMProvider, create_llm_provider
from src.rag.result import Err, Ok, Result
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.keyword import KeywordRetriever
from src.retrieval.semantic import SemanticRetriever
from src.retrieval.store import VectorStore


class RAGPipeline:
    """Production RAG pipeline with pluggable components.

    Usage:
        config = RAGConfig(mode=RunMode.MOCK)
        pipeline = RAGPipeline(config)

        # Ingest documents
        pipeline.ingest([Document(content="...", source="file.txt")])

        # Query
        result = pipeline.query("What is X?")
    """

    def __init__(
        self,
        config: Optional[RAGConfig] = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
        llm_provider: Optional[LLMProvider] = None,
    ) -> None:
        self._config = config or RAGConfig()

        # Dependency injection with sensible defaults
        self._embeddings = embedding_provider or create_embedding_provider(self._config)
        self._llm = llm_provider or create_llm_provider(self._config)

        # Initialize components
        self._store = VectorStore(self._embeddings, self._config)
        self._keyword_retriever = KeywordRetriever()
        self._semantic_retriever = SemanticRetriever(self._store)
        self._hybrid_retriever = HybridRetriever(
            self._semantic_retriever, self._keyword_retriever, self._config
        )

        # Track all chunks for keyword retrieval
        self._all_chunks: list[Chunk] = []

        # Select chunking strategy
        self._chunker = self._create_chunker()

    def _create_chunker(self) -> ChunkingStrategy:
        """Create the configured chunking strategy."""
        chunk_size = self._config.chunk_size
        overlap = min(self._config.chunk_overlap, chunk_size - 1)

        strategy_map: dict[ChunkingMethod, ChunkingStrategy] = {
            ChunkingMethod.FIXED: FixedSizeChunker(
                chunk_size=chunk_size, overlap=overlap
            ),
            ChunkingMethod.RECURSIVE: RecursiveChunker(
                chunk_size=chunk_size, overlap=overlap
            ),
            ChunkingMethod.SEMANTIC: SemanticChunker(chunk_size=chunk_size),
            ChunkingMethod.SLIDING_WINDOW: SlidingWindowChunker(
                window_size=chunk_size, step_size=max(1, chunk_size - overlap)
            ),
        }
        return strategy_map[self._config.chunking_method]

    def ingest(self, documents: list[Document]) -> Result[int, str]:
        """Ingest documents into the pipeline.

        Chunks documents and indexes them for both semantic and keyword retrieval.

        Returns:
            Result with total number of chunks indexed, or error message.
        """
        if not documents:
            return Ok(0)

        total_chunks = 0

        for doc in documents:
            chunks = self._chunker.chunk(doc)
            if not chunks:
                continue

            # Add to vector store
            store_result = self._store.add_chunks(chunks)
            if store_result.is_err():
                return Err(f"Vector store indexing failed: {store_result.error}")  # type: ignore[union-attr]

            # Track for keyword retrieval
            self._all_chunks.extend(chunks)
            total_chunks += len(chunks)

        # Rebuild BM25 index with all chunks
        kw_result = self._keyword_retriever.index(self._all_chunks)
        if kw_result.is_err():
            return Err(f"Keyword indexing failed: {kw_result.error}")  # type: ignore[union-attr]

        return Ok(total_chunks)

    def query(self, query: str, top_k: Optional[int] = None) -> Result[GenerationResult, str]:
        """Query the pipeline and generate an answer.

        Args:
            query: The user's question
            top_k: Override number of chunks to retrieve

        Returns:
            Result with GenerationResult or error message.
        """
        start_time = time.monotonic()
        k = top_k or self._config.top_k

        # Retrieve relevant chunks
        retrieval_result = self._retrieve(query, k)
        if retrieval_result.is_err():
            return Err(f"Retrieval failed: {retrieval_result.error}")  # type: ignore[union-attr]

        retrieved_chunks = retrieval_result.unwrap()

        # Generate answer
        gen_result = self._llm.generate(query, retrieved_chunks)
        if gen_result.is_err():
            return Err(f"Generation failed: {gen_result.error}")  # type: ignore[union-attr]

        answer = gen_result.unwrap()
        elapsed_ms = (time.monotonic() - start_time) * 1000

        return Ok(
            GenerationResult(
                answer=answer,
                query=query,
                retrieved_chunks=retrieved_chunks,
                model=self._config.llm_model if self._config.mode != RunMode.MOCK else "mock",
                latency_ms=elapsed_ms,
            )
        )

    def _retrieve(
        self, query: str, top_k: int
    ) -> Result[list[RetrievedChunk], str]:
        """Retrieve chunks using the configured retrieval strategy."""
        method = self._config.retrieval_method

        if method == RetrievalMethod.SEMANTIC:
            return self._semantic_retriever.retrieve(query, top_k=top_k)
        elif method == RetrievalMethod.KEYWORD:
            return self._keyword_retriever.retrieve(query, top_k=top_k)
        else:  # HYBRID
            return self._hybrid_retriever.retrieve(query, top_k=top_k)

    @property
    def document_count(self) -> int:
        """Return the number of indexed chunks."""
        return self._store.count

    def clear(self) -> Result[None, str]:
        """Clear all indexed documents."""
        self._all_chunks = []
        self._keyword_retriever.clear()
        return self._store.clear()
