"""Vector store abstraction over ChromaDB.

Provides document storage and similarity search with
dependency injection for the embedding provider.
"""

from __future__ import annotations

from typing import Optional

import chromadb
from chromadb.config import Settings

from src.rag.config import RAGConfig
from src.rag.document import Chunk, RetrievedChunk
from src.rag.embeddings import EmbeddingProvider
from src.rag.result import Err, Ok, Result


class VectorStore:
    """ChromaDB-backed vector store for document chunks."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        config: RAGConfig,
        client: Optional[chromadb.ClientAPI] = None,
    ) -> None:
        self._embeddings = embedding_provider
        self._config = config

        if client is not None:
            self._client = client
        else:
            self._client = chromadb.Client(
                Settings(
                    persist_directory=config.chroma_persist_dir,
                    anonymized_telemetry=False,
                )
            )

        self._collection = self._client.get_or_create_collection(
            name=config.chroma_collection,
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def count(self) -> int:
        """Return the number of documents in the store."""
        return self._collection.count()

    def add_chunks(self, chunks: list[Chunk]) -> Result[int, str]:
        """Add chunks to the vector store."""
        if not chunks:
            return Ok(0)

        texts = [c.content for c in chunks]
        embed_result = self._embeddings.embed_texts(texts)

        if embed_result.is_err():
            return Err(f"Embedding failed: {embed_result.error}")  # type: ignore[union-attr]

        embeddings = embed_result.unwrap()

        ids = [c.chunk_id for c in chunks]
        metadatas = [
            {
                "doc_id": c.doc_id,
                "chunk_index": c.chunk_index,
                "source": c.source,
                "token_count": c.token_count,
                **{k: str(v) for k, v in c.metadata.items()},
            }
            for c in chunks
        ]

        try:
            self._collection.add(
                ids=ids,
                embeddings=embeddings,  # type: ignore[arg-type]
                documents=texts,
                metadatas=metadatas,  # type: ignore[arg-type]
            )
            return Ok(len(chunks))
        except Exception as e:
            return Err(f"ChromaDB add failed: {e}")

    def search(
        self, query: str, top_k: Optional[int] = None
    ) -> Result[list[RetrievedChunk], str]:
        """Search for similar chunks using vector similarity."""
        k = top_k or self._config.top_k

        query_embed_result = self._embeddings.embed_query(query)
        if query_embed_result.is_err():
            return Err(f"Query embedding failed: {query_embed_result.error}")  # type: ignore[union-attr]

        query_embedding = query_embed_result.unwrap()

        try:
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=min(k, self._collection.count()) if self._collection.count() > 0 else k,
                include=["documents", "metadatas", "distances"],
            )

            if not results["documents"] or not results["documents"][0]:
                return Ok([])

            retrieved: list[RetrievedChunk] = []
            for i, doc in enumerate(results["documents"][0]):
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}  # type: ignore[index]
                distance = results["distances"][0][i] if results["distances"] else 0.0  # type: ignore[index]

                # Convert cosine distance to similarity score
                score = max(0.0, min(1.0, 1.0 - distance))

                chunk = Chunk(
                    content=doc,
                    doc_id=str(metadata.get("doc_id", "")),  # type: ignore[union-attr]
                    chunk_index=int(metadata.get("chunk_index", 0)),  # type: ignore[union-attr]
                    source=str(metadata.get("source", "")),  # type: ignore[union-attr]
                    token_count=int(metadata.get("token_count", 0)),  # type: ignore[union-attr]
                )

                retrieved.append(
                    RetrievedChunk(
                        chunk=chunk,
                        score=score,
                        retrieval_method="semantic",
                    )
                )

            return Ok(retrieved)
        except Exception as e:
            return Err(f"ChromaDB search failed: {e}")

    def clear(self) -> Result[None, str]:
        """Clear all documents from the store."""
        try:
            self._client.delete_collection(self._config.chroma_collection)
            self._collection = self._client.get_or_create_collection(
                name=self._config.chroma_collection,
                metadata={"hnsw:space": "cosine"},
            )
            return Ok(None)
        except Exception as e:
            return Err(f"Clear failed: {e}")
