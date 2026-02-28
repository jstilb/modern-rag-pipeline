"""Qdrant vector store backend.

Provides document storage and similarity search using Qdrant,
mirroring the VectorStore (ChromaDB) interface for drop-in compatibility.

Supports both in-memory mode (for testing and development) and
remote Qdrant Cloud / self-hosted instances.
"""

from __future__ import annotations

import uuid
from typing import Any, Optional

from src.rag.config import RAGConfig
from src.rag.document import Chunk, RetrievedChunk
from src.rag.embeddings import EmbeddingProvider
from src.rag.result import Err, Ok, Result


class QdrantVectorStore:
    """Qdrant-backed vector store for document chunks.

    Mirrors the VectorStore (ChromaDB) interface so the two backends
    are interchangeable. Use location=":memory:" for in-process testing.

    Args:
        embedding_provider: Provider used to generate embedding vectors.
        config: RAG pipeline configuration.
        location: Qdrant server URL or ":memory:" for in-process storage.
            Defaults to ":memory:" for safe operation without a running server.
        collection_name: Name of the Qdrant collection. Defaults to config value.
        timeout: Request timeout in seconds for Qdrant API calls.
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        config: RAGConfig,
        location: str = ":memory:",
        collection_name: Optional[str] = None,
        timeout: float = 30.0,
    ) -> None:
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
        except ImportError as exc:
            raise ImportError(
                "qdrant-client is required for QdrantVectorStore. "
                "Install it with: pip install 'qdrant-client>=1.7.0'"
            ) from exc

        self._embeddings = embedding_provider
        self._config = config
        self._collection_name = collection_name or config.chroma_collection
        self._timeout = timeout

        # Initialise Qdrant client
        if location == ":memory:":
            self._client = QdrantClient(location=":memory:")
        else:
            self._client = QdrantClient(url=location, timeout=timeout)

        # Create collection if it does not already exist
        existing = {c.name for c in self._client.get_collections().collections}
        if self._collection_name not in existing:
            self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config=VectorParams(
                    size=config.embedding_dimensions,
                    distance=Distance.COSINE,
                ),
            )

        # Keep a local list of chunk objects for retrieval reconstruction
        # (Qdrant payloads store metadata; we reconstruct Chunk from payload)
        self._id_to_chunk: dict[str, Chunk] = {}

    @property
    def count(self) -> int:
        """Return the number of points in the Qdrant collection."""
        info = self._client.get_collection(self._collection_name)
        return info.points_count or 0

    def add_chunks(self, chunks: list[Chunk]) -> Result[int, str]:
        """Add chunks to the Qdrant collection.

        Args:
            chunks: List of Chunk objects to index.

        Returns:
            Ok(number of chunks added) or Err(error message).
        """
        if not chunks:
            return Ok(0)

        from qdrant_client.models import PointStruct

        texts = [c.content for c in chunks]
        embed_result = self._embeddings.embed_texts(texts)

        if embed_result.is_err():
            return Err(f"Embedding failed: {embed_result.error}")  # type: ignore[union-attr]

        embeddings = embed_result.unwrap()

        points: list[PointStruct] = []
        for chunk, vector in zip(chunks, embeddings, strict=True):
            # Qdrant requires integer or UUID point IDs
            point_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.chunk_id))
            payload: dict[str, Any] = {
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "chunk_index": chunk.chunk_index,
                "source": chunk.source,
                "token_count": chunk.token_count,
                "content": chunk.content,
                **{k: str(v) for k, v in chunk.metadata.items()},
            }
            points.append(PointStruct(id=point_uuid, vector=vector, payload=payload))
            self._id_to_chunk[point_uuid] = chunk

        try:
            self._client.upsert(
                collection_name=self._collection_name,
                points=points,
                wait=True,
            )
            return Ok(len(chunks))
        except Exception as exc:
            return Err(f"Qdrant upsert failed: {exc}")

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> Result[list[RetrievedChunk], str]:
        """Search for similar chunks using Qdrant vector similarity.

        Args:
            query: The query string to embed and search with.
            top_k: Number of results to return. Defaults to config.top_k.

        Returns:
            Ok(list of RetrievedChunk) or Err(error message).
        """
        k = top_k or self._config.top_k

        query_embed_result = self._embeddings.embed_query(query)
        if query_embed_result.is_err():
            return Err(
                f"Query embedding failed: {query_embed_result.error}"  # type: ignore[union-attr]
            )

        query_vector = query_embed_result.unwrap()

        try:
            results = self._client.search(
                collection_name=self._collection_name,
                query_vector=query_vector,
                limit=k,
                with_payload=True,
            )
        except Exception as exc:
            return Err(f"Qdrant search failed: {exc}")

        retrieved: list[RetrievedChunk] = []
        for hit in results:
            payload = hit.payload or {}
            score = float(hit.score)
            # Qdrant cosine scores are already in [-1, 1]; normalise to [0, 1]
            normalised_score = max(0.0, min(1.0, (score + 1.0) / 2.0))

            chunk = Chunk(
                content=str(payload.get("content", "")),
                doc_id=str(payload.get("doc_id", "")),
                chunk_index=int(payload.get("chunk_index", 0)),
                source=str(payload.get("source", "")),
                token_count=int(payload.get("token_count", 0)),
                chunk_id=str(payload.get("chunk_id", str(hit.id))),
            )
            retrieved.append(
                RetrievedChunk(
                    chunk=chunk,
                    score=normalised_score,
                    retrieval_method="qdrant_semantic",
                )
            )

        return Ok(retrieved)

    def clear(self) -> Result[None, str]:
        """Delete and recreate the Qdrant collection."""
        try:
            from qdrant_client.models import Distance, VectorParams

            self._client.delete_collection(self._collection_name)
            self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config=VectorParams(
                    size=self._config.embedding_dimensions,
                    distance=Distance.COSINE,
                ),
            )
            self._id_to_chunk.clear()
            return Ok(None)
        except Exception as exc:
            return Err(f"Clear failed: {exc}")
