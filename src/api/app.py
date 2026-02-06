"""FastAPI REST API for the RAG pipeline.

Provides async endpoints for document ingestion, querying,
and health checks. Supports both production and mock modes.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.rag.config import RAGConfig, RunMode
from src.rag.document import Document
from src.rag.pipeline import RAGPipeline


# --- Request/Response Models ---


class IngestRequest(BaseModel):
    """Request body for document ingestion."""

    documents: list[DocumentInput]


class DocumentInput(BaseModel):
    """A single document to ingest."""

    content: str = Field(..., min_length=1, description="Document text content")
    source: str = Field(..., min_length=1, description="Document source identifier")
    metadata: dict[str, str | int | float | bool] = Field(
        default_factory=dict, description="Optional metadata"
    )


class IngestResponse(BaseModel):
    """Response from document ingestion."""

    chunks_indexed: int
    total_documents: int
    status: str = "success"


class QueryRequest(BaseModel):
    """Request body for querying the pipeline."""

    query: str = Field(..., min_length=1, description="The question to answer")
    top_k: Optional[int] = Field(default=None, ge=1, le=20, description="Number of chunks")


class QueryResponse(BaseModel):
    """Response from a query."""

    answer: str
    query: str
    sources: list[SourceInfo]
    model: str
    latency_ms: float


class SourceInfo(BaseModel):
    """Information about a retrieved source chunk."""

    content: str
    source: str
    score: float
    retrieval_method: str


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    mode: str
    document_count: int
    version: str = "0.1.0"


# --- Application ---

_pipeline: Optional[RAGPipeline] = None


def get_pipeline() -> RAGPipeline:
    """Get or create the global pipeline instance."""
    global _pipeline
    if _pipeline is None:
        config = RAGConfig()
        _pipeline = RAGPipeline(config)
    return _pipeline


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler."""
    # Startup: initialize pipeline
    get_pipeline()
    yield
    # Shutdown: cleanup


def create_app(config: Optional[RAGConfig] = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        config: Optional RAGConfig. Defaults to environment-based config.
    """
    app = FastAPI(
        title="Modern RAG Pipeline",
        description="Production-ready RAG pipeline with hybrid search and evaluation",
        version="0.1.0",
        lifespan=lifespan,
    )

    if config is not None:
        global _pipeline
        _pipeline = RAGPipeline(config)

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        """Health check endpoint."""
        pipeline = get_pipeline()
        return HealthResponse(
            status="healthy",
            mode=pipeline._config.mode.value,
            document_count=pipeline.document_count,
        )

    @app.post("/ingest", response_model=IngestResponse)
    async def ingest(request: IngestRequest) -> IngestResponse:
        """Ingest documents into the RAG pipeline."""
        pipeline = get_pipeline()

        documents = [
            Document(
                content=doc.content,
                source=doc.source,
                metadata=doc.metadata,
            )
            for doc in request.documents
        ]

        result = pipeline.ingest(documents)
        if result.is_err():
            raise HTTPException(status_code=500, detail=str(result.error))  # type: ignore[union-attr]

        return IngestResponse(
            chunks_indexed=result.unwrap(),
            total_documents=len(documents),
        )

    @app.post("/query", response_model=QueryResponse)
    async def query(request: QueryRequest) -> QueryResponse:
        """Query the RAG pipeline."""
        pipeline = get_pipeline()

        result = pipeline.query(request.query, top_k=request.top_k)
        if result.is_err():
            raise HTTPException(status_code=500, detail=str(result.error))  # type: ignore[union-attr]

        gen_result = result.unwrap()

        sources = [
            SourceInfo(
                content=rc.chunk.content[:200],
                source=rc.chunk.source,
                score=rc.score,
                retrieval_method=rc.retrieval_method,
            )
            for rc in gen_result.retrieved_chunks
        ]

        return QueryResponse(
            answer=gen_result.answer,
            query=gen_result.query,
            sources=sources,
            model=gen_result.model,
            latency_ms=gen_result.latency_ms,
        )

    return app


# Default app instance for uvicorn
app = create_app()
