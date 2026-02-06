"""Basic RAG pipeline example.

Demonstrates the core ingest-and-query workflow using mock mode.
No API keys required.

Usage:
    python examples/basic_pipeline.py
"""

from src.rag.config import RAGConfig, RunMode
from src.rag.document import Document
from src.rag.pipeline import RAGPipeline


def main() -> None:
    # 1. Configure pipeline in mock mode (no API keys needed)
    config = RAGConfig(
        mode=RunMode.MOCK,
        chunk_size=128,
        top_k=3,
    )
    pipeline = RAGPipeline(config)

    # 2. Create sample documents
    documents = [
        Document(
            content=(
                "FastAPI is a modern Python web framework for building APIs. "
                "It provides automatic OpenAPI documentation, type validation "
                "with Pydantic, and high performance using async/await. "
                "FastAPI is built on Starlette and is one of the fastest "
                "Python frameworks available."
            ),
            source="fastapi-docs.md",
            metadata={"topic": "web_framework"},
        ),
        Document(
            content=(
                "Docker containers package applications with their dependencies "
                "for consistent deployment across environments. A Dockerfile "
                "defines the build steps, and docker-compose orchestrates "
                "multi-container applications. Containers are lightweight "
                "compared to virtual machines."
            ),
            source="docker-guide.md",
            metadata={"topic": "devops"},
        ),
    ]

    # 3. Ingest documents
    ingest_result = pipeline.ingest(documents)
    if ingest_result.is_err():
        print(f"Ingest failed: {ingest_result.error}")
        return

    print(f"Ingested {ingest_result.unwrap()} chunks from {len(documents)} documents")

    # 4. Query the pipeline
    queries = [
        "What is FastAPI?",
        "How do Docker containers work?",
        "What web framework should I use for Python APIs?",
    ]

    for query in queries:
        result = pipeline.query(query)
        if result.is_ok():
            gen = result.unwrap()
            print(f"\nQ: {query}")
            print(f"A: {gen.answer[:200]}...")
            print(f"   Sources: {len(gen.retrieved_chunks)} chunks, {gen.latency_ms:.1f}ms")
        else:
            print(f"\nQ: {query}")
            print(f"   Error: {result.error}")


if __name__ == "__main__":
    main()
