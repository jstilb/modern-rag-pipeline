"""CLI interface for the RAG pipeline.

Provides command-line access to pipeline operations:
- ingest: Add documents from files
- query: Ask questions
- demo: Run a complete demo with sample data
- serve: Start the FastAPI server
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.rag.config import RAGConfig, RunMode
from src.rag.document import Document
from src.rag.pipeline import RAGPipeline


SAMPLE_DOCUMENTS = [
    Document(
        content=(
            "Retrieval-Augmented Generation (RAG) is a technique that combines "
            "information retrieval with text generation. It was introduced by "
            "Lewis et al. in 2020. RAG systems first retrieve relevant documents "
            "from a knowledge base, then use those documents as context for a "
            "language model to generate answers. This approach reduces hallucination "
            "and allows the model to access up-to-date information beyond its "
            "training data. RAG has become a fundamental pattern in enterprise "
            "AI applications."
        ),
        source="rag-overview.md",
        metadata={"topic": "RAG", "type": "overview"},
    ),
    Document(
        content=(
            "Vector databases are specialized storage systems optimized for "
            "similarity search over high-dimensional vectors. Popular options "
            "include ChromaDB, Pinecone, Weaviate, and Qdrant. ChromaDB is an "
            "open-source embedding database that runs locally or in the cloud. "
            "It supports cosine similarity, L2 distance, and inner product "
            "metrics. Vector databases enable efficient nearest-neighbor search "
            "which is critical for RAG retrieval performance."
        ),
        source="vector-databases.md",
        metadata={"topic": "vector_db", "type": "reference"},
    ),
    Document(
        content=(
            "Chunking strategies determine how documents are split into smaller "
            "pieces for retrieval. Fixed-size chunking splits by token count. "
            "Recursive chunking respects natural boundaries like paragraphs and "
            "sentences. Semantic chunking groups content by topic similarity. "
            "The choice of chunking strategy significantly affects retrieval "
            "quality. Smaller chunks enable more precise retrieval but may lose "
            "context. Larger chunks preserve context but may include irrelevant "
            "information. A typical chunk size is 256-512 tokens with 10-20% overlap."
        ),
        source="chunking-strategies.md",
        metadata={"topic": "chunking", "type": "guide"},
    ),
    Document(
        content=(
            "Hybrid search combines semantic search (vector similarity) with "
            "keyword search (BM25) to get the best of both approaches. Semantic "
            "search excels at understanding meaning and paraphrases but can miss "
            "exact keyword matches. BM25 excels at finding specific terms, acronyms, "
            "and proper nouns. Reciprocal Rank Fusion (RRF) is a popular method "
            "for combining results from multiple retrieval systems. It assigns "
            "scores based on rank position rather than raw scores, making it "
            "robust to score scale differences."
        ),
        source="hybrid-search.md",
        metadata={"topic": "retrieval", "type": "guide"},
    ),
    Document(
        content=(
            "RAG evaluation requires measuring both retrieval quality and generation "
            "quality. Retrieval metrics include precision@k, recall@k, MRR (Mean "
            "Reciprocal Rank), and NDCG (Normalized Discounted Cumulative Gain). "
            "Generation metrics include faithfulness (is the answer grounded in "
            "context?), answer relevance (does it address the question?), and "
            "context utilization (how much context was used?). End-to-end evaluation "
            "combines both to assess overall system quality. Frameworks like RAGAS "
            "provide automated evaluation pipelines."
        ),
        source="evaluation-metrics.md",
        metadata={"topic": "evaluation", "type": "reference"},
    ),
]


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Modern RAG Pipeline - Production-ready RAG with hybrid search"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run a complete demo")
    demo_parser.add_argument(
        "--query",
        default="What is RAG and how does it work?",
        help="Query to demo",
    )

    # Query command
    query_parser = subparsers.add_parser("query", help="Query the pipeline")
    query_parser.add_argument("question", help="Question to ask")
    query_parser.add_argument("--top-k", type=int, default=5, help="Number of chunks")

    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents")
    ingest_parser.add_argument("files", nargs="+", help="Files to ingest")

    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Start API server")
    serve_parser.add_argument("--host", default="0.0.0.0", help="Host")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port")

    args = parser.parse_args()

    if args.command == "demo":
        run_demo(args.query)
    elif args.command == "query":
        run_query(args.question, args.top_k)
    elif args.command == "ingest":
        run_ingest(args.files)
    elif args.command == "serve":
        run_serve(args.host, args.port)
    else:
        parser.print_help()
        sys.exit(1)


def run_demo(query: str) -> None:
    """Run a complete demo with sample data."""
    print("=" * 60)
    print("Modern RAG Pipeline - Demo Mode")
    print("=" * 60)
    print()

    config = RAGConfig(mode=RunMode.MOCK)
    pipeline = RAGPipeline(config)

    print(f"[1/3] Ingesting {len(SAMPLE_DOCUMENTS)} sample documents...")
    result = pipeline.ingest(SAMPLE_DOCUMENTS)
    if result.is_err():
        print(f"ERROR: {result.error}")  # type: ignore[union-attr]
        sys.exit(1)
    print(f"      Indexed {result.unwrap()} chunks")
    print()

    print(f'[2/3] Querying: "{query}"')
    print()

    gen_result = pipeline.query(query)
    if gen_result.is_err():
        print(f"ERROR: {gen_result.error}")  # type: ignore[union-attr]
        sys.exit(1)

    output = gen_result.unwrap()

    print("[3/3] Results:")
    print("-" * 60)
    print(f"Answer: {output.answer}")
    print()
    print(f"Model: {output.model}")
    print(f"Latency: {output.latency_ms:.1f}ms")
    print(f"Sources: {len(output.retrieved_chunks)} chunks retrieved")
    print()
    for i, rc in enumerate(output.retrieved_chunks, 1):
        print(f"  [{i}] {rc.chunk.source} (score: {rc.score:.3f}, method: {rc.retrieval_method})")
        print(f"      {rc.chunk.content[:100]}...")
        print()
    print("=" * 60)

    # JSON output for programmatic use
    json_output = {
        "answer": output.answer,
        "query": output.query,
        "model": output.model,
        "latency_ms": output.latency_ms,
        "sources": [
            {
                "source": rc.chunk.source,
                "score": rc.score,
                "method": rc.retrieval_method,
            }
            for rc in output.retrieved_chunks
        ],
    }
    print("JSON output:")
    print(json.dumps(json_output, indent=2))


def run_query(question: str, top_k: int) -> None:
    """Query an existing pipeline."""
    config = RAGConfig(mode=RunMode.MOCK)
    pipeline = RAGPipeline(config)

    # For standalone queries, load sample data
    pipeline.ingest(SAMPLE_DOCUMENTS)

    result = pipeline.query(question, top_k=top_k)
    if result.is_err():
        print(f"ERROR: {result.error}")  # type: ignore[union-attr]
        sys.exit(1)

    output = result.unwrap()
    print(json.dumps({
        "answer": output.answer,
        "query": output.query,
        "model": output.model,
        "latency_ms": output.latency_ms,
        "source_count": len(output.retrieved_chunks),
    }, indent=2))


def run_ingest(files: list[str]) -> None:
    """Ingest documents from files."""
    config = RAGConfig(mode=RunMode.MOCK)
    pipeline = RAGPipeline(config)

    documents = []
    for file_path in files:
        path = Path(file_path)
        if not path.exists():
            print(f"WARNING: File not found: {file_path}")
            continue
        content = path.read_text()
        documents.append(Document(content=content, source=str(path)))

    if documents:
        result = pipeline.ingest(documents)
        if result.is_err():
            print(f"ERROR: {result.error}")  # type: ignore[union-attr]
            sys.exit(1)
        print(f"Indexed {result.unwrap()} chunks from {len(documents)} documents")
    else:
        print("No valid documents to ingest")
        sys.exit(1)


def run_serve(host: str, port: int) -> None:
    """Start the FastAPI server."""
    import uvicorn
    uvicorn.run("src.api.app:app", host=host, port=port, reload=True)


if __name__ == "__main__":
    main()
