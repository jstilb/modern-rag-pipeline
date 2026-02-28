#!/usr/bin/env python3
"""Wikipedia RAG example.

Demonstrates the modern-rag-pipeline end-to-end using hardcoded Wikipedia
article text about retrieval-augmented generation. Runs in mock mode by
default — no API keys or internet access required.

Usage::

    python examples/wikipedia_rag.py
    python examples/wikipedia_rag.py --mock
    python examples/wikipedia_rag.py --query "What is RAG?"
"""

from __future__ import annotations

import argparse
import sys


# ---------------------------------------------------------------------------
# Hardcoded Wikipedia article text (no live API calls)
# Source: https://en.wikipedia.org/wiki/Retrieval-augmented_generation
# ---------------------------------------------------------------------------

WIKIPEDIA_ARTICLE = """
Retrieval-augmented generation (RAG) is a technique for enhancing the accuracy
and reliability of generative AI models with facts fetched from external sources.

RAG is an approach where the model is given a mechanism to retrieve relevant
information before generating an answer. This differs from pure generative
models, which rely solely on the information encoded during training.

== How RAG Works ==

The RAG architecture typically consists of three components:

1. Retriever: Indexes a corpus of documents and fetches the most relevant
   passages for a given query. Common retrieval methods include dense
   vector search (semantic retrieval), sparse keyword search (BM25), and
   hybrid combinations.

2. Reader/Generator: A large language model (LLM) that takes the retrieved
   passages as context and generates a grounded answer to the query.

3. Knowledge Base: The external document corpus that provides up-to-date
   information beyond the LLM's training cutoff.

== Retrieval Methods ==

Dense retrieval encodes documents and queries into dense vector representations
using a bi-encoder neural network. Similarity is computed using cosine distance
or inner product in a high-dimensional embedding space.

Sparse retrieval (BM25) uses term-frequency weighting to score documents based
on lexical overlap with the query. It excels at exact keyword matching.

Hybrid retrieval combines both approaches, often using Reciprocal Rank Fusion
(RRF) to merge the ranked lists. Studies show hybrid retrieval consistently
outperforms either method alone, with NDCG improvements of 10–20%.

== Chunking Strategies ==

Documents are split into chunks before indexing. Common strategies include:

- Fixed-size chunking: splits text into fixed-length windows (e.g., 512 tokens).
- Recursive chunking: splits on paragraph/sentence boundaries recursively.
- Semantic chunking: groups sentences by semantic similarity.
- Sliding window: uses overlapping windows to preserve context at boundaries.

The choice of chunking strategy significantly affects retrieval quality.
Recursive chunking is generally considered the best default.

== Applications ==

RAG is widely used in:
- Enterprise Q&A systems over private documents
- Customer support chatbots grounded in product documentation
- Legal and medical question answering
- Code documentation search
- Scientific literature exploration

== Evaluation ==

RAG systems are evaluated on:
- Faithfulness: Is the answer supported by the retrieved context?
- Answer Relevance: Does the answer address the query?
- Context Precision: Are the retrieved chunks relevant to the query?
- NDCG: Normalized Discounted Cumulative Gain for retrieval quality.

The RAGAS framework provides automated evaluation of all four dimensions.
"""

# Hardcoded benchmark questions with expected answer hints
BENCHMARK_QUESTIONS = [
    "What is retrieval-augmented generation?",
    "How does hybrid search improve RAG performance?",
    "What chunking strategy is best for most documents?",
    "What are the main components of a RAG architecture?",
    "How is faithfulness evaluated in RAG systems?",
]


def run_wikipedia_rag(query: str, mock: bool = True) -> str:
    """Run the RAG pipeline on the Wikipedia article and return an answer.

    Args:
        query: The question to answer.
        mock: If True, run in mock mode (no API keys needed).

    Returns:
        The generated answer string.
    """
    import sys
    import os

    # Ensure the project root is on the path
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from src.rag.config import RAGConfig, RunMode
    from src.rag.document import Document
    from src.rag.pipeline import RAGPipeline

    mode = RunMode.MOCK if mock else RunMode.PRODUCTION
    config = RAGConfig(mode=mode)
    pipeline = RAGPipeline(config)

    # Ingest the Wikipedia article
    doc = Document(
        content=WIKIPEDIA_ARTICLE,
        source="wikipedia:Retrieval-augmented_generation",
        metadata={"topic": "RAG", "language": "en"},
    )
    ingest_result = pipeline.ingest([doc])
    if ingest_result.is_err():
        raise RuntimeError(f"Ingestion failed: {ingest_result.error}")  # type: ignore[union-attr]

    # Run the query
    query_result = pipeline.query(query)
    if query_result.is_err():
        raise RuntimeError(f"Query failed: {query_result.error}")  # type: ignore[union-attr]

    return query_result.unwrap().answer


def main() -> None:
    """Main entry point for the Wikipedia RAG example."""
    parser = argparse.ArgumentParser(
        description="Wikipedia RAG example — answers questions about RAG using mock mode."
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        default=True,
        help="Run in mock mode (default: True, no API keys needed)",
    )
    parser.add_argument(
        "--no-mock",
        action="store_false",
        dest="mock",
        help="Run in production mode (requires OPENAI_API_KEY)",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Custom question to answer (default: runs all benchmark questions)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Wikipedia RAG Example")
    print(f"Mode: {'mock (no API keys)' if args.mock else 'production'}")
    print("=" * 60)

    questions = [args.query] if args.query else BENCHMARK_QUESTIONS

    for question in questions:
        print(f"\nQ: {question}")
        answer = run_wikipedia_rag(question, mock=args.mock)
        print(f"A: {answer}")
        print("-" * 40)

    print("\nWikipedia RAG example completed successfully.")
    sys.exit(0)


if __name__ == "__main__":
    main()
