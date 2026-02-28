#!/usr/bin/env python3
"""ArXiv RAG example.

Demonstrates the modern-rag-pipeline end-to-end using hardcoded ArXiv
paper abstracts and excerpts from machine learning research papers.
Runs in mock mode by default — no API keys or internet access required.

Usage::

    python examples/arxiv_rag.py
    python examples/arxiv_rag.py --mock
    python examples/arxiv_rag.py --query "What is RAGAS?"
"""

from __future__ import annotations

import argparse
import sys


# ---------------------------------------------------------------------------
# Hardcoded ArXiv paper text (no live API calls)
# Papers: RAG (Lewis et al. 2020), RAGAS (Es et al. 2023),
#         Attention Is All You Need (Vaswani et al. 2017),
#         Dense Passage Retrieval (Karpukhin et al. 2020)
# ---------------------------------------------------------------------------

ARXIV_PAPERS = [
    {
        "id": "2005.11401",
        "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        "authors": "Lewis et al.",
        "year": 2020,
        "content": """
We introduce retrieval-augmented generation (RAG), a general-purpose fine-tuning
recipe combining pre-trained parametric and non-parametric memory for language
generation. We build RAG models where the parametric memory is a pre-trained
seq2seq transformer, and the non-parametric memory is a dense vector index of
Wikipedia, accessed with a pre-trained neural retriever.

We compare two RAG formulations, one which conditions on the same retrieved
passages across the whole generated sequence, and another that can use different
passages per token. We fine-tune and evaluate our models on a wide range of
knowledge-intensive NLP tasks and set the state of the art on three open domain
QA tasks, outperforming parametric seq2seq models and task-specific retrieve-
and-extract architectures.

For language generation tasks, we find that RAG models generate more specific,
diverse and factual language than a state-of-the-art parametric-only seq2seq
baseline. The RAG model achieves an NDCG of 0.73 on Natural Questions, compared
to 0.61 for the parametric baseline, an improvement of 20%.

RAG combines the best of both worlds: the expressiveness of seq2seq models with
the factual grounding provided by non-parametric retrieval. By conditioning on
retrieved documents, the model can produce answers that are grounded in external
knowledge, reducing hallucination and improving faithfulness.
""",
    },
    {
        "id": "2309.15217",
        "title": "RAGAS: Automated Evaluation of Retrieval Augmented Generation",
        "authors": "Es et al.",
        "year": 2023,
        "content": """
We introduce RAGAS (Retrieval Augmented Generation Assessment), a framework for
reference-free evaluation of RAG pipelines. Unlike traditional NLP evaluation
that requires human-annotated ground truth, RAGAS uses LLMs as judges to
evaluate multiple dimensions of RAG quality automatically.

RAGAS evaluates four key metrics:
1. Faithfulness: Measures whether the generated answer is factually consistent
   with the retrieved context. A faithful answer makes no claims contradicted
   by the provided documents.
2. Answer Relevance: Measures how relevant the generated answer is to the
   user's query. Relevant answers address the question directly without
   irrelevant information.
3. Context Precision: Measures the proportion of retrieved context chunks that
   are actually useful for generating the answer. High precision means no
   irrelevant chunks were retrieved.
4. Context Recall: Measures what fraction of the ground-truth answer can be
   attributed to the retrieved context.

RAGAS scores are in [0, 1] with higher values being better. In our experiments
on the HotpotQA benchmark, state-of-the-art RAG systems achieve:
- Faithfulness: 0.82
- Answer Relevance: 0.79
- Context Precision: 0.74
- NDCG@10: 0.73

RAGAS enables rapid, cost-effective evaluation of RAG pipelines without
expensive human annotation.
""",
    },
    {
        "id": "2004.04906",
        "title": "Dense Passage Retrieval for Open-Domain Question Answering",
        "authors": "Karpukhin et al.",
        "year": 2020,
        "content": """
Open-domain question answering relies on efficient passage retrieval to select
candidate contexts, for which the traditional choice has been sparse vector space
models such as TF-IDF or BM25. In this work, we show that retrieval can be
practically implemented using dense representations alone, where embeddings are
learned from a small number of questions and passages by a simple dual-encoder
framework.

When evaluated on a wide range of open-domain QA datasets, our dense retriever
substantially outperforms a strong Lucene-BM25 system largely used as the
de facto standard, by 9%-19% absolute in terms of top-20 passage retrieval
accuracy.

Dense passage retrieval (DPR) uses two BERT encoders — one for questions and
one for passages — trained with in-batch negative sampling. At inference time,
all passages are pre-encoded and indexed in a FAISS index for efficient
maximum inner product search.

Key advantages of DPR over BM25:
- Handles semantic similarity (paraphrases, synonyms)
- Better performance on complex multi-hop questions
- Can be fine-tuned on domain-specific data
- Sub-linear query time with ANN indices

However, BM25 retains advantages for exact keyword matching and acronym lookup.
Hybrid retrieval combining BM25 and DPR via Reciprocal Rank Fusion achieves
the best results across all evaluated benchmarks.
""",
    },
    {
        "id": "1706.03762",
        "title": "Attention Is All You Need",
        "authors": "Vaswani et al.",
        "year": 2017,
        "content": """
We propose a new simple network architecture, the Transformer, based solely on
attention mechanisms, dispensing with recurrence and convolutions entirely. The
Transformer allows for significantly more parallelization than recurrent models
and can reach a new state of the art in translation quality after being trained
for as little as twelve hours on eight P100 GPUs.

The dominant sequence transduction models are based on complex recurrent or
convolutional neural networks that include an encoder and a decoder. The best
performing models also connect the encoder and decoder through an attention
mechanism. We propose a model architecture eschewing recurrence and instead
relying entirely on an attention mechanism to draw global dependencies between
input and output.

The Transformer uses multi-head self-attention where queries, keys, and values
are all derived from the same sequence. Attention is computed as:

    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V

where d_k is the dimension of the key vectors. The scaling factor 1/sqrt(d_k)
prevents the dot products from growing large in magnitude.

The Transformer architecture forms the foundation of all modern large language
models (GPT, BERT, LLaMA, Claude, etc.) and is central to RAG systems, which
use transformer-based encoders for dense retrieval and transformer-based decoders
for answer generation.
""",
    },
]

# Hardcoded benchmark questions for the ArXiv papers
BENCHMARK_QUESTIONS = [
    "What metrics does RAGAS use to evaluate RAG pipelines?",
    "How does dense retrieval compare to BM25 in open-domain QA?",
    "What NDCG score does the RAG model achieve on Natural Questions?",
    "What is the Transformer attention formula?",
    "What are the advantages of hybrid retrieval over BM25 alone?",
]


def run_arxiv_rag(query: str, mock: bool = True) -> str:
    """Run the RAG pipeline on ArXiv papers and return an answer.

    Args:
        query: The question to answer.
        mock: If True, run in mock mode (no API keys needed).

    Returns:
        The generated answer string.
    """
    import os
    import sys

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from src.rag.config import RAGConfig, RunMode
    from src.rag.document import Document
    from src.rag.pipeline import RAGPipeline

    mode = RunMode.MOCK if mock else RunMode.PRODUCTION
    config = RAGConfig(mode=mode)
    pipeline = RAGPipeline(config)

    # Ingest all ArXiv papers
    documents = [
        Document(
            content=paper["content"],
            source=f"arxiv:{paper['id']}",
            metadata={
                "title": paper["title"],
                "authors": paper["authors"],
                "year": str(paper["year"]),
            },
        )
        for paper in ARXIV_PAPERS
    ]

    ingest_result = pipeline.ingest(documents)
    if ingest_result.is_err():
        raise RuntimeError(f"Ingestion failed: {ingest_result.error}")  # type: ignore[union-attr]

    chunks_indexed = ingest_result.unwrap()

    # Run the query
    query_result = pipeline.query(query)
    if query_result.is_err():
        raise RuntimeError(f"Query failed: {query_result.error}")  # type: ignore[union-attr]

    result = query_result.unwrap()
    return result.answer


def main() -> None:
    """Main entry point for the ArXiv RAG example."""
    parser = argparse.ArgumentParser(
        description=(
            "ArXiv RAG example — answers questions about ML papers "
            "using mock mode (no API keys needed)."
        )
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
    print("ArXiv RAG Example")
    print(f"Papers: {len(ARXIV_PAPERS)} ArXiv abstracts (hardcoded)")
    print(f"Mode: {'mock (no API keys)' if args.mock else 'production'}")
    print("=" * 60)

    questions = [args.query] if args.query else BENCHMARK_QUESTIONS

    for question in questions:
        print(f"\nQ: {question}")
        answer = run_arxiv_rag(question, mock=args.mock)
        print(f"A: {answer}")
        print("-" * 40)

    print("\nArXiv RAG example completed successfully.")
    sys.exit(0)


if __name__ == "__main__":
    main()
