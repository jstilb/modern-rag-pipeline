"""RAG evaluation metrics for retrieval and generation quality.

Implements standard IR metrics (MRR, NDCG, Precision, Recall) and
generation quality metrics (faithfulness, answer relevance).
No external eval framework dependencies - pure Python implementation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class RetrievalMetrics:
    """Metrics for retrieval quality."""

    precision_at_k: float
    recall_at_k: float
    mrr: float  # Mean Reciprocal Rank
    ndcg_at_k: float  # Normalized Discounted Cumulative Gain
    hit_rate: float
    k: int


@dataclass(frozen=True, slots=True)
class GenerationMetrics:
    """Metrics for generation quality."""

    faithfulness: float  # How grounded is the answer in context
    answer_relevance: float  # How relevant is the answer to the query
    context_utilization: float  # How much of the context was used


@dataclass(frozen=True, slots=True)
class RAGMetrics:
    """Combined retrieval and generation metrics."""

    retrieval: RetrievalMetrics
    generation: GenerationMetrics
    overall_score: float  # Weighted combination


def evaluate_retrieval(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int = 5,
) -> RetrievalMetrics:
    """Evaluate retrieval quality against known relevant documents.

    Args:
        retrieved_ids: Ordered list of retrieved document IDs
        relevant_ids: Set of known relevant document IDs
        k: Number of results to evaluate at
    """
    top_k = retrieved_ids[:k]

    # Precision@k
    relevant_in_top_k = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    precision = relevant_in_top_k / k if k > 0 else 0.0

    # Recall@k
    recall = (
        relevant_in_top_k / len(relevant_ids) if relevant_ids else 0.0
    )

    # MRR (Mean Reciprocal Rank)
    mrr = 0.0
    for i, doc_id in enumerate(top_k):
        if doc_id in relevant_ids:
            mrr = 1.0 / (i + 1)
            break

    # NDCG@k
    ndcg = _compute_ndcg(top_k, relevant_ids, k)

    # Hit Rate
    hit_rate = 1.0 if any(doc_id in relevant_ids for doc_id in top_k) else 0.0

    return RetrievalMetrics(
        precision_at_k=precision,
        recall_at_k=recall,
        mrr=mrr,
        ndcg_at_k=ndcg,
        hit_rate=hit_rate,
        k=k,
    )


def _compute_ndcg(
    retrieved_ids: list[str], relevant_ids: set[str], k: int
) -> float:
    """Compute Normalized Discounted Cumulative Gain at k."""
    # DCG
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_ids[:k]):
        rel = 1.0 if doc_id in relevant_ids else 0.0
        dcg += rel / math.log2(i + 2)  # +2 because log2(1) = 0

    # Ideal DCG
    ideal_rels = sorted(
        [1.0 if doc_id in relevant_ids else 0.0 for doc_id in retrieved_ids[:k]],
        reverse=True,
    )
    # Add any remaining relevant docs not in retrieved
    remaining = len(relevant_ids) - sum(1 for d in retrieved_ids[:k] if d in relevant_ids)
    ideal_rels.extend([1.0] * remaining)
    ideal_rels = ideal_rels[:k]

    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal_rels))

    return dcg / idcg if idcg > 0 else 0.0


def evaluate_generation(
    answer: str,
    query: str,
    context_chunks: list[str],
) -> GenerationMetrics:
    """Evaluate generation quality using heuristic metrics.

    For production use, these would be replaced with LLM-as-judge
    evaluation. These heuristics provide a reasonable baseline.

    Args:
        answer: Generated answer text
        query: Original query
        context_chunks: Retrieved context used for generation
    """
    # Faithfulness: word overlap between answer and context
    answer_words = set(answer.lower().split())
    context_text = " ".join(context_chunks)
    context_words = set(context_text.lower().split())

    if answer_words:
        common_words = answer_words & context_words
        # Remove stop words from calculation
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "shall", "can",
            "to", "of", "in", "for", "on", "with", "at", "by", "from",
            "as", "into", "through", "during", "before", "after", "and",
            "but", "or", "nor", "not", "so", "yet", "both", "either",
            "neither", "each", "every", "all", "any", "few", "more",
            "most", "other", "some", "such", "no", "only", "own", "same",
            "than", "too", "very", "just", "because", "this", "that",
            "these", "those", "it", "its",
        }
        meaningful_answer = answer_words - stop_words
        meaningful_common = common_words - stop_words
        faithfulness = (
            len(meaningful_common) / len(meaningful_answer)
            if meaningful_answer
            else 0.0
        )
    else:
        faithfulness = 0.0

    # Answer relevance: word overlap between answer and query
    query_words = set(query.lower().split()) - {
        "what", "how", "why", "when", "where", "who", "which", "is", "are",
        "the", "a", "an", "do", "does", "can", "could", "would", "should",
    }
    if query_words:
        query_overlap = len(answer_words & query_words) / len(query_words)
        answer_relevance = min(1.0, query_overlap)
    else:
        answer_relevance = 0.0

    # Context utilization: how many chunks contributed words to the answer
    if context_chunks:
        chunks_used = sum(
            1
            for chunk in context_chunks
            if len(set(chunk.lower().split()) & answer_words) > 3
        )
        context_utilization = chunks_used / len(context_chunks)
    else:
        context_utilization = 0.0

    return GenerationMetrics(
        faithfulness=min(1.0, faithfulness),
        answer_relevance=min(1.0, answer_relevance),
        context_utilization=min(1.0, context_utilization),
    )
