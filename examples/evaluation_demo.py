"""Evaluation metrics demo.

Demonstrates how to evaluate RAG retrieval and generation quality
using built-in metrics. No API keys required.

Usage:
    python examples/evaluation_demo.py
"""

from src.evaluation.metrics import evaluate_generation, evaluate_retrieval


def main() -> None:
    print("=" * 60)
    print("RAG Evaluation Metrics Demo")
    print("=" * 60)

    # --- Retrieval Evaluation ---
    print("\n--- Retrieval Metrics ---\n")

    # Simulate retrieval results
    retrieved_ids = ["doc_1", "doc_3", "doc_5", "doc_2", "doc_7"]
    relevant_ids = {"doc_1", "doc_2", "doc_3"}

    metrics = evaluate_retrieval(retrieved_ids, relevant_ids, k=5)

    print(f"Retrieved: {retrieved_ids}")
    print(f"Relevant:  {relevant_ids}")
    print(f"k = {metrics.k}")
    print()
    print(f"  Precision@k: {metrics.precision_at_k:.3f}")
    print(f"  Recall@k:    {metrics.recall_at_k:.3f}")
    print(f"  MRR:         {metrics.mrr:.3f}")
    print(f"  NDCG@k:      {metrics.ndcg_at_k:.3f}")
    print(f"  Hit Rate:    {metrics.hit_rate:.3f}")

    # Perfect retrieval
    print("\n--- Perfect Retrieval ---")
    perfect = evaluate_retrieval(["a", "b", "c"], {"a", "b", "c"}, k=3)
    print(f"  Precision@3: {perfect.precision_at_k:.3f} (expecting 1.0)")
    print(f"  Recall@3:    {perfect.recall_at_k:.3f} (expecting 1.0)")

    # --- Generation Evaluation ---
    print("\n\n--- Generation Metrics ---\n")

    query = "What is machine learning?"
    answer = "Machine learning is a field of AI that enables systems to learn from data patterns."
    context = [
        "Machine learning is a subset of artificial intelligence focused on data-driven learning.",
        "ML systems improve their performance on tasks through experience and data patterns.",
    ]

    gen_metrics = evaluate_generation(answer, query, context)

    print(f"Query:   {query}")
    print(f"Answer:  {answer}")
    print(f"Context: {len(context)} chunks")
    print()
    print(f"  Faithfulness:        {gen_metrics.faithfulness:.3f}")
    print(f"  Answer Relevance:    {gen_metrics.answer_relevance:.3f}")
    print(f"  Context Utilization: {gen_metrics.context_utilization:.3f}")

    # Unfaithful answer (hallucination)
    print("\n--- Unfaithful Answer (Hallucination) ---")
    hallucinated = "Quantum computing will replace all classical computers by 2030."
    hal_metrics = evaluate_generation(hallucinated, query, context)
    print(f"  Faithfulness: {hal_metrics.faithfulness:.3f} (low = hallucinated)")

    print("\n" + "=" * 60)
    print("Evaluation complete.")


if __name__ == "__main__":
    main()
