"""Tests for evaluation metrics."""

import pytest

from src.evaluation.metrics import evaluate_generation, evaluate_retrieval


class TestEvaluateRetrieval:
    def test_perfect_retrieval(self) -> None:
        retrieved = ["a", "b", "c"]
        relevant = {"a", "b", "c"}
        metrics = evaluate_retrieval(retrieved, relevant, k=3)
        assert metrics.precision_at_k == 1.0
        assert metrics.recall_at_k == 1.0
        assert metrics.mrr == 1.0
        assert metrics.hit_rate == 1.0

    def test_no_relevant_retrieved(self) -> None:
        retrieved = ["x", "y", "z"]
        relevant = {"a", "b", "c"}
        metrics = evaluate_retrieval(retrieved, relevant, k=3)
        assert metrics.precision_at_k == 0.0
        assert metrics.recall_at_k == 0.0
        assert metrics.mrr == 0.0
        assert metrics.hit_rate == 0.0

    def test_partial_retrieval(self) -> None:
        retrieved = ["a", "x", "b"]
        relevant = {"a", "b", "c"}
        metrics = evaluate_retrieval(retrieved, relevant, k=3)
        assert metrics.precision_at_k == pytest.approx(2 / 3)
        assert metrics.recall_at_k == pytest.approx(2 / 3)
        assert metrics.mrr == 1.0  # First relevant at position 1

    def test_mrr_late_hit(self) -> None:
        retrieved = ["x", "y", "a"]
        relevant = {"a"}
        metrics = evaluate_retrieval(retrieved, relevant, k=3)
        assert metrics.mrr == pytest.approx(1 / 3)

    def test_empty_relevant(self) -> None:
        retrieved = ["a", "b"]
        relevant: set[str] = set()
        metrics = evaluate_retrieval(retrieved, relevant, k=2)
        assert metrics.recall_at_k == 0.0

    def test_k_parameter(self) -> None:
        retrieved = ["a", "b", "c", "d"]
        relevant = {"c", "d"}
        metrics_2 = evaluate_retrieval(retrieved, relevant, k=2)
        metrics_4 = evaluate_retrieval(retrieved, relevant, k=4)
        assert metrics_2.precision_at_k == 0.0
        assert metrics_4.precision_at_k == 0.5

    def test_ndcg_perfect(self) -> None:
        retrieved = ["a", "b"]
        relevant = {"a", "b"}
        metrics = evaluate_retrieval(retrieved, relevant, k=2)
        assert metrics.ndcg_at_k == pytest.approx(1.0)


class TestEvaluateGeneration:
    def test_high_faithfulness(self) -> None:
        answer = "Machine learning systems learn from data patterns"
        context = ["Machine learning is about systems that learn patterns from data"]
        metrics = evaluate_generation(answer, "What is ML?", context)
        assert metrics.faithfulness > 0.3

    def test_low_faithfulness(self) -> None:
        answer = "Quantum computing uses qubits for computation"
        context = ["Machine learning is about pattern recognition"]
        metrics = evaluate_generation(answer, "What is ML?", context)
        assert metrics.faithfulness < 0.5

    def test_answer_relevance(self) -> None:
        answer = "Machine learning algorithms learn from data"
        metrics = evaluate_generation(
            answer, "machine learning algorithms", ["context"]
        )
        assert metrics.answer_relevance > 0.0

    def test_empty_context(self) -> None:
        metrics = evaluate_generation("answer", "query", [])
        assert metrics.context_utilization == 0.0

    def test_empty_answer(self) -> None:
        metrics = evaluate_generation("", "query", ["context"])
        assert metrics.faithfulness == 0.0
