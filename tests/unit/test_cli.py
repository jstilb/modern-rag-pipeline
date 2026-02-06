"""Tests for CLI interface."""

import json
from io import StringIO
from unittest.mock import patch

import pytest

from src.rag.cli import main, run_demo, run_query, SAMPLE_DOCUMENTS


class TestCLI:
    def test_demo_runs_successfully(self, capsys: pytest.CaptureFixture[str]) -> None:
        run_demo("What is RAG?")
        captured = capsys.readouterr()
        assert "Modern RAG Pipeline - Demo Mode" in captured.out
        assert "Ingesting" in captured.out
        assert "Results:" in captured.out
        assert "JSON output:" in captured.out

    def test_demo_custom_query(self, capsys: pytest.CaptureFixture[str]) -> None:
        run_demo("How does hybrid search work?")
        captured = capsys.readouterr()
        assert "hybrid search" in captured.out.lower() or "Results:" in captured.out

    def test_query_runs_successfully(self, capsys: pytest.CaptureFixture[str]) -> None:
        run_query("What is a vector database?", top_k=3)
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert "answer" in data
        assert data["query"] == "What is a vector database?"
        assert data["source_count"] > 0

    def test_sample_documents_valid(self) -> None:
        assert len(SAMPLE_DOCUMENTS) >= 3
        for doc in SAMPLE_DOCUMENTS:
            assert len(doc.content) > 0
            assert len(doc.source) > 0

    def test_main_no_args(self) -> None:
        with pytest.raises(SystemExit):
            with patch("sys.argv", ["rag-pipeline"]):
                main()

    def test_main_demo(self, capsys: pytest.CaptureFixture[str]) -> None:
        with patch("sys.argv", ["rag-pipeline", "demo"]):
            main()
        captured = capsys.readouterr()
        assert "Modern RAG Pipeline" in captured.out

    def test_main_query(self, capsys: pytest.CaptureFixture[str]) -> None:
        with patch("sys.argv", ["rag-pipeline", "query", "test question"]):
            main()
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert "answer" in data
