"""Integration tests for the FastAPI REST API."""

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.rag.config import RAGConfig, RunMode


@pytest.fixture
def client() -> TestClient:
    config = RAGConfig(mode=RunMode.MOCK, chunk_size=50)
    app = create_app(config)
    return TestClient(app)


class TestHealthEndpoint:
    def test_health(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["mode"] == "mock"
        assert "version" in data


class TestIngestEndpoint:
    def test_ingest_documents(self, client: TestClient) -> None:
        response = client.post(
            "/ingest",
            json={
                "documents": [
                    {
                        "content": "Python is great for data science and AI.",
                        "source": "test.md",
                    },
                    {
                        "content": "TypeScript adds types to JavaScript for safety.",
                        "source": "test2.md",
                    },
                ]
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["total_documents"] == 2
        assert data["chunks_indexed"] > 0

    def test_ingest_empty(self, client: TestClient) -> None:
        response = client.post("/ingest", json={"documents": []})
        assert response.status_code == 200
        assert response.json()["chunks_indexed"] == 0

    def test_ingest_invalid_document(self, client: TestClient) -> None:
        response = client.post(
            "/ingest",
            json={
                "documents": [
                    {"content": "", "source": "test.md"},
                ]
            },
        )
        assert response.status_code == 422  # Validation error


class TestQueryEndpoint:
    def test_query(self, client: TestClient) -> None:
        # First ingest
        client.post(
            "/ingest",
            json={
                "documents": [
                    {
                        "content": "Machine learning uses algorithms to find patterns in data.",
                        "source": "ml.md",
                    }
                ]
            },
        )

        # Then query
        response = client.post(
            "/query",
            json={"query": "What is machine learning?"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert data["query"] == "What is machine learning?"
        assert "sources" in data
        assert data["latency_ms"] > 0

    def test_query_with_top_k(self, client: TestClient) -> None:
        client.post(
            "/ingest",
            json={
                "documents": [
                    {"content": "Doc one about testing.", "source": "a.md"},
                    {"content": "Doc two about testing.", "source": "b.md"},
                    {"content": "Doc three about testing.", "source": "c.md"},
                ]
            },
        )

        response = client.post(
            "/query",
            json={"query": "testing", "top_k": 2},
        )
        assert response.status_code == 200
        assert len(response.json()["sources"]) <= 2

    def test_query_empty_string(self, client: TestClient) -> None:
        response = client.post("/query", json={"query": ""})
        assert response.status_code == 422
