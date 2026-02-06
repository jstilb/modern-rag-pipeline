.PHONY: install dev test lint type-check format demo serve clean docker-build docker-run

# Installation
install:
	pip install -e .

dev:
	pip install -e ".[dev]"

# Testing
test:
	python -m pytest tests/ --cov=src --cov-report=term-missing -v

test-unit:
	python -m pytest tests/unit/ -v

test-integration:
	python -m pytest tests/integration/ -v

# Code quality
lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

type-check:
	mypy src/ --ignore-missing-imports

format:
	ruff check --fix src/ tests/
	ruff format src/ tests/

# Running
demo:
	python -m src.rag.cli demo

serve:
	python -m src.rag.cli serve --port 8000

# Docker
docker-build:
	docker build -t modern-rag-pipeline .

docker-run:
	docker run -p 8000:8000 --env-file .env modern-rag-pipeline

# Cleanup
clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	rm -rf dist build *.egg-info
	rm -rf .chroma_data
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# All checks
check: lint type-check test
