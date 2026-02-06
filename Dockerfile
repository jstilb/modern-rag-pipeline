FROM python:3.11-slim AS base

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml README.md LICENSE ./
COPY src/ src/

# Install Python dependencies
RUN pip install --no-cache-dir -e .

# Default to mock mode (no API keys needed)
ENV RAG_MODE=mock
ENV RAG_API_HOST=0.0.0.0
ENV RAG_API_PORT=8000

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import httpx; r = httpx.get('http://localhost:8000/health'); r.raise_for_status()"

CMD ["python", "-m", "src.rag.cli", "serve", "--host", "0.0.0.0", "--port", "8000"]
