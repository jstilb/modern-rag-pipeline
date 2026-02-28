# Production Concerns for Modern RAG Pipeline

This document covers failure modes, degradation scenarios, and mitigation strategies for
running the modern-rag-pipeline in production environments.

---

## 1. Latency Failure Modes

### 1.1 Embedding Generation Latency

**Problem:** Embedding calls to OpenAI/Cohere can add 100–500 ms per query depending on
batch size and model. Under high load, this creates a latency multiplier that cascades
through the pipeline.

**Symptoms:**
- P95 query latency exceeds 2 seconds
- Timeout errors from embedding API under concurrent load

**Mitigations:**
- **Batch embedding:** Accumulate queries and embed in batches (size 8–32) when throughput
  exceeds 20 QPS.
- **Embedding cache:** Cache query embeddings with a TTL of 60 seconds. For repeated
  queries (common in chatbot UX), cache hit rates of 30–50% are typical.
- **Async pipeline:** Use `asyncio` throughout; avoid blocking the event loop on embedding
  calls. The FastAPI app in this project is already async.
- **Timeout circuit breaker:** Set hard timeouts (2 s) on embedding calls; return a
  degraded response (BM25-only retrieval) rather than failing the entire request.

### 1.2 Vector Search Latency

**Problem:** ChromaDB / Qdrant nearest-neighbor search latency grows with collection size.
At 1M vectors, naive HNSW search can take 50–200 ms.

**Mitigations:**
- Set `ef_search` (HNSW parameter) to 50 for p50 latency, 128 for recall-optimized mode.
- Pre-filter by metadata (date, source, tenant) before ANN search to reduce candidate set.
- Monitor `collection.count()` and shard at 500K vectors per collection.

### 1.3 LLM Generation Latency

**Problem:** GPT-4 Turbo generates ~30 tokens/sec. For 500-token answers, generation alone
takes ~16 seconds, which is unacceptable for interactive use.

**Mitigations:**
- **Stream responses:** Use `stream=True` in OpenAI calls; forward SSE to clients
  immediately. The FastAPI app supports this via `StreamingResponse`.
- **Token budget enforcement:** Cap `max_tokens` at 300 for interactive mode; 1000 for
  document summarization mode.
- **Model tiering:** Use GPT-3.5-turbo for low-stakes queries; GPT-4 only for complex
  multi-hop reasoning.

---

## 2. Embedding API Outage Handling

### 2.1 Total Outage Scenario

**Problem:** OpenAI embedding API (or Cohere) becomes unavailable. All new ingestion and
semantic search fails.

**Detection:**
```python
# EmbeddingProvider already wraps errors in Result type
result = embedder.embed(texts)
if result.is_err():
    # Log, alert, switch to fallback
    metrics.increment("embedding_api_errors")
```

**Mitigations:**
- **BM25 fallback:** When embedding is unavailable, automatically route queries to
  keyword-only retrieval. Quality degrades (~15% NDCG reduction) but the pipeline stays
  available.
- **Multi-provider failover:** Configure primary (OpenAI) and secondary (Cohere) embedding
  providers. Failover triggers after 3 consecutive errors or a 5-second timeout.
- **Pre-computed embedding cache:** Persist embedding vectors to disk (already supported via
  `chroma_persist_dir`). Cached vectors survive API outages for existing documents.
- **Circuit breaker pattern:** After 5 failures in 60 seconds, open the circuit; return
  BM25-only results with a `degraded_mode: true` flag in the API response.

### 2.2 Partial Degradation (Rate Limiting)

**Problem:** Embedding API returns 429 (rate limit) under burst load.

**Mitigations:**
- Implement exponential backoff: `wait = min(2^attempt * 0.5, 30)` seconds.
- Use a token bucket to pre-limit request rate (100 RPM for OpenAI `text-embedding-3-small`).
- Queue ingestion jobs with a priority queue; deprioritize bulk ingestion vs. real-time queries.

---

## 3. Vector Store Timeouts

### 3.1 ChromaDB Timeout Scenarios

**Problem:** ChromaDB running locally (SQLite backend) can lock during concurrent writes.
Under high ingestion + query load, read queries can wait indefinitely for write locks.

**Symptoms:**
- `sqlite3.OperationalError: database is locked`
- Query latency spikes >5 seconds during bulk ingestion

**Mitigations:**
- Use ChromaDB's HTTP client mode (`chromadb.HttpClient`) for production — separate the
  server process from the application to avoid GIL contention.
- Set `SQLITE_BUSY_TIMEOUT=5000` (5 seconds) to prevent indefinite lock waits.
- Separate read and write replicas: ingest to a staging collection, merge to production
  during low-traffic windows.

### 3.2 Qdrant Timeout Scenarios

**Problem:** Qdrant cloud instances can throttle under sustained load or during index
rebuilding after bulk uploads.

**Symptoms:**
- HTTP 429 from Qdrant Cloud after 10K points/second insert rate
- `timeout` errors during HNSW index construction

**Mitigations:**
- Set `timeout=30` on all Qdrant client calls (already configurable in `QdrantVectorStore`).
- Use Qdrant's batch upload API (`upload_points`) with `batch_size=100` rather than
  single-point inserts.
- Monitor Qdrant collection status; pause ingestion if `status == "yellow"` (index
  optimization in progress).

### 3.3 General Vector Store Resilience

```python
# Example: timeout wrapper for vector store operations
import asyncio
from typing import TypeVar, Callable, Awaitable

T = TypeVar("T")

async def with_timeout(
    coro: Awaitable[T],
    timeout_seconds: float = 5.0,
    fallback: T | None = None,
) -> T | None:
    try:
        return await asyncio.wait_for(coro, timeout=timeout_seconds)
    except asyncio.TimeoutError:
        metrics.increment("vector_store_timeouts")
        return fallback
```

---

## 4. Chunk Retrieval Degradation Scenarios

### 4.1 Low-Recall Retrieval

**Problem:** The hybrid retriever returns fewer than `top_k` chunks, or all retrieved chunks
have low relevance scores. This produces hallucinated or vague answers.

**Root Causes:**
- Query is out-of-domain (vocabulary mismatch with indexed documents)
- Chunking strategy produced chunks too large or too small for the query
- Collection is too small (fewer than 50 documents)

**Detection:**
```python
# Check retrieval quality in the pipeline
result = retriever.retrieve(query, top_k=5)
if result.is_ok():
    chunks = result.value
    if len(chunks) < 3 or all(c.score < 0.3 for c in chunks):
        # Log low-recall event
        metrics.increment("low_recall_retrievals")
        # Optionally: widen search
        result = retriever.retrieve(query, top_k=15)
```

**Mitigations:**
- **Dynamic top-k:** If average chunk score < 0.35, double `top_k` and re-rank.
- **Query expansion:** For sparse queries (<5 words), use the LLM to generate 2 alternative
  phrasings and retrieve against all three.
- **Minimum confidence threshold:** If no chunk scores above 0.2, respond with "I don't have
  information about this topic" rather than generating a hallucinated answer.

### 4.2 Chunking Strategy Mismatch

**Problem:** Fixed-size chunking breaks sentences mid-thought, causing retrieved chunks to
lack context. Semantic chunking may produce overly large chunks that dilute relevance.

**Symptoms:**
- Answer quality drops for multi-sentence factual questions
- Context window fills up with irrelevant padding from large chunks

**Mitigations:**
- Use the chunking comparison notebook (`notebooks/chunking_comparison.ipynb`) to
  empirically select the best strategy for your document type.
- Default to `recursive` chunking (chunk_size=512, overlap=50) as the best general-purpose
  strategy (see benchmark results in README).
- For long-form documents (books, legal docs): use `semantic` chunking.
- For structured data (tables, lists): use `fixed` chunking with small size (128 tokens).

### 4.3 RRF Fusion Failure

**Problem:** Reciprocal Rank Fusion can fail to surface the best results when one retrieval
modality (semantic or BM25) has very high scores and the other has very low scores.

**Mitigations:**
- Monitor per-modality recall separately. If BM25 consistently returns 0 results for a
  query type, disable BM25 for that category.
- Tune the RRF `k` parameter (default 60): lower values weight high-rank results more
  heavily; higher values smooth across ranks.

### 4.4 Context Window Overflow

**Problem:** When `top_k` is large (e.g., 20) and chunks are big (512 tokens each), the
context passed to the LLM exceeds the model's context window (128K for GPT-4, 200K for
Claude).

**Mitigations:**
- Enforce `max_context_tokens = 8000` (conservative for GPT-4 Turbo) in the pipeline config.
- Apply a map-reduce pattern for long contexts: summarize each chunk independently, then
  generate the final answer from summaries.
- Use the reranker (cross-encoder or Cohere) to select the top 3–5 most relevant chunks
  before passing to the LLM, regardless of `top_k` retrieved.

---

## 5. Monitoring Checklist

| Metric | Alert Threshold | Action |
|--------|----------------|--------|
| Embedding API error rate | >5% over 5 min | Switch to BM25 fallback |
| Query P95 latency | >3 seconds | Scale up embedding cache |
| Vector store timeout rate | >1% over 5 min | Check DB lock contention |
| Low-recall retrieval rate | >10% over 1 hour | Trigger re-indexing |
| LLM error rate | >2% over 5 min | Circuit break + return error |
| Chunk count per collection | >500K | Begin sharding |

---

## 6. Deployment Checklist

- [ ] Set `OPENAI_API_KEY` / `COHERE_API_KEY` in environment (never in code)
- [ ] Configure `chroma_persist_dir` or Qdrant URL for persistent storage
- [ ] Set `MOCK_MODE=false` for production
- [ ] Enable request-level logging with correlation IDs
- [ ] Configure health check endpoint (`/health`) in load balancer
- [ ] Set up Prometheus metrics scraping at `/metrics`
- [ ] Pre-warm embedding cache with common queries before launch
