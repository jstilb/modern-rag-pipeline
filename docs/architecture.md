# Architecture

## System Overview

```
                         +------------------+
                         |   FastAPI REST   |
                         |  /health /ingest |
                         |     /query       |
                         +--------+---------+
                                  |
                         +--------+---------+
                         |   RAG Pipeline   |
                         |   (Orchestrator) |
                         +--+-----+-----+--+
                            |     |     |
              +-------------+  +--+--+  +--------------+
              |                |     |                  |
     +--------+------+  +-----+--+  +-------+--------+ |
     |   Chunking    |  | Retrieval|  |   Generation  | |
     |   Strategies  |  | Engine   |  |    (LLM)      | |
     +-------+-------+  +----+----+  +-------+-------+ |
             |               |                |         |
     +-------+-------+  +---+---+    +-------+-------+ |
     | Fixed/Recursive|  |Hybrid |    | OpenAI / Mock | |
     | Semantic/Window|  |Search |    +---------------+ |
     +----------------+  +--+---+                       |
                            |                           |
                   +--------+--------+                  |
                   |                 |         +--------+-------+
             +-----+----+    +------+----+    |   Evaluation   |
             | Semantic  |    | Keyword   |    |    Metrics     |
             | (ChromaDB)|    | (BM25)    |    +----------------+
             +-----+-----+   +-----------+
                   |
             +-----+------+
             |  Embedding  |
             |  Provider   |
             | (OpenAI/Mock)|
             +-------------+
```

## Data Flow

### Ingestion Pipeline

```
Document --> Chunker --> Chunks --> Embedding Provider --> Vector Store
                                                     --> BM25 Index
```

1. **Document** arrives via API or CLI
2. **Chunker** splits into sized pieces using configured strategy
3. **Embedding Provider** generates vectors for each chunk
4. **Vector Store** (ChromaDB) indexes embeddings for similarity search
5. **BM25 Index** indexes tokens for keyword search

### Query Pipeline

```
Query --> Retrieval Engine --> Top-K Chunks --> LLM Provider --> Answer
             |                                     |
             +-- Semantic Search (ChromaDB)        +-- Context Assembly
             +-- Keyword Search (BM25)             +-- Prompt Construction
             +-- RRF Fusion                        +-- Generation
```

1. **Query** arrives via API or CLI
2. **Retrieval Engine** runs configured strategy:
   - Semantic: embed query, cosine similarity search
   - Keyword: BM25 token matching
   - Hybrid: Both + Reciprocal Rank Fusion
3. **Top-K Chunks** are assembled as context
4. **LLM Provider** generates answer from context + query

## Component Details

### Chunking Strategies

| Strategy | Best For | Trade-off |
|----------|----------|-----------|
| Fixed Size | Uniform documents | Simple but may break mid-sentence |
| Recursive | Prose/technical docs | Preserves structure, variable sizes |
| Semantic | Topic-diverse docs | Best boundaries, slower |
| Sliding Window | Context-heavy docs | Maximum overlap, more chunks |

### Retrieval Strategies

| Strategy | Strengths | Weaknesses |
|----------|-----------|------------|
| Semantic | Paraphrasing, meaning | Misses exact keywords |
| Keyword (BM25) | Exact matches, acronyms | No semantic understanding |
| Hybrid (RRF) | Best of both | Slightly more latency |

### Dependency Injection

All external services use interface abstraction:

```python
# Production: real APIs
config = RAGConfig(mode=RunMode.PRODUCTION, openai_api_key="sk-...")
pipeline = RAGPipeline(config)

# Mock: no API keys needed
config = RAGConfig(mode=RunMode.MOCK)
pipeline = RAGPipeline(config)

# Custom: inject your own providers
pipeline = RAGPipeline(
    config=config,
    embedding_provider=MyCustomEmbeddings(),
    llm_provider=MyCustomLLM(),
)
```

### Error Handling

Uses Result[T, E] pattern (Rust-inspired):

```python
result = pipeline.query("question")
if result.is_ok():
    answer = result.unwrap()
else:
    error = result.error
```

No exceptions for expected failures. Explicit handling required.
