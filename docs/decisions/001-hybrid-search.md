# ADR-001: Hybrid Search with Reciprocal Rank Fusion

## Status
Accepted

## Date
2026-02-06

## Context

RAG systems need to retrieve the most relevant document chunks for a given query. Two main approaches exist:

1. **Semantic search** (vector similarity): Excels at understanding meaning and paraphrasing but can miss exact keyword matches, especially for acronyms, proper nouns, and domain-specific terms.

2. **Keyword search** (BM25): Excels at exact term matching but has no semantic understanding and fails with paraphrased queries.

Production RAG systems frequently encounter queries that would benefit from both approaches.

## Decision

Implement hybrid retrieval combining semantic (ChromaDB) and keyword (BM25) search, fused using Reciprocal Rank Fusion (RRF).

**Why RRF over alternatives:**
- Linear combination requires score normalization across different scales
- RRF is rank-based, naturally handling score scale differences
- Proven effective in information retrieval literature
- Simple to implement and reason about
- Configurable via semantic_weight and keyword_weight parameters

## Consequences

### Positive
- Better retrieval quality across diverse query types
- Catches both semantic matches and exact keyword hits
- Configurable weights allow tuning for specific use cases
- Each retriever can be used independently if needed

### Negative
- Slightly higher latency (two search passes)
- More memory usage (both vector store and BM25 index)
- Additional complexity in retrieval pipeline

### Mitigations
- Retrieval method is configurable (can use semantic-only or keyword-only)
- Both indexes are maintained in-memory for fast access
- Latency overhead is typically <50ms for reasonable corpus sizes
