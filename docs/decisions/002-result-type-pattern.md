# ADR-002: Result Type Pattern for Error Handling

## Status
Accepted

## Date
2026-02-06

## Context

The RAG pipeline has many operations that can fail: embedding API calls, vector store operations, LLM generation, document parsing. We need a consistent error handling strategy that:

1. Makes failure explicit in type signatures
2. Forces callers to handle both success and error cases
3. Avoids exception-based control flow for expected failures
4. Composes well across multiple fallible operations

## Decision

Implement a Rust-inspired Result[T, E] type with Ok and Err variants. All operations that can fail return Result instead of raising exceptions.

```python
result = pipeline.query("question")
if result.is_ok():
    answer = result.unwrap()
else:
    handle_error(result.error)
```

Exceptions are reserved for truly unexpected conditions (programmer errors, system failures).

## Consequences

### Positive
- Failure is visible in function signatures
- Callers cannot accidentally ignore errors
- Composable via map/map_err operations
- No try/except boilerplate for expected failures
- Better testability -- can assert on specific error types

### Negative
- More verbose than exception-based error handling
- Requires discipline to use consistently
- Some Python developers may find it unfamiliar
- Type checker support is imperfect for Union types

### Mitigations
- unwrap_or provides convenient defaults for non-critical operations
- map chains allow clean composition without nested conditionals
- Consistent usage throughout the codebase establishes the pattern
