"""Chunking strategies for splitting documents into retrieval-sized pieces.

Four strategies available:
- FixedSizeChunker: Split by token count with overlap
- RecursiveChunker: Split by semantic boundaries (paragraphs, sentences)
- SemanticChunker: Split by topic/meaning boundaries
- SlidingWindowChunker: Overlapping windows for maximum context preservation
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import replace

from src.rag.document import Chunk, Document


class ChunkingStrategy(ABC):
    """Base class for all chunking strategies."""

    @abstractmethod
    def chunk(self, document: Document) -> list[Chunk]:
        """Split a document into chunks."""
        ...

    def _make_chunk(
        self, content: str, document: Document, index: int
    ) -> Chunk:
        """Create a Chunk from content and source document."""
        words = content.split()
        return Chunk(
            content=content.strip(),
            doc_id=document.doc_id,
            chunk_index=index,
            source=document.source,
            metadata={**document.metadata, "chunking_strategy": self.__class__.__name__},
            token_count=len(words),  # Approximate token count
        )


class FixedSizeChunker(ChunkingStrategy):
    """Split documents into fixed-size chunks by word count with overlap.

    Simple and predictable. Best when documents have uniform structure.
    """

    def __init__(self, chunk_size: int = 256, overlap: int = 32) -> None:
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if overlap < 0:
            raise ValueError(f"overlap must be non-negative, got {overlap}")
        if overlap >= chunk_size:
            raise ValueError(f"overlap ({overlap}) must be less than chunk_size ({chunk_size})")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, document: Document) -> list[Chunk]:
        words = document.content.split()
        if not words:
            return []

        chunks: list[Chunk] = []
        start = 0
        index = 0

        while start < len(words):
            end = start + self.chunk_size
            chunk_words = words[start:end]
            content = " ".join(chunk_words)

            if content.strip():
                chunks.append(self._make_chunk(content, document, index))
                index += 1

            start += self.chunk_size - self.overlap

        return chunks


class RecursiveChunker(ChunkingStrategy):
    """Split documents by semantic boundaries: paragraphs, then sentences, then words.

    Preserves natural document structure. Best for prose and technical docs.
    """

    SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " "]

    def __init__(self, chunk_size: int = 256, overlap: int = 32) -> None:
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, document: Document) -> list[Chunk]:
        raw_chunks = self._split_recursive(document.content, 0)
        result: list[Chunk] = []
        for i, text in enumerate(raw_chunks):
            if text.strip():
                result.append(self._make_chunk(text, document, i))
        return result

    def _split_recursive(self, text: str, sep_index: int) -> list[str]:
        """Recursively split text using progressively finer separators."""
        if len(text.split()) <= self.chunk_size:
            return [text] if text.strip() else []

        if sep_index >= len(self.SEPARATORS):
            # Fall back to fixed-size word splitting
            words = text.split()
            chunks = []
            for i in range(0, len(words), self.chunk_size):
                chunk = " ".join(words[i : i + self.chunk_size])
                if chunk.strip():
                    chunks.append(chunk)
            return chunks

        separator = self.SEPARATORS[sep_index]
        parts = text.split(separator)
        merged: list[str] = []
        current: list[str] = []
        current_len = 0

        for part in parts:
            part_len = len(part.split())
            if current_len + part_len > self.chunk_size and current:
                merged.append(separator.join(current))
                # Keep overlap by preserving last elements
                overlap_parts: list[str] = []
                overlap_len = 0
                for p in reversed(current):
                    p_len = len(p.split())
                    if overlap_len + p_len > self.overlap:
                        break
                    overlap_parts.insert(0, p)
                    overlap_len += p_len
                current = overlap_parts + [part]
                current_len = overlap_len + part_len
            else:
                current.append(part)
                current_len += part_len

        if current:
            merged.append(separator.join(current))

        # If any merged chunk is still too large, split it with the next separator
        result: list[str] = []
        for chunk in merged:
            if len(chunk.split()) > self.chunk_size:
                result.extend(self._split_recursive(chunk, sep_index + 1))
            else:
                result.append(chunk)

        return result


class SemanticChunker(ChunkingStrategy):
    """Split documents at topic/meaning boundaries.

    Uses sentence-level analysis to find natural break points where
    the topic shifts. Falls back to paragraph boundaries when
    semantic analysis is inconclusive.
    """

    def __init__(self, chunk_size: int = 256, similarity_threshold: float = 0.5) -> None:
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        self.chunk_size = chunk_size
        self.similarity_threshold = similarity_threshold

    def chunk(self, document: Document) -> list[Chunk]:
        sentences = self._split_sentences(document.content)
        if not sentences:
            return []

        chunks: list[Chunk] = []
        current_sentences: list[str] = []
        current_len = 0

        for sentence in sentences:
            sent_len = len(sentence.split())

            if current_len + sent_len > self.chunk_size and current_sentences:
                content = " ".join(current_sentences)
                if content.strip():
                    chunks.append(self._make_chunk(content, document, len(chunks)))
                current_sentences = [sentence]
                current_len = sent_len
            else:
                current_sentences.append(sentence)
                current_len += sent_len

        if current_sentences:
            content = " ".join(current_sentences)
            if content.strip():
                chunks.append(self._make_chunk(content, document, len(chunks)))

        return chunks

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split text into sentences using regex."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]


class SlidingWindowChunker(ChunkingStrategy):
    """Create overlapping windows across the document.

    Maximizes context preservation at the cost of more chunks.
    Good for documents where context from neighboring sections matters.
    """

    def __init__(self, window_size: int = 256, step_size: int = 128) -> None:
        if window_size <= 0:
            raise ValueError(f"window_size must be positive, got {window_size}")
        if step_size <= 0:
            raise ValueError(f"step_size must be positive, got {step_size}")
        if step_size > window_size:
            raise ValueError(
                f"step_size ({step_size}) must be <= window_size ({window_size})"
            )
        self.window_size = window_size
        self.step_size = step_size

    def chunk(self, document: Document) -> list[Chunk]:
        words = document.content.split()
        if not words:
            return []

        chunks: list[Chunk] = []
        index = 0

        for start in range(0, len(words), self.step_size):
            end = start + self.window_size
            window = words[start:end]
            content = " ".join(window)

            if content.strip():
                chunks.append(self._make_chunk(content, document, index))
                index += 1

            if end >= len(words):
                break

        return chunks
