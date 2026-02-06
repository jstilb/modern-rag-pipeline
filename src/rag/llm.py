"""LLM providers with dependency injection for mock mode.

Supports:
- OpenAI LLM (production)
- Mock LLM (demo/testing - returns template-based responses)
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod

from src.rag.config import RAGConfig, RunMode
from src.rag.document import RetrievedChunk
from src.rag.result import Err, Ok, Result


class LLMProvider(ABC):
    """Abstract LLM provider interface."""

    @abstractmethod
    def generate(
        self, query: str, context_chunks: list[RetrievedChunk]
    ) -> Result[str, str]:
        """Generate an answer given a query and retrieved context."""
        ...


class MockLLMProvider(LLMProvider):
    """Deterministic mock LLM for testing and demos.

    Generates plausible-looking answers based on the retrieved context.
    """

    TEMPLATES = [
        "Based on the provided context, {summary}. The key information comes from "
        "the retrieved documents which discuss {topics}.",
        "According to the available sources, {summary}. This is supported by "
        "evidence from {source_count} retrieved passages covering {topics}.",
        "The retrieved context indicates that {summary}. Multiple sources "
        "({source_count} passages) corroborate this finding, particularly "
        "regarding {topics}.",
    ]

    def generate(
        self, query: str, context_chunks: list[RetrievedChunk]
    ) -> Result[str, str]:
        if not context_chunks:
            return Ok(
                "I don't have enough context to answer this question. "
                "Please provide relevant documents first."
            )

        try:
            # Extract key information from context
            all_text = " ".join(rc.chunk.content for rc in context_chunks)
            words = all_text.split()

            # Create a deterministic summary from context
            key_words = list(dict.fromkeys(words[:50]))  # First 50 unique words
            summary = " ".join(key_words[:20])

            # Extract topics from query
            topics = ", ".join(query.split()[:5])

            # Select template deterministically
            template_index = (
                int(hashlib.md5(query.encode()).hexdigest()[:4], 16)
                % len(self.TEMPLATES)
            )
            template = self.TEMPLATES[template_index]

            answer = template.format(
                summary=summary,
                topics=topics,
                source_count=len(context_chunks),
            )

            return Ok(answer)
        except Exception as e:
            return Err(f"Mock generation failed: {e}")


class OpenAILLMProvider(LLMProvider):
    """OpenAI API LLM provider for production use."""

    def __init__(self, config: RAGConfig) -> None:
        self._config = config

    def generate(
        self, query: str, context_chunks: list[RetrievedChunk]
    ) -> Result[str, str]:
        try:
            from langchain_openai import ChatOpenAI

            llm = ChatOpenAI(
                model=self._config.llm_model,
                temperature=self._config.llm_temperature,
                max_tokens=self._config.llm_max_tokens,
                openai_api_key=self._config.openai_api_key,
            )

            context = "\n\n---\n\n".join(
                f"[Source: {rc.chunk.source}, Score: {rc.score:.3f}]\n{rc.chunk.content}"
                for rc in context_chunks
            )

            prompt = (
                f"Answer the following question based ONLY on the provided context. "
                f"If the context doesn't contain enough information, say so.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {query}\n\n"
                f"Answer:"
            )

            response = llm.invoke(prompt)
            return Ok(str(response.content))
        except ImportError:
            return Err("langchain-openai not installed")
        except Exception as e:
            return Err(f"OpenAI generation failed: {e}")


def create_llm_provider(config: RAGConfig) -> LLMProvider:
    """Factory function to create the appropriate LLM provider."""
    if config.mode in (RunMode.MOCK, RunMode.HYBRID):
        return MockLLMProvider()
    return OpenAILLMProvider(config)
