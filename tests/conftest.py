"""Shared fixtures: an offline fake InfoLang client and context helpers.

The fake stands in for ``AsyncInfoLang`` so the whole suite runs offline with no
network and no real API key. It records calls and lets each test set the recall
result, an error, or an artificial delay (to exercise the recall timeout).
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from infolang import Chunk, RecallResult, RememberResult
from pipecat.processors.aggregators.llm_context import LLMContext


class FakeInfoLang:
    """In-memory stand-in for ``AsyncInfoLang`` used by the processor."""

    def __init__(self) -> None:
        self.recall_result = RecallResult(chunks=[])
        self.recall_error: Exception | None = None
        self.recall_delay: float = 0.0
        self.remember_error: Exception | None = None
        self.recall_calls: list[dict[str, Any]] = []
        self.remember_calls: list[dict[str, Any]] = []
        self.closed = False

    def set_chunks(self, *chunks: tuple[str, float | None]) -> None:
        self.recall_result = RecallResult(
            chunks=[
                Chunk(id=str(i), text=text, score=score)
                for i, (text, score) in enumerate(chunks)
            ]
        )

    async def recall(
        self,
        query: str,
        *,
        namespace: str | None = None,
        top_k: int | None = None,
        **_: Any,
    ) -> RecallResult:
        self.recall_calls.append({"query": query, "namespace": namespace, "top_k": top_k})
        if self.recall_delay:
            await asyncio.sleep(self.recall_delay)
        if self.recall_error is not None:
            raise self.recall_error
        return self.recall_result

    async def remember(
        self,
        text: str,
        *,
        namespace: str | None = None,
        source: str | None = None,
        tags: str | None = None,
        **_: Any,
    ) -> RememberResult:
        self.remember_calls.append(
            {"text": text, "namespace": namespace, "source": source, "tags": tags}
        )
        if self.remember_error is not None:
            raise self.remember_error
        return RememberResult(memory_id="mem-fake")

    async def aclose(self) -> None:
        self.closed = True


@pytest.fixture
def fake_client() -> FakeInfoLang:
    return FakeInfoLang()


def make_context(*messages: dict[str, Any]) -> LLMContext:
    """Build an ``LLMContext`` seeded with ``messages``."""

    return LLMContext(list(messages))
