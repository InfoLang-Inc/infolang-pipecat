"""Unit and integration tests for ``InfoLangMemoryProcessor``.

Unit tests drive the processor's methods directly (deterministic, no pipeline).
Integration tests run it inside a real Pipecat pipeline via
``pipecat.tests.utils.run_test`` to verify frame routing end to end. Everything
runs offline against the ``FakeInfoLang`` client.
"""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import Any

from pipecat.frames.frames import (
    LLMContextAssistantTurnFrame,
    LLMContextFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.tests.utils import SleepFrame, run_test

from infolang_pipecat import InfoLangMemoryProcessor
from tests.conftest import FakeInfoLang, make_context


def _capture_tasks(processor: InfoLangMemoryProcessor) -> list[asyncio.Task[Any]]:
    """Patch ``create_task`` so scheduled retains run without a live pipeline."""

    tasks: list[asyncio.Task[Any]] = []

    def fake_create_task(
        coro: Coroutine[Any, Any, Any], name: str | None = None
    ) -> asyncio.Task[Any]:
        task = asyncio.ensure_future(coro)
        tasks.append(task)
        return task

    processor.create_task = fake_create_task  # type: ignore[method-assign]
    return tasks


async def _settle(tasks: list[asyncio.Task[Any]]) -> None:
    if tasks:
        await asyncio.gather(*tasks)


# --- recall + injection -------------------------------------------------


async def test_recall_injects_system_message(fake_client: FakeInfoLang) -> None:
    fake_client.set_chunks(("Caller prefers window seats", 0.95))
    processor = InfoLangMemoryProcessor(client=fake_client, namespace="ns-a", retain_enabled=False)
    context = make_context(
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "book me a flight"},
    )

    await processor._recall_and_inject(context, "book me a flight")

    messages = context.get_messages()
    assert messages[0]["role"] == "system"
    assert "window seats" in messages[0]["content"]
    assert fake_client.recall_calls[0]["namespace"] == "ns-a"
    assert fake_client.recall_calls[0]["top_k"] == 6


async def test_recall_injects_as_user_message_when_configured(fake_client: FakeInfoLang) -> None:
    fake_client.set_chunks(("some memory", 0.9))
    processor = InfoLangMemoryProcessor(
        client=fake_client, add_as_system_message=False, memory_position=1
    )
    context = make_context({"role": "user", "content": "hello there friend"})

    await processor._recall_and_inject(context, "hello there friend")

    injected = context.get_messages()[1]
    assert injected["role"] == "user"
    assert "some memory" in injected["content"]


async def test_recall_refreshes_single_injected_message(fake_client: FakeInfoLang) -> None:
    processor = InfoLangMemoryProcessor(client=fake_client, retain_enabled=False)
    context = make_context({"role": "user", "content": "first question"})

    fake_client.set_chunks(("memory one", 0.95))
    await processor._recall_and_inject(context, "first question")
    fake_client.set_chunks(("memory two", 0.95))
    await processor._recall_and_inject(context, "first question again")

    system_messages = [m for m in context.get_messages() if m["role"] == "system"]
    assert len(system_messages) == 1
    assert "memory two" in system_messages[0]["content"]
    assert "memory one" not in system_messages[0]["content"]


async def test_no_chunks_removes_stale_injection(fake_client: FakeInfoLang) -> None:
    processor = InfoLangMemoryProcessor(client=fake_client, retain_enabled=False)
    context = make_context({"role": "user", "content": "a question"})

    fake_client.set_chunks(("stale memory", 0.95))
    await processor._recall_and_inject(context, "a question")
    assert any(m["role"] == "system" for m in context.get_messages())

    fake_client.recall_result.chunks = []
    await processor._recall_and_inject(context, "a question")
    assert not any(m["role"] == "system" for m in context.get_messages())


async def test_score_floor_filters_low_confidence(fake_client: FakeInfoLang) -> None:
    fake_client.set_chunks(("weak match", 0.4), ("strong match", 0.95))
    processor = InfoLangMemoryProcessor(client=fake_client, score_floor=0.85, retain_enabled=False)
    context = make_context({"role": "user", "content": "the question"})

    await processor._recall_and_inject(context, "the question")

    content = next(m["content"] for m in context.get_messages() if m["role"] == "system")
    assert "strong match" in content
    assert "weak match" not in content


async def test_recall_timeout_is_fail_open(fake_client: FakeInfoLang) -> None:
    fake_client.recall_delay = 0.2
    processor = InfoLangMemoryProcessor(
        client=fake_client, recall_timeout_s=0.01, retain_enabled=False
    )
    context = make_context({"role": "user", "content": "slow question"})

    await processor._recall_and_inject(context, "slow question")

    assert not any(m["role"] == "system" for m in context.get_messages())


async def test_recall_error_is_fail_open(fake_client: FakeInfoLang) -> None:
    fake_client.recall_error = RuntimeError("boom")
    processor = InfoLangMemoryProcessor(client=fake_client, retain_enabled=False)
    context = make_context({"role": "user", "content": "another question"})

    await processor._recall_and_inject(context, "another question")

    assert not any(m["role"] == "system" for m in context.get_messages())


# --- context frame orchestration ---------------------------------------


async def test_handle_context_frame_recalls_and_retains(fake_client: FakeInfoLang) -> None:
    fake_client.set_chunks(("remembered fact", 0.95))
    processor = InfoLangMemoryProcessor(client=fake_client, namespace="ns-b")
    tasks = _capture_tasks(processor)
    context = make_context(
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "what did I say before"},
    )

    await processor._handle_context_frame(LLMContextFrame(context=context))
    await _settle(tasks)

    assert fake_client.recall_calls[0]["query"] == "what did I say before"
    assert fake_client.remember_calls == [
        {
            "text": "Caller: what did I say before",
            "namespace": "ns-b",
            "source": "pipecat",
            "tags": "voice,user",
        }
    ]


async def test_no_user_message_is_a_noop(fake_client: FakeInfoLang) -> None:
    processor = InfoLangMemoryProcessor(client=fake_client)
    _capture_tasks(processor)
    context = make_context({"role": "system", "content": "system only"})

    await processor._handle_context_frame(LLMContextFrame(context=context))

    assert fake_client.recall_calls == []
    assert fake_client.remember_calls == []


async def test_short_query_skips_recall_but_still_retains(fake_client: FakeInfoLang) -> None:
    processor = InfoLangMemoryProcessor(client=fake_client, min_query_chars=5)
    tasks = _capture_tasks(processor)
    context = make_context({"role": "user", "content": "hi"})

    await processor._handle_context_frame(LLMContextFrame(context=context))
    await _settle(tasks)

    assert fake_client.recall_calls == []
    assert fake_client.remember_calls[0]["text"] == "Caller: hi"


async def test_user_turn_retained_once(fake_client: FakeInfoLang) -> None:
    processor = InfoLangMemoryProcessor(client=fake_client, recall_enabled=False)
    tasks = _capture_tasks(processor)
    context = make_context({"role": "user", "content": "remember this once"})
    frame = LLMContextFrame(context=context)

    await processor._handle_context_frame(frame)
    await processor._handle_context_frame(frame)
    await _settle(tasks)

    assert len(fake_client.remember_calls) == 1


async def test_recall_disabled(fake_client: FakeInfoLang) -> None:
    fake_client.set_chunks(("unused", 0.95))
    processor = InfoLangMemoryProcessor(client=fake_client, recall_enabled=False)
    tasks = _capture_tasks(processor)
    context = make_context({"role": "user", "content": "a real question"})

    await processor._handle_context_frame(LLMContextFrame(context=context))
    await _settle(tasks)

    assert fake_client.recall_calls == []
    assert not any(m["role"] == "system" for m in context.get_messages())
    assert fake_client.remember_calls  # retain still happens


# --- retain -------------------------------------------------------------


async def test_retain_disabled_still_recalls(fake_client: FakeInfoLang) -> None:
    fake_client.set_chunks(("a stored fact", 0.95))
    processor = InfoLangMemoryProcessor(client=fake_client, retain_enabled=False)
    _capture_tasks(processor)
    context = make_context({"role": "user", "content": "recall for me please"})

    await processor._handle_context_frame(LLMContextFrame(context=context))

    assert fake_client.recall_calls[0]["query"] == "recall for me please"
    assert fake_client.remember_calls == []


async def test_retain_labels_and_tags_turns(fake_client: FakeInfoLang) -> None:
    processor = InfoLangMemoryProcessor(client=fake_client, namespace="ns-c", source="my-bot")

    await processor._retain("assistant", "how can I help?")

    assert fake_client.remember_calls == [
        {
            "text": "Assistant: how can I help?",
            "namespace": "ns-c",
            "source": "my-bot",
            "tags": "voice,assistant",
        }
    ]


async def test_retain_without_labels_or_tags(fake_client: FakeInfoLang) -> None:
    processor = InfoLangMemoryProcessor(client=fake_client, label_turns=False, retain_tags="")

    await processor._retain("user", "plain text")

    call = fake_client.remember_calls[0]
    assert call["text"] == "plain text"
    assert call["tags"] == "user"


async def test_retain_swallows_errors(fake_client: FakeInfoLang) -> None:
    fake_client.remember_error = RuntimeError("write failed")
    processor = InfoLangMemoryProcessor(client=fake_client)

    await processor._retain("user", "will fail")  # must not raise


async def test_assistant_turn_dedup_by_timestamp(fake_client: FakeInfoLang) -> None:
    processor = InfoLangMemoryProcessor(client=fake_client)
    tasks = _capture_tasks(processor)
    frame = LLMContextAssistantTurnFrame(text="same turn", timestamp="2026-01-01T00:00:00Z")

    processor._retain_assistant_turn(frame)
    processor._retain_assistant_turn(frame)
    await _settle(tasks)

    assert len(fake_client.remember_calls) == 1


async def test_assistant_turn_empty_text_skipped(fake_client: FakeInfoLang) -> None:
    processor = InfoLangMemoryProcessor(client=fake_client)
    _capture_tasks(processor)

    processor._retain_assistant_turn(
        LLMContextAssistantTurnFrame(text="   ", timestamp="2026-01-01T00:00:01Z")
    )

    assert fake_client.remember_calls == []


async def test_assistant_turn_retain_disabled(fake_client: FakeInfoLang) -> None:
    processor = InfoLangMemoryProcessor(client=fake_client, retain_enabled=False)
    _capture_tasks(processor)

    processor._retain_assistant_turn(
        LLMContextAssistantTurnFrame(text="ignored", timestamp="2026-01-01T00:00:02Z")
    )

    assert fake_client.remember_calls == []


# --- multimodal + helpers ----------------------------------------------


async def test_multimodal_user_content_extracts_text(fake_client: FakeInfoLang) -> None:
    processor = InfoLangMemoryProcessor(client=fake_client, recall_enabled=False)
    tasks = _capture_tasks(processor)
    context = make_context(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "describe this"},
                {"type": "image_url", "image_url": {"url": "data:..."}},
            ],
        }
    )

    await processor._handle_context_frame(LLMContextFrame(context=context))
    await _settle(tasks)

    assert fake_client.remember_calls[0]["text"] == "Caller: describe this"


async def test_non_text_user_content_is_a_noop(fake_client: FakeInfoLang) -> None:
    processor = InfoLangMemoryProcessor(client=fake_client)
    _capture_tasks(processor)
    context = make_context({"role": "user", "content": {"unexpected": "shape"}})

    await processor._handle_context_frame(LLMContextFrame(context=context))

    assert fake_client.recall_calls == []
    assert fake_client.remember_calls == []


async def test_remove_injected_when_nothing_injected(fake_client: FakeInfoLang) -> None:
    processor = InfoLangMemoryProcessor(client=fake_client, retain_enabled=False)
    context = make_context({"role": "user", "content": "no memories exist yet"})

    # recall returns nothing -> _remove_injected runs with no prior injection
    await processor._recall_and_inject(context, "no memories exist yet")

    assert [m["role"] for m in context.get_messages()] == ["user"]


# --- lifecycle ----------------------------------------------------------


def test_from_api_key_owns_client() -> None:
    processor = InfoLangMemoryProcessor.from_api_key("il_test_key", namespace="ns-d")
    assert processor.namespace == "ns-d"
    assert processor._owns_client is True


async def test_cleanup_closes_owned_client(fake_client: FakeInfoLang) -> None:
    processor = InfoLangMemoryProcessor(client=fake_client)
    processor._owns_client = True

    await processor.cleanup()

    assert fake_client.closed is True


async def test_cleanup_leaves_borrowed_client_open(fake_client: FakeInfoLang) -> None:
    processor = InfoLangMemoryProcessor(client=fake_client)

    await processor.cleanup()

    assert fake_client.closed is False


# --- integration via a real pipeline -----------------------------------


async def test_pipeline_recalls_and_retains_downstream(fake_client: FakeInfoLang) -> None:
    fake_client.set_chunks(("the caller is named Ada", 0.96))
    processor = InfoLangMemoryProcessor(client=fake_client, namespace="ns-e")
    context = make_context(
        {"role": "system", "content": "You are a helpful voice agent."},
        {"role": "user", "content": "do you remember my name"},
    )

    received_down, _ = await run_test(
        processor,
        frames_to_send=[LLMContextFrame(context=context), SleepFrame(0.1)],
        expected_down_frames=[LLMContextFrame],
    )

    assert isinstance(received_down[0], LLMContextFrame)
    messages = context.get_messages()
    assert any(m["role"] == "system" and "Ada" in m["content"] for m in messages)
    assert any(c["tags"] == "voice,user" for c in fake_client.remember_calls)


async def test_pipeline_retains_assistant_turn_upstream(fake_client: FakeInfoLang) -> None:
    processor = InfoLangMemoryProcessor(client=fake_client, namespace="ns-f")

    await run_test(
        processor,
        frames_to_send=[
            LLMContextAssistantTurnFrame(text="Nice to meet you", timestamp="2026-01-01T00:00:03Z"),
            SleepFrame(0.1),
        ],
        frames_to_send_direction=FrameDirection.UPSTREAM,
        expected_up_frames=[LLMContextAssistantTurnFrame],
    )

    assert fake_client.remember_calls == [
        {
            "text": "Assistant: Nice to meet you",
            "namespace": "ns-f",
            "source": "pipecat",
            "tags": "voice,assistant",
        }
    ]
