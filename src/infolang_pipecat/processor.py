"""``InfoLangMemoryProcessor`` -- an InfoLang-backed memory ``FrameProcessor``.

Drop this processor into a Pipecat pipeline **between the user context
aggregator and the LLM service**::

    context_aggregator = llm.create_context_aggregator(context)
    memory = InfoLangMemoryProcessor(client=client, namespace=caller_ns)

    pipeline = Pipeline([
        transport.input(),
        stt,
        context_aggregator.user(),
        memory,          # <-- recall before the LLM, retain after each turn
        llm,
        tts,
        transport.output(),
        context_aggregator.assistant(),
    ])

At that position the processor sees the ``LLMContextFrame`` the user aggregator
emits right before LLM inference (Pipecat 1.x), and -- because the assistant
aggregator *broadcasts* ``LLMContextAssistantTurnFrame`` upstream -- it also sees
each completed assistant turn.

Two things happen:

* **Recall (before the LLM).** On each ``LLMContextFrame`` the latest user
  message is used to recall relevant memories, which are injected as a single
  refreshed system message. Recall is awaited inline (it must land before
  inference) but bounded by ``recall_timeout_s`` and *fail-open*: a slow or
  failing recall never blocks or breaks the turn.
* **Retain (after each turn).** User turns (from the context frame) and
  assistant turns (from ``LLMContextAssistantTurnFrame``) are written back to
  InfoLang in background tasks, so the audio path is never blocked.
"""

from __future__ import annotations

import asyncio
from typing import Any

from infolang import AsyncInfoLang, RecallResult
from loguru import logger
from pipecat.frames.frames import (
    Frame,
    LLMContextAssistantTurnFrame,
    LLMContextFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

__all__ = ["DEFAULT_SOURCE", "DEFAULT_SYSTEM_PROMPT", "InfoLangMemoryProcessor"]

DEFAULT_SOURCE = "pipecat"
DEFAULT_SYSTEM_PROMPT = "Relevant details the caller shared in earlier conversations:"

_ROLE_LABELS = {"user": "Caller", "assistant": "Assistant"}


class InfoLangMemoryProcessor(FrameProcessor):
    """Recall InfoLang memories before the LLM and retain turns after."""

    def __init__(
        self,
        *,
        client: AsyncInfoLang,
        namespace: str | None = None,
        source: str = DEFAULT_SOURCE,
        top_k: int = 6,
        recall_timeout_s: float = 2.0,
        min_query_chars: int = 3,
        recall_enabled: bool = True,
        retain_enabled: bool = True,
        add_as_system_message: bool = True,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        memory_position: int = 0,
        score_floor: float | None = None,
        retain_tags: str = "voice",
        label_turns: bool = True,
        name: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Create the processor.

        Args:
            client: An ``AsyncInfoLang`` client. Its ``workspace`` selects the
                tenant; ``namespace`` here selects the per-caller/session bank.
            namespace: InfoLang namespace (bank) for reads and writes. When
                ``None`` the client's own default namespace applies. For voice,
                build this per caller/session (see ``namespaces.voice_namespace``).
            source: ``source`` provenance tag written on every retained turn.
            top_k: Max memories to recall and inject per turn.
            recall_timeout_s: Hard cap on recall latency. On timeout the turn
                proceeds with no injected memory (fail-open).
            min_query_chars: Skip recall for user text shorter than this (after
                stripping) -- avoids spending the latency budget on "hi"/"yes".
            recall_enabled: Toggle the recall/injection path.
            retain_enabled: Toggle the background retain path.
            add_as_system_message: Inject recalled memory as a ``system`` message
                (default) or a ``user`` message when ``False``.
            system_prompt: Prefix line for the injected memory message.
            memory_position: Index at which to insert the memory message in the
                message list (``0`` = front).
            score_floor: If set, only inject chunks scoring at or above this.
            retain_tags: Comma-separated tag prefix stored on retained turns; the
                turn role is appended (e.g. ``"voice,user"``).
            label_turns: Prefix retained text with a speaker label ("Caller: ",
                "Assistant: ") so recalled memories are self-describing.
            name: Optional processor name.
            **kwargs: Forwarded to :class:`FrameProcessor`.
        """

        super().__init__(name=name, **kwargs)
        self._client = client
        self._namespace = namespace
        self._source = source
        self._top_k = top_k
        self._recall_timeout_s = recall_timeout_s
        self._min_query_chars = min_query_chars
        self._recall_enabled = recall_enabled
        self._retain_enabled = retain_enabled
        self._add_as_system_message = add_as_system_message
        self._system_prompt = system_prompt
        self._memory_position = memory_position
        self._score_floor = score_floor
        self._retain_tags = retain_tags
        self._label_turns = label_turns

        self._owns_client = False
        self._injected_message: dict[str, Any] | None = None
        self._retained_user_ids: set[int] = set()
        self._last_assistant_timestamp: str | None = None

    # --- constructors ---------------------------------------------------

    @classmethod
    def from_api_key(
        cls,
        api_key: str,
        *,
        namespace: str | None = None,
        workspace: str | None = None,
        source: str = DEFAULT_SOURCE,
        **kwargs: Any,
    ) -> InfoLangMemoryProcessor:
        """Build a processor that owns a freshly constructed managed client.

        The processor takes ownership of the client and closes it in
        :meth:`cleanup` (called by Pipecat on pipeline teardown).
        """

        client = AsyncInfoLang.from_api_key(api_key, namespace=namespace, workspace=workspace)
        processor = cls(client=client, namespace=namespace, source=source, **kwargs)
        processor._owns_client = True
        return processor

    @property
    def namespace(self) -> str | None:
        """The namespace (bank) reads and writes target."""

        return self._namespace

    # --- frame handling -------------------------------------------------

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Recall before ``LLMContextFrame`` and retain assistant turns."""

        await super().process_frame(frame, direction)

        if isinstance(frame, LLMContextFrame):
            await self._handle_context_frame(frame)
            await self.push_frame(frame, direction)
        elif isinstance(frame, LLMContextAssistantTurnFrame):
            self._retain_assistant_turn(frame)
            await self.push_frame(frame, direction)
        else:
            await self.push_frame(frame, direction)

    async def _handle_context_frame(self, frame: LLMContextFrame) -> None:
        context = frame.context
        try:
            messages = context.get_messages()
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"{self}: could not read context messages: {exc}")
            return

        user_message = self._latest_user_message(messages)
        if user_message is None:
            return
        user_text = _message_text(user_message)
        if not user_text:
            return

        if self._retain_enabled:
            self._retain_user_message(user_message, user_text)

        if self._recall_enabled and len(user_text.strip()) >= self._min_query_chars:
            await self._recall_and_inject(context, user_text)

    # --- recall / injection ---------------------------------------------

    async def _recall_and_inject(self, context: LLMContext, query: str) -> None:
        try:
            result = await asyncio.wait_for(
                self._client.recall(query, namespace=self._namespace, top_k=self._top_k),
                timeout=self._recall_timeout_s,
            )
        except TimeoutError:
            logger.warning(
                f"{self}: recall exceeded {self._recall_timeout_s}s budget; "
                "proceeding without injected memory"
            )
            return
        except Exception as exc:
            logger.warning(f"{self}: recall failed ({exc}); proceeding without injected memory")
            return

        memory_text = self._format_memories(result)
        if memory_text is None:
            self._remove_injected(context)
            return
        self._inject(context, memory_text)

    def _format_memories(self, result: RecallResult) -> str | None:
        chunks = [
            chunk
            for chunk in result.chunks
            if chunk.text
            and (
                self._score_floor is None
                or (chunk.score is not None and chunk.score >= self._score_floor)
            )
        ]
        if not chunks:
            return None
        lines = [self._system_prompt.rstrip("\n")]
        lines.extend(f"{i}. {chunk.text}" for i, chunk in enumerate(chunks, 1))
        return "\n".join(lines)

    def _inject(self, context: LLMContext, memory_text: str) -> None:
        messages: list[Any] = [
            m for m in context.get_messages() if m is not self._injected_message
        ]
        role = "system" if self._add_as_system_message else "user"
        message: dict[str, Any] = {"role": role, "content": memory_text}
        position = max(0, min(self._memory_position, len(messages)))
        messages.insert(position, message)
        context.set_messages(messages)
        self._injected_message = message

    def _remove_injected(self, context: LLMContext) -> None:
        if self._injected_message is None:
            return
        messages: list[Any] = [
            m for m in context.get_messages() if m is not self._injected_message
        ]
        context.set_messages(messages)
        self._injected_message = None

    # --- retain ---------------------------------------------------------

    def _retain_user_message(self, message: dict[str, Any], text: str) -> None:
        key = id(message)
        if key in self._retained_user_ids:
            return
        self._retained_user_ids.add(key)
        self._schedule_retain("user", text)

    def _retain_assistant_turn(self, frame: LLMContextAssistantTurnFrame) -> None:
        if not self._retain_enabled:
            return
        text = (frame.text or "").strip()
        if not text:
            return
        timestamp = getattr(frame, "timestamp", None)
        if timestamp and timestamp == self._last_assistant_timestamp:
            return
        self._last_assistant_timestamp = timestamp
        self._schedule_retain("assistant", text)

    def _schedule_retain(self, role: str, text: str) -> None:
        self.create_task(self._retain(role, text), name=f"infolang_retain_{role}")

    async def _retain(self, role: str, text: str) -> None:
        content = f"{_ROLE_LABELS.get(role, role.title())}: {text}" if self._label_turns else text
        tags = f"{self._retain_tags},{role}" if self._retain_tags else role
        try:
            await self._client.remember(
                content, namespace=self._namespace, source=self._source, tags=tags
            )
        except Exception as exc:
            logger.warning(f"{self}: retain of {role} turn failed: {exc}")

    # --- helpers --------------------------------------------------------

    @staticmethod
    def _latest_user_message(messages: list[Any]) -> dict[str, Any] | None:
        for message in reversed(messages):
            if isinstance(message, dict) and message.get("role") == "user":
                return message
        return None

    # --- lifecycle ------------------------------------------------------

    async def cleanup(self) -> None:
        """Close the owned InfoLang client, if this processor created it."""

        await super().cleanup()  # type: ignore[no-untyped-call]
        if self._owns_client:
            await self._client.aclose()


def _message_text(message: dict[str, Any]) -> str:
    """Extract plain text from a chat message ``content`` (str or parts list)."""

    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        ]
        return " ".join(p for p in parts if p).strip()
    return ""
