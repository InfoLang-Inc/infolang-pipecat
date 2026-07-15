"""InfoLang semantic memory for the Pipecat voice-agent framework.

Quickstart::

    from infolang import AsyncInfoLang
    from infolang_pipecat import InfoLangMemoryProcessor, voice_namespace

    client = AsyncInfoLang.from_api_key("il_live_...")
    memory = InfoLangMemoryProcessor(
        client=client,
        namespace=voice_namespace(caller_phone_number),
    )

    pipeline = Pipeline([
        transport.input(),
        stt,
        context_aggregator.user(),
        memory,            # recall before the LLM, retain after each turn
        llm,
        tts,
        transport.output(),
        context_aggregator.assistant(),
    ])
"""

from __future__ import annotations

from ._version import __version__
from .namespaces import DEFAULT_PREFIX, normalize_identifier, voice_namespace
from .processor import (
    DEFAULT_SOURCE,
    DEFAULT_SYSTEM_PROMPT,
    InfoLangMemoryProcessor,
)

__all__ = [
    "__version__",
    "InfoLangMemoryProcessor",
    "DEFAULT_SOURCE",
    "DEFAULT_SYSTEM_PROMPT",
    "voice_namespace",
    "normalize_identifier",
    "DEFAULT_PREFIX",
]
