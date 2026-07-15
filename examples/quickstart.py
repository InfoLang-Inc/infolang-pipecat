"""Wire InfoLang memory into a Pipecat voice pipeline.

This is an illustrative sketch (transport/STT/TTS/LLM services are elided) that
shows *where* the ``InfoLangMemoryProcessor`` goes: right after the user context
aggregator and before the LLM. Set ``INFOLANG_API_KEY`` and run inside your own
Pipecat app.

    pip install infolang-pipecat
"""

from __future__ import annotations

import os

from infolang import AsyncInfoLang
from pipecat.pipeline.pipeline import Pipeline

from infolang_pipecat import InfoLangMemoryProcessor, voice_namespace


def build_pipeline(caller_phone_number: str, *, transport, stt, llm, tts):  # type: ignore[no-untyped-def]
    """Return a Pipeline with per-caller InfoLang memory."""

    client = AsyncInfoLang.from_api_key(os.environ["INFOLANG_API_KEY"])

    # One namespace (bank) per caller -> a returning caller recalls their own
    # history. Set the client's `workspace=` to scope by tenant/customer.
    memory = InfoLangMemoryProcessor(
        client=client,
        namespace=voice_namespace(caller_phone_number),
        top_k=6,
        recall_timeout_s=2.0,
    )

    context_aggregator = llm.create_context_aggregator(  # provided by your LLM service
        llm.create_context()
    )

    return Pipeline(
        [
            transport.input(),
            stt,
            context_aggregator.user(),
            memory,  # recall before the LLM, retain after each turn
            llm,
            tts,
            transport.output(),
            context_aggregator.assistant(),
        ]
    )
