"""Optional live smoke test against the real InfoLang API.

Skipped unless ``INFOLANG_API_KEY`` is set -- NOT part of the default ``pytest``
run and excluded from the coverage gate. Only touches namespaces prefixed
``ittest-pipecat-`` and cleans up in a ``finally`` block, so it is safe to run
against a shared account.

Run it with::

    INFOLANG_API_KEY=il_live_... pytest tests/test_live_smoke.py -v
"""

from __future__ import annotations

import os
import uuid

import pytest
from infolang import AsyncInfoLang

from infolang_pipecat import voice_namespace

pytestmark = pytest.mark.skipif(
    not os.environ.get("INFOLANG_API_KEY"),
    reason="live smoke test requires INFOLANG_API_KEY",
)


async def test_live_recall_round_trip() -> None:
    namespace = voice_namespace(uuid.uuid4().hex[:8], prefix="ittest-pipecat")
    memory_id = ""
    async with AsyncInfoLang.from_api_key(os.environ["INFOLANG_API_KEY"]) as client:
        try:
            result = await client.remember(
                "Caller: the passphrase is blue giraffe",
                namespace=namespace,
                source="pipecat-smoke",
                tags="voice,user",
            )
            memory_id = result.memory_id or ""
            assert memory_id

            recalled = await client.recall("passphrase", namespace=namespace, top_k=5)
            assert any("giraffe" in chunk.text for chunk in recalled.chunks)
        finally:
            if memory_id:
                await client.forget(memory_id, namespace=namespace)
