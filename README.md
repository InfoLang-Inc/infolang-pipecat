# infolang-pipecat

InfoLang semantic memory for [Pipecat](https://github.com/pipecat-ai/pipecat)
voice agents. A single `FrameProcessor` recalls relevant memories **before the
LLM** on each user turn and retains user + assistant turns **after** — so a
caller who hangs up and calls back is remembered.

Built entirely on the **public** InfoLang Python SDK
([`infolang`](https://pypi.org/project/infolang/)). No runtime or engine
internals.

- **Verified against:** `pipecat-ai` 1.5.0, `infolang` 0.2.0, Python 3.12.
- **Status:** Alpha. Mock-tested offline; live probe is opt-in (see below).

## Install

```bash
pip install infolang-pipecat
```

This pulls in `infolang>=0.2,<0.3` and `pipecat-ai>=1.5,<2`.

## Quickstart

Place `InfoLangMemoryProcessor` **between the user context aggregator and the
LLM**. That is where Pipecat 1.x emits the `LLMContextFrame` that carries the
conversation into the LLM, so the processor can inject recalled memory right
before inference. Because the assistant aggregator *broadcasts* its turn frame
upstream, the same processor also sees each completed assistant turn.

```python
from infolang import AsyncInfoLang
from pipecat.pipeline.pipeline import Pipeline

from infolang_pipecat import InfoLangMemoryProcessor, voice_namespace

client = AsyncInfoLang.from_api_key("il_live_...")

memory = InfoLangMemoryProcessor(
    client=client,
    namespace=voice_namespace(caller_phone_number),  # one bank per caller
    top_k=6,
    recall_timeout_s=2.0,   # hard latency cap; fail-open on timeout
)

pipeline = Pipeline([
    transport.input(),
    stt,
    context_aggregator.user(),
    memory,                 # <-- recall before the LLM, retain after each turn
    llm,
    tts,
    transport.output(),
    context_aggregator.assistant(),
])
```

Or let the processor build and own the client:

```python
memory = InfoLangMemoryProcessor.from_api_key(
    "il_live_...", namespace=voice_namespace(caller_phone_number)
)
```

## How it works

- **Recall (before the LLM).** On each `LLMContextFrame`, the latest user
  message is used as a recall query. Results are injected as a *single, refreshed*
  system message at the front of the context (old injection is replaced, so the
  context never grows unbounded). Recall is awaited inline — it must land before
  inference — but is bounded by `recall_timeout_s` and **fail-open**: a slow or
  failing recall proceeds with no injected memory rather than blocking or
  breaking the turn.
- **Retain (after each turn).** The user turn (from the context frame) and the
  assistant turn (from `LLMContextAssistantTurnFrame`) are written back to
  InfoLang in **background tasks**, so the audio path is never blocked. Each turn
  is stored once (user turns deduped by message identity, assistant turns by
  timestamp), labeled (`Caller:` / `Assistant:`) and tagged (`voice,user` /
  `voice,assistant`).

### Key options

| Option | Default | Purpose |
| --- | --- | --- |
| `namespace` | client default | Per-caller/session bank for reads + writes |
| `top_k` | `6` | Max memories recalled and injected per turn |
| `recall_timeout_s` | `2.0` | Hard cap on recall latency (fail-open) |
| `min_query_chars` | `3` | Skip recall for trivial utterances ("hi", "yes") |
| `score_floor` | `None` | Only inject chunks scoring at/above this |
| `recall_enabled` / `retain_enabled` | `True` | Toggle each path |
| `add_as_system_message` | `True` | Inject as `system` (else `user`) message |
| `source` | `"pipecat"` | Provenance tag on retained turns |

## Workspace vs. namespace

InfoLang scopes memory on two axes:

- **workspace = tenant.** Set once on the `AsyncInfoLang` client
  (`workspace=...` or `INFOLANG_WORKSPACE`). Use it to isolate customers/apps.
  A managed API key must allowlist the workspace.
- **namespace = bank.** The per-recall/per-write partition. For voice this is
  the **session mapping**: `voice_namespace(caller_id)` gives each caller a
  stable bank so returning callers recall their own history and nobody else's.
  Use `voice_namespace(call_id, prefix="voice-call")` for per-call, ephemeral
  memory instead.

A managed API key honors the `namespace` argument on both reads and writes; a
self-hosted dev key is pinned to a single namespace.

## Security & privacy

- Voice transcripts are **PII**. Each turn is stored verbatim in the namespace
  you choose — segregate callers with per-caller namespaces and tenants with
  workspaces, and apply your own retention policy (memories can be removed with
  `client.forget(...)`).
- The processor talks only to the InfoLang public API over HTTPS using the SDK's
  auth (managed API key or dev key). No credentials are logged. Recall/retain
  failures are logged at `warning` without payloads.
- Retain runs off the audio path; recall is time-boxed and fail-open, so InfoLang
  availability never degrades call quality.

## Development

```bash
pip install -e ".[dev]"
ruff check .
mypy
pytest -q          # offline; mocks the InfoLang client and Pipecat frames
```

The default suite is fully offline. The opt-in live probe hits the real API and
only runs when `INFOLANG_API_KEY` is set:

```bash
INFOLANG_API_KEY=il_live_... pytest tests/test_live_smoke.py -v
```

Coverage gate: **90%** line + branch on `infolang_pipecat` (enforced in CI).

## License

Apache-2.0.
