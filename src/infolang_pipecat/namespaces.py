"""Session -> namespace mapping for voice memory.

InfoLang scopes memory on two axes:

* **workspace** (tenant) -- set once on the ``AsyncInfoLang`` client and applied
  to every request. Pick it per deployment/customer, not per call.
* **namespace** (bank) -- the per-recall/per-write partition. For voice this is
  where the session mapping lives: each caller (or call) gets a stable namespace
  so a returning caller recalls their own history and nobody else's.

The helpers below turn a raw caller/session identifier (a phone number, a SIP
URI, a user id, a random call id) into a stable, filesystem-and-URL-safe
namespace string. They are pure so a mapping is trivial to unit test and reason
about.

Typical schemes:

* **Per caller** (persistent memory across calls) -- key by the caller's phone
  number or user id::

      ns = voice_namespace(caller_phone_number)  # "voice-caller-15551234567"

* **Per call** (ephemeral, one call only) -- key by the Pipecat/transport call
  id::

      ns = voice_namespace(call_id, prefix="voice-call")
"""

from __future__ import annotations

import re

__all__ = ["DEFAULT_PREFIX", "normalize_identifier", "voice_namespace"]

DEFAULT_PREFIX = "voice-caller"

_UNSAFE = re.compile(r"[^a-z0-9]+")
_TRIM = re.compile(r"(^-+)|(-+$)")


def normalize_identifier(identifier: str) -> str:
    """Lower-case ``identifier`` and collapse unsafe runs into single hyphens.

    Digits and ASCII letters survive; everything else (``+``, spaces, ``@``,
    ``:``, ``/`` from SIP URIs, etc.) becomes a hyphen, with leading/trailing
    hyphens trimmed. Two identifiers that differ only by formatting (``+1 (555)
    123-4567`` vs ``+15551234567``) are *not* forced to collapse to the same
    value -- normalization is lexical, not semantic -- so normalize phone
    numbers upstream (e.g. to E.164) if you need that.

    Raises:
        ValueError: If ``identifier`` is empty or normalizes to an empty string.
    """

    if not identifier or not identifier.strip():
        raise ValueError("identifier must be a non-empty string")
    slug = _TRIM.sub("", _UNSAFE.sub("-", identifier.strip().lower()))
    if not slug:
        raise ValueError(f"identifier {identifier!r} has no usable characters")
    return slug


def voice_namespace(identifier: str, *, prefix: str = DEFAULT_PREFIX) -> str:
    """Build a stable namespace for a caller/session ``identifier``.

    Args:
        identifier: The caller or session key (phone number, user id, SIP URI,
            call id, ...).
        prefix: Namespace prefix that groups all voice banks. Normalized the
            same way as ``identifier``.

    Returns:
        ``"<prefix>-<identifier>"``, normalized to ``[a-z0-9-]``.

    Raises:
        ValueError: If ``prefix`` or ``identifier`` normalizes to empty.
    """

    return f"{normalize_identifier(prefix)}-{normalize_identifier(identifier)}"
