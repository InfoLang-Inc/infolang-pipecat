"""Tests for the session -> namespace mapping helpers."""

from __future__ import annotations

import pytest

from infolang_pipecat import DEFAULT_PREFIX, normalize_identifier, voice_namespace


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("+1 (555) 123-4567", "1-555-123-4567"),
        ("+15551234567", "15551234567"),
        ("USER_42", "user-42"),
        ("sip:alice@example.com", "sip-alice-example-com"),
        ("--Weird__Id!!", "weird-id"),
        ("  spaced  out  ", "spaced-out"),
    ],
)
def test_normalize_identifier(raw: str, expected: str) -> None:
    assert normalize_identifier(raw) == expected


@pytest.mark.parametrize("bad", ["", "   ", "+++", "@@@"])
def test_normalize_identifier_rejects_empty(bad: str) -> None:
    with pytest.raises(ValueError):
        normalize_identifier(bad)


def test_voice_namespace_default_prefix() -> None:
    assert voice_namespace("+15551234567") == f"{DEFAULT_PREFIX}-15551234567"


def test_voice_namespace_custom_prefix() -> None:
    assert voice_namespace("call-abc123", prefix="voice-call") == "voice-call-call-abc123"


def test_voice_namespace_is_stable() -> None:
    assert voice_namespace("+1 (555) 123-4567") == voice_namespace("+1 (555) 123-4567")


def test_voice_namespace_rejects_empty_identifier() -> None:
    with pytest.raises(ValueError):
        voice_namespace("   ")
