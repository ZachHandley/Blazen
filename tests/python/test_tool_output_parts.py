"""Tests for ``LlmPayload.parts(...)`` and the ``parts_value`` getter.

Covers the newly-added multipart payload classmethod factory and round-trip
inspection. Wraps the payload in a ``ToolOutput`` to confirm the full
two-channel tool-result surface accepts the new variant.

No API keys or network access required.
"""

from blazen import ContentPart, LlmPayload, ToolOutput


# ---------------------------------------------------------------------------
# 1. LlmPayload.parts(...) construction
# ---------------------------------------------------------------------------


def test_llm_payload_parts_constructs_without_error():
    """``LlmPayload.parts([...])`` accepts a list of ContentPart instances."""
    parts = [
        ContentPart.text(text="hi"),
        ContentPart.image_url(url="https://example.com/x.png"),
    ]
    payload = LlmPayload.parts(parts)
    assert payload is not None


# ---------------------------------------------------------------------------
# 2. payload.kind == "parts"
# ---------------------------------------------------------------------------


def test_llm_payload_parts_kind_is_parts():
    """The variant tag for a parts payload is ``"parts"``."""
    payload = LlmPayload.parts(
        [
            ContentPart.text(text="hi"),
            ContentPart.image_url(url="https://example.com/x.png"),
        ]
    )
    assert payload.kind == "parts"


# ---------------------------------------------------------------------------
# 3. parts_value returns a list of two ContentPart instances
# ---------------------------------------------------------------------------


def test_llm_payload_parts_value_round_trips():
    """``parts_value`` returns the list of ContentPart instances passed in."""
    payload = LlmPayload.parts(
        [
            ContentPart.text(text="hi"),
            ContentPart.image_url(url="https://example.com/x.png"),
        ]
    )
    parts = payload.parts_value
    assert parts is not None
    assert len(parts) == 2
    for p in parts:
        assert isinstance(p, ContentPart)


# ---------------------------------------------------------------------------
# 4. First part is text, second is image (verified via __repr__ discriminator)
# ---------------------------------------------------------------------------


def test_llm_payload_parts_value_variant_discriminators():
    """Inspect each part's variant via ``repr()`` (which encodes the variant tag).

    ``PyContentPart`` does not expose a public ``kind`` getter, so the variant
    discriminator is recovered from ``__repr__`` (see
    ``crates/blazen-py/src/types/message.rs``):

      * ``ContentPart.text(text='...')`` for ``Text``
      * ``ContentPart(image)`` for ``Image``
      * ``ContentPart(audio)`` for ``Audio``
      * ``ContentPart(video)`` for ``Video``
      * ``ContentPart(file)`` for ``File``
    """
    payload = LlmPayload.parts(
        [
            ContentPart.text(text="hi"),
            ContentPart.image_url(url="https://example.com/x.png"),
        ]
    )
    parts = payload.parts_value
    assert parts is not None and len(parts) == 2

    first_repr = repr(parts[0])
    second_repr = repr(parts[1])

    assert "ContentPart.text(" in first_repr, (
        f"first part should be a text variant, got repr={first_repr!r}"
    )
    assert "hi" in first_repr

    assert "image" in second_repr.lower(), (
        f"second part should be an image variant, got repr={second_repr!r}"
    )


# ---------------------------------------------------------------------------
# 5. ToolOutput wrapping with a parts override
# ---------------------------------------------------------------------------


def test_tool_output_accepts_parts_override():
    """``ToolOutput(data=..., llm_override=LlmPayload.parts(...))`` constructs.

    Confirms the full two-channel tool-result surface accepts a parts override
    (the structured ``data`` channel is independent of the LLM-visible payload).
    """
    payload = LlmPayload.parts(
        [
            ContentPart.text(text="hi"),
            ContentPart.image_url(url="https://example.com/x.png"),
        ]
    )
    output = ToolOutput(data={"ok": True}, llm_override=payload)

    assert output.data == {"ok": True}
    override = output.llm_override
    assert override is not None
    assert override.kind == "parts"
    round_trip_parts = override.parts_value
    assert round_trip_parts is not None
    assert len(round_trip_parts) == 2


# ---------------------------------------------------------------------------
# 6. Non-parts payloads return None for parts_value
# ---------------------------------------------------------------------------


def test_llm_payload_text_returns_none_for_parts_value():
    """A ``Text`` payload returns ``None`` for the ``parts_value`` getter.

    Confirms ``parts_value`` is variant-discriminated and does not leak across
    payload kinds.
    """
    payload = LlmPayload.text("just text")
    assert payload.kind == "text"
    assert payload.parts_value is None
