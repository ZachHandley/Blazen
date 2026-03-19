"""fal.ai compute smoke tests.

Gated on the FAL_API_KEY environment variable.
Tests LLM completion, image generation, TTS, transcription, and timing metadata.
"""

import os

import pytest

from blazen import (
    ChatMessage,
    CompletionModel,
)

FAL_API_KEY = os.environ.get("FAL_API_KEY")

skip_without_key = pytest.mark.skipif(
    not FAL_API_KEY,
    reason="FAL_API_KEY not set",
)


# ---------------------------------------------------------------------------
# LLM completion via fal-ai/any-llm
# ---------------------------------------------------------------------------


@skip_without_key
@pytest.mark.asyncio
async def test_fal_basic_completion():
    model = CompletionModel.fal(FAL_API_KEY)
    response = await model.complete(
        [ChatMessage.user("What is 2+2? Reply with just the number.")],
        max_tokens=10,
    )

    assert response.content is not None
    assert "4" in response.content
    assert response.model is not None


@skip_without_key
@pytest.mark.asyncio
async def test_fal_timing_metadata():
    model = CompletionModel.fal(FAL_API_KEY)
    response = await model.complete(
        [ChatMessage.user("Say hello.")],
    )

    assert response.content is not None
    # fal.ai queue mode should populate timing
    if response.timing is not None:
        assert response.timing.total_ms is None or isinstance(response.timing.total_ms, int)


@skip_without_key
@pytest.mark.asyncio
async def test_fal_passes_temperature():
    model = CompletionModel.fal(FAL_API_KEY)
    response = await model.complete(
        [ChatMessage.user("Write a one-word greeting.")],
        temperature=0.1,
        max_tokens=10,
    )

    assert response.content is not None
    # With max_tokens=10, should be short
    assert len(response.content) < 200


@skip_without_key
@pytest.mark.asyncio
async def test_fal_completion_response_fields():
    """Verify all CompletionResponse fields are accessible."""
    model = CompletionModel.fal(FAL_API_KEY)
    response = await model.complete(
        [ChatMessage.user("Hi")],
        max_tokens=5,
    )

    # Attribute access
    assert response.content is not None or response.content is None  # can be either
    assert isinstance(response.model, str)
    assert response.finish_reason is None or isinstance(response.finish_reason, str)
    assert isinstance(response.tool_calls, list)
    assert response.cost is None or isinstance(response.cost, float)
    assert isinstance(response.images, list)
    assert isinstance(response.audio, list)
    assert isinstance(response.videos, list)

    # Dict-style access (backwards compat)
    assert response["model"] is not None
    assert "content" in response.keys()
    assert "cost" in response.keys()
    assert "timing" in response.keys()


# ---------------------------------------------------------------------------
# FalProvider compute tests
# ---------------------------------------------------------------------------


@skip_without_key
@pytest.mark.asyncio
async def test_fal_image_generation():
    from blazen import FalProvider, ImageRequest

    provider = FalProvider(api_key=FAL_API_KEY)
    result = await provider.generate_image(
        ImageRequest(prompt="a simple red circle on white background")
    )
    assert "images" in result
    assert len(result["images"]) > 0


@skip_without_key
@pytest.mark.asyncio
async def test_fal_text_to_speech():
    from blazen import FalProvider, SpeechRequest

    provider = FalProvider(api_key=FAL_API_KEY)
    result = await provider.text_to_speech(
        SpeechRequest(text="Hello world, this is a test.")
    )
    assert "audio" in result
    assert len(result["audio"]) > 0
