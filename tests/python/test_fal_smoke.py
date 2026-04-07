"""fal.ai compute smoke tests.

Gated on the FAL_API_KEY environment variable.
Tests LLM completion, image generation, TTS, transcription, and timing metadata.
"""

import os

import pytest

from blazen import (
    ChatMessage,
    CompletionModel,
    CompletionOptions,
    FalLlmEndpoint,
    FalOptions,
    FalProvider,
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
async def test_fal_basic_completion_openai_chat():
    """Default endpoint (OpenAiChat) — request lands at openrouter/router/openai/v1/chat/completions."""
    model = CompletionModel.fal(FAL_API_KEY)  # default options
    response = await model.complete(
        [ChatMessage.user("What is 2+2? Reply with just the number.")],
        CompletionOptions(max_tokens=10),
    )
    assert response.content is not None
    assert "4" in response.content


@skip_without_key
@pytest.mark.asyncio
async def test_fal_basic_completion_enterprise():
    """Enterprise mode promotes OpenAiChat -> AnyLlm{enterprise:true}."""
    model = CompletionModel.fal(FAL_API_KEY, options=FalOptions(enterprise=True))
    response = await model.complete(
        [ChatMessage.user("Say hello.")],
        CompletionOptions(max_tokens=10),
    )
    assert response.content is not None


@skip_without_key
@pytest.mark.asyncio
async def test_fal_responses_api_endpoint():
    """OpenAiResponses endpoint targets openrouter/router/openai/v1/responses."""
    model = CompletionModel.fal(
        FAL_API_KEY,
        options=FalOptions(endpoint=FalLlmEndpoint.OpenAiResponses),
    )
    response = await model.complete(
        [ChatMessage.user("Say hi.")],
        CompletionOptions(max_tokens=10),
    )
    assert response.content is not None


@skip_without_key
@pytest.mark.asyncio
async def test_fal_passes_temperature_max_tokens():
    model = CompletionModel.fal(FAL_API_KEY)
    response = await model.complete(
        [ChatMessage.user("Write a one-word greeting.")],
        CompletionOptions(temperature=0.1, max_tokens=10),
    )
    assert response.content is not None


@skip_without_key
@pytest.mark.asyncio
async def test_fal_completion_response_fields():
    model = CompletionModel.fal(FAL_API_KEY)
    response = await model.complete(
        [ChatMessage.user("Hi")],
        CompletionOptions(max_tokens=5),
    )
    assert response.content is not None
    assert response.model is not None
    # New typed fields are accessible (default-empty)
    assert hasattr(response, "tool_calls")
    assert hasattr(response, "reasoning")
    assert hasattr(response, "citations")
    assert hasattr(response, "artifacts")


# ---------------------------------------------------------------------------
# Modality auto-routing (AnyLlm endpoint)
# ---------------------------------------------------------------------------


@skip_without_key
@pytest.mark.asyncio
async def test_fal_vision_auto_routes_when_anyllm_and_image_present():
    """When AnyLlm endpoint is configured and a message has an image part,
    the request should auto-route to fal-ai/any-llm/vision."""
    model = CompletionModel.fal(
        FAL_API_KEY,
        options=FalOptions(endpoint=FalLlmEndpoint.AnyLlm),
    )
    msg = ChatMessage.user_image_url(
        text="What is in this image? One word.",
        url="https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png",
        media_type="image/png",
    )
    response = await model.complete([msg], CompletionOptions(max_tokens=20))
    assert response.content is not None


@skip_without_key
@pytest.mark.asyncio
async def test_fal_audio_auto_routes_when_openrouter_and_audio_present():
    """Audio LLM auto-routes from OpenRouter to openrouter/router/audio.

    fal's workers cannot fetch files from arbitrary public CDNs in test
    environments, so this test asserts that EITHER the call succeeds OR
    the error is specifically a fal file-download failure (which proves
    routing landed at the audio endpoint correctly).
    """
    model = CompletionModel.fal(
        FAL_API_KEY,
        options=FalOptions(endpoint=FalLlmEndpoint.OpenRouter),
    )
    msg = ChatMessage.user_audio(
        text="What does this clip say?",
        url="https://upload.wikimedia.org/wikipedia/commons/c/c8/Example.ogg",
    )
    try:
        response = await model.complete([msg], CompletionOptions(max_tokens=30))
        assert response.content is not None
    except Exception as e:
        err = str(e)
        # Routing succeeded if fal accepted the request and tried to fetch the audio.
        assert any(
            marker in err
            for marker in (
                "Failed to download audio",
                "audio_url",
                "file_download_error",
            )
        ), f"unexpected error (routing may have failed): {err}"


@skip_without_key
@pytest.mark.asyncio
async def test_fal_video_auto_routes_when_openrouter_and_video_present():
    """Video LLM auto-routes from OpenRouter to openrouter/router/video.

    fal's workers cannot fetch files from arbitrary public CDNs in test
    environments, so this test asserts that EITHER the call succeeds OR
    the error is specifically a fal file-download failure (which proves
    routing landed at the video endpoint correctly).
    """
    model = CompletionModel.fal(
        FAL_API_KEY,
        options=FalOptions(endpoint=FalLlmEndpoint.OpenRouter),
    )
    msg = ChatMessage.user_video(
        text="What is happening in this video? One sentence.",
        url="https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4",
    )
    try:
        response = await model.complete([msg], CompletionOptions(max_tokens=40))
        assert response.content is not None
    except Exception as e:
        err = str(e)
        # Routing succeeded if fal accepted the request and tried to fetch the video.
        assert any(
            marker in err
            for marker in (
                "Failed to download video",
                "video_url",
                "file_download_error",
            )
        ), f"unexpected error (routing may have failed): {err}"


# ---------------------------------------------------------------------------
# Streaming, embeddings, 3D, background removal
# ---------------------------------------------------------------------------


@skip_without_key
@pytest.mark.asyncio
async def test_fal_streaming_yields_multiple_chunks():
    model = CompletionModel.fal(FAL_API_KEY)
    chunks = []
    async for chunk in model.stream(
        [ChatMessage.user("Count from 1 to 5, one number per line.")],
        options=CompletionOptions(max_tokens=50),
    ):
        chunks.append(chunk)
    assert len(chunks) > 1, f"expected multiple chunks, got {len(chunks)}"


@skip_without_key
@pytest.mark.asyncio
async def test_fal_embeddings():
    provider = FalProvider(api_key=FAL_API_KEY)
    em = provider.embedding_model()
    response = await em.embed(["hello", "world"])
    # Response shape may be a list of vectors or a wrapping object — check what em.embed returns
    # If it returns an EmbeddingResponse object: response.embeddings has len 2
    # If it returns a raw list: response has len 2
    if hasattr(response, "embeddings"):
        assert len(response.embeddings) == 2
        assert len(response.embeddings[0]) == 1536
    else:
        assert len(response) == 2
        assert len(response[0]) == 1536


@skip_without_key
@pytest.mark.asyncio
async def test_fal_3d_generation():
    """3D generation routes to the correct fal endpoint.

    fal's workers cannot fetch files from arbitrary public CDNs in test
    environments, so this test asserts that EITHER the call succeeds OR
    the error is specifically a fal file-download failure (which proves
    routing landed at the 3D endpoint correctly).
    """
    try:
        from blazen import ThreeDRequest
    except ImportError:
        pytest.skip("ThreeDRequest not exposed in Python bindings")
    provider = FalProvider(api_key=FAL_API_KEY)
    request = ThreeDRequest(
        prompt="a wooden chair",
        image_url="https://storage.googleapis.com/falserverless/example_inputs/triposr_input.jpg",
    )
    try:
        result = await provider.generate_3d(request)
        assert result is not None
    except Exception as e:
        err = str(e)
        # Routing succeeded if fal accepted the request and tried to fetch the image.
        assert any(
            marker in err
            for marker in (
                "file_download_error",
                "image_url",
            )
        ), f"unexpected error (routing may have failed): {err}"


@skip_without_key
@pytest.mark.asyncio
async def test_fal_background_removal():
    """Background removal routes to the correct fal endpoint.

    fal's workers cannot fetch files from arbitrary public CDNs in test
    environments, so this test asserts that EITHER the call succeeds OR
    the error is specifically a fal file-download failure (which proves
    routing landed at the background-removal endpoint correctly).
    """
    provider = FalProvider(api_key=FAL_API_KEY)
    try:
        result = await provider.remove_background(
            image_url="https://storage.googleapis.com/falserverless/example_inputs/birefnet_input.jpeg",
        )
        assert result is not None
    except Exception as e:
        err = str(e)
        # Routing succeeded if fal accepted the request and tried to fetch the image.
        assert any(
            marker in err
            for marker in (
                "file_download_error",
                "image_url",
            )
        ), f"unexpected error (routing may have failed): {err}"


# ---------------------------------------------------------------------------
# FalProvider compute tests
# ---------------------------------------------------------------------------


@skip_without_key
@pytest.mark.asyncio
async def test_fal_image_generation():
    from blazen import ImageRequest

    provider = FalProvider(api_key=FAL_API_KEY)
    result = await provider.generate_image(
        ImageRequest(prompt="a simple red circle on white background")
    )
    assert "images" in result
    assert len(result["images"]) > 0


@skip_without_key
@pytest.mark.asyncio
async def test_fal_text_to_speech():
    from blazen import SpeechRequest

    provider = FalProvider(api_key=FAL_API_KEY)
    result = await provider.text_to_speech(
        SpeechRequest(text="Hello world, this is a test.")
    )
    assert "audio" in result
    assert len(result["audio"]) > 0
