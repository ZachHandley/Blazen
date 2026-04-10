"""fal.ai compute smoke tests.

Gated on the FAL_API_KEY environment variable.
Tests LLM completion, image generation, TTS, transcription, and timing metadata.
"""

import os

import pytest

from blazen import (
    BackgroundRemovalRequest,
    ChatMessage,
    CompletionModel,
    CompletionOptions,
    ComputeRequest,
    FalLlmEndpointKind,
    FalOptions,
    FalProvider,
    ImageRequest,
    MusicRequest,
    RetryConfig,
    SpeechRequest,
    ThreeDRequest,
    TranscriptionRequest,
    VideoRequest,
)

FAL_API_KEY = os.environ.get("FAL_API_KEY")

skip_without_key = pytest.mark.skipif(
    not FAL_API_KEY,
    reason="FAL_API_KEY not set",
)


async def _fal_or_skip(coro):
    """Await a fal compute coroutine; skip on transient infra errors."""
    try:
        return await coro
    except Exception as e:
        err = str(e)
        if any(s in err for s in ("502", "503", "504", "Bad Gateway", "service_unavailable")):
            pytest.skip(f"fal.ai infra error: {err}")
        raise


# ---------------------------------------------------------------------------
# LLM completion via fal-ai/any-llm
# ---------------------------------------------------------------------------


@skip_without_key
@pytest.mark.asyncio
async def test_fal_completion_openai_chat():
    """Default endpoint (OpenAiChat) with temperature, max_tokens, and response field checks."""
    model = CompletionModel.fal(options=FalOptions(api_key=FAL_API_KEY)).with_retry(RetryConfig(max_retries=2))
    response = await model.complete(
        [ChatMessage.user("What is 2+2? Reply with just the number.")],
        CompletionOptions(temperature=0.1, max_tokens=10),
    )
    assert response.content is not None
    assert "4" in response.content
    assert response.model is not None
    assert hasattr(response, "tool_calls")
    assert hasattr(response, "reasoning")
    assert hasattr(response, "citations")
    assert hasattr(response, "artifacts")


@skip_without_key
@pytest.mark.asyncio
async def test_fal_basic_completion_enterprise():
    """Enterprise mode promotes OpenAiChat -> AnyLlm{enterprise:true}."""
    model = CompletionModel.fal(options=FalOptions(api_key=FAL_API_KEY, enterprise=True)).with_retry(RetryConfig(max_retries=2))
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
        options=FalOptions(api_key=FAL_API_KEY, endpoint=FalLlmEndpointKind.OpenAiResponses),
    ).with_retry(RetryConfig(max_retries=2))
    response = await model.complete(
        [ChatMessage.user("Say hi.")],
        CompletionOptions(max_tokens=10),
    )
    assert response.content is not None


# ---------------------------------------------------------------------------
# Modality auto-routing (AnyLlm endpoint)
# ---------------------------------------------------------------------------


@skip_without_key
@pytest.mark.asyncio
async def test_fal_vision_auto_routes_when_anyllm_and_image_present():
    """When AnyLlm endpoint is configured and a message has an image part,
    the request should auto-route to fal-ai/any-llm/vision."""
    model = CompletionModel.fal(
        options=FalOptions(api_key=FAL_API_KEY, endpoint=FalLlmEndpointKind.AnyLlm),
    ).with_retry(RetryConfig(max_retries=2))
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
        options=FalOptions(api_key=FAL_API_KEY, endpoint=FalLlmEndpointKind.OpenRouter),
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
        options=FalOptions(api_key=FAL_API_KEY, endpoint=FalLlmEndpointKind.OpenRouter),
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
    model = CompletionModel.fal(options=FalOptions(api_key=FAL_API_KEY)).with_retry(RetryConfig(max_retries=2))
    chunks = []
    async for chunk in model.stream(
        [ChatMessage.user("Count from 1 to 5, one number per line.")],
        options=CompletionOptions(max_tokens=50),
    ):
        chunks.append(chunk)
    assert len(chunks) > 1, f"expected multiple chunks, got {len(chunks)}"


@skip_without_key
@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_fal_embeddings():
    provider = FalProvider(options=FalOptions(api_key=FAL_API_KEY))
    em = provider.embedding_model()
    response = await _fal_or_skip(em.embed(["hello", "world"]))
    assert len(response["embeddings"]) == 2
    assert len(response["embeddings"][0]) == 1536


@skip_without_key
@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_fal_3d_generation():
    """3D generation routes to the correct fal endpoint.

    fal's workers cannot fetch files from arbitrary public CDNs in test
    environments, so this test asserts that EITHER the call succeeds OR
    the error is specifically a fal file-download failure (which proves
    routing landed at the 3D endpoint correctly).
    """
    provider = FalProvider(options=FalOptions(api_key=FAL_API_KEY))
    request = ThreeDRequest(
        prompt="a wooden chair",
        image_url="https://storage.googleapis.com/falserverless/example_inputs/triposr_input.jpg",
    )
    try:
        result = await _fal_or_skip(provider.generate_3d(request))
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
@pytest.mark.timeout(300)
async def test_fal_background_removal():
    """Background removal routes to the correct fal endpoint.

    fal's workers cannot fetch files from arbitrary public CDNs in test
    environments, so this test asserts that EITHER the call succeeds OR
    the error is specifically a fal file-download failure (which proves
    routing landed at the background-removal endpoint correctly).
    """
    provider = FalProvider(options=FalOptions(api_key=FAL_API_KEY))
    try:
        result = await _fal_or_skip(provider.remove_background(
            BackgroundRemovalRequest(image_url="https://storage.googleapis.com/falserverless/example_inputs/birefnet_input.jpeg"),
        ))
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
@pytest.mark.timeout(300)
async def test_fal_image_generation():
    provider = FalProvider(options=FalOptions(api_key=FAL_API_KEY))
    result = await _fal_or_skip(provider.generate_image(
        ImageRequest(prompt="a simple red circle on white background"),
    ))
    assert result.images is not None
    assert len(result.images) > 0


@skip_without_key
@pytest.mark.asyncio
@pytest.mark.timeout(90)
async def test_fal_text_to_speech():
    provider = FalProvider(options=FalOptions(api_key=FAL_API_KEY))
    result = await _fal_or_skip(provider.text_to_speech(
        SpeechRequest(text="Hello world, this is a test."),
    ))
    assert result.audio is not None
    assert len(result.audio) > 0


# ---------------------------------------------------------------------------
# Compute tests (music, transcription, video, raw run, job lifecycle)
# ---------------------------------------------------------------------------


@skip_without_key
@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_fal_generate_music():
    """Generate music and verify the response contains audio data."""
    provider = FalProvider(options=FalOptions(api_key=FAL_API_KEY))
    result = await _fal_or_skip(provider.generate_music(
        MusicRequest(prompt="happy upbeat jingle", duration_seconds=5.0),
    ))
    assert result.audio is not None
    assert len(result.audio) > 0


@skip_without_key
@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_fal_transcribe():
    """Transcribe a short audio clip and verify text is returned."""
    provider = FalProvider(options=FalOptions(api_key=FAL_API_KEY))
    result = await _fal_or_skip(provider.transcribe(
        TranscriptionRequest(audio_url="https://cdn.openai.com/API/docs/audio/alloy.wav"),
    ))
    assert result.text is not None
    assert len(result.text) > 0


@skip_without_key
@pytest.mark.asyncio
@pytest.mark.timeout(600)
async def test_fal_text_to_video():
    """Generate a video from text (slow, ~30-60s)."""
    provider = FalProvider(options=FalOptions(api_key=FAL_API_KEY))
    try:
        result = await _fal_or_skip(provider.text_to_video(
            VideoRequest(prompt="a cat walking"),
        ))
    except Exception as e:
        err = str(e).lower()
        if any(
            marker in err
            for marker in (
                "downstream_service_unavailable",
                "service unavailable",
                "service_unavailable",
            )
        ):
            pytest.skip(f"fal.ai video service transiently unavailable: {e}")
        raise
    assert result.videos is not None
    assert len(result.videos) > 0


@skip_without_key
@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_fal_raw_compute_run():
    """Run a raw compute job via the generic run() method."""
    provider = FalProvider(options=FalOptions(api_key=FAL_API_KEY))
    result = await _fal_or_skip(provider.run(
        ComputeRequest(
            model="fal-ai/flux/schnell",
            input={"prompt": "blue sky", "image_size": "square_hd"},
        ),
    ))
    assert result is not None
    assert result.output is not None


@skip_without_key
@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_fal_job_submit():
    """Submit a job and verify a job handle is returned."""
    provider = FalProvider(options=FalOptions(api_key=FAL_API_KEY))
    job = await _fal_or_skip(provider.submit(
        ComputeRequest(
            model="fal-ai/flux/schnell",
            input={"prompt": "green forest"},
        ),
    ))
    assert job.id is not None
    assert isinstance(job.id, str)
    assert len(job.id) > 0
