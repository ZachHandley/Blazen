"""fal.ai compute smoke tests: image, video, audio, transcription, job lifecycle.

Gated on the FAL_API_KEY environment variable.
"""

import os

import pytest

from blazen import FalProvider

FAL_API_KEY = os.environ.get("FAL_API_KEY")

skip_without_key = pytest.mark.skipif(
    not FAL_API_KEY,
    reason="FAL_API_KEY not set",
)


# ---------------------------------------------------------------------------
# Image generation
# ---------------------------------------------------------------------------


@skip_without_key
@pytest.mark.asyncio
async def test_fal_generate_image():
    """Generate an image and verify the response contains image data."""
    provider = FalProvider(api_key=FAL_API_KEY)
    result = await provider.generate_image(
        {"prompt": "a red circle on white background"}
    )

    assert "images" in result
    assert len(result["images"]) > 0


# ---------------------------------------------------------------------------
# Text-to-speech
# ---------------------------------------------------------------------------


@skip_without_key
@pytest.mark.asyncio
async def test_fal_text_to_speech():
    """Synthesize speech and verify the response contains audio data."""
    provider = FalProvider(api_key=FAL_API_KEY)
    result = await provider.text_to_speech(
        {"text": "Hello world."}
    )

    assert "audio" in result
    assert len(result["audio"]) > 0


# ---------------------------------------------------------------------------
# Music generation
# ---------------------------------------------------------------------------


@skip_without_key
@pytest.mark.asyncio
async def test_fal_generate_music():
    """Generate music and verify the response contains audio data."""
    provider = FalProvider(api_key=FAL_API_KEY)
    result = await provider.generate_music(
        {"prompt": "happy upbeat jingle", "duration_seconds": 5.0}
    )

    assert "audio" in result
    assert len(result["audio"]) > 0


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------


@skip_without_key
@pytest.mark.asyncio
async def test_fal_transcribe():
    """Transcribe a short audio clip and verify text is returned."""
    provider = FalProvider(api_key=FAL_API_KEY)
    result = await provider.transcribe(
        {"audio_url": "https://cdn.openai.com/API/docs/audio/alloy.wav"}
    )

    assert "text" in result
    assert len(result["text"]) > 0


# ---------------------------------------------------------------------------
# Text-to-video (slow -- extended timeout)
# ---------------------------------------------------------------------------


@skip_without_key
@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_fal_text_to_video():
    """Generate a video from text (slow, ~30-60s)."""
    provider = FalProvider(api_key=FAL_API_KEY)
    try:
        result = await provider.text_to_video(
            {"prompt": "a cat walking"}
        )
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

    assert "videos" in result
    assert len(result["videos"]) > 0


# ---------------------------------------------------------------------------
# Raw compute run
# ---------------------------------------------------------------------------


@skip_without_key
@pytest.mark.asyncio
async def test_fal_raw_compute_run():
    """Run a raw compute job via the generic run() method."""
    provider = FalProvider(api_key=FAL_API_KEY)
    result = await provider.run(
        model="fal-ai/flux/schnell",
        input={"prompt": "blue sky", "image_size": "square_hd"},
    )

    assert result is not None
    assert "output" in result or "images" in result


# ---------------------------------------------------------------------------
# Job lifecycle: submit + status
# ---------------------------------------------------------------------------


@skip_without_key
@pytest.mark.asyncio
async def test_fal_job_submit():
    """Submit a job and verify a job handle is returned."""
    provider = FalProvider(api_key=FAL_API_KEY)

    job = await provider.submit(
        model="fal-ai/flux/schnell",
        input={"prompt": "green forest"},
    )

    assert "id" in job
    assert isinstance(job["id"], str)
    assert len(job["id"]) > 0
