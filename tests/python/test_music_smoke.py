"""Smoke tests for the `MusicModel` / `MusicChunk` / `MusicStream` surface.

The construction-free smoke (`test_music_module_smoke`) always runs and
verifies the typed surface is reachable from Python — it requires only
that blazen-py was built with at least one of the
`audio-music-musicgen` / `audio-music-audiogen` / `audio-music-stable-audio`
features.

The live `MusicGen` test is opt-in via `BLAZEN_RUN_MUSIC_TESTS=1` because
it downloads a ~2 GB Hugging Face checkpoint.
"""

import os

import pytest
import blazen as bz

pytestmark = pytest.mark.asyncio


def test_music_module_smoke():
    """Construction-free smoke: types are importable when the music
    features are compiled in."""
    if not hasattr(bz, "MusicChunk"):
        pytest.skip("blazen built without any audio-music feature")
    assert hasattr(bz, "MusicChunk")
    assert hasattr(bz, "MusicStream")
    assert hasattr(bz, "MusicModel")
    assert hasattr(bz, "MusicGenError")
    # MusicGenError is a subclass of ProviderError.
    assert issubclass(bz.MusicGenError, bz.ProviderError)


@pytest.mark.skipif(
    not hasattr(bz, "MusicModel") or os.getenv("BLAZEN_RUN_MUSIC_TESTS") != "1",
    reason="requires audio-music-musicgen feature and BLAZEN_RUN_MUSIC_TESTS=1 "
    "(downloads ~2 GB MusicGen checkpoint from Hugging Face)",
)
async def test_musicgen_smoke():
    """End-to-end live test against the MusicGen small variant."""
    model = bz.MusicModel.musicgen(variant="small")
    assert model.sample_rate == 32_000
    assert model.id == "musicgen-small"
    chunk = await model.generate_music("upbeat synth loop", duration_seconds=1.0)
    assert isinstance(chunk, bz.MusicChunk)
    assert chunk.is_final
    assert chunk.sample_count > 0
    assert chunk.sample_rate == 32_000
