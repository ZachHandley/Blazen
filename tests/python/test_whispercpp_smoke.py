"""Smoke tests for the whisper.cpp local transcription backend.

These tests require a whisper model download on first run (~150MB for
the default Base variant, up to ~3GB for LargeV3) and a user-supplied
16 kHz mono WAV file.  Gated on two environment variables to avoid
surprises in CI:

    BLAZEN_TEST_WHISPERCPP=1                  # run the tests
    BLAZEN_WHISPER_AUDIO_PATH=/path/to/x.wav  # audio file to transcribe

The whisper.cpp backend is behind a Cargo feature gate.  If the wheel
was not built with ``--features whispercpp``, the entire module is
skipped.

Convert a source audio file to the required format with:

    ffmpeg -i input.mp3 -ar 16000 -ac 1 -c:a pcm_s16le output.wav

Run manually:

    BLAZEN_TEST_WHISPERCPP=1 \\
    BLAZEN_WHISPER_AUDIO_PATH=/path/to/audio.wav \\
        uv run pytest tests/python/test_whispercpp_smoke.py -v
"""

import os

import pytest

from blazen import Transcription, TranscriptionRequest

# WhisperOptions/WhisperModel live behind the `whispercpp` Cargo feature
# gate.  If the installed wheel was not built with that feature, skip
# the entire module rather than failing with an ImportError.
try:
    from blazen import WhisperModel, WhisperOptions
except ImportError:
    pytest.skip(
        "blazen was not built with the whispercpp feature "
        "(rebuild with: maturin develop -m crates/blazen-py/Cargo.toml --features whispercpp)",
        allow_module_level=True,
    )

skip_without_flag = pytest.mark.skipif(
    not os.environ.get("BLAZEN_TEST_WHISPERCPP"),
    reason="BLAZEN_TEST_WHISPERCPP not set",
)

AUDIO_PATH = os.environ.get("BLAZEN_WHISPER_AUDIO_PATH")

skip_without_audio = pytest.mark.skipif(
    not AUDIO_PATH or not os.path.exists(AUDIO_PATH or ""),
    reason="BLAZEN_WHISPER_AUDIO_PATH not set to an existing 16 kHz mono WAV file",
)


@skip_without_flag
def test_whispercpp_options_roundtrip():
    """WhisperOptions accepts the expected keyword arguments."""
    opts = WhisperOptions(
        model=WhisperModel.Base,
        device="cpu",
        language="en",
    )
    assert opts.language == "en"
    assert opts.device == "cpu"


@skip_without_flag
def test_whispercpp_factory_constructs_provider():
    """Transcription.whispercpp builds a provider with defaults."""
    opts = WhisperOptions(model=WhisperModel.Base)
    transcriber = Transcription.whispercpp(options=opts)
    assert transcriber.provider_id == "whispercpp"


@skip_without_flag
def test_transcription_request_from_file_sets_no_url():
    """TranscriptionRequest.from_file produces a request with an empty audio_url."""
    req = TranscriptionRequest.from_file("/tmp/fake.wav")
    # When constructed from a local file, audio_source carries the path
    # and audio_url is left empty.
    assert req.audio_url == ""


@skip_without_flag
@skip_without_audio
@pytest.mark.asyncio
async def test_whispercpp_transcribe_local_file():
    """whisper.cpp transcribes a local WAV file to non-empty text."""
    opts = WhisperOptions(model=WhisperModel.Base)
    transcriber = Transcription.whispercpp(options=opts)

    result = await transcriber.transcribe(
        TranscriptionRequest.from_file(AUDIO_PATH),
    )

    assert result.text is not None
    assert len(result.text) > 0
    # The backend should produce at least one aligned segment.
    assert len(result.segments) >= 1
    # Every segment should have monotonically ordered timestamps.
    for segment in result.segments:
        assert segment.end >= segment.start
        assert segment.text is not None


@skip_without_flag
@skip_without_audio
@pytest.mark.asyncio
async def test_whispercpp_transcribe_with_language_hint():
    """Passing a language hint to WhisperOptions is respected."""
    opts = WhisperOptions(model=WhisperModel.Base, language="en")
    transcriber = Transcription.whispercpp(options=opts)

    result = await transcriber.transcribe(
        TranscriptionRequest.from_file(AUDIO_PATH),
    )

    assert result.text is not None
    assert result.language is not None
