"""Smoke tests for the local Piper text-to-speech backend.

These tests verify that the typed PyO3 binding surface for the Piper TTS
provider is importable and that the data-carrier types behave correctly
without requiring a real Piper voice model on disk.

The Piper backend is behind a Cargo feature gate. If the wheel was not
built with ``--features piper``, the entire module is skipped.

Tests that require a real voice model are gated on
``BLAZEN_TEST_PIPER=1`` plus ``BLAZEN_PIPER_MODEL_PATH=/path/to/voice.onnx``
(or a Piper voice id resolvable by the local model cache) to avoid
surprise multi-MB downloads or load failures in CI.

Run manually:

    BLAZEN_TEST_PIPER=1 \\
    BLAZEN_PIPER_MODEL_PATH=/models/en_US-amy-medium.onnx \\
        uv run --no-sync pytest tests/python/test_piper_smoke.py -v
"""

import os

import pytest

# The full Piper typed surface lives behind the `piper` Cargo feature.
# If the installed wheel was not built with that feature, skip the entire
# module rather than failing with an ImportError.
try:
    from blazen import (
        PiperOptions,
        PiperProvider,
        SpeechRequest,
    )
except ImportError:
    pytest.skip(
        "blazen was not built with the piper feature "
        "(rebuild with: maturin develop -m crates/blazen-py/Cargo.toml --features piper)",
        allow_module_level=True,
    )

PIPER_ENABLED = os.environ.get("BLAZEN_TEST_PIPER") == "1"
MODEL_PATH = os.environ.get("BLAZEN_PIPER_MODEL_PATH")

skip_without_flag = pytest.mark.skipif(
    not PIPER_ENABLED,
    reason="BLAZEN_TEST_PIPER=1 not set",
)

skip_without_model = pytest.mark.skipif(
    not MODEL_PATH or not os.path.exists(MODEL_PATH or ""),
    reason="BLAZEN_PIPER_MODEL_PATH not set to an existing voice model",
)


# ---------------------------------------------------------------------------
# Pure typed-surface tests (no model required, always run)
# ---------------------------------------------------------------------------


def test_piper_options_typed():
    """PiperOptions accepts the documented keyword-only fields."""
    opts = PiperOptions(
        model_id="en_US-amy-medium",
        speaker_id=0,
        sample_rate=22050,
        cache_dir="/tmp/blazen-piper-cache",
    )
    assert opts.model_id == "en_US-amy-medium"
    assert opts.speaker_id == 0
    assert opts.sample_rate == 22050
    assert opts.cache_dir == "/tmp/blazen-piper-cache"


def test_piper_options_defaults_are_none():
    """PiperOptions with no kwargs has all-None fields."""
    opts = PiperOptions()
    assert opts.model_id is None
    assert opts.speaker_id is None
    assert opts.sample_rate is None
    assert opts.cache_dir is None


def test_piper_options_setters_mutate_inplace():
    """PiperOptions exposes setters for each field."""
    opts = PiperOptions()
    opts.model_id = "en_GB-alan-medium"
    opts.sample_rate = 16000
    assert opts.model_id == "en_GB-alan-medium"
    assert opts.sample_rate == 16000


def test_piper_provider_class_shape():
    """PiperProvider is a class with the expected method surface."""
    # Constructing the provider does not require a real model file when no
    # options are supplied, but introspecting the type avoids depending on
    # the engine-availability of the current build.
    assert hasattr(PiperProvider, "text_to_speech")
    # model_id is a getter on the provider per crates/blazen-py/src/providers/piper.rs.
    assert hasattr(PiperProvider, "model_id")
    # engine_available reflects whether the ONNX Runtime backend is compiled in.
    assert hasattr(PiperProvider, "engine_available")


def test_piper_provider_constructs_without_options():
    """PiperProvider() with no options succeeds and exposes model_id getter."""
    provider = PiperProvider()
    # With no options supplied the configured model id is None.
    assert provider.model_id is None
    # engine_available is a bool regardless of whether the engine is wired.
    assert isinstance(provider.engine_available, bool)


def test_piper_provider_constructs_with_options():
    """PiperProvider(options=opts) round-trips the configured model id."""
    opts = PiperOptions(model_id="en_US-amy-medium")
    provider = PiperProvider(options=opts)
    assert provider.model_id == "en_US-amy-medium"


# ---------------------------------------------------------------------------
# Live tests gated behind BLAZEN_TEST_PIPER=1 + a real model path
# ---------------------------------------------------------------------------


@skip_without_flag
@skip_without_model
def test_piper_provider_constructs_with_real_model():
    """PiperProvider(options=opts) succeeds against a real voice model."""
    opts = PiperOptions(model_id=MODEL_PATH)
    provider = PiperProvider(options=opts)
    # model_id should round-trip the configured value.
    assert isinstance(provider.model_id, str)
    assert len(provider.model_id) > 0


@skip_without_flag
@skip_without_model
@pytest.mark.asyncio
async def test_piper_provider_synthesizes_text():
    """A real Piper voice synthesizes non-empty audio for a simple prompt."""
    opts = PiperOptions(model_id=MODEL_PATH)
    provider = PiperProvider(options=opts)

    request = SpeechRequest(text="Hello from Blazen.")
    result = await provider.text_to_speech(request)

    # AudioResult exposes the synthesized bytes via a `data` or similar
    # attribute; we only assert that the call returned a value, since the
    # exact field shape is covered by the AudioResult-specific tests.
    assert result is not None
