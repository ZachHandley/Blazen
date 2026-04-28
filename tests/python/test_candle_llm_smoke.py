"""Smoke tests for the local candle LLM backend.

These tests verify that the typed PyO3 binding surface for the candle
provider is importable and that the data-carrier types behave correctly
without requiring real model weights on disk or in the HuggingFace cache.

The candle backend is behind a Cargo feature gate. If the wheel was not
built with ``--features candle``, the entire module is skipped.

Tests that require a real model are gated on ``BLAZEN_TEST_CANDLE=1``
plus ``BLAZEN_CANDLE_MODEL_PATH=/path/or/hf-id`` to avoid surprise
multi-GB downloads or load failures in CI.

Run manually:

    BLAZEN_TEST_CANDLE=1 \\
    BLAZEN_CANDLE_MODEL_PATH=meta-llama/Llama-3.2-1B \\
        uv run --no-sync pytest tests/python/test_candle_llm_smoke.py -v
"""

import os

import pytest

# The full candle typed surface lives behind the `candle` Cargo feature.
# If the installed wheel was not built with that feature, skip the entire
# module rather than failing with an ImportError.
try:
    from blazen import (
        CandleInferenceResult,
        CandleLlmOptions,
        CandleLlmProvider,
        ChatMessage,
    )
except ImportError:
    pytest.skip(
        "blazen was not built with the candle feature "
        "(rebuild with: maturin develop -m crates/blazen-py/Cargo.toml --features candle)",
        allow_module_level=True,
    )

CANDLE_ENABLED = os.environ.get("BLAZEN_TEST_CANDLE") == "1"
MODEL_PATH = os.environ.get("BLAZEN_CANDLE_MODEL_PATH")

skip_without_flag = pytest.mark.skipif(
    not CANDLE_ENABLED,
    reason="BLAZEN_TEST_CANDLE=1 not set",
)

skip_without_model = pytest.mark.skipif(
    not MODEL_PATH,
    reason="BLAZEN_CANDLE_MODEL_PATH not set to a model id or local path",
)


# ---------------------------------------------------------------------------
# Pure typed-surface tests (no model required, always run)
# ---------------------------------------------------------------------------


def test_candle_options_typed_kwargs():
    """CandleLlmOptions accepts the documented keyword-only fields."""
    opts = CandleLlmOptions(
        model_id="meta-llama/Llama-3.2-1B",
        device="cpu",
        quantization="q4_k_m",
        revision="main",
        context_length=2048,
        cache_dir="/tmp/blazen-candle-cache",
    )
    assert opts.model_id == "meta-llama/Llama-3.2-1B"
    assert opts.device == "cpu"
    assert opts.quantization == "q4_k_m"
    assert opts.revision == "main"
    assert opts.context_length == 2048
    assert opts.cache_dir == "/tmp/blazen-candle-cache"


def test_candle_options_defaults_are_none():
    """CandleLlmOptions with no kwargs has all-None fields."""
    opts = CandleLlmOptions()
    assert opts.model_id is None
    assert opts.device is None
    assert opts.quantization is None
    assert opts.revision is None
    assert opts.context_length is None
    assert opts.cache_dir is None


def test_candle_options_setters_mutate_inplace():
    """CandleLlmOptions exposes setters for each field."""
    opts = CandleLlmOptions()
    opts.model_id = "meta-llama/Llama-3.2-1B"
    opts.context_length = 4096
    assert opts.model_id == "meta-llama/Llama-3.2-1B"
    assert opts.context_length == 4096


def test_candle_inference_result_constructor():
    """CandleInferenceResult exposes content / token counts / total_time_secs."""
    result = CandleInferenceResult(
        content="hi there",
        prompt_tokens=5,
        completion_tokens=7,
        total_time_secs=1.5,
    )
    assert result.content == "hi there"
    assert result.prompt_tokens == 5
    assert result.completion_tokens == 7
    # Float comparison with tolerance for f32 round-trip.
    assert abs(result.total_time_secs - 1.5) < 1e-5


def test_candle_provider_class_shape():
    """CandleLlmProvider is a class with the expected method surface."""
    # Constructing the provider eagerly loads weights, so we can't
    # instantiate without a real model. But we can introspect the type.
    assert hasattr(CandleLlmProvider, "complete")
    assert hasattr(CandleLlmProvider, "stream")
    assert hasattr(CandleLlmProvider, "load")
    assert hasattr(CandleLlmProvider, "unload")
    assert hasattr(CandleLlmProvider, "is_loaded")
    # model_id is a getter on the provider per crates/blazen-py/src/providers/candle_llm.rs.
    assert hasattr(CandleLlmProvider, "model_id")


# ---------------------------------------------------------------------------
# Live tests gated behind BLAZEN_TEST_CANDLE=1 + a real model id / path
# ---------------------------------------------------------------------------


@skip_without_flag
@skip_without_model
def test_candle_provider_constructs_with_real_model():
    """CandleLlmProvider(options=opts) succeeds against a real model."""
    opts = CandleLlmOptions(model_id=MODEL_PATH)
    provider = CandleLlmProvider(options=opts)
    # model_id should be a non-empty string identifying the loaded model.
    assert isinstance(provider.model_id, str)
    assert len(provider.model_id) > 0


@skip_without_flag
@skip_without_model
@pytest.mark.asyncio
async def test_candle_provider_completes_prompt():
    """A real candle model produces non-empty content for a simple prompt."""
    opts = CandleLlmOptions(model_id=MODEL_PATH)
    provider = CandleLlmProvider(options=opts)

    response = await provider.complete([
        ChatMessage.user("What is 2+2? Answer with just the number."),
    ])

    assert response.content is not None
    assert len(response.content) > 0
