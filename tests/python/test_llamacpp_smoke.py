"""Smoke tests for the local llama.cpp LLM backend.

These tests verify that the typed PyO3 binding surface for the llama.cpp
provider is importable and that the data-carrier types behave correctly
without requiring a real GGUF model file on disk.

The llama.cpp backend is behind a Cargo feature gate. If the wheel was not
built with ``--features llamacpp``, the entire module is skipped.

Tests that require a real model file are gated on
``BLAZEN_TEST_LLAMACPP=1`` plus ``BLAZEN_LLAMACPP_MODEL_PATH=/path/to.gguf``
to avoid surprise multi-GB downloads or load failures in CI.

Run manually:

    BLAZEN_TEST_LLAMACPP=1 \\
    BLAZEN_LLAMACPP_MODEL_PATH=/models/llama-3.2-1b-q4_k_m.gguf \\
        uv run --no-sync pytest tests/python/test_llamacpp_smoke.py -v
"""

import os

import pytest

# The full llama.cpp typed surface lives behind the `llamacpp` Cargo feature.
# If the installed wheel was not built with that feature, skip the entire
# module rather than failing with an ImportError.
try:
    from blazen import (
        ChatMessage,
        LlamaCppChatMessageInput,
        LlamaCppChatRole,
        LlamaCppInferenceChunk,
        LlamaCppInferenceChunkStream,
        LlamaCppInferenceResult,
        LlamaCppInferenceUsage,
        LlamaCppOptions,
        LlamaCppProvider,
    )
except ImportError:
    pytest.skip(
        "blazen was not built with the llamacpp feature "
        "(rebuild with: maturin develop -m crates/blazen-py/Cargo.toml --features llamacpp)",
        allow_module_level=True,
    )

LLAMACPP_ENABLED = os.environ.get("BLAZEN_TEST_LLAMACPP") == "1"
MODEL_PATH = os.environ.get("BLAZEN_LLAMACPP_MODEL_PATH")

skip_without_flag = pytest.mark.skipif(
    not LLAMACPP_ENABLED,
    reason="BLAZEN_TEST_LLAMACPP=1 not set",
)

skip_without_model = pytest.mark.skipif(
    not MODEL_PATH or not os.path.exists(MODEL_PATH or ""),
    reason="BLAZEN_LLAMACPP_MODEL_PATH not set to an existing GGUF file",
)


# ---------------------------------------------------------------------------
# Pure typed-surface tests (no model required, always run)
# ---------------------------------------------------------------------------


def test_llamacpp_options_typed_kwargs():
    """LlamaCppOptions accepts the documented keyword-only fields."""
    opts = LlamaCppOptions(
        model_path="/dev/null",
        device="cpu",
        quantization="q4_k_m",
        context_length=2048,
        n_gpu_layers=0,
        cache_dir="/tmp/blazen-llamacpp-cache",
    )
    assert opts.model_path == "/dev/null"
    assert opts.device == "cpu"
    assert opts.quantization == "q4_k_m"
    assert opts.context_length == 2048
    assert opts.n_gpu_layers == 0
    assert opts.cache_dir == "/tmp/blazen-llamacpp-cache"


def test_llamacpp_options_defaults_are_none():
    """LlamaCppOptions with no kwargs has all-None fields."""
    opts = LlamaCppOptions()
    assert opts.model_path is None
    assert opts.device is None
    assert opts.quantization is None
    assert opts.context_length is None
    assert opts.n_gpu_layers is None
    assert opts.cache_dir is None


def test_llamacpp_options_setters_mutate_inplace():
    """LlamaCppOptions exposes setters for each field."""
    opts = LlamaCppOptions()
    opts.model_path = "/models/foo.gguf"
    opts.context_length = 4096
    assert opts.model_path == "/models/foo.gguf"
    assert opts.context_length == 4096


def test_llamacpp_chat_role_variants_exist():
    """LlamaCppChatRole exposes the four expected variants."""
    assert LlamaCppChatRole.System != LlamaCppChatRole.User
    assert LlamaCppChatRole.Assistant != LlamaCppChatRole.Tool
    # Round-trip equality.
    assert LlamaCppChatRole.User == LlamaCppChatRole.User


def test_llamacpp_chat_message_input_constructor():
    """LlamaCppChatMessageInput exposes role and text getters."""
    msg = LlamaCppChatMessageInput(
        role=LlamaCppChatRole.User,
        text="Hello, llama.",
    )
    assert msg.role == LlamaCppChatRole.User
    assert msg.text == "Hello, llama."


def test_llamacpp_chat_message_input_create_factory():
    """LlamaCppChatMessageInput.create is a positional staticmethod factory."""
    msg = LlamaCppChatMessageInput.create(LlamaCppChatRole.System, "be terse")
    assert msg.role == LlamaCppChatRole.System
    assert msg.text == "be terse"


def test_llamacpp_inference_usage_constructor():
    """LlamaCppInferenceUsage stores token counts and timing."""
    usage = LlamaCppInferenceUsage(
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
        total_time_sec=1.5,
    )
    assert usage.prompt_tokens == 10
    assert usage.completion_tokens == 20
    assert usage.total_tokens == 30
    # Float comparison with tolerance for f32 round-trip.
    assert abs(usage.total_time_sec - 1.5) < 1e-5


def test_llamacpp_inference_usage_defaults_are_zero():
    """LlamaCppInferenceUsage with no args is all-zero."""
    usage = LlamaCppInferenceUsage()
    assert usage.prompt_tokens == 0
    assert usage.completion_tokens == 0
    assert usage.total_tokens == 0
    assert usage.total_time_sec == 0.0


def test_llamacpp_inference_result_constructor():
    """LlamaCppInferenceResult exposes content/finish_reason/model/usage."""
    usage = LlamaCppInferenceUsage(prompt_tokens=5, completion_tokens=7, total_tokens=12)
    result = LlamaCppInferenceResult(
        finish_reason="stop",
        model="llama-3.2-1b",
        usage=usage,
        content="hi there",
    )
    assert result.content == "hi there"
    assert result.finish_reason == "stop"
    assert result.model == "llama-3.2-1b"
    # Usage round-trips through the result.
    assert result.usage.total_tokens == 12


def test_llamacpp_inference_result_content_optional():
    """LlamaCppInferenceResult.content may be None when omitted."""
    usage = LlamaCppInferenceUsage()
    result = LlamaCppInferenceResult(
        finish_reason="length",
        model="test-model",
        usage=usage,
    )
    assert result.content is None
    assert result.finish_reason == "length"


def test_llamacpp_inference_chunk_constructor():
    """LlamaCppInferenceChunk exposes optional delta/finish_reason."""
    chunk = LlamaCppInferenceChunk(delta="tok", finish_reason=None)
    assert chunk.delta == "tok"
    assert chunk.finish_reason is None

    final = LlamaCppInferenceChunk(delta=None, finish_reason="stop")
    assert final.delta is None
    assert final.finish_reason == "stop"


def test_llamacpp_inference_chunk_stream_class_exists():
    """LlamaCppInferenceChunkStream is exported as a class with the async-iter protocol."""
    # We can't construct one directly without a live engine, but we can
    # confirm the class is importable and exposes the async iterator dunder.
    assert hasattr(LlamaCppInferenceChunkStream, "__aiter__")
    assert hasattr(LlamaCppInferenceChunkStream, "__anext__")


def test_llamacpp_provider_class_shape():
    """LlamaCppProvider is a class with the expected method surface."""
    # Constructing the provider eagerly loads the model, so we can't
    # instantiate without a real GGUF file. But we can introspect the type.
    assert hasattr(LlamaCppProvider, "complete")
    assert hasattr(LlamaCppProvider, "stream")
    assert hasattr(LlamaCppProvider, "load")
    assert hasattr(LlamaCppProvider, "unload")
    assert hasattr(LlamaCppProvider, "is_loaded")
    # model_id is a getter on the provider per crates/blazen-py/src/providers/llamacpp.rs.
    assert hasattr(LlamaCppProvider, "model_id")


# ---------------------------------------------------------------------------
# Live tests gated behind BLAZEN_TEST_LLAMACPP=1 + a real model path
# ---------------------------------------------------------------------------


@skip_without_flag
@skip_without_model
def test_llamacpp_provider_constructs_with_real_model():
    """LlamaCppProvider(options=opts) succeeds against a real GGUF file."""
    opts = LlamaCppOptions(model_path=MODEL_PATH)
    provider = LlamaCppProvider(options=opts)
    # model_id should be a non-empty string identifying the loaded model.
    assert isinstance(provider.model_id, str)
    assert len(provider.model_id) > 0


@skip_without_flag
@skip_without_model
@pytest.mark.asyncio
async def test_llamacpp_provider_completes_prompt():
    """A real llama.cpp model produces non-empty content for a simple prompt."""
    opts = LlamaCppOptions(model_path=MODEL_PATH)
    provider = LlamaCppProvider(options=opts)

    response = await provider.complete([
        ChatMessage.user("What is 2+2? Answer with just the number."),
    ])

    assert response.content is not None
    assert len(response.content) > 0
