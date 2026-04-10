"""Smoke tests for the mistral.rs local LLM backend.

These tests require a model download on first run (several GB for even a small
quantised model).  Gated on the BLAZEN_TEST_MISTRALRS=1 environment variable
to avoid unexpected downloads in CI.

The mistral.rs backend is behind a Cargo feature gate.  If the wheel was not
built with ``--features mistralrs``, the entire module is skipped.

Run manually:
    BLAZEN_TEST_MISTRALRS=1 uv run pytest tests/python/test_mistralrs_smoke.py -v
"""

import os

import pytest

from blazen import ChatMessage, CompletionModel

# MistralRsOptions lives behind the `mistralrs` Cargo feature gate.
# If the installed wheel was not built with that feature, skip the
# entire module rather than failing with an ImportError.
try:
    from blazen import MistralRsOptions
except ImportError:
    pytest.skip(
        "blazen was not built with the mistralrs feature "
        "(rebuild with: maturin develop -m crates/blazen-py/Cargo.toml --features mistralrs)",
        allow_module_level=True,
    )

MODEL_ID = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"

skip_without_flag = pytest.mark.skipif(
    not os.environ.get("BLAZEN_TEST_MISTRALRS"),
    reason="BLAZEN_TEST_MISTRALRS not set",
)


@skip_without_flag
@pytest.mark.asyncio
async def test_mistralrs_complete():
    """mistral.rs model completes a prompt and returns non-empty content."""
    opts = MistralRsOptions(MODEL_ID)
    model = CompletionModel.mistralrs(options=opts)

    response = await model.complete([
        ChatMessage.user("What is 2+2? Answer with just the number."),
    ])

    assert response.content is not None
    assert len(response.content) > 0


@skip_without_flag
@pytest.mark.asyncio
async def test_mistralrs_model_id():
    """MistralRsOptions exposes model_id correctly."""
    opts = MistralRsOptions(MODEL_ID)
    assert opts.model_id == MODEL_ID

    model = CompletionModel.mistralrs(options=opts)
    assert model.model_id is not None
    assert len(model.model_id) > 0


@skip_without_flag
@pytest.mark.asyncio
async def test_mistralrs_with_system_message():
    """mistral.rs model handles system + user message pairs."""
    opts = MistralRsOptions(MODEL_ID)
    model = CompletionModel.mistralrs(options=opts)

    response = await model.complete([
        ChatMessage.system("You are a helpful assistant. Be concise."),
        ChatMessage.user("What is the capital of France?"),
    ])

    assert response.content is not None
    assert len(response.content) > 0
