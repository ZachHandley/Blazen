"""Per-provider LLM completion smoke tests.

Each test is gated on its own API key environment variable
and can be skipped independently.
"""

import os

import pytest

from blazen import ChatMessage, CompletionModel, CompletionOptions

# ---------------------------------------------------------------------------
# API keys
# ---------------------------------------------------------------------------

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY")
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
FIREWORKS_API_KEY = os.environ.get("FIREWORKS_API_KEY")
PERPLEXITY_API_KEY = os.environ.get("PERPLEXITY_API_KEY")
XAI_API_KEY = os.environ.get("XAI_API_KEY")
COHERE_API_KEY = os.environ.get("COHERE_API_KEY")

# ---------------------------------------------------------------------------
# Skip markers
# ---------------------------------------------------------------------------

skip_without_openai = pytest.mark.skipif(not OPENAI_API_KEY, reason="OPENAI_API_KEY not set")
skip_without_anthropic = pytest.mark.skipif(not ANTHROPIC_API_KEY, reason="ANTHROPIC_API_KEY not set")
skip_without_gemini = pytest.mark.skipif(not GEMINI_API_KEY, reason="GEMINI_API_KEY not set")
skip_without_groq = pytest.mark.skipif(not GROQ_API_KEY, reason="GROQ_API_KEY not set")
skip_without_together = pytest.mark.skipif(not TOGETHER_API_KEY, reason="TOGETHER_API_KEY not set")
skip_without_mistral = pytest.mark.skipif(not MISTRAL_API_KEY, reason="MISTRAL_API_KEY not set")
skip_without_deepseek = pytest.mark.skipif(not DEEPSEEK_API_KEY, reason="DEEPSEEK_API_KEY not set")
skip_without_fireworks = pytest.mark.skipif(not FIREWORKS_API_KEY, reason="FIREWORKS_API_KEY not set")
skip_without_perplexity = pytest.mark.skipif(not PERPLEXITY_API_KEY, reason="PERPLEXITY_API_KEY not set")
skip_without_xai = pytest.mark.skipif(not XAI_API_KEY, reason="XAI_API_KEY not set")
skip_without_cohere = pytest.mark.skipif(not COHERE_API_KEY, reason="COHERE_API_KEY not set")

# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

PROMPT = [ChatMessage.user("What is 2+2? Reply with just the number.")]


async def _assert_completion(model: CompletionModel) -> None:
    """Send a simple completion and verify the response."""
    response = await model.complete(PROMPT, CompletionOptions(max_tokens=10))
    assert response.content is not None
    assert "4" in response.content
    assert response.model is not None


# ---------------------------------------------------------------------------
# Provider tests
# ---------------------------------------------------------------------------


@skip_without_openai
@pytest.mark.asyncio
async def test_openai_completion():
    model = CompletionModel.openai(OPENAI_API_KEY)
    await _assert_completion(model)


@skip_without_anthropic
@pytest.mark.asyncio
async def test_anthropic_completion():
    model = CompletionModel.anthropic(ANTHROPIC_API_KEY)
    await _assert_completion(model)


@skip_without_gemini
@pytest.mark.asyncio
async def test_gemini_completion():
    model = CompletionModel.gemini(GEMINI_API_KEY)
    await _assert_completion(model)


@skip_without_groq
@pytest.mark.asyncio
async def test_groq_completion():
    model = CompletionModel.groq(GROQ_API_KEY)
    await _assert_completion(model)


@skip_without_together
@pytest.mark.asyncio
async def test_together_completion():
    model = CompletionModel.together(TOGETHER_API_KEY)
    await _assert_completion(model)


@skip_without_mistral
@pytest.mark.asyncio
async def test_mistral_completion():
    model = CompletionModel.mistral(MISTRAL_API_KEY)
    await _assert_completion(model)


@skip_without_deepseek
@pytest.mark.asyncio
async def test_deepseek_completion():
    model = CompletionModel.deepseek(DEEPSEEK_API_KEY)
    await _assert_completion(model)


@skip_without_fireworks
@pytest.mark.asyncio
async def test_fireworks_completion():
    model = CompletionModel.fireworks(FIREWORKS_API_KEY)
    await _assert_completion(model)


@skip_without_perplexity
@pytest.mark.asyncio
async def test_perplexity_completion():
    model = CompletionModel.perplexity(PERPLEXITY_API_KEY)
    await _assert_completion(model)


@skip_without_xai
@pytest.mark.asyncio
async def test_xai_completion():
    model = CompletionModel.xai(XAI_API_KEY)
    await _assert_completion(model)


@skip_without_cohere
@pytest.mark.asyncio
async def test_cohere_completion():
    model = CompletionModel.cohere(COHERE_API_KEY)
    await _assert_completion(model)
