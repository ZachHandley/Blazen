"""Smoke tests for the embed local embedding backend.

These tests require a model download on first run (~33 MB for the default
BAAI/bge-small-en-v1.5 model).  Gated on the BLAZEN_TEST_EMBED=1
environment variable to avoid unexpected downloads in CI.

The embed backend is behind a Cargo feature gate.  If the wheel was not
built with ``--features embed``, the entire module is skipped.

Run manually:
    BLAZEN_TEST_EMBED=1 uv run pytest tests/python/test_embed_smoke.py -v
"""

import os

import pytest

from blazen import EmbeddingModel

# EmbedOptions lives behind the `embed` Cargo feature gate.
# If the installed wheel was not built with that feature, skip the
# entire module rather than failing with an ImportError.
try:
    from blazen import EmbedOptions
except ImportError:
    pytest.skip(
        "blazen was not built with the embed feature "
        "(rebuild with: maturin develop -m crates/blazen-py/Cargo.toml --features embed)",
        allow_module_level=True,
    )

skip_without_flag = pytest.mark.skipif(
    not os.environ.get("BLAZEN_TEST_EMBED"),
    reason="BLAZEN_TEST_EMBED not set",
)


@skip_without_flag
@pytest.mark.asyncio
async def test_embed_default_model():
    """Default embed model embeds texts and reports correct dimensions."""
    model = EmbeddingModel.local()
    result = await model.embed(["hello", "world"])

    assert len(result.embeddings) == 2
    assert all(len(v) > 0 for v in result.embeddings)
    assert model.dimensions > 0
    assert model.model_id is not None


@skip_without_flag
@pytest.mark.asyncio
async def test_embed_vector_dimensionality():
    """Each embedding vector has exactly model.dimensions elements."""
    model = EmbeddingModel.local()
    result = await model.embed(["a single sentence"])

    assert len(result.embeddings) == 1
    assert len(result.embeddings[0]) == model.dimensions


@skip_without_flag
@pytest.mark.asyncio
async def test_embed_with_options():
    """EmbedOptions are accepted without error."""
    opts = EmbedOptions(show_download_progress=True)
    model = EmbeddingModel.local(options=opts)
    result = await model.embed(["test"])

    assert len(result.embeddings) == 1
    assert len(result.embeddings[0]) > 0


@skip_without_flag
@pytest.mark.asyncio
async def test_embed_batch_ordering():
    """Embedding order matches input order (different texts give different vectors)."""
    model = EmbeddingModel.local()
    texts = ["the cat sat on the mat", "quantum mechanics is fascinating"]
    result = await model.embed(texts)

    assert len(result.embeddings) == 2
    # The two vectors should not be identical (different semantic content).
    assert result.embeddings[0] != result.embeddings[1]
