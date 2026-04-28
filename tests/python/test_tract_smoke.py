"""Smoke tests for the local tract (pure-Rust ONNX) embedding backend.

These tests verify that the typed PyO3 binding surface for the tract
provider is importable and that the data-carrier types behave correctly
without requiring a real model download.

The tract backend is behind a Cargo feature gate. If the wheel was not
built with ``--features tract``, the entire module is skipped.

Tract is a drop-in equivalent of :class:`FastEmbedModel` for environments
where the prebuilt ONNX Runtime binaries that back fastembed cannot be
linked (musl-libc Linux distributions, sandboxed targets, ...). It loads
the same fastembed model catalog via tract-onnx instead.

Tests that require a real model download are gated on
``BLAZEN_TEST_TRACT=1`` (and optionally ``BLAZEN_TRACT_MODEL_NAME`` to
override the default ``BGESmallENV15`` variant) to avoid surprise
multi-MB downloads in CI.

Run manually:

    BLAZEN_TEST_TRACT=1 \\
        uv run --no-sync pytest tests/python/test_tract_smoke.py -v
"""

import os

import pytest

# The full tract typed surface lives behind the `tract` Cargo feature.
# If the installed wheel was not built with that feature, skip the entire
# module rather than failing with an ImportError.
try:
    from blazen import (
        TractEmbedModel,
        TractError,
        TractOptions,
        TractResponse,
    )
except ImportError:
    pytest.skip(
        "blazen was not built with the tract feature "
        "(rebuild with: maturin develop -m crates/blazen-py/Cargo.toml --features tract)",
        allow_module_level=True,
    )

TRACT_ENABLED = os.environ.get("BLAZEN_TEST_TRACT") == "1"
MODEL_NAME = os.environ.get("BLAZEN_TRACT_MODEL_NAME")

skip_without_flag = pytest.mark.skipif(
    not TRACT_ENABLED,
    reason="BLAZEN_TEST_TRACT=1 not set",
)


# ---------------------------------------------------------------------------
# Pure typed-surface tests (no model required, always run)
# ---------------------------------------------------------------------------


def test_tract_options_typed():
    """TractOptions accepts the documented keyword-only fields and round-trips."""
    opts = TractOptions(
        model_name="BGESmallENV15",
        cache_dir="/tmp/blazen-tract-cache",
        max_batch_size=32,
        show_download_progress=True,
    )
    assert opts.model_name == "BGESmallENV15"
    assert opts.cache_dir == "/tmp/blazen-tract-cache"
    assert opts.max_batch_size == 32
    assert opts.show_download_progress is True


def test_tract_options_defaults_are_none():
    """TractOptions with no kwargs has all-None fields."""
    opts = TractOptions()
    assert opts.model_name is None
    assert opts.cache_dir is None
    assert opts.max_batch_size is None
    assert opts.show_download_progress is None


def test_tract_options_setters_mutate_inplace():
    """TractOptions exposes setters for each field."""
    opts = TractOptions()
    opts.model_name = "BGESmallENV15"
    opts.max_batch_size = 64
    opts.show_download_progress = False
    opts.cache_dir = "/var/cache/blazen"
    assert opts.model_name == "BGESmallENV15"
    assert opts.max_batch_size == 64
    assert opts.show_download_progress is False
    assert opts.cache_dir == "/var/cache/blazen"


def test_tract_embed_model_class_shape():
    """TractEmbedModel is a class with the expected method/getter surface."""
    # Constructing the model eagerly downloads/loads ONNX weights, so we
    # can't instantiate without a real model. Introspect the type instead.
    assert hasattr(TractEmbedModel, "embed")
    assert hasattr(TractEmbedModel, "model_id")
    assert hasattr(TractEmbedModel, "dimensions")


def test_tract_response_class_shape():
    """TractResponse exposes embeddings + model getters as a frozen class.

    Per crates/blazen-py/src/providers/tract.rs, ``TractResponse`` is a
    ``frozen`` pyclass with no Python-side ``__new__``: it is constructed
    only via ``TractEmbedModel.embed(...)``. Verify the attribute surface
    without instantiating.
    """
    assert hasattr(TractResponse, "embeddings")
    assert hasattr(TractResponse, "model")


def test_tract_error_is_exception():
    """TractError is an exception type subclassing the provider error hierarchy."""
    assert isinstance(TractError, type)
    assert issubclass(TractError, Exception)


# ---------------------------------------------------------------------------
# Live tests gated behind BLAZEN_TEST_TRACT=1 (downloads model on first run)
# ---------------------------------------------------------------------------


@skip_without_flag
def test_tract_embed_model_constructs_with_real_model():
    """TractEmbedModel(options=opts) succeeds and exposes model_id/dimensions."""
    opts = TractOptions(model_name=MODEL_NAME) if MODEL_NAME else TractOptions()
    model = TractEmbedModel(options=opts)
    assert isinstance(model.model_id, str)
    assert len(model.model_id) > 0
    assert isinstance(model.dimensions, int)
    assert model.dimensions > 0


@skip_without_flag
@pytest.mark.asyncio
async def test_tract_embed_returns_typed_response():
    """A real tract model produces a TractResponse with embeddings + model id."""
    opts = TractOptions(model_name=MODEL_NAME) if MODEL_NAME else TractOptions()
    model = TractEmbedModel(options=opts)

    response = await model.embed(["hello", "world"])

    # TractResponse round-trips its data-carrier surface.
    assert isinstance(response, TractResponse)
    assert isinstance(response.model, str)
    assert len(response.model) > 0

    assert len(response.embeddings) == 2
    assert all(len(v) == model.dimensions for v in response.embeddings)
    # Different inputs should produce different vectors.
    assert response.embeddings[0] != response.embeddings[1]


@skip_without_flag
@pytest.mark.asyncio
async def test_tract_embed_vector_dimensionality():
    """Each embedding vector has exactly model.dimensions elements."""
    opts = TractOptions(model_name=MODEL_NAME) if MODEL_NAME else TractOptions()
    model = TractEmbedModel(options=opts)

    response = await model.embed(["a single sentence"])

    assert len(response.embeddings) == 1
    assert len(response.embeddings[0]) == model.dimensions
