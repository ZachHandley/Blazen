"""Smoke tests for the local diffusion-rs image generation backend.

These tests verify that the typed PyO3 binding surface for the diffusion
provider is importable and that the data-carrier types behave correctly
without requiring a real diffusion model on disk.

The diffusion backend is behind a Cargo feature gate. If the wheel was not
built with ``--features diffusion``, the entire module is skipped.

Tests that require a real model are gated on
``BLAZEN_TEST_DIFFUSION=1`` plus ``BLAZEN_DIFFUSION_MODEL_ID=<hf-repo-id>``
to avoid surprise multi-GB downloads or load failures in CI.

Run manually:

    BLAZEN_TEST_DIFFUSION=1 \\
    BLAZEN_DIFFUSION_MODEL_ID=stabilityai/stable-diffusion-2-1 \\
        uv run --no-sync pytest tests/python/test_diffusion_smoke.py -v
"""

import os

import pytest

# The full diffusion typed surface lives behind the `diffusion` Cargo feature.
# If the installed wheel was not built with that feature, skip the entire
# module rather than failing with an ImportError.
try:
    from blazen import (
        DiffusionOptions,
        DiffusionProvider,
        DiffusionScheduler,
        ImageRequest,
    )
except ImportError:
    pytest.skip(
        "blazen was not built with the diffusion feature "
        "(rebuild with: maturin develop -m crates/blazen-py/Cargo.toml --features diffusion)",
        allow_module_level=True,
    )

DIFFUSION_ENABLED = os.environ.get("BLAZEN_TEST_DIFFUSION") == "1"
MODEL_ID = os.environ.get("BLAZEN_DIFFUSION_MODEL_ID")

skip_without_flag = pytest.mark.skipif(
    not DIFFUSION_ENABLED,
    reason="BLAZEN_TEST_DIFFUSION=1 not set",
)

skip_without_model = pytest.mark.skipif(
    not MODEL_ID,
    reason="BLAZEN_DIFFUSION_MODEL_ID not set to a HuggingFace repo id",
)


# ---------------------------------------------------------------------------
# Pure typed-surface tests (no model required, always run)
# ---------------------------------------------------------------------------


def test_diffusion_options_typed_kwargs():
    """DiffusionOptions accepts the documented keyword-only fields."""
    opts = DiffusionOptions(
        model_id="stabilityai/stable-diffusion-2-1",
        device="cpu",
        width=768,
        height=768,
        num_inference_steps=25,
        guidance_scale=7.5,
        scheduler=DiffusionScheduler.EulerA,
        cache_dir="/tmp/blazen-diffusion-cache",
    )
    assert opts.model_id == "stabilityai/stable-diffusion-2-1"
    assert opts.device == "cpu"
    assert opts.width == 768
    assert opts.height == 768
    assert opts.num_inference_steps == 25
    # Float comparison with tolerance for f32 round-trip.
    assert abs(opts.guidance_scale - 7.5) < 1e-5
    assert opts.scheduler == DiffusionScheduler.EulerA
    assert opts.cache_dir == "/tmp/blazen-diffusion-cache"


def test_diffusion_options_defaults_are_none_or_default_scheduler():
    """DiffusionOptions with no kwargs has all-None fields except scheduler."""
    opts = DiffusionOptions()
    assert opts.model_id is None
    assert opts.device is None
    assert opts.width is None
    assert opts.height is None
    assert opts.num_inference_steps is None
    assert opts.guidance_scale is None
    assert opts.cache_dir is None
    # scheduler is a non-Optional enum getter; default per docs is EulerA.
    assert opts.scheduler == DiffusionScheduler.EulerA


def test_diffusion_options_setters_mutate_inplace():
    """DiffusionOptions exposes setters for each field."""
    opts = DiffusionOptions()
    opts.model_id = "runwayml/stable-diffusion-v1-5"
    opts.width = 1024
    opts.height = 1024
    opts.num_inference_steps = 50
    opts.scheduler = DiffusionScheduler.Dpm
    assert opts.model_id == "runwayml/stable-diffusion-v1-5"
    assert opts.width == 1024
    assert opts.height == 1024
    assert opts.num_inference_steps == 50
    assert opts.scheduler == DiffusionScheduler.Dpm


def test_diffusion_scheduler_variants_exist():
    """DiffusionScheduler exposes the four expected variants."""
    assert DiffusionScheduler.Euler != DiffusionScheduler.EulerA
    assert DiffusionScheduler.Dpm != DiffusionScheduler.Ddim
    assert DiffusionScheduler.EulerA != DiffusionScheduler.Dpm
    # Round-trip equality.
    assert DiffusionScheduler.Euler == DiffusionScheduler.Euler
    assert DiffusionScheduler.EulerA == DiffusionScheduler.EulerA
    assert DiffusionScheduler.Dpm == DiffusionScheduler.Dpm
    assert DiffusionScheduler.Ddim == DiffusionScheduler.Ddim


def test_diffusion_provider_class_shape():
    """DiffusionProvider is a class with the expected method/getter surface."""
    # generate_image is the async image-gen entry point per blazen.pyi.
    assert hasattr(DiffusionProvider, "generate_image")
    # Resolved-config getters exposed on the provider.
    assert hasattr(DiffusionProvider, "width")
    assert hasattr(DiffusionProvider, "height")
    assert hasattr(DiffusionProvider, "num_inference_steps")
    assert hasattr(DiffusionProvider, "guidance_scale")
    assert hasattr(DiffusionProvider, "scheduler")


def test_diffusion_provider_default_construction():
    """DiffusionProvider() with no options resolves default config getters."""
    # No options means the provider falls back to its documented defaults
    # (512x512, 20 steps, 7.5 guidance, EulerA scheduler) without touching
    # disk or downloading a model.
    provider = DiffusionProvider()
    assert provider.width == 512
    assert provider.height == 512
    assert provider.num_inference_steps == 20
    assert abs(provider.guidance_scale - 7.5) < 1e-5
    assert provider.scheduler == DiffusionScheduler.EulerA


def test_diffusion_provider_constructs_with_options():
    """DiffusionProvider(options=opts) reflects the provided options."""
    opts = DiffusionOptions(
        width=640,
        height=480,
        num_inference_steps=30,
        guidance_scale=8.0,
        scheduler=DiffusionScheduler.Ddim,
    )
    provider = DiffusionProvider(options=opts)
    assert provider.width == 640
    assert provider.height == 480
    assert provider.num_inference_steps == 30
    assert abs(provider.guidance_scale - 8.0) < 1e-5
    assert provider.scheduler == DiffusionScheduler.Ddim


# ---------------------------------------------------------------------------
# Live tests gated behind BLAZEN_TEST_DIFFUSION=1 + a real HF model id
# ---------------------------------------------------------------------------


@skip_without_flag
@skip_without_model
def test_diffusion_provider_constructs_with_real_model_id():
    """DiffusionProvider(options=opts) accepts a real HF repo id without raising."""
    opts = DiffusionOptions(model_id=MODEL_ID)
    provider = DiffusionProvider(options=opts)
    # Resolved width/height should be the documented default (512) when not
    # overridden, even if the upstream engine integration is still in progress.
    assert provider.width == 512
    assert provider.height == 512


@skip_without_flag
@skip_without_model
@pytest.mark.asyncio
async def test_diffusion_provider_generate_image_runs_or_raises_diffusion_error():
    """generate_image either returns an ImageResult or raises DiffusionError.

    The blazen.pyi docstring documents that the upstream engine integration
    is in progress and ``generate_image`` may raise :class:`DiffusionError`.
    This test accepts either outcome so it remains valid as the integration
    lands.
    """
    opts = DiffusionOptions(model_id=MODEL_ID)
    provider = DiffusionProvider(options=opts)

    request = ImageRequest(
        prompt="a small red cube on a white background",
        width=512,
        height=512,
        num_images=1,
    )

    try:
        result = await provider.generate_image(request)
    except Exception as exc:
        # DiffusionError is the documented failure mode while integration
        # is in progress. Any other exception type is unexpected.
        assert type(exc).__name__ == "DiffusionError", (
            f"unexpected exception {type(exc).__name__}: {exc}"
        )
        return

    # If the call succeeded, the result should be a non-None ImageResult.
    assert result is not None
