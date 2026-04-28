//! Python wrapper for the local diffusion-rs image generation provider.
//!
//! The diffusion-rs engine integration is in progress in the Rust crate
//! (Phase 5.3 per the upstream roadmap); the Python class is exposed now
//! so callers can construct providers and surface engine-availability
//! errors with the same shape as the other local backends.

use std::sync::Arc;

use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use crate::compute::request_types::PyImageRequest;
use crate::compute::result_types::PyImageResult;
use crate::error::DiffusionError;
use crate::providers::options::{PyDiffusionOptions, PyDiffusionScheduler};
use blazen_llm::DiffusionProvider;

// ---------------------------------------------------------------------------
// PyDiffusionProvider
// ---------------------------------------------------------------------------

/// A local diffusion-rs image generation provider.
///
/// Runs Stable Diffusion inference fully on-device via the diffusion-rs
/// engine. No API key is required.
///
/// The underlying Rust integration is in progress; calls to
/// :meth:`generate_image` currently raise :class:`DiffusionError` until
/// the pipeline wiring lands.
///
/// Example:
///     >>> opts = DiffusionOptions(model_id="stabilityai/stable-diffusion-2-1")
///     >>> provider = DiffusionProvider(options=opts)
#[gen_stub_pyclass]
#[pyclass(name = "DiffusionProvider", from_py_object)]
#[derive(Clone)]
pub struct PyDiffusionProvider {
    inner: Arc<DiffusionProvider>,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyDiffusionProvider {
    /// Create a new diffusion-rs provider.
    ///
    /// Args:
    ///     options: Optional :class:`DiffusionOptions` for model id, device,
    ///         dimensions, inference steps, guidance scale, scheduler, and
    ///         cache directory.
    #[new]
    #[pyo3(signature = (*, options=None))]
    fn new(options: Option<PyRef<'_, PyDiffusionOptions>>) -> PyResult<Self> {
        let opts = options.map(|o| o.inner.clone()).unwrap_or_default();
        let provider = DiffusionProvider::from_options(opts)
            .map_err(|e| DiffusionError::new_err(e.to_string()))?;
        Ok(Self {
            inner: Arc::new(provider),
        })
    }

    /// Resolved output image width (default 512).
    #[getter]
    fn width(&self) -> u32 {
        self.inner.width()
    }

    /// Resolved output image height (default 512).
    #[getter]
    fn height(&self) -> u32 {
        self.inner.height()
    }

    /// Resolved number of inference steps (default 20).
    #[getter]
    fn num_inference_steps(&self) -> u32 {
        self.inner.num_inference_steps()
    }

    /// Resolved guidance scale (default 7.5).
    #[getter]
    fn guidance_scale(&self) -> f32 {
        self.inner.guidance_scale()
    }

    /// Configured noise scheduler.
    #[getter]
    fn scheduler(&self) -> PyDiffusionScheduler {
        self.inner.scheduler().into()
    }

    /// Generate images from a text prompt.
    ///
    /// Args:
    ///     request: An :class:`ImageRequest` with prompt and parameters.
    ///
    /// Returns:
    ///     An :class:`ImageResult`.
    ///
    /// Raises:
    ///     DiffusionError: While the upstream engine integration is in
    ///         progress, this call always raises.
    #[gen_stub(override_return_type(type_repr = "typing.Coroutine[typing.Any, typing.Any, ImageResult]", imports = ("typing",)))]
    fn generate_image<'py>(
        &self,
        py: Python<'py>,
        _request: PyImageRequest,
    ) -> PyResult<Bound<'py, PyAny>> {
        // Capture the resolved configuration so the error message tells the
        // caller exactly what the provider would have rendered. This also
        // keeps the method consuming `&self`, matching the eventual wired
        // version's signature.
        let summary = format!(
            "diffusion-rs image generation is not yet wired to the engine pipeline \
             (would have rendered {}x{} with {} steps, scheduler={})",
            self.inner.width(),
            self.inner.height(),
            self.inner.num_inference_steps(),
            self.inner.scheduler(),
        );
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            Err::<PyImageResult, _>(DiffusionError::new_err(summary))
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "DiffusionProvider(width={}, height={}, steps={})",
            self.inner.width(),
            self.inner.height(),
            self.inner.num_inference_steps()
        )
    }
}
