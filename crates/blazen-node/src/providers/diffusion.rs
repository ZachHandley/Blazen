//! JavaScript bindings for the local diffusion-rs image provider.
//!
//! Exposes [`JsDiffusionProvider`] as a NAPI class with a factory
//! constructor.
//!
//! The actual `ImageGeneration` surface is not yet wired up in
//! [`blazen_image_diffusion::DiffusionProvider`] -- the upstream crate
//! currently validates options and stages the pipeline handle for the
//! Phase 5.3 engine integration. Once that lands, `generateImage` will be
//! added here.

#![cfg(feature = "diffusion")]

use std::sync::Arc;

use napi::bindgen_prelude::*;
use napi_derive::napi;

use blazen_llm::{DiffusionOptions, DiffusionProvider, DiffusionScheduler};

// ---------------------------------------------------------------------------
// JsDiffusionScheduler
// ---------------------------------------------------------------------------

/// Noise schedulers available for the diffusion process.
///
/// Different schedulers trade off between generation speed and output
/// quality. `eulerA` is a good default for most use cases.
#[napi(string_enum)]
pub enum JsDiffusionScheduler {
    #[napi(value = "euler")]
    Euler,
    #[napi(value = "eulerA")]
    EulerA,
    #[napi(value = "dpm")]
    Dpm,
    #[napi(value = "ddim")]
    Ddim,
}

impl From<JsDiffusionScheduler> for DiffusionScheduler {
    fn from(s: JsDiffusionScheduler) -> Self {
        match s {
            JsDiffusionScheduler::Euler => Self::Euler,
            JsDiffusionScheduler::EulerA => Self::EulerA,
            JsDiffusionScheduler::Dpm => Self::Dpm,
            JsDiffusionScheduler::Ddim => Self::Ddim,
        }
    }
}

// ---------------------------------------------------------------------------
// JsDiffusionOptions
// ---------------------------------------------------------------------------

/// Options for the local diffusion-rs image generation backend.
///
/// All fields are optional. Defaults: 512x512, 20 steps, 7.5 guidance,
/// `eulerA` scheduler.
///
/// ```javascript
/// const provider = DiffusionProvider.create({
///   modelId: "stabilityai/stable-diffusion-2-1",
///   width: 1024,
///   height: 1024,
///   numInferenceSteps: 30,
/// });
/// ```
#[napi(object)]
pub struct JsDiffusionOptions {
    /// `HuggingFace` model repository ID.
    #[napi(js_name = "modelId")]
    pub model_id: Option<String>,
    /// Hardware device specifier (e.g. `"cpu"`, `"cuda:0"`, `"metal"`).
    pub device: Option<String>,
    /// Output image width in pixels.
    pub width: Option<u32>,
    /// Output image height in pixels.
    pub height: Option<u32>,
    /// Number of denoising steps to run.
    #[napi(js_name = "numInferenceSteps")]
    pub num_inference_steps: Option<u32>,
    /// Classifier-free guidance scale.
    #[napi(js_name = "guidanceScale")]
    pub guidance_scale: Option<f64>,
    /// Noise scheduler to use.
    pub scheduler: Option<JsDiffusionScheduler>,
    /// Path to cache downloaded models.
    #[napi(js_name = "cacheDir")]
    pub cache_dir: Option<String>,
}

impl From<JsDiffusionOptions> for DiffusionOptions {
    fn from(val: JsDiffusionOptions) -> Self {
        Self {
            model_id: val.model_id,
            device: val.device,
            width: val.width,
            height: val.height,
            num_inference_steps: val.num_inference_steps,
            #[allow(clippy::cast_possible_truncation)]
            guidance_scale: val.guidance_scale.map(|v| v as f32),
            scheduler: val.scheduler.map(Into::into).unwrap_or_default(),
            cache_dir: val.cache_dir.map(std::path::PathBuf::from),
        }
    }
}

// ---------------------------------------------------------------------------
// JsDiffusionProvider NAPI class
// ---------------------------------------------------------------------------

/// A local diffusion-rs image generation provider.
///
/// ```javascript
/// const provider = DiffusionProvider.create({
///   modelId: "stabilityai/stable-diffusion-2-1",
/// });
/// ```
#[napi(js_name = "DiffusionProvider")]
pub struct JsDiffusionProvider {
    inner: Arc<DiffusionProvider>,
}

#[napi]
#[allow(clippy::must_use_candidate, clippy::missing_errors_doc)]
impl JsDiffusionProvider {
    // -----------------------------------------------------------------
    // Factory
    // -----------------------------------------------------------------

    /// Create a new diffusion provider.
    #[napi(factory)]
    pub fn create(options: Option<JsDiffusionOptions>) -> Result<Self> {
        let opts: DiffusionOptions = options.map(Into::into).unwrap_or_default();
        Ok(Self {
            inner: Arc::new(
                DiffusionProvider::from_options(opts)
                    .map_err(|e| napi::Error::new(napi::Status::GenericFailure, e.to_string()))?,
            ),
        })
    }

    // -----------------------------------------------------------------
    // Resolved options
    // -----------------------------------------------------------------

    /// The resolved width (user-specified or default 512).
    #[napi(getter)]
    pub fn width(&self) -> u32 {
        self.inner.width()
    }

    /// The resolved height (user-specified or default 512).
    #[napi(getter)]
    pub fn height(&self) -> u32 {
        self.inner.height()
    }

    /// The resolved number of inference steps (user-specified or default 20).
    #[napi(js_name = "numInferenceSteps", getter)]
    pub fn num_inference_steps(&self) -> u32 {
        self.inner.num_inference_steps()
    }

    /// The resolved guidance scale (user-specified or default 7.5).
    #[napi(js_name = "guidanceScale", getter)]
    pub fn guidance_scale(&self) -> f64 {
        f64::from(self.inner.guidance_scale())
    }
}
