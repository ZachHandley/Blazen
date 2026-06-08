//! The [`DiffusionProvider`] type.
//!
//! Without the `engine` cargo feature this is a pure-stub provider: it
//! validates options and exposes accessors but cannot actually run image
//! generation. With `engine`, the inherent [`DiffusionProvider::generate_image`]
//! method lazily initialises a [`crate::engine::Engine`] and runs the
//! stable-diffusion.cpp pipeline through `diffusion-rs`.

use std::fmt;

use crate::DiffusionOptions;

/// Error type for diffusion-rs operations.
#[derive(Debug)]
pub enum DiffusionError {
    /// A required option was missing or invalid.
    InvalidOptions(String),
    /// The model file could not be downloaded or found.
    ModelLoad(String),
    /// An image generation operation failed.
    Generation(String),
    /// The crate was built without the `engine` feature so the underlying
    /// diffusion-rs runtime is not linked. Surface this distinctly from
    /// generic generation failures so bindings can map it to a clear error.
    EngineNotAvailable,
}

impl fmt::Display for DiffusionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidOptions(msg) => write!(f, "diffusion-rs invalid options: {msg}"),
            Self::ModelLoad(msg) => write!(f, "diffusion-rs model load failed: {msg}"),
            Self::Generation(msg) => write!(f, "diffusion-rs generation failed: {msg}"),
            Self::EngineNotAvailable => f.write_str(
                "diffusion-rs runtime is not linked -- rebuild blazen-image-diffusion \
                 with the `engine` feature (or a forwarding feature such as `cuda` / \
                 `metal`) to enable image generation",
            ),
        }
    }
}

impl std::error::Error for DiffusionError {}

/// A local image generation provider backed by [`diffusion-rs`](https://github.com/newfla/diffusion-rs).
///
/// Constructed via [`DiffusionProvider::from_options`]. With the `engine`
/// feature on, [`DiffusionProvider::generate_image`] lazily initialises the
/// underlying pipeline on first call and runs the synchronous
/// stable-diffusion.cpp generation inside [`tokio::task::spawn_blocking`].
pub struct DiffusionProvider {
    /// Full options preserved for deferred engine initialisation.
    options: DiffusionOptions,
    #[cfg(feature = "engine")]
    engine: tokio::sync::OnceCell<std::sync::Arc<crate::engine::Engine>>,
}

impl DiffusionProvider {
    /// Create a new provider from the given options.
    ///
    /// This currently validates the options and stores them. The actual
    /// diffusion-rs pipeline will be initialised in Phase 5.3.
    ///
    /// # Errors
    ///
    /// Returns [`DiffusionError::InvalidOptions`] if any option is present but
    /// invalid (e.g. an empty device string, zero dimensions, or zero steps).
    pub fn from_options(opts: DiffusionOptions) -> Result<Self, DiffusionError> {
        if let Some(ref device) = opts.device
            && device.is_empty()
        {
            return Err(DiffusionError::InvalidOptions(
                "device must not be empty when specified".into(),
            ));
        }

        if let Some(ref model_id) = opts.model_id
            && model_id.is_empty()
        {
            return Err(DiffusionError::InvalidOptions(
                "model_id must not be empty when specified".into(),
            ));
        }

        if let Some(width) = opts.width
            && width == 0
        {
            return Err(DiffusionError::InvalidOptions(
                "width must be greater than zero".into(),
            ));
        }

        if let Some(height) = opts.height
            && height == 0
        {
            return Err(DiffusionError::InvalidOptions(
                "height must be greater than zero".into(),
            ));
        }

        if let Some(steps) = opts.num_inference_steps
            && steps == 0
        {
            return Err(DiffusionError::InvalidOptions(
                "num_inference_steps must be greater than zero".into(),
            ));
        }

        if let Some(scale) = opts.guidance_scale
            && scale <= 0.0
        {
            return Err(DiffusionError::InvalidOptions(
                "guidance_scale must be positive".into(),
            ));
        }

        Ok(Self {
            options: opts,
            #[cfg(feature = "engine")]
            engine: tokio::sync::OnceCell::new(),
        })
    }

    /// The resolved device string (`"cpu"` when unset).
    #[must_use]
    pub fn device_str(&self) -> &str {
        self.options.device.as_deref().unwrap_or("cpu")
    }

    /// The configured model identifier (or `"sd-1.5"` when unset).
    #[must_use]
    pub fn model_id(&self) -> &str {
        self.options.model_id.as_deref().unwrap_or("sd-1.5")
    }

    /// The resolved width (user-specified or default 512).
    #[must_use]
    pub fn width(&self) -> u32 {
        self.options.width.unwrap_or(512)
    }

    /// The resolved height (user-specified or default 512).
    #[must_use]
    pub fn height(&self) -> u32 {
        self.options.height.unwrap_or(512)
    }

    /// The resolved number of inference steps (user-specified or default 20).
    #[must_use]
    pub fn num_inference_steps(&self) -> u32 {
        self.options.num_inference_steps.unwrap_or(20)
    }

    /// The resolved guidance scale (user-specified or default 7.5).
    #[must_use]
    pub fn guidance_scale(&self) -> f32 {
        self.options.guidance_scale.unwrap_or(7.5)
    }

    /// The scheduler configured for this provider.
    #[must_use]
    pub const fn scheduler(&self) -> crate::DiffusionScheduler {
        self.options.scheduler
    }

    /// Eagerly warm the underlying diffusion-rs pipeline.
    ///
    /// Without the `engine` feature this returns
    /// [`DiffusionError::EngineNotAvailable`]. With it, this is idempotent
    /// and safe to call from multiple tasks concurrently.
    ///
    /// # Errors
    ///
    /// Returns [`DiffusionError::ModelLoad`] if pipeline construction or the
    /// output-directory bootstrap fails.
    #[allow(clippy::unused_async)] // async to mirror LocalModel and the engine path
    pub async fn load(&self) -> Result<(), DiffusionError> {
        #[cfg(feature = "engine")]
        {
            let opts = self.options.clone();
            self.engine
                .get_or_try_init(|| async move {
                    tokio::task::spawn_blocking(move || crate::engine::Engine::new(&opts))
                        .await
                        .map_err(|e| DiffusionError::ModelLoad(format!("join: {e}")))?
                        .map(std::sync::Arc::new)
                })
                .await?;
            Ok(())
        }
        #[cfg(not(feature = "engine"))]
        {
            Err(DiffusionError::EngineNotAvailable)
        }
    }

    /// Best-effort unload. Always succeeds.
    ///
    /// `diffusion-rs` does not expose a "drop weights" entry point and the
    /// cached pipeline lives behind a [`tokio::sync::OnceCell`] shared via
    /// `&self`, so we cannot evict it from interior mutability alone.
    /// Callers that require strict resource release should `drop` the
    /// entire [`DiffusionProvider`] and construct a fresh one.
    ///
    /// # Errors
    ///
    /// Never errors today; the `Result` is kept to match the
    /// [`blazen_llm::LocalModel::unload`] trait signature so the
    /// bridge can forward without contortions.
    #[allow(clippy::unused_async)]
    pub async fn unload(&self) -> Result<(), DiffusionError> {
        Ok(())
    }

    /// `true` if a pipeline has been warmed via [`Self::load`] or the first
    /// generate call.
    #[allow(clippy::unused_async)]
    pub async fn is_loaded(&self) -> bool {
        #[cfg(feature = "engine")]
        {
            self.engine.initialized()
        }
        #[cfg(not(feature = "engine"))]
        {
            false
        }
    }
}

#[cfg(feature = "engine")]
impl DiffusionProvider {
    /// Inherent text-to-image entry point used by the
    /// [`blazen_llm::ImageGeneration`] trait impl in
    /// `blazen-llm::backends::diffusion`. Kept as an inherent method so the
    /// engine type does not leak across the `blazen-llm` boundary.
    ///
    /// # Errors
    ///
    /// Returns a [`DiffusionError`] if the engine cannot be initialised or
    /// generation fails.
    pub async fn generate_image_inherent(
        &self,
        prompt: String,
        negative_prompt: Option<String>,
        width: Option<u32>,
        height: Option<u32>,
    ) -> Result<crate::engine::GeneratedImage, DiffusionError> {
        // Lazy init via the shared OnceCell.
        let opts = self.options.clone();
        let engine = self
            .engine
            .get_or_try_init(|| async move {
                tokio::task::spawn_blocking(move || crate::engine::Engine::new(&opts))
                    .await
                    .map_err(|e| DiffusionError::ModelLoad(format!("join: {e}")))?
                    .map(std::sync::Arc::new)
            })
            .await?
            .clone();

        let w = width.unwrap_or_else(|| self.width());
        let h = height.unwrap_or_else(|| self.height());
        let steps = self.num_inference_steps();
        let scale = self.guidance_scale();

        tokio::task::spawn_blocking(move || {
            engine.txt2img(&prompt, negative_prompt.as_deref(), w, h, steps, scale)
        })
        .await
        .map_err(|e| DiffusionError::Generation(format!("join: {e}")))?
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DiffusionOptions, DiffusionScheduler};

    #[test]
    fn from_options_with_defaults() {
        let opts = DiffusionOptions::default();
        let provider = DiffusionProvider::from_options(opts).expect("should succeed");
        assert_eq!(provider.width(), 512);
        assert_eq!(provider.height(), 512);
        assert_eq!(provider.num_inference_steps(), 20);
        assert!((provider.guidance_scale() - 7.5).abs() < f32::EPSILON);
        assert_eq!(provider.scheduler(), DiffusionScheduler::EulerA);
    }

    /// GPU/inference smoke: download a fast diffusion model and generate one
    /// image. Gated on `engine` (the real generate path) and `#[ignore]`'d so
    /// the beastpc-e2e `--run-ignored only` step runs it. Uses SD-Turbo at a
    /// single step for speed. Runs on CPU unless the binary enables
    /// `diffusion-rs/cuda` (the crate-level `cuda` feature is a marker — see
    /// Cargo.toml).
    #[cfg(feature = "engine")]
    #[tokio::test]
    #[ignore = "downloads an SD-Turbo diffusion model + generates an image"]
    async fn smoke_generate_image() {
        let opts = DiffusionOptions {
            model_id: Some("sd-turbo".into()),
            num_inference_steps: Some(1),
            ..DiffusionOptions::default()
        };
        let provider = DiffusionProvider::from_options(opts).expect("options valid");
        let image = provider
            .generate_image_inherent("a red square".into(), None, Some(512), Some(512))
            .await
            .expect("image generation should succeed");
        assert!(
            !image.bytes.is_empty(),
            "should produce non-empty image bytes"
        );
        assert!(
            image.width > 0 && image.height > 0,
            "image should have positive dimensions, got {}x{}",
            image.width,
            image.height
        );
    }

    #[test]
    fn from_options_with_custom_values() {
        let opts = DiffusionOptions {
            model_id: Some("stabilityai/stable-diffusion-2-1".into()),
            width: Some(1024),
            height: Some(768),
            num_inference_steps: Some(30),
            guidance_scale: Some(10.0),
            scheduler: DiffusionScheduler::Dpm,
            ..DiffusionOptions::default()
        };
        let provider = DiffusionProvider::from_options(opts).expect("should succeed");
        assert_eq!(provider.width(), 1024);
        assert_eq!(provider.height(), 768);
        assert_eq!(provider.num_inference_steps(), 30);
        assert!((provider.guidance_scale() - 10.0).abs() < f32::EPSILON);
        assert_eq!(provider.scheduler(), DiffusionScheduler::Dpm);
    }

    #[test]
    fn from_options_rejects_empty_device() {
        let opts = DiffusionOptions {
            device: Some(String::new()),
            ..DiffusionOptions::default()
        };
        assert!(DiffusionProvider::from_options(opts).is_err());
    }

    #[test]
    fn from_options_rejects_empty_model_id() {
        let opts = DiffusionOptions {
            model_id: Some(String::new()),
            ..DiffusionOptions::default()
        };
        assert!(DiffusionProvider::from_options(opts).is_err());
    }

    #[test]
    fn from_options_rejects_zero_width() {
        let opts = DiffusionOptions {
            width: Some(0),
            ..DiffusionOptions::default()
        };
        assert!(DiffusionProvider::from_options(opts).is_err());
    }

    #[test]
    fn from_options_rejects_zero_height() {
        let opts = DiffusionOptions {
            height: Some(0),
            ..DiffusionOptions::default()
        };
        assert!(DiffusionProvider::from_options(opts).is_err());
    }

    #[test]
    fn from_options_rejects_zero_steps() {
        let opts = DiffusionOptions {
            num_inference_steps: Some(0),
            ..DiffusionOptions::default()
        };
        assert!(DiffusionProvider::from_options(opts).is_err());
    }

    #[test]
    fn from_options_rejects_non_positive_guidance() {
        let opts = DiffusionOptions {
            guidance_scale: Some(0.0),
            ..DiffusionOptions::default()
        };
        assert!(DiffusionProvider::from_options(opts).is_err());

        let opts = DiffusionOptions {
            guidance_scale: Some(-1.0),
            ..DiffusionOptions::default()
        };
        assert!(DiffusionProvider::from_options(opts).is_err());
    }

    #[test]
    fn from_options_accepts_valid_device() {
        let opts = DiffusionOptions {
            device: Some("cuda:0".into()),
            ..DiffusionOptions::default()
        };
        let provider = DiffusionProvider::from_options(opts).expect("should succeed");
        assert_eq!(provider.width(), 512);
    }
}
