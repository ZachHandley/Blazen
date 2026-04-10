//! The [`DiffusionProvider`] type -- stub for Phase 5.1-5.2.
//!
//! The actual `ImageGeneration` trait implementation will be added in Phase 5.3
//! once the diffusion-rs engine API is wired up.

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
}

impl fmt::Display for DiffusionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidOptions(msg) => write!(f, "diffusion-rs invalid options: {msg}"),
            Self::ModelLoad(msg) => write!(f, "diffusion-rs model load failed: {msg}"),
            Self::Generation(msg) => write!(f, "diffusion-rs generation failed: {msg}"),
        }
    }
}

impl std::error::Error for DiffusionError {}

/// A local image generation provider backed by [`diffusion-rs`](https://github.com/huggingface/diffusion-rs).
///
/// Constructed via [`DiffusionProvider::from_options`]. The `ImageGeneration`
/// trait implementation will be added in Phase 5.3.
pub struct DiffusionProvider {
    /// Full options preserved for deferred engine initialisation.
    #[allow(dead_code)]
    options: DiffusionOptions,
    // pipeline: ... -- will hold the diffusion-rs pipeline once wired (Phase 5.3)
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

        Ok(Self { options: opts })
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
