//! Configuration options for the diffusion-rs local image generation backend.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// Options for constructing a [`DiffusionProvider`](crate::DiffusionProvider).
///
/// All fields are optional and have sensible defaults. The scheduler defaults
/// to [`DiffusionScheduler::EulerA`] and image dimensions default to 512x512.
///
/// # Examples
///
/// ```
/// use blazen_image_diffusion::{DiffusionOptions, DiffusionScheduler};
///
/// // Use defaults (512x512, EulerA scheduler, 20 steps)
/// let opts = DiffusionOptions::default();
/// assert_eq!(opts.scheduler, DiffusionScheduler::EulerA);
///
/// // Override specific fields
/// let opts = DiffusionOptions {
///     width: Some(1024),
///     height: Some(1024),
///     num_inference_steps: Some(30),
///     ..DiffusionOptions::default()
/// };
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DiffusionOptions {
    /// `HuggingFace` model repository ID (e.g. `"stabilityai/stable-diffusion-2-1"`).
    ///
    /// When `None`, a sensible default model will be selected in Phase 5.3.
    pub model_id: Option<String>,

    /// Hardware device specifier string (e.g. `"cpu"`, `"cuda:0"`, `"metal"`).
    ///
    /// Accepts the same format strings as `blazen_llm::Device::parse`:
    /// `"cpu"`, `"cuda"`, `"cuda:N"`, `"metal"`.
    ///
    /// When `None`, defaults to `"cpu"`.
    pub device: Option<String>,

    /// Output image width in pixels.
    ///
    /// When `None`, defaults to 512.
    pub width: Option<u32>,

    /// Output image height in pixels.
    ///
    /// When `None`, defaults to 512.
    pub height: Option<u32>,

    /// Number of denoising steps to run.
    ///
    /// More steps generally produce higher quality images at the cost of
    /// longer generation time. When `None`, defaults to 20.
    pub num_inference_steps: Option<u32>,

    /// Classifier-free guidance scale.
    ///
    /// Higher values make the output more closely follow the prompt but may
    /// reduce diversity. Typical values range from 5.0 to 15.0.
    /// When `None`, defaults to 7.5.
    pub guidance_scale: Option<f32>,

    /// The noise scheduler to use during the diffusion process.
    ///
    /// Defaults to [`DiffusionScheduler::EulerA`].
    #[serde(default)]
    pub scheduler: DiffusionScheduler,

    /// Path to cache downloaded models.
    ///
    /// When `None`, falls back to `blazen-model-cache`'s default cache
    /// directory (`$BLAZEN_CACHE_DIR` or `~/.cache/blazen/models`).
    pub cache_dir: Option<PathBuf>,
}

/// Noise schedulers available for the diffusion process.
///
/// Different schedulers trade off between generation speed and output quality.
/// [`EulerA`](Self::EulerA) is a good default for most use cases.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum DiffusionScheduler {
    /// Euler discrete scheduler.
    Euler,
    /// Euler ancestral discrete scheduler (stochastic, good default).
    #[default]
    EulerA,
    /// DPM-Solver++ multistep scheduler (fast, high quality).
    #[serde(rename = "DPM")]
    Dpm,
    /// Denoising Diffusion Implicit Models scheduler.
    #[serde(rename = "DDIM")]
    Ddim,
}

impl std::fmt::Display for DiffusionScheduler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            Self::Euler => "euler",
            Self::EulerA => "euler_a",
            Self::Dpm => "dpm",
            Self::Ddim => "ddim",
        };
        f.write_str(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_scheduler_is_euler_a() {
        assert_eq!(DiffusionScheduler::default(), DiffusionScheduler::EulerA);
    }

    #[test]
    fn default_options_has_euler_a_scheduler() {
        let opts = DiffusionOptions::default();
        assert_eq!(opts.scheduler, DiffusionScheduler::EulerA);
        assert!(opts.model_id.is_none());
        assert!(opts.device.is_none());
        assert!(opts.width.is_none());
        assert!(opts.height.is_none());
        assert!(opts.num_inference_steps.is_none());
        assert!(opts.guidance_scale.is_none());
        assert!(opts.cache_dir.is_none());
    }

    #[test]
    fn struct_update_syntax_works() {
        let opts = DiffusionOptions {
            width: Some(1024),
            height: Some(1024),
            num_inference_steps: Some(30),
            ..DiffusionOptions::default()
        };
        assert_eq!(opts.width, Some(1024));
        assert_eq!(opts.height, Some(1024));
        assert_eq!(opts.num_inference_steps, Some(30));
        assert!(opts.model_id.is_none());
        assert!(opts.device.is_none());
    }

    #[test]
    fn display_impl() {
        assert_eq!(DiffusionScheduler::Euler.to_string(), "euler");
        assert_eq!(DiffusionScheduler::EulerA.to_string(), "euler_a");
        assert_eq!(DiffusionScheduler::Dpm.to_string(), "dpm");
        assert_eq!(DiffusionScheduler::Ddim.to_string(), "ddim");
    }

    #[test]
    fn serde_roundtrip_options() {
        let opts = DiffusionOptions {
            model_id: Some("stabilityai/stable-diffusion-2-1".into()),
            device: Some("cuda:0".into()),
            width: Some(768),
            height: Some(768),
            num_inference_steps: Some(25),
            guidance_scale: Some(7.5),
            scheduler: DiffusionScheduler::Dpm,
            cache_dir: Some(PathBuf::from("/tmp/diffusion-cache")),
        };
        let json = serde_json::to_string(&opts).expect("serialize");
        let parsed: DiffusionOptions = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(
            parsed.model_id.as_deref(),
            Some("stabilityai/stable-diffusion-2-1")
        );
        assert_eq!(parsed.device.as_deref(), Some("cuda:0"));
        assert_eq!(parsed.width, Some(768));
        assert_eq!(parsed.height, Some(768));
        assert_eq!(parsed.num_inference_steps, Some(25));
        assert_eq!(parsed.guidance_scale, Some(7.5));
        assert_eq!(parsed.scheduler, DiffusionScheduler::Dpm);
        assert_eq!(
            parsed.cache_dir.as_deref(),
            Some(std::path::Path::new("/tmp/diffusion-cache"))
        );
    }

    #[test]
    fn serde_roundtrip_scheduler_enum() {
        for scheduler in [
            DiffusionScheduler::Euler,
            DiffusionScheduler::EulerA,
            DiffusionScheduler::Dpm,
            DiffusionScheduler::Ddim,
        ] {
            let json = serde_json::to_string(&scheduler).expect("serialize");
            let parsed: DiffusionScheduler = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(parsed, scheduler);
        }
    }
}
