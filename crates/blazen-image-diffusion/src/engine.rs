//! Thin wrapper around the [`diffusion-rs`] runtime, gated on the
//! [`engine`](crate#feature-flags) feature so the bulk of the crate stays
//! dep-free for downstream consumers that only need the type stubs.
//!
//! diffusion-rs is a safe-Rust wrapper over `stable-diffusion.cpp`. It
//! exposes `api::gen_img(&Config, &mut ModelConfig)` which performs a
//! synchronous, blocking generation that writes the result to
//! `Config::output` on disk. There is also `api::gen_img_with_progress`
//! which takes a `std::sync::mpsc::Sender<Progress>` for per-step updates,
//! but the [`Progress`] type's fields are private so callers can only log
//! it via `Debug`; we intentionally use the non-progress variant here and
//! document the limitation in [`crate`].

use std::path::{Path, PathBuf};

use diffusion_rs::api::gen_img;
use diffusion_rs::preset::{Preset, PresetBuilder};
use image::ImageReader;

use crate::DiffusionError;
use crate::options::DiffusionOptions;

/// Outcome of a single `txt2img` invocation.
///
/// Re-exported via [`crate::GeneratedImage`] because the [`blazen-llm`]
/// bridge crate needs to read these fields to populate
/// `blazen_llm::media::GeneratedImage`.
#[derive(Debug, Clone)]
pub struct GeneratedImage {
    /// Raw bytes of the encoded image file (PNG by default -- diffusion-rs
    /// picks the codec from the output path's extension).
    pub bytes: Vec<u8>,
    /// Detected width in pixels.
    pub width: u32,
    /// Detected height in pixels.
    pub height: u32,
}

/// Owned diffusion-rs pipeline handle. Lazily constructed by
/// [`DiffusionProvider`](crate::DiffusionProvider) on first generation.
pub struct Engine {
    preset: Preset,
    output_dir: PathBuf,
}

impl Engine {
    /// Resolve options into a ready-to-run engine.
    ///
    /// This validates the preset selection and ensures the output directory
    /// exists, but defers weight download / pipeline construction until the
    /// first `txt2img` call -- diffusion-rs has no separate "warm up"
    /// entry point, weights are fetched inside `gen_img`.
    ///
    /// # Errors
    ///
    /// Returns [`DiffusionError::InvalidOptions`] if `opts.model_id` is not
    /// a recognised preset alias, or [`DiffusionError::ModelLoad`] if the
    /// output directory cannot be created.
    pub fn new(opts: &DiffusionOptions) -> Result<Self, DiffusionError> {
        let preset = resolve_preset(opts.model_id.as_deref())?;
        let output_dir = opts.cache_dir.clone().unwrap_or_else(default_output_dir);
        std::fs::create_dir_all(&output_dir)
            .map_err(|e| DiffusionError::ModelLoad(format!("create output dir: {e}")))?;
        Ok(Self { preset, output_dir })
    }

    /// Synchronous text-to-image generation. **Blocks the calling thread**
    /// inside the stable-diffusion.cpp FFI -- callers must invoke this from
    /// a [`tokio::task::spawn_blocking`] context if they hold an async
    /// runtime, otherwise the runtime will stall.
    ///
    /// # Errors
    ///
    /// - [`DiffusionError::ModelLoad`] if the diffusion-rs builder rejects
    ///   the configuration (e.g. invalid preset+modifier combination).
    /// - [`DiffusionError::Generation`] if `gen_img` fails or the output
    ///   file cannot be read back from disk.
    pub fn txt2img(
        &self,
        prompt: &str,
        neg: Option<&str>,
        width: u32,
        height: u32,
        steps: u32,
        guidance: f32,
    ) -> Result<GeneratedImage, DiffusionError> {
        let output_path = unique_output_path(&self.output_dir);

        // `PresetBuilder` uses derive_builder's `pattern = "owned"`, so every
        // setter consumes and returns `Self`. Width/height/steps/cfg_scale
        // and the output path are exposed on the inner `ConfigBuilder` (the
        // `.0` of `ConfigsBuilder`); we set them via a modifier so the
        // preset's curated defaults override only where the user opted in.
        let neg_owned = neg.map(str::to_owned);
        let out_for_modifier = output_path.clone();
        let (config, mut model_config) = PresetBuilder::default()
            .preset(self.preset)
            .prompt(prompt.to_owned())
            .with_modifier(move |mut cfgs| {
                #[allow(clippy::cast_possible_wrap)]
                {
                    cfgs.0
                        .width(width as i32)
                        .height(height as i32)
                        .steps(steps as i32)
                        .cfg_scale(guidance)
                        .output(out_for_modifier);
                }
                if let Some(n) = neg_owned {
                    cfgs.0.negative_prompt(n);
                }
                Ok(cfgs)
            })
            .build()
            .map_err(|e| DiffusionError::ModelLoad(format!("preset build: {e}")))?;

        tracing::info!(
            ?self.preset,
            output = %output_path.display(),
            width,
            height,
            steps,
            "running diffusion-rs txt2img"
        );

        gen_img(&config, &mut model_config)
            .map_err(|e| DiffusionError::Generation(format!("gen_img: {e}")))?;

        let bytes = std::fs::read(&output_path)
            .map_err(|e| DiffusionError::Generation(format!("read output: {e}")))?;

        // Use image's dimension-only fast path so we don't have to decode
        // the whole bitmap just to populate `width`/`height` metadata.
        let (got_w, got_h) = ImageReader::open(&output_path)
            .and_then(ImageReader::with_guessed_format)
            .map_err(|e| DiffusionError::Generation(format!("open output: {e}")))?
            .into_dimensions()
            .unwrap_or((width, height));

        // Best-effort cleanup; failing to remove the cache file is not a
        // generation failure.
        let _ = std::fs::remove_file(&output_path);

        Ok(GeneratedImage {
            bytes,
            width: got_w,
            height: got_h,
        })
    }
}

/// Map a Blazen-side `model_id` string into a [`Preset`]. Only the most
/// common presets are wired by name; anything else is rejected with a clear
/// error so we never silently fall back to a different model.
fn resolve_preset(model_id: Option<&str>) -> Result<Preset, DiffusionError> {
    let id = model_id.unwrap_or("sd-1.5").to_ascii_lowercase();
    let preset = match id.as_str() {
        "default" | "sd1.5" | "sd-1.5" | "stable-diffusion-1.5" => Preset::StableDiffusion1_5,
        "sd1.4" | "sd-1.4" | "stable-diffusion-1.4" => Preset::StableDiffusion1_4,
        "sd2.1" | "sd-2.1" | "stable-diffusion-2.1" => Preset::StableDiffusion2_1,
        "sdxl" | "sdxl-base" | "sdxl-base-1.0" => Preset::SDXLBase1_0,
        "sd-turbo" | "sdturbo" => Preset::SDTurbo,
        "sdxl-turbo" | "sdxl-turbo-1.0" => Preset::SDXLTurbo1_0,
        "sd3-medium" | "sd3" => Preset::StableDiffusion3Medium,
        "juggernaut-xl" | "juggernaut-xl-11" => Preset::JuggernautXL11,
        "segmind-vega" => Preset::SegmindVega,
        "dreamshaper-xl" | "dreamshaper-xl-2.1-turbo" => Preset::DreamShaperXL2_1Turbo,
        other => {
            return Err(DiffusionError::InvalidOptions(format!(
                "unrecognized diffusion preset `{other}` -- supported keys: \
                 sd-1.4, sd-1.5, sd-2.1, sdxl, sd-turbo, sdxl-turbo, sd3-medium, \
                 juggernaut-xl, segmind-vega, dreamshaper-xl"
            )));
        }
    };
    Ok(preset)
}

fn default_output_dir() -> PathBuf {
    blazen_model_cache::ModelCache::new()
        .map_or_else(
            |_| std::env::temp_dir().join("blazen-diffusion"),
            |c| c.cache_dir().to_path_buf(),
        )
        .join("diffusion-out")
}

/// Pick an output filename that is unlikely to collide with a concurrent
/// generation in the same process. `gen_img` infers the codec from the
/// extension; `.png` keeps everything lossless.
fn unique_output_path(dir: &Path) -> PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let seq = COUNTER.fetch_add(1, Ordering::Relaxed);
    let pid = std::process::id();
    dir.join(format!("blazen-diffusion-{pid}-{seq}.png"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_preset_default_is_sd15() {
        assert!(matches!(
            resolve_preset(None).unwrap(),
            Preset::StableDiffusion1_5
        ));
    }

    #[test]
    fn resolve_preset_known_aliases() {
        assert!(matches!(
            resolve_preset(Some("sdxl")).unwrap(),
            Preset::SDXLBase1_0
        ));
        assert!(matches!(
            resolve_preset(Some("SD-Turbo")).unwrap(),
            Preset::SDTurbo
        ));
        assert!(matches!(
            resolve_preset(Some("stable-diffusion-2.1")).unwrap(),
            Preset::StableDiffusion2_1
        ));
    }

    #[test]
    fn resolve_preset_unknown_is_rejected() {
        let err = resolve_preset(Some("midjourney")).unwrap_err();
        assert!(
            matches!(err, DiffusionError::InvalidOptions(_)),
            "expected InvalidOptions, got {err:?}"
        );
    }

    #[test]
    fn unique_output_path_is_unique() {
        let dir = std::env::temp_dir();
        let a = unique_output_path(&dir);
        let b = unique_output_path(&dir);
        assert_ne!(a, b);
        assert_eq!(a.extension().and_then(|s| s.to_str()), Some("png"));
    }
}
