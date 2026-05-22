//! `DINOv2`-base image encoder wrapper around `blazen-3d-core` (Wave T.2).
//!
//! `TripoSR` conditions its triplane transformer on per-patch features
//! extracted from a `DINOv2`-base `ViT` (see
//! <https://github.com/facebookresearch/dinov2>). The upstream candle
//! port hard-codes a `14 px` patch grid, so the canonical input size
//! used by `TripoSR` is `518 x 518` (=> `37 x 37 = 1369` patch tokens
//! plus the CLS token for a total of `1370` tokens, each of dim `768`).
//!
//! This module is a thin wrapper that:
//!
//! 1. Loads the `DINOv2` weights from a safetensors file via
//!    [`candle_nn::VarBuilder::from_mmaped_safetensors`].
//! 2. Resizes pre-decoded `RGB` pixel data to `518 x 518`.
//! 3. Normalises with the standard `ImageNet` mean / std.
//! 4. Drives [`blazen_3d_core::image_encoders::DinoV2Encoder::encode`]
//!    on a `(1, 3, 518, 518)` tensor.
//!
//! # Wave T.2 caveats
//!
//! - Resize uses **nearest-neighbour** interpolation. `DINOv2`'s
//!   robust `ViT` features tolerate the visual artefact at the
//!   scaffolding level, and a `candle`-native bilinear resize is a
//!   non-trivial follow-up. Bilinear is the obvious enhancement.
//! - [`blazen_3d_core::image_encoders::DinoV2Encoder::encode`] returns
//!   the upstream candle `DINOv2` classifier logits (`(1, 1000)`) today;
//!   the per-patch-token output that `TripoSR`'s triplane transformer
//!   actually consumes (`(1, 1370, 768)`) will land alongside an
//!   `get_intermediate_layers`-style helper in `blazen-3d-core` in a
//!   follow-up. The signature here already returns a [`Tensor`] so
//!   that future change is source-compatible for callers in this
//!   crate.

// Wave T.2 scaffolding: this module's public surface is consumed by
// `pipeline.rs` in a follow-up wave. Until that lands, every item
// here looks "dead" to the compiler. The sibling `triplane_transformer`
// module uses the same allow for the same reason.
#![allow(
    dead_code,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap
)]

use std::path::Path;

use blazen_3d_core::image_encoders::{
    DinoV2Config, DinoV2Encoder, ImageEncoderError as Core3dImageEncoderError,
};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use thiserror::Error;

/// Side length (in pixels) of the resized image that is fed to
/// `DINOv2`-base. `518 = 37 * 14`, giving a `37 x 37` patch grid at
/// the upstream candle `PATCH_SIZE = 14` setting.
pub const DINOV2_INPUT_SIZE: usize = 518;

/// Per-channel `ImageNet` mean used by `DINOv2` (and most `ViT`
/// checkpoints) for input normalisation.
pub const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];

/// Per-channel `ImageNet` standard deviation used by `DINOv2` (and most
/// `ViT` checkpoints) for input normalisation.
pub const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

/// Errors raised while loading or running the `TripoSR` `DINOv2` image
/// encoder.
#[derive(Debug, Error)]
pub enum TripoSrEncoderError {
    /// Filesystem error while reading the safetensors weights file.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    /// Model construction failed (bad shapes, missing keys, …).
    #[error("model load failed: {0}")]
    ModelLoad(String),
    /// Inference failure surfaced from this wrapper's own preprocessing
    /// / shape-validation path (as opposed to a raw `candle` error,
    /// which is reported via the [`TripoSrEncoderError::Candle`]
    /// variant).
    #[error("inference error: {0}")]
    Inference(String),
    /// A candle tensor operation failed.
    #[error("candle error: {0}")]
    Candle(#[from] candle_core::Error),
    /// An error surfaced from `blazen-3d-core`'s `DinoV2Encoder`
    /// wrapper. Stored as a string because
    /// [`Core3dImageEncoderError`] does not implement `Clone` and we
    /// want a flat, `Send + Sync + 'static` payload.
    #[error("blazen-3d-core error: {0}")]
    Core3d(String),
}

impl From<Core3dImageEncoderError> for TripoSrEncoderError {
    fn from(err: Core3dImageEncoderError) -> Self {
        match err {
            Core3dImageEncoderError::Candle(e) => Self::Candle(e),
            other => Self::Core3d(other.to_string()),
        }
    }
}

/// `DINOv2`-base image encoder used by the `TripoSR` pipeline.
///
/// Construct with [`TripoSrImageEncoder::load`]. The instance owns the
/// underlying [`DinoV2Encoder`] and stays on the device the weights
/// were materialised on.
#[derive(Debug)]
pub struct TripoSrImageEncoder {
    encoder: DinoV2Encoder,
    device: Device,
}

impl TripoSrImageEncoder {
    /// Load the `DINOv2`-base encoder from a safetensors file.
    ///
    /// The file is memory-mapped via
    /// [`VarBuilder::from_mmaped_safetensors`], so weights stream from
    /// disk on demand rather than being copied up-front.
    #[allow(unsafe_code)]
    pub fn load(weights_path: &Path, device: &Device) -> Result<Self, TripoSrEncoderError> {
        if !weights_path.exists() {
            return Err(TripoSrEncoderError::ModelLoad(format!(
                "DINOv2 weights file does not exist: {}",
                weights_path.display()
            )));
        }
        // SAFETY: `VarBuilder::from_mmaped_safetensors` is `unsafe`
        // because the underlying mmap relies on the file not being
        // mutated for the lifetime of the resulting `VarBuilder`. The
        // safetensors checkpoint here is opened read-only and is not
        // externally modified by Blazen, matching the standard
        // candle-backend pattern (see e.g. `blazen-audio-tts/.../bark`).
        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, device) }
                .map_err(|e| {
                    TripoSrEncoderError::ModelLoad(format!(
                        "failed to mmap safetensors {}: {e}",
                        weights_path.display()
                    ))
                })?;
        let encoder = DinoV2Encoder::load(&vb, DinoV2Config::base_default())?;
        Ok(Self {
            encoder,
            device: device.clone(),
        })
    }

    /// Encode a pre-decoded RGB image.
    ///
    /// `image_rgb` is expected to be in standard interleaved byte order
    /// (`R, G, B, R, G, B, ...`) at `(height, width, 3)` -- exactly
    /// what `image::DynamicImage::to_rgb8().into_raw()` produces. The
    /// caller is responsible for the PNG / JPEG decode; this method
    /// only resizes and normalises.
    ///
    /// Returns the encoder output tensor on the device the encoder was
    /// loaded onto.
    pub fn encode(
        &self,
        image_rgb: &[u8],
        width: u32,
        height: u32,
    ) -> Result<Tensor, TripoSrEncoderError> {
        let input = self.preprocess(image_rgb, width, height)?;
        let out = self.encoder.encode(&input)?;
        Ok(out)
    }

    /// Preprocess pre-decoded RGB pixel data into a normalised
    /// `(1, 3, 518, 518)` f32 tensor on the encoder's device.
    ///
    /// Exposed at `pub(crate)` so the pipeline-level integration in
    /// sibling modules can re-use it (and so the inline tests can
    /// exercise it without a real safetensors checkpoint).
    pub(crate) fn preprocess(
        &self,
        image_rgb: &[u8],
        width: u32,
        height: u32,
    ) -> Result<Tensor, TripoSrEncoderError> {
        preprocess_to_dinov2_tensor(image_rgb, width, height, &self.device)
    }

    /// The device the encoder's weights live on.
    #[must_use]
    pub fn device(&self) -> &Device {
        &self.device
    }
}

/// Preprocess interleaved RGB bytes (`H x W x 3`) into the normalised
/// `(1, 3, 518, 518)` f32 tensor that `DINOv2`-base expects.
///
/// Split out as a free function (rather than a method) so the tests
/// below can exercise it without touching the real `DinoV2Encoder` /
/// loaded weights.
fn preprocess_to_dinov2_tensor(
    image_rgb: &[u8],
    width: u32,
    height: u32,
    device: &Device,
) -> Result<Tensor, TripoSrEncoderError> {
    let w = width as usize;
    let h = height as usize;
    let expected = w
        .checked_mul(h)
        .and_then(|n| n.checked_mul(3))
        .ok_or_else(|| {
            TripoSrEncoderError::Inference(format!(
                "image dimensions overflow usize: width={width}, height={height}"
            ))
        })?;
    if image_rgb.len() != expected {
        return Err(TripoSrEncoderError::Inference(format!(
            "image_rgb length mismatch: got {} bytes, expected {expected} for {width}x{height}x3",
            image_rgb.len()
        )));
    }
    if w == 0 || h == 0 {
        return Err(TripoSrEncoderError::Inference(format!(
            "image dimensions must be non-zero: got {width}x{height}"
        )));
    }

    // Nearest-neighbour resize from (h, w, 3) -> (DINOV2_INPUT_SIZE,
    // DINOV2_INPUT_SIZE, 3), converting to f32 / 255.0 and applying
    // per-channel ImageNet normalisation in the same pass. The output
    // is laid out as a flat channels-first buffer of length
    // `3 * DINOV2_INPUT_SIZE * DINOV2_INPUT_SIZE` so we can hand it
    // straight to `Tensor::from_vec` with shape `(1, 3, S, S)`.
    let s = DINOV2_INPUT_SIZE;
    let plane = s * s;
    let mut chw = vec![0.0_f32; 3 * plane];

    let scale_x = w as f32 / s as f32;
    let scale_y = h as f32 / s as f32;

    for dy in 0..s {
        let src_row_f = (dy as f32 + 0.5) * scale_y - 0.5;
        let src_row = clamp_index(src_row_f, h);
        for dx in 0..s {
            let src_col_f = (dx as f32 + 0.5) * scale_x - 0.5;
            let src_col = clamp_index(src_col_f, w);
            let src_off = (src_row * w + src_col) * 3;
            // Channels-first: chw[c * plane + dy * s + dx]
            let dst_pixel = dy * s + dx;
            for c in 0..3 {
                let v = f32::from(image_rgb[src_off + c]) / 255.0;
                let normed = (v - IMAGENET_MEAN[c]) / IMAGENET_STD[c];
                chw[c * plane + dst_pixel] = normed;
            }
        }
    }

    let tensor =
        Tensor::from_vec(chw, (1, 3, s, s), device).map_err(TripoSrEncoderError::Candle)?;
    Ok(tensor)
}

/// Round a (potentially negative) source coordinate to the nearest
/// in-bounds integer index for nearest-neighbour sampling.
fn clamp_index(coord: f32, len: usize) -> usize {
    if coord <= 0.0 {
        return 0;
    }
    let idx = coord.round() as isize;
    if idx < 0 {
        0
    } else if (idx as usize) >= len {
        len - 1
    } else {
        idx as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::IndexOp;

    #[test]
    fn preprocess_resizes_to_dinov2_input_size() {
        let device = Device::Cpu;
        // 256x256 dummy image — gradient so each pixel differs.
        let (w, h) = (256_u32, 256_u32);
        let mut img = Vec::with_capacity((w * h * 3) as usize);
        for y in 0..h {
            for x in 0..w {
                img.push((x % 256) as u8);
                img.push((y % 256) as u8);
                img.push(((x + y) % 256) as u8);
            }
        }
        let t = preprocess_to_dinov2_tensor(&img, w, h, &device).expect("preprocess ok");
        assert_eq!(
            t.dims(),
            &[1, 3, DINOV2_INPUT_SIZE, DINOV2_INPUT_SIZE],
            "expected (1, 3, 518, 518); got {:?}",
            t.dims()
        );
    }

    #[test]
    fn preprocess_applies_imagenet_normalization() {
        let device = Device::Cpu;
        // 14x14 all-white image (255 per byte).
        let (w, h) = (14_u32, 14_u32);
        let img = vec![255_u8; (w * h * 3) as usize];
        let t = preprocess_to_dinov2_tensor(&img, w, h, &device).expect("preprocess ok");

        // For an all-white input every channel should be a constant
        // `(1.0 - mean[c]) / std[c]` after normalisation, so the mean
        // across the (518 x 518) plane equals that constant.
        let plane_mean: Vec<f32> = (0..3)
            .map(|c| {
                let channel = t
                    .i((0, c, .., ..))
                    .expect("slice channel")
                    .to_vec2::<f32>()
                    .expect("to_vec2");
                let n = channel.len() * channel[0].len();
                let sum: f32 = channel.iter().flatten().copied().sum();
                sum / n as f32
            })
            .collect();

        for c in 0..3 {
            let expected = (1.0 - IMAGENET_MEAN[c]) / IMAGENET_STD[c];
            assert!(
                (plane_mean[c] - expected).abs() < 0.01,
                "channel {c}: expected {expected}, got {} (delta {})",
                plane_mean[c],
                (plane_mean[c] - expected).abs()
            );
        }
    }

    #[test]
    fn preprocess_rejects_byte_length_mismatch() {
        let device = Device::Cpu;
        let img = vec![0_u8; 10]; // way too small for any non-trivial (w, h)
        let err = preprocess_to_dinov2_tensor(&img, 16, 16, &device).unwrap_err();
        match err {
            TripoSrEncoderError::Inference(msg) => {
                assert!(msg.contains("length mismatch"), "unexpected message: {msg}");
            }
            other => panic!("expected Inference variant, got {other:?}"),
        }
    }

    #[test]
    fn load_surfaces_missing_file_as_model_load_error() {
        let bogus = Path::new("/nonexistent/blazen-triposr-dinov2-does-not-exist.safetensors");
        let device = Device::Cpu;
        let err = TripoSrImageEncoder::load(bogus, &device).unwrap_err();
        match err {
            TripoSrEncoderError::ModelLoad(msg) => {
                assert!(msg.contains("does not exist"), "unexpected message: {msg}");
            }
            other => panic!("expected ModelLoad, got {other:?}"),
        }
    }
}
