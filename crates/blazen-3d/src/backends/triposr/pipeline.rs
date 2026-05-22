//! End-to-end orchestration: image -> `DINOv2` -> triplane -> `NeRF` -> marching cubes -> GLB (Wave T.2).
//!
//! This module owns the public `TripoSrPipeline` surface that the
//! eventual `impl blazen_llm::compute::traits::ThreeDGeneration for
//! TripoSrBackend` block (landed from `blazen-llm`, see
//! [`crate::backends::triposr`] cycle-resolution notes) drives.
//!
//! # End-to-end flow
//!
//! 1. Decode caller-supplied `RGB` bytes -> `DINOv2` image-token tensor
//!    via [`super::image_encoder::TripoSrImageEncoder::encode`].
//! 2. Project image tokens into a [`blazen_3d_core::triplane::Triplane`]
//!    via [`super::triplane_transformer::TripoSrTransformer::forward`].
//! 3. Sample a `(R, R, R)` density grid from the triplane via the
//!    `NeRF` field (`super::nerf_field`) and feed it to
//!    [`blazen_3d_core::marching_cubes::MarchingCubes::extract`].
//! 4. Re-query the `NeRF` field at every extracted vertex to grab a
//!    vertex colour, then pack `POSITION + COLOR_0 + INDICES` into a
//!    minimal glTF 2.0 binary (GLB) container via [`pack_glb`].
//!
//! # GLB packing strategy
//!
//! The output is a tiny single-mesh GLB: one `POSITION` vec3 f32
//! attribute, one `COLOR_0` vec3 u8 normalised attribute, and a flat
//! triangle-list `u32` index buffer. No textures, no materials, no
//! animations, no skins. Pulling in a full glTF writer crate for this
//! footprint would dwarf the actual mesh data on disk and add a
//! transitive maintenance surface, so we hand-roll the writer here in
//! ~150 LOC of pure stdlib (the JSON chunk goes through `serde_json`,
//! which is already a workspace dep, and the binary chunk is just
//! `f32::to_le_bytes` / `u32::to_le_bytes` concatenation).
//!
//! # Wave T.2 wiring status
//!
//! The sibling [`super::image_encoder`] and
//! [`super::triplane_transformer`] modules ship their full Wave T.2
//! implementations. The remaining components -- the safetensors weight
//! loader (`super::weights`) and the triplane-conditioned `NeRF` field
//! (`super::nerf_field`) -- are scheduled for follow-up commits inside
//! the same wave. Until those land, [`TripoSrPipeline::load_from_paths`]
//! and [`TripoSrPipeline::load_from_hf`] surface
//! [`TripoSrPipelineError::Weights`] with a clear "pending Wave T.2"
//! message; [`TripoSrPipeline::image_to_glb`] still validates its input
//! dimensions eagerly so callers get a fast error on obviously-bad
//! requests. [`pack_glb`] itself is fully implemented and inline-tested
//! -- once the `NeRF` + weights modules land, only the body of
//! `image_to_glb` needs to be filled in; the public surface of this
//! module stays stable.

// Wave T.2 lands the consumer pipeline ahead of the NeRF + weights
// sibling modules; the public types here are intended to be the public
// surface of the backend even before the full pipeline body is wired,
// so keep the symbols visible to consumers in `mod.rs`.
#![allow(dead_code)]

use std::path::Path;

use candle_core::Device;
use thiserror::Error;

/// Errors raised by [`TripoSrPipeline`] construction and orchestration.
#[derive(Debug, Error)]
pub enum TripoSrPipelineError {
    /// Filesystem error (missing weight file, unreadable cache dir, ...).
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    /// Image-encoder construction or forward pass failed. Wraps
    /// [`super::image_encoder::TripoSrEncoderError`] as a flat string so
    /// the error type stays `Send + Sync + 'static` without pulling the
    /// underlying error's transitive `From` impls into the pipeline
    /// surface.
    #[error("image encoder error: {0}")]
    Encoder(String),
    /// Triplane transformer construction or forward pass failed.
    #[error("triplane transformer error: {0}")]
    Transformer(String),
    /// `NeRF` field construction or query failed.
    #[error("nerf field error: {0}")]
    NerfField(String),
    /// Weight loading (local file or HF download) failed.
    #[error("weights error: {0}")]
    Weights(String),
    /// Marching-cubes mesh extraction failed.
    #[error("marching cubes error: {0}")]
    MarchingCubes(String),
    /// GLB packing failed (overflow in chunk lengths, bad input lengths,
    /// JSON encoding error).
    #[error("glb pack error: {0}")]
    GltfPack(String),
    /// A candle tensor / module operation failed.
    #[error("candle error: {0}")]
    Candle(#[from] candle_core::Error),
}

/// End-to-end `TripoSR` orchestration pipeline.
///
/// Owns the candle [`Device`] every component runs on, plus (eventually)
/// the loaded image encoder, triplane transformer, and `NeRF` field
/// handles. Cheap to wrap in [`std::sync::Arc`] for cross-clone sharing
/// once the component modules land.
///
/// See the module-level docs for the Wave T.2 wiring status.
#[derive(Debug)]
pub struct TripoSrPipeline {
    image_encoder: super::image_encoder::TripoSrImageEncoder,
    transformer: super::triplane_transformer::TripoSrTransformer,
    nerf_field: super::nerf_field::TripoSrNerfField,
    device: Device,
}

impl TripoSrPipeline {
    /// Load the full pipeline from a directory of local weight files.
    ///
    /// `weights_path` is the directory containing the `DINOv2`, triplane
    /// transformer, and `NeRF` safetensors checkpoints. The weights loader
    /// in `super::weights` resolves the individual file names off this
    /// root.
    ///
    /// # Errors
    ///
    /// Returns [`TripoSrPipelineError::Weights`] while the
    /// `super::weights` module is still pending in Wave T.2. Once it
    /// lands, returns the matching `Encoder` / `Transformer` /
    /// `NerfField` variants when an individual component fails to load.
    pub fn load_from_paths(
        weights_path: &Path,
        device: &Device,
    ) -> Result<Self, TripoSrPipelineError> {
        let weights = super::weights::load_weights_from_path(weights_path, device)
            .map_err(|e| TripoSrPipelineError::Weights(e.to_string()))?;
        let image_encoder = super::image_encoder::TripoSrImageEncoder::load_from_var_builder(
            weights.image_encoder_vb,
            device,
        )
        .map_err(|e| TripoSrPipelineError::Encoder(e.to_string()))?;
        let transformer = super::triplane_transformer::TripoSrTransformer::load_from_var_builder(
            weights.transformer_vb,
            super::triplane_transformer::TripoSrTransformerConfig::base_default(),
        )
        .map_err(|e| TripoSrPipelineError::Transformer(e.to_string()))?;
        let nerf_field = super::nerf_field::TripoSrNerfField::load_from_var_builder(
            weights.nerf_field_vb,
            super::nerf_field::NerfFieldConfig::base_default(),
        )
        .map_err(|e| TripoSrPipelineError::NerfField(e.to_string()))?;
        Ok(Self {
            image_encoder,
            transformer,
            nerf_field,
            device: device.clone(),
        })
    }

    /// Download `TripoSR` weights from Hugging Face and load the pipeline.
    ///
    /// `hf_repo_id` is the HF repo (e.g. `"stabilityai/TripoSR"`).
    /// `revision` pins a specific git revision / tag / branch on the
    /// repo; `None` defaults to the repo's `main` branch.
    ///
    /// # Errors
    ///
    /// Returns [`TripoSrPipelineError::Weights`] while the
    /// `super::weights::load_weights_from_hf` helper is still pending
    /// in Wave T.2.
    //
    // The signature is intentionally `async` to match the eventual
    // `super::weights::load_weights_from_hf` HF-download call: Wave T.2
    // will fill in that body and the signature here will stay stable.
    pub async fn load_from_hf(
        hf_repo_id: &str,
        revision: Option<&str>,
        device: &Device,
    ) -> Result<Self, TripoSrPipelineError> {
        let weights = super::weights::load_weights(hf_repo_id, revision, device)
            .await
            .map_err(|e| TripoSrPipelineError::Weights(e.to_string()))?;
        let image_encoder = super::image_encoder::TripoSrImageEncoder::load_from_var_builder(
            weights.image_encoder_vb,
            device,
        )
        .map_err(|e| TripoSrPipelineError::Encoder(e.to_string()))?;
        let transformer = super::triplane_transformer::TripoSrTransformer::load_from_var_builder(
            weights.transformer_vb,
            super::triplane_transformer::TripoSrTransformerConfig::base_default(),
        )
        .map_err(|e| TripoSrPipelineError::Transformer(e.to_string()))?;
        let nerf_field = super::nerf_field::TripoSrNerfField::load_from_var_builder(
            weights.nerf_field_vb,
            super::nerf_field::NerfFieldConfig::base_default(),
        )
        .map_err(|e| TripoSrPipelineError::NerfField(e.to_string()))?;
        Ok(Self {
            image_encoder,
            transformer,
            nerf_field,
            device: device.clone(),
        })
    }

    /// Run the full image -> GLB pipeline.
    ///
    /// `image_rgb` is interleaved `R, G, B, R, G, B, ...` byte order at
    /// `(height, width, 3)` -- exactly what
    /// `image::DynamicImage::to_rgb8().into_raw()` produces.
    /// `mesh_resolution` is the side length of the density grid sampled
    /// from the triplane; `256` matches the upstream `TripoSR`
    /// reference.
    ///
    /// # Errors
    ///
    /// - [`TripoSrPipelineError::Encoder`] when `width == 0`,
    ///   `height == 0`, or `image_rgb.len()` does not equal
    ///   `width * height * 3`.
    /// - [`TripoSrPipelineError::NerfField`] while the triplane `NeRF`
    ///   module is still pending in Wave T.2.
    //
    // `&self` is unused today (every meaningful code path errors out
    // before touching the device) but the eventual full body needs
    // self.image_encoder / self.transformer / self.nerf_field, so the
    // receiver is part of the stable public surface.
    pub fn image_to_glb(
        &self,
        image_rgb: &[u8],
        width: u32,
        height: u32,
        mesh_resolution: usize,
    ) -> Result<Vec<u8>, TripoSrPipelineError> {
        // Fast-fail on obviously-invalid input *before* touching any of
        // the unimplemented downstream components. Matches the
        // validation contract of `TripoSrImageEncoder::preprocess` and
        // gives callers a clear error on zero-sized inputs even while
        // the rest of the pipeline body is pending Wave T.2.
        if width == 0 || height == 0 {
            return Err(TripoSrPipelineError::Encoder(format!(
                "image dimensions must be non-zero: got {width}x{height}"
            )));
        }
        let expected = (width as usize)
            .checked_mul(height as usize)
            .and_then(|n| n.checked_mul(3))
            .ok_or_else(|| {
                TripoSrPipelineError::Encoder(format!(
                    "image dimensions overflow usize: width={width}, height={height}"
                ))
            })?;
        if image_rgb.len() != expected {
            return Err(TripoSrPipelineError::Encoder(format!(
                "image_rgb length mismatch: got {} bytes, expected {expected} for {width}x{height}x3",
                image_rgb.len()
            )));
        }
        if mesh_resolution < 2 {
            return Err(TripoSrPipelineError::MarchingCubes(format!(
                "mesh_resolution must be >= 2; got {mesh_resolution}"
            )));
        }

        // 1. Image -> DINOv2 tokens.
        let image_tokens = self
            .image_encoder
            .encode(image_rgb, width, height)
            .map_err(|e| TripoSrPipelineError::Encoder(e.to_string()))?;

        // 2. Tokens -> Triplane via cross-attention transformer.
        let triplane = self
            .transformer
            .forward(&image_tokens)
            .map_err(|e| TripoSrPipelineError::Transformer(e.to_string()))?;

        // 3. Density grid on the unit cube; marching cubes -> mesh.
        let density = self
            .nerf_field
            .density_grid(&triplane, mesh_resolution)
            .map_err(|e| TripoSrPipelineError::NerfField(e.to_string()))?;
        let (vertices, indices) = blazen_3d_core::marching_cubes::MarchingCubes::extract(
            &density,
            0.5,
            ([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]),
        )
        .map_err(|e| TripoSrPipelineError::MarchingCubes(e.to_string()))?;

        // 4. Color sample at vertex positions via NeRF field.
        let n = vertices.len();
        let mut flat = Vec::with_capacity(n * 3);
        for v in &vertices {
            flat.extend_from_slice(v);
        }
        let vertex_points = candle_core::Tensor::from_vec(flat, (n, 3), &self.device)
            .map_err(TripoSrPipelineError::Candle)?;
        let nerf_out = self
            .nerf_field
            .query(&triplane, &vertex_points)
            .map_err(|e| TripoSrPipelineError::NerfField(e.to_string()))?;
        let colors_u8 = nerf_out
            .color
            .clamp(0.0_f32, 1.0_f32)
            .map_err(TripoSrPipelineError::Candle)?
            .affine(255.0, 0.0)
            .map_err(TripoSrPipelineError::Candle)?
            .to_dtype(candle_core::DType::U8)
            .map_err(TripoSrPipelineError::Candle)?
            .flatten_all()
            .map_err(TripoSrPipelineError::Candle)?
            .to_vec1::<u8>()
            .map_err(TripoSrPipelineError::Candle)?;

        // 5. Pack into GLB.
        pack_glb(&vertices, &indices, &colors_u8)
    }

    /// The candle device every loaded component runs on.
    #[must_use]
    pub fn device(&self) -> &Device {
        &self.device
    }
}

// ---------------------------------------------------------------------------
// GLB packing
// ---------------------------------------------------------------------------

/// glTF 2.0 binary container magic (`"glTF"` in ASCII).
const GLB_MAGIC: u32 = 0x4654_6C67;
/// glTF binary container version we emit.
const GLB_VERSION: u32 = 2;
/// Chunk type tag for the JSON chunk (`"JSON"` in ASCII, little-endian).
const GLB_CHUNK_TYPE_JSON: u32 = 0x4E4F_534A;
/// Chunk type tag for the binary chunk (`"BIN\0"` in ASCII, little-endian).
const GLB_CHUNK_TYPE_BIN: u32 = 0x004E_4942;

/// glTF accessor `componentType` constants.
mod gl {
    /// Unsigned byte (`GL_UNSIGNED_BYTE`).
    pub const UNSIGNED_BYTE: u32 = 5121;
    /// Unsigned int (`GL_UNSIGNED_INT`).
    pub const UNSIGNED_INT: u32 = 5125;
    /// Single-precision float (`GL_FLOAT`).
    pub const FLOAT: u32 = 5126;
    /// glTF bufferView target: `ARRAY_BUFFER` (vertex attributes).
    pub const ARRAY_BUFFER: u32 = 34_962;
    /// glTF bufferView target: `ELEMENT_ARRAY_BUFFER` (indices).
    pub const ELEMENT_ARRAY_BUFFER: u32 = 34_963;
}

/// Pack a vertex-coloured triangle mesh into a self-contained GLB byte
/// stream.
///
/// `vertices` is the position list (one `[x, y, z]` per vertex, f32),
/// `indices` is a flat triangle-list buffer (each consecutive triple of
/// `u32` is one CCW triangle), and `colors` is the flat per-vertex
/// `R, G, B` byte buffer (`3 * vertices.len()` entries). Returns the
/// raw GLB binary, ready to be written straight to disk or shipped over
/// the wire.
///
/// # Errors
///
/// Returns [`TripoSrPipelineError::GltfPack`] when the input buffers
/// have inconsistent lengths, when the index buffer is not a multiple
/// of 3, when an internal chunk length overflows `u32`, or when the
/// embedded JSON cannot be encoded.
//
// The body is intentionally one straight-line function (validate ->
// build BIN -> build JSON -> assemble GLB) so the offsets / chunk
// boundaries stay obviously consistent. Splitting it just to satisfy
// the 100-line ceiling would force shared mutable state across helpers
// without making the spec mapping easier to audit.
#[allow(clippy::too_many_lines)]
pub fn pack_glb(
    vertices: &[[f32; 3]],
    indices: &[u32],
    colors: &[u8],
) -> Result<Vec<u8>, TripoSrPipelineError> {
    if vertices.is_empty() {
        return Err(TripoSrPipelineError::GltfPack(
            "vertices buffer must be non-empty".to_owned(),
        ));
    }
    if indices.is_empty() {
        return Err(TripoSrPipelineError::GltfPack(
            "indices buffer must be non-empty".to_owned(),
        ));
    }
    if !indices.len().is_multiple_of(3) {
        return Err(TripoSrPipelineError::GltfPack(format!(
            "indices length must be a multiple of 3; got {}",
            indices.len()
        )));
    }
    let expected_color_bytes = vertices.len().checked_mul(3).ok_or_else(|| {
        TripoSrPipelineError::GltfPack(format!(
            "vertex count overflows usize when sized for COLOR_0: {}",
            vertices.len()
        ))
    })?;
    if colors.len() != expected_color_bytes {
        return Err(TripoSrPipelineError::GltfPack(format!(
            "colors length mismatch: got {}, expected {expected_color_bytes} for {} RGB vertices",
            colors.len(),
            vertices.len()
        )));
    }
    let max_index = u32::try_from(vertices.len()).map_err(|_| {
        TripoSrPipelineError::GltfPack(format!(
            "vertex count {} exceeds GLB u32 index range",
            vertices.len()
        ))
    })?;
    if let Some(&bad) = indices.iter().find(|&&i| i >= max_index) {
        return Err(TripoSrPipelineError::GltfPack(format!(
            "index {bad} out of range for {} vertices",
            vertices.len()
        )));
    }

    // ---- Build the binary chunk: positions + colours + indices. ----
    //
    // glTF requires each bufferView's `byteOffset` to be aligned to the
    // size of its component type. Positions (f32) are 4-byte aligned by
    // construction; colours (u8) have no alignment constraint; indices
    // (u32) are 4-byte aligned, which means the colour view (which is
    // `3 * vertex_count` bytes long) must be padded out to a 4-byte
    // boundary before the index view starts.
    let mut bin: Vec<u8> = Vec::new();

    let positions_offset: u32 = 0;
    for v in vertices {
        for component in v {
            bin.extend_from_slice(&component.to_le_bytes());
        }
    }
    let positions_len = u32::try_from(bin.len() - positions_offset as usize).map_err(|_| {
        TripoSrPipelineError::GltfPack("positions byteLength overflows u32".to_owned())
    })?;

    let colors_offset = u32::try_from(bin.len()).map_err(|_| {
        TripoSrPipelineError::GltfPack("colors byteOffset overflows u32".to_owned())
    })?;
    bin.extend_from_slice(colors);
    let colors_len_raw = u32::try_from(bin.len() - colors_offset as usize).map_err(|_| {
        TripoSrPipelineError::GltfPack("colors byteLength overflows u32".to_owned())
    })?;
    // Pad to a 4-byte boundary so the following u32 index view aligns.
    while !bin.len().is_multiple_of(4) {
        bin.push(0);
    }

    let indices_offset = u32::try_from(bin.len()).map_err(|_| {
        TripoSrPipelineError::GltfPack("indices byteOffset overflows u32".to_owned())
    })?;
    for &i in indices {
        bin.extend_from_slice(&i.to_le_bytes());
    }
    let indices_len = u32::try_from(bin.len() - indices_offset as usize).map_err(|_| {
        TripoSrPipelineError::GltfPack("indices byteLength overflows u32".to_owned())
    })?;

    let bin_total_len = u32::try_from(bin.len())
        .map_err(|_| TripoSrPipelineError::GltfPack("binary chunk overflows u32".to_owned()))?;

    // Compute POSITION min/max -- glTF spec requires these on
    // POSITION accessors so loaders can build a bounding box without
    // walking the whole vertex buffer.
    let (pos_min, pos_max) = vertex_bounds(vertices);

    // ---- Build the JSON chunk. ----
    let json = serde_json::json!({
        "asset": { "version": "2.0", "generator": "blazen-3d/triposr" },
        "scene": 0,
        "scenes": [ { "nodes": [0] } ],
        "nodes":  [ { "mesh": 0 } ],
        "meshes": [ {
            "primitives": [ {
                "attributes": {
                    "POSITION": 0,
                    "COLOR_0": 1
                },
                "indices": 2,
                "mode": 4
            } ]
        } ],
        "accessors": [
            {
                "bufferView": 0,
                "byteOffset": 0,
                "componentType": gl::FLOAT,
                "count": vertices.len(),
                "type": "VEC3",
                "min": pos_min,
                "max": pos_max
            },
            {
                "bufferView": 1,
                "byteOffset": 0,
                "componentType": gl::UNSIGNED_BYTE,
                "normalized": true,
                "count": vertices.len(),
                "type": "VEC3"
            },
            {
                "bufferView": 2,
                "byteOffset": 0,
                "componentType": gl::UNSIGNED_INT,
                "count": indices.len(),
                "type": "SCALAR"
            }
        ],
        "bufferViews": [
            {
                "buffer": 0,
                "byteOffset": positions_offset,
                "byteLength": positions_len,
                "target": gl::ARRAY_BUFFER
            },
            {
                "buffer": 0,
                "byteOffset": colors_offset,
                "byteLength": colors_len_raw,
                "target": gl::ARRAY_BUFFER
            },
            {
                "buffer": 0,
                "byteOffset": indices_offset,
                "byteLength": indices_len,
                "target": gl::ELEMENT_ARRAY_BUFFER
            }
        ],
        "buffers": [ { "byteLength": bin_total_len } ]
    });
    let mut json_bytes = serde_json::to_vec(&json).map_err(|e| {
        TripoSrPipelineError::GltfPack(format!("failed to encode glTF JSON chunk: {e}"))
    })?;
    // Pad the JSON chunk with ASCII spaces to a 4-byte boundary
    // (glTF binary spec section 4.4.2).
    while !json_bytes.len().is_multiple_of(4) {
        json_bytes.push(b' ');
    }
    let json_len = u32::try_from(json_bytes.len())
        .map_err(|_| TripoSrPipelineError::GltfPack("JSON chunk overflows u32".to_owned()))?;

    // ---- Assemble the full GLB. ----
    //
    // Layout: 12-byte header + (8-byte chunk header + JSON bytes)
    //                       + (8-byte chunk header + BIN bytes).
    let total_len = 12_u64 + 8 + u64::from(json_len) + 8 + u64::from(bin_total_len);
    let total_len_u32 = u32::try_from(total_len).map_err(|_| {
        TripoSrPipelineError::GltfPack(format!("total GLB length {total_len} overflows u32"))
    })?;

    let cap = usize::try_from(total_len).map_err(|_| {
        TripoSrPipelineError::GltfPack(format!(
            "total GLB length {total_len} does not fit in usize on this platform"
        ))
    })?;
    let mut out = Vec::with_capacity(cap);
    out.extend_from_slice(&GLB_MAGIC.to_le_bytes());
    out.extend_from_slice(&GLB_VERSION.to_le_bytes());
    out.extend_from_slice(&total_len_u32.to_le_bytes());

    // JSON chunk.
    out.extend_from_slice(&json_len.to_le_bytes());
    out.extend_from_slice(&GLB_CHUNK_TYPE_JSON.to_le_bytes());
    out.extend_from_slice(&json_bytes);

    // Binary chunk.
    out.extend_from_slice(&bin_total_len.to_le_bytes());
    out.extend_from_slice(&GLB_CHUNK_TYPE_BIN.to_le_bytes());
    out.extend_from_slice(&bin);

    debug_assert_eq!(
        out.len() as u64,
        total_len,
        "assembled GLB length disagrees with header"
    );
    Ok(out)
}

/// Compute the component-wise min and max of a vertex list (returned as
/// `Vec` rather than `[f32; 3]` so they serialise straight into the
/// JSON accessor's expected `[x, y, z]` array form).
fn vertex_bounds(vertices: &[[f32; 3]]) -> (Vec<f32>, Vec<f32>) {
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    for v in vertices {
        for c in 0..3 {
            if v[c] < min[c] {
                min[c] = v[c];
            }
            if v[c] > max[c] {
                max[c] = v[c];
            }
        }
    }
    (min.to_vec(), max.to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 4-vertex tetrahedron with hand-picked positions and per-vertex
    /// colours. Used by the GLB-shape test below.
    fn tetrahedron() -> (Vec<[f32; 3]>, Vec<u32>, Vec<u8>) {
        let vertices = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        let indices = vec![0, 1, 2, 2, 1, 3, 0, 2, 3, 0, 3, 1];
        let colors = vec![
            255, 0, 0, // v0
            0, 255, 0, // v1
            0, 0, 255, // v2
            255, 255, 0, // v3
        ];
        (vertices, indices, colors)
    }

    #[test]
    fn pack_glb_emits_valid_glb_header() {
        let (vertices, indices, colors) = tetrahedron();
        let bytes = pack_glb(&vertices, &indices, &colors).expect("pack_glb ok");

        // 12-byte header + at minimum two 8-byte chunk headers.
        assert!(
            bytes.len() >= 12 + 8 + 8,
            "glb output too short: {} bytes",
            bytes.len()
        );

        // Magic must be the ASCII `glTF` bytes -- not just the u32
        // value, since the spec uses a byte-level magic check.
        assert_eq!(&bytes[0..4], b"glTF", "expected glTF magic prefix");

        // Version.
        let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        assert_eq!(version, 2, "expected glTF version 2; got {version}");

        // Total length in the header must match the actual buffer
        // length.
        let total_len = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        assert_eq!(
            total_len as usize,
            bytes.len(),
            "header total length {total_len} disagrees with actual buffer length {}",
            bytes.len()
        );

        // First chunk header is JSON.
        let json_chunk_len = u32::from_le_bytes(bytes[12..16].try_into().unwrap());
        let json_chunk_type = u32::from_le_bytes(bytes[16..20].try_into().unwrap());
        assert_eq!(
            json_chunk_type, GLB_CHUNK_TYPE_JSON,
            "first chunk must be JSON"
        );
        // JSON chunk length must be a multiple of 4 per spec section
        // 4.4.2 (padded with spaces).
        assert!(
            json_chunk_len.is_multiple_of(4),
            "JSON chunk length {json_chunk_len} must be 4-byte aligned"
        );

        // Second chunk header is BIN, and starts 20 + json_chunk_len bytes in.
        let bin_chunk_off = 20 + json_chunk_len as usize;
        let bin_chunk_type = u32::from_le_bytes(
            bytes[bin_chunk_off + 4..bin_chunk_off + 8]
                .try_into()
                .unwrap(),
        );
        assert_eq!(
            bin_chunk_type, GLB_CHUNK_TYPE_BIN,
            "second chunk must be BIN"
        );
    }

    #[test]
    fn pack_glb_rejects_color_length_mismatch() {
        let (vertices, indices, mut colors) = tetrahedron();
        colors.pop(); // now 11 bytes instead of 12 -- off by one.
        let err = pack_glb(&vertices, &indices, &colors)
            .expect_err("color length mismatch must surface an error");
        match err {
            TripoSrPipelineError::GltfPack(msg) => {
                assert!(msg.contains("colors length mismatch"), "msg={msg}");
            }
            other => panic!("expected GltfPack variant, got {other:?}"),
        }
    }

    #[test]
    fn pack_glb_rejects_out_of_range_index() {
        let (vertices, _indices, colors) = tetrahedron();
        // 4 is one past the last valid index for a 4-vertex mesh.
        let bad_indices = vec![0, 1, 4];
        let err = pack_glb(&vertices, &bad_indices, &colors)
            .expect_err("out-of-range index must surface an error");
        match err {
            TripoSrPipelineError::GltfPack(msg) => {
                assert!(msg.contains("out of range"), "msg={msg}");
            }
            other => panic!("expected GltfPack variant, got {other:?}"),
        }
    }
}
