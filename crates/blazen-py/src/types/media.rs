//! Python wrappers for media types and generated media output types.

use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

// ---------------------------------------------------------------------------
// PyMediaType
// ---------------------------------------------------------------------------

/// Media type constants for identifying file formats.
///
/// Example:
///     >>> MediaType.PNG   # "image/png"
///     >>> MediaType.MP4   # "video/mp4"
///     >>> MediaType.MP3   # "audio/mpeg"
#[gen_stub_pyclass]
#[pyclass(name = "MediaType", frozen)]
pub struct PyMediaType;

#[gen_stub_pymethods]
#[pymethods]
impl PyMediaType {
    // Images
    #[classattr]
    const PNG: &'static str = "image/png";
    #[classattr]
    const JPEG: &'static str = "image/jpeg";
    #[classattr]
    const WEBP: &'static str = "image/webp";
    #[classattr]
    const GIF: &'static str = "image/gif";
    #[classattr]
    const SVG: &'static str = "image/svg+xml";
    #[classattr]
    const BMP: &'static str = "image/bmp";
    #[classattr]
    const TIFF: &'static str = "image/tiff";
    #[classattr]
    const AVIF: &'static str = "image/avif";

    // Video
    #[classattr]
    const MP4: &'static str = "video/mp4";
    #[classattr]
    const WEBM: &'static str = "video/webm";
    #[classattr]
    const MOV: &'static str = "video/quicktime";

    // Audio
    #[classattr]
    const MP3: &'static str = "audio/mpeg";
    #[classattr]
    const WAV: &'static str = "audio/wav";
    #[classattr]
    const OGG: &'static str = "audio/ogg";
    #[classattr]
    const FLAC: &'static str = "audio/flac";
    #[classattr]
    const AAC: &'static str = "audio/aac";
    #[classattr]
    const M4A: &'static str = "audio/mp4";

    // 3D Models
    #[classattr]
    const GLB: &'static str = "model/gltf-binary";
    #[classattr]
    const GLTF: &'static str = "model/gltf+json";
    #[classattr]
    const OBJ: &'static str = "model/obj";
    #[classattr]
    const USDZ: &'static str = "model/vnd.usdz+zip";
    #[classattr]
    const FBX: &'static str = "application/octet-stream";
    #[classattr]
    const STL: &'static str = "model/stl";

    // Documents
    #[classattr]
    const PDF: &'static str = "application/pdf";
}

// Re-export core media types directly.
pub use blazen_llm::media::{
    Generated3DModel, GeneratedAudio, GeneratedImage, GeneratedVideo, MediaOutput,
};
