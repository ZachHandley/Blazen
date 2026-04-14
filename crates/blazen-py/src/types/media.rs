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

// ---------------------------------------------------------------------------
// Generated media output wrappers
// ---------------------------------------------------------------------------

use blazen_llm::media::{
    Generated3DModel, GeneratedAudio, GeneratedImage, GeneratedVideo, MediaOutput,
};

/// A single piece of generated media content.
///
/// At least one of `url`, `base64`, or `raw_content` will be populated.
/// `raw_content` is used for text-based formats like SVG, OBJ, and GLTF JSON.
#[gen_stub_pyclass]
#[pyclass(name = "MediaOutput", frozen)]
pub struct PyMediaOutput {
    pub(crate) inner: MediaOutput,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyMediaOutput {
    /// URL where the media can be downloaded.
    #[getter]
    fn url(&self) -> Option<String> {
        self.inner.url.clone()
    }

    /// Base64-encoded media data.
    #[getter]
    fn base64(&self) -> Option<String> {
        self.inner.base64.clone()
    }

    /// Raw text content for text-based formats (SVG, OBJ, GLTF JSON).
    #[getter]
    fn raw_content(&self) -> Option<String> {
        self.inner.raw_content.clone()
    }

    /// The media type as a MIME string (e.g. `"image/png"`).
    #[getter]
    fn media_type(&self) -> String {
        self.inner.media_type.mime().to_owned()
    }

    /// File size in bytes, if known.
    #[getter]
    fn file_size(&self) -> Option<u64> {
        self.inner.file_size
    }

    /// Arbitrary provider-specific metadata as a Python dict.
    #[getter]
    #[gen_stub(override_return_type(type_repr = "dict[str, typing.Any]", imports = ("typing",)))]
    fn metadata(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        crate::convert::json_to_py(py, &self.inner.metadata)
    }

    fn __repr__(&self) -> String {
        format!(
            "MediaOutput(media_type={:?}, has_url={})",
            self.inner.media_type.mime(),
            self.inner.url.is_some()
        )
    }
}

py_result_type!(
    /// A single generated image with optional dimension metadata.
    "GeneratedImage", PyGeneratedImage, GeneratedImage,
    fields: {
        /// The underlying media output.
        media: wrap(PyMediaOutput),
        /// Image width in pixels, if known.
        width: copy(Option<u32>),
        /// Image height in pixels, if known.
        height: copy(Option<u32>),
    },
    repr: "GeneratedImage(width={:?}, height={:?})", width, height,
);

py_result_type!(
    /// A single generated video with optional metadata.
    "GeneratedVideo", PyGeneratedVideo, GeneratedVideo,
    fields: {
        /// The underlying media output.
        media: wrap(PyMediaOutput),
        /// Video width in pixels, if known.
        width: copy(Option<u32>),
        /// Video height in pixels, if known.
        height: copy(Option<u32>),
        /// Duration in seconds, if known.
        duration_seconds: copy(Option<f32>),
        /// Frames per second, if known.
        fps: copy(Option<f32>),
    },
    repr: "GeneratedVideo(width={:?}, height={:?}, duration_seconds={:?}, fps={:?})",
        width, height, duration_seconds, fps,
);

py_result_type!(
    /// A single generated audio clip with optional metadata.
    "GeneratedAudio", PyGeneratedAudio, GeneratedAudio,
    fields: {
        /// The underlying media output.
        media: wrap(PyMediaOutput),
        /// Duration in seconds, if known.
        duration_seconds: copy(Option<f32>),
        /// Sample rate in Hz, if known.
        sample_rate: copy(Option<u32>),
        /// Number of audio channels, if known.
        channels: copy(Option<u8>),
    },
    repr: "GeneratedAudio(duration_seconds={:?}, sample_rate={:?}, channels={:?})",
        duration_seconds, sample_rate, channels,
);

py_result_type!(
    /// A single generated 3D model with optional mesh metadata.
    "Generated3DModel", PyGenerated3DModel, Generated3DModel,
    fields: {
        /// The underlying media output.
        media: wrap(PyMediaOutput),
        /// Total vertex count, if known.
        vertex_count: copy(Option<u64>),
        /// Total face/triangle count, if known.
        face_count: copy(Option<u64>),
        /// Whether the model includes texture data.
        has_textures: copy(bool),
        /// Whether the model includes animation data.
        has_animations: copy(bool),
    },
    repr: "Generated3DModel(vertex_count={:?}, face_count={:?}, has_textures={}, has_animations={})",
        vertex_count, face_count, has_textures, has_animations,
);
