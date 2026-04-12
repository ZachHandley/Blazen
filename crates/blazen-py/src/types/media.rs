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

/// A single generated image with optional dimension metadata.
#[gen_stub_pyclass]
#[pyclass(name = "GeneratedImage", frozen)]
pub struct PyGeneratedImage {
    pub(crate) inner: GeneratedImage,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyGeneratedImage {
    /// The underlying media output.
    #[getter]
    fn media(&self) -> PyMediaOutput {
        PyMediaOutput {
            inner: self.inner.media.clone(),
        }
    }

    /// Image width in pixels, if known.
    #[getter]
    fn width(&self) -> Option<u32> {
        self.inner.width
    }

    /// Image height in pixels, if known.
    #[getter]
    fn height(&self) -> Option<u32> {
        self.inner.height
    }

    fn __repr__(&self) -> String {
        format!(
            "GeneratedImage(width={:?}, height={:?}, media_type={:?})",
            self.inner.width,
            self.inner.height,
            self.inner.media.media_type.mime(),
        )
    }
}

/// A single generated video with optional metadata.
#[gen_stub_pyclass]
#[pyclass(name = "GeneratedVideo", frozen)]
pub struct PyGeneratedVideo {
    pub(crate) inner: GeneratedVideo,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyGeneratedVideo {
    /// The underlying media output.
    #[getter]
    fn media(&self) -> PyMediaOutput {
        PyMediaOutput {
            inner: self.inner.media.clone(),
        }
    }

    /// Video width in pixels, if known.
    #[getter]
    fn width(&self) -> Option<u32> {
        self.inner.width
    }

    /// Video height in pixels, if known.
    #[getter]
    fn height(&self) -> Option<u32> {
        self.inner.height
    }

    /// Duration in seconds, if known.
    #[getter]
    fn duration_seconds(&self) -> Option<f32> {
        self.inner.duration_seconds
    }

    /// Frames per second, if known.
    #[getter]
    fn fps(&self) -> Option<f32> {
        self.inner.fps
    }

    fn __repr__(&self) -> String {
        format!(
            "GeneratedVideo(width={:?}, height={:?}, duration_seconds={:?}, fps={:?})",
            self.inner.width, self.inner.height, self.inner.duration_seconds, self.inner.fps,
        )
    }
}

/// A single generated audio clip with optional metadata.
#[gen_stub_pyclass]
#[pyclass(name = "GeneratedAudio", frozen)]
pub struct PyGeneratedAudio {
    pub(crate) inner: GeneratedAudio,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyGeneratedAudio {
    /// The underlying media output.
    #[getter]
    fn media(&self) -> PyMediaOutput {
        PyMediaOutput {
            inner: self.inner.media.clone(),
        }
    }

    /// Duration in seconds, if known.
    #[getter]
    fn duration_seconds(&self) -> Option<f32> {
        self.inner.duration_seconds
    }

    /// Sample rate in Hz, if known.
    #[getter]
    fn sample_rate(&self) -> Option<u32> {
        self.inner.sample_rate
    }

    /// Number of audio channels, if known.
    #[getter]
    fn channels(&self) -> Option<u8> {
        self.inner.channels
    }

    fn __repr__(&self) -> String {
        format!(
            "GeneratedAudio(duration_seconds={:?}, sample_rate={:?}, channels={:?})",
            self.inner.duration_seconds, self.inner.sample_rate, self.inner.channels,
        )
    }
}

/// A single generated 3D model with optional mesh metadata.
#[gen_stub_pyclass]
#[pyclass(name = "Generated3DModel", frozen)]
pub struct PyGenerated3DModel {
    pub(crate) inner: Generated3DModel,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyGenerated3DModel {
    /// The underlying media output.
    #[getter]
    fn media(&self) -> PyMediaOutput {
        PyMediaOutput {
            inner: self.inner.media.clone(),
        }
    }

    /// Total vertex count, if known.
    #[getter]
    fn vertex_count(&self) -> Option<u64> {
        self.inner.vertex_count
    }

    /// Total face/triangle count, if known.
    #[getter]
    fn face_count(&self) -> Option<u64> {
        self.inner.face_count
    }

    /// Whether the model includes texture data.
    #[getter]
    fn has_textures(&self) -> bool {
        self.inner.has_textures
    }

    /// Whether the model includes animation data.
    #[getter]
    fn has_animations(&self) -> bool {
        self.inner.has_animations
    }

    fn __repr__(&self) -> String {
        format!(
            "Generated3DModel(vertex_count={:?}, face_count={:?}, has_textures={}, has_animations={})",
            self.inner.vertex_count,
            self.inner.face_count,
            self.inner.has_textures,
            self.inner.has_animations,
        )
    }
}
