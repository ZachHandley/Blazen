//! Python wrappers for media types and generated media output types.

use pyo3::prelude::*;

// ---------------------------------------------------------------------------
// PyMediaType
// ---------------------------------------------------------------------------

/// Media type constants for identifying file formats.
///
/// Example:
///     >>> MediaType.PNG   # "image/png"
///     >>> MediaType.MP4   # "video/mp4"
///     >>> MediaType.MP3   # "audio/mpeg"
#[pyclass(name = "MediaType", frozen)]
pub struct PyMediaType;

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
// PyMediaOutput
// ---------------------------------------------------------------------------

/// A single piece of generated media content.
///
/// At least one of url, base64, or raw_content will be populated.
///
/// Example:
///     >>> output.url
///     >>> output.media_type
///     >>> output.file_size
#[pyclass(name = "MediaOutput", frozen, from_py_object)]
#[derive(Clone)]
pub struct PyMediaOutput {
    pub(crate) inner: blazen_llm::media::MediaOutput,
}

#[pymethods]
impl PyMediaOutput {
    /// URL where the media can be downloaded.
    #[getter]
    fn url(&self) -> Option<&str> {
        self.inner.url.as_deref()
    }

    /// Base64-encoded media data.
    #[getter]
    fn base64(&self) -> Option<&str> {
        self.inner.base64.as_deref()
    }

    /// Raw text content for text-based formats.
    #[getter]
    fn raw_content(&self) -> Option<&str> {
        self.inner.raw_content.as_deref()
    }

    /// The MIME type string of the media.
    #[getter]
    fn media_type(&self) -> String {
        self.inner.media_type.mime().to_owned()
    }

    /// File size in bytes, if known.
    #[getter]
    fn file_size(&self) -> Option<u64> {
        self.inner.file_size
    }

    /// Provider-specific metadata as a dict.
    #[getter]
    fn metadata(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        crate::workflow::event::json_to_py(py, &self.inner.metadata)
    }

    fn __getitem__(&self, py: Python<'_>, key: &str) -> PyResult<Py<PyAny>> {
        match key {
            "url" => match &self.inner.url {
                Some(s) => Ok(s.clone().into_pyobject(py)?.into_any().unbind()),
                None => Ok(py.None()),
            },
            "base64" => match &self.inner.base64 {
                Some(s) => Ok(s.clone().into_pyobject(py)?.into_any().unbind()),
                None => Ok(py.None()),
            },
            "raw_content" => match &self.inner.raw_content {
                Some(s) => Ok(s.clone().into_pyobject(py)?.into_any().unbind()),
                None => Ok(py.None()),
            },
            "media_type" => Ok(self
                .inner
                .media_type
                .mime()
                .to_owned()
                .into_pyobject(py)?
                .into_any()
                .unbind()),
            "file_size" => Ok(self.inner.file_size.into_pyobject(py)?.into_any().unbind()),
            "metadata" => crate::workflow::event::json_to_py(py, &self.inner.metadata),
            _ => Err(pyo3::exceptions::PyKeyError::new_err(key.to_owned())),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "MediaOutput(media_type='{}', url={:?})",
            self.inner.media_type.mime(),
            self.inner.url,
        )
    }
}

// ---------------------------------------------------------------------------
// PyGeneratedImage
// ---------------------------------------------------------------------------

/// A single generated image with optional dimension metadata.
///
/// Example:
///     >>> img.media.url
///     >>> img.width, img.height
#[pyclass(name = "GeneratedImage", frozen, from_py_object)]
#[derive(Clone)]
pub struct PyGeneratedImage {
    pub(crate) inner: blazen_llm::media::GeneratedImage,
}

#[pymethods]
impl PyGeneratedImage {
    /// The image media output.
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

    fn __getitem__(&self, py: Python<'_>, key: &str) -> PyResult<Py<PyAny>> {
        match key {
            "width" => Ok(self.inner.width.into_pyobject(py)?.into_any().unbind()),
            "height" => Ok(self.inner.height.into_pyobject(py)?.into_any().unbind()),
            "media" => {
                let json = serde_json::to_value(&self.inner.media)
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
                crate::workflow::event::json_to_py(py, &json)
            }
            _ => Err(pyo3::exceptions::PyKeyError::new_err(key.to_owned())),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "GeneratedImage(width={:?}, height={:?})",
            self.inner.width, self.inner.height,
        )
    }
}

// ---------------------------------------------------------------------------
// PyGeneratedVideo
// ---------------------------------------------------------------------------

/// A single generated video with optional metadata.
///
/// Example:
///     >>> vid.media.url
///     >>> vid.duration_seconds, vid.fps
#[pyclass(name = "GeneratedVideo", frozen, from_py_object)]
#[derive(Clone)]
pub struct PyGeneratedVideo {
    pub(crate) inner: blazen_llm::media::GeneratedVideo,
}

#[pymethods]
impl PyGeneratedVideo {
    #[getter]
    fn media(&self) -> PyMediaOutput {
        PyMediaOutput {
            inner: self.inner.media.clone(),
        }
    }

    #[getter]
    fn width(&self) -> Option<u32> {
        self.inner.width
    }

    #[getter]
    fn height(&self) -> Option<u32> {
        self.inner.height
    }

    #[getter]
    fn duration_seconds(&self) -> Option<f32> {
        self.inner.duration_seconds
    }

    #[getter]
    fn fps(&self) -> Option<f32> {
        self.inner.fps
    }

    fn __getitem__(&self, py: Python<'_>, key: &str) -> PyResult<Py<PyAny>> {
        match key {
            "width" => Ok(self.inner.width.into_pyobject(py)?.into_any().unbind()),
            "height" => Ok(self.inner.height.into_pyobject(py)?.into_any().unbind()),
            "duration_seconds" => Ok(self
                .inner
                .duration_seconds
                .into_pyobject(py)?
                .into_any()
                .unbind()),
            "fps" => Ok(self.inner.fps.into_pyobject(py)?.into_any().unbind()),
            "media" => {
                let json = serde_json::to_value(&self.inner.media)
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
                crate::workflow::event::json_to_py(py, &json)
            }
            _ => Err(pyo3::exceptions::PyKeyError::new_err(key.to_owned())),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "GeneratedVideo(width={:?}, height={:?}, duration_seconds={:?}, fps={:?})",
            self.inner.width, self.inner.height, self.inner.duration_seconds, self.inner.fps,
        )
    }
}

// ---------------------------------------------------------------------------
// PyGeneratedAudio
// ---------------------------------------------------------------------------

/// A single generated audio clip with optional metadata.
///
/// Example:
///     >>> audio.media.url
///     >>> audio.duration_seconds, audio.sample_rate
#[pyclass(name = "GeneratedAudio", frozen, from_py_object)]
#[derive(Clone)]
pub struct PyGeneratedAudio {
    pub(crate) inner: blazen_llm::media::GeneratedAudio,
}

#[pymethods]
impl PyGeneratedAudio {
    #[getter]
    fn media(&self) -> PyMediaOutput {
        PyMediaOutput {
            inner: self.inner.media.clone(),
        }
    }

    #[getter]
    fn duration_seconds(&self) -> Option<f32> {
        self.inner.duration_seconds
    }

    #[getter]
    fn sample_rate(&self) -> Option<u32> {
        self.inner.sample_rate
    }

    #[getter]
    fn channels(&self) -> Option<u8> {
        self.inner.channels
    }

    fn __getitem__(&self, py: Python<'_>, key: &str) -> PyResult<Py<PyAny>> {
        match key {
            "duration_seconds" => Ok(self
                .inner
                .duration_seconds
                .into_pyobject(py)?
                .into_any()
                .unbind()),
            "sample_rate" => Ok(self
                .inner
                .sample_rate
                .into_pyobject(py)?
                .into_any()
                .unbind()),
            "channels" => Ok(self.inner.channels.into_pyobject(py)?.into_any().unbind()),
            "media" => {
                let json = serde_json::to_value(&self.inner.media)
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
                crate::workflow::event::json_to_py(py, &json)
            }
            _ => Err(pyo3::exceptions::PyKeyError::new_err(key.to_owned())),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "GeneratedAudio(duration_seconds={:?}, sample_rate={:?}, channels={:?})",
            self.inner.duration_seconds, self.inner.sample_rate, self.inner.channels,
        )
    }
}

// ---------------------------------------------------------------------------
// PyGenerated3DModel
// ---------------------------------------------------------------------------

/// A single generated 3D model with optional mesh metadata.
///
/// Example:
///     >>> model.media.url
///     >>> model.vertex_count, model.has_textures
#[pyclass(name = "Generated3DModel", frozen, from_py_object)]
#[derive(Clone)]
pub struct PyGenerated3DModel {
    pub(crate) inner: blazen_llm::media::Generated3DModel,
}

#[pymethods]
impl PyGenerated3DModel {
    #[getter]
    fn media(&self) -> PyMediaOutput {
        PyMediaOutput {
            inner: self.inner.media.clone(),
        }
    }

    #[getter]
    fn vertex_count(&self) -> Option<u64> {
        self.inner.vertex_count
    }

    #[getter]
    fn face_count(&self) -> Option<u64> {
        self.inner.face_count
    }

    #[getter]
    fn has_textures(&self) -> bool {
        self.inner.has_textures
    }

    #[getter]
    fn has_animations(&self) -> bool {
        self.inner.has_animations
    }

    fn __getitem__(&self, py: Python<'_>, key: &str) -> PyResult<Py<PyAny>> {
        match key {
            "vertex_count" => Ok(self
                .inner
                .vertex_count
                .into_pyobject(py)?
                .into_any()
                .unbind()),
            "face_count" => Ok(self.inner.face_count.into_pyobject(py)?.into_any().unbind()),
            "has_textures" => Ok(self
                .inner
                .has_textures
                .into_pyobject(py)?
                .to_owned()
                .into_any()
                .unbind()),
            "has_animations" => Ok(self
                .inner
                .has_animations
                .into_pyobject(py)?
                .to_owned()
                .into_any()
                .unbind()),
            "media" => {
                let json = serde_json::to_value(&self.inner.media)
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
                crate::workflow::event::json_to_py(py, &json)
            }
            _ => Err(pyo3::exceptions::PyKeyError::new_err(key.to_owned())),
        }
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
