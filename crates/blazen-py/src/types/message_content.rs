//! Standalone Python wrappers for the multimedia message content types.
//!
//! `PyContentPart` (in `message.rs`) carries factories that hide the
//! per-modality structs. These wrappers expose the underlying types
//! (`ImageContent`, `AudioContent`, `VideoContent`, `FileContent`) and the
//! shared `ImageSource` enum as standalone classes so callers can construct
//! and inspect them directly.

use std::path::PathBuf;

use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use blazen_llm::{
    AudioContent, ContentPart, FileContent, ImageContent, ImageSource, MessageContent, VideoContent,
};

// ---------------------------------------------------------------------------
// PyImageSource
// ---------------------------------------------------------------------------

/// How a piece of media is provided. Variants are constructed via the
/// classmethod factories ``url(...)``, ``base64(...)``, and ``file(...)``.
///
/// Inspect via the ``kind`` getter (``"url"``, ``"base64"``, or ``"file"``)
/// and the per-variant getters (``url``, ``data``, ``path``).
#[gen_stub_pyclass]
#[pyclass(name = "ImageSource", from_py_object)]
#[derive(Clone)]
pub struct PyImageSource {
    pub(crate) inner: ImageSource,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyImageSource {
    /// Build a URL source.
    #[staticmethod]
    fn url(url: String) -> Self {
        Self {
            inner: ImageSource::Url { url },
        }
    }

    /// Build a base64 source.
    #[staticmethod]
    fn base64(data: String) -> Self {
        Self {
            inner: ImageSource::Base64 { data },
        }
    }

    /// Build a local-file source. Only honoured by local backends; cloud
    /// providers reject this variant.
    #[staticmethod]
    fn file(path: PathBuf) -> Self {
        Self {
            inner: ImageSource::File { path },
        }
    }

    /// Variant tag: ``"url"``, ``"base64"``, or ``"file"``.
    #[getter]
    fn kind(&self) -> &'static str {
        match &self.inner {
            ImageSource::Url { .. } => "url",
            ImageSource::Base64 { .. } => "base64",
            ImageSource::File { .. } => "file",
        }
    }

    /// URL string for ``Url`` variants. ``None`` otherwise.
    #[getter]
    fn url_value(&self) -> Option<&str> {
        if let ImageSource::Url { url } = &self.inner {
            Some(url)
        } else {
            None
        }
    }

    /// Base64-encoded payload for ``Base64`` variants. ``None`` otherwise.
    #[getter]
    fn data(&self) -> Option<&str> {
        if let ImageSource::Base64 { data } = &self.inner {
            Some(data)
        } else {
            None
        }
    }

    /// Local file path for ``File`` variants. ``None`` otherwise.
    #[getter]
    fn path(&self) -> Option<PathBuf> {
        if let ImageSource::File { path } = &self.inner {
            Some(path.clone())
        } else {
            None
        }
    }

    fn __repr__(&self) -> String {
        match &self.inner {
            ImageSource::Url { url } => format!("ImageSource.url({url:?})"),
            ImageSource::Base64 { data } => {
                format!("ImageSource.base64(<{} chars>)", data.len())
            }
            ImageSource::File { path } => format!("ImageSource.file({})", path.display()),
        }
    }
}

impl From<ImageSource> for PyImageSource {
    fn from(inner: ImageSource) -> Self {
        Self { inner }
    }
}

// ---------------------------------------------------------------------------
// PyImageContent
// ---------------------------------------------------------------------------

/// Image content for multimodal messages.
#[gen_stub_pyclass]
#[pyclass(name = "ImageContent", from_py_object)]
#[derive(Clone)]
pub struct PyImageContent {
    pub(crate) inner: ImageContent,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyImageContent {
    /// Construct an image content block from a typed source.
    #[new]
    #[pyo3(signature = (*, source, media_type=None))]
    fn new(source: PyImageSource, media_type: Option<String>) -> Self {
        Self {
            inner: ImageContent {
                source: source.inner,
                media_type,
            },
        }
    }

    /// The underlying media source.
    #[getter]
    fn source(&self) -> PyImageSource {
        PyImageSource {
            inner: self.inner.source.clone(),
        }
    }

    /// MIME type, e.g. ``"image/png"``.
    #[getter]
    fn media_type(&self) -> Option<&str> {
        self.inner.media_type.as_deref()
    }

    fn __repr__(&self) -> String {
        format!("ImageContent(media_type={:?})", self.inner.media_type)
    }
}

impl From<ImageContent> for PyImageContent {
    fn from(inner: ImageContent) -> Self {
        Self { inner }
    }
}

// ---------------------------------------------------------------------------
// PyAudioContent
// ---------------------------------------------------------------------------

/// Audio content for multimodal messages (Gemini, gpt-4o-audio-preview).
#[gen_stub_pyclass]
#[pyclass(name = "AudioContent", from_py_object)]
#[derive(Clone)]
pub struct PyAudioContent {
    pub(crate) inner: AudioContent,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyAudioContent {
    /// Construct an audio content block.
    #[new]
    #[pyo3(signature = (*, source, media_type=None, duration_seconds=None))]
    fn new(
        source: PyImageSource,
        media_type: Option<String>,
        duration_seconds: Option<f32>,
    ) -> Self {
        Self {
            inner: AudioContent {
                source: source.inner,
                media_type,
                duration_seconds,
            },
        }
    }

    /// Build an audio block from a public URL.
    #[staticmethod]
    fn from_url(url: String) -> Self {
        Self {
            inner: AudioContent::from_url(url),
        }
    }

    /// Build an audio block from base64-encoded data with an explicit MIME.
    #[staticmethod]
    fn from_base64(data: String, media_type: String) -> Self {
        Self {
            inner: AudioContent::from_base64(data, media_type),
        }
    }

    #[getter]
    fn source(&self) -> PyImageSource {
        PyImageSource {
            inner: self.inner.source.clone(),
        }
    }

    #[getter]
    fn media_type(&self) -> Option<&str> {
        self.inner.media_type.as_deref()
    }

    #[getter]
    fn duration_seconds(&self) -> Option<f32> {
        self.inner.duration_seconds
    }

    fn __repr__(&self) -> String {
        format!(
            "AudioContent(media_type={:?}, duration_seconds={:?})",
            self.inner.media_type, self.inner.duration_seconds
        )
    }
}

impl From<AudioContent> for PyAudioContent {
    fn from(inner: AudioContent) -> Self {
        Self { inner }
    }
}

// ---------------------------------------------------------------------------
// PyVideoContent
// ---------------------------------------------------------------------------

/// Video content for multimodal messages (Gemini).
#[gen_stub_pyclass]
#[pyclass(name = "VideoContent", from_py_object)]
#[derive(Clone)]
pub struct PyVideoContent {
    pub(crate) inner: VideoContent,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyVideoContent {
    /// Construct a video content block.
    #[new]
    #[pyo3(signature = (*, source, media_type=None, duration_seconds=None))]
    fn new(
        source: PyImageSource,
        media_type: Option<String>,
        duration_seconds: Option<f32>,
    ) -> Self {
        Self {
            inner: VideoContent {
                source: source.inner,
                media_type,
                duration_seconds,
            },
        }
    }

    /// Build a video block from a public URL.
    #[staticmethod]
    fn from_url(url: String) -> Self {
        Self {
            inner: VideoContent::from_url(url),
        }
    }

    /// Build a video block from base64-encoded data with an explicit MIME.
    #[staticmethod]
    fn from_base64(data: String, media_type: String) -> Self {
        Self {
            inner: VideoContent::from_base64(data, media_type),
        }
    }

    #[getter]
    fn source(&self) -> PyImageSource {
        PyImageSource {
            inner: self.inner.source.clone(),
        }
    }

    #[getter]
    fn media_type(&self) -> Option<&str> {
        self.inner.media_type.as_deref()
    }

    #[getter]
    fn duration_seconds(&self) -> Option<f32> {
        self.inner.duration_seconds
    }

    fn __repr__(&self) -> String {
        format!(
            "VideoContent(media_type={:?}, duration_seconds={:?})",
            self.inner.media_type, self.inner.duration_seconds
        )
    }
}

impl From<VideoContent> for PyVideoContent {
    fn from(inner: VideoContent) -> Self {
        Self { inner }
    }
}

// ---------------------------------------------------------------------------
// PyFileContent
// ---------------------------------------------------------------------------

/// File / document content (PDFs and other documents) for multimodal messages.
#[gen_stub_pyclass]
#[pyclass(name = "FileContent", from_py_object)]
#[derive(Clone)]
pub struct PyFileContent {
    pub(crate) inner: FileContent,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyFileContent {
    /// Construct a file content block.
    #[new]
    #[pyo3(signature = (*, source, media_type, filename=None))]
    fn new(source: PyImageSource, media_type: String, filename: Option<String>) -> Self {
        Self {
            inner: FileContent {
                source: source.inner,
                media_type,
                filename,
            },
        }
    }

    #[getter]
    fn source(&self) -> PyImageSource {
        PyImageSource {
            inner: self.inner.source.clone(),
        }
    }

    #[getter]
    fn media_type(&self) -> &str {
        &self.inner.media_type
    }

    #[getter]
    fn filename(&self) -> Option<&str> {
        self.inner.filename.as_deref()
    }

    fn __repr__(&self) -> String {
        format!(
            "FileContent(media_type={:?}, filename={:?})",
            self.inner.media_type, self.inner.filename
        )
    }
}

impl From<FileContent> for PyFileContent {
    fn from(inner: FileContent) -> Self {
        Self { inner }
    }
}

// ---------------------------------------------------------------------------
// PyMessageContent
// ---------------------------------------------------------------------------

/// The content payload of a [`ChatMessage`]. One of:
/// - ``"text"`` — plain string content.
/// - ``"image"`` — a single image (legacy shorthand).
/// - ``"parts"`` — a multi-part list of typed [`ContentPart`] values.
///
/// Construct via the factory classmethods. Inspect via ``kind`` plus the
/// per-variant getters (``text``, ``image``, ``parts``).
#[gen_stub_pyclass]
#[pyclass(name = "MessageContent", from_py_object)]
#[derive(Clone)]
pub struct PyMessageContent {
    pub(crate) inner: MessageContent,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyMessageContent {
    /// Build a plain-text content payload.
    #[staticmethod]
    fn text(text: String) -> Self {
        Self {
            inner: MessageContent::Text(text),
        }
    }

    /// Build a single-image content payload.
    #[staticmethod]
    fn image(image: PyImageContent) -> Self {
        Self {
            inner: MessageContent::Image(image.inner),
        }
    }

    /// Build a multi-part content payload from a list of [`ContentPart`] values.
    #[staticmethod]
    fn parts(parts: Vec<PyRef<'_, super::PyContentPart>>) -> Self {
        let inner_parts: Vec<ContentPart> = parts.iter().map(|p| p.inner.clone()).collect();
        Self {
            inner: MessageContent::Parts(inner_parts),
        }
    }

    /// Variant tag: ``"text"``, ``"image"``, or ``"parts"``.
    #[getter]
    fn kind(&self) -> &'static str {
        match &self.inner {
            MessageContent::Text(_) => "text",
            MessageContent::Image(_) => "image",
            MessageContent::Parts(_) => "parts",
        }
    }

    /// Text body for ``Text`` variants. ``None`` otherwise.
    #[getter]
    fn text_value(&self) -> Option<String> {
        if let MessageContent::Text(s) = &self.inner {
            Some(s.clone())
        } else {
            None
        }
    }

    /// Image body for ``Image`` variants. ``None`` otherwise.
    #[getter]
    fn image_value(&self) -> Option<PyImageContent> {
        if let MessageContent::Image(img) = &self.inner {
            Some(PyImageContent { inner: img.clone() })
        } else {
            None
        }
    }

    /// Parts body for ``Parts`` variants. Returns an empty list otherwise.
    #[getter]
    fn parts_value(&self) -> Vec<super::PyContentPart> {
        match &self.inner {
            MessageContent::Parts(parts) => parts
                .iter()
                .map(|p| super::PyContentPart { inner: p.clone() })
                .collect(),
            _ => Vec::new(),
        }
    }

    /// Concatenate any text segments, returning ``None`` if no text exists.
    fn text_content(&self) -> Option<String> {
        self.inner.text_content()
    }

    /// Convert any variant into a list of typed [`ContentPart`]s.
    fn as_parts(&self) -> Vec<super::PyContentPart> {
        self.inner
            .as_parts()
            .into_iter()
            .map(|p| super::PyContentPart { inner: p })
            .collect()
    }

    /// Whether this message contains any image parts.
    fn has_images(&self) -> bool {
        self.inner.has_images()
    }

    /// Whether this message contains any audio parts.
    fn has_audio(&self) -> bool {
        self.inner.has_audio()
    }

    /// Whether this message contains any video parts.
    fn has_video(&self) -> bool {
        self.inner.has_video()
    }

    /// Whether this message contains any file parts.
    fn has_files(&self) -> bool {
        self.inner.has_files()
    }

    fn __repr__(&self) -> String {
        format!("MessageContent(kind={:?})", self.kind())
    }
}

impl From<MessageContent> for PyMessageContent {
    fn from(inner: MessageContent) -> Self {
        Self { inner }
    }
}
