//! Python wrappers for compute request types, result types, and media types.
//!
//! Exposes typed request classes for image, video, audio, transcription,
//! and 3D generation operations, plus the [`MediaType`](blazen_llm::MediaType)
//! constants. Also provides wrapper types for job handles, job status,
//! media outputs, and generated media objects.

use pyo3::prelude::*;

use blazen_llm::compute::{self as compute_types, requests};

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
// PyImageRequest
// ---------------------------------------------------------------------------

/// Request to generate images from a text prompt.
///
/// Example:
///     >>> req = ImageRequest(prompt="a cat in space", width=1024, height=1024, num_images=2)
#[pyclass(name = "ImageRequest", from_py_object)]
#[derive(Clone)]
pub struct PyImageRequest {
    pub(crate) inner: requests::ImageRequest,
}

#[pymethods]
impl PyImageRequest {
    #[new]
    #[pyo3(signature = (*, prompt, negative_prompt=None, width=None, height=None, num_images=None, model=None))]
    fn new(
        prompt: &str,
        negative_prompt: Option<String>,
        width: Option<u32>,
        height: Option<u32>,
        num_images: Option<u32>,
        model: Option<String>,
    ) -> Self {
        let mut req = requests::ImageRequest::new(prompt);
        req.negative_prompt = negative_prompt;
        req.width = width;
        req.height = height;
        req.num_images = num_images;
        req.model = model;
        Self { inner: req }
    }

    #[getter]
    fn prompt(&self) -> &str {
        &self.inner.prompt
    }

    #[getter]
    fn negative_prompt(&self) -> Option<&str> {
        self.inner.negative_prompt.as_deref()
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
    fn num_images(&self) -> Option<u32> {
        self.inner.num_images
    }

    #[getter]
    fn model(&self) -> Option<&str> {
        self.inner.model.as_deref()
    }

    fn __repr__(&self) -> String {
        format!(
            "ImageRequest(prompt='{}', width={:?}, height={:?}, num_images={:?})",
            self.inner.prompt, self.inner.width, self.inner.height, self.inner.num_images,
        )
    }
}

// ---------------------------------------------------------------------------
// PyUpscaleRequest
// ---------------------------------------------------------------------------

/// Request to upscale an image.
///
/// Example:
///     >>> req = UpscaleRequest(image_url="https://...", scale=4.0)
#[pyclass(name = "UpscaleRequest", from_py_object)]
#[derive(Clone)]
pub struct PyUpscaleRequest {
    pub(crate) inner: requests::UpscaleRequest,
}

#[pymethods]
impl PyUpscaleRequest {
    #[new]
    #[pyo3(signature = (*, image_url, scale, model=None))]
    fn new(image_url: &str, scale: f32, model: Option<String>) -> Self {
        let mut req = requests::UpscaleRequest::new(image_url, scale);
        req.model = model;
        Self { inner: req }
    }

    #[getter]
    fn image_url(&self) -> &str {
        &self.inner.image_url
    }

    #[getter]
    fn scale(&self) -> f32 {
        self.inner.scale
    }

    #[getter]
    fn model(&self) -> Option<&str> {
        self.inner.model.as_deref()
    }

    fn __repr__(&self) -> String {
        format!(
            "UpscaleRequest(image_url='{}', scale={})",
            self.inner.image_url, self.inner.scale,
        )
    }
}

// ---------------------------------------------------------------------------
// PyVideoRequest
// ---------------------------------------------------------------------------

/// Request to generate a video.
///
/// Example:
///     >>> req = VideoRequest(prompt="a sunset timelapse", duration_seconds=5.0)
///     >>> req = VideoRequest(prompt="animate this", image_url="https://...")
#[pyclass(name = "VideoRequest", from_py_object)]
#[derive(Clone)]
pub struct PyVideoRequest {
    pub(crate) inner: requests::VideoRequest,
}

#[pymethods]
impl PyVideoRequest {
    #[new]
    #[pyo3(signature = (*, prompt, image_url=None, duration_seconds=None, negative_prompt=None, width=None, height=None, model=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        prompt: &str,
        image_url: Option<String>,
        duration_seconds: Option<f32>,
        negative_prompt: Option<String>,
        width: Option<u32>,
        height: Option<u32>,
        model: Option<String>,
    ) -> Self {
        let mut req = requests::VideoRequest::new(prompt);
        req.image_url = image_url;
        req.duration_seconds = duration_seconds;
        req.negative_prompt = negative_prompt;
        req.width = width;
        req.height = height;
        req.model = model;
        Self { inner: req }
    }

    #[getter]
    fn prompt(&self) -> &str {
        &self.inner.prompt
    }

    #[getter]
    fn image_url(&self) -> Option<&str> {
        self.inner.image_url.as_deref()
    }

    #[getter]
    fn duration_seconds(&self) -> Option<f32> {
        self.inner.duration_seconds
    }

    #[getter]
    fn negative_prompt(&self) -> Option<&str> {
        self.inner.negative_prompt.as_deref()
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
    fn model(&self) -> Option<&str> {
        self.inner.model.as_deref()
    }

    fn __repr__(&self) -> String {
        format!(
            "VideoRequest(prompt='{}', duration_seconds={:?}, width={:?}, height={:?})",
            self.inner.prompt, self.inner.duration_seconds, self.inner.width, self.inner.height,
        )
    }
}

// ---------------------------------------------------------------------------
// PySpeechRequest
// ---------------------------------------------------------------------------

/// Request to generate speech from text.
///
/// Example:
///     >>> req = SpeechRequest(text="Hello world", voice="alloy", speed=1.2)
#[pyclass(name = "SpeechRequest", from_py_object)]
#[derive(Clone)]
pub struct PySpeechRequest {
    pub(crate) inner: requests::SpeechRequest,
}

#[pymethods]
impl PySpeechRequest {
    #[new]
    #[pyo3(signature = (*, text, voice=None, voice_url=None, language=None, speed=None, model=None))]
    fn new(
        text: &str,
        voice: Option<String>,
        voice_url: Option<String>,
        language: Option<String>,
        speed: Option<f32>,
        model: Option<String>,
    ) -> Self {
        let mut req = requests::SpeechRequest::new(text);
        req.voice = voice;
        req.voice_url = voice_url;
        req.language = language;
        req.speed = speed;
        req.model = model;
        Self { inner: req }
    }

    #[getter]
    fn text(&self) -> &str {
        &self.inner.text
    }

    #[getter]
    fn voice(&self) -> Option<&str> {
        self.inner.voice.as_deref()
    }

    #[getter]
    fn voice_url(&self) -> Option<&str> {
        self.inner.voice_url.as_deref()
    }

    #[getter]
    fn language(&self) -> Option<&str> {
        self.inner.language.as_deref()
    }

    #[getter]
    fn speed(&self) -> Option<f32> {
        self.inner.speed
    }

    #[getter]
    fn model(&self) -> Option<&str> {
        self.inner.model.as_deref()
    }

    fn __repr__(&self) -> String {
        format!(
            "SpeechRequest(text='{}', voice={:?}, speed={:?})",
            self.inner.text, self.inner.voice, self.inner.speed,
        )
    }
}

// ---------------------------------------------------------------------------
// PyMusicRequest
// ---------------------------------------------------------------------------

/// Request to generate music or sound effects.
///
/// Example:
///     >>> req = MusicRequest(prompt="upbeat jazz", duration_seconds=30.0)
#[pyclass(name = "MusicRequest", from_py_object)]
#[derive(Clone)]
pub struct PyMusicRequest {
    pub(crate) inner: requests::MusicRequest,
}

#[pymethods]
impl PyMusicRequest {
    #[new]
    #[pyo3(signature = (*, prompt, duration_seconds=None, model=None))]
    fn new(prompt: &str, duration_seconds: Option<f32>, model: Option<String>) -> Self {
        let mut req = requests::MusicRequest::new(prompt);
        req.duration_seconds = duration_seconds;
        req.model = model;
        Self { inner: req }
    }

    #[getter]
    fn prompt(&self) -> &str {
        &self.inner.prompt
    }

    #[getter]
    fn duration_seconds(&self) -> Option<f32> {
        self.inner.duration_seconds
    }

    #[getter]
    fn model(&self) -> Option<&str> {
        self.inner.model.as_deref()
    }

    fn __repr__(&self) -> String {
        format!(
            "MusicRequest(prompt='{}', duration_seconds={:?})",
            self.inner.prompt, self.inner.duration_seconds,
        )
    }
}

// ---------------------------------------------------------------------------
// PyTranscriptionRequest
// ---------------------------------------------------------------------------

/// Request to transcribe audio to text.
///
/// Example:
///     >>> req = TranscriptionRequest(audio_url="https://...", language="en", diarize=True)
#[pyclass(name = "TranscriptionRequest", from_py_object)]
#[derive(Clone)]
pub struct PyTranscriptionRequest {
    pub(crate) inner: requests::TranscriptionRequest,
}

#[pymethods]
impl PyTranscriptionRequest {
    #[new]
    #[pyo3(signature = (*, audio_url, language=None, diarize=None, model=None))]
    fn new(
        audio_url: &str,
        language: Option<String>,
        diarize: Option<bool>,
        model: Option<String>,
    ) -> Self {
        let mut req = requests::TranscriptionRequest::new(audio_url);
        req.language = language;
        if let Some(d) = diarize {
            req.diarize = d;
        }
        req.model = model;
        Self { inner: req }
    }

    #[getter]
    fn audio_url(&self) -> &str {
        &self.inner.audio_url
    }

    #[getter]
    fn language(&self) -> Option<&str> {
        self.inner.language.as_deref()
    }

    #[getter]
    fn diarize(&self) -> bool {
        self.inner.diarize
    }

    #[getter]
    fn model(&self) -> Option<&str> {
        self.inner.model.as_deref()
    }

    fn __repr__(&self) -> String {
        format!(
            "TranscriptionRequest(audio_url='{}', language={:?}, diarize={})",
            self.inner.audio_url, self.inner.language, self.inner.diarize,
        )
    }
}

// ---------------------------------------------------------------------------
// PyThreeDRequest
// ---------------------------------------------------------------------------

/// Request to generate a 3D model.
///
/// Example:
///     >>> req = ThreeDRequest(prompt="a 3D cat", format="glb")
///     >>> req = ThreeDRequest(image_url="https://...", format="obj")
#[pyclass(name = "ThreeDRequest", from_py_object)]
#[derive(Clone)]
pub struct PyThreeDRequest {
    pub(crate) inner: requests::ThreeDRequest,
}

#[pymethods]
impl PyThreeDRequest {
    #[new]
    #[pyo3(signature = (*, prompt=None, image_url=None, format=None, model=None))]
    fn new(
        prompt: Option<&str>,
        image_url: Option<String>,
        format: Option<String>,
        model: Option<String>,
    ) -> Self {
        let mut req = if let Some(p) = prompt {
            requests::ThreeDRequest::new(p)
        } else {
            // If no prompt, create with empty string -- image_url will be set below
            requests::ThreeDRequest::new("")
        };
        req.image_url = image_url;
        req.format = format;
        req.model = model;
        Self { inner: req }
    }

    #[getter]
    fn prompt(&self) -> &str {
        &self.inner.prompt
    }

    #[getter]
    fn image_url(&self) -> Option<&str> {
        self.inner.image_url.as_deref()
    }

    #[getter]
    fn format(&self) -> Option<&str> {
        self.inner.format.as_deref()
    }

    #[getter]
    fn model(&self) -> Option<&str> {
        self.inner.model.as_deref()
    }

    fn __repr__(&self) -> String {
        format!(
            "ThreeDRequest(prompt='{}', image_url={:?}, format={:?})",
            self.inner.prompt, self.inner.image_url, self.inner.format,
        )
    }
}

// ===========================================================================
// Job types
// ===========================================================================

// ---------------------------------------------------------------------------
// PyJobStatus
// ---------------------------------------------------------------------------

/// Job status constants.
///
/// Example:
///     >>> JobStatus.QUEUED    # "queued"
///     >>> JobStatus.RUNNING   # "running"
///     >>> JobStatus.COMPLETED # "completed"
#[pyclass(name = "JobStatus", frozen)]
pub struct PyJobStatus;

#[pymethods]
impl PyJobStatus {
    #[classattr]
    const QUEUED: &'static str = "queued";
    #[classattr]
    const RUNNING: &'static str = "running";
    #[classattr]
    const COMPLETED: &'static str = "completed";
    #[classattr]
    const FAILED: &'static str = "failed";
    #[classattr]
    const CANCELLED: &'static str = "cancelled";
}

// ---------------------------------------------------------------------------
// PyJobHandle
// ---------------------------------------------------------------------------

/// A handle to a submitted compute job.
///
/// Example:
///     >>> handle = await fal.submit(model="fal-ai/flux/dev", input={...})
///     >>> print(handle.id, handle.model)
#[pyclass(name = "JobHandle", frozen, from_py_object)]
#[derive(Clone)]
pub struct PyJobHandle {
    pub(crate) inner: compute_types::JobHandle,
}

#[pymethods]
impl PyJobHandle {
    /// The provider-assigned job identifier.
    #[getter]
    fn id(&self) -> &str {
        &self.inner.id
    }

    /// The provider name (e.g. "fal").
    #[getter]
    fn provider(&self) -> &str {
        &self.inner.provider
    }

    /// The model/endpoint that was invoked.
    #[getter]
    fn model(&self) -> &str {
        &self.inner.model
    }

    /// When the job was submitted (ISO 8601 string).
    #[getter]
    fn submitted_at(&self) -> String {
        self.inner.submitted_at.to_rfc3339()
    }

    fn __getitem__(&self, py: Python<'_>, key: &str) -> PyResult<Py<PyAny>> {
        match key {
            "id" => Ok(self.inner.id.clone().into_pyobject(py)?.into_any().unbind()),
            "provider" => Ok(self
                .inner
                .provider
                .clone()
                .into_pyobject(py)?
                .into_any()
                .unbind()),
            "model" => Ok(self
                .inner
                .model
                .clone()
                .into_pyobject(py)?
                .into_any()
                .unbind()),
            "submitted_at" => Ok(self
                .inner
                .submitted_at
                .to_rfc3339()
                .into_pyobject(py)?
                .into_any()
                .unbind()),
            _ => Err(pyo3::exceptions::PyKeyError::new_err(key.to_owned())),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "JobHandle(id='{}', provider='{}', model='{}')",
            self.inner.id, self.inner.provider, self.inner.model,
        )
    }
}

// ---------------------------------------------------------------------------
// PyComputeRequest
// ---------------------------------------------------------------------------

/// Input for a raw compute job.
///
/// Example:
///     >>> req = ComputeRequest(model="fal-ai/flux/dev", input={"prompt": "a cat"})
#[pyclass(name = "ComputeRequest", from_py_object)]
#[derive(Clone)]
pub struct PyComputeRequest {
    pub(crate) inner: compute_types::ComputeRequest,
}

#[pymethods]
impl PyComputeRequest {
    #[new]
    #[pyo3(signature = (*, model, input))]
    fn new(py: Python<'_>, model: &str, input: &Bound<'_, PyAny>) -> PyResult<Self> {
        let input_json = crate::event::py_to_json(py, input)?;
        Ok(Self {
            inner: compute_types::ComputeRequest {
                model: model.to_owned(),
                input: input_json,
                webhook: None,
            },
        })
    }

    #[getter]
    fn model(&self) -> &str {
        &self.inner.model
    }

    #[getter]
    fn input(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        crate::event::json_to_py(py, &self.inner.input)
    }

    fn __repr__(&self) -> String {
        format!("ComputeRequest(model='{}')", self.inner.model)
    }
}

// ===========================================================================
// Media output types
// ===========================================================================

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
        crate::event::json_to_py(py, &self.inner.metadata)
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
            "metadata" => crate::event::json_to_py(py, &self.inner.metadata),
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
                crate::event::json_to_py(py, &json)
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
                crate::event::json_to_py(py, &json)
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
                crate::event::json_to_py(py, &json)
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
                crate::event::json_to_py(py, &json)
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
