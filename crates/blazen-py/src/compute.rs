//! Python wrappers for compute request types and media types.
//!
//! Exposes typed request classes for image, video, audio, transcription,
//! and 3D generation operations, plus the [`MediaType`](blazen_llm::MediaType)
//! constants.

use pyo3::prelude::*;

use blazen_llm::compute::requests;

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
