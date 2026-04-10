//! Typed wrappers for compute request types.

use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use blazen_llm::compute::requests::{
    BackgroundRemovalRequest, ImageRequest, MusicRequest, SpeechRequest, ThreeDRequest,
    TranscriptionRequest, UpscaleRequest, VideoRequest,
};

/// Parse an optional Python object into a `serde_json::Value` (for the
/// `parameters` escape hatch on every compute request type).
fn parse_parameters(py: Python<'_>, obj: Option<Py<PyAny>>) -> PyResult<serde_json::Value> {
    match obj {
        Some(o) => crate::convert::py_to_json(py, o.bind(py)),
        None => Ok(serde_json::Value::Null),
    }
}

// ---------------------------------------------------------------------------
// ImageRequest
// ---------------------------------------------------------------------------

/// Typed wrapper for a text-to-image request.
#[gen_stub_pyclass]
#[pyclass(name = "ImageRequest", from_py_object)]
#[derive(Clone)]
pub struct PyImageRequest {
    pub(crate) inner: ImageRequest,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyImageRequest {
    #[new]
    #[pyo3(signature = (*, prompt, negative_prompt=None, width=None, height=None, num_images=None, model=None, parameters=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        prompt: String,
        negative_prompt: Option<String>,
        width: Option<u32>,
        height: Option<u32>,
        num_images: Option<u32>,
        model: Option<String>,
        parameters: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: ImageRequest {
                prompt,
                negative_prompt,
                width,
                height,
                num_images,
                model,
                parameters: parse_parameters(py, parameters)?,
            },
        })
    }

    #[getter]
    fn prompt(&self) -> &str {
        &self.inner.prompt
    }
    #[getter]
    fn negative_prompt(&self) -> Option<String> {
        self.inner.negative_prompt.clone()
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
    fn model(&self) -> Option<String> {
        self.inner.model.clone()
    }

    fn __repr__(&self) -> String {
        format!("ImageRequest(prompt={:?})", self.inner.prompt)
    }
}

// ---------------------------------------------------------------------------
// UpscaleRequest
// ---------------------------------------------------------------------------

/// Typed wrapper for an image upscale request.
#[gen_stub_pyclass]
#[pyclass(name = "UpscaleRequest", from_py_object)]
#[derive(Clone)]
pub struct PyUpscaleRequest {
    pub(crate) inner: UpscaleRequest,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyUpscaleRequest {
    #[new]
    #[pyo3(signature = (*, image_url, scale, model=None, parameters=None))]
    fn new(
        py: Python<'_>,
        image_url: String,
        scale: f32,
        model: Option<String>,
        parameters: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: UpscaleRequest {
                image_url,
                scale,
                model,
                parameters: parse_parameters(py, parameters)?,
            },
        })
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
    fn model(&self) -> Option<String> {
        self.inner.model.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "UpscaleRequest(image_url={:?}, scale={})",
            self.inner.image_url, self.inner.scale
        )
    }
}

// ---------------------------------------------------------------------------
// VideoRequest
// ---------------------------------------------------------------------------

/// Typed wrapper for a video generation request.
#[gen_stub_pyclass]
#[pyclass(name = "VideoRequest", from_py_object)]
#[derive(Clone)]
pub struct PyVideoRequest {
    pub(crate) inner: VideoRequest,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyVideoRequest {
    #[new]
    #[pyo3(signature = (*, prompt, image_url=None, duration_seconds=None, negative_prompt=None, width=None, height=None, model=None, parameters=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        prompt: String,
        image_url: Option<String>,
        duration_seconds: Option<f32>,
        negative_prompt: Option<String>,
        width: Option<u32>,
        height: Option<u32>,
        model: Option<String>,
        parameters: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: VideoRequest {
                prompt,
                image_url,
                duration_seconds,
                negative_prompt,
                width,
                height,
                model,
                parameters: parse_parameters(py, parameters)?,
            },
        })
    }

    #[getter]
    fn prompt(&self) -> &str {
        &self.inner.prompt
    }
    #[getter]
    fn image_url(&self) -> Option<String> {
        self.inner.image_url.clone()
    }
    #[getter]
    fn duration_seconds(&self) -> Option<f32> {
        self.inner.duration_seconds
    }
    #[getter]
    fn negative_prompt(&self) -> Option<String> {
        self.inner.negative_prompt.clone()
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
    fn model(&self) -> Option<String> {
        self.inner.model.clone()
    }

    fn __repr__(&self) -> String {
        format!("VideoRequest(prompt={:?})", self.inner.prompt)
    }
}

// ---------------------------------------------------------------------------
// SpeechRequest
// ---------------------------------------------------------------------------

/// Typed wrapper for a text-to-speech request.
#[gen_stub_pyclass]
#[pyclass(name = "SpeechRequest", from_py_object)]
#[derive(Clone)]
pub struct PySpeechRequest {
    pub(crate) inner: SpeechRequest,
}

#[gen_stub_pymethods]
#[pymethods]
impl PySpeechRequest {
    #[new]
    #[pyo3(signature = (*, text, voice=None, voice_url=None, language=None, speed=None, model=None, parameters=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        text: String,
        voice: Option<String>,
        voice_url: Option<String>,
        language: Option<String>,
        speed: Option<f32>,
        model: Option<String>,
        parameters: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: SpeechRequest {
                text,
                voice,
                voice_url,
                language,
                speed,
                model,
                parameters: parse_parameters(py, parameters)?,
            },
        })
    }

    #[getter]
    fn text(&self) -> &str {
        &self.inner.text
    }
    #[getter]
    fn voice(&self) -> Option<String> {
        self.inner.voice.clone()
    }
    #[getter]
    fn voice_url(&self) -> Option<String> {
        self.inner.voice_url.clone()
    }
    #[getter]
    fn language(&self) -> Option<String> {
        self.inner.language.clone()
    }
    #[getter]
    fn speed(&self) -> Option<f32> {
        self.inner.speed
    }
    #[getter]
    fn model(&self) -> Option<String> {
        self.inner.model.clone()
    }

    fn __repr__(&self) -> String {
        format!("SpeechRequest(text={:?})", self.inner.text)
    }
}

// ---------------------------------------------------------------------------
// MusicRequest
// ---------------------------------------------------------------------------

/// Typed wrapper for a music/SFX generation request.
#[gen_stub_pyclass]
#[pyclass(name = "MusicRequest", from_py_object)]
#[derive(Clone)]
pub struct PyMusicRequest {
    pub(crate) inner: MusicRequest,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyMusicRequest {
    #[new]
    #[pyo3(signature = (*, prompt, duration_seconds=None, model=None, parameters=None))]
    fn new(
        py: Python<'_>,
        prompt: String,
        duration_seconds: Option<f32>,
        model: Option<String>,
        parameters: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: MusicRequest {
                prompt,
                duration_seconds,
                model,
                parameters: parse_parameters(py, parameters)?,
            },
        })
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
    fn model(&self) -> Option<String> {
        self.inner.model.clone()
    }

    fn __repr__(&self) -> String {
        format!("MusicRequest(prompt={:?})", self.inner.prompt)
    }
}

// ---------------------------------------------------------------------------
// TranscriptionRequest
// ---------------------------------------------------------------------------

/// Typed wrapper for an audio transcription request.
#[gen_stub_pyclass]
#[pyclass(name = "TranscriptionRequest", from_py_object)]
#[derive(Clone)]
pub struct PyTranscriptionRequest {
    pub(crate) inner: TranscriptionRequest,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyTranscriptionRequest {
    #[new]
    #[pyo3(signature = (*, audio_url, language=None, diarize=false, model=None, parameters=None))]
    fn new(
        py: Python<'_>,
        audio_url: String,
        language: Option<String>,
        diarize: bool,
        model: Option<String>,
        parameters: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: TranscriptionRequest {
                audio_url,
                audio_source: None,
                language,
                diarize,
                model,
                parameters: parse_parameters(py, parameters)?,
            },
        })
    }

    /// Create a transcription request from a local file path.
    ///
    /// This is the preferred constructor for local backends like whisper.cpp
    /// that read audio directly from disk. It sets ``audio_source`` to a
    /// ``MediaSource::File`` and leaves ``audio_url`` empty.
    ///
    /// Args:
    ///     path: Absolute or relative path to a local audio file
    ///         (16-bit PCM mono 16 kHz WAV for whisper.cpp).
    ///     language: Optional ISO 639-1 language hint (e.g. "en").
    ///     diarize: Enable speaker diarization (provider-dependent).
    ///     model: Optional provider-specific model override.
    ///
    /// Example:
    ///     >>> req = TranscriptionRequest.from_file("/path/to/audio.wav")
    ///     >>> req = TranscriptionRequest.from_file(
    ///     ...     "/path/to/audio.wav", language="en"
    ///     ... )
    #[staticmethod]
    #[pyo3(signature = (path, *, language=None, diarize=false, model=None))]
    fn from_file(
        path: String,
        language: Option<String>,
        diarize: bool,
        model: Option<String>,
    ) -> Self {
        let mut inner = TranscriptionRequest::from_file(path);
        inner.language = language;
        inner.diarize = diarize;
        inner.model = model;
        Self { inner }
    }

    #[getter]
    fn audio_url(&self) -> &str {
        &self.inner.audio_url
    }
    #[getter]
    fn language(&self) -> Option<String> {
        self.inner.language.clone()
    }
    #[getter]
    fn diarize(&self) -> bool {
        self.inner.diarize
    }
    #[getter]
    fn model(&self) -> Option<String> {
        self.inner.model.clone()
    }

    fn __repr__(&self) -> String {
        format!("TranscriptionRequest(audio_url={:?})", self.inner.audio_url)
    }
}

// ---------------------------------------------------------------------------
// ThreeDRequest
// ---------------------------------------------------------------------------

/// Typed wrapper for a 3D model generation request.
#[gen_stub_pyclass]
#[pyclass(name = "ThreeDRequest", from_py_object)]
#[derive(Clone)]
pub struct PyThreeDRequest {
    pub(crate) inner: ThreeDRequest,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyThreeDRequest {
    #[new]
    #[pyo3(signature = (*, prompt=String::new(), image_url=None, format=None, model=None, parameters=None))]
    fn new(
        py: Python<'_>,
        prompt: String,
        image_url: Option<String>,
        format: Option<String>,
        model: Option<String>,
        parameters: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: ThreeDRequest {
                prompt,
                image_url,
                format,
                model,
                parameters: parse_parameters(py, parameters)?,
            },
        })
    }

    #[getter]
    fn prompt(&self) -> &str {
        &self.inner.prompt
    }
    #[getter]
    fn image_url(&self) -> Option<String> {
        self.inner.image_url.clone()
    }
    #[getter]
    fn format(&self) -> Option<String> {
        self.inner.format.clone()
    }
    #[getter]
    fn model(&self) -> Option<String> {
        self.inner.model.clone()
    }

    fn __repr__(&self) -> String {
        format!("ThreeDRequest(prompt={:?})", self.inner.prompt)
    }
}

// ---------------------------------------------------------------------------
// BackgroundRemovalRequest
// ---------------------------------------------------------------------------

/// Typed wrapper for a background removal request.
#[gen_stub_pyclass]
#[pyclass(name = "BackgroundRemovalRequest", from_py_object)]
#[derive(Clone)]
pub struct PyBackgroundRemovalRequest {
    pub(crate) inner: BackgroundRemovalRequest,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyBackgroundRemovalRequest {
    #[new]
    #[pyo3(signature = (*, image_url, model=None, parameters=None))]
    fn new(
        py: Python<'_>,
        image_url: String,
        model: Option<String>,
        parameters: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: BackgroundRemovalRequest {
                image_url,
                model,
                parameters: parse_parameters(py, parameters)?,
            },
        })
    }

    #[getter]
    fn image_url(&self) -> &str {
        &self.inner.image_url
    }
    #[getter]
    fn model(&self) -> Option<String> {
        self.inner.model.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "BackgroundRemovalRequest(image_url={:?})",
            self.inner.image_url
        )
    }
}
