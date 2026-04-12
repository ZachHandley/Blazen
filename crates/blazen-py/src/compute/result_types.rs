//! Typed wrappers for compute result types.
//!
//! These wrappers expose the Rust `*Result` shapes to Python as frozen
//! pyclasses. They are *result types* — produced by the library, never
//! constructed by users — so no `#[new]` constructors are provided.
//!
//! The nested media collections (`images`, `videos`, `audio`, `models`) return
//! typed wrappers (`PyGeneratedImage`, etc.) from `crate::types::media`.
//! Timing fields return `PyRequestTiming` from `crate::types`.

use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use blazen_llm::compute::job::ComputeResult;
use blazen_llm::compute::results::{
    AudioResult, ImageResult, ThreeDResult, TranscriptionResult, TranscriptionSegment, VideoResult,
    VoiceHandle,
};

use crate::types::PyRequestTiming;
use crate::types::media::{
    PyGenerated3DModel, PyGeneratedAudio, PyGeneratedImage, PyGeneratedVideo,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Convert any `Serialize` value to a Python object via JSON.
fn serialize_to_py<T: serde::Serialize>(py: Python<'_>, value: &T) -> PyResult<Py<PyAny>> {
    let v = serde_json::to_value(value)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    crate::convert::json_to_py(py, &v)
}

// ---------------------------------------------------------------------------
// TranscriptionSegment
// ---------------------------------------------------------------------------

/// A single segment within a transcription.
#[gen_stub_pyclass]
#[pyclass(name = "TranscriptionSegment", frozen)]
pub struct PyTranscriptionSegment {
    pub(crate) inner: TranscriptionSegment,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyTranscriptionSegment {
    /// The transcribed text for this segment.
    #[getter]
    fn text(&self) -> String {
        self.inner.text.clone()
    }

    /// Start time in seconds.
    #[getter]
    fn start(&self) -> f64 {
        self.inner.start
    }

    /// End time in seconds.
    #[getter]
    fn end(&self) -> f64 {
        self.inner.end
    }

    /// Speaker label, if diarization was enabled.
    #[getter]
    fn speaker(&self) -> Option<String> {
        self.inner.speaker.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "TranscriptionSegment(text={:?}, start={}, end={}, speaker={:?})",
            self.inner.text, self.inner.start, self.inner.end, self.inner.speaker
        )
    }
}

// ---------------------------------------------------------------------------
// TranscriptionResult
// ---------------------------------------------------------------------------

/// Result of a transcription operation.
#[gen_stub_pyclass]
#[pyclass(name = "TranscriptionResult", frozen)]
pub struct PyTranscriptionResult {
    pub(crate) inner: TranscriptionResult,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyTranscriptionResult {
    /// The full transcribed text.
    #[getter]
    fn text(&self) -> String {
        self.inner.text.clone()
    }

    /// Time-aligned segments, if available.
    #[getter]
    fn segments(&self) -> Vec<PyTranscriptionSegment> {
        self.inner
            .segments
            .iter()
            .map(|s| PyTranscriptionSegment { inner: s.clone() })
            .collect()
    }

    /// Detected or specified language code (e.g. "en", "fr").
    #[getter]
    fn language(&self) -> Option<String> {
        self.inner.language.clone()
    }

    /// Request timing breakdown.
    #[getter]
    fn timing(&self) -> PyRequestTiming {
        PyRequestTiming {
            inner: self.inner.timing.clone(),
        }
    }

    /// Cost in USD, if reported by the provider.
    #[getter]
    fn cost(&self) -> Option<f64> {
        self.inner.cost
    }

    /// Arbitrary provider-specific metadata (as a dict).
    #[getter]
    #[gen_stub(override_return_type(type_repr = "dict[str, typing.Any]", imports = ("typing",)))]
    fn metadata(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        crate::convert::json_to_py(py, &self.inner.metadata)
    }
}

// ---------------------------------------------------------------------------
// ImageResult
// ---------------------------------------------------------------------------

/// Result of an image generation or upscale operation.
#[gen_stub_pyclass]
#[pyclass(name = "ImageResult", frozen)]
pub struct PyImageResult {
    pub(crate) inner: ImageResult,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyImageResult {
    /// The generated or upscaled images.
    #[getter]
    fn images(&self) -> Vec<PyGeneratedImage> {
        self.inner
            .images
            .iter()
            .map(|i| PyGeneratedImage { inner: i.clone() })
            .collect()
    }

    /// Request timing breakdown.
    #[getter]
    fn timing(&self) -> PyRequestTiming {
        PyRequestTiming {
            inner: self.inner.timing.clone(),
        }
    }

    /// Cost in USD, if reported by the provider.
    #[getter]
    fn cost(&self) -> Option<f64> {
        self.inner.cost
    }

    /// Arbitrary provider-specific metadata (as a dict).
    #[getter]
    #[gen_stub(override_return_type(type_repr = "dict[str, typing.Any]", imports = ("typing",)))]
    fn metadata(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        crate::convert::json_to_py(py, &self.inner.metadata)
    }
}

// ---------------------------------------------------------------------------
// VideoResult
// ---------------------------------------------------------------------------

/// Result of a video generation operation.
#[gen_stub_pyclass]
#[pyclass(name = "VideoResult", frozen)]
pub struct PyVideoResult {
    pub(crate) inner: VideoResult,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyVideoResult {
    /// The generated videos.
    #[getter]
    fn videos(&self) -> Vec<PyGeneratedVideo> {
        self.inner
            .videos
            .iter()
            .map(|v| PyGeneratedVideo { inner: v.clone() })
            .collect()
    }

    /// Request timing breakdown.
    #[getter]
    fn timing(&self) -> PyRequestTiming {
        PyRequestTiming {
            inner: self.inner.timing.clone(),
        }
    }

    /// Cost in USD, if reported by the provider.
    #[getter]
    fn cost(&self) -> Option<f64> {
        self.inner.cost
    }

    /// Arbitrary provider-specific metadata (as a dict).
    #[getter]
    #[gen_stub(override_return_type(type_repr = "dict[str, typing.Any]", imports = ("typing",)))]
    fn metadata(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        crate::convert::json_to_py(py, &self.inner.metadata)
    }
}

// ---------------------------------------------------------------------------
// AudioResult
// ---------------------------------------------------------------------------

/// Result of an audio generation or TTS operation.
#[gen_stub_pyclass]
#[pyclass(name = "AudioResult", frozen)]
pub struct PyAudioResult {
    pub(crate) inner: AudioResult,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyAudioResult {
    /// The generated audio clips.
    #[getter]
    fn audio(&self) -> Vec<PyGeneratedAudio> {
        self.inner
            .audio
            .iter()
            .map(|a| PyGeneratedAudio { inner: a.clone() })
            .collect()
    }

    /// Request timing breakdown.
    #[getter]
    fn timing(&self) -> PyRequestTiming {
        PyRequestTiming {
            inner: self.inner.timing.clone(),
        }
    }

    /// Cost in USD, if reported by the provider.
    #[getter]
    fn cost(&self) -> Option<f64> {
        self.inner.cost
    }

    /// Arbitrary provider-specific metadata (as a dict).
    #[getter]
    #[gen_stub(override_return_type(type_repr = "dict[str, typing.Any]", imports = ("typing",)))]
    fn metadata(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        crate::convert::json_to_py(py, &self.inner.metadata)
    }
}

// ---------------------------------------------------------------------------
// ThreeDResult
// ---------------------------------------------------------------------------

/// Result of a 3D model generation operation.
#[gen_stub_pyclass]
#[pyclass(name = "ThreeDResult", frozen)]
pub struct PyThreeDResult {
    pub(crate) inner: ThreeDResult,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyThreeDResult {
    /// The generated 3D models.
    #[getter]
    fn models(&self) -> Vec<PyGenerated3DModel> {
        self.inner
            .models
            .iter()
            .map(|m| PyGenerated3DModel { inner: m.clone() })
            .collect()
    }

    /// Request timing breakdown.
    #[getter]
    fn timing(&self) -> PyRequestTiming {
        PyRequestTiming {
            inner: self.inner.timing.clone(),
        }
    }

    /// Cost in USD, if reported by the provider.
    #[getter]
    fn cost(&self) -> Option<f64> {
        self.inner.cost
    }

    /// Arbitrary provider-specific metadata (as a dict).
    #[getter]
    #[gen_stub(override_return_type(type_repr = "dict[str, typing.Any]", imports = ("typing",)))]
    fn metadata(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        crate::convert::json_to_py(py, &self.inner.metadata)
    }
}

// ---------------------------------------------------------------------------
// VoiceHandle
// ---------------------------------------------------------------------------

/// A persisted voice identifier returned by a voice-cloning provider.
///
/// Unlike the other result types in this module, ``VoiceHandle`` is both
/// produced by provider methods (``clone_voice``, ``list_voices``) and
/// passed back into provider methods (``delete_voice``), so it is
/// constructible from Python via ``from_py_object``.
///
/// Example:
///     >>> handle = await provider.clone_voice(VoiceCloneRequest(
///     ...     name="rachel-clone",
///     ...     reference_urls=["https://example.com/rachel.wav"],
///     ... ))
///     >>> await provider.delete_voice(handle)
#[gen_stub_pyclass]
#[pyclass(name = "VoiceHandle", from_py_object)]
#[derive(Clone)]
pub struct PyVoiceHandle {
    pub(crate) inner: VoiceHandle,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyVoiceHandle {
    #[new]
    #[pyo3(signature = (*, id, name, provider, language=None, description=None, metadata=None))]
    fn new(
        py: Python<'_>,
        id: String,
        name: String,
        provider: String,
        language: Option<String>,
        description: Option<String>,
        metadata: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        let metadata_json = match metadata {
            Some(m) => crate::convert::py_to_json(py, m.bind(py))?,
            None => serde_json::Value::Object(serde_json::Map::new()),
        };
        Ok(Self {
            inner: VoiceHandle {
                id,
                name,
                provider,
                language,
                description,
                metadata: metadata_json,
            },
        })
    }

    /// Provider-specific voice identifier (e.g. ElevenLabs ``voice_id``).
    #[getter]
    fn id(&self) -> &str {
        &self.inner.id
    }

    /// Human-readable name for the voice.
    #[getter]
    fn name(&self) -> &str {
        &self.inner.name
    }

    /// Which provider owns this voice (e.g. ``"elevenlabs"``).
    #[getter]
    fn provider(&self) -> &str {
        &self.inner.provider
    }

    /// Optional language code (e.g. ``"en"``) for language-specific voices.
    #[getter]
    fn language(&self) -> Option<String> {
        self.inner.language.clone()
    }

    /// Optional description of the voice.
    #[getter]
    fn description(&self) -> Option<String> {
        self.inner.description.clone()
    }

    /// Arbitrary provider-specific metadata (as a dict).
    #[getter]
    #[gen_stub(override_return_type(type_repr = "dict[str, typing.Any]", imports = ("typing",)))]
    fn metadata(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        crate::convert::json_to_py(py, &self.inner.metadata)
    }

    fn __repr__(&self) -> String {
        format!(
            "VoiceHandle(id={:?}, name={:?}, provider={:?})",
            self.inner.id, self.inner.name, self.inner.provider
        )
    }
}

// ---------------------------------------------------------------------------
// ComputeResult
// ---------------------------------------------------------------------------

/// Result of a completed compute job (generic, untyped output).
#[gen_stub_pyclass]
#[pyclass(name = "ComputeResult", frozen)]
pub struct PyComputeResult {
    pub(crate) inner: ComputeResult,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyComputeResult {
    /// The job handle that produced this result, if available (as a dict).
    #[getter]
    fn job(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match &self.inner.job {
            Some(job) => serialize_to_py(py, job),
            None => Ok(py.None()),
        }
    }

    /// Output data (model-specific JSON).
    #[getter]
    #[gen_stub(override_return_type(type_repr = "dict[str, typing.Any]", imports = ("typing",)))]
    fn output(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        crate::convert::json_to_py(py, &self.inner.output)
    }

    /// Request timing breakdown.
    #[getter]
    fn timing(&self) -> PyRequestTiming {
        PyRequestTiming {
            inner: self.inner.timing.clone(),
        }
    }

    /// Cost in USD, if reported by the provider.
    #[getter]
    fn cost(&self) -> Option<f64> {
        self.inner.cost
    }

    /// Raw provider-specific metadata (as a dict).
    #[getter]
    #[gen_stub(override_return_type(type_repr = "dict[str, typing.Any]", imports = ("typing",)))]
    fn metadata(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        crate::convert::json_to_py(py, &self.inner.metadata)
    }
}
