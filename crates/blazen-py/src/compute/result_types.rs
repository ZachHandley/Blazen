//! Typed wrappers for compute result types.
//!
//! These wrappers expose the Rust `*Result` shapes to Python as frozen
//! pyclasses. They are *result types* — produced by the library, never
//! constructed by users — so no `#[new]` constructors are provided.
//!
//! The nested media collections (`images`, `videos`, `audio`, `models`) are
//! currently exposed as plain Python lists of dicts via JSON serialization.
//! This keeps this module decoupled from the typed `PyGeneratedImage` etc.
//! wrappers being added in a parallel task.

use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use blazen_llm::compute::job::ComputeResult;
use blazen_llm::compute::results::{
    AudioResult, ImageResult, ThreeDResult, TranscriptionResult, TranscriptionSegment, VideoResult,
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

    /// Request timing breakdown (as a dict).
    #[getter]
    fn timing(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        serialize_to_py(py, &self.inner.timing)
    }

    /// Cost in USD, if reported by the provider.
    #[getter]
    fn cost(&self) -> Option<f64> {
        self.inner.cost
    }

    /// Arbitrary provider-specific metadata (as a dict).
    #[getter]
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
    /// The generated or upscaled images (as a list of dicts).
    #[getter]
    fn images(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        serialize_to_py(py, &self.inner.images)
    }

    /// Request timing breakdown (as a dict).
    #[getter]
    fn timing(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        serialize_to_py(py, &self.inner.timing)
    }

    /// Cost in USD, if reported by the provider.
    #[getter]
    fn cost(&self) -> Option<f64> {
        self.inner.cost
    }

    /// Arbitrary provider-specific metadata (as a dict).
    #[getter]
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
    /// The generated videos (as a list of dicts).
    #[getter]
    fn videos(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        serialize_to_py(py, &self.inner.videos)
    }

    /// Request timing breakdown (as a dict).
    #[getter]
    fn timing(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        serialize_to_py(py, &self.inner.timing)
    }

    /// Cost in USD, if reported by the provider.
    #[getter]
    fn cost(&self) -> Option<f64> {
        self.inner.cost
    }

    /// Arbitrary provider-specific metadata (as a dict).
    #[getter]
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
    /// The generated audio clips (as a list of dicts).
    #[getter]
    fn audio(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        serialize_to_py(py, &self.inner.audio)
    }

    /// Request timing breakdown (as a dict).
    #[getter]
    fn timing(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        serialize_to_py(py, &self.inner.timing)
    }

    /// Cost in USD, if reported by the provider.
    #[getter]
    fn cost(&self) -> Option<f64> {
        self.inner.cost
    }

    /// Arbitrary provider-specific metadata (as a dict).
    #[getter]
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
    /// The generated 3D models (as a list of dicts).
    #[getter]
    fn models(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        serialize_to_py(py, &self.inner.models)
    }

    /// Request timing breakdown (as a dict).
    #[getter]
    fn timing(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        serialize_to_py(py, &self.inner.timing)
    }

    /// Cost in USD, if reported by the provider.
    #[getter]
    fn cost(&self) -> Option<f64> {
        self.inner.cost
    }

    /// Arbitrary provider-specific metadata (as a dict).
    #[getter]
    fn metadata(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        crate::convert::json_to_py(py, &self.inner.metadata)
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
    fn output(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        crate::convert::json_to_py(py, &self.inner.output)
    }

    /// Request timing breakdown (as a dict).
    #[getter]
    fn timing(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        serialize_to_py(py, &self.inner.timing)
    }

    /// Cost in USD, if reported by the provider.
    #[getter]
    fn cost(&self) -> Option<f64> {
        self.inner.cost
    }

    /// Raw provider-specific metadata (as a dict).
    #[getter]
    fn metadata(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        crate::convert::json_to_py(py, &self.inner.metadata)
    }
}
